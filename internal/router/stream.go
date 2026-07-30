package router

import (
	"context"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
	"github.com/harsha-moparthy/llmgw/internal/provider"
)

// StreamSink is what the server passes to ExecuteStream to receive frames. The
// router does not know about HTTP or SSE; it hands the caller each chunk and the
// caller decides how to write it.
//
// The critical contract is FirstContent: the sink must report whether any
// content byte has already reached the client, because that is what determines
// whether a failure is still in the transparent-failover window. The server
// implements this by holding the response headers unwritten until the first
// content frame; until it writes, FirstContent stays false and the router may
// retry on another provider.
type StreamSink interface {
	// OnChunk is called for each content chunk. Returning an error (e.g. the
	// client disconnected) aborts the stream.
	OnChunk(chunk *apiv1.ChatChunk, raw []byte) error
	// Contentful reports whether any content has been written to the client yet.
	// While false, a failure is transparently retryable; once true, it is not.
	Contentful() bool
	// OnError is called with a failure that is being surfaced to the client
	// (i.e. one that occurred after content began). The sink writes an error
	// frame. It is NOT called for a failure that the router will transparently
	// retry.
	OnError(f *provider.Failure)
	// OnDone is called when the stream completed cleanly.
	OnDone()
}

// ExecuteStream runs the streaming path down the failover chain, honouring the
// transparent-failover window described in the package doc.
func (r *Router) ExecuteStream(ctx context.Context, tenant string, route *Route, req *apiv1.ChatRequest, sink StreamSink, opt ExecOptions) (*AttemptResult, *provider.Failure) {
	deadline := time.Now().Add(r.requestDeadline)
	if d, ok := ctx.Deadline(); ok && d.Before(deadline) {
		deadline = d
	}

	targets := r.selectable(route, opt.DegradeToCheaper, opt.EstPromptTokens, opt.EstMaxTokens)
	if len(targets) == 0 {
		r.stats.NoProvider.Add(1)
		return &AttemptResult{}, &provider.Failure{Class: provider.ClassUnknown, Err: provider.ErrNoProviders}
	}

	maxAttempts := route.MaxAttempts
	if maxAttempts <= 0 || maxAttempts > len(targets) {
		maxAttempts = len(targets)
	}
	if !route.AllowFailover {
		maxAttempts = 1
	}

	ar := &AttemptResult{}
	var lastFail *provider.Failure

	for i := 0; i < maxAttempts && i < len(targets); i++ {
		if err := ctx.Err(); err != nil {
			return ar, &provider.Failure{Class: provider.ClassCancelled, Err: err}
		}
		if time.Until(deadline) <= 0 {
			ar.DeadlineExceeded = true
			break
		}

		t := targets[i]
		attemptNo := i + 1
		r.stats.Attempts.Add(1)
		ar.Attempts = attemptNo

		attemptCtx, cancel := context.WithDeadline(ctx, deadline)
		attemptCtx = provider.WithAttempt(attemptCtx, attemptNo)
		start := time.Now()
		t.inflightAdd(1)
		stream, fail := t.Provider.ChatStream(attemptCtx, req, t.Model)

		if fail != nil {
			// Failed before the stream even opened. Nothing was written to the
			// client, so this is squarely in the transparent window.
			t.inflightAdd(-1)
			cancel()
			fail.Provider, fail.Model = t.Provider.Name(), t.Model
			r.recordHealth(t, fail)
			r.report(opt.Sink, r.streamOutcome(tenant, route, t, attemptNo, fail, nil, time.Since(start)))
			lastFail = fail
			if !fail.Retryable() || !route.AllowFailover {
				break
			}
			continue
		}

		// Consume the stream. relayStream returns the failure if one occurred and
		// whether any content had reached the client at that point.
		streamFail, usage := r.relayStream(stream, sink)
		stream.Close()
		t.inflightAdd(-1)
		cancel()
		latency := time.Since(start)

		if streamFail == nil {
			r.recordHealth(t, nil)
			okOutcome := r.streamOutcome(tenant, route, t, attemptNo, nil, usage, latency)
			okOutcome.Served = true
			r.report(opt.Sink, okOutcome)
			ar.ServedBy, ar.ServedModel = t.Provider.Name(), t.Model
			if attemptNo > 1 {
				r.stats.Failovers.Add(1)
			}
			sink.OnDone()
			return ar, nil
		}

		streamFail.Provider, streamFail.Model = t.Provider.Name(), t.Model
		if usage != nil {
			streamFail.UsageAtFailure = usage
		}
		r.recordHealth(t, streamFail)
		outcome := r.streamOutcome(tenant, route, t, attemptNo, streamFail, usage, latency)
		outcome.Served = sink.Contentful()
		r.report(opt.Sink, outcome)
		lastFail = streamFail

		// The failover decision now depends on whether content already reached
		// the client. If it did, Retryable() is false (BytesStreamed was set from
		// the sink), and we surface the error rather than splicing a second
		// model's output onto the first.
		if sink.Contentful() {
			streamFail.BytesStreamed = 1 // mark as past the transparent window
			r.stats.StreamSurfaced.Add(1)
			sink.OnError(streamFail)
			ar.ServedBy, ar.ServedModel = t.Provider.Name(), t.Model
			return ar, streamFail
		}
		if !streamFail.Retryable() || !route.AllowFailover {
			break
		}
		// No content yet and the failure is retryable: loop to the next target,
		// transparently.
	}

	if lastFail == nil {
		lastFail = &provider.Failure{Class: provider.ClassUnknown, Err: provider.ErrNoProviders}
	}
	// If we exhausted the chain before any content, surface the last failure to
	// the client so it is not left hanging.
	if !sink.Contentful() {
		sink.OnError(lastFail)
	}
	r.stats.Exhausted.Add(1)
	return ar, lastFail
}

// relayStream forwards a provider stream to the sink, returning the terminal
// failure (nil on clean completion) and the accumulated usage.
func (r *Router) relayStream(stream provider.Stream, sink StreamSink) (*provider.Failure, *apiv1.Usage) {
	for ev := range stream.Events() {
		if ev.Err != nil {
			usage, _ := stream.Usage()
			return ev.Err, usage
		}
		if ev.Done {
			usage, _ := stream.Usage()
			return nil, usage
		}
		if ev.Chunk == nil && ev.Raw == nil {
			continue // keep-alive
		}
		if err := sink.OnChunk(ev.Chunk, ev.Raw); err != nil {
			// The sink failed to write — client disconnected or could not keep
			// up. This is not a provider failure; it is the client leaving, so
			// classify it as cancelled and stop. The partial usage is still
			// returned so it can be billed.
			usage, _ := stream.Usage()
			return &provider.Failure{Class: provider.ClassCancelled, Err: err, BytesStreamed: 1}, usage
		}
	}
	// Channel closed without an explicit Done/Err. Treat as a clean end using
	// whatever usage was seen; the provider stream contract closes the channel
	// after a terminal event, so this is belt-and-braces.
	usage, _ := stream.Usage()
	return nil, usage
}

func (r *Router) streamOutcome(tenant string, route *Route, t *Target, attempt int, fail *provider.Failure, usage *apiv1.Usage, latency time.Duration) AttemptOutcome {
	o := AttemptOutcome{
		Tenant: tenant, Alias: route.Alias, Provider: t.Provider.Name(),
		UpstreamModel: t.Model, Attempt: attempt, FailedOver: attempt > 1,
		Latency: latency, Streamed: true, Usage: usage, Failure: fail,
	}
	if usage != nil {
		o.Cost = r.cost(t.Model, usage)
	}
	return o
}
