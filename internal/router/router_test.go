package router

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
	"github.com/harsha-moparthy/llmgw/internal/breaker"
	"github.com/harsha-moparthy/llmgw/internal/provider"
)

// fakeProvider is a fully controllable provider for driving router paths. Each
// call records that it happened and returns a scripted result or failure, so a
// test can assert not only the outcome but exactly which providers were tried
// and in what order.
type fakeProvider struct {
	name  string
	calls atomic.Int64

	// chat returns the scripted non-streaming outcome.
	chat func(attempt int64) (*provider.Result, *provider.Failure)
	// stream returns the scripted streaming outcome.
	stream func(attempt int64) (provider.Stream, *provider.Failure)
}

func (f *fakeProvider) Name() string   { return f.name }
func (f *fakeProvider) Vendor() string { return "fake" }
func (f *fakeProvider) Chat(ctx context.Context, req *apiv1.ChatRequest, model string) (*provider.Result, *provider.Failure) {
	n := f.calls.Add(1)
	if f.chat != nil {
		return f.chat(n)
	}
	return okResult(), nil
}
func (f *fakeProvider) ChatStream(ctx context.Context, req *apiv1.ChatRequest, model string) (provider.Stream, *provider.Failure) {
	n := f.calls.Add(1)
	if f.stream != nil {
		return f.stream(n)
	}
	return newScriptedStream([]string{"hi"}, okUsage(), nil), nil
}

func okResult() *provider.Result {
	stop := apiv1.FinishStop
	return &provider.Result{
		Response: &apiv1.ChatResponse{
			ID: "c1", Choices: []apiv1.Choice{{Message: &apiv1.Message{Content: apiv1.NewTextContent("ok")}, FinishReason: &stop}},
			Usage: okUsage(),
		},
		Usage: okUsage(),
	}
}
func okUsage() *apiv1.Usage {
	return &apiv1.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15}
}

// scriptedStream implements provider.Stream from a fixed list of content chunks
// plus an optional terminal failure.
type scriptedStream struct {
	events chan provider.StreamEvent
	usage  *apiv1.Usage
	closed atomic.Bool
}

func newScriptedStream(contents []string, usage *apiv1.Usage, terminalFail *provider.Failure) *scriptedStream {
	s := &scriptedStream{events: make(chan provider.StreamEvent, len(contents)+2), usage: usage}
	// Opening role chunk.
	s.events <- provider.StreamEvent{Chunk: &apiv1.ChatChunk{Choices: []apiv1.Choice{{Delta: &apiv1.Message{Role: apiv1.RoleAssistant}}}}, Raw: []byte("role")}
	for _, c := range contents {
		s.events <- provider.StreamEvent{Chunk: &apiv1.ChatChunk{Choices: []apiv1.Choice{{Delta: &apiv1.Message{Content: apiv1.NewTextContent(c)}}}}, Raw: []byte(c)}
	}
	if terminalFail != nil {
		s.events <- provider.StreamEvent{Err: terminalFail}
	} else {
		s.events <- provider.StreamEvent{Done: true}
	}
	close(s.events)
	return s
}
func (s *scriptedStream) Events() <-chan provider.StreamEvent { return s.events }
func (s *scriptedStream) Usage() (*apiv1.Usage, bool)         { return s.usage, false }
func (s *scriptedStream) Close() error                        { s.closed.Store(true); return nil }

// recordingSink captures router attempt outcomes.
type recordingSink struct {
	mu       sync.Mutex
	outcomes []AttemptOutcome
}

func (r *recordingSink) RecordAttempt(o AttemptOutcome) {
	r.mu.Lock()
	r.outcomes = append(r.outcomes, o)
	r.mu.Unlock()
}
func (r *recordingSink) count() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.outcomes)
}

func routeOf(targets ...*Target) *Route {
	return &Route{Alias: "gw", Targets: targets, AllowFailover: true, MaxAttempts: len(targets)}
}
func target(p provider.Provider) *Target {
	return &Target{Provider: p, Model: "m", inflight: new(atomic.Int64)}
}

func newRouter() *Router { return New(nil, Options{RequestDeadline: 5 * time.Second}) }

// TestFailoverOnFirstProviderDown is the headline behaviour: the first provider
// fails with a retryable error, the second succeeds, and the request is served.
// The assertion on attempt count is what makes this a real failover test — a
// test that only checks the response succeeded would pass even if failover never
// happened.
func TestFailoverOnFirstProviderDown(t *testing.T) {
	p1 := &fakeProvider{name: "p1", chat: func(int64) (*provider.Result, *provider.Failure) {
		return nil, &provider.Failure{Class: provider.ClassConnect}
	}}
	p2 := &fakeProvider{name: "p2"} // succeeds
	route := routeOf(target(p1), target(p2))

	sink := &recordingSink{}
	res, ar, fail := newRouter().Execute(context.Background(), "t", route, &apiv1.ChatRequest{}, ExecOptions{Sink: sink})
	if fail != nil {
		t.Fatalf("expected success after failover, got %v", fail)
	}
	if res.Response.Choices[0].Message.Content.Text() != "ok" {
		t.Errorf("unexpected content")
	}
	if ar.Attempts != 2 {
		t.Errorf("attempts = %d, want 2 (failover occurred)", ar.Attempts)
	}
	if ar.ServedBy != "p2" {
		t.Errorf("served by %q, want p2", ar.ServedBy)
	}
	if p1.calls.Load() != 1 || p2.calls.Load() != 1 {
		t.Errorf("call counts: p1=%d p2=%d, want 1 each", p1.calls.Load(), p2.calls.Load())
	}
	// Two attempts => two ledger rows: the failed one is billed too.
	if sink.count() != 2 {
		t.Errorf("ledger rows = %d, want 2 (the failed attempt gets a row)", sink.count())
	}
}

// TestNonRetryableNotFailedOver is the inverse and equally important: a
// non-retryable failure (a 400) must NOT walk the chain. The assertion that the
// second provider was never called is the whole point.
func TestNonRetryableNotFailedOver(t *testing.T) {
	p1 := &fakeProvider{name: "p1", chat: func(int64) (*provider.Result, *provider.Failure) {
		return nil, &provider.Failure{Class: provider.ClassBadRequest}
	}}
	p2 := &fakeProvider{name: "p2"}
	route := routeOf(target(p1), target(p2))

	_, ar, fail := newRouter().Execute(context.Background(), "t", route, &apiv1.ChatRequest{}, ExecOptions{})
	if fail == nil {
		t.Fatal("expected the bad-request failure to be returned")
	}
	if fail.Class != provider.ClassBadRequest {
		t.Errorf("class = %s, want bad_request", fail.Class)
	}
	if ar.Attempts != 1 {
		t.Errorf("attempts = %d, want 1 (no failover on a non-retryable error)", ar.Attempts)
	}
	if p2.calls.Load() != 0 {
		t.Errorf("p2 was called %d times; a non-retryable failure must not fail over", p2.calls.Load())
	}
}

func TestAllProvidersDown(t *testing.T) {
	down := func(int64) (*provider.Result, *provider.Failure) {
		return nil, &provider.Failure{Class: provider.ClassUpstream5xx}
	}
	p1 := &fakeProvider{name: "p1", chat: down}
	p2 := &fakeProvider{name: "p2", chat: down}
	p3 := &fakeProvider{name: "p3", chat: down}
	route := routeOf(target(p1), target(p2), target(p3))

	_, ar, fail := newRouter().Execute(context.Background(), "t", route, &apiv1.ChatRequest{}, ExecOptions{})
	if fail == nil {
		t.Fatal("expected a failure when every provider is down")
	}
	if ar.Attempts != 3 {
		t.Errorf("attempts = %d, want 3 (the whole chain was tried)", ar.Attempts)
	}
}

// TestBreakerOpenSkipsProvider verifies an open breaker takes a provider out of
// rotation without an upstream call.
func TestBreakerOpenSkipsProvider(t *testing.T) {
	br, err := breaker.New("p1", breaker.DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	br.Trip() // force open

	p1 := &fakeProvider{name: "p1"} // would succeed, but should be skipped
	p2 := &fakeProvider{name: "p2"}
	t1 := &Target{Provider: p1, Model: "m", Breaker: br, inflight: new(atomic.Int64)}
	t2 := target(p2)
	route := routeOf(t1, t2)

	_, ar, fail := newRouter().Execute(context.Background(), "t", route, &apiv1.ChatRequest{}, ExecOptions{})
	if fail != nil {
		t.Fatalf("expected success via p2, got %v", fail)
	}
	if p1.calls.Load() != 0 {
		t.Errorf("p1 was called despite an open breaker (%d times)", p1.calls.Load())
	}
	if ar.ServedBy != "p2" {
		t.Errorf("served by %q, want p2", ar.ServedBy)
	}
}

// TestBadRequestDoesNotTripBreaker is the multi-tenant isolation property: a
// client hammering malformed requests must not trip a provider's breaker and
// deny service to everyone else.
func TestBadRequestDoesNotTripBreaker(t *testing.T) {
	br, _ := breaker.New("p1", breaker.DefaultConfig())
	p1 := &fakeProvider{name: "p1", chat: func(int64) (*provider.Result, *provider.Failure) {
		return nil, &provider.Failure{Class: provider.ClassBadRequest}
	}}
	t1 := &Target{Provider: p1, Model: "m", Breaker: br, inflight: new(atomic.Int64)}
	route := &Route{Alias: "gw", Targets: []*Target{t1}, AllowFailover: false, MaxAttempts: 1}
	rt := newRouter()

	for i := 0; i < 200; i++ {
		rt.Execute(context.Background(), "t", route, &apiv1.ChatRequest{}, ExecOptions{})
	}
	if br.State() != breaker.StateClosed {
		t.Errorf("breaker state = %v after 200 bad requests, want Closed", br.State())
	}
}

// TestServerErrorTripsBreaker is the positive control: genuine provider failures
// DO count and eventually open the breaker.
func TestServerErrorTripsBreaker(t *testing.T) {
	cfg := breaker.DefaultConfig()
	br, _ := breaker.New("p1", cfg)
	p1 := &fakeProvider{name: "p1", chat: func(int64) (*provider.Result, *provider.Failure) {
		return nil, &provider.Failure{Class: provider.ClassUpstream5xx}
	}}
	t1 := &Target{Provider: p1, Model: "m", Breaker: br, inflight: new(atomic.Int64)}
	route := &Route{Alias: "gw", Targets: []*Target{t1}, AllowFailover: false, MaxAttempts: 1}
	rt := newRouter()

	for i := 0; i < 100 && br.State() == breaker.StateClosed; i++ {
		rt.Execute(context.Background(), "t", route, &apiv1.ChatRequest{}, ExecOptions{})
	}
	if br.State() == breaker.StateClosed {
		t.Error("breaker never opened despite sustained 5xx failures")
	}
}

// --- streaming ---

// streamSink is a test StreamSink that records chunks and can be told at which
// point content "reaches the client".
type streamSink struct {
	mu         sync.Mutex
	chunks     []string
	contentful bool
	errored    *provider.Failure
	done       bool
	// failOnChunk, when >0, makes OnChunk return an error on that chunk index,
	// simulating a client disconnect mid-stream.
	failOnChunk int
	seen        int
}

func (s *streamSink) OnChunk(chunk *apiv1.ChatChunk, raw []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.seen++
	if s.failOnChunk > 0 && s.seen >= s.failOnChunk {
		return context.Canceled
	}
	text := ""
	if chunk != nil {
		text = chunk.DeltaText()
	}
	if text != "" {
		s.chunks = append(s.chunks, text)
		s.contentful = true // content has now reached the client
	}
	return nil
}
func (s *streamSink) Contentful() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.contentful
}
func (s *streamSink) OnError(f *provider.Failure) {
	s.mu.Lock()
	s.errored = f
	s.mu.Unlock()
}
func (s *streamSink) OnDone() {
	s.mu.Lock()
	s.done = true
	s.mu.Unlock()
}
func (s *streamSink) text() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := ""
	for _, c := range s.chunks {
		out += c
	}
	return out
}

// TestStreamFailoverBeforeFirstByte: a provider that fails to open its stream is
// transparently retried, because nothing reached the client. This is the clean-
// failover window.
func TestStreamFailoverBeforeFirstByte(t *testing.T) {
	p1 := &fakeProvider{name: "p1", stream: func(int64) (provider.Stream, *provider.Failure) {
		return nil, &provider.Failure{Class: provider.ClassConnect}
	}}
	p2 := &fakeProvider{name: "p2", stream: func(int64) (provider.Stream, *provider.Failure) {
		return newScriptedStream([]string{"Hel", "lo"}, okUsage(), nil), nil
	}}
	route := routeOf(target(p1), target(p2))

	sink := &streamSink{}
	ar, fail := newRouter().ExecuteStream(context.Background(), "t", route, &apiv1.ChatRequest{}, sink, ExecOptions{})
	if fail != nil {
		t.Fatalf("expected transparent failover to succeed, got %v", fail)
	}
	if ar.Attempts != 2 {
		t.Errorf("attempts = %d, want 2", ar.Attempts)
	}
	if sink.text() != "Hello" {
		t.Errorf("text = %q, want Hello", sink.text())
	}
	if !sink.done {
		t.Error("OnDone was not called")
	}
	if sink.errored != nil {
		t.Errorf("an error was surfaced despite successful failover: %v", sink.errored)
	}
}

// TestStreamFailureAfterFirstByteSurfaced is the honesty boundary: a provider
// that dies AFTER emitting content must NOT be transparently retried; the error
// is surfaced to the client and the second provider is never called.
func TestStreamFailureAfterFirstByteSurfaced(t *testing.T) {
	p1 := &fakeProvider{name: "p1", stream: func(int64) (provider.Stream, *provider.Failure) {
		// Emits content, then dies with a truncation.
		return newScriptedStream([]string{"Par", "tial"}, okUsage(), &provider.Failure{Class: provider.ClassTimeout}), nil
	}}
	p2 := &fakeProvider{name: "p2"}
	route := routeOf(target(p1), target(p2))

	sink := &streamSink{}
	ar, fail := newRouter().ExecuteStream(context.Background(), "t", route, &apiv1.ChatRequest{}, sink, ExecOptions{})
	if fail == nil {
		t.Fatal("expected the mid-stream failure to be surfaced, not swallowed")
	}
	if p2.calls.Load() != 0 {
		t.Errorf("p2 was called %d times; a mid-stream failure must NOT fail over", p2.calls.Load())
	}
	if sink.errored == nil {
		t.Error("OnError was not called; the client would see a silent truncation")
	}
	// The partial content did reach the client.
	if sink.text() != "Partial" {
		t.Errorf("partial text = %q, want Partial", sink.text())
	}
	if ar.Attempts != 1 {
		t.Errorf("attempts = %d, want 1", ar.Attempts)
	}
}

// TestStreamClientDisconnect: when the sink reports a write error (client left),
// the router stops and does not fail over, and the partial usage is available for
// billing.
func TestStreamClientDisconnect(t *testing.T) {
	served := newScriptedStream([]string{"a", "b", "c", "d"}, okUsage(), nil)
	p1 := &fakeProvider{name: "p1", stream: func(int64) (provider.Stream, *provider.Failure) { return served, nil }}
	p2 := &fakeProvider{name: "p2"}
	route := routeOf(target(p1), target(p2))

	sink := &streamSink{failOnChunk: 2} // disconnect after the first content chunk
	_, fail := newRouter().ExecuteStream(context.Background(), "t", route, &apiv1.ChatRequest{}, sink, ExecOptions{})
	if fail == nil || fail.Class != provider.ClassCancelled {
		t.Fatalf("expected a cancelled failure on client disconnect, got %v", fail)
	}
	if p2.calls.Load() != 0 {
		t.Error("client disconnect must not trigger failover")
	}
	if !served.closed.Load() {
		t.Error("the upstream stream was not closed after client disconnect")
	}
}

func TestDeadlineExhaustionMidChain(t *testing.T) {
	slow := func(int64) (*provider.Result, *provider.Failure) {
		time.Sleep(40 * time.Millisecond)
		return nil, &provider.Failure{Class: provider.ClassUpstream5xx}
	}
	p1 := &fakeProvider{name: "p1", chat: slow}
	p2 := &fakeProvider{name: "p2", chat: slow}
	p3 := &fakeProvider{name: "p3", chat: slow}
	route := routeOf(target(p1), target(p2), target(p3))

	rt := New(nil, Options{RequestDeadline: 50 * time.Millisecond})
	_, ar, fail := rt.Execute(context.Background(), "t", route, &apiv1.ChatRequest{}, ExecOptions{})
	if fail == nil {
		t.Fatal("expected a failure")
	}
	// The deadline should cut the chain short before all three are tried.
	if ar.Attempts == 3 {
		t.Errorf("all 3 attempts ran despite a 50ms deadline and 40ms/attempt; deadline not enforced")
	}
}

// TestConcurrentExecute runs many Execute calls at once to shake out races in
// the shared stats and per-target inflight counters.
func TestConcurrentExecute(t *testing.T) {
	p1 := &fakeProvider{name: "p1", chat: func(n int64) (*provider.Result, *provider.Failure) {
		if n%3 == 0 {
			return nil, &provider.Failure{Class: provider.ClassConnect}
		}
		return okResult(), nil
	}}
	p2 := &fakeProvider{name: "p2"}
	route := routeOf(target(p1), target(p2))
	rt := newRouter()

	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 20; j++ {
				sink := &recordingSink{}
				rt.Execute(context.Background(), "t", route, &apiv1.ChatRequest{}, ExecOptions{Sink: sink})
			}
		}()
	}
	wg.Wait()
	if rt.Stats().Attempts.Load() == 0 {
		t.Error("no attempts recorded")
	}
	// Every target's inflight must return to zero after all requests complete.
	for _, tgt := range route.Targets {
		if got := tgt.Inflight(); got != 0 {
			t.Errorf("target %s inflight = %d after all requests, want 0", tgt.Provider.Name(), got)
		}
	}
}
