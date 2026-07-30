package provider

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"sync"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
	"github.com/harsha-moparthy/llmgw/internal/sse"
)

// openaiStream adapts an OpenAI-shaped SSE response body to the Stream
// interface.
//
// The two responsibilities that make this more than a loop:
//
//   - Truncation detection. A stream that ends without [DONE] is a provider that
//     died, and it must surface as a Failure carrying the usage seen so far, not
//     as a clean channel close. sse.Decoder already distinguishes the two; this
//     type propagates that distinction rather than flattening it.
//
//   - Never leaking the connection. The response body must be closed on every
//     exit path, and Close must be idempotent, or an abandoned stream holds an
//     upstream connection out of the pool for as long as the provider keeps it
//     open — which under load exhausts the pool and looks like the provider
//     being slow.
type openaiStream struct {
	provider string
	model    string

	body   io.ReadCloser
	dec    *sse.Decoder
	events chan StreamEvent

	mu        sync.Mutex
	closed    bool
	usage     *apiv1.Usage
	estimated bool
	done      chan struct{}
}

func newOpenAIStream(ctx context.Context, providerName, model string, resp *http.Response) *openaiStream {
	s := &openaiStream{
		provider: providerName,
		model:    model,
		body:     resp.Body,
		dec:      sse.NewDecoder(resp.Body),
		events:   make(chan StreamEvent, 16),
		done:     make(chan struct{}),
	}
	go s.run(ctx)
	return s
}

// run reads frames until the stream terminates, translating each into a
// StreamEvent. It is the only writer to the events channel and closes it exactly
// once on exit.
func (s *openaiStream) run(ctx context.Context) {
	defer close(s.events)
	for {
		// Honour cancellation between frames so a client disconnect promptly
		// stops us reading a provider that is still generating — otherwise the
		// tenant keeps paying for tokens no one will read.
		select {
		case <-ctx.Done():
			s.emit(StreamEvent{Err: &Failure{
				Class: ClassCancelled, Provider: s.provider, Model: s.model,
				Err: ctx.Err(), UsageAtFailure: s.snapshotUsage(),
			}})
			return
		default:
		}

		ev, err := s.dec.Next()
		if err != nil {
			if errors.Is(err, io.EOF) {
				// EOF here means the loop never returned via the [DONE] sentinel
				// below — an OpenAI-compatible stream that ends WITHOUT [DONE] is
				// truncated, even when its last SSE frame was well-formed and the
				// EOF is "clean" at the framing level. A provider that dies after
				// flushing a complete frame produces exactly this, and treating
				// it as a normal end would serve a half-generated answer as a
				// whole one. So it is a failure, carrying the usage seen so far.
				s.emit(StreamEvent{Err: &Failure{
					Class: ClassTimeout, Provider: s.provider, Model: s.model,
					Err: io.ErrUnexpectedEOF, UsageAtFailure: s.snapshotUsage(),
				}})
				return
			}
			class := ClassUpstream5xx
			if errors.Is(err, sse.ErrTruncated) {
				class = ClassTimeout // a stream that dies mid-frame is, to us, a timeout-class transport failure
			}
			s.emit(StreamEvent{Err: &Failure{
				Class: class, Provider: s.provider, Model: s.model,
				Err: err, UsageAtFailure: s.snapshotUsage(),
			}})
			return
		}
		if ev.IsComment() {
			// Keep-alive. Forward it so a caller can reset a read deadline, but
			// it carries no content.
			s.emit(StreamEvent{Raw: nil})
			continue
		}
		if ev.Done {
			s.emit(StreamEvent{Done: true})
			return
		}

		chunk, uerr := parseChunk(ev.Data)
		if uerr != nil {
			// A malformed JSON payload mid-stream. This is a provider or proxy
			// defect; surface it rather than forwarding garbage the client's SDK
			// will choke on.
			s.emit(StreamEvent{Err: &Failure{
				Class: ClassUpstream5xx, Provider: s.provider, Model: s.model,
				Err: uerr, Body: safeExcerpt(ev.Data), UsageAtFailure: s.snapshotUsage(),
			}})
			return
		}
		if chunk != nil && chunk.Usage != nil {
			s.setUsage(chunk.Usage, false)
		}
		s.emit(StreamEvent{Chunk: chunk, Raw: append([]byte(nil), ev.Data...)})
	}
}

func (s *openaiStream) emit(ev StreamEvent) {
	select {
	case s.events <- ev:
	case <-s.done:
	}
}

func (s *openaiStream) setUsage(u *apiv1.Usage, estimated bool) {
	s.mu.Lock()
	s.usage = u
	s.estimated = estimated
	s.mu.Unlock()
}

func (s *openaiStream) snapshotUsage() *apiv1.Usage {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.usage
}

// Events implements Stream.
func (s *openaiStream) Events() <-chan StreamEvent { return s.events }

// Usage implements Stream.
func (s *openaiStream) Usage() (*apiv1.Usage, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.usage, s.estimated
}

// Close implements Stream. Idempotent, and it both signals run to stop and
// closes the body so the connection is released.
func (s *openaiStream) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	close(s.done)
	s.mu.Unlock()
	// Closing the body unblocks a run goroutine parked in a Read on a stalled
	// provider, which is what makes Close safe to call on a hung stream.
	return s.body.Close()
}

// parseChunk unmarshals a chunk payload.
func parseChunk(data []byte) (*apiv1.ChatChunk, error) {
	var c apiv1.ChatChunk
	if err := json.Unmarshal(data, &c); err != nil {
		return nil, err
	}
	return &c, nil
}

func safeExcerpt(b []byte) string {
	const max = 256
	if len(b) > max {
		return string(b[:max]) + "..."
	}
	return string(b)
}

// readErrorBody reads a bounded excerpt of an error response body, so a
// diagnostic never pulls a multi-megabyte body into memory.
func readErrorBody(r io.Reader) string {
	b, _ := io.ReadAll(io.LimitReader(r, 4<<10))
	return string(b)
}

// bufferedBody wraps a response body with a bufio.Reader, used where an adapter
// needs to peek. Not currently required by the OpenAI path but kept for the
// Anthropic translator, which reads event-typed frames.
func bufferedBody(rc io.ReadCloser) *bufio.Reader { return bufio.NewReaderSize(rc, 16<<10) }
