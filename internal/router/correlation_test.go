package router

import (
	"context"
	"sync/atomic"
	"testing"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
	"github.com/harsha-moparthy/llmgw/internal/provider"
)

// This file pins the correlation bug that the LIVE failover reconciliation
// found: the router must stamp the ACTUAL attempt number into each attempt's
// context, so the number a provider logs is the number the ledger recorded.
//
// The reconciliation keys on (request id, attempt). If the ledger records a
// failed-over request as attempt 2 but the provider that served it was told
// attempt 1, the rows never pair up and a perfectly correct bill fails to
// reconcile. No unit test caught this — it only manifests when a provider
// actually fails over — so it gets one here.

// correlationSpy records the correlation each attempt was given, by reading it
// back out of the context the router passed in.
type correlationSpy struct {
	name  string
	calls atomic.Int64
	// seen accumulates the (requestID, attempt) each call observed.
	seen []provider.Correlation
	fail func(attempt int64) *provider.Failure
}

func (s *correlationSpy) Name() string   { return s.name }
func (s *correlationSpy) Vendor() string { return "spy" }

func (s *correlationSpy) record(ctx context.Context) {
	// Read back exactly what a real adapter would stamp onto the upstream
	// request headers.
	c, _ := provider.CorrelationFrom(ctx)
	s.seen = append(s.seen, c)
}

func (s *correlationSpy) Chat(ctx context.Context, req *apiv1.ChatRequest, model string) (*provider.Result, *provider.Failure) {
	n := s.calls.Add(1)
	s.record(ctx)
	if s.fail != nil {
		if f := s.fail(n); f != nil {
			return nil, f
		}
	}
	return okResult(), nil
}

func (s *correlationSpy) ChatStream(ctx context.Context, req *apiv1.ChatRequest, model string) (provider.Stream, *provider.Failure) {
	n := s.calls.Add(1)
	s.record(ctx)
	if s.fail != nil {
		if f := s.fail(n); f != nil {
			return nil, f
		}
	}
	return newScriptedStream([]string{"ok"}, okUsage(), nil), nil
}

// TestAttemptNumberIsStampedPerAttempt is the regression test. Attempt 1 must be
// told "1" and attempt 2 must be told "2"; before the fix both were told "1".
func TestAttemptNumberIsStampedPerAttempt(t *testing.T) {
	p1 := &correlationSpy{name: "p1", fail: func(int64) *provider.Failure {
		return &provider.Failure{Class: provider.ClassConnect} // retryable: forces failover
	}}
	p2 := &correlationSpy{name: "p2"} // succeeds
	route := routeOf(target(p1), target(p2))

	ctx := provider.WithCorrelation(context.Background(), provider.Correlation{RequestID: "gw-req-1"})
	_, ar, fail := newRouter().Execute(ctx, "t", route, &apiv1.ChatRequest{}, ExecOptions{})
	if fail != nil {
		t.Fatalf("expected failover to succeed, got %v", fail)
	}
	if ar.Attempts != 2 {
		t.Fatalf("attempts = %d, want 2 (the test needs a real failover to be meaningful)", ar.Attempts)
	}

	if len(p1.seen) != 1 || len(p2.seen) != 1 {
		t.Fatalf("expected one call each; p1=%d p2=%d", len(p1.seen), len(p2.seen))
	}
	if got := p1.seen[0].Attempt; got != 1 {
		t.Errorf("first attempt was told attempt=%d, want 1", got)
	}
	// This is the assertion that failed before the fix: the second provider was
	// told attempt 1, so its log row could never match the ledger's attempt 2.
	if got := p2.seen[0].Attempt; got != 2 {
		t.Errorf("second attempt was told attempt=%d, want 2; the provider's log row "+
			"will not match the ledger's attempt-2 row and the reconciliation will fail", got)
	}
	// The request id must be carried through unchanged on both attempts, or the
	// two sides key on different requests entirely.
	for i, c := range [][]provider.Correlation{p1.seen, p2.seen} {
		if c[0].RequestID != "gw-req-1" {
			t.Errorf("provider %d saw request id %q, want gw-req-1", i+1, c[0].RequestID)
		}
	}
}

// TestAttemptStampingOnStreamingPath: the streaming path has its own attempt loop
// and needed the same fix, so it needs the same test.
func TestAttemptStampingOnStreamingPath(t *testing.T) {
	p1 := &correlationSpy{name: "p1", fail: func(int64) *provider.Failure {
		return &provider.Failure{Class: provider.ClassConnect}
	}}
	p2 := &correlationSpy{name: "p2"}
	route := routeOf(target(p1), target(p2))

	ctx := provider.WithCorrelation(context.Background(), provider.Correlation{RequestID: "gw-req-2"})
	sink := &streamSink{}
	ar, fail := newRouter().ExecuteStream(ctx, "t", route, &apiv1.ChatRequest{}, sink, ExecOptions{})
	if fail != nil {
		t.Fatalf("expected transparent failover to succeed, got %v", fail)
	}
	if ar.Attempts != 2 {
		t.Fatalf("attempts = %d, want 2", ar.Attempts)
	}
	if len(p2.seen) != 1 {
		t.Fatalf("second provider was called %d times, want 1", len(p2.seen))
	}
	if got := p2.seen[0].Attempt; got != 2 {
		t.Errorf("streaming second attempt was told attempt=%d, want 2", got)
	}
}

// TestWithAttemptPreservesRequestID guards the helper itself: overwriting the
// attempt must not drop or change the request id.
func TestWithAttemptPreservesRequestID(t *testing.T) {
	ctx := provider.WithCorrelation(context.Background(), provider.Correlation{RequestID: "abc", Attempt: 1})
	got := mustCorrelation(provider.WithAttempt(ctx, 7))
	if got.RequestID != "abc" {
		t.Errorf("request id = %q after WithAttempt, want abc", got.RequestID)
	}
	if got.Attempt != 7 {
		t.Errorf("attempt = %d, want 7", got.Attempt)
	}
	// Called on a context with no correlation at all, it must not panic and must
	// still carry the attempt.
	bare := mustCorrelation(provider.WithAttempt(context.Background(), 3))
	if bare.Attempt != 3 {
		t.Errorf("attempt = %d on a bare context, want 3", bare.Attempt)
	}
}

func mustCorrelation(ctx context.Context) provider.Correlation {
	c, _ := provider.CorrelationFrom(ctx)
	return c
}
