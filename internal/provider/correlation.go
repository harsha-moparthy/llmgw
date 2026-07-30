package provider

import (
	"context"
	"net/http"
	"strconv"
)

// Correlation headers the gateway sends upstream so a provider's own log can be
// reconciled against the gateway's ledger on the SAME key.
//
// This is the mechanism that makes "reconciles exactly against mock-provider
// logs" possible. The two sides compute cost independently, but they must agree
// on WHICH request each row describes, and neither can read the other's
// internal id. So the gateway mints a request id, sends it (and the attempt
// number) as headers, and the mock provider records exactly those. Real
// providers ignore unknown headers, so sending them is harmless in production
// and load-bearing for the reconciliation demo.
const (
	HeaderRequestID = "X-Llmgw-Request-Id"
	HeaderAttempt   = "X-Llmgw-Attempt"
)

type correlationKey struct{}

// Correlation carries the request id and attempt number for one upstream call.
type Correlation struct {
	RequestID string
	Attempt   int
}

// WithCorrelation attaches correlation info to a context, read by the adapters
// when they build the upstream request.
func WithCorrelation(ctx context.Context, c Correlation) context.Context {
	return context.WithValue(ctx, correlationKey{}, c)
}

// WithAttempt returns a context carrying the same request id as its parent but a
// new attempt number.
//
// This exists because the reconciliation keys on (request id, attempt): if the
// gateway's ledger records a failed-over request as attempt 2 but the provider
// that served it was told attempt 1, the two rows never match and a perfectly
// correct bill fails to reconcile. The router calls this per attempt so the
// number the provider logs is the number the ledger recorded. The bug this
// prevents is invisible until a provider actually fails over under load, which
// is exactly when it was found.
func WithAttempt(ctx context.Context, attempt int) context.Context {
	c, _ := correlationFrom(ctx)
	c.Attempt = attempt
	return context.WithValue(ctx, correlationKey{}, c)
}

// correlationFrom extracts correlation info, if any.
func correlationFrom(ctx context.Context) (Correlation, bool) {
	c, ok := ctx.Value(correlationKey{}).(Correlation)
	return c, ok
}

// applyCorrelation sets the correlation headers on an upstream request.
func applyCorrelation(ctx context.Context, req *http.Request) {
	if c, ok := correlationFrom(ctx); ok {
		if c.RequestID != "" {
			req.Header.Set(HeaderRequestID, c.RequestID)
		}
		if c.Attempt > 0 {
			req.Header.Set(HeaderAttempt, strconv.Itoa(c.Attempt))
		}
	}
}
