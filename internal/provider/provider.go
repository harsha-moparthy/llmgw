// Package provider defines the contract every upstream adapter implements, and
// the vocabulary of upstream failure that routing and failover decisions are
// made from.
//
// The design commitment in this file is the distinction between a failure that
// may be retried on another provider and one that may not. Getting that
// backwards is the classic gateway bug in both directions:
//
//   - Retrying a non-idempotent or client-caused failure (a 400 for a malformed
//     tool schema) walks the same doomed request down every provider in the
//     chain, multiplying latency and cost to produce the same 400 the first
//     provider produced instantly.
//   - Refusing to retry a genuinely transient failure defeats the entire point
//     of the gateway.
//
// So Failure carries an explicit Retryable decision made at the adapter, where
// the provider's status codes and error bodies are understood, rather than
// being re-derived by a switch statement further up the stack that has to guess.
package provider

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
)

// Class categorises an upstream failure. The zero value is ClassUnknown, which
// is treated as retryable-but-suspicious: safer to fail over than to hard-fail
// a request on an error the adapter did not recognise.
type Class int

// Failure classes, ordered roughly from "our fault" to "their fault".
const (
	// ClassUnknown is an unrecognised error. Retryable.
	ClassUnknown Class = iota
	// ClassConnect is a failure to establish a connection: DNS, refused, TLS.
	// Retryable, and importantly it means zero tokens were consumed upstream.
	ClassConnect
	// ClassTimeout is a request that exceeded its deadline. Retryable, but it
	// may have consumed tokens upstream — see Failure.MayHaveBilled.
	ClassTimeout
	// ClassRateLimit is a 429. Retryable on a different provider, and carries
	// RetryAfter when the provider supplied one.
	ClassRateLimit
	// ClassUpstream5xx is a 500/502/503/504. Retryable.
	ClassUpstream5xx
	// ClassOverloaded is an explicit capacity rejection (Anthropic's 529,
	// or a 503 with an overloaded body). Retryable.
	ClassOverloaded
	// ClassBadRequest is a 400/404/422: the request itself is wrong. NOT
	// retryable — every provider will reject it, so failing over only burns
	// latency and hides the real error from the client.
	ClassBadRequest
	// ClassAuth is a 401/403: our credential for this provider is bad. NOT
	// retryable on this provider, but it says nothing about the others, so the
	// router may still try a different one.
	ClassAuth
	// ClassContentFilter is a provider-side safety refusal. NOT retryable:
	// another provider may well also refuse, and silently shopping a refused
	// prompt around providers is not behaviour a gateway should have by default.
	ClassContentFilter
	// ClassContextLength is a prompt that exceeds the model's window. NOT
	// retryable on an equivalent model, though a router with a larger-window
	// fallback could act on it. Distinguished from ClassBadRequest because it
	// is the one 400 that a *differently configured* route could satisfy.
	ClassContextLength
	// ClassCancelled is the client going away. Never retried.
	ClassCancelled
)

// String renders the class for logs and metric labels.
func (c Class) String() string {
	switch c {
	case ClassConnect:
		return "connect"
	case ClassTimeout:
		return "timeout"
	case ClassRateLimit:
		return "rate_limit"
	case ClassUpstream5xx:
		return "upstream_5xx"
	case ClassOverloaded:
		return "overloaded"
	case ClassBadRequest:
		return "bad_request"
	case ClassAuth:
		return "auth"
	case ClassContentFilter:
		return "content_filter"
	case ClassContextLength:
		return "context_length"
	case ClassCancelled:
		return "cancelled"
	default:
		return "unknown"
	}
}

// retryableByDefault encodes the retry decision per class. It is a table rather
// than a chain of ifs so that the policy is auditable in one glance and so the
// must-fail control in selfcheck can assert the whole table.
var retryableByDefault = map[Class]bool{
	ClassUnknown:       true,
	ClassConnect:       true,
	ClassTimeout:       true,
	ClassRateLimit:     true,
	ClassUpstream5xx:   true,
	ClassOverloaded:    true,
	ClassBadRequest:    false,
	ClassAuth:          false,
	ClassContentFilter: false,
	ClassContextLength: false,
	ClassCancelled:     false,
}

// Retryable reports whether a failure of this class should be failed over to
// another provider.
func (c Class) Retryable() bool { return retryableByDefault[c] }

// CountsAgainstHealth reports whether a failure of this class is evidence that
// the *provider* is unhealthy, as opposed to evidence that the request was bad.
//
// This distinction is what keeps a client looping on malformed requests from
// tripping the circuit breaker and taking a healthy provider out of rotation
// for every other tenant. It is the single most important line in this file.
func (c Class) CountsAgainstHealth() bool {
	switch c {
	case ClassBadRequest, ClassContentFilter, ClassContextLength, ClassCancelled:
		return false
	case ClassAuth:
		// A 401 *is* a provider-level problem — the route is unusable until an
		// operator fixes the key — so it counts. It is simultaneously
		// non-retryable on this provider and a health signal about it.
		return true
	default:
		return true
	}
}

// Failure is a classified upstream error.
type Failure struct {
	// Class is the adapter's classification.
	Class Class
	// Provider is the provider instance name that produced the failure.
	Provider string
	// Model is the upstream model name that was attempted.
	Model string
	// StatusCode is the upstream HTTP status, or 0 for transport failures.
	StatusCode int
	// RetryAfter is the provider's requested backoff, if it supplied one.
	RetryAfter time.Duration
	// Body is a bounded excerpt of the upstream error body, for diagnostics.
	Body string
	// Err is the underlying error, if any.
	Err error
	// BytesStreamed is how many response bytes had already been written to the
	// client when the failure occurred. Non-zero means failover is no longer
	// transparent: see the mid-stream discussion in internal/server.
	BytesStreamed int64
	// UsageAtFailure is whatever token usage was known when the failure hit.
	// A stream that dies after 400 completion tokens still cost 400 completion
	// tokens, and the ledger must record them or the reconciliation will not
	// balance against the provider's own logs.
	UsageAtFailure *apiv1.Usage
}

// Error implements error.
func (f *Failure) Error() string {
	if f == nil {
		return "<nil provider failure>"
	}
	base := fmt.Sprintf("provider %s (model %s): %s", f.Provider, f.Model, f.Class)
	if f.StatusCode != 0 {
		base += fmt.Sprintf(" status=%d", f.StatusCode)
	}
	if f.Err != nil {
		base += ": " + f.Err.Error()
	} else if f.Body != "" {
		base += ": " + f.Body
	}
	return base
}

// Unwrap exposes the underlying error to errors.Is/As.
func (f *Failure) Unwrap() error { return f.Err }

// Retryable reports whether this specific failure may be failed over.
//
// Mid-stream is the subtle case. Once bytes have reached the client, a silent
// retry would splice a second provider's continuation onto the first's partial
// output, producing a response that no single model ever generated — sentences
// interrupted mid-clause, contradicted facts, duplicated tool calls. That is
// worse than an honest error, so the default is to refuse. The server can still
// choose to fail over when it knows no *content* was emitted yet (only the
// role-opening frame), and it signals that by leaving BytesStreamed at zero.
func (f *Failure) Retryable() bool {
	if f == nil {
		return false
	}
	if f.BytesStreamed > 0 {
		return false
	}
	return f.Class.Retryable()
}

// MayHaveBilled reports whether the upstream provider plausibly charged us for
// this failed attempt.
//
// A connection that never established cannot have been billed. A request that
// timed out after the provider began generating almost certainly was. The
// ledger uses this to decide whether a failed attempt gets a cost row, because
// silently treating every failure as free is exactly the assumption that makes
// a reconciliation drift from the provider's invoice.
func (f *Failure) MayHaveBilled() bool {
	if f == nil {
		return false
	}
	if f.UsageAtFailure != nil {
		return true
	}
	switch f.Class {
	case ClassConnect, ClassRateLimit, ClassAuth, ClassBadRequest:
		// Rejected before any generation happened.
		return false
	case ClassTimeout, ClassCancelled:
		// Generation may have been underway. Both are billed by real providers.
		return true
	default:
		return f.StatusCode == 0 || f.StatusCode >= 500
	}
}

// ClassifyStatus maps an HTTP status to a Class. Adapters may override with
// body-specific knowledge (e.g. distinguishing a context-length 400).
func ClassifyStatus(code int) Class {
	switch code {
	case http.StatusTooManyRequests:
		return ClassRateLimit
	case http.StatusUnauthorized, http.StatusForbidden:
		return ClassAuth
	case http.StatusBadRequest, http.StatusNotFound, http.StatusUnprocessableEntity:
		return ClassBadRequest
	case 529: // Anthropic's non-standard "overloaded".
		return ClassOverloaded
	}
	switch {
	case code >= 500:
		return ClassUpstream5xx
	case code >= 200 && code < 300:
		return ClassUnknown // caller should not be classifying a success
	default:
		return ClassUnknown
	}
}

// Result is the outcome of a non-streaming call.
type Result struct {
	Response *apiv1.ChatResponse
	// Usage is the authoritative token usage, taken from the provider's own
	// accounting when it supplies one.
	Usage *apiv1.Usage
	// UsageIsEstimated is true when the provider did not report usage and the
	// gateway had to estimate it. It propagates all the way to the ledger and
	// into the reconciliation report, because a bill built partly from
	// estimates is a different object from one built entirely from provider
	// counts, and conflating them is how a reconciliation appears to pass.
	UsageIsEstimated bool
	// UpstreamLatency is time to full response.
	UpstreamLatency time.Duration
}

// StreamEvent is one event from a streaming call.
//
// A single channel of a sum type is used rather than separate data/error
// channels so that ordering between the last chunk and a terminal error cannot
// be raced — a mid-stream failure must be observed strictly after the chunks
// that preceded it, and two channels cannot guarantee that.
type StreamEvent struct {
	// Chunk is a parsed delta frame, or nil for non-chunk events.
	Chunk *apiv1.ChatChunk
	// Raw is the exact bytes of the upstream `data:` payload. Preserved so the
	// pass-through path can forward provider fields the gateway does not model
	// without re-serialising (which would also reorder keys).
	Raw []byte
	// Done marks the upstream [DONE] sentinel.
	Done bool
	// Err is a terminal failure. When set, no further events follow.
	Err *Failure
}

// Stream is an in-progress streaming response.
type Stream interface {
	// Events returns the event channel. It is closed after a Done or Err event.
	Events() <-chan StreamEvent
	// Usage returns the final usage once the stream has terminated, and
	// reports whether it came from the provider (false) or was estimated
	// (true). Calling it before termination returns nil.
	Usage() (*apiv1.Usage, bool)
	// Close releases the stream. Safe to call multiple times, and required on
	// every path — an abandoned SSE body holds a connection out of the pool for
	// as long as the provider keeps it open.
	Close() error
}

// Provider is one upstream endpoint the gateway can route to.
//
// An instance is identified by Name (a config-chosen instance label such as
// "openai-primary"), not by vendor, because a realistic deployment fronts the
// same vendor several times — different keys, regions, or quota pools — and
// each of those needs its own health state and its own circuit breaker.
type Provider interface {
	// Name is the config-assigned instance name, unique across the gateway.
	Name() string
	// Vendor is the wire protocol family: "openai", "anthropic", "mock".
	Vendor() string
	// Chat performs a non-streaming completion.
	Chat(ctx context.Context, req *apiv1.ChatRequest, model string) (*Result, *Failure)
	// ChatStream performs a streaming completion. The returned Stream is owned
	// by the caller, which must Close it.
	ChatStream(ctx context.Context, req *apiv1.ChatRequest, model string) (Stream, *Failure)
}

// HealthProbe is implemented by providers that expose an active health check.
// Optional: a provider without one is judged solely by the circuit breaker's
// passive observation of real traffic.
type HealthProbe interface {
	Probe(ctx context.Context) error
}

// ErrNoProviders is returned when no route has a healthy provider.
var ErrNoProviders = errors.New("no healthy provider available for model")
