package provider

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
)

func TestClassRetryTable(t *testing.T) {
	// The retry decision is a table, and this pins the whole table so a future
	// edit to one class is a deliberate, visible change rather than an accident.
	tests := []struct {
		class        Class
		retryable    bool
		countsHealth bool
	}{
		{ClassUnknown, true, true},
		{ClassConnect, true, true},
		{ClassTimeout, true, true},
		{ClassRateLimit, true, true},
		{ClassUpstream5xx, true, true},
		{ClassOverloaded, true, true},
		{ClassBadRequest, false, false},
		{ClassAuth, false, true}, // not retryable on this provider, but IS a health signal about it
		{ClassContentFilter, false, false},
		{ClassContextLength, false, false},
		{ClassCancelled, false, false},
	}
	for _, tc := range tests {
		t.Run(tc.class.String(), func(t *testing.T) {
			if got := tc.class.Retryable(); got != tc.retryable {
				t.Errorf("%s.Retryable() = %v, want %v", tc.class, got, tc.retryable)
			}
			if got := tc.class.CountsAgainstHealth(); got != tc.countsHealth {
				t.Errorf("%s.CountsAgainstHealth() = %v, want %v", tc.class, got, tc.countsHealth)
			}
		})
	}
}

// TestMidStreamFailureNotRetryable is the honesty property: once bytes have
// reached the client, a failure is not transparently retryable, because
// splicing a second model's output onto the first produces a response no model
// generated.
func TestMidStreamFailureNotRetryable(t *testing.T) {
	f := &Failure{Class: ClassTimeout, BytesStreamed: 0}
	if !f.Retryable() {
		t.Error("a timeout with no bytes streamed should be retryable")
	}
	f.BytesStreamed = 128
	if f.Retryable() {
		t.Error("a failure after bytes reached the client must NOT be retryable")
	}
}

func TestMayHaveBilled(t *testing.T) {
	tests := []struct {
		name string
		f    *Failure
		want bool
	}{
		{"connect refused was never billed", &Failure{Class: ClassConnect}, false},
		{"rate limit rejected before generation", &Failure{Class: ClassRateLimit}, false},
		{"auth rejected before generation", &Failure{Class: ClassAuth}, false},
		{"bad request rejected before generation", &Failure{Class: ClassBadRequest}, false},
		{"timeout may have generated tokens", &Failure{Class: ClassTimeout}, true},
		{"cancelled mid-generation", &Failure{Class: ClassCancelled}, true},
		{"usage present means billed", &Failure{Class: ClassConnect, UsageAtFailure: &usageStub}, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.f.MayHaveBilled(); got != tc.want {
				t.Errorf("MayHaveBilled() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestClassifyStatus(t *testing.T) {
	tests := map[int]Class{
		429: ClassRateLimit,
		401: ClassAuth,
		403: ClassAuth,
		400: ClassBadRequest,
		404: ClassBadRequest,
		422: ClassBadRequest,
		529: ClassOverloaded,
		500: ClassUpstream5xx,
		503: ClassUpstream5xx,
	}
	for code, want := range tests {
		if got := ClassifyStatus(code); got != want {
			t.Errorf("ClassifyStatus(%d) = %s, want %s", code, got, want)
		}
	}
}

// TestParseRetryAfter covers both the seconds form and the HTTP-date form, since
// real providers use both and a gateway that understood only one would back off
// wrongly against the other.
func TestParseRetryAfter(t *testing.T) {
	fixed := time.Date(2026, 7, 30, 12, 0, 0, 0, time.UTC)
	p := &OpenAIProvider{now: func() time.Time { return fixed }}

	if got := p.parseRetryAfter("5"); got != 5*time.Second {
		t.Errorf("seconds form: got %v, want 5s", got)
	}
	if got := p.parseRetryAfter(""); got != 0 {
		t.Errorf("empty: got %v, want 0", got)
	}
	if got := p.parseRetryAfter("-3"); got != 0 {
		t.Errorf("negative seconds: got %v, want 0", got)
	}
	// HTTP-date 30 seconds in the future.
	future := fixed.Add(30 * time.Second).UTC().Format(http.TimeFormat)
	if got := p.parseRetryAfter(future); got < 29*time.Second || got > 31*time.Second {
		t.Errorf("http-date form: got %v, want ~30s", got)
	}
	// A date in the past clamps to zero rather than going negative.
	past := fixed.Add(-time.Hour).UTC().Format(http.TimeFormat)
	if got := p.parseRetryAfter(past); got != 0 {
		t.Errorf("past http-date: got %v, want 0", got)
	}
}

var usageStub = apiv1.Usage{PromptTokens: 5, CompletionTokens: 3, TotalTokens: 8}

func TestTransportDefaults(t *testing.T) {
	c := Transport{}.Client()
	if c.Timeout != 0 {
		t.Errorf("client Timeout = %v, want 0 (streaming must not be killed by a whole-exchange timeout)", c.Timeout)
	}
	tr, ok := c.Transport.(*http.Transport)
	if !ok {
		t.Fatal("transport is not *http.Transport")
	}
	if tr.MaxIdleConnsPerHost < 64 {
		t.Errorf("MaxIdleConnsPerHost = %d; a gateway needs far more than the stdlib default of 2", tr.MaxIdleConnsPerHost)
	}
	if tr.ResponseHeaderTimeout == 0 {
		t.Error("ResponseHeaderTimeout is 0; a hung provider would never be detected")
	}
}

func TestContextCancelClassification(t *testing.T) {
	p := NewOpenAIProvider(OpenAIConfig{Name: "x", BaseURL: "http://127.0.0.1:1"})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	f := p.transportFailure(ctx, "m", context.Canceled)
	if f.Class != ClassCancelled {
		t.Errorf("cancelled context classified as %s, want cancelled", f.Class)
	}
}
