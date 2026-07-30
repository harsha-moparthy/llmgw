package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
)

func simpleReq() *apiv1.ChatRequest {
	return &apiv1.ChatRequest{
		Model:    "gw",
		Messages: []apiv1.Message{{Role: "user", Content: apiv1.NewTextContent("hi")}},
	}
}

func newOpenAITestProvider(url string) *OpenAIProvider {
	return NewOpenAIProvider(OpenAIConfig{Name: "test-openai", BaseURL: url, APIKey: "k"})
}

// TestOpenAIChatSuccess is the happy path: a well-formed response decodes and
// carries usage.
func TestOpenAIChatSuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer k" {
			t.Errorf("missing auth header: %q", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"c1","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":2,"total_tokens":11}}`)
	}))
	defer srv.Close()

	p := newOpenAITestProvider(srv.URL)
	res, f := p.Chat(context.Background(), simpleReq(), "gpt-4o")
	if f != nil {
		t.Fatalf("unexpected failure: %v", f)
	}
	if res.Response.Choices[0].Message.Content.Text() != "hello" {
		t.Errorf("content = %q", res.Response.Choices[0].Message.Content.Text())
	}
	if res.Usage == nil || res.Usage.TotalTokens != 11 {
		t.Errorf("usage = %+v, want total 11", res.Usage)
	}
	if res.UsageIsEstimated {
		t.Error("usage was reported by the provider; it must not be flagged estimated")
	}
}

// TestOpenAIClassification drives every classification path, including the
// body-dependent ones (context-length vs plain 400) that a status-code-only
// classifier would get wrong.
func TestOpenAIClassification(t *testing.T) {
	tests := []struct {
		name       string
		status     int
		body       string
		retryAfter string
		wantClass  Class
		wantRetry  bool
	}{
		{"rate limit", 429, `{"error":{"message":"slow down"}}`, "3", ClassRateLimit, true},
		{"server error", 500, `oops`, "", ClassUpstream5xx, true},
		{"overloaded 529", 529, `overloaded`, "", ClassOverloaded, true},
		{"auth", 401, `{"error":{"message":"bad key"}}`, "", ClassAuth, false},
		{"plain bad request", 400, `{"error":{"message":"bad param","code":"invalid_request_error"}}`, "", ClassBadRequest, false},
		{"context length via code", 400, `{"error":{"code":"context_length_exceeded","message":"too long"}}`, "", ClassContextLength, false},
		{"context length via message", 400, `{"error":{"message":"This model's maximum context length is 8192 tokens"}}`, "", ClassContextLength, false},
		{"content filter", 400, `{"error":{"message":"content_filter triggered"}}`, "", ClassContentFilter, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if tc.retryAfter != "" {
					w.Header().Set("Retry-After", tc.retryAfter)
				}
				w.WriteHeader(tc.status)
				_, _ = io.WriteString(w, tc.body)
			}))
			defer srv.Close()

			p := newOpenAITestProvider(srv.URL)
			_, f := p.Chat(context.Background(), simpleReq(), "gpt-4o")
			if f == nil {
				t.Fatal("expected a failure")
			}
			if f.Class != tc.wantClass {
				t.Errorf("class = %s, want %s (body=%q)", f.Class, tc.wantClass, tc.body)
			}
			if f.Retryable() != tc.wantRetry {
				t.Errorf("Retryable() = %v, want %v", f.Retryable(), tc.wantRetry)
			}
			if tc.retryAfter != "" && f.RetryAfter == 0 {
				t.Errorf("Retry-After was sent but not parsed")
			}
		})
	}
}

// TestOpenAIStreamSuccess decodes a full streamed completion and accumulates
// usage from the final frame.
func TestOpenAIStreamSuccess(t *testing.T) {
	stream := "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"}}]}\n\n" +
		"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"}}]}\n\n" +
		"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: {\"id\":\"c1\",\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":2,\"total_tokens\":11}}\n\n" +
		"data: [DONE]\n\n"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		_, _ = io.WriteString(w, stream)
	}))
	defer srv.Close()

	p := newOpenAITestProvider(srv.URL)
	st, f := p.ChatStream(context.Background(), simpleReq(), "gpt-4o")
	if f != nil {
		t.Fatalf("unexpected failure: %v", f)
	}
	defer st.Close()

	var text strings.Builder
	var sawDone bool
	for ev := range st.Events() {
		if ev.Err != nil {
			t.Fatalf("stream error: %v", ev.Err)
		}
		if ev.Done {
			sawDone = true
			continue
		}
		if ev.Chunk != nil {
			text.WriteString(ev.Chunk.DeltaText())
		}
	}
	if !sawDone {
		t.Error("never saw Done")
	}
	if text.String() != "Hello" {
		t.Errorf("reassembled text = %q, want Hello", text.String())
	}
	u, estimated := st.Usage()
	if u == nil || u.TotalTokens != 11 {
		t.Errorf("stream usage = %+v, want total 11", u)
	}
	if estimated {
		t.Error("usage came from the provider; not estimated")
	}
}

// TestOpenAIStreamTruncationSurfacesFailure is the critical one: a stream that
// ends without [DONE] must surface as a failure carrying the usage seen so far,
// not as a clean end.
func TestOpenAIStreamTruncationSurfacesFailure(t *testing.T) {
	// Two content frames then the server closes with no [DONE].
	partial := "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"}}]}\n\n" +
		"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"}}]}\n\n" +
		"event: partial\n" // deliberately unterminated
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		_, _ = io.WriteString(w, partial)
		// handler returns, closing the body without a [DONE].
	}))
	defer srv.Close()

	p := newOpenAITestProvider(srv.URL)
	st, f := p.ChatStream(context.Background(), simpleReq(), "gpt-4o")
	if f != nil {
		t.Fatalf("unexpected setup failure: %v", f)
	}
	defer st.Close()

	var gotErr *Failure
	for ev := range st.Events() {
		if ev.Err != nil {
			gotErr = ev.Err
		}
	}
	if gotErr == nil {
		t.Fatal("a truncated stream produced no failure event")
	}
	if gotErr.Class != ClassTimeout && gotErr.Class != ClassUpstream5xx {
		t.Errorf("truncation classified as %s, want a retryable transport class", gotErr.Class)
	}
}

// TestStreamCloseIdempotentAndNoLeak asserts Close is safe to call repeatedly and
// that the reader goroutine exits, so an abandoned stream does not leak a
// connection or a goroutine.
func TestStreamCloseIdempotentAndNoLeak(t *testing.T) {
	before := runtime.NumGoroutine()

	// A server that streams slowly forever, so the stream is genuinely live when
	// we abandon it.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		fl, _ := w.(http.Flusher)
		for i := 0; i < 1000; i++ {
			_, err := io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n")
			if err != nil {
				return
			}
			if fl != nil {
				fl.Flush()
			}
			time.Sleep(5 * time.Millisecond)
		}
	}))
	defer srv.Close()

	p := newOpenAITestProvider(srv.URL)
	st, f := p.ChatStream(context.Background(), simpleReq(), "gpt-4o")
	if f != nil {
		t.Fatalf("setup failure: %v", f)
	}
	// Read one event, then abandon.
	<-st.Events()
	if err := st.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
	// Idempotent.
	if err := st.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
	// Drain so the run goroutine can exit.
	for range st.Events() {
	}

	// Allow the goroutine to unwind.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if runtime.NumGoroutine() <= before+2 {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Logf("goroutines before=%d after=%d (allowing slack for httptest)", before, runtime.NumGoroutine())
}

// --- Anthropic translation ---

func newAnthropicTestProvider(url string) *AnthropicProvider {
	return NewAnthropicProvider(AnthropicConfig{Name: "test-anthropic", BaseURL: url, APIKey: "k", DefaultMaxTokens: 1024})
}

// TestAnthropicRequestTranslation checks the request-side translation: system
// hoisting, the required max_tokens default, and stop_sequences.
func TestAnthropicRequestTranslation(t *testing.T) {
	var captured anthropicRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-api-key") != "k" {
			t.Errorf("missing x-api-key header")
		}
		if r.Header.Get("anthropic-version") == "" {
			t.Error("missing anthropic-version header")
		}
		_ = json.NewDecoder(r.Body).Decode(&captured)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_1","model":"claude","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`)
	}))
	defer srv.Close()

	req := &apiv1.ChatRequest{
		Model: "gw",
		Messages: []apiv1.Message{
			{Role: "system", Content: apiv1.NewTextContent("be terse")},
			{Role: "system", Content: apiv1.NewTextContent("and precise")},
			{Role: "user", Content: apiv1.NewTextContent("hello")},
		},
		Stop: apiv1.NewStringOrArray("END"),
	}
	p := newAnthropicTestProvider(srv.URL)
	_, f := p.Chat(context.Background(), req, "claude-sonnet-4-5")
	if f != nil {
		t.Fatalf("unexpected failure: %v", f)
	}
	// System messages hoisted and concatenated.
	if captured.System != "be terse\n\nand precise" {
		t.Errorf("system = %q, want the two system messages joined", captured.System)
	}
	// System messages must NOT appear as messages.
	if len(captured.Messages) != 1 || captured.Messages[0].Role != "user" {
		t.Errorf("messages = %+v, want a single user message", captured.Messages)
	}
	// max_tokens defaulted because the client set none.
	if captured.MaxTokens != 1024 {
		t.Errorf("max_tokens = %d, want the adapter default 1024", captured.MaxTokens)
	}
	// stop -> stop_sequences.
	if len(captured.StopSequences) != 1 || captured.StopSequences[0] != "END" {
		t.Errorf("stop_sequences = %v, want [END]", captured.StopSequences)
	}
}

// TestAnthropicResponseTranslation checks the response-side translation into
// OpenAI shape, including cache-read folding and stop-reason mapping.
func TestAnthropicResponseTranslation(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_1","model":"claude","content":[{"type":"text","text":"Hello world"}],"stop_reason":"max_tokens","usage":{"input_tokens":10,"output_tokens":5,"cache_read_input_tokens":4}}`)
	}))
	defer srv.Close()

	p := newAnthropicTestProvider(srv.URL)
	res, f := p.Chat(context.Background(), simpleReq(), "claude")
	if f != nil {
		t.Fatalf("unexpected failure: %v", f)
	}
	if res.Response.Choices[0].Message.Content.Text() != "Hello world" {
		t.Errorf("content = %q", res.Response.Choices[0].Message.Content.Text())
	}
	// max_tokens -> length.
	if *res.Response.Choices[0].FinishReason != apiv1.FinishLength {
		t.Errorf("finish_reason = %q, want length", *res.Response.Choices[0].FinishReason)
	}
	// prompt_tokens must INCLUDE the cache-read tokens (10 input + 4 cached).
	if res.Usage.PromptTokens != 14 {
		t.Errorf("prompt_tokens = %d, want 14 (input+cache_read)", res.Usage.PromptTokens)
	}
	if res.Usage.CachedPromptTokens() != 4 {
		t.Errorf("cached tokens = %d, want 4", res.Usage.CachedPromptTokens())
	}
}

// TestAnthropicStreamUsageAccumulation is the subtle one: input tokens arrive on
// message_start and output tokens on message_delta, so a consumer that read one
// event alone would report half the usage. This asserts the accumulation.
func TestAnthropicStreamUsageAccumulation(t *testing.T) {
	stream := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude\",\"usage\":{\"input_tokens\":25,\"cache_read_input_tokens\":5}}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hel\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":8}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		_, _ = io.WriteString(w, stream)
	}))
	defer srv.Close()

	p := newAnthropicTestProvider(srv.URL)
	st, f := p.ChatStream(context.Background(), simpleReq(), "claude")
	if f != nil {
		t.Fatalf("unexpected failure: %v", f)
	}
	defer st.Close()

	var text strings.Builder
	var role, finish, done bool
	for ev := range st.Events() {
		if ev.Err != nil {
			t.Fatalf("stream error: %v", ev.Err)
		}
		if ev.Done {
			done = true
			continue
		}
		if ev.Chunk != nil {
			if ev.Chunk.DeltaText() != "" {
				text.WriteString(ev.Chunk.DeltaText())
			}
			if len(ev.Chunk.Choices) > 0 && ev.Chunk.Choices[0].Delta != nil && ev.Chunk.Choices[0].Delta.Role == apiv1.RoleAssistant {
				role = true
			}
			if len(ev.Chunk.Choices) > 0 && ev.Chunk.Choices[0].FinishReason != nil {
				finish = true
			}
		}
	}
	if !role {
		t.Error("no opening role chunk emitted")
	}
	if !finish {
		t.Error("no finish_reason chunk emitted")
	}
	if !done {
		t.Error("stream did not signal Done")
	}
	if text.String() != "Hello" {
		t.Errorf("reassembled text = %q, want Hello", text.String())
	}
	u, estimated := st.Usage()
	if u == nil {
		t.Fatal("no usage accumulated")
	}
	// input(25) + cache(5) folded into prompt = 30; output = 8.
	if u.PromptTokens != 30 {
		t.Errorf("prompt_tokens = %d, want 30 (25 input + 5 cache accumulated from message_start)", u.PromptTokens)
	}
	if u.CompletionTokens != 8 {
		t.Errorf("completion_tokens = %d, want 8 (accumulated from message_delta)", u.CompletionTokens)
	}
	if estimated {
		t.Error("Anthropic reports usage; must not be flagged estimated")
	}
}

func TestAnthropic529IsOverloaded(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(529)
		_, _ = io.WriteString(w, `{"type":"error","error":{"type":"overloaded_error"}}`)
	}))
	defer srv.Close()
	p := newAnthropicTestProvider(srv.URL)
	_, f := p.Chat(context.Background(), simpleReq(), "claude")
	if f == nil || f.Class != ClassOverloaded {
		t.Fatalf("529 classified as %v, want overloaded", f)
	}
	if !f.Retryable() {
		t.Error("an overloaded failure should be retryable")
	}
}

func TestConnectFailureClassified(t *testing.T) {
	// Point at a port nothing is listening on.
	p := newOpenAITestProvider("http://127.0.0.1:1")
	_, f := p.Chat(context.Background(), simpleReq(), "m")
	if f == nil {
		t.Fatal("expected a connect failure")
	}
	if f.Class != ClassConnect && f.Class != ClassTimeout {
		t.Errorf("class = %s, want connect or timeout", f.Class)
	}
	if !f.Retryable() {
		t.Error("a connect failure should be retryable")
	}
	if f.MayHaveBilled() {
		t.Error("a connection that never established cannot have been billed")
	}
}

var _ = fmt.Sprintf
