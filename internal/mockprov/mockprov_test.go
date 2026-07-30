package mockprov

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
	"github.com/harsha-moparthy/llmgw/internal/tokens"
)

// testConfig returns a small valid config with one fast model and one reasoning
// model.
func testConfig() *Config {
	return &Config{
		Listen:       "127.0.0.1:0",
		DefaultModel: "fast",
		Models: map[string]*ModelConfig{
			"fast": {
				TTFBMillis: 0, InterTokenMillis: 0, CompletionTokens: 16,
				PromptPricePerM: "0.15", CachedPromptPricePerM: "0.075", CompletionPricePerM: "0.60",
			},
			"thinker": {
				TTFBMillis: 0, InterTokenMillis: 0, CompletionTokens: 12, ReasoningTokens: 40, Reasoning: true,
				PromptPricePerM: "1.10", CachedPromptPricePerM: "0.55", CompletionPricePerM: "4.40",
			},
			"cachey": {
				TTFBMillis: 0, InterTokenMillis: 0, CompletionTokens: 10, CachedPromptFraction: 0.5,
				PromptPricePerM: "0.15", CachedPromptPricePerM: "0.0375", CompletionPricePerM: "0.60",
			},
		},
	}
}

// captureLog is an in-memory io.WriteCloser so tests can inspect the request log
// without touching the filesystem.
type captureLog struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (c *captureLog) Write(p []byte) (int, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.buf.Write(p)
}
func (c *captureLog) Close() error { return nil }
func (c *captureLog) records(t *testing.T) []RequestRecord {
	t.Helper()
	c.mu.Lock()
	defer c.mu.Unlock()
	var out []RequestRecord
	sc := bufio.NewScanner(bytes.NewReader(c.buf.Bytes()))
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var r RequestRecord
		if err := json.Unmarshal(line, &r); err != nil {
			t.Fatalf("bad log line %q: %v", line, err)
		}
		out = append(out, r)
	}
	return out
}

func newTestServer(t *testing.T, cfg *Config) (*Server, *captureLog, *httptest.Server) {
	t.Helper()
	log := &captureLog{}
	srv, err := New(cfg, Options{LogWriter: log, Now: func() time.Time { return time.Unix(1700000000, 0) }})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	hs := httptest.NewServer(srv.Handler())
	t.Cleanup(hs.Close)
	return srv, log, hs
}

func chatBody(model, prompt string, stream bool) []byte {
	req := map[string]any{
		"model": model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
	}
	if stream {
		req["stream"] = true
		req["stream_options"] = map[string]any{"include_usage": true}
	}
	b, _ := json.Marshal(req)
	return b
}

func post(t *testing.T, url string, body []byte) *http.Response {
	t.Helper()
	resp, err := http.Post(url+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	return resp
}

// TestDeterminism is the property the whole measurement rig depends on: the same
// request produces byte-identical output every time.
func TestDeterminism(t *testing.T) {
	_, _, hs := newTestServer(t, testConfig())

	body := chatBody("fast", "what is the gateway", false)
	var first []byte
	for i := 0; i < 5; i++ {
		resp := post(t, hs.URL, body)
		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if i == 0 {
			first = b
			continue
		}
		if !bytes.Equal(first, b) {
			t.Fatalf("response %d differs from the first:\n  %s\n  %s", i, first, b)
		}
	}

	// A different prompt must (almost certainly) produce different output.
	other := post(t, hs.URL, chatBody("fast", "a completely different prompt", false))
	ob, _ := io.ReadAll(other.Body)
	other.Body.Close()
	if bytes.Equal(first, ob) {
		t.Error("two different prompts produced identical output")
	}
}

// TestTokenCountsMatchReferenceIndependently checks the mock's reported usage
// against tokens.Reference computed here in the test. This is the ground truth
// the reconciliation trusts, so it must be verifiable without trusting the mock.
func TestTokenCountsMatchReference(t *testing.T) {
	_, _, hs := newTestServer(t, testConfig())
	prompt := "count these tokens exactly please"
	resp := post(t, hs.URL, chatBody("fast", prompt, false))
	var cr apiv1.ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&cr); err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	// Independently recompute the prompt token count from the request the mock
	// received.
	req := apiv1.ChatRequest{Model: "fast", Messages: []apiv1.Message{{Role: "user", Content: apiv1.NewTextContent(prompt)}}}
	wantPrompt := tokens.Reference.CountRequest(&req)
	if cr.Usage.PromptTokens != wantPrompt {
		t.Errorf("prompt_tokens = %d, reference says %d", cr.Usage.PromptTokens, wantPrompt)
	}
	// The completion text must tokenise to exactly the reported completion count.
	replyTokens := tokens.Reference.Count(cr.Choices[0].Message.Content.Text())
	if cr.Usage.CompletionTokens != replyTokens {
		t.Errorf("completion_tokens = %d, but the reply text tokenises to %d", cr.Usage.CompletionTokens, replyTokens)
	}
	if cr.Usage.TotalTokens != cr.Usage.PromptTokens+cr.Usage.CompletionTokens {
		t.Errorf("total_tokens is not prompt+completion")
	}
}

// TestReasoningTokensIncludedNotAdded verifies the reasoning-model accounting:
// reasoning tokens are inside completion_tokens, not on top. Getting this wrong
// in the instrument would make the gateway's double-billing bug invisible.
func TestReasoningTokensIncludedNotAdded(t *testing.T) {
	_, _, hs := newTestServer(t, testConfig())
	resp := post(t, hs.URL, chatBody("thinker", "solve this", false))
	var cr apiv1.ChatResponse
	json.NewDecoder(resp.Body).Decode(&cr)
	resp.Body.Close()

	replyTokens := tokens.Reference.Count(cr.Choices[0].Message.Content.Text())
	reasoning := cr.Usage.ReasoningTokens()
	if reasoning != 40 {
		t.Errorf("reasoning_tokens = %d, want 40", reasoning)
	}
	// completion_tokens must equal visible text + reasoning, i.e. reasoning is
	// included.
	if cr.Usage.CompletionTokens != replyTokens+reasoning {
		t.Errorf("completion_tokens = %d, want text(%d)+reasoning(%d)=%d",
			cr.Usage.CompletionTokens, replyTokens, reasoning, replyTokens+reasoning)
	}
}

// TestCachedTokensIncludedInPrompt verifies prompt_tokens includes cached_tokens.
func TestCachedTokensIncludedInPrompt(t *testing.T) {
	_, _, hs := newTestServer(t, testConfig())
	resp := post(t, hs.URL, chatBody("cachey", "a prompt that is partly cached and reasonably long", false))
	var cr apiv1.ChatResponse
	json.NewDecoder(resp.Body).Decode(&cr)
	resp.Body.Close()

	cached := cr.Usage.CachedPromptTokens()
	if cached <= 0 {
		t.Fatalf("expected some cached tokens, got %d", cached)
	}
	if cached > cr.Usage.PromptTokens {
		t.Errorf("cached_tokens %d exceeds prompt_tokens %d", cached, cr.Usage.PromptTokens)
	}
}

// TestIndependentLogMatchesResponse checks the request log records the same
// counts the response reports — the mock is internally consistent, which is a
// precondition for it being a trustworthy reconciliation source.
func TestIndependentLogMatchesResponse(t *testing.T) {
	_, log, hs := newTestServer(t, testConfig())
	resp := post(t, hs.URL, chatBody("fast", "log consistency check", false))
	var cr apiv1.ChatResponse
	json.NewDecoder(resp.Body).Decode(&cr)
	resp.Body.Close()

	recs := log.records(t)
	if len(recs) != 1 {
		t.Fatalf("expected 1 log record, got %d", len(recs))
	}
	r := recs[0]
	if r.Tokens.Prompt != cr.Usage.PromptTokens || r.Tokens.Completion != cr.Usage.CompletionTokens {
		t.Errorf("log tokens (%d/%d) disagree with response (%d/%d)",
			r.Tokens.Prompt, r.Tokens.Completion, cr.Usage.PromptTokens, cr.Usage.CompletionTokens)
	}
	if r.CostPico <= 0 {
		t.Errorf("log cost = %d, want positive", r.CostPico)
	}
	// Recompute the cost from the mock's own rates and check the log matches.
	wantCost := testConfig().Models["fast"].cost(cr.Usage)
	// resolve() must run for the rates to be populated.
	mc := testConfig().Models["fast"]
	if err := mc.resolve("fast"); err != nil {
		t.Fatal(err)
	}
	wantCost = mc.cost(cr.Usage)
	if int64(wantCost) != r.CostPico {
		t.Errorf("log cost %d, independent recompute %d", r.CostPico, int64(wantCost))
	}
}

func TestStreamingShape(t *testing.T) {
	_, _, hs := newTestServer(t, testConfig())
	resp := post(t, hs.URL, chatBody("fast", "stream this response please", true))
	defer resp.Body.Close()

	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("Content-Type = %q, want text/event-stream", ct)
	}

	frames, sawRole, sawFinish, sawUsage, sawDone := parseStream(t, resp.Body)
	if !sawRole {
		t.Error("first content frame did not carry the assistant role")
	}
	if !sawFinish {
		t.Error("no finish_reason frame")
	}
	if !sawUsage {
		t.Error("include_usage was set but no usage frame was emitted")
	}
	if !sawDone {
		t.Error("stream did not end with [DONE]")
	}
	if frames < 3 {
		t.Errorf("only %d frames; expected role + deltas + finish", frames)
	}
}

// parseStream reads an SSE completion stream and reports what it saw.
func parseStream(t *testing.T, body io.Reader) (frames int, role, finish, usage, done bool) {
	t.Helper()
	sc := bufio.NewScanner(body)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			done = true
			continue
		}
		frames++
		var chunk apiv1.ChatChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			t.Fatalf("bad chunk JSON %q: %v", payload, err)
		}
		if len(chunk.Choices) > 0 {
			if chunk.Choices[0].Delta != nil && chunk.Choices[0].Delta.Role == apiv1.RoleAssistant {
				role = true
			}
			if chunk.Choices[0].FinishReason != nil {
				finish = true
			}
		}
		if chunk.Usage != nil {
			usage = true
		}
	}
	return
}

// TestStreamReassemblesToNonStream checks the streamed deltas concatenate to the
// same text the non-streaming path returns, so a client gets the same answer
// either way — which is also why the cache can serve one form from the other.
func TestStreamReassemblesToNonStream(t *testing.T) {
	_, _, hs := newTestServer(t, testConfig())

	ns := post(t, hs.URL, chatBody("fast", "identical prompt for both paths", false))
	var cr apiv1.ChatResponse
	json.NewDecoder(ns.Body).Decode(&cr)
	ns.Body.Close()
	nonStreamText := cr.Choices[0].Message.Content.Text()

	sresp := post(t, hs.URL, chatBody("fast", "identical prompt for both paths", true))
	defer sresp.Body.Close()
	var sb strings.Builder
	sc := bufio.NewScanner(sresp.Body)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			break
		}
		var chunk apiv1.ChatChunk
		if json.Unmarshal([]byte(payload), &chunk) == nil {
			sb.WriteString(chunk.DeltaText())
		}
	}
	if sb.String() != nonStreamText {
		t.Errorf("streamed text != non-streamed text:\n  stream:    %q\n  nonstream: %q", sb.String(), nonStreamText)
	}
}

func TestFaultErrorStatus(t *testing.T) {
	srv, _, hs := newTestServer(t, testConfig())
	if err := srv.Faults().Set("status_code", "429"); err != nil {
		t.Fatal(err)
	}
	if err := srv.Faults().Set("retry_after_s", "7"); err != nil {
		t.Fatal(err)
	}
	resp := post(t, hs.URL, chatBody("fast", "x", false))
	defer resp.Body.Close()
	if resp.StatusCode != 429 {
		t.Errorf("status = %d, want 429", resp.StatusCode)
	}
	if ra := resp.Header.Get("Retry-After"); ra != "7" {
		t.Errorf("Retry-After = %q, want 7", ra)
	}
	// The error body must be an OpenAI error envelope.
	var env apiv1.ErrorEnvelope
	if err := json.NewDecoder(resp.Body).Decode(&env); err != nil {
		t.Fatalf("error body is not an OpenAI envelope: %v", err)
	}
	if env.Err.Type == "" {
		t.Error("error envelope has no type")
	}
}

// TestMidStreamAbort is the most important fault: the mock emits some frames then
// closes without [DONE], and the test asserts exactly that. The gateway's job is
// to detect this; this test only proves the instrument produces it.
func TestMidStreamAbort(t *testing.T) {
	cfg := testConfig()
	cfg.Models["fast"].CompletionTokens = 20
	srv, _, hs := newTestServer(t, cfg)
	if err := srv.Faults().Set("mid_stream_abort_after", "3"); err != nil {
		t.Fatal(err)
	}
	resp := post(t, hs.URL, chatBody("fast", "abort midway", true))
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	s := string(body)
	if strings.Contains(s, "[DONE]") {
		t.Error("aborted stream should NOT contain [DONE]")
	}
	// It should have emitted a bounded number of content frames before dying.
	dataFrames := strings.Count(s, "data: ")
	if dataFrames == 0 {
		t.Error("aborted stream emitted no frames at all")
	}
	if dataFrames > 6 { // role + ~3 deltas, generously bounded
		t.Errorf("aborted after 3 but emitted %d frames", dataFrames)
	}
}

func TestFaultDownClosesConnection(t *testing.T) {
	srv, _, hs := newTestServer(t, testConfig())
	srv.Faults().Set("down", "true")
	// A down provider drops the connection; the client sees a transport error or
	// a non-200. Either way it must not get a normal completion.
	resp, err := http.Post(hs.URL+"/v1/chat/completions", "application/json", bytes.NewReader(chatBody("fast", "x", false)))
	if err != nil {
		return // connection reset — the expected outcome
	}
	defer resp.Body.Close()
	if resp.StatusCode == 200 {
		t.Error("a down provider returned a 200")
	}
}

func TestAdminUnknownFaultRejected(t *testing.T) {
	srv, _, _ := newTestServer(t, testConfig())
	if err := srv.Faults().Set("teleport", "true"); err == nil {
		t.Error("setting an unknown fault field should error, not silently no-op")
	}
}

func TestConfigValidation(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*Config)
		wantErr bool
	}{
		{name: "valid", mutate: func(*Config) {}},
		{name: "no models", mutate: func(c *Config) { c.Models = nil }, wantErr: true},
		{name: "default not in models", mutate: func(c *Config) { c.DefaultModel = "ghost" }, wantErr: true},
		{name: "inexact price", mutate: func(c *Config) { c.Models["fast"].PromptPricePerM = "0.1234567" }, wantErr: true},
		{name: "cached fraction out of range", mutate: func(c *Config) { c.Models["fast"].CachedPromptFraction = 2 }, wantErr: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := testConfig()
			tc.mutate(cfg)
			err := cfg.Validate()
			if (err != nil) != tc.wantErr {
				t.Errorf("Validate err = %v, wantErr = %v", err, tc.wantErr)
			}
		})
	}
}

// TestStartAndShutdown exercises the real listener path and graceful shutdown.
func TestStartAndShutdown(t *testing.T) {
	cfg := testConfig()
	cfg.AdminListen = "127.0.0.1:0"
	srv, err := New(cfg, Options{LogWriter: &captureLog{}})
	if err != nil {
		t.Fatal(err)
	}
	if err := srv.Start(); err != nil {
		t.Fatal(err)
	}
	addr := srv.Addr()
	resp, err := http.Post("http://"+addr+"/v1/chat/completions", "application/json", bytes.NewReader(chatBody("fast", "x", false)))
	if err != nil {
		t.Fatalf("request to started server: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Errorf("status = %d, want 200", resp.StatusCode)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		t.Errorf("Shutdown: %v", err)
	}
}

// TestMockHonoursMaxTokens pins a bug found by audit: the mock ignored the
// client's max_tokens and generated its configured length regardless. A real
// provider cannot exceed the cap, so the mock was emitting more completion
// tokens than any real upstream would — which made the gateway's deliberately
// conservative pre-flight estimate look like an UNDER-estimate in the cost
// reconciliation. The instrument was wrong, not the estimator.
func TestMockHonoursMaxTokens(t *testing.T) {
	cfg := testConfig()
	cfg.Models["fast"].CompletionTokens = 48 // more than the cap below
	_, _, hs := newTestServer(t, cfg)

	body, _ := json.Marshal(map[string]any{
		"model":      "fast",
		"messages":   []map[string]string{{"role": "user", "content": "cap me"}},
		"max_tokens": 32,
	})
	resp := post(t, hs.URL, body)
	defer resp.Body.Close()
	var cr apiv1.ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&cr); err != nil {
		t.Fatal(err)
	}
	if cr.Usage.CompletionTokens > 32 {
		t.Errorf("completion_tokens = %d, exceeds the client's max_tokens of 32; "+
			"no real provider may do this", cr.Usage.CompletionTokens)
	}
	// Stopping at the cap is finish_reason "length", not "stop".
	if got := *cr.Choices[0].FinishReason; got != apiv1.FinishLength {
		t.Errorf("finish_reason = %q, want %q when the cap truncated the response",
			got, apiv1.FinishLength)
	}

	// And with a cap ABOVE the configured length, the model finishes naturally.
	body2, _ := json.Marshal(map[string]any{
		"model":      "fast",
		"messages":   []map[string]string{{"role": "user", "content": "room to finish"}},
		"max_tokens": 4096,
	})
	r2 := post(t, hs.URL, body2)
	defer r2.Body.Close()
	var cr2 apiv1.ChatResponse
	if err := json.NewDecoder(r2.Body).Decode(&cr2); err != nil {
		t.Fatal(err)
	}
	if got := *cr2.Choices[0].FinishReason; got != apiv1.FinishStop {
		t.Errorf("finish_reason = %q, want %q when the cap was not reached", got, apiv1.FinishStop)
	}
}
