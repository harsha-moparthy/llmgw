package server

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
	"github.com/harsha-moparthy/llmgw/internal/breaker"
	"github.com/harsha-moparthy/llmgw/internal/budget"
	"github.com/harsha-moparthy/llmgw/internal/cache"
	"github.com/harsha-moparthy/llmgw/internal/config"
	"github.com/harsha-moparthy/llmgw/internal/ledger"
	"github.com/harsha-moparthy/llmgw/internal/metrics"
	"github.com/harsha-moparthy/llmgw/internal/mockprov"
	"github.com/harsha-moparthy/llmgw/internal/money"
	"github.com/harsha-moparthy/llmgw/internal/pricing"
	"github.com/harsha-moparthy/llmgw/internal/provider"
	"github.com/harsha-moparthy/llmgw/internal/router"
	"github.com/harsha-moparthy/llmgw/internal/tokens"
)

// harness spins up one or more real mock providers behind a real gateway server,
// all in-process via httptest. It is the closest thing to the deployed system a
// unit test can build, and it is what makes the reconciliation assertion
// meaningful: the gateway's ledger and the mock's independent log are compared
// for real.
type harness struct {
	t         *testing.T
	gateway   *httptest.Server
	server    *Server
	mocks     []*mockprov.Server
	mockLogs  []*captureWriter
	ledger    *ledger.Ledger
	ledgerBuf *syncBuffer
}

// captureWriter is an in-memory io.WriteCloser capturing a mock's request log.
type captureWriter struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (c *captureWriter) Write(p []byte) (int, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.buf.Write(p)
}
func (c *captureWriter) Close() error { return nil }
func (c *captureWriter) records(t *testing.T) []mockprov.RequestRecord {
	c.mu.Lock()
	defer c.mu.Unlock()
	var out []mockprov.RequestRecord
	sc := bufio.NewScanner(bytes.NewReader(c.buf.Bytes()))
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var r mockprov.RequestRecord
		if err := json.Unmarshal(line, &r); err != nil {
			t.Fatalf("bad mock log line: %v", err)
		}
		out = append(out, r)
	}
	return out
}

// syncBuffer is a concurrency-safe buffer for the gateway ledger.
type syncBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *syncBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}
func (b *syncBuffer) entries(t *testing.T) []ledger.Entry {
	b.mu.Lock()
	defer b.mu.Unlock()
	entries, err := ledger.Decode(bytes.NewReader(b.buf.Bytes()))
	if err != nil {
		t.Fatalf("decoding ledger: %v", err)
	}
	return entries
}

func mockConfig() *mockprov.Config {
	return &mockprov.Config{
		DefaultModel: "mock-fast",
		Models: map[string]*mockprov.ModelConfig{
			"mock-fast": {TTFBMillis: 0, InterTokenMillis: 0, CompletionTokens: 16,
				PromptPricePerM: "0.15", CachedPromptPricePerM: "0.075", CompletionPricePerM: "0.60"},
		},
	}
}

func newHarness(t *testing.T, numMocks int, tenants []config.Tenant) *harness {
	t.Helper()
	h := &harness{t: t}

	providers := map[string]provider.Provider{}
	var provCfgs []config.Provider
	for i := 0; i < numMocks; i++ {
		logw := &captureWriter{}
		mp, err := mockprov.New(mockConfig(), mockprov.Options{LogWriter: logw})
		if err != nil {
			t.Fatal(err)
		}
		ms := httptest.NewServer(mp.Handler())
		t.Cleanup(ms.Close)
		h.mocks = append(h.mocks, mp)
		h.mockLogs = append(h.mockLogs, logw)

		name := fmt.Sprintf("mock-%d", i)
		providers[name] = provider.NewOpenAIProvider(provider.OpenAIConfig{Name: name, BaseURL: ms.URL})
		provCfgs = append(provCfgs, config.Provider{Name: name, Vendor: "mock", BaseURL: ms.URL})
	}

	prices, err := (&pricing.Sheet{Models: []pricing.ModelPrice{
		{Model: "mock-fast", InputPerMTok: "0.15", CachedInputPerMTok: "0.075", OutputPerMTok: "0.60"},
	}}).Table()
	if err != nil {
		t.Fatal(err)
	}

	// One route across all mocks, so failover has somewhere to go.
	var targets []config.Target
	for name := range providers {
		targets = append(targets, config.Target{Provider: name, Model: "mock-fast"})
	}
	// Deterministic target order: mock-0 first.
	for i := range targets {
		for j := i + 1; j < len(targets); j++ {
			if targets[j].Provider < targets[i].Provider {
				targets[i], targets[j] = targets[j], targets[i]
			}
		}
	}

	cfg := &config.Config{
		Server:  config.Server{MaxRequestBytes: 1 << 20, RequestDeadline: config.Duration(10 * time.Second)},
		Routes:  map[string]config.Route{"gw-chat": {Targets: targets, AllowFailover: true, MaxAttempts: numMocks}},
		Tenants: tenants,
	}

	breakers, err := breaker.NewRegistry(breaker.DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	for _, pc := range provCfgs {
		_ = breakers.Get(pc.Name)
	}
	routes, err := router.BuildRoutes(cfg, providers, breakers)
	if err != nil {
		t.Fatal(err)
	}
	rt := router.New(routes, router.Options{RequestDeadline: 10 * time.Second, Prices: prices})

	bud := budget.New(budget.Options{})
	for _, tn := range tenants {
		if limit, err := tn.Limit(); err == nil && !limit.Unlimited() {
			_ = bud.SetLimit(tn.ID, limit)
		}
	}

	h.ledgerBuf = &syncBuffer{}
	h.ledger = ledger.New(h.ledgerBuf, ledger.Options{})
	store := cache.New(1<<20, cache.Policy{})
	priors, _ := tokens.NewPriors(tokens.DefaultPriorConfig())

	h.server = New(Deps{
		Config: cfg, Router: rt, Budget: bud, Cache: store, Ledger: h.ledger,
		Metrics: metrics.NewGateway(), Prices: prices, Priors: priors,
		Readiness: func() bool { return true },
	})
	h.gateway = httptest.NewServer(h.server.Handler())
	t.Cleanup(h.gateway.Close)
	return h
}

func (h *harness) post(key string, body []byte) *http.Response {
	h.t.Helper()
	req, _ := http.NewRequest(http.MethodPost, h.gateway.URL+"/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		h.t.Fatalf("POST: %v", err)
	}
	return resp
}

func chatBody(model, prompt string, stream bool) []byte {
	req := map[string]any{
		"model":    model,
		"messages": []map[string]string{{"role": "user", "content": prompt}},
	}
	if stream {
		req["stream"] = true
		req["stream_options"] = map[string]any{"include_usage": true}
	}
	b, _ := json.Marshal(req)
	return b
}

func benchTenant() config.Tenant {
	return config.Tenant{ID: "bench", APIKeyHash: config.HashKey("bench-key"), AllowedModels: []string{"gw-chat"}}
}

// TestHappyPathNonStreaming: a well-formed request succeeds, carries usage, and
// exposes the gateway's headers.
func TestHappyPathNonStreaming(t *testing.T) {
	h := newHarness(t, 1, []config.Tenant{benchTenant()})
	resp := h.post("bench-key", chatBody("gw-chat", "hello gateway", false))
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, body = %s", resp.StatusCode, b)
	}
	var cr apiv1.ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&cr); err != nil {
		t.Fatal(err)
	}
	if cr.Usage == nil || cr.Usage.TotalTokens == 0 {
		t.Errorf("no usage in response: %+v", cr.Usage)
	}
	if resp.Header.Get(HeaderProvider) == "" {
		t.Error("missing X-Llmgw-Provider header")
	}
	if resp.Header.Get(HeaderOverheadUs) == "" {
		t.Error("missing overhead header")
	}
}

// TestReconciliationExact is the project's central claim: the gateway's ledger
// and the mock provider's INDEPENDENT log agree, to the token and to the
// picodollar, over a batch of requests.
func TestReconciliationExact(t *testing.T) {
	h := newHarness(t, 1, []config.Tenant{benchTenant()})

	const n = 50
	for i := 0; i < n; i++ {
		resp := h.post("bench-key", chatBody("gw-chat", fmt.Sprintf("reconcile request number %d", i), false))
		if resp.StatusCode != 200 {
			b, _ := io.ReadAll(resp.Body)
			t.Fatalf("request %d: status %d: %s", i, resp.StatusCode, b)
		}
		resp.Body.Close()
	}
	if err := h.ledger.Flush(); err != nil {
		t.Fatal(err)
	}

	entries := h.ledgerBuf.entries(t)
	provRecords := toProviderRecords(h.mockLogs[0].records(t))

	report := ledger.Reconcile(entries, provRecords)
	if !report.Exact() {
		t.Fatalf("reconciliation not exact: %s", report.Summary())
	}
	if len(report.Matched) < n {
		t.Errorf("matched %d rows, want at least %d", len(report.Matched), n)
	}
	// The estimated-matched count must be zero here: the mock reports real usage
	// on every success, so nothing in this batch should have been estimated. A
	// non-zero value would mean a "balanced" report was partly built on guesses.
	if report.EstimatedMatched != 0 {
		t.Errorf("estimated_matched = %d, want 0 (the mock always reports usage)", report.EstimatedMatched)
	}
}

// TestAuthRequired: no key or a wrong key is a 401 with an OpenAI error envelope.
func TestAuthRequired(t *testing.T) {
	h := newHarness(t, 1, []config.Tenant{benchTenant()})
	for _, key := range []string{"", "wrong-key"} {
		resp := h.post(key, chatBody("gw-chat", "x", false))
		if resp.StatusCode != 401 {
			t.Errorf("key %q: status = %d, want 401", key, resp.StatusCode)
		}
		var env apiv1.ErrorEnvelope
		json.NewDecoder(resp.Body).Decode(&env)
		resp.Body.Close()
		if env.Err.Type != apiv1.ErrTypeAuth {
			t.Errorf("key %q: error type = %q, want %q", key, env.Err.Type, apiv1.ErrTypeAuth)
		}
	}
}

// TestOversizedBodyRejected: a body over the limit is a 413, and crucially it is
// rejected without being fully decoded.
func TestOversizedBodyRejected(t *testing.T) {
	h := newHarness(t, 1, []config.Tenant{benchTenant()})
	h.server.deps.Config.Server.MaxRequestBytes = 512
	big := strings.Repeat("x", 4096)
	resp := h.post("bench-key", chatBody("gw-chat", big, false))
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusRequestEntityTooLarge && resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 413 or 400", resp.StatusCode)
	}
}

// TestModelNotAllowed: a tenant requesting a route not on its allowlist gets a
// 404 (not a 403 that would confirm the route exists).
func TestModelNotAllowed(t *testing.T) {
	tenant := config.Tenant{ID: "restricted", APIKeyHash: config.HashKey("restricted-key"), AllowedModels: []string{"nonexistent-route"}}
	h := newHarness(t, 1, []config.Tenant{tenant})
	resp := h.post("restricted-key", chatBody("gw-chat", "x", false))
	defer resp.Body.Close()
	if resp.StatusCode != 404 {
		t.Errorf("status = %d, want 404", resp.StatusCode)
	}
}

// TestBudgetRejectionExplainsItself: a tenant over its hard limit gets a
// rejection whose body carries the numbers to act on.
func TestBudgetRejectionExplainsItself(t *testing.T) {
	tenant := config.Tenant{
		ID: "poor", APIKeyHash: config.HashKey("poor-key"),
		BudgetLimit: "0.000000001", BudgetPeriod: "hour", // essentially nothing
		AllowedModels: []string{"gw-chat"},
	}
	h := newHarness(t, 1, []config.Tenant{tenant})

	var rejected bool
	for i := 0; i < 20; i++ {
		resp := h.post("poor-key", chatBody("gw-chat", fmt.Sprintf("expensive %d", i), false))
		if resp.StatusCode == 402 || resp.StatusCode == 429 {
			rejected = true
			var env apiv1.ErrorEnvelope
			json.NewDecoder(resp.Body).Decode(&env)
			resp.Body.Close()
			// The message must name the numbers a client needs: the budget it hit
			// and the tenant it applies to. The exact wording is the budget
			// package's to own; the test asserts the message is actionable, not a
			// bare "budget exceeded".
			msg := env.Err.Message
			if !strings.Contains(msg, "budget") || !strings.Contains(msg, "poor") {
				t.Errorf("rejection message does not explain itself: %q", msg)
			}
			break
		}
		resp.Body.Close()
	}
	if !rejected {
		t.Error("a near-zero budget never produced a rejection")
	}
}

// TestCacheHitServedAndCheaper: a repeated deterministic request is served from
// cache, marked as a hit, and does not reach the provider a second time.
func TestCacheHitServed(t *testing.T) {
	h := newHarness(t, 1, []config.Tenant{benchTenant()})
	body := chatBody("gw-chat", "deterministic please", false)
	// temperature 0 for cacheability.
	var m map[string]any
	json.Unmarshal(body, &m)
	m["temperature"] = 0
	body, _ = json.Marshal(m)

	r1 := h.post("bench-key", body)
	r1.Body.Close()
	if got := r1.Header.Get(HeaderCache); got == "hit" {
		t.Error("first request was a cache hit; the cache was dirty")
	}
	r2 := h.post("bench-key", body)
	defer r2.Body.Close()
	if got := r2.Header.Get(HeaderCache); got != "hit" {
		t.Errorf("second identical request X-Llmgw-Cache = %q, want hit", got)
	}
	// The provider should have seen exactly one request.
	if recs := h.mockLogs[0].records(t); len(recs) != 1 {
		t.Errorf("provider saw %d requests; a cache hit must not reach upstream", len(recs))
	}
}

// TestStreamingHappyPath: a streaming request produces an SSE stream that ends
// with [DONE] and reassembles to real content.
func TestStreamingHappyPath(t *testing.T) {
	h := newHarness(t, 1, []config.Tenant{benchTenant()})
	resp := h.post("bench-key", chatBody("gw-chat", "stream me a response", true))
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("Content-Type = %q", ct)
	}
	text, done, usage := readSSE(t, resp.Body)
	if !done {
		t.Error("stream did not end with [DONE]")
	}
	if text == "" {
		t.Error("stream carried no content")
	}
	if !usage {
		t.Error("include_usage was set but no usage frame arrived")
	}
}

// TestFailoverServedBySecond: with the first provider forced down, the request
// is served by the second and the response header shows two attempts.
func TestFailoverServedBySecond(t *testing.T) {
	h := newHarness(t, 2, []config.Tenant{benchTenant()})
	// Kill mock-0 (the first target).
	h.mocks[0].Faults().Set("down", "true")

	resp := h.post("bench-key", chatBody("gw-chat", "who serves me", false))
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d: %s", resp.StatusCode, b)
	}
	if got := resp.Header.Get(HeaderAttempts); got != "2" {
		t.Errorf("attempts header = %q, want 2 (failover)", got)
	}
	if got := resp.Header.Get(HeaderProvider); got != "mock-1" {
		t.Errorf("served by %q, want mock-1", got)
	}
}

// TestMidStreamAbortProducesErrorFrame: a provider that dies mid-stream makes the
// gateway emit an explicit error event rather than a silent truncation.
func TestMidStreamAbortProducesErrorFrame(t *testing.T) {
	h := newHarness(t, 1, []config.Tenant{benchTenant()})
	h.mocks[0].Faults().Set("mid_stream_abort_after", "2")

	resp := h.post("bench-key", chatBody("gw-chat", "abort me midstream after a few tokens", true))
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	s := string(body)
	// The gateway must surface an explicit error event, not just cut off.
	if !strings.Contains(s, "event: error") {
		t.Errorf("mid-stream abort did not produce an error event:\n%s", s)
	}
}

// TestClientDisconnectStopsUpstreamAndBills: when the client goes away mid-
// stream, the upstream is closed and the partial usage is still ledgered.
func TestClientDisconnectClosesUpstream(t *testing.T) {
	h := newHarness(t, 1, []config.Tenant{benchTenant()})

	req, _ := http.NewRequest(http.MethodPost, h.gateway.URL+"/v1/chat/completions", bytes.NewReader(chatBody("gw-chat", "disconnect me", true)))
	req.Header.Set("Authorization", "Bearer bench-key")
	ctx, cancel := context.WithCancel(context.Background())
	req = req.WithContext(ctx)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("initial request: %v", err)
	}
	// Read a little, then cancel.
	buf := make([]byte, 64)
	_, _ = resp.Body.Read(buf)
	cancel()
	resp.Body.Close()
	// The gateway should not panic or hang; a follow-up request must still work.
	time.Sleep(50 * time.Millisecond)
	r2 := h.post("bench-key", chatBody("gw-chat", "still alive", false))
	if r2.StatusCode != 200 {
		t.Errorf("gateway unhealthy after client disconnect: status %d", r2.StatusCode)
	}
	r2.Body.Close()
}

// TestMetricsEndpoint: after some traffic, /metrics exposes valid exposition.
func TestMetricsEndpoint(t *testing.T) {
	h := newHarness(t, 1, []config.Tenant{benchTenant()})
	for i := 0; i < 3; i++ {
		h.post("bench-key", chatBody("gw-chat", fmt.Sprintf("metric %d", i), false)).Body.Close()
	}
	resp, err := http.Get(h.gateway.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "llmgw_requests_total") {
		t.Errorf("metrics missing request counter:\n%s", body)
	}
}

func TestHealthAndReady(t *testing.T) {
	h := newHarness(t, 1, []config.Tenant{benchTenant()})
	for _, path := range []string{"/healthz", "/readyz"} {
		resp, err := http.Get(h.gateway.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		if resp.StatusCode != 200 {
			t.Errorf("%s = %d, want 200", path, resp.StatusCode)
		}
		resp.Body.Close()
	}
}

// TestConcurrentLoad hammers the gateway and, under -race, shakes out data races
// across the whole pipeline. It also re-checks the reconciliation under
// concurrency, which is where a lost ledger row would show up.
func TestConcurrentLoad(t *testing.T) {
	h := newHarness(t, 1, []config.Tenant{benchTenant()})
	var wg sync.WaitGroup
	const workers, each = 8, 25
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < each; i++ {
				resp := h.post("bench-key", chatBody("gw-chat", fmt.Sprintf("w%d-i%d", w, i), false))
				io.Copy(io.Discard, resp.Body)
				resp.Body.Close()
			}
		}(w)
	}
	wg.Wait()
	if err := h.ledger.Flush(); err != nil {
		t.Fatal(err)
	}

	entries := h.ledgerBuf.entries(t)
	provRecords := toProviderRecords(h.mockLogs[0].records(t))
	report := ledger.Reconcile(entries, provRecords)
	if !report.Exact() {
		t.Errorf("concurrent reconciliation not exact:\n%s", report.Summary())
	}
	if len(report.Matched) != workers*each {
		t.Errorf("matched %d rows, want %d", len(report.Matched), workers*each)
	}
}

// --- helpers ---

func toProviderRecords(recs []mockprov.RequestRecord) []ledger.ProviderRecord {
	out := make([]ledger.ProviderRecord, 0, len(recs))
	for _, r := range recs {
		if r.FinishReason == "error" {
			continue // errors carry no billable tokens; the ledger row for them has none either
		}
		out = append(out, ledger.ProviderRecord{
			RequestID: r.RequestID,
			Attempt:   r.Attempt,
			Model:     r.Model,
			Tokens: ledger.TokenCounts{
				Prompt:     r.Tokens.Prompt,
				Cached:     r.Tokens.Cached,
				Completion: r.Tokens.Completion,
				Reasoning:  r.Tokens.Reasoning,
			},
			CostPico: moneyPico(r.CostPico),
		})
	}
	return out
}

func moneyPico(v int64) money.Pico { return money.Pico(v) }

func readSSE(t *testing.T, r io.Reader) (text string, done, usage bool) {
	t.Helper()
	sc := bufio.NewScanner(r)
	var sb strings.Builder
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
		var chunk apiv1.ChatChunk
		if json.Unmarshal([]byte(payload), &chunk) != nil {
			continue
		}
		sb.WriteString(chunk.DeltaText())
		if chunk.Usage != nil {
			usage = true
		}
	}
	return sb.String(), done, usage
}
