package mockprov

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
	"github.com/harsha-moparthy/llmgw/internal/money"
	"github.com/harsha-moparthy/llmgw/internal/sse"
	"github.com/harsha-moparthy/llmgw/internal/tokens"
)

// Server is a running mock provider.
type Server struct {
	cfg    *Config
	faults *FaultStore
	clock  clock

	logMu sync.Mutex
	logW  io.WriteCloser

	// listener is held so Fault "down" can close and reopen it.
	mu       sync.Mutex
	ln       net.Listener
	httpSrv  *http.Server
	adminSrv *http.Server
}

// Options configures a Server beyond its Config.
type Options struct {
	// Now overrides the clock for tests. The response body never depends on it;
	// only the log's timestamps do.
	Now func() time.Time
	// LogWriter overrides the log destination. When nil, LogPath is opened.
	LogWriter io.WriteCloser
}

// New builds a Server. It does not start listening; call Start.
func New(cfg *Config, opt Options) (*Server, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	s := &Server{
		cfg:    cfg,
		faults: NewFaultStore(cfg.Faults),
		clock:  clock{now: opt.Now},
	}
	if opt.LogWriter != nil {
		s.logW = opt.LogWriter
	} else if cfg.LogPath != "" {
		f, err := os.OpenFile(cfg.LogPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
		if err != nil {
			return nil, fmt.Errorf("mockprov: opening log %q: %w", cfg.LogPath, err)
		}
		s.logW = f
	}
	return s, nil
}

// Faults returns the live fault store, so a test or the admin endpoint can
// change behaviour at runtime.
func (s *Server) Faults() *FaultStore { return s.faults }

// Handler returns the main HTTP handler, exposed so tests can mount it on an
// httptest.Server rather than binding a port.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", s.handleChat)
	mux.HandleFunc("/v1/models", s.handleModels)
	return mux
}

// AdminHandler returns the admin HTTP handler.
func (s *Server) AdminHandler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/admin/health", func(w http.ResponseWriter, r *http.Request) {
		// Health reflects the "down" fault: a killed provider must fail its
		// health check, or the gateway's prober would keep it in rotation and
		// the failover demo would not fire.
		if s.faults.Load().Down {
			http.Error(w, "down", http.StatusServiceUnavailable)
			return
		}
		_, _ = io.WriteString(w, "ok")
	})
	mux.HandleFunc("/admin/fault", s.handleFault)
	return mux
}

// Start binds the listeners and serves in the background. It returns once the
// main listener is accepting, so a caller can immediately send traffic without
// racing startup.
func (s *Server) Start() error {
	ln, err := net.Listen("tcp", s.cfg.Listen)
	if err != nil {
		return fmt.Errorf("mockprov: listen %q: %w", s.cfg.Listen, err)
	}
	s.mu.Lock()
	s.ln = ln
	s.httpSrv = &http.Server{Handler: s.Handler()}
	s.mu.Unlock()
	go func() { _ = s.httpSrv.Serve(ln) }()

	if s.cfg.AdminListen != "" {
		aln, err := net.Listen("tcp", s.cfg.AdminListen)
		if err != nil {
			return fmt.Errorf("mockprov: admin listen %q: %w", s.cfg.AdminListen, err)
		}
		s.mu.Lock()
		s.adminSrv = &http.Server{Handler: s.AdminHandler()}
		s.mu.Unlock()
		go func() { _ = s.adminSrv.Serve(aln) }()
	}
	return nil
}

// Addr returns the main listener's address, useful when Listen was ":0".
func (s *Server) Addr() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ln != nil {
		return s.ln.Addr().String()
	}
	return s.cfg.Listen
}

// Shutdown stops both servers and closes the log.
func (s *Server) Shutdown(ctx context.Context) error {
	s.mu.Lock()
	main, admin := s.httpSrv, s.adminSrv
	s.mu.Unlock()
	var err error
	if main != nil {
		err = main.Shutdown(ctx)
	}
	if admin != nil {
		_ = admin.Shutdown(ctx)
	}
	s.logMu.Lock()
	if s.logW != nil {
		_ = s.logW.Close()
		s.logW = nil
	}
	s.logMu.Unlock()
	return err
}

func (s *Server) handleModels(w http.ResponseWriter, _ *http.Request) {
	type model struct {
		ID     string `json:"id"`
		Object string `json:"object"`
	}
	list := make([]model, 0, len(s.cfg.Models))
	for name := range s.cfg.Models {
		list = append(list, model{ID: name, Object: "model"})
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": list})
}

func (s *Server) handleFault(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	// Accept faults either as query params (?down=true) for quick toggling from
	// a benchmark, or as a JSON body for a full replacement.
	if r.URL.RawQuery != "" {
		applied := 0
		for k, vs := range r.URL.Query() {
			if len(vs) == 0 {
				continue
			}
			if err := s.faults.Set(k, vs[0]); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			applied++
		}
		writeJSON(w, http.StatusOK, map[string]any{"applied": applied, "faults": s.faults.Load()})
		return
	}
	var f Faults
	if err := json.NewDecoder(r.Body).Decode(&f); err != nil {
		http.Error(w, "invalid fault body: "+err.Error(), http.StatusBadRequest)
		return
	}
	s.faults.Replace(f)
	writeJSON(w, http.StatusOK, map[string]any{"faults": s.faults.Load()})
}

// handleChat is the core: it applies faults, generates a deterministic reply,
// counts and prices it independently, logs the record, and responds streaming or
// not.
func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	start := s.clock.Now()
	faults := s.faults.Load()

	// A killed provider closes the client connection abruptly rather than
	// answering. http.Server does not expose a clean "reset" so hijacking and
	// closing the raw conn is the faithful way to produce the connection-refused
	// / reset that a real dead provider gives the gateway.
	if faults.Down {
		if hj, ok := w.(http.Hijacker); ok {
			if conn, _, err := hj.Hijack(); err == nil {
				_ = conn.Close()
				return
			}
		}
		http.Error(w, "provider down", http.StatusServiceUnavailable)
		return
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, 8<<20))
	if err != nil {
		writeErr(w, http.StatusBadRequest, apiv1.ErrTypeInvalidRequest, "cannot read body")
		return
	}
	var req apiv1.ChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeErr(w, http.StatusBadRequest, apiv1.ErrTypeInvalidRequest, "invalid JSON: "+err.Error())
		return
	}
	model := req.Model
	mc, ok := s.cfg.Models[model]
	if !ok {
		mc, ok = s.cfg.Models[s.cfg.DefaultModel], true
		model = s.cfg.DefaultModel
	}

	seed := requestSeed(&req, model)
	// The gateway sends a correlation id and attempt number so this log can be
	// reconciled against the gateway's ledger on the same key. When present they
	// win over the seed-derived id; when absent (a direct client) the seed id is
	// used so the log is still self-consistent.
	corr := correlation{
		requestID: r.Header.Get("X-Llmgw-Request-Id"),
		attempt:   atoiDefault(r.Header.Get("X-Llmgw-Attempt"), 1),
	}
	if corr.requestID == "" {
		corr.requestID = requestID(seed)
	}

	// Error faults, evaluated against the per-request deterministic jitter so a
	// given error-rate reproduces the same failing requests across runs.
	if faults.StatusCode != 0 && (faults.ErrorRate == 0 || deterministicJitter(seed) < faults.ErrorRate) {
		s.serveError(w, faults, &req, model, corr, start)
		return
	}
	if faults.StatusCode == 0 && faults.ErrorRate > 0 && deterministicJitter(seed) < faults.ErrorRate {
		faults.StatusCode = http.StatusServiceUnavailable
		s.serveError(w, faults, &req, model, corr, start)
		return
	}

	// Latency: constant plus a seeded tail spike.
	latency := time.Duration(faults.AddedLatencyMillis) * time.Millisecond
	if faults.P99SpikeMillis > 0 && deterministicJitter(seed^0x9e3779b9) < faults.P99SpikeRate {
		latency += time.Duration(faults.P99SpikeMillis) * time.Millisecond
	}

	promptTokens := tokens.Reference.CountRequest(&req)

	// Honour the client's completion cap. A real provider CANNOT exceed
	// max_tokens — it stops and reports finish_reason "length" — so a mock that
	// ignored the cap would be a dishonest instrument: it made the gateway's
	// (correct, deliberately conservative) pre-flight estimate look like an
	// under-estimate in the cost reconciliation, when in fact the mock was
	// generating more tokens than any real provider would have been allowed to.
	// That is exactly backwards from the property the estimator guarantees, and
	// it produced a false "the estimate under-counts" signal in a committed
	// benchmark artifact.
	target := mc.CompletionTokens
	cappedByClient := false
	if cap := req.EffectiveMaxTokens(); cap > 0 && cap < target {
		target = cap
		cappedByClient = true
	}
	replyText := generateReply(seed, target)
	completionTextTokens := tokens.Reference.Count(replyText)

	finish := apiv1.FinishStop
	if cappedByClient {
		// Stopped because the cap was reached, not because the model was done.
		finish = apiv1.FinishLength
	}
	if faults.TruncateAtLength {
		finish = apiv1.FinishLength
	}

	if req.Stream {
		s.serveStream(w, &req, model, mc, replyText, promptTokens, completionTextTokens, finish, faults, latency, seed, corr, start)
		return
	}
	s.serveNonStream(w, &req, model, mc, replyText, promptTokens, completionTextTokens, finish, latency, seed, corr, start)
}

// correlation ties a mock response to the gateway's ledger row.
type correlation struct {
	requestID string
	attempt   int
}

func atoiDefault(s string, def int) int {
	if s == "" {
		return def
	}
	n, err := strconv.Atoi(s)
	if err != nil || n <= 0 {
		return def
	}
	return n
}

func (s *Server) serveError(w http.ResponseWriter, faults Faults, req *apiv1.ChatRequest, model string, corr correlation, start time.Time) {
	if faults.RetryAfterSeconds > 0 {
		w.Header().Set("Retry-After", strconv.Itoa(faults.RetryAfterSeconds))
	}
	body := faults.StatusBody
	if body == "" {
		body = http.StatusText(faults.StatusCode)
	}
	// An error still costs zero tokens and is logged as such, so the
	// reconciliation sees that a rejected request produced no charge.
	s.appendLog(RequestRecord{
		RequestID:      corr.requestID,
		Attempt:        corr.attempt,
		Model:          model,
		Stream:         req.Stream,
		Fault:          fmt.Sprintf("status_%d", faults.StatusCode),
		FinishReason:   "error",
		ReceivedAt:     start.UTC().Format(time.RFC3339Nano),
		DurationMicros: s.clock.Now().Sub(start).Microseconds(),
	})
	writeErr(w, faults.StatusCode, apiv1.ErrTypeUpstream, body)
}

func (s *Server) serveNonStream(w http.ResponseWriter, req *apiv1.ChatRequest, model string, mc *ModelConfig, replyText string, promptTokens, completionTextTokens int, finish string, latency time.Duration, seed uint64, corr correlation, start time.Time) {
	if latency > 0 {
		time.Sleep(latency)
	}
	usage := mc.usage(promptTokens, completionTextTokens)
	cost := mc.cost(usage)

	created := start.Unix()
	resp := apiv1.ChatResponse{
		ID:      "chatcmpl-" + corr.requestID,
		Object:  apiv1.ObjectChatCompletion,
		Created: created,
		Model:   model,
		Choices: []apiv1.Choice{{
			Index:        0,
			Message:      &apiv1.Message{Role: apiv1.RoleAssistant, Content: apiv1.NewTextContent(replyText)},
			FinishReason: &finish,
		}},
		Usage: usage,
	}
	s.logResponse(model, usage, cost, false, finish, corr, start)
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) serveStream(w http.ResponseWriter, req *apiv1.ChatRequest, model string, mc *ModelConfig, replyText string, promptTokens, completionTextTokens int, finish string, faults Faults, latency time.Duration, seed uint64, corr correlation, start time.Time) {
	enc, err := sse.NewEncoder(w)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, apiv1.ErrTypeServer, "streaming unsupported")
		return
	}
	sse.WriteHeaders(w.Header())
	w.WriteHeader(http.StatusOK)

	created := start.Unix()
	id := "chatcmpl-" + corr.requestID

	// TTFB delay is applied before the first frame; inter-token delay between
	// frames. Together they reproduce the two-phase timing a real streaming
	// model has.
	if ttfb := time.Duration(mc.TTFBMillis)*time.Millisecond + latency; ttfb > 0 {
		time.Sleep(ttfb)
	}

	if faults.MalformedSSE {
		// A frame that violates framing: no "data:" prefix. Tests that the
		// gateway's decoder surfaces this rather than forwarding garbage.
		_, _ = io.WriteString(w, "this is not a valid sse frame\n\n")
		enc.Flush()
		return
	}

	// Opening role frame.
	roleFrame, _ := marshalChunk(id, model, created, apiv1.RoleAssistant, "", nil, nil)
	_ = enc.WriteData(roleFrame)

	toks := tokens.Reference.Tokenize(replyText)
	emitted := 0
	for i, tok := range toks {
		if faults.MidStreamAbortAfter > 0 && emitted >= faults.MidStreamAbortAfter {
			// Abort: stop writing and return WITHOUT a [DONE]. The gateway must
			// detect the truncated stream. This is the fault the whole streaming-
			// failover story hinges on, so it is deliberately abrupt.
			s.logResponse(model, mc.usage(promptTokens, emitted), 0, true, "abort", corr, start)
			return
		}
		if faults.MidStreamStallAfter > 0 && emitted >= faults.MidStreamStallAfter {
			// Stall: hang so the gateway's read deadline fires. Bounded so a test
			// does not hang forever.
			time.Sleep(30 * time.Second)
			return
		}
		if faults.MalformedJSON && i == len(toks)/2 {
			_ = enc.WriteData([]byte("{not valid json"))
		}
		frame, _ := marshalChunk(id, model, created, "", tok, nil, nil)
		if err := enc.WriteData(frame); err != nil {
			// Client went away. Log the partial and stop — the provider stops
			// spending when the reader leaves.
			s.logResponse(model, mc.usage(promptTokens, emitted), 0, true, "client_gone", corr, start)
			return
		}
		emitted++
		if mc.InterTokenMillis > 0 {
			time.Sleep(time.Duration(mc.InterTokenMillis) * time.Millisecond)
		}
	}

	// Terminal frame with finish_reason.
	finFrame, _ := marshalChunk(id, model, created, "", "", &finish, nil)
	_ = enc.WriteData(finFrame)

	usage := mc.usage(promptTokens, completionTextTokens)
	cost := mc.cost(usage)

	// Usage frame when the client asked for it, matching OpenAI's include_usage.
	if req.WantsUsage() {
		usageFrame, _ := marshalChunk(id, model, created, "", "", nil, usage)
		_ = enc.WriteData(usageFrame)
	}
	_ = enc.WriteDone()
	s.logResponse(model, usage, cost, true, finish, corr, start)
}

// logResponse appends the mock's independent request record — the provider side
// of the reconciliation.
func (s *Server) logResponse(model string, usage *apiv1.Usage, cost money.Pico, stream bool, finish string, corr correlation, start time.Time) {
	rec := RequestRecord{
		RequestID: corr.requestID,
		Attempt:   corr.attempt,
		Model:     model,
		Tokens: TokenCounts{
			Prompt:     usage.PromptTokens,
			Cached:     usage.CachedPromptTokens(),
			Completion: usage.CompletionTokens,
			Reasoning:  usage.ReasoningTokens(),
		},
		CostPico:       int64(cost),
		Stream:         stream,
		FinishReason:   finish,
		ReceivedAt:     start.UTC().Format(time.RFC3339Nano),
		DurationMicros: s.clock.Now().Sub(start).Microseconds(),
	}
	if finish == "abort" || finish == "client_gone" {
		rec.Fault = finish
	}
	s.appendLog(rec)
}

// appendLog writes one JSONL record. Serialised behind a mutex because many
// requests log concurrently and interleaved writes would corrupt the file the
// reconciliation reads. One write(2) per record, flushed by the OS — a mock
// under benchmark load writes hundreds of thousands of these, and buffering is
// left to the file's own layer rather than risking a lost tail on shutdown.
func (s *Server) appendLog(rec RequestRecord) {
	s.logMu.Lock()
	defer s.logMu.Unlock()
	if s.logW == nil {
		return
	}
	b, err := json.Marshal(rec)
	if err != nil {
		return
	}
	b = append(b, '\n')
	_, _ = s.logW.Write(b)
}

// requestID derives a short stable id from the seed, so a log record can be
// matched to a response and — because the gateway forwards nothing that would
// let it share this id — the reconciliation matches on the id the mock assigns
// and echoes.
func requestID(seed uint64) string {
	return strconv.FormatUint(seed, 16)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, status int, typ, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(apiv1.NewError(typ, "", msg))
}

// applyFault sets one fault field from a string value, used by the admin query-
// param path. Unknown fields are an error rather than a silent no-op, so a typo
// in a benchmark script fails loudly instead of quietly leaving the fault off.
func applyFault(f *Faults, field, value string) error {
	parseBool := func() (bool, error) { return strconv.ParseBool(value) }
	parseInt := func() (int, error) { return strconv.Atoi(value) }
	parseFloat := func() (float64, error) { return strconv.ParseFloat(value, 64) }
	switch field {
	case "down":
		b, err := parseBool()
		if err != nil {
			return fmt.Errorf("fault down: %w", err)
		}
		f.Down = b
	case "status_code":
		n, err := parseInt()
		if err != nil {
			return fmt.Errorf("fault status_code: %w", err)
		}
		f.StatusCode = n
	case "status_body":
		f.StatusBody = value
	case "retry_after_s":
		n, err := parseInt()
		if err != nil {
			return fmt.Errorf("fault retry_after_s: %w", err)
		}
		f.RetryAfterSeconds = n
	case "added_latency_ms":
		n, err := parseInt()
		if err != nil {
			return fmt.Errorf("fault added_latency_ms: %w", err)
		}
		f.AddedLatencyMillis = n
	case "p99_spike_ms":
		n, err := parseInt()
		if err != nil {
			return fmt.Errorf("fault p99_spike_ms: %w", err)
		}
		f.P99SpikeMillis = n
	case "p99_spike_rate":
		v, err := parseFloat()
		if err != nil {
			return fmt.Errorf("fault p99_spike_rate: %w", err)
		}
		f.P99SpikeRate = v
	case "mid_stream_abort_after":
		n, err := parseInt()
		if err != nil {
			return fmt.Errorf("fault mid_stream_abort_after: %w", err)
		}
		f.MidStreamAbortAfter = n
	case "mid_stream_stall_after":
		n, err := parseInt()
		if err != nil {
			return fmt.Errorf("fault mid_stream_stall_after: %w", err)
		}
		f.MidStreamStallAfter = n
	case "malformed_sse":
		b, err := parseBool()
		if err != nil {
			return fmt.Errorf("fault malformed_sse: %w", err)
		}
		f.MalformedSSE = b
	case "malformed_json":
		b, err := parseBool()
		if err != nil {
			return fmt.Errorf("fault malformed_json: %w", err)
		}
		f.MalformedJSON = b
	case "truncate_at_length":
		b, err := parseBool()
		if err != nil {
			return fmt.Errorf("fault truncate_at_length: %w", err)
		}
		f.TruncateAtLength = b
	case "error_rate":
		v, err := parseFloat()
		if err != nil {
			return fmt.Errorf("fault error_rate: %w", err)
		}
		f.ErrorRate = v
	default:
		return fmt.Errorf("unknown fault field %q", field)
	}
	return nil
}
