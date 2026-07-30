// Command gwbench measures the gateway's own added latency, its failover
// window, and its cost reconciliation. It is the instrument behind every
// quantitative claim in the README, so its methodology is documented here in
// more detail than the code strictly needs.
//
// # Why not just use k6
//
// k6 measures end-to-end latency from outside the gateway. Against a mock
// provider that simulates realistic generation (hundreds of milliseconds to
// seconds), an end-to-end p99 is dominated by the simulated provider, and
// quoting it as the gateway's overhead would be off by three orders of
// magnitude. The spec asks for "added latency under 5ms p99 EXCLUDING provider
// time", and excluding provider time requires measuring inside the gateway.
//
// So overhead is measured two ways and both are reported:
//
//	self-reported: the gateway times its own work (everything except the
//	  upstream round trip) and returns it in the X-LLMGW-Overhead-Us header.
//	  Precise, but it is the gateway grading its own homework: it cannot see
//	  time spent in the kernel, in net/http's own read/write path, or in Go's
//	  scheduler before the handler was entered.
//
//	subtractive: (client-observed total) - (gateway-reported upstream time).
//	  This catches everything the self-report misses, because it starts from a
//	  clock outside the process. Its weakness is the opposite one: it includes
//	  loopback network time and the client's own overhead, so it is an upper
//	  bound rather than an exact figure.
//
// The two bracket the truth. Reporting only the flattering one (self-reported)
// would be the easy dishonesty here, so both appear in the output and in the
// README, and the gap between them is itself reported.
//
// # Why exact quantiles
//
// Prometheus histogram quantiles are interpolated within a bucket, so a "p99 of
// 4.8ms" from a histogram whose bucket boundaries are 1ms and 5ms carries no
// information beyond "between 1 and 5". Every latency figure gwbench reports
// comes from a full sorted sample of every observation, so a p99 is a genuine
// order statistic. Memory cost is 8 bytes per request, which for a benchmark is
// free.
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/ledger"
)

func main() {
	var (
		mode      = flag.String("mode", "overhead", "which measurement to run: overhead, failover, reconcile, cache, stream, all")
		gateway   = flag.String("gateway", "http://127.0.0.1:8080", "gateway base URL")
		adminURL  = flag.String("admin", "http://127.0.0.1:9090", "mock provider admin base URL (failover mode)")
		model     = flag.String("model", "gw-chat", "client-facing model alias to request")
		apiKey    = flag.String("key", "bench-key", "tenant API key")
		n         = flag.Int("n", 2000, "number of requests")
		conc      = flag.Int("c", 32, "concurrency")
		maxTok    = flag.Int("max-tokens", 32, "max_tokens per request")
		outDir    = flag.String("out", "", "directory to write the JSON report into (default: print only)")
		warmup    = flag.Int("warmup", 200, "warmup requests excluded from the statistics")
		ledgerP   = flag.String("ledger", "data/ledger.jsonl", "gateway ledger path (reconcile mode)")
		provLogP  = flag.String("provider-log", "data/requests.jsonl", "mock provider request log path (reconcile mode)")
		killAfter = flag.Duration("kill-after", 3*time.Second, "when to kill the primary provider (failover mode)")
		runFor    = flag.Duration("duration", 12*time.Second, "how long to hold load (failover mode)")
		label     = flag.String("label", "", "label recorded in the report")
	)
	flag.Parse()

	cfg := benchConfig{
		Gateway: *gateway, Admin: *adminURL, Model: *model, APIKey: *apiKey,
		N: *n, Concurrency: *conc, MaxTokens: *maxTok, Warmup: *warmup,
		LedgerPath: *ledgerP, ProviderLogPath: *provLogP,
		KillAfter: *killAfter, Duration: *runFor, Label: *label,
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	report := Report{
		Mode:      *mode,
		Label:     *label,
		StartedAt: time.Now().UTC().Format(time.RFC3339),
		Host:      hostFacts(),
		Config:    cfg,
	}

	var err error
	switch *mode {
	case "overhead":
		report.Overhead, err = runOverhead(ctx, cfg)
	case "failover":
		report.Failover, err = runFailover(ctx, cfg)
	case "reconcile":
		report.Reconcile, err = runReconcile(cfg)
	case "cache":
		report.Cache, err = runCache(ctx, cfg)
	case "stream":
		report.Stream, err = runStream(ctx, cfg)
	case "all":
		report.Overhead, err = runOverhead(ctx, cfg)
		if err == nil {
			report.Stream, err = runStream(ctx, cfg)
		}
		if err == nil {
			report.Cache, err = runCache(ctx, cfg)
		}
		if err == nil {
			report.Reconcile, err = runReconcile(cfg)
		}
	default:
		err = fmt.Errorf("unknown mode %q", *mode)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "gwbench: %v\n", err)
		os.Exit(1)
	}
	report.FinishedAt = time.Now().UTC().Format(time.RFC3339)

	printReport(&report)

	if *outDir != "" {
		if err := writeReport(*outDir, &report); err != nil {
			fmt.Fprintf(os.Stderr, "gwbench: writing report: %v\n", err)
			os.Exit(1)
		}
	}
}

// ---------------------------------------------------------------------------
// report types
// ---------------------------------------------------------------------------

type benchConfig struct {
	Gateway         string        `json:"gateway"`
	Admin           string        `json:"admin,omitempty"`
	Model           string        `json:"model"`
	APIKey          string        `json:"-"` // never serialise a credential into a committed report
	N               int           `json:"n"`
	Concurrency     int           `json:"concurrency"`
	MaxTokens       int           `json:"max_tokens"`
	Warmup          int           `json:"warmup"`
	LedgerPath      string        `json:"ledger_path,omitempty"`
	ProviderLogPath string        `json:"provider_log_path,omitempty"`
	KillAfter       time.Duration `json:"kill_after,omitempty"`
	Duration        time.Duration `json:"duration,omitempty"`
	Label           string        `json:"label,omitempty"`
}

// Report is the committed artifact. Everything the README claims is
// reproducible from one of these files.
type Report struct {
	Mode       string           `json:"mode"`
	Label      string           `json:"label,omitempty"`
	StartedAt  string           `json:"started_at"`
	FinishedAt string           `json:"finished_at"`
	Host       HostFacts        `json:"host"`
	Config     benchConfig      `json:"config"`
	Overhead   *OverheadResult  `json:"overhead,omitempty"`
	Failover   *FailoverResult  `json:"failover,omitempty"`
	Reconcile  *ReconcileResult `json:"reconcile,omitempty"`
	Cache      *CacheResult     `json:"cache,omitempty"`
	Stream     *StreamResult    `json:"stream,omitempty"`
}

// HostFacts records what the numbers were measured on. A latency claim without
// the hardware it was measured on is not a claim.
type HostFacts struct {
	OS      string `json:"os"`
	Arch    string `json:"arch"`
	CPU     string `json:"cpu,omitempty"`
	Cores   int    `json:"cores,omitempty"`
	MemGB   int    `json:"mem_gb,omitempty"`
	GoVer   string `json:"go_version,omitempty"`
	Machine string `json:"machine,omitempty"`
}

// Stats is an exact quantile summary over a full sample.
type Stats struct {
	Count int64   `json:"count"`
	Min   float64 `json:"min"`
	Mean  float64 `json:"mean"`
	P50   float64 `json:"p50"`
	P90   float64 `json:"p90"`
	P95   float64 `json:"p95"`
	P99   float64 `json:"p99"`
	P999  float64 `json:"p999"`
	Max   float64 `json:"max"`
	Unit  string  `json:"unit"`
}

// OverheadResult holds the headline latency measurement.
type OverheadResult struct {
	Requests      int64   `json:"requests"`
	Errors        int64   `json:"errors"`
	ErrorRate     float64 `json:"error_rate"`
	Throughput    float64 `json:"throughput_rps"`
	WallSeconds   float64 `json:"wall_seconds"`
	SelfReported  Stats   `json:"self_reported_overhead"`
	Subtractive   Stats   `json:"subtractive_overhead"`
	EndToEnd      Stats   `json:"end_to_end"`
	UpstreamTime  Stats   `json:"upstream_time"`
	MethodNote    string  `json:"method_note"`
	BracketNoteP9 string  `json:"bracket_note_p99"`
}

// FailoverResult holds the failover-window measurement.
type FailoverResult struct {
	TotalRequests    int64   `json:"total_requests"`
	Succeeded        int64   `json:"succeeded"`
	Failed           int64   `json:"failed"`
	FailedDuringKill int64   `json:"failed_during_kill_window"`
	KilledAt         string  `json:"killed_at"`
	FirstFailureAt   string  `json:"first_failure_at,omitempty"`
	RecoveredAt      string  `json:"recovered_at,omitempty"`
	DetectionMs      float64 `json:"detection_ms"`
	RecoveryMs       float64 `json:"recovery_ms"`
	WindowMs         float64 `json:"failover_window_ms"`
	FailoverAttempts int64   `json:"requests_that_failed_over"`
	MethodNote       string  `json:"method_note"`
}

// ReconcileResult compares the gateway ledger against the provider's own log.
type ReconcileResult struct {
	LedgerRows        int `json:"ledger_rows"`
	ProviderRows      int `json:"provider_rows"`
	Matched           int `json:"matched"`
	TokenMismatches   int `json:"token_mismatches"`
	CostMismatches    int `json:"cost_mismatches"`
	MissingInLedger   int `json:"missing_in_ledger"`
	MissingInProvider int `json:"missing_in_provider"`
	EstimatedRows     int `json:"estimated_usage_rows"`
	// EstimatedMismatches counts mismatched rows whose ledger side was an
	// estimate rather than a measurement. In a benchmark these are almost always
	// requests the harness itself cancelled at its -duration deadline while the
	// provider had already completed them: the gateway correctly recorded an
	// estimated (over-)charge, and it simply does not equal the provider's exact
	// count. They are separated out because they are an artifact of shutting the
	// load generator down, not a billing discrepancy in the gateway.
	EstimatedMismatches int      `json:"estimated_mismatches"`
	SettledMismatches   int      `json:"settled_mismatches"`
	Exact               bool     `json:"exact"`
	ExactSettled        bool     `json:"exact_among_settled"`
	Examples            []string `json:"mismatch_examples,omitempty"`
	MethodNote          string   `json:"method_note"`
}

// CacheResult measures what the cache actually buys.
type CacheResult struct {
	ColdRequests int64   `json:"cold_requests"`
	WarmRequests int64   `json:"warm_requests"`
	ColdHits     int64   `json:"cold_pass_hits"`
	Errors       int64   `json:"errors"`
	HitRate      float64 `json:"hit_rate"`
	ColdLatency  Stats   `json:"cold_latency"`
	WarmLatency  Stats   `json:"warm_latency"`
	SpeedupX     float64 `json:"speedup_x"`
	MethodNote   string  `json:"method_note"`
}

// StreamResult measures the streaming path's time-to-first-token and the
// gateway's per-frame relay cost.
type StreamResult struct {
	Requests    int64   `json:"requests"`
	Errors      int64   `json:"errors"`
	TTFB        Stats   `json:"ttfb"`
	Total       Stats   `json:"total"`
	InterFrame  Stats   `json:"inter_frame_gap"`
	FramesTotal int64   `json:"frames_total"`
	CleanEnds   int64   `json:"clean_ends"`
	Truncated   int64   `json:"truncated"`
	MeanFrames  float64 `json:"mean_frames_per_request"`
	MethodNote  string  `json:"method_note"`
}

// ---------------------------------------------------------------------------
// overhead
// ---------------------------------------------------------------------------

func runOverhead(ctx context.Context, cfg benchConfig) (*OverheadResult, error) {
	client := newBenchClient(cfg.Concurrency)

	// Warmup exists for a specific reason worth stating: the first requests pay
	// for TLS/TCP setup, Go's lazily-populated connection pool, and the JIT-free
	// but still cold code paths and heap growth. Including them would put a
	// handful of multi-millisecond samples into a distribution whose p99 we are
	// claiming is under 5ms, at which point the p99 measures process startup
	// rather than steady-state overhead. They are excluded and the exclusion is
	// reported.
	if cfg.Warmup > 0 {
		_ = drive(ctx, client, cfg, cfg.Warmup, 8, nil)
	}

	var (
		mu       sync.Mutex
		selfUs   []float64
		subUs    []float64
		e2eUs    []float64
		upUs     []float64
		errCount atomic.Int64
	)

	start := time.Now()
	total := drive(ctx, client, cfg, cfg.N, cfg.Concurrency, func(s sample) {
		if s.err != nil || s.status != 200 {
			errCount.Add(1)
			return
		}
		mu.Lock()
		e2eUs = append(e2eUs, float64(s.elapsed.Microseconds()))
		if s.overheadUs > 0 {
			selfUs = append(selfUs, float64(s.overheadUs))
		}
		if s.upstreamUs > 0 {
			upUs = append(upUs, float64(s.upstreamUs))
			sub := float64(s.elapsed.Microseconds() - s.upstreamUs)
			// A negative subtractive value means the gateway's upstream timer
			// and the client's wall clock disagree — usually because the
			// gateway timed a period that overlaps its own response write.
			// Clamping to zero would hide that, so they are kept and surfaced:
			// if this happens the two brackets will visibly cross and the
			// method note says so.
			subUs = append(subUs, sub)
		}
		mu.Unlock()
	})
	wall := time.Since(start)

	if len(e2eUs) == 0 {
		return nil, fmt.Errorf("no successful requests out of %d; is the gateway running at %s?", total, cfg.Gateway)
	}

	res := &OverheadResult{
		Requests:     total,
		Errors:       errCount.Load(),
		ErrorRate:    float64(errCount.Load()) / float64(total),
		WallSeconds:  wall.Seconds(),
		Throughput:   float64(total) / wall.Seconds(),
		SelfReported: summarize(selfUs, "us"),
		Subtractive:  summarize(subUs, "us"),
		EndToEnd:     summarize(e2eUs, "us"),
		UpstreamTime: summarize(upUs, "us"),
		MethodNote: "self_reported = gateway's own timer for everything but the upstream round trip " +
			"(X-LLMGW-Overhead-Us). subtractive = client-observed total minus the gateway's reported " +
			"upstream time, so it also contains loopback and client cost and is an upper bound. " +
			fmt.Sprintf("Exact order statistics over the full sample; %d warmup requests excluded.", cfg.Warmup),
	}
	if len(selfUs) == 0 {
		res.MethodNote += " WARNING: the gateway reported no overhead header, so self_reported is empty."
	}
	res.BracketNoteP9 = fmt.Sprintf(
		"p99 overhead is bracketed by [%.0fus self-reported, %.0fus subtractive]; the %.0fus gap is loopback + client + in-process time the gateway cannot see.",
		res.SelfReported.P99, res.Subtractive.P99, res.Subtractive.P99-res.SelfReported.P99)
	return res, nil
}

// ---------------------------------------------------------------------------
// failover
// ---------------------------------------------------------------------------

// runFailover holds steady load, kills the primary provider mid-flight through
// the mock's admin endpoint, and measures how long the gateway takes to route
// around it.
//
// The measurement that matters is the recovery window: from the instant the
// provider stops serving to the instant the gateway is serving successfully
// again. Reporting only "failover happened" would be worthless, and reporting
// the breaker's configured threshold as if it were a measurement would be
// worse — it is an input, not a result.
func runFailover(ctx context.Context, cfg benchConfig) (*FailoverResult, error) {
	client := newBenchClient(cfg.Concurrency)

	type event struct {
		at       time.Time
		ok       bool
		attempts int
	}
	var (
		mu     sync.Mutex
		events []event
	)

	runCtx, cancel := context.WithTimeout(ctx, cfg.Duration)
	defer cancel()

	// Continuous load for the whole window.
	var wg sync.WaitGroup
	for i := 0; i < cfg.Concurrency; i++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			iter := 0
			for runCtx.Err() == nil {
				s := doRequest(runCtx, client, cfg, fmt.Sprintf("fo-%d-%d", worker, iter), false)
				iter++
				if runCtx.Err() != nil {
					break
				}
				mu.Lock()
				events = append(events, event{at: s.at, ok: s.err == nil && s.status == 200, attempts: s.attempts})
				mu.Unlock()
			}
		}(i)
	}

	// Kill the primary after the configured delay, then leave it dead. The
	// gateway is expected to route to the secondary and stay there.
	time.Sleep(cfg.KillAfter)
	killedAt := time.Now()
	if err := mockAdmin(ctx, cfg.Admin, "down", "true"); err != nil {
		cancel()
		wg.Wait()
		return nil, fmt.Errorf("killing primary provider via %s: %w", cfg.Admin, err)
	}

	wg.Wait()

	mu.Lock()
	defer mu.Unlock()
	sort.Slice(events, func(i, j int) bool { return events[i].at.Before(events[j].at) })

	res := &FailoverResult{
		TotalRequests: int64(len(events)),
		KilledAt:      killedAt.UTC().Format(time.RFC3339Nano),
		MethodNote: "Load held constant while the primary provider is killed via the mock's admin endpoint. " +
			"detection_ms = kill to first client-visible failure. recovery_ms = kill to the last failure " +
			"(after which the gateway serves from the secondary continuously). failover_window_ms = the " +
			"span during which any client saw an error. A run with zero failures means the breaker absorbed " +
			"the outage entirely and the window is 0.",
	}

	var (
		firstFail, lastFail time.Time
		sawFail             bool
	)
	for _, e := range events {
		if e.ok {
			res.Succeeded++
		} else {
			res.Failed++
			if e.at.After(killedAt) {
				res.FailedDuringKill++
			}
			if !sawFail || e.at.Before(firstFail) {
				firstFail, sawFail = e.at, true
			}
			if e.at.After(lastFail) {
				lastFail = e.at
			}
		}
		if e.attempts > 1 {
			res.FailoverAttempts++
		}
	}

	if sawFail {
		res.FirstFailureAt = firstFail.UTC().Format(time.RFC3339Nano)
		res.DetectionMs = float64(firstFail.Sub(killedAt).Microseconds()) / 1000
		res.RecoveredAt = lastFail.UTC().Format(time.RFC3339Nano)
		res.RecoveryMs = float64(lastFail.Sub(killedAt).Microseconds()) / 1000
		res.WindowMs = float64(lastFail.Sub(firstFail).Microseconds()) / 1000
	}
	return res, nil
}

// mockAdmin flips a fault switch on the mock provider.
func mockAdmin(ctx context.Context, base, key, val string) error {
	u := strings.TrimRight(base, "/") + "/admin/fault?" + key + "=" + val
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer func() {
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
	}()
	if resp.StatusCode/100 != 2 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("admin %s=%s returned %d: %s", key, val, resp.StatusCode, bytes.TrimSpace(b))
	}
	return nil
}

// ---------------------------------------------------------------------------
// reconcile
// ---------------------------------------------------------------------------

// runReconcile diffs the gateway's ledger against the mock provider's
// independently computed request log.
//
// The independence is the whole point. If both sides computed cost with the same
// code, a bug in that code would cancel out and the reconciliation would pass
// while every invoice was wrong. The mock provider owns its own price list and
// its own token counter, so agreement is evidence.
//
// It delegates the actual comparison to ledger.Reconcile — the SAME code the
// gateway's own tests use — rather than re-implementing the field mapping here.
// A second, subtly-different reconciler would be a place for the two to disagree
// about what "agree" means, which is exactly the bug class this project is about.
// The provider-log path accepts a comma-separated list, because a failover run
// spreads records across every provider that served, and the reconciliation must
// see all of them.
func runReconcile(cfg benchConfig) (*ReconcileResult, error) {
	entries, err := ledger.DecodeFile(cfg.LedgerPath)
	if err != nil {
		return nil, fmt.Errorf("reading gateway ledger: %w", err)
	}

	var records []ledger.ProviderRecord
	for _, path := range strings.Split(cfg.ProviderLogPath, ",") {
		path = strings.TrimSpace(path)
		if path == "" {
			continue
		}
		f, err := os.Open(path)
		if err != nil {
			return nil, fmt.Errorf("reading provider log %q: %w", path, err)
		}
		recs, err := ledger.DecodeProviderLog(f)
		_ = f.Close()
		if err != nil {
			return nil, fmt.Errorf("decoding provider log %q: %w", path, err)
		}
		records = append(records, recs...)
	}

	report := ledger.Reconcile(entries, records)

	estMismatch, settledMismatch := 0, 0
	for _, m := range report.Mismatched {
		if m.LedgerEstimated {
			estMismatch++
		} else {
			settledMismatch++
		}
	}
	res := &ReconcileResult{
		LedgerRows:          report.LedgerRows,
		ProviderRows:        report.UpstreamRows,
		Matched:             len(report.Matched),
		TokenMismatches:     countMismatch(report.Mismatched, true),
		CostMismatches:      countMismatch(report.Mismatched, false),
		MissingInProvider:   len(report.MissingUpstream),
		MissingInLedger:     len(report.MissingLocal),
		EstimatedRows:       report.EstimatedMatched,
		EstimatedMismatches: estMismatch,
		SettledMismatches:   settledMismatch,
		Exact:               report.Exact(),
		// Exact among settled rows: every mismatch that remains is an estimated
		// row (a request the harness cancelled after the provider completed it),
		// and there are no orphans. This is the honest headline for a run that
		// includes a load-generator shutdown; the strict Exact is also reported.
		ExactSettled: settledMismatch == 0 && len(report.MissingLocal) == 0 && len(report.MissingUpstream) == 0 && len(report.Matched) > 0,
		MethodNote: "Delegates to ledger.Reconcile, the same comparator the gateway's own tests use. " +
			"The gateway ledger and the mock provider's log are produced by independent code paths with " +
			"independent price tables; agreement is therefore evidence, not a tautology. Matching is on " +
			"(request id, attempt). No tolerance: a single token or picodollar of disagreement is a mismatch. " +
			"Multiple provider logs (a failover run spreads records across providers) are merged before comparison.",
	}
	for _, m := range report.Mismatched {
		if len(res.Examples) >= 10 {
			break
		}
		var parts []string
		for _, d := range m.Deltas {
			parts = append(parts, d.String())
		}
		res.Examples = append(res.Examples, m.Key.String()+": "+strings.Join(parts, ", "))
	}
	for _, o := range report.MissingLocal {
		if len(res.Examples) >= 10 {
			break
		}
		res.Examples = append(res.Examples, "provider row "+o.Key.String()+" has no ledger counterpart ("+o.Reason+")")
	}
	return res, nil
}

// countMismatch counts mismatches touching a token field (tokens=true) or the
// cost field (tokens=false). A mismatch can touch both, so it is counted in each
// category it touches; the two counts can therefore exceed len(Mismatched).
func countMismatch(ms []ledger.Mismatch, tokens bool) int {
	n := 0
	for _, m := range ms {
		touchedToken, touchedCost := false, false
		for _, d := range m.Deltas {
			switch d.Field {
			case ledger.FieldPromptTokens, ledger.FieldCachedTokens,
				ledger.FieldCompletionTokens, ledger.FieldReasoningTokens:
				touchedToken = true
			case ledger.FieldCost:
				touchedCost = true
			}
		}
		if tokens && touchedToken {
			n++
		}
		if !tokens && touchedCost {
			n++
		}
	}
	return n
}

// ---------------------------------------------------------------------------
// cache
// ---------------------------------------------------------------------------

// runCache measures the cache honestly: a cold pass over N distinct prompts,
// then a warm pass over the same N. Reporting a cache speedup by sending the
// same prompt N times would measure nothing but the second-request path.
func runCache(ctx context.Context, cfg benchConfig) (*CacheResult, error) {
	client := newBenchClient(cfg.Concurrency)
	// Cacheable requests must be deterministic, so temperature is pinned to 0.
	// A non-zero temperature is deliberately not cached by the gateway, and a
	// cache benchmark run at temperature 0.7 would report a 0% hit rate and
	// look like a bug rather than the policy it is.
	cc := cfg
	cc.N = min(cfg.N, 400)
	// Raise the completion cap above the mock's configured reply length. With the
	// default -max-tokens=32 the provider stops AT the cap and reports
	// finish_reason "length", and the cache correctly refuses to store a truncated
	// response — so the whole warm pass would miss and the benchmark would measure
	// the cache's safety policy instead of the cache. Documented rather than
	// silently reported as a 0% hit rate.
	if cc.MaxTokens < 512 {
		cc.MaxTokens = 512
	}

	var (
		mu           sync.Mutex
		cold, warm   []float64
		coldHits     atomic.Int64
		warmHits     atomic.Int64
		coldN, warmN atomic.Int64
		errsCold     atomic.Int64
		errsWarm     atomic.Int64
	)
	distinctPrompts := cc.N

	// collect returns a sampler that records into one pass's slice. The two
	// passes are counted separately on purpose: a cold pass that already reports
	// hits means the cache was dirty from a previous run, and the hit rate would
	// be flattered by requests this benchmark did not issue. Reporting cold hits
	// makes that visible instead of silently folding it in.
	collect := func(dst *[]float64, cnt, hits, errs *atomic.Int64) func(sample) {
		return func(s sample) {
			if s.err != nil || s.status != 200 {
				errs.Add(1)
				return
			}
			cnt.Add(1)
			if s.cacheHit {
				hits.Add(1)
			}
			mu.Lock()
			*dst = append(*dst, float64(s.elapsed.Microseconds()))
			mu.Unlock()
		}
	}

	// Cold pass: N distinct deterministic prompts at temperature 0.
	driveDeterministic(ctx, client, cc, distinctPrompts, cc.Concurrency, 0,
		collect(&cold, &coldN, &coldHits, &errsCold))
	// Warm pass: the identical N requests, so every one should hit.
	driveDeterministic(ctx, client, cc, distinctPrompts, cc.Concurrency, 0,
		collect(&warm, &warmN, &warmHits, &errsWarm))

	res := &CacheResult{
		ColdRequests: coldN.Load(),
		WarmRequests: warmN.Load(),
		ColdHits:     coldHits.Load(),
		Errors:       errsCold.Load() + errsWarm.Load(),
		ColdLatency:  summarize(cold, "us"),
		WarmLatency:  summarize(warm, "us"),
		MethodNote: "Cold pass sends N distinct deterministic (temperature=0) prompts; the warm pass repeats " +
			"the identical N. Hit rate is measured on the warm pass only. Reusing one prompt N times would " +
			"measure the cache against itself.",
	}
	if res.WarmRequests > 0 {
		res.HitRate = float64(warmHits.Load()) / float64(res.WarmRequests)
	}
	if res.WarmLatency.P50 > 0 {
		res.SpeedupX = res.ColdLatency.P50 / res.WarmLatency.P50
	}
	if res.ColdHits > 0 {
		res.MethodNote += fmt.Sprintf(
			" WARNING: %d of the cold pass's requests were already cached from an earlier run, "+
				"so the cold latency is not purely cold.", res.ColdHits)
	}
	return res, nil
}

// ---------------------------------------------------------------------------
// stream
// ---------------------------------------------------------------------------

// runStream measures the streaming path from a real incremental reader, which is
// what k6 cannot do. Time-to-first-byte and the inter-frame gap distribution are
// the numbers that matter for a streaming proxy: a gateway that buffers the
// whole response and then flushes it has a perfect total latency and a useless
// TTFB, and only a frame-level reader can tell the difference.
func runStream(ctx context.Context, cfg benchConfig) (*StreamResult, error) {
	client := newBenchClient(cfg.Concurrency)
	n := min(cfg.N, 400)

	var (
		mu                       sync.Mutex
		ttfb, total, gaps        []float64
		frames, cleanEnds, trunc atomic.Int64
		errCount, reqCount       atomic.Int64
		wg                       sync.WaitGroup
		sem                      = make(chan struct{}, max(1, cfg.Concurrency/2))
	)

	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			st, err := streamOnce(ctx, client, cfg, fmt.Sprintf("st-%d", i))
			reqCount.Add(1)
			if err != nil {
				errCount.Add(1)
				return
			}
			frames.Add(int64(st.frames))
			if st.clean {
				cleanEnds.Add(1)
			} else {
				trunc.Add(1)
			}
			mu.Lock()
			if st.ttfb > 0 {
				ttfb = append(ttfb, float64(st.ttfb.Microseconds()))
			}
			total = append(total, float64(st.total.Microseconds()))
			gaps = append(gaps, st.gaps...)
			mu.Unlock()
		}(i)
	}
	wg.Wait()

	res := &StreamResult{
		Requests:    reqCount.Load(),
		Errors:      errCount.Load(),
		TTFB:        summarize(ttfb, "us"),
		Total:       summarize(total, "us"),
		InterFrame:  summarize(gaps, "us"),
		FramesTotal: frames.Load(),
		CleanEnds:   cleanEnds.Load(),
		Truncated:   trunc.Load(),
		MethodNote: "Measured with an incremental SSE reader, so TTFB is the time to the first content " +
			"frame rather than to the full response. A stream is 'clean' only if it ended with the [DONE] " +
			"sentinel; anything else is counted as truncated, which is what makes a silently-cut stream " +
			"visible instead of passing as a success.",
	}
	if res.Requests > 0 {
		res.MeanFrames = float64(res.FramesTotal) / float64(res.Requests)
	}
	return res, nil
}

type streamStat struct {
	ttfb   time.Duration
	total  time.Duration
	frames int
	clean  bool
	gaps   []float64
}

func streamOnce(ctx context.Context, client *http.Client, cfg benchConfig, tag string) (streamStat, error) {
	var st streamStat
	body := buildBody(cfg, tag, true, 0.7)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		strings.TrimRight(cfg.Gateway, "/")+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return st, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	req.Header.Set("Accept", "text/event-stream")

	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		return st, err
	}
	defer func() {
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
	}()
	if resp.StatusCode != http.StatusOK {
		return st, fmt.Errorf("stream status %d", resp.StatusCode)
	}

	// A minimal SSE reader. internal/sse has the real one, but the benchmark
	// deliberately does not import it: an instrument that shares a parser with
	// the thing it measures cannot detect that parser's bugs.
	br := bufio.NewReaderSize(resp.Body, 16<<10)
	last := start
	for {
		line, err := br.ReadString('\n')
		if len(line) > 0 {
			trimmed := strings.TrimRight(line, "\r\n")
			if strings.HasPrefix(trimmed, "data: ") {
				payload := strings.TrimPrefix(trimmed, "data: ")
				if payload == "[DONE]" {
					st.clean = true
					st.total = time.Since(start)
					return st, nil
				}
				now := time.Now()
				if st.frames == 0 {
					st.ttfb = now.Sub(start)
				} else {
					st.gaps = append(st.gaps, float64(now.Sub(last).Microseconds()))
				}
				last = now
				st.frames++
			}
		}
		if err != nil {
			// EOF without [DONE] is a truncated stream, not a clean end.
			st.total = time.Since(start)
			if errors.Is(err, io.EOF) {
				return st, nil
			}
			return st, nil
		}
	}
}

// ---------------------------------------------------------------------------
// request driving
// ---------------------------------------------------------------------------

type sample struct {
	at         time.Time
	elapsed    time.Duration
	status     int
	overheadUs int64
	upstreamUs int64
	attempts   int
	cacheHit   bool
	err        error
}

func newBenchClient(conc int) *http.Client {
	// The benchmark client must never be the bottleneck. The stdlib default of
	// 2 idle connections per host would serialise the load generator behind
	// connection setup and produce a latency distribution that is mostly the
	// client's own queueing — a classic way to accidentally benchmark your
	// benchmark.
	idle := conc * 2
	if idle < 64 {
		idle = 64
	}
	return &http.Client{
		Transport: &http.Transport{
			MaxIdleConns:          idle,
			MaxIdleConnsPerHost:   idle,
			MaxConnsPerHost:       0,
			IdleConnTimeout:       90 * time.Second,
			DisableCompression:    true, // gzip would add CPU to both sides and distort the overhead measurement
			ResponseHeaderTimeout: 60 * time.Second,
			ForceAttemptHTTP2:     false, // measure the HTTP/1.1 path most deployments actually use
		},
		// No client Timeout: a long legitimate stream must not be killed by the
		// instrument. Deadlines come from the context.
		Timeout: 0,
	}
}

func buildBody(cfg benchConfig, tag string, stream bool, temp float64) []byte {
	type msg struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	payload := map[string]any{
		"model": cfg.Model,
		"messages": []msg{
			{Role: "system", Content: "You are a concise assistant."},
			{Role: "user", Content: "Summarise the state of request " + tag + " in one sentence."},
		},
		"max_tokens":  cfg.MaxTokens,
		"temperature": temp,
	}
	if stream {
		payload["stream"] = true
		payload["stream_options"] = map[string]any{"include_usage": true}
	}
	b, _ := json.Marshal(payload)
	return b
}

func doRequest(ctx context.Context, client *http.Client, cfg benchConfig, tag string, stream bool) sample {
	return doRequestTemp(ctx, client, cfg, tag, stream, 0.7)
}

func doRequestTemp(ctx context.Context, client *http.Client, cfg benchConfig, tag string, stream bool, temp float64) sample {
	s := sample{at: time.Now()}
	body := buildBody(cfg, tag, stream, temp)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		strings.TrimRight(cfg.Gateway, "/")+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		s.err = err
		return s
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+cfg.APIKey)

	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		s.elapsed = time.Since(start)
		s.err = err
		return s
	}
	// The body must be fully drained before stopping the clock, or the
	// measurement excludes the response transfer and the connection is not
	// returned to the pool.
	_, copyErr := io.Copy(io.Discard, resp.Body)
	s.elapsed = time.Since(start)
	_ = resp.Body.Close()
	if copyErr != nil {
		s.err = copyErr
	}
	s.status = resp.StatusCode
	s.overheadUs = headerInt(resp.Header, "X-Llmgw-Overhead-Us")
	s.upstreamUs = headerInt(resp.Header, "X-Llmgw-Upstream-Us")
	s.attempts = int(headerInt(resp.Header, "X-Llmgw-Attempts"))
	s.cacheHit = strings.EqualFold(resp.Header.Get("X-Llmgw-Cache"), "hit")
	return s
}

// drive issues n requests at the given concurrency with unique prompts.
func drive(ctx context.Context, client *http.Client, cfg benchConfig, n, conc int, onSample func(sample)) int64 {
	return driveDeterministic(ctx, client, cfg, n, conc, 0.7, onSample)
}

// driveDeterministic issues n requests whose prompts are a deterministic
// function of the index, so a second pass with the same n produces
// byte-identical requests. That is what makes the cache pass meaningful.
func driveDeterministic(ctx context.Context, client *http.Client, cfg benchConfig, n, conc int, temp float64, onSample func(sample)) int64 {
	if conc < 1 {
		conc = 1
	}
	var (
		wg    sync.WaitGroup
		idx   atomic.Int64
		count atomic.Int64
	)
	for w := 0; w < conc; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				i := idx.Add(1) - 1
				if i >= int64(n) || ctx.Err() != nil {
					return
				}
				s := doRequestTemp(ctx, client, cfg, "bench-"+strconv.FormatInt(i, 10), false, temp)
				count.Add(1)
				if onSample != nil {
					onSample(s)
				}
			}
		}()
	}
	wg.Wait()
	return count.Load()
}

func headerInt(h http.Header, key string) int64 {
	v := h.Get(key)
	if v == "" {
		return 0
	}
	n, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64)
	if err != nil {
		return 0
	}
	return n
}

// ---------------------------------------------------------------------------
// statistics
// ---------------------------------------------------------------------------

// summarize computes exact order statistics over the full sample.
//
// Quantile convention: nearest-rank on the sorted sample (the p99 is a real
// observed value, not an interpolation between two). Stating the convention
// matters because "p99" is ambiguous across tools by a fraction of a percent,
// and a claim near a 5ms threshold should not turn on an undocumented
// interpolation rule.
func summarize(xs []float64, unit string) Stats {
	st := Stats{Unit: unit, Count: int64(len(xs))}
	if len(xs) == 0 {
		return st
	}
	s := make([]float64, len(xs))
	copy(s, xs)
	sort.Float64s(s)
	var sum float64
	for _, v := range s {
		sum += v
	}
	st.Min, st.Max = s[0], s[len(s)-1]
	st.Mean = sum / float64(len(s))
	st.P50 = quantile(s, 0.50)
	st.P90 = quantile(s, 0.90)
	st.P95 = quantile(s, 0.95)
	st.P99 = quantile(s, 0.99)
	st.P999 = quantile(s, 0.999)
	return st
}

func quantile(sorted []float64, q float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	if len(sorted) == 1 {
		return sorted[0]
	}
	// Nearest-rank: ceil(q*N)-th smallest, 1-indexed.
	rank := int(math.Ceil(q * float64(len(sorted))))
	if rank < 1 {
		rank = 1
	}
	if rank > len(sorted) {
		rank = len(sorted)
	}
	return sorted[rank-1]
}

// ---------------------------------------------------------------------------
// misc helpers
// ---------------------------------------------------------------------------

func hostFacts() HostFacts {
	h := HostFacts{
		OS:    runtime.GOOS,
		Arch:  runtime.GOARCH,
		GoVer: strings.TrimPrefix(runtime.Version(), "go"),
		Cores: runtime.NumCPU(),
	}
	// sysctl is macOS/BSD-specific. On other platforms these simply stay empty
	// rather than the report claiming hardware it could not read.
	if b, err := exec.Command("sysctl", "-n", "machdep.cpu.brand_string").Output(); err == nil {
		h.CPU = strings.TrimSpace(string(b))
	}
	if b, err := exec.Command("sysctl", "-n", "hw.memsize").Output(); err == nil {
		if n, err := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64); err == nil {
			h.MemGB = int(n / (1 << 30))
		}
	}
	if b, err := exec.Command("sysctl", "-n", "hw.model").Output(); err == nil {
		h.Machine = strings.TrimSpace(string(b))
	}
	return h
}

func printReport(r *Report) {
	fmt.Printf("\n=== gwbench: %s ===\n", r.Mode)
	if r.Label != "" {
		fmt.Printf("label: %s\n", r.Label)
	}
	fmt.Printf("host:  %s %s, %d cores", r.Host.OS, r.Host.Arch, r.Host.Cores)
	if r.Host.CPU != "" {
		fmt.Printf(", %s", r.Host.CPU)
	}
	if r.Host.Machine != "" {
		fmt.Printf(" (%s)", r.Host.Machine)
	}
	fmt.Printf("\n")

	if o := r.Overhead; o != nil {
		fmt.Printf("\n-- gateway overhead (%d requests, c=%d, %.0f rps) --\n",
			o.Requests, r.Config.Concurrency, o.Throughput)
		fmt.Printf("errors: %d (%.3f%%)\n", o.Errors, o.ErrorRate*100)
		printStats("self-reported overhead", o.SelfReported)
		printStats("subtractive overhead ", o.Subtractive)
		printStats("upstream time        ", o.UpstreamTime)
		printStats("end-to-end           ", o.EndToEnd)
		fmt.Printf("\n%s\n", wrap(o.BracketNoteP9, 96))
		fmt.Printf("\nmethod: %s\n", wrap(o.MethodNote, 96))
	}
	if f := r.Failover; f != nil {
		fmt.Printf("\n-- failover --\n")
		fmt.Printf("requests: %d  succeeded: %d  failed: %d (%d after the kill)\n",
			f.TotalRequests, f.Succeeded, f.Failed, f.FailedDuringKill)
		fmt.Printf("requests that failed over to a second provider: %d\n", f.FailoverAttempts)
		if f.Failed > 0 {
			fmt.Printf("detection: %.1f ms after kill\nrecovery:  %.1f ms after kill\nwindow:    %.1f ms\n",
				f.DetectionMs, f.RecoveryMs, f.WindowMs)
		} else {
			fmt.Printf("no client-visible failure: the gateway absorbed the outage entirely\n")
		}
		fmt.Printf("\nmethod: %s\n", wrap(f.MethodNote, 96))
	}
	if rc := r.Reconcile; rc != nil {
		fmt.Printf("\n-- cost reconciliation --\n")
		fmt.Printf("ledger rows: %d   provider rows: %d\n", rc.LedgerRows, rc.ProviderRows)
		fmt.Printf("matched: %d   token mismatches: %d   cost mismatches: %d\n",
			rc.Matched, rc.TokenMismatches, rc.CostMismatches)
		fmt.Printf("missing in ledger: %d   missing in provider log: %d\n",
			rc.MissingInLedger, rc.MissingInProvider)
		fmt.Printf("settled mismatches: %d   estimated-row mismatches: %d (harness-cancelled tail)\n",
			rc.SettledMismatches, rc.EstimatedMismatches)
		fmt.Printf("EXACT (strict): %v    EXACT among settled rows: %v\n", rc.Exact, rc.ExactSettled)
		for _, e := range rc.Examples {
			fmt.Printf("  ! %s\n", e)
		}
		fmt.Printf("\nmethod: %s\n", wrap(rc.MethodNote, 96))
	}
	if c := r.Cache; c != nil {
		fmt.Printf("\n-- cache --\n")
		fmt.Printf("cold: %d   warm: %d   hit rate on the warm pass: %.1f%%\n",
			c.ColdRequests, c.WarmRequests, c.HitRate*100)
		printStats("cold latency", c.ColdLatency)
		printStats("warm latency", c.WarmLatency)
		fmt.Printf("median speedup: %.1fx\n", c.SpeedupX)
		fmt.Printf("\nmethod: %s\n", wrap(c.MethodNote, 96))
	}
	if s := r.Stream; s != nil {
		fmt.Printf("\n-- streaming --\n")
		fmt.Printf("requests: %d   errors: %d   frames: %d (mean %.1f/request)\n",
			s.Requests, s.Errors, s.FramesTotal, s.MeanFrames)
		fmt.Printf("clean ends: %d   truncated: %d\n", s.CleanEnds, s.Truncated)
		printStats("time to first frame", s.TTFB)
		printStats("inter-frame gap    ", s.InterFrame)
		printStats("total              ", s.Total)
		fmt.Printf("\nmethod: %s\n", wrap(s.MethodNote, 96))
	}
	fmt.Println()
}

func printStats(name string, s Stats) {
	if s.Count == 0 {
		fmt.Printf("%s: no samples\n", name)
		return
	}
	div := 1000.0 // us -> ms for display
	fmt.Printf("%s: n=%d  min=%.3f  p50=%.3f  p90=%.3f  p95=%.3f  p99=%.3f  p99.9=%.3f  max=%.3f  (ms)\n",
		name, s.Count, s.Min/div, s.P50/div, s.P90/div, s.P95/div, s.P99/div, s.P999/div, s.Max/div)
}

func wrap(s string, width int) string {
	words := strings.Fields(s)
	var b strings.Builder
	col := 0
	for i, w := range words {
		if col > 0 && col+1+len(w) > width {
			b.WriteString("\n        ")
			col = 8
		} else if i > 0 {
			b.WriteByte(' ')
			col++
		}
		b.WriteString(w)
		col += len(w)
	}
	return b.String()
}

func writeReport(dir string, r *Report) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	name := fmt.Sprintf("%s.json", r.Mode)
	path := filepath.Join(dir, name)
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	if err := os.WriteFile(path, b, 0o644); err != nil {
		return err
	}
	fmt.Printf("wrote %s\n", path)
	return nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
