package server

import (
	"net/http"
	"strconv"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
	"github.com/harsha-moparthy/llmgw/internal/budget"
	"github.com/harsha-moparthy/llmgw/internal/cache"
	"github.com/harsha-moparthy/llmgw/internal/ledger"
	"github.com/harsha-moparthy/llmgw/internal/metrics"
	"github.com/harsha-moparthy/llmgw/internal/money"
	"github.com/harsha-moparthy/llmgw/internal/router"
)

// writeAttemptHeaders exposes what the router did, so a client or the benchmark
// can read it off the response rather than querying separately.
func (s *Server) writeAttemptHeaders(w http.ResponseWriter, ar *router.AttemptResult, _ time.Time) {
	if ar == nil {
		return
	}
	if ar.ServedBy != "" {
		w.Header().Set(HeaderProvider, ar.ServedBy)
		w.Header().Set(HeaderModel, ar.ServedModel)
	}
	w.Header().Set(HeaderAttempts, strconv.Itoa(ar.Attempts))
}

// setOverheadHeader writes the gateway's own overhead (total minus upstream) in
// microseconds. This is the number cmd/gwbench reads to isolate gateway time
// from provider time — the "added latency excluding provider time" the spec asks
// for.
func (s *Server) setOverheadHeader(w http.ResponseWriter, upstreamUs int64, start time.Time) {
	total := s.now().Sub(start).Microseconds()
	overhead := total - upstreamUs
	if overhead < 0 {
		overhead = 0
	}
	w.Header().Set(HeaderOverheadUs, strconv.FormatInt(overhead, 10))
	w.Header().Set(HeaderUpstreamUs, strconv.FormatInt(upstreamUs, 10))
}

// writeBudgetRejection writes a rejection whose body carries the numbers a
// client needs to act: the limit, what has been spent, what remains, and when
// the window resets. "budget exceeded" with no numbers forces a client to guess.
func (s *Server) writeBudgetRejection(w http.ResponseWriter, d budget.Decision) {
	if ra := d.RetryAfter(); ra > 0 {
		w.Header().Set("Retry-After", strconv.FormatInt(int64(ra.Seconds()), 10))
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(d.HTTPStatus())
	// d.Message already includes limit/spent/remaining/reset; the budget package
	// owns that wording so the status and the explanation stay consistent.
	env := apiv1.NewError(apiv1.ErrTypeBudget, d.Reason, d.Message())
	_ = encodeJSON(w, env)
}

func (s *Server) recordCacheLookup(hit bool) {
	if s.deps.Metrics == nil {
		return
	}
	result := "miss"
	if hit {
		result = "hit"
	}
	s.deps.Metrics.CacheLookups.With(metrics.Labels{"result": result}).Inc()
	s.publishCacheStats()
}

func (s *Server) publishCacheStats() {
	if s.deps.Metrics == nil || s.deps.Cache == nil {
		return
	}
	st := s.deps.Cache.Stats()
	s.deps.Metrics.CacheEntries.With(nil).Set(float64(st.Entries))
	s.deps.Metrics.CacheBytes.With(nil).Set(float64(st.Bytes))
	s.deps.Metrics.CacheSavedPico.With(nil).Set(float64(st.CostSaved))
}

func (s *Server) recordRequestMetric(tenant, model string, status int, start time.Time) {
	if s.deps.Metrics == nil {
		return
	}
	model = s.safeModelLabel(model)
	s.deps.Metrics.Requests.With(metrics.Labels{
		"tenant": tenant, "model": model, "status": metrics.StatusClass(status),
	}).Inc()
	metrics.ObserveSeconds(s.deps.Metrics.RequestLatency.With(metrics.Labels{"model": model}), s.now().Sub(start))
}

// safeModelLabel collapses a model name that is not a configured route into a
// single constant.
//
// The `model` field comes from the request body, and a metric label is a
// permanently retained series. Without this, any authenticated tenant could mint
// unbounded series — and therefore unbounded memory — by sending a fresh random
// model name per request, which is a denial of service with a 404 as its only
// visible symptom. Cardinality is bounded by configuration, which is the rule the
// metrics package documents for itself; this is the one place client input
// reached a label and it needed enforcing here rather than trusted at call sites.
func (s *Server) safeModelLabel(model string) string {
	if model == "" {
		return ""
	}
	if _, ok := s.deps.Config.Routes[model]; ok {
		return model
	}
	return "unknown"
}

func (s *Server) recordAttemptMetrics(o router.AttemptOutcome) {
	if s.deps.Metrics == nil {
		return
	}
	m := s.deps.Metrics
	if o.Failure != nil {
		m.UpstreamErrors.With(metrics.Labels{"provider": o.Provider, "class": o.Failure.Class.String()}).Inc()
		if o.FailedOver {
			m.Failovers.With(metrics.Labels{"class": o.Failure.Class.String()}).Inc()
		}
	}
	if o.Usage != nil {
		labels := metrics.Labels{"tenant": o.Tenant, "model": o.UpstreamModel}
		m.PromptTokens.With(labels).Add(float64(o.Usage.PromptTokens))
		m.CompletionTokens.With(labels).Add(float64(o.Usage.CompletionTokens))
		if rt := o.Usage.ReasoningTokens(); rt > 0 {
			m.ReasoningTokens.With(labels).Add(float64(rt))
		}
		if o.UsageEstimated {
			m.CostEstimatedPico.With(labels).Add(float64(o.Cost))
		} else {
			m.CostPico.With(labels).Add(float64(o.Cost))
		}
	}
	metrics.ObserveSeconds(m.UpstreamLatency.With(metrics.Labels{"provider": o.Provider}), o.Latency)
}

func (s *Server) recordSuccessMetrics(tenant, model string, ar *router.AttemptResult, res *providerResult, actualCost money.Pico, upstreamUs int64, start time.Time) {
	if s.deps.Metrics == nil {
		return
	}
	m := s.deps.Metrics
	s.recordRequestMetric(tenant, model, http.StatusOK, start)
	m.Attempts.With(metrics.Labels{"model": model}).Observe(float64(ar.Attempts))
	overhead := s.now().Sub(start) - time.Duration(upstreamUs)*time.Microsecond
	if overhead > 0 {
		metrics.ObserveSeconds(m.Overhead.With(metrics.Labels{"model": model}), overhead)
	}
	s.publishBudget(tenant)
}

func (s *Server) publishBudget(tenant string) {
	if s.deps.Metrics == nil || s.deps.Budget == nil {
		return
	}
	d := s.deps.Budget.Status(tenant)
	if d.Unlimited {
		return
	}
	s.deps.Metrics.BudgetSpent.With(metrics.Labels{"tenant": tenant}).Set(float64(d.Spent))
	s.deps.Metrics.BudgetRemaining.With(metrics.Labels{"tenant": tenant}).Set(float64(d.Remaining))
}

// serveCacheHit serves a cached entry, as JSON or as a synthetic SSE stream.
func (s *Server) serveCacheHit(w http.ResponseWriter, req *apiv1.ChatRequest, entry *cache.Entry, reqID string, start time.Time) {
	w.Header().Set(HeaderCache, "hit")
	w.Header().Set(HeaderProvider, entry.Provider)
	w.Header().Set(HeaderModel, entry.Model)
	w.Header().Set(HeaderAttempts, "0")
	w.Header().Set(HeaderRequestID, reqID)
	w.Header().Set(HeaderOverheadUs, strconv.FormatInt(s.now().Sub(start).Microseconds(), 10))
	w.Header().Set(HeaderUpstreamUs, "0")

	if !req.Stream {
		writeJSON(w, http.StatusOK, entry.Response)
		s.recordRequestMetric("", req.Model, http.StatusOK, start)
		return
	}
	// Replay the cached response as a stream. The chunk boundaries are not the
	// original ones (they cannot be — the cache stores assembled text), which is
	// documented in the cache package.
	if err := s.streamCachedEntry(w, req, entry); err != nil {
		s.deps.Logger.Warn("serving cached stream", "err", err)
	}
}

func (s *Server) ledgerCacheHit(tenant, alias string, entry *cache.Entry, reqID string, now time.Time) {
	if s.deps.Ledger == nil {
		return
	}
	e := ledger.Entry{
		RequestID:      reqID,
		Tenant:         tenant,
		RequestedModel: alias,
		Provider:       entry.Provider,
		UpstreamModel:  entry.Model,
		Attempt:        1,
		Status:         ledger.StatusCacheHit,
		CacheHit:       true,
		// A cache hit consumed no upstream tokens, so cost is zero and the tokens
		// are recorded as source "none": the tenant did get the value, but the
		// gateway paid the provider nothing, and billing for hits is a separate
		// pricing policy the ledger must not silently apply.
		UsageSource: ledger.SourceNone,
		Billable:    false,
		StartedAt:   now,
		EndedAt:     now,
	}
	if err := s.deps.Ledger.Append(e); err != nil {
		s.deps.Logger.Warn("ledger cache-hit append failed", "err", err)
	}
}

func (s *Server) ledgerReject(tenant, alias string, now time.Time) {
	if s.deps.Ledger == nil {
		return
	}
	e := ledger.Entry{
		RequestID:      s.requestID(),
		Tenant:         tenant,
		RequestedModel: alias,
		Attempt:        1,
		Status:         ledger.StatusRejected,
		UsageSource:    ledger.SourceNone,
		Billable:       false,
		StartedAt:      now,
		EndedAt:        now,
	}
	_ = s.deps.Ledger.Append(e)
}

// maybeCache stores a successful response when the request was cacheable.
func (s *Server) maybeCache(req *apiv1.ChatRequest, res *providerResult, ar *router.AttemptResult, scope cache.Scope, cost money.Pico) {
	if s.deps.Cache == nil || res == nil || res.Response == nil {
		return
	}
	cacheable, _ := s.deps.Cache.Cacheable(req)
	if !cacheable {
		return
	}
	key := cache.ComputeKey(req, scope)
	entry := &cache.Entry{
		Response:    res.Response,
		Usage:       res.Usage,
		CostAvoided: cost,
		Provider:    ar.ServedBy,
		Model:       ar.ServedModel,
	}
	// A Put that the policy refuses (e.g. a truncated response) returns an error
	// we intentionally ignore: not caching is the correct outcome, not a failure.
	_ = s.deps.Cache.Put(key, entry, req)
	s.publishCacheStats()
}

func encodeJSON(w http.ResponseWriter, v any) error {
	return jsonEncode(w, v)
}
