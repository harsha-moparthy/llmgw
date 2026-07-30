package metrics

import "time"

// Gateway is the gateway's metric set, defined in one place so that a metric's
// name, help text, labels and bucket layout are decided once rather than
// scattered across the handlers that record them.
//
// Every vector is pre-resolved at construction, so recording an observation on
// the request path is an atomic add on a pointer that was looked up at startup —
// not a map lookup keyed by a freshly-built label string.
//
// # Label cardinality
//
// Cardinality is the way a metrics layer takes down the process it is measuring:
// every distinct label combination is a retained series, so a label whose values
// come from client input is an unbounded memory leak. The rules here:
//
//   - tenant, model, provider: bounded by configuration, so safe.
//   - status: the HTTP status class, not the code, and never a message.
//   - class: provider.Class values, a closed enum.
//
// Never a request id, never a user id, never an error string.
type Gateway struct {
	reg *Registry

	// Request-level.
	Requests        *CounterVec
	RequestLatency  *HistogramVec
	Overhead        *HistogramVec
	UpstreamLatency *HistogramVec
	InFlight        *GaugeVec

	// Tokens and money.
	PromptTokens     *CounterVec
	CompletionTokens *CounterVec
	ReasoningTokens  *CounterVec
	CostPico         *CounterVec
	// CostEstimatedPico is tracked separately from CostPico rather than being
	// folded in with a label, because an operator asking "what did we spend"
	// must not be handed a total that silently mixes measured and estimated
	// numbers. Two counters make the split impossible to miss.
	CostEstimatedPico *CounterVec

	// Routing and failure.
	Attempts       *HistogramVec
	Failovers      *CounterVec
	UpstreamErrors *CounterVec
	NoProvider     *CounterVec

	// Breaker.
	BreakerState       *GaugeVec
	BreakerTransitions *CounterVec
	BreakerRejections  *CounterVec

	// Budget.
	BudgetRejections *CounterVec
	BudgetDegraded   *CounterVec
	BudgetSpent      *GaugeVec
	BudgetRemaining  *GaugeVec

	// Cache.
	CacheLookups   *CounterVec
	CacheRejected  *CounterVec
	CacheEntries   *GaugeVec
	CacheBytes     *GaugeVec
	CacheSavedPico *GaugeVec

	// Streaming.
	StreamTTFB      *HistogramVec
	StreamFrames    *CounterVec
	StreamTruncated *CounterVec
	StreamAbandoned *CounterVec
}

// NewGateway builds the gateway metric set on a fresh registry.
func NewGateway() *Gateway {
	r := NewRegistry()
	return NewGatewayOn(r)
}

// NewGatewayOn builds the metric set on an existing registry.
func NewGatewayOn(r *Registry) *Gateway {
	lat := DefaultLatencyBucketsSeconds
	return &Gateway{
		reg: r,

		Requests: r.Counter("llmgw_requests_total",
			"Client requests by tenant, requested model, and HTTP status class."),
		RequestLatency: r.Histogram("llmgw_request_duration_seconds",
			"End-to-end request duration as the gateway sees it, including upstream time.", lat),
		Overhead: r.Histogram("llmgw_overhead_seconds",
			"Gateway-added latency: total request time minus the upstream round trip. "+
				"NOTE: quantiles from this histogram are interpolated and bucket-bounded; "+
				"the exact figures reported by this project come from cmd/gwbench.", lat),
		UpstreamLatency: r.Histogram("llmgw_upstream_duration_seconds",
			"Time spent waiting on the upstream provider.", lat),
		InFlight: r.Gauge("llmgw_requests_in_flight",
			"Requests currently being served."),

		PromptTokens: r.Counter("llmgw_prompt_tokens_total",
			"Prompt tokens billed, by tenant and upstream model."),
		CompletionTokens: r.Counter("llmgw_completion_tokens_total",
			"Completion tokens billed, by tenant and upstream model. "+
				"Includes reasoning tokens, matching provider billing semantics."),
		ReasoningTokens: r.Counter("llmgw_reasoning_tokens_total",
			"Hidden reasoning tokens, reported for visibility. Already counted in "+
				"llmgw_completion_tokens_total; do not add the two."),
		CostPico: r.Counter("llmgw_cost_picodollars_total",
			"Cost in picodollars computed from provider-reported usage."),
		CostEstimatedPico: r.Counter("llmgw_cost_estimated_picodollars_total",
			"Cost in picodollars computed from ESTIMATED usage, where the provider "+
				"reported none. Kept separate so a spend total is never silently part guess."),

		Attempts: r.Histogram("llmgw_attempts_per_request",
			"Upstream attempts per client request. A value above 1 means failover occurred.",
			[]float64{1, 2, 3, 4, 5, 10}),
		Failovers: r.Counter("llmgw_failovers_total",
			"Failovers to another provider, by the failure class that caused them."),
		UpstreamErrors: r.Counter("llmgw_upstream_errors_total",
			"Upstream failures by provider and classified failure class."),
		NoProvider: r.Counter("llmgw_no_provider_total",
			"Requests that found no usable provider for the requested model."),

		BreakerState: r.Gauge("llmgw_breaker_state",
			"Circuit breaker state per provider: 0=closed, 1=half-open, 2=open."),
		BreakerTransitions: r.Counter("llmgw_breaker_transitions_total",
			"Circuit breaker state transitions per provider."),
		BreakerRejections: r.Counter("llmgw_breaker_rejections_total",
			"Attempts rejected without an upstream call because the breaker was open."),

		BudgetRejections: r.Counter("llmgw_budget_rejections_total",
			"Requests rejected for exceeding a tenant's budget."),
		BudgetDegraded: r.Counter("llmgw_budget_degraded_total",
			"Requests routed to a cheaper model because a tenant crossed its soft threshold."),
		BudgetSpent: r.Gauge("llmgw_budget_spent_picodollars",
			"Settled spend in the current budget window, per tenant."),
		BudgetRemaining: r.Gauge("llmgw_budget_remaining_picodollars",
			"Remaining budget in the current window, per tenant, net of outstanding holds."),

		CacheLookups: r.Counter("llmgw_cache_lookups_total",
			"Cache lookups by result: hit or miss."),
		CacheRejected: r.Counter("llmgw_cache_rejected_total",
			"Responses the cache declined to store, by reason. Distinct from a miss."),
		CacheEntries: r.Gauge("llmgw_cache_entries",
			"Entries currently cached."),
		CacheBytes: r.Gauge("llmgw_cache_bytes",
			"Approximate bytes held by the cache."),
		CacheSavedPico: r.Gauge("llmgw_cache_saved_picodollars",
			"Upstream cost avoided by cache hits, in picodollars."),

		StreamTTFB: r.Histogram("llmgw_stream_ttfb_seconds",
			"Time from client request to the first content frame forwarded.", lat),
		StreamFrames: r.Counter("llmgw_stream_frames_total",
			"SSE frames forwarded to clients."),
		StreamTruncated: r.Counter("llmgw_stream_truncated_total",
			"Streams that ended without the terminating sentinel, i.e. an upstream that died mid-response."),
		StreamAbandoned: r.Counter("llmgw_stream_abandoned_total",
			"Streams abandoned because the client could not keep up or disconnected."),
	}
}

// Registry returns the underlying registry, for the /metrics handler.
func (g *Gateway) Registry() *Registry { return g.reg }

// ObserveSeconds is a small helper so callers pass a Duration and the metric
// stays in the seconds unit Prometheus conventions require.
//
// Recording milliseconds into a metric named _seconds is a classic and
// surprisingly durable bug: every dashboard and alert threshold silently becomes
// wrong by 1000x, and nothing fails. Funnelling every duration through one
// helper is how this codebase avoids having that conversion written out in
// twenty places.
func ObserveSeconds(h *Histogram, d time.Duration) {
	if h == nil {
		return
	}
	h.Observe(d.Seconds())
}

// StatusClass maps an HTTP status to a low-cardinality label value.
//
// Using the exact status code as a label would be finer-grained and would also
// let a client mint new series at will by provoking unusual codes. The class is
// what alerting actually keys on.
func StatusClass(code int) string {
	switch {
	case code >= 200 && code < 300:
		return "2xx"
	case code >= 300 && code < 400:
		return "3xx"
	case code >= 400 && code < 500:
		return "4xx"
	case code >= 500:
		return "5xx"
	default:
		return "other"
	}
}
