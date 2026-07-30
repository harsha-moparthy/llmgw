package server

import (
	"errors"
	"net/http"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
	"github.com/harsha-moparthy/llmgw/internal/budget"
	"github.com/harsha-moparthy/llmgw/internal/cache"
	"github.com/harsha-moparthy/llmgw/internal/config"
	"github.com/harsha-moparthy/llmgw/internal/ledger"
	"github.com/harsha-moparthy/llmgw/internal/metrics"
	"github.com/harsha-moparthy/llmgw/internal/money"
	"github.com/harsha-moparthy/llmgw/internal/provider"
	"github.com/harsha-moparthy/llmgw/internal/router"
	"github.com/harsha-moparthy/llmgw/internal/tokens"
)

// Response headers the gateway sets so a client (and the benchmark harness) can
// see what actually happened without a separate query.
const (
	HeaderProvider   = "X-Llmgw-Provider"
	HeaderModel      = "X-Llmgw-Upstream-Model"
	HeaderAttempts   = "X-Llmgw-Attempts"
	HeaderOverheadUs = "X-Llmgw-Overhead-Us"
	HeaderUpstreamUs = "X-Llmgw-Upstream-Us"
	HeaderCache      = "X-Llmgw-Cache"
	HeaderRequestID  = "X-Llmgw-Request-Id"
)

// handleChat is the main request path. Each step is ordered deliberately and the
// ordering is load-bearing; see the inline comments.
func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	start := s.now()

	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, apiv1.ErrTypeInvalidRequest, "", "method not allowed")
		return
	}

	// 1. Body size limit BEFORE decoding. Decoding an unbounded body is a memory
	// DoS: a client can stream gigabytes into json.Decode before any of the
	// checks below run. MaxBytesReader caps it at the source.
	maxBytes := s.deps.Config.Server.MaxRequestBytes
	if maxBytes <= 0 {
		maxBytes = 1 << 20
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxBytes)

	// 2. Authenticate, constant-time. Before we spend any effort parsing.
	tenant, ok := s.tenantFromAuth(r)
	if !ok {
		s.recordRequestMetric("", "", http.StatusUnauthorized, start)
		writeError(w, http.StatusUnauthorized, apiv1.ErrTypeAuth, "", "invalid or missing API key")
		return
	}

	// 3. Decode + validate.
	var req apiv1.ChatRequest
	if err := decodeBody(r, &req); err != nil {
		status := http.StatusBadRequest
		if isBodyTooLarge(err) {
			status = http.StatusRequestEntityTooLarge
		}
		s.recordRequestMetric(tenant.ID, "", status, start)
		writeError(w, status, apiv1.ErrTypeInvalidRequest, "", err.Error())
		return
	}
	if err := req.Validate(); err != nil {
		s.recordRequestMetric(tenant.ID, req.Model, http.StatusBadRequest, start)
		writeError(w, http.StatusBadRequest, apiv1.ErrTypeInvalidRequest, "", err.Error())
		return
	}

	alias := req.Model

	// 4. Authorise the model against the tenant's allowlist. 404, not 403: a
	// tenant should not be able to probe which models exist but are forbidden to
	// it, so an unknown route and a forbidden one look the same from outside.
	route, exists := s.deps.Router.Route(alias)
	if !exists || !router.AllowedFor(tenant.AllowedModels, alias) {
		s.recordRequestMetric(tenant.ID, alias, http.StatusNotFound, start)
		writeError(w, http.StatusNotFound, apiv1.ErrTypeInvalidRequest, "model_not_found",
			"model "+alias+" is not available to this tenant")
		return
	}

	// 5. Estimate prompt and completion tokens for the budget pre-flight. The
	// estimate is conservative by construction (see internal/tokens): it can
	// only over-reserve, which can reject a request that would have fit but can
	// never admit one that will not.
	prior := s.priorFor(tenant.ID, route)
	estReq := tokens.EstimateAll(&req, alias, prior)
	estPrompt := int64(estReq.Prompt.TokenCount)
	estMax := int64(estReq.Completion.TokenCount)
	estCost := s.estimateCost(route, estPrompt, estMax)

	// 6. Budget reserve. Rich decision so a rejection can explain itself.
	var reservation budget.Reservation
	degrade := false
	if s.deps.Budget != nil {
		res, decision := s.deps.Budget.Reserve(tenant.ID, estCost)
		switch decision.Outcome {
		case budget.Reject:
			s.ledgerReject(tenant.ID, alias, start)
			if s.deps.Metrics != nil {
				s.deps.Metrics.BudgetRejections.With(metrics.Labels{"tenant": tenant.ID}).Inc()
			}
			s.writeBudgetRejection(w, decision)
			s.recordRequestMetric(tenant.ID, alias, decision.HTTPStatus(), start)
			return
		case budget.AllowDegraded:
			degrade = true
			if s.deps.Metrics != nil {
				s.deps.Metrics.BudgetDegraded.With(metrics.Labels{"tenant": tenant.ID}).Inc()
			}
		}
		reservation = res
	}
	// From here on the reservation must be resolved on every path — committed
	// with the real cost on success, released on failure — or it leaks and
	// eventually rejects everything. A defer guarantees it.
	committed := false
	defer func() {
		if !committed && reservation.Valid() && s.deps.Budget != nil {
			_ = s.deps.Budget.Release(reservation)
		}
	}()

	reqID := s.requestID()

	// 7. Cache lookup, tenant-scoped. A hit costs nothing upstream and is served
	// directly, including as a synthetic stream for streaming clients.
	scope := s.cacheScope(tenant)
	if s.deps.Cache != nil {
		if cacheable, _ := s.deps.Cache.Cacheable(&req); cacheable {
			key := cache.ComputeKey(&req, scope)
			if entry, hit := s.deps.Cache.Get(key); hit {
				s.recordCacheLookup(true)
				actual := reservation
				if reservation.Valid() && s.deps.Budget != nil {
					// A cache hit consumed no upstream budget, so commit zero to
					// release the whole hold cleanly.
					_, _ = s.deps.Budget.Commit(reservation, 0)
					committed = true
				}
				_ = actual
				s.serveCacheHit(w, &req, entry, reqID, start)
				s.ledgerCacheHit(tenant.ID, alias, entry, reqID, start)
				return
			}
			s.recordCacheLookup(false)
		}
	}

	// 8+9. Route (with failover), then true-up the budget and write the ledger.
	collector := &attemptCollector{}
	execOpt := router.ExecOptions{
		Sink:             collector,
		DegradeToCheaper: degrade,
		EstPromptTokens:  estPrompt,
		EstMaxTokens:     estMax,
	}

	if req.Stream {
		s.serveStreaming(w, r, tenant, route, &req, reqID, reservation, &committed, collector, execOpt, scope, start)
		return
	}
	s.serveNonStreaming(w, r, tenant, route, &req, reqID, reservation, &committed, collector, execOpt, scope, start)
}

// serveNonStreaming runs the non-streaming path.
func (s *Server) serveNonStreaming(w http.ResponseWriter, r *http.Request, tenant tenantT, route *router.Route, req *apiv1.ChatRequest, reqID string, reservation budget.Reservation, committed *bool, collector *attemptCollector, execOpt router.ExecOptions, scope cache.Scope, start time.Time) {
	// Seed the correlation with the request id; the router overwrites the attempt
	// number per attempt (provider.WithAttempt), so the number a provider logs is
	// the number the ledger records. That pairing is what the reconciliation
	// joins on — see TestAttemptNumberIsStampedPerAttempt.
	ctx := provider.WithCorrelation(r.Context(), provider.Correlation{RequestID: reqID, Attempt: 1})

	res, ar, fail := s.deps.Router.Execute(ctx, tenant.ID, route, req, execOpt)

	s.writeAttemptHeaders(w, ar, start)
	upstreamUs, billedCost := s.flushLedger(collector, reqID, req.Model, tokenEstimate{prompt: execOpt.EstPromptTokens, max: execOpt.EstMaxTokens}, start)

	if fail != nil {
		status := statusForFailure(fail)
		if s.deps.Budget != nil && reservation.Valid() {
			// A failed request is NOT automatically free. Some failures — a
			// timeout, a 5xx after generation began, a client cancellation —
			// are billed by the provider, and flushLedger has just recorded
			// exactly that cost (measured where the provider reported usage,
			// estimated where it did not). Releasing the whole hold here would
			// let a tenant burn unlimited budget through a failing upstream and
			// never be rejected, because the ledger would show spend the budget
			// never saw. So commit what was actually charged; Commit(0) is
			// equivalent to a full release when nothing was billed.
			_, _ = s.deps.Budget.Commit(reservation, billedCost)
			*committed = true
		}
		s.recordRequestMetric(tenant.ID, req.Model, status, start)
		writeError(w, status, apiv1.ErrTypeUpstream, fail.Class.String(), fail.Error())
		return
	}

	// True-up: commit the ACTUAL cost.
	actualCost := s.costOf(ar.ServedModel, res.Usage)
	if s.deps.Budget != nil && reservation.Valid() {
		_, _ = s.deps.Budget.Commit(reservation, actualCost)
		*committed = true
	}
	// Update the completion-length prior with the real output length, so future
	// estimates for this tenant/model sharpen.
	s.updatePrior(tenant.ID, ar.ServedModel, res.Usage)

	// Populate the cache when cacheable.
	s.maybeCache(req, res, ar, scope, actualCost)

	s.recordSuccessMetrics(tenant.ID, req.Model, ar, res, actualCost, upstreamUs, start)
	s.setOverheadHeader(w, upstreamUs, start)

	writeJSON(w, http.StatusOK, res.Response)
}

// --- small helpers shared by the paths ---

// tenantT is an alias so signatures read cleanly.
type tenantT = config.Tenant

func (s *Server) priorFor(tenantID string, route *router.Route) *tokens.Prior {
	if s.deps.Priors == nil {
		return nil
	}
	model := route.Alias
	if len(route.Targets) > 0 {
		model = route.Targets[0].Model
	}
	// Lookup, never create: model comes from the request body, and a
	// get-or-create here would let a caller allocate a prior per random model
	// name on the admission path. A nil prior is fine — the estimator falls back
	// to its conservative default.
	return s.deps.Priors.Lookup(tenantID, model)
}

func (s *Server) updatePrior(tenantID, model string, u *apiv1.Usage) {
	if s.deps.Priors == nil || u == nil {
		return
	}
	// Observe runs only after a request has completed upstream, which is the one
	// path allowed to allocate a new prior.
	_ = s.deps.Priors.ObserveUsage(tenantID, model, u)
}

func (s *Server) estimateCost(route *router.Route, promptTokens, maxTokens int64) money.Pico {
	if s.deps.Prices == nil || len(route.Targets) == 0 {
		return 0
	}
	// Estimate against the FIRST target — the one that will serve unless it is
	// down — since that is the cost the request will most likely incur.
	est, err := s.deps.Prices.Estimate(route.Targets[0].Model, promptTokens, maxTokens)
	if err != nil {
		return 0
	}
	return est
}

func (s *Server) costOf(model string, u *apiv1.Usage) money.Pico {
	if s.deps.Prices == nil || u == nil {
		return 0
	}
	b, err := s.deps.Prices.Cost(model, u)
	if err != nil {
		return 0
	}
	return b.Total
}

func (s *Server) cacheScope(t tenantT) cache.Scope {
	if t.CachePool != "" {
		return cache.SharedPoolScope(t.CachePool)
	}
	return cache.TenantScope(t.ID)
}

// flushLedger writes every collected attempt to the ledger and returns the
// upstream latency of the served attempt (for the overhead header).
//
// est is the request's pre-flight token estimate, used for the one case that
// would otherwise under-bill: an attempt that failed as billable (a timeout or
// a client cancellation after the provider began generating) but reported no
// usage. The provider charged for those tokens; recording zero would make the
// gateway's total drift below the invoice exactly when things go wrong. So such
// a row is written with an ESTIMATED cost, flagged SourceEstimated, which is the
// ledger's documented NeedsEstimate contract.
// It returns the served attempt's upstream latency and the TOTAL cost the ledger
// attributed to this request across every attempt — which the caller commits
// against the tenant's budget, so a billed failure is not silently free.
func (s *Server) flushLedger(collector *attemptCollector, reqID, requestedModel string, est tokenEstimate, now time.Time) (int64, money.Pico) {
	collector.mu.Lock()
	outcomes := collector.outcomes
	collector.mu.Unlock()

	var upstreamUs int64
	var billed money.Pico
	for _, o := range outcomes {
		if o.Failure == nil {
			upstreamUs = o.Latency.Microseconds()
		}
		if s.deps.Ledger == nil {
			continue
		}
		entry := ledgerFromOutcome(o, reqID, requestedModel, s.deps.Prices, s.now())
		s.applyEstimateIfBilled(&entry, o, est)
		if err := s.deps.Ledger.Append(entry); err != nil {
			s.deps.Logger.Warn("ledger append failed", "err", err, "request_id", reqID)
		}
		if sum, err := money.Add(billed, entry.CostPico); err == nil {
			billed = sum
		}
		s.recordAttemptMetrics(o)
	}
	return upstreamUs, billed
}

// tokenEstimate is the request's pre-flight estimate, carried to the ledger flush
// for the billable-but-usage-less case.
type tokenEstimate struct {
	prompt int64
	max    int64
}

// applyEstimateIfBilled fills an estimated cost into a failed row that the
// provider plausibly billed but reported no usage for.
func (s *Server) applyEstimateIfBilled(e *ledger.Entry, o router.AttemptOutcome, est tokenEstimate) {
	if o.Failure == nil || o.Usage != nil {
		return // a success, or a failure that already carried usage, needs no estimate
	}
	rec := ledger.ClassifyFailure(o.Failure, o.FailedOver)
	if !rec.NeedsEstimate || s.deps.Prices == nil {
		return
	}
	// Price the estimated prompt+completion at the attempted model's rate. The
	// estimate is an over-estimate by construction (see internal/tokens), which
	// is the safe direction for a bill: it can overstate what a cancelled request
	// cost, never understate it. The row is flagged estimated so a reconciliation
	// cannot mistake it for a measured charge.
	estUsage := &apiv1.Usage{
		PromptTokens:     int(est.prompt),
		CompletionTokens: int(est.max),
		TotalTokens:      int(est.prompt + est.max),
	}
	b, err := s.deps.Prices.Cost(o.UpstreamModel, estUsage)
	if err != nil {
		return
	}
	e.Tokens = ledger.FromUsage(estUsage)
	e.CostPico = b.Total
	e.Breakdown = ledger.CostBreakdown{
		PromptPico:     b.InputCost,
		CachedPico:     b.CachedInputCost,
		CompletionPico: b.OutputCost,
	}
	e.UsageSource = ledger.SourceEstimated
	e.Billable = b.Total > 0
}

func statusForFailure(f *provider.Failure) int {
	switch f.Class {
	case provider.ClassRateLimit:
		return http.StatusTooManyRequests
	case provider.ClassAuth:
		return http.StatusBadGateway // our upstream credential problem, not the client's
	case provider.ClassBadRequest, provider.ClassContextLength:
		return http.StatusBadRequest
	case provider.ClassContentFilter:
		return http.StatusBadRequest
	case provider.ClassCancelled:
		return 499 // client closed request (nginx convention)
	default:
		return http.StatusBadGateway
	}
}

func isBodyTooLarge(err error) bool {
	var maxErr *http.MaxBytesError
	return errors.As(err, &maxErr)
}
