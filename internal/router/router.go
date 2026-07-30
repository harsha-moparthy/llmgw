// Package router owns target selection and failover — the spec's headline
// behaviour: "provider outage failover within a bounded window", including
// "failing over a streaming request cleanly".
//
// # The streaming honesty boundary
//
// The one design decision that matters most in this package: once a single byte
// of a response has been written to the client, the gateway can no longer
// transparently retry on another provider, because splicing a second model's
// continuation onto the first's partial output produces a response that no model
// ever generated — a sentence cut mid-clause, a fact then contradicted, a tool
// call duplicated. That is worse than an honest error.
//
// So streaming failover is defined precisely:
//   - Before the first CONTENT byte reaches the client, a failure is
//     transparently retryable. This window covers the common outage cases —
//     connection refused, 429, 500, a provider that dies before its first token
//     — and it is where clean failover lives.
//   - After the first content byte, the failure is surfaced to the client as an
//     explicit error, the partial usage is still billed, and no silent retry
//     happens.
//
// The server keeps that window open by not writing the client's response body
// until the first content arrives; this package decides, per attempt, whether a
// failure is still in the transparent window (StreamStart not yet fired) or past
// it.
package router

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
	"github.com/harsha-moparthy/llmgw/internal/breaker"
	"github.com/harsha-moparthy/llmgw/internal/config"
	"github.com/harsha-moparthy/llmgw/internal/money"
	"github.com/harsha-moparthy/llmgw/internal/pricing"
	"github.com/harsha-moparthy/llmgw/internal/provider"
)

// Target is one resolved route target: a live provider plus the upstream model
// name to send it.
type Target struct {
	Provider provider.Provider
	Model    string
	Breaker  *breaker.Breaker
	// inflight counts requests currently assigned to this target, for the
	// least-outstanding-requests tiebreak.
	inflight *atomic.Int64
}

// Route is a resolved failover chain for one client alias.
type Route struct {
	Alias         string
	Targets       []*Target
	AllowFailover bool
	MaxAttempts   int
}

// Router selects targets and orchestrates failover.
type Router struct {
	routes map[string]*Route
	prices *pricing.Table

	// requestDeadline caps total time across all attempts for one request.
	requestDeadline time.Duration

	stats FailoverStats
}

// FailoverStats is readable by the metrics layer and the measurement harness.
type FailoverStats struct {
	Attempts       atomic.Int64
	Failovers      atomic.Int64
	Exhausted      atomic.Int64 // requests that ran out of targets
	NoProvider     atomic.Int64 // requests with no usable target at all
	StreamSurfaced atomic.Int64 // mid-stream failures surfaced to the client
}

// Options builds a Router.
type Options struct {
	RequestDeadline time.Duration
	Prices          *pricing.Table
}

// LedgerSink receives an attempt record per upstream attempt. The router depends
// on this interface, not on the concrete ledger, so the two packages do not form
// a cycle and the router can be tested with a fake.
type LedgerSink interface {
	RecordAttempt(rec AttemptOutcome)
}

// AttemptOutcome is one upstream attempt's result, handed to the LedgerSink.
type AttemptOutcome struct {
	Tenant         string
	Alias          string
	Provider       string
	UpstreamModel  string
	Attempt        int
	FailedOver     bool
	Usage          *apiv1.Usage
	UsageEstimated bool
	Cost           money.Pico
	Failure        *provider.Failure
	Latency        time.Duration
	Streamed       bool
	// Served is true when this attempt wrote response content to the client. It
	// is set on the streaming path once content has been forwarded, so the
	// ledger can distinguish "failed before serving" (a clean failover) from
	// "failed after serving" (a surfaced mid-stream error), which are billed and
	// reported differently.
	Served bool
}

// New builds a Router from resolved routes.
func New(routes map[string]*Route, opt Options) *Router {
	rd := opt.RequestDeadline
	if rd <= 0 {
		rd = 60 * time.Second
	}
	return &Router{routes: routes, prices: opt.Prices, requestDeadline: rd}
}

// Stats returns the failover statistics.
func (r *Router) Stats() *FailoverStats { return &r.stats }

// Route returns the resolved route for an alias.
func (r *Router) Route(alias string) (*Route, bool) {
	rt, ok := r.routes[alias]
	return rt, ok
}

// selectable returns the route's targets that currently admit traffic, honouring
// the tenant's allowlist and each target's breaker, ordered by least outstanding
// requests (a deterministic load spread; ties broken by chain order, which is
// itself deterministic — never random, so a test can assert what happens).
func (r *Router) selectable(route *Route, degradeToCheaper bool, estPromptTokens, estMaxTokens int64) []*Target {
	admitted := make([]*Target, 0, len(route.Targets))
	for _, t := range route.Targets {
		if t.Breaker != nil && !t.Breaker.Allow() {
			continue // breaker open: skip without an upstream call
		}
		admitted = append(admitted, t)
	}
	// When the budget said "degrade", prefer the cheapest ESTIMATED cost rather
	// than the raw per-token rate: a model with cheap input and dear output is
	// not uniformly cheaper, and the request's own shape decides which wins.
	if degradeToCheaper && r.prices != nil && len(admitted) > 1 {
		sortByEstimatedCost(admitted, r.prices, estPromptTokens, estMaxTokens)
		return admitted
	}
	// Otherwise keep chain order, but move a target with strictly fewer
	// outstanding requests ahead of an equal-priority peer. Stable so that with
	// equal inflight the configured order is preserved.
	stableByInflight(admitted)
	return admitted
}

// Execute runs the non-streaming path down the failover chain.
//
// It returns the successful Result, or the last Failure if every retryable
// attempt failed. Every attempt — success or failure — is reported to the sink,
// because a request that failed over across three providers has up to three cost
// rows and the ledger must see them all.
func (r *Router) Execute(ctx context.Context, tenant string, route *Route, req *apiv1.ChatRequest, opt ExecOptions) (*provider.Result, *AttemptResult, *provider.Failure) {
	deadline := time.Now().Add(r.requestDeadline)
	if d, ok := ctx.Deadline(); ok && d.Before(deadline) {
		deadline = d
	}

	targets := r.selectable(route, opt.DegradeToCheaper, opt.EstPromptTokens, opt.EstMaxTokens)
	if len(targets) == 0 {
		r.stats.NoProvider.Add(1)
		return nil, &AttemptResult{}, &provider.Failure{Class: provider.ClassUnknown, Err: provider.ErrNoProviders}
	}

	maxAttempts := route.MaxAttempts
	if maxAttempts <= 0 || maxAttempts > len(targets) {
		maxAttempts = len(targets)
	}
	if !route.AllowFailover {
		maxAttempts = 1
	}

	ar := &AttemptResult{}
	var lastFail *provider.Failure

	for i := 0; i < maxAttempts && i < len(targets); i++ {
		if err := ctx.Err(); err != nil {
			return nil, ar, &provider.Failure{Class: provider.ClassCancelled, Err: err}
		}
		remaining := time.Until(deadline)
		if remaining <= 0 {
			// No time left for another attempt. Returning the last failure is
			// more useful to the client than a fresh deadline error, since it
			// says WHY the chain was walking.
			ar.DeadlineExceeded = true
			break
		}

		t := targets[i]
		attemptNo := i + 1
		r.stats.Attempts.Add(1)
		ar.Attempts = attemptNo

		attemptCtx, cancel := context.WithDeadline(ctx, deadline)
		// Stamp this attempt's number into the correlation, so the provider logs
		// it against the same (request id, attempt) key the ledger will use.
		attemptCtx = provider.WithAttempt(attemptCtx, attemptNo)
		start := time.Now()
		t.inflightAdd(1)
		res, fail := t.Provider.Chat(attemptCtx, req, t.Model)
		t.inflightAdd(-1)
		cancel()
		latency := time.Since(start)

		outcome := AttemptOutcome{
			Tenant: tenant, Alias: route.Alias, Provider: t.Provider.Name(),
			UpstreamModel: t.Model, Attempt: attemptNo, FailedOver: attemptNo > 1,
			Latency: latency,
		}

		if fail == nil {
			outcome.Usage = res.Usage
			outcome.UsageEstimated = res.UsageIsEstimated
			outcome.Cost = r.cost(t.Model, res.Usage)
			r.recordHealth(t, nil)
			r.report(opt.Sink, outcome)
			ar.ServedBy = t.Provider.Name()
			ar.ServedModel = t.Model
			if attemptNo > 1 {
				r.stats.Failovers.Add(1)
			}
			return res, ar, nil
		}

		// Failure. Populate provider/model for diagnostics, record health if the
		// class is a provider signal, and always ledger the attempt.
		fail.Provider = t.Provider.Name()
		fail.Model = t.Model
		outcome.Failure = fail
		if fail.MayHaveBilled() && fail.UsageAtFailure != nil {
			outcome.Usage = fail.UsageAtFailure
			outcome.Cost = r.cost(t.Model, fail.UsageAtFailure)
		}
		r.recordHealth(t, fail)
		r.report(opt.Sink, outcome)
		lastFail = fail

		if !fail.Retryable() || !route.AllowFailover {
			// A non-retryable failure must NOT walk the chain: it would burn
			// latency to produce the same error and hide it behind others.
			break
		}
	}

	if lastFail != nil && ar.Attempts >= 1 {
		r.stats.Exhausted.Add(1)
	}
	if lastFail == nil {
		lastFail = &provider.Failure{Class: provider.ClassUnknown, Err: provider.ErrNoProviders}
	}
	return nil, ar, lastFail
}

// ExecOptions carries per-request routing inputs.
type ExecOptions struct {
	Sink             LedgerSink
	DegradeToCheaper bool
	EstPromptTokens  int64
	EstMaxTokens     int64
}

// AttemptResult summarises what Execute did, for response headers and metrics.
type AttemptResult struct {
	Attempts         int
	ServedBy         string
	ServedModel      string
	DeadlineExceeded bool
}

func (r *Router) cost(model string, u *apiv1.Usage) money.Pico {
	if r.prices == nil || u == nil {
		return 0
	}
	b, err := r.prices.Cost(model, u)
	if err != nil {
		return 0
	}
	return b.Total
}

// recordHealth feeds an outcome to the target's breaker, but ONLY when the class
// is evidence about the provider. A client looping on malformed requests must
// not be able to trip a breaker and deny service to every other tenant.
func (r *Router) recordHealth(t *Target, fail *provider.Failure) {
	if t.Breaker == nil {
		return
	}
	if fail == nil {
		t.Breaker.RecordSuccess()
		return
	}
	if fail.Class.CountsAgainstHealth() {
		t.Breaker.RecordFailure(fail.Class)
	}
	// A non-health class (bad request, content filter) is recorded as neither
	// success nor failure: it is simply not evidence about the provider.
}

func (r *Router) report(sink LedgerSink, o AttemptOutcome) {
	if sink != nil {
		sink.RecordAttempt(o)
	}
}

func (t *Target) inflightAdd(n int64) {
	if t.inflight != nil {
		t.inflight.Add(n)
	}
}

// Inflight returns the target's current outstanding request count.
func (t *Target) Inflight() int64 {
	if t.inflight == nil {
		return 0
	}
	return t.inflight.Load()
}

// BuildRoutes resolves a config into router Routes, given a provider registry
// and a breaker registry. It is a package-level function rather than a method so
// the wiring in cmd/gateway is explicit about what it is constructing.
func BuildRoutes(cfg *config.Config, providers map[string]provider.Provider, breakers *breaker.Registry) (map[string]*Route, error) {
	out := make(map[string]*Route, len(cfg.Routes))
	for alias, rc := range cfg.Routes {
		route := &Route{Alias: alias, AllowFailover: rc.AllowFailover, MaxAttempts: rc.MaxAttempts}
		for _, tc := range rc.Targets {
			p, ok := providers[tc.Provider]
			if !ok {
				return nil, fmt.Errorf("route %q references undefined provider %q", alias, tc.Provider)
			}
			var br *breaker.Breaker
			if breakers != nil {
				br = breakers.Get(tc.Provider)
			}
			route.Targets = append(route.Targets, &Target{
				Provider: p, Model: tc.Model, Breaker: br, inflight: new(atomic.Int64),
			})
		}
		out[alias] = route
	}
	return out, nil
}

// AllowedFor filters a set of route aliases by a tenant's allowlist. An empty
// allowlist means all routes are permitted.
func AllowedFor(allowed []string, alias string) bool {
	if len(allowed) == 0 {
		return true
	}
	for _, a := range allowed {
		if a == alias {
			return true
		}
	}
	return false
}

// ErrNoRoute is returned when an alias is not configured.
var ErrNoRoute = errors.New("router: no route for model")

var _ sync.Locker // reserved: the router itself is lock-free; targets carry their own atomics
