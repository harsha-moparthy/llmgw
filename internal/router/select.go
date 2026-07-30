package router

import (
	"sort"

	"github.com/harsha-moparthy/llmgw/internal/pricing"
)

// stableByInflight orders targets by outstanding request count ascending, stably
// so that equal-inflight targets keep their configured chain order.
//
// Least-outstanding-requests is chosen over round-robin or random because it is
// both a good load spread AND deterministic: a test can set up the inflight
// counts and assert exactly which target is chosen, which random selection makes
// impossible. Determinism in the selection policy is what lets the failover
// tests assert "the second provider was tried", rather than hoping.
func stableByInflight(targets []*Target) {
	sort.SliceStable(targets, func(i, j int) bool {
		return targets[i].Inflight() < targets[j].Inflight()
	})
}

// sortByEstimatedCost orders targets cheapest-first by the ESTIMATED cost of
// THIS request, not by raw per-token rate.
//
// The distinction matters: a model with cheap input and expensive output is not
// uniformly cheaper than one with the reverse, so "which is cheaper" depends on
// the request's own prompt/completion shape. A budget-degraded request should go
// to whichever target actually costs less for what is being asked, and that can
// only be known by pricing the specific request. A target whose model is
// unpriced sorts last, because routing a degraded (cost-sensitive) request to a
// model whose cost is unknown defeats the purpose of degrading.
func sortByEstimatedCost(targets []*Target, prices *pricing.Table, promptTokens, maxTokens int64) {
	type scored struct {
		t    *Target
		cost int64
		ok   bool
	}
	scoredTargets := make([]scored, len(targets))
	for i, t := range targets {
		est, err := prices.Estimate(t.Model, promptTokens, maxTokens)
		scoredTargets[i] = scored{t: t, cost: int64(est), ok: err == nil}
	}
	sort.SliceStable(scoredTargets, func(i, j int) bool {
		a, b := scoredTargets[i], scoredTargets[j]
		// Priced targets before unpriced ones.
		if a.ok != b.ok {
			return a.ok
		}
		if !a.ok && !b.ok {
			return false // keep order among unpriced
		}
		return a.cost < b.cost
	})
	for i := range scoredTargets {
		targets[i] = scoredTargets[i].t
	}
}
