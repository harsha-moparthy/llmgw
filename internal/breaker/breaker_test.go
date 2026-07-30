package breaker

import (
	"errors"
	"fmt"
	"math/rand/v2"
	"sync"
	"testing"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/provider"
)

// fakeClock is a mutex-guarded manual clock. Guarded rather than plain because
// the concurrency tests read it from many goroutines, and a data race in the
// test harness would be reported as a race in the code under test.
type fakeClock struct {
	mu sync.Mutex
	t  time.Time
}

func newClock() *fakeClock {
	return &fakeClock{t: time.Date(2026, 3, 1, 12, 0, 0, 0, time.UTC)}
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.t = c.t.Add(d)
}

// testConfig is DefaultConfig with the clock pinned and jitter removed, so
// every timing assertion below is an equality rather than a range.
func testConfig(clk *fakeClock) Config {
	cfg := DefaultConfig()
	cfg.Now = clk.Now
	cfg.Jitter = func() float64 { return 0 }
	return cfg
}

func newTestBreaker(t *testing.T, cfg Config) *Breaker {
	t.Helper()
	b, err := New("openai-primary", cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return b
}

// recordFailures drives n counting failures through the breaker, honouring
// Allow so that the sequence is what the real request path would produce.
func recordFailures(b *Breaker, n int, class provider.Class) {
	for i := 0; i < n; i++ {
		if !b.Allow() {
			continue
		}
		b.RecordFailure(class)
	}
}

func recordSuccesses(b *Breaker, n int) {
	for i := 0; i < n; i++ {
		if !b.Allow() {
			continue
		}
		b.RecordSuccess()
	}
}

func TestConfigValidate(t *testing.T) {
	base := DefaultConfig()
	mut := func(f func(*Config)) Config {
		c := base
		f(&c)
		return c
	}

	tests := []struct {
		name    string
		cfg     Config
		wantErr bool
	}{
		{"default is valid", base, false},
		{"zero value rejected", Config{}, true},
		{"window zero", mut(func(c *Config) { c.WindowSize = 0 }), true},
		{"window negative", mut(func(c *Config) { c.WindowSize = -1 }), true},
		{"min samples zero", mut(func(c *Config) { c.MinSamples = 0 }), true},
		// The whole point of MinSamples is that it is reachable; if it exceeds
		// the window the breaker can never trip and is decorative.
		{"min samples exceeds window", mut(func(c *Config) { c.MinSamples = 21 }), true},
		{"min samples equals window ok", mut(func(c *Config) { c.MinSamples = 20 }), false},
		{"ratio zero", mut(func(c *Config) { c.FailureRatio = 0 }), true},
		{"ratio negative", mut(func(c *Config) { c.FailureRatio = -0.1 }), true},
		{"ratio one ok", mut(func(c *Config) { c.FailureRatio = 1 }), false},
		{"ratio above one", mut(func(c *Config) { c.FailureRatio = 1.01 }), true},
		{"cooldown zero", mut(func(c *Config) { c.Cooldown = 0 }), true},
		{"max cooldown below cooldown", mut(func(c *Config) { c.MaxCooldown = time.Second }), true},
		{"max cooldown equal ok", mut(func(c *Config) { c.MaxCooldown = c.Cooldown }), false},
		{"backoff below one", mut(func(c *Config) { c.BackoffFactor = 0.9 }), true},
		{"backoff exactly one ok", mut(func(c *Config) { c.BackoffFactor = 1 }), false},
		{"jitter negative", mut(func(c *Config) { c.JitterFraction = -0.1 }), true},
		{"jitter zero ok", mut(func(c *Config) { c.JitterFraction = 0 }), false},
		{"jitter one ok", mut(func(c *Config) { c.JitterFraction = 1 }), false},
		{"jitter above one", mut(func(c *Config) { c.JitterFraction = 1.5 }), true},
		{"half open probes zero", mut(func(c *Config) { c.HalfOpenProbes = 0 }), true},
		{"half open successes zero", mut(func(c *Config) { c.HalfOpenSuccesses = 0 }), true},
		{"half open timeout zero", mut(func(c *Config) { c.HalfOpenTimeout = 0 }), true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.cfg.Validate()
			if (err != nil) != tt.wantErr {
				t.Fatalf("Validate() err = %v, wantErr = %v", err, tt.wantErr)
			}
			// New must agree with Validate, or a bad config reaches the request
			// path through the constructor.
			_, nerr := New("p", tt.cfg)
			if (nerr != nil) != tt.wantErr {
				t.Fatalf("New() err = %v, wantErr = %v", nerr, tt.wantErr)
			}
		})
	}
}

func TestNewRejectsEmptyName(t *testing.T) {
	if _, err := New("", DefaultConfig()); err == nil {
		t.Fatal("New with empty name: want error, got nil")
	}
}

func TestStateString(t *testing.T) {
	tests := []struct {
		s    State
		want string
	}{
		{StateClosed, "closed"},
		{StateOpen, "open"},
		{StateHalfOpen, "half_open"},
		{State(99), "closed"},
	}
	for _, tt := range tests {
		if got := tt.s.String(); got != tt.want {
			t.Errorf("State(%d).String() = %q, want %q", int(tt.s), got, tt.want)
		}
	}
}

// TestZeroValueStateIsClosed pins the choice that made StateClosed the zero
// value: a Breaker whose state field was never written must pass traffic, not
// reject everything.
func TestZeroValueStateIsClosed(t *testing.T) {
	var s State
	if s != StateClosed {
		t.Fatalf("zero State = %v, want StateClosed", s)
	}
}

// TestNonCountingClassesCannotTripBreaker is the most important test in the
// package. A client that can send a malformed request can send ten thousand of
// them; if those counted as provider health evidence, that client could open
// the breaker on a perfectly healthy provider and deny service to every other
// tenant. 10,000 consecutive rejections must leave the breaker Closed with an
// empty window.
func TestNonCountingClassesCannotTripBreaker(t *testing.T) {
	nonCounting := []provider.Class{
		provider.ClassBadRequest,
		provider.ClassContentFilter,
		provider.ClassContextLength,
		provider.ClassCancelled,
	}

	for _, class := range nonCounting {
		t.Run(class.String(), func(t *testing.T) {
			// Guard against the test silently proving nothing if the frozen
			// contract ever changed underneath it.
			if class.CountsAgainstHealth() {
				t.Fatalf("precondition: %v now counts against health; this test is testing the wrong class", class)
			}

			clk := newClock()
			b := newTestBreaker(t, testConfig(clk))

			const n = 10000
			for i := 0; i < n; i++ {
				if !b.Allow() {
					t.Fatalf("Allow() = false at request %d: breaker rejected traffic on client-caused errors", i)
				}
				b.RecordFailure(class)
			}

			snap := b.Snapshot()
			if snap.State != StateClosed {
				t.Fatalf("state = %v after %d %v failures, want closed", snap.State, n, class)
			}
			if snap.Samples != 0 || snap.Failures != 0 {
				t.Fatalf("window = %d samples / %d failures, want 0/0: non-counting failures must not enter the window", snap.Samples, snap.Failures)
			}
			if snap.Ignored != n {
				t.Fatalf("Ignored = %d, want %d: the drops must be observable, not silent", snap.Ignored, n)
			}
			if snap.Transitions != 0 || snap.Opens != 0 {
				t.Fatalf("transitions = %d, opens = %d, want 0/0", snap.Transitions, snap.Opens)
			}
		})
	}
}

// TestCountingClassesDoTripBreaker is the other half: if the filter above were
// implemented as "ignore everything", the previous test would still pass. Every
// class that provider says is health evidence must actually open the breaker.
func TestCountingClassesDoTripBreaker(t *testing.T) {
	counting := []provider.Class{
		provider.ClassUnknown,
		provider.ClassConnect,
		provider.ClassTimeout,
		provider.ClassRateLimit,
		provider.ClassUpstream5xx,
		provider.ClassOverloaded,
		provider.ClassAuth,
	}

	for _, class := range counting {
		t.Run(class.String(), func(t *testing.T) {
			if !class.CountsAgainstHealth() {
				t.Fatalf("precondition: %v no longer counts against health", class)
			}
			clk := newClock()
			b := newTestBreaker(t, testConfig(clk))
			recordFailures(b, 20, class)
			if got := b.State(); got != StateOpen {
				t.Fatalf("state = %v after 20 %v failures, want open", got, class)
			}
		})
	}
}

// TestMixedTenantsCannotBeStarvedByOneBadClient is the scenario the class filter
// exists for, played out end to end: a flood of malformed requests interleaved
// with a healthy tenant's successful traffic.
func TestMixedTenantsCannotBeStarvedByOneBadClient(t *testing.T) {
	clk := newClock()
	b := newTestBreaker(t, testConfig(clk))

	for i := 0; i < 5000; i++ {
		// Bad tenant.
		if !b.Allow() {
			t.Fatalf("healthy provider taken out of rotation at iteration %d", i)
		}
		b.RecordFailure(provider.ClassBadRequest)
		// Good tenant.
		if !b.Allow() {
			t.Fatalf("healthy provider taken out of rotation at iteration %d", i)
		}
		b.RecordSuccess()
	}

	snap := b.Snapshot()
	if snap.State != StateClosed {
		t.Fatalf("state = %v, want closed", snap.State)
	}
	if snap.Failures != 0 {
		t.Fatalf("window failures = %d, want 0", snap.Failures)
	}
	if snap.Samples != snap.Successes {
		t.Fatalf("samples = %d but successes = %d", snap.Samples, snap.Successes)
	}
}

// TestMinSampleCountGatesRatio pins the reason MinSamples exists: with ratio
// alone, one failure into a one-sample window is a 100% failure rate and a
// healthy provider loses all its traffic over a single blip.
func TestMinSampleCountGatesRatio(t *testing.T) {
	clk := newClock()

	t.Run("single failure does not trip with min samples", func(t *testing.T) {
		b := newTestBreaker(t, testConfig(clk))
		recordFailures(b, 1, provider.ClassUpstream5xx)
		snap := b.Snapshot()
		if snap.State != StateClosed {
			t.Fatalf("state = %v after one failure, want closed", snap.State)
		}
		if snap.FailureRatio != 1.0 {
			t.Fatalf("FailureRatio = %v, want 1.0: the ratio is genuinely 100%%, which is exactly why it must be gated", snap.FailureRatio)
		}
	})

	t.Run("trips at exactly min samples when all fail", func(t *testing.T) {
		cfg := testConfig(clk)
		b := newTestBreaker(t, cfg)
		recordFailures(b, cfg.MinSamples-1, provider.ClassUpstream5xx)
		if got := b.State(); got != StateClosed {
			t.Fatalf("state = %v after %d failures, want closed", got, cfg.MinSamples-1)
		}
		recordFailures(b, 1, provider.ClassUpstream5xx)
		if got := b.State(); got != StateOpen {
			t.Fatalf("state = %v after %d failures, want open", got, cfg.MinSamples)
		}
	})

	t.Run("min samples of one does trip immediately", func(t *testing.T) {
		// Proves the gate is a real gate and not the only thing preventing the
		// trip: lower MinSamples to 1 and the same single failure opens it.
		cfg := testConfig(clk)
		cfg.MinSamples = 1
		b := newTestBreaker(t, cfg)
		recordFailures(b, 1, provider.ClassUpstream5xx)
		if got := b.State(); got != StateOpen {
			t.Fatalf("state = %v with MinSamples=1, want open", got)
		}
	})
}

// TestRatioTripBoundary walks the ratio comparison across its boundary. The
// integer-permille comparison must be exact at ratios that are not
// representable in binary floating point.
func TestRatioTripBoundary(t *testing.T) {
	tests := []struct {
		name      string
		ratio     float64
		window    int
		min       int
		successes int
		failures  int
		wantOpen  bool
	}{
		// 1/3 quantises to 333 permille, which is *below* 1/3, so 2 of 6
		// (333.3 permille) trips. This is the quantisation the permille
		// comparison documents, and it errs toward tripping — the conservative
		// direction for a breaker.
		{"one third, 2 of 6 trips after quantisation", 1.0 / 3.0, 6, 6, 4, 2, true},
		{"just above one third, 2 of 6 is below", 0.34, 6, 6, 4, 2, false},
		{"just above one third, 3 of 6 trips", 0.34, 6, 6, 3, 3, true},
		{"half, 3 of 6 is exactly at threshold", 0.5, 6, 6, 3, 3, true},
		{"half, 2 of 6 is below", 0.5, 6, 6, 4, 2, false},
		{"ratio 1.0 needs every sample to fail", 1.0, 4, 4, 1, 3, false},
		{"ratio 1.0 with all failures", 1.0, 4, 4, 0, 4, true},
		{"tiny ratio still needs min samples", 0.001, 10, 10, 9, 1, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clk := newClock()
			cfg := testConfig(clk)
			cfg.WindowSize = tt.window
			cfg.MinSamples = tt.min
			cfg.FailureRatio = tt.ratio
			b := newTestBreaker(t, cfg)

			// Successes first so the failures land in a window that is exactly
			// as full as the case describes.
			recordSuccesses(b, tt.successes)
			recordFailures(b, tt.failures, provider.ClassUpstream5xx)

			gotOpen := b.State() == StateOpen
			if gotOpen != tt.wantOpen {
				snap := b.Snapshot()
				t.Fatalf("open = %v, want %v (samples=%d failures=%d ratio=%v)",
					gotOpen, tt.wantOpen, snap.Samples, snap.Failures, snap.FailureRatio)
			}
		})
	}
}

func TestRatioPermille(t *testing.T) {
	tests := []struct {
		in   float64
		want int64
	}{
		{0.5, 500},
		{1.0, 1000},
		{1.0 / 3.0, 333},
		{0.001, 1},
		{0.0001, 0}, // quantised away; below the documented 0.1% resolution.
		{0.9995, 1000},
	}
	for _, tt := range tests {
		if got := ratioPermille(tt.in); got != tt.want {
			t.Errorf("ratioPermille(%v) = %d, want %d", tt.in, got, tt.want)
		}
	}
}

// TestSlidingWindowEvicts checks the ring's eviction accounting directly. A
// failure count that drifted from the ring contents would make the breaker trip
// early or never, and neither is visible from the outside without this test.
func TestSlidingWindowEvicts(t *testing.T) {
	clk := newClock()
	cfg := testConfig(clk)
	cfg.WindowSize = 4
	cfg.MinSamples = 4
	cfg.FailureRatio = 1.0 // only an all-failure window trips
	b := newTestBreaker(t, cfg)

	// Alternate so the ring wraps several times; with ratio 1.0 it must never
	// trip, and the failure count must track the window exactly.
	for i := 0; i < 40; i++ {
		if i%2 == 0 {
			recordFailures(b, 1, provider.ClassTimeout)
		} else {
			recordSuccesses(b, 1)
		}
		snap := b.Snapshot()
		if snap.State != StateClosed {
			t.Fatalf("iteration %d: state = %v, want closed", i, snap.State)
		}
		if snap.Samples > cfg.WindowSize {
			t.Fatalf("iteration %d: samples = %d, exceeds window %d", i, snap.Samples, cfg.WindowSize)
		}
		if snap.Failures+snap.Successes != snap.Samples {
			t.Fatalf("iteration %d: failures %d + successes %d != samples %d",
				i, snap.Failures, snap.Successes, snap.Samples)
		}
		if snap.Failures > snap.Samples {
			t.Fatalf("iteration %d: failures %d exceeds samples %d — eviction accounting drifted",
				i, snap.Failures, snap.Samples)
		}
	}

	// Now four in a row must trip it, proving the alternating traffic left the
	// window in a genuinely usable state rather than a wedged one.
	recordFailures(b, 4, provider.ClassTimeout)
	if got := b.State(); got != StateOpen {
		t.Fatalf("state = %v after 4 consecutive failures with ratio 1.0, want open", got)
	}
}

func TestOpenRejectsWithoutUpstreamCall(t *testing.T) {
	clk := newClock()
	cfg := testConfig(clk)
	b := newTestBreaker(t, cfg)
	recordFailures(b, 20, provider.ClassConnect)

	if got := b.State(); got != StateOpen {
		t.Fatalf("state = %v, want open", got)
	}
	for i := 0; i < 100; i++ {
		if b.Allow() {
			t.Fatalf("Allow() = true while open at attempt %d", i)
		}
	}
	snap := b.Snapshot()
	if snap.Rejected < 100 {
		t.Fatalf("Rejected = %d, want >= 100", snap.Rejected)
	}
	if snap.UntilNextProbe != cfg.Cooldown {
		t.Fatalf("UntilNextProbe = %v, want %v", snap.UntilNextProbe, cfg.Cooldown)
	}
}

func TestOpenToHalfOpenExactlyAtCooldown(t *testing.T) {
	clk := newClock()
	cfg := testConfig(clk)
	b := newTestBreaker(t, cfg)
	recordFailures(b, 20, provider.ClassConnect)

	// One nanosecond short: still Open. This is the assertion that a sleepy test
	// could not make.
	clk.Advance(cfg.Cooldown - time.Nanosecond)
	if got := b.State(); got != StateOpen {
		t.Fatalf("state = %v 1ns before cooldown, want open", got)
	}
	if got := b.UntilNextProbe(); got != time.Nanosecond {
		t.Fatalf("UntilNextProbe = %v, want 1ns", got)
	}

	clk.Advance(time.Nanosecond)
	if got := b.State(); got != StateHalfOpen {
		t.Fatalf("state = %v exactly at cooldown, want half open", got)
	}
	if got := b.UntilNextProbe(); got != 0 {
		t.Fatalf("UntilNextProbe = %v in half open, want 0", got)
	}
}

func TestHalfOpenAdmitsBoundedTrials(t *testing.T) {
	clk := newClock()
	cfg := testConfig(clk)
	cfg.HalfOpenProbes = 3
	cfg.HalfOpenSuccesses = 100 // never close during this test
	b := newTestBreaker(t, cfg)

	recordFailures(b, 20, provider.ClassConnect)
	clk.Advance(cfg.Cooldown)

	for i := 0; i < cfg.HalfOpenProbes; i++ {
		if !b.Allow() {
			t.Fatalf("Allow() = false for trial %d, want the first %d admitted", i, cfg.HalfOpenProbes)
		}
	}
	if b.Allow() {
		t.Fatalf("Allow() = true for trial %d, want rejected past the trial budget", cfg.HalfOpenProbes)
	}
	if got := b.Snapshot().HalfOpenInFlight; got != cfg.HalfOpenProbes {
		t.Fatalf("HalfOpenInFlight = %d, want %d", got, cfg.HalfOpenProbes)
	}

	// Resolving one trial frees exactly one slot.
	b.RecordSuccess()
	if got := b.Snapshot().HalfOpenInFlight; got != cfg.HalfOpenProbes-1 {
		t.Fatalf("HalfOpenInFlight = %d after one resolution, want %d", got, cfg.HalfOpenProbes-1)
	}
	if !b.Allow() {
		t.Fatal("Allow() = false after a trial slot was freed")
	}
}

func TestHalfOpenClosesAfterConsecutiveSuccesses(t *testing.T) {
	clk := newClock()
	cfg := testConfig(clk)
	cfg.HalfOpenSuccesses = 3
	cfg.HalfOpenProbes = 1
	b := newTestBreaker(t, cfg)

	recordFailures(b, 20, provider.ClassConnect)
	clk.Advance(cfg.Cooldown)

	for i := 0; i < cfg.HalfOpenSuccesses-1; i++ {
		if !b.Allow() {
			t.Fatalf("Allow() = false for trial %d", i)
		}
		b.RecordSuccess()
		if got := b.State(); got != StateHalfOpen {
			t.Fatalf("state = %v after %d of %d successes, want half open", got, i+1, cfg.HalfOpenSuccesses)
		}
	}
	if !b.Allow() {
		t.Fatal("Allow() = false for the closing trial")
	}
	b.RecordSuccess()

	snap := b.Snapshot()
	if snap.State != StateClosed {
		t.Fatalf("state = %v after %d successes, want closed", snap.State, cfg.HalfOpenSuccesses)
	}
	if snap.ConsecutiveOpens != 0 {
		t.Fatalf("ConsecutiveOpens = %d after close, want 0: closing must reset the backoff", snap.ConsecutiveOpens)
	}
	if snap.Samples != 0 {
		t.Fatalf("samples = %d after close, want a clean window", snap.Samples)
	}
}

// TestHalfOpenSingleFailureReopensWithBackoff pins the backoff schedule
// exactly: 5s, 10s, 20s, 40s, 80s, then capped at 120s forever.
func TestHalfOpenSingleFailureReopensWithBackoff(t *testing.T) {
	clk := newClock()
	cfg := testConfig(clk)
	cfg.HalfOpenProbes = 1
	b := newTestBreaker(t, cfg)

	want := []time.Duration{
		5 * time.Second,
		10 * time.Second,
		20 * time.Second,
		40 * time.Second,
		80 * time.Second,
		120 * time.Second, // capped by MaxCooldown
		120 * time.Second,
		120 * time.Second,
	}

	// First trip comes from real traffic; the rest from failed trials.
	recordFailures(b, 20, provider.ClassUpstream5xx)
	for i, wantCooldown := range want {
		snap := b.Snapshot()
		if snap.State != StateOpen {
			t.Fatalf("open %d: state = %v, want open", i+1, snap.State)
		}
		if snap.Cooldown != wantCooldown {
			t.Fatalf("open %d: cooldown = %v, want %v", i+1, snap.Cooldown, wantCooldown)
		}
		if snap.ConsecutiveOpens != i+1 {
			t.Fatalf("open %d: ConsecutiveOpens = %d, want %d", i+1, snap.ConsecutiveOpens, i+1)
		}
		if got := snap.NextProbeAt.Sub(snap.OpenedAt); got != wantCooldown {
			t.Fatalf("open %d: NextProbeAt - OpenedAt = %v, want %v", i+1, got, wantCooldown)
		}

		clk.Advance(wantCooldown)
		if got := b.State(); got != StateHalfOpen {
			t.Fatalf("open %d: state = %v after cooldown, want half open", i+1, got)
		}
		if !b.Allow() {
			t.Fatalf("open %d: Allow() = false in half open", i+1)
		}
		// A single trial failure re-opens; no ratio is consulted, because we
		// already have fresh evidence and more traffic would be more damage.
		b.RecordFailure(provider.ClassUpstream5xx)
	}

	// And it must still recover once the provider does, rather than being stuck
	// at the cap.
	clk.Advance(120 * time.Second)
	if !b.Allow() {
		t.Fatal("Allow() = false after the capped cooldown: a recovered provider would never be retried")
	}
	b.RecordSuccess()
	if !b.Allow() {
		t.Fatal("Allow() = false for the second trial")
	}
	b.RecordSuccess()
	if got := b.State(); got != StateClosed {
		t.Fatalf("state = %v, want closed", got)
	}
}

func TestBackoffFactorOneIsFlat(t *testing.T) {
	clk := newClock()
	cfg := testConfig(clk)
	cfg.BackoffFactor = 1
	cfg.HalfOpenProbes = 1
	b := newTestBreaker(t, cfg)

	recordFailures(b, 20, provider.ClassUpstream5xx)
	for i := 0; i < 5; i++ {
		if got := b.Snapshot().Cooldown; got != cfg.Cooldown {
			t.Fatalf("open %d: cooldown = %v, want flat %v", i+1, got, cfg.Cooldown)
		}
		clk.Advance(cfg.Cooldown)
		if !b.Allow() {
			t.Fatalf("open %d: Allow() = false in half open", i+1)
		}
		b.RecordFailure(provider.ClassUpstream5xx)
	}
}

// TestJitterBounds proves jitter is additive: the configured cooldown is a
// floor, and the maximum is cooldown*(1+JitterFraction). A subtractive jitter
// would let the breaker probe before the backoff it just computed.
func TestJitterBounds(t *testing.T) {
	tests := []struct {
		name   string
		jitter float64
		want   time.Duration
	}{
		{"pinned low", 0, 5 * time.Second},
		{"pinned mid", 0.5, 5*time.Second + 500*time.Millisecond},
		{"pinned just below one", 0.999999, 5*time.Second + 999999*time.Microsecond},
		// Out-of-contract jitter sources are clamped rather than trusted: a
		// cooldown must never exceed the documented maximum or dip below the
		// configured floor, whatever the source returns.
		{"clamped above one", 5.0, 6 * time.Second},
		{"clamped below zero", -3.0, 5 * time.Second},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clk := newClock()
			cfg := testConfig(clk)
			cfg.JitterFraction = 0.2
			cfg.Jitter = func() float64 { return tt.jitter }
			b := newTestBreaker(t, cfg)

			recordFailures(b, 20, provider.ClassUpstream5xx)
			got := b.Snapshot().Cooldown

			if got < cfg.Cooldown {
				t.Fatalf("cooldown = %v, below the configured floor %v", got, cfg.Cooldown)
			}
			maxAllowed := time.Duration(float64(cfg.Cooldown) * (1 + cfg.JitterFraction))
			if got > maxAllowed {
				t.Fatalf("cooldown = %v, above the documented maximum %v", got, maxAllowed)
			}
			// Allow a nanosecond of float slop on the fractional cases.
			if d := got - tt.want; d > time.Microsecond || d < -time.Microsecond {
				t.Fatalf("cooldown = %v, want ~%v", got, tt.want)
			}
		})
	}
}

// TestJitterDesynchronisesReplicas is the thundering-herd test. Many gateway
// replicas observe the same outage at the same instant; without jitter they all
// probe the recovering provider at the same instant and re-kill it. With jitter
// their probe times must be spread.
func TestJitterDesynchronisesReplicas(t *testing.T) {
	const replicas = 200
	clk := newClock()

	openAll := func(jitterFrac float64) map[time.Time]int {
		times := make(map[time.Time]int)
		// A seeded PRNG, not the global one: the spread being asserted is a
		// property of the jitter design, and the test should not be able to fail
		// once in a thousand runs on an unlucky global seed.
		r := rand.New(rand.NewPCG(1, 2))
		for i := 0; i < replicas; i++ {
			cfg := testConfig(clk)
			cfg.JitterFraction = jitterFrac
			cfg.Jitter = r.Float64
			b := newTestBreaker(t, cfg)
			recordFailures(b, 20, provider.ClassUpstream5xx)
			times[b.Snapshot().NextProbeAt]++
		}
		return times
	}

	noJitter := openAll(0)
	if len(noJitter) != 1 {
		t.Fatalf("without jitter, distinct probe instants = %d, want 1 (the herd is the baseline this test is contrasting against)", len(noJitter))
	}

	withJitter := openAll(0.2)
	// Nanosecond resolution over a one-second jitter span: near-total spread is
	// expected, and anything below half would mean the jitter is not reaching
	// the deadline.
	if len(withJitter) < replicas/2 {
		t.Fatalf("with jitter, distinct probe instants = %d of %d replicas, want > %d", len(withJitter), replicas, replicas/2)
	}
	for at, n := range withJitter {
		if n > replicas/10 {
			t.Fatalf("%d of %d replicas would probe at the same instant %v: jitter is not breaking the phase lock", n, replicas, at)
		}
	}
}

func TestHalfOpenTimeoutReopens(t *testing.T) {
	clk := newClock()
	cfg := testConfig(clk)
	cfg.HalfOpenProbes = 2
	b := newTestBreaker(t, cfg)

	recordFailures(b, 20, provider.ClassUpstream5xx)
	clk.Advance(cfg.Cooldown)
	if !b.Allow() {
		t.Fatal("Allow() = false in half open")
	}
	// The caller vanishes without ever calling Record — a hung upstream, or a
	// panic above the Record call. Without the timeout the trial slot leaks and
	// the breaker wedges in HalfOpen forever.
	clk.Advance(cfg.HalfOpenTimeout - time.Nanosecond)
	if got := b.State(); got != StateHalfOpen {
		t.Fatalf("state = %v 1ns before the half-open timeout, want half open", got)
	}
	clk.Advance(time.Nanosecond)

	snap := b.Snapshot()
	if snap.State != StateOpen {
		t.Fatalf("state = %v after the half-open timeout, want open", snap.State)
	}
	if snap.HalfOpenInFlight != 0 {
		t.Fatalf("HalfOpenInFlight = %d after re-open, want 0: the leaked slot must be reclaimed", snap.HalfOpenInFlight)
	}
	if snap.ConsecutiveOpens != 2 {
		t.Fatalf("ConsecutiveOpens = %d, want 2: an unresolved trial is the signature of a hung provider and must back off", snap.ConsecutiveOpens)
	}
}

// TestHalfOpenNonCountingFailureDoesNotReopen: a malformed request that happens
// to land on a trial slot must release the slot without deciding anything. If it
// re-opened the breaker, one bad client could hold a provider Open forever by
// racing every HalfOpen window.
func TestHalfOpenNonCountingFailureDoesNotReopen(t *testing.T) {
	clk := newClock()
	cfg := testConfig(clk)
	cfg.HalfOpenProbes = 2
	cfg.HalfOpenSuccesses = 2
	b := newTestBreaker(t, cfg)

	recordFailures(b, 20, provider.ClassUpstream5xx)
	clk.Advance(cfg.Cooldown)

	for i := 0; i < 50; i++ {
		if !b.Allow() {
			t.Fatalf("iteration %d: Allow() = false; trial slots leaked on non-counting outcomes", i)
		}
		b.RecordFailure(provider.ClassBadRequest)
		snap := b.Snapshot()
		if snap.State != StateHalfOpen {
			t.Fatalf("iteration %d: state = %v, want half open", i, snap.State)
		}
		if snap.HalfOpenInFlight != 0 {
			t.Fatalf("iteration %d: HalfOpenInFlight = %d, want 0", i, snap.HalfOpenInFlight)
		}
		if snap.HalfOpenSuccesses != 0 {
			t.Fatalf("iteration %d: HalfOpenSuccesses = %d, want 0: a bad request is not progress toward closing", i, snap.HalfOpenSuccesses)
		}
	}
	if got := b.Snapshot().Ignored; got != 50 {
		t.Fatalf("Ignored = %d, want 50", got)
	}
}

// TestStaleOutcomeWhileOpenIsDropped: outcomes from requests admitted before the
// trip arrive after it. A stale success must not undo the trip.
func TestStaleOutcomeWhileOpenIsDropped(t *testing.T) {
	clk := newClock()
	cfg := testConfig(clk)
	b := newTestBreaker(t, cfg)
	recordFailures(b, 20, provider.ClassUpstream5xx)

	for i := 0; i < 100; i++ {
		b.RecordSuccess()
	}
	snap := b.Snapshot()
	if snap.State != StateOpen {
		t.Fatalf("state = %v after 100 stale successes, want open", snap.State)
	}
	if snap.Samples != 0 {
		t.Fatalf("samples = %d, want 0: stale outcomes must not enter the window", snap.Samples)
	}
	if snap.Stale != 100 {
		t.Fatalf("Stale = %d, want 100", snap.Stale)
	}
	if snap.UntilNextProbe != cfg.Cooldown {
		t.Fatalf("UntilNextProbe = %v, want the cooldown to be untouched at %v", snap.UntilNextProbe, cfg.Cooldown)
	}
}

// TestWindowIsClearedOnTransition: if the trip left the failing window in place,
// the first counting failure after the breaker eventually closed would re-trip
// it instantly, and the breaker could never stay closed after its first outage.
func TestWindowIsClearedOnTransition(t *testing.T) {
	clk := newClock()
	cfg := testConfig(clk)
	cfg.HalfOpenProbes = 1
	b := newTestBreaker(t, cfg)

	recordFailures(b, 20, provider.ClassUpstream5xx)
	if got := b.Snapshot().Samples; got != 0 {
		t.Fatalf("samples = %d immediately after open, want a cleared window", got)
	}

	clk.Advance(cfg.Cooldown)
	for i := 0; i < cfg.HalfOpenSuccesses; i++ {
		if !b.Allow() {
			t.Fatalf("Allow() = false for trial %d", i)
		}
		b.RecordSuccess()
	}
	if got := b.State(); got != StateClosed {
		t.Fatalf("state = %v, want closed", got)
	}

	// A single blip must not re-trip the freshly closed breaker.
	recordFailures(b, 1, provider.ClassUpstream5xx)
	if got := b.State(); got != StateClosed {
		t.Fatalf("state = %v after one post-recovery failure, want closed", got)
	}
	if got := b.Snapshot().Opens; got != 1 {
		t.Fatalf("Opens = %d, want 1", got)
	}
}

func TestRecordNilFailureIsSuccess(t *testing.T) {
	clk := newClock()
	b := newTestBreaker(t, testConfig(clk))
	for i := 0; i < 30; i++ {
		if !b.Allow() {
			t.Fatalf("Allow() = false at %d", i)
		}
		b.Record(nil)
	}
	snap := b.Snapshot()
	if snap.State != StateClosed {
		t.Fatalf("state = %v, want closed", snap.State)
	}
	if snap.Failures != 0 {
		t.Fatalf("failures = %d, want 0", snap.Failures)
	}
}

func TestRecordUsesFailureClass(t *testing.T) {
	tests := []struct {
		name     string
		failure  *provider.Failure
		wantOpen bool
	}{
		{"nil is success", nil, false},
		{"bad request is ignored", &provider.Failure{Class: provider.ClassBadRequest, StatusCode: 400}, false},
		{"5xx counts", &provider.Failure{Class: provider.ClassUpstream5xx, StatusCode: 503}, true},
		{"connect counts", &provider.Failure{Class: provider.ClassConnect, Err: errors.New("dial tcp: refused")}, true},
		{"auth counts", &provider.Failure{Class: provider.ClassAuth, StatusCode: 401}, true},
		{"content filter is ignored", &provider.Failure{Class: provider.ClassContentFilter}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clk := newClock()
			b := newTestBreaker(t, testConfig(clk))
			for i := 0; i < 40; i++ {
				if !b.Allow() {
					break
				}
				b.Record(tt.failure)
			}
			gotOpen := b.State() == StateOpen
			if gotOpen != tt.wantOpen {
				t.Fatalf("open = %v, want %v", gotOpen, tt.wantOpen)
			}
		})
	}
}

func TestWorstCaseFailuresToTrip(t *testing.T) {
	tests := []struct {
		name   string
		window int
		min    int
		ratio  float64
		want   int
	}{
		{"default geometry", 20, 5, 0.5, 10},
		{"min samples dominates", 20, 15, 0.5, 15},
		{"ratio one needs the whole window", 10, 1, 1.0, 10},
		{"tiny ratio still needs min samples", 100, 8, 0.01, 8},
		{"tiny ratio, min samples of one", 100, 1, 0.01, 1},
		{"window of one", 1, 1, 1.0, 1},
		// 2/3 quantises up to 667 permille, so 8 of 12 (666.7) is not quite
		// enough and the ninth failure is required.
		{"two thirds of twelve", 12, 3, 2.0 / 3.0, 9},
		{"three quarters of twelve", 12, 3, 0.75, 9},
		{"invalid config yields zero", 0, 0, 0, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := DefaultConfig()
			cfg.WindowSize = tt.window
			cfg.MinSamples = tt.min
			cfg.FailureRatio = tt.ratio
			if got := cfg.WorstCaseFailuresToTrip(); got != tt.want {
				t.Fatalf("WorstCaseFailuresToTrip() = %d, want %d", got, tt.want)
			}
		})
	}
}

// TestDetectionLatencyBound is the headline measurement. A provider under steady
// load starts failing; the number asserted is the interval on the injected clock
// between the first failure and the breaker opening, which must not exceed
// Config.DetectionBound for that load.
func TestDetectionLatencyBound(t *testing.T) {
	const interval = 20 * time.Millisecond // 50 rps to this provider

	tests := []struct {
		name string
		warm int // successes recorded before the outage begins
	}{
		{"cold breaker, no prior traffic", 0},
		{"partly warmed window", 7},
		{"window full of successes (worst case)", 20},
		{"window wrapped many times", 137},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clk := newClock()
			cfg := testConfig(clk)
			b := newTestBreaker(t, cfg)

			for i := 0; i < tt.warm; i++ {
				clk.Advance(interval)
				if !b.Allow() {
					t.Fatalf("Allow() = false during healthy warmup at %d", i)
				}
				b.RecordSuccess()
			}

			// The bound the breaker itself claims, before the outage starts.
			predicted := b.FailuresToTrip()
			if predicted <= 0 {
				t.Fatalf("FailuresToTrip() = %d before the outage, want > 0", predicted)
			}
			if worst := cfg.WorstCaseFailuresToTrip(); predicted > worst {
				t.Fatalf("FailuresToTrip() = %d exceeds WorstCaseFailuresToTrip() = %d: the advertised bound is wrong", predicted, worst)
			}

			// The outage begins. The provider hard-fails every request.
			var firstFailureAt, openedAt time.Time
			failures := 0
			for i := 0; i < 1000; i++ {
				clk.Advance(interval)
				if !b.Allow() {
					t.Fatalf("Allow() = false at request %d before the breaker opened", i)
				}
				if failures == 0 {
					firstFailureAt = clk.Now()
				}
				b.RecordFailure(provider.ClassUpstream5xx)
				failures++
				if b.State() == StateOpen {
					openedAt = clk.Now()
					break
				}
			}
			if openedAt.IsZero() {
				t.Fatal("breaker never opened under a total outage")
			}

			if failures != predicted {
				t.Fatalf("took %d failures to open, but FailuresToTrip() predicted %d", failures, predicted)
			}

			// Detection latency measured on the injected clock: the interval
			// between the first failure and the trip.
			latency := openedAt.Sub(firstFailureAt)
			wantLatency := time.Duration(failures-1) * interval
			if latency != wantLatency {
				t.Fatalf("detection latency = %v, want exactly %v", latency, wantLatency)
			}
			if bound := cfg.DetectionBound(interval); latency > bound {
				t.Fatalf("detection latency %v exceeds the advertised bound %v", latency, bound)
			}
		})
	}
}

// TestDetectionBoundHoldsForArbitraryWindowStates is the property version: for
// any reachable window contents, driving WorstCaseFailuresToTrip() failures must
// open the breaker. A bound that is merely usually right would pass the
// fixed-warmup cases above and fail here.
func TestDetectionBoundHoldsForArbitraryWindowStates(t *testing.T) {
	geometries := []struct {
		window, min int
		ratio       float64
	}{
		{20, 5, 0.5},
		{10, 10, 1.0},
		{50, 3, 0.25},
		{7, 2, 2.0 / 3.0},
		{1, 1, 1.0},
	}
	r := rand.New(rand.NewPCG(7, 11))

	for _, g := range geometries {
		name := fmt.Sprintf("w%d_m%d_r%.3f", g.window, g.min, g.ratio)
		t.Run(name, func(t *testing.T) {
			cfg := testConfig(newClock())
			cfg.WindowSize, cfg.MinSamples, cfg.FailureRatio = g.window, g.min, g.ratio
			worst := cfg.WorstCaseFailuresToTrip()
			if worst <= 0 {
				t.Fatalf("WorstCaseFailuresToTrip() = %d for a valid config", worst)
			}

			for trial := 0; trial < 300; trial++ {
				b := newTestBreaker(t, cfg)
				// Random reachable prefix of successes and failures that did not
				// itself trip the breaker.
				for i := 0; i < r.IntN(3*g.window+1); i++ {
					if b.State() != StateClosed {
						break
					}
					if r.IntN(2) == 0 {
						b.RecordSuccess()
					} else {
						b.RecordFailure(provider.ClassTimeout)
					}
				}
				if b.State() != StateClosed {
					continue // already tripped; nothing to bound
				}

				predicted := b.FailuresToTrip()
				if predicted < 1 || predicted > worst {
					t.Fatalf("trial %d: FailuresToTrip() = %d, outside [1, %d]", trial, predicted, worst)
				}
				for i := 0; i < predicted-1; i++ {
					b.RecordFailure(provider.ClassTimeout)
					if b.State() != StateClosed {
						t.Fatalf("trial %d: opened after %d failures, predicted %d", trial, i+1, predicted)
					}
				}
				b.RecordFailure(provider.ClassTimeout)
				if b.State() != StateOpen {
					t.Fatalf("trial %d: still closed after the predicted %d failures", trial, predicted)
				}
			}
		})
	}
}

func TestFailuresToTripIsZeroWhenNotClosed(t *testing.T) {
	clk := newClock()
	cfg := testConfig(clk)
	b := newTestBreaker(t, cfg)
	recordFailures(b, 20, provider.ClassUpstream5xx)

	if got := b.FailuresToTrip(); got != 0 {
		t.Fatalf("FailuresToTrip() = %d while open, want 0", got)
	}
	clk.Advance(cfg.Cooldown)
	if b.State() != StateHalfOpen {
		t.Fatal("expected half open")
	}
	if got := b.FailuresToTrip(); got != 0 {
		t.Fatalf("FailuresToTrip() = %d while half open, want 0", got)
	}
	if got := b.Snapshot().FailuresToTrip; got != 0 {
		t.Fatalf("Snapshot().FailuresToTrip = %d while half open, want 0", got)
	}
}

// TestFailuresToTripDoesNotMutateWindow: the projection replays hypothetical
// failures against the live ring. If it wrote to it, reading the metric would
// trip the breaker.
func TestFailuresToTripDoesNotMutateWindow(t *testing.T) {
	clk := newClock()
	b := newTestBreaker(t, testConfig(clk))
	recordSuccesses(b, 12)
	recordFailures(b, 3, provider.ClassTimeout)

	before := b.Snapshot()
	first := b.FailuresToTrip()
	for i := 0; i < 1000; i++ {
		if got := b.FailuresToTrip(); got != first {
			t.Fatalf("FailuresToTrip() = %d on call %d, was %d on the first call: reading the metric mutated state", got, i, first)
		}
	}
	after := b.Snapshot()
	if after.Samples != before.Samples || after.Failures != before.Failures || after.State != before.State {
		t.Fatalf("window changed from %d/%d %v to %d/%d %v after reading FailuresToTrip",
			before.Samples, before.Failures, before.State, after.Samples, after.Failures, after.State)
	}
}

func TestSnapshotFields(t *testing.T) {
	clk := newClock()
	cfg := testConfig(clk)
	b := newTestBreaker(t, cfg)

	recordSuccesses(b, 6)
	recordFailures(b, 2, provider.ClassTimeout)
	recordFailures(b, 3, provider.ClassBadRequest)

	snap := b.Snapshot()
	if snap.Name != "openai-primary" {
		t.Errorf("Name = %q", snap.Name)
	}
	if snap.State != StateClosed {
		t.Errorf("State = %v, want closed", snap.State)
	}
	if snap.Samples != 8 || snap.Failures != 2 || snap.Successes != 6 {
		t.Errorf("samples/failures/successes = %d/%d/%d, want 8/2/6", snap.Samples, snap.Failures, snap.Successes)
	}
	if snap.FailureRatio != 0.25 {
		t.Errorf("FailureRatio = %v, want 0.25", snap.FailureRatio)
	}
	if snap.Ignored != 3 {
		t.Errorf("Ignored = %d, want 3", snap.Ignored)
	}
	if snap.FailuresToTrip <= 0 {
		t.Errorf("FailuresToTrip = %d, want > 0 while closed", snap.FailuresToTrip)
	}
	if !snap.OpenedAt.IsZero() || !snap.NextProbeAt.IsZero() {
		t.Errorf("OpenedAt/NextProbeAt = %v/%v, want zero while closed", snap.OpenedAt, snap.NextProbeAt)
	}
	if snap.Transitions != 0 {
		t.Errorf("Transitions = %d, want 0", snap.Transitions)
	}

	// Empty window must not divide by zero.
	b.Reset()
	if got := b.Snapshot().FailureRatio; got != 0 {
		t.Errorf("FailureRatio on an empty window = %v, want 0", got)
	}
}

func TestTransitionCounter(t *testing.T) {
	clk := newClock()
	cfg := testConfig(clk)
	cfg.HalfOpenProbes = 1
	cfg.HalfOpenSuccesses = 1
	b := newTestBreaker(t, cfg)

	recordFailures(b, 20, provider.ClassUpstream5xx) // closed -> open: 1
	if got := b.Transitions(); got != 1 {
		t.Fatalf("Transitions = %d, want 1", got)
	}
	clk.Advance(cfg.Cooldown) // open -> half open: 2
	if got := b.State(); got != StateHalfOpen {
		t.Fatalf("state = %v", got)
	}
	if got := b.Transitions(); got != 2 {
		t.Fatalf("Transitions = %d, want 2", got)
	}
	if !b.Allow() {
		t.Fatal("Allow() = false in half open")
	}
	b.RecordFailure(provider.ClassUpstream5xx) // half open -> open: 3
	if got := b.Transitions(); got != 3 {
		t.Fatalf("Transitions = %d, want 3", got)
	}
	clk.Advance(b.Snapshot().Cooldown) // open -> half open: 4
	if !b.Allow() {
		t.Fatal("Allow() = false in half open")
	}
	b.RecordSuccess() // half open -> closed: 5
	if got := b.Transitions(); got != 5 {
		t.Fatalf("Transitions = %d, want 5", got)
	}
	if got := b.Snapshot().Opens; got != 2 {
		t.Fatalf("Opens = %d, want 2", got)
	}
}

func TestTripAndReset(t *testing.T) {
	clk := newClock()
	cfg := testConfig(clk)
	b := newTestBreaker(t, cfg)

	if got := b.State(); got != StateClosed {
		t.Fatalf("state = %v, want closed", got)
	}
	b.Trip()
	snap := b.Snapshot()
	if snap.State != StateOpen {
		t.Fatalf("state = %v after Trip, want open", snap.State)
	}
	if snap.Opens != 1 {
		t.Fatalf("Opens = %d after Trip, want 1: a manual intervention must be visible in the same counter", snap.Opens)
	}
	if b.Allow() {
		t.Fatal("Allow() = true after Trip")
	}

	// Repeated manual trips back off, same as automatic ones.
	b.Trip()
	if got := b.Snapshot().Cooldown; got != 2*cfg.Cooldown {
		t.Fatalf("cooldown after second Trip = %v, want %v", got, 2*cfg.Cooldown)
	}

	b.Reset()
	snap = b.Snapshot()
	if snap.State != StateClosed {
		t.Fatalf("state = %v after Reset, want closed", snap.State)
	}
	if snap.ConsecutiveOpens != 0 || snap.Cooldown != 0 {
		t.Fatalf("ConsecutiveOpens = %d, Cooldown = %v after Reset, want 0/0", snap.ConsecutiveOpens, snap.Cooldown)
	}
	if !b.Allow() {
		t.Fatal("Allow() = false after Reset")
	}
}

func TestDefaultClockAndJitterAreUsable(t *testing.T) {
	// New must supply working defaults; a nil Now would panic on the first call
	// and a nil Jitter on the first trip.
	b, err := New("p", DefaultConfig())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !b.Allow() {
		t.Fatal("Allow() = false on a fresh breaker")
	}
	b.RecordSuccess()
	b.Trip()
	snap := b.Snapshot()
	if snap.State != StateOpen {
		t.Fatalf("state = %v, want open", snap.State)
	}
	if snap.Cooldown < DefaultConfig().Cooldown {
		t.Fatalf("cooldown = %v, below the configured floor", snap.Cooldown)
	}
	if snap.UntilNextProbe <= 0 {
		t.Fatalf("UntilNextProbe = %v, want > 0 with a real clock", snap.UntilNextProbe)
	}
}

// TestConcurrentRequestPath runs the real request-path pattern from many
// goroutines. Its job is to be run under -race; the assertions afterwards check
// that the counters remained internally consistent rather than merely that
// nothing crashed.
func TestConcurrentRequestPath(t *testing.T) {
	clk := newClock()
	cfg := testConfig(clk)
	b := newTestBreaker(t, cfg)

	const goroutines = 32
	const perGoroutine = 500

	var wg sync.WaitGroup
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < perGoroutine; i++ {
				switch (g + i) % 4 {
				case 0:
					if b.Allow() {
						b.RecordSuccess()
					}
				case 1:
					if b.Allow() {
						b.RecordFailure(provider.ClassUpstream5xx)
					}
				case 2:
					if b.Allow() {
						b.RecordFailure(provider.ClassBadRequest)
					}
				default:
					// Readers race with writers too: metrics scrapes and the
					// router's health check both read while traffic flows.
					_ = b.Snapshot()
					_ = b.State()
					_ = b.UntilNextProbe()
					_ = b.FailuresToTrip()
					_ = b.Transitions()
				}
				// A concurrently advancing clock is realistic and exercises the
				// lazy Open -> HalfOpen edge from many goroutines at once.
				if i%37 == 0 {
					clk.Advance(cfg.Cooldown / 4)
				}
			}
		}(g)
	}
	wg.Wait()

	snap := b.Snapshot()
	if snap.Samples < 0 || snap.Samples > cfg.WindowSize {
		t.Fatalf("samples = %d, outside [0, %d]", snap.Samples, cfg.WindowSize)
	}
	if snap.Failures < 0 || snap.Failures > snap.Samples {
		t.Fatalf("failures = %d, outside [0, %d]", snap.Failures, snap.Samples)
	}
	if snap.Failures+snap.Successes != snap.Samples {
		t.Fatalf("failures %d + successes %d != samples %d", snap.Failures, snap.Successes, snap.Samples)
	}
	if snap.HalfOpenInFlight < 0 || snap.HalfOpenInFlight > cfg.HalfOpenProbes {
		t.Fatalf("HalfOpenInFlight = %d, outside [0, %d]", snap.HalfOpenInFlight, cfg.HalfOpenProbes)
	}
	if snap.Transitions == 0 {
		t.Fatal("Transitions = 0: this mix should have opened the breaker at least once, so the test is not exercising the state machine")
	}
}

// TestRequestPathDoesNotAllocate guards the claim in the package doc. Allow and
// Record run on every request; an allocation here is an allocation multiplied by
// the gateway's entire throughput.
func TestRequestPathDoesNotAllocate(t *testing.T) {
	clk := newClock()
	b := newTestBreaker(t, testConfig(clk))

	got := testing.AllocsPerRun(2000, func() {
		if b.Allow() {
			b.RecordSuccess()
		}
	})
	if got != 0 {
		t.Fatalf("Allow+RecordSuccess allocated %v objects per call, want 0", got)
	}

	// The failing path must be allocation-free too, including the trip itself.
	got = testing.AllocsPerRun(2000, func() {
		if b.Allow() {
			b.RecordFailure(provider.ClassUpstream5xx)
		}
		b.Reset()
	})
	if got != 0 {
		t.Fatalf("Allow+RecordFailure allocated %v objects per call, want 0", got)
	}
}

func TestRegistrySharesOneBreakerPerProvider(t *testing.T) {
	clk := newClock()
	r, err := NewRegistry(testConfig(clk))
	if err != nil {
		t.Fatalf("NewRegistry: %v", err)
	}

	a1 := r.Get("openai-primary")
	a2 := r.Get("openai-primary")
	if a1 != a2 {
		t.Fatal("Get returned two breakers for the same provider: its health state would be split")
	}
	bb := r.Get("openai-secondary")
	if bb == a1 {
		t.Fatal("Get returned the same breaker for two providers: one expired key would take the healthy one out of rotation")
	}
	if bb.Name() != "openai-secondary" {
		t.Fatalf("Name = %q", bb.Name())
	}

	// Opening one must not affect the other.
	recordFailures(a1, 20, provider.ClassAuth)
	if a1.State() != StateOpen {
		t.Fatal("primary should be open")
	}
	if bb.State() != StateClosed {
		t.Fatal("secondary should be unaffected")
	}

	snaps := r.Snapshots()
	if len(snaps) != 2 {
		t.Fatalf("Snapshots length = %d, want 2", len(snaps))
	}
	byName := map[string]Snapshot{}
	for _, s := range snaps {
		byName[s.Name] = s
	}
	if byName["openai-primary"].State != StateOpen {
		t.Fatalf("primary snapshot state = %v", byName["openai-primary"].State)
	}
	if byName["openai-secondary"].State != StateClosed {
		t.Fatalf("secondary snapshot state = %v", byName["openai-secondary"].State)
	}
	if r.Config().WindowSize != testConfig(clk).WindowSize {
		t.Fatal("Config() did not round-trip")
	}
}

func TestRegistryRejectsBadConfig(t *testing.T) {
	if _, err := NewRegistry(Config{}); err == nil {
		t.Fatal("NewRegistry with a zero config: want error, got nil")
	}
}

// TestRegistryConcurrentGet: two goroutines missing the read lock at once must
// not each create a breaker, or the provider's health state is split between
// them and neither ever accumulates enough samples to trip.
func TestRegistryConcurrentGet(t *testing.T) {
	clk := newClock()
	r, err := NewRegistry(testConfig(clk))
	if err != nil {
		t.Fatalf("NewRegistry: %v", err)
	}

	const goroutines = 64
	got := make([]*Breaker, goroutines)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			got[i] = r.Get("contended")
			got[i].Allow()
			got[i].RecordFailure(provider.ClassTimeout)
			_ = r.Snapshots()
		}(i)
	}
	close(start)
	wg.Wait()

	for i, b := range got {
		if b != got[0] {
			t.Fatalf("goroutine %d got a different breaker instance", i)
		}
	}
	if n := len(r.Snapshots()); n != 1 {
		t.Fatalf("registry holds %d breakers, want 1", n)
	}
	// 64 counting failures through one breaker must have opened it. If the
	// registry had handed out separate instances this would be Closed.
	if st := got[0].State(); st != StateOpen {
		t.Fatalf("state = %v after 64 concurrent counting failures, want open", st)
	}
}

func TestErrOpenSentinel(t *testing.T) {
	// Routing code wraps this; if it were not comparable with errors.Is the
	// wrapping would silently stop matching.
	wrapped := fmt.Errorf("route to openai-primary: %w", ErrOpen)
	if !errors.Is(wrapped, ErrOpen) {
		t.Fatal("errors.Is(wrapped, ErrOpen) = false")
	}
}
