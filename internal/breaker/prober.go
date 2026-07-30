package breaker

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/provider"
)

// ProbeFunc performs one health check. A nil error is a healthy provider. A
// non-nil error should be a *provider.Failure so the prober can classify it;
// anything else is treated as provider.ClassUnknown, which counts against
// health — an unrecognised probe error is more likely a broken provider than a
// broken probe, and treating it as evidence is the safe direction.
type ProbeFunc func(ctx context.Context) error

// ProbePolicy decides when active probing is worth its cost. Probes are real
// requests: they burn quota, they cost money on a metered provider, and on a
// rate-limited one they compete with paying traffic. So the policy is explicit
// rather than "probe every N seconds forever".
type ProbePolicy int

const (
	// ProbeWhenNotClosed is the default and the one you almost always want:
	// probe only while the breaker is Open or HalfOpen. Passive observation of
	// real traffic is free and strictly better evidence than a synthetic probe,
	// so while the breaker is Closed the prober does nothing. Its whole job is
	// recovery detection for a provider that, by definition, is receiving no
	// traffic to be judged by.
	ProbeWhenNotClosed ProbePolicy = iota

	// ProbeAlways probes on every tick regardless of state. Worth the quota for
	// a provider that gets so little organic traffic that the sliding window
	// never reaches MinSamples — without it, such a provider's outage is not
	// detected until a real request happens to hit it. Note the cost: this
	// makes probe failures able to open the breaker, which is the point.
	ProbeAlways

	// ProbeDisabled never issues a probe at all, but still ticks.
	//
	// That is not a no-op, and it is the subtlest policy here. The breaker's
	// Open -> HalfOpen edge is evaluated lazily, on observation, so a ticking
	// prober that only *reads* the breaker is enough to move an idle Open
	// breaker into HalfOpen and make it eligible for real traffic again — at
	// zero quota cost. Use it for a provider whose health endpoint is a weaker
	// signal than real traffic (a /models list can return 200 while inference is
	// down), where you want the recovery timer without the synthetic check.
	ProbeDisabled
)

// String renders the policy for logs and config round-tripping.
func (p ProbePolicy) String() string {
	switch p {
	case ProbeAlways:
		return "always"
	case ProbeDisabled:
		return "disabled"
	default:
		return "when_not_closed"
	}
}

// ProberConfig configures one background health prober.
type ProberConfig struct {
	// Interval is the tick period. Must be > 0.
	Interval time.Duration

	// Timeout bounds one probe. Must be > 0 and should be well under Interval,
	// so a hung provider cannot stack probes. A probe that hangs forever is the
	// classic way an active health checker turns into a goroutine leak.
	Timeout time.Duration

	// Policy decides whether a given tick actually probes.
	Policy ProbePolicy

	// AlignToCooldown makes the prober skip ticks while the breaker is Open and
	// still inside its cooldown, instead of probing on the raw interval.
	//
	// This is what makes the exponential backoff mean anything. The breaker's
	// backoff says "do not touch this provider for the next 90 seconds"; a
	// prober ticking every 5 seconds and probing anyway would hit it 18 times
	// in that window and the backoff would be decorative. Defaults to on
	// (the field is Disable-shaped for exactly that reason).
	DisableCooldownAlignment bool

	// Now is the clock, defaulting to time.Now. Injected only for the prober's
	// own bookkeeping; the ticker itself is supplied by Ticks for tests that
	// need to step time exactly.
	Now func() time.Time

	// Ticks, when non-nil, replaces the internal time.Ticker. Tests send on it
	// to drive exactly one tick, which makes every assertion deterministic
	// instead of a sleep. Production leaves it nil.
	Ticks <-chan time.Time
}

// DefaultProberConfig probes a provider every 5 seconds with a 2-second timeout,
// only while its breaker is not Closed.
func DefaultProberConfig() ProberConfig {
	return ProberConfig{
		Interval: 5 * time.Second,
		Timeout:  2 * time.Second,
		Policy:   ProbeWhenNotClosed,
	}
}

// Validate checks the prober config.
func (c ProberConfig) Validate() error {
	switch {
	case c.Interval <= 0 && c.Ticks == nil:
		return fmt.Errorf("prober: Interval must be > 0, got %v", c.Interval)
	case c.Timeout <= 0:
		return fmt.Errorf("prober: Timeout must be > 0, got %v", c.Timeout)
	case c.Policy < ProbeWhenNotClosed || c.Policy > ProbeDisabled:
		return fmt.Errorf("prober: unknown Policy %d", int(c.Policy))
	}
	return nil
}

// Prober actively health-checks one provider in the background and feeds the
// results into its breaker.
//
// It is the other half of the recovery story. The breaker's Open -> HalfOpen
// edge only fires when someone calls Allow, and a provider that has been taken
// out of rotation may get no calls at all — the router skips it, so nothing asks
// the breaker whether it is time to try again, and it can sit Open indefinitely
// with the gateway insisting the provider is down long after it came back. The
// prober is what guarantees an idle Open breaker is still re-examined.
type Prober struct {
	br    *Breaker
	probe ProbeFunc
	cfg   ProberConfig
	now   func() time.Time
	ticks <-chan time.Time
	align bool

	// done closes when the run loop has returned, so Wait can prove the
	// goroutine is gone rather than assume it.
	done chan struct{}
	once sync.Once

	probes  atomic.Uint64
	skipped atomic.Uint64
	ok      atomic.Uint64
	failed  atomic.Uint64
	handled atomic.Uint64
}

// NewProber builds a prober. It does not start anything; call Run.
func NewProber(br *Breaker, probe ProbeFunc, cfg ProberConfig) (*Prober, error) {
	if br == nil {
		return nil, errors.New("prober: breaker must not be nil")
	}
	if probe == nil {
		return nil, errors.New("prober: probe func must not be nil")
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	now := cfg.Now
	if now == nil {
		now = time.Now
	}
	return &Prober{
		br:    br,
		probe: probe,
		cfg:   cfg,
		now:   now,
		ticks: cfg.Ticks,
		align: !cfg.DisableCooldownAlignment,
		done:  make(chan struct{}),
	}, nil
}

// ProbeFromHealthProbe adapts a provider that implements provider.HealthProbe.
// Returned separately from NewProber so a provider without a health endpoint is
// a compile-time-visible absence rather than a nil check at runtime.
func ProbeFromHealthProbe(hp provider.HealthProbe) ProbeFunc {
	return hp.Probe
}

// Run drives the probe loop until ctx is cancelled, then returns. It blocks, so
// callers own the goroutine — `go p.Run(ctx)` — which keeps the lifetime visible
// at the call site instead of hidden inside a constructor that started a
// goroutine nobody can see or join.
func (p *Prober) Run(ctx context.Context) {
	defer p.once.Do(func() { close(p.done) })

	ticks := p.ticks
	if ticks == nil {
		t := time.NewTicker(p.cfg.Interval)
		defer t.Stop()
		ticks = t.C
	}

	for {
		select {
		case <-ctx.Done():
			return
		case _, open := <-ticks:
			if !open {
				// An injected tick channel that closed. Treat it as shutdown
				// rather than spinning on a closed channel forever, which is
				// how a test helper turns into a pegged CPU core.
				return
			}
			p.tick(ctx)
		}
	}
}

// Wait blocks until the run loop has returned. Tests use it to assert the
// goroutine actually exits on cancellation; shutdown code uses it to avoid
// racing a final probe against process teardown.
func (p *Prober) Wait() { <-p.done }

// Done is closed when the run loop has returned, for callers that want to select
// on it rather than block.
func (p *Prober) Done() <-chan struct{} { return p.done }

// tick runs at most one probe and feeds its result to the breaker.
func (p *Prober) tick(ctx context.Context) {
	// Incremented last, after the outcome has reached the breaker, so that
	// "ticks handled" is a happens-after signal for the whole tick. A metrics
	// scrape that saw Probes incremented but the breaker not yet updated would
	// be reporting a half-applied tick, and the tests would have to poll for
	// the breaker to catch up instead of asserting on it directly.
	defer p.handled.Add(1)

	if !p.shouldProbe() {
		p.skipped.Add(1)
		return
	}
	// The breaker gates admission even for probes, so a HalfOpen trial budget of
	// 2 is not silently exceeded by the prober racing real traffic. If the
	// breaker says no, this tick is a skip, not a probe.
	if !p.br.Allow() {
		p.skipped.Add(1)
		return
	}

	pctx, cancel := context.WithTimeout(ctx, p.cfg.Timeout)
	err := p.probe(pctx)
	cancel()

	p.probes.Add(1)
	if err == nil {
		p.ok.Add(1)
		p.br.RecordSuccess()
		return
	}
	p.failed.Add(1)

	// A cancelled parent context means the *gateway* is shutting down, not that
	// the provider is unhealthy. Recording it as a failure would leave every
	// breaker Open across a restart, so the gateway would come back up
	// convinced all its providers were dead. Release the admission slot with a
	// class that carries no health signal.
	if ctx.Err() != nil {
		p.br.RecordFailure(provider.ClassCancelled)
		return
	}

	var f *provider.Failure
	if errors.As(err, &f) {
		p.br.RecordFailure(f.Class)
		return
	}
	// An unclassified error, including a probe that exceeded Timeout: unknown
	// counts against health, which is what a probe that will not answer in two
	// seconds deserves.
	p.br.RecordFailure(provider.ClassUnknown)
}

// shouldProbe applies the policy. Note that it calls Snapshot unconditionally,
// including under ProbeDisabled: that read is what advances an idle Open breaker
// to HalfOpen, and it is the entire mechanism behind ProbeDisabled being useful.
func (p *Prober) shouldProbe() bool {
	snap := p.br.Snapshot()
	switch p.cfg.Policy {
	case ProbeDisabled:
		return false
	case ProbeAlways:
		// Still respect the cooldown when Open: probing inside a backoff window
		// is exactly the load a backed-off provider is being protected from.
	default: // ProbeWhenNotClosed
		if snap.State == StateClosed {
			return false
		}
	}
	if p.align && snap.State == StateOpen && snap.UntilNextProbe > 0 {
		return false
	}
	return true
}

// ProberStats is the prober's own counters, for metrics. Separate from
// Snapshot because "how often did we probe" and "what state is the breaker in"
// are answers to different questions and conflating them hides a prober that is
// skipping every tick.
type ProberStats struct {
	// Probes is the number of probe calls actually made.
	Probes uint64
	// Skipped is ticks that did not probe: policy said no, the cooldown had not
	// elapsed, or the breaker refused admission.
	Skipped uint64
	// OK and Failed partition Probes by result.
	OK     uint64
	Failed uint64
	// Ticks is the number of ticks fully handled: Probes + Skipped, but read
	// after the outcome reached the breaker. It is the counter to watch to tell
	// "the prober is not running" apart from "the prober is running and
	// skipping every tick".
	Ticks uint64
}

// Stats reads the prober's counters. Uses atomics rather than the breaker's
// mutex so a metrics scrape cannot contend with the probe loop.
func (p *Prober) Stats() ProberStats {
	// Ticks is loaded first, so that a caller which observes an incremented
	// Ticks is guaranteed to see the rest of that tick's counters too. Loading
	// it last would let it report a tick whose Probes increment had not yet
	// been read.
	return ProberStats{
		Ticks:   p.handled.Load(),
		Probes:  p.probes.Load(),
		Skipped: p.skipped.Load(),
		OK:      p.ok.Load(),
		Failed:  p.failed.Load(),
	}
}

// ProberSet runs one prober per provider and shuts them all down together.
type ProberSet struct {
	mu      sync.Mutex
	probers map[string]*Prober
	wg      sync.WaitGroup
	started bool
}

// NewProberSet returns an empty set.
func NewProberSet() *ProberSet {
	return &ProberSet{probers: make(map[string]*Prober)}
}

// Add registers a prober for a provider instance. Must be called before Start;
// adding after Start returns an error rather than silently never running the
// prober, which is the failure mode you would not notice until an outage.
func (s *ProberSet) Add(br *Breaker, probe ProbeFunc, cfg ProberConfig) error {
	p, err := NewProber(br, probe, cfg)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.started {
		return errors.New("prober set: Add after Start")
	}
	if _, dup := s.probers[br.Name()]; dup {
		return fmt.Errorf("prober set: duplicate prober for provider %q", br.Name())
	}
	s.probers[br.Name()] = p
	return nil
}

// Start launches every registered prober.
func (s *ProberSet) Start(ctx context.Context) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.started {
		return
	}
	s.started = true
	for _, p := range s.probers {
		s.wg.Add(1)
		go func(p *Prober) {
			defer s.wg.Done()
			p.Run(ctx)
		}(p)
	}
}

// Wait blocks until every prober's loop has returned.
func (s *ProberSet) Wait() { s.wg.Wait() }

// Stats returns per-provider prober counters.
func (s *ProberSet) Stats() map[string]ProberStats {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make(map[string]ProberStats, len(s.probers))
	for name, p := range s.probers {
		out[name] = p.Stats()
	}
	return out
}
