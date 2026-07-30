package breaker

import (
	"context"
	"errors"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/provider"
)

// probeRecorder is a scriptable ProbeFunc. Calls are counted atomically so the
// leak tests can read them from the test goroutine while the prober runs.
type probeRecorder struct {
	mu      sync.Mutex
	results []error // consumed in order; the last one repeats forever
	calls   atomic.Int64
	// block, when non-nil, is received from inside the probe so a test can hold
	// a probe in flight and inspect the breaker mid-probe.
	block chan struct{}
	// sawDeadline records whether the probe context carried the timeout, which
	// is the only way to prove the timeout is actually applied.
	sawDeadline atomic.Bool
	// sawCancel records whether the probe observed its context cancelled.
	sawCancel atomic.Bool
}

func newProbeRecorder(results ...error) *probeRecorder {
	if len(results) == 0 {
		results = []error{nil}
	}
	return &probeRecorder{results: results}
}

func (p *probeRecorder) fn(ctx context.Context) error {
	n := p.calls.Add(1)
	if _, ok := ctx.Deadline(); ok {
		p.sawDeadline.Store(true)
	}
	if p.block != nil {
		select {
		case <-p.block:
		case <-ctx.Done():
			p.sawCancel.Store(true)
			return ctx.Err()
		}
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	idx := int(n) - 1
	if idx >= len(p.results) {
		idx = len(p.results) - 1
	}
	return p.results[idx]
}

func (p *probeRecorder) Calls() int64 { return p.calls.Load() }

// tickDriver drives a prober one tick at a time. Every prober test below steps
// this rather than sleeping, so the assertions are about the prober's logic and
// not about the scheduler.
type tickDriver struct {
	ch chan time.Time
}

func newTickDriver() *tickDriver { return &tickDriver{ch: make(chan time.Time)} }

// tick delivers one tick and waits until the prober has fully handled it,
// including feeding the outcome to the breaker. Stats().Ticks is incremented
// last, so observing it is a happens-after signal for the whole tick — that
// handshake, rather than a sleep, is what makes every assertion below exact.
func (d *tickDriver) tick(t *testing.T, p *Prober) {
	t.Helper()
	before := p.Stats().Ticks
	select {
	case d.ch <- time.Now():
	case <-time.After(2 * time.Second):
		t.Fatal("prober did not accept a tick within 2s")
	}
	deadline := time.Now().Add(2 * time.Second)
	for p.Stats().Ticks == before {
		if time.Now().After(deadline) {
			t.Fatal("prober did not complete a tick within 2s")
		}
		runtime.Gosched()
	}
}

func startProber(t *testing.T, p *Prober) (cancel context.CancelFunc) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	go p.Run(ctx)
	t.Cleanup(func() {
		cancel()
		p.Wait()
	})
	return cancel
}

func testProberConfig(d *tickDriver) ProberConfig {
	cfg := DefaultProberConfig()
	cfg.Ticks = d.ch
	return cfg
}

func TestProberConfigValidate(t *testing.T) {
	base := DefaultProberConfig()
	mut := func(f func(*ProberConfig)) ProberConfig {
		c := base
		f(&c)
		return c
	}
	ticks := make(chan time.Time)

	tests := []struct {
		name    string
		cfg     ProberConfig
		wantErr bool
	}{
		{"default is valid", base, false},
		{"zero value rejected", ProberConfig{}, true},
		{"interval zero", mut(func(c *ProberConfig) { c.Interval = 0 }), true},
		// An injected tick channel supplies the schedule, so Interval is moot.
		{"interval zero with injected ticks", mut(func(c *ProberConfig) { c.Interval = 0; c.Ticks = ticks }), false},
		{"timeout zero", mut(func(c *ProberConfig) { c.Timeout = 0 }), true},
		{"timeout negative", mut(func(c *ProberConfig) { c.Timeout = -time.Second }), true},
		{"policy always", mut(func(c *ProberConfig) { c.Policy = ProbeAlways }), false},
		{"policy disabled", mut(func(c *ProberConfig) { c.Policy = ProbeDisabled }), false},
		{"policy out of range", mut(func(c *ProberConfig) { c.Policy = ProbePolicy(99) }), true},
		{"policy negative", mut(func(c *ProberConfig) { c.Policy = ProbePolicy(-1) }), true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.cfg.Validate(); (err != nil) != tt.wantErr {
				t.Fatalf("Validate() err = %v, wantErr = %v", err, tt.wantErr)
			}
			clk := newClock()
			b := newTestBreaker(t, testConfig(clk))
			_, err := NewProber(b, newProbeRecorder().fn, tt.cfg)
			if (err != nil) != tt.wantErr {
				t.Fatalf("NewProber() err = %v, wantErr = %v", err, tt.wantErr)
			}
		})
	}
}

func TestProbePolicyString(t *testing.T) {
	tests := []struct {
		p    ProbePolicy
		want string
	}{
		{ProbeWhenNotClosed, "when_not_closed"},
		{ProbeAlways, "always"},
		{ProbeDisabled, "disabled"},
		{ProbePolicy(42), "when_not_closed"},
	}
	for _, tt := range tests {
		if got := tt.p.String(); got != tt.want {
			t.Errorf("ProbePolicy(%d).String() = %q, want %q", int(tt.p), got, tt.want)
		}
	}
}

func TestNewProberRejectsNilArguments(t *testing.T) {
	clk := newClock()
	b := newTestBreaker(t, testConfig(clk))
	if _, err := NewProber(nil, newProbeRecorder().fn, DefaultProberConfig()); err == nil {
		t.Error("NewProber(nil breaker): want error")
	}
	if _, err := NewProber(b, nil, DefaultProberConfig()); err == nil {
		t.Error("NewProber(nil probe): want error")
	}
}

// TestProberExitsOnContextCancellationWithoutLeaking is the goroutine-leak test.
// It asserts the loop actually returns — via Wait, which is only closed by the
// loop's own deferred close — and that the goroutine count returns to baseline.
func TestProberExitsOnContextCancellationWithoutLeaking(t *testing.T) {
	clk := newClock()
	b := newTestBreaker(t, testConfig(clk))
	rec := newProbeRecorder(nil)

	// A real ticker, not an injected one, so the production path is what exits.
	cfg := DefaultProberConfig()
	cfg.Interval = time.Millisecond
	cfg.Timeout = 50 * time.Millisecond
	cfg.Policy = ProbeAlways
	p, err := NewProber(b, rec.fn, cfg)
	if err != nil {
		t.Fatalf("NewProber: %v", err)
	}

	baseline := runtime.NumGoroutine()
	ctx, cancel := context.WithCancel(context.Background())
	returned := make(chan struct{})
	go func() {
		p.Run(ctx)
		close(returned)
	}()

	// Let it actually get going, so the test is not proving that a loop which
	// never started also never leaks.
	deadline := time.Now().Add(2 * time.Second)
	for rec.Calls() == 0 {
		if time.Now().After(deadline) {
			t.Fatal("prober never probed")
		}
		time.Sleep(time.Millisecond)
	}

	cancel()
	select {
	case <-returned:
	case <-time.After(2 * time.Second):
		t.Fatal("Run did not return within 2s of cancellation: the prober goroutine is leaked")
	}
	select {
	case <-p.Done():
	case <-time.After(time.Second):
		t.Fatal("Done() not closed after Run returned")
	}
	p.Wait() // must not block once Run has returned

	// The ticker goroutine and the run goroutine must both be gone. Poll,
	// because the runtime reaps asynchronously.
	deadline = time.Now().Add(2 * time.Second)
	for runtime.NumGoroutine() > baseline {
		if time.Now().After(deadline) {
			t.Fatalf("goroutine count %d still above baseline %d", runtime.NumGoroutine(), baseline)
		}
		time.Sleep(time.Millisecond)
	}
}

// TestProberWaitIsIdempotent: Wait is called by shutdown code and by tests;
// closing done twice would panic.
func TestProberWaitIsIdempotent(t *testing.T) {
	clk := newClock()
	b := newTestBreaker(t, testConfig(clk))
	d := newTickDriver()
	p, err := NewProber(b, newProbeRecorder(nil).fn, testProberConfig(d))
	if err != nil {
		t.Fatalf("NewProber: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	go p.Run(ctx)
	cancel()
	p.Wait()
	p.Wait()
	p.Wait()
}

// TestProberExitsOnClosedTickChannel: a test helper that closes its tick channel
// must not spin the loop on a permanently-ready closed channel, which is how an
// injected ticker pegs a CPU core.
func TestProberExitsOnClosedTickChannel(t *testing.T) {
	clk := newClock()
	b := newTestBreaker(t, testConfig(clk))
	ticks := make(chan time.Time)
	cfg := DefaultProberConfig()
	cfg.Ticks = ticks
	p, err := NewProber(b, newProbeRecorder(nil).fn, cfg)
	if err != nil {
		t.Fatalf("NewProber: %v", err)
	}
	go p.Run(context.Background())
	close(ticks)
	select {
	case <-p.Done():
	case <-time.After(2 * time.Second):
		t.Fatal("Run did not return on a closed tick channel")
	}
}

// TestProberPolicySkipsWhenClosed pins the quota policy: while the breaker is
// Closed there is real traffic being observed for free, so a synthetic probe is
// pure waste.
func TestProberPolicySkipsWhenClosed(t *testing.T) {
	tests := []struct {
		name       string
		policy     ProbePolicy
		wantProbes int64
	}{
		{"when not closed skips a closed breaker", ProbeWhenNotClosed, 0},
		{"always probes a closed breaker", ProbeAlways, 5},
		{"disabled never probes", ProbeDisabled, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clk := newClock()
			b := newTestBreaker(t, testConfig(clk))
			rec := newProbeRecorder(nil)
			d := newTickDriver()
			cfg := testProberConfig(d)
			cfg.Policy = tt.policy
			p, err := NewProber(b, rec.fn, cfg)
			if err != nil {
				t.Fatalf("NewProber: %v", err)
			}
			startProber(t, p)

			for i := 0; i < 5; i++ {
				d.tick(t, p)
			}
			if got := rec.Calls(); got != tt.wantProbes {
				t.Fatalf("probe calls = %d, want %d", got, tt.wantProbes)
			}
			stats := p.Stats()
			if stats.Probes+stats.Skipped != 5 {
				t.Fatalf("probes %d + skipped %d != 5 ticks", stats.Probes, stats.Skipped)
			}
			if b.State() != StateClosed {
				t.Fatalf("breaker state = %v, want closed (healthy probes must not move it)", b.State())
			}
		})
	}
}

// TestProberAlwaysCanDetectOutageOnIdleProvider is the reason ProbeAlways
// exists: a provider with no organic traffic never fills the window, so its
// outage would only be found by an unlucky real request. Synthetic probes must be
// able to open the breaker on their own.
func TestProberAlwaysCanDetectOutageOnIdleProvider(t *testing.T) {
	clk := newClock()
	bcfg := testConfig(clk)
	b := newTestBreaker(t, bcfg)
	rec := newProbeRecorder(&provider.Failure{Class: provider.ClassConnect, Err: errors.New("connection refused")})
	d := newTickDriver()
	cfg := testProberConfig(d)
	cfg.Policy = ProbeAlways
	p, err := NewProber(b, rec.fn, cfg)
	if err != nil {
		t.Fatalf("NewProber: %v", err)
	}
	startProber(t, p)

	// The breaker's own live projection, not WorstCaseFailuresToTrip: the window
	// starts empty, so MinSamples dominates and five probes suffice. Asserting
	// the worst case here would be asserting a bound the cold breaker beats.
	want := b.FailuresToTrip()
	if worst := bcfg.WorstCaseFailuresToTrip(); want > worst {
		t.Fatalf("FailuresToTrip() = %d exceeds the advertised worst case %d", want, worst)
	}
	for i := 0; i < want; i++ {
		if b.State() != StateClosed {
			t.Fatalf("breaker opened after %d probes, expected %d", i, want)
		}
		d.tick(t, p)
	}
	if got := b.State(); got != StateOpen {
		t.Fatalf("state = %v after %d failing probes, want open", got, want)
	}
	if got := p.Stats().Failed; got != uint64(want) {
		t.Fatalf("Stats().Failed = %d, want %d", got, want)
	}
}

// TestProberRecoversAnIdleOpenBreaker is the prober's core job: nothing calls
// Allow on an Open provider (the router skips it), so without the prober the
// breaker could sit Open forever with a provider that has long since recovered.
func TestProberRecoversAnIdleOpenBreaker(t *testing.T) {
	clk := newClock()
	bcfg := testConfig(clk)
	bcfg.HalfOpenProbes = 1
	bcfg.HalfOpenSuccesses = 2
	b := newTestBreaker(t, bcfg)
	rec := newProbeRecorder(nil)
	d := newTickDriver()
	p, err := NewProber(b, rec.fn, testProberConfig(d))
	if err != nil {
		t.Fatalf("NewProber: %v", err)
	}
	startProber(t, p)

	recordFailures(b, 20, provider.ClassUpstream5xx)
	if b.State() != StateOpen {
		t.Fatal("expected open")
	}

	// Inside the cooldown the prober must not touch the provider: that is what
	// makes the breaker's backoff mean anything.
	for i := 0; i < 3; i++ {
		d.tick(t, p)
	}
	if got := rec.Calls(); got != 0 {
		t.Fatalf("probe calls = %d inside the cooldown, want 0: the backoff would be decorative", got)
	}
	if b.State() != StateOpen {
		t.Fatal("breaker should still be open inside the cooldown")
	}

	// Cooldown expires. Now probes are allowed, and two successes close it.
	clk.Advance(bcfg.Cooldown)
	d.tick(t, p)
	if got := rec.Calls(); got != 1 {
		t.Fatalf("probe calls = %d after the cooldown, want 1", got)
	}
	if got := b.State(); got != StateHalfOpen {
		t.Fatalf("state = %v after one successful probe, want half open (2 successes required)", got)
	}
	d.tick(t, p)
	if got := b.State(); got != StateClosed {
		t.Fatalf("state = %v after two successful probes, want closed", got)
	}

	// And with the breaker Closed the default policy stops probing again.
	callsAtClose := rec.Calls()
	for i := 0; i < 3; i++ {
		d.tick(t, p)
	}
	if got := rec.Calls(); got != callsAtClose {
		t.Fatalf("probe calls = %d after close, want %d: probing a healthy provider wastes quota", got, callsAtClose)
	}
}

// TestProbeDisabledStillAdvancesRecoveryTimer proves the non-obvious property
// documented on ProbeDisabled: reading the breaker is enough to move an idle Open
// breaker to HalfOpen, so the policy buys the recovery timer at zero quota cost.
func TestProbeDisabledStillAdvancesRecoveryTimer(t *testing.T) {
	clk := newClock()
	bcfg := testConfig(clk)
	b := newTestBreaker(t, bcfg)
	rec := newProbeRecorder(nil)
	d := newTickDriver()
	cfg := testProberConfig(d)
	cfg.Policy = ProbeDisabled
	p, err := NewProber(b, rec.fn, cfg)
	if err != nil {
		t.Fatalf("NewProber: %v", err)
	}
	startProber(t, p)

	recordFailures(b, 20, provider.ClassUpstream5xx)
	clk.Advance(bcfg.Cooldown)
	d.tick(t, p)

	if got := rec.Calls(); got != 0 {
		t.Fatalf("probe calls = %d, want 0 under ProbeDisabled", got)
	}
	// Read the raw state field rather than State(), which would itself perform
	// the refresh and make the assertion tautological.
	b.mu.Lock()
	raw := b.state
	b.mu.Unlock()
	if raw != StateHalfOpen {
		t.Fatalf("internal state = %v, want half open: the tick's read should have advanced the timer", raw)
	}
	if got := p.Stats().Skipped; got != 1 {
		t.Fatalf("Skipped = %d, want 1", got)
	}
}

// TestProberCooldownAlignmentCanBeDisabled: the alignment is the default but it
// is a policy, so the opposite must actually be selectable.
func TestProberCooldownAlignmentCanBeDisabled(t *testing.T) {
	clk := newClock()
	bcfg := testConfig(clk)
	b := newTestBreaker(t, bcfg)
	rec := newProbeRecorder(&provider.Failure{Class: provider.ClassConnect})
	d := newTickDriver()
	cfg := testProberConfig(d)
	cfg.DisableCooldownAlignment = true
	p, err := NewProber(b, rec.fn, cfg)
	if err != nil {
		t.Fatalf("NewProber: %v", err)
	}
	startProber(t, p)

	recordFailures(b, 20, provider.ClassUpstream5xx)
	if got := b.Snapshot().UntilNextProbe; got <= 0 {
		t.Fatalf("UntilNextProbe = %v, want > 0", got)
	}
	d.tick(t, p)
	// Alignment off, so the probe happens despite the cooldown. The breaker's
	// own Allow still refuses admission while Open, so it counts as a skip —
	// which is the layered defence: even with alignment off, the breaker gates.
	if got := p.Stats().Skipped; got != 1 {
		t.Fatalf("Skipped = %d, want 1: the breaker's own Allow must still gate a probe inside the cooldown", got)
	}
	if got := rec.Calls(); got != 0 {
		t.Fatalf("probe calls = %d, want 0: Allow refused admission", got)
	}
}

// TestProberRespectsHalfOpenTrialBudget: the prober goes through Allow, so it
// cannot exceed the trial budget by racing real traffic.
func TestProberRespectsHalfOpenTrialBudget(t *testing.T) {
	clk := newClock()
	bcfg := testConfig(clk)
	bcfg.HalfOpenProbes = 1
	bcfg.HalfOpenSuccesses = 100 // never close during the test
	b := newTestBreaker(t, bcfg)
	rec := newProbeRecorder(nil)
	rec.block = make(chan struct{})
	d := newTickDriver()
	p, err := NewProber(b, rec.fn, testProberConfig(d))
	if err != nil {
		t.Fatalf("NewProber: %v", err)
	}
	startProber(t, p)

	recordFailures(b, 20, provider.ClassUpstream5xx)
	clk.Advance(bcfg.Cooldown)

	// Real traffic takes the single trial slot first.
	if !b.Allow() {
		t.Fatal("Allow() = false for the real request")
	}
	d.tick(t, p)
	if got := rec.Calls(); got != 0 {
		t.Fatalf("probe calls = %d, want 0: the prober must not exceed the trial budget", got)
	}
	if got := p.Stats().Skipped; got != 1 {
		t.Fatalf("Skipped = %d, want 1", got)
	}

	// Release the slot; now the prober may take it.
	b.RecordSuccess()
	close(rec.block)
	d.tick(t, p)
	if got := rec.Calls(); got != 1 {
		t.Fatalf("probe calls = %d, want 1 once the slot was freed", got)
	}
}

// TestProberClassifiesProbeErrors is the classification test. A probe error that
// carries a non-counting class must be as harmless as one on the real request
// path — a health endpoint returning 400 because the gateway's probe payload is
// wrong is a gateway bug, not a provider outage, and must not open the breaker.
func TestProberClassifiesProbeErrors(t *testing.T) {
	tests := []struct {
		name        string
		err         error
		wantOpen    bool
		wantIgnored bool
	}{
		{"nil is healthy", nil, false, false},
		{"connect failure counts", &provider.Failure{Class: provider.ClassConnect}, true, false},
		{"5xx counts", &provider.Failure{Class: provider.ClassUpstream5xx, StatusCode: 503}, true, false},
		{"auth counts", &provider.Failure{Class: provider.ClassAuth, StatusCode: 401}, true, false},
		{"bad request does not count", &provider.Failure{Class: provider.ClassBadRequest, StatusCode: 400}, false, true},
		{"content filter does not count", &provider.Failure{Class: provider.ClassContentFilter}, false, true},
		// An unclassified error is treated as ClassUnknown, which counts: a
		// probe that fails in a way the adapter did not model is more likely a
		// broken provider than a broken probe.
		{"plain error counts as unknown", errors.New("something broke"), true, false},
		// Wrapped, because real adapters return wrapped failures.
		{"wrapped failure is unwrapped", errWrap(&provider.Failure{Class: provider.ClassOverloaded}), true, false},
		{"wrapped non-counting failure stays non-counting", errWrap(&provider.Failure{Class: provider.ClassContextLength}), false, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clk := newClock()
			bcfg := testConfig(clk)
			b := newTestBreaker(t, bcfg)
			rec := newProbeRecorder(tt.err)
			d := newTickDriver()
			cfg := testProberConfig(d)
			cfg.Policy = ProbeAlways
			p, err := NewProber(b, rec.fn, cfg)
			if err != nil {
				t.Fatalf("NewProber: %v", err)
			}
			startProber(t, p)

			for i := 0; i < 40 && b.State() == StateClosed; i++ {
				d.tick(t, p)
			}

			gotOpen := b.State() == StateOpen
			if gotOpen != tt.wantOpen {
				t.Fatalf("open = %v, want %v", gotOpen, tt.wantOpen)
			}
			snap := b.Snapshot()
			if tt.wantIgnored && snap.Ignored == 0 {
				t.Fatal("Ignored = 0, want > 0: the non-counting probe failures must be visible")
			}
			if !tt.wantIgnored && snap.Ignored != 0 {
				t.Fatalf("Ignored = %d, want 0", snap.Ignored)
			}
			stats := p.Stats()
			if tt.err == nil {
				if stats.OK == 0 || stats.Failed != 0 {
					t.Fatalf("OK = %d, Failed = %d, want OK > 0 and Failed = 0", stats.OK, stats.Failed)
				}
			} else if stats.Failed == 0 || stats.OK != 0 {
				t.Fatalf("OK = %d, Failed = %d, want Failed > 0 and OK = 0", stats.OK, stats.Failed)
			}
		})
	}
}

func errWrap(err error) error {
	return &wrapped{err}
}

type wrapped struct{ err error }

func (w *wrapped) Error() string { return "wrapped: " + w.err.Error() }
func (w *wrapped) Unwrap() error { return w.err }

// TestProbeTimeoutIsApplied: a probe that hangs must be bounded, or a hung
// provider stalls the probe loop forever and the breaker never learns anything.
func TestProbeTimeoutIsApplied(t *testing.T) {
	clk := newClock()
	bcfg := testConfig(clk)
	b := newTestBreaker(t, bcfg)
	// A probe that blocks until its context dies.
	rec := newProbeRecorder(nil)
	rec.block = make(chan struct{}) // never closed

	d := newTickDriver()
	cfg := testProberConfig(d)
	cfg.Policy = ProbeAlways
	cfg.Timeout = 10 * time.Millisecond
	p, err := NewProber(b, rec.fn, cfg)
	if err != nil {
		t.Fatalf("NewProber: %v", err)
	}
	startProber(t, p)

	d.tick(t, p)

	if !rec.sawDeadline.Load() {
		t.Fatal("probe context had no deadline: Timeout is not being applied")
	}
	if !rec.sawCancel.Load() {
		t.Fatal("probe was not cancelled: a hung provider would stall the loop forever")
	}
	// The timeout returned context.DeadlineExceeded, which is not a
	// *provider.Failure, so it is ClassUnknown and counts against health. A
	// provider that will not answer in 10ms deserves that.
	if got := p.Stats().Failed; got != 1 {
		t.Fatalf("Failed = %d, want 1", got)
	}
	if got := b.Snapshot().Failures; got != 1 {
		t.Fatalf("window failures = %d, want 1: a timed-out probe is health evidence", got)
	}
}

// TestProbeCancellationDuringShutdownIsNotHealthEvidence: a probe that fails
// because the gateway is shutting down must not be recorded as a provider
// failure, or every breaker comes back Open across a restart and the gateway
// boots convinced all its providers are dead.
func TestProbeCancellationDuringShutdownIsNotHealthEvidence(t *testing.T) {
	clk := newClock()
	bcfg := testConfig(clk)
	bcfg.MinSamples = 1
	bcfg.FailureRatio = 1.0
	b := newTestBreaker(t, bcfg)

	started := make(chan struct{})
	rec := &probeRecorder{results: []error{nil}}
	probe := func(ctx context.Context) error {
		rec.calls.Add(1)
		close(started)
		<-ctx.Done() // parent cancelled by shutdown
		return ctx.Err()
	}

	d := newTickDriver()
	cfg := testProberConfig(d)
	cfg.Policy = ProbeAlways
	cfg.Timeout = time.Hour // so the deadline cannot be the cause
	p, err := NewProber(b, probe, cfg)
	if err != nil {
		t.Fatalf("NewProber: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	go p.Run(ctx)

	d.ch <- time.Now()
	<-started
	cancel()
	p.Wait()

	snap := b.Snapshot()
	if snap.State != StateClosed {
		t.Fatalf("state = %v after a shutdown-cancelled probe, want closed", snap.State)
	}
	if snap.Failures != 0 || snap.Samples != 0 {
		t.Fatalf("window = %d samples / %d failures, want 0/0: shutdown is not a provider failure", snap.Samples, snap.Failures)
	}
	if snap.Ignored != 1 {
		t.Fatalf("Ignored = %d, want 1", snap.Ignored)
	}
}

func TestProbeFromHealthProbe(t *testing.T) {
	hp := &fakeHealthProbe{err: errors.New("down")}
	fn := ProbeFromHealthProbe(hp)
	if err := fn(context.Background()); err == nil {
		t.Fatal("probe returned nil, want the health probe's error")
	}
	if hp.calls != 1 {
		t.Fatalf("health probe calls = %d, want 1", hp.calls)
	}

	// It must satisfy the frozen provider.HealthProbe contract, not a lookalike.
	var _ provider.HealthProbe = hp
}

type fakeHealthProbe struct {
	err   error
	calls int
}

func (f *fakeHealthProbe) Probe(ctx context.Context) error {
	f.calls++
	return f.err
}

func TestProberStatsAccounting(t *testing.T) {
	clk := newClock()
	b := newTestBreaker(t, testConfig(clk))
	// Alternate healthy and unhealthy probes.
	rec := newProbeRecorder(nil, &provider.Failure{Class: provider.ClassBadRequest}, nil, &provider.Failure{Class: provider.ClassBadRequest})
	d := newTickDriver()
	cfg := testProberConfig(d)
	cfg.Policy = ProbeAlways
	p, err := NewProber(b, rec.fn, cfg)
	if err != nil {
		t.Fatalf("NewProber: %v", err)
	}
	startProber(t, p)

	for i := 0; i < 4; i++ {
		d.tick(t, p)
	}
	stats := p.Stats()
	if stats.Probes != 4 {
		t.Fatalf("Probes = %d, want 4", stats.Probes)
	}
	if stats.OK != 2 || stats.Failed != 2 {
		t.Fatalf("OK = %d, Failed = %d, want 2/2", stats.OK, stats.Failed)
	}
	if stats.OK+stats.Failed != stats.Probes {
		t.Fatalf("OK %d + Failed %d != Probes %d", stats.OK, stats.Failed, stats.Probes)
	}
	if stats.Skipped != 0 {
		t.Fatalf("Skipped = %d, want 0", stats.Skipped)
	}
}

func TestProberSetLifecycle(t *testing.T) {
	clk := newClock()
	bcfg := testConfig(clk)
	b1 := mustBreaker(t, "openai-primary", bcfg)
	b2 := mustBreaker(t, "anthropic-primary", bcfg)

	rec1 := newProbeRecorder(nil)
	rec2 := newProbeRecorder(&provider.Failure{Class: provider.ClassConnect})
	d1, d2 := newTickDriver(), newTickDriver()

	set := NewProberSet()
	cfg1 := testProberConfig(d1)
	cfg1.Policy = ProbeAlways
	cfg2 := testProberConfig(d2)
	cfg2.Policy = ProbeAlways
	if err := set.Add(b1, rec1.fn, cfg1); err != nil {
		t.Fatalf("Add: %v", err)
	}
	if err := set.Add(b2, rec2.fn, cfg2); err != nil {
		t.Fatalf("Add: %v", err)
	}
	// A duplicate would silently replace one provider's prober.
	if err := set.Add(b1, rec1.fn, cfg1); err == nil {
		t.Fatal("Add duplicate: want error")
	}
	// A bad config must be rejected at Add, not swallowed.
	if err := set.Add(mustBreaker(t, "third", bcfg), rec1.fn, ProberConfig{}); err == nil {
		t.Fatal("Add with an invalid config: want error")
	}

	ctx, cancel := context.WithCancel(context.Background())
	set.Start(ctx)
	set.Start(ctx) // idempotent; a second Start must not double the goroutines

	// Adding after Start would create a prober nobody ever runs.
	if err := set.Add(mustBreaker(t, "late", bcfg), rec1.fn, cfg1); err == nil {
		t.Fatal("Add after Start: want error")
	}

	// Wait on the tick handshake, not on the probe function's own counter: the
	// probe increments its counter on entry, so rec.Calls() != 0 does not imply
	// the prober has finished accounting for the tick.
	set.mu.Lock()
	p1, p2 := set.probers["openai-primary"], set.probers["anthropic-primary"]
	set.mu.Unlock()
	d1.tick(t, p1)
	d2.tick(t, p2)
	if rec1.Calls() == 0 || rec2.Calls() == 0 {
		t.Fatalf("probers did not both run: rec1=%d rec2=%d", rec1.Calls(), rec2.Calls())
	}

	stats := set.Stats()
	if len(stats) != 2 {
		t.Fatalf("Stats length = %d, want 2", len(stats))
	}
	if stats["openai-primary"].Probes == 0 || stats["anthropic-primary"].Probes == 0 {
		t.Fatalf("per-provider probes = %+v, want both non-zero", stats)
	}

	cancel()
	done := make(chan struct{})
	go func() { set.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("ProberSet.Wait did not return within 2s: prober goroutines leaked")
	}
}

func mustBreaker(t *testing.T, name string, cfg Config) *Breaker {
	t.Helper()
	b, err := New(name, cfg)
	if err != nil {
		t.Fatalf("New(%q): %v", name, err)
	}
	return b
}

func TestProberSetNoGoroutineLeakAcrossManyProviders(t *testing.T) {
	clk := newClock()
	bcfg := testConfig(clk)
	set := NewProberSet()
	const n = 24
	for i := 0; i < n; i++ {
		cfg := DefaultProberConfig()
		cfg.Interval = time.Millisecond
		cfg.Timeout = 10 * time.Millisecond
		cfg.Policy = ProbeAlways
		name := "p" + string(rune('a'+i))
		if err := set.Add(mustBreaker(t, name, bcfg), newProbeRecorder(nil).fn, cfg); err != nil {
			t.Fatalf("Add: %v", err)
		}
	}

	baseline := runtime.NumGoroutine()
	ctx, cancel := context.WithCancel(context.Background())
	set.Start(ctx)
	time.Sleep(20 * time.Millisecond) // let them tick a few times
	cancel()
	set.Wait()

	deadline := time.Now().Add(2 * time.Second)
	for runtime.NumGoroutine() > baseline {
		if time.Now().After(deadline) {
			t.Fatalf("goroutine count %d still above baseline %d after Wait", runtime.NumGoroutine(), baseline)
		}
		time.Sleep(time.Millisecond)
	}
}

// TestProberConcurrentWithRequestTraffic runs the prober against a breaker that
// is simultaneously taking real traffic. Intended for -race: the prober and the
// request path share the breaker, and the prober's own counters are read by a
// metrics scrape at the same time.
func TestProberConcurrentWithRequestTraffic(t *testing.T) {
	clk := newClock()
	bcfg := testConfig(clk)
	b := newTestBreaker(t, bcfg)
	rec := newProbeRecorder(nil, &provider.Failure{Class: provider.ClassTimeout})
	cfg := DefaultProberConfig()
	cfg.Interval = 100 * time.Microsecond
	cfg.Timeout = 10 * time.Millisecond
	cfg.Policy = ProbeAlways
	p, err := NewProber(b, rec.fn, cfg)
	if err != nil {
		t.Fatalf("NewProber: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	go p.Run(ctx)

	var wg sync.WaitGroup
	stop := make(chan struct{})
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; ; i++ {
				select {
				case <-stop:
					return
				default:
				}
				if b.Allow() {
					if (g+i)%3 == 0 {
						b.RecordFailure(provider.ClassUpstream5xx)
					} else {
						b.RecordSuccess()
					}
				}
				_ = b.Snapshot()
				_ = p.Stats()
				if i%50 == 0 {
					clk.Advance(bcfg.Cooldown / 2)
				}
			}
		}(g)
	}
	time.Sleep(50 * time.Millisecond)
	close(stop)
	wg.Wait()
	cancel()
	p.Wait()

	stats := p.Stats()
	if stats.OK+stats.Failed != stats.Probes {
		t.Fatalf("OK %d + Failed %d != Probes %d", stats.OK, stats.Failed, stats.Probes)
	}
	if stats.Probes == 0 {
		t.Fatal("Probes = 0: the prober never ran, so this test proved nothing")
	}
	snap := b.Snapshot()
	if snap.Failures+snap.Successes != snap.Samples {
		t.Fatalf("failures %d + successes %d != samples %d", snap.Failures, snap.Successes, snap.Samples)
	}
	if snap.HalfOpenInFlight < 0 || snap.HalfOpenInFlight > bcfg.HalfOpenProbes {
		t.Fatalf("HalfOpenInFlight = %d, outside [0, %d]", snap.HalfOpenInFlight, bcfg.HalfOpenProbes)
	}
}
