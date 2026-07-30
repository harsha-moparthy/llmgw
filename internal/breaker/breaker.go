// Package breaker is the gateway's failover trigger: a per-provider circuit
// breaker whose detection latency is a computable bound rather than a hope,
// plus an active prober that can recover a provider without live traffic.
//
// The headline claim this package has to support is "provider outage failover
// within a bounded window". A bound is only real if you can name it, so the
// breaker exposes WorstCaseFailuresToTrip and FailuresToTrip: the number of
// health-counting failures that can possibly be needed before the breaker
// opens, given the window geometry. Multiply by the inter-arrival time of
// requests to that provider and you have the detection latency, in closed form,
// with no reference to how the code "feels" in a demo.
//
// The design commitments, ordered by how much damage getting them wrong does:
//
//  1. Only evidence about the *provider* moves the breaker. Every outcome
//     arrives with a provider.Class and is filtered through
//     Class.CountsAgainstHealth(). One tenant looping on a malformed tool
//     schema generates an unbounded stream of 400s; if those counted, that
//     tenant could open the breaker on a perfectly healthy provider and deny
//     service to every other tenant. That is a denial-of-service vector handed
//     to any client that can send a bad request, which is all of them.
//
//  2. Failure *ratio* over a sliding window, gated by a minimum sample count.
//     Ratio alone is meaningless on a nearly empty window: one failure out of
//     one sample is a 100% failure rate, and a breaker that trusts it removes a
//     healthy provider from rotation over a single blip.
//
//  3. Re-opening backs off exponentially to a cap, and every cooldown carries
//     additive jitter. A provider that has been down for an hour must not be
//     probed every five seconds for that hour, and N gateway replicas must not
//     all probe the instant the cooldown expires — that synchronised burst is
//     precisely what re-kills a provider that is halfway through recovering.
//
//  4. A mutex, not a lock-free scheme. Allow and Record are on the request
//     path, but their critical sections are a handful of integer operations
//     with no allocation, no syscalls and no callbacks, so an uncontended
//     sync.Mutex costs about one atomic CAS. A lock-free ring plus atomic state
//     machine would have to make the window counts, the state, and the
//     transition counter mutually consistent without a lock, which is a genuine
//     research problem for a component whose whole value is that its behaviour
//     is provable. The lock stays.
package breaker

import (
	"errors"
	"fmt"
	"math/rand/v2"
	"sync"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/provider"
)

// ErrOpen is returned by callers that want an error rather than a boolean from
// a refused admission. The breaker itself never returns it; it is exported so
// that routing code has one canonical sentinel to wrap and test for.
var ErrOpen = errors.New("circuit breaker open")

// State is the breaker's position in the three-state machine.
//
// The transition rules, in full:
//
//	Closed   -> Open:     the sliding window holds at least MinSamples
//	                      health-counting outcomes and the failure ratio over
//	                      the window is >= FailureRatio.
//	Open     -> HalfOpen: the (jittered, backed-off) cooldown has elapsed.
//	                      Evaluated lazily on the next Allow/State/Snapshot, so
//	                      an idle breaker costs nothing — no timers per provider.
//	HalfOpen -> Closed:   HalfOpenSuccesses consecutive counting successes.
//	HalfOpen -> Open:     one counting failure, or HalfOpenTimeout elapses with
//	                      the trial set unresolved. Cooldown grows by
//	                      BackoffFactor, capped at MaxCooldown.
//
// There is deliberately no Closed -> HalfOpen edge and no HalfOpen -> HalfOpen
// edge: a trial round either concludes or the breaker goes back to Open.
type State int

// The three states. StateClosed is the zero value so a Breaker that failed to
// be configured cannot default to "reject everything".
const (
	// StateClosed passes all traffic and watches outcomes.
	StateClosed State = iota
	// StateOpen rejects immediately without an upstream call.
	StateOpen
	// StateHalfOpen admits a bounded number of trial requests.
	StateHalfOpen
)

// String renders the state for logs and metric labels.
func (s State) String() string {
	switch s {
	case StateOpen:
		return "open"
	case StateHalfOpen:
		return "half_open"
	default:
		return "closed"
	}
}

// Config is the breaker's tuning. The zero value is not usable; start from
// DefaultConfig and override.
type Config struct {
	// WindowSize is how many health-counting outcomes the sliding window
	// remembers. It is a count, not a duration: a count-based window has a
	// fixed memory footprint, needs no expiry sweep, and — the reason that
	// matters here — degrades sanely for a provider receiving one request a
	// minute, where a 10-second time window is empty almost always.
	WindowSize int

	// MinSamples is the fewest outcomes in the window before FailureRatio is
	// consulted at all. Must be <= WindowSize.
	MinSamples int

	// FailureRatio is the fraction of the window that must be failures to trip.
	// Stored internally as an integer permille to keep the trip test exact;
	// see New.
	FailureRatio float64

	// Cooldown is the first Open duration, before backoff and jitter.
	Cooldown time.Duration

	// MaxCooldown caps the exponential backoff. Without a cap the backoff walks
	// off to hours and a provider that came back never gets probed.
	MaxCooldown time.Duration

	// BackoffFactor multiplies the cooldown on each consecutive re-open. Must
	// be >= 1; 1 means a flat cooldown.
	BackoffFactor float64

	// JitterFraction is the maximum fraction of the cooldown added as random
	// jitter, in [0, 1]. Jitter is purely additive — see nextCooldownLocked.
	JitterFraction float64

	// HalfOpenProbes is how many trial requests may be in flight at once in
	// HalfOpen. Bounded because the point of HalfOpen is to risk a few requests,
	// not to re-point the entire load at a provider that may still be dead.
	HalfOpenProbes int

	// HalfOpenSuccesses is how many consecutive trial successes close the
	// breaker. Greater than one so that a provider serving one request out of
	// ten does not get the full load back.
	HalfOpenSuccesses int

	// HalfOpenTimeout bounds how long HalfOpen may last without resolving.
	// It exists because Allow and Record are separate calls: a caller that is
	// admitted and then never reports (client hung up, goroutine panicked
	// upstream of Record) would otherwise hold a trial slot forever and wedge
	// the breaker in HalfOpen. On expiry the breaker re-opens *with* backoff,
	// because a trial that never returned is itself the signature of a hung
	// provider. Scale it to your upstream request timeout.
	HalfOpenTimeout time.Duration

	// Now is the clock. Defaults to time.Now. Injected so that every timing
	// assertion in the tests is exact rather than a sleep.
	Now func() time.Time

	// Jitter returns a value in [0, 1). Defaults to rand.Float64. Injected so
	// tests can pin the cooldown to an exact instant.
	Jitter func() float64
}

// DefaultConfig is a starting point tuned for an interactive LLM gateway: with
// a provider taking steady traffic, ten consecutive failures trip the breaker,
// which at 50 rps is 200ms of detection latency and at 1 rps is ten seconds.
func DefaultConfig() Config {
	return Config{
		WindowSize:        20,
		MinSamples:        5,
		FailureRatio:      0.5,
		Cooldown:          5 * time.Second,
		MaxCooldown:       2 * time.Minute,
		BackoffFactor:     2,
		JitterFraction:    0.2,
		HalfOpenProbes:    2,
		HalfOpenSuccesses: 2,
		HalfOpenTimeout:   30 * time.Second,
	}
}

// Validate reports whether the config describes a breaker that can actually
// open and close. Called by New; exported so config loading can reject a bad
// gateway config at startup instead of at the first outage.
func (c Config) Validate() error {
	switch {
	case c.WindowSize <= 0:
		return fmt.Errorf("breaker: WindowSize must be > 0, got %d", c.WindowSize)
	case c.MinSamples <= 0:
		return fmt.Errorf("breaker: MinSamples must be > 0, got %d", c.MinSamples)
	case c.MinSamples > c.WindowSize:
		// Otherwise the window can never hold enough samples to trip and the
		// breaker is decorative.
		return fmt.Errorf("breaker: MinSamples (%d) must be <= WindowSize (%d)", c.MinSamples, c.WindowSize)
	case c.FailureRatio <= 0 || c.FailureRatio > 1:
		return fmt.Errorf("breaker: FailureRatio must be in (0, 1], got %v", c.FailureRatio)
	case c.Cooldown <= 0:
		return fmt.Errorf("breaker: Cooldown must be > 0, got %v", c.Cooldown)
	case c.MaxCooldown < c.Cooldown:
		return fmt.Errorf("breaker: MaxCooldown (%v) must be >= Cooldown (%v)", c.MaxCooldown, c.Cooldown)
	case c.BackoffFactor < 1:
		return fmt.Errorf("breaker: BackoffFactor must be >= 1, got %v", c.BackoffFactor)
	case c.JitterFraction < 0 || c.JitterFraction > 1:
		return fmt.Errorf("breaker: JitterFraction must be in [0, 1], got %v", c.JitterFraction)
	case c.HalfOpenProbes <= 0:
		return fmt.Errorf("breaker: HalfOpenProbes must be > 0, got %d", c.HalfOpenProbes)
	case c.HalfOpenSuccesses <= 0:
		return fmt.Errorf("breaker: HalfOpenSuccesses must be > 0, got %d", c.HalfOpenSuccesses)
	case c.HalfOpenTimeout <= 0:
		return fmt.Errorf("breaker: HalfOpenTimeout must be > 0, got %v", c.HalfOpenTimeout)
	}
	return nil
}

// WorstCaseFailuresToTrip is the largest number of consecutive
// health-counting failures that can be required to open the breaker, from any
// window state.
//
// The worst case is a window completely full of successes: each failure then
// evicts a success, so after k failures the ratio is exactly k/WindowSize and
// the breaker trips at the first k with k/WindowSize >= FailureRatio. A window
// that is only partly full trips no later than that, because the denominator is
// smaller. MinSamples can dominate when the window starts empty.
//
// This is the multiplier in the failover bound: detection latency <=
// WorstCaseFailuresToTrip() * (inter-arrival time of requests to the provider).
func (c Config) WorstCaseFailuresToTrip() int {
	if err := c.Validate(); err != nil {
		return 0
	}
	permille := ratioPermille(c.FailureRatio)
	k := 0
	for k < c.WindowSize {
		k++
		if int64(k)*1000 >= permille*int64(c.WindowSize) {
			break
		}
	}
	if k < c.MinSamples {
		k = c.MinSamples
	}
	return k
}

// DetectionBound converts WorstCaseFailuresToTrip into wall-clock time for a
// provider receiving a request every interval. This is the number the failover
// harness asserts against.
func (c Config) DetectionBound(interval time.Duration) time.Duration {
	return time.Duration(c.WorstCaseFailuresToTrip()) * interval
}

// ratioPermille converts a fractional ratio to integer permille, rounding to
// nearest. The trip test is then pure integer arithmetic (failures*1000 >=
// permille*samples), which matters at the boundary: with a float ratio,
// "2 failures out of 6 >= 1.0/3.0" is a coin flip decided by the last bit of
// the division, and a breaker whose trip point depends on rounding mode is not
// a breaker whose behaviour you can state in a README. The cost is that ratios
// are quantised to 0.1% — far finer than any window size worth configuring.
func ratioPermille(r float64) int64 {
	return int64(r*1000 + 0.5)
}

// Breaker is one provider instance's circuit breaker. Safe for concurrent use.
type Breaker struct {
	name string

	// Immutable after New, so read without the lock.
	window       int
	minSamples   int
	permille     int64
	baseCooldown time.Duration
	maxCooldown  time.Duration
	backoff      float64
	jitterFrac   float64
	halfOpenMax  int
	halfOpenWant int
	halfOpenTTL  time.Duration
	now          func() time.Time
	jitter       func() float64

	mu sync.Mutex
	// ring[i] is true when outcome i was a health-counting failure. Allocated
	// once in New and never grown, so Record allocates nothing. A []bool rather
	// than a bitset: one byte per sample for a window of tens of entries is
	// noise, and the bit twiddling would obscure the eviction accounting, which
	// is the part that is easy to get wrong.
	ring     []bool
	next     int
	filled   int
	failures int

	state            State
	openedAt         time.Time
	nextProbeAt      time.Time
	halfOpenExpires  time.Time
	consecutiveOpens int
	cooldown         time.Duration
	hoInFlight       int
	hoSuccesses      int

	transitions uint64
	opens       uint64
	rejected    uint64
	ignored     uint64
	stale       uint64
}

// New builds a breaker for the named provider instance.
func New(name string, cfg Config) (*Breaker, error) {
	if name == "" {
		return nil, errors.New("breaker: name must not be empty")
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	now := cfg.Now
	if now == nil {
		now = time.Now
	}
	jitter := cfg.Jitter
	if jitter == nil {
		// rand.Float64 from math/rand/v2 is safe for concurrent use and needs
		// no seeding, so there is no shared *rand.Rand to lock around.
		jitter = rand.Float64
	}
	return &Breaker{
		name:         name,
		window:       cfg.WindowSize,
		minSamples:   cfg.MinSamples,
		permille:     ratioPermille(cfg.FailureRatio),
		baseCooldown: cfg.Cooldown,
		maxCooldown:  cfg.MaxCooldown,
		backoff:      cfg.BackoffFactor,
		jitterFrac:   cfg.JitterFraction,
		halfOpenMax:  cfg.HalfOpenProbes,
		halfOpenWant: cfg.HalfOpenSuccesses,
		halfOpenTTL:  cfg.HalfOpenTimeout,
		now:          now,
		jitter:       jitter,
		ring:         make([]bool, cfg.WindowSize),
		state:        StateClosed,
	}, nil
}

// Name is the provider instance this breaker guards.
func (b *Breaker) Name() string { return b.name }

// Allow reports whether a request may be sent upstream right now, and moves the
// state machine forward if a deadline has passed.
//
// Every true return in HalfOpen consumes a trial slot that is released by the
// matching Record. Callers must therefore pair every Allow()==true with exactly
// one Record; HalfOpenTimeout is the safety net for callers that do not.
func (b *Breaker) Allow() bool {
	now := b.now()
	b.mu.Lock()
	defer b.mu.Unlock()
	b.refreshLocked(now)

	switch b.state {
	case StateOpen:
		b.rejected++
		return false
	case StateHalfOpen:
		if b.hoInFlight >= b.halfOpenMax {
			b.rejected++
			return false
		}
		b.hoInFlight++
		return true
	default:
		return true
	}
}

// Record feeds one classified outcome back in. A nil failure is a success.
// This is the form the request path has on hand, since provider.Provider
// returns *provider.Failure.
func (b *Breaker) Record(f *provider.Failure) {
	if f == nil {
		b.RecordSuccess()
		return
	}
	b.RecordFailure(f.Class)
}

// RecordSuccess reports a completed upstream call.
func (b *Breaker) RecordSuccess() { b.record(true, provider.ClassUnknown) }

// RecordFailure reports a failed upstream call.
//
// The class is not decoration. A failure whose Class.CountsAgainstHealth() is
// false never reaches the window and never resolves a HalfOpen trial: it only
// releases the trial slot. This is what stops a client hammering malformed
// requests from opening the breaker on a healthy provider and taking it away
// from every other tenant.
func (b *Breaker) RecordFailure(class provider.Class) { b.record(false, class) }

func (b *Breaker) record(ok bool, class provider.Class) {
	now := b.now()
	counts := ok || class.CountsAgainstHealth()

	b.mu.Lock()
	defer b.mu.Unlock()
	b.refreshLocked(now)

	switch b.state {
	case StateHalfOpen:
		// Release the slot first, unconditionally: the request is over whether
		// or not its outcome tells us anything.
		if b.hoInFlight > 0 {
			b.hoInFlight--
		}
		if !counts {
			b.ignored++
			return
		}
		if !ok {
			// One trial failure is enough. Insisting on a ratio here would send
			// more real traffic into a provider we already have fresh evidence
			// is broken.
			b.openLocked(now)
			return
		}
		b.hoSuccesses++
		if b.hoSuccesses >= b.halfOpenWant {
			b.closeLocked()
		}
	case StateOpen:
		// An outcome arriving while Open is from a request admitted before the
		// trip (or from a caller ignoring Allow). It says nothing about now, and
		// letting a stale success close the breaker would undo the trip. Drop it
		// and count it so the drop is visible.
		//
		// Distinguishing "stale" from "current" properly would mean handing a
		// generation token out of Allow and back into Record. That is the
		// textbook design, and it is rejected here because it puts a token in
		// every call site on the request path to fix an accounting rounding
		// error: the only outcomes it would recover are ones that arrive after
		// the breaker has already decided.
		b.stale++
	default:
		if !counts {
			b.ignored++
			return
		}
		b.pushLocked(!ok)
		if b.shouldTripLocked() {
			b.openLocked(now)
		}
	}
}

// pushLocked appends one outcome to the ring, evicting the oldest when full and
// keeping the failure count in step with the eviction.
func (b *Breaker) pushLocked(failure bool) {
	if b.filled == b.window {
		if b.ring[b.next] {
			b.failures--
		}
	} else {
		b.filled++
	}
	b.ring[b.next] = failure
	if failure {
		b.failures++
	}
	b.next = (b.next + 1) % b.window
}

func (b *Breaker) shouldTripLocked() bool {
	if b.filled < b.minSamples {
		return false
	}
	return int64(b.failures)*1000 >= b.permille*int64(b.filled)
}

// refreshLocked applies any time-driven transition. Called at the top of every
// public entry point so that the state a caller observes is never stale, and so
// that an idle provider needs no timer goroutine of its own.
func (b *Breaker) refreshLocked(now time.Time) {
	switch b.state {
	case StateOpen:
		if !now.Before(b.nextProbeAt) {
			b.toHalfOpenLocked(now)
		}
	case StateHalfOpen:
		if !now.Before(b.halfOpenExpires) {
			b.openLocked(now)
		}
	}
}

func (b *Breaker) openLocked(now time.Time) {
	b.consecutiveOpens++
	b.cooldown = b.nextCooldownLocked()
	b.state = StateOpen
	b.openedAt = now
	b.nextProbeAt = now.Add(b.cooldown)
	b.hoInFlight = 0
	b.hoSuccesses = 0
	// Clear the window. Keeping it would leave the ring full of the failures
	// that caused the trip, so the first counting failure after the breaker
	// eventually closed would re-trip it instantly — the breaker would be
	// unable to stay closed after its first outage.
	b.resetWindowLocked()
	b.transitions++
	b.opens++
}

func (b *Breaker) toHalfOpenLocked(now time.Time) {
	b.state = StateHalfOpen
	b.hoInFlight = 0
	b.hoSuccesses = 0
	b.halfOpenExpires = now.Add(b.halfOpenTTL)
	b.transitions++
}

func (b *Breaker) closeLocked() {
	b.state = StateClosed
	b.consecutiveOpens = 0
	b.cooldown = 0
	b.openedAt = time.Time{}
	b.nextProbeAt = time.Time{}
	b.halfOpenExpires = time.Time{}
	b.hoInFlight = 0
	b.hoSuccesses = 0
	b.resetWindowLocked()
	b.transitions++
}

func (b *Breaker) resetWindowLocked() {
	for i := range b.ring {
		b.ring[i] = false
	}
	b.next = 0
	b.filled = 0
	b.failures = 0
}

// nextCooldownLocked computes the Open duration for the current consecutive-open
// count: base * BackoffFactor^(opens-1), capped at MaxCooldown, plus additive
// jitter of up to JitterFraction of that.
//
// Two deliberate choices.
//
// Exponential-with-a-cap, because a provider that has been down for an hour
// should not be probed every five seconds for that hour: each failed probe is a
// real request against a broken upstream, and across many replicas that is a
// steady load aimed at the one machine that is trying to recover. The cap is
// what guarantees a provider that *does* come back is noticed within a bounded
// time instead of after a backoff that walked off to hours.
//
// Jitter is additive, never subtractive, so the configured cooldown is a floor
// the breaker will not probe before. Without jitter, N replicas that observed
// the same outage open at nearly the same instant and therefore probe at the
// same instant; the recovering provider takes a synchronised burst of N probes,
// fails them, and every replica re-opens in lockstep — a thundering herd that
// converts a recovering provider back into a dead one, and keeps the replicas
// phase-locked so it happens again on the next cycle. Spreading the probes over
// [cooldown, cooldown*(1+JitterFraction)) breaks the phase lock.
//
// The multiplication is iterative rather than math.Pow because the loop can
// stop the moment it reaches the cap, which also means it cannot overflow
// time.Duration no matter how many consecutive opens accumulate.
func (b *Breaker) nextCooldownLocked() time.Duration {
	d := b.baseCooldown
	for i := 1; i < b.consecutiveOpens; i++ {
		d = time.Duration(float64(d) * b.backoff)
		if d >= b.maxCooldown {
			d = b.maxCooldown
			break
		}
	}
	if d > b.maxCooldown {
		d = b.maxCooldown
	}
	if b.jitterFrac > 0 {
		j := b.jitter()
		if j < 0 {
			j = 0
		} else if j >= 1 {
			// Keep the interval half-open so a jitter source pinned at 1 cannot
			// return a cooldown longer than the documented maximum.
			j = 1 - 1e-9
		}
		d += time.Duration(float64(d) * b.jitterFrac * j)
	}
	return d
}

// State is the breaker's current state, after applying any elapsed deadline.
func (b *Breaker) State() State {
	now := b.now()
	b.mu.Lock()
	defer b.mu.Unlock()
	b.refreshLocked(now)
	return b.state
}

// UntilNextProbe is how long until the breaker will admit a request again. Zero
// when it will admit one now (Closed, or HalfOpen with a free trial slot).
func (b *Breaker) UntilNextProbe() time.Duration {
	now := b.now()
	b.mu.Lock()
	defer b.mu.Unlock()
	b.refreshLocked(now)
	return b.untilNextProbeLocked(now)
}

func (b *Breaker) untilNextProbeLocked(now time.Time) time.Duration {
	if b.state != StateOpen {
		return 0
	}
	d := b.nextProbeAt.Sub(now)
	if d < 0 {
		return 0
	}
	return d
}

// Transitions is the total number of state changes since New. The metrics layer
// exports it as a counter; a breaker flapping between Open and HalfOpen shows up
// as a rate here long before it shows up in error budgets.
func (b *Breaker) Transitions() uint64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.transitions
}

// FailuresToTrip is how many further consecutive health-counting failures would
// open the breaker from its current window contents, or 0 if it is not Closed.
//
// It is the live version of WorstCaseFailuresToTrip and it is what makes the
// failover bound checkable at runtime rather than only on paper.
func (b *Breaker) FailuresToTrip() int {
	now := b.now()
	b.mu.Lock()
	defer b.mu.Unlock()
	b.refreshLocked(now)
	if b.state != StateClosed {
		return 0
	}
	return b.failuresToTripLocked()
}

// Snapshot is a consistent view of the breaker for metrics, admin endpoints and
// the failover measurement. Every field is read under one lock acquisition,
// because a metrics scrape that reported State=Closed alongside a full window
// of failures would be actively misleading during the exact incident it exists
// to explain.
type Snapshot struct {
	Name  string
	State State

	// Samples is the number of health-counting outcomes in the window,
	// Failures how many of those were failures.
	Samples   int
	Failures  int
	Successes int
	// FailureRatio is Failures/Samples, or 0 for an empty window.
	FailureRatio float64
	// FailuresToTrip is how many more consecutive counting failures would open
	// the breaker; 0 when it is not Closed.
	FailuresToTrip int

	// OpenedAt is when the current Open period began; zero unless Open.
	OpenedAt time.Time
	// NextProbeAt is when Open becomes HalfOpen; zero unless Open.
	NextProbeAt time.Time
	// UntilNextProbe is NextProbeAt minus now, floored at zero.
	UntilNextProbe time.Duration
	// Cooldown is the jittered, backed-off duration of the current Open period.
	Cooldown time.Duration
	// ConsecutiveOpens drives the backoff; reset when the breaker closes.
	ConsecutiveOpens int

	// HalfOpenInFlight is the number of trial requests currently admitted,
	// HalfOpenSuccesses the consecutive successes accumulated so far.
	HalfOpenInFlight  int
	HalfOpenSuccesses int

	// Transitions counts every state change. Opens counts entries into Open.
	Transitions uint64
	Opens       uint64
	// Rejected counts admissions refused (Open, or HalfOpen at its trial limit).
	Rejected uint64
	// Ignored counts failures discarded because the class was not evidence
	// about the provider. A large and growing value here means a client is
	// generating bad requests, not that a provider is sick — the single most
	// useful disambiguation during an incident.
	Ignored uint64
	// Stale counts outcomes dropped because they arrived while Open.
	Stale uint64
}

// Snapshot reads the breaker's observable state.
func (b *Breaker) Snapshot() Snapshot {
	now := b.now()
	b.mu.Lock()
	defer b.mu.Unlock()
	b.refreshLocked(now)

	s := Snapshot{
		Name:              b.name,
		State:             b.state,
		Samples:           b.filled,
		Failures:          b.failures,
		Successes:         b.filled - b.failures,
		OpenedAt:          b.openedAt,
		NextProbeAt:       b.nextProbeAt,
		UntilNextProbe:    b.untilNextProbeLocked(now),
		Cooldown:          b.cooldown,
		ConsecutiveOpens:  b.consecutiveOpens,
		HalfOpenInFlight:  b.hoInFlight,
		HalfOpenSuccesses: b.hoSuccesses,
		Transitions:       b.transitions,
		Opens:             b.opens,
		Rejected:          b.rejected,
		Ignored:           b.ignored,
		Stale:             b.stale,
	}
	if b.filled > 0 {
		s.FailureRatio = float64(b.failures) / float64(b.filled)
	}
	if b.state == StateClosed {
		s.FailuresToTrip = b.failuresToTripLocked()
	}
	return s
}

// failuresToTripLocked replays hypothetical failures against a read-only view
// of the live ring — the same eviction arithmetic as pushLocked, with no copy
// and no allocation — so both FailuresToTrip and Snapshot can compute it inside
// a single critical section.
func (b *Breaker) failuresToTripLocked() int {
	f, s, idx := b.failures, b.filled, b.next
	for k := 1; k <= 2*b.window; k++ {
		if s == b.window {
			if b.ring[idx] {
				f--
			}
		} else {
			s++
		}
		f++
		idx = (idx + 1) % b.window
		if s >= b.minSamples && int64(f)*1000 >= b.permille*int64(s) {
			return k
		}
	}
	// Unreachable for a validated config: after WindowSize failures the window
	// holds nothing but failures, so the ratio is 1.0 and MinSamples (<=
	// WindowSize) is satisfied.
	return 0
}

// Trip forces the breaker Open, applying the normal backoff and jitter. For an
// operator taking a provider out of rotation by hand; it counts as an open, so
// a human intervention is visible in the same counter as an automatic one.
func (b *Breaker) Trip() {
	now := b.now()
	b.mu.Lock()
	defer b.mu.Unlock()
	b.openLocked(now)
}

// Reset forces the breaker Closed and clears the window and the backoff. For an
// operator who has fixed the underlying problem (rotated a bad API key, say)
// and does not want to wait out a cooldown that is by then irrelevant.
func (b *Breaker) Reset() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.closeLocked()
}

// Registry holds one breaker per provider instance, all sharing a Config.
//
// Provider instances are identified by name and not by vendor (see
// provider.Provider): fronting the same vendor twice with different keys or
// regions gives two independently failing routes, and one breaker across both
// would take a healthy key out of rotation because the other one expired.
type Registry struct {
	cfg Config

	mu       sync.RWMutex
	breakers map[string]*Breaker
}

// NewRegistry validates the shared config once, up front, so a
// misconfiguration surfaces at startup rather than on the first Get.
func NewRegistry(cfg Config) (*Registry, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &Registry{cfg: cfg, breakers: make(map[string]*Breaker)}, nil
}

// Get returns the breaker for a provider instance, creating it on first use.
// Safe for concurrent use; the fast path takes only a read lock, because on the
// request path this is a lookup of an existing breaker essentially always.
func (r *Registry) Get(name string) *Breaker {
	r.mu.RLock()
	b := r.breakers[name]
	r.mu.RUnlock()
	if b != nil {
		return b
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	// Re-check: two goroutines can both miss the read lock, and handing them
	// different breakers for the same provider would split its health state.
	if b = r.breakers[name]; b != nil {
		return b
	}
	b, err := New(name, r.cfg)
	if err != nil {
		// Unreachable: NewRegistry validated the config, and the only other
		// failure mode is an empty name.
		panic("breaker: registry misconfigured: " + err.Error())
	}
	r.breakers[name] = b
	return b
}

// Snapshots returns a snapshot per breaker, for a metrics scrape.
func (r *Registry) Snapshots() []Snapshot {
	r.mu.RLock()
	bs := make([]*Breaker, 0, len(r.breakers))
	for _, b := range r.breakers {
		bs = append(bs, b)
	}
	r.mu.RUnlock()

	// Snapshot outside the registry lock: each breaker takes its own, and
	// holding the registry lock across all of them would make a scrape
	// contend with every provider's request path at once.
	out := make([]Snapshot, 0, len(bs))
	for _, b := range bs {
		out = append(out, b.Snapshot())
	}
	return out
}

// Config returns the shared config the registry hands to new breakers.
func (r *Registry) Config() Config { return r.cfg }
