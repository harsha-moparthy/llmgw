// Package budget enforces per-tenant spend limits with reserve-then-true-up
// mechanics, and its headline claim is an over-admission bound of zero.
//
// THE BOUND. At every instant, for every tenant, the sum of the spend admitted
// against a fixed window and the holds outstanding against it satisfies
//
//	spent + reserved <= limit
//
// with no slack term, under arbitrarily many concurrent Reserve calls. That is
// achieved the boring way: the "is there room?" test and the "take the room"
// mutation are one critical section under one mutex, so there is no window in
// which two callers can both observe the same headroom. The alternative that
// looks cleaner — an atomic read of the remaining budget followed by an atomic
// add — admits N concurrent callers into headroom that only fits one, and the
// overshoot scales with concurrency exactly when a tenant is being hammered.
// TestOverAdmissionBoundUnderConcurrency fires hundreds of concurrent Reserves
// at a budget that fits three of them and asserts the bound directly.
//
// The bound is on ADMISSION, not on final spend, and the two places it can be
// exceeded are both named and tested:
//
//  1. An under-estimate. Commit records the true cost, which may exceed the hold
//     it replaces. The excess is bounded by the sum of (actual - estimate) over
//     requests in flight. Because internal/tokens over-estimates by
//     construction, this term is zero in normal operation — but it is a real
//     term, so the package does not pretend otherwise.
//  2. A hold swept while its request is still running. See Options.HoldTTL: the
//     TTL must exceed the longest request deadline, or the sweep itself becomes
//     a source of over-admission.
//
// THE COST OF BEING CONSERVATIVE. Because the estimate is an upper bound, a
// tenant is admission-limited by a number larger than what they will actually be
// charged: with an estimate k times the true cost, only 1/k of the true
// concurrent capacity is admitted at once. Nobody is ever *charged* for the
// over-estimate — Commit returns the unused hold immediately — so sustained
// throughput is unaffected; only simultaneity is. TestOverEstimateCostsAdmission
// NotMoney quantifies both halves of that sentence.
package budget

import (
	"errors"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/money"
)

// Period is a fixed-window length.
type Period int

// Periods. PeriodUnset is the zero value and is invalid for a non-zero limit,
// rather than silently defaulting to one of the others: a config that forgot to
// say "daily" must be rejected at load time, not guessed at.
const (
	PeriodUnset Period = iota
	PeriodHourly
	PeriodDaily
	PeriodMonthly
)

// String renders the period for logs, metric labels, and error bodies.
func (p Period) String() string {
	switch p {
	case PeriodHourly:
		return "hourly"
	case PeriodDaily:
		return "daily"
	case PeriodMonthly:
		return "monthly"
	default:
		return "unset"
	}
}

// Valid reports whether the period names a real window.
func (p Period) Valid() bool {
	return p == PeriodHourly || p == PeriodDaily || p == PeriodMonthly
}

// ParsePeriod parses a config string.
func ParsePeriod(s string) (Period, error) {
	switch s {
	case "hourly", "hour", "1h":
		return PeriodHourly, nil
	case "daily", "day", "1d":
		return PeriodDaily, nil
	case "monthly", "month", "1mo":
		return PeriodMonthly, nil
	default:
		return PeriodUnset, fmt.Errorf("budget: unknown period %q (want hourly, daily, or monthly)", s)
	}
}

// WindowStart returns the start of the fixed window containing t.
//
// THE BOUNDARY RULE, stated once so nobody has to infer it: windows are aligned
// to UTC and are half-open, [start, end). An instant exactly on a boundary
// belongs to the window that is starting, never to the one that is ending.
//
// UTC and not tenant-local, deliberately. A tenant-local boundary needs a
// timezone per tenant, and then a daily window under DST is either 23 or 25
// hours long and an hourly window can repeat, so "which window am I in?" stops
// having one answer during the repeated hour. That is a genuinely nasty class of
// billing bug in exchange for a cosmetic benefit, so the trade is refused here
// and the reset instant is reported to the client instead (see Decision.ResetIn),
// which is the thing a client actually needs.
//
// Monthly windows are calendar months, not 30-day blocks, because a monthly
// spend cap is a finance concept and finance means February.
func (p Period) WindowStart(t time.Time) time.Time {
	u := t.UTC()
	switch p {
	case PeriodHourly:
		return time.Date(u.Year(), u.Month(), u.Day(), u.Hour(), 0, 0, 0, time.UTC)
	case PeriodDaily:
		return time.Date(u.Year(), u.Month(), u.Day(), 0, 0, 0, 0, time.UTC)
	case PeriodMonthly:
		return time.Date(u.Year(), u.Month(), 1, 0, 0, 0, 0, time.UTC)
	default:
		return u
	}
}

// WindowEnd returns the exclusive end of the window starting at start.
func (p Period) WindowEnd(start time.Time) time.Time {
	switch p {
	case PeriodHourly:
		return start.Add(time.Hour)
	case PeriodDaily:
		return start.AddDate(0, 0, 1)
	case PeriodMonthly:
		return start.AddDate(0, 1, 0)
	default:
		return start
	}
}

// Limit is one tenant's spend cap.
type Limit struct {
	// Amount is the hard cap per window. Zero or negative means unlimited; see
	// Budget.SetLimit for why that is the fail-open default.
	Amount money.Pico
	// Period is the window length.
	Period Period
	// SoftBasisPoints is the utilisation at which Reserve starts returning
	// AllowDegraded, in basis points of Amount (8000 = 80%). Zero disables the
	// soft threshold, so a tenant is either fine or rejected with nothing in
	// between.
	SoftBasisPoints int
}

// Unlimited reports whether this limit imposes no cap.
func (l Limit) Unlimited() bool { return l.Amount <= 0 }

// SoftThreshold is the absolute spend at which degradation begins, or 0 when the
// soft threshold is disabled.
//
// The multiplication is done before the division so the only precision loss is a
// single truncation of the final basis-point remainder, which moves the
// threshold by less than a picodollar.
func (l Limit) SoftThreshold() money.Pico {
	if l.Unlimited() || l.SoftBasisPoints <= 0 {
		return 0
	}
	if l.SoftBasisPoints >= 10000 {
		return l.Amount
	}
	return money.Pico(int64(l.Amount) * int64(l.SoftBasisPoints) / 10000)
}

// Validate checks the limit is usable.
func (l Limit) Validate() error {
	if l.Amount < 0 {
		return fmt.Errorf("budget: negative limit %d pico", int64(l.Amount))
	}
	if l.Amount > 0 && !l.Period.Valid() {
		return errors.New("budget: a non-zero limit needs a period (hourly, daily, or monthly)")
	}
	if l.SoftBasisPoints < 0 || l.SoftBasisPoints > 10000 {
		return fmt.Errorf("budget: soft threshold %d basis points is outside 0..10000",
			l.SoftBasisPoints)
	}
	return nil
}

// Outcome is the admission decision.
type Outcome int

// Outcomes.
const (
	// Allow: there is room, proceed normally.
	Allow Outcome = iota
	// AllowDegraded: there is room, but the tenant is past the soft threshold.
	//
	// This is a ROUTING HINT AND NOTHING MORE. The budget package does not know
	// which models are cheap, does not know whether a cheaper model can serve
	// this request, and does not refuse anything here — the request is admitted
	// either way. The router MAY act on it by picking a cheaper route; if it
	// ignores the hint the only consequence is that the tenant reaches the hard
	// limit sooner. Encoding it as an outcome rather than as a separate
	// out-of-band signal keeps the numbers that justify it attached to the
	// decision that produced them.
	AllowDegraded
	// Reject: the hard limit would be exceeded. The request is not admitted and
	// no hold was taken.
	Reject
)

// String renders the outcome for logs and metric labels.
func (o Outcome) String() string {
	switch o {
	case Allow:
		return "allow"
	case AllowDegraded:
		return "allow_degraded"
	case Reject:
		return "reject"
	default:
		return "unknown"
	}
}

// Admitted reports whether the request may proceed.
func (o Outcome) Admitted() bool { return o == Allow || o == AllowDegraded }

// Decision is the full justification for an admission outcome.
//
// It is a struct and not a bool because the spec asks for "clear rejection
// semantics": a client that gets a 429 needs to be told the limit, what has been
// spent against it, what is held for in-flight requests, and above all WHEN it
// resets, or its only recourse is to poll. Every field here exists to be
// rendered into a 4xx body or a log line.
type Decision struct {
	Outcome Outcome `json:"outcome"`
	Tenant  string  `json:"tenant"`

	Limit money.Pico `json:"limit_pico"`
	// Spent is settled spend against the current window.
	Spent money.Pico `json:"spent_pico"`
	// Reserved is the total of outstanding holds against the current window,
	// including this request's hold when it was admitted.
	Reserved money.Pico `json:"reserved_pico"`
	// Estimate is the amount this call asked to hold.
	Estimate money.Pico `json:"estimate_pico"`
	// Remaining is limit - spent - reserved evaluated AFTER this decision was
	// applied. On Allow it therefore already excludes this request's own hold,
	// which is the number a caller wants for a "you have $X left" header; on
	// Reject nothing was applied, so it is the headroom that proved insufficient.
	Remaining money.Pico `json:"remaining_pico"`
	// SoftThreshold is the absolute spend at which degradation starts, 0 if off.
	SoftThreshold money.Pico `json:"soft_threshold_pico,omitempty"`

	Period      Period    `json:"period"`
	PeriodStart time.Time `json:"period_start"`
	PeriodEnd   time.Time `json:"period_end"`
	// ResetIn is the time until the window rolls. This is the field that makes a
	// rejection actionable rather than merely rude.
	ResetIn time.Duration `json:"reset_in"`

	// Unlimited means no cap is configured for this tenant.
	Unlimited bool `json:"unlimited,omitempty"`
	// Reason is a short machine-stable explanation.
	Reason string `json:"reason,omitempty"`
}

// Reason codes. Stable strings so a dashboard can group on them.
const (
	ReasonWithinLimit    = "within_limit"
	ReasonNoLimit        = "no_limit_configured"
	ReasonSoftThreshold  = "soft_threshold_exceeded"
	ReasonHardLimit      = "hard_limit_exceeded"
	ReasonEstimateTooBig = "estimate_exceeds_limit"
	ReasonOverflow       = "arithmetic_overflow"
)

// HTTPStatus is the status a rejected request should be answered with.
//
// 429, not 402, and the choice matters because the two say different things to a
// client's retry logic.
//
// 402 Payment Required means "a human must do something before this will ever
// work". OpenAI-compatible SDKs treat a non-429 4xx as terminal and stop, which
// is correct for a genuine billing failure and wrong here: this limit heals by
// itself at a known instant, with no human involved. Answering 402 would make
// every client abandon a request that would succeed on its own in an hour.
//
// 429 says "not now, try later", which is precisely true. Its weakness is that
// clients read it as "retry with backoff in a second or two", and a budget may
// not reset for a month — so Retry-After is MANDATORY on this path, not
// optional, and RetryAfter below returns the real distance to the window
// boundary rather than a polite small number. A client that honours Retry-After
// then does exactly the right thing: it stops until the budget actually resets.
// A client that ignores it burns cheap 429s instead of expensive upstream calls,
// which is the correct failure mode to have.
//
// (If this gateway ever grew a non-resetting lifetime cap, THAT would be a 402:
// the distinction is not "budget vs rate" but "does waiting fix it".)
func (d Decision) HTTPStatus() int {
	if d.Outcome == Reject {
		return http.StatusTooManyRequests
	}
	return http.StatusOK
}

// RetryAfter is the value for the Retry-After header, rounded up to whole
// seconds because the header has second granularity and rounding DOWN would
// invite a retry that is still inside the exhausted window.
func (d Decision) RetryAfter() time.Duration {
	if d.Outcome != Reject || d.ResetIn <= 0 {
		return 0
	}
	secs := (d.ResetIn + time.Second - 1) / time.Second
	return secs * time.Second
}

// Message is a client-facing explanation, precise about both the numbers and the
// reset instant.
func (d Decision) Message() string {
	switch d.Outcome {
	case Reject:
		if d.Reason == ReasonEstimateTooBig {
			return fmt.Sprintf(
				"request rejected: its estimated cost %s exceeds the entire %s budget of %s for tenant %q; "+
					"no retry will help — reduce max_tokens or the prompt size",
				money.FormatUSD(d.Estimate), d.Period, money.FormatUSD(d.Limit), d.Tenant)
		}
		return fmt.Sprintf(
			"%s budget exceeded for tenant %q: limit %s, spent %s, reserved for in-flight requests %s, "+
				"remaining %s; estimated cost of this request %s. Budget resets at %s (in %s).",
			d.Period, d.Tenant, money.FormatUSD(d.Limit), money.FormatUSD(d.Spent),
			money.FormatUSD(d.Reserved), money.FormatUSD(d.Remaining),
			money.FormatUSD(d.Estimate),
			d.PeriodEnd.Format(time.RFC3339), d.ResetIn.Round(time.Second))
	case AllowDegraded:
		return fmt.Sprintf(
			"tenant %q is past its soft budget threshold %s (spent %s of %s); "+
				"routing to a cheaper model is advised",
			d.Tenant, money.FormatUSD(d.SoftThreshold), money.FormatUSD(d.Spent),
			money.FormatUSD(d.Limit))
	default:
		if d.Unlimited {
			return fmt.Sprintf("tenant %q has no configured budget", d.Tenant)
		}
		return fmt.Sprintf("tenant %q within budget: %s of %s remaining",
			d.Tenant, money.FormatUSD(d.Remaining), money.FormatUSD(d.Limit))
	}
}

// Reservation is a hold on a tenant's remaining budget.
//
// It is a value type carrying an opaque ID rather than a pointer to internal
// state, so a caller cannot mutate a live hold, and a Commit for a hold that has
// already been resolved is detected instead of corrupting the counters.
type Reservation struct {
	ID     uint64     `json:"id"`
	Tenant string     `json:"tenant"`
	Amount money.Pico `json:"amount_pico"`
	// WindowStart identifies the window that admitted this hold. Spend is
	// attributed back to it even if the window has since rolled; see Commit.
	WindowStart time.Time `json:"window_start"`
	ExpiresAt   time.Time `json:"expires_at"`
}

// Valid reports whether this is a real hold. A rejected Reserve returns the zero
// value, for which Valid is false.
func (r Reservation) Valid() bool { return r.ID != 0 }

// Errors returned by the resolution paths.
var (
	// ErrNoReservation is returned when Commit or Release is handed a zero or
	// unknown reservation. Most often it means double resolution: a handler with
	// both a deferred Release and an explicit Commit.
	ErrNoReservation = errors.New("budget: unknown or already-resolved reservation")
	// ErrExpiredReservation is returned when Commit arrives after the sweep has
	// already released the hold. The cost is still recorded — see Commit — so
	// this is a warning about handler latency, not a lost charge.
	ErrExpiredReservation = errors.New("budget: reservation had already expired and been swept")
)

// hold is the internal record of a reservation. The prev/next pointers make it a
// node in the global expiry queue so a sweep is O(expired) and a Commit unlinks
// in O(1) — a sweep that scanned every outstanding hold would get slower exactly
// as concurrency rose.
type hold struct {
	id          uint64
	tenant      string
	amount      money.Pico
	windowStart int64
	expiresAt   time.Time
	prev, next  *hold
}

// window is the per-tenant, per-period counter pair.
type window struct {
	start    time.Time
	end      time.Time
	spent    money.Pico
	reserved money.Pico
	// holds is the number of outstanding reservations admitted by this window.
	// A non-current window is kept alive only while this is non-zero, which is
	// what stops a period rollover from losing an in-flight hold.
	holds int
}

type tenantState struct {
	limit Limit
	cur   *window
	// past holds windows that have rolled but still have outstanding holds,
	// keyed by window start. Bounded in practice by HoldTTL/period length + 1,
	// i.e. one entry, because a hold cannot outlive its TTL.
	past map[int64]*window
}

// Stats are package-level counters for the metrics endpoint.
type Stats struct {
	Reserved  int64 `json:"reserved"`
	Allowed   int64 `json:"allowed"`
	Degraded  int64 `json:"degraded"`
	Rejected  int64 `json:"rejected"`
	Committed int64 `json:"committed"`
	Released  int64 `json:"released"`
	// Expired counts holds reclaimed by the sweep, i.e. handlers that never
	// resolved their reservation. A non-zero and growing value is a leak in the
	// request path, and it is exported rather than merely logged because the
	// sweep silently papering over it is how the leak stays invisible.
	Expired int64 `json:"expired"`
	// LateCommits counts Commits that arrived after their hold was swept.
	LateCommits int64 `json:"late_commits"`
	// Unreserved counts Charge calls: spend recorded with no prior hold.
	Unreserved int64 `json:"unreserved_charges"`
	// InvalidEstimates counts Reserve calls with a negative estimate, which is a
	// caller bug. Clamped to zero rather than rejected — refusing traffic over an
	// accounting mistake is the wrong trade — but counted so the bug surfaces.
	InvalidEstimates int64 `json:"invalid_estimates"`
	// Rollovers counts period boundaries crossed.
	Rollovers int64 `json:"rollovers"`
}

// TenantUsage is a snapshot of one tenant's position.
type TenantUsage struct {
	Tenant        string        `json:"tenant"`
	Limit         money.Pico    `json:"limit_pico"`
	Spent         money.Pico    `json:"spent_pico"`
	Reserved      money.Pico    `json:"reserved_pico"`
	Remaining     money.Pico    `json:"remaining_pico"`
	SoftThreshold money.Pico    `json:"soft_threshold_pico,omitempty"`
	Holds         int           `json:"holds"`
	Period        Period        `json:"period"`
	PeriodStart   time.Time     `json:"period_start"`
	PeriodEnd     time.Time     `json:"period_end"`
	ResetIn       time.Duration `json:"reset_in"`
	Unlimited     bool          `json:"unlimited,omitempty"`
	// CarriedHolds is the number of outstanding holds admitted by an earlier
	// window and therefore not charged against this one.
	CarriedHolds int `json:"carried_holds,omitempty"`
}

// Snapshot is the whole enforcer's state for the metrics endpoint.
type Snapshot struct {
	Stats    Stats                  `json:"stats"`
	Tenants  map[string]TenantUsage `json:"tenants"`
	Holds    int                    `json:"outstanding_holds"`
	Now      time.Time              `json:"now"`
	Defaults Limit                  `json:"default_limit"`
}

// DefaultHoldTTL is the default lifetime of an unresolved reservation.
const DefaultHoldTTL = 2 * time.Minute

// Options configures a Budget.
type Options struct {
	// Now is the clock, defaulting to time.Now. Injected so the window
	// boundary and the expiry sweep are tested by moving time, not by sleeping.
	Now func() time.Time

	// DefaultLimit applies to tenants with no explicit limit.
	DefaultLimit Limit

	// HoldTTL is how long an unresolved reservation survives before the sweep
	// reclaims it.
	//
	// IT MUST EXCEED THE LONGEST POSSIBLE REQUEST DEADLINE. A hold swept while
	// its request is still generating tokens is released while the money is
	// still being spent, which frees headroom for a request that should not have
	// been admitted — the sweep becomes the one internal source of
	// over-admission. The default is deliberately generous relative to any sane
	// upstream timeout, because the cost of a too-long TTL is a briefly
	// over-cautious budget, while the cost of a too-short one is a broken bound.
	HoldTTL time.Duration
}

// Budget is a concurrency-safe per-tenant spend enforcer.
type Budget struct {
	now func() time.Time
	ttl time.Duration

	mu      sync.Mutex
	nextID  uint64
	holds   map[uint64]*hold
	qhead   *hold
	qtail   *hold
	tenants map[string]*tenantState
	limits  map[string]Limit
	def     Limit
	stats   Stats
}

// New builds a Budget.
func New(opt Options) *Budget {
	now := opt.Now
	if now == nil {
		now = time.Now
	}
	ttl := opt.HoldTTL
	if ttl <= 0 {
		ttl = DefaultHoldTTL
	}
	return &Budget{
		now:     now,
		ttl:     ttl,
		holds:   make(map[uint64]*hold),
		tenants: make(map[string]*tenantState),
		limits:  make(map[string]Limit),
		def:     opt.DefaultLimit,
	}
}

// SetLimit installs a tenant's limit, taking effect on the next Reserve.
//
// Changing a limit does not touch settled spend or outstanding holds: lowering a
// limit below what a tenant has already spent leaves them at zero remaining
// until the window rolls, which is the intended meaning of "your budget is now
// smaller" and is why the check compares against live counters rather than
// caching a remaining figure.
func (b *Budget) SetLimit(tenant string, l Limit) error {
	if tenant == "" {
		return errors.New("budget: empty tenant")
	}
	if err := l.Validate(); err != nil {
		return err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.limits[tenant] = l
	if st, ok := b.tenants[tenant]; ok {
		st.limit = l
	}
	return nil
}

// SetDefaultLimit installs the limit for tenants with no explicit entry.
func (b *Budget) SetDefaultLimit(l Limit) error {
	if err := l.Validate(); err != nil {
		return err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.def = l
	return nil
}

// LimitFor returns the limit in force for a tenant.
func (b *Budget) LimitFor(tenant string) Limit {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.limitLocked(tenant)
}

func (b *Budget) limitLocked(tenant string) Limit {
	if l, ok := b.limits[tenant]; ok {
		return l
	}
	// Fail OPEN for an unconfigured tenant, which is worth defending because it
	// looks like the wrong default. A budget is a cost control, not a security
	// control: authentication already decided this tenant may call the gateway.
	// Failing closed here means one missing config entry turns into a total
	// outage for that tenant, trading a bounded, visible, refundable overspend
	// for an unbounded availability incident. The overspend is not invisible —
	// Decision.Unlimited and TenantUsage.Unlimited both say so, and the ledger
	// records every picodollar regardless.
	return b.def
}

// Reserve holds an estimated cost against a tenant's remaining budget.
//
// On Allow or AllowDegraded the returned Reservation is valid and the caller
// MUST eventually Commit or Release it; an unresolved hold is a slow leak that
// reduces the tenant's effective budget until the sweep reclaims it. On Reject
// the Reservation is the zero value and nothing was held.
func (b *Budget) Reserve(tenant string, estimate money.Pico) (Reservation, Decision) {
	now := b.now()

	b.mu.Lock()
	defer b.mu.Unlock()

	// Sweeping here, on the request path, rather than only from a background
	// ticker: a crashed handler's hold must be reclaimed even in a process with
	// no ticker running, and the sweep is O(expired) so the common case where
	// nothing has expired costs one pointer comparison.
	b.sweepLocked(now)

	if estimate < 0 {
		b.stats.InvalidEstimates++
		estimate = 0
	}

	st := b.stateLocked(tenant, now)
	w := st.cur
	d := Decision{
		Tenant:        tenant,
		Estimate:      estimate,
		Limit:         st.limit.Amount,
		Spent:         w.spent,
		Reserved:      w.reserved,
		SoftThreshold: st.limit.SoftThreshold(),
		Period:        st.limit.Period,
		PeriodStart:   w.start,
		PeriodEnd:     w.end,
		ResetIn:       w.end.Sub(now),
	}
	if st.limit.Period == PeriodUnset {
		// No window to reset: report nothing rather than a misleading instant.
		d.PeriodStart, d.PeriodEnd, d.ResetIn = time.Time{}, time.Time{}, 0
	}

	b.stats.Reserved++

	if st.limit.Unlimited() {
		// A hold is still taken so the caller's Commit/Release path is uniform
		// and so in-flight spend is observable for an unlimited tenant too.
		res := b.takeLocked(tenant, estimate, w, now)
		d.Outcome = Allow
		d.Unlimited = true
		d.Reason = ReasonNoLimit
		d.Reserved = w.reserved
		b.stats.Allowed++
		return res, d
	}

	// An estimate larger than the entire limit can never be admitted, in this
	// window or any future one. Saying so explicitly turns an infinite retry
	// loop into an actionable error.
	if estimate > st.limit.Amount {
		d.Outcome = Reject
		d.Reason = ReasonEstimateTooBig
		d.Remaining = st.limit.Amount - w.spent - w.reserved
		b.stats.Rejected++
		return Reservation{}, d
	}

	committed, err := money.Add(w.spent, w.reserved)
	if err == nil {
		_, err = money.Add(committed, estimate)
	}
	if err != nil {
		// Overflow can only happen with an absurd configuration, but silently
		// wrapping would produce a negative total and admit everything.
		d.Outcome = Reject
		d.Reason = ReasonOverflow
		d.Remaining = 0
		b.stats.Rejected++
		return Reservation{}, d
	}

	projected := committed + estimate
	if projected > st.limit.Amount {
		d.Outcome = Reject
		d.Reason = ReasonHardLimit
		d.Remaining = st.limit.Amount - committed
		b.stats.Rejected++
		return Reservation{}, d
	}

	res := b.takeLocked(tenant, estimate, w, now)
	d.Reserved = w.reserved
	d.Remaining = st.limit.Amount - projected

	// The soft threshold is evaluated on the PROJECTED position including this
	// request's own hold. Evaluating it on the position before the hold would
	// let the request that actually crosses the threshold sail through
	// undegraded, so the hint would always arrive one request late.
	if soft := st.limit.SoftThreshold(); soft > 0 && projected >= soft {
		d.Outcome = AllowDegraded
		d.Reason = ReasonSoftThreshold
		b.stats.Degraded++
	} else {
		d.Outcome = Allow
		d.Reason = ReasonWithinLimit
		b.stats.Allowed++
	}
	return res, d
}

// takeLocked creates the hold and links it into the expiry queue.
func (b *Budget) takeLocked(tenant string, amount money.Pico, w *window, now time.Time) Reservation {
	b.nextID++
	h := &hold{
		id:          b.nextID,
		tenant:      tenant,
		amount:      amount,
		windowStart: w.start.UnixNano(),
		expiresAt:   now.Add(b.ttl),
	}
	w.reserved += amount
	w.holds++
	b.holds[h.id] = h
	// Appending to the tail keeps the queue sorted by expiry, because the TTL is
	// a single constant: every hold created later expires later. If HoldTTL ever
	// became per-tenant this invariant would break and the queue would need to
	// become a heap — hence the constant, and hence this comment.
	if b.qtail == nil {
		b.qhead, b.qtail = h, h
	} else {
		h.prev = b.qtail
		b.qtail.next = h
		b.qtail = h
	}
	return Reservation{
		ID:          h.id,
		Tenant:      tenant,
		Amount:      amount,
		WindowStart: w.start,
		ExpiresAt:   h.expiresAt,
	}
}

// Commit resolves a reservation with the true cost, releasing the difference.
//
// The actual cost is attributed to the window that ADMITTED the request, not to
// whichever window happens to be current when the response arrives. A request
// approved at 10:59:59 that finishes at 11:00:01 spent budget the ten-o'clock
// window authorised; charging it to the eleven-o'clock window would let a burst
// admitted under one window's headroom eat the next window's, and would break
// the property that a window's spend equals the spend it admitted.
//
// Commit is the only path that can push settled spend past the limit, and only
// when actual exceeds the estimate. That is allowed on purpose: the money was
// already spent upstream and refusing to record it would make the ledger and the
// budget disagree, which is worse than a temporarily over-limit tenant who is
// now rejected until the window rolls.
func (b *Budget) Commit(res Reservation, actual money.Pico) (Decision, error) {
	now := b.now()
	if actual < 0 {
		return Decision{}, fmt.Errorf("budget: negative actual cost %d pico", int64(actual))
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	h, ok := b.holds[res.ID]
	if !ok || res.ID == 0 {
		// Either a double resolution or a hold the sweep already reclaimed. The
		// spend is real either way, so record it rather than lose it.
		if res.Valid() && res.Tenant != "" {
			b.stats.LateCommits++
			b.chargeLocked(res.Tenant, res.WindowStart, actual, now)
			d := b.usageDecisionLocked(res.Tenant, now)
			return d, fmt.Errorf("%w (id %d, cost %s recorded as unreserved spend)",
				ErrExpiredReservation, res.ID, money.FormatUSD(actual))
		}
		return Decision{}, ErrNoReservation
	}
	b.unlinkLocked(h)
	st := b.stateLocked(h.tenant, now)
	w := b.windowForLocked(st, h.windowStart, now)
	w.reserved -= h.amount
	w.holds--
	w.spent += actual
	b.stats.Committed++
	b.pruneLocked(st)
	return b.usageDecisionLocked(h.tenant, now), nil
}

// Release returns a hold in full, for the failure path where no cost was
// incurred. A request that failed after burning tokens must use Commit with the
// real cost instead; Release is not "the error path", it is "the free path".
func (b *Budget) Release(res Reservation) error {
	now := b.now()
	b.mu.Lock()
	defer b.mu.Unlock()

	h, ok := b.holds[res.ID]
	if !ok || res.ID == 0 {
		return ErrNoReservation
	}
	b.unlinkLocked(h)
	st := b.stateLocked(h.tenant, now)
	w := b.windowForLocked(st, h.windowStart, now)
	w.reserved -= h.amount
	w.holds--
	b.stats.Released++
	b.pruneLocked(st)
	return nil
}

// Charge records spend with no prior reservation.
//
// This exists for the failover case the ledger models: one client request can
// produce several billable upstream attempts, and an attempt that died after
// consuming tokens has a cost that was never individually reserved. Charging it
// keeps the budget's view of spend equal to the ledger's, which is the whole
// point of having both.
func (b *Budget) Charge(tenant string, amount money.Pico) error {
	if tenant == "" {
		return errors.New("budget: empty tenant")
	}
	if amount < 0 {
		return fmt.Errorf("budget: negative charge %d pico", int64(amount))
	}
	now := b.now()
	b.mu.Lock()
	defer b.mu.Unlock()
	b.stats.Unreserved++
	b.chargeLocked(tenant, time.Time{}, amount, now)
	return nil
}

// chargeLocked adds settled spend, preferring the named window when it is still
// live and falling back to the current one.
func (b *Budget) chargeLocked(tenant string, windowStart time.Time, amount money.Pico, now time.Time) {
	st := b.stateLocked(tenant, now)
	w := st.cur
	if !windowStart.IsZero() {
		// The admitting window may have been pruned once its last hold was
		// resolved. Falling back to the current window over-constrains the new
		// window slightly, which is the conservative direction: the alternative
		// is discarding a real charge.
		w = b.windowForLocked(st, windowStart.UnixNano(), now)
	}
	w.spent += amount
}

// Sweep reclaims holds whose TTL has passed and returns how many it released.
//
// Reserve already sweeps, so calling this is optional; a background ticker
// wanting to keep an idle tenant's leaked holds from lingering, and wanting the
// Expired counter to move promptly, should call it.
func (b *Budget) Sweep() int {
	now := b.now()
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.sweepLocked(now)
}

func (b *Budget) sweepLocked(now time.Time) int {
	n := 0
	for h := b.qhead; h != nil && !h.expiresAt.After(now); h = b.qhead {
		b.unlinkLocked(h)
		st := b.stateLocked(h.tenant, now)
		w := b.windowForLocked(st, h.windowStart, now)
		w.reserved -= h.amount
		w.holds--
		b.pruneLocked(st)
		b.stats.Expired++
		n++
	}
	return n
}

func (b *Budget) unlinkLocked(h *hold) {
	if h.prev != nil {
		h.prev.next = h.next
	} else {
		b.qhead = h.next
	}
	if h.next != nil {
		h.next.prev = h.prev
	} else {
		b.qtail = h.prev
	}
	h.prev, h.next = nil, nil
	delete(b.holds, h.id)
}

// stateLocked fetches a tenant's state, rolling the window if the period has
// advanced.
func (b *Budget) stateLocked(tenant string, now time.Time) *tenantState {
	st, ok := b.tenants[tenant]
	if !ok {
		st = &tenantState{limit: b.limitLocked(tenant)}
		st.cur = newWindow(st.limit.Period, now)
		b.tenants[tenant] = st
		return st
	}
	st.limit = b.limitLocked(tenant)
	start := st.limit.Period.WindowStart(now)
	if st.cur.start.Equal(start) {
		return st
	}
	// The window rolled. The old window is retained only while it still has
	// outstanding holds, so a rollover cannot lose an in-flight reservation, and
	// a rolled window with nothing in flight is dropped immediately rather than
	// accumulating one map entry per elapsed hour forever.
	old := st.cur
	st.cur = newWindow(st.limit.Period, now)
	b.stats.Rollovers++
	if old.holds > 0 {
		if st.past == nil {
			st.past = make(map[int64]*window, 1)
		}
		st.past[old.start.UnixNano()] = old
	}
	return st
}

// windowForLocked resolves the window a hold was admitted by.
func (b *Budget) windowForLocked(st *tenantState, startNano int64, now time.Time) *window {
	if st.cur.start.UnixNano() == startNano {
		return st.cur
	}
	if w, ok := st.past[startNano]; ok {
		return w
	}
	// The window is gone. Charge the current one: over-constraining the live
	// window is the conservative failure, and it is reachable only when a
	// reservation outlives both its window and the retention rule above.
	return st.cur
}

// pruneLocked drops retained past windows that have no outstanding holds.
func (b *Budget) pruneLocked(st *tenantState) {
	for k, w := range st.past {
		if w.holds <= 0 {
			delete(st.past, k)
		}
	}
	if len(st.past) == 0 {
		st.past = nil
	}
}

func newWindow(p Period, now time.Time) *window {
	start := p.WindowStart(now)
	return &window{start: start, end: p.WindowEnd(start)}
}

// Status reports a tenant's current position without reserving anything. Safe to
// call from a read-only endpoint; it does roll an elapsed window and sweep
// expired holds, because reporting stale counters would be worse than the write.
func (b *Budget) Status(tenant string) Decision {
	now := b.now()
	b.mu.Lock()
	defer b.mu.Unlock()
	b.sweepLocked(now)
	return b.usageDecisionLocked(tenant, now)
}

func (b *Budget) usageDecisionLocked(tenant string, now time.Time) Decision {
	st := b.stateLocked(tenant, now)
	w := st.cur
	d := Decision{
		Tenant:        tenant,
		Limit:         st.limit.Amount,
		Spent:         w.spent,
		Reserved:      w.reserved,
		SoftThreshold: st.limit.SoftThreshold(),
		Period:        st.limit.Period,
		PeriodStart:   w.start,
		PeriodEnd:     w.end,
		ResetIn:       w.end.Sub(now),
		Outcome:       Allow,
		Reason:        ReasonWithinLimit,
	}
	if st.limit.Unlimited() {
		d.Unlimited = true
		d.Reason = ReasonNoLimit
		d.PeriodStart, d.PeriodEnd, d.ResetIn = time.Time{}, time.Time{}, 0
		return d
	}
	d.Remaining = st.limit.Amount - w.spent - w.reserved
	if d.Remaining <= 0 {
		d.Outcome = Reject
		d.Reason = ReasonHardLimit
	} else if soft := d.SoftThreshold; soft > 0 && w.spent+w.reserved >= soft {
		d.Outcome = AllowDegraded
		d.Reason = ReasonSoftThreshold
	}
	return d
}

// Usage returns a tenant's snapshot.
func (b *Budget) Usage(tenant string) TenantUsage {
	now := b.now()
	b.mu.Lock()
	defer b.mu.Unlock()
	b.sweepLocked(now)
	return b.usageLocked(tenant, now)
}

func (b *Budget) usageLocked(tenant string, now time.Time) TenantUsage {
	st := b.stateLocked(tenant, now)
	w := st.cur
	u := TenantUsage{
		Tenant:        tenant,
		Limit:         st.limit.Amount,
		Spent:         w.spent,
		Reserved:      w.reserved,
		SoftThreshold: st.limit.SoftThreshold(),
		Holds:         w.holds,
		Period:        st.limit.Period,
		PeriodStart:   w.start,
		PeriodEnd:     w.end,
		ResetIn:       w.end.Sub(now),
		Unlimited:     st.limit.Unlimited(),
	}
	if !u.Unlimited {
		u.Remaining = st.limit.Amount - w.spent - w.reserved
	} else {
		u.PeriodStart, u.PeriodEnd, u.ResetIn = time.Time{}, time.Time{}, 0
	}
	for _, pw := range st.past {
		u.CarriedHolds += pw.holds
	}
	return u
}

// Stats returns the counters.
func (b *Budget) Stats() Stats {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.stats
}

// Snapshot copies the whole state, so the metrics encoder can format without
// holding the lock that the request path needs.
func (b *Budget) Snapshot() Snapshot {
	now := b.now()
	b.mu.Lock()
	defer b.mu.Unlock()
	s := Snapshot{
		Stats:    b.stats,
		Tenants:  make(map[string]TenantUsage, len(b.tenants)),
		Holds:    len(b.holds),
		Now:      now,
		Defaults: b.def,
	}
	// Deliberately does NOT roll windows or sweep: a snapshot is an observation,
	// and an observer that mutates state makes a scrape interval a functional
	// input to the system. The counters may therefore describe a window that has
	// just elapsed, which PeriodEnd makes visible.
	for name := range b.tenants {
		st := b.tenants[name]
		w := st.cur
		u := TenantUsage{
			Tenant:        name,
			Limit:         st.limit.Amount,
			Spent:         w.spent,
			Reserved:      w.reserved,
			SoftThreshold: st.limit.SoftThreshold(),
			Holds:         w.holds,
			Period:        st.limit.Period,
			PeriodStart:   w.start,
			PeriodEnd:     w.end,
			ResetIn:       w.end.Sub(now),
			Unlimited:     st.limit.Unlimited(),
		}
		if !u.Unlimited {
			u.Remaining = st.limit.Amount - w.spent - w.reserved
		} else {
			u.PeriodStart, u.PeriodEnd, u.ResetIn = time.Time{}, time.Time{}, 0
		}
		for _, pw := range st.past {
			u.CarriedHolds += pw.holds
		}
		s.Tenants[name] = u
	}
	return s
}

// OutstandingHolds returns the number of unresolved reservations. Primarily a
// test and debug affordance: a number that only grows is the leak the sweep
// exists to bound.
func (b *Budget) OutstandingHolds() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.holds)
}
