package budget

import (
	"errors"
	"fmt"
	"math"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/money"
)

// manualClock is a settable clock. Reads are atomic so the concurrency tests can
// share one instance without a data race of their own.
type manualClock struct {
	ns atomic.Int64
}

func newClock(t time.Time) *manualClock {
	c := &manualClock{}
	c.ns.Store(t.UnixNano())
	return c
}

func (c *manualClock) Now() time.Time { return time.Unix(0, c.ns.Load()).UTC() }

func (c *manualClock) Advance(d time.Duration) { c.ns.Add(int64(d)) }

func (c *manualClock) Set(t time.Time) { c.ns.Store(t.UnixNano()) }

// mid-window instant, deliberately not on a boundary.
var noon = time.Date(2026, 3, 15, 12, 30, 0, 0, time.UTC)

func TestPeriodParseAndString(t *testing.T) {
	tests := []struct {
		in      string
		want    Period
		wantErr bool
	}{
		{"hourly", PeriodHourly, false},
		{"hour", PeriodHourly, false},
		{"1h", PeriodHourly, false},
		{"daily", PeriodDaily, false},
		{"day", PeriodDaily, false},
		{"monthly", PeriodMonthly, false},
		{"month", PeriodMonthly, false},
		{"", PeriodUnset, true},
		{"weekly", PeriodUnset, true},
		{"HOURLY", PeriodUnset, true},
	}
	for _, tc := range tests {
		t.Run(tc.in, func(t *testing.T) {
			got, err := ParsePeriod(tc.in)
			if (err != nil) != tc.wantErr {
				t.Fatalf("ParsePeriod(%q) err = %v, wantErr %t", tc.in, err, tc.wantErr)
			}
			if got != tc.want {
				t.Fatalf("ParsePeriod(%q) = %v, want %v", tc.in, got, tc.want)
			}
		})
	}
	if PeriodUnset.Valid() {
		t.Error("PeriodUnset must be invalid: a config that forgot the period must be rejected")
	}
	for _, p := range []Period{PeriodHourly, PeriodDaily, PeriodMonthly} {
		if !p.Valid() {
			t.Errorf("%v reported invalid", p)
		}
		if p.String() == "unset" {
			t.Errorf("%d has no name", int(p))
		}
	}
	if Period(99).String() != "unset" {
		t.Error("unknown period should render as unset")
	}
}

func TestWindowBoundaryRule(t *testing.T) {
	// The documented rule: UTC-aligned, half-open [start, end). An instant
	// exactly on a boundary belongs to the window that is STARTING.
	tests := []struct {
		name      string
		period    Period
		at        time.Time
		wantStart time.Time
		wantEnd   time.Time
	}{
		{
			"hourly mid",
			PeriodHourly,
			time.Date(2026, 3, 15, 12, 30, 45, 123, time.UTC),
			time.Date(2026, 3, 15, 12, 0, 0, 0, time.UTC),
			time.Date(2026, 3, 15, 13, 0, 0, 0, time.UTC),
		},
		{
			"hourly exactly on the boundary belongs to the new window",
			PeriodHourly,
			time.Date(2026, 3, 15, 13, 0, 0, 0, time.UTC),
			time.Date(2026, 3, 15, 13, 0, 0, 0, time.UTC),
			time.Date(2026, 3, 15, 14, 0, 0, 0, time.UTC),
		},
		{
			"hourly one nanosecond before the boundary is the old window",
			PeriodHourly,
			time.Date(2026, 3, 15, 12, 59, 59, 999999999, time.UTC),
			time.Date(2026, 3, 15, 12, 0, 0, 0, time.UTC),
			time.Date(2026, 3, 15, 13, 0, 0, 0, time.UTC),
		},
		{
			"daily",
			PeriodDaily,
			time.Date(2026, 3, 15, 23, 59, 59, 0, time.UTC),
			time.Date(2026, 3, 15, 0, 0, 0, 0, time.UTC),
			time.Date(2026, 3, 16, 0, 0, 0, 0, time.UTC),
		},
		{
			// A non-UTC input must be normalised, not taken at face value.
			"non-UTC input is normalised to UTC",
			PeriodDaily,
			time.Date(2026, 3, 15, 23, 30, 0, 0, time.FixedZone("UTC+5", 5*3600)),
			time.Date(2026, 3, 15, 0, 0, 0, 0, time.UTC),
			time.Date(2026, 3, 16, 0, 0, 0, 0, time.UTC),
		},
		{
			// Calendar months, so February is 28 days and not 30.
			"monthly February in a non-leap year",
			PeriodMonthly,
			time.Date(2026, 2, 14, 5, 0, 0, 0, time.UTC),
			time.Date(2026, 2, 1, 0, 0, 0, 0, time.UTC),
			time.Date(2026, 3, 1, 0, 0, 0, 0, time.UTC),
		},
		{
			"monthly February in a leap year",
			PeriodMonthly,
			time.Date(2028, 2, 29, 23, 0, 0, 0, time.UTC),
			time.Date(2028, 2, 1, 0, 0, 0, 0, time.UTC),
			time.Date(2028, 3, 1, 0, 0, 0, 0, time.UTC),
		},
		{
			"monthly December rolls the year",
			PeriodMonthly,
			time.Date(2026, 12, 31, 23, 59, 0, 0, time.UTC),
			time.Date(2026, 12, 1, 0, 0, 0, 0, time.UTC),
			time.Date(2027, 1, 1, 0, 0, 0, 0, time.UTC),
		},
		{
			// A daily window across a DST transition in some local zone is still
			// exactly 24 hours, which is the entire point of aligning to UTC.
			"daily across a US DST spring-forward is still 24h",
			PeriodDaily,
			time.Date(2026, 3, 8, 10, 0, 0, 0, time.UTC),
			time.Date(2026, 3, 8, 0, 0, 0, 0, time.UTC),
			time.Date(2026, 3, 9, 0, 0, 0, 0, time.UTC),
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			start := tc.period.WindowStart(tc.at)
			if !start.Equal(tc.wantStart) {
				t.Errorf("WindowStart = %v, want %v", start, tc.wantStart)
			}
			end := tc.period.WindowEnd(start)
			if !end.Equal(tc.wantEnd) {
				t.Errorf("WindowEnd = %v, want %v", end, tc.wantEnd)
			}
			// Half-open: the end instant must belong to the NEXT window.
			if next := tc.period.WindowStart(end); !next.Equal(end) {
				t.Errorf("the end instant %v maps back to window %v; windows must be half-open", end, next)
			}
		})
	}
	// The DST claim, checked rather than asserted in prose.
	d := PeriodDaily
	s := d.WindowStart(time.Date(2026, 3, 8, 10, 0, 0, 0, time.UTC))
	if got := d.WindowEnd(s).Sub(s); got != 24*time.Hour {
		t.Errorf("daily window across DST = %v, want 24h", got)
	}
}

func TestLimitValidateAndSoftThreshold(t *testing.T) {
	tests := []struct {
		name     string
		limit    Limit
		wantErr  string
		wantSoft money.Pico
	}{
		{
			name:     "valid daily with soft",
			limit:    Limit{Amount: money.USD(10), Period: PeriodDaily, SoftBasisPoints: 8000},
			wantSoft: money.USD(8),
		},
		{
			name:  "unlimited needs no period",
			limit: Limit{},
		},
		{
			name:    "limit without a period",
			limit:   Limit{Amount: money.USD(10)},
			wantErr: "needs a period",
		},
		{
			name:    "negative limit",
			limit:   Limit{Amount: -1, Period: PeriodDaily},
			wantErr: "negative limit",
		},
		{
			name:    "soft above 10000 bp",
			limit:   Limit{Amount: money.USD(1), Period: PeriodDaily, SoftBasisPoints: 10001},
			wantErr: "outside 0..10000",
		},
		{
			name:    "negative soft",
			limit:   Limit{Amount: money.USD(1), Period: PeriodDaily, SoftBasisPoints: -1},
			wantErr: "outside 0..10000",
		},
		{
			name:     "soft at 10000 bp equals the limit",
			limit:    Limit{Amount: money.USD(3), Period: PeriodDaily, SoftBasisPoints: 10000},
			wantSoft: money.USD(3),
		},
		{
			name:     "soft disabled",
			limit:    Limit{Amount: money.USD(3), Period: PeriodDaily},
			wantSoft: 0,
		},
		{
			// Truncation must be sub-picodollar, not a visible shift.
			name:     "awkward basis points truncate by less than a pico",
			limit:    Limit{Amount: 10000000000001, Period: PeriodDaily, SoftBasisPoints: 3333},
			wantSoft: money.Pico(10000000000001 * 3333 / 10000),
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.limit.Validate()
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("Validate = %v, want nil", err)
				}
			} else {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("Validate = %v, want it to contain %q", err, tc.wantErr)
				}
				return
			}
			if got := tc.limit.SoftThreshold(); got != tc.wantSoft {
				t.Errorf("SoftThreshold = %d, want %d", got, tc.wantSoft)
			}
		})
	}
	if !(Limit{}).Unlimited() {
		t.Error("the zero Limit must be unlimited")
	}
	if (Limit{Amount: 1, Period: PeriodDaily}).Unlimited() {
		t.Error("a positive limit is not unlimited")
	}
}

func newBudget(t *testing.T, clk *manualClock, l Limit) *Budget {
	t.Helper()
	b := New(Options{Now: clk.Now, DefaultLimit: l})
	return b
}

func TestReserveCommitTrueUpReturnsTheDifference(t *testing.T) {
	clk := newClock(noon)
	b := newBudget(t, clk, Limit{Amount: money.USD(1), Period: PeriodHourly})

	res, d := b.Reserve("acme", money.Cents(50))
	if d.Outcome != Allow {
		t.Fatalf("outcome = %v, want Allow: %s", d.Outcome, d.Message())
	}
	if !res.Valid() {
		t.Fatal("an admitted Reserve must return a valid reservation")
	}
	if d.Reserved != money.Cents(50) {
		t.Errorf("Reserved = %s, want $0.50", money.FormatUSD(d.Reserved))
	}
	// Remaining excludes this request's own hold, which is what a
	// "you have $X left" header needs.
	if d.Remaining != money.Cents(50) {
		t.Errorf("Remaining = %s, want $0.50", money.FormatUSD(d.Remaining))
	}

	// The true-up: the request actually cost a tenth of the estimate.
	if _, err := b.Commit(res, money.Cents(5)); err != nil {
		t.Fatalf("Commit: %v", err)
	}
	u := b.Usage("acme")
	if u.Spent != money.Cents(5) {
		t.Errorf("Spent = %s, want $0.05", money.FormatUSD(u.Spent))
	}
	if u.Reserved != 0 {
		t.Errorf("Reserved = %s after Commit, want 0: the hold must be fully released",
			money.FormatUSD(u.Reserved))
	}
	if u.Remaining != money.Cents(95) {
		t.Errorf("Remaining = %s, want $0.95", money.FormatUSD(u.Remaining))
	}
	if u.Holds != 0 {
		t.Errorf("Holds = %d, want 0", u.Holds)
	}
	if b.OutstandingHolds() != 0 {
		t.Errorf("OutstandingHolds = %d, want 0", b.OutstandingHolds())
	}
}

func TestReleaseReturnsTheWholeHold(t *testing.T) {
	clk := newClock(noon)
	b := newBudget(t, clk, Limit{Amount: money.USD(1), Period: PeriodHourly})

	res, _ := b.Reserve("acme", money.Cents(90))
	if err := b.Release(res); err != nil {
		t.Fatalf("Release: %v", err)
	}
	u := b.Usage("acme")
	if u.Spent != 0 || u.Reserved != 0 {
		t.Fatalf("after Release: spent=%s reserved=%s, want 0/0",
			money.FormatUSD(u.Spent), money.FormatUSD(u.Reserved))
	}
	// The headroom must be genuinely usable again.
	if _, d := b.Reserve("acme", money.Cents(95)); d.Outcome != Allow {
		t.Fatalf("a released hold did not free headroom: %s", d.Message())
	}
}

func TestDoubleResolutionIsDetected(t *testing.T) {
	// A handler with both a deferred Release and an explicit Commit is the
	// classic bug; it must not double-credit the tenant.
	clk := newClock(noon)
	b := newBudget(t, clk, Limit{Amount: money.USD(1), Period: PeriodHourly})

	res, _ := b.Reserve("acme", money.Cents(50))
	if _, err := b.Commit(res, money.Cents(10)); err != nil {
		t.Fatal(err)
	}
	if err := b.Release(res); !errors.Is(err, ErrNoReservation) {
		t.Fatalf("Release after Commit = %v, want ErrNoReservation", err)
	}
	if u := b.Usage("acme"); u.Reserved != 0 {
		t.Fatalf("Reserved = %s after a double resolution; the counter was corrupted",
			money.FormatUSD(u.Reserved))
	}

	// A second Commit records the spend as unreserved rather than losing it: the
	// money was really spent upstream.
	before := b.Usage("acme").Spent
	if _, err := b.Commit(res, money.Cents(7)); !errors.Is(err, ErrExpiredReservation) {
		t.Fatalf("second Commit = %v, want ErrExpiredReservation", err)
	}
	if got := b.Usage("acme").Spent; got != before+money.Cents(7) {
		t.Errorf("Spent = %s, want %s: a late Commit must still record real spend",
			money.FormatUSD(got), money.FormatUSD(before+money.Cents(7)))
	}
}

func TestResolvingTheZeroReservation(t *testing.T) {
	clk := newClock(noon)
	b := newBudget(t, clk, Limit{Amount: money.USD(1), Period: PeriodHourly})
	if err := b.Release(Reservation{}); !errors.Is(err, ErrNoReservation) {
		t.Errorf("Release(zero) = %v, want ErrNoReservation", err)
	}
	if _, err := b.Commit(Reservation{}, money.Cents(1)); !errors.Is(err, ErrNoReservation) {
		t.Errorf("Commit(zero) = %v, want ErrNoReservation", err)
	}
	if (Reservation{}).Valid() {
		t.Error("the zero Reservation must not be Valid")
	}
	if _, err := b.Commit(Reservation{ID: 1, Tenant: "acme"}, -1); err == nil {
		t.Error("Commit with a negative cost was accepted")
	}
}

func TestRejectAtHardLimitCarriesTheJustifyingNumbers(t *testing.T) {
	clk := newClock(noon)
	b := newBudget(t, clk, Limit{Amount: money.USD(1), Period: PeriodHourly})

	res, _ := b.Reserve("acme", money.Cents(60))
	if _, err := b.Commit(res, money.Cents(60)); err != nil {
		t.Fatal(err)
	}
	held, d := b.Reserve("acme", money.Cents(30))
	if d.Outcome != Allow {
		t.Fatalf("second reserve: %s", d.Message())
	}
	_ = held

	// $0.60 spent + $0.30 held leaves $0.10; asking for $0.20 must be rejected.
	_, d = b.Reserve("acme", money.Cents(20))
	if d.Outcome != Reject {
		t.Fatalf("outcome = %v, want Reject", d.Outcome)
	}
	if d.Reason != ReasonHardLimit {
		t.Errorf("Reason = %q, want %q", d.Reason, ReasonHardLimit)
	}
	if d.Limit != money.USD(1) {
		t.Errorf("Limit = %s", money.FormatUSD(d.Limit))
	}
	if d.Spent != money.Cents(60) {
		t.Errorf("Spent = %s, want $0.60", money.FormatUSD(d.Spent))
	}
	if d.Reserved != money.Cents(30) {
		t.Errorf("Reserved = %s, want $0.30 of in-flight holds", money.FormatUSD(d.Reserved))
	}
	if d.Remaining != money.Cents(10) {
		t.Errorf("Remaining = %s, want $0.10 (the headroom that proved insufficient)",
			money.FormatUSD(d.Remaining))
	}
	if d.Estimate != money.Cents(20) {
		t.Errorf("Estimate = %s, want $0.20", money.FormatUSD(d.Estimate))
	}
	if !d.PeriodEnd.Equal(time.Date(2026, 3, 15, 13, 0, 0, 0, time.UTC)) {
		t.Errorf("PeriodEnd = %v, want 13:00Z", d.PeriodEnd)
	}
	if d.ResetIn != 30*time.Minute {
		t.Errorf("ResetIn = %v, want 30m", d.ResetIn)
	}
	// A rejection must not consume budget.
	if u := b.Usage("acme"); u.Reserved != money.Cents(30) {
		t.Errorf("a rejected Reserve took a hold: Reserved = %s", money.FormatUSD(u.Reserved))
	}

	msg := d.Message()
	for _, want := range []string{"$1", "$0.6", "$0.3", "$0.1", "resets at", "30m0s"} {
		if !strings.Contains(msg, want) {
			t.Errorf("message lacks %q, so the client cannot see why or when: %s", want, msg)
		}
	}
}

func TestRejectStatusIs429WithARealRetryAfter(t *testing.T) {
	// 429 and not 402: this limit heals by itself at a known instant, and an
	// SDK treats a non-429 4xx as terminal. Retry-After must be the real
	// distance to the boundary, rounded UP, or a client retries inside the
	// still-exhausted window.
	clk := newClock(time.Date(2026, 3, 15, 12, 0, 0, 0, time.UTC))
	b := newBudget(t, clk, Limit{Amount: money.Cents(10), Period: PeriodHourly})
	res, _ := b.Reserve("acme", money.Cents(10))
	if _, err := b.Commit(res, money.Cents(10)); err != nil {
		t.Fatal(err)
	}

	_, d := b.Reserve("acme", money.Cents(1))
	if d.Outcome != Reject {
		t.Fatalf("want Reject, got %v", d.Outcome)
	}
	if got := d.HTTPStatus(); got != http.StatusTooManyRequests {
		t.Errorf("HTTPStatus = %d, want 429 (402 would make clients give up on a "+
			"limit that resets by itself)", got)
	}
	if got := d.RetryAfter(); got != time.Hour {
		t.Errorf("RetryAfter = %v, want 1h", got)
	}

	// A fractional second must round UP.
	clk.Advance(time.Hour - 1500*time.Millisecond)
	_, d = b.Reserve("acme", money.Cents(1))
	if d.Outcome != Reject {
		t.Fatalf("want Reject, got %v", d.Outcome)
	}
	if got := d.RetryAfter(); got != 2*time.Second {
		t.Errorf("RetryAfter = %v, want 2s (rounding down would retry inside the "+
			"exhausted window)", got)
	}

	// A monthly budget's Retry-After is honestly enormous, not a polite lie.
	clk.Set(time.Date(2026, 3, 2, 0, 0, 0, 0, time.UTC))
	bm := newBudget(t, clk, Limit{Amount: money.Cents(1), Period: PeriodMonthly})
	r2, _ := bm.Reserve("acme", money.Cents(1))
	if _, err := bm.Commit(r2, money.Cents(1)); err != nil {
		t.Fatal(err)
	}
	_, d = bm.Reserve("acme", money.Cents(1))
	if got := d.RetryAfter(); got < 29*24*time.Hour {
		t.Errorf("monthly RetryAfter = %v, want ~29 days; a small polite number "+
			"would make clients hammer a budget that will not reset for weeks", got)
	}

	// Non-rejections carry no Retry-After and a 200.
	allowed := Decision{Outcome: Allow, ResetIn: time.Hour}
	if allowed.RetryAfter() != 0 || allowed.HTTPStatus() != http.StatusOK {
		t.Error("an Allow decision must not carry Retry-After or a 4xx")
	}
}

func TestEstimateLargerThanTheWholeLimitIsUnretryable(t *testing.T) {
	// Rejecting with ReasonHardLimit would invite an infinite retry loop: no
	// amount of waiting makes this request fit.
	clk := newClock(noon)
	b := newBudget(t, clk, Limit{Amount: money.Cents(10), Period: PeriodHourly})
	_, d := b.Reserve("acme", money.USD(5))
	if d.Outcome != Reject {
		t.Fatalf("want Reject, got %v", d.Outcome)
	}
	if d.Reason != ReasonEstimateTooBig {
		t.Errorf("Reason = %q, want %q", d.Reason, ReasonEstimateTooBig)
	}
	if !strings.Contains(d.Message(), "no retry will help") {
		t.Errorf("message must say retrying is pointless: %s", d.Message())
	}
	if !strings.Contains(d.Message(), "max_tokens") {
		t.Errorf("message should tell the client what to change: %s", d.Message())
	}
	// An estimate exactly equal to the limit IS admissible.
	if _, d := b.Reserve("acme", money.Cents(10)); d.Outcome != Allow && d.Outcome != AllowDegraded {
		t.Errorf("an estimate equal to the limit was rejected: %s", d.Message())
	}
}

func TestSoftThresholdIsARoutingHintNotARejection(t *testing.T) {
	clk := newClock(noon)
	b := newBudget(t, clk, Limit{
		Amount: money.USD(1), Period: PeriodHourly, SoftBasisPoints: 8000,
	})

	// Below the threshold: plain Allow.
	res, d := b.Reserve("acme", money.Cents(70))
	if d.Outcome != Allow {
		t.Fatalf("outcome = %v, want Allow at 70%%", d.Outcome)
	}
	if _, err := b.Commit(res, money.Cents(70)); err != nil {
		t.Fatal(err)
	}

	// Crossing it: degraded, but still admitted with a valid hold.
	res, d = b.Reserve("acme", money.Cents(15))
	if d.Outcome != AllowDegraded {
		t.Fatalf("outcome = %v, want AllowDegraded at 85%%: %s", d.Outcome, d.Message())
	}
	if !d.Outcome.Admitted() {
		t.Error("AllowDegraded must be admitted; it is a hint, not a rejection")
	}
	if !res.Valid() {
		t.Fatal("AllowDegraded must still return a usable hold")
	}
	if d.SoftThreshold != money.Cents(80) {
		t.Errorf("SoftThreshold = %s, want $0.80", money.FormatUSD(d.SoftThreshold))
	}
	if d.Reason != ReasonSoftThreshold {
		t.Errorf("Reason = %q", d.Reason)
	}
	if !strings.Contains(d.Message(), "cheaper model") {
		t.Errorf("the hint must name the suggested action: %s", d.Message())
	}
	if _, err := b.Commit(res, money.Cents(15)); err != nil {
		t.Fatal(err)
	}

	// The request that CROSSES the threshold must itself be degraded, not the one
	// after it: the hint arriving one request late is the bug this checks.
	clk.Advance(time.Hour)
	b2 := newBudget(t, clk, Limit{Amount: money.USD(1), Period: PeriodHourly, SoftBasisPoints: 8000})
	_, d = b2.Reserve("acme", money.Cents(80))
	if d.Outcome != AllowDegraded {
		t.Fatalf("the request whose own hold reaches the threshold must be degraded, got %v", d.Outcome)
	}
	// One picodollar short of the threshold must NOT degrade: the boundary is >=.
	b3 := newBudget(t, clk, Limit{Amount: money.USD(1), Period: PeriodHourly, SoftBasisPoints: 8000})
	if _, d := b3.Reserve("acme", money.Cents(80)-1); d.Outcome != Allow {
		t.Errorf("outcome = %v one pico below the threshold, want Allow", d.Outcome)
	}
}

func TestSoftThresholdDisabledMeansNoDegradation(t *testing.T) {
	clk := newClock(noon)
	b := newBudget(t, clk, Limit{Amount: money.USD(1), Period: PeriodHourly})
	res, d := b.Reserve("acme", money.Cents(99))
	if d.Outcome != Allow {
		t.Fatalf("outcome = %v, want Allow with no soft threshold configured", d.Outcome)
	}
	if d.SoftThreshold != 0 {
		t.Errorf("SoftThreshold = %s, want 0", money.FormatUSD(d.SoftThreshold))
	}
	if _, err := b.Commit(res, money.Cents(99)); err != nil {
		t.Fatal(err)
	}
	// Fine right up to the limit, then a hard Reject with nothing in between.
	if _, d := b.Reserve("acme", money.Cents(1)); d.Outcome != Allow {
		t.Errorf("outcome = %v at exactly the limit, want Allow", d.Outcome)
	}
}

// TestOverAdmissionBoundUnderConcurrency is the headline correctness claim:
// spent + reserved <= limit at all times, with no slack, however many callers
// race. Run with -race.
func TestOverAdmissionBoundUnderConcurrency(t *testing.T) {
	const (
		callers  = 400
		estimate = money.Pico(money.PicoPerCent) // $0.01 each
	)
	// A budget that fits exactly three of the 400 concurrent requests. If the
	// check-then-act were not atomic, dozens would be admitted.
	limit := money.Pico(3 * money.PicoPerCent)

	clk := newClock(noon)
	b := New(Options{
		Now:          clk.Now,
		DefaultLimit: Limit{Amount: limit, Period: PeriodHourly},
		HoldTTL:      time.Hour, // must not expire during the test
	})

	var admitted atomic.Int64
	var rejected atomic.Int64
	var mu sync.Mutex
	var reservations []Reservation

	var start sync.WaitGroup
	var done sync.WaitGroup
	start.Add(1)
	for i := 0; i < callers; i++ {
		done.Add(1)
		go func() {
			defer done.Done()
			start.Wait() // release everyone at once to maximise the race
			res, d := b.Reserve("acme", estimate)
			if d.Outcome.Admitted() {
				admitted.Add(1)
				mu.Lock()
				reservations = append(reservations, res)
				mu.Unlock()
				if !res.Valid() {
					t.Errorf("admitted without a valid reservation")
				}
			} else {
				rejected.Add(1)
				if res.Valid() {
					t.Errorf("rejected but a hold was taken: %+v", res)
				}
			}
		}()
	}
	start.Done()
	done.Wait()

	// THE BOUND.
	u := b.Usage("acme")
	if u.Spent+u.Reserved > limit {
		t.Fatalf("OVER-ADMISSION: spent %s + reserved %s = %s exceeds limit %s",
			money.FormatUSD(u.Spent), money.FormatUSD(u.Reserved),
			money.FormatUSD(u.Spent+u.Reserved), money.FormatUSD(limit))
	}
	if got := admitted.Load(); got != 3 {
		t.Fatalf("admitted %d of %d callers, want exactly 3 (the budget fits 3)", got, callers)
	}
	if got := rejected.Load(); got != callers-3 {
		t.Fatalf("rejected %d, want %d", got, callers-3)
	}
	if u.Reserved != limit {
		t.Errorf("Reserved = %s, want the full limit %s held", money.FormatUSD(u.Reserved), money.FormatUSD(limit))
	}

	// Now commit them all at the true cost and check the bound still holds and
	// the headroom comes back.
	for _, res := range reservations {
		if _, err := b.Commit(res, estimate/2); err != nil {
			t.Fatalf("Commit: %v", err)
		}
	}
	u = b.Usage("acme")
	if u.Reserved != 0 {
		t.Errorf("Reserved = %s after committing everything", money.FormatUSD(u.Reserved))
	}
	if u.Spent != 3*estimate/2 {
		t.Errorf("Spent = %s, want %s", money.FormatUSD(u.Spent), money.FormatUSD(3*estimate/2))
	}
	if u.Spent+u.Reserved > limit {
		t.Fatalf("bound violated after commit: %s > %s",
			money.FormatUSD(u.Spent+u.Reserved), money.FormatUSD(limit))
	}
}

// TestOverAdmissionBoundUnderMixedConcurrentOperations races Reserve against
// Commit, Release, Sweep, Status and Snapshot, checking the bound continuously
// from an observer goroutine rather than only at the end. A bound that only
// holds at quiescence is not the bound that was claimed.
func TestOverAdmissionBoundUnderMixedConcurrentOperations(t *testing.T) {
	const workers = 24
	const iterations = 200
	limit := money.Cents(50)

	clk := newClock(noon)
	b := New(Options{
		Now:          clk.Now,
		DefaultLimit: Limit{Amount: limit, Period: PeriodHourly, SoftBasisPoints: 7000},
		HoldTTL:      time.Hour,
	})

	stop := make(chan struct{})
	var violations atomic.Int64
	var observerDone sync.WaitGroup
	observerDone.Add(1)
	go func() {
		defer observerDone.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			u := b.Usage("acme")
			// The estimate never under-shoots in this test (actual == estimate/2),
			// so the bound must hold at every observation.
			if u.Spent+u.Reserved > limit {
				violations.Add(1)
			}
			_ = b.Snapshot()
			_ = b.Status("acme")
			_ = b.Stats()
		}
	}()

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				res, d := b.Reserve("acme", money.Cents(1))
				if !d.Outcome.Admitted() {
					continue
				}
				switch (w + i) % 3 {
				case 0:
					if _, err := b.Commit(res, money.Cents(1)/2); err != nil {
						t.Errorf("Commit: %v", err)
					}
				case 1:
					if err := b.Release(res); err != nil {
						t.Errorf("Release: %v", err)
					}
				default:
					// Leak it deliberately; the sweep must clean it up and the
					// bound must hold in the meantime.
				}
			}
		}(w)
	}
	wg.Wait()
	close(stop)
	observerDone.Wait()

	if n := violations.Load(); n != 0 {
		t.Fatalf("%d observations violated spent+reserved <= limit", n)
	}
	u := b.Usage("acme")
	if u.Spent+u.Reserved > limit {
		t.Fatalf("final: spent %s + reserved %s > limit %s",
			money.FormatUSD(u.Spent), money.FormatUSD(u.Reserved), money.FormatUSD(limit))
	}
	// The deliberately leaked holds must be reclaimable.
	clk.Advance(2 * time.Hour)
	b.Sweep()
	if got := b.OutstandingHolds(); got != 0 {
		t.Errorf("OutstandingHolds = %d after the TTL elapsed, want 0", got)
	}
}

func TestCommitOfAnUnderEstimateIsTheDocumentedBoundExcess(t *testing.T) {
	// The one legitimate way settled spend passes the limit: the estimate was
	// too low. The money was really spent upstream, so recording it is correct;
	// refusing to record it would make the ledger and the budget disagree.
	clk := newClock(noon)
	b := newBudget(t, clk, Limit{Amount: money.Cents(10), Period: PeriodHourly})

	res, d := b.Reserve("acme", money.Cents(1))
	if d.Outcome != Allow {
		t.Fatal("setup")
	}
	if _, err := b.Commit(res, money.Cents(30)); err != nil {
		t.Fatalf("Commit: %v", err)
	}
	u := b.Usage("acme")
	if u.Spent != money.Cents(30) {
		t.Fatalf("Spent = %s, want the real $0.30", money.FormatUSD(u.Spent))
	}
	if u.Remaining >= 0 {
		t.Errorf("Remaining = %s, want negative: the tenant is genuinely over",
			money.FormatUSD(u.Remaining))
	}
	// And the over-limit tenant is now rejected until the window rolls.
	if _, d := b.Reserve("acme", 1); d.Outcome != Reject {
		t.Errorf("an over-limit tenant was admitted: %s", d.Message())
	}
	if st := b.Status("acme"); st.Outcome != Reject {
		t.Errorf("Status outcome = %v, want Reject", st.Outcome)
	}
	// The window rolling clears it.
	clk.Advance(time.Hour)
	if _, d := b.Reserve("acme", money.Cents(5)); d.Outcome != Allow {
		t.Errorf("the new window did not clear the overspend: %s", d.Message())
	}
}

// TestOverEstimateCostsAdmissionNotMoney quantifies the conservatism the package
// documents: an estimate k times the true cost admits 1/k of the true concurrent
// capacity, while charging the tenant nothing extra and leaving sustained
// throughput untouched.
func TestOverEstimateCostsAdmissionNotMoney(t *testing.T) {
	const overEstimateFactor = 4
	trueCost := money.Cents(1)
	estimate := trueCost * overEstimateFactor
	limit := money.Cents(20) // fits 20 true requests, or 5 estimates at once

	clk := newClock(noon)
	b := newBudget(t, clk, Limit{Amount: limit, Period: PeriodHourly})

	// Phase 1: simultaneity. Only limit/estimate = 5 fit at once, even though the
	// true cost of 5 requests is a quarter of the budget.
	var live []Reservation
	for {
		res, d := b.Reserve("acme", estimate)
		if !d.Outcome.Admitted() {
			break
		}
		live = append(live, res)
	}
	if len(live) != 5 {
		t.Fatalf("admitted %d concurrent requests, want 5 (limit/estimate)", len(live))
	}
	trueSpendOfLive := trueCost * money.Pico(len(live))
	if trueSpendOfLive != money.Cents(5) {
		t.Fatalf("test arithmetic: %s", money.FormatUSD(trueSpendOfLive))
	}
	// The quantified cost of conservatism: 75% of the budget is held against
	// spend that will never happen.
	heldButUnspent := b.Usage("acme").Reserved - trueSpendOfLive
	if heldButUnspent != money.Cents(15) {
		t.Errorf("held-but-never-spent = %s, want $0.15", money.FormatUSD(heldButUnspent))
	}

	// Phase 2: nobody is charged for the over-estimate.
	for _, res := range live {
		if _, err := b.Commit(res, trueCost); err != nil {
			t.Fatal(err)
		}
	}
	u := b.Usage("acme")
	if u.Spent != trueSpendOfLive {
		t.Fatalf("Spent = %s, want the true %s: the tenant must not be charged the estimate",
			money.FormatUSD(u.Spent), money.FormatUSD(trueSpendOfLive))
	}
	if u.Reserved != 0 {
		t.Fatalf("Reserved = %s, want the whole over-estimate returned", money.FormatUSD(u.Reserved))
	}

	// Phase 3: sustained throughput is unaffected. Serially, the window still
	// admits the full 20 true requests despite the 4x estimate, because each
	// Commit returns the unused hold before the next Reserve.
	admitted := len(live)
	for {
		res, d := b.Reserve("acme", estimate)
		if !d.Outcome.Admitted() {
			break
		}
		if _, err := b.Commit(res, trueCost); err != nil {
			t.Fatal(err)
		}
		admitted++
		if admitted > 100 {
			t.Fatal("runaway loop")
		}
	}
	// 16 requests fit ($0.16 spent) before the 4-cent estimate no longer does.
	if admitted != 17 {
		t.Errorf("serial throughput = %d requests, want 17; the over-estimate must "+
			"limit simultaneity, not sustained throughput", admitted)
	}
	if got := b.Usage("acme").Spent; got != money.Cents(17) {
		t.Errorf("Spent = %s, want $0.17", money.FormatUSD(got))
	}
}

func TestExpirySweepReclaimsAbandonedHolds(t *testing.T) {
	// A handler that crashed between Reserve and Commit leaks a hold, and a
	// leaked hold eventually rejects everything. The sweep is the bound on that.
	clk := newClock(noon)
	b := New(Options{
		Now:          clk.Now,
		DefaultLimit: Limit{Amount: money.Cents(10), Period: PeriodHourly},
		HoldTTL:      30 * time.Second,
	})

	// Abandon the whole budget.
	for i := 0; i < 10; i++ {
		if _, d := b.Reserve("acme", money.Cents(1)); !d.Outcome.Admitted() {
			t.Fatalf("setup reserve %d: %s", i, d.Message())
		}
	}
	if _, d := b.Reserve("acme", money.Cents(1)); d.Outcome != Reject {
		t.Fatal("a fully-held budget should reject")
	}
	if got := b.OutstandingHolds(); got != 10 {
		t.Fatalf("OutstandingHolds = %d, want 10", got)
	}

	// One nanosecond before the TTL: nothing is reclaimed. The sweep must not be
	// eager, or it becomes a source of over-admission.
	clk.Advance(30*time.Second - time.Nanosecond)
	if n := b.Sweep(); n != 0 {
		t.Fatalf("swept %d holds before the TTL elapsed", n)
	}
	if _, d := b.Reserve("acme", money.Cents(1)); d.Outcome != Reject {
		t.Fatal("a premature sweep freed headroom")
	}

	// At the TTL exactly: everything is reclaimed.
	clk.Advance(time.Nanosecond)
	if n := b.Sweep(); n != 10 {
		t.Fatalf("swept %d holds at the TTL, want 10", n)
	}
	u := b.Usage("acme")
	if u.Reserved != 0 || u.Holds != 0 {
		t.Fatalf("after the sweep: reserved=%s holds=%d, want 0/0",
			money.FormatUSD(u.Reserved), u.Holds)
	}
	if u.Spent != 0 {
		t.Errorf("Spent = %s; a swept hold is released, not charged", money.FormatUSD(u.Spent))
	}
	if _, d := b.Reserve("acme", money.Cents(10)); !d.Outcome.Admitted() {
		t.Fatalf("the sweep did not restore headroom: %s", d.Message())
	}
	if s := b.Stats(); s.Expired != 10 {
		t.Errorf("Stats.Expired = %d, want 10: a leak must be visible, not papered over", s.Expired)
	}
}

func TestReserveSweepsWithoutABackgroundTicker(t *testing.T) {
	// The sweep must work in a process with no ticker: a crashed handler's hold
	// is reclaimed by the next Reserve.
	clk := newClock(noon)
	b := New(Options{
		Now:          clk.Now,
		DefaultLimit: Limit{Amount: money.Cents(2), Period: PeriodHourly},
		HoldTTL:      time.Minute,
	})
	if _, d := b.Reserve("acme", money.Cents(2)); !d.Outcome.Admitted() {
		t.Fatal("setup")
	}
	clk.Advance(2 * time.Minute)
	_, d := b.Reserve("acme", money.Cents(2))
	if !d.Outcome.Admitted() {
		t.Fatalf("Reserve did not sweep the expired hold: %s", d.Message())
	}
	if d.Reserved != money.Cents(2) {
		t.Errorf("Reserved = %s, want only this request's own hold", money.FormatUSD(d.Reserved))
	}
}

func TestCommitAfterSweepStillRecordsTheSpend(t *testing.T) {
	// A slow handler whose hold was swept still spent real money; losing the
	// charge would silently under-bill exactly the slowest requests.
	clk := newClock(noon)
	b := New(Options{
		Now:          clk.Now,
		DefaultLimit: Limit{Amount: money.USD(1), Period: PeriodHourly},
		HoldTTL:      time.Second,
	})
	res, _ := b.Reserve("acme", money.Cents(10))
	clk.Advance(2 * time.Second)
	if n := b.Sweep(); n != 1 {
		t.Fatalf("swept %d, want 1", n)
	}
	d, err := b.Commit(res, money.Cents(12))
	if !errors.Is(err, ErrExpiredReservation) {
		t.Fatalf("Commit = %v, want ErrExpiredReservation", err)
	}
	if d.Spent != money.Cents(12) {
		t.Errorf("Spent = %s, want the real $0.12", money.FormatUSD(d.Spent))
	}
	if s := b.Stats(); s.LateCommits != 1 {
		t.Errorf("Stats.LateCommits = %d, want 1", s.LateCommits)
	}
	if got := b.Usage("acme").Reserved; got != 0 {
		t.Errorf("Reserved = %s; a late Commit must not re-decrement a swept hold",
			money.FormatUSD(got))
	}
}

func TestSweepQueueIsOrderedAndCommitUnlinksCleanly(t *testing.T) {
	// The expiry queue is a linked list kept in expiry order. Unlinking from the
	// head, tail and middle must all leave it traversable, or a later sweep
	// either misses holds or walks off the end.
	clk := newClock(noon)
	b := New(Options{
		Now:          clk.Now,
		DefaultLimit: Limit{Amount: money.USD(10), Period: PeriodHourly},
		HoldTTL:      time.Minute,
	})
	var res []Reservation
	for i := 0; i < 5; i++ {
		r, d := b.Reserve("acme", money.Cents(1))
		if !d.Outcome.Admitted() {
			t.Fatal("setup")
		}
		res = append(res, r)
		clk.Advance(time.Second) // stagger the expiries
	}
	// Unlink the head, the tail, and one from the middle.
	for _, i := range []int{0, 4, 2} {
		if err := b.Release(res[i]); err != nil {
			t.Fatalf("Release(%d): %v", i, err)
		}
	}
	if got := b.OutstandingHolds(); got != 2 {
		t.Fatalf("OutstandingHolds = %d, want 2", got)
	}
	// The remaining two must still be sweepable.
	clk.Advance(2 * time.Minute)
	if n := b.Sweep(); n != 2 {
		t.Fatalf("swept %d, want the 2 survivors", n)
	}
	if got := b.OutstandingHolds(); got != 0 {
		t.Fatalf("OutstandingHolds = %d after the sweep", got)
	}
	if got := b.Usage("acme").Reserved; got != 0 {
		t.Errorf("Reserved = %s, want 0", money.FormatUSD(got))
	}
}

func TestPeriodRolloverDoesNotLoseInFlightReservations(t *testing.T) {
	// A request admitted at 12:59 finishing at 13:01 must be charged to the
	// twelve-o'clock window that authorised it. Charging the new window would let
	// a burst eat the next window's headroom.
	clk := newClock(time.Date(2026, 3, 15, 12, 59, 30, 0, time.UTC))
	b := New(Options{
		Now:          clk.Now,
		DefaultLimit: Limit{Amount: money.Cents(10), Period: PeriodHourly},
		HoldTTL:      10 * time.Minute,
	})

	res, d := b.Reserve("acme", money.Cents(8))
	if d.Outcome != Allow {
		t.Fatalf("setup: %s", d.Message())
	}
	if !res.WindowStart.Equal(time.Date(2026, 3, 15, 12, 0, 0, 0, time.UTC)) {
		t.Fatalf("reservation window = %v, want 12:00Z", res.WindowStart)
	}

	clk.Advance(90 * time.Second) // now 13:01, a new window
	u := b.Usage("acme")
	if u.PeriodStart.Hour() != 13 {
		t.Fatalf("window did not roll: %v", u.PeriodStart)
	}
	// The new window is fresh: the old hold does not encumber it...
	if u.Reserved != 0 {
		t.Errorf("new window Reserved = %s, want 0", money.FormatUSD(u.Reserved))
	}
	// ...but the hold is not lost either; it is carried against the old window.
	if u.CarriedHolds != 1 {
		t.Errorf("CarriedHolds = %d, want 1: the in-flight hold must survive the rollover", u.CarriedHolds)
	}
	if got := b.OutstandingHolds(); got != 1 {
		t.Errorf("OutstandingHolds = %d, want 1", got)
	}

	// The commit lands on the window that admitted it, so the new window stays
	// clean and the full new budget is available.
	if _, err := b.Commit(res, money.Cents(8)); err != nil {
		t.Fatalf("Commit: %v", err)
	}
	u = b.Usage("acme")
	if u.Spent != 0 {
		t.Errorf("new window Spent = %s, want 0: the charge belongs to the 12:00 window",
			money.FormatUSD(u.Spent))
	}
	if u.CarriedHolds != 0 {
		t.Errorf("CarriedHolds = %d after Commit, want 0", u.CarriedHolds)
	}
	if _, d := b.Reserve("acme", money.Cents(10)); d.Outcome != Allow {
		t.Errorf("the new window did not offer its full budget: %s", d.Message())
	}
}

func TestRolledWindowsWithNoHoldsAreNotRetainedForever(t *testing.T) {
	// One retained window per elapsed hour would be a slow leak in a
	// long-running process.
	clk := newClock(noon)
	b := New(Options{
		Now:          clk.Now,
		DefaultLimit: Limit{Amount: money.USD(1), Period: PeriodHourly},
		HoldTTL:      time.Minute,
	})
	for i := 0; i < 200; i++ {
		res, d := b.Reserve("acme", money.Cents(1))
		if !d.Outcome.Admitted() {
			t.Fatalf("iteration %d: %s", i, d.Message())
		}
		if _, err := b.Commit(res, money.Cents(1)); err != nil {
			t.Fatal(err)
		}
		clk.Advance(time.Hour)
	}
	b.mu.Lock()
	past := len(b.tenants["acme"].past)
	b.mu.Unlock()
	if past != 0 {
		t.Fatalf("retained %d past windows after 200 rollovers, want 0", past)
	}
	if s := b.Stats(); s.Rollovers < 190 {
		t.Errorf("Rollovers = %d, want ~199", s.Rollovers)
	}
}

func TestSpendResetsAtTheWindowBoundary(t *testing.T) {
	clk := newClock(time.Date(2026, 3, 15, 12, 0, 0, 0, time.UTC))
	b := newBudget(t, clk, Limit{Amount: money.Cents(10), Period: PeriodHourly})

	res, _ := b.Reserve("acme", money.Cents(10))
	if _, err := b.Commit(res, money.Cents(10)); err != nil {
		t.Fatal(err)
	}
	if _, d := b.Reserve("acme", 1); d.Outcome != Reject {
		t.Fatal("should be exhausted")
	}

	// One nanosecond before the boundary: still exhausted.
	clk.Set(time.Date(2026, 3, 15, 12, 59, 59, 999999999, time.UTC))
	if _, d := b.Reserve("acme", 1); d.Outcome != Reject {
		t.Error("budget reset before the boundary")
	}
	// Exactly on the boundary: the new window, per the half-open rule.
	clk.Set(time.Date(2026, 3, 15, 13, 0, 0, 0, time.UTC))
	_, d := b.Reserve("acme", money.Cents(10))
	if d.Outcome != Allow {
		t.Fatalf("budget did not reset exactly on the boundary: %s", d.Message())
	}
	if d.Spent != 0 {
		t.Errorf("Spent = %s in the new window, want 0", money.FormatUSD(d.Spent))
	}
	if !d.PeriodStart.Equal(time.Date(2026, 3, 15, 13, 0, 0, 0, time.UTC)) {
		t.Errorf("PeriodStart = %v, want 13:00Z", d.PeriodStart)
	}
}

func TestTenantsAreIsolated(t *testing.T) {
	clk := newClock(noon)
	b := newBudget(t, clk, Limit{Amount: money.Cents(10), Period: PeriodHourly})

	res, _ := b.Reserve("acme", money.Cents(10))
	if _, err := b.Commit(res, money.Cents(10)); err != nil {
		t.Fatal(err)
	}
	if _, d := b.Reserve("acme", 1); d.Outcome != Reject {
		t.Fatal("acme should be exhausted")
	}
	if _, d := b.Reserve("globex", money.Cents(10)); d.Outcome != Allow {
		t.Fatalf("globex was affected by acme's spend: %s", d.Message())
	}
	if u := b.Usage("globex"); u.Spent != 0 {
		t.Errorf("globex Spent = %s", money.FormatUSD(u.Spent))
	}
}

func TestPerTenantLimitsOverrideTheDefault(t *testing.T) {
	clk := newClock(noon)
	b := newBudget(t, clk, Limit{Amount: money.Cents(10), Period: PeriodHourly})

	if err := b.SetLimit("whale", Limit{Amount: money.USD(100), Period: PeriodMonthly}); err != nil {
		t.Fatal(err)
	}
	if got := b.LimitFor("whale").Amount; got != money.USD(100) {
		t.Errorf("whale limit = %s", money.FormatUSD(got))
	}
	if got := b.LimitFor("minnow").Amount; got != money.Cents(10) {
		t.Errorf("minnow limit = %s, want the default", money.FormatUSD(got))
	}
	_, d := b.Reserve("whale", money.USD(50))
	if d.Outcome != Allow {
		t.Fatalf("whale rejected under its own limit: %s", d.Message())
	}
	if d.Period != PeriodMonthly {
		t.Errorf("Period = %v, want monthly", d.Period)
	}

	// Bad limits are refused, and the old one stays in force.
	if err := b.SetLimit("whale", Limit{Amount: money.USD(1)}); err == nil {
		t.Error("SetLimit accepted a limit with no period")
	}
	if err := b.SetLimit("", Limit{}); err == nil {
		t.Error("SetLimit accepted an empty tenant")
	}
	if got := b.LimitFor("whale").Amount; got != money.USD(100) {
		t.Errorf("a rejected SetLimit changed the limit to %s", money.FormatUSD(got))
	}
	if err := b.SetDefaultLimit(Limit{Amount: -5, Period: PeriodDaily}); err == nil {
		t.Error("SetDefaultLimit accepted a negative amount")
	}
}

func TestLoweringALimitBelowCurrentSpendRejectsUntilTheWindowRolls(t *testing.T) {
	clk := newClock(noon)
	b := newBudget(t, clk, Limit{Amount: money.USD(1), Period: PeriodHourly})

	res, _ := b.Reserve("acme", money.Cents(80))
	if _, err := b.Commit(res, money.Cents(80)); err != nil {
		t.Fatal(err)
	}
	if err := b.SetLimit("acme", Limit{Amount: money.Cents(50), Period: PeriodHourly}); err != nil {
		t.Fatal(err)
	}
	// Settled spend is not rewritten; the tenant is simply over.
	u := b.Usage("acme")
	if u.Spent != money.Cents(80) {
		t.Errorf("Spent = %s, want the unchanged $0.80", money.FormatUSD(u.Spent))
	}
	if u.Limit != money.Cents(50) {
		t.Errorf("Limit = %s, want the new $0.50", money.FormatUSD(u.Limit))
	}
	if _, d := b.Reserve("acme", 1); d.Outcome != Reject {
		t.Error("a lowered limit did not take effect")
	}
	clk.Advance(time.Hour)
	if _, d := b.Reserve("acme", money.Cents(50)); d.Outcome != Allow {
		t.Errorf("the new window did not apply the new limit: %s", d.Message())
	}
}

func TestUnlimitedTenantIsAdmittedButStillTracked(t *testing.T) {
	// Fail OPEN for an unconfigured tenant: a budget is a cost control, not a
	// security control, and one missing config entry must not be an outage.
	clk := newClock(noon)
	b := New(Options{Now: clk.Now}) // no default limit at all

	res, d := b.Reserve("stranger", money.USD(1000))
	if d.Outcome != Allow {
		t.Fatalf("an unconfigured tenant was rejected: %s", d.Message())
	}
	if !d.Unlimited {
		t.Error("Unlimited must be set so the overspend is not invisible")
	}
	if d.Reason != ReasonNoLimit {
		t.Errorf("Reason = %q, want %q", d.Reason, ReasonNoLimit)
	}
	if !strings.Contains(d.Message(), "no configured budget") {
		t.Errorf("message = %q", d.Message())
	}
	// A hold is still taken, so in-flight spend is observable and the caller's
	// Commit/Release path is uniform.
	if !res.Valid() {
		t.Fatal("an unlimited tenant must still get a resolvable reservation")
	}
	if u := b.Usage("stranger"); u.Reserved != money.USD(1000) || u.Holds != 1 {
		t.Errorf("unlimited tenant not tracked: reserved=%s holds=%d",
			money.FormatUSD(u.Reserved), u.Holds)
	}
	if _, err := b.Commit(res, money.USD(900)); err != nil {
		t.Fatal(err)
	}
	u := b.Usage("stranger")
	if u.Spent != money.USD(900) {
		t.Errorf("Spent = %s, want $900 recorded even with no limit", money.FormatUSD(u.Spent))
	}
	if !u.Unlimited {
		t.Error("TenantUsage.Unlimited must say so too")
	}
	// No window to reset, so no misleading instant is reported.
	if !u.PeriodEnd.IsZero() || u.ResetIn != 0 {
		t.Errorf("unlimited tenant reports a reset at %v in %v; there is no window",
			u.PeriodEnd, u.ResetIn)
	}
}

func TestChargeRecordsUnreservedSpend(t *testing.T) {
	// The failover case: one client request, several billable upstream attempts,
	// only one of which was individually reserved.
	clk := newClock(noon)
	b := newBudget(t, clk, Limit{Amount: money.USD(1), Period: PeriodHourly})

	res, _ := b.Reserve("acme", money.Cents(20))
	// Attempt 1 died after burning tokens; the cost was never reserved.
	if err := b.Charge("acme", money.Cents(3)); err != nil {
		t.Fatal(err)
	}
	// Attempt 2 succeeded under the original hold.
	if _, err := b.Commit(res, money.Cents(15)); err != nil {
		t.Fatal(err)
	}
	u := b.Usage("acme")
	if u.Spent != money.Cents(18) {
		t.Errorf("Spent = %s, want $0.18: the failed attempt's cost must count",
			money.FormatUSD(u.Spent))
	}
	if s := b.Stats(); s.Unreserved != 1 {
		t.Errorf("Stats.Unreserved = %d, want 1", s.Unreserved)
	}
	if err := b.Charge("", money.Cents(1)); err == nil {
		t.Error("Charge accepted an empty tenant")
	}
	if err := b.Charge("acme", -1); err == nil {
		t.Error("Charge accepted a negative amount")
	}
}

func TestNegativeEstimateIsClampedAndCounted(t *testing.T) {
	// A negative estimate is a caller bug. Refusing traffic over an accounting
	// mistake is the wrong trade, but it must not silently CREDIT the tenant.
	clk := newClock(noon)
	b := newBudget(t, clk, Limit{Amount: money.Cents(10), Period: PeriodHourly})

	res, d := b.Reserve("acme", -money.Cents(50))
	if d.Outcome != Allow {
		t.Fatalf("outcome = %v, want Allow", d.Outcome)
	}
	if d.Estimate != 0 {
		t.Errorf("Estimate = %s, want it clamped to 0", money.FormatUSD(d.Estimate))
	}
	if u := b.Usage("acme"); u.Reserved != 0 {
		t.Fatalf("Reserved = %s; a negative estimate must not create headroom",
			money.FormatUSD(u.Reserved))
	}
	if s := b.Stats(); s.InvalidEstimates != 1 {
		t.Errorf("Stats.InvalidEstimates = %d, want 1: the bug must surface", s.InvalidEstimates)
	}
	if _, err := b.Commit(res, money.Cents(1)); err != nil {
		t.Fatal(err)
	}
	if got := b.Usage("acme").Spent; got != money.Cents(1) {
		t.Errorf("Spent = %s, want $0.01", money.FormatUSD(got))
	}
}

func TestReserveRejectsOnOverflowRatherThanWrapping(t *testing.T) {
	// A wrapped total would go negative and sail past the limit check, admitting
	// everything. Reaching the guard needs a limit above MaxInt64/2, so that a
	// second hold of the same size wraps: this is the only configuration that can
	// get there, which is exactly why the guard is easy to leave untested.
	clk := newClock(noon)
	huge := money.Pico(math.MaxInt64)
	b := New(Options{
		Now:          clk.Now,
		DefaultLimit: Limit{Amount: huge, Period: PeriodHourly},
		HoldTTL:      time.Hour,
	})
	if _, d := b.Reserve("acme", huge); !d.Outcome.Admitted() {
		t.Fatalf("setup: an estimate equal to the limit must be admitted: %s", d.Message())
	}
	_, d := b.Reserve("acme", huge)
	if d.Outcome != Reject {
		t.Fatalf("outcome = %v, want Reject", d.Outcome)
	}
	if d.Reason != ReasonOverflow {
		t.Errorf("Reason = %q, want %q", d.Reason, ReasonOverflow)
	}
	if d.Remaining != 0 {
		t.Errorf("Remaining = %d, want 0 rather than a wrapped figure", d.Remaining)
	}
	u := b.Usage("acme")
	if u.Reserved < 0 || u.Spent < 0 {
		t.Fatalf("counters wrapped negative: reserved=%d spent=%d", u.Reserved, u.Spent)
	}
	if u.Reserved+u.Spent > huge {
		t.Fatalf("bound violated: %d > %d", u.Reserved+u.Spent, huge)
	}
}

func TestStatsAccountForEveryDecision(t *testing.T) {
	clk := newClock(noon)
	b := newBudget(t, clk, Limit{Amount: money.Cents(10), Period: PeriodHourly, SoftBasisPoints: 5000})

	var allowed, degraded, rejected int
	var live []Reservation
	for i := 0; i < 20; i++ {
		res, d := b.Reserve("acme", money.Cents(1))
		switch d.Outcome {
		case Allow:
			allowed++
			live = append(live, res)
		case AllowDegraded:
			degraded++
			live = append(live, res)
		case Reject:
			rejected++
		}
	}
	s := b.Stats()
	if int(s.Reserved) != 20 {
		t.Errorf("Stats.Reserved = %d, want 20", s.Reserved)
	}
	if int(s.Allowed) != allowed || int(s.Degraded) != degraded || int(s.Rejected) != rejected {
		t.Errorf("stats %+v disagree with observed %d/%d/%d", s, allowed, degraded, rejected)
	}
	if int(s.Allowed+s.Degraded+s.Rejected) != int(s.Reserved) {
		t.Errorf("the outcome counters do not partition Reserved: %+v", s)
	}
	if degraded == 0 {
		t.Error("no request was degraded despite a 50% soft threshold")
	}
	if rejected == 0 {
		t.Error("no request was rejected despite a 10-cent budget and 20 requests")
	}
	for i, res := range live {
		if i%2 == 0 {
			if _, err := b.Commit(res, money.Cents(1)/2); err != nil {
				t.Fatal(err)
			}
		} else if err := b.Release(res); err != nil {
			t.Fatal(err)
		}
	}
	s = b.Stats()
	if int(s.Committed+s.Released) != len(live) {
		t.Errorf("Committed %d + Released %d != %d holds", s.Committed, s.Released, len(live))
	}
}

func TestSnapshotIsAnObservationNotAMutation(t *testing.T) {
	// A scrape interval must not be a functional input to the system, so Snapshot
	// deliberately does not roll windows or sweep.
	clk := newClock(noon)
	b := New(Options{
		Now:          clk.Now,
		DefaultLimit: Limit{Amount: money.Cents(10), Period: PeriodHourly},
		HoldTTL:      time.Second,
	})
	if _, d := b.Reserve("acme", money.Cents(5)); !d.Outcome.Admitted() {
		t.Fatal("setup")
	}
	clk.Advance(2 * time.Hour) // both the window and the TTL have elapsed

	s := b.Snapshot()
	if s.Holds != 1 {
		t.Errorf("Snapshot.Holds = %d, want 1; Snapshot must not sweep", s.Holds)
	}
	u, ok := s.Tenants["acme"]
	if !ok {
		t.Fatal("acme missing from the snapshot")
	}
	if u.PeriodStart.Hour() != 12 {
		t.Errorf("Snapshot rolled the window to %v; it must only observe", u.PeriodStart)
	}
	if u.ResetIn >= 0 {
		t.Errorf("ResetIn = %v, want negative to make the elapsed window visible", u.ResetIn)
	}
	if !s.Now.Equal(clk.Now()) {
		t.Errorf("Snapshot.Now = %v, want the injected clock's %v", s.Now, clk.Now())
	}
	if s.Defaults.Amount != money.Cents(10) {
		t.Errorf("Snapshot.Defaults = %+v", s.Defaults)
	}
	// A read that IS allowed to mutate says so, and does.
	if u := b.Usage("acme"); u.PeriodStart.Hour() != 14 || u.Holds != 0 {
		t.Errorf("Usage did not roll and sweep: %+v", u)
	}
	if s2 := b.Snapshot(); s2.Holds != 0 {
		t.Errorf("after Usage swept, Snapshot.Holds = %d, want 0", s2.Holds)
	}
}

func TestSnapshotTenantsAreCopies(t *testing.T) {
	// The metrics encoder formats without the lock, so the map it gets must not
	// alias live state.
	clk := newClock(noon)
	b := newBudget(t, clk, Limit{Amount: money.USD(1), Period: PeriodHourly})
	res, _ := b.Reserve("acme", money.Cents(10))
	s := b.Snapshot()
	before := s.Tenants["acme"].Reserved
	if _, err := b.Commit(res, money.Cents(10)); err != nil {
		t.Fatal(err)
	}
	if got := s.Tenants["acme"].Reserved; got != before {
		t.Fatalf("the snapshot changed under us: %s then %s",
			money.FormatUSD(before), money.FormatUSD(got))
	}
}

func TestStatusDoesNotReserve(t *testing.T) {
	clk := newClock(noon)
	b := newBudget(t, clk, Limit{Amount: money.Cents(10), Period: PeriodHourly, SoftBasisPoints: 5000})

	d := b.Status("acme")
	if d.Outcome != Allow || d.Reason != ReasonWithinLimit {
		t.Errorf("fresh Status = %v/%q", d.Outcome, d.Reason)
	}
	if d.Remaining != money.Cents(10) {
		t.Errorf("Remaining = %s, want the whole budget", money.FormatUSD(d.Remaining))
	}
	if b.OutstandingHolds() != 0 {
		t.Fatal("Status took a hold")
	}

	res, _ := b.Reserve("acme", money.Cents(6))
	if d := b.Status("acme"); d.Outcome != AllowDegraded {
		t.Errorf("Status outcome = %v past the soft threshold, want AllowDegraded", d.Outcome)
	}
	if _, err := b.Commit(res, money.Cents(10)); err != nil {
		t.Fatal(err)
	}
	if d := b.Status("acme"); d.Outcome != Reject || d.Reason != ReasonHardLimit {
		t.Errorf("Status = %v/%q at the limit, want Reject/hard_limit", d.Outcome, d.Reason)
	}
	// Status on an unknown tenant must not panic and must reflect the default.
	if d := b.Status("nobody"); d.Limit != money.Cents(10) {
		t.Errorf("Status for an unknown tenant: limit = %s", money.FormatUSD(d.Limit))
	}
}

func TestOutcomeStringAndAdmitted(t *testing.T) {
	tests := []struct {
		o        Outcome
		str      string
		admitted bool
	}{
		{Allow, "allow", true},
		{AllowDegraded, "allow_degraded", true},
		{Reject, "reject", false},
		{Outcome(42), "unknown", false},
	}
	for _, tc := range tests {
		if got := tc.o.String(); got != tc.str {
			t.Errorf("Outcome(%d).String() = %q, want %q", int(tc.o), got, tc.str)
		}
		if got := tc.o.Admitted(); got != tc.admitted {
			t.Errorf("Outcome(%d).Admitted() = %t, want %t", int(tc.o), got, tc.admitted)
		}
	}
}

func TestDefaultHoldTTLApplies(t *testing.T) {
	clk := newClock(noon)
	b := New(Options{Now: clk.Now, DefaultLimit: Limit{Amount: money.USD(1), Period: PeriodHourly}})
	res, _ := b.Reserve("acme", money.Cents(1))
	if got := res.ExpiresAt.Sub(noon); got != DefaultHoldTTL {
		t.Errorf("hold TTL = %v, want the default %v", got, DefaultHoldTTL)
	}
	// A non-positive TTL must fall back to the default, not create holds that
	// expire instantly — an immediately-expiring hold is a broken bound.
	b2 := New(Options{Now: clk.Now, HoldTTL: -time.Second,
		DefaultLimit: Limit{Amount: money.USD(1), Period: PeriodHourly}})
	res2, _ := b2.Reserve("acme", money.Cents(1))
	if got := res2.ExpiresAt.Sub(noon); got != DefaultHoldTTL {
		t.Errorf("negative HoldTTL gave %v, want the default %v", got, DefaultHoldTTL)
	}
	if n := b2.Sweep(); n != 0 {
		t.Fatalf("a fresh hold was swept immediately (%d)", n)
	}
}

func TestNewWithoutAClockUsesTimeNow(t *testing.T) {
	b := New(Options{DefaultLimit: Limit{Amount: money.USD(1), Period: PeriodHourly}})
	before := time.Now()
	res, d := b.Reserve("acme", money.Cents(1))
	if d.Outcome != Allow {
		t.Fatalf("outcome = %v", d.Outcome)
	}
	if res.ExpiresAt.Before(before) {
		t.Errorf("ExpiresAt %v predates the call at %v", res.ExpiresAt, before)
	}
	if err := b.Release(res); err != nil {
		t.Fatal(err)
	}
}

func TestManyTenantsConcurrentlyAreIsolatedAndBounded(t *testing.T) {
	// Run with -race. Each tenant's bound must hold independently, and the
	// per-tenant map must not race on first insert.
	const tenants = 16
	const perTenant = 60
	limit := money.Cents(10)

	clk := newClock(noon)
	b := New(Options{
		Now:          clk.Now,
		DefaultLimit: Limit{Amount: limit, Period: PeriodHourly},
		HoldTTL:      time.Hour,
	})

	var wg sync.WaitGroup
	for i := 0; i < tenants; i++ {
		for j := 0; j < perTenant; j++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				tenant := fmt.Sprintf("t%d", i)
				res, d := b.Reserve(tenant, money.Cents(1))
				if !d.Outcome.Admitted() {
					return
				}
				if _, err := b.Commit(res, money.Cents(1)); err != nil {
					t.Errorf("Commit: %v", err)
				}
			}(i)
		}
	}
	wg.Wait()

	for i := 0; i < tenants; i++ {
		tenant := fmt.Sprintf("t%d", i)
		u := b.Usage(tenant)
		if u.Spent+u.Reserved > limit {
			t.Errorf("%s over-admitted: spent %s + reserved %s > %s", tenant,
				money.FormatUSD(u.Spent), money.FormatUSD(u.Reserved), money.FormatUSD(limit))
		}
		if u.Spent != limit {
			t.Errorf("%s Spent = %s, want the full %s (60 requests of $0.01 into $0.10)",
				tenant, money.FormatUSD(u.Spent), money.FormatUSD(limit))
		}
	}
	if got := b.OutstandingHolds(); got != 0 {
		t.Errorf("OutstandingHolds = %d, want 0", got)
	}
}

func BenchmarkReserveCommit(b *testing.B) {
	bg := New(Options{
		DefaultLimit: Limit{Amount: money.USD(1_000_000), Period: PeriodHourly},
		HoldTTL:      time.Hour,
	})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, d := bg.Reserve("acme", money.Cents(1))
		if !d.Outcome.Admitted() {
			b.Fatal("rejected")
		}
		if _, err := bg.Commit(res, money.Cents(1)/2); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkReserveCommitParallel(b *testing.B) {
	bg := New(Options{
		DefaultLimit: Limit{Amount: money.USD(1_000_000_000), Period: PeriodHourly},
		HoldTTL:      time.Hour,
	})
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			res, d := bg.Reserve("acme", money.Cents(1))
			if !d.Outcome.Admitted() {
				continue
			}
			if _, err := bg.Commit(res, 1); err != nil {
				b.Error(err)
			}
		}
	})
}
