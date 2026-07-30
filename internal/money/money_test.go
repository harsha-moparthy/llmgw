package money

import (
	"errors"
	"math"
	"strings"
	"testing"
)

func TestUSDAndCents(t *testing.T) {
	tests := []struct {
		name string
		got  Pico
		want Pico
	}{
		{"one dollar", USD(1), 1_000_000_000_000},
		{"zero dollars", USD(0), 0},
		{"negative dollars", USD(-3), -3_000_000_000_000},
		{"one cent", Cents(1), 10_000_000_000},
		{"a hundred cents is a dollar", Cents(100), USD(1)},
		{"negative cents", Cents(-25), -250_000_000_000},
		{"the constants agree", Pico(PicoPerUSD), Pico(PicoPerCent) * 100},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if tc.got != tc.want {
				t.Fatalf("got %d, want %d", tc.got, tc.want)
			}
		})
	}
}

// TestExactnessOfRepresentativePrices is the claim the whole package exists to
// support: the amounts that appear in real LLM billing are exact in picodollars,
// and the float64 arithmetic they replace is not.
func TestExactnessOfRepresentativePrices(t *testing.T) {
	// The canonical float64 counterexample, in the units this gateway bills in.
	a, b := usdOrFatal(t, "0.1"), usdOrFatal(t, "0.2")
	sum, err := Add(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if want := usdOrFatal(t, "0.3"); sum != want {
		t.Fatalf("0.1 + 0.2 = %d, want %d", sum, want)
	}
	// Deliberately through float64 variables and not constants: Go evaluates
	// untyped constant arithmetic at arbitrary precision, so `0.1+0.2 == 0.3`
	// written literally is true at compile time and would prove nothing about the
	// runtime arithmetic this package replaces.
	f1, f2, f3 := 0.1, 0.2, 0.3
	if f1+f2 == f3 {
		t.Fatal("float64 has apparently been fixed; this package's premise needs revisiting")
	}

	// A thousand additions of a third of a cent must land exactly on $3.33...,
	// which is where a float64 accumulator would have drifted.
	third := usdOrFatal(t, "0.0033")
	var acc Pico
	for i := 0; i < 1000; i++ {
		if acc, err = Add(acc, third); err != nil {
			t.Fatal(err)
		}
	}
	if want := usdOrFatal(t, "3.30"); acc != want {
		t.Fatalf("accumulated %s, want %s", FormatUSD(acc), FormatUSD(want))
	}

	// The end-to-end shape of a real charge: 1,000,000 tokens at $0.15/1M must
	// cost exactly $0.15, with no rounding anywhere in the chain.
	perToken, err := PerToken(usdOrFatal(t, "0.15"))
	if err != nil {
		t.Fatal(err)
	}
	cost, err := Cost(perToken, TokensPerPriceUnit)
	if err != nil {
		t.Fatal(err)
	}
	if want := usdOrFatal(t, "0.15"); cost != want {
		t.Fatalf("1M tokens at $0.15/1M cost %s, want %s", FormatUSD(cost), FormatUSD(want))
	}
}

func usdOrFatal(t *testing.T, s string) Pico {
	t.Helper()
	p, err := ParseUSD(s)
	if err != nil {
		t.Fatalf("ParseUSD(%q): %v", s, err)
	}
	return p
}

func TestMul(t *testing.T) {
	tests := []struct {
		name    string
		p       Pico
		n       int64
		want    Pico
		wantErr error
		errHas  string
	}{
		{name: "typical", p: 150_000, n: 1000, want: 150_000_000},
		{name: "by zero", p: 150_000, n: 0, want: 0},
		{name: "zero price by a huge count", p: 0, n: math.MaxInt64, want: 0},
		{name: "by one", p: 12345, n: 1, want: 12345},
		{name: "negative amount", p: -150_000, n: 3, want: -450_000},
		{
			// A negative count is a programming error, not an overflow, and gets a
			// distinct message so it is not mistaken for one.
			name: "negative multiplier is rejected", p: 1, n: -1, errHas: "negative multiplier",
		},
		{
			name: "negative multiplier with a zero amount is still rejected",
			p:    0, n: -1, errHas: "negative multiplier",
		},
		{
			// 2^40 tokens at a real per-token price is only $1.6e5 in pico terms,
			// well inside the range: the guard must NOT fire here, or a legitimate
			// (if enormous) request would be rejected as an overflow.
			name: "an enormous but representable token count is allowed",
			p:    150_000, n: 1 << 40, want: 150_000 * (1 << 40),
		},
		{
			// The hostile case the docstring names: 2^45 tokens at the same price
			// exceeds the guard, and must produce an error rather than a wrapped
			// negative charge.
			name: "hostile token count overflows", p: 150_000, n: 1 << 45, wantErr: ErrOverflow,
		},
		{name: "negative amount overflows too", p: -(1 << 40), n: 1 << 30, wantErr: ErrOverflow},
		{name: "max int multiplier", p: 2, n: math.MaxInt64, wantErr: ErrOverflow},
		{
			// The guard limit is 1<<62, so the largest product it admits is
			// exactly 1<<62.
			name: "at the guard limit", p: 1 << 62, n: 1, want: 1 << 62,
		},
		{
			// Deliberately conservative: 3<<61 fits in an int64 (it is 6.9e18,
			// under MaxInt64) but the 1<<62 guard rejects it. Refusing a product
			// that would have fit is a $4.6M-scale false negative in a gateway
			// that bills fractions of a cent, and it buys a branch-free check on
			// the hot path. This test pins the trade-off so nobody "fixes" the
			// guard without noticing it was chosen.
			name: "conservative rejection above the guard but inside int64",
			p:    3 << 60, n: 2, wantErr: ErrOverflow,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := Mul(tc.p, tc.n)
			switch {
			case tc.wantErr != nil:
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("Mul(%d, %d) err = %v, want %v", tc.p, tc.n, err, tc.wantErr)
				}
			case tc.errHas != "":
				if err == nil || !strings.Contains(err.Error(), tc.errHas) {
					t.Fatalf("Mul(%d, %d) err = %v, want one mentioning %q", tc.p, tc.n, err, tc.errHas)
				}
			default:
				if err != nil {
					t.Fatalf("Mul(%d, %d): unexpected error %v", tc.p, tc.n, err)
				}
				if got != tc.want {
					t.Fatalf("Mul(%d, %d) = %d, want %d", tc.p, tc.n, got, tc.want)
				}
				return
			}
			// Every error path must also yield a zero, so a caller that ignores
			// the error (badly, but it happens) records nothing rather than
			// recording a wrapped number that looks like money.
			if got != 0 {
				t.Fatalf("Mul(%d, %d) returned %d alongside an error; want 0", tc.p, tc.n, got)
			}
		})
	}
}

// TestMulNeverWraps sweeps the region around the guard to prove no input
// produces a value whose sign or magnitude contradicts the operands. A wrapped
// product is a negative invoice, so a spot check is not enough.
func TestMulNeverWraps(t *testing.T) {
	amounts := []Pico{0, 1, -1, 150_000, -150_000, 1 << 30, 1 << 45, 1 << 61, 1 << 62,
		-(1 << 61), Pico(math.MaxInt64), Pico(math.MinInt64 + 1)}
	counts := []int64{0, 1, 2, 3, 1 << 20, 1 << 31, 1 << 40, 1 << 62, math.MaxInt64}
	for _, p := range amounts {
		for _, n := range counts {
			got, err := Mul(p, n)
			if err != nil {
				if got != 0 {
					t.Fatalf("Mul(%d, %d) = %d with error %v; want 0", p, n, got, err)
				}
				continue
			}
			// No error means the product must be exactly right, which we verify
			// by dividing back out.
			if n != 0 && p != 0 && got/Pico(n) != p {
				t.Fatalf("Mul(%d, %d) = %d, which does not divide back to %d", p, n, got, p)
			}
			if (p > 0 && n > 0 && got < 0) || (p < 0 && n > 0 && got > 0) {
				t.Fatalf("Mul(%d, %d) = %d has the wrong sign", p, n, got)
			}
		}
	}
}

func TestAdd(t *testing.T) {
	const maxP = Pico(math.MaxInt64)
	const minP = Pico(math.MinInt64)
	tests := []struct {
		name    string
		a, b    Pico
		want    Pico
		wantErr bool
	}{
		{name: "typical", a: 1_500_000, b: 2_500_000, want: 4_000_000},
		{name: "zero identity", a: 0, b: 0, want: 0},
		{name: "negative plus positive nets out", a: -500, b: 500, want: 0},
		{name: "negative plus negative", a: -500, b: -250, want: -750},
		{name: "positive plus negative stays negative", a: 250, b: -1000, want: -750},
		{name: "at the ceiling exactly", a: maxP - 1, b: 1, want: maxP},
		{name: "at the floor exactly", a: minP + 1, b: -1, want: minP},
		{name: "one past the ceiling", a: maxP, b: 1, wantErr: true},
		{name: "doubling the ceiling", a: maxP, b: maxP, wantErr: true},
		{name: "one past the floor", a: minP, b: -1, wantErr: true},
		{name: "doubling the floor", a: minP, b: minP, wantErr: true},
		{
			// Mixed signs can never overflow, so this must succeed even though
			// both operands are extreme.
			name: "mixed extremes cannot overflow", a: maxP, b: minP, want: -1,
		},
		{name: "adding zero to the ceiling", a: maxP, b: 0, want: maxP},
		{name: "adding zero to the floor", a: minP, b: 0, want: minP},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := Add(tc.a, tc.b)
			if tc.wantErr {
				if !errors.Is(err, ErrOverflow) {
					t.Fatalf("Add(%d, %d) err = %v, want ErrOverflow", tc.a, tc.b, err)
				}
				if got != 0 {
					t.Fatalf("Add(%d, %d) = %d alongside an error; want 0", tc.a, tc.b, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("Add(%d, %d): unexpected error %v", tc.a, tc.b, err)
			}
			if got != tc.want {
				t.Fatalf("Add(%d, %d) = %d, want %d", tc.a, tc.b, got, tc.want)
			}
		})
	}
}

func TestAddIsCommutative(t *testing.T) {
	vals := []Pico{0, 1, -1, 150_000, -2_500_000, Pico(math.MaxInt64), Pico(math.MinInt64),
		Pico(math.MaxInt64) - 1, Pico(math.MinInt64) + 1}
	for _, a := range vals {
		for _, b := range vals {
			ab, errAB := Add(a, b)
			ba, errBA := Add(b, a)
			if (errAB == nil) != (errBA == nil) || ab != ba {
				t.Fatalf("Add(%d,%d) = %d,%v but Add(%d,%d) = %d,%v", a, b, ab, errAB, b, a, ba, errBA)
			}
		}
	}
}

func TestPerToken(t *testing.T) {
	tests := []struct {
		name       string
		perMillion string // as an operator would write it on a config line
		want       Pico
		wantErr    bool
		errHas     string
	}{
		{name: "a whole dollar per million", perMillion: "1", want: 1_000_000},
		{name: "the common gpt-4o input rate", perMillion: "2.50", want: 2_500_000},
		{name: "the cheapest common rate", perMillion: "0.15", want: 150_000},
		{name: "a half-cent rate", perMillion: "0.005", want: 5_000},
		{name: "four decimal places is still exact", perMillion: "0.3125", want: 312_500},
		{name: "free", perMillion: "0", want: 0},
		{
			// The exactness boundary: $0.000001 per 1M tokens is 1e6 pico, which
			// divides to exactly 1 pico per token. One digit finer does not.
			name: "at the exactness boundary", perMillion: "0.000001", want: 1,
		},
		{
			name:       "one digit past the boundary is rejected, not rounded to zero",
			perMillion: "0.0000001", wantErr: true, errHas: "not exactly representable per token",
		},
		{
			name:       "an inexact price mentions how to fix it",
			perMillion: "0.0000015", wantErr: true, errHas: "at most 6 decimal places",
		},
		{
			// Would round to 3 pico/token under any rounding mode, which is
			// precisely what must not happen silently.
			name:       "a price that would round nicely is still rejected",
			perMillion: "0.0000035", wantErr: true, errHas: "not exactly representable",
		},
		{name: "negative price", perMillion: "-1", wantErr: true, errHas: "negative price"},
		{
			name:       "a tiny negative price is rejected as negative, not as inexact",
			perMillion: "-0.0000001", wantErr: true, errHas: "negative price",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			in := usdOrFatal(t, tc.perMillion)
			got, err := PerToken(in)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("PerToken(%s) = %d, want an error", tc.perMillion, got)
				}
				if !strings.Contains(err.Error(), tc.errHas) {
					t.Fatalf("error %q does not mention %q", err, tc.errHas)
				}
				if got != 0 {
					t.Fatalf("PerToken(%s) = %d alongside an error; want 0", tc.perMillion, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("PerToken(%s): unexpected error %v", tc.perMillion, err)
			}
			if got != tc.want {
				t.Fatalf("PerToken(%s) = %d, want %d", tc.perMillion, got, tc.want)
			}
			// The defining property: the per-token price, multiplied back by the
			// quote unit, is the original price exactly. This is what makes
			// downstream cost arithmetic rounding-free.
			back, err := Mul(got, TokensPerPriceUnit)
			if err != nil {
				t.Fatal(err)
			}
			if back != in {
				t.Fatalf("PerToken(%s) does not round-trip: %d * %d = %d, want %d",
					tc.perMillion, got, TokensPerPriceUnit, back, in)
			}
		})
	}
}

// TestPerTokenRejectionIsNotVacuous guards against the check silently always
// passing. If the modulo test in PerToken were removed, every value in this
// sweep would be accepted; as written, exactly the multiples of
// TokensPerPriceUnit are.
func TestPerTokenRejectionIsNotVacuous(t *testing.T) {
	accepted, rejected := 0, 0
	for v := Pico(0); v <= 3*TokensPerPriceUnit; v += 250_000 {
		_, err := PerToken(v)
		exact := v%TokensPerPriceUnit == 0
		if exact {
			if err != nil {
				t.Fatalf("PerToken(%d) rejected an exact price: %v", v, err)
			}
			accepted++
			continue
		}
		if err == nil {
			t.Fatalf("PerToken(%d) accepted an inexact price", v)
		}
		rejected++
	}
	if accepted == 0 || rejected == 0 {
		t.Fatalf("sweep is not discriminating: %d accepted, %d rejected", accepted, rejected)
	}
}

func TestParseUSD(t *testing.T) {
	tests := []struct {
		name   string
		in     string
		want   Pico
		bad    bool
		errHas string
	}{
		{name: "whole dollars", in: "1", want: 1_000_000_000_000},
		{name: "two decimals", in: "1.50", want: 1_500_000_000_000},
		{name: "leading dollar sign", in: "$2.50", want: 2_500_000_000_000},
		{name: "negative with a dollar sign", in: "-$2.50", want: -2_500_000_000_000},
		{name: "negative without a dollar sign", in: "-2", want: -2_000_000_000_000},
		{name: "explicit plus", in: "+1.25", want: 1_250_000_000_000},
		{name: "surrounding whitespace", in: "  0.15\t", want: 150_000_000_000},
		{name: "no integer part", in: ".15", want: 150_000_000_000},
		{name: "zero", in: "0", want: 0},
		{name: "negative zero is just zero", in: "-0.00", want: 0},
		{name: "trailing zeros do not change the value", in: "0.1500", want: 150_000_000_000},
		{name: "a full picodollar", in: "0.000000000001", want: 1},
		{name: "twelve decimals is the boundary", in: "1.123456789012", want: 1_123_456_789_012},
		{name: "a per-million list price", in: "0.075", want: 75_000_000_000},
		{
			name: "thirteen decimals is rejected rather than rounded",
			in:   "1.1234567890123", bad: true, errHas: "at most 12",
		},
		{name: "empty", in: "", bad: true, errHas: "empty amount"},
		{name: "whitespace only", in: "   ", bad: true, errHas: "empty amount"},
		{name: "a bare sign", in: "-", bad: true, errHas: "no digits"},
		{name: "a bare dollar sign", in: "$", bad: true, errHas: "no digits"},
		{name: "trailing decimal point", in: "1.", bad: true, errHas: "trailing decimal point"},
		{name: "not a number", in: "one dollar", bad: true, errHas: "bad integer part"},
		{name: "two decimal points", in: "1.5.5", bad: true, errHas: "bad fractional part"},
		{name: "embedded space", in: "1 500", bad: true, errHas: "bad integer part"},
		{name: "underscores are not accepted", in: "1_000", bad: true, errHas: "bad integer part"},
		{name: "hex is not accepted", in: "0x10", bad: true, errHas: "bad integer part"},
		{
			// The int64 ceiling in dollars is about $9.2M, and the guard trips at
			// half that. An amount this large in a pricing config is a units
			// mistake, and an error beats a wrapped negative.
			name: "an absurd amount overflows", in: "9999999999", bad: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ParseUSD(tc.in)
			if tc.bad {
				if err == nil {
					t.Fatalf("ParseUSD(%q) = %d, want an error", tc.in, got)
				}
				if tc.errHas != "" && !strings.Contains(err.Error(), tc.errHas) {
					t.Fatalf("error %q does not mention %q", err, tc.errHas)
				}
				if got != 0 {
					t.Fatalf("ParseUSD(%q) = %d alongside an error; want 0", tc.in, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseUSD(%q): unexpected error %v", tc.in, err)
			}
			if got != tc.want {
				t.Fatalf("ParseUSD(%q) = %d, want %d", tc.in, got, tc.want)
			}
		})
	}
}

func TestFormatUSD(t *testing.T) {
	tests := []struct {
		name string
		in   Pico
		want string
	}{
		{name: "zero", in: 0, want: "$0"},
		{name: "whole dollars omit the decimal part", in: USD(5), want: "$5"},
		{name: "trailing zeros are trimmed", in: 1_500_000_000_000, want: "$1.5"},
		{name: "cents", in: Cents(7), want: "$0.07"},
		{name: "a sub-cent cost still shows its magnitude", in: 3_000_000, want: "$0.000003"},
		{name: "one picodollar", in: 1, want: "$0.000000000001"},
		{name: "negative", in: -1_500_000_000_000, want: "-$1.5"},
		{name: "negative sub-cent", in: -3_000_000, want: "-$0.000003"},
		{name: "negative one picodollar", in: -1, want: "-$0.000000000001"},
		{name: "all twelve digits are significant", in: 1_123_456_789_012, want: "$1.123456789012"},
		{name: "interior zeros are preserved", in: 1_000_000_000_001, want: "$1.000000000001"},
		{name: "large amount", in: USD(4_611_686), want: "$4611686"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := FormatUSD(tc.in)
			if got != tc.want {
				t.Fatalf("FormatUSD(%d) = %q, want %q", tc.in, got, tc.want)
			}
			// The docstring promises a round trip; a formatter that loses a digit
			// would make a reconciliation report unusable as evidence.
			back, err := ParseUSD(got)
			if err != nil {
				t.Fatalf("ParseUSD(FormatUSD(%d)) = %q: %v", tc.in, got, err)
			}
			if back != tc.in {
				t.Fatalf("round trip: %d -> %q -> %d", tc.in, got, back)
			}
		})
	}
}

// TestFormatParseRoundTripSweep exercises the round trip across magnitudes and
// digit patterns, including the values a naive trailing-zero trim would corrupt.
func TestFormatParseRoundTripSweep(t *testing.T) {
	vals := []Pico{
		0, 1, -1, 10, 100, 999, 1_000_000_000_000, -1_000_000_000_000,
		100_000_000_000, 10_000_000_000, 1_000_000_000, 100_000_000, 10_000_000,
		150_000, 75_000, 312_500, 1, 999_999_999_999, -999_999_999_999,
		4_611_686_000_000_000_000, -4_611_686_000_000_000_000,
		123_456_789_012_345, -123_456_789_012_345,
	}
	for _, v := range vals {
		s := FormatUSD(v)
		back, err := ParseUSD(s)
		if err != nil {
			t.Fatalf("ParseUSD(%q) from %d: %v", s, v, err)
		}
		if back != v {
			t.Fatalf("round trip %d -> %q -> %d", v, s, back)
		}
	}
}

func TestFormatUSDPrec(t *testing.T) {
	tests := []struct {
		name     string
		in       Pico
		decimals int
		want     string
	}{
		{name: "two decimals, exact", in: Cents(150), decimals: 2, want: "$1.50"},
		{name: "two decimals rounds a sub-cent up at the half", in: 5_000_000_000, decimals: 2, want: "$0.01"},
		{name: "just below the half rounds down", in: 4_999_999_999, decimals: 2, want: "$0.00"},
		{name: "just above the half rounds up", in: 5_000_000_001, decimals: 2, want: "$0.01"},
		{
			// Half away from zero, not banker's rounding: $0.015 -> $0.02 and
			// $0.025 -> $0.03, where half-to-even would give $0.02 for both.
			name: "half away from zero at an odd digit", in: 15_000_000_000, decimals: 2, want: "$0.02",
		},
		{
			name: "half away from zero at an even digit", in: 25_000_000_000, decimals: 2, want: "$0.03",
		},
		{name: "negative rounds away from zero too", in: -5_000_000_000, decimals: 2, want: "-$0.01"},
		{name: "negative just below the half", in: -4_999_999_999, decimals: 2, want: "$0.00"},
		{
			// A value that rounds to zero must not print "-$0.00": a minus sign on
			// a zero reads as a credit in a rollup.
			name: "a negative that rounds to zero drops the sign", in: -1, decimals: 2, want: "$0.00",
		},
		{name: "zero decimals rounds at the half dollar", in: 500_000_000_000, decimals: 0, want: "$1"},
		{name: "zero decimals just below the half", in: 499_999_999_999, decimals: 0, want: "$0"},
		{name: "zero decimals on a whole amount", in: USD(7), decimals: 0, want: "$7"},
		{name: "six decimals, the per-token scale", in: 3_000_000, decimals: 6, want: "$0.000003"},
		{name: "six decimals rounds at its own half", in: 500_000, decimals: 6, want: "$0.000001"},
		{name: "six decimals just below its half", in: 499_999, decimals: 6, want: "$0.000000"},
		{name: "twelve decimals is lossless", in: 1_123_456_789_012, decimals: 12, want: "$1.123456789012"},
		{name: "more than twelve decimals is clamped", in: 1, decimals: 99, want: "$0.000000000001"},
		{name: "negative decimals are clamped to zero", in: USD(2), decimals: -5, want: "$2"},
		{name: "leading zeros in the fraction are padded", in: 1_000_000_001, decimals: 4, want: "$0.0010"},
		{name: "carry across the decimal point", in: 999_999_999_999, decimals: 2, want: "$1.00"},
		{name: "negative carry across the decimal point", in: -999_999_999_999, decimals: 2, want: "-$1.00"},
		{name: "zero at two decimals", in: 0, decimals: 2, want: "$0.00"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := FormatUSDPrec(tc.in, tc.decimals); got != tc.want {
				t.Fatalf("FormatUSDPrec(%d, %d) = %q, want %q", tc.in, tc.decimals, got, tc.want)
			}
		})
	}
}

// TestFormatUSDPrecNeverPrintsNegativeZero sweeps the whole band of amounts that
// round to zero at two decimals. "-$0.00" in a cost rollup is read as a refund.
func TestFormatUSDPrecNeverPrintsNegativeZero(t *testing.T) {
	for v := Pico(-4_999_999_999); v <= 0; v += 37_000_000 {
		s := FormatUSDPrec(v, 2)
		if strings.HasPrefix(s, "-") {
			t.Fatalf("FormatUSDPrec(%d, 2) = %q; a value rounding to zero must not carry a sign", v, s)
		}
	}
	// And the first value that does not round to zero must carry the sign.
	if s := FormatUSDPrec(-5_000_000_000, 2); s != "-$0.01" {
		t.Fatalf("FormatUSDPrec(-5e9, 2) = %q, want %q", s, "-$0.01")
	}
}

// TestFormatUSDPrecIsSignSymmetric checks that rounding is symmetric about zero,
// which is what "half away from zero" means and what keeps a debit and its
// reversal from failing to cancel in a report.
func TestFormatUSDPrecIsSignSymmetric(t *testing.T) {
	vals := []Pico{1, 499_999_999_999, 500_000_000_000, 5_000_000_000, 999_999_999_999,
		Cents(150), USD(3), 15_000_000_000, 25_000_000_000}
	for _, decimals := range []int{0, 2, 6} {
		for _, v := range vals {
			pos := FormatUSDPrec(v, decimals)
			neg := FormatUSDPrec(-v, decimals)
			if pos == "$0" || pos == "$0.00" || pos == "$0.000000" {
				continue // the rounds-to-zero band is asymmetric by design
			}
			if neg != "-"+pos {
				t.Fatalf("FormatUSDPrec(%d, %d) = %q but FormatUSDPrec(%d, %d) = %q",
					v, decimals, pos, -v, decimals, neg)
			}
		}
	}
}

// TestNegativeAmounts covers signed arithmetic end to end. Negative money is
// legitimate here — a ledger correction or a refund row — so it must behave, not
// merely fail to crash.
func TestNegativeAmounts(t *testing.T) {
	refund := usdOrFatal(t, "-1.25")
	charge := usdOrFatal(t, "1.25")

	net, err := Add(charge, refund)
	if err != nil {
		t.Fatal(err)
	}
	if net != 0 {
		t.Fatalf("a charge and its refund net to %s, want $0", FormatUSD(net))
	}

	// Scaling a refund keeps it negative.
	scaled, err := Mul(refund, 4)
	if err != nil {
		t.Fatal(err)
	}
	if want := usdOrFatal(t, "-5.00"); scaled != want {
		t.Fatalf("4 x %s = %s, want %s", FormatUSD(refund), FormatUSD(scaled), FormatUSD(want))
	}

	// A negative amount must survive a format/parse round trip with its sign.
	if back := usdOrFatal(t, FormatUSD(scaled)); back != scaled {
		t.Fatalf("round trip lost the sign: %d -> %q -> %d", scaled, FormatUSD(scaled), back)
	}

	// But a negative *price* is a config error, and both entry points refuse it
	// rather than producing a negative bill.
	if _, err := PerToken(refund); err == nil {
		t.Fatal("PerToken accepted a negative price")
	}
	if _, err := Cost(charge, -1); err == nil {
		t.Fatal("Cost accepted a negative token count")
	}
}

func TestCostDelegatesToMul(t *testing.T) {
	tests := []struct {
		name     string
		perToken Pico
		tokens   int64
		want     Pico
		wantErr  bool
	}{
		{name: "a realistic charge", perToken: 150_000, tokens: 4096, want: 614_400_000},
		{name: "no tokens is free", perToken: 150_000, tokens: 0, want: 0},
		{name: "a free model", perToken: 0, tokens: 1_000_000, want: 0},
		{name: "one token at the cheapest representable rate", perToken: 1, tokens: 1, want: 1},
		{name: "negative token count", perToken: 150_000, tokens: -1, wantErr: true},
		{name: "an absurd token count overflows", perToken: 10_000_000, tokens: 1 << 45, wantErr: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := Cost(tc.perToken, tc.tokens)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("Cost(%d, %d) = %d, want an error", tc.perToken, tc.tokens, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("Cost(%d, %d): %v", tc.perToken, tc.tokens, err)
			}
			if got != tc.want {
				t.Fatalf("Cost(%d, %d) = %d, want %d", tc.perToken, tc.tokens, got, tc.want)
			}
		})
	}
}

func BenchmarkCost(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := Cost(150_000, 4096); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFormatUSD(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = FormatUSD(1_123_456_789_012)
	}
}

// TestScaleByBasisPointsNoOverflow pins the overflow class that an audit found
// silently breaking budget degradation. A picodollar is 1e-12 USD, so the obvious
// `amount * bps / 10000` wraps int64 for any amount above roughly $922 — and it
// wrapped NEGATIVE, so a $5,000 budget's 80% soft threshold computed as $310.65
// and degradation fired at 6% of the intended spend.
func TestScaleByBasisPointsNoOverflow(t *testing.T) {
	for _, usd := range []int64{1, 100, 922, 1000, 5000, 100_000, 9_000_000} {
		amount := USD(usd)
		got := ScaleByBasisPoints(amount, 8000)
		want := USD(usd) / 10000 * 8000 // exact for whole-dollar inputs
		if got != want {
			t.Errorf("ScaleByBasisPoints($%d, 8000) = %s, want %s",
				usd, FormatUSD(got), FormatUSD(want))
		}
		if got < 0 {
			t.Errorf("ScaleByBasisPoints($%d, 8000) went NEGATIVE (%d): int64 wrap", usd, int64(got))
		}
	}
	// Boundary behaviours.
	if got := ScaleByBasisPoints(USD(10), 0); got != 0 {
		t.Errorf("0 bps should scale to 0, got %s", FormatUSD(got))
	}
	if got := ScaleByBasisPoints(USD(10), 10000); got != USD(10) {
		t.Errorf("10000 bps should be the identity, got %s", FormatUSD(got))
	}
	// Exactness: no truncation beyond the single final division. 10000 pico at
	// 3333 bps is exactly 3333 pico.
	if got := ScaleByBasisPoints(Pico(10000), 3333); got != Pico(3333) {
		t.Errorf("ScaleByBasisPoints(10000 pico, 3333) = %d, want 3333 (precision lost)", int64(got))
	}
	// Negatives scale symmetrically.
	if got := ScaleByBasisPoints(USD(-1000), 5000); got != USD(-500) {
		t.Errorf("negative scaling = %s, want %s", FormatUSD(got), FormatUSD(USD(-500)))
	}
}

// TestFormatMinInt64IsWellFormed pins a formatting bug an audit found: the usual
// `v = -v` leaves math.MinInt64 negative, so both the whole and fractional parts
// rendered with their own minus signs — "-$-9223372.-36854775808". That value
// reaches users through Entry.CostUSD.
func TestFormatMinInt64IsWellFormed(t *testing.T) {
	got := FormatUSD(minPico)
	if strings.Count(got, "-") != 1 {
		t.Errorf("FormatUSD(MinInt64) = %q: exactly one minus sign expected, digits must not carry their own", got)
	}
	if strings.Contains(got, ".-") || strings.Contains(got, "$-") {
		t.Errorf("FormatUSD(MinInt64) = %q: malformed sign placement", got)
	}
	prec := FormatUSDPrec(minPico, 2)
	if strings.Count(prec, "-") != 1 || strings.Contains(prec, ".-") {
		t.Errorf("FormatUSDPrec(MinInt64, 2) = %q: malformed", prec)
	}
}
