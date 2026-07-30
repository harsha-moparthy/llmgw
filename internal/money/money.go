// Package money is integer-only currency arithmetic for the gateway's cost
// accounting.
//
// The spec this project answers requires cost accounting that "reconciles
// exactly against mock-provider logs". Exactly is a strong word and it rules
// out float64: 0.1+0.2 != 0.3 in binary floating point, and a gateway that
// accumulates a fraction of a cent of error per request has no defence when a
// customer disputes an invoice. So every monetary value in this codebase is an
// int64 count of picodollars, and the only division that happens is division
// that has been proven exact at config-load time (see pricing.Validate).
package money

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// Pico is a signed count of picodollars (1e-12 USD).
//
// Why pico and not nano or micro: list prices are quoted per million tokens,
// so the natural unit of a price is "per token = price_per_million / 1e6".
// A representative price of $0.15 per 1M tokens is 0.15 picodollars... no:
// it is 150000 picodollars per token, an exact integer. At nanodollar
// resolution the same price is 0.15 nanodollars per token, which is not an
// integer, and truncating it to 0 would make the cheapest models free. Pico is
// the coarsest unit at which every realistic list price divides exactly.
//
// Range: int64 holds +/-9.22e18 pico, i.e. about +/-$9.2 million. That is
// ample for this project, and Mul below refuses to silently wrap past it.
type Pico int64

// PicoPerUSD is the number of picodollars in one US dollar.
const PicoPerUSD = 1_000_000_000_000

// PicoPerCent is the number of picodollars in one US cent.
const PicoPerCent = PicoPerUSD / 100

// TokensPerPriceUnit is the token count that list prices are quoted against.
// Providers quote "$X per 1M tokens", so a price divided by this constant is a
// per-token price.
const TokensPerPriceUnit = 1_000_000

// minPico is the most negative representable amount. It has no positive
// counterpart, so several formatting paths need it called out explicitly.
const minPico = Pico(math.MinInt64)

// ErrOverflow is returned when an operation would exceed the int64 range.
// It is returned rather than panicking so that a hostile request (a caller
// asking for 2^40 tokens) degrades to a 400 instead of taking the process down.
var ErrOverflow = errors.New("money: int64 overflow")

// USD converts whole dollars to Pico. Intended for constants in tests and
// config defaults, not for parsing untrusted input — use ParseUSD for that.
func USD(d int64) Pico { return Pico(d) * PicoPerUSD }

// Cents converts whole cents to Pico.
func Cents(c int64) Pico { return Pico(c) * PicoPerCent }

// Mul multiplies a Pico amount by a non-negative integer count, reporting
// overflow rather than wrapping.
//
// This is the operation that turns a per-token price into a request cost, so it
// is on the hot path of every single request and it is the one place where a
// silent wrap would produce a negative bill.
func Mul(p Pico, n int64) (Pico, error) {
	if n < 0 {
		return 0, fmt.Errorf("money: negative multiplier %d", n)
	}
	if p == 0 || n == 0 {
		return 0, nil
	}
	// Overflow check by division: exact for integers, and cheaper than
	// promoting to big.Int on a path this hot.
	limit := int64(1) << 62
	if p > 0 && int64(p) > limit/n {
		return 0, ErrOverflow
	}
	if p < 0 && -int64(p) > limit/n {
		return 0, ErrOverflow
	}
	return p * Pico(n), nil
}

// Add sums two amounts, reporting overflow rather than wrapping.
func Add(a, b Pico) (Pico, error) {
	s := a + b
	// Overflow happened iff the operands share a sign that the sum does not.
	if (a > 0 && b > 0 && s < 0) || (a < 0 && b < 0 && s >= 0) {
		return 0, ErrOverflow
	}
	return s, nil
}

// PerToken converts a price quoted per million tokens into a per-token price.
//
// It fails rather than rounding when the division is not exact. Refusing
// inexact prices at load time is what lets the rest of the codebase treat cost
// arithmetic as exact: there is no rounding mode to argue about downstream,
// because no rounding ever occurs. A price with more than six decimal places
// of precision in dollars is rejected, and the operator is told to express it
// differently rather than being silently charged a rounded version of it.
func PerToken(perMillion Pico) (Pico, error) {
	if perMillion < 0 {
		return 0, fmt.Errorf("money: negative price %d", int64(perMillion))
	}
	if perMillion%TokensPerPriceUnit != 0 {
		return 0, fmt.Errorf(
			"money: price %s per 1M tokens is not exactly representable per token "+
				"(%d pico %% %d = %d); express the price with at most 6 decimal places of a dollar",
			FormatUSD(perMillion), int64(perMillion), TokensPerPriceUnit,
			int64(perMillion)%TokensPerPriceUnit,
		)
	}
	return perMillion / TokensPerPriceUnit, nil
}

// Cost returns the cost of n tokens at a per-token price.
func Cost(perToken Pico, n int64) (Pico, error) { return Mul(perToken, n) }

// FormatUSD renders an amount as a dollar string with full picodollar
// precision, trailing zeros trimmed. Round-trips through ParseUSD.
//
// Cost figures in this project routinely land far below a cent (a 20-token
// completion on a cheap model costs about $0.000003), so a 2-decimal
// presentation would print every interesting number as "$0.00". This formatter
// never lies about magnitude: it either prints the exact value or, for the
// aggregate views, callers ask for FormatUSDPrec explicitly.
func FormatUSD(p Pico) string {
	// math.MinInt64 has no positive counterpart, so the usual `v = -v` leaves it
	// negative and both the whole and fractional parts then render with their own
	// minus signs — "-$-9223372.-36854775808", which is not parseable and reaches
	// users through Entry.CostUSD. Handled explicitly rather than left to produce
	// nonsense.
	if p == minPico {
		return "-$9223372.036854775808"
	}
	neg := p < 0
	v := int64(p)
	if neg {
		v = -v
	}
	whole := v / PicoPerUSD
	frac := v % PicoPerUSD
	var sb strings.Builder
	if neg {
		sb.WriteByte('-')
	}
	sb.WriteByte('$')
	sb.WriteString(strconv.FormatInt(whole, 10))
	if frac == 0 {
		return sb.String()
	}
	// 12 fractional digits, zero-padded, then trimmed.
	fs := strconv.FormatInt(frac, 10)
	fs = strings.Repeat("0", 12-len(fs)) + fs
	fs = strings.TrimRight(fs, "0")
	sb.WriteByte('.')
	sb.WriteString(fs)
	return sb.String()
}

// FormatUSDPrec renders an amount with a fixed number of decimal places,
// rounding half away from zero. Used for human-facing rollups only; never for
// anything that is reconciled.
func FormatUSDPrec(p Pico, decimals int) string {
	if decimals < 0 {
		decimals = 0
	}
	if decimals > 12 {
		decimals = 12
	}
	if p == minPico {
		// See FormatUSD: MinInt64 cannot be negated. Format the neighbouring
		// value, which differs by one picodollar — far below any precision this
		// function is asked for — rather than emitting malformed digits.
		p = minPico + 1
	}
	neg := p < 0
	v := int64(p)
	if neg {
		v = -v
	}
	// Scale down to the requested precision with half-up rounding.
	div := int64(1)
	for i := 0; i < 12-decimals; i++ {
		div *= 10
	}
	scaled := v / div
	if div > 1 && v%div >= div/2 {
		scaled++
	}
	unit := int64(1)
	for i := 0; i < decimals; i++ {
		unit *= 10
	}
	whole := scaled / unit
	frac := scaled % unit
	var sb strings.Builder
	if neg && (whole != 0 || frac != 0) {
		sb.WriteByte('-')
	}
	sb.WriteByte('$')
	sb.WriteString(strconv.FormatInt(whole, 10))
	if decimals > 0 {
		fs := strconv.FormatInt(frac, 10)
		fs = strings.Repeat("0", decimals-len(fs)) + fs
		sb.WriteByte('.')
		sb.WriteString(fs)
	}
	return sb.String()
}

// ParseUSD parses a dollar amount such as "1.50", "$0.000015", or "-2" into
// Pico. It accepts at most 12 decimal places and rejects anything finer rather
// than rounding, for the same reason PerToken does.
func ParseUSD(s string) (Pico, error) {
	t := strings.TrimSpace(s)
	if t == "" {
		return 0, errors.New("money: empty amount")
	}
	neg := false
	switch t[0] {
	case '-':
		neg = true
		t = t[1:]
	case '+':
		t = t[1:]
	}
	t = strings.TrimPrefix(t, "$")
	if t == "" {
		return 0, fmt.Errorf("money: %q has no digits", s)
	}
	intPart, fracPart, hasFrac := strings.Cut(t, ".")
	if intPart == "" {
		intPart = "0"
	}
	if hasFrac && fracPart == "" {
		return 0, fmt.Errorf("money: %q has a trailing decimal point", s)
	}
	if len(fracPart) > 12 {
		return 0, fmt.Errorf(
			"money: %q has %d decimal places; at most 12 (picodollar) are representable",
			s, len(fracPart))
	}
	whole, err := strconv.ParseInt(intPart, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("money: bad integer part in %q: %w", s, err)
	}
	if whole > (1<<62)/PicoPerUSD {
		return 0, ErrOverflow
	}
	total := whole * PicoPerUSD
	if hasFrac {
		padded := fracPart + strings.Repeat("0", 12-len(fracPart))
		frac, err := strconv.ParseInt(padded, 10, 64)
		if err != nil {
			return 0, fmt.Errorf("money: bad fractional part in %q: %w", s, err)
		}
		total += frac
	}
	if neg {
		total = -total
	}
	return Pico(total), nil
}

// ScaleByBasisPoints returns p * bps / 10000, computed without overflowing.
//
// This exists because the obvious expression — int64(p) * int64(bps) / 10000 —
// overflows int64 for entirely realistic inputs, and does so SILENTLY. A
// picodollar is 1e-12 USD, so $1,000 is 1e15 pico; multiplying that by 8000
// basis points needs 8e18, which is within a factor of ~1.15 of the int64
// ceiling. A $5,000 budget at an 80% soft threshold therefore wrapped to $310
// instead of $4,000, firing budget degradation at 6% of the intended spend.
//
// Dividing first would avoid the overflow but truncate: (p/10000)*bps loses up
// to 9,999 pico of precision per call, and this codebase's whole premise is that
// money arithmetic is exact. So the computation is split into a quotient part
// that cannot overflow and a remainder part small enough that scaling it is
// always safe — which is exact for every input, with no wrap and no truncation
// beyond the single unavoidable final division.
func ScaleByBasisPoints(p Pico, bps int) Pico {
	const denom = 10000
	if bps <= 0 || p == 0 {
		return 0
	}
	if bps >= denom {
		return p
	}
	neg := p < 0
	v := int64(p)
	if neg {
		v = -v
	}
	// v = q*denom + r, so v*bps/denom = q*bps + r*bps/denom exactly.
	// q*bps cannot overflow for any p within int64 because q <= v/10000, and
	// r*bps <= 9999*9999 which is trivially small.
	q, r := v/denom, v%denom
	out := q*int64(bps) + r*int64(bps)/denom
	if neg {
		out = -out
	}
	return Pico(out)
}
