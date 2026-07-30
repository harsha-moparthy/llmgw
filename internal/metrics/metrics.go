// Package metrics is a bounded reimplementation of the Prometheus text
// exposition format.
//
// Using the official client_golang would be the conventional choice and this
// package is not an argument that it is wrong. It exists because this project
// has a zero-dependency rule (a gateway sits in the path of every LLM call a
// company makes, so its dependency tree is its attack surface), and because the
// subset of Prometheus a gateway actually needs — counters, gauges, one
// histogram shape — is small enough to implement correctly and test properly.
// What this package deliberately does NOT implement: exemplars, native
// histograms, the protobuf negotiation, collector registration hooks, or
// pushgateway support.
//
// # The quantile honesty problem
//
// A Prometheus histogram quantile is interpolated within a bucket. If the
// buckets around 5ms are [1ms, 5ms, 10ms], then a reported "p99 = 4.8ms" carries
// no information beyond "somewhere between 1 and 5" — the 4.8 is an artifact of
// where the interpolation lands, and moving a bucket boundary would move the
// reported p99 without any change in the system.
//
// This project's headline claim is "added latency under 5ms p99", which sits
// exactly where that artifact would live. So this package provides two things
// and keeps them clearly separate:
//
//   - Histogram, for the /metrics endpoint: cheap, bounded memory, interpolated
//     quantiles. What you scrape and alert on.
//   - Sample, for the benchmark harness: retains every observation and computes
//     exact order statistics. What the README's numbers come from.
//
// Conflating the two is how a project ends up reporting a p99 that is a property
// of its bucket layout. The distinction is stated in the README too.
package metrics

import (
	"fmt"
	"io"
	"math"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
)

// MetricType is the Prometheus type of a metric family.
type MetricType string

// Prometheus metric types this package emits.
const (
	TypeCounter   MetricType = "counter"
	TypeGauge     MetricType = "gauge"
	TypeHistogram MetricType = "histogram"
)

// Labels is a label set. Keys must be valid Prometheus label names; values are
// escaped on output.
type Labels map[string]string

// canonical renders a label set into a stable string used as the map key.
//
// Stability requires sorting: Go's map iteration order is randomised, so
// building the key by ranging over the map would give the same label set a
// different identity on every call and leak one series per observation. That is
// a slow memory leak with a plausible-looking metrics output, which is the worst
// combination.
func (l Labels) canonical() string {
	if len(l) == 0 {
		return ""
	}
	keys := make([]string, 0, len(l))
	for k := range l {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var sb strings.Builder
	for i, k := range keys {
		if i > 0 {
			sb.WriteByte(',')
		}
		sb.WriteString(k)
		sb.WriteByte('=')
		sb.WriteString(l[k])
	}
	return sb.String()
}

func (l Labels) clone() Labels {
	if len(l) == 0 {
		return nil
	}
	out := make(Labels, len(l))
	for k, v := range l {
		out[k] = v
	}
	return out
}

// Counter is a monotonically increasing value.
type Counter struct {
	v atomic.Uint64 // float64 bits, so fractional increments are possible
}

// Inc adds one.
func (c *Counter) Inc() { c.Add(1) }

// Add adds delta, which must be non-negative. A negative delta is ignored
// rather than panicking: a metrics bug must never be able to take down the
// request path it is measuring.
func (c *Counter) Add(delta float64) {
	if delta < 0 || math.IsNaN(delta) {
		return
	}
	for {
		old := c.v.Load()
		next := math.Float64bits(math.Float64frombits(old) + delta)
		if c.v.CompareAndSwap(old, next) {
			return
		}
	}
}

// Value returns the current value.
func (c *Counter) Value() float64 { return math.Float64frombits(c.v.Load()) }

// Gauge is a value that can go up and down.
type Gauge struct {
	v atomic.Uint64
}

// Set replaces the value.
func (g *Gauge) Set(v float64) { g.v.Store(math.Float64bits(v)) }

// Add adds delta, which may be negative.
func (g *Gauge) Add(delta float64) {
	for {
		old := g.v.Load()
		next := math.Float64bits(math.Float64frombits(old) + delta)
		if g.v.CompareAndSwap(old, next) {
			return
		}
	}
}

// Inc and Dec adjust by one, for in-flight counters.
func (g *Gauge) Inc() { g.Add(1) }
func (g *Gauge) Dec() { g.Add(-1) }

// Value returns the current value.
func (g *Gauge) Value() float64 { return math.Float64frombits(g.v.Load()) }

// Histogram is a bucketed distribution.
//
// Buckets are cumulative on output (Prometheus requires le-cumulative counts)
// but stored per-bucket, so an observation is one atomic add rather than a walk
// over every bucket above it.
type Histogram struct {
	bounds  []float64 // upper bounds, ascending, excluding +Inf
	buckets []atomic.Uint64
	inf     atomic.Uint64
	sum     atomic.Uint64 // float64 bits
	count   atomic.Uint64
}

// DefaultLatencyBucketsSeconds spans the range a gateway's own overhead lives in.
//
// The bucket layout is chosen around the claim being made. Because the target is
// "under 5ms p99", the resolution is deliberately dense between 100us and 10ms:
// a coarse layout there would make the interpolated p99 an artifact of the
// boundaries rather than a measurement. Note that even with this layout the
// exact figure in the README comes from Sample, not from here.
var DefaultLatencyBucketsSeconds = []float64{
	0.000_05, 0.000_1, 0.000_25, 0.000_5,
	0.001, 0.002, 0.003, 0.004, 0.005, 0.0075,
	0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60,
}

// DefaultTokenBuckets spans realistic prompt and completion sizes.
var DefaultTokenBuckets = []float64{
	16, 64, 256, 1024, 4096, 16384, 65536, 262144,
}

// NewHistogram returns a Histogram with the given upper bounds. The bounds are
// sorted and de-duplicated; +Inf is implicit.
func NewHistogram(bounds []float64) *Histogram {
	b := make([]float64, 0, len(bounds))
	seen := make(map[float64]struct{}, len(bounds))
	for _, v := range bounds {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			continue
		}
		if _, dup := seen[v]; dup {
			continue
		}
		seen[v] = struct{}{}
		b = append(b, v)
	}
	sort.Float64s(b)
	return &Histogram{bounds: b, buckets: make([]atomic.Uint64, len(b))}
}

// Observe records a value.
func (h *Histogram) Observe(v float64) {
	if math.IsNaN(v) {
		return
	}
	// Binary search for the first bound >= v. Linear scan would be fine at 22
	// buckets, but Observe is called on every request and every provider call,
	// so the log factor is free to take.
	i := sort.SearchFloat64s(h.bounds, v)
	if i < len(h.buckets) {
		h.buckets[i].Add(1)
	} else {
		h.inf.Add(1)
	}
	h.count.Add(1)
	for {
		old := h.sum.Load()
		next := math.Float64bits(math.Float64frombits(old) + v)
		if h.sum.CompareAndSwap(old, next) {
			break
		}
	}
}

// Count returns the number of observations.
func (h *Histogram) Count() uint64 { return h.count.Load() }

// Sum returns the sum of observations.
func (h *Histogram) Sum() float64 { return math.Float64frombits(h.sum.Load()) }

// cumulative returns the le-cumulative bucket counts plus the +Inf total.
func (h *Histogram) cumulative() ([]uint64, uint64) {
	out := make([]uint64, len(h.buckets))
	var running uint64
	for i := range h.buckets {
		running += h.buckets[i].Load()
		out[i] = running
	}
	return out, running + h.inf.Load()
}

// Quantile returns an INTERPOLATED quantile, with the same linear-interpolation
// rule Prometheus uses.
//
// The name is deliberately not "P99": the value is approximate, bounded by the
// surrounding bucket edges, and must not be used for a headline claim. Use
// Sample for that. This method exists so a dashboard can plot something.
func (h *Histogram) Quantile(q float64) float64 {
	if q <= 0 {
		q = 0
	}
	if q >= 1 {
		q = 1
	}
	cum, total := h.cumulative()
	if total == 0 {
		return 0
	}
	want := q * float64(total)
	prevCount := uint64(0)
	prevBound := 0.0
	for i, c := range cum {
		if float64(c) >= want {
			// Linear interpolation within [prevBound, bounds[i]].
			span := h.bounds[i] - prevBound
			inBucket := float64(c - prevCount)
			if inBucket <= 0 {
				return h.bounds[i]
			}
			frac := (want - float64(prevCount)) / inBucket
			return prevBound + span*frac
		}
		prevCount = c
		prevBound = h.bounds[i]
	}
	// Everything at or above the top bound: the only honest answer is the top
	// bound, since the +Inf bucket carries no magnitude information at all.
	if len(h.bounds) > 0 {
		return h.bounds[len(h.bounds)-1]
	}
	return 0
}

// Sample retains every observation and computes exact order statistics.
//
// This is the recorder behind every latency number this project reports. It is
// not exposed on /metrics — unbounded retention is fine for a benchmark run and
// unacceptable for a long-lived process, and the distinction is the whole reason
// both types exist.
type Sample struct {
	mu sync.Mutex
	xs []float64
}

// NewSample returns a Sample with capacity hint n.
func NewSample(n int) *Sample {
	if n < 0 {
		n = 0
	}
	return &Sample{xs: make([]float64, 0, n)}
}

// Observe records a value.
func (s *Sample) Observe(v float64) {
	if math.IsNaN(v) {
		return
	}
	s.mu.Lock()
	s.xs = append(s.xs, v)
	s.mu.Unlock()
}

// Len returns the number of observations.
func (s *Sample) Len() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.xs)
}

// Quantile returns the exact q-quantile by nearest-rank on the sorted sample.
//
// Nearest-rank means the returned value is one that was actually observed, not
// an interpolation between two. The convention is stated because "p99" differs
// between tools by a fraction of a percent, and a claim sitting near a threshold
// should not turn on an undocumented interpolation rule.
func (s *Sample) Quantile(q float64) float64 {
	s.mu.Lock()
	xs := make([]float64, len(s.xs))
	copy(xs, s.xs)
	s.mu.Unlock()
	if len(xs) == 0 {
		return 0
	}
	sort.Float64s(xs)
	if q <= 0 {
		return xs[0]
	}
	if q >= 1 {
		return xs[len(xs)-1]
	}
	rank := int(math.Ceil(q * float64(len(xs))))
	if rank < 1 {
		rank = 1
	}
	if rank > len(xs) {
		rank = len(xs)
	}
	return xs[rank-1]
}

// Mean returns the arithmetic mean.
func (s *Sample) Mean() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.xs) == 0 {
		return 0
	}
	var sum float64
	for _, v := range s.xs {
		sum += v
	}
	return sum / float64(len(s.xs))
}

// Reset discards all observations.
func (s *Sample) Reset() {
	s.mu.Lock()
	s.xs = s.xs[:0]
	s.mu.Unlock()
}

// family is one metric name with its help text, type, and label-keyed children.
type family struct {
	name string
	help string
	typ  MetricType

	// buckets is the histogram layout for TypeHistogram families.
	buckets []float64

	mu       sync.RWMutex
	labelSet map[string]*child
	// order preserves first-seen order so the exposition output is stable
	// across scrapes. Stable output makes a diff of two scrapes readable, which
	// is worth one slice.
	order []string
}

type child struct {
	labels Labels
	ctr    *Counter
	gauge  *Gauge
	hist   *Histogram
}

// Registry holds metric families and renders the exposition format.
type Registry struct {
	mu       sync.RWMutex
	families map[string]*family
	order    []string
}

// NewRegistry returns an empty Registry.
func NewRegistry() *Registry {
	return &Registry{families: make(map[string]*family)}
}

func (r *Registry) family(name, help string, typ MetricType, buckets []float64) *family {
	r.mu.RLock()
	f, ok := r.families[name]
	r.mu.RUnlock()
	if ok {
		return f
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if f, ok = r.families[name]; ok {
		return f
	}
	f = &family{
		name: name, help: help, typ: typ, buckets: buckets,
		labelSet: make(map[string]*child),
	}
	r.families[name] = f
	r.order = append(r.order, name)
	return f
}

func (f *family) child(l Labels) *child {
	key := l.canonical()
	f.mu.RLock()
	c, ok := f.labelSet[key]
	f.mu.RUnlock()
	if ok {
		return c
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if c, ok = f.labelSet[key]; ok {
		return c
	}
	c = &child{labels: l.clone()}
	switch f.typ {
	case TypeCounter:
		c.ctr = &Counter{}
	case TypeGauge:
		c.gauge = &Gauge{}
	case TypeHistogram:
		c.hist = NewHistogram(f.buckets)
	}
	f.labelSet[key] = c
	f.order = append(f.order, key)
	return c
}

// CounterVec is a pre-resolved handle to a counter family.
//
// The two-step API (resolve a label set once, then Inc on the handle) exists
// because the naive alternative — reg.Counter("name", Labels{...}).Inc() on
// every request — does a map allocation, a sort, and a string concatenation per
// observation, on the hot path, purely to look up a pointer that never changes.
// Pre-resolving turns the steady-state cost into a single atomic add.
type CounterVec struct{ f *family }

// Counter returns a counter family handle.
func (r *Registry) Counter(name, help string) *CounterVec {
	return &CounterVec{f: r.family(name, help, TypeCounter, nil)}
}

// With resolves a label set to a Counter.
func (cv *CounterVec) With(l Labels) *Counter { return cv.f.child(l).ctr }

// GaugeVec is a pre-resolved handle to a gauge family.
type GaugeVec struct{ f *family }

// Gauge returns a gauge family handle.
func (r *Registry) Gauge(name, help string) *GaugeVec {
	return &GaugeVec{f: r.family(name, help, TypeGauge, nil)}
}

// With resolves a label set to a Gauge.
func (gv *GaugeVec) With(l Labels) *Gauge { return gv.f.child(l).gauge }

// HistogramVec is a pre-resolved handle to a histogram family.
type HistogramVec struct{ f *family }

// Histogram returns a histogram family handle with the given buckets.
func (r *Registry) Histogram(name, help string, buckets []float64) *HistogramVec {
	return &HistogramVec{f: r.family(name, help, TypeHistogram, buckets)}
}

// With resolves a label set to a Histogram.
func (hv *HistogramVec) With(l Labels) *Histogram { return hv.f.child(l).hist }

// WriteTo renders the exposition format.
//
// Ordering is first-seen for both families and label sets, so two scrapes of an
// unchanged process produce byte-identical output and a diff shows only what
// actually moved.
func (r *Registry) WriteTo(w io.Writer) (int64, error) {
	var sb strings.Builder
	r.mu.RLock()
	names := make([]string, len(r.order))
	copy(names, r.order)
	fams := make([]*family, 0, len(names))
	for _, n := range names {
		fams = append(fams, r.families[n])
	}
	r.mu.RUnlock()

	for _, f := range fams {
		f.mu.RLock()
		keys := make([]string, len(f.order))
		copy(keys, f.order)
		children := make([]*child, 0, len(keys))
		for _, k := range keys {
			children = append(children, f.labelSet[k])
		}
		f.mu.RUnlock()

		if len(children) == 0 {
			continue
		}
		if f.help != "" {
			sb.WriteString("# HELP ")
			sb.WriteString(f.name)
			sb.WriteByte(' ')
			sb.WriteString(escapeHelp(f.help))
			sb.WriteByte('\n')
		}
		sb.WriteString("# TYPE ")
		sb.WriteString(f.name)
		sb.WriteByte(' ')
		sb.WriteString(string(f.typ))
		sb.WriteByte('\n')

		for _, c := range children {
			switch f.typ {
			case TypeCounter:
				writeSample(&sb, f.name, c.labels, nil, c.ctr.Value())
			case TypeGauge:
				writeSample(&sb, f.name, c.labels, nil, c.gauge.Value())
			case TypeHistogram:
				cum, total := c.hist.cumulative()
				for i, bound := range c.hist.bounds {
					writeSample(&sb, f.name+"_bucket", c.labels,
						map[string]string{"le": formatFloat(bound)}, float64(cum[i]))
				}
				writeSample(&sb, f.name+"_bucket", c.labels,
					map[string]string{"le": "+Inf"}, float64(total))
				writeSample(&sb, f.name+"_sum", c.labels, nil, c.hist.Sum())
				writeSample(&sb, f.name+"_count", c.labels, nil, float64(total))
			}
		}
	}
	n, err := io.WriteString(w, sb.String())
	return int64(n), err
}

func writeSample(sb *strings.Builder, name string, labels Labels, extra map[string]string, v float64) {
	sb.WriteString(name)
	if len(labels) > 0 || len(extra) > 0 {
		sb.WriteByte('{')
		first := true
		// Sorted for stable output.
		keys := make([]string, 0, len(labels))
		for k := range labels {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			if !first {
				sb.WriteByte(',')
			}
			first = false
			sb.WriteString(k)
			sb.WriteString(`="`)
			sb.WriteString(escapeLabelValue(labels[k]))
			sb.WriteByte('"')
		}
		ekeys := make([]string, 0, len(extra))
		for k := range extra {
			ekeys = append(ekeys, k)
		}
		sort.Strings(ekeys)
		for _, k := range ekeys {
			if !first {
				sb.WriteByte(',')
			}
			first = false
			sb.WriteString(k)
			sb.WriteString(`="`)
			sb.WriteString(escapeLabelValue(extra[k]))
			sb.WriteByte('"')
		}
		sb.WriteByte('}')
	}
	sb.WriteByte(' ')
	sb.WriteString(formatFloat(v))
	sb.WriteByte('\n')
}

// escapeLabelValue escapes a label value per the exposition format.
//
// This is a security boundary, not a cosmetic one. Label values here include
// model names and provider names that originate in a client request, and an
// unescaped newline or quote would let a caller inject additional metric lines
// into the scrape output — corrupting a monitoring system through a chat
// completion request. TestHostileLabelValue pins it.
func escapeLabelValue(s string) string {
	if !strings.ContainsAny(s, "\\\"\n\r") {
		return s
	}
	var sb strings.Builder
	sb.Grow(len(s) + 8)
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '\\':
			sb.WriteString(`\\`)
		case '"':
			sb.WriteString(`\"`)
		case '\n':
			sb.WriteString(`\n`)
		case '\r':
			// Not in the spec's escape list; dropped rather than passed through,
			// since a bare CR can still break a line-oriented parser.
			sb.WriteString(`\r`)
		default:
			sb.WriteByte(s[i])
		}
	}
	return sb.String()
}

func escapeHelp(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	s = strings.ReplaceAll(s, "\n", `\n`)
	return strings.ReplaceAll(s, "\r", `\r`)
}

// formatFloat renders a value the way Prometheus expects, including the special
// values.
func formatFloat(v float64) string {
	switch {
	case math.IsInf(v, 1):
		return "+Inf"
	case math.IsInf(v, -1):
		return "-Inf"
	case math.IsNaN(v):
		return "NaN"
	}
	// An integral value is written without a decimal point, which is what every
	// Prometheus client does and what makes counter output readable.
	if v == math.Trunc(v) && math.Abs(v) < 1e15 {
		return strconv.FormatInt(int64(v), 10)
	}
	return strconv.FormatFloat(v, 'g', -1, 64)
}

// String renders the exposition format to a string.
func (r *Registry) String() string {
	var sb strings.Builder
	_, _ = r.WriteTo(&sb)
	return sb.String()
}

// ServeHTTP implements http.Handler for the /metrics endpoint.
func (r *Registry) ServeHTTP(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	_, _ = r.WriteTo(w)
}

// Validate reports label names that Prometheus would reject, so a typo is
// caught at wiring time rather than corrupting a scrape.
func Validate(name string, l Labels) error {
	if !validMetricName(name) {
		return fmt.Errorf("metrics: %q is not a valid metric name", name)
	}
	for k := range l {
		if !validLabelName(k) {
			return fmt.Errorf("metrics: %q is not a valid label name", k)
		}
	}
	return nil
}

func validMetricName(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		ok := c == '_' || c == ':' ||
			(c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
			(i > 0 && c >= '0' && c <= '9')
		if !ok {
			return false
		}
	}
	return true
}

func validLabelName(s string) bool {
	if s == "" {
		return false
	}
	// The __ prefix is reserved for Prometheus internals.
	if strings.HasPrefix(s, "__") {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		ok := c == '_' ||
			(c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
			(i > 0 && c >= '0' && c <= '9')
		if !ok {
			return false
		}
	}
	return true
}
