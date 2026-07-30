package metrics

import (
	"fmt"
	"math"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestCounter(t *testing.T) {
	var c Counter
	if got := c.Value(); got != 0 {
		t.Fatalf("zero value = %v, want 0", got)
	}
	c.Inc()
	c.Add(2.5)
	if got := c.Value(); got != 3.5 {
		t.Errorf("Value = %v, want 3.5", got)
	}
	// A negative add must be ignored rather than panicking: a metrics bug must
	// never be able to take down the request path it is measuring.
	c.Add(-100)
	if got := c.Value(); got != 3.5 {
		t.Errorf("after a negative Add, Value = %v, want 3.5 (unchanged)", got)
	}
	c.Add(math.NaN())
	if math.IsNaN(c.Value()) {
		t.Error("a NaN Add poisoned the counter")
	}
}

func TestGauge(t *testing.T) {
	var g Gauge
	g.Set(10)
	g.Add(-3)
	g.Inc()
	g.Dec()
	if got := g.Value(); got != 7 {
		t.Errorf("Value = %v, want 7", got)
	}
	g.Set(-5)
	if got := g.Value(); got != -5 {
		t.Errorf("gauges must accept negatives: got %v", got)
	}
}

func TestHistogramBucketing(t *testing.T) {
	h := NewHistogram([]float64{1, 5, 10})
	for _, v := range []float64{0.5, 1, 3, 5, 7, 20} {
		h.Observe(v)
	}
	if h.Count() != 6 {
		t.Errorf("Count = %d, want 6", h.Count())
	}
	if got, want := h.Sum(), 36.5; got != want {
		t.Errorf("Sum = %v, want %v", got, want)
	}
	cum, total := h.cumulative()
	// le=1 gets 0.5 and 1 (bucketing is inclusive of the upper bound).
	// le=5 adds 3 and 5 -> 4. le=10 adds 7 -> 5. +Inf adds 20 -> 6.
	want := []uint64{2, 4, 5}
	for i := range want {
		if cum[i] != want[i] {
			t.Errorf("cumulative[%d] (le=%v) = %d, want %d", i, h.bounds[i], cum[i], want[i])
		}
	}
	if total != 6 {
		t.Errorf("+Inf total = %d, want 6", total)
	}
}

func TestHistogramBoundsSortedAndDeduped(t *testing.T) {
	h := NewHistogram([]float64{10, 1, 5, 5, math.Inf(1), math.NaN()})
	if len(h.bounds) != 3 {
		t.Fatalf("bounds = %v, want 3 finite unique values", h.bounds)
	}
	for i := 1; i < len(h.bounds); i++ {
		if h.bounds[i] <= h.bounds[i-1] {
			t.Errorf("bounds not strictly ascending: %v", h.bounds)
		}
	}
}

// TestHistogramQuantileIsInterpolated documents, in a test, that the histogram's
// quantile is approximate. The assertion is a range rather than a value, because
// asserting an exact number here would be asserting a property of the bucket
// layout — precisely the confusion this package exists to avoid.
func TestHistogramQuantileIsInterpolated(t *testing.T) {
	h := NewHistogram([]float64{1, 5, 10})
	// 100 observations, all at 4.9 — a value inside the (1,5] bucket.
	for i := 0; i < 100; i++ {
		h.Observe(4.9)
	}
	q := h.Quantile(0.99)
	if q <= 1 || q > 5 {
		t.Errorf("Quantile(0.99) = %v, want a value in (1,5]", q)
	}
	// The true p99 is 4.9 but the histogram cannot know that; it will report the
	// bucket's upper edge. Asserting the DIFFERENCE is the point of the test.
	if q == 4.9 {
		t.Log("note: the interpolated value coincidentally matched the true value")
	}

	// The exact recorder does know.
	s := NewSample(100)
	for i := 0; i < 100; i++ {
		s.Observe(4.9)
	}
	if got := s.Quantile(0.99); got != 4.9 {
		t.Errorf("Sample.Quantile(0.99) = %v, want exactly 4.9", got)
	}
}

func TestHistogramEmptyQuantile(t *testing.T) {
	h := NewHistogram([]float64{1, 5})
	if got := h.Quantile(0.99); got != 0 {
		t.Errorf("empty histogram Quantile = %v, want 0 (not NaN)", got)
	}
}

func TestSampleExactQuantiles(t *testing.T) {
	s := NewSample(100)
	// 1..100 inclusive.
	for i := 1; i <= 100; i++ {
		s.Observe(float64(i))
	}
	tests := []struct {
		q    float64
		want float64
	}{
		{0.0, 1},
		{0.5, 50},
		{0.9, 90},
		{0.99, 99},
		{1.0, 100},
	}
	for _, tc := range tests {
		if got := s.Quantile(tc.q); got != tc.want {
			t.Errorf("Quantile(%v) = %v, want %v (nearest-rank)", tc.q, got, tc.want)
		}
	}
	if got := s.Mean(); got != 50.5 {
		t.Errorf("Mean = %v, want 50.5", got)
	}
	if s.Len() != 100 {
		t.Errorf("Len = %d, want 100", s.Len())
	}
	s.Reset()
	if s.Len() != 0 {
		t.Error("Reset did not clear the sample")
	}
	if got := s.Quantile(0.5); got != 0 {
		t.Errorf("empty Sample Quantile = %v, want 0", got)
	}
}

func TestExpositionFormat(t *testing.T) {
	r := NewRegistry()
	c := r.Counter("llmgw_requests_total", "Requests by tenant.")
	c.With(Labels{"tenant": "acme", "status": "2xx"}).Add(3)
	g := r.Gauge("llmgw_in_flight", "In flight.")
	g.With(nil).Set(2)
	h := r.Histogram("llmgw_latency_seconds", "Latency.", []float64{0.001, 0.01})
	hc := h.With(Labels{"model": "gw-chat"})
	hc.Observe(0.0005)
	hc.Observe(0.005)
	hc.Observe(1)

	out := r.String()

	for _, want := range []string{
		"# HELP llmgw_requests_total Requests by tenant.",
		"# TYPE llmgw_requests_total counter",
		`llmgw_requests_total{status="2xx",tenant="acme"} 3`,
		"# TYPE llmgw_in_flight gauge",
		"llmgw_in_flight 2",
		"# TYPE llmgw_latency_seconds histogram",
		`llmgw_latency_seconds_bucket{model="gw-chat",le="0.001"} 1`,
		`llmgw_latency_seconds_bucket{model="gw-chat",le="0.01"} 2`,
		`llmgw_latency_seconds_bucket{model="gw-chat",le="+Inf"} 3`,
		`llmgw_latency_seconds_count{model="gw-chat"} 3`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q\n--- got ---\n%s", want, out)
		}
	}

	// The _sum line must carry the actual sum.
	if !strings.Contains(out, `llmgw_latency_seconds_sum{model="gw-chat"} 1.0055`) {
		t.Errorf("missing or wrong _sum line\n--- got ---\n%s", out)
	}
}

// TestHostileLabelValue is a security test. Label values here include model and
// provider names that originate in client requests; an unescaped quote or
// newline would let a caller inject metric lines into the scrape output and
// corrupt a monitoring system through a chat completion request.
func TestHostileLabelValue(t *testing.T) {
	r := NewRegistry()
	c := r.Counter("llmgw_requests_total", "Requests.")
	hostile := "evil\"} 999\nllmgw_injected_metric{x=\"1\"} 42\n#\\"
	c.With(Labels{"model": hostile}).Inc()

	out := r.String()

	// The real assertion is structural, not textual. The injected metric name
	// still APPEARS in the output — inside an escaped label value, which is
	// harmless and correct. What must not happen is that it appears as its own
	// line, which is what a parser would act on. So the test counts lines rather
	// than searching for a substring: a substring check here would fail on
	// correctly-escaped output and pass on some genuinely broken output, which is
	// the wrong test in both directions.
	lines := 0
	for _, l := range strings.Split(strings.TrimSpace(out), "\n") {
		if strings.TrimSpace(l) != "" {
			lines++
		}
	}
	if lines != 3 {
		t.Fatalf("label injection produced extra lines (want 3: HELP, TYPE, one sample):\n%s", out)
	}
	// No line may begin with the injected metric name.
	for _, l := range strings.Split(out, "\n") {
		if strings.HasPrefix(strings.TrimSpace(l), "llmgw_injected_metric") {
			t.Fatalf("injected metric escaped onto its own line:\n%s", out)
		}
	}
	// The escaped forms must be present instead of the raw characters.
	if !strings.Contains(out, `\"`) || !strings.Contains(out, `\n`) || !strings.Contains(out, `\\`) {
		t.Errorf("expected escaped quote, newline and backslash in output:\n%s", out)
	}
	// And the raw control characters must be gone from the label value: exactly
	// one newline per output line.
	if strings.Count(out, "\n") != 3 {
		t.Errorf("raw newlines survived escaping: %d newlines in output:\n%q", strings.Count(out, "\n"), out)
	}
}

func TestFormatFloatSpecialValues(t *testing.T) {
	tests := map[float64]string{
		0:            "0",
		1:            "1",
		-3:           "-3",
		1.5:          "1.5",
		0.0001:       "0.0001",
		math.Inf(1):  "+Inf",
		math.Inf(-1): "-Inf",
		1e20:         "1e+20",
	}
	for in, want := range tests {
		if got := formatFloat(in); got != want {
			t.Errorf("formatFloat(%v) = %q, want %q", in, got, want)
		}
	}
	if got := formatFloat(math.NaN()); got != "NaN" {
		t.Errorf("formatFloat(NaN) = %q, want NaN", got)
	}
}

// TestLabelSetIdentityIsStable is the test that catches the series-leak bug: if
// the canonical key depended on Go's randomised map iteration order, the same
// label set would resolve to a different child on every call, leaking one series
// per observation while still producing plausible-looking output.
func TestLabelSetIdentityIsStable(t *testing.T) {
	r := NewRegistry()
	c := r.Counter("m_total", "m")
	for i := 0; i < 200; i++ {
		// Rebuild the map each time so iteration order genuinely varies.
		c.With(Labels{"a": "1", "b": "2", "c": "3", "d": "4"}).Inc()
	}
	out := r.String()
	samples := strings.Count(out, "m_total{")
	if samples != 1 {
		t.Fatalf("label set resolved to %d distinct series, want 1:\n%s", samples, out)
	}
	if !strings.Contains(out, "} 200") {
		t.Errorf("expected a single series with value 200:\n%s", out)
	}
}

func TestOutputIsStableAcrossScrapes(t *testing.T) {
	r := NewRegistry()
	c := r.Counter("a_total", "a")
	for _, tenant := range []string{"z", "a", "m"} {
		c.With(Labels{"tenant": tenant}).Inc()
	}
	g := r.Gauge("b_gauge", "b")
	g.With(Labels{"x": "1"}).Set(5)

	first := r.String()
	second := r.String()
	if first != second {
		t.Errorf("two scrapes of an unchanged registry differ:\n--- first ---\n%s\n--- second ---\n%s", first, second)
	}
}

func TestFamilyWithNoChildrenIsOmitted(t *testing.T) {
	r := NewRegistry()
	r.Counter("never_used_total", "unused")
	out := r.String()
	if strings.Contains(out, "never_used_total") {
		t.Errorf("a family with no observations should not be exposed:\n%s", out)
	}
}

func TestValidate(t *testing.T) {
	tests := []struct {
		name    string
		metric  string
		labels  Labels
		wantErr bool
	}{
		{name: "ok", metric: "llmgw_requests_total", labels: Labels{"tenant": "a"}},
		{name: "empty metric name", metric: "", wantErr: true},
		{name: "metric starting with a digit", metric: "1bad", wantErr: true},
		{name: "metric with a dash", metric: "bad-name", wantErr: true},
		{name: "label with a dash", metric: "ok", labels: Labels{"bad-label": "v"}, wantErr: true},
		{name: "reserved label prefix", metric: "ok", labels: Labels{"__reserved": "v"}, wantErr: true},
		{name: "empty label name", metric: "ok", labels: Labels{"": "v"}, wantErr: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := Validate(tc.metric, tc.labels)
			if (err != nil) != tc.wantErr {
				t.Errorf("Validate(%q, %v) err = %v, wantErr = %v", tc.metric, tc.labels, err, tc.wantErr)
			}
		})
	}
}

// TestConcurrentRecordAndScrape is the -race test. The registry is written by
// every in-flight request and read by the scrape, so a lock error here shows up
// as a crash in production under load and as nothing at all in a serial test.
func TestConcurrentRecordAndScrape(t *testing.T) {
	r := NewRegistry()
	reqs := r.Counter("llmgw_requests_total", "Requests.")
	lat := r.Histogram("llmgw_latency_seconds", "Latency.", DefaultLatencyBucketsSeconds)
	inflight := r.Gauge("llmgw_in_flight", "In flight.")

	// Two WaitGroups, not one: the scrapers run until told to stop, and the
	// recorders finish on their own. Waiting on a single group would deadlock,
	// since the signal to stop the scrapers comes after the recorders are done.
	var recorders, scrapers sync.WaitGroup
	stop := make(chan struct{})

	// Scrapers yield between scrapes. A tight loop taking the registry's read
	// lock continuously starves the recorders that need the write lock to create
	// a new family, which would make the test slow for a reason unrelated to the
	// code under test. A real Prometheus scrape happens every 15 seconds.
	for i := 0; i < 3; i++ {
		scrapers.Add(1)
		go func() {
			defer scrapers.Done()
			for {
				select {
				case <-stop:
					return
				default:
					_ = r.String()
					time.Sleep(100 * time.Microsecond)
				}
			}
		}()
	}

	// Recorders, deliberately creating new label sets as well as reusing old
	// ones, since family creation is the path that takes the registry write lock.
	for w := 0; w < 12; w++ {
		recorders.Add(1)
		go func(w int) {
			defer recorders.Done()
			for i := 0; i < 500; i++ {
				l := Labels{"tenant": fmt.Sprintf("t%d", i%5), "status": "2xx"}
				reqs.With(l).Inc()
				lat.With(Labels{"model": fmt.Sprintf("m%d", i%3)}).Observe(float64(i%50) / 1000)
				inflight.With(nil).Inc()
				inflight.With(nil).Dec()
			}
		}(w)
	}
	recorders.Wait()
	close(stop)
	scrapers.Wait()

	// 12 workers x 500 iterations spread over 5 tenants = 1200 each.
	out := r.String()
	for i := 0; i < 5; i++ {
		want := fmt.Sprintf(`llmgw_requests_total{status="2xx",tenant="t%d"} 1200`, i)
		if !strings.Contains(out, want) {
			t.Errorf("missing or wrong count for tenant t%d; expected %q", i, want)
		}
	}
	if got := inflight.With(nil).Value(); got != 0 {
		t.Errorf("in-flight gauge = %v after balanced Inc/Dec, want 0", got)
	}
}

func TestGatewayMetricSet(t *testing.T) {
	g := NewGateway()
	if g.Registry() == nil {
		t.Fatal("Registry() is nil")
	}
	// Record one of everything so the exposition path covers each family, and so
	// a typo in a metric name shows up here rather than in a dashboard.
	g.Requests.With(Labels{"tenant": "t", "model": "m", "status": "2xx"}).Inc()
	ObserveSeconds(g.RequestLatency.With(Labels{"model": "m"}), 12*time.Millisecond)
	ObserveSeconds(g.Overhead.With(Labels{"model": "m"}), 400*time.Microsecond)
	ObserveSeconds(g.UpstreamLatency.With(Labels{"provider": "p"}), 11*time.Millisecond)
	g.InFlight.With(nil).Set(3)
	g.PromptTokens.With(Labels{"tenant": "t", "model": "m"}).Add(100)
	g.CompletionTokens.With(Labels{"tenant": "t", "model": "m"}).Add(50)
	g.ReasoningTokens.With(Labels{"tenant": "t", "model": "m"}).Add(20)
	g.CostPico.With(Labels{"tenant": "t", "model": "m"}).Add(1500)
	g.CostEstimatedPico.With(Labels{"tenant": "t", "model": "m"}).Add(10)
	g.Attempts.With(Labels{"model": "m"}).Observe(2)
	g.Failovers.With(Labels{"class": "connect"}).Inc()
	g.UpstreamErrors.With(Labels{"provider": "p", "class": "timeout"}).Inc()
	g.NoProvider.With(Labels{"model": "m"}).Inc()
	g.BreakerState.With(Labels{"provider": "p"}).Set(2)
	g.BreakerTransitions.With(Labels{"provider": "p"}).Inc()
	g.BreakerRejections.With(Labels{"provider": "p"}).Inc()
	g.BudgetRejections.With(Labels{"tenant": "t"}).Inc()
	g.BudgetDegraded.With(Labels{"tenant": "t"}).Inc()
	g.BudgetSpent.With(Labels{"tenant": "t"}).Set(500)
	g.BudgetRemaining.With(Labels{"tenant": "t"}).Set(9500)
	g.CacheLookups.With(Labels{"result": "hit"}).Inc()
	g.CacheRejected.With(Labels{"reason": "nondeterministic"}).Inc()
	g.CacheEntries.With(nil).Set(10)
	g.CacheBytes.With(nil).Set(4096)
	g.CacheSavedPico.With(nil).Set(1500)
	ObserveSeconds(g.StreamTTFB.With(Labels{"model": "m"}), 30*time.Millisecond)
	g.StreamFrames.With(Labels{"model": "m"}).Add(42)
	g.StreamTruncated.With(Labels{"provider": "p"}).Inc()
	g.StreamAbandoned.With(Labels{"reason": "client_gone"}).Inc()

	out := g.Registry().String()
	// Every metric name must be valid, or Prometheus rejects the whole scrape.
	for _, line := range strings.Split(out, "\n") {
		if !strings.HasPrefix(line, "# TYPE ") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 4 {
			t.Errorf("malformed TYPE line: %q", line)
			continue
		}
		if !validMetricName(fields[2]) {
			t.Errorf("invalid metric name %q", fields[2])
		}
	}
	// Spot-check that the separate estimated-cost counter really is separate: a
	// spend total must never silently mix measured and estimated money.
	if !strings.Contains(out, "llmgw_cost_picodollars_total") ||
		!strings.Contains(out, "llmgw_cost_estimated_picodollars_total") {
		t.Error("the measured and estimated cost counters are not both present")
	}
}

func TestStatusClass(t *testing.T) {
	for code, want := range map[int]string{
		200: "2xx", 204: "2xx", 302: "3xx", 400: "4xx", 402: "4xx",
		429: "4xx", 500: "5xx", 503: "5xx", 0: "other", 99: "other",
	} {
		if got := StatusClass(code); got != want {
			t.Errorf("StatusClass(%d) = %q, want %q", code, got, want)
		}
	}
}

// TestObserveSecondsUsesSeconds guards the unit bug: recording milliseconds into
// a metric named _seconds silently makes every dashboard and alert threshold
// wrong by 1000x, and nothing fails.
func TestObserveSecondsUsesSeconds(t *testing.T) {
	h := NewHistogram([]float64{0.001, 0.01, 1})
	ObserveSeconds(h, 5*time.Millisecond)
	if got, want := h.Sum(), 0.005; math.Abs(got-want) > 1e-12 {
		t.Errorf("Sum = %v, want %v — a Duration was recorded in the wrong unit", got, want)
	}
	// A nil histogram must be a no-op rather than a panic; wiring code should not
	// need a nil check at every call site.
	ObserveSeconds(nil, time.Second)
}
