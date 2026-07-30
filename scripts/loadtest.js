// k6 load test for the LLM gateway.
//
// This script measures the gateway under concurrent load and is the source of
// the throughput and error-rate figures in the README. It deliberately does NOT
// produce the headline p99-overhead number: k6 measures end-to-end latency from
// outside, which includes the mock provider's own simulated generation time
// (seconds), so a p99 taken here would be dominated by the instrument rather
// than the gateway. The overhead figure comes from cmd/gwbench, which reads the
// gateway's own X-LLMGW-Overhead-Us header and so isolates gateway time from
// provider time. Two tools because they answer two different questions.
//
// Usage:
//   k6 run scripts/loadtest.js
//   k6 run -e SCENARIO=failover -e GATEWAY=http://127.0.0.1:8080 scripts/loadtest.js
//
// Scenarios:
//   steady    (default) constant arrival rate, non-streaming, healthy providers
//   streaming constant arrival rate of streaming requests
//   ramp      ramping arrival rate, to find the knee
//   failover  steady load intended to be run while a provider is killed
//   budget    a single tenant hammering a small budget, to observe rejection
//             semantics under concurrency

import http from 'k6/http'
import { check, fail } from 'k6'
import { Counter, Trend, Rate } from 'k6/metrics'

const GATEWAY = __ENV.GATEWAY || 'http://127.0.0.1:8080'
const SCENARIO = __ENV.SCENARIO || 'steady'
const MODEL = __ENV.MODEL || 'gw-chat'
const RATE = parseInt(__ENV.RATE || '200', 10)
const DURATION = __ENV.DURATION || '30s'

// Tenant keys must match configs/gateway.example.json. The load test
// authenticates for real rather than bypassing auth, because auth is on the
// measured path and excluding it would understate the gateway's overhead.
const TENANTS = {
  bench: __ENV.BENCH_KEY || 'bench-key',
  small: __ENV.SMALL_KEY || 'small-budget-key',
}

// Custom metrics. The gateway reports its own overhead per request in a header.
// It is tracked here for visibility but is NOT the source of the README's
// overhead figure: k6 runs on the same machine as the gateway, so under load the
// two contend for CPU and the gateway's self-timed overhead inflates for a
// reason that has nothing to do with the gateway's code. cmd/gwbench measures it
// in isolation. The Trend is NOT marked isTime (its unit is microseconds, and
// k6's time formatting assumes milliseconds — marking it isTime is exactly the
// us-vs-ms bug this project is otherwise careful to avoid).
const gwOverheadUs = new Trend('gw_overhead_us')
const gwAttempts = new Trend('gw_attempts')
const failovers = new Counter('gw_failovers')
const budgetRejects = new Counter('gw_budget_rejects')
const upstream5xx = new Counter('gw_upstream_5xx')
const streamTTFB = new Trend('gw_stream_ttfb_ms', true)
const okRate = new Rate('gw_success_rate')

const scenarios = {
  steady: {
    executor: 'constant-arrival-rate',
    rate: RATE,
    timeUnit: '1s',
    duration: DURATION,
    preAllocatedVUs: Math.max(50, RATE),
    maxVUs: Math.max(200, RATE * 4),
    exec: 'nonStreaming',
  },
  streaming: {
    executor: 'constant-arrival-rate',
    rate: Math.max(1, Math.floor(RATE / 4)),
    timeUnit: '1s',
    duration: DURATION,
    preAllocatedVUs: Math.max(50, RATE),
    maxVUs: Math.max(200, RATE * 4),
    exec: 'streaming',
  },
  ramp: {
    executor: 'ramping-arrival-rate',
    startRate: 20,
    timeUnit: '1s',
    preAllocatedVUs: 100,
    maxVUs: 2000,
    stages: [
      { target: 100, duration: '15s' },
      { target: 400, duration: '15s' },
      { target: 1000, duration: '15s' },
      { target: 1000, duration: '15s' },
    ],
    exec: 'nonStreaming',
  },
  failover: {
    executor: 'constant-arrival-rate',
    rate: RATE,
    timeUnit: '1s',
    duration: DURATION,
    preAllocatedVUs: Math.max(50, RATE),
    maxVUs: Math.max(200, RATE * 4),
    exec: 'nonStreaming',
  },
  budget: {
    executor: 'constant-arrival-rate',
    rate: 50,
    timeUnit: '1s',
    duration: '10s',
    preAllocatedVUs: 50,
    maxVUs: 200,
    exec: 'budgetPressure',
  },
}

if (!scenarios[SCENARIO]) {
  fail(`unknown SCENARIO=${SCENARIO}; expected one of ${Object.keys(scenarios).join(', ')}`)
}

// Thresholds are intentionally asymmetric per scenario. The failover scenario
// EXPECTS non-2xx responses (a provider is being killed under it), so asserting
// a low error rate there would fail the run for doing exactly what it is meant
// to do. Instead it asserts that failures stay bounded and that failover
// actually happened — a failover test that records zero failovers proves
// nothing and must not pass.
const thresholds = {
  steady: {
    'http_req_failed': ['rate<0.01'],
    // No gw_overhead_us threshold here: see the metric's comment. Under
    // co-located load k6 and the gateway contend for the same cores, so this
    // number is not a clean measurement of gateway overhead — cmd/gwbench is.
    // The success-rate and body-shape checks are what this scenario gates on.
    'gw_success_rate': ['rate>0.99'],
  },
  streaming: {
    'http_req_failed': ['rate<0.01'],
  },
  ramp: {},
  failover: {
    // Bounded, not zero: some requests legitimately fail while a provider dies.
    'http_req_failed': ['rate<0.20'],
    'gw_failovers': ['count>0'],
  },
  budget: {
    'gw_budget_rejects': ['count>0'],
  },
}

export const options = {
  scenarios: { [SCENARIO]: scenarios[SCENARIO] },
  thresholds: thresholds[SCENARIO] || {},
  summaryTrendStats: ['avg', 'min', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
  // A gateway is a proxy; connection reuse is part of what is being measured.
  noConnectionReuse: false,
  discardResponseBodies: false,
}

function headers(tenantKey) {
  return {
    'Authorization': `Bearer ${tenantKey}`,
    'Content-Type': 'application/json',
  }
}

// Vary the prompt per iteration so the response cache does not turn the load
// test into a measurement of the cache. A cache-hit-only load test reports a
// beautiful p99 that says nothing about the gateway's real path. The cache is
// measured separately and deliberately in cmd/gwbench.
function body(stream, extra) {
  const uniq = `${__VU}-${__ITER}`
  return JSON.stringify(Object.assign({
    model: MODEL,
    messages: [
      { role: 'system', content: 'You are a concise assistant.' },
      { role: 'user', content: `Summarise the state of request ${uniq} in one sentence.` },
    ],
    max_tokens: 64,
    temperature: 0.7,
    stream: stream,
  }, extra || {}))
}

// recordGatewayHeaders pulls the gateway's self-reported telemetry off the
// response. These headers are the cross-check between what the gateway thinks
// it did and what the client observed.
function recordGatewayHeaders(res) {
  const ov = res.headers['X-Llmgw-Overhead-Us'] || res.headers['X-LLMGW-Overhead-Us']
  if (ov !== undefined) {
    const v = parseInt(ov, 10)
    if (!isNaN(v)) gwOverheadUs.add(v)
  }
  const at = res.headers['X-Llmgw-Attempts'] || res.headers['X-LLMGW-Attempts']
  if (at !== undefined) {
    const n = parseInt(at, 10)
    if (!isNaN(n)) {
      gwAttempts.add(n)
      if (n > 1) failovers.add(n - 1)
    }
  }
}

export function nonStreaming() {
  const res = http.post(`${GATEWAY}/v1/chat/completions`, body(false), {
    headers: headers(TENANTS.bench),
    tags: { kind: 'nonstream' },
  })
  recordGatewayHeaders(res)
  okRate.add(res.status === 200)
  if (res.status >= 500) upstream5xx.add(1)
  if (res.status === 402 || res.status === 429) budgetRejects.add(1)

  check(res, {
    'status is 200 or a classified refusal': (r) =>
      r.status === 200 || r.status === 402 || r.status === 429 || r.status === 503,
    // A 200 must actually carry a completion and a usage record. Checking only
    // the status code lets an empty-bodied 200 pass, which is the failure mode
    // a broken passthrough produces.
    'successful body has content and usage': (r) => {
      if (r.status !== 200) return true
      let j
      try { j = r.json() } catch (e) { return false }
      return j && j.choices && j.choices.length > 0 &&
        j.usage && j.usage.total_tokens > 0
    },
    // Any non-2xx must be a well-formed OpenAI error envelope, because clients
    // switch on error.type. A bare text/plain 503 from a proxy breaks SDKs.
    'errors are OpenAI-shaped': (r) => {
      if (r.status === 200) return true
      let j
      try { j = r.json() } catch (e) { return false }
      return j && j.error && typeof j.error.type === 'string'
    },
  })
}

export function streaming() {
  const start = Date.now()
  // k6's http module buffers the whole SSE response rather than exposing frames
  // incrementally, so TTFB here is approximated by waiting_time. Frame-level
  // streaming assertions (ordering, [DONE], error frames, inter-token gaps) are
  // made in Go, in internal/server's tests and in cmd/gwbench, where a real
  // incremental reader is available. Noting the limitation rather than
  // reporting a TTFB that is actually a full-response time.
  const res = http.post(`${GATEWAY}/v1/chat/completions`, body(true), {
    headers: headers(TENANTS.bench),
    tags: { kind: 'stream' },
  })
  recordGatewayHeaders(res)
  okRate.add(res.status === 200)
  streamTTFB.add(res.timings.waiting)
  if (res.status >= 500) upstream5xx.add(1)

  check(res, {
    'stream status is 200 or a classified refusal': (r) =>
      r.status === 200 || r.status === 402 || r.status === 429 || r.status === 503,
    'stream is SSE': (r) =>
      r.status !== 200 || String(r.headers['Content-Type'] || '').indexOf('text/event-stream') === 0,
    // A clean stream must terminate with the sentinel. Without this check a
     // truncated stream passes as a success, which is precisely the bug the
    // mid-stream-abort fault exists to catch.
    'stream terminated with [DONE]': (r) => {
      if (r.status !== 200) return true
      return String(r.body).indexOf('data: [DONE]') !== -1
    },
    'stream carried at least one delta': (r) => {
      if (r.status !== 200) return true
      return String(r.body).indexOf('"delta"') !== -1
    },
  })
  // Guard against a mis-tagged duration if the scenario is misconfigured.
  if (Date.now() - start < 0) fail('negative elapsed time')
}

// budgetPressure drives one small-budget tenant hard, to observe that rejection
// is clean and informative under concurrency rather than a 500 or a hang. The
// interesting assertion is not that requests are rejected — it is that the
// rejection body tells the client the limit, the spend and the reset time, and
// that the gateway never admits more than the limit (proved exactly in Go, in
// internal/budget's concurrency test; observed end-to-end here).
export function budgetPressure() {
  const res = http.post(`${GATEWAY}/v1/chat/completions`, body(false), {
    headers: headers(TENANTS.small),
    tags: { kind: 'budget' },
  })
  recordGatewayHeaders(res)
  const rejected = res.status === 402 || res.status === 429
  if (rejected) budgetRejects.add(1)
  okRate.add(res.status === 200)

  check(res, {
    'budget outcome is 200 or a budget refusal': (r) =>
      r.status === 200 || r.status === 402 || r.status === 429,
    'rejection explains itself': (r) => {
      if (!rejected) return true
      let j
      try { j = r.json() } catch (e) { return false }
      if (!j || !j.error) return false
      const m = String(j.error.message)
      // The message must carry the numbers a client needs to act, not just
      // "budget exceeded".
      return m.indexOf('limit') !== -1 && m.indexOf('spent') !== -1
    },
    'rejection is retry-informative': (r) => {
      if (!rejected) return true
      // Either a Retry-After header or a reset time in the body — a client that
      // cannot tell when to come back will hot-loop.
      return r.headers['Retry-After'] !== undefined ||
        String(r.body).indexOf('reset') !== -1
    },
  })
}
