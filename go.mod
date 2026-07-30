module github.com/harsha-moparthy/llmgw

go 1.26

// No dependencies, deliberately. Everything here — SSE framing, the circuit
// breaker, the Prometheus exposition format, the HTTP surface — is stdlib.
// A gateway sits in the request path of every LLM call a company makes, so its
// dependency tree is its attack surface and its p99 is its reputation.
