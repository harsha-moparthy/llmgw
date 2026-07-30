package cache

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"sort"
	"strconv"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
)

// Scope determines which requests may share a cache entry.
//
// The type exists so that sharing has to be asked for by name. A cache whose
// default is "shared" leaks one customer's prompts to another as cache hits,
// which is a data-isolation breach rather than a performance regression, and it
// is invisible in testing because a hit looks exactly like a correct response.
// Making the isolated scope the zero value means the safe behaviour is what you
// get by forgetting to think about it.
type Scope struct {
	// tenant namespaces the key. Empty only for an explicitly shared pool.
	tenant string
	// pool is a named shared namespace, used when several tenants have
	// consented to share (same org, same trust boundary).
	pool string
}

// TenantScope returns a scope private to one tenant. This is the default and
// the only one the server uses unless a tenant's config opts into a pool.
func TenantScope(tenantID string) Scope { return Scope{tenant: tenantID} }

// SharedPoolScope returns a scope shared by every tenant configured with the
// same pool name.
//
// Deliberately verbose, and deliberately requiring a name rather than accepting
// a bare boolean: an operator enabling this is making a data-sharing decision
// and the config should read like one.
func SharedPoolScope(pool string) Scope { return Scope{pool: pool} }

// String renders the scope for metrics and debugging. It never includes prompt
// content.
func (s Scope) String() string {
	if s.pool != "" {
		return "pool:" + s.pool
	}
	return "tenant:" + s.tenant
}

// namespace returns the bytes that prefix every key computed in this scope.
func (s Scope) namespace() (kind byte, name string) {
	if s.pool != "" {
		return 'p', s.pool
	}
	return 't', s.tenant
}

// Key is a cache key: a scope plus a digest of the semantically relevant
// request fields.
type Key struct {
	// Digest is the hex SHA-256 over the canonical form.
	Digest string
	// Scope is carried alongside so metrics can attribute a hit without
	// re-deriving it.
	Scope Scope
}

// String returns the storage key.
func (k Key) String() string {
	kind, name := k.Scope.namespace()
	return string(kind) + ":" + name + ":" + k.Digest
}

// ComputeKey derives the cache key for a request within a scope.
//
// # What is included and why
//
// Everything that can change the model's output: model, the full message list,
// temperature, top_p, stop, seed, tools, tool_choice, response_format, and the
// completion cap. Two requests with the same key must be substitutable for one
// another, so omitting a field that changes the answer would serve a wrong
// response, and including a field that cannot change the answer only costs hit
// rate.
//
// # What is deliberately excluded
//
// The `stream` flag. The same question asked streaming and non-streaming has
// the same answer — streaming is a transport choice, not a semantic one — and a
// key that includes it halves the hit rate for no correctness benefit. The
// entry stores the complete response, and the streaming path replays it as
// frames (see Entry.Replay).
//
// Also excluded: `user` (an end-user attribution tag that does not reach the
// model as content), and `stream_options`.
//
// # Canonicalisation
//
// The digest is computed over a length-prefixed encoding, not over concatenated
// strings. This matters more than it looks: with plain concatenation the
// requests ("ab", "c") and ("a", "bc") produce the same byte stream and
// therefore the same key, which is a cross-request collision that serves one
// user's answer to another's question. Every variable-length field is preceded
// by its length, so the encoding is unambiguous. TestKeyBoundaryShiftDoesNotCollide
// pins this.
func ComputeKey(req *apiv1.ChatRequest, scope Scope) Key {
	h := sha256.New()

	// A version tag so that a future change to what is canonicalised invalidates
	// old entries instead of silently reusing them under a new interpretation.
	writeField(h, []byte("llmgw-cache-v1"))

	kind, name := scope.namespace()
	writeField(h, []byte{kind})
	writeField(h, []byte(name))

	writeField(h, []byte(req.Model))

	writeLen(h, len(req.Messages))
	for i := range req.Messages {
		m := &req.Messages[i]
		writeField(h, []byte(m.Role))
		writeField(h, []byte(m.Name))
		// The raw content JSON is hashed rather than the flattened text, so that
		// array-form content with image parts does not collide with the
		// text-only request that happens to have the same words. Those are
		// genuinely different requests with genuinely different answers.
		raw, err := m.Content.MarshalJSON()
		if err != nil {
			raw = []byte("null")
		}
		writeField(h, raw)
		writeField(h, m.ToolCalls)
		writeField(h, []byte(m.ToolCallID))
	}

	// Sampling parameters. A nil pointer and an explicitly-set default are
	// hashed differently because the provider may treat them differently, and
	// guessing that they are equivalent would be a correctness assumption about
	// every upstream at once.
	writeOptFloat(h, req.Temperature)
	writeOptFloat(h, req.TopP)
	writeOptFloat(h, req.PresencePenalty)
	writeOptFloat(h, req.FrequencyPenalty)
	writeOptInt(h, req.N)
	if req.Seed != nil {
		writeField(h, []byte("seed"))
		writeField(h, []byte(strconv.FormatInt(*req.Seed, 10)))
	} else {
		writeField(h, nil)
	}

	// max_tokens changes the answer: the same prompt capped at 10 tokens and at
	// 1000 tokens yields different responses, and serving the short one for the
	// long request is a visible truncation bug.
	writeField(h, []byte(strconv.Itoa(req.EffectiveMaxTokens())))

	// Stop sequences are order-significant to some providers, so the original
	// order is preserved rather than sorted.
	stops := req.Stop.Values()
	writeLen(h, len(stops))
	for _, s := range stops {
		writeField(h, []byte(s))
	}

	// Tool and format definitions are canonicalised so that a semantically
	// identical definition with different key ordering still hits. JSON object
	// key order is not semantic, and clients built from different SDKs emit
	// different orders for the same tool set.
	writeField(h, canonicalJSON(req.Tools))
	writeField(h, canonicalJSON(req.ToolChoice))
	writeField(h, canonicalJSON(req.ResponseFormat))

	// Unmodelled fields are included, sorted by name. A client setting
	// logit_bias is changing the output, and a cache that ignored it would serve
	// an unbiased response to a biased request.
	if len(req.Extra) > 0 {
		keys := make([]string, 0, len(req.Extra))
		for k := range req.Extra {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		writeLen(h, len(keys))
		for _, k := range keys {
			writeField(h, []byte(k))
			writeField(h, canonicalJSON(req.Extra[k]))
		}
	} else {
		writeLen(h, 0)
	}

	return Key{Digest: hex.EncodeToString(h.Sum(nil)), Scope: scope}
}

// writeField writes a length-prefixed field. The length prefix is what makes the
// encoding injective.
func writeField(h interface{ Write([]byte) (int, error) }, b []byte) {
	writeLen(h, len(b))
	_, _ = h.Write(b)
}

func writeLen(h interface{ Write([]byte) (int, error) }, n int) {
	var buf [8]byte
	binary.BigEndian.PutUint64(buf[:], uint64(n))
	_, _ = h.Write(buf[:])
}

// writeOptFloat hashes an optional float, distinguishing absent from present.
//
// strconv.FormatFloat with 'b' (binary exponent) is used rather than a decimal
// format because it is exact and injective for float64: two distinct float64
// values can share a short decimal representation, and hashing the decimal form
// would map them to one key.
func writeOptFloat(h interface{ Write([]byte) (int, error) }, f *float64) {
	if f == nil {
		writeField(h, nil)
		return
	}
	writeField(h, []byte(strconv.FormatFloat(*f, 'b', -1, 64)))
}

func writeOptInt(h interface{ Write([]byte) (int, error) }, n *int) {
	if n == nil {
		writeField(h, nil)
		return
	}
	writeField(h, []byte(strconv.Itoa(*n)))
}

// canonicalJSON re-encodes a JSON value with object keys sorted, so that two
// encodings of the same value hash identically.
//
// encoding/json already sorts map[string]any keys on marshal, so decoding into
// `any` and re-marshalling is sufficient and avoids hand-rolling a canonical
// JSON writer. Invalid or absent JSON is passed through unchanged rather than
// being treated as an error: the cache's job is not to validate the request.
func canonicalJSON(raw json.RawMessage) []byte {
	if len(raw) == 0 {
		return nil
	}
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		return raw
	}
	out, err := json.Marshal(v)
	if err != nil {
		return raw
	}
	return out
}
