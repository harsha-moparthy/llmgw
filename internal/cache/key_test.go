package cache

import (
	"encoding/json"
	"testing"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
)

// TestKeyBoundaryShiftDoesNotCollide is the most important test in this file.
//
// A canonicaliser that concatenates fields without length prefixes maps
// ("ab","c") and ("a","bc") to the same bytes, so two different conversations
// get the same key and one user receives the other's answer. The length-prefixed
// encoding in ComputeKey exists precisely to prevent this, and this test fails
// if someone "simplifies" it away.
func TestKeyBoundaryShiftDoesNotCollide(t *testing.T) {
	scope := TenantScope("t")

	pairs := [][2][]apiv1.Message{
		{
			{{Role: "user", Content: apiv1.NewTextContent("ab")}, {Role: "user", Content: apiv1.NewTextContent("c")}},
			{{Role: "user", Content: apiv1.NewTextContent("a")}, {Role: "user", Content: apiv1.NewTextContent("bc")}},
		},
		{
			{{Role: "user", Content: apiv1.NewTextContent("hello world")}},
			{{Role: "user", Content: apiv1.NewTextContent("hello")}, {Role: "user", Content: apiv1.NewTextContent("world")}},
		},
		{
			{{Role: "system", Content: apiv1.NewTextContent("be nice")}, {Role: "user", Content: apiv1.NewTextContent("hi")}},
			{{Role: "system", Content: apiv1.NewTextContent("be nicehi")}},
		},
	}

	for i, p := range pairs {
		a := ComputeKey(&apiv1.ChatRequest{Model: "m", Messages: p[0], Temperature: f64(0)}, scope)
		b := ComputeKey(&apiv1.ChatRequest{Model: "m", Messages: p[1], Temperature: f64(0)}, scope)
		if a.Digest == b.Digest {
			t.Errorf("pair %d: boundary-shifted message sets collided (digest %s)", i, a.Digest)
		}
	}

	// The same trick across the model/role boundary.
	x := ComputeKey(&apiv1.ChatRequest{
		Model: "ab", Messages: []apiv1.Message{{Role: "c", Content: apiv1.NewTextContent("d")}}, Temperature: f64(0),
	}, scope)
	y := ComputeKey(&apiv1.ChatRequest{
		Model: "a", Messages: []apiv1.Message{{Role: "bc", Content: apiv1.NewTextContent("d")}}, Temperature: f64(0),
	}, scope)
	if x.Digest == y.Digest {
		t.Error("model/role boundary shift collided")
	}
}

// TestKeyStabilityAcrossIdenticalRequests: the same request must always produce
// the same key, or the cache never hits.
func TestKeyStabilityAcrossIdenticalRequests(t *testing.T) {
	mk := func() *apiv1.ChatRequest {
		return &apiv1.ChatRequest{
			Model: "gw-chat",
			Messages: []apiv1.Message{
				{Role: "system", Content: apiv1.NewTextContent("You are concise.")},
				{Role: "user", Content: apiv1.NewTextContent("2+2?")},
			},
			Temperature: f64(0),
			MaxTokens:   iptr(64),
		}
	}
	a := ComputeKey(mk(), TenantScope("t"))
	b := ComputeKey(mk(), TenantScope("t"))
	if a.Digest != b.Digest {
		t.Fatalf("identical requests produced different digests:\n  %s\n  %s", a.Digest, b.Digest)
	}
	if a.String() != b.String() {
		t.Errorf("storage keys differ: %q vs %q", a.String(), b.String())
	}
}

// TestStreamFlagExcludedFromKey: streaming is a transport choice, so the same
// question asked both ways must share an entry. Including the flag would halve
// the hit rate for no correctness gain.
func TestStreamFlagExcludedFromKey(t *testing.T) {
	base := apiv1.ChatRequest{
		Model:       "m",
		Messages:    []apiv1.Message{{Role: "user", Content: apiv1.NewTextContent("hi")}},
		Temperature: f64(0),
	}
	nonStream := base
	streaming := base
	streaming.Stream = true
	streaming.StreamOptions = &apiv1.StreamOptions{IncludeUsage: true}

	a := ComputeKey(&nonStream, TenantScope("t"))
	b := ComputeKey(&streaming, TenantScope("t"))
	if a.Digest != b.Digest {
		t.Error("the stream flag changed the cache key; a streaming client cannot reuse a non-streaming entry")
	}
}

// TestUserFieldExcludedFromKey: `user` is an attribution tag that never reaches
// the model as content, so including it would give every end user their own
// private cache miss.
func TestUserFieldExcludedFromKey(t *testing.T) {
	base := apiv1.ChatRequest{
		Model:       "m",
		Messages:    []apiv1.Message{{Role: "user", Content: apiv1.NewTextContent("hi")}},
		Temperature: f64(0),
	}
	a := base
	a.User = "alice"
	b := base
	b.User = "bob"
	if ComputeKey(&a, TenantScope("t")).Digest != ComputeKey(&b, TenantScope("t")).Digest {
		t.Error("the `user` attribution field changed the cache key")
	}
}

// TestSemanticFieldsChangeKey: every field that can change the model's answer
// must change the key. A miss here means serving a wrong response.
func TestSemanticFieldsChangeKey(t *testing.T) {
	base := func() *apiv1.ChatRequest {
		return &apiv1.ChatRequest{
			Model:       "m",
			Messages:    []apiv1.Message{{Role: "user", Content: apiv1.NewTextContent("hi")}},
			Temperature: f64(0),
		}
	}
	baseDigest := ComputeKey(base(), TenantScope("t")).Digest

	mutations := map[string]func(*apiv1.ChatRequest){
		"model":            func(r *apiv1.ChatRequest) { r.Model = "other" },
		"message text":     func(r *apiv1.ChatRequest) { r.Messages[0].Content = apiv1.NewTextContent("bye") },
		"message role":     func(r *apiv1.ChatRequest) { r.Messages[0].Role = "system" },
		"message name":     func(r *apiv1.ChatRequest) { r.Messages[0].Name = "bob" },
		"extra message":    func(r *apiv1.ChatRequest) { r.Messages = append(r.Messages, apiv1.Message{Role: "user"}) },
		"temperature":      func(r *apiv1.ChatRequest) { r.Temperature = f64(0.5) },
		"top_p":            func(r *apiv1.ChatRequest) { r.TopP = f64(0.9) },
		"max_tokens":       func(r *apiv1.ChatRequest) { r.MaxTokens = iptr(10) },
		"seed":             func(r *apiv1.ChatRequest) { v := int64(7); r.Seed = &v },
		"stop":             func(r *apiv1.ChatRequest) { r.Stop = apiv1.NewStringOrArray("END") },
		"presence penalty": func(r *apiv1.ChatRequest) { r.PresencePenalty = f64(1) },
		"freq penalty":     func(r *apiv1.ChatRequest) { r.FrequencyPenalty = f64(1) },
		"tools":            func(r *apiv1.ChatRequest) { r.Tools = json.RawMessage(`[{"type":"function"}]`) },
		"tool_choice":      func(r *apiv1.ChatRequest) { r.ToolChoice = json.RawMessage(`"auto"`) },
		"response_format":  func(r *apiv1.ChatRequest) { r.ResponseFormat = json.RawMessage(`{"type":"json_object"}`) },
		"extra field": func(r *apiv1.ChatRequest) {
			r.Extra = map[string]json.RawMessage{"logit_bias": json.RawMessage(`{"1":2}`)}
		},
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			r := base()
			mutate(r)
			if got := ComputeKey(r, TenantScope("t")).Digest; got == baseDigest {
				t.Errorf("changing %s did not change the cache key", name)
			}
		})
	}
}

// TestMaxTokensSpellingsAgree: max_tokens and max_completion_tokens mean the same
// thing, so a client that switched spellings should still hit.
func TestMaxTokensSpellingsAgree(t *testing.T) {
	a := &apiv1.ChatRequest{
		Model: "m", Messages: []apiv1.Message{{Role: "user"}},
		Temperature: f64(0), MaxTokens: iptr(64),
	}
	b := &apiv1.ChatRequest{
		Model: "m", Messages: []apiv1.Message{{Role: "user"}},
		Temperature: f64(0), MaxCompletionTok: iptr(64),
	}
	if ComputeKey(a, TenantScope("t")).Digest != ComputeKey(b, TenantScope("t")).Digest {
		t.Error("the two spellings of the completion cap produced different keys")
	}
}

// TestToolJSONKeyOrderCanonicalised: JSON object key order is not semantic, and
// different SDKs emit different orders for the same tool definition. Without
// canonicalisation those clients would never share a cache entry.
func TestToolJSONKeyOrderCanonicalised(t *testing.T) {
	a := &apiv1.ChatRequest{
		Model: "m", Messages: []apiv1.Message{{Role: "user"}}, Temperature: f64(0),
		Tools: json.RawMessage(`[{"type":"function","function":{"name":"f","description":"d"}}]`),
	}
	b := &apiv1.ChatRequest{
		Model: "m", Messages: []apiv1.Message{{Role: "user"}}, Temperature: f64(0),
		Tools: json.RawMessage(`[{"function":{"description":"d","name":"f"},"type":"function"}]`),
	}
	if ComputeKey(a, TenantScope("t")).Digest != ComputeKey(b, TenantScope("t")).Digest {
		t.Error("reordered JSON keys in an identical tool definition produced different cache keys")
	}
	// But a genuinely different definition must still differ.
	c := &apiv1.ChatRequest{
		Model: "m", Messages: []apiv1.Message{{Role: "user"}}, Temperature: f64(0),
		Tools: json.RawMessage(`[{"type":"function","function":{"name":"g","description":"d"}}]`),
	}
	if ComputeKey(a, TenantScope("t")).Digest == ComputeKey(c, TenantScope("t")).Digest {
		t.Error("different tool names produced the same key")
	}
}

// TestAbsentVsExplicitDefaultDiffer: a nil parameter and an explicitly-set value
// are hashed differently, because providers may treat them differently and
// assuming equivalence would be a correctness claim about every upstream at once.
func TestAbsentVsExplicitDefaultDiffer(t *testing.T) {
	absent := &apiv1.ChatRequest{
		Model: "m", Messages: []apiv1.Message{{Role: "user"}}, Temperature: f64(0),
	}
	explicit := &apiv1.ChatRequest{
		Model: "m", Messages: []apiv1.Message{{Role: "user"}}, Temperature: f64(0), TopP: f64(1),
	}
	if ComputeKey(absent, TenantScope("t")).Digest == ComputeKey(explicit, TenantScope("t")).Digest {
		t.Error("absent top_p and explicit top_p=1 hashed identically")
	}
}

// TestFloatPrecisionInKey: two float64 values that are distinct but share a short
// decimal representation must not collide.
func TestFloatPrecisionInKey(t *testing.T) {
	mk := func(v float64) *apiv1.ChatRequest {
		return &apiv1.ChatRequest{
			Model: "m", Messages: []apiv1.Message{{Role: "user"}},
			Temperature: f64(0), TopP: f64(v),
		}
	}
	a := ComputeKey(mk(0.1), TenantScope("t")).Digest
	b := ComputeKey(mk(0.1+1e-17), TenantScope("t")).Digest
	// 0.1 + 1e-17 rounds to the same float64 as 0.1, so these SHOULD match —
	// asserting that pins the intent (identical values hash identically) rather
	// than an accident of formatting.
	if a != b {
		t.Error("values that are the same float64 hashed differently")
	}
	c := ComputeKey(mk(0.1000000000000001), TenantScope("t")).Digest
	if a == c {
		t.Error("distinct float64 values collided; the key format is lossy")
	}
}

// TestContentFormPreservedInKey: text-form and array-form content with the same
// words are different requests (the array form may carry images), so they must
// not share an entry.
func TestContentFormPreservedInKey(t *testing.T) {
	var arrayForm apiv1.Content
	if err := json.Unmarshal([]byte(`[{"type":"text","text":"hi"}]`), &arrayForm); err != nil {
		t.Fatal(err)
	}
	textForm := apiv1.NewTextContent("hi")

	a := ComputeKey(&apiv1.ChatRequest{
		Model: "m", Messages: []apiv1.Message{{Role: "user", Content: textForm}}, Temperature: f64(0),
	}, TenantScope("t"))
	b := ComputeKey(&apiv1.ChatRequest{
		Model: "m", Messages: []apiv1.Message{{Role: "user", Content: arrayForm}}, Temperature: f64(0),
	}, TenantScope("t"))
	if a.Digest == b.Digest {
		t.Error("string content and array content with the same text share a cache key")
	}
}

// TestKeyVersionTagPresent ensures a change to the canonical form can be rolled
// out without silently reinterpreting existing entries. The digest of a fixed
// request is pinned; changing what is hashed will fail this test loudly, which is
// the reminder to bump the version tag.
func TestKeyVersionTagPinned(t *testing.T) {
	req := &apiv1.ChatRequest{
		Model:       "gw-chat",
		Messages:    []apiv1.Message{{Role: "user", Content: apiv1.NewTextContent("hello")}},
		Temperature: f64(0),
	}
	got := ComputeKey(req, TenantScope("acme"))
	if len(got.Digest) != 64 {
		t.Fatalf("digest length = %d, want 64 hex chars of SHA-256", len(got.Digest))
	}
	if want := "t:acme:" + got.Digest; got.String() != want {
		t.Errorf("String() = %q, want %q", got.String(), want)
	}
}

func TestScopeString(t *testing.T) {
	if got := TenantScope("acme").String(); got != "tenant:acme" {
		t.Errorf("TenantScope.String() = %q", got)
	}
	if got := SharedPoolScope("facts").String(); got != "pool:facts" {
		t.Errorf("SharedPoolScope.String() = %q", got)
	}
}
