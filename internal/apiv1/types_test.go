package apiv1

import (
	"bytes"
	"encoding/json"
	"testing"
)

// TestContentUnionRoundTrips is the reason this package has custom
// (un)marshalers: `content` is legally either a bare string or an array of typed
// parts, and a gateway that re-emits the string form as a one-element array is a
// visible behaviour change to any client that diffs its own payloads. Each form
// must round-trip byte-faithfully.
func TestContentUnionRoundTrips(t *testing.T) {
	tests := []struct {
		name     string
		in       string
		wantText string
		wantStr  bool
	}{
		{"bare string", `"hello world"`, "hello world", true},
		{"empty string", `""`, "", true},
		{"null content", `null`, "", false},
		{"text parts array", `[{"type":"text","text":"a"},{"type":"text","text":"b"}]`, "a\nb", false},
		{"array with image part", `[{"type":"text","text":"describe"},{"type":"image_url","image_url":{"url":"http://x"}}]`, "describe", false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var c Content
			if err := json.Unmarshal([]byte(tc.in), &c); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if c.Text() != tc.wantText {
				t.Errorf("Text() = %q, want %q", c.Text(), tc.wantText)
			}
			if c.IsString() != tc.wantStr {
				t.Errorf("IsString() = %v, want %v", c.IsString(), tc.wantStr)
			}
			// Re-marshal must reproduce the ORIGINAL bytes, not a normalised form.
			out, err := json.Marshal(c)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			if !bytes.Equal(bytes.TrimSpace(out), []byte(tc.in)) {
				t.Errorf("round trip changed the wire form:\n  in:  %s\n  out: %s", tc.in, out)
			}
		})
	}
}

func TestContentHasNonTextParts(t *testing.T) {
	tests := []struct {
		in   string
		want bool
	}{
		{`"just text"`, false},
		{`[{"type":"text","text":"a"}]`, false},
		{`[{"type":"text","text":"a"},{"type":"image_url","image_url":{"url":"x"}}]`, true},
		{`null`, false},
	}
	for _, tc := range tests {
		var c Content
		if err := json.Unmarshal([]byte(tc.in), &c); err != nil {
			t.Fatal(err)
		}
		if got := c.HasNonTextParts(); got != tc.want {
			t.Errorf("HasNonTextParts(%s) = %v, want %v", tc.in, got, tc.want)
		}
	}
}

// TestStopUnionRoundTrips: `stop` is a string or an array of strings, and its
// order is significant to some providers, so it must survive in the form it
// arrived.
func TestStopUnionRoundTrips(t *testing.T) {
	tests := []struct {
		in     string
		values []string
	}{
		{`"END"`, []string{"END"}},
		{`["\n\n","STOP"]`, []string{"\n\n", "STOP"}},
		{`null`, nil},
	}
	for _, tc := range tests {
		var s StringOrArray
		if err := json.Unmarshal([]byte(tc.in), &s); err != nil {
			t.Fatalf("unmarshal %s: %v", tc.in, err)
		}
		if len(s.Values()) != len(tc.values) {
			t.Fatalf("%s: got %d values, want %d", tc.in, len(s.Values()), len(tc.values))
		}
		for i := range tc.values {
			if s.Values()[i] != tc.values[i] {
				t.Errorf("%s: value[%d] = %q, want %q", tc.in, i, s.Values()[i], tc.values[i])
			}
		}
		out, _ := json.Marshal(s)
		if !bytes.Equal(bytes.TrimSpace(out), []byte(tc.in)) {
			t.Errorf("round trip: in %s, out %s", tc.in, out)
		}
	}
}

// TestExtraFieldsPreserved: a client field the gateway does not model (e.g.
// logit_bias) must survive the round trip to the provider. Dropping it would
// give the client a successful response that silently ignored its request.
func TestExtraFieldsPreserved(t *testing.T) {
	in := `{"model":"m","messages":[{"role":"user","content":"hi"}],"logit_bias":{"50256":-100},"custom_flag":true}`
	var req ChatRequest
	if err := json.Unmarshal([]byte(in), &req); err != nil {
		t.Fatal(err)
	}
	if _, ok := req.Extra["logit_bias"]; !ok {
		t.Error("logit_bias was not captured into Extra")
	}
	if _, ok := req.Extra["custom_flag"]; !ok {
		t.Error("custom_flag was not captured into Extra")
	}
	out, err := json.Marshal(&req)
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]json.RawMessage
	json.Unmarshal(out, &got)
	if _, ok := got["logit_bias"]; !ok {
		t.Errorf("logit_bias was dropped on re-marshal: %s", out)
	}
	if _, ok := got["custom_flag"]; !ok {
		t.Errorf("custom_flag was dropped on re-marshal: %s", out)
	}
}

// TestExtraCannotShadowModelledField: a modelled field must always win over an
// Extra of the same name, or a stale Extra could override the typed value.
func TestExtraCannotShadowModelledField(t *testing.T) {
	req := ChatRequest{
		Model:    "real-model",
		Messages: []Message{{Role: "user"}},
		Extra:    map[string]json.RawMessage{"model": json.RawMessage(`"shadow-model"`)},
	}
	out, err := json.Marshal(&req)
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]json.RawMessage
	json.Unmarshal(out, &got)
	if string(got["model"]) != `"real-model"` {
		t.Errorf("Extra shadowed the modelled model field: got %s", got["model"])
	}
}

func TestEffectiveMaxTokens(t *testing.T) {
	i := func(v int) *int { return &v }
	tests := []struct {
		name             string
		maxTokens        *int
		maxCompletionTok *int
		want             int
	}{
		{"neither set", nil, nil, 0},
		{"max_tokens only", i(64), nil, 64},
		{"max_completion_tokens preferred", i(64), i(128), 128},
		{"zero max_tokens treated as unset", i(0), nil, 0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := ChatRequest{MaxTokens: tc.maxTokens, MaxCompletionTok: tc.maxCompletionTok}
			if got := r.EffectiveMaxTokens(); got != tc.want {
				t.Errorf("EffectiveMaxTokens() = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestValidate(t *testing.T) {
	i := func(v int) *int { return &v }
	tests := []struct {
		name    string
		req     ChatRequest
		wantErr bool
	}{
		{"ok", ChatRequest{Model: "m", Messages: []Message{{Role: "user"}}}, false},
		{"no model", ChatRequest{Messages: []Message{{Role: "user"}}}, true},
		{"no messages", ChatRequest{Model: "m"}, true},
		{"message without role", ChatRequest{Model: "m", Messages: []Message{{}}}, true},
		{"n>1 unsupported", ChatRequest{Model: "m", Messages: []Message{{Role: "user"}}, N: i(2)}, true},
		{"n=1 ok", ChatRequest{Model: "m", Messages: []Message{{Role: "user"}}, N: i(1)}, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.req.Validate(); (err != nil) != tc.wantErr {
				t.Errorf("Validate() err = %v, wantErr = %v", err, tc.wantErr)
			}
		})
	}
}

// TestUsageContainmentAccessors: reasoning tokens are inside completion, cached
// inside prompt. The accessors must reflect that so downstream cost code does
// not double-count.
func TestUsageAccessors(t *testing.T) {
	u := &Usage{
		PromptTokens:            100,
		CompletionTokens:        50,
		PromptTokensDetails:     &PromptTokensDetails{CachedTokens: 30},
		CompletionTokensDetails: &CompletionTokensDetails{ReasoningTokens: 20},
	}
	if u.CachedPromptTokens() != 30 {
		t.Errorf("CachedPromptTokens() = %d, want 30", u.CachedPromptTokens())
	}
	if u.ReasoningTokens() != 20 {
		t.Errorf("ReasoningTokens() = %d, want 20", u.ReasoningTokens())
	}
	// Nil-safe.
	var nilU *Usage
	if nilU.CachedPromptTokens() != 0 || nilU.ReasoningTokens() != 0 {
		t.Error("nil Usage accessors should return 0, not panic")
	}
}

func TestChunkDeltaText(t *testing.T) {
	c := &ChatChunk{Choices: []Choice{{Delta: &Message{Content: NewTextContent("tok")}}}}
	if c.DeltaText() != "tok" {
		t.Errorf("DeltaText() = %q, want tok", c.DeltaText())
	}
	// Nil-safe on an empty chunk.
	empty := &ChatChunk{}
	if empty.DeltaText() != "" {
		t.Errorf("empty chunk DeltaText() = %q, want empty", empty.DeltaText())
	}
}

func TestContentArrayFormPreservesImagePartsOnRemarshal(t *testing.T) {
	// The whole raw array — including the image part the estimator cannot count —
	// must survive to the provider, or a multimodal request silently loses its
	// image.
	in := `[{"type":"text","text":"look"},{"type":"image_url","image_url":{"url":"http://img"}}]`
	var c Content
	if err := json.Unmarshal([]byte(in), &c); err != nil {
		t.Fatal(err)
	}
	out, _ := json.Marshal(c)
	if !bytes.Equal(bytes.TrimSpace(out), []byte(in)) {
		t.Errorf("image part lost on remarshal:\n  in:  %s\n  out: %s", in, out)
	}
}
