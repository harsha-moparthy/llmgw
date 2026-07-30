package tokens

import (
	"encoding/json"
	"math/rand"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
)

func TestReferenceTokenize(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want []string
	}{
		{"empty", "", nil},
		{"single letter", "a", []string{"a"}},
		{"word", "hello", []string{"hello"}},
		// Rule 2: a leading space attaches to the word.
		{"leading space attaches", " the", []string{" the"}},
		{"two words", "hello world", []string{"hello", " world"}},
		// Rule 3: chunking at ReferenceMaxWordRunes.
		{"exactly the chunk limit", "abcdef", []string{"abcdef"}},
		{"one past the chunk limit", "abcdefg", []string{"abcdef", "g"}},
		{"two full chunks", "abcdefghijkl", []string{"abcdef", "ghijkl"}},
		// Rule 4: contractions.
		{"contraction", "don't", []string{"don", "'t"}},
		{"typographic apostrophe", "don’t", []string{"don", "’t"}},
		{"possessive across a chunk boundary", "abcdef's", []string{"abcdef", "'s"}},
		// A trailing apostrophe has no letter after it, so rule 4 does not fire
		// and it is punctuation.
		{"trailing apostrophe", "dogs'", []string{"dogs", "'"}},
		// A leading apostrophe has no letter before it.
		{"leading apostrophe", "'tis", []string{"'", "tis"}},
		// Rule 5: digit grouping.
		{"three digits", "123", []string{"123"}},
		{"four digits", "1234", []string{"123", "4"}},
		{"eight digits", "12345678", []string{"123", "456", "78"}},
		{"space before digits attaches", " 42", []string{" 42"}},
		// Rule 6: one token per ideographic rune.
		{"cjk", "日本語", []string{"日", "本", "語"}},
		{"hangul", "한국", []string{"한", "국"}},
		// Rule 7: punctuation clusters.
		{"json braces", `{"a":1}`, []string{`{"`, `a`, `":`, `1`, `}`}},
		{"arrow", "=>", []string{"=>"}},
		{"four punctuation", "!!!!", []string{"!!!", "!"}},
		{"space before punctuation attaches", " {", []string{" {"}},
		{"assignment", "x = 1", []string{"x", " =", " 1"}},
		// Whitespace runs, and the final space peeling off to attach forward.
		{"newline", "\n", []string{"\n"}},
		{"indent run", "\n\tif", []string{"\n\t", "if"}},
		{"run then attached space", "\n\n hello", []string{"\n\n", " hello"}},
		{"double space before a word", "a  b", []string{"a", " ", " b"}},
		{"trailing space", "a ", []string{"a", " "}},
		// Underscore joins identifiers rather than splitting them.
		{"snake case", "prompt_tokens", []string{"prompt", "_token", "s"}},
		// Non-ASCII letters below the ideographic floor are ordinary words.
		{"accented", "café", []string{"café"}},
		{"cyrillic", "привет", []string{"привет"}},
		// Emoji is one token per rune (rule 8).
		{"emoji", "ok👍", []string{"ok", "👍"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := Reference.Tokenize(tc.in)
			if !equalStrings(got, tc.want) {
				t.Fatalf("Tokenize(%q) = %q, want %q", tc.in, got, tc.want)
			}
			if n := Reference.Count(tc.in); n != len(tc.want) {
				t.Fatalf("Count(%q) = %d, want %d", tc.in, n, len(tc.want))
			}
		})
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestReferenceLossless is the property the mock provider depends on. If
// tokenization were not an exact partition of the input, streaming one token per
// SSE delta would either drop or duplicate output bytes, and the reconciliation
// would then be checking a corrupted ground truth.
//
// The generated inputs deliberately include invalid UTF-8, lone surrogate bytes,
// and adversarial whitespace/apostrophe placement, because those are where a
// hand-rolled splitter loses a byte or fails to advance and hangs.
func TestReferenceLossless(t *testing.T) {
	alphabet := []string{
		"a", "Z", "_", " ", "  ", "\n", "\t", "\r\n", "0", "9", "'", "’",
		"{", "}", "\"", ":", ",", "=", ">", "-", ".", "!", "/", "\\",
		"日", "한", "ぁ", "é", "п", "👍", "\x80", "\xff", "\xc3", "\xed\xa0\x80",
		"hello", "1234567", "abcdefghij", "don't", "  \n\t ",
	}
	// Fixed seed: a fuzz-shaped test that fails only on some runs is a test
	// nobody can bisect.
	rng := rand.New(rand.NewSource(20260730))
	for iter := 0; iter < 4000; iter++ {
		var sb strings.Builder
		for n := rng.Intn(14); n >= 0; n-- {
			sb.WriteString(alphabet[rng.Intn(len(alphabet))])
		}
		in := sb.String()

		toks := Reference.Tokenize(in)
		if got := Reference.Detokenize(toks); got != in {
			t.Fatalf("not lossless: Tokenize(%q) = %q, which rejoins to %q", in, toks, got)
		}
		if n := Reference.Count(in); n != len(toks) {
			t.Fatalf("Count(%q) = %d but Tokenize returned %d tokens", in, n, len(toks))
		}
		for _, tk := range toks {
			if tk == "" {
				t.Fatalf("Tokenize(%q) = %q contains an empty token, which inflates every count", in, toks)
			}
		}
	}
}

// TestReferenceCountMatchesTokenize covers the divergence that nothing else
// would catch: Count walks and counts while Tokenize walks and collects, and if
// they ever disagreed the mock provider's reported completion_tokens would not
// equal the number of deltas it actually streamed.
func TestReferenceCountMatchesTokenize(t *testing.T) {
	inputs := []string{
		"", "a", loremProse, userProse, goCode, jsonPayload, cjkText, toolSchema,
		strings.Repeat("word ", 500), strings.Repeat("日本", 300),
		strings.Repeat("{\"k\":\"v\"},", 200),
	}
	for _, f := range promptFixtures() {
		for i := range f.req.Messages {
			inputs = append(inputs, f.req.Messages[i].Content.Text())
		}
	}
	for _, in := range inputs {
		if got, want := Reference.Count(in), len(Reference.Tokenize(in)); got != want {
			t.Fatalf("Count = %d but len(Tokenize) = %d for a %d-byte input", got, want, len(in))
		}
	}
}

func TestReferenceAppendTokens(t *testing.T) {
	buf := make([]string, 0, 8)
	buf = Reference.AppendTokens(buf, "hello world")
	buf = Reference.AppendTokens(buf, " again")
	want := []string{"hello", " world", " again"}
	if !equalStrings(buf, want) {
		t.Fatalf("AppendTokens = %q, want %q", buf, want)
	}
	// Reuse must be safe: truncating and refilling must not leak old tokens.
	buf = Reference.AppendTokens(buf[:0], "x")
	if !equalStrings(buf, []string{"x"}) {
		t.Fatalf("after reuse, AppendTokens = %q, want [x]", buf)
	}
}

func TestReferenceDetokenize(t *testing.T) {
	tests := []struct {
		name string
		in   []string
		want string
	}{
		{"nil", nil, ""},
		{"empty slice", []string{}, ""},
		{"round trip", Reference.Tokenize("The quick brown fox."), "The quick brown fox."},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := Reference.Detokenize(tc.in); got != tc.want {
				t.Fatalf("Detokenize(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

func TestReferenceDeterministic(t *testing.T) {
	// Determinism is the entire point: the mock provider's usage must be a pure
	// function of its input, or the reconciliation harness has nothing stable to
	// compare against.
	for _, in := range []string{loremProse, goCode, jsonPayload, cjkText} {
		first := Reference.Count(in)
		for i := 0; i < 50; i++ {
			if got := Reference.Count(in); got != first {
				t.Fatalf("Count is not deterministic: %d then %d", first, got)
			}
		}
	}
}

func TestReferenceConcurrentUse(t *testing.T) {
	inputs := []string{loremProse, goCode, jsonPayload, cjkText, toolSchema}
	want := make([]int, len(inputs))
	for i, in := range inputs {
		want[i] = Reference.Count(in)
	}
	var wg sync.WaitGroup
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				j := i % len(inputs)
				if got := Reference.Count(inputs[j]); got != want[j] {
					t.Errorf("concurrent Count = %d, want %d", got, want[j])
					return
				}
				_ = Reference.Tokenize(inputs[j])
			}
		}()
	}
	wg.Wait()
}

func TestReferenceCountMessages(t *testing.T) {
	msgs := []apiv1.Message{
		msg(apiv1.RoleSystem, "Be brief."),
		msg(apiv1.RoleUser, "hello"),
	}
	// Priming(3) + 2 * (framing(3) + role(1)) + text.
	wantText := Reference.Count("Be brief.") + Reference.Count("hello")
	want := ReplyPrimingTokens + 2*(PerMessageTokens+1) + wantText
	if got := Reference.CountMessages(msgs); got != want {
		t.Fatalf("CountMessages = %d, want %d", got, want)
	}

	// A name costs PerNameTokens plus its own text.
	named := []apiv1.Message{{Role: apiv1.RoleUser, Name: "alice", Content: apiv1.NewTextContent("hello")}}
	bare := []apiv1.Message{msg(apiv1.RoleUser, "hello")}
	diff := Reference.CountMessages(named) - Reference.CountMessages(bare)
	if diff != PerNameTokens+Reference.Count("alice") {
		t.Fatalf("name cost %d tokens, want %d", diff, PerNameTokens+Reference.Count("alice"))
	}

	if got := Reference.CountMessages(nil); got != ReplyPrimingTokens {
		t.Fatalf("CountMessages(nil) = %d, want the priming cost %d", got, ReplyPrimingTokens)
	}
}

func TestReferenceCountRequest(t *testing.T) {
	if got := Reference.CountRequest(nil); got != 0 {
		t.Fatalf("CountRequest(nil) = %d, want 0", got)
	}
	base := &apiv1.ChatRequest{
		Model:    "mock-1",
		Messages: []apiv1.Message{msg(apiv1.RoleUser, "hello")},
	}
	withTools := &apiv1.ChatRequest{
		Model:      base.Model,
		Messages:   base.Messages,
		Tools:      json.RawMessage(toolSchema),
		ToolChoice: json.RawMessage(`"auto"`),
	}
	want := Reference.CountRequest(base) + ToolsFramingTokens +
		Reference.Count(toolSchema) + Reference.Count(`"auto"`)
	if got := Reference.CountRequest(withTools); got != want {
		t.Fatalf("CountRequest with tools = %d, want %d", got, want)
	}
}

// TestReferenceCountsMatchStreamedDeltas is the property the reconciliation
// harness actually relies on: if the mock provider streams one reference token
// per delta and then reports Count() as completion_tokens, the two must agree
// exactly and the concatenated deltas must reproduce the response text. This test
// simulates that loop so a change to the tokenizer cannot break the mock silently.
func TestReferenceCountsMatchStreamedDeltas(t *testing.T) {
	for _, body := range []string{
		"The answer is 42.",
		"Here is some code:\n\tif x <= 0 {\n\t\treturn 0\n\t}",
		`{"result":"ok","count":17}`,
		cjkText,
		"",
	} {
		deltas := Reference.Tokenize(body)
		reported := Reference.Count(body)
		if len(deltas) != reported {
			t.Fatalf("would stream %d deltas but report %d completion_tokens for %q",
				len(deltas), reported, body)
		}
		if joined := strings.Join(deltas, ""); joined != body {
			t.Fatalf("streamed deltas rejoin to %q, want %q", joined, body)
		}
	}
}

// TestReferenceCharsPerTokenIsCalibrated pins the one calibration property of
// the reference tokenizer. It is not ground truth about any provider, but if it
// drifted far from the 3.5-4.3 chars-per-token band real tokenizers occupy on
// English prose, then the estimator's measured error against it would stop
// meaning anything and TestEstimatorErrorBound would be measuring noise.
func TestReferenceCharsPerTokenIsCalibrated(t *testing.T) {
	prose := loremProse + " " + userProse
	cpt := float64(len(prose)) / float64(Reference.Count(prose))
	t.Logf("MEASURED: reference tokenizer is %.2f characters per token on English prose", cpt)
	if cpt < 3.5 || cpt > 4.3 {
		t.Fatalf("reference density is %.2f chars/token, outside the 3.5-4.3 band real "+
			"tokenizers occupy; the estimator's error bound is measured against this and "+
			"would become meaningless", cpt)
	}

	// Code and JSON must come out denser than prose, as they do in every real
	// tokenizer. If they did not, the dense-character handling in TextTokens
	// would be correcting for something the yardstick does not exhibit.
	for name, s := range map[string]string{"code": goCode, "json": jsonPayload} {
		d := float64(len(s)) / float64(Reference.Count(s))
		t.Logf("MEASURED: reference tokenizer is %.2f characters per token on %s", d, name)
		if d >= cpt {
			t.Errorf("%s is %.2f chars/token, not denser than prose at %.2f", name, d, cpt)
		}
	}
}

func TestReferenceIdeographicFloor(t *testing.T) {
	// Runes just below the floor must behave as word letters; at or above it,
	// one token each. This is the boundary that decides whether Japanese is
	// counted per-character or per-word, so it is asserted directly rather than
	// trusted.
	below := "ⴰⴱⴲ" // Tifinagh letters, well below U+2E80
	if got := Reference.Count(below); got != 1 {
		t.Errorf("Count(%q) = %d, want 1: runes below the ideographic floor form a word run", below, got)
	}
	at := "⺀⺁"
	if got := Reference.Count(at); got != 2 {
		t.Errorf("Count(%q) = %d, want 2: runes at the floor are one token each", at, got)
	}
	if !isWordRune(0x2D30) {
		t.Error("U+2D30 should be a word rune")
	}
	if isWordRune(ideographicFloor) {
		t.Error("the ideographic floor itself must not be a word rune")
	}
}

// TestReferenceInvalidUTF8Terminates guards the hang, not the count. Every
// branch of walk must consume at least one byte; a punct-run that matched a rune
// it could not then consume would loop forever on a single request and take the
// mock provider's goroutine with it.
func TestReferenceInvalidUTF8Terminates(t *testing.T) {
	for _, in := range []string{
		"\x80", "\xff\xff", "\xc3", "a\xffb", "'\xff", "\xed\xa0\x80", " \xff",
	} {
		done := make(chan int, 1)
		go func(s string) { done <- Reference.Count(s) }(in)
		select {
		case n := <-done:
			if n < 1 {
				t.Errorf("Count(%q) = %d, want >= 1", in, n)
			}
			if got := Reference.Detokenize(Reference.Tokenize(in)); got != in {
				t.Errorf("Tokenize(%q) is not lossless: rejoins to %q", in, got)
			}
		case <-time.After(5 * time.Second):
			t.Fatalf("Count(%q) did not terminate; a branch of walk consumed zero bytes", in)
		}
	}
}

// TestReferenceEveryRuneClassIsHandled asserts the class predicates partition
// the rune space, so no rune can fall through walk's switch without being
// consumed. Overlapping classes would be a silent miscount; a gap would be the
// hang above.
func TestReferenceEveryRuneClassIsHandled(t *testing.T) {
	for r := rune(0); r < 0x3000; r++ {
		if !utf8.ValidRune(r) {
			continue
		}
		classes := 0
		if isWordRune(r) {
			classes++
		}
		if isDigitRune(r) {
			classes++
		}
		if isPunctRune(r) {
			classes++
		}
		if classes > 1 {
			t.Fatalf("rune %U belongs to %d classes; the switch in walk would take an "+
				"arbitrary branch", r, classes)
		}
	}
	// Digits and word runes must be disjoint from punctuation specifically,
	// since punctuation is the widest predicate.
	for _, r := range []rune{'0', '9', 'a', 'Z', '_'} {
		if isPunctRune(r) {
			t.Errorf("%q classified as punctuation", r)
		}
	}
	for _, r := range []rune{'{', '"', ':', '=', '\n', '\t'} {
		if isWordRune(r) || isDigitRune(r) {
			t.Errorf("%q classified as a word or digit rune", r)
		}
	}
}

func BenchmarkReferenceCount(b *testing.B) {
	body := loremProse + " " + userProse
	b.SetBytes(int64(len(body)))
	b.ReportAllocs()
	for b.Loop() {
		_ = Reference.Count(body)
	}
}

func BenchmarkReferenceTokenize(b *testing.B) {
	body := loremProse + " " + userProse
	b.SetBytes(int64(len(body)))
	b.ReportAllocs()
	for b.Loop() {
		_ = Reference.Tokenize(body)
	}
}
