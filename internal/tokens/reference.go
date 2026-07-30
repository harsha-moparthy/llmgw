package tokens

import (
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
)

// ReferenceTokenizer is a deterministic word/subword splitter.
//
// It is NOT a BPE tokenizer and it is not an attempt to reproduce one. It has no
// vocabulary, no merge table, and no training data; it will disagree with
// tiktoken, with Anthropic's counting endpoint, and with SentencePiece on real
// text, sometimes substantially. Nothing in this gateway bills a customer from
// its output.
//
// Its purpose is narrower and it is worth stating exactly, because a "reference
// tokenizer" that was mistaken for a real one would be a serious trap: the mock
// provider needs token counts that are *reproducible*, so that the cost
// reconciliation harness has an exact ground truth to check the ledger against.
// If the mock invented plausible-looking counts, a reconciliation that "passed"
// would prove only that two guesses agreed. With this tokenizer the mock's
// reported usage is a pure function of the bytes it received and emitted, so a
// mismatch in the ledger is unambiguously a ledger bug.
//
// The rules, in full — the point is that they fit in a comment, which a real BPE
// never could:
//
//  1. Every byte of the input lands in exactly one token, in order, so
//     strings.Join(Tokenize(s), "") == s for any s, including invalid UTF-8.
//     Losslessness is what lets the mock provider stream one reference token per
//     SSE delta and have the delta count equal the reported completion_tokens.
//  2. A single space attaches to whatever follows it — a word, a number, or a
//     punctuation cluster — so " the" and " {" are each one token. This mirrors
//     the one behaviour of GPT-family tokenizers that most affects counts, and
//     it must apply to punctuation too: charging a separate token for the space
//     in `x = 1` makes this tokenizer 50% denser than any real one on source
//     code, which is a large share of the traffic a gateway sees. A run of two
//     or more whitespace characters is one token, less its final space when that
//     space attaches forward.
//  3. A run of letters is one token, split into chunks of at most
//     ReferenceMaxWordRunes runes so that a very long identifier costs more than
//     a short word. The chunk size is 6 rather than a rounder number because it
//     is the value that puts this tokenizer at ~3.96 characters per token on
//     English prose, which is the middle of the band published tokenizers
//     occupy. That is the one property worth calibrating: a ground truth that
//     was 30% denser than every real tokenizer would make the estimator's
//     measured error meaningless.
//  4. An apostrophe between letters attaches to the letters after it, so
//     "don't" is "don" + "'t" — again matching observed BPE behaviour on
//     contractions, which are frequent enough in chat traffic to matter.
//  5. A run of ASCII digits is split into chunks of at most
//     ReferenceMaxDigitRunes, which is what cl100k_base and o200k_base do.
//  6. Ideographic and syllabic scripts (Han, Kana, Hangul, and everything else
//     at or above U+2E80) are one token per rune. Real tokenizers land between
//     0.5 and 2 tokens per CJK character; one is the honest middle.
//  7. A run of ASCII punctuation and symbols is split into chunks of at most
//     ReferenceMaxPunctRunes. Not one-per-rune: real BPE vocabularies contain
//     the punctuation clusters that JSON and code are made of (`":"`, `},`,
//     `=>`), and one-token-per-punctuation-mark would over-count a JSON payload
//     by nearly a factor of two — which would then make this tokenizer a
//     misleading yardstick for the estimator.
//  8. Everything else — non-ASCII symbols, emoji, invalid bytes — is one token
//     per rune.
//
// The zero value is ready to use; there is no state, which is also why it is
// trivially safe for concurrent use.
type ReferenceTokenizer struct{}

// Reference is the shared instance. Stateless, so one value serves the whole
// process.
var Reference ReferenceTokenizer

// Chunk limits for rules 3, 5 and 7. Exported so a test asserting the boundary
// cannot drift from the implementation by hardcoding the number.
const (
	ReferenceMaxWordRunes  = 6
	ReferenceMaxDigitRunes = 3
	ReferenceMaxPunctRunes = 3
)

// ideographicFloor is the code point at or above which a letter is treated as
// one token per rune. U+2E80 is the start of the CJK Radicals Supplement, the
// first block of the contiguous CJK/Kana/Hangul region; every alphabetic script
// whose words are letter-runs (Latin, Greek, Cyrillic, Hebrew, Arabic,
// Devanagari, ...) sits below it.
const ideographicFloor = 0x2E80

// Count returns the number of reference tokens in s without allocating a slice.
//
// This is the function on the mock provider's hot path, so it walks the string
// once and counts, rather than tokenizing and taking the length. The two must
// agree exactly; TestReferenceCountMatchesTokenize asserts it on every fixture,
// because a divergence between them would corrupt the reconciliation ground
// truth in a way nothing else would catch.
func (t ReferenceTokenizer) Count(s string) int {
	n := 0
	t.walk(s, func(string) { n++ })
	return n
}

// Tokenize splits s into reference tokens. The concatenation of the result is s.
func (t ReferenceTokenizer) Tokenize(s string) []string {
	if s == "" {
		return nil
	}
	// One token per ~4 bytes is the observed density on prose; over-allocating
	// slightly beats growing the slice repeatedly for a long completion.
	out := make([]string, 0, len(s)/4+8)
	t.walk(s, func(tok string) { out = append(out, tok) })
	return out
}

// AppendTokens appends the tokens of s to dst, letting a caller that tokenizes
// many strings (the mock provider building a scripted response) reuse a buffer.
func (t ReferenceTokenizer) AppendTokens(dst []string, s string) []string {
	t.walk(s, func(tok string) { dst = append(dst, tok) })
	return dst
}

// walk is the single implementation of the tokenization rules. Count and
// Tokenize both go through it so they cannot drift apart.
//
// The callback is a func rather than an interface method so the compiler can
// inline the counting case; the closure does not escape.
func (t ReferenceTokenizer) walk(s string, emit func(string)) {
	i := 0
	for i < len(s) {
		start := i
		r, size := utf8.DecodeRuneInString(s[i:])

		switch {
		case unicode.IsSpace(r):
			// Rule 2. A lone space glues to a following word or digit run;
			// every other whitespace run is a token of its own.
			if r == ' ' {
				next, nsize := utf8.DecodeRuneInString(s[i+size:])
				if nsize > 0 && isAttachable(next) {
					i += size
					i = t.consumeAttached(s, start, i, next, emit)
					continue
				}
			}
			i += size
			for i < len(s) {
				r2, sz := utf8.DecodeRuneInString(s[i:])
				if !unicode.IsSpace(r2) || (r2 == ' ' && startsAttachable(s[i+sz:])) {
					break
				}
				i += sz
			}
			emit(s[start:i])

		case isWordRune(r):
			i = t.emitWordRun(s, start, i, emit)

		case isDigitRune(r):
			i = t.emitDigitRun(s, start, i, emit)

		case isPunctRune(r), r == '\'':
			// A stray apostrophe with no letter before it (rule 4 does not
			// apply) is ordinary punctuation.
			i = t.emitPunctRun(s, start, start, emit)

		default:
			// Rules 6 and 8 coincide: one token per rune. Ideographic letters
			// reach here because isWordRune excludes them, as do emoji and the
			// U+FFFD that DecodeRuneInString yields for an invalid byte.
			i += size
			emit(s[start:i])
		}
	}
}

// startsAttachable reports whether the text after a space begins with something
// the space would attach to, which is how a whitespace run knows to stop before
// its final space: "\n\n hello" is "\n\n" + " hello", not "\n\n " + "hello".
func startsAttachable(rest string) bool {
	r, size := utf8.DecodeRuneInString(rest)
	if size == 0 {
		return false
	}
	return isAttachable(r)
}

// isAttachable reports whether a preceding space glues onto r. Everything a
// space can precede in text qualifies except more whitespace and the
// one-token-per-rune classes (ideographic, emoji), where a merged token would
// not correspond to anything a real vocabulary holds.
func isAttachable(r rune) bool {
	return isWordRune(r) || isDigitRune(r) || isPunctRune(r) || r == '\''
}

// consumeAttached emits the token beginning at tokStart, whose word or digit run
// begins at runStart.
func (t ReferenceTokenizer) consumeAttached(s string, tokStart, runStart int, first rune, emit func(string)) int {
	switch {
	case isDigitRune(first):
		return t.emitDigitRun(s, tokStart, runStart, emit)
	case isWordRune(first):
		return t.emitWordRun(s, tokStart, runStart, emit)
	default:
		return t.emitPunctRun(s, tokStart, runStart, emit)
	}
}

// emitWordRun consumes a letter run starting at runStart, emitting chunks of at
// most ReferenceMaxWordRunes runes. The first chunk starts at tokStart so an
// attached leading space is included in it rather than being counted twice.
func (t ReferenceTokenizer) emitWordRun(s string, tokStart, runStart int, emit func(string)) int {
	i := runStart
	chunkStart := tokStart
	runes := 0
	sawWord := false
	for i < len(s) {
		r, sz := utf8.DecodeRuneInString(s[i:])
		if !isWordRune(r) {
			// Rule 4: an apostrophe with a letter on each side starts a new
			// token rather than ending the word. sawWord rather than runes > 0,
			// because runes resets at a chunk boundary and "abcdefgh's" should
			// still split as "abcdefgh" + "'s".
			if isApostrophe(r) && sawWord {
				if nr, nsz := utf8.DecodeRuneInString(s[i+sz:]); nsz > 0 && isWordRune(nr) {
					// i == chunkStart when the run just closed a full chunk
					// ("abcdefgh's"); emitting there would produce an empty
					// token and inflate the count.
					if i > chunkStart {
						emit(s[chunkStart:i])
					}
					// The apostrophe leads the next chunk ("'t"), so reset the
					// chunk to start at it and keep going.
					chunkStart = i
					i += sz
					runes = 0
					continue
				}
			}
			break
		}
		i += sz
		runes++
		sawWord = true
		if runes == ReferenceMaxWordRunes {
			emit(s[chunkStart:i])
			chunkStart = i
			runes = 0
		}
	}
	if i > chunkStart {
		emit(s[chunkStart:i])
	}
	return i
}

// emitDigitRun consumes a digit run, emitting chunks of at most
// ReferenceMaxDigitRunes digits.
func (t ReferenceTokenizer) emitDigitRun(s string, tokStart, runStart int, emit func(string)) int {
	i := runStart
	chunkStart := tokStart
	digits := 0
	for i < len(s) {
		r, sz := utf8.DecodeRuneInString(s[i:])
		if !isDigitRune(r) {
			break
		}
		i += sz
		digits++
		if digits == ReferenceMaxDigitRunes {
			emit(s[chunkStart:i])
			chunkStart = i
			digits = 0
		}
	}
	if i > chunkStart {
		emit(s[chunkStart:i])
	}
	return i
}

// emitPunctRun consumes a run of ASCII punctuation, emitting chunks of at most
// ReferenceMaxPunctRunes runes. Every rune here is one byte, so the chunking can
// index directly.
func (t ReferenceTokenizer) emitPunctRun(s string, tokStart, runStart int, emit func(string)) int {
	i := runStart
	chunkStart := tokStart
	n := 0
	for i < len(s) && isPunctOrASCIIQuote(rune(s[i])) {
		i++
		n++
		if n == ReferenceMaxPunctRunes {
			emit(s[chunkStart:i])
			chunkStart = i
			n = 0
		}
	}
	if i > chunkStart {
		emit(s[chunkStart:i])
	}
	return i
}

// isWordRune reports whether r participates in a letter run. Underscore is
// included so that snake_case identifiers — which fill the code-heavy prompts
// this gateway sees — tokenize as one run rather than three.
func isWordRune(r rune) bool {
	if r < utf8.RuneSelf {
		return r == '_' || (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z')
	}
	if r >= ideographicFloor {
		return false
	}
	return unicode.IsLetter(r)
}

// isDigitRune is ASCII-only on purpose: digit grouping is a property of the
// tokenizers being imitated, which group ASCII digits and treat other numeral
// systems as ordinary symbols.
func isDigitRune(r rune) bool { return r >= '0' && r <= '9' }

// isPunctOrASCIIQuote is isPunctRune widened to include the straight
// apostrophe, which emitPunctRun must be able to consume: rule 4 only claims an
// apostrophe that has a letter before it and a letter after it, and a stray one
// ("'tis", "x'") is ordinary punctuation. Without this the punct run would
// consume zero bytes and walk would spin forever — the sort of hang that only a
// fuzz-shaped input finds, which is why TestReferenceLossless runs one.
func isPunctOrASCIIQuote(r rune) bool { return isPunctRune(r) || r == '\'' }

// isPunctRune reports whether r is ASCII punctuation or an ASCII symbol.
// Underscore is excluded (it is a word rune) and so is the apostrophe, which
// rule 4 owns; a stray apostrophe with no letter before it still reaches here
// via the default branch as a single token.
func isPunctRune(r rune) bool {
	if r >= utf8.RuneSelf || r <= ' ' || r == '_' || r == '\'' {
		return false
	}
	return !isDigitRune(r) && !(r >= 'a' && r <= 'z') && !(r >= 'A' && r <= 'Z')
}

// isApostrophe covers the straight quote and the typographic one, because chat
// clients emit both and they should count the same.
func isApostrophe(r rune) bool { return r == '\'' || r == '’' }

// CountMessages returns the reference token count of a message list including
// the same structural framing EstimatePrompt applies.
//
// It shares the framing constants with the estimator deliberately: the mock
// provider's usage and the gateway's estimate then differ only in how the *text*
// was counted, which is the one variable the error-bound test is trying to
// isolate.
func (t ReferenceTokenizer) CountMessages(msgs []apiv1.Message) int {
	total := ReplyPrimingTokens
	for i := range msgs {
		m := &msgs[i]
		total += PerMessageTokens
		total += roleTokens(m.Role, DefaultRatio)
		if m.Name != "" {
			total += PerNameTokens + t.Count(m.Name)
		}
		total += t.Count(m.Content.Text())
		if m.ToolCallID != "" {
			total += t.Count(m.ToolCallID)
		}
		if len(m.ToolCalls) > 0 {
			total += t.Count(string(m.ToolCalls))
		}
	}
	return total
}

// CountRequest returns the reference prompt token count of a whole request,
// counting exactly the fields EstimatePrompt counts. This is what the mock
// provider reports as prompt_tokens.
func (t ReferenceTokenizer) CountRequest(req *apiv1.ChatRequest) int {
	if req == nil {
		return 0
	}
	total := t.CountMessages(req.Messages)
	if len(req.Tools) > 0 {
		total += ToolsFramingTokens + t.Count(string(req.Tools))
	}
	if len(req.ToolChoice) > 0 {
		total += t.Count(string(req.ToolChoice))
	}
	if len(req.ResponseFormat) > 0 {
		total += t.Count(string(req.ResponseFormat))
	}
	return total
}

// Detokenize concatenates tokens back into text. Exact inverse of Tokenize.
func (t ReferenceTokenizer) Detokenize(toks []string) string {
	if len(toks) == 0 {
		return ""
	}
	n := 0
	for _, tk := range toks {
		n += len(tk)
	}
	var sb strings.Builder
	sb.Grow(n)
	for _, tk := range toks {
		sb.WriteString(tk)
	}
	return sb.String()
}
