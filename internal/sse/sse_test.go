package sse

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// byteReader returns exactly one byte per Read call.
//
// This is the single most valuable test fixture in the package. Every SSE
// decoder works when fed a whole frame at once; the bugs live at Read
// boundaries, and a one-byte reader puts a boundary between every pair of bytes
// in the input, including inside "data:", inside a UTF-8 rune, and between CR
// and LF.
type byteReader struct {
	data []byte
	pos  int
}

func (b *byteReader) Read(p []byte) (int, error) {
	if b.pos >= len(b.data) {
		return 0, io.EOF
	}
	if len(p) == 0 {
		return 0, nil
	}
	p[0] = b.data[b.pos]
	b.pos++
	return 1, nil
}

// chunkReader returns bytes in a repeating cycle of pathological chunk sizes,
// so boundaries fall at irregular offsets rather than the regular ones a
// one-byte reader produces.
type chunkReader struct {
	data   []byte
	pos    int
	sizes  []int
	sizeIx int
}

func (c *chunkReader) Read(p []byte) (int, error) {
	if c.pos >= len(c.data) {
		return 0, io.EOF
	}
	n := c.sizes[c.sizeIx%len(c.sizes)]
	c.sizeIx++
	if n <= 0 {
		n = 1
	}
	if n > len(p) {
		n = len(p)
	}
	if c.pos+n > len(c.data) {
		n = len(c.data) - c.pos
	}
	copy(p, c.data[c.pos:c.pos+n])
	c.pos += n
	return n, nil
}

func TestDecoderBasicFrames(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  []Event
	}{
		{
			name:  "single data frame",
			input: "data: hello\n\n",
			want:  []Event{{Data: []byte("hello")}},
		},
		{
			name:  "no space after colon",
			input: "data:hello\n\n",
			want:  []Event{{Data: []byte("hello")}},
		},
		{
			name: "only one leading space is stripped",
			// A payload whose own first character is a space must keep it.
			input: "data:  hello\n\n",
			want:  []Event{{Data: []byte(" hello")}},
		},
		{
			name:  "multi-line data joined with newline",
			input: "data: line1\ndata: line2\ndata: line3\n\n",
			want:  []Event{{Data: []byte("line1\nline2\nline3")}},
		},
		{
			name:  "event type",
			input: "event: content_block_delta\ndata: {}\n\n",
			want:  []Event{{Type: "content_block_delta", Data: []byte("{}")}},
		},
		{
			name:  "id and retry",
			input: "id: 42\nretry: 3000\ndata: x\n\n",
			want:  []Event{{ID: "42", Retry: 3 * time.Second, HasRetry: true, Data: []byte("x")}},
		},
		{
			name:  "done sentinel is flagged",
			input: "data: [DONE]\n\n",
			want:  []Event{{Data: []byte("[DONE]"), Done: true}},
		},
		{
			name:  "comment keepalive",
			input: ": ping\n\n",
			want:  []Event{{Comment: "ping"}},
		},
		{
			name:  "CRLF line endings",
			input: "data: hello\r\n\r\n",
			want:  []Event{{Data: []byte("hello")}},
		},
		{
			name:  "lone CR line endings",
			input: "data: hello\r\r",
			want:  []Event{{Data: []byte("hello")}},
		},
		{
			name:  "empty data value",
			input: "data:\n\n",
			want:  []Event{{Data: []byte("")}},
		},
		{
			name:  "unknown field ignored",
			input: "flavour: vanilla\ndata: x\n\n",
			want:  []Event{{Data: []byte("x")}},
		},
		{
			name:  "leading blank lines ignored",
			input: "\n\n\ndata: x\n\n",
			want:  []Event{{Data: []byte("x")}},
		},
		{
			name:  "two frames",
			input: "data: a\n\ndata: b\n\n",
			want:  []Event{{Data: []byte("a")}, {Data: []byte("b")}},
		},
		{
			name:  "non-numeric retry ignored not fatal",
			input: "retry: soon\ndata: x\n\n",
			want:  []Event{{Data: []byte("x")}},
		},
		{
			// An event with no terminating blank line is truncation, not a
			// final event, so it yields nothing. The rule is uniform — see
			// TestDecoderTruncationIsNotCleanEOF for why this project prefers
			// one consistent rule over a "last event may omit its terminator"
			// special case.
			name:  "unterminated final frame yields no event",
			input: "data: x\n",
			want:  nil,
		},
	}

	// Each case runs against three readers: a plain buffer, a one-byte reader,
	// and an irregular-chunk reader. A decoder that passes only the first is the
	// broken-in-production case this triple exists to catch.
	readers := []struct {
		name string
		make func(string) io.Reader
	}{
		{"buffer", func(s string) io.Reader { return strings.NewReader(s) }},
		{"one-byte", func(s string) io.Reader { return &byteReader{data: []byte(s)} }},
		{"irregular", func(s string) io.Reader {
			return &chunkReader{data: []byte(s), sizes: []int{1, 3, 2, 7, 1, 5}}
		}},
	}

	for _, tc := range tests {
		for _, rd := range readers {
			t.Run(tc.name+"/"+rd.name, func(t *testing.T) {
				d := NewDecoder(rd.make(tc.input))
				var got []Event
				for {
					ev, err := d.Next()
					if err != nil {
						if !errors.Is(err, io.EOF) && !errors.Is(err, ErrTruncated) {
							t.Fatalf("Next: unexpected error: %v", err)
						}
						break
					}
					got = append(got, *ev)
				}
				if len(got) != len(tc.want) {
					t.Fatalf("got %d events, want %d: %+v", len(got), len(tc.want), got)
				}
				for i := range tc.want {
					if got[i].Type != tc.want[i].Type {
						t.Errorf("event %d Type = %q, want %q", i, got[i].Type, tc.want[i].Type)
					}
					if !bytes.Equal(got[i].Data, tc.want[i].Data) {
						t.Errorf("event %d Data = %q, want %q", i, got[i].Data, tc.want[i].Data)
					}
					if got[i].ID != tc.want[i].ID {
						t.Errorf("event %d ID = %q, want %q", i, got[i].ID, tc.want[i].ID)
					}
					if got[i].Retry != tc.want[i].Retry {
						t.Errorf("event %d Retry = %v, want %v", i, got[i].Retry, tc.want[i].Retry)
					}
					if got[i].Done != tc.want[i].Done {
						t.Errorf("event %d Done = %v, want %v", i, got[i].Done, tc.want[i].Done)
					}
					if got[i].Comment != tc.want[i].Comment {
						t.Errorf("event %d Comment = %q, want %q", i, got[i].Comment, tc.want[i].Comment)
					}
				}
			})
		}
	}
}

// TestDecoderSplitMidUTF8 checks that a multi-byte rune split across Read calls
// survives. The decoder is byte-oriented so this should hold by construction,
// but "should hold by construction" is exactly the claim that deserves a test:
// a future optimisation that decodes runes would break it silently, producing
// replacement characters in customer-visible model output.
func TestDecoderSplitMidUTF8(t *testing.T) {
	payload := "héllo → 世界 🎉"
	input := "data: " + payload + "\n\n"
	d := NewDecoder(&byteReader{data: []byte(input)})
	ev, err := d.Next()
	if err != nil {
		t.Fatalf("Next: %v", err)
	}
	if string(ev.Data) != payload {
		t.Errorf("Data = %q, want %q", ev.Data, payload)
	}
}

// TestDecoderTruncationIsNotCleanEOF is the test that protects the gateway's
// central honesty property: a stream that dies mid-event must be reported as a
// failure. If this returns io.EOF instead, a half-generated completion is served
// and billed as a whole one.
func TestDecoderTruncationIsNotCleanEOF(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantErr error
	}{
		{
			name:    "clean end after terminated event",
			input:   "data: a\n\n",
			wantErr: io.EOF,
		},
		{
			name: "truncated: field line then EOF with no blank line",
			// "data: a\n" alone is returned as a final event (a provider may omit
			// the last blank line), so truncation is tested with a partial
			// SECOND event after a complete first one.
			input:   "data: a\n\nevent: partial\n",
			wantErr: ErrTruncated,
		},
		{
			name:    "truncated mid-field-name",
			input:   "data: a\n\ndat",
			wantErr: ErrTruncated,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			d := NewDecoder(strings.NewReader(tc.input))
			var lastErr error
			for {
				_, err := d.Next()
				if err != nil {
					lastErr = err
					break
				}
			}
			if !errors.Is(lastErr, tc.wantErr) {
				t.Errorf("terminal error = %v, want %v", lastErr, tc.wantErr)
			}
		})
	}
}

// TestDecoderErrorIsSticky verifies a terminal error is returned on every
// subsequent call. A decoder that recovers after a truncation would let a caller
// loop forever on a dead connection.
func TestDecoderErrorIsSticky(t *testing.T) {
	d := NewDecoder(strings.NewReader("data: a\n\nevent: x\n"))
	if _, err := d.Next(); err != nil {
		t.Fatalf("first Next: %v", err)
	}
	for i := 0; i < 3; i++ {
		_, err := d.Next()
		if !errors.Is(err, ErrTruncated) {
			t.Fatalf("call %d after truncation: err = %v, want ErrTruncated", i, err)
		}
	}
}

func TestDecoderMaxEventSize(t *testing.T) {
	// A single data field larger than the bound.
	big := strings.Repeat("x", 200)
	d := NewDecoderSize(strings.NewReader("data: "+big+"\n\n"), 64)
	_, err := d.Next()
	if !errors.Is(err, ErrEventTooLarge) {
		t.Fatalf("err = %v, want ErrEventTooLarge", err)
	}

	// Accumulation across many small data fields must also be bounded, or the
	// limit is trivially bypassed by sending 10,000 one-byte data lines.
	var sb strings.Builder
	for i := 0; i < 500; i++ {
		sb.WriteString("data: xxxxxxxxxx\n")
	}
	sb.WriteString("\n")
	d2 := NewDecoderSize(strings.NewReader(sb.String()), 256)
	if _, err := d2.Next(); !errors.Is(err, ErrEventTooLarge) {
		t.Fatalf("accumulated: err = %v, want ErrEventTooLarge", err)
	}
}

// TestDecoderLineLongerThanBuffer exercises the ReadSlice/ErrBufferFull path,
// where a single line exceeds the internal 16 KiB read buffer.
func TestDecoderLineLongerThanBuffer(t *testing.T) {
	payload := strings.Repeat("abcdefghij", 5000) // 50 KB, over the 16 KiB buffer
	d := NewDecoder(strings.NewReader("data: " + payload + "\n\n"))
	ev, err := d.Next()
	if err != nil {
		t.Fatalf("Next: %v", err)
	}
	if string(ev.Data) != payload {
		t.Errorf("payload mangled: got %d bytes, want %d", len(ev.Data), len(payload))
	}
}

// TestDecoderReturnedDataIsNotAliased guards the buffer-reuse optimisation. If
// buildEvent handed out the internal buffer instead of a copy, the second
// event's decode would overwrite the first event's Data — a bug that manifests
// as one response's text appearing inside another's.
func TestDecoderReturnedDataIsNotAliased(t *testing.T) {
	d := NewDecoder(strings.NewReader("data: first\n\ndata: second-and-longer\n\n"))
	e1, err := d.Next()
	if err != nil {
		t.Fatalf("first: %v", err)
	}
	saved := string(e1.Data)
	if _, err := d.Next(); err != nil {
		t.Fatalf("second: %v", err)
	}
	if string(e1.Data) != saved {
		t.Errorf("first event's Data was mutated by the second decode: %q became %q", saved, e1.Data)
	}
}

func TestDecoderLastEventIDPersists(t *testing.T) {
	// Per the spec an event with no id inherits the last one seen. A gateway
	// forwarding ids for client-side resumption would otherwise drop them.
	d := NewDecoder(strings.NewReader("id: 7\ndata: a\n\ndata: b\n\n"))
	e1, _ := d.Next()
	e2, _ := d.Next()
	if e1.ID != "7" || e2.ID != "7" {
		t.Errorf("ids = %q, %q; want both 7", e1.ID, e2.ID)
	}
}

// TestRoundTrip is the encoder/decoder inverse property, including the payloads
// that break naive implementations.
func TestRoundTrip(t *testing.T) {
	payloads := []string{
		"hello",
		"",
		" leading space",
		"trailing space ",
		":",
		"::",
		": looks like a comment",
		"data: looks like a field",
		"line1\nline2",
		"line1\r\nline2",
		"line1\rline2",
		"\n",
		"\n\n",
		"trailing newline\n",
		"\nleading newline",
		`{"choices":[{"delta":{"content":"multi\nline JSON string"}}]}`,
		"[DONE]",
		"héllo → 世界 🎉",
		strings.Repeat("x", 5000),
	}
	for _, p := range payloads {
		t.Run(fmt.Sprintf("%q", truncate(p, 24)), func(t *testing.T) {
			var buf bytes.Buffer
			enc := NewEncoderNoFlush(&buf)
			if err := enc.WriteData([]byte(p)); err != nil {
				t.Fatalf("WriteData: %v", err)
			}
			dec := NewDecoder(bytes.NewReader(buf.Bytes()))
			ev, err := dec.Next()
			if err != nil {
				t.Fatalf("Next: %v (encoded: %q)", err, buf.String())
			}
			// The encoder normalises CR and CRLF terminators to LF, because SSE
			// data fields are joined with LF by every conforming decoder and
			// there is no way to express "this was a CR" in the wire format.
			// That is a documented, lossless-modulo-newline-flavour property,
			// not silent corruption, so the comparison normalises too.
			wantNorm := normalizeNewlines(p)
			if string(ev.Data) != wantNorm {
				t.Errorf("round trip lost data:\n  sent %q\n  got  %q\n  wire %q", p, ev.Data, buf.String())
			}
		})
	}
}

func normalizeNewlines(s string) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	return strings.ReplaceAll(s, "\r", "\n")
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// TestEncoderMultilinePayloadDoesNotSplitEvent is the specific bug this design
// guards against: writing a payload containing a newline as a single data line
// would terminate the event early, and the second half would be parsed as a new
// event. That desynchronises the entire stream from that point on.
func TestEncoderMultilinePayloadDoesNotSplitEvent(t *testing.T) {
	var buf bytes.Buffer
	enc := NewEncoderNoFlush(&buf)
	if err := enc.WriteData([]byte("first\nsecond")); err != nil {
		t.Fatal(err)
	}
	if err := enc.WriteData([]byte("third")); err != nil {
		t.Fatal(err)
	}
	dec := NewDecoder(bytes.NewReader(buf.Bytes()))
	var got []string
	for {
		ev, err := dec.Next()
		if err != nil {
			break
		}
		got = append(got, string(ev.Data))
	}
	want := []string{"first\nsecond", "third"}
	if len(got) != len(want) {
		t.Fatalf("got %d events %q, want %d %q", len(got), got, len(want), want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("event %d = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestEncoderRequiresFlusher(t *testing.T) {
	// A plain buffer is not flushable, and constructing a flushing encoder over
	// it must fail loudly rather than silently buffering a stream.
	if _, err := NewEncoder(&bytes.Buffer{}); !errors.Is(err, ErrNoFlusher) {
		t.Errorf("NewEncoder(non-flusher) err = %v, want ErrNoFlusher", err)
	}
	rec := httptest.NewRecorder()
	if _, err := NewEncoder(rec); err != nil {
		t.Errorf("NewEncoder(ResponseRecorder) err = %v, want nil", err)
	}
}

// TestEncoderFlushesEveryFrame verifies the per-frame flush. Without it the
// gateway is a buffering proxy that happens to speak SSE, which is the failure
// mode users experience as "the response appears all at once at the end".
func TestEncoderFlushesEveryFrame(t *testing.T) {
	fw := &countingFlushWriter{}
	enc, err := NewEncoder(fw)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 5; i++ {
		if err := enc.WriteData([]byte("tok")); err != nil {
			t.Fatal(err)
		}
	}
	if err := enc.WriteDone(); err != nil {
		t.Fatal(err)
	}
	if fw.flushes != 6 {
		t.Errorf("flushes = %d, want 6 (one per frame)", fw.flushes)
	}
}

type countingFlushWriter struct {
	bytes.Buffer
	flushes int
}

func (c *countingFlushWriter) Flush() { c.flushes++ }

func TestEncoderWrittenCount(t *testing.T) {
	// The router's mid-stream failover decision depends on this count being
	// exact: a non-zero value means bytes reached the client and a transparent
	// retry is no longer possible.
	var buf bytes.Buffer
	enc := NewEncoderNoFlush(&buf)
	if enc.Written() != 0 {
		t.Fatalf("Written() = %d before any write, want 0", enc.Written())
	}
	if err := enc.WriteData([]byte("hello")); err != nil {
		t.Fatal(err)
	}
	if got, want := enc.Written(), int64(buf.Len()); got != want {
		t.Errorf("Written() = %d, want %d (actual bytes emitted)", got, want)
	}
}

func TestEncoderCommentAndEventSanitisation(t *testing.T) {
	// A newline in an event name or comment would inject an extra field line and
	// let a value forge a frame. Sanitisation must neutralise it.
	var buf bytes.Buffer
	enc := NewEncoderNoFlush(&buf)
	if err := enc.WriteEvent("evil\ndata: injected", []byte("real")); err != nil {
		t.Fatal(err)
	}
	dec := NewDecoder(bytes.NewReader(buf.Bytes()))
	ev, err := dec.Next()
	if err != nil {
		t.Fatal(err)
	}
	if string(ev.Data) != "real" {
		t.Errorf("Data = %q, want %q — injection succeeded", ev.Data, "real")
	}
	if strings.Contains(ev.Type, "\n") {
		t.Errorf("event type still contains a newline: %q", ev.Type)
	}
	// And there must be exactly one event, not two.
	if _, err := dec.Next(); !errors.Is(err, io.EOF) {
		t.Errorf("expected exactly one event; got another (err=%v)", err)
	}
}

func TestEncoderNamedEventRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	enc := NewEncoderNoFlush(&buf)
	if err := enc.WriteEvent("content_block_delta", []byte(`{"x":1}`)); err != nil {
		t.Fatal(err)
	}
	dec := NewDecoder(bytes.NewReader(buf.Bytes()))
	ev, err := dec.Next()
	if err != nil {
		t.Fatal(err)
	}
	if ev.Type != "content_block_delta" || string(ev.Data) != `{"x":1}` {
		t.Errorf("got type=%q data=%q", ev.Type, ev.Data)
	}
}

func TestWriteHeaders(t *testing.T) {
	h := http.Header{}
	WriteHeaders(h)
	for k, want := range map[string]string{
		"Content-Type":      "text/event-stream; charset=utf-8",
		"Cache-Control":     "no-cache, no-transform",
		"X-Accel-Buffering": "no",
	} {
		if got := h.Get(k); got != want {
			t.Errorf("%s = %q, want %q", k, got, want)
		}
	}
}

func TestWriteJSON(t *testing.T) {
	var buf bytes.Buffer
	enc := NewEncoderNoFlush(&buf)
	type payload struct {
		// A string containing a newline is the interesting case: json.Marshal
		// escapes it to \n so the wire form is single-line, but the test pins
		// that rather than assuming it.
		Text string `json:"text"`
	}
	if err := enc.WriteJSON(payload{Text: "a\nb"}); err != nil {
		t.Fatal(err)
	}
	dec := NewDecoder(bytes.NewReader(buf.Bytes()))
	ev, err := dec.Next()
	if err != nil {
		t.Fatal(err)
	}
	if string(ev.Data) != `{"text":"a\nb"}` {
		t.Errorf("Data = %q", ev.Data)
	}
}

// FuzzRoundTrip asserts Encode->Decode is lossless for arbitrary payloads.
//
// Fuzzing earns its place here because the encoder's newline handling has more
// cases than a table test naturally covers (a payload ending in CR, a payload
// that is only terminators, CRLF split across the boundary of a chunk) and the
// consequence of a miss is silent corruption of customer-visible output.
func FuzzRoundTrip(f *testing.F) {
	seeds := []string{
		"hello", "", " ", ":", "\n", "\r", "\r\n", "\n\r",
		"a\nb", "a\r\nb", "a\rb", "data: x", "[DONE]",
		"trailing\n", "\nleading", "\r\n\r\n", "é世🎉",
		`{"delta":{"content":"x\ny"}}`,
	}
	for _, s := range seeds {
		f.Add([]byte(s))
	}
	f.Fuzz(func(t *testing.T, payload []byte) {
		var buf bytes.Buffer
		enc := NewEncoderNoFlush(&buf)
		if err := enc.WriteData(payload); err != nil {
			t.Fatalf("WriteData(%q): %v", payload, err)
		}
		dec := NewDecoderSize(bytes.NewReader(buf.Bytes()), 1<<24)
		ev, err := dec.Next()
		if err != nil {
			t.Fatalf("Next after encoding %q (wire %q): %v", payload, buf.String(), err)
		}
		want := normalizeNewlines(string(payload))
		if string(ev.Data) != want {
			t.Fatalf("round trip mismatch:\n  in   %q\n  want %q\n  got  %q\n  wire %q",
				payload, want, ev.Data, buf.String())
		}
		// There must be exactly one event: a payload must never be able to forge
		// a frame boundary.
		if _, err := dec.Next(); !errors.Is(err, io.EOF) {
			t.Fatalf("payload %q produced more than one event (err=%v, wire %q)", payload, err, buf.String())
		}
	})
}

// TestRealisticProviderStream decodes a stream shaped exactly like OpenAI's, as
// an end-to-end sanity check that the pieces compose.
func TestRealisticProviderStream(t *testing.T) {
	stream := "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n" +
		": keep-alive\n\n" +
		"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: {\"id\":\"c1\",\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":2,\"total_tokens\":11}}\n\n" +
		"data: [DONE]\n\n"

	d := NewDecoder(&byteReader{data: []byte(stream)})
	var dataFrames, comments int
	var sawDone bool
	for {
		ev, err := d.Next()
		if err != nil {
			if !errors.Is(err, io.EOF) {
				t.Fatalf("Next: %v", err)
			}
			break
		}
		switch {
		case ev.IsComment():
			comments++
		case ev.Done:
			sawDone = true
		default:
			dataFrames++
		}
	}
	if dataFrames != 5 {
		t.Errorf("data frames = %d, want 5", dataFrames)
	}
	if comments != 1 {
		t.Errorf("comments = %d, want 1", comments)
	}
	if !sawDone {
		t.Error("did not see the [DONE] sentinel")
	}
}
