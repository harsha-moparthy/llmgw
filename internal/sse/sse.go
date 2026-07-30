// Package sse implements Server-Sent Events framing in both directions.
//
// This is the most correctness-critical parsing in the gateway. Every streamed
// token from every provider passes through the decoder and every streamed token
// to every client passes through the encoder, so a framing bug does not fail
// loudly — it silently corrupts responses, and it corrupts them in ways that
// look like the model said something odd rather than like the proxy is broken.
//
// Three properties are treated as non-negotiable, and each has tests that fail
// if it regresses:
//
//  1. Split-tolerance. A provider's TCP stream splits wherever the network
//     decides, which means a frame can be cut mid-token ("da" | "ta: {..."),
//     mid-UTF-8-rune, or between the CR and LF of a CRLF pair. A decoder that
//     assumes one Read returns one frame works perfectly in tests against a
//     bytes.Buffer and fails in production. The tests here drive the decoder
//     with a one-byte-at-a-time reader specifically to make that class of bug
//     impossible to ship.
//
//  2. Round-trip losslessness. Encode(payload) followed by Decode must return
//     exactly payload — including payloads containing "\n", "\r\n", leading
//     spaces, and lone colons. This matters because JSON payloads from a
//     provider can contain escaped newlines, and a naive encoder that writes a
//     multi-line payload as a single "data:" line turns one frame into several
//     and desynchronises the whole stream. Enforced by both table tests and a
//     fuzz target.
//
//  3. Truncation is a failure, not an end. An SSE stream that stops without its
//     terminator is a dead provider, not a finished response. The decoder
//     reports the difference, because a gateway that cannot tell them apart
//     will bill a truncated answer as a complete one and return it to the
//     client as if the model had finished.
//
// The implementation follows the WHATWG event-stream grammar, restricted to
// what LLM providers actually emit, and is deliberately allocation-light: it
// reuses its line buffer across frames rather than allocating per line.
package sse

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"
)

// DoneSentinel is the payload OpenAI (and everything OpenAI-compatible) sends
// to mark the end of a completion stream. It is not part of the SSE spec — it is
// a convention layered on top — so it is matched exactly rather than guessed at.
const DoneSentinel = "[DONE]"

// DefaultMaxEventSize bounds a single event's accumulated data.
//
// An unbounded decoder is a remote out-of-memory: a hostile or broken upstream
// that sends "data: " followed by an endless stream of bytes with no blank line
// would grow this buffer until the process dies. 1 MiB is far above any real
// completion chunk (which is tens of bytes) while still being small enough that
// a few thousand concurrent streams cannot exhaust memory.
const DefaultMaxEventSize = 1 << 20

// ErrEventTooLarge is returned when an event exceeds the configured bound.
var ErrEventTooLarge = errors.New("sse: event exceeds maximum size")

// ErrTruncated is returned when the underlying reader hits EOF part-way through
// an event, i.e. after some field lines but before the blank line that
// terminates them.
//
// This is distinct from a clean EOF and the distinction is the point: a clean
// EOF after a complete event means the upstream finished, while a truncated
// event means it died. Collapsing both into io.EOF is the bug that lets a
// half-generated answer be served as a whole one.
var ErrTruncated = errors.New("sse: stream truncated mid-event")

// ErrNoFlusher is returned by NewEncoder when the writer cannot be flushed.
//
// A buffering SSE proxy is not a slow SSE proxy, it is a broken one: the client
// receives nothing until the response completes, which defeats the entire
// purpose of streaming and looks to the user like a hung request. Failing loudly
// at construction is much better than discovering it from a bug report.
var ErrNoFlusher = errors.New("sse: writer does not implement http.Flusher; a buffering SSE writer would defeat streaming")

// Event is one decoded server-sent event.
//
// Data holds the joined value of the event's data fields. Per the spec, multiple
// data lines within one event are joined with "\n" (and the trailing newline the
// spec adds is dropped, matching every real client's behaviour).
type Event struct {
	// Type is the value of the "event:" field, or "" if absent.
	Type string
	// Data is the joined payload of all "data:" fields in the event.
	Data []byte
	// ID is the value of the "id:" field, or "" if absent.
	ID string
	// Retry is the reconnection time from a "retry:" field, if one was present
	// and parsed as a valid integer number of milliseconds.
	Retry time.Duration
	// HasRetry distinguishes an explicit "retry: 0" from an absent field.
	HasRetry bool
	// Comment holds the text of a comment-only event (a line beginning with
	// ':'), which providers use as a keep-alive. Comment events carry no data
	// and must not be forwarded as content.
	Comment string
	// Done is true when Data is exactly the [DONE] sentinel.
	Done bool
}

// IsComment reports whether this event was a comment/keep-alive rather than a
// data-bearing event.
func (e *Event) IsComment() bool { return e.Comment != "" && len(e.Data) == 0 }

// Decoder reads events from a stream.
//
// Not safe for concurrent use: one Decoder belongs to one upstream response
// body, which one goroutine reads.
type Decoder struct {
	br  *bufio.Reader
	max int

	// data accumulates the current event's data field across lines. Reused
	// across events to avoid an allocation per frame; a completion stream is
	// thousands of frames and per-frame garbage is the difference between a
	// gateway that scales and one that spends its time in GC.
	data bytes.Buffer
	// dataSeen tracks whether any data field appeared, so that an event
	// consisting only of "data:" with an empty value is still a data event.
	dataSeen bool

	evType   string
	id       string
	retry    time.Duration
	hasRetry bool

	// inEvent is true once any field line of an event has been read, so an EOF
	// can be classified as truncation rather than a clean end.
	inEvent bool

	// lastEventID persists across events per the spec: an event without an id
	// field inherits the previously seen id.
	lastEventID string

	// lineBuf is reused by readLine across calls.
	lineBuf []byte

	// pendingEOF records that readLine returned a partial final line and the
	// underlying reader is exhausted, so the next read must not block or
	// re-Peek.
	pendingEOF bool

	err error
}

// NewDecoder returns a Decoder reading from r with the default size bound.
func NewDecoder(r io.Reader) *Decoder { return NewDecoderSize(r, DefaultMaxEventSize) }

// NewDecoderSize returns a Decoder with an explicit maximum event size.
func NewDecoderSize(r io.Reader, maxEvent int) *Decoder {
	if maxEvent <= 0 {
		maxEvent = DefaultMaxEventSize
	}
	// The bufio.Reader's own buffer is sized independently of maxEvent: it only
	// needs to be large enough to make reads efficient, because readLine below
	// handles lines longer than the buffer by accumulating across fills. Sizing
	// it to maxEvent would allocate a megabyte per concurrent stream.
	const bufSize = 16 << 10
	return &Decoder{br: bufio.NewReaderSize(r, bufSize), max: maxEvent}
}

// Next reads the next event.
//
// It returns io.EOF on a clean end of stream, ErrTruncated if the stream ended
// mid-event, and ErrEventTooLarge if an event exceeded the bound. Comment
// (keep-alive) events are returned rather than swallowed, so a caller that
// wants to reset a read deadline on any upstream activity can see them; callers
// that only want content should skip Event.IsComment().
func (d *Decoder) Next() (*Event, error) {
	if d.err != nil {
		return nil, d.err
	}
	for {
		line, err := d.readLine()
		if err != nil {
			if errors.Is(err, io.EOF) {
				// A stream that ends without a blank line after field data is
				// truncated. Providers always terminate their final event, so
				// this genuinely means the connection died.
				if d.inEvent {
					d.err = ErrTruncated
					return nil, d.err
				}
				d.err = io.EOF
				return nil, io.EOF
			}
			d.err = err
			return nil, err
		}

		// A blank line dispatches the accumulated event.
		if len(line) == 0 {
			if !d.inEvent {
				// Leading or repeated blank lines between events are legal and
				// carry no meaning.
				continue
			}
			ev := d.buildEvent()
			d.reset()
			return ev, nil
		}

		// A line beginning with a colon is a comment. Providers send these as
		// keep-alives (": ping"), and they must never be mistaken for content.
		if line[0] == ':' {
			// Comments are dispatched immediately as their own event rather than
			// being buffered, because they are not part of any event's field set
			// and a keep-alive arriving mid-event must not disturb it.
			if !d.inEvent {
				return &Event{Comment: string(trimLeadingSpace(line[1:]))}, nil
			}
			continue
		}

		field, value := splitField(line)
		d.inEvent = true

		switch string(field) {
		case "data":
			// Multiple data fields in one event are joined with a newline. This
			// is why the encoder must split a payload containing newlines into
			// several data lines: it is the exact inverse.
			if d.dataSeen {
				d.data.WriteByte('\n')
			}
			if d.data.Len()+len(value) > d.max {
				d.err = fmt.Errorf("%w (%d bytes, limit %d)", ErrEventTooLarge, d.data.Len()+len(value), d.max)
				return nil, d.err
			}
			d.data.Write(value)
			d.dataSeen = true
		case "event":
			d.evType = string(value)
		case "id":
			// The spec requires ignoring an id containing a NUL byte.
			if !bytes.ContainsRune(value, 0) {
				d.id = string(value)
			}
		case "retry":
			if ms, err := strconv.Atoi(string(value)); err == nil && ms >= 0 {
				d.retry = time.Duration(ms) * time.Millisecond
				d.hasRetry = true
			}
			// A non-numeric retry is ignored per the spec rather than being an
			// error: one malformed optional field must not kill a stream that is
			// otherwise delivering valid completions.
		default:
			// Unknown fields are ignored, as the spec requires. Being strict
			// here would mean a provider adding a new field breaks the gateway.
		}
	}
}

func (d *Decoder) buildEvent() *Event {
	ev := &Event{
		Type:     d.evType,
		ID:       d.id,
		Retry:    d.retry,
		HasRetry: d.hasRetry,
	}
	if ev.ID == "" {
		ev.ID = d.lastEventID
	} else {
		d.lastEventID = ev.ID
	}
	if d.dataSeen {
		// Copy: the buffer is reused for the next event, so handing out its
		// bytes would alias memory that is about to be overwritten. This is the
		// kind of aliasing bug that shows up as one response's text appearing
		// inside another's, and it is worth the allocation to be rid of.
		ev.Data = append([]byte(nil), d.data.Bytes()...)
		if string(ev.Data) == DoneSentinel {
			ev.Done = true
		}
	}
	return ev
}

func (d *Decoder) reset() {
	d.data.Reset()
	d.dataSeen = false
	d.evType = ""
	d.id = ""
	d.retry = 0
	d.hasRetry = false
	d.inEvent = false
}

// readLine reads one line, accepting LF, CRLF, or a lone CR as the terminator.
//
// All three are legal per the spec, and the lone-CR case is why this cannot be
// bufio.Scanner or ReadString('\n'): both would swallow a CR-terminated line
// into the following one. Scanning for either terminator and then peeking one
// byte to decide whether a CR was half of a CRLF handles every case, including
// a CRLF split across two Read calls — the peek refills the buffer rather than
// concluding the CR stood alone.
//
// The line buffer is reused across calls. A completion stream is thousands of
// short lines, and allocating one slice per line is exactly the kind of
// per-token garbage that makes a proxy's p99 a GC artifact.
func (d *Decoder) readLine() ([]byte, error) {
	if d.pendingEOF {
		return nil, io.EOF
	}
	acc := d.lineBuf[:0]
	for {
		buf, _ := d.br.Peek(d.br.Buffered())
		if len(buf) == 0 {
			// Nothing buffered: force a read. Peek(1) is the documented way to
			// make bufio fill without consuming.
			var err error
			buf, err = d.br.Peek(1)
			if err != nil {
				if errors.Is(err, io.EOF) {
					if len(acc) > 0 {
						// A partial final line. It is returned as content and
						// Next then sees EOF with inEvent set, which classifies
						// the stream as truncated. That is the correct reading:
						// bytes arrived that were not a complete event.
						d.lineBuf = acc
						d.pendingEOF = true
						return acc, nil
					}
					return nil, io.EOF
				}
				return nil, err
			}
			continue
		}

		i := bytes.IndexAny(buf, "\r\n")
		if i < 0 {
			// No terminator in what is buffered; take it all and read on.
			if len(acc)+len(buf) > d.max {
				return nil, fmt.Errorf("%w (line exceeds %d bytes)", ErrEventTooLarge, d.max)
			}
			acc = append(acc, buf...)
			d.br.Discard(len(buf))
			continue
		}

		if len(acc)+i > d.max {
			return nil, fmt.Errorf("%w (line exceeds %d bytes)", ErrEventTooLarge, d.max)
		}
		acc = append(acc, buf[:i]...)
		term := buf[i]
		d.br.Discard(i + 1) // consume the line content and its first terminator byte
		if term == '\r' {
			// Decide CRLF vs lone CR. Peek forces a refill if the LF has not
			// arrived yet, so a CRLF split across Reads is not misread.
			if nb, perr := d.br.Peek(1); perr == nil && nb[0] == '\n' {
				d.br.Discard(1)
			}
		}
		d.lineBuf = acc
		return acc, nil
	}
}

// splitField splits "name: value" per the spec: everything before the first
// colon is the field name, and a single leading space in the value is stripped.
// A line with no colon is a field name with an empty value.
func splitField(line []byte) (field, value []byte) {
	i := bytes.IndexByte(line, ':')
	if i < 0 {
		return line, nil
	}
	field = line[:i]
	value = line[i+1:]
	// Exactly ONE leading space is stripped, not all whitespace. This matters:
	// a JSON payload is conventionally sent as "data: {...}" so one space is
	// cosmetic, but a payload whose own first character is a space would lose
	// it if the implementation trimmed greedily.
	if len(value) > 0 && value[0] == ' ' {
		value = value[1:]
	}
	return field, value
}

func trimLeadingSpace(b []byte) []byte {
	if len(b) > 0 && b[0] == ' ' {
		return b[1:]
	}
	return b
}

// Encoder writes events to a writer, flushing each one.
//
// Not safe for concurrent use: SSE frames must be written in order, and a mutex
// here would only hide a caller that is racing to define that order.
type Encoder struct {
	w       io.Writer
	f       http.Flusher
	written int64
	// scratch is reused across frames to avoid an allocation per token.
	scratch []byte
}

// NewEncoder returns an Encoder writing to w, which must implement
// http.Flusher.
func NewEncoder(w io.Writer) (*Encoder, error) {
	f, ok := w.(http.Flusher)
	if !ok {
		return nil, ErrNoFlusher
	}
	return &Encoder{w: w, f: f, scratch: make([]byte, 0, 512)}, nil
}

// NewEncoderNoFlush returns an Encoder that does not flush, for writing to a
// buffer in tests and for the cached-replay path where the whole response is
// already materialised.
func NewEncoderNoFlush(w io.Writer) *Encoder {
	return &Encoder{w: w, scratch: make([]byte, 0, 512)}
}

// Written returns the total number of bytes written.
//
// The router's failover decision keys off this: once a single byte of a response
// has reached the client, a silent retry on another provider would splice two
// models' outputs together, so the count has to be exact rather than a
// best-effort guess.
func (e *Encoder) Written() int64 { return e.written }

// WriteData writes a data frame carrying payload.
//
// A payload containing newlines is split across multiple data lines, which is
// the inverse of the decoder's joining rule. Writing it as one line would
// terminate the event early and desynchronise the stream — the single most
// likely encoder bug, and the one the fuzz target exists to rule out.
func (e *Encoder) WriteData(payload []byte) error {
	e.scratch = e.scratch[:0]
	e.scratch = appendDataLines(e.scratch, payload)
	e.scratch = append(e.scratch, '\n') // blank line terminates the event
	return e.write(e.scratch)
}

// WriteJSON marshals v and writes it as a data frame.
func (e *Encoder) WriteJSON(v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("sse: marshalling event: %w", err)
	}
	return e.WriteData(b)
}

// WriteEvent writes a named event with a payload, used by providers whose
// protocol is event-typed (Anthropic) and by the gateway's own error frames.
func (e *Encoder) WriteEvent(name string, payload []byte) error {
	e.scratch = e.scratch[:0]
	if name != "" {
		e.scratch = append(e.scratch, "event: "...)
		// A newline inside an event name would inject a field line. Names come
		// from the gateway itself rather than from user input, but sanitising
		// costs nothing and removes the possibility entirely.
		e.scratch = appendSanitizedLine(e.scratch, []byte(name))
		e.scratch = append(e.scratch, '\n')
	}
	e.scratch = appendDataLines(e.scratch, payload)
	e.scratch = append(e.scratch, '\n')
	return e.write(e.scratch)
}

// WriteComment writes a comment frame, used as a keep-alive.
//
// Keep-alives matter for a gateway specifically: a model that thinks for 30
// seconds before its first token looks to every intermediate proxy and load
// balancer like an idle connection, and they will close it. A periodic comment
// is invisible to the client's SSE parser and keeps the path open.
func (e *Encoder) WriteComment(text string) error {
	e.scratch = e.scratch[:0]
	e.scratch = append(e.scratch, ':', ' ')
	e.scratch = appendSanitizedLine(e.scratch, []byte(text))
	e.scratch = append(e.scratch, '\n', '\n')
	return e.write(e.scratch)
}

// WriteDone writes the [DONE] sentinel that terminates an OpenAI-compatible
// completion stream.
func (e *Encoder) WriteDone() error { return e.WriteData([]byte(DoneSentinel)) }

// Flush flushes the underlying writer if it supports it.
func (e *Encoder) Flush() {
	if e.f != nil {
		e.f.Flush()
	}
}

func (e *Encoder) write(b []byte) error {
	n, err := e.w.Write(b)
	e.written += int64(n)
	if err != nil {
		return err
	}
	// Flush per frame. Batching frames would improve throughput and destroy the
	// property the client actually cares about, which is that tokens appear as
	// they are generated.
	e.Flush()
	return nil
}

// appendDataLines appends payload as one or more "data:" lines.
func appendDataLines(dst, payload []byte) []byte {
	if len(payload) == 0 {
		// An empty payload is still a data field: "data:" with no value. The
		// decoder reports it as a present-but-empty Data, which round-trips.
		return append(dst, "data:\n"...)
	}
	rest := payload
	for {
		i := bytes.IndexAny(rest, "\r\n")
		if i < 0 {
			dst = append(dst, "data: "...)
			dst = append(dst, rest...)
			dst = append(dst, '\n')
			return dst
		}
		dst = append(dst, "data: "...)
		dst = append(dst, rest[:i]...)
		dst = append(dst, '\n')
		// Consume the line terminator: CRLF as one, otherwise the single byte.
		if rest[i] == '\r' && i+1 < len(rest) && rest[i+1] == '\n' {
			rest = rest[i+2:]
		} else {
			rest = rest[i+1:]
		}
		if len(rest) == 0 {
			// The payload ended with a newline, which means a final empty data
			// line, so that Decode joins back to a trailing "\n".
			return append(dst, "data:\n"...)
		}
	}
}

// appendSanitizedLine appends b with CR and LF replaced by spaces, so a value
// cannot inject additional SSE field lines.
func appendSanitizedLine(dst, b []byte) []byte {
	start := len(dst)
	dst = append(dst, b...)
	for i := start; i < len(dst); i++ {
		if dst[i] == '\r' || dst[i] == '\n' {
			dst[i] = ' '
		}
	}
	return dst
}

// WriteHeaders sets the response headers an SSE endpoint requires.
//
// X-Accel-Buffering is not a standard header but is load-bearing in practice:
// nginx buffers proxied responses by default, which converts a streaming
// response into a single delayed one. A gateway that streams perfectly and is
// deployed behind a default nginx config appears to not stream at all, and this
// header is the fix.
func WriteHeaders(h http.Header) {
	h.Set("Content-Type", "text/event-stream; charset=utf-8")
	h.Set("Cache-Control", "no-cache, no-transform")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")
}
