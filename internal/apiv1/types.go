// Package apiv1 is the gateway's wire vocabulary: the OpenAI-compatible
// /v1/chat/completions shapes, used both as the public API and as the
// gateway's internal normalized form.
//
// Choosing the OpenAI schema as the *internal* representation is not laziness,
// it is the same call LiteLLM and Portkey make: it is the format clients
// already emit, so the common path costs zero translation, and each provider
// adapter owns exactly one translation instead of every component owning a
// little of it.
//
// The care in this file is concentrated on the union-typed fields —
// `content` (string or array of parts) and `stop` (string or array). Those are
// where real gateways break, because a client sending the legal-but-unusual
// form gets a 400 from the gateway for a request the provider would have
// accepted. Both round-trip in whichever form they arrived in: re-emitting a
// string as a one-element array is a visible behaviour change to a client that
// diffs its own payloads, and being byte-faithful is cheap here.
package apiv1

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
)

// Role constants for chat messages.
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleTool      = "tool"
)

// Object type discriminators emitted in responses.
const (
	ObjectChatCompletion      = "chat.completion"
	ObjectChatCompletionChunk = "chat.completion.chunk"
)

// Finish reasons.
const (
	FinishStop          = "stop"
	FinishLength        = "length"
	FinishToolCalls     = "tool_calls"
	FinishContentFilter = "content_filter"
)

// ChatRequest is an OpenAI-compatible chat completion request.
//
// Unknown fields are preserved in Extra and re-emitted to the upstream provider
// on the pass-through path. A gateway that silently drops a field the client
// set is worse than one that rejects it: the client gets a successful response
// that quietly ignored its `logit_bias`.
type ChatRequest struct {
	Model            string          `json:"model"`
	Messages         []Message       `json:"messages"`
	MaxTokens        *int            `json:"max_tokens,omitempty"`
	MaxCompletionTok *int            `json:"max_completion_tokens,omitempty"`
	Temperature      *float64        `json:"temperature,omitempty"`
	TopP             *float64        `json:"top_p,omitempty"`
	N                *int            `json:"n,omitempty"`
	Stream           bool            `json:"stream,omitempty"`
	StreamOptions    *StreamOptions  `json:"stream_options,omitempty"`
	Stop             StringOrArray   `json:"stop,omitempty"`
	PresencePenalty  *float64        `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64        `json:"frequency_penalty,omitempty"`
	Seed             *int64          `json:"seed,omitempty"`
	User             string          `json:"user,omitempty"`
	Tools            json.RawMessage `json:"tools,omitempty"`
	ToolChoice       json.RawMessage `json:"tool_choice,omitempty"`
	ResponseFormat   json.RawMessage `json:"response_format,omitempty"`

	// Extra holds fields the gateway does not model, so they survive the round
	// trip to the provider. Populated by UnmarshalJSON, re-emitted by
	// MarshalJSON.
	Extra map[string]json.RawMessage `json:"-"`
}

// StreamOptions mirrors OpenAI's stream_options.
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

// knownChatRequestFields lists the JSON keys ChatRequest models explicitly, so
// UnmarshalJSON can route everything else into Extra.
var knownChatRequestFields = map[string]struct{}{
	"model": {}, "messages": {}, "max_tokens": {}, "max_completion_tokens": {},
	"temperature": {}, "top_p": {}, "n": {}, "stream": {}, "stream_options": {},
	"stop": {}, "presence_penalty": {}, "frequency_penalty": {}, "seed": {},
	"user": {}, "tools": {}, "tool_choice": {}, "response_format": {},
}

type chatRequestAlias ChatRequest

// UnmarshalJSON decodes a request and captures unmodelled fields into Extra.
func (r *ChatRequest) UnmarshalJSON(b []byte) error {
	var alias chatRequestAlias
	if err := json.Unmarshal(b, &alias); err != nil {
		return err
	}
	var all map[string]json.RawMessage
	if err := json.Unmarshal(b, &all); err != nil {
		return err
	}
	for k, v := range all {
		if _, known := knownChatRequestFields[k]; known {
			continue
		}
		if alias.Extra == nil {
			alias.Extra = make(map[string]json.RawMessage, 4)
		}
		alias.Extra[k] = v
	}
	*r = ChatRequest(alias)
	return nil
}

// MarshalJSON encodes a request, merging Extra back in at the top level.
func (r ChatRequest) MarshalJSON() ([]byte, error) {
	b, err := json.Marshal(chatRequestAlias(r))
	if err != nil {
		return nil, err
	}
	if len(r.Extra) == 0 {
		return b, nil
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	for k, v := range r.Extra {
		if _, known := knownChatRequestFields[k]; known {
			// Never let Extra shadow a modelled field; the typed value wins.
			continue
		}
		m[k] = v
	}
	return json.Marshal(m)
}

// EffectiveMaxTokens returns the completion cap the client asked for, honouring
// either spelling, or 0 if unset.
//
// OpenAI deprecated max_tokens in favour of max_completion_tokens but both are
// in the wild, and the budget pre-flight needs a single answer.
func (r *ChatRequest) EffectiveMaxTokens() int {
	if r.MaxCompletionTok != nil && *r.MaxCompletionTok > 0 {
		return *r.MaxCompletionTok
	}
	if r.MaxTokens != nil && *r.MaxTokens > 0 {
		return *r.MaxTokens
	}
	return 0
}

// Validate checks the request is well-formed enough to route.
//
// This is deliberately permissive about anything the provider can validate
// better than the gateway can (temperature ranges, tool schemas). The gateway
// only rejects what it must understand itself in order to do its job: which
// model, and what text to count tokens for.
func (r *ChatRequest) Validate() error {
	if r.Model == "" {
		return errors.New("field 'model' is required")
	}
	if len(r.Messages) == 0 {
		return errors.New("field 'messages' must contain at least one message")
	}
	for i, m := range r.Messages {
		if m.Role == "" {
			return fmt.Errorf("messages[%d].role is required", i)
		}
	}
	if n := r.N; n != nil && *n > 1 {
		// Honest limitation rather than a silent wrong answer: the cost model
		// and the stream multiplexer both assume one choice.
		return errors.New("field 'n' > 1 is not supported by this gateway")
	}
	if mt := r.EffectiveMaxTokens(); mt < 0 {
		return errors.New("field 'max_tokens' must be non-negative")
	}
	return nil
}

// WantsUsage reports whether the client asked for a usage record on the
// streaming path.
func (r *ChatRequest) WantsUsage() bool {
	return r.StreamOptions != nil && r.StreamOptions.IncludeUsage
}

// Message is a single chat message.
type Message struct {
	Role       string          `json:"role"`
	Content    Content         `json:"content,omitempty"`
	Name       string          `json:"name,omitempty"`
	ToolCalls  json.RawMessage `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
}

// Content is a message body that may arrive either as a bare string or as an
// array of typed parts (text, image_url, ...).
//
// The zero value marshals to nothing (the field is omitted). A present-but-null
// content — legal for assistant messages that only carry tool_calls — round
// trips as null rather than being coerced to "".
type Content struct {
	// raw is the exact JSON that arrived, or nil if absent.
	raw json.RawMessage
	// text is the flattened text, computed once at decode time because the
	// token estimator needs it on the hot path.
	text     string
	isString bool
}

// NewTextContent builds string-form content.
func NewTextContent(s string) Content {
	b, _ := json.Marshal(s)
	return Content{raw: b, text: s, isString: true}
}

// UnmarshalJSON accepts a string, null, or an array of parts.
func (c *Content) UnmarshalJSON(b []byte) error {
	trimmed := bytes.TrimSpace(b)
	c.raw = append(c.raw[:0], trimmed...)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		c.text, c.isString = "", false
		return nil
	}
	switch trimmed[0] {
	case '"':
		var s string
		if err := json.Unmarshal(trimmed, &s); err != nil {
			return err
		}
		c.text, c.isString = s, true
		return nil
	case '[':
		var parts []contentPart
		if err := json.Unmarshal(trimmed, &parts); err != nil {
			return fmt.Errorf("content array: %w", err)
		}
		// Flatten only the text parts. Image parts contribute tokens too, but
		// their count is model-specific and unknowable from the URL alone;
		// tokens.Estimate reports that as an explicit unknown rather than
		// pretending the image is free.
		var sb bytes.Buffer
		for _, p := range parts {
			if p.Type == "text" || (p.Type == "" && p.Text != "") {
				if sb.Len() > 0 {
					sb.WriteByte('\n')
				}
				sb.WriteString(p.Text)
			}
		}
		c.text, c.isString = sb.String(), false
		return nil
	default:
		return fmt.Errorf("content must be a string, null, or an array; got %q", firstByte(trimmed))
	}
}

// MarshalJSON re-emits content in exactly the form it arrived in.
func (c Content) MarshalJSON() ([]byte, error) {
	if len(c.raw) == 0 {
		return []byte("null"), nil
	}
	return c.raw, nil
}

// Text returns the flattened text of the content.
func (c Content) Text() string { return c.text }

// IsString reports whether the content arrived as a bare JSON string.
func (c Content) IsString() bool { return c.isString }

// Present reports whether the field was set to anything other than null.
func (c Content) Present() bool {
	return len(c.raw) > 0 && !bytes.Equal(c.raw, []byte("null"))
}

// HasNonTextParts reports whether array-form content carried parts the token
// estimator cannot count (images, audio).
func (c Content) HasNonTextParts() bool {
	if c.isString || !c.Present() || c.raw[0] != '[' {
		return false
	}
	var parts []contentPart
	if err := json.Unmarshal(c.raw, &parts); err != nil {
		return false
	}
	for _, p := range parts {
		if p.Type != "text" && p.Type != "" {
			return true
		}
	}
	return false
}

type contentPart struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

// StringOrArray models fields such as `stop` that accept either form.
type StringOrArray struct {
	raw    json.RawMessage
	values []string
}

// NewStringOrArray builds the array form from values.
func NewStringOrArray(vs ...string) StringOrArray {
	if len(vs) == 0 {
		return StringOrArray{}
	}
	b, _ := json.Marshal(vs)
	return StringOrArray{raw: b, values: vs}
}

// UnmarshalJSON accepts a string, an array of strings, or null.
func (s *StringOrArray) UnmarshalJSON(b []byte) error {
	trimmed := bytes.TrimSpace(b)
	s.raw = append(s.raw[:0], trimmed...)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		s.values = nil
		return nil
	}
	switch trimmed[0] {
	case '"':
		var one string
		if err := json.Unmarshal(trimmed, &one); err != nil {
			return err
		}
		s.values = []string{one}
		return nil
	case '[':
		return json.Unmarshal(trimmed, &s.values)
	default:
		return fmt.Errorf("expected a string or an array of strings; got %q", firstByte(trimmed))
	}
}

// MarshalJSON re-emits the original form.
func (s StringOrArray) MarshalJSON() ([]byte, error) {
	if len(s.raw) == 0 {
		return []byte("null"), nil
	}
	return s.raw, nil
}

// Values returns the normalized slice form.
func (s StringOrArray) Values() []string { return s.values }

// Len returns the number of values.
func (s StringOrArray) Len() int { return len(s.values) }

// ChatResponse is a non-streaming chat completion response.
type ChatResponse struct {
	ID                string   `json:"id"`
	Object            string   `json:"object"`
	Created           int64    `json:"created"`
	Model             string   `json:"model"`
	Choices           []Choice `json:"choices"`
	Usage             *Usage   `json:"usage,omitempty"`
	SystemFingerprint string   `json:"system_fingerprint,omitempty"`
}

// Choice is one completion alternative.
type Choice struct {
	Index        int      `json:"index"`
	Message      *Message `json:"message,omitempty"`
	Delta        *Message `json:"delta,omitempty"`
	FinishReason *string  `json:"finish_reason"`
	Logprobs     any      `json:"logprobs,omitempty"`
}

// ChatChunk is one streamed delta frame.
type ChatChunk struct {
	ID                string   `json:"id"`
	Object            string   `json:"object"`
	Created           int64    `json:"created"`
	Model             string   `json:"model"`
	Choices           []Choice `json:"choices"`
	Usage             *Usage   `json:"usage,omitempty"`
	SystemFingerprint string   `json:"system_fingerprint,omitempty"`
}

// DeltaText returns the text carried by the first choice's delta, if any.
func (c *ChatChunk) DeltaText() string {
	if c == nil || len(c.Choices) == 0 {
		return ""
	}
	d := c.Choices[0].Delta
	if d == nil {
		return ""
	}
	return d.Content.Text()
}

// Usage is the token accounting a provider reports.
//
// PromptTokens and CompletionTokens are what the gateway bills against.
// ReasoningTokens is carried separately because reasoning/thinking tokens are
// billed as *output* by every provider that emits them, and a cost model that
// forgets to fold them into the completion count under-bills a thinking model
// by more than an order of magnitude.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`

	PromptTokensDetails     *PromptTokensDetails     `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *CompletionTokensDetails `json:"completion_tokens_details,omitempty"`
}

// PromptTokensDetails breaks the prompt count down.
type PromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
}

// CompletionTokensDetails breaks the completion count down.
type CompletionTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// CachedPromptTokens returns the prompt tokens the provider served from its own
// prefix cache, which are usually billed at a discount.
func (u *Usage) CachedPromptTokens() int {
	if u == nil || u.PromptTokensDetails == nil {
		return 0
	}
	return u.PromptTokensDetails.CachedTokens
}

// ReasoningTokens returns hidden reasoning tokens included in the completion
// count.
func (u *Usage) ReasoningTokens() int {
	if u == nil || u.CompletionTokensDetails == nil {
		return 0
	}
	return u.CompletionTokensDetails.ReasoningTokens
}

// Error is the OpenAI error envelope: {"error": {...}}.
type Error struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   string `json:"param,omitempty"`
	Code    string `json:"code,omitempty"`
}

// ErrorEnvelope wraps an Error for the wire.
type ErrorEnvelope struct {
	Err Error `json:"error"`
}

// Error type discriminators. These match the strings OpenAI clients switch on,
// so an SDK's retry logic behaves the same against this gateway as against the
// provider it fronts.
const (
	ErrTypeInvalidRequest = "invalid_request_error"
	ErrTypeAuth           = "authentication_error"
	ErrTypeRateLimit      = "rate_limit_error"
	ErrTypeServer         = "server_error"
	ErrTypeBudget         = "budget_exceeded_error"
	ErrTypeUpstream       = "upstream_error"
)

// NewError builds an error envelope.
func NewError(typ, code, msg string) ErrorEnvelope {
	return ErrorEnvelope{Err: Error{Message: msg, Type: typ, Code: code}}
}

func firstByte(b []byte) string {
	if len(b) == 0 {
		return ""
	}
	return string(b[:1])
}
