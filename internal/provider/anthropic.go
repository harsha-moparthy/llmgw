package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
	"github.com/harsha-moparthy/llmgw/internal/sse"
)

// AnthropicProvider translates between the gateway's OpenAI-shaped internal form
// and Anthropic's Messages API, in both directions.
//
// Unlike the OpenAI adapter this is a real translation, because the wire formats
// genuinely differ in ways that matter:
//
//   - The system prompt is a top-level field, not a message with role "system".
//   - max_tokens is REQUIRED; Anthropic rejects a request without it, so the
//     adapter must supply a default when the client omitted one.
//   - Stop sequences are "stop_sequences", not "stop".
//   - The streaming protocol is event-typed (message_start, content_block_delta,
//     message_delta, message_stop) rather than OpenAI's uniform chunks, and the
//     token usage arrives split across two events — input on message_start,
//     output on message_delta — so it must be ACCUMULATED, not read from one
//     place. A gateway that read usage from a single event would under-report
//     every Anthropic request.
//
// Getting the usage accumulation wrong is the subtle failure here, so it has a
// dedicated test.
type AnthropicProvider struct {
	name       string
	baseURL    string
	apiKey     string
	version    string
	defaultMax int
	client     *http.Client
	now        func() time.Time
}

// AnthropicConfig configures an AnthropicProvider.
type AnthropicConfig struct {
	Name    string
	BaseURL string
	APIKey  string
	// Version is the anthropic-version header. Defaults to a known-good value.
	Version string
	// DefaultMaxTokens is used when the client did not set a completion cap,
	// since Anthropic requires one. Documented as an adapter policy rather than
	// a silent magic number.
	DefaultMaxTokens int
	Transport        Transport
	Now              func() time.Time
}

// NewAnthropicProvider builds an Anthropic adapter.
func NewAnthropicProvider(cfg AnthropicConfig) *AnthropicProvider {
	now := cfg.Now
	if now == nil {
		now = time.Now
	}
	version := cfg.Version
	if version == "" {
		version = "2023-06-01"
	}
	maxTok := cfg.DefaultMaxTokens
	if maxTok <= 0 {
		maxTok = 4096
	}
	return &AnthropicProvider{
		name:       cfg.Name,
		baseURL:    strings.TrimRight(cfg.BaseURL, "/"),
		apiKey:     cfg.APIKey,
		version:    version,
		defaultMax: maxTok,
		client:     cfg.Transport.Client(),
		now:        now,
	}
}

// Name implements Provider.
func (p *AnthropicProvider) Name() string { return p.name }

// Vendor implements Provider.
func (p *AnthropicProvider) Vendor() string { return "anthropic" }

// anthropicRequest is the Messages API request shape.
type anthropicRequest struct {
	Model         string             `json:"model"`
	System        string             `json:"system,omitempty"`
	Messages      []anthropicMessage `json:"messages"`
	MaxTokens     int                `json:"max_tokens"`
	Temperature   *float64           `json:"temperature,omitempty"`
	TopP          *float64           `json:"top_p,omitempty"`
	StopSequences []string           `json:"stop_sequences,omitempty"`
	Stream        bool               `json:"stream,omitempty"`
}

type anthropicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// toAnthropic translates an OpenAI-shaped request into Anthropic's form.
func (p *AnthropicProvider) toAnthropic(req *apiv1.ChatRequest, model string, stream bool) (*anthropicRequest, error) {
	out := &anthropicRequest{
		Model:       model,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      stream,
	}
	// max_tokens is required by Anthropic. Honour the client's cap, else the
	// adapter default.
	if mt := req.EffectiveMaxTokens(); mt > 0 {
		out.MaxTokens = mt
	} else {
		out.MaxTokens = p.defaultMax
	}
	if req.Stop.Len() > 0 {
		out.StopSequences = req.Stop.Values()
	}
	// Hoist system messages into the top-level system field, concatenating
	// multiple system turns (Anthropic takes a single system string).
	var systemParts []string
	for i := range req.Messages {
		m := &req.Messages[i]
		if m.Role == apiv1.RoleSystem {
			systemParts = append(systemParts, m.Content.Text())
			continue
		}
		role := m.Role
		if role == apiv1.RoleAssistant {
			role = "assistant"
		} else {
			role = "user" // tool and user roles both map to user turns for this subset
		}
		out.Messages = append(out.Messages, anthropicMessage{Role: role, Content: m.Content.Text()})
	}
	out.System = strings.Join(systemParts, "\n\n")
	if len(out.Messages) == 0 {
		return nil, fmt.Errorf("anthropic: request has no non-system messages")
	}
	return out, nil
}

func (p *AnthropicProvider) buildRequest(ctx context.Context, body []byte, stream bool) (*http.Request, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("anthropic-version", p.version)
	if p.apiKey != "" {
		httpReq.Header.Set("x-api-key", p.apiKey)
	}
	if stream {
		httpReq.Header.Set("Accept", "text/event-stream")
	}
	applyCorrelation(ctx, httpReq)
	return httpReq, nil
}

// Chat implements Provider for the non-streaming path.
func (p *AnthropicProvider) Chat(ctx context.Context, req *apiv1.ChatRequest, model string) (*Result, *Failure) {
	start := p.now()
	areq, err := p.toAnthropic(req, model, false)
	if err != nil {
		return nil, &Failure{Class: ClassBadRequest, Provider: p.name, Model: model, Err: err}
	}
	body, err := json.Marshal(areq)
	if err != nil {
		return nil, &Failure{Class: ClassBadRequest, Provider: p.name, Model: model, Err: err}
	}
	httpReq, err := p.buildRequest(ctx, body, false)
	if err != nil {
		return nil, &Failure{Class: ClassBadRequest, Provider: p.name, Model: model, Err: err}
	}
	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, p.transportFailure(ctx, model, err)
	}
	defer func() {
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
	}()
	if resp.StatusCode != http.StatusOK {
		return nil, p.classifyResponse(model, resp)
	}
	var ar anthropicResponse
	if err := json.NewDecoder(resp.Body).Decode(&ar); err != nil {
		return nil, &Failure{Class: ClassUpstream5xx, Provider: p.name, Model: model, Err: fmt.Errorf("decoding response: %w", err)}
	}
	return &Result{
		Response:         ar.toOpenAI(model, p.now().Unix()),
		Usage:            ar.usage(),
		UsageIsEstimated: false,
		UpstreamLatency:  p.now().Sub(start),
	}, nil
}

// ChatStream implements Provider for the streaming path.
func (p *AnthropicProvider) ChatStream(ctx context.Context, req *apiv1.ChatRequest, model string) (Stream, *Failure) {
	areq, err := p.toAnthropic(req, model, true)
	if err != nil {
		return nil, &Failure{Class: ClassBadRequest, Provider: p.name, Model: model, Err: err}
	}
	body, err := json.Marshal(areq)
	if err != nil {
		return nil, &Failure{Class: ClassBadRequest, Provider: p.name, Model: model, Err: err}
	}
	httpReq, err := p.buildRequest(ctx, body, true)
	if err != nil {
		return nil, &Failure{Class: ClassBadRequest, Provider: p.name, Model: model, Err: err}
	}
	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, p.transportFailure(ctx, model, err)
	}
	if resp.StatusCode != http.StatusOK {
		f := p.classifyResponse(model, resp)
		_ = resp.Body.Close()
		return nil, f
	}
	return newAnthropicStream(ctx, p.name, model, resp, p.now().Unix()), nil
}

func (p *AnthropicProvider) transportFailure(ctx context.Context, model string, err error) *Failure {
	class := ClassConnect
	if ctx.Err() != nil {
		class = ClassCancelled
	} else if isTimeout(err) {
		class = ClassTimeout
	}
	return &Failure{Class: class, Provider: p.name, Model: model, Err: err}
}

func (p *AnthropicProvider) classifyResponse(model string, resp *http.Response) *Failure {
	body := readErrorBody(resp.Body)
	class := ClassifyStatus(resp.StatusCode)
	// Anthropic returns 529 for overload, which ClassifyStatus already maps, and
	// signals context overflow in the error message on a 400.
	if resp.StatusCode == http.StatusBadRequest {
		low := strings.ToLower(body)
		if strings.Contains(low, "prompt is too long") || strings.Contains(low, "context") {
			class = ClassContextLength
		}
	}
	f := &Failure{Class: class, Provider: p.name, Model: model, StatusCode: resp.StatusCode, Body: body}
	if class == ClassRateLimit || class == ClassOverloaded {
		f.RetryAfter = (&OpenAIProvider{now: p.now}).parseRetryAfter(resp.Header.Get("Retry-After"))
	}
	return f
}

// anthropicResponse is the non-streaming Messages response.
type anthropicResponse struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	StopReason string `json:"stop_reason"`
	Usage      struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
		// Anthropic reports cache reads/writes separately; the cache-read count
		// maps to OpenAI's cached prompt tokens.
		CacheReadInputTokens int `json:"cache_read_input_tokens"`
	} `json:"usage"`
}

func (ar *anthropicResponse) text() string {
	var sb strings.Builder
	for _, c := range ar.Content {
		if c.Type == "text" {
			sb.WriteString(c.Text)
		}
	}
	return sb.String()
}

func (ar *anthropicResponse) usage() *apiv1.Usage {
	u := &apiv1.Usage{
		PromptTokens:     ar.Usage.InputTokens,
		CompletionTokens: ar.Usage.OutputTokens,
		TotalTokens:      ar.Usage.InputTokens + ar.Usage.OutputTokens,
	}
	if ar.Usage.CacheReadInputTokens > 0 {
		// Anthropic reports cache reads SEPARATELY from input_tokens, unlike
		// OpenAI where cached is a subset of prompt. Fold it in so the gateway's
		// single accounting model holds: prompt_tokens must include cached.
		u.PromptTokens += ar.Usage.CacheReadInputTokens
		u.TotalTokens += ar.Usage.CacheReadInputTokens
		u.PromptTokensDetails = &apiv1.PromptTokensDetails{CachedTokens: ar.Usage.CacheReadInputTokens}
	}
	return u
}

// toOpenAI converts an Anthropic response into OpenAI shape.
//
// created must be supplied by the caller: Anthropic's Messages API does not
// return a creation timestamp, but OpenAI's schema has a required `created`
// field that clients display as the completion time. Leaving it at the zero
// value emits "created":0, which every client renders as 1 January 1970 — so the
// adapter stamps its own clock rather than passing along a plausible-looking lie.
func (ar *anthropicResponse) toOpenAI(model string, created int64) *apiv1.ChatResponse {
	finish := mapStopReason(ar.StopReason)
	return &apiv1.ChatResponse{
		ID:      ar.ID,
		Object:  apiv1.ObjectChatCompletion,
		Created: created,
		Model:   model,
		Choices: []apiv1.Choice{{
			Index:        0,
			Message:      &apiv1.Message{Role: apiv1.RoleAssistant, Content: apiv1.NewTextContent(ar.text())},
			FinishReason: &finish,
		}},
		Usage: ar.usage(),
	}
}

// mapStopReason maps Anthropic stop reasons to OpenAI finish reasons.
func mapStopReason(r string) string {
	switch r {
	case "end_turn", "stop_sequence":
		return apiv1.FinishStop
	case "max_tokens":
		return apiv1.FinishLength
	case "tool_use":
		return apiv1.FinishToolCalls
	default:
		return apiv1.FinishStop
	}
}

// anthropicStream translates the event-typed Messages stream into OpenAI chunks.
//
// The state machine is small but the usage accumulation is the reason it exists:
// input_tokens arrives on message_start, output_tokens on message_delta, and a
// consumer that read either one alone would report half the usage.
type anthropicStream struct {
	provider string
	model    string
	body     io.ReadCloser
	dec      *sse.Decoder
	events   chan StreamEvent

	mu     sync.Mutex
	closed bool
	usage  *apiv1.Usage
	done   chan struct{}

	id           string
	created      int64
	inputTokens  int
	outputTokens int
	cachedTokens int
}

// newAnthropicStream builds the translating stream. created is the timestamp the
// emitted OpenAI chunks carry: Anthropic's stream does not supply one, and a zero
// there renders as 1970 in every client, so the adapter stamps its own clock once
// at stream start and uses it for every chunk (a single value, so the frames of
// one response agree with each other and with the non-streaming path).
func newAnthropicStream(ctx context.Context, providerName, model string, resp *http.Response, created int64) *anthropicStream {
	s := &anthropicStream{
		provider: providerName,
		model:    model,
		created:  created,
		body:     resp.Body,
		dec:      sse.NewDecoder(resp.Body),
		events:   make(chan StreamEvent, 16),
		done:     make(chan struct{}),
	}
	go s.run(ctx)
	return s
}

// anthropic stream event payloads, only the fields we consume.
type anthropicEvent struct {
	Type    string `json:"type"`
	Message *struct {
		ID    string `json:"id"`
		Model string `json:"model"`
		Usage *struct {
			InputTokens          int `json:"input_tokens"`
			CacheReadInputTokens int `json:"cache_read_input_tokens"`
		} `json:"usage"`
	} `json:"message"`
	Index int `json:"index"`
	Delta *struct {
		Type       string `json:"type"`
		Text       string `json:"text"`
		StopReason string `json:"stop_reason"`
	} `json:"delta"`
	Usage *struct {
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

func (s *anthropicStream) run(ctx context.Context) {
	defer close(s.events)
	sentRole := false

	for {
		select {
		case <-ctx.Done():
			s.emit(StreamEvent{Err: &Failure{Class: ClassCancelled, Provider: s.provider, Model: s.model, Err: ctx.Err(), UsageAtFailure: s.snapshotUsage()}})
			return
		default:
		}

		ev, err := s.dec.Next()
		if err != nil {
			if errors.Is(err, io.EOF) {
				// Anthropic terminates with message_stop, and that case returns from
				// this loop directly. Reaching EOF here means the stop never arrived,
				// so the response is TRUNCATED even though the framing was clean —
				// exactly the OpenAI [DONE] situation. Emitting Done would serve a
				// half-generated Claude response as a complete one, so it is a
				// failure carrying the usage accumulated so far.
				s.emit(StreamEvent{Err: &Failure{
					Class: ClassTimeout, Provider: s.provider, Model: s.model,
					Err: io.ErrUnexpectedEOF, UsageAtFailure: s.snapshotUsage(),
				}})
				return
			}
			class := ClassUpstream5xx
			if strings.Contains(err.Error(), "truncat") {
				class = ClassTimeout
			}
			s.emit(StreamEvent{Err: &Failure{Class: class, Provider: s.provider, Model: s.model, Err: err, UsageAtFailure: s.snapshotUsage()}})
			return
		}
		if ev.IsComment() || len(ev.Data) == 0 {
			continue
		}

		var ae anthropicEvent
		if err := json.Unmarshal(ev.Data, &ae); err != nil {
			s.emit(StreamEvent{Err: &Failure{Class: ClassUpstream5xx, Provider: s.provider, Model: s.model, Err: err, Body: safeExcerpt(ev.Data), UsageAtFailure: s.snapshotUsage()}})
			return
		}

		switch ae.Type {
		case "message_start":
			if ae.Message != nil {
				s.id = ae.Message.ID
				if ae.Message.Usage != nil {
					s.setInput(ae.Message.Usage.InputTokens, ae.Message.Usage.CacheReadInputTokens)
				}
			}
			// Emit an OpenAI-style opening chunk carrying the role.
			if !sentRole {
				chunk := s.openAIChunk(s.created, apiv1.RoleAssistant, "", nil)
				s.emitChunk(chunk)
				sentRole = true
			}
		case "content_block_delta":
			if ae.Delta != nil && ae.Delta.Type == "text_delta" && ae.Delta.Text != "" {
				chunk := s.openAIChunk(s.created, "", ae.Delta.Text, nil)
				s.emitChunk(chunk)
			}
		case "message_delta":
			// Output tokens accumulate here, and the stop reason arrives here.
			if ae.Usage != nil {
				s.setOutput(ae.Usage.OutputTokens)
			}
			if ae.Delta != nil && ae.Delta.StopReason != "" {
				finish := mapStopReason(ae.Delta.StopReason)
				chunk := s.openAIChunk(s.created, "", "", &finish)
				s.emitChunk(chunk)
			}
		case "message_stop":
			// Terminal. Emit a usage-bearing chunk then Done, mirroring OpenAI's
			// include_usage behaviour so the gateway's ledger gets real counts.
			u := s.snapshotUsage()
			usageChunk := &apiv1.ChatChunk{
				ID: s.id, Object: apiv1.ObjectChatCompletionChunk, Created: s.created,
				Model: s.model, Choices: []apiv1.Choice{}, Usage: u,
			}
			b, _ := json.Marshal(usageChunk)
			s.emit(StreamEvent{Chunk: usageChunk, Raw: b})
			s.emit(StreamEvent{Done: true})
			return
		case "error":
			s.emit(StreamEvent{Err: &Failure{Class: ClassUpstream5xx, Provider: s.provider, Model: s.model, Body: safeExcerpt(ev.Data), UsageAtFailure: s.snapshotUsage()}})
			return
		default:
			// ping, content_block_start, content_block_stop: ignored.
		}
	}
}

func (s *anthropicStream) openAIChunk(created int64, role, content string, finish *string) *apiv1.ChatChunk {
	delta := &apiv1.Message{}
	if role != "" {
		delta.Role = role
	}
	if content != "" {
		delta.Content = apiv1.NewTextContent(content)
	}
	return &apiv1.ChatChunk{
		ID:      s.id,
		Object:  apiv1.ObjectChatCompletionChunk,
		Created: created,
		Model:   s.model,
		Choices: []apiv1.Choice{{Index: 0, Delta: delta, FinishReason: finish}},
	}
}

func (s *anthropicStream) emitChunk(c *apiv1.ChatChunk) {
	b, _ := json.Marshal(c)
	s.emit(StreamEvent{Chunk: c, Raw: b})
}

func (s *anthropicStream) emit(ev StreamEvent) {
	select {
	case s.events <- ev:
	case <-s.done:
	}
}

func (s *anthropicStream) setInput(input, cached int) {
	s.mu.Lock()
	s.inputTokens = input + cached
	s.cachedTokens = cached
	s.recomputeUsageLocked()
	s.mu.Unlock()
}

func (s *anthropicStream) setOutput(output int) {
	s.mu.Lock()
	s.outputTokens = output
	s.recomputeUsageLocked()
	s.mu.Unlock()
}

func (s *anthropicStream) recomputeUsageLocked() {
	u := &apiv1.Usage{
		PromptTokens:     s.inputTokens,
		CompletionTokens: s.outputTokens,
		TotalTokens:      s.inputTokens + s.outputTokens,
	}
	if s.cachedTokens > 0 {
		u.PromptTokensDetails = &apiv1.PromptTokensDetails{CachedTokens: s.cachedTokens}
	}
	s.usage = u
}

func (s *anthropicStream) snapshotUsage() *apiv1.Usage {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.usage
}

// Events implements Stream.
func (s *anthropicStream) Events() <-chan StreamEvent { return s.events }

// Usage implements Stream. Anthropic always reports usage, so it is never
// estimated.
func (s *anthropicStream) Usage() (*apiv1.Usage, bool) {
	return s.snapshotUsage(), false
}

// Close implements Stream, idempotently.
func (s *anthropicStream) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	close(s.done)
	s.mu.Unlock()
	return s.body.Close()
}
