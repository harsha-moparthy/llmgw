package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
)

// OpenAIProvider speaks the OpenAI /v1/chat/completions protocol.
//
// Because the gateway's internal representation IS the OpenAI schema, this
// adapter is nearly a pass-through: it forwards the request (including the Extra
// fields the client set that the gateway does not model), attaches auth, and —
// the part that is actually work — classifies failures. The classification uses
// the response body as well as the status code, because the two 400s a router
// must treat differently (a malformed request vs a prompt that exceeds the
// context window) are indistinguishable by status alone.
type OpenAIProvider struct {
	name    string
	baseURL string
	apiKey  string
	client  *http.Client
	// now is injected so tests can pin Retry-After HTTP-date parsing.
	now func() time.Time
}

// OpenAIConfig configures an OpenAIProvider.
type OpenAIConfig struct {
	Name      string
	BaseURL   string
	APIKey    string
	Transport Transport
	Now       func() time.Time
}

// NewOpenAIProvider builds an OpenAI adapter.
func NewOpenAIProvider(cfg OpenAIConfig) *OpenAIProvider {
	now := cfg.Now
	if now == nil {
		now = time.Now
	}
	return &OpenAIProvider{
		name:    cfg.Name,
		baseURL: strings.TrimRight(cfg.BaseURL, "/"),
		apiKey:  cfg.APIKey,
		client:  cfg.Transport.Client(),
		now:     now,
	}
}

// Name implements Provider.
func (p *OpenAIProvider) Name() string { return p.name }

// Vendor implements Provider.
func (p *OpenAIProvider) Vendor() string { return "openai" }

// buildRequest constructs the upstream HTTP request, rewriting only the model
// field to the upstream model name and forwarding everything else.
func (p *OpenAIProvider) buildRequest(ctx context.Context, req *apiv1.ChatRequest, model string, stream bool) (*http.Request, error) {
	// Copy so that rewriting the model and stream flag does not mutate the
	// caller's request — a subtle aliasing bug that would corrupt a retry on a
	// different provider.
	upstream := *req
	upstream.Model = model
	upstream.Stream = stream
	if stream {
		// Always ask the upstream for usage on the streaming path, so the ledger
		// has real token counts even when the client did not request them. The
		// gateway decides separately whether to forward the usage frame.
		if upstream.StreamOptions == nil {
			upstream.StreamOptions = &apiv1.StreamOptions{}
		}
		upstream.StreamOptions.IncludeUsage = true
	}
	body, err := json.Marshal(&upstream)
	if err != nil {
		return nil, fmt.Errorf("marshalling request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	}
	if stream {
		httpReq.Header.Set("Accept", "text/event-stream")
	}
	applyCorrelation(ctx, httpReq)
	return httpReq, nil
}

// Chat implements Provider for the non-streaming path.
func (p *OpenAIProvider) Chat(ctx context.Context, req *apiv1.ChatRequest, model string) (*Result, *Failure) {
	start := p.now()
	httpReq, err := p.buildRequest(ctx, req, model, false)
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

	var out apiv1.ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, &Failure{Class: ClassUpstream5xx, Provider: p.name, Model: model, Err: fmt.Errorf("decoding response: %w", err)}
	}
	return &Result{
		Response:         &out,
		Usage:            out.Usage,
		UsageIsEstimated: out.Usage == nil,
		UpstreamLatency:  p.now().Sub(start),
	}, nil
}

// ChatStream implements Provider for the streaming path.
func (p *OpenAIProvider) ChatStream(ctx context.Context, req *apiv1.ChatRequest, model string) (Stream, *Failure) {
	httpReq, err := p.buildRequest(ctx, req, model, true)
	if err != nil {
		return nil, &Failure{Class: ClassBadRequest, Provider: p.name, Model: model, Err: err}
	}
	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, p.transportFailure(ctx, model, err)
	}
	if resp.StatusCode != http.StatusOK {
		// Read the error body before returning; the connection is not going to
		// become a stream.
		f := p.classifyResponse(model, resp)
		_ = resp.Body.Close()
		return nil, f
	}
	return newOpenAIStream(ctx, p.name, model, resp), nil
}

// transportFailure classifies a Client.Do error, distinguishing a cancelled
// context (client left) from a genuine connect/timeout failure.
func (p *OpenAIProvider) transportFailure(ctx context.Context, model string, err error) *Failure {
	class := ClassConnect
	if ctx.Err() != nil {
		class = ClassCancelled
	} else if isTimeout(err) {
		class = ClassTimeout
	}
	return &Failure{Class: class, Provider: p.name, Model: model, Err: err}
}

// classifyResponse turns a non-2xx response into a Failure, inspecting the body
// where the status code alone is ambiguous.
func (p *OpenAIProvider) classifyResponse(model string, resp *http.Response) *Failure {
	body := readErrorBody(resp.Body)
	class := ClassifyStatus(resp.StatusCode)

	// A 400 is ambiguous: it can be a malformed request (not retryable, not a
	// health signal) or a context-length overflow (a different route with a
	// larger window could satisfy it). OpenAI signals the latter with the error
	// code "context_length_exceeded"; treat that specially rather than lumping
	// it in with genuine client errors.
	if resp.StatusCode == http.StatusBadRequest {
		if code := openAIErrorCode(body); code == "context_length_exceeded" ||
			strings.Contains(strings.ToLower(body), "maximum context length") {
			class = ClassContextLength
		}
	}
	// A 400/403 can also carry a content-filter signal.
	if strings.Contains(strings.ToLower(body), "content_filter") ||
		strings.Contains(strings.ToLower(body), "content management policy") {
		class = ClassContentFilter
	}

	f := &Failure{
		Class:      class,
		Provider:   p.name,
		Model:      model,
		StatusCode: resp.StatusCode,
		Body:       body,
	}
	if class == ClassRateLimit || class == ClassOverloaded {
		f.RetryAfter = p.parseRetryAfter(resp.Header.Get("Retry-After"))
	}
	return f
}

// parseRetryAfter handles both forms the header can take: an integer number of
// seconds, and an HTTP-date. Real providers use both, and a gateway that only
// understood one would back off wrongly against the other.
func (p *OpenAIProvider) parseRetryAfter(v string) time.Duration {
	v = strings.TrimSpace(v)
	if v == "" {
		return 0
	}
	if secs, err := strconv.Atoi(v); err == nil {
		if secs < 0 {
			return 0
		}
		return time.Duration(secs) * time.Second
	}
	if t, err := http.ParseTime(v); err == nil {
		d := t.Sub(p.now())
		if d < 0 {
			return 0
		}
		return d
	}
	return 0
}

// openAIErrorCode extracts error.code from an OpenAI error envelope, returning
// "" if the body is not one.
func openAIErrorCode(body string) string {
	var env struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal([]byte(body), &env); err != nil {
		return ""
	}
	return env.Error.Code
}

// isTimeout reports whether an error is a timeout, via the net.Error interface
// that both dial and response-header timeouts implement.
func isTimeout(err error) bool {
	type timeout interface{ Timeout() bool }
	var t timeout
	if errors.As(err, &t) {
		return t.Timeout()
	}
	return strings.Contains(strings.ToLower(err.Error()), "timeout")
}
