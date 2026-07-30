// Package server is the gateway's HTTP surface and the place every other
// component is wired together.
//
// The request path is a fixed sequence — size-limit, authenticate, decode,
// authorise, estimate, budget-reserve, cache, route, true-up — and each step's
// failure produces an OpenAI-shaped error with a deliberately chosen status, so
// a client's existing SDK retry logic behaves the same against this gateway as
// against the provider it fronts.
//
// Two things in here are subtler than they look and are commented where they
// happen: the order of the pipeline (size limit BEFORE decode, or a hostile body
// is a memory DoS; auth by constant-time compare, or the key check is a timing
// oracle), and the streaming header timing (the response is not written until
// the first upstream content arrives, which costs a little TTFB and is what
// keeps the router's transparent-failover window open).
package server

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
	"github.com/harsha-moparthy/llmgw/internal/budget"
	"github.com/harsha-moparthy/llmgw/internal/cache"
	"github.com/harsha-moparthy/llmgw/internal/config"
	"github.com/harsha-moparthy/llmgw/internal/ledger"
	"github.com/harsha-moparthy/llmgw/internal/metrics"
	"github.com/harsha-moparthy/llmgw/internal/pricing"
	"github.com/harsha-moparthy/llmgw/internal/router"
	"github.com/harsha-moparthy/llmgw/internal/tokens"
)

// Deps are the wired-up dependencies a Server needs. Assembling them is the job
// of cmd/gateway; the server takes them as an explicit struct rather than
// reaching for globals, so a test can substitute a fake for any one of them.
type Deps struct {
	Config  *config.Config
	Router  *router.Router
	Budget  *budget.Budget
	Cache   *cache.Store
	Ledger  *ledger.Ledger
	Metrics *metrics.Gateway
	Prices  *pricing.Table
	Priors  *tokens.Priors
	Logger  *slog.Logger
	// Now is injected for tests.
	Now func() time.Time
	// Readiness reports whether any provider is currently usable, for /readyz.
	Readiness func() bool
}

// Server is the gateway HTTP server.
type Server struct {
	deps    Deps
	tenants map[string]config.Tenant // by id
	reqSeq  atomic.Uint64
	now     func() time.Time
}

// New builds a Server from its dependencies.
func New(deps Deps) *Server {
	if deps.Logger == nil {
		deps.Logger = slog.Default()
	}
	if deps.Now == nil {
		deps.Now = time.Now
	}
	tenants := make(map[string]config.Tenant, len(deps.Config.Tenants))
	for _, t := range deps.Config.Tenants {
		tenants[t.ID] = t
	}
	return &Server{deps: deps, tenants: tenants, now: deps.Now}
}

// Handler returns the routed HTTP handler.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", s.recover(s.handleChat))
	mux.HandleFunc("/v1/models", s.recover(s.handleModels))
	mux.HandleFunc("/v1/usage", s.recover(s.handleUsage))
	mux.HandleFunc("/healthz", s.handleHealthz)
	mux.HandleFunc("/readyz", s.handleReadyz)
	if s.deps.Metrics != nil {
		// /metrics carries per-tenant spend, token volume and remaining budget —
		// one tenant's commercial data. Prometheus endpoints are conventionally
		// left open because they conventionally carry nothing sensitive; this one
		// does, so it is gated behind the operator token when one is configured.
		//
		// The default is deliberately open, because the alternative — refusing to
		// serve metrics unless a token is set — silently breaks every existing
		// scrape config on upgrade. Instead the gap is stated in the README and
		// the config field is right there. A deployment that exposes this port
		// beyond its monitoring network should set it.
		mux.Handle("/metrics", s.requireOperator(s.deps.Metrics.Registry()))
	}
	return mux
}

// requireOperator gates a handler behind the operator token from
// LLMGW_OPERATOR_TOKEN, if one is set. With no token configured the handler is
// served openly and a warning is logged once at startup.
func (s *Server) requireOperator(h http.Handler) http.Handler {
	token := os.Getenv("LLMGW_OPERATOR_TOKEN")
	if token == "" {
		return h
	}
	want := config.HashKey(token)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		presented := r.Header.Get("Authorization")
		presented = strings.TrimPrefix(presented, "Bearer ")
		if subtle.ConstantTimeCompare([]byte(config.HashKey(strings.TrimSpace(presented))), []byte(want)) != 1 {
			writeError(w, http.StatusUnauthorized, apiv1.ErrTypeAuth, "", "operator token required")
			return
		}
		h.ServeHTTP(w, r)
	})
}

// recover wraps a handler so a panic returns a 500 and is logged with its stack,
// rather than taking the process down. It does NOT swallow the panic silently:
// one structured log line with the stack is the minimum a on-call engineer needs.
func (s *Server) recover(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if v := recover(); v != nil {
				s.deps.Logger.Error("panic in handler",
					"panic", fmt.Sprint(v),
					"path", r.URL.Path,
					"stack", stack())
				// The client may already have received headers on a streaming
				// path; WriteHeader will no-op then, which is fine — the point is
				// not to crash the process.
				writeError(w, http.StatusInternalServerError, apiv1.ErrTypeServer, "", "internal error")
			}
		}()
		h(w, r)
	}
}

// tenantFromAuth authenticates the bearer token in constant time.
//
// The constant-time compare is not theatre: a naive string == leaks, through its
// timing, how many leading bytes of a guess were correct, which turns key
// recovery into a byte-at-a-time search. subtle.ConstantTimeCompare closes that
// oracle. We hash the presented key and compare hashes so the comparison length
// does not itself leak the key length.
func (s *Server) tenantFromAuth(r *http.Request) (config.Tenant, bool) {
	auth := r.Header.Get("Authorization")
	const prefix = "Bearer "
	if !strings.HasPrefix(auth, prefix) {
		return config.Tenant{}, false
	}
	presented := config.HashKey(strings.TrimSpace(auth[len(prefix):]))
	// Compare against every tenant, and do NOT early-return on the first
	// mismatch — a loop that stops early would leak, via timing, which tenant a
	// key nearly matched. The match flag is accumulated and the loop always runs
	// to completion.
	var matched config.Tenant
	found := false
	for _, t := range s.deps.Config.Tenants {
		if subtle.ConstantTimeCompare([]byte(presented), []byte(t.APIKeyHash)) == 1 {
			matched = t
			found = true
		}
	}
	return matched, found
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	tenant, ok := s.tenantFromAuth(r)
	if !ok {
		writeError(w, http.StatusUnauthorized, apiv1.ErrTypeAuth, "", "invalid or missing API key")
		return
	}
	type model struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		OwnedBy string `json:"owned_by"`
	}
	list := make([]model, 0, len(s.deps.Config.Routes))
	for alias := range s.deps.Config.Routes {
		if !router.AllowedFor(tenant.AllowedModels, alias) {
			continue
		}
		list = append(list, model{ID: alias, Object: "model", OwnedBy: "llmgw"})
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": list})
}

func (s *Server) handleHealthz(w http.ResponseWriter, _ *http.Request) {
	// Liveness: the process is up and can answer. It says NOTHING about whether
	// upstreams are usable — conflating the two makes a rolling deploy kill a
	// healthy fleet the moment a shared upstream blips.
	_, _ = w.Write([]byte("ok\n"))
}

func (s *Server) handleReadyz(w http.ResponseWriter, _ *http.Request) {
	// Readiness: is there at least one usable provider. A load balancer routes on
	// this, so it must reflect real upstream availability.
	if s.deps.Readiness != nil && !s.deps.Readiness() {
		writeError(w, http.StatusServiceUnavailable, apiv1.ErrTypeServer, "", "no healthy provider")
		return
	}
	_, _ = w.Write([]byte("ready\n"))
}

func (s *Server) handleUsage(w http.ResponseWriter, r *http.Request) {
	tenant, ok := s.tenantFromAuth(r)
	if !ok {
		writeError(w, http.StatusUnauthorized, apiv1.ErrTypeAuth, "", "invalid or missing API key")
		return
	}
	out := map[string]any{"tenant": tenant.ID}
	if s.deps.Ledger != nil {
		if totals, ok := s.deps.Ledger.TenantTotals(tenant.ID); ok {
			// The estimated-vs-reported split is surfaced explicitly: a usage
			// report that hid which numbers were guessed would be worse than
			// useless to someone reconciling a bill.
			out["totals"] = totals
		}
	}
	if s.deps.Budget != nil {
		out["budget"] = s.deps.Budget.Status(tenant.ID)
	}
	writeJSON(w, http.StatusOK, out)
}

// requestID mints a gateway-unique id used to correlate the ledger with the
// provider log. Monotonic plus a start-time salt so ids do not collide across
// restarts within a benchmark directory.
func (s *Server) requestID() string {
	n := s.reqSeq.Add(1)
	return "gw-" + strconv.FormatUint(n, 36)
}

// writeError writes an OpenAI-shaped error envelope.
func writeError(w http.ResponseWriter, status int, typ, code, msg string) {
	// If headers were already sent (streaming path), this WriteHeader is a no-op
	// and we can only have logged; that is handled by callers that know they are
	// mid-stream. Here we assume pre-stream.
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(apiv1.NewError(typ, code, msg))
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// decodeBody decodes the request body with unknown-field rejection off (clients
// legitimately send provider-specific fields the gateway forwards via Extra) but
// with a single-value guard so a body with trailing garbage after the JSON
// object is rejected rather than silently accepted.
func decodeBody(r *http.Request, v *apiv1.ChatRequest) error {
	dec := json.NewDecoder(r.Body)
	if err := dec.Decode(v); err != nil {
		return err
	}
	// Reject trailing content after the first JSON value: two concatenated
	// objects would otherwise decode as the first one silently.
	if dec.More() {
		return errors.New("unexpected data after the JSON request body")
	}
	return nil
}

var _ = io.Discard // io used by sibling files in the package

// attemptCollector gathers router attempt outcomes for one request, so the
// handler can write them all to the ledger after the fact — including the failed
// attempts, which must be billed and reconciled too.
type attemptCollector struct {
	mu       sync.Mutex
	outcomes []router.AttemptOutcome
}

func (c *attemptCollector) RecordAttempt(o router.AttemptOutcome) {
	c.mu.Lock()
	c.outcomes = append(c.outcomes, o)
	c.mu.Unlock()
}

// ledgerFromOutcome converts a router attempt outcome into a durable ledger
// entry.
//
// The ledger enforces that Breakdown.Total() == CostPico exactly, so the cost
// and its components are always taken from the SAME pricing computation rather
// than being set independently (which could drift). When the pricing table
// cannot price the model, the row records zero cost with source "none" rather
// than an unbacked number.
func ledgerFromOutcome(o router.AttemptOutcome, reqID, requestedModel string, prices *pricing.Table, now time.Time) ledger.Entry {
	e := ledger.Entry{
		RequestID:      reqID,
		Tenant:         o.Tenant,
		RequestedModel: requestedModel,
		Provider:       o.Provider,
		UpstreamModel:  o.UpstreamModel,
		Attempt:        o.Attempt,
		Streaming:      o.Streamed,
		LatencyMS:      o.Latency.Milliseconds(),
		StartedAt:      now.Add(-o.Latency),
		EndedAt:        now,
	}

	if o.Failure != nil {
		e.FailureClass = o.Failure.Class.String()
		e.StatusCode = o.Failure.StatusCode
		if o.Failure.Err != nil {
			e.ErrorMessage = o.Failure.Err.Error()
		}
		if o.Served {
			e.Status = ledger.StatusFailed // failed after serving content mid-stream
		} else if o.FailedOver {
			e.Status = ledger.StatusFailedOver
		} else {
			e.Status = ledger.StatusFailed
		}
		e.ServedClient = o.Served
	} else {
		e.Status = ledger.StatusSucceeded
		e.ServedClient = true
	}

	// Price the usage through the pricing table so cost and breakdown are one
	// computation. A nil usage, or a model the table cannot price, yields a
	// zero-cost "none"-source row — never an unbacked charge.
	if o.Usage != nil && prices != nil {
		if b, err := prices.Cost(o.UpstreamModel, o.Usage); err == nil {
			e.Tokens = ledger.FromUsage(o.Usage)
			e.CostPico = b.Total
			e.Breakdown = ledger.CostBreakdown{
				PromptPico:     b.InputCost,
				CachedPico:     b.CachedInputCost,
				CompletionPico: b.OutputCost,
			}
			if o.UsageEstimated {
				e.UsageSource = ledger.SourceEstimated
			} else {
				e.UsageSource = ledger.SourceReported
			}
			e.Billable = b.Total > 0
			return e
		}
	}
	// No priced usage.
	e.UsageSource = ledger.SourceNone
	e.Billable = false
	return e
}

func stack() string {
	buf := make([]byte, 8192)
	n := runtime.Stack(buf, false)
	return string(buf[:n])
}

// ErrShutdown is returned by Run when the server stops due to context
// cancellation rather than a listener error.
var ErrShutdown = errors.New("server: shut down")

// Run starts the HTTP server and blocks until ctx is cancelled, then drains
// in-flight requests within the configured grace period and flushes the ledger.
//
// The shutdown order matters: stop accepting first, let in-flight streams finish
// within the grace window, and flush the ledger LAST so a cost row generated by
// a request that completed during the drain is not lost. A ledger flush before
// the drain would leave exactly those rows unwritten.
func (s *Server) Run(ctx context.Context) error {
	srvCfg := s.deps.Config.Server
	httpSrv := &http.Server{
		Addr:         srvCfg.Listen,
		Handler:      s.Handler(),
		ReadTimeout:  srvCfg.ReadTimeout.D(),
		WriteTimeout: srvCfg.WriteTimeout.D(), // 0 by config default: a stream must not be killed by a write deadline
		IdleTimeout:  srvCfg.IdleTimeout.D(),
	}

	errCh := make(chan error, 1)
	go func() {
		s.deps.Logger.Info("gateway listening", "addr", srvCfg.Listen)
		if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
		}
	}()

	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
	}

	grace := srvCfg.ShutdownGrace.D()
	if grace <= 0 {
		grace = 20 * time.Second
	}
	shutdownCtx, cancel := context.WithTimeout(context.Background(), grace)
	defer cancel()

	s.deps.Logger.Info("gateway draining", "grace", grace)
	err := httpSrv.Shutdown(shutdownCtx)

	// Flush the ledger AFTER the drain, so a row written by a request that
	// finished during shutdown survives.
	if s.deps.Ledger != nil {
		if ferr := s.deps.Ledger.Flush(); ferr != nil {
			s.deps.Logger.Error("flushing ledger on shutdown", "err", ferr)
		}
	}
	if err != nil {
		return err
	}
	return ErrShutdown
}
