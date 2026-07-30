package provider

import (
	"net"
	"net/http"
	"time"
)

// Transport builds the *http.Client an adapter uses to reach one upstream
// instance.
//
// # The connection pool is the point
//
// The stdlib's default MaxIdleConnsPerHost is 2. For a program that occasionally
// calls an API that is fine; for a gateway it is a self-inflicted bottleneck,
// because every request beyond the second concurrent one to the same provider
// waits for a TCP+TLS handshake before it can even send. A gateway's whole job
// is to multiplex many clients onto an upstream, so the pool has to be sized for
// concurrency, and this is one of the highest-leverage numbers in the codebase.
//
// # No client Timeout on a streaming path
//
// http.Client.Timeout bounds the ENTIRE exchange including reading the body. On
// a streaming completion the body is read for as long as the model generates,
// which is legitimately tens of seconds, so a client Timeout would sever healthy
// long streams. The right controls are per-phase: a dial timeout, a TLS
// handshake timeout, and a response-header timeout bound the parts that should
// be fast, while the streaming read is bounded by the request context instead.
type Transport struct {
	// ConnectTimeout bounds establishing the TCP connection.
	ConnectTimeout time.Duration
	// TLSHandshakeTimeout bounds the TLS handshake.
	TLSHandshakeTimeout time.Duration
	// ResponseHeaderTimeout bounds waiting for the response headers — i.e. the
	// upstream's time-to-first-byte. This is what catches a provider that
	// accepted the connection but is not answering, without killing a stream
	// that has started flowing.
	ResponseHeaderTimeout time.Duration
	// MaxIdleConnsPerHost sizes the connection pool per upstream host.
	MaxIdleConnsPerHost int
	// IdleConnTimeout is how long an idle pooled connection is kept.
	IdleConnTimeout time.Duration
}

// DefaultTransport returns sensible gateway defaults.
func DefaultTransport() Transport {
	return Transport{
		ConnectTimeout:        5 * time.Second,
		TLSHandshakeTimeout:   5 * time.Second,
		ResponseHeaderTimeout: 30 * time.Second,
		MaxIdleConnsPerHost:   64,
		IdleConnTimeout:       90 * time.Second,
	}
}

// Client builds an *http.Client from the transport settings.
func (t Transport) Client() *http.Client {
	if t.MaxIdleConnsPerHost <= 0 {
		t.MaxIdleConnsPerHost = DefaultTransport().MaxIdleConnsPerHost
	}
	if t.ConnectTimeout <= 0 {
		t.ConnectTimeout = DefaultTransport().ConnectTimeout
	}
	if t.TLSHandshakeTimeout <= 0 {
		t.TLSHandshakeTimeout = DefaultTransport().TLSHandshakeTimeout
	}
	if t.IdleConnTimeout <= 0 {
		t.IdleConnTimeout = DefaultTransport().IdleConnTimeout
	}
	if t.ResponseHeaderTimeout <= 0 {
		// A zero here would mean "wait forever for the first byte", so a hung
		// provider would never be detected on the non-streaming path. Default it
		// rather than trusting every caller to set it.
		t.ResponseHeaderTimeout = DefaultTransport().ResponseHeaderTimeout
	}
	dialer := &net.Dialer{Timeout: t.ConnectTimeout, KeepAlive: 30 * time.Second}
	tr := &http.Transport{
		DialContext:           dialer.DialContext,
		MaxIdleConns:          t.MaxIdleConnsPerHost * 4,
		MaxIdleConnsPerHost:   t.MaxIdleConnsPerHost,
		IdleConnTimeout:       t.IdleConnTimeout,
		TLSHandshakeTimeout:   t.TLSHandshakeTimeout,
		ResponseHeaderTimeout: t.ResponseHeaderTimeout,
		// ExpectContinueTimeout guards against a provider that never sends the
		// 100-continue we might wait for.
		ExpectContinueTimeout: 1 * time.Second,
		ForceAttemptHTTP2:     true,
	}
	return &http.Client{
		Transport: tr,
		// Deliberately no Timeout. See the type doc: a whole-exchange timeout
		// kills long legitimate streams; deadlines come from the context.
		Timeout: 0,
	}
}
