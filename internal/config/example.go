package config

import (
	"encoding/json"

	"github.com/harsha-moparthy/llmgw/internal/pricing"
)

// Example returns a complete, realistic, valid configuration.
//
// It is the source of the committed configs/gateway.example.json (the cmd/gateway
// binary writes it out with -print-example), and the round-trip test pins that
// the example this package produces both validates and survives a
// marshal/unmarshal cycle. That matters because an example config that has
// drifted out of validity is worse than none: it is the first thing a new
// operator copies.
//
// The stack it describes is the local demo: two mock-provider instances behind a
// single "gw-chat" route, so failover has somewhere to go, plus a "gw-smart"
// route and a budget-degradation example. Every value here runs offline with no
// credentials.
func Example() *Config {
	return &Config{
		Server: Server{
			Listen:          "127.0.0.1:8080",
			ReadTimeout:     Duration(15_000_000_000),  // 15s
			WriteTimeout:    Duration(0),               // 0: no socket write deadline, so a long stream is not killed
			IdleTimeout:     Duration(120_000_000_000), // 120s
			ShutdownGrace:   Duration(20_000_000_000),  // 20s
			MaxRequestBytes: 1 << 20,                   // 1 MiB
			RequestDeadline: Duration(60_000_000_000),  // 60s across all failover attempts
		},
		Providers: []Provider{
			{
				Name:                  "mock-primary",
				Vendor:                "mock",
				BaseURL:               "http://127.0.0.1:9001",
				MaxIdleConnsPerHost:   64,
				ResponseHeaderTimeout: Duration(30_000_000_000),
				Breaker: BreakerCfg{
					WindowSize: 20, MinSamples: 5, FailureRatio: 0.5,
					Cooldown: Duration(2_000_000_000), MaxCooldown: Duration(30_000_000_000),
					HalfOpenProbes: 2,
				},
				Probe: ProbeCfg{Enabled: true, Interval: Duration(5_000_000_000), Timeout: Duration(2_000_000_000), Path: "/admin/health"},
			},
			{
				Name:                  "mock-secondary",
				Vendor:                "mock",
				BaseURL:               "http://127.0.0.1:9002",
				MaxIdleConnsPerHost:   64,
				ResponseHeaderTimeout: Duration(30_000_000_000),
				Breaker: BreakerCfg{
					WindowSize: 20, MinSamples: 5, FailureRatio: 0.5,
					Cooldown: Duration(2_000_000_000), MaxCooldown: Duration(30_000_000_000),
					HalfOpenProbes: 2,
				},
				Probe: ProbeCfg{Enabled: true, Interval: Duration(5_000_000_000), Timeout: Duration(2_000_000_000), Path: "/admin/health"},
			},
		},
		Routes: map[string]Route{
			// The headline route: primary then secondary. Killing the primary
			// under load is the failover demo, and the secondary is where it
			// fails over TO.
			"gw-chat": {
				Targets: []Target{
					{Provider: "mock-primary", Model: "mock-fast"},
					{Provider: "mock-secondary", Model: "mock-fast"},
				},
				AllowFailover: true,
				MaxAttempts:   3,
			},
			// A more expensive route, used to demonstrate cost control and the
			// budget-degradation fallback (gw-smart degrades to the cheaper
			// mock-fast when a tenant crosses its soft threshold).
			"gw-smart": {
				Targets: []Target{
					{Provider: "mock-primary", Model: "mock-smart"},
					{Provider: "mock-secondary", Model: "mock-smart"},
					{Provider: "mock-primary", Model: "mock-fast"},
				},
				AllowFailover: true,
				MaxAttempts:   3,
			},
		},
		Tenants: []Tenant{
			{
				ID: "bench",
				// Hash of "bench-key". The plaintext lives only in the load-test
				// scripts and .env.example, never in a committed config.
				APIKeyHash:    HashKey("bench-key"),
				AllowedModels: []string{"gw-chat", "gw-smart"},
			},
			{
				ID:         "small",
				APIKeyHash: HashKey("small-budget-key"),
				// A deliberately tiny hourly budget so the budget-pressure demo
				// actually trips within a short k6 run: at ~$0.00003 per cheap
				// request, $0.005/hour is exhausted in ~150 requests. A realistic
				// production budget would be orders of magnitude larger; this one
				// is sized to make the rejection semantics observable in seconds.
				BudgetLimit:      "0.005",
				BudgetPeriod:     "hour",
				SoftThresholdPct: 80,
				AllowedModels:    []string{"gw-chat", "gw-smart"},
			},
		},
		// The gateway's own price sheet for the mock models. These prices are the
		// gateway's side of the cost reconciliation; the mock provider carries an
		// independent copy in configs/mockprovider.json, and the reconciliation
		// proves they agree. They are expressed as decimal strings, so
		// money.ParseUSD converts them exactly with no float rounding.
		Pricing: []pricing.ModelPrice{
			{Model: "mock-fast", InputPerMTok: "0.15", CachedInputPerMTok: "0.075", OutputPerMTok: "0.60",
				Note: "demo price for the mock provider; not a real model"},
			{Model: "mock-smart", InputPerMTok: "2.50", CachedInputPerMTok: "1.25", OutputPerMTok: "10.00",
				Note: "demo price for the mock provider; not a real model"},
			{Model: "mock-thinking", InputPerMTok: "1.10", CachedInputPerMTok: "0.55", OutputPerMTok: "4.40",
				Note: "demo reasoning model; output rate covers reasoning tokens too"},
			{Model: "mock-cached", InputPerMTok: "0.15", CachedInputPerMTok: "0.0375", OutputPerMTok: "0.60",
				Note: "demo model that reports cached prompt tokens"},
		},
		Cache: Cache{
			Enabled:       true,
			MaxBytes:      64 << 20,
			TTL:           Duration(600_000_000_000), // 10m
			MaxEntryBytes: 1 << 20,
			SweepInterval: Duration(60_000_000_000),
		},
	}
}

// MarshalExample returns the indented JSON of the example config.
func MarshalExample() ([]byte, error) {
	b, err := json.MarshalIndent(Example(), "", "  ")
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}
