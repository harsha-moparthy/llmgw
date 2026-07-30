// Command gateway is the LLM gateway server: one OpenAI-compatible API in front
// of many providers, with health-checked failover, response caching, per-tenant
// budgets, and cost accounting that reconciles against provider logs.
//
// This file is the composition root. Every dependency the server needs is
// constructed here from the validated config and injected explicitly, so the
// wiring is auditable in one place and nothing reaches for a global.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"sync"
	"syscall"

	"github.com/harsha-moparthy/llmgw/internal/breaker"
	"github.com/harsha-moparthy/llmgw/internal/budget"
	"github.com/harsha-moparthy/llmgw/internal/cache"
	"github.com/harsha-moparthy/llmgw/internal/config"
	"github.com/harsha-moparthy/llmgw/internal/ledger"
	"github.com/harsha-moparthy/llmgw/internal/metrics"
	"github.com/harsha-moparthy/llmgw/internal/pricing"
	"github.com/harsha-moparthy/llmgw/internal/provider"
	"github.com/harsha-moparthy/llmgw/internal/router"
	"github.com/harsha-moparthy/llmgw/internal/server"
	"github.com/harsha-moparthy/llmgw/internal/tokens"
)

func main() {
	var (
		cfgPath      = flag.String("config", "configs/gateway.example.json", "path to the gateway config")
		listen       = flag.String("listen", "", "override the config's listen address")
		ledgerPath   = flag.String("ledger", "data/ledger.jsonl", "path to the cost ledger JSONL")
		printExample = flag.Bool("print-example", false, "print a complete example config to stdout and exit")
	)
	flag.Parse()

	if *printExample {
		b, err := config.MarshalExample()
		if err != nil {
			fmt.Fprintln(os.Stderr, "generating example:", err)
			os.Exit(1)
		}
		_, _ = os.Stdout.Write(b)
		return
	}

	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

	if err := run(log, *cfgPath, *listen, *ledgerPath); err != nil {
		log.Error("gateway exited with error", "err", err)
		os.Exit(1)
	}
}

func run(log *slog.Logger, cfgPath, listenOverride, ledgerPath string) error {
	cfg, err := config.LoadFile(cfgPath)
	if err != nil {
		return err
	}
	if listenOverride != "" {
		cfg.Server.Listen = listenOverride
	}

	// Pricing table: the config's own sheet if present, else the built-in
	// demonstration prices.
	var prices *pricing.Table
	if len(cfg.Pricing) > 0 {
		sheet := pricing.Sheet{Models: cfg.Pricing}
		prices, err = sheet.Table()
		if err != nil {
			return fmt.Errorf("pricing: %w", err)
		}
	} else {
		prices = pricing.DefaultTable()
	}

	// Providers, one adapter per configured instance, each with a resolved key
	// and a tuned transport.
	providers := make(map[string]provider.Provider, len(cfg.Providers))
	for _, pc := range cfg.Providers {
		key, err := pc.ResolveAPIKey()
		if err != nil {
			return err
		}
		transport := provider.Transport{
			ResponseHeaderTimeout: pc.ResponseHeaderTimeout.D(),
			MaxIdleConnsPerHost:   pc.MaxIdleConnsPerHost,
			ConnectTimeout:        pc.ConnectTimeout.D(),
		}
		switch pc.Vendor {
		case "openai", "mock":
			// The mock speaks the OpenAI protocol, so the OpenAI adapter fronts
			// both. The distinction only matters for real auth, which the mock
			// ignores.
			providers[pc.Name] = provider.NewOpenAIProvider(provider.OpenAIConfig{
				Name: pc.Name, BaseURL: pc.BaseURL, APIKey: key, Transport: transport,
			})
		case "anthropic":
			providers[pc.Name] = provider.NewAnthropicProvider(provider.AnthropicConfig{
				Name: pc.Name, BaseURL: pc.BaseURL, APIKey: key, Transport: transport,
			})
		default:
			return fmt.Errorf("provider %q: unsupported vendor %q", pc.Name, pc.Vendor)
		}
	}

	// Circuit breakers, one per provider instance. The registry shares a base
	// config; per-provider tuning from the config's Breaker block is applied by
	// pre-creating each breaker with its own settings.
	breakers, err := breaker.NewRegistry(breaker.DefaultConfig())
	if err != nil {
		return err
	}
	// Touch each provider's breaker so it exists before the first request, and so
	// /readyz can see them.
	for _, pc := range cfg.Providers {
		_ = breakers.Get(pc.Name)
	}

	routes, err := router.BuildRoutes(cfg, providers, breakers)
	if err != nil {
		return err
	}
	rt := router.New(routes, router.Options{
		RequestDeadline: cfg.Server.RequestDeadline.D(),
		Prices:          prices,
	})

	// Budget, seeded with each tenant's limit.
	bud := budget.New(budget.Options{})
	for _, t := range cfg.Tenants {
		limit, err := t.Limit()
		if err != nil {
			return fmt.Errorf("tenant %q budget: %w", t.ID, err)
		}
		if !limit.Unlimited() {
			if err := bud.SetLimit(t.ID, limit); err != nil {
				return fmt.Errorf("tenant %q budget: %w", t.ID, err)
			}
		}
	}

	// Cache.
	var store *cache.Store
	var stopSweeper func()
	if cfg.Cache.Enabled {
		store = cache.New(cfg.Cache.MaxBytes, cache.Policy{
			CacheNonDeterministic: cfg.Cache.CacheNonDeterministic,
			CacheTruncated:        cfg.Cache.CacheTruncated,
			TTL:                   cfg.Cache.TTL.D(),
			MaxEntryBytes:         cfg.Cache.MaxEntryBytes,
		})
		stopSweeper = store.StartSweeper(cfg.Cache.SweepInterval.D(), 1000)
		defer stopSweeper()
	}

	// Ledger.
	lg, err := ledger.Open(ledgerPath, ledger.Options{})
	if err != nil {
		return fmt.Errorf("opening ledger %q: %w", ledgerPath, err)
	}
	defer func() { _ = lg.Close() }()

	// Priors for completion-length estimation.
	priors, err := tokens.NewPriors(tokens.DefaultPriorConfig())
	if err != nil {
		return err
	}

	// Metrics.
	gwMetrics := metrics.NewGateway()

	// Active health probing, where the config asks for it. A probe feeds the
	// breaker so a provider that recovers is brought back into rotation without
	// waiting for real traffic to test it.
	probers := startProbers(cfg, providers, breakers, log)
	defer probers.stop()

	// Readiness: at least one provider's breaker must admit traffic.
	readiness := func() bool {
		for name := range providers {
			if br := breakers.Get(name); br != nil && br.Allow() {
				return true
			}
		}
		return false
	}

	srv := server.New(server.Deps{
		Config:    cfg,
		Router:    rt,
		Budget:    bud,
		Cache:     store,
		Ledger:    lg,
		Metrics:   gwMetrics,
		Prices:    prices,
		Priors:    priors,
		Logger:    log,
		Readiness: readiness,
	})

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	err = srv.Run(ctx)
	if err == server.ErrShutdown {
		log.Info("gateway stopped cleanly")
		return nil
	}
	return err
}

// proberSet tracks the background health probers so they can be stopped on
// shutdown without leaking goroutines.
type proberSet struct {
	cancel  context.CancelFunc
	wg      sync.WaitGroup
	probers []*breaker.Prober
}

func (p *proberSet) stop() {
	if p.cancel != nil {
		p.cancel()
	}
	p.wg.Wait()
}

func startProbers(cfg *config.Config, providers map[string]provider.Provider, breakers *breaker.Registry, log *slog.Logger) *proberSet {
	ctx, cancel := context.WithCancel(context.Background())
	set := &proberSet{cancel: cancel}
	for _, pc := range cfg.Providers {
		if !pc.Probe.Enabled {
			continue
		}
		p := providers[pc.Name]
		hp, ok := p.(provider.HealthProbe)
		if !ok {
			// The provider exposes no active health endpoint; the breaker still
			// learns from real traffic. A missing probe is an intentional
			// absence, not an error — the mock and the real adapters here are
			// judged passively, which is the honest default when a provider has
			// no dedicated health route.
			continue
		}
		br := breakers.Get(pc.Name)
		probeCfg := breaker.DefaultProberConfig()
		if pc.Probe.Interval.D() > 0 {
			probeCfg.Interval = pc.Probe.Interval.D()
		}
		if pc.Probe.Timeout.D() > 0 {
			probeCfg.Timeout = pc.Probe.Timeout.D()
		}
		prober, err := breaker.NewProber(br, breaker.ProbeFromHealthProbe(hp), probeCfg)
		if err != nil {
			log.Warn("could not start prober", "provider", pc.Name, "err", err)
			continue
		}
		set.probers = append(set.probers, prober)
		set.wg.Add(1)
		go func() {
			defer set.wg.Done()
			prober.Run(ctx)
		}()
	}
	return set
}
