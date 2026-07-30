// Command mockprovider runs a programmable, OpenAI-compatible mock LLM provider.
//
// It is the upstream the gateway is measured against. Two instances of it behind
// the gateway give failover somewhere to go; its admin endpoint lets a benchmark
// kill a provider mid-flight; and its independently-computed request log is one
// side of the cost reconciliation. See internal/mockprov for the design.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/mockprov"
)

func main() {
	var (
		listen  = flag.String("listen", "127.0.0.1:9001", "address for the OpenAI-compatible endpoint")
		admin   = flag.String("admin", "127.0.0.1:9091", "address for the fault-control admin endpoint")
		cfgPath = flag.String("config", "configs/mockprovider.json", "path to the model/fault config")
		logPath = flag.String("log", "", "path to the JSONL request log (overrides the config)")
	)
	flag.Parse()

	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

	cfg, err := loadConfig(*cfgPath)
	if err != nil {
		log.Error("loading config", "err", err)
		os.Exit(1)
	}
	// Command-line addresses win over the config so the stack script can place
	// two instances on different ports from one config file.
	cfg.Listen = *listen
	cfg.AdminListen = *admin
	if *logPath != "" {
		cfg.LogPath = *logPath
	}

	srv, err := mockprov.New(cfg, mockprov.Options{})
	if err != nil {
		log.Error("building server", "err", err)
		os.Exit(1)
	}
	if err := srv.Start(); err != nil {
		log.Error("starting server", "err", err)
		os.Exit(1)
	}
	log.Info("mock provider up",
		"endpoint", "http://"+cfg.Listen,
		"admin", "http://"+cfg.AdminListen,
		"log", cfg.LogPath,
		"kill_hint", fmt.Sprintf("curl -XPOST 'http://%s/admin/fault?down=true'", cfg.AdminListen),
	)

	// Wait for a signal, then shut down gracefully so the request log's tail is
	// not lost.
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	<-ctx.Done()

	log.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Error("shutdown", "err", err)
		os.Exit(1)
	}
}

func loadConfig(path string) (*mockprov.Config, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	var cfg mockprov.Config
	dec := json.NewDecoder(f)
	if err := dec.Decode(&cfg); err != nil {
		return nil, fmt.Errorf("parsing %s: %w", path, err)
	}
	return &cfg, nil
}
