// Package config is the gateway's configuration: JSON in, a validated in-memory
// model out.
//
// # Why JSON and not YAML
//
// YAML is the conventional choice for this kind of file and it would be more
// pleasant to hand-edit. It is not used here for one reason: this project has a
// zero-dependency rule, the standard library has no YAML parser, and a
// hand-rolled YAML subset is a worse liability than JSON's verbosity — YAML's
// surprising corners (the Norway problem, significant whitespace, anchors) are
// exactly where a config parser bug hides, and a config parser bug is a
// production outage. JSON's grammar is small enough that encoding/json is
// trustworthy.
//
// # Validation is the point of this package
//
// A gateway that boots with a broken route and discovers it on the first
// production request has failed at the one job config validation exists to do.
// So Validate is exhaustive and every error names the exact offending path
// (routes["gpt-4o"].targets[1].provider), because "invalid config" with no
// location is a config file the operator now has to bisect by hand at 3am.
//
// # Credentials are never in the file
//
// A provider's API key is named by an environment variable, never written in the
// config. A config file gets committed to git, pasted into a ticket, and shipped
// inside a container image; each of those is a place a key does not belong. The
// file carries the variable name; the process resolves it at startup and fails
// loudly if it is unset.
package config

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/budget"
	"github.com/harsha-moparthy/llmgw/internal/money"
	"github.com/harsha-moparthy/llmgw/internal/pricing"
)

// Config is the whole gateway configuration.
type Config struct {
	Server    Server               `json:"server"`
	Providers []Provider           `json:"providers"`
	Routes    map[string]Route     `json:"routes"`
	Tenants   []Tenant             `json:"tenants"`
	Pricing   []pricing.ModelPrice `json:"pricing"`
	Cache     Cache                `json:"cache"`
}

// Server holds the HTTP server settings.
type Server struct {
	Listen          string   `json:"listen"`
	ReadTimeout     Duration `json:"read_timeout"`
	WriteTimeout    Duration `json:"write_timeout"`
	IdleTimeout     Duration `json:"idle_timeout"`
	ShutdownGrace   Duration `json:"shutdown_grace"`
	MaxRequestBytes int64    `json:"max_request_bytes"`
	// RequestDeadline caps the total time a single client request may spend
	// across all failover attempts. It is separate from WriteTimeout because a
	// streaming response legitimately outlives any single write, so the whole-
	// request bound has to be expressed in wall time the handler enforces via
	// context, not in a socket timeout.
	RequestDeadline Duration `json:"request_deadline"`
}

// Provider is one upstream instance.
//
// The instance is identified by Name, not by vendor, because a real deployment
// fronts the same vendor several times — different keys, regions, or quota pools
// — and each of those needs its own health state and circuit breaker. Two
// entries with vendor "openai" and names "openai-primary"/"openai-backup" are
// two independent providers as far as failover and health are concerned.
type Provider struct {
	Name    string `json:"name"`
	Vendor  string `json:"vendor"`
	BaseURL string `json:"base_url"`
	// APIKeyEnv is the NAME of the environment variable holding the key, not the
	// key. See the package doc.
	APIKeyEnv string `json:"api_key_env"`

	// Timeouts. Zero means a sensible default is applied at build time.
	ConnectTimeout        Duration `json:"connect_timeout"`
	ResponseHeaderTimeout Duration `json:"response_header_timeout"`
	// MaxIdleConnsPerHost sizes the connection pool. The stdlib default is 2,
	// which serialises a gateway's upstream calls behind connection setup; this
	// is one of the highest-leverage numbers in the whole configuration.
	MaxIdleConnsPerHost int `json:"max_idle_conns_per_host"`

	Breaker BreakerCfg `json:"breaker"`
	Probe   ProbeCfg   `json:"probe"`
}

// BreakerCfg mirrors the breaker's tunables in serialisable form. Zero fields
// fall back to breaker.DefaultConfig at build time.
type BreakerCfg struct {
	WindowSize     int      `json:"window_size"`
	MinSamples     int      `json:"min_samples"`
	FailureRatio   float64  `json:"failure_ratio"`
	Cooldown       Duration `json:"cooldown"`
	MaxCooldown    Duration `json:"max_cooldown"`
	HalfOpenProbes int      `json:"half_open_probes"`
}

// ProbeCfg configures active health probing.
type ProbeCfg struct {
	Enabled  bool     `json:"enabled"`
	Interval Duration `json:"interval"`
	Timeout  Duration `json:"timeout"`
	Path     string   `json:"path"`
}

// Route maps a client-facing model alias to an ordered failover chain.
type Route struct {
	// Targets are tried in order. The first with a healthy breaker and adequate
	// budget serves the request; the rest are the failover chain.
	Targets []Target `json:"targets"`
	// AllowFailover gates cross-target retries for this route. Some routes (a
	// single-provider model with no equivalent) legitimately have no failover,
	// and forcing a retry there only burns latency.
	AllowFailover bool `json:"allow_failover"`
	// MaxAttempts caps attempts for this route, independent of the chain length,
	// so a 20-target chain does not become a 20x latency amplifier on a bad day.
	MaxAttempts int `json:"max_attempts"`
}

// Target is one (provider instance, upstream model) pair in a route.
type Target struct {
	Provider string `json:"provider"`
	Model    string `json:"model"`
}

// Tenant is one API client.
type Tenant struct {
	ID string `json:"id"`
	// APIKeyHash is the hex SHA-256 of the tenant's bearer key. The plaintext
	// key is never stored: a leaked config must not be a leaked set of
	// credentials, and auth compares hashes in constant time (see HashKey and
	// the server's auth path).
	APIKeyHash string `json:"api_key_hash"`

	BudgetLimit  string `json:"budget_limit"`  // dollar amount, e.g. "50.00"; empty = unlimited
	BudgetPeriod string `json:"budget_period"` // hour|day|month
	// SoftThresholdPct is the utilisation at which the gateway begins routing to
	// cheaper models, as a whole-number percentage. 0 disables graceful
	// degradation, so the tenant goes straight from fine to rejected.
	SoftThresholdPct int `json:"soft_threshold_pct"`

	// AllowedModels restricts which route aliases this tenant may call. Empty
	// means all routes are allowed.
	AllowedModels []string `json:"allowed_models"`

	// CachePool names a shared cache pool this tenant consents to. Empty means
	// the tenant's cache is isolated to itself, which is the safe default — a
	// shared cache leaks prompts between tenants as cache hits.
	CachePool string `json:"cache_pool"`
}

// Cache holds the response-cache settings.
type Cache struct {
	Enabled               bool     `json:"enabled"`
	MaxBytes              int      `json:"max_bytes"`
	TTL                   Duration `json:"ttl"`
	MaxEntryBytes         int      `json:"max_entry_bytes"`
	CacheNonDeterministic bool     `json:"cache_non_deterministic"`
	CacheTruncated        bool     `json:"cache_truncated"`
	SweepInterval         Duration `json:"sweep_interval"`
}

// Duration is a time.Duration that (de)serialises as a Go duration string
// ("500ms", "30s") rather than as a nanosecond integer.
//
// The stdlib marshals time.Duration as an int64 of nanoseconds, which is
// unreadable in a config file and a rich source of "why is my 30-second timeout
// actually 30 nanoseconds" bugs. This type makes the config say what it means.
type Duration time.Duration

// MarshalJSON renders the duration as a string.
func (d Duration) MarshalJSON() ([]byte, error) {
	return json.Marshal(time.Duration(d).String())
}

// UnmarshalJSON accepts a duration string or a plain number of seconds.
func (d *Duration) UnmarshalJSON(b []byte) error {
	var v any
	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}
	switch t := v.(type) {
	case string:
		parsed, err := time.ParseDuration(t)
		if err != nil {
			return fmt.Errorf("invalid duration %q: %w", t, err)
		}
		*d = Duration(parsed)
	case float64:
		// A bare number is interpreted as seconds, which is the least surprising
		// reading of "timeout: 30" and matches how most ops tools behave.
		*d = Duration(time.Duration(t * float64(time.Second)))
	default:
		return fmt.Errorf("duration must be a string or a number, got %T", v)
	}
	return nil
}

// D returns the value as a time.Duration.
func (d Duration) D() time.Duration { return time.Duration(d) }

// Load parses and validates a config from r.
func Load(r io.Reader) (*Config, error) {
	dec := json.NewDecoder(r)
	// Reject unknown fields. A typo'd key ("timout") in a config that otherwise
	// parses is a silent misconfiguration — the intended setting keeps its
	// default and nothing complains. Failing on unknown fields turns that into a
	// startup error, which is where it belongs.
	dec.DisallowUnknownFields()
	var c Config
	if err := dec.Decode(&c); err != nil {
		return nil, fmt.Errorf("config: parsing: %w", err)
	}
	if err := c.Validate(); err != nil {
		return nil, err
	}
	return &c, nil
}

// LoadFile reads and validates a config file.
func LoadFile(path string) (*Config, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("config: %w", err)
	}
	defer func() { _ = f.Close() }()
	return Load(f)
}

// Validate checks the whole config for internal consistency.
//
// The checks are ordered from structural (is this a well-formed set of names)
// to referential (do the cross-references resolve), because a referential error
// message is only useful once the names it refers to are known to be valid.
func (c *Config) Validate() error {
	var errs []string
	add := func(format string, args ...any) { errs = append(errs, fmt.Sprintf(format, args...)) }

	// Providers: unique names, known vendor, resolvable key.
	provNames := map[string]bool{}
	for i, p := range c.Providers {
		path := fmt.Sprintf("providers[%d]", i)
		if p.Name == "" {
			add("%s.name is required", path)
			continue
		}
		if provNames[p.Name] {
			add("%s.name %q is a duplicate", path, p.Name)
		}
		provNames[p.Name] = true
		switch p.Vendor {
		case "openai", "anthropic", "mock":
		case "":
			add("%s (%q).vendor is required", path, p.Name)
		default:
			add("%s (%q).vendor %q is not one of openai, anthropic, mock", path, p.Name, p.Vendor)
		}
		if p.Vendor != "mock" && p.BaseURL == "" {
			add("%s (%q).base_url is required for a %s provider", path, p.Name, p.Vendor)
		}
		if p.Breaker.FailureRatio < 0 || p.Breaker.FailureRatio > 1 {
			add("%s (%q).breaker.failure_ratio %g is not in [0,1]", path, p.Name, p.Breaker.FailureRatio)
		}
	}

	// Build the pricing table once so a route can be checked against it. An
	// invalid pricing entry is reported here rather than deferred, because a
	// route pointing at an unpriced model is a billing hole.
	var priceTable *pricing.Table
	if len(c.Pricing) > 0 {
		sheet := pricing.Sheet{Models: c.Pricing}
		t, err := sheet.Table()
		if err != nil {
			add("pricing: %v", err)
		} else {
			priceTable = t
		}
	} else {
		priceTable = pricing.DefaultTable()
	}

	// Routes: non-empty, targets reference known providers, priced models.
	if len(c.Routes) == 0 {
		add("routes: at least one route is required or the gateway can serve nothing")
	}
	// Sort alias names so the error output is deterministic — a map ranged in
	// random order produces errors in random order, which makes a diff of two
	// validation runs unreadable.
	aliases := make([]string, 0, len(c.Routes))
	for alias := range c.Routes {
		aliases = append(aliases, alias)
	}
	sort.Strings(aliases)
	for _, alias := range aliases {
		route := c.Routes[alias]
		rp := fmt.Sprintf("routes[%q]", alias)
		if len(route.Targets) == 0 {
			add("%s has no targets", rp)
			continue
		}
		if route.MaxAttempts < 0 {
			add("%s.max_attempts %d is negative", rp, route.MaxAttempts)
		}
		for j, tgt := range route.Targets {
			tp := fmt.Sprintf("%s.targets[%d]", rp, j)
			if tgt.Provider == "" {
				add("%s.provider is required", tp)
			} else if !provNames[tgt.Provider] {
				add("%s.provider %q is not a defined provider instance", tp, tgt.Provider)
			}
			if tgt.Model == "" {
				add("%s.model is required", tp)
				continue
			}
			if priceTable != nil {
				if _, err := priceTable.Lookup(tgt.Model); err != nil {
					add("%s.model %q is not priced; a route to an unpriced model is a billing hole", tp, tgt.Model)
				}
			}
		}
	}

	// Tenants: unique ids, valid key hash, coherent budget, allowlist points at
	// real routes.
	tenantIDs := map[string]bool{}
	for i, t := range c.Tenants {
		tp := fmt.Sprintf("tenants[%d]", i)
		if t.ID == "" {
			add("%s.id is required", tp)
			continue
		}
		if tenantIDs[t.ID] {
			add("%s.id %q is a duplicate", tp, t.ID)
		}
		tenantIDs[t.ID] = true
		if t.APIKeyHash == "" {
			add("%s (%q).api_key_hash is required", tp, t.ID)
		} else if len(t.APIKeyHash) != 64 || !isHex(t.APIKeyHash) {
			add("%s (%q).api_key_hash must be a 64-char hex SHA-256 (use HashKey to produce one)", tp, t.ID)
		}
		if t.BudgetLimit != "" {
			amt, err := money.ParseUSD(t.BudgetLimit)
			if err != nil {
				add("%s (%q).budget_limit %q: %v", tp, t.ID, t.BudgetLimit, err)
			} else if amt < 0 {
				add("%s (%q).budget_limit is negative", tp, t.ID)
			}
			if t.BudgetPeriod == "" {
				add("%s (%q) has a budget_limit but no budget_period", tp, t.ID)
			} else if _, err := budget.ParsePeriod(t.BudgetPeriod); err != nil {
				add("%s (%q).budget_period %q: %v", tp, t.ID, t.BudgetPeriod, err)
			}
		}
		if t.SoftThresholdPct < 0 || t.SoftThresholdPct > 100 {
			add("%s (%q).soft_threshold_pct %d is not in [0,100]", tp, t.ID, t.SoftThresholdPct)
		}
		for _, m := range t.AllowedModels {
			if _, ok := c.Routes[m]; !ok {
				add("%s (%q).allowed_models references %q, which is not a defined route", tp, t.ID, m)
			}
		}
	}

	if c.Server.MaxRequestBytes < 0 {
		add("server.max_request_bytes is negative")
	}

	if len(errs) > 0 {
		sort.Strings(errs)
		return fmt.Errorf("config: %d validation error(s):\n  - %s",
			len(errs), strings.Join(errs, "\n  - "))
	}
	return nil
}

// Limit builds a budget.Limit from a tenant's fields. The tenant must have
// passed Validate.
func (t Tenant) Limit() (budget.Limit, error) {
	if t.BudgetLimit == "" {
		return budget.Limit{}, nil // unlimited
	}
	amt, err := money.ParseUSD(t.BudgetLimit)
	if err != nil {
		return budget.Limit{}, err
	}
	period, err := budget.ParsePeriod(t.BudgetPeriod)
	if err != nil {
		return budget.Limit{}, err
	}
	return budget.Limit{
		Amount:          amt,
		Period:          period,
		SoftBasisPoints: t.SoftThresholdPct * 100,
	}, nil
}

// ResolveAPIKey reads the provider's key from its named environment variable,
// failing loudly if unset. Called at startup, never on the request path.
func (p Provider) ResolveAPIKey() (string, error) {
	if p.APIKeyEnv == "" {
		if p.Vendor == "mock" {
			return "", nil // the mock accepts any key
		}
		return "", fmt.Errorf("provider %q: api_key_env is required for a %s provider", p.Name, p.Vendor)
	}
	v := os.Getenv(p.APIKeyEnv)
	if v == "" && p.Vendor != "mock" {
		return "", fmt.Errorf("provider %q: environment variable %s is unset", p.Name, p.APIKeyEnv)
	}
	return v, nil
}

func isHex(s string) bool {
	for _, c := range s {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
			return false
		}
	}
	return true
}

// HashKey returns the hex SHA-256 of a plaintext API key, which is what belongs
// in a tenant's api_key_hash. The plaintext is never stored, so a leaked config
// is not a leaked credential set — and auth compares these hashes in constant
// time.
func HashKey(plaintext string) string {
	sum := sha256.Sum256([]byte(plaintext))
	return hex.EncodeToString(sum[:])
}

// ErrEmpty is returned when there is nothing to parse.
var ErrEmpty = errors.New("config: empty input")
