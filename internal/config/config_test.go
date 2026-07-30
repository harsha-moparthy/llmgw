package config

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/pricing"
)

// TestExampleValidatesAndRoundTrips is the load-bearing test for this package.
// The example config is the first thing a new operator copies, so an example
// that has drifted out of validity is worse than none. This asserts it both
// validates and survives a marshal/unmarshal cycle unchanged.
func TestExampleValidatesAndRoundTrips(t *testing.T) {
	ex := Example()
	if err := ex.Validate(); err != nil {
		t.Fatalf("the example config does not validate: %v", err)
	}
	b, err := MarshalExample()
	if err != nil {
		t.Fatalf("MarshalExample: %v", err)
	}
	loaded, err := Load(bytes.NewReader(b))
	if err != nil {
		t.Fatalf("the marshalled example does not load: %v", err)
	}
	if err := loaded.Validate(); err != nil {
		t.Fatalf("the reloaded example does not validate: %v", err)
	}
	// The round trip must preserve the structure. Re-marshalling the loaded copy
	// must produce identical bytes.
	b2, err := json.MarshalIndent(loaded, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	b2 = append(b2, '\n')
	if !bytes.Equal(b, b2) {
		t.Errorf("round trip changed the config bytes:\n--- first ---\n%s\n--- second ---\n%s", b, b2)
	}
}

// TestDurationParsing pins the human-readable duration handling: the whole
// reason this type exists is that the stdlib marshals a duration as a
// nanosecond integer, which is where "my 30s timeout is actually 30ns" bugs come
// from.
func TestDurationParsing(t *testing.T) {
	tests := []struct {
		in   string
		want time.Duration
	}{
		{`"30s"`, 30 * time.Second},
		{`"500ms"`, 500 * time.Millisecond},
		{`"2m"`, 2 * time.Minute},
		{`0`, 0},
		{`30`, 30 * time.Second}, // a bare number is seconds
		{`1.5`, 1500 * time.Millisecond},
	}
	for _, tc := range tests {
		var d Duration
		if err := json.Unmarshal([]byte(tc.in), &d); err != nil {
			t.Errorf("Unmarshal(%s): %v", tc.in, err)
			continue
		}
		if d.D() != tc.want {
			t.Errorf("Unmarshal(%s) = %v, want %v", tc.in, d.D(), tc.want)
		}
	}
	// Round trip through a string form.
	d := Duration(90 * time.Second)
	b, _ := json.Marshal(d)
	if string(b) != `"1m30s"` {
		t.Errorf("Marshal(90s) = %s, want \"1m30s\"", b)
	}
}

// baseValid returns a minimal valid config that each invalid-case test mutates
// into one specific kind of broken.
func baseValid() *Config {
	return &Config{
		Providers: []Provider{
			{Name: "p1", Vendor: "mock", BaseURL: "http://127.0.0.1:9001"},
		},
		Routes: map[string]Route{
			"chat": {Targets: []Target{{Provider: "p1", Model: "mock-fast"}}, MaxAttempts: 2},
		},
		Tenants: []Tenant{
			{ID: "t1", APIKeyHash: HashKey("k")},
		},
		Pricing: []pricing.ModelPrice{
			{Model: "mock-fast", InputPerMTok: "0.15", OutputPerMTok: "0.60"},
		},
	}
}

// TestValidateNamesTheOffendingPath is the core promise of the package: every
// validation error points at the exact place in the config that is wrong, so an
// operator does not have to bisect the file by hand.
func TestValidateNamesTheOffendingPath(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*Config)
		wantSub string // a substring the error must contain
	}{
		{
			name:    "route points at an undefined provider",
			mutate:  func(c *Config) { c.Routes["chat"] = Route{Targets: []Target{{Provider: "ghost", Model: "mock-fast"}}} },
			wantSub: `routes["chat"].targets[0].provider "ghost"`,
		},
		{
			name: "duplicate provider name",
			mutate: func(c *Config) {
				c.Providers = append(c.Providers, Provider{Name: "p1", Vendor: "mock", BaseURL: "x"})
			},
			wantSub: `providers[1].name "p1" is a duplicate`,
		},
		{
			name:    "unknown vendor",
			mutate:  func(c *Config) { c.Providers[0].Vendor = "acme-llm" },
			wantSub: `vendor "acme-llm"`,
		},
		{
			name: "route to an unpriced model",
			mutate: func(c *Config) {
				c.Routes["chat"] = Route{Targets: []Target{{Provider: "p1", Model: "unpriced-model"}}}
			},
			wantSub: `model "unpriced-model" is not priced`,
		},
		{
			name:    "tenant with a budget but no period",
			mutate:  func(c *Config) { c.Tenants[0].BudgetLimit = "10.00" },
			wantSub: "budget_limit but no budget_period",
		},
		{
			name:    "tenant allowlist references an undefined route",
			mutate:  func(c *Config) { c.Tenants[0].AllowedModels = []string{"nonexistent"} },
			wantSub: `allowed_models references "nonexistent"`,
		},
		{
			name:    "duplicate tenant id",
			mutate:  func(c *Config) { c.Tenants = append(c.Tenants, Tenant{ID: "t1", APIKeyHash: HashKey("x")}) },
			wantSub: `tenants[1].id "t1" is a duplicate`,
		},
		{
			name:    "malformed api key hash",
			mutate:  func(c *Config) { c.Tenants[0].APIKeyHash = "not-a-hash" },
			wantSub: "api_key_hash must be a 64-char hex",
		},
		{
			name:    "no routes at all",
			mutate:  func(c *Config) { c.Routes = nil },
			wantSub: "at least one route is required",
		},
		{
			name:    "route with no targets",
			mutate:  func(c *Config) { c.Routes["chat"] = Route{} },
			wantSub: `routes["chat"] has no targets`,
		},
		{
			name:    "soft threshold out of range",
			mutate:  func(c *Config) { c.Tenants[0].SoftThresholdPct = 150 },
			wantSub: "soft_threshold_pct 150 is not in [0,100]",
		},
		{
			name:    "breaker failure ratio out of range",
			mutate:  func(c *Config) { c.Providers[0].Breaker.FailureRatio = 1.5 },
			wantSub: "failure_ratio 1.5 is not in [0,1]",
		},
		{
			name:    "invalid budget period string",
			mutate:  func(c *Config) { c.Tenants[0].BudgetLimit = "10.00"; c.Tenants[0].BudgetPeriod = "fortnight" },
			wantSub: `budget_period "fortnight"`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			c := baseValid()
			tc.mutate(c)
			err := c.Validate()
			if err == nil {
				t.Fatalf("expected a validation error mentioning %q, got nil", tc.wantSub)
			}
			if !strings.Contains(err.Error(), tc.wantSub) {
				t.Errorf("error does not name the offending path.\n  want substring: %q\n  got: %v", tc.wantSub, err)
			}
		})
	}
}

func TestValidBaseConfigPasses(t *testing.T) {
	if err := baseValid().Validate(); err != nil {
		t.Fatalf("the base valid config should validate, got: %v", err)
	}
}

func TestUnknownFieldRejected(t *testing.T) {
	// A typo'd key must be a startup error, not a silently-ignored setting.
	bad := `{"server":{"listen":"x","timout":"30s"},"routes":{"c":{"targets":[{"provider":"p","model":"m"}]}}}`
	_, err := Load(strings.NewReader(bad))
	if err == nil || !strings.Contains(err.Error(), "timout") {
		t.Errorf("expected an unknown-field error mentioning the typo, got: %v", err)
	}
}

func TestTenantLimit(t *testing.T) {
	// Unlimited when no budget is set.
	unlimited := Tenant{ID: "t", APIKeyHash: HashKey("k")}
	l, err := unlimited.Limit()
	if err != nil {
		t.Fatal(err)
	}
	if !l.Unlimited() {
		t.Errorf("a tenant with no budget should be unlimited, got %+v", l)
	}

	limited := Tenant{ID: "t", APIKeyHash: HashKey("k"), BudgetLimit: "50.00", BudgetPeriod: "day", SoftThresholdPct: 80}
	l2, err := limited.Limit()
	if err != nil {
		t.Fatal(err)
	}
	if l2.Unlimited() {
		t.Error("a tenant with a budget should not be unlimited")
	}
	if l2.SoftBasisPoints != 8000 {
		t.Errorf("SoftBasisPoints = %d, want 8000 (80%%)", l2.SoftBasisPoints)
	}
	if err := l2.Validate(); err != nil {
		t.Errorf("the derived limit does not validate: %v", err)
	}
}

func TestResolveAPIKey(t *testing.T) {
	// A mock provider needs no key.
	mock := Provider{Name: "m", Vendor: "mock"}
	if _, err := mock.ResolveAPIKey(); err != nil {
		t.Errorf("mock provider should not require a key: %v", err)
	}

	// A real provider with an unset env var must fail loudly.
	real := Provider{Name: "o", Vendor: "openai", APIKeyEnv: "LLMGW_TEST_KEY_DEFINITELY_UNSET"}
	if _, err := real.ResolveAPIKey(); err == nil {
		t.Error("expected an error for an unset API key env var")
	}

	// And succeed when set.
	t.Setenv("LLMGW_TEST_KEY", "secret")
	real.APIKeyEnv = "LLMGW_TEST_KEY"
	got, err := real.ResolveAPIKey()
	if err != nil || got != "secret" {
		t.Errorf("ResolveAPIKey = %q, %v; want \"secret\", nil", got, err)
	}
}

func TestHashKeyStableAnd64Hex(t *testing.T) {
	h := HashKey("bench-key")
	if len(h) != 64 || !isHex(h) {
		t.Errorf("HashKey produced %q, want 64 hex chars", h)
	}
	if HashKey("bench-key") != h {
		t.Error("HashKey is not deterministic")
	}
	if HashKey("other") == h {
		t.Error("HashKey collided on different inputs")
	}
}
