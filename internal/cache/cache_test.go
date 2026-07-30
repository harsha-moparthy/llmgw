package cache

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
	"github.com/harsha-moparthy/llmgw/internal/money"
)

func f64(v float64) *float64 { return &v }
func iptr(v int) *int        { return &v }

// detReq builds a deterministic (cacheable) request.
func detReq(text string) *apiv1.ChatRequest {
	return &apiv1.ChatRequest{
		Model:       "gw-chat",
		Messages:    []apiv1.Message{{Role: apiv1.RoleUser, Content: apiv1.NewTextContent(text)}},
		Temperature: f64(0),
	}
}

func resp(id, text string, finish string) *apiv1.ChatResponse {
	return &apiv1.ChatResponse{
		ID:      id,
		Object:  apiv1.ObjectChatCompletion,
		Created: 1700000000,
		Model:   "mock-fast",
		Choices: []apiv1.Choice{{
			Index:        0,
			Message:      &apiv1.Message{Role: apiv1.RoleAssistant, Content: apiv1.NewTextContent(text)},
			FinishReason: &finish,
		}},
	}
}

func entry(id, text, finish string) *Entry {
	return &Entry{
		Response:    resp(id, text, finish),
		Usage:       &apiv1.Usage{PromptTokens: 10, CompletionTokens: 20, TotalTokens: 30},
		CostAvoided: money.Pico(1500),
		Provider:    "mock-primary",
		Model:       "mock-fast",
	}
}

func TestGetPutHitAndMiss(t *testing.T) {
	s := New(1<<20, Policy{})
	req := detReq("hello")
	k := ComputeKey(req, TenantScope("acme"))

	if _, ok := s.Get(k); ok {
		t.Fatal("empty cache returned a hit")
	}
	if err := s.Put(k, entry("c1", "hi there", apiv1.FinishStop), req); err != nil {
		t.Fatalf("Put: %v", err)
	}
	got, ok := s.Get(k)
	if !ok {
		t.Fatal("expected a hit after Put")
	}
	if got.Response.ID != "c1" {
		t.Errorf("got response %q, want c1", got.Response.ID)
	}
	st := s.Stats()
	if st.Hits != 1 || st.Misses != 1 || st.Stores != 1 {
		t.Errorf("stats = %+v; want 1 hit, 1 miss, 1 store", st)
	}
	if st.CostSaved != money.Pico(1500) {
		t.Errorf("CostSaved = %d, want 1500", st.CostSaved)
	}
}

// TestTenantIsolation is the security property of this package. Two tenants
// asking a byte-identical question must not see each other's responses.
func TestTenantIsolation(t *testing.T) {
	s := New(1<<20, Policy{})
	req := detReq("what is our Q3 revenue")

	acme := ComputeKey(req, TenantScope("acme"))
	globex := ComputeKey(req, TenantScope("globex"))

	if acme.Digest == globex.Digest {
		t.Fatal("identical prompts from different tenants produced the same digest: cross-tenant leak")
	}
	if err := s.Put(acme, entry("acme-1", "42 million", apiv1.FinishStop), req); err != nil {
		t.Fatal(err)
	}
	if _, ok := s.Get(globex); ok {
		t.Fatal("tenant globex read tenant acme's cached response")
	}
	if _, ok := s.Get(acme); !ok {
		t.Fatal("tenant acme could not read its own entry")
	}
}

// TestSharedPoolHitsAcrossTenants verifies the opt-in path works — an isolation
// default that cannot be overridden would just be a broken cache.
func TestSharedPoolHitsAcrossTenants(t *testing.T) {
	s := New(1<<20, Policy{})
	req := detReq("what is the capital of France")
	pool := SharedPoolScope("public-facts")

	k1 := ComputeKey(req, pool)
	k2 := ComputeKey(req, pool)
	if k1.Digest != k2.Digest {
		t.Fatal("same pool + same request produced different digests")
	}
	if err := s.Put(k1, entry("p1", "Paris", apiv1.FinishStop), req); err != nil {
		t.Fatal(err)
	}
	if _, ok := s.Get(k2); !ok {
		t.Fatal("shared pool did not produce a hit")
	}
	// A pool and a tenant with the same name must still not collide.
	if ComputeKey(req, SharedPoolScope("acme")).String() == ComputeKey(req, TenantScope("acme")).String() {
		t.Error("pool 'acme' and tenant 'acme' share a storage key")
	}
}

func TestCacheablePolicy(t *testing.T) {
	tests := []struct {
		name     string
		req      *apiv1.ChatRequest
		policy   Policy
		wantOK   bool
		wantWord string
	}{
		{
			name:   "temperature 0 is cacheable",
			req:    detReq("x"),
			wantOK: true,
		},
		{
			name: "temperature 0.7 is not cacheable by default",
			req: &apiv1.ChatRequest{
				Model: "m", Messages: []apiv1.Message{{Role: "user"}}, Temperature: f64(0.7),
			},
			wantOK:   false,
			wantWord: "non-deterministic",
		},
		{
			// Unset temperature means the provider's sampling default, which is
			// not deterministic. Treating absent as 0 would cache most real
			// traffic in violation of the policy.
			name:     "unset temperature is not cacheable by default",
			req:      &apiv1.ChatRequest{Model: "m", Messages: []apiv1.Message{{Role: "user"}}},
			wantOK:   false,
			wantWord: "unset",
		},
		{
			name: "top_p below 1 is not cacheable even at temperature 0",
			req: &apiv1.ChatRequest{
				Model: "m", Messages: []apiv1.Message{{Role: "user"}},
				Temperature: f64(0), TopP: f64(0.5),
			},
			wantOK:   false,
			wantWord: "top_p",
		},
		{
			name: "operator opt-in allows non-deterministic",
			req: &apiv1.ChatRequest{
				Model: "m", Messages: []apiv1.Message{{Role: "user"}}, Temperature: f64(0.9),
			},
			policy: Policy{CacheNonDeterministic: true},
			wantOK: true,
		},
		{
			name: "n > 1 is not cached",
			req: &apiv1.ChatRequest{
				Model: "m", Messages: []apiv1.Message{{Role: "user"}},
				Temperature: f64(0), N: iptr(2),
			},
			wantOK:   false,
			wantWord: "n > 1",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := New(1<<20, tc.policy)
			ok, reason := s.Cacheable(tc.req)
			if ok != tc.wantOK {
				t.Fatalf("Cacheable = %v (%q), want %v", ok, reason, tc.wantOK)
			}
			if !ok && tc.wantWord != "" && !strings.Contains(reason, tc.wantWord) {
				t.Errorf("reason %q does not mention %q", reason, tc.wantWord)
			}
		})
	}
}

// TestPutEnforcesPolicyIndependently checks that Put refuses a forbidden entry
// even if the caller skipped Cacheable. A policy enforced in one place only is a
// policy that gets bypassed.
func TestPutEnforcesPolicyIndependently(t *testing.T) {
	s := New(1<<20, Policy{})
	hot := &apiv1.ChatRequest{
		Model: "m", Messages: []apiv1.Message{{Role: "user"}}, Temperature: f64(1.0),
	}
	k := ComputeKey(hot, TenantScope("t"))
	err := s.Put(k, entry("c", "text", apiv1.FinishStop), hot)
	if err == nil {
		t.Fatal("Put stored a non-deterministic response despite the policy")
	}
	if s.Stats().Rejected != 1 {
		t.Errorf("Rejected = %d, want 1", s.Stats().Rejected)
	}
}

// TestTruncatedResponseNotCached protects against the worst cache bug: a
// response cut off at the token cap, cached, and then served forever.
func TestTruncatedResponseNotCached(t *testing.T) {
	req := detReq("write me an essay")

	s := New(1<<20, Policy{})
	k := ComputeKey(req, TenantScope("t"))
	if err := s.Put(k, entry("c", "half an essa", apiv1.FinishLength), req); err == nil {
		t.Fatal("a truncated response was cached under the default policy")
	}
	if _, ok := s.Get(k); ok {
		t.Fatal("truncated response is retrievable")
	}

	// With the opt-in it is allowed, so the default is a policy rather than an
	// inability.
	s2 := New(1<<20, Policy{CacheTruncated: true})
	if err := s2.Put(k, entry("c", "half an essa", apiv1.FinishLength), req); err != nil {
		t.Errorf("with CacheTruncated the store should accept it: %v", err)
	}
}

func TestTTLExpiry(t *testing.T) {
	now := time.Unix(1700000000, 0)
	s := New(1<<20, Policy{TTL: time.Minute})
	s.SetClock(func() time.Time { return now })

	req := detReq("x")
	k := ComputeKey(req, TenantScope("t"))
	if err := s.Put(k, entry("c", "v", apiv1.FinishStop), req); err != nil {
		t.Fatal(err)
	}
	if _, ok := s.Get(k); !ok {
		t.Fatal("fresh entry missing")
	}

	now = now.Add(59 * time.Second)
	if _, ok := s.Get(k); !ok {
		t.Fatal("entry expired one second early")
	}

	now = now.Add(2 * time.Second) // past the TTL
	if _, ok := s.Get(k); ok {
		t.Fatal("expired entry was returned")
	}
	if st := s.Stats(); st.Expirations != 1 {
		t.Errorf("Expirations = %d, want 1", st.Expirations)
	}
	if st := s.Stats(); st.Entries != 0 {
		t.Errorf("expired entry was not removed: Entries = %d", st.Entries)
	}
}

// TestByteBoundEviction verifies the bound is on bytes, not entries: entries of
// wildly different sizes must be evicted so that total bytes stays under the
// limit.
func TestByteBoundEviction(t *testing.T) {
	// Small bound so a handful of entries crosses it.
	s := New(8<<10, Policy{})
	req := detReq("x")

	// Each entry is ~512 overhead + response JSON. Store enough to force
	// eviction, and check the invariant rather than a specific eviction count
	// (which would be an assertion about the overhead constant, not behaviour).
	for i := 0; i < 60; i++ {
		k := ComputeKey(detReq(fmt.Sprintf("prompt-%d", i)), TenantScope("t"))
		e := entry(fmt.Sprintf("c%d", i), strings.Repeat("a", 200), apiv1.FinishStop)
		if err := s.Put(k, e, req); err != nil {
			t.Fatalf("Put %d: %v", i, err)
		}
		if got := s.Stats().Bytes; got > 8<<10 {
			t.Fatalf("after %d puts, bytes = %d exceeds the 8192 bound", i, got)
		}
	}
	st := s.Stats()
	if st.Evictions == 0 {
		t.Error("no evictions occurred despite exceeding the bound")
	}
	if st.Entries == 0 {
		t.Error("everything was evicted")
	}
	t.Logf("after 60 puts: %d entries, %d bytes, %d evictions", st.Entries, st.Bytes, st.Evictions)
}

// TestLRUOrderEvictsLeastRecentlyUsed checks the replacement policy is actually
// LRU: a repeatedly-read entry must survive while cold ones are evicted.
func TestLRUOrderEvictsLeastRecentlyUsed(t *testing.T) {
	s := New(4<<10, Policy{})
	req := detReq("x")

	hotKey := ComputeKey(detReq("hot"), TenantScope("t"))
	if err := s.Put(hotKey, entry("hot", strings.Repeat("h", 100), apiv1.FinishStop), req); err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 40; i++ {
		// Touch the hot entry before each insert so it stays at the front.
		if _, ok := s.Get(hotKey); !ok {
			t.Fatalf("hot entry evicted at iteration %d despite being read every round", i)
		}
		k := ComputeKey(detReq(fmt.Sprintf("cold-%d", i)), TenantScope("t"))
		if err := s.Put(k, entry(fmt.Sprintf("c%d", i), strings.Repeat("c", 100), apiv1.FinishStop), req); err != nil {
			t.Fatal(err)
		}
	}
	if _, ok := s.Get(hotKey); !ok {
		t.Error("the most-recently-used entry was evicted: replacement is not LRU")
	}
}

func TestPutReplaceAdjustsByteTotal(t *testing.T) {
	s := New(1<<20, Policy{})
	req := detReq("x")
	k := ComputeKey(req, TenantScope("t"))

	if err := s.Put(k, entry("c1", strings.Repeat("a", 100), apiv1.FinishStop), req); err != nil {
		t.Fatal(err)
	}
	first := s.Stats()

	// Replacing the same key must not double-count bytes or add an entry.
	if err := s.Put(k, entry("c2", strings.Repeat("a", 100), apiv1.FinishStop), req); err != nil {
		t.Fatal(err)
	}
	second := s.Stats()
	if second.Entries != 1 {
		t.Errorf("Entries = %d after replacing one key, want 1", second.Entries)
	}
	if second.Bytes != first.Bytes {
		t.Errorf("Bytes = %d after replacing with an equal-size entry, want %d", second.Bytes, first.Bytes)
	}
	got, _ := s.Get(k)
	if got.Response.ID != "c2" {
		t.Errorf("replacement did not take effect: got %q", got.Response.ID)
	}
}

func TestOversizedEntryRejected(t *testing.T) {
	s := New(1<<20, Policy{MaxEntryBytes: 1024})
	req := detReq("x")
	k := ComputeKey(req, TenantScope("t"))
	err := s.Put(k, entry("big", strings.Repeat("x", 4096), apiv1.FinishStop), req)
	if !errors.Is(err, ErrTooLarge) {
		t.Fatalf("err = %v, want ErrTooLarge", err)
	}
	if s.Stats().Entries != 0 {
		t.Error("oversized entry was stored anyway")
	}
}

func TestSweep(t *testing.T) {
	now := time.Unix(1700000000, 0)
	s := New(1<<20, Policy{TTL: time.Minute})
	s.SetClock(func() time.Time { return now })
	req := detReq("x")

	for i := 0; i < 10; i++ {
		k := ComputeKey(detReq(fmt.Sprintf("p%d", i)), TenantScope("t"))
		if err := s.Put(k, entry(fmt.Sprintf("c%d", i), "v", apiv1.FinishStop), req); err != nil {
			t.Fatal(err)
		}
	}
	if n := s.Sweep(100); n != 0 {
		t.Errorf("Sweep removed %d fresh entries, want 0", n)
	}
	now = now.Add(2 * time.Minute)
	if n := s.Sweep(100); n != 10 {
		t.Errorf("Sweep removed %d expired entries, want 10", n)
	}
	if s.Stats().Entries != 0 {
		t.Errorf("Entries = %d after sweeping everything", s.Stats().Entries)
	}
}

// TestSweeperGoroutineExits asserts StartSweeper's stop function actually joins
// the goroutine. A sweeper that outlives its store is a leak that only shows up
// as slowly growing goroutine counts in production.
func TestSweeperGoroutineExits(t *testing.T) {
	s := New(1<<20, Policy{TTL: time.Millisecond})
	stop := s.StartSweeper(time.Millisecond, 10)
	done := make(chan struct{})
	go func() {
		stop()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("stop() did not return: the sweeper goroutine is leaked")
	}
	// Calling stop twice must be safe.
	stop()
}

// TestConcurrentAccess hammers the store from many goroutines. Run under -race
// this is what proves the mutex discipline is right; the byte-total invariant is
// checked at the end because a lost update there manifests as a cache that
// slowly exceeds its memory bound.
func TestConcurrentAccess(t *testing.T) {
	const maxBytes = 32 << 10
	s := New(maxBytes, Policy{})
	req := detReq("x")

	var wg sync.WaitGroup
	for w := 0; w < 16; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				k := ComputeKey(detReq(fmt.Sprintf("w%d-i%d", w, i%50)), TenantScope("t"))
				switch i % 3 {
				case 0:
					_ = s.Put(k, entry(fmt.Sprintf("c%d-%d", w, i), strings.Repeat("x", 120), apiv1.FinishStop), req)
				case 1:
					s.Get(k)
				default:
					s.Sweep(10)
				}
			}
		}(w)
	}
	wg.Wait()

	st := s.Stats()
	if st.Bytes > maxBytes {
		t.Errorf("Bytes = %d exceeds the bound %d after concurrent access", st.Bytes, maxBytes)
	}
	if st.Bytes < 0 {
		t.Errorf("Bytes went negative: %d", st.Bytes)
	}
	t.Logf("concurrent result: %+v", st)
}

func TestReplayFrameShape(t *testing.T) {
	e := entry("c1", "Hello there, this is a longer cached completion body.", apiv1.FinishStop)
	e.StoredAt = time.Now()
	chunks := e.Replay(8, true)

	if len(chunks) < 4 {
		t.Fatalf("got %d chunks, want at least 4", len(chunks))
	}
	// First frame carries the role, as OpenAI's opening chunk does.
	if d := chunks[0].Choices[0].Delta; d == nil || d.Role != apiv1.RoleAssistant {
		t.Errorf("first chunk does not carry the assistant role: %+v", chunks[0].Choices[0].Delta)
	}
	// Reassembled content must equal the original exactly.
	var sb strings.Builder
	for _, c := range chunks {
		sb.WriteString(c.DeltaText())
	}
	want := "Hello there, this is a longer cached completion body."
	if sb.String() != want {
		t.Errorf("replayed text = %q, want %q", sb.String(), want)
	}
	// A finish_reason must appear exactly once.
	finishes := 0
	for _, c := range chunks {
		if len(c.Choices) > 0 && c.Choices[0].FinishReason != nil {
			finishes++
		}
	}
	if finishes != 1 {
		t.Errorf("finish_reason appeared %d times, want 1", finishes)
	}
	// Usage frame present when requested.
	last := chunks[len(chunks)-1]
	if last.Usage == nil {
		t.Error("include_usage requested but no usage frame was emitted")
	}
	// And absent when not.
	if got := e.Replay(8, false); got[len(got)-1].Usage != nil {
		t.Error("usage frame emitted despite include_usage=false")
	}
	if chunks[0].Object != apiv1.ObjectChatCompletionChunk {
		t.Errorf("object = %q, want %q", chunks[0].Object, apiv1.ObjectChatCompletionChunk)
	}
}

// TestReplayNeverSplitsRunes guards against emitting invalid UTF-8, which would
// render as replacement characters in the client.
func TestReplayNeverSplitsRunes(t *testing.T) {
	text := "héllo → 世界 🎉 emoji and accents everywhere ñ"
	e := entry("c1", text, apiv1.FinishStop)
	for _, size := range []int{1, 2, 3, 5, 7} {
		chunks := e.Replay(size, false)
		var sb strings.Builder
		for _, c := range chunks {
			frag := c.DeltaText()
			if !utf8Valid(frag) {
				t.Errorf("chunkRunes=%d produced invalid UTF-8 fragment %q", size, frag)
			}
			sb.WriteString(frag)
		}
		if sb.String() != text {
			t.Errorf("chunkRunes=%d reassembled to %q, want %q", size, sb.String(), text)
		}
	}
}

func utf8Valid(s string) bool {
	for _, r := range s {
		if r == '�' {
			return false
		}
	}
	return true
}

func TestSummarizeForHeaderCarriesNoContent(t *testing.T) {
	secret := "the client's confidential prompt text"
	e := entry("c1", secret, apiv1.FinishStop)
	e.StoredAt = time.Now()
	got := e.SummarizeForHeader()
	if strings.Contains(got, secret) {
		t.Errorf("header summary leaked response content: %q", got)
	}
	if !strings.Contains(got, "mock-primary") {
		t.Errorf("header summary %q does not name the provider", got)
	}
}

func TestPurge(t *testing.T) {
	s := New(1<<20, Policy{})
	req := detReq("x")
	for i := 0; i < 5; i++ {
		k := ComputeKey(detReq(fmt.Sprintf("p%d", i)), TenantScope("t"))
		if err := s.Put(k, entry("c", "v", apiv1.FinishStop), req); err != nil {
			t.Fatal(err)
		}
	}
	s.Purge()
	st := s.Stats()
	if st.Entries != 0 || st.Bytes != 0 {
		t.Errorf("after Purge: entries=%d bytes=%d, want 0/0", st.Entries, st.Bytes)
	}
}

func TestHitRate(t *testing.T) {
	var s Stats
	if s.HitRate() != 0 {
		t.Error("empty stats should report a 0 hit rate, not NaN")
	}
	s.Hits, s.Misses = 3, 1
	if got := s.HitRate(); got != 0.75 {
		t.Errorf("HitRate = %v, want 0.75", got)
	}
}
