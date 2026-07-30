// Package cache is the gateway's response cache.
//
// Three decisions in here are worth more than the code that implements them.
//
// # Tenant isolation is the default, not an option
//
// Cache keys are namespaced per tenant unless an operator explicitly names a
// shared pool. A gateway whose cache defaults to shared serves one customer's
// completion as another customer's cache hit; that is a data-isolation breach,
// and it is invisible in testing because a hit is indistinguishable from a
// correct response. See Scope in key.go — the safe scope is the zero value.
//
// # Only deterministic requests are cached by default
//
// A request with temperature > 0 asked for sampled output. Serving the same
// bytes to every such request silently changes the semantics the client
// requested: an agent that retries a generation hoping for a different answer
// would loop forever on the cached one. So the default policy caches only
// requests that are deterministic by construction, and an operator who prefers
// the cost saving must opt in and accept that trade knowingly.
//
// # The bound is bytes, not entries
//
// Entries here range from a 200-byte refusal to a 200 KB long-form completion,
// three orders of magnitude apart. An entry-count bound cannot protect memory
// against that spread — 10,000 entries is either 2 MB or 2 GB — so the LRU is
// bounded by total serialised bytes.
package cache

import (
	"container/list"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
	"github.com/harsha-moparthy/llmgw/internal/money"
)

// Policy controls what may be cached.
type Policy struct {
	// CacheNonDeterministic allows caching requests with temperature > 0.
	//
	// Off by default. Turning it on trades the client's requested sampling
	// semantics for a cost saving; it is a legitimate choice for a
	// high-volume, low-variance workload and a wrong one for an agent that
	// depends on resampling.
	CacheNonDeterministic bool

	// CacheTruncated allows caching responses that stopped at the token cap
	// (finish_reason "length").
	//
	// Off by default, and this default is load-bearing. A truncated answer
	// cached with a long TTL is a permanently wrong response served from memory:
	// the client cannot retry its way out of it, and it looks like a model
	// defect rather than a cache defect. Correctness beats hit rate here.
	CacheTruncated bool

	// TTL is how long an entry stays valid. Zero means DefaultTTL.
	TTL time.Duration

	// MaxEntryBytes caps a single entry, so one enormous response cannot evict
	// the entire working set.
	MaxEntryBytes int
}

// Defaults for the policy and store.
const (
	DefaultTTL           = 10 * time.Minute
	DefaultMaxBytes      = 256 << 20 // 256 MiB
	DefaultMaxEntryBytes = 1 << 20   // 1 MiB
)

// ErrTooLarge is returned by Put when an entry exceeds MaxEntryBytes.
var ErrTooLarge = errors.New("cache: entry exceeds the per-entry size limit")

// Entry is a cached response.
type Entry struct {
	// Response is the complete non-streaming response body.
	Response *apiv1.ChatResponse
	// Usage is the token usage the original upstream call reported.
	//
	// Kept so a hit can still be attributed to the tenant's usage record: a
	// cache hit costs nothing upstream, but the tenant did consume the value of
	// those tokens and a usage report that shows zero for cached traffic hides
	// most of what the tenant actually did.
	Usage *apiv1.Usage
	// CostAvoided is what the upstream call would have cost, used to report the
	// cache's savings as a measured figure rather than an estimate.
	CostAvoided money.Pico
	// Provider and Model record which upstream produced this, for the response
	// headers on a hit.
	Provider string
	Model    string
	// StoredAt and ExpiresAt bound validity.
	StoredAt  time.Time
	ExpiresAt time.Time
	// bytes is the serialised size, used for the byte bound.
	bytes int
}

// Stats is a snapshot of cache behaviour.
type Stats struct {
	Hits        int64
	Misses      int64
	Stores      int64
	Evictions   int64
	Expirations int64
	// Rejected counts entries refused by policy (non-deterministic, truncated,
	// oversized). Reported separately from Misses because "we chose not to
	// cache this" and "we looked and it was not there" are different facts, and
	// a hit rate that conflates them cannot be reasoned about.
	Rejected  int64
	Bytes     int
	Entries   int
	CostSaved money.Pico
}

// HitRate returns hits / (hits + misses), or 0 with no lookups.
func (s Stats) HitRate() float64 {
	total := s.Hits + s.Misses
	if total == 0 {
		return 0
	}
	return float64(s.Hits) / float64(total)
}

// Store is a byte-bounded LRU cache with per-entry TTL.
//
// Safe for concurrent use. A single mutex guards both the map and the LRU list:
// they must move together, and a fancier scheme (sharded locks, lock-free list)
// would buy throughput the gateway does not need — the critical section is a map
// lookup and two pointer writes — at the cost of the correctness that matters
// most here.
type Store struct {
	mu      sync.Mutex
	entries map[string]*list.Element
	lru     *list.List // front = most recently used
	bytes   int

	maxBytes int
	policy   Policy
	now      func() time.Time

	stats Stats
}

type lruItem struct {
	key   string
	entry *Entry
}

// New returns a Store with the given byte bound and policy.
func New(maxBytes int, policy Policy) *Store {
	if maxBytes <= 0 {
		maxBytes = DefaultMaxBytes
	}
	if policy.TTL <= 0 {
		policy.TTL = DefaultTTL
	}
	if policy.MaxEntryBytes <= 0 {
		policy.MaxEntryBytes = DefaultMaxEntryBytes
	}
	return &Store{
		entries:  make(map[string]*list.Element),
		lru:      list.New(),
		maxBytes: maxBytes,
		policy:   policy,
		now:      time.Now,
	}
}

// SetClock overrides the time source, for deterministic TTL tests. Injecting the
// clock rather than sleeping is what makes the expiry tests exact and instant.
func (s *Store) SetClock(now func() time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.now = now
}

// Policy returns the store's policy.
func (s *Store) Policy() Policy { return s.policy }

// Cacheable reports whether a request may be cached under the store's policy,
// and why not when it may not.
//
// The reason string is returned so the server can put it in a response header
// during development: "why did this not cache" is otherwise one of the more
// annoying things to debug in a gateway.
func (s *Store) Cacheable(req *apiv1.ChatRequest) (bool, string) {
	if !s.policy.CacheNonDeterministic {
		// An unset temperature means the provider's default, which for every
		// current provider is non-zero sampling. Treating unset as deterministic
		// would cache the majority of real traffic in violation of the policy,
		// so absent is treated as non-deterministic.
		if req.Temperature == nil {
			return false, "temperature unset (provider default is non-deterministic)"
		}
		if *req.Temperature != 0 {
			return false, fmt.Sprintf("temperature %g is non-deterministic", *req.Temperature)
		}
		// top_p < 1 also samples, so temperature 0 alone is not sufficient.
		if req.TopP != nil && *req.TopP != 1 {
			return false, fmt.Sprintf("top_p %g is non-deterministic", *req.TopP)
		}
	}
	if req.N != nil && *req.N > 1 {
		return false, "n > 1 is not cached"
	}
	return true, ""
}

// Get looks up an entry.
func (s *Store) Get(k Key) (*Entry, bool) {
	ks := k.String()
	s.mu.Lock()
	defer s.mu.Unlock()

	el, ok := s.entries[ks]
	if !ok {
		s.stats.Misses++
		return nil, false
	}
	it := el.Value.(*lruItem)
	if !it.entry.ExpiresAt.After(s.now()) {
		// Lazy expiry on read. A goroutine or timer per entry would be far more
		// expensive than checking on the path that already holds the lock.
		s.removeElement(el)
		s.stats.Expirations++
		s.stats.Misses++
		return nil, false
	}
	s.lru.MoveToFront(el)
	s.stats.Hits++
	s.stats.CostSaved += it.entry.CostAvoided
	return it.entry, true
}

// Put stores a response.
//
// The policy checks live here as well as in Cacheable so that a caller who
// forgets to consult Cacheable cannot store something the policy forbids. A
// policy enforced in only one place is a policy that will eventually be
// bypassed.
func (s *Store) Put(k Key, e *Entry, req *apiv1.ChatRequest) error {
	if ok, reason := s.Cacheable(req); !ok {
		s.mu.Lock()
		s.stats.Rejected++
		s.mu.Unlock()
		return fmt.Errorf("cache: not cacheable: %s", reason)
	}
	if !s.policy.CacheTruncated && finishedByLength(e.Response) {
		s.mu.Lock()
		s.stats.Rejected++
		s.mu.Unlock()
		return errors.New("cache: response was truncated at the token cap (finish_reason=length)")
	}

	size, err := entrySize(e)
	if err != nil {
		return err
	}
	if size > s.policy.MaxEntryBytes {
		s.mu.Lock()
		s.stats.Rejected++
		s.mu.Unlock()
		return fmt.Errorf("%w (%d bytes, limit %d)", ErrTooLarge, size, s.policy.MaxEntryBytes)
	}
	e.bytes = size

	s.mu.Lock()
	defer s.mu.Unlock()

	now := s.now()
	e.StoredAt = now
	if e.ExpiresAt.IsZero() {
		e.ExpiresAt = now.Add(s.policy.TTL)
	}

	ks := k.String()
	if el, ok := s.entries[ks]; ok {
		// Replacing an existing entry: adjust the byte total by the delta rather
		// than adding twice.
		old := el.Value.(*lruItem)
		s.bytes -= old.entry.bytes
		old.entry = e
		s.bytes += size
		s.lru.MoveToFront(el)
	} else {
		el := s.lru.PushFront(&lruItem{key: ks, entry: e})
		s.entries[ks] = el
		s.bytes += size
	}
	s.stats.Stores++
	s.evictLocked()
	return nil
}

// evictLocked drops least-recently-used entries until the byte bound holds.
func (s *Store) evictLocked() {
	for s.bytes > s.maxBytes {
		el := s.lru.Back()
		if el == nil {
			// Should be unreachable: bytes cannot exceed the bound with no
			// entries. Guarded anyway rather than spinning forever if a future
			// accounting bug makes s.bytes drift.
			s.bytes = 0
			return
		}
		s.removeElement(el)
		s.stats.Evictions++
	}
}

func (s *Store) removeElement(el *list.Element) {
	it := el.Value.(*lruItem)
	s.lru.Remove(el)
	delete(s.entries, it.key)
	s.bytes -= it.entry.bytes
	if s.bytes < 0 {
		s.bytes = 0
	}
}

// Sweep removes expired entries, bounded to at most limit examinations so a
// large cache cannot hold the lock for an unbounded period.
//
// A single bounded sweeper plus lazy expiry on read is the whole expiry
// strategy. The alternative — a timer per entry — costs a goroutine per cached
// response, which for a gateway at any real volume is thousands of goroutines
// doing nothing.
func (s *Store) Sweep(limit int) int {
	if limit <= 0 {
		limit = 1000
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.now()
	removed := 0
	// Walk from the back: least-recently-used entries are the likeliest to have
	// expired, so the bounded walk finds the most garbage per unit of work.
	el := s.lru.Back()
	for i := 0; i < limit && el != nil; i++ {
		prev := el.Prev()
		if !el.Value.(*lruItem).entry.ExpiresAt.After(now) {
			s.removeElement(el)
			s.stats.Expirations++
			removed++
		}
		el = prev
	}
	return removed
}

// StartSweeper runs Sweep on an interval until stop is closed. Returns a
// function that stops it and waits for the goroutine to exit, so a test can
// assert no goroutine is leaked.
func (s *Store) StartSweeper(interval time.Duration, limit int) (stop func()) {
	if interval <= 0 {
		interval = time.Minute
	}
	done := make(chan struct{})
	exited := make(chan struct{})
	go func() {
		defer close(exited)
		t := time.NewTicker(interval)
		defer t.Stop()
		for {
			select {
			case <-done:
				return
			case <-t.C:
				s.Sweep(limit)
			}
		}
	}()
	var once sync.Once
	return func() {
		once.Do(func() { close(done) })
		<-exited
	}
}

// Stats returns a snapshot.
func (s *Store) Stats() Stats {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := s.stats
	out.Bytes = s.bytes
	out.Entries = len(s.entries)
	return out
}

// Purge removes everything. Used by tests and by an operator endpoint.
func (s *Store) Purge() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.entries = make(map[string]*list.Element)
	s.lru.Init()
	s.bytes = 0
}

// entrySize returns the serialised size of an entry, which is what the byte
// bound is expressed in.
//
// Marshalling to measure is not free, but it happens once per store (not per
// read) and it is the only figure that actually corresponds to memory held. The
// alternative — summing string lengths — undercounts by whatever the Go runtime
// adds in headers and slack, which is exactly the drift that turns a "256 MB"
// cache into an OOM.
func entrySize(e *Entry) (int, error) {
	if e == nil || e.Response == nil {
		return 0, errors.New("cache: entry has no response")
	}
	b, err := json.Marshal(e.Response)
	if err != nil {
		return 0, fmt.Errorf("cache: sizing entry: %w", err)
	}
	// A fixed allowance for the entry struct, its usage record and the map/list
	// bookkeeping. Approximate by nature, and deliberately an over-estimate:
	// erring high means the cache holds slightly less than its bound, which is
	// the safe direction.
	const overhead = 512
	return len(b) + overhead, nil
}

func finishedByLength(r *apiv1.ChatResponse) bool {
	if r == nil {
		return false
	}
	for _, c := range r.Choices {
		if c.FinishReason != nil && *c.FinishReason == apiv1.FinishLength {
			return true
		}
	}
	return false
}

// Replay converts a cached response into the sequence of streaming chunks a
// streaming client expects.
//
// The chunk boundaries are NOT the original ones. They cannot be: the cache
// stores the assembled text, and the token boundaries the provider used are not
// recoverable from it. This function splits on a fixed rune count, which means a
// client that measured inter-token timing or counted frames would see something
// different from the original stream. That is a real, if minor, fidelity loss
// and it is stated here rather than glossed over — the alternative (storing every
// original frame) triples the entry size to preserve a property no client
// depends on.
//
// The frame shape is faithful: a leading role frame, content frames, a
// finish_reason frame, and an optional usage frame, matching what OpenAI emits.
func (e *Entry) Replay(chunkRunes int, includeUsage bool) []*apiv1.ChatChunk {
	if chunkRunes <= 0 {
		chunkRunes = 24
	}
	if e == nil || e.Response == nil || len(e.Response.Choices) == 0 {
		return nil
	}
	resp := e.Response
	var text string
	if m := resp.Choices[0].Message; m != nil {
		text = m.Content.Text()
	}

	base := func() *apiv1.ChatChunk {
		return &apiv1.ChatChunk{
			ID:                resp.ID,
			Object:            apiv1.ObjectChatCompletionChunk,
			Created:           resp.Created,
			Model:             resp.Model,
			SystemFingerprint: resp.SystemFingerprint,
		}
	}

	out := make([]*apiv1.ChatChunk, 0, len(text)/chunkRunes+3)

	// Opening frame carries the role and empty content, as OpenAI's does.
	first := base()
	first.Choices = []apiv1.Choice{{
		Index: 0,
		Delta: &apiv1.Message{Role: apiv1.RoleAssistant, Content: apiv1.NewTextContent("")},
	}}
	out = append(out, first)

	// Split on rune boundaries, never mid-rune: a chunk cut inside a multi-byte
	// character would emit invalid UTF-8 and render as a replacement character
	// in the client.
	runes := []rune(text)
	for i := 0; i < len(runes); i += chunkRunes {
		end := i + chunkRunes
		if end > len(runes) {
			end = len(runes)
		}
		c := base()
		c.Choices = []apiv1.Choice{{
			Index: 0,
			Delta: &apiv1.Message{Content: apiv1.NewTextContent(string(runes[i:end]))},
		}}
		out = append(out, c)
	}

	// Terminal frame carries the finish reason and an empty delta.
	fin := base()
	reason := apiv1.FinishStop
	if r := resp.Choices[0].FinishReason; r != nil {
		reason = *r
	}
	fin.Choices = []apiv1.Choice{{Index: 0, Delta: &apiv1.Message{}, FinishReason: &reason}}
	out = append(out, fin)

	if includeUsage && e.Usage != nil {
		u := base()
		u.Choices = []apiv1.Choice{}
		u.Usage = e.Usage
		out = append(out, u)
	}
	return out
}

// SummarizeForHeader renders a short, content-free description of a hit for a
// response header. It must never include prompt or completion text.
func (e *Entry) SummarizeForHeader() string {
	if e == nil {
		return ""
	}
	var sb strings.Builder
	sb.WriteString("provider=")
	sb.WriteString(e.Provider)
	sb.WriteString(" model=")
	sb.WriteString(e.Model)
	sb.WriteString(" age=")
	sb.WriteString(time.Since(e.StoredAt).Truncate(time.Millisecond).String())
	return sb.String()
}
