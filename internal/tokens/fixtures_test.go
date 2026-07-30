package tokens

import (
	"encoding/json"
	"testing"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
)

// promptFixture is one realistic prompt shape used by the error-bound test.
//
// The shapes are chosen to cover the cases where a chars-per-token estimator is
// known to be worst, not just the case where it is best:
//
//	prose            — the case it is tuned for.
//	many-short-turns — framing overhead dominates; the case that punishes an
//	                   estimator that drops the per-message constants.
//	code             — dense punctuation and identifiers, fewer chars per token
//	                   than prose in every real tokenizer.
//	json-payload     — the pathological end of the same axis.
//	cjk              — where byte counting over-counts by ~3x.
//	tool-heavy       — schemas larger than the messages.
type promptFixture struct {
	name string
	req  *apiv1.ChatRequest
}

func msg(role, text string) apiv1.Message {
	return apiv1.Message{Role: role, Content: apiv1.NewTextContent(text)}
}

const loremProse = `You are a meticulous technical editor. Rewrite the passage below so that it ` +
	`is clearer and shorter, but do not change any of the technical claims it makes. Preserve the ` +
	`author's voice. If a sentence contains a factual error, leave it in place and add a bracketed ` +
	`note after it rather than silently correcting it, because the author needs to see what you ` +
	`disagreed with. Return only the rewritten passage.`

const userProse = `The gateway sits in front of every model call the company makes, so its ` +
	`dependency tree is its attack surface and its tail latency is its reputation. We measured a ` +
	`p99 of about four milliseconds of added latency, excluding the time the provider itself spends ` +
	`generating, and we think that is the honest number to quote because an end-to-end figure ` +
	`against a mock provider would be dominated by the mock.`

const goCode = `func (r Ratio) Tokens(chars int) int {
	if chars <= 0 {
		return 0
	}
	if r <= 0 {
		r = DefaultRatio
	}
	n := (int64(chars)*100 + int64(r) - 1) / int64(r)
	if n < 1 {
		n = 1
	}
	return int(n)
}`

const jsonPayload = `{"id":"cmpl-7f3a91","object":"chat.completion","created":1735689600,` +
	`"model":"gpt-4o-mini-2024-07-18","choices":[{"index":0,"message":{"role":"assistant",` +
	`"content":"Done."},"finish_reason":"stop"}],"usage":{"prompt_tokens":142,` +
	`"completion_tokens":3,"total_tokens":145}}`

const cjkText = `ゲートウェイは会社が行うすべてのモデル呼び出しの前に位置するため、依存関係ツリーが攻撃対象領域となり、テールレイテンシがその評判となります。`

const toolSchema = `[{"type":"function","function":{"name":"search_orders","description":` +
	`"Search a customer's order history by date range, status, and free-text query.",` +
	`"parameters":{"type":"object","properties":{"customer_id":{"type":"string",` +
	`"description":"Opaque customer identifier."},"from":{"type":"string","format":"date"},` +
	`"to":{"type":"string","format":"date"},"status":{"type":"string","enum":["placed",` +
	`"shipped","delivered","cancelled","refunded"]},"query":{"type":"string"},` +
	`"limit":{"type":"integer","minimum":1,"maximum":100}},"required":["customer_id"]}}}]`

func promptFixtures() []promptFixture {
	return []promptFixture{
		{
			name: "prose-system-plus-user",
			req: &apiv1.ChatRequest{
				Model: "gpt-4o",
				Messages: []apiv1.Message{
					msg(apiv1.RoleSystem, loremProse),
					msg(apiv1.RoleUser, userProse),
				},
			},
		},
		{
			name: "many-short-turns",
			req: &apiv1.ChatRequest{
				Model: "gpt-4o",
				Messages: []apiv1.Message{
					msg(apiv1.RoleSystem, "Be terse."),
					msg(apiv1.RoleUser, "hi"),
					msg(apiv1.RoleAssistant, "Hello."),
					msg(apiv1.RoleUser, "status?"),
					msg(apiv1.RoleAssistant, "All green."),
					msg(apiv1.RoleUser, "thanks"),
					msg(apiv1.RoleAssistant, "Sure."),
					msg(apiv1.RoleUser, "bye"),
				},
			},
		},
		{
			name: "code-review",
			req: &apiv1.ChatRequest{
				Model: "claude-sonnet-4-5",
				Messages: []apiv1.Message{
					msg(apiv1.RoleSystem, "You review Go for correctness bugs only."),
					msg(apiv1.RoleUser, goCode),
				},
			},
		},
		{
			name: "json-payload",
			req: &apiv1.ChatRequest{
				Model: "gpt-4o-mini",
				Messages: []apiv1.Message{
					msg(apiv1.RoleUser, "Extract the total token count from this response:\n"+jsonPayload),
				},
			},
		},
		{
			name: "cjk",
			req: &apiv1.ChatRequest{
				Model: "gemini-2.5-flash",
				Messages: []apiv1.Message{
					msg(apiv1.RoleUser, cjkText),
				},
			},
		},
		{
			name: "tool-heavy-agent",
			req: &apiv1.ChatRequest{
				Model:      "gpt-5",
				Messages:   []apiv1.Message{msg(apiv1.RoleUser, "Any refunds for cust_812 last month?")},
				Tools:      json.RawMessage(toolSchema),
				ToolChoice: json.RawMessage(`"auto"`),
			},
		},
		{
			name: "long-single-turn",
			req: &apiv1.ChatRequest{
				Model: "llama-3.3-70b",
				Messages: []apiv1.Message{
					msg(apiv1.RoleSystem, loremProse),
					msg(apiv1.RoleUser, userProse+"\n\n"+userProse+"\n\n"+goCode),
				},
			},
		},
		{
			name: "named-participants",
			req: &apiv1.ChatRequest{
				Model: "mistral-large",
				Messages: []apiv1.Message{
					{Role: apiv1.RoleSystem, Content: apiv1.NewTextContent("Moderate the thread."), Name: "moderator"},
					{Role: apiv1.RoleUser, Content: apiv1.NewTextContent("The deploy broke staging again."), Name: "alice"},
					{Role: apiv1.RoleUser, Content: apiv1.NewTextContent("Rolling back now."), Name: "bob"},
				},
			},
		},
	}
}

// mustUnmarshalReq decodes a request from JSON, so that fixtures exercising the
// union-typed content forms go through the real decoder rather than being
// hand-built (Content's private fields make hand-building the array form
// impossible from here, which is exactly the property the estimator relies on).
func mustUnmarshalReq(t *testing.T, body string) *apiv1.ChatRequest {
	t.Helper()
	var r apiv1.ChatRequest
	if err := json.Unmarshal([]byte(body), &r); err != nil {
		t.Fatalf("fixture does not decode: %v", err)
	}
	return &r
}
