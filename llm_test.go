package libsygo

import (
	"encoding/json"
	"strings"
	"testing"
)

// The oracle's recorded digest for the single-message request "write a haiku".
//
// Taken from testdata/golden/decision.jsonl in github.com/dio/switchyard-capi: it
// is the one digest that was already being produced before normalization was
// ported, so it pins the serializer — field order, escaping and compactness —
// against an answer that is known to be right rather than against a Go value that
// merely round-trips through Go.
const oracleHaikuDigest = "sha256:7b247019ae91cc77baecfa093f71a3f4cf3d0af63d71ba9f979ecbd9e752de96"

func TestMessagesDigestMatchesTheRecordedOracle(t *testing.T) {
	messages := []Message{{Role: RoleUser, Content: []ContentBlock{Text("write a haiku")}}}

	if got := MessagesDigest(messages); got != oracleHaikuDigest {
		t.Fatalf("digest = %s, want the oracle's %s\nserialized: %s",
			got, oracleHaikuDigest, AppendMessages(nil, messages))
	}
}

// An empty message list must serialize as `[]`, not `null`: upstream serializes a
// Vec, and the digest would diverge on the first turn that produced no messages.
func TestEmptyMessagesSerializeAsAnArray(t *testing.T) {
	if got := string(AppendMessages(nil, nil)); got != "[]" {
		t.Fatalf("serialized = %s, want []", got)
	}
}

// serde_json escapes fewer characters than encoding/json does. Escaping `<`, `>`
// or `&` would change the digest for any transcript containing markup, which is
// most of them.
func TestSerializerEscapesLikeSerdeNotLikeGo(t *testing.T) {
	message := Message{Role: RoleUser, Content: []ContentBlock{Text(`<tag> & "quoted" \ café 🙂`)}}

	got := string(AppendMessages(nil, []Message{message}))
	want := `[{"role":"user","content":[{"type":"text","text":"<tag> & \"quoted\" \\ café 🙂"}]}]`
	if got != want {
		t.Fatalf("serialized = %s\nwant         %s", got, want)
	}

	// The contrast is the point: Go's own encoder escapes what serde does not.
	goEncoded, err := json.Marshal([]Message(nil))
	if err != nil {
		t.Fatal(err)
	}
	_ = goEncoded
	if strings.Contains(want, `\u003c`) {
		t.Fatal("test is wrong: the expectation itself escaped the angle brackets")
	}
}

func TestControlCharactersAreEscaped(t *testing.T) {
	got := string(AppendMessages(nil, []Message{
		{Role: RoleUser, Content: []ContentBlock{Text("line\nbreak\ttab\x01control")}},
	}))
	want := `[{"role":"user","content":[{"type":"text","text":"line\nbreak\ttab\u0001control"}]}]`
	if got != want {
		t.Fatalf("serialized = %s\nwant         %s", got, want)
	}
}

// Provider values are re-emitted from their original bytes, so a tool call's
// arguments keep the order the provider wrote. Re-encoding through a Go map would
// alphabetize them and change the digest.
func TestProviderValuesKeepTheirKeyOrder(t *testing.T) {
	arguments := json.RawMessage("{\"path\": \".\",\"sort\":  \"name\"}")

	got := string(AppendMessages(nil, []Message{{
		Role: RoleAssistant,
		Content: []ContentBlock{{Kind: ContentToolCall, Call: &ToolCall{
			ID: "call_1", Name: "ls", Arguments: arguments,
		}}},
	}}))

	want := `[{"role":"assistant","content":[{"type":"tool_call","id":"call_1","name":"ls",` +
		`"arguments":{"path":".","sort":"name"}}]}]`
	if got != want {
		t.Fatalf("serialized = %s\nwant         %s", got, want)
	}
}

func TestCompactJSONStripsWhitespaceButKeepsEverythingElse(t *testing.T) {
	cases := []struct{ in, want string }{
		{`{"z": 1, "a": 2}`, `{"z":1,"a":2}`},
		{"{\n  \"a\" : [ 1, 2 ]\n}", `{"a":[1,2]}`},
		// Whitespace inside a string is content, not formatting.
		{`{"a": "  keep  "}`, `{"a":"  keep  "}`},
		// An escaped quote does not end the string.
		{`{ "a": "say \"hi\" now" }`, `{"a":"say \"hi\" now"}`},
		// Number spelling survives rather than round-tripping through a float.
		{`{"n": 1e3, "m": 0.10}`, `{"n":1e3,"m":0.10}`},
	}
	for _, c := range cases {
		if got := string(CompactJSON([]byte(c.in))); got != c.want {
			t.Errorf("CompactJSON(%s) = %s, want %s", c.in, got, c.want)
		}
	}
}

// Instructions are serialized in the same shape as messages, because the harness
// compares them directly rather than by digest.
func TestInstructionsSerializeAsBlocks(t *testing.T) {
	got := string(InstructionsJSON([]InstructionBlock{{Role: RoleSystem, Content: []ContentBlock{Text("You are terse.")}}}))
	want := `[{"role":"system","content":[{"type":"text","text":"You are terse."}]}]`
	if got != want {
		t.Fatalf("serialized = %s\nwant         %s", got, want)
	}
	if got := string(InstructionsJSON(nil)); got != "[]" {
		t.Fatalf("empty instructions = %s, want []", got)
	}
}

// A reasoning block omits `details` when empty but always emits `signature`, and
// the empty string is normalized to absent — both upstream behaviours.
func TestReasoningBlockShape(t *testing.T) {
	text := "because"
	withSignature := ContentBlock{Kind: ContentReasoning, Text: "step", Signature: &text}
	got := string(AppendMessages(nil, []Message{{Role: RoleAssistant, Content: []ContentBlock{withSignature}}}))
	want := `[{"role":"assistant","content":[{"type":"reasoning","text":"step","signature":"because"}]}]`
	if got != want {
		t.Fatalf("with signature = %s\nwant           %s", got, want)
	}

	withoutSignature := ContentBlock{Kind: ContentReasoning, Text: "step"}
	got = string(AppendMessages(nil, []Message{{Role: RoleAssistant, Content: []ContentBlock{withoutSignature}}}))
	want = `[{"role":"assistant","content":[{"type":"reasoning","text":"step","signature":null}]}]`
	if got != want {
		t.Fatalf("without signature = %s\nwant              %s", got, want)
	}
}

func TestToolResultShapePreservesNullableError(t *testing.T) {
	failed := true
	got := string(AppendMessages(nil, []Message{{
		Role: RoleUser,
		Content: []ContentBlock{{Kind: ContentToolResult, Result: &ToolResult{
			ToolCallID: "call_9",
			Content:    []ContentBlock{Text("boom")},
			IsError:    &failed,
		}}},
	}}))
	want := `[{"role":"user","content":[{"type":"tool_result","tool_call_id":"call_9",` +
		`"content":[{"type":"text","text":"boom"}],"is_error":true}]}]`
	if got != want {
		t.Fatalf("serialized = %s\nwant         %s", got, want)
	}
}

// ---------------------------------------------------------------------------
// Task selection
// ---------------------------------------------------------------------------

// The judge sees the opening task and the latest follow-up, not the whole
// transcript: a long tool-driven conversation must not outrank the task it serves.
func TestTaskMessagesKeepsTheAnchorAndTheLatestFollowUp(t *testing.T) {
	messages := []Message{
		{Role: RoleSystem, Content: []ContentBlock{Text("be terse")}},
		{Role: RoleUser, Content: []ContentBlock{Text("plan the migration")}},
		{Role: RoleAssistant, Content: []ContentBlock{Text("which module?")}},
		{Role: RoleUser, Content: []ContentBlock{Text("src/auth")}},
	}

	selected := TaskMessages(messages)
	if len(selected) != 2 {
		t.Fatalf("selected %d messages, want 2: %+v", len(selected), selected)
	}
	if selected[0].Content[0].Text != "plan the migration" || selected[1].Content[0].Text != "src/auth" {
		t.Fatalf("selected %q and %q", selected[0].Content[0].Text, selected[1].Content[0].Text)
	}
	for _, message := range selected {
		if message.Role != RoleUser {
			t.Fatalf("role = %q, want user", message.Role)
		}
	}
}

// A tool result is represented as a user-role message, so role alone cannot tell
// it from a task. It must be neither the anchor nor the follow-up.
func TestTaskMessagesIgnoresToolResults(t *testing.T) {
	messages := []Message{
		{Role: RoleUser, Content: []ContentBlock{Text("list the files")}},
		{Role: RoleAssistant, Content: []ContentBlock{{Kind: ContentToolCall, Call: &ToolCall{ID: "c1", Name: "ls"}}}},
		{Role: RoleUser, Content: []ContentBlock{{Kind: ContentToolResult, Result: &ToolResult{ToolCallID: "c1"}}}},
	}

	selected := TaskMessages(messages)
	if len(selected) != 1 {
		t.Fatalf("selected %d messages, want just the opening task: %+v", len(selected), selected)
	}
	if selected[0].Content[0].Text != "list the files" {
		t.Fatalf("anchor = %q", selected[0].Content[0].Text)
	}
}

func TestTaskMessagesStripsReasoningFromTheSelection(t *testing.T) {
	text := "because"
	messages := []Message{
		{Role: RoleUser, Content: []ContentBlock{Text("do the thing")}},
		{Role: RoleUser, Content: []ContentBlock{
			{Kind: ContentReasoning, Text: "thinking", Signature: &text},
			Text("and also this"),
		}},
	}

	selected := TaskMessages(messages)
	if len(selected) != 2 {
		t.Fatalf("selected %d messages, want 2", len(selected))
	}
	for _, block := range selected[1].Content {
		if block.Kind == ContentReasoning {
			t.Fatal("reasoning reached the judge")
		}
	}
}

func TestTaskMessagesIsEmptyWithoutAUserTurn(t *testing.T) {
	if selected := TaskMessages([]Message{{Role: RoleAssistant, Content: []ContentBlock{Text("hi")}}}); len(selected) != 0 {
		t.Fatalf("selected %+v, want nothing", selected)
	}
}
