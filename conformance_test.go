package libsygo

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/dio/libsygo/jev"
)

// The corpus and its recorded golden are canonical here, in libsygo, because they
// are this package's conformance spec — and a spec it cannot replay is not much of
// one. The Rust oracle that *records* the golden lives in github.com/dio/switchyard-capi,
// along with the upstream pin and the drift job, because recording needs a Rust
// toolchain and a network fetch and this package must never need either.
//
// So this test is the hermetic half: pure Go, no toolchain, no network, run on every
// commit. It decides whether this package still does what it did, not whether
// upstream has moved — that is the drift job's question, and only the oracle can
// answer it.
//
// Regenerate the golden with:
//
//	cd ../switchyard-capi && make verify   # the oracle still matches what is recorded
//	cd ../switchyard-capi && cargo run -q -p libsy-conform -- --mode record

// maxPortedTier is the highest corpus tier this package implements.
//
// Anything above it belongs to a family that is not ported. Replaying those would
// report a failure that is a missing feature rather than a divergence, which would
// train everyone to ignore the test.
const maxPortedTier = 1

// corpusCase mirrors the harness's case schema and the decider_responses fixtures.
type corpusCase struct {
	CaseID     string            `json:"case_id"`
	Tier       *int              `json:"tier"`
	Profile    Profile           `json:"profile"`
	Models     Models            `json:"models"`
	Headers    map[string]string `json:"headers"`
	Prompt     *string           `json:"prompt"`
	Model      *string           `json:"model"`
	RawRequest json.RawMessage   `json:"raw_request"`
	WireFormat string            `json:"wire_format"`
	Decider    *corpusDecider    `json:"decider_responses"`
	Turns      []corpusTurn      `json:"turns"`
}

type corpusTurn struct {
	Headers    map[string]string `json:"headers"`
	Prompt     *string           `json:"prompt"`
	Model      *string           `json:"model"`
	RawRequest json.RawMessage   `json:"raw_request"`
	WireFormat string            `json:"wire_format"`
	Decider    *corpusDecider    `json:"decider_responses"`
}

type corpusDecider struct {
	ByIndex []corpusScripted `json:"by_index"`
}

type corpusScripted struct {
	Kind     string          `json:"kind"`
	Response json.RawMessage `json:"response"`
	Message  string          `json:"message"`
}

// goldenTurn is one recorded oracle result. Only the fields this package is
// responsible for are read; a field that appears in the golden but not here is one
// another component owns.
//
// Reading the schema is deliberate: if the harness changes a field, this test stops
// compiling or fails rather than passing while comparing less than it claims.
type goldenTurn struct {
	CaseID             string          `json:"case_id"`
	Turn               int             `json:"turn"`
	Steps              []string        `json:"steps"`
	SelectedModelIDs   []string        `json:"selected_model_ids"`
	SelectedModelID    *string         `json:"selected_model_id"`
	RewrittenModel     *string         `json:"rewritten_model"`
	Instructions       json.RawMessage `json:"instructions"`
	MessagesDigest     string          `json:"messages_digest"`
	ResponseFormat     json.RawMessage `json:"response_format"`
	JudgeTask          json.RawMessage `json:"judge_task"`
	Evidence           json.RawMessage `json:"evidence"`
	Algorithm          *string         `json:"algorithm"`
	ResponsePresent    bool            `json:"response_present"`
	UnconsumedScripted int             `json:"unconsumed_scripted"`
	Error              *string         `json:"error"`
}

func (c corpusCase) turns() []corpusTurn {
	if len(c.Turns) > 0 {
		return c.Turns
	}
	return []corpusTurn{{
		Headers:    c.Headers,
		Prompt:     c.Prompt,
		Model:      c.Model,
		RawRequest: c.RawRequest,
		WireFormat: c.WireFormat,
		Decider:    c.Decider,
	}}
}

func (t corpusTurn) fixtures() []corpusScripted {
	if t.Decider == nil {
		return nil
	}
	return t.Decider.ByIndex
}

func (t corpusTurn) session() string { return t.Headers["x-switchyard-session-id"] }

func (t corpusTurn) promptText() string {
	if t.Prompt == nil {
		return ""
	}
	return *t.Prompt
}

// request builds the turn's normalized request the same way the harness does, so a
// divergence here is a real one rather than a difference in how a case is read.
func (t corpusTurn) request() (LlmRequest, error) {
	if len(t.RawRequest) == 0 {
		request := TextRequest(t.promptText())
		// The model the request arrived addressed to, which routing replaces. An
		// algorithm that does not route still reads it.
		if t.Model != nil {
			request.Model = *t.Model
		}
		return request, nil
	}
	switch t.WireFormat {
	case "", "openai_chat":
		return DecodeOpenAIChat(t.RawRequest)
	case "anthropic_messages":
		return DecodeAnthropicMessages(t.RawRequest)
	case "openai_responses":
		return DecodeResponses(t.RawRequest)
	default:
		return LlmRequest{}, fmt.Errorf("wire format %q is not ported", t.WireFormat)
	}
}

// scriptedDecider serves judge calls from a case's fixtures, in order, counting what
// it consumed so an unconsumed fixture is visible as a routing surprise.
type scriptedDecider struct {
	fixtures []corpusScripted
	consumed int
	calls    int
	// judgeTask is what the first judge call of the turn showed: the turn's *selection*
	// rather than the conversation it was drawn from.
	judgeTask json.RawMessage
}

func (d *scriptedDecider) Call(_ context.Context, request jev.DecisionRequest) (jev.DecisionResponse, error) {
	d.calls++
	if d.judgeTask == nil {
		var context struct {
			Task json.RawMessage `json:"task"`
		}
		if err := json.Unmarshal(request.Context, &context); err == nil {
			d.judgeTask = context.Task
		}
	}
	if d.consumed >= len(d.fixtures) {
		return jev.DecisionResponse{}, fmt.Errorf("unexpected judge call: the case scripted no reply")
	}
	reply := d.fixtures[d.consumed]
	d.consumed++
	switch reply.Kind {
	case "decision":
		var response jev.DecisionResponse
		if err := json.Unmarshal(reply.Response, &response); err != nil {
			return jev.DecisionResponse{}, err
		}
		return response, nil
	case "decision_error":
		return jev.DecisionResponse{}, fmt.Errorf("%s", reply.Message)
	default:
		return jev.DecisionResponse{}, fmt.Errorf("scripted step %q cannot serve a judge call", reply.Kind)
	}
}

func (d *scriptedDecider) unconsumed() int { return len(d.fixtures) - d.consumed }

// TestCorpusReplaysTheRecordedGolden is the hermetic conformance check: every tier
// the port covers, replayed against what the oracle recorded.
func TestCorpusReplaysTheRecordedGolden(t *testing.T) {
	pin, err := os.ReadFile(filepath.Join("testdata", "corpus", "PIN"))
	if err != nil {
		t.Fatalf("the corpus pin is missing: %v", err)
	}
	t.Logf("golden recorded from Switchyard %s", strings.TrimSpace(string(pin)))

	suites, err := filepath.Glob(filepath.Join("testdata", "corpus", "*.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if len(suites) == 0 {
		t.Fatal("no corpus suites found")
	}
	sort.Strings(suites)

	compared, skipped := 0, 0
	for _, path := range suites {
		suite := strings.TrimSuffix(filepath.Base(path), ".jsonl")
		cases := loadCorpus(t, path)
		golden := loadGolden(t, filepath.Join("testdata", "golden", suite+".jsonl"))

		for _, c := range cases {
			if c.Tier == nil || *c.Tier > maxPortedTier {
				skipped++
				continue
			}
			for index, result := range replayCase(c) {
				key := goldenKey{c.CaseID, index}
				want, recorded := golden[key]
				if !recorded {
					t.Errorf("%s turn %d: no recorded result", c.CaseID, index)
					continue
				}
				compared++
				compareTurn(t, c.CaseID, index, want, result)
			}
		}
	}

	if compared == 0 {
		t.Fatal("nothing was compared: the tier filter and the corpus disagree")
	}
	t.Logf("compared %d turn(s); skipped %d case(s) above tier %d", compared, skipped, maxPortedTier)
}

// replayCase runs a case's turns against one router, so session state persists
// across turns exactly as it would in production.
func replayCase(c corpusCase) []goldenTurn {
	router, err := New(c.Profile, c.Models)
	if err != nil {
		// A profile this package refuses is a rejection, which is what the oracle
		// records for the same input.
		return []goldenTurn{rejectedTurn(c.CaseID)}
	}

	var results []goldenTurn
	for index, turn := range c.turns() {
		results = append(results, replayTurn(router, c.CaseID, index, turn))
	}
	return results
}

func replayTurn(router *Router, caseID string, index int, turn corpusTurn) goldenTurn {
	request, err := turn.request()
	if err != nil {
		return rejectedTurn(caseID)
	}

	decider := &scriptedDecider{fixtures: turn.fixtures()}
	outcome := router.Route(context.Background(), Turn{
		Request: request,
		Session: turn.session(),
	}, decider)

	result := goldenTurn{
		CaseID:             caseID,
		Turn:               index,
		UnconsumedScripted: decider.unconsumed(),
	}
	if outcome.Error != "" {
		result.Steps = []string{"Error"}
		if outcome.Calls > 0 {
			result.Steps = []string{"CallDecision", "Error"}
		}
		result.SelectedModelIDs = []string{}
		result.Error = &outcome.Error
		// A failed turn still showed the judge something, and the recorded result keeps
		// it: the error path is not a reason to lose what was asked.
		result.JudgeTask = decider.judgeTask
		return result
	}

	result.Steps = []string{"Done"}
	if outcome.Calls > 0 {
		result.Steps = []string{"CallDecision", "Done"}
	}
	result.SelectedModelIDs = outcome.SelectedModelIDs
	result.SelectedModelID = &outcome.Model
	result.RewrittenModel = &outcome.Request.Model
	// The rewritten request, not the one that went in: a decision can splice a tier
	// system prompt into the instructions, so the input is not what the model gets.
	result.Instructions = InstructionsJSON(outcome.Request.Instructions)
	result.MessagesDigest = MessagesDigest(outcome.Request.Messages)
	result.ResponseFormat = outcome.Request.Output.ResponseFormat
	result.JudgeTask = decider.judgeTask
	result.Evidence = outcome.Evidence
	result.ResponsePresent = outcome.Responded
	algorithm := outcome.Algorithm
	result.Algorithm = &algorithm
	return result
}

// rejectedTurn is the recorded shape of a request refused before it could route.
// The oracle's wording is not compared anywhere — only the label — because two
// implementations will not phrase a refusal identically.
func rejectedTurn(caseID string) goldenTurn {
	rejected := "request_rejected"
	return goldenTurn{
		CaseID:           caseID,
		Steps:            []string{"Error"},
		SelectedModelIDs: []string{},
		Instructions:     json.RawMessage("null"),
		Error:            &rejected,
	}
}

func compareTurn(t *testing.T, caseID string, index int, want, got goldenTurn) {
	t.Helper()
	at := fmt.Sprintf("%s turn %d", caseID, index)

	if diff := strings.Join(want.Steps, ","); diff != strings.Join(got.Steps, ",") {
		t.Errorf("%s: steps = %v, want %v", at, got.Steps, want.Steps)
	}
	if diff := strings.Join(want.SelectedModelIDs, ","); diff != strings.Join(got.SelectedModelIDs, ",") {
		t.Errorf("%s: selected_model_ids = %v, want %v", at, got.SelectedModelIDs, want.SelectedModelIDs)
	}
	compareOptionalString(t, at, "selected_model_id", want.SelectedModelID, got.SelectedModelID)
	compareOptionalString(t, at, "rewritten_model", want.RewrittenModel, got.RewrittenModel)
	compareOptionalString(t, at, "algorithm", want.Algorithm, got.Algorithm)
	compareOptionalString(t, at, "error", want.Error, got.Error)

	if want.MessagesDigest != got.MessagesDigest {
		t.Errorf("%s: messages_digest = %s, want %s", at, got.MessagesDigest, want.MessagesDigest)
	}
	if want.ResponsePresent != got.ResponsePresent {
		t.Errorf("%s: response_present = %v, want %v", at, got.ResponsePresent, want.ResponsePresent)
	}
	if want.UnconsumedScripted != got.UnconsumedScripted {
		t.Errorf("%s: unconsumed_scripted = %d, want %d", at, got.UnconsumedScripted, want.UnconsumedScripted)
	}

	// Compared structurally but not semantically: object key order is ignored, because
	// the oracle serializes with preserve_order and this package does not, while scalars
	// are compared *as written*. That matters for numbers — serde writes a float-valued
	// zero as 0.0 and Go writes 0, and the harness compares numbers as serialized — so a
	// port that spells its floats differently has to fail here too rather than only in
	// the Rust harness.
	compareJSON(t, at, "instructions", want.Instructions, got.Instructions)
	compareJSON(t, at, "response_format", want.ResponseFormat, got.ResponseFormat)
	compareJSON(t, at, "judge_task", want.JudgeTask, got.JudgeTask)
	compareJSON(t, at, "evidence", want.Evidence, got.Evidence)
}

func compareOptionalString(t *testing.T, at, field string, want, got *string) {
	t.Helper()
	switch {
	case want == nil && got == nil:
	case want == nil || got == nil:
		t.Errorf("%s: %s = %v, want %v", at, field, render(got), render(want))
	case *want != *got:
		t.Errorf("%s: %s = %q, want %q", at, field, *got, *want)
	}
}

func render(value *string) string {
	if value == nil {
		return "null"
	}
	return fmt.Sprintf("%q", *value)
}

func compareJSON(t *testing.T, at, field string, want, got json.RawMessage) {
	t.Helper()
	if !sameJSON(want, got) {
		t.Errorf("%s: %s = %s, want %s", at, field, compactOrNull(got), compactOrNull(want))
	}
}

// sameJSON compares two values the way the harness does: recursively, ignoring object
// key order, and with scalars compared as they are written.
func sameJSON(a, b json.RawMessage) bool {
	left, right := bytes.TrimSpace(a), bytes.TrimSpace(b)
	// An absent value and an explicit null are the same thing to the schema: the
	// oracle's fields are optional, so both spell as null on its side.
	if len(left) == 0 {
		left = []byte("null")
	}
	if len(right) == 0 {
		right = []byte("null")
	}
	switch left[0] {
	case '{':
		var leftObject, rightObject map[string]json.RawMessage
		if json.Unmarshal(left, &leftObject) != nil || json.Unmarshal(right, &rightObject) != nil {
			return false
		}
		if len(leftObject) != len(rightObject) {
			return false
		}
		for key, value := range leftObject {
			other, present := rightObject[key]
			if !present || !sameJSON(value, other) {
				return false
			}
		}
		return true
	case '[':
		var leftArray, rightArray []json.RawMessage
		if json.Unmarshal(left, &leftArray) != nil || json.Unmarshal(right, &rightArray) != nil {
			return false
		}
		if len(leftArray) != len(rightArray) {
			return false
		}
		for index := range leftArray {
			if !sameJSON(leftArray[index], rightArray[index]) {
				return false
			}
		}
		return true
	case '"':
		var leftText, rightText string
		if json.Unmarshal(left, &leftText) != nil || json.Unmarshal(right, &rightText) != nil {
			return false
		}
		return leftText == rightText
	default:
		return string(left) == string(right)
	}
}

func compactOrNull(raw json.RawMessage) string {
	if len(raw) == 0 {
		return "null"
	}
	return string(CompactJSON(raw))
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

type goldenKey struct {
	caseID string
	turn   int
}

func loadCorpus(t *testing.T, path string) []corpusCase {
	t.Helper()
	var cases []corpusCase
	eachJSONLine(t, path, func(line []byte) {
		var c corpusCase
		if err := json.Unmarshal(line, &c); err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		cases = append(cases, c)
	})
	return cases
}

func loadGolden(t *testing.T, path string) map[goldenKey]goldenTurn {
	t.Helper()
	golden := map[goldenKey]goldenTurn{}
	eachJSONLine(t, path, func(line []byte) {
		var turn goldenTurn
		if err := json.Unmarshal(line, &turn); err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		golden[goldenKey{turn.CaseID, turn.Turn}] = turn
	})
	return golden
}

// eachJSONLine reads a JSONL suite, ignoring blank lines and `//` comments, which is
// how the corpus is authored.
func eachJSONLine(t *testing.T, path string, visit func([]byte)) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	for number, line := range strings.Split(string(raw), "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "//") {
			continue
		}
		if !json.Valid([]byte(trimmed)) {
			t.Fatalf("%s:%d: not valid JSON", path, number+1)
		}
		visit([]byte(trimmed))
	}
}
