package libsygo

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/dio/libsygo/jev"
)

// fakeDecider answers with a chosen advantage, so the routing can be driven
// without a judge or a network.
type fakeDecider struct {
	advantage float64
	err       error
	calls     int
}

func (f *fakeDecider) Call(_ context.Context, _ jev.DecisionRequest) (jev.DecisionResponse, error) {
	f.calls++
	if f.err != nil {
		return jev.DecisionResponse{}, f.err
	}
	data, err := json.Marshal(map[string]any{
		"selected":      "advantage",
		"probabilities": map[string]float64{"advantage": f.advantage},
	})
	if err != nil {
		return jev.DecisionResponse{}, err
	}
	return jev.DecisionResponse{
		Answers: map[string]jev.DecisionAnswer{
			"route": {Value: jev.Tagged{Type: "choice", Data: data}},
		},
	}, nil
}

func testProfile() Profile {
	failOpen := true
	return Profile{
		Algorithm:       "llm_classifier",
		Mode:            "capability",
		FailOpen:        &failOpen,
		SessionAffinity: true,
		Judge: &Judge{
			Type:       "decision",
			Cutoff:     0.4,
			Candidates: map[string]string{"a": "smart", "b": "cheap"},
			Evidence:   json.RawMessage(`{"candidates":{"a":"capable","b":"efficient"}}`),
		},
	}
}

func testModels() Models {
	return Models{
		"capable":   {"smart"},
		"efficient": {"cheap"},
		"judge":     {"jev-latest"},
		"any":       {"cheap", "smart"},
	}
}

func newTestRouter(t *testing.T) *Router {
	t.Helper()
	router, err := New(testProfile(), testModels())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return router
}

// The cutoff is strictly greater, so a score exactly at it does not select the
// capable tier.
func TestCutoffIsStrictlyGreater(t *testing.T) {
	for _, tc := range []struct {
		advantage float64
		want      string
	}{
		{0.83, "smart"},
		{0.41, "smart"},
		{0.4, "cheap"},
		{0.21, "cheap"},
	} {
		router := newTestRouter(t)
		outcome := router.Route(context.Background(),
			Turn{Request: TextRequest("anything"), Session: "s"},
			&fakeDecider{advantage: tc.advantage})

		if outcome.Model != tc.want {
			t.Errorf("advantage %v selected %q, want %q", tc.advantage, outcome.Model, tc.want)
		}
	}
}

// The fallback order: the pick, then the rest of its category, then every other
// routable target.
func TestSelectedModelIDsCarryTheFallbackOrder(t *testing.T) {
	router := newTestRouter(t)
	outcome := router.Route(context.Background(),
		Turn{Request: TextRequest("anything")},
		&fakeDecider{advantage: 0.83})

	want := []string{"smart", "cheap"}
	if len(outcome.SelectedModelIDs) != len(want) {
		t.Fatalf("selected = %v, want %v", outcome.SelectedModelIDs, want)
	}
	for i := range want {
		if outcome.SelectedModelIDs[i] != want[i] {
			t.Fatalf("selected = %v, want %v", outcome.SelectedModelIDs, want)
		}
	}
}

// The decision carries its reasoning, so it is auditable after the fact.
func TestEvidenceNamesTheRule(t *testing.T) {
	router := newTestRouter(t)
	outcome := router.Route(context.Background(),
		Turn{Request: TextRequest("anything")},
		&fakeDecider{advantage: 0.83})

	var evidence struct {
		Source    string  `json:"source"`
		Verdict   string  `json:"verdict"`
		Score     float64 `json:"score"`
		Threshold float64 `json:"threshold"`
	}
	if err := json.Unmarshal(outcome.Evidence, &evidence); err != nil {
		t.Fatal(err)
	}
	if evidence.Source != "decision_classifier" || evidence.Verdict != "relative_advantage" {
		t.Fatalf("evidence = %+v", evidence)
	}
	if evidence.Score != 0.83 || evidence.Threshold != 0.4 {
		t.Fatalf("evidence = %+v", evidence)
	}
}

// A judge failure with fail-open routes to the capable tier and says so, rather
// than leaving the request unrouted.
func TestJudgeFailureFailsOpenByDefault(t *testing.T) {
	router := newTestRouter(t)
	outcome := router.Route(context.Background(),
		Turn{Request: TextRequest("anything")},
		&fakeDecider{err: errors.New("judge unavailable")})

	if outcome.Error != "" {
		t.Fatalf("Error = %q, want the request to be routed", outcome.Error)
	}
	if outcome.Model != "smart" {
		t.Fatalf("model = %q; fail-open lands on the capable tier", outcome.Model)
	}
	if !json.Valid(outcome.Evidence) {
		t.Fatalf("evidence is not JSON: %s", outcome.Evidence)
	}
	var evidence struct {
		Source string `json:"source"`
		Reason string `json:"reason_code"`
	}
	if err := json.Unmarshal(outcome.Evidence, &evidence); err != nil {
		t.Fatal(err)
	}
	if evidence.Source != "fail_open" || evidence.Reason != "call_error" {
		t.Fatalf("evidence = %+v", evidence)
	}
}

// fail_open = false propagates instead, so a tenant can choose to fail rather than
// be routed on a guess.
func TestJudgeFailurePropagatesWhenFailClosed(t *testing.T) {
	profile := testProfile()
	failOpen := false
	profile.FailOpen = &failOpen

	router, err := New(profile, testModels())
	if err != nil {
		t.Fatal(err)
	}
	outcome := router.Route(context.Background(),
		Turn{Request: TextRequest("anything")},
		&fakeDecider{err: errors.New("judge unavailable")})

	if outcome.Error == "" {
		t.Fatal("expected the failure to propagate")
	}
	if outcome.Model != "" {
		t.Fatalf("model = %q, want none selected", outcome.Model)
	}
}

// A verdict with no usable score is a failed verdict, not a crash.
func TestMissingScoreFailsOpenAsInvalidVerdict(t *testing.T) {
	router := newTestRouter(t)

	// An answer that is not a choice at all.
	decider := stubDecider{response: jev.DecisionResponse{
		Answers: map[string]jev.DecisionAnswer{
			"route": {Value: jev.Tagged{Type: "score", Data: json.RawMessage(`{}`)}},
		},
	}}
	outcome := router.Route(context.Background(), Turn{Request: TextRequest("anything")}, decider)

	var evidence struct {
		Source string `json:"source"`
		Reason string `json:"reason_code"`
	}
	if err := json.Unmarshal(outcome.Evidence, &evidence); err != nil {
		t.Fatal(err)
	}
	if evidence.Source != "fail_open" || evidence.Reason != "invalid_verdict" {
		t.Fatalf("evidence = %+v", outcome.Evidence)
	}
}

// The latch reuses a session's target without consulting the judge again.
func TestSessionLatchSkipsTheJudge(t *testing.T) {
	router := newTestRouter(t)
	decider := &fakeDecider{advantage: 0.83}

	first := router.Route(context.Background(), Turn{Request: TextRequest("one"), Session: "tenant-a/s-1"}, decider)
	second := router.Route(context.Background(), Turn{Request: TextRequest("two"), Session: "tenant-a/s-1"}, decider)

	if decider.calls != 1 {
		t.Fatalf("judge calls = %d, want 1; the second turn should be latched", decider.calls)
	}
	if first.Model != second.Model {
		t.Fatalf("latched turn returned %q, want %q", second.Model, first.Model)
	}
	if !second.Latched {
		t.Fatal("the second turn should report itself latched")
	}
	if !containsJSON(second.Evidence, `"source":"retained"`) {
		t.Fatalf("evidence = %s", second.Evidence)
	}
}

// A different session is judged on its own.
func TestDistinctSessionsAreJudgedSeparately(t *testing.T) {
	router := newTestRouter(t)
	decider := &fakeDecider{advantage: 0.83}

	router.Route(context.Background(), Turn{Request: TextRequest("one"), Session: "tenant-a/s-1"}, decider)
	router.Route(context.Background(), Turn{Request: TextRequest("two"), Session: "tenant-a/s-2"}, decider)

	if decider.calls != 2 {
		t.Fatalf("judge calls = %d, want 2", decider.calls)
	}
}

// Forget drops a session, so an ended session does not retain state.
func TestForgetReleasesALatchedSession(t *testing.T) {
	router := newTestRouter(t)
	decider := &fakeDecider{advantage: 0.83}

	router.Route(context.Background(), Turn{Request: TextRequest("one"), Session: "s"}, decider)
	router.Forget("s")
	router.Route(context.Background(), Turn{Request: TextRequest("two"), Session: "s"}, decider)

	if decider.calls != 2 {
		t.Fatalf("judge calls = %d, want 2 after Forget", decider.calls)
	}
}

// An unimplemented family is refused, not approximated.
func TestUnimplementedProfilesAreRefused(t *testing.T) {
	for _, tc := range []struct {
		name    string
		mutate  func(*Profile)
		wantErr string
	}{
		{"signal router", func(p *Profile) { p.Algorithm = "stage_router" }, "llm_classifier only"},
		{"other mode", func(p *Profile) { p.Mode = "escalation" }, "capability only"},
		{"other judge", func(p *Profile) { p.Judge.Type = "llm" }, "decision judge only"},
		{"no judge", func(p *Profile) { p.Judge = nil }, "decision judge only"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			profile := testProfile()
			tc.mutate(&profile)
			if _, err := New(profile, testModels()); err == nil {
				t.Fatal("expected construction to fail")
			}
		})
	}
}

// The pool invariant is enforced at construction, not mid-route.
func TestPoolMustListEveryTargetUnderAny(t *testing.T) {
	for _, tc := range []struct {
		name   string
		models Models
	}{
		{"missing any", Models{"capable": {"smart"}}},
		{"tier outside any", Models{"capable": {"smart"}, "any": {"cheap"}}},
		{"no judge", Models{"any": {"cheap", "smart"}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := New(testProfile(), tc.models); err == nil {
				t.Fatal("expected the pool to be rejected")
			}
		})
	}
}

// The judge is asked the route question the decision contract expects.
func TestJudgeIsAskedTheRouteChoice(t *testing.T) {
	router := newTestRouter(t)
	probe := &probeDecider{}
	router.Route(context.Background(), Turn{Request: TextRequest("refactor everything")}, probe)

	if probe.seen == nil {
		t.Fatal("the judge was not consulted")
	}
	if probe.seen.Model == nil || *probe.seen.Model != "jev-latest" {
		t.Fatalf("judge model = %v, want jev-latest", probe.seen.Model)
	}
	question, ok := probe.seen.Questions["route"]
	if !ok {
		t.Fatal("the route question was not asked")
	}
	if question.Kind.Type != "choice" {
		t.Fatalf("question kind = %q, want choice", question.Kind.Type)
	}
	var kind struct {
		Options []struct {
			ID string `json:"id"`
		} `json:"options"`
	}
	if err := json.Unmarshal(question.Kind.Data, &kind); err != nil {
		t.Fatal(err)
	}
	ids := []string{}
	for _, option := range kind.Options {
		ids = append(ids, option.ID)
	}
	if len(ids) != 2 || ids[0] != "advantage" || ids[1] != "no_advantage" {
		t.Fatalf("options = %v", ids)
	}
}

type stubDecider struct {
	response jev.DecisionResponse
}

func (s stubDecider) Call(context.Context, jev.DecisionRequest) (jev.DecisionResponse, error) {
	return s.response, nil
}

type probeDecider struct {
	seen *jev.DecisionRequest
}

func (p *probeDecider) Call(_ context.Context, request jev.DecisionRequest) (jev.DecisionResponse, error) {
	p.seen = &request
	return jev.DecisionResponse{}, errors.New("not judging")
}

func containsJSON(raw json.RawMessage, want string) bool {
	return strings.Contains(string(raw), want)
}
