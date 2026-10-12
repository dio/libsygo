package libsygo

import (
	"context"
	"errors"
	"math"
	"testing"
)

// stubClassifier returns a fixed verdict, so cascade order can be tested without
// a judge.
type stubClassifier struct {
	classification Classification
	err            error
}

func (s stubClassifier) Classify(context.Context, *State, *Turn, *Driver) (Classification, error) {
	return s.classification, s.err
}

func driverFor(models Models) *Driver { return &Driver{models: models} }

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

func TestArgmaxPicksTheHighestConfidence(t *testing.T) {
	classification := Scores(
		Score{Target: "weak", Confidence: 0.2},
		Score{Target: "strong", Confidence: 0.9},
		Score{Target: "mid", Confidence: 0.5},
	)

	best, err := classification.Argmax(false)
	if err != nil {
		t.Fatal(err)
	}
	if best == nil || best.Target != "strong" {
		t.Fatalf("best = %+v, want strong", best)
	}
}

// Ties keep the earlier score, so cascade order breaks them rather than the
// comparison being unstable.
func TestArgmaxKeepsTheEarlierOnATie(t *testing.T) {
	best, err := Scores(
		Score{Target: "first", Confidence: 0.7},
		Score{Target: "second", Confidence: 0.7},
	).Argmax(false)
	if err != nil {
		t.Fatal(err)
	}
	if best == nil || best.Target != "first" {
		t.Fatalf("best = %+v, want first", best)
	}
}

func TestArgmaxAbstainsOnAnEmptySet(t *testing.T) {
	best, err := Scores().Argmax(false)
	if err != nil {
		t.Fatal(err)
	}
	if best != nil {
		t.Fatalf("best = %+v, want an abstention", best)
	}
}

// Ambiguity means "do not choose", unless the caller opts to ignore it.
func TestAmbiguityAbstainsUnlessIgnored(t *testing.T) {
	ambiguous := Ambiguity()

	best, err := ambiguous.Argmax(false)
	if err != nil {
		t.Fatal(err)
	}
	if best != nil {
		t.Fatalf("ambiguous classification chose %+v", best)
	}
}

// A NaN confidence has no defined ordering, so it surfaces rather than being
// silently skipped.
func TestArgmaxErrorsOnNaNConfidence(t *testing.T) {
	_, err := Scores(
		Score{Target: "weak", Confidence: 0.3},
		Score{Target: "broken", Confidence: math.NaN()},
	).Argmax(false)
	if err == nil {
		t.Fatal("expected an error for a NaN confidence")
	}
}

// ---------------------------------------------------------------------------
// Cascade
// ---------------------------------------------------------------------------

// The first classifier to score decides; later ones are not consulted.
func TestCascadeFirstToScoreWins(t *testing.T) {
	second := &countingClassifier{verdict: Scores(Score{Target: "cheap", Confidence: 1})}
	cascade := NewFallThrough("test", testModels()).
		WithClassifier(stubClassifier{classification: Scores()}).
		WithClassifier(second)

	outcome, err := cascade.route(context.Background(), Turn{Request: TextRequest("anything")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if outcome.Model != "cheap" {
		t.Fatalf("model = %q, want cheap", outcome.Model)
	}
	if second.calls != 1 {
		t.Fatalf("second classifier consulted %d times, want 1", second.calls)
	}
}

// A classifier that abstains passes the turn on, which is how fail-open reaches
// the default classifier.
func TestCascadeAbstentionFallsThroughToTheDefault(t *testing.T) {
	cascade := NewFallThrough("test", testModels()).
		WithClassifier(stubClassifier{classification: Ambiguity()}).
		WithClassifier(stubClassifier{classification: Scores()}).
		WithClassifier(DefaultCategoryClassifier{Category: categoryCapable})

	outcome, err := cascade.route(context.Background(), Turn{Request: TextRequest("anything")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if outcome.Model != "smart" {
		t.Fatalf("model = %q, want the default capable tier", outcome.Model)
	}
	// The default is a fallback, not a judgement: zero confidence, matching
	// upstream's FallOpen.
	if outcome.Calls != 0 {
		t.Fatalf("calls = %d, want 0", outcome.Calls)
	}
}

// A cascade whose classifiers all abstain cannot route, and says so rather than
// picking arbitrarily.
func TestCascadeWithNoDeciderFails(t *testing.T) {
	cascade := NewFallThrough("test", testModels()).
		WithClassifier(stubClassifier{classification: Ambiguity()})

	if _, err := cascade.route(context.Background(), Turn{}, nil); err == nil {
		t.Fatal("expected an error when every classifier abstains")
	}
}

// A classifier error propagates, which is how fail-closed reaches the caller.
func TestCascadePropagatesAClassifierError(t *testing.T) {
	cascade := NewFallThrough("test", testModels()).
		WithClassifier(stubClassifier{err: errors.New("judge unavailable")})

	_, err := cascade.route(context.Background(), Turn{}, nil)
	if err == nil || err.Error() != "judge unavailable" {
		t.Fatalf("err = %v, want the classifier's error", err)
	}
}

// Processors see the request and then the decision, so a stateful one can bind
// the selection.
func TestProcessorsObserveRequestThenDecision(t *testing.T) {
	recorder := &recordingProcessor{}
	cascade := NewFallThrough("test", testModels()).
		WithProcessor(recorder).
		WithClassifier(stubClassifier{classification: Scores(Score{Target: "smart", Category: categoryCapable, Confidence: 1})})

	if _, err := cascade.route(context.Background(), Turn{Request: TextRequest("hi")}, nil); err != nil {
		t.Fatal(err)
	}
	if len(recorder.kinds) != 2 || recorder.kinds[0] != EventRequest || recorder.kinds[1] != EventDecision {
		t.Fatalf("events = %v, want [Request Decision]", recorder.kinds)
	}
	if recorder.decided != "smart" {
		t.Fatalf("decision replayed as %q, want smart", recorder.decided)
	}
}

// ---------------------------------------------------------------------------
// Affinity
// ---------------------------------------------------------------------------

// The write side retains the first assignment for an identity; the read side
// then decides the turn.
func TestAffinityRetainsAndScores(t *testing.T) {
	affinity := NewAffinityRouter()
	turn := &Turn{Session: "s-1"}

	if err := affinity.Process(newState(), Event{Kind: EventDecision, SelectedModel: "smart", Category: categoryCapable, Turn: turn}); err != nil {
		t.Fatal(err)
	}

	driver := driverFor(testModels())
	classification, err := affinity.Classify(context.Background(), newState(), turn, driver)
	if err != nil {
		t.Fatal(err)
	}
	best, err := classification.Argmax(false)
	if err != nil {
		t.Fatal(err)
	}
	if best == nil || best.Target != "smart" || best.Confidence != 1 {
		t.Fatalf("best = %+v, want smart at confidence 1", best)
	}
	if source := evidenceSource(driver.Evidence()); source != "retained" {
		t.Fatalf("evidence source = %q, want retained", source)
	}
}

// An unknown identity abstains, so the cascade consults the next classifier.
func TestAffinityAbstainsForAnUnknownSession(t *testing.T) {
	affinity := NewAffinityRouter()

	classification, err := affinity.Classify(context.Background(), newState(), &Turn{Session: "unseen"}, driverFor(testModels()))
	if err != nil {
		t.Fatal(err)
	}
	best, err := classification.Argmax(false)
	if err != nil {
		t.Fatal(err)
	}
	if best != nil {
		t.Fatalf("unseen session scored %+v", best)
	}
}

// The first assignment wins: a later decision does not overwrite it.
func TestAffinityKeepsTheFirstAssignment(t *testing.T) {
	affinity := NewAffinityRouter()
	turn := &Turn{Session: "s-1"}

	affinity.Process(newState(), Event{Kind: EventDecision, SelectedModel: "smart", Category: categoryCapable, Turn: turn})
	affinity.Process(newState(), Event{Kind: EventDecision, SelectedModel: "cheap", Category: categoryEfficient, Turn: turn})

	if held := affinity.assignments["s-1"]; held.model != "smart" {
		t.Fatalf("retained %q, want the first assignment smart", held.model)
	}
}

// LatchOnly retains some targets and not others, which is the escalation latch:
// hold the strong tier, never the weak one.
func TestAffinityLatchOnlyRefusesOtherModels(t *testing.T) {
	affinity := NewAffinityRouter().LatchOnly("smart")
	turn := &Turn{Session: "s-1"}

	affinity.Process(newState(), Event{Kind: EventDecision, SelectedModel: "cheap", Category: categoryEfficient, Turn: turn})
	if _, held := affinity.assignments["s-1"]; held {
		t.Fatal("cheap should not have been retained")
	}

	affinity.Process(newState(), Event{Kind: EventDecision, SelectedModel: "smart", Category: categoryCapable, Turn: turn})
	if _, held := affinity.assignments["s-1"]; !held {
		t.Fatal("smart should have been retained")
	}
}

// The read side abstains without an identity, so unkeyed traffic is not latched
// onto an arbitrary target.
func TestAffinityAbstainsWithoutAnIdentity(t *testing.T) {
	affinity := NewAffinityRouter()
	affinity.Process(newState(), Event{Kind: EventDecision, SelectedModel: "smart", Turn: &Turn{}})

	classification, err := affinity.Classify(context.Background(), newState(), &Turn{}, driverFor(testModels()))
	if err != nil {
		t.Fatal(err)
	}
	best, err := classification.Argmax(false)
	if err != nil {
		t.Fatal(err)
	}
	if best != nil {
		t.Fatalf("a turn with no identity scored %+v", best)
	}
}

type countingClassifier struct {
	verdict Classification
	calls   int
}

func (c *countingClassifier) Classify(context.Context, *State, *Turn, *Driver) (Classification, error) {
	c.calls++
	return c.verdict, nil
}

type recordingProcessor struct {
	kinds   []EventKind
	decided string
}

func (p *recordingProcessor) Process(_ *State, event Event) error {
	p.kinds = append(p.kinds, event.Kind)
	if event.Kind == EventDecision {
		p.decided = event.SelectedModel
	}
	return nil
}
