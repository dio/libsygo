package libsygo

import (
	"context"
	"sync"
)

// ClassifyTrigger says how often a session's target is re-decided.
type ClassifyTrigger string

const (
	// EveryRequest judges every turn, tool continuations included.
	EveryRequest ClassifyTrigger = "every_request"
	// UserTurn judges each new user message and holds the target across the tool
	// calls between. Needs the normalized request to tell a human turn from a tool
	// continuation, so it lands with the signal router.
	UserTurn ClassifyTrigger = "user_turn"
	// NewSession judges once and reuses the target for the session.
	NewSession ClassifyTrigger = "new_session"
)

// AffinityRouter retains the first target chosen for a session and forces it on
// later turns sharing that session.
//
// It is one object playing both roles — the write side is a [Processor], the read
// side is a [Classifier] — so registering it as both cannot drift apart. Upstream
// makes the same choice, and for the same reason.
type AffinityRouter struct {
	// onlyModels narrows which targets are retained. A decision for any other
	// model routes normally but is not latched: the escalation latch, which
	// retains the strong tier and never the weak one.
	onlyModels map[string]bool

	mu          sync.Mutex
	assignments map[string]assignment
}

type assignment struct {
	model    string
	category string
}

// NewAffinityRouter retains every decision.
func NewAffinityRouter() *AffinityRouter {
	return &AffinityRouter{assignments: map[string]assignment{}}
}

// LatchOnly narrows retention to the given models. A decision for any other model
// routes normally but is not retained.
func (a *AffinityRouter) LatchOnly(models ...string) *AffinityRouter {
	a.onlyModels = make(map[string]bool, len(models))
	for _, model := range models {
		a.onlyModels[model] = true
	}
	return a
}

// Process is the write side: the first target chosen for a session is retained.
func (a *AffinityRouter) Process(_ *State, event Event) error {
	if event.Kind != EventDecision || !a.shouldLatch(event.SelectedModel) {
		return nil
	}
	session := sessionKey(event.Turn)
	if session == "" {
		return nil
	}

	a.mu.Lock()
	defer a.mu.Unlock()
	if _, held := a.assignments[session]; !held {
		a.assignments[session] = assignment{model: event.SelectedModel, category: event.Category}
	}
	return nil
}

// Classify is the read side: score the retained target at full confidence, or
// abstain so the cascade consults the next classifier.
func (a *AffinityRouter) Classify(
	_ context.Context, _ *State, turn *Turn, driver *Driver,
) (Classification, error) {
	session := sessionKey(turn)
	if session == "" {
		return Scores(), nil
	}

	a.mu.Lock()
	held, ok := a.assignments[session]
	a.mu.Unlock()
	if !ok {
		return Scores(), nil
	}

	// The evidence names the latch as the reason, which is what makes a skipped
	// judge call legible after the fact.
	driver.SetEvidenceIfEmpty(map[string]any{"source": "retained"})
	return Scores(Score{Target: held.model, Category: held.category, Confidence: 1}), nil
}

// Forget drops a session's retained target, so an ended session does not retain
// state. Honouring a session-final signal keeps a long-lived gateway bounded.
func (a *AffinityRouter) Forget(session string) {
	if session == "" {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	delete(a.assignments, session)
}

func (a *AffinityRouter) shouldLatch(model string) bool {
	if a.onlyModels == nil {
		return true
	}
	return a.onlyModels[model]
}

// sessionKey is the identity a turn's state is retained under.
//
// Upstream derives it from correlation metadata and refines it to session+agent
// for delegated child-agent work. This port takes it directly, so the refinement
// lands with the sub-agent family.
func sessionKey(turn *Turn) string {
	if turn == nil {
		return ""
	}
	return turn.Session
}
