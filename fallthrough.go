package libsygo

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// sessionTTL bounds how long an idle session's state is retained. Upstream uses
// the same hour.
const sessionTTL = time.Hour

// maxSessions bounds retained sessions, so a long-lived gateway stays bounded
// even when a caller never signals session end. Upstream evicts the oldest entry
// once the bound is reached, and so does this.
const maxSessions = 4096

// FallThrough routes each turn through a processor chain and a classifier
// cascade, retaining one private state per session.
//
// Each turn: request-side processors fold facts into state; the cascade is
// consulted in order and the first classifier to score decides; the selection is
// replayed to the processors so stateful ones can bind it.
type FallThrough struct {
	name        string
	models      Models
	processors  []Processor
	classifiers []Classifier
	sessions    *sessionStore
}

// NewFallThrough builds an empty cascade over a model pool.
func NewFallThrough(name string, models Models) *FallThrough {
	return &FallThrough{name: name, models: models, sessions: newSessionStore()}
}

// WithProcessor appends a processor to the request-side chain.
func (f *FallThrough) WithProcessor(processor Processor) *FallThrough {
	f.processors = append(f.processors, processor)
	return f
}

// WithClassifier appends a classifier to the cascade. Order matters: the first to
// score decides.
func (f *FallThrough) WithClassifier(classifier Classifier) *FallThrough {
	f.classifiers = append(f.classifiers, classifier)
	return f
}

// Forget drops a session's retained state.
func (f *FallThrough) Forget(session string) { f.sessions.forget(session) }

// route runs one turn.
//
// It returns an Outcome even alongside an error, because a failed turn still made
// calls worth reporting.
func (f *FallThrough) route(ctx context.Context, turn Turn, judge Decider) (Outcome, error) {
	state := f.sessions.stateFor(turn.Session)
	driver := &Driver{models: f.models, judge: judge}

	// 1. Request-side processors accumulate facts.
	for _, processor := range f.processors {
		if err := processor.Process(state, Event{Kind: EventRequest, Turn: &turn}); err != nil {
			return Outcome{Calls: driver.Calls()}, err
		}
	}

	// 2. The cascade: the first classifier to score decides.
	var winner *Score
	for _, classifier := range f.classifiers {
		classification, err := classifier.Classify(ctx, state, &turn, driver)
		if err != nil {
			return Outcome{Calls: driver.Calls()}, err
		}
		score, err := classification.Argmax(false)
		if err != nil {
			return Outcome{Calls: driver.Calls()}, err
		}
		if score != nil {
			winner = score
			break
		}
	}
	if winner == nil {
		return Outcome{Calls: driver.Calls()}, fmt.Errorf("%s: every classifier abstained", f.name)
	}

	// 3. Post-decision replay: stateful processors bind the selection, and may
	// rewrite the turn for the model that will receive it.
	for _, processor := range f.processors {
		event := Event{
			Kind:          EventDecision,
			Turn:          &turn,
			SelectedModel: winner.Target,
			Category:      winner.Category,
		}
		if err := processor.Process(state, event); err != nil {
			return Outcome{Calls: driver.Calls()}, err
		}
	}

	evidence := driver.Evidence()
	outcome := Outcome{
		SelectedModelIDs: orderedCandidates(f.models, winner.Target, winner.Category),
		Model:            winner.Target,
		Category:         winner.Category,
		Evidence:         evidence,
		Calls:            driver.Calls(),
		Latched:          evidenceSource(evidence) == "retained",
	}
	return outcome, nil
}

// orderedCandidates reproduces libsy's fallback order: the pick, then the rest of
// the category it was drawn from, then every other runtime target.
func orderedCandidates(models Models, selected, category string) []string {
	out := []string{selected}
	add := func(list []string) {
		for _, model := range list {
			if model == selected || contains(out, model) {
				continue
			}
			out = append(out, model)
		}
	}
	add(models[category])
	add(models[categoryAny])
	return out
}

// sessionStore retains one state per session, bounded and expiring.
//
// Eviction is lazy — on access — rather than a periodic task, so the library
// spawns no goroutine to maintain it. Behaviour is the same for routing, which
// only ever reads state on access.
type sessionStore struct {
	mu     sync.Mutex
	states map[string]*retained
}

type retained struct {
	state   *State
	touched time.Time
}

func newSessionStore() *sessionStore {
	return &sessionStore{states: map[string]*retained{}}
}

// stateFor returns a session's retained state, or an unretained one when the turn
// carries no session.
func (s *sessionStore) stateFor(session string) *State {
	if session == "" {
		return newState()
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	now := time.Now()
	if held, ok := s.states[session]; ok && now.Sub(held.touched) < sessionTTL {
		held.touched = now
		return held.state
	}

	state := newState()
	s.states[session] = &retained{state: state, touched: now}
	s.evictOldestLocked()
	return state
}

func (s *sessionStore) forget(session string) {
	if session == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.states, session)
}

// evictOldestLocked drops the least-recently-used session once the bound is
// reached. The scan is linear, but it only runs when the bound is exceeded.
func (s *sessionStore) evictOldestLocked() {
	if len(s.states) <= maxSessions {
		return
	}
	oldest, oldestAt := "", time.Time{}
	for session, held := range s.states {
		if oldestAt.IsZero() || held.touched.Before(oldestAt) {
			oldest, oldestAt = session, held.touched
		}
	}
	delete(s.states, oldest)
}

// classifyTrigger maps a profile's session affinity onto a trigger. Affinity is
// one trigger of several: a profile that judges every request has no latch at all.
func classifyTrigger(sessionAffinity bool) ClassifyTrigger {
	if sessionAffinity {
		return NewSession
	}
	return EveryRequest
}
