package libsygo

import (
	"context"
	"encoding/json"
	"fmt"
)

// Outcome is the terminal result of routing one turn.
type Outcome struct {
	// SelectedModelIDs is libsy's fallback order: the chosen model first, then
	// the rest of the category it was drawn from, then every other routable
	// model. A caller that cannot reach the first may try the rest in order.
	SelectedModelIDs []string

	// Model is the head of SelectedModelIDs.
	Model string

	// Category the model was drawn from: "capable" or "efficient".
	Category string

	// Evidence explains the decision and is what makes it auditable:
	// `{source, verdict, score, threshold}` on a judgement, `{source: "retained"}`
	// on a latched session, `{source: "fail_open", reason_code}` on a fallback.
	Evidence json.RawMessage

	// Latched is true when a retained session decided this turn and no judge call
	// was made.
	Latched bool

	// Error is set when the turn failed rather than mis-routed: a judge failure
	// with fail-open disabled. The caller should surface it, not route.
	Error string

	// Calls is how many judge calls this turn made (0 when latched).
	Calls int
}

// Router is a built profile: a composition over the model pool.
//
// It retains per-session state, so one Router serves one profile — cache it by
// (tenant, profile ID, revision) and namespace session ids per tenant. A Router
// is safe for concurrent use.
//
// The judge [Decider] is supplied per Route call rather than at construction,
// because it is usually tied to a single request (the host's callout for that
// stream). The Router is the long-lived, shared part.
type Router struct {
	cascade  *FallThrough
	affinity *AffinityRouter
}

// New builds a router, refusing anything not implemented rather than
// approximating it.
func New(profile Profile, models Models) (*Router, error) {
	if profile.Algorithm != "llm_classifier" {
		return nil, fmt.Errorf(
			"profile algorithm %q is not implemented: only llm_classifier with a decision judge is",
			profile.Algorithm)
	}
	if profile.Mode != "capability" {
		return nil, fmt.Errorf("llm_classifier mode %q is not implemented: only capability is", profile.Mode)
	}
	if profile.Judge == nil || profile.Judge.Type != "decision" {
		return nil, fmt.Errorf("judge is not implemented: only the decision (System One) judge is")
	}
	if err := validatePool(models); err != nil {
		return nil, err
	}
	if len(models[categoryJudge]) == 0 {
		return nil, fmt.Errorf("models.judge must name at least one judge model")
	}

	instructions := profile.Judge.Instructions
	if len(instructions) == 0 {
		instructions = json.RawMessage(relativeAdvantagePrompt)
	}
	failOpen := true
	if profile.FailOpen != nil {
		failOpen = *profile.FailOpen
	}

	capability := &capabilityClassifier{
		judge:        profile.Judge,
		instructions: instructions,
		failOpen:     failOpen,
	}

	cascade := NewFallThrough("llm_classifier", models)
	var affinity *AffinityRouter
	if classifyTrigger(profile.SessionAffinity) != EveryRequest {
		// The latch leads the cascade: a retained session decides without a judge
		// call. Registered as both a processor and a classifier so the two roles
		// share one assignment map.
		affinity = NewAffinityRouter()
		cascade.WithProcessor(affinity).WithClassifier(affinity)
	}
	cascade.
		WithClassifier(capability).
		// Closes the cascade, so a turn is never left unrouted — which is what
		// makes fail-open land on the capable tier.
		WithClassifier(DefaultCategoryClassifier{Category: categoryCapable})

	return &Router{cascade: cascade, affinity: affinity}, nil
}

// Route decides one turn, making at most one judge call through decider.
//
// A latched session returns without a judge call. Otherwise decider is consulted;
// a verdict that is missing, malformed, or the result of a failed call falls back
// per FailOpen. A nil decider is treated as a failed call rather than a panic.
func (r *Router) Route(ctx context.Context, turn Turn, decider Decider) Outcome {
	outcome, err := r.cascade.route(ctx, turn, decider)
	if err != nil {
		return Outcome{
			SelectedModelIDs: []string{},
			Error:            err.Error(),
			Calls:            outcome.Calls,
		}
	}
	return outcome
}

// Forget drops everything retained for a session — its composition state and its
// latch assignment — so an ended session stops being routed by a stale target.
// Honouring a session-final signal keeps a long-lived gateway bounded.
//
// This is a deliberate divergence from upstream, which clears the composition
// state on a session-final signal but leaves the latch assignment to its own
// bound. A Forget that does not release the target is a footgun, and the latch is
// the retained state that actually changes routing.
func (r *Router) Forget(session string) {
	r.cascade.Forget(session)
	if r.affinity != nil {
		r.affinity.Forget(session)
	}
}

// validatePool mirrors switchyard-runner: the selected target is resolved against
// "any", so a group outside it can never be served.
func validatePool(models Models) error {
	any := models[categoryAny]
	if len(any) == 0 {
		return fmt.Errorf(
			"models.any must list every routable target: the selected target is resolved against it")
	}
	for name, ids := range models {
		if name == categoryAny || name == categoryJudge {
			continue
		}
		for _, id := range ids {
			if !contains(any, id) {
				return fmt.Errorf("models.%s lists %q, which must also appear in models.any", name, id)
			}
		}
	}
	return nil
}

func contains(list []string, value string) bool {
	for _, item := range list {
		if item == value {
			return true
		}
	}
	return false
}

func mustJSON(value any) json.RawMessage {
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil
	}
	return encoded
}
