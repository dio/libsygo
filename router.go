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
	// on a latched session, `{source: "fall_open", reason_code}` on a fallback.
	Evidence json.RawMessage

	// Request is the request as routing left it: the selected model is stamped
	// into Model, and any processor that binds the decision — a tier system
	// prompt, a handoff note — has already spliced itself in. This is what the
	// model would receive, and it is not the request that was passed in.
	Request LlmRequest

	// Algorithm names what decided, for telemetry that has to attribute a
	// decision to the route that made it.
	Algorithm string

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

// Algorithms this package builds.
const (
	// AlgorithmCapability is the capability route: ask a judge whether the cheap
	// tier can handle the task.
	AlgorithmCapability = "llm_classifier"
	// AlgorithmStage is the signal router: read tool activity and pick a tier
	// without asking anyone.
	AlgorithmStage = "stage_router"
)

// New builds a router, refusing anything not implemented rather than
// approximating it.
func New(profile Profile, models Models) (*Router, error) {
	switch profile.Algorithm {
	case AlgorithmCapability:
		return newCapabilityRouter(profile, models)
	case AlgorithmStage:
		return newStageRouter(profile, models)
	default:
		return nil, fmt.Errorf("profile algorithm %q is not implemented: %q and %q are",
			profile.Algorithm, AlgorithmCapability, AlgorithmStage)
	}
}

// newCapabilityRouter assembles the capability route.
func newCapabilityRouter(profile Profile, models Models) (*Router, error) {
	if profile.Mode != "capability" {
		return nil, fmt.Errorf("llm_classifier mode %q is not implemented: only capability is", profile.Mode)
	}
	capability, err := newCapabilityClassifier(profile, models)
	if err != nil {
		return nil, err
	}

	cascade := NewFallThrough(AlgorithmName, models)
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

// newCapabilityClassifier builds the decision-judge classifier a profile configures.
func newCapabilityClassifier(profile Profile, models Models) (*capabilityClassifier, error) {
	if profile.Judge == nil || profile.Judge.Type != "decision" {
		return nil, fmt.Errorf("judge is not implemented: only the decision (System One) judge is")
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
	return &capabilityClassifier{
		judge:        profile.Judge,
		instructions: instructions,
		failOpen:     failOpen,
	}, nil
}

// newStageRouter assembles the signal router.
//
// The cascade is the signals, then the signals' own classifier, then the capability
// judge when one is configured, and finally the picker's tier closing it — so a turn
// the signals cannot decide gets the same verdict the capability route would give it,
// and a turn nothing can decide is still routed.
func newStageRouter(profile Profile, models Models) (*Router, error) {
	if profile.ConfidenceThreshold < 0 || profile.ConfidenceThreshold > 1 {
		return nil, fmt.Errorf("confidence_threshold must be between 0 and 1, got %v",
			profile.ConfidenceThreshold)
	}
	defaultTier, err := profile.Picker.defaultTier()
	if err != nil {
		return nil, err
	}
	if err := validatePool(models); err != nil {
		return nil, err
	}

	recentWindow := defaultRecentWindow
	if profile.RecentWindow != nil {
		recentWindow = *profile.RecentWindow
	}

	semantics := ToolSemantics{}
	if profile.ToolSemantics != nil {
		semantics = *profile.ToolSemantics
		if err := semantics.Validate(); err != nil {
			return nil, err
		}
	}
	var holdTurns uint32 = defaultCapableHoldTurns
	if profile.CapableHoldTurns != nil {
		holdTurns = *profile.CapableHoldTurns
	}

	cascade := NewFallThrough(AlgorithmStage, models).
		WithProcessor(toolSignalProcessor{recentWindow: recentWindow, semantics: semantics}).
		// The signals decide the turns they are confident about; the rest of the
		// cascade catches the ones they are not.
		WithClassifier(&stageClassifier{
			mode:                profile.Picker,
			confidenceThreshold: profile.ConfidenceThreshold,
			capableHoldTurns:    holdTurns,
		})

	if profile.Judge != nil {
		fallback, err := newCapabilityClassifier(profile, models)
		if err != nil {
			return nil, err
		}
		cascade.WithClassifier(sourceStamp{inner: fallback, source: SourceLlmClassifier})
	}

	cascade.WithClassifier(sourceStamp{
		inner:  fallOpenClassifier{defaultTier: defaultTier},
		source: SourceFallOpen,
	})

	if profile.CapableSystemPrompt != "" || profile.EfficientSystemPrompt != "" {
		cascade.WithProcessor(tierPromptProcessor{
			capable:   profile.CapableSystemPrompt,
			efficient: profile.EfficientSystemPrompt,
		})
	}

	return &Router{cascade: cascade}, nil
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
