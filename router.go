package libsygo

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/dio/libsygo/jev"
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

	// Latched is true when a retained session target decided this turn and no
	// judge call was made.
	Latched bool

	// Error is set when the turn failed rather than mis-routed: a judge failure
	// with FailOpen disabled. The caller should surface it, not route.
	Error string

	// Calls is how many judge calls this turn made (0 when latched).
	Calls int
}

// Router is a built profile. It holds the per-session latch, so one Router serves
// one tenant's profile: cache by (tenant, profile ID, revision) and namespace
// session ids per tenant. A Router is safe for concurrent use.
//
// The judge [Decider] is supplied per Route call, not at construction, because it
// is usually tied to a single request (the host's callout for that stream). The
// Router itself is the long-lived, shared part.
type Router struct {
	judge           *Judge
	instructions    json.RawMessage
	failOpen        bool
	sessionAffinity bool
	models          Models

	latched map[string]latch
}

type latch struct {
	model    string
	category string
}

// New builds a router, rejecting anything outside tier 0/1 rather than guessing.
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

	return &Router{
		judge:           profile.Judge,
		instructions:    instructions,
		failOpen:        failOpen,
		sessionAffinity: profile.SessionAffinity,
		models:          models,
		latched:         map[string]latch{},
	}, nil
}

// Route decides one turn, making at most one judge call through decider.
//
// A latched session returns without a judge call. Otherwise decider is consulted;
// a verdict that is missing, malformed, or the result of a failed call falls back
// per FailOpen. A nil decider is treated as a failed call rather than a panic.
func (r *Router) Route(ctx context.Context, turn Turn, decider Decider) Outcome {
	if r.sessionAffinity && turn.Session != "" {
		if held, ok := r.latched[turn.Session]; ok {
			outcome := r.decide(held.model, held.category)
			outcome.Latched = true
			outcome.Evidence = json.RawMessage(`{"source":"retained"}`)
			return outcome
		}
	}

	if decider == nil {
		return r.fallback(reasonCallError, "no judge decider was supplied for this call")
	}

	request, err := r.buildJudgeRequest(turn.Prompt)
	if err != nil {
		return r.fallback(reasonCallError, err.Error())
	}
	response, err := decider.Call(ctx, request)
	if err != nil {
		return r.fallback(reasonCallError, err.Error())
	}

	score, ok := advantage(response)
	if !ok {
		return r.fallback(reasonInvalidVerdict, "")
	}

	model, category := r.tierFor(score)
	if r.sessionAffinity && turn.Session != "" {
		r.latched[turn.Session] = latch{model: model, category: category}
	}
	outcome := r.decide(model, category)
	outcome.Calls = 1
	outcome.Evidence = mustJSON(map[string]any{
		"source":    "decision_classifier",
		"verdict":   "relative_advantage",
		"score":     score,
		"threshold": r.judge.Cutoff,
	})
	return outcome
}

// Forget drops a session's retained target, so an ended session does not
// accumulate. Honouring a session-final signal from the caller keeps a
// long-lived gateway bounded.
func (r *Router) Forget(session string) {
	delete(r.latched, session)
}

// tierFor applies the cutoff: strictly greater selects capable, so a score
// exactly at the cutoff does not.
func (r *Router) tierFor(score float64) (model, category string) {
	if score > r.judge.Cutoff {
		return r.first(categoryCapable), categoryCapable
	}
	return r.first(categoryEfficient), categoryEfficient
}

// fallback handles a judge that produced no usable verdict.
func (r *Router) fallback(reason, message string) Outcome {
	if !r.failOpen {
		return Outcome{SelectedModelIDs: []string{}, Error: message, Calls: 1}
	}
	outcome := r.decide(r.first(categoryCapable), categoryCapable)
	outcome.Calls = 1
	outcome.Evidence = mustJSON(map[string]any{"source": "fail_open", "reason_code": reason})
	return outcome
}

// decide builds an outcome with libsy's fallback ordering.
func (r *Router) decide(model, category string) Outcome {
	return Outcome{
		SelectedModelIDs: r.ordered(model, category),
		Model:            model,
		Category:         category,
	}
}

// ordered reproduces libsy's fallback order: the pick, then the rest of the
// category it was drawn from, then every other runtime target.
func (r *Router) ordered(selected, category string) []string {
	out := []string{selected}
	add := func(list []string) {
		for _, model := range list {
			if model == selected || contains(out, model) {
				continue
			}
			out = append(out, model)
		}
	}
	add(r.models[category])
	add(r.models[categoryAny])
	return out
}

func (r *Router) first(category string) string {
	if list := r.models[category]; len(list) > 0 {
		return list[0]
	}
	return ""
}

// advantage reads p(advantage) from the judge's verdict, mirroring
// DecisionClassifier: only a Choice answer with a probability in [0,1] counts.
func advantage(response jev.DecisionResponse) (float64, bool) {
	answer, ok := response.Answers["route"]
	if !ok || answer.Value.Type != "choice" {
		return 0, false
	}
	var data struct {
		Probabilities map[string]float64 `json:"probabilities"`
	}
	if err := json.Unmarshal(answer.Value.Data, &data); err != nil {
		return 0, false
	}
	score, ok := data.Probabilities["advantage"]
	if !ok || score < 0 || score > 1 {
		return 0, false
	}
	return score, true
}

func mustJSON(value any) json.RawMessage {
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil
	}
	return encoded
}
