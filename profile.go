// Package libsygo is a pure-Go implementation of Switchyard's libsy: the
// algorithms that decide which model a request should route to.
//
// It performs no I/O of its own. The judge call is supplied by the host through
// [Decider], so a dataplane that owns its own egress — Envoy, or anything with
// its own HTTP stack — keeps that ownership: TLS, credentials, retries and
// timeouts all stay host-side. A host with no such stack can pass a [jev.Client].
//
// # Scope
//
// Implemented: the capability classifier over a decision (System One / JEV)
// judge, with the composition it needs — the cascade, the per-session latch, the
// cutoff, fail-open, candidate ordering, and evidence.
//
// Not implemented, and refused at construction rather than approximated: the
// signal router (tool signals), escalation, advisor gate, plan/execute,
// composite, sub-agent, custom classifiers, and LLM-text judges.
//
// # Conformance
//
// Verified turn-for-turn against upstream libsy by the conformance harness in
// github.com/dio/switchyard-capi: same corpus, same recorded golden, diffed on
// tier, ordered candidates, evidence, and the rewritten request. Upstream's tuned
// constants are not derivable, which is why the harness exists and why every
// behaviour here is a ported one rather than an invented one.
package libsygo

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"sort"

	"github.com/dio/libsygo/jev"
)

//go:embed prompts/relative_advantage.json
var relativeAdvantagePrompt []byte

// AlgorithmName is the telemetry name this package reports, matching libsy's
// `Algorithm::name` so decisions are attributable across both implementations.
const AlgorithmName = "llm_task_classifier"

// Capability-judge answer descriptions, from DecisionClassifier::new.
const (
	advantageDescription   = "The capable candidate succeeds AND the efficient candidate fails."
	noAdvantageDescription = "The efficient candidate succeeds OR the capable candidate fails, including shared failure."
)

// Routable categories. `judge` is never a destination.
const (
	categoryCapable   = "capable"
	categoryEfficient = "efficient"
	categoryJudge     = "judge"
	categoryAny       = "any"
)

// Profile is one routing setup: the algorithm, its configuration, and the judge
// it consults. The model pool is passed separately because it is per request.
//
// A profile is materialized upstream and arrives whole; this package does no
// inheritance or cascading.
type Profile struct {
	// ID and Revision identify the materialization. They are not interpreted
	// here — use them as cache keys, so a revision bump builds a fresh Router.
	ID       string `json:"id,omitempty"`
	Revision int    `json:"revision,omitempty"`

	Algorithm string `json:"algorithm"`
	Mode      string `json:"mode"`
	Judge     *Judge `json:"judge"`

	// FailOpen routes an unusable verdict to the capable tier instead of
	// failing the request. Defaults to true. See Route.
	FailOpen *bool `json:"fail_open"`

	// SessionAffinity retains the first target chosen for a session and reuses
	// it, without calling the judge again.
	SessionAffinity bool `json:"session_affinity"`
}

// Judge describes how the decider obtains a verdict.
type Judge struct {
	// Type is the judge family. Only "decision" (System One / JEV) is
	// implemented; an LLM-text judge is not ported.
	Type string `json:"type"`

	// Binding names a host-owned judge endpoint and credential. This package
	// never sees the URL or the key — that is the host's to resolve.
	Binding string `json:"binding,omitempty"`

	// Cutoff routes to the capable tier only when the advantage score is
	// strictly greater than this value.
	Cutoff float64 `json:"cutoff"`

	// Candidates labels the compared models so evidence and judgements can talk
	// about them without brand names. Every compared target needs a label.
	Candidates map[string]string `json:"candidates"`

	// Evidence is passed to the judge unchanged.
	Evidence json.RawMessage `json:"evidence"`

	// Instructions replaces the packaged judge prompt when set.
	Instructions json.RawMessage `json:"instructions"`
}

// Models is the routable pool, per category, best-first.
//
// "any" must list every routable target, because the selected target is resolved
// against it; New rejects a pool that violates this rather than failing later.
// For the capability classifier the reference order is [efficient..., capable...].
type Models map[string][]string

// Turn is one request to route.
type Turn struct {
	// Prompt is the user text the judge evaluates.
	Prompt string

	// Session keys the latch. It MUST be tenant-namespaced by the caller:
	// two tenants sharing a session id would share retained routing state.
	Session string
}

// Decider performs one judge call. The host supplies it, which is what keeps
// egress out of this package.
type Decider interface {
	Call(ctx context.Context, request jev.DecisionRequest) (jev.DecisionResponse, error)
}

// fail point reasons reported in evidence.
const (
	reasonInvalidVerdict = "invalid_verdict"
	reasonCallError      = "call_error"
	reasonRetained       = "retained"
)

// buildJudgeRequest assembles the context and question that libsy's
// DecisionClassifier sends to the judge.
func (r *Router) buildJudgeRequest(prompt string) (jev.DecisionRequest, error) {
	comparison, err := r.comparison()
	if err != nil {
		return jev.DecisionRequest{}, err
	}

	context, err := json.Marshal(map[string]any{
		"task": []map[string]any{{
			"role":    "user",
			"content": []map[string]any{{"type": "text", "text": prompt}},
		}},
		"candidates": r.candidateLabels(),
		"comparison": comparison,
		"evidence":   r.judge.Evidence,
	})
	if err != nil {
		return jev.DecisionRequest{}, err
	}

	options, err := json.Marshal(map[string]any{
		"options": []map[string]any{
			{"id": "advantage", "description": advantageDescription},
			{"id": "no_advantage", "description": noAdvantageDescription},
		},
	})
	if err != nil {
		return jev.DecisionRequest{}, err
	}

	model := r.first(categoryJudge)
	return jev.DecisionRequest{
		Model:   &model,
		Context: context,
		Questions: map[string]jev.DecisionQuestion{
			"route": {Instructions: r.instructions, Kind: jev.Tagged{Type: "choice", Data: options}},
		},
	}, nil
}

// candidateLabels are the labels in the order libsy's BTreeMap yields them.
func (r *Router) candidateLabels() []string {
	labels := make([]string, 0, len(r.judge.Candidates))
	for label := range r.judge.Candidates {
		labels = append(labels, label)
	}
	sort.Strings(labels)
	return labels
}

// comparison names the labels of the runtime capable and efficient targets.
func (r *Router) comparison() (map[string]string, error) {
	capable, err := r.labelFor(r.first(categoryCapable))
	if err != nil {
		return nil, err
	}
	efficient, err := r.labelFor(r.first(categoryEfficient))
	if err != nil {
		return nil, err
	}
	return map[string]string{"capable": capable, "efficient": efficient}, nil
}

func (r *Router) labelFor(model string) (string, error) {
	for _, label := range r.candidateLabels() {
		if r.judge.Candidates[label] == model {
			return label, nil
		}
	}
	return "", fmt.Errorf("decision candidate is missing for target %q", model)
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
