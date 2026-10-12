// Package libsygo is a pure-Go implementation of Switchyard's libsy: the
// algorithms that decide which model a request should route to.
//
// It performs no I/O of its own. The judge call is supplied by the host through
// [Decider], so a dataplane that owns its own egress — Envoy, or anything with
// its own HTTP stack — keeps that ownership: TLS, credentials, retries and
// timeouts all stay host-side. A host with no such stack can pass a [jev.Client].
//
// # Composition
//
// A profile is a [FallThrough]: a processor chain that accumulates facts into
// per-session state, then a classifier cascade in which the first classifier to
// score decides. The capability classifier is one member of that cascade and the
// session latch is another, which is why those pieces are exported — the
// remaining algorithm families are new classifiers and processors on the same
// machinery, not new routers.
//
// # Scope
//
// Implemented:
//
//   - `llm_classifier` in `capability` mode: a judge is asked whether the cheap tier
//     can handle the task, over the decision (System One / JEV) judge.
//   - `stage_router` as far as its fall-open path: the picker's default tier, the
//     `fall_open` evidence, and the per-tier system prompts.
//   - The composition both rest on — the cascade, the per-session latch, the cutoff,
//     fail-open, candidate ordering, and evidence — and request normalization.
//
// Not implemented, and refused rather than approximated: **the stage router's tool
// signals**. That one is worth stating precisely, because the refusal is the feature:
// a turn carrying tool traffic is rejected rather than scored as signal-free. "No
// signals" is a legitimate state that falls open to the picker's tier, so a port that
// could not derive signals would reach it on every coding session and route all of
// them to the default tier while looking like it worked.
//
// Also not implemented, and refused at construction: escalation, advisor gate,
// plan/execute, composite, sub-agent, custom classifiers, LLM-text judges, and the
// noop/random/passthrough algorithms.
//
// The tool vocabulary is ported and tested — [ClassifyToolCall], [IsRetrievalTool] and
// the shell rules they rest on, plus [SeverityOf] and the failure-identity rules
// beside it — but nothing walks a conversation to derive signals from them yet, so
// they feed no decision. They are exported because a host that already parses tool
// calls should be able to ask the same questions this package will.
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

	// --- stage_router ---

	// Picker is the tier an undecided turn falls open to: "capable_first" or
	// "efficient_first".
	Picker PickerMode `json:"picker"`

	// ConfidenceThreshold is how much corroboration a signal-driven pick needs, in
	// [0, 1]. Signals that do not clear it abstain and the turn falls through.
	ConfidenceThreshold float64 `json:"confidence_threshold"`

	// RecentWindow is how many trailing tool results the signals are computed over.
	// Nil uses the built-in default; an explicit zero is not the same as absent.
	RecentWindow *int `json:"recent_window"`

	// CapableHoldTurns keeps a session on the capable tier for this many requests
	// after an escalation. Nil uses the built-in default; zero disables the hold.
	CapableHoldTurns *uint32 `json:"capable_hold_turns"`

	// CapableSystemPrompt and EfficientSystemPrompt are handed to the model a turn
	// routes to, ahead of the client's own instructions.
	CapableSystemPrompt   string `json:"capable_system_prompt"`
	EfficientSystemPrompt string `json:"efficient_system_prompt"`

	// EscalationNote and DeescalationNote are spliced into the request when the
	// signals move a session between tiers.
	EscalationNote   string `json:"escalation_note"`
	DeescalationNote string `json:"deescalation_note"`
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
	// Request is the normalized request the algorithms read. Decode a provider
	// body into it with [DecodeOpenAIChat], or build one with [TextRequest] when
	// all a caller has is a prompt.
	Request LlmRequest

	// Session keys the latch. It MUST be tenant-namespaced by the caller:
	// two tenants sharing a session id would share retained routing state.
	Session string
}

// Decider performs one judge call. The host supplies it, which is what keeps
// egress out of this package.
type Decider interface {
	Call(ctx context.Context, request jev.DecisionRequest) (jev.DecisionResponse, error)
}
