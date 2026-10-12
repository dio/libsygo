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
//
//   - `stage_router`: tool signals derived from the conversation — the tool vocabulary,
//     the shell rules, severity bands and failure identity, the counts and their
//     windows, error streaks, test detection and compaction — scored onto the capable
//     and efficient tiers, with the picker's default, the three override reasons, the
//     confidence threshold and the capable hold.
//
//   - The composition both rest on — the cascade, the per-session latch, the cutoff,
//     fail-open, candidate ordering, and evidence.
//
//   - Request normalization for all three wire formats: OpenAI chat, Anthropic
//     Messages, and OpenAI Responses.
//
//   - `passthrough` selects the first routable target and offers the rest as fallbacks.
//
//   - `noop` answers with a synthetic response, leaving the request's own model in place.
//     It is the only algorithm here that responds rather than selects, which is what
//     [Outcome.Responded] reports.
//
// Not implemented, and refused at construction: escalation, advisor gate, plan/execute,
// composite, sub-agent, custom classifiers, and LLM-text judges.
//
// `random` is refused for a different reason, and one worth stating: a seeded pick is the
// upstream RNG's *stream* — ChaCha12, seeded and sampled by a specific algorithm — rather
// than a distribution this could reproduce. An unseeded pick is a distribution this could
// reproduce, but it cannot be recorded deterministically, so there would be no gate for
// it. Refusing is the honest option: a seeded profile that silently picked differently
// would look like it worked.
//
// Declared gaps inside `stage_router`, each unverified rather than known wrong:
//
//   - The capable hold is keyed per session. Upstream refines it per agent id, which
//     the ported request does not carry.
//   - A tool namespace's *extension* record is not kept, so a namespaced call's bare
//     name is not resolved when the signals look one up. The qualification itself — a
//     call spelled the way its definition was — is applied.
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

	// OnlyOnWrongSignalEscalation restricts the escalation note to escalations the
	// signals made. Nil means yes, which is the safe default: an ungated note can tell
	// the capable model the efficient one was stalling when it was not.
	OnlyOnWrongSignalEscalation *bool `json:"only_on_wrong_signal_escalation"`

	// ToolSemantics adds tool names to the built-in vocabulary. Additive: a built-in
	// name cannot be reclassified, because the recorded signals were measured against
	// the built-ins.
	ToolSemantics *ToolSemantics `json:"tool_semantics"`
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
