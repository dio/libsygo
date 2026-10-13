package libsygo

import "strings"

// The harness headers a host supplies out of band.
//
// They are not part of the conversation, which is why they travel separately: which session
// a request belongs to, which agent is making it, and whether the harness is signalling
// anything about either. A host that sends none of them still works — everything here is
// optional, and silence is distinguished from a negative answer.
const (
	headerSessionID     = "x-switchyard-session-id"
	headerAgentID       = "x-switchyard-agent-id"
	headerParentAgentID = "x-switchyard-parent-agent-id"
	headerIsSubagent    = "x-switchyard-is-subagent"
	headerAgentKind     = "x-switchyard-agent-kind"
	headerAgentRole     = "x-switchyard-agent-role"
	headerTaskID        = "x-switchyard-task-id"
	headerTaskKind      = "x-switchyard-task-kind"
	headerTurnID        = "x-switchyard-turn-id"
	headerSessionFinal  = "x-switchyard-session-final"
	headerRequestID     = "x-switchyard-request-id"
)

// Metadata is the request context that travels beside the conversation.
//
// A faithful port of the upstream struct, including the fields this package does not read
// yet: it is the host contract, and a host handing over a request should not have to know
// which of its fields the router currently consults.
type Metadata struct {
	// SessionID is a stable id for a multi-request conversation.
	SessionID string

	// AgentID is the agent making this request.
	AgentID string

	// ParentAgentID is set when the request comes from a child agent.
	ParentAgentID string

	// IsSubagent says the harness identified the request as a child's. It is *not* what
	// routing keys on: see IsDelegatedWork.
	IsSubagent bool

	// IsDelegatedWork says the request carries work delegated to a sub-agent, and is what
	// the sub-agent route keys on. Computed from raw harness signals only, deliberately
	// independent of AgentKind — an operator may label an agent without delegating work to
	// it, and routing on that label would send ordinary traffic to a worker model.
	IsDelegatedWork bool

	// AgentKind, AgentRole, TaskID, TaskKind and TurnID are harness vocabulary the router
	// passes through without interpreting.
	AgentKind string
	AgentRole string
	TaskID    string
	TaskKind  string
	TurnID    string

	// SessionFinal says the harness is ending the session, so a host may evict per-session
	// state. Nil when the harness said nothing, which is not the same as saying no.
	SessionFinal *bool

	// CorrelationID joins this request with the host's own telemetry.
	CorrelationID string
}

// MetadataFromHeaders reads what a harness supplied, case-insensitively.
//
// An unknown or unparseable header is absent rather than an error: metadata is advisory,
// and a host that sends something this build does not understand should still be routed.
func MetadataFromHeaders(headers map[string]string) Metadata {
	lowered := make(map[string]string, len(headers))
	for name, value := range headers {
		lowered[strings.ToLower(name)] = value
	}
	value := func(name string) string {
		return strings.TrimSpace(lowered[name])
	}

	metadata := Metadata{
		SessionID:     value(headerSessionID),
		AgentID:       value(headerAgentID),
		ParentAgentID: value(headerParentAgentID),
		AgentKind:     value(headerAgentKind),
		AgentRole:     value(headerAgentRole),
		TaskID:        value(headerTaskID),
		TaskKind:      value(headerTaskKind),
		TurnID:        value(headerTurnID),
		CorrelationID: value(headerRequestID),
	}
	if subagent, ok := parseBoolHeader(value(headerIsSubagent)); ok {
		metadata.IsSubagent = subagent
		// An explicit harness signal is the one case where the two flags agree. The
		// harness-based detection a real deployment needs is the sub-agent route's, not
		// this struct's.
		metadata.IsDelegatedWork = subagent
	}
	if final, ok := parseBoolHeader(value(headerSessionFinal)); ok {
		metadata.SessionFinal = &final
	}
	return metadata
}

// parseBoolHeader reads the spellings a boolean header may take, reporting whether the
// header said anything at all.
func parseBoolHeader(value string) (bool, bool) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "true", "1", "yes":
		return true, true
	case "false", "0", "no":
		return false, true
	}
	return false, false
}
