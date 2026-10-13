package libsygo

import "context"

// EventKind names what a processor is observing.
type EventKind int

const (
	// EventRequest is the inbound turn, before the cascade runs.
	EventRequest EventKind = iota
	// EventDecision is a routing decision, paired with the turn that produced it.
	EventDecision
	// EventModelResponse is a buffered response received from a model.
	EventModelResponse
)

// Event is what a processor observes as the algorithm runs.
//
// Request-bearing events carry the driver and the context, because a processor may consult
// a classifier: the composite tier setter runs a judge before the cascade decides, and a
// processor with no way to reach the host could not.
//
// Request-bearing events carry a rewritable turn, so a processor may add
// instructions bound to the routing outcome — a tier system prompt, a handoff
// note — and the edit reaches the model.
type Event struct {
	Kind EventKind

	// Turn is set on EventRequest and EventDecision.
	Turn *Turn

	// Context and Driver are set on EventRequest, and are what a processor uses to
	// consult a classifier.
	Context context.Context
	Driver  *Driver

	// SelectedModel and Category are set on EventDecision.
	SelectedModel string
	Category      string
}

// Processor accumulates facts into session state as the algorithm runs.
//
// Processors run twice per turn: once with the request, before the cascade, and
// once with the decision, so a stateful processor can bind the selection.
type Processor interface {
	Process(state *State, event Event) error
}
