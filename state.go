package libsygo

// State is the routing state shared by every processor and classifier in a
// composition, retained per session.
type State struct {
	// TurnCount is how many turns this session has routed.
	TurnCount uint32

	// Extra holds algorithm-specific facts keyed by stable internal names.
	//
	// Values are upstream's `StateValue` variants: string, uint32, int32, or
	// float32. Keeping the value untyped rather than a tagged union is the one
	// place this port deviates from upstream's shape; the alternative was a
	// four-field struct where every user reads one of them.
	//
	// Upstream's tool signals land here with the signal router, which is not
	// ported yet.
	Extra map[string]any
}

func newState() *State {
	return &State{Extra: map[string]any{}}
}
