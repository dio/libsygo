package libsygo

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"
)

// Composite stacks a judge over a stage router.
//
// The judge is not a third opinion. It runs as a *processor*: it sets the tier the stage
// router falls open to and picks no target itself, so the signals decide the turns they
// can and the judge decides the rest — rather than two routers disagreeing about the same
// turn.
//
// The verdict is retained per routing identity and replayed into state on later requests,
// so a session the judge has already placed does not get judged again by default.

// fallOpenKey is the state key a default-tier override is recorded under.
const fallOpenKey = "fall_open"

// setFallOpen overrides the tier undecided turns fall open to.
//
// Held in session state, so it survives later requests only when they carry a session —
// which is what makes a judged session sticky without retaining it anywhere else.
//
// The tier's *label* is recorded rather than its category name, because that is what
// upstream records and it is what keeps an override meaningful if the categories are ever
// renamed.
func setFallOpen(state *State, tier Tier) {
	state.Extra[fallOpenKey] = tier.tierLabel()
}

func fallOpenTier(state *State) (Tier, bool) {
	label, ok := state.Extra[fallOpenKey].(string)
	if !ok {
		return "", false
	}
	return tierFromLabel(label)
}

// isUserTurn reports a human turn rather than a tool continuation.
//
// Role alone cannot tell them apart: providers carry tool results as user-role messages,
// so Anthropic's transcript is full of user-role turns nobody said. A message is a human
// turn unless *all* of its content is tool results — which also means an empty user
// message is not one, since it says nothing.
func isUserTurn(message Message) bool {
	if message.Role != RoleUser {
		return false
	}
	onlyResults := true
	for _, block := range message.Content {
		if block.Kind != ContentToolResult {
			onlyResults = false
			break
		}
	}
	return !onlyResults
}

// hasNewUserTurn reports whether a request ends on a human turn, which is what the
// user-turn trigger keys on: a request ending in tool results is a continuation of the
// turn already judged.
func hasNewUserTurn(messages []Message) bool {
	return len(messages) > 0 && isUserTurn(messages[len(messages)-1])
}

// tierSetter runs a judge ahead of the cascade and records its verdict as the tier the
// stage router falls open to.
type tierSetter struct {
	judge               Classifier
	trigger             ClassifyTrigger
	messageHashFallback bool

	mu    sync.Mutex
	tiers map[string]Tier
}

// Process implements Processor, on the request side so the tier is set before the cascade
// reads it.
func (t *tierSetter) Process(state *State, event Event) error {
	if event.Kind != EventRequest || event.Turn == nil || event.Driver == nil {
		return nil
	}

	identity := t.retentionKey(event.Turn)
	if t.isDue(identity, event.Turn) {
		classification, err := t.judge.Classify(event.Context, state, event.Turn, event.Driver)
		if err != nil {
			return err
		}
		winner, err := classification.Argmax(false)
		if err != nil {
			return err
		}
		if winner != nil {
			if tier, ok := tierForCategory(winner.Category); ok {
				setFallOpen(state, tier)
				if identity != "" {
					t.retain(identity, tier)
				}
				return nil
			}
		}
	}

	// Not this request's turn, or the judge had no verdict: the last tier stands rather
	// than dropping back to the picker's default, which would undo a judgement on every
	// turn the trigger skips.
	if tier, ok := t.retained(identity); ok {
		setFallOpen(state, tier)
		event.Driver.SetEvidenceIfEmpty(map[string]any{"source": "retained"})
	}
	return nil
}

// isDue reports whether this request is one the trigger judges.
//
// Two requests for one identity can both pass this and both judge, since a judge call sits
// between here and [tierSetter.retain]. The later verdict wins.
func (t *tierSetter) isDue(identity string, turn *Turn) bool {
	switch t.trigger {
	case UserTurn:
		return hasNewUserTurn(turn.Request.Messages)
	case NewSession:
		// Unkeyed requests cannot be told apart, so every one is treated as a new session.
		if identity == "" {
			return true
		}
		t.mu.Lock()
		defer t.mu.Unlock()
		_, held := t.tiers[identity]
		return !held
	}
	return true
}

// retain records a verdict. A user-turn trigger re-decides, so it overwrites; a session
// keeps its first verdict, which is how the affinity latch behaves too.
func (t *tierSetter) retain(identity string, tier Tier) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.trigger != UserTurn {
		if _, held := t.tiers[identity]; held {
			return
		}
	}
	if len(t.tiers) >= maxSessions {
		// Bounded like every other retained map here: a long-lived gateway must not grow
		// without limit on identities it will never see again.
		for key := range t.tiers {
			delete(t.tiers, key)
			break
		}
	}
	t.tiers[identity] = tier
}

func (t *tierSetter) retained(identity string) (Tier, bool) {
	if identity == "" {
		return "", false
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	tier, held := t.tiers[identity]
	return tier, held
}

// retentionKey is the identity a verdict is retained under.
//
// A session keys it directly. Without one, and only when the fallback is enabled, the
// opening task keys it — so a caller that never sends a session id still gets one task
// held to one tier instead of every request judging afresh.
//
// The hash only ever groups requests within one process, so any stable hash of the opening
// user text does upstream's job; what matters is that two requests carrying the same task
// land on the same key, not which hash produced it.
func (t *tierSetter) retentionKey(turn *Turn) string {
	if turn.Session != "" {
		return turn.Session
	}
	if !t.messageHashFallback {
		return ""
	}
	for _, message := range turn.Request.Messages {
		if message.Role != RoleUser {
			continue
		}
		text, _ := message.TextContent("")
		sum := sha256.Sum256([]byte(text))
		return hex.EncodeToString(sum[:8])
	}
	return ""
}

// tierForCategory maps a classifier's category onto a tier, reporting whether it named one
// at all: a verdict for some other category is not a tier this router can fall open to.
func tierForCategory(category string) (Tier, bool) {
	switch Tier(category) {
	case TierCapable:
		return TierCapable, true
	case TierEfficient:
		return TierEfficient, true
	}
	return "", false
}

// parseTrigger reads a profile's classify trigger. An absent one judges every request,
// which is the standalone route's historical behaviour.
func parseTrigger(value string) (ClassifyTrigger, error) {
	switch ClassifyTrigger(value) {
	case EveryRequest, UserTurn, NewSession:
		return ClassifyTrigger(value), nil
	case "":
		return EveryRequest, nil
	}
	return "", fmt.Errorf("unknown classify_trigger %q", value)
}

// newCompositeRouter assembles a judge over a stage router.
func newCompositeRouter(profile Profile, models Models) (*Router, error) {
	trigger, err := parseTrigger(profile.ClassifyTrigger)
	if err != nil {
		return nil, err
	}
	if trigger == EveryRequest {
		// Every request judging would make the retention meaningless, and the signals
		// would never get to decide a turn the judge had already placed.
		return nil, fmt.Errorf("composite: classify_trigger must be %q or %q",
			UserTurn, NewSession)
	}

	// Only the judge's classifier face is used, so the standalone route's pairing rules
	// must not run: their trigger belongs to this router, not to the judge.
	judge, err := newCapabilityClassifier(profile, models)
	if err != nil {
		return nil, err
	}

	// The stage route is built without a cascade fallback: this profile's judge is the
	// tier setter's, and wiring it in twice would put it in front of its own verdict.
	cascade, err := buildStageRoute(AlgorithmComposite, profile, models, nil)
	if err != nil {
		return nil, err
	}
	cascade.WithProcessor(&tierSetter{
		judge:               judge,
		trigger:             trigger,
		messageHashFallback: profile.MessageHashFallback,
		tiers:               map[string]Tier{},
	})
	return &Router{cascade: cascade}, nil
}
