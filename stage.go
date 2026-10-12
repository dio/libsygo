package libsygo

import (
	"context"
	"errors"
	"fmt"
)

// defaultRecentWindow is how many trailing tool results the signals are computed over
// when a profile does not say. Upstream's DEFAULT_RECENT_WINDOW.
const defaultRecentWindow = 3

// Tier is one of the two routing tiers a stage router chooses between.
type Tier string

// The routing tiers. The values are the model-pool category names, so a tier
// doubles as a category key.
const (
	TierCapable   Tier = "capable"
	TierEfficient Tier = "efficient"
)

// tierLabel is the stable, target-independent name a tier is recorded under.
//
// Deliberately not the category name: the label is what a deployment sees in stats
// and what a default-tier override is encoded as, so it has to stay put if the
// categories are ever renamed. The capability route reports the same strings, so a
// deployment running both sees one vocabulary.
func (t Tier) tierLabel() string {
	if t == TierCapable {
		return "strong"
	}
	return "weak"
}

func tierFromLabel(label string) (Tier, bool) {
	switch label {
	case "strong":
		return TierCapable, true
	case "weak":
		return TierEfficient, true
	}
	return "", false
}

// PickerMode is which tier a turn falls open to when the signals are not confident.
type PickerMode string

// The picker modes.
const (
	// PickerCapableFirst defaults to capable unless the signals confidently pick
	// efficient. The cautious default: an undecided turn gets the stronger model.
	PickerCapableFirst PickerMode = "capable_first"
	// PickerEfficientFirst defaults to efficient unless the signals confidently
	// pick capable. The cheap default.
	PickerEfficientFirst PickerMode = "efficient_first"
)

// defaultTier is where an undecided turn lands.
func (p PickerMode) defaultTier() (Tier, error) {
	switch p {
	case PickerCapableFirst:
		return TierCapable, nil
	case PickerEfficientFirst:
		return TierEfficient, nil
	}
	return "", fmt.Errorf("stage_router picker %q is not implemented: want %q or %q",
		p, PickerCapableFirst, PickerEfficientFirst)
}

// DecisionSource names what decided a turn. Each is stamped by the component that
// knows, so a turn's label names whoever actually decided it.
type DecisionSource string

// The decision sources.
const (
	// SourceDimensions is the signal scorer crossing its confidence threshold.
	SourceDimensions DecisionSource = "dimensions"
	// SourceOverride is a hard override from the signals.
	SourceOverride DecisionSource = "override"
	// SourceCapableHold is a recent escalation still being held on the capable tier.
	SourceCapableHold DecisionSource = "capable_hold"
	// SourceAmbiguous means the signals had no opinion, so the turn was passed on.
	SourceAmbiguous DecisionSource = "ambiguous"
	// SourceLlmClassifier is the capability judge deciding a turn the signals left
	// undecided.
	SourceLlmClassifier DecisionSource = "llm-classifier"
	// SourceFallOpen means nothing resolved the turn, so it landed on the picker's
	// default tier.
	SourceFallOpen DecisionSource = "fall_open"
)

// decisionSourceKey is the state key a turn's decision source is recorded under.
const decisionSourceKey = "decision_source"

func recordDecisionSource(state *State, source DecisionSource) {
	state.Extra[decisionSourceKey] = string(source)
}

// DecisionSource reports what decided the turn, if anything has been recorded yet.
func (s *State) DecisionSource() (DecisionSource, bool) {
	value, ok := s.Extra[decisionSourceKey].(string)
	return DecisionSource(value), ok
}

// fallOpenClassifier closes a stage cascade on a tier.
//
// Zero confidence is deliberate: this is not a judgement, it is the absence of one,
// and it is what keeps an undecided turn routable. Upstream names the same choice —
// the fallback exists so nothing is left unrouted, not so that it looks decided.
type fallOpenClassifier struct {
	defaultTier Tier
}

func (c fallOpenClassifier) Classify(
	_ context.Context, _ *State, _ *Turn, driver *Driver,
) (Classification, error) {
	target := driver.FirstModel(string(c.defaultTier))
	if target == "" {
		// Upstream propagates this rather than abstaining: a pool with no model for
		// the default tier cannot route undecided turns at all, which is a
		// configuration fault, not a turn that needs another opinion.
		return Classification{}, fmt.Errorf(
			"stage_router: no %s model is configured to fall open to", c.defaultTier)
	}
	return Scores(Score{
		Target:     target,
		Category:   string(c.defaultTier),
		Confidence: 0,
	}), nil
}

// sourceStamp credits the classifier that decided a turn.
//
// Only a deciding classifier is stamped — an abstaining one passes the turn on, and
// the decision is not its to claim. The evidence is set only when empty, so the first
// component with something to say about the turn owns the explanation.
type sourceStamp struct {
	inner  Classifier
	source DecisionSource
}

func (s sourceStamp) Classify(
	ctx context.Context, state *State, turn *Turn, driver *Driver,
) (Classification, error) {
	classification, err := s.inner.Classify(ctx, state, turn, driver)
	if err != nil {
		return Classification{}, err
	}
	winner, err := classification.Argmax(false)
	if err != nil {
		return Classification{}, err
	}
	if winner != nil {
		recordDecisionSource(state, s.source)
		driver.SetEvidenceIfEmpty(map[string]any{"source": string(s.source)})
	}
	return classification, nil
}

// tierPromptProcessor hands the routed model the tier's system prompt.
//
// It runs on the decision rather than the request because the tier is what the
// decision determines: the same conversation gets a different prompt depending on
// where it was sent.
type tierPromptProcessor struct {
	capable   string
	efficient string
}

func (p tierPromptProcessor) Process(_ *State, event Event) error {
	if event.Kind != EventDecision || event.Turn == nil {
		return nil
	}
	prompt := ""
	switch Tier(event.Category) {
	case TierCapable:
		prompt = p.capable
	case TierEfficient:
		prompt = p.efficient
	}
	if prompt != "" {
		prependSystemPrompt(&event.Turn.Request, prompt)
	}
	return nil
}

// prependSystemPrompt puts a system instruction ahead of the client's own.
//
// Ahead rather than appended because the tier's instruction frames how the task is
// approached, and providers give earlier system content more weight. It is a separate
// instruction block, not a message, so it cannot be mistaken for something the user
// said.
func prependSystemPrompt(request *LlmRequest, prompt string) {
	instructions := make([]InstructionBlock, 0, len(request.Instructions)+1)
	instructions = append(instructions, InstructionBlock{
		Role:    RoleSystem,
		Content: []ContentBlock{Text(prompt)},
	})
	request.Instructions = append(instructions, request.Instructions...)
}

// toolSignalProcessor derives the signals a stage router scores.
//
// Signals come from the conversation's tool traffic — which is the whole reason this
// family reads messages rather than a prompt. Deriving them is the next slice of this
// port, and until it lands a turn that carries tool traffic is *refused* rather than
// scored as signal-free.
//
// The distinction matters more than it looks. "No signals" is a legitimate state that
// means "the signals have no opinion", and upstream falls open to the picker's tier
// on it. A port that could not derive signals would reach that same state on every
// coding session, and route all of them to the default tier while looking exactly
// like it was working.
type toolSignalProcessor struct {
	recentWindow int
}

func (p toolSignalProcessor) Process(_ *State, event Event) error {
	if event.Kind != EventRequest || event.Turn == nil {
		return nil
	}
	if !hasToolTraffic(event.Turn.Request.Messages) {
		return nil
	}
	return errors.New(
		"stage_router: tool signals are not ported yet, so a turn carrying tool " +
			"traffic cannot be scored — refusing rather than falling open on it")
}

// hasToolTraffic reports whether a conversation carries anything the signal scorer
// would read: a tool call or a tool result.
func hasToolTraffic(messages []Message) bool {
	for _, message := range messages {
		for _, block := range message.Content {
			switch block.Kind {
			case ContentToolCall, ContentToolResult:
				return true
			}
		}
	}
	return false
}
