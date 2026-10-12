package libsygo

import (
	"context"
	"fmt"
	"math"
)

// The scorer's constants, ported as-is.
const (
	// scoreGain spreads the raw sum into a usable range. One maxed signal is ±0.10,
	// two corroborating signals ±0.20, which the threshold reads as: roughly 0.3
	// escalates on one signal, 0.5 needs about one and a half, 0.7 needs two.
	scoreGain = 5.0
	// signalUnit is what one maxed signal is worth.
	signalUnit = 0.10
	// hardSeverity normalises severity onto the same unit, so it lands beside the
	// stall dimensions rather than dominating them.
	hardSeverity = 0.7
	// stallMinTurnDepth is how deep a conversation has to be before "producing
	// nothing" means anything. It counts messages, so it is wire-format dependent.
	stallMinTurnDepth = 8
)

// dimensions is the two-axis view of a signal: how badly it is going, and what the
// agent is doing about it.
type dimensions struct {
	// severity is the windowed worst failure in [0, 1].
	severity float64
	// spinning is 1 when a deep turn has no recognized observation, mutation, plan or
	// new activity: activity without a purpose.
	spinning float64
	// exploring is 1 when a deep turn observes or plans but does not produce.
	exploring float64
	// productionIntensity is the fraction of recent operations that produced code.
	productionIntensity float64
}

// scoreResult is a signed score in (-1, +1) with its magnitude.
//
// Positive is capable, negative efficient, and the magnitude is the confidence — so
// the threshold is a confidence gate rather than a rate.
type scoreResult struct {
	score      float64
	confidence float64
}

// probability remaps the score onto a capable-tier probability.
func (r scoreResult) probability() float64 {
	return (r.score + 1.0) / 2.0
}

// pickOutcome is either a decided tier or a request to consult what comes next.
type pickOutcome struct {
	// resolved reports whether the signals decided.
	resolved bool
	// tier and source describe a resolved decision.
	tier   Tier
	source DecisionSource
	// probability is the capable-tier probability; 0.5 for the hard rules, which
	// bypass the scorer.
	probability float64
	// confidence is the scorer's certainty, nil where the scorer did not run.
	confidence *float64
	// defaultTier is where an undecided turn falls open to.
	defaultTier Tier
}

// dimensionsFromSignal projects a signal onto the two-axis space.
//
// The two stall dimensions partition "not producing" by investigative activity, so at
// most one fires and the production axis is never double-counted.
func dimensionsFromSignal(signal ToolSignals) dimensions {
	recentOps := signal.RecentWriteCount + signal.RecentEditCount +
		signal.RecentReadCount + signal.RecentTodowriteCount
	deepEnough := signal.TurnDepth >= stallMinTurnDepth
	noProduction := signal.RecentWriteCount == 0 && signal.RecentEditCount == 0
	investigating := signal.RecentReadCount >= 1 || signal.RecentTodowriteCount >= 1
	newActivity := signal.RecentNewCount >= 1

	return dimensions{
		// Widened from f32 deliberately: the storage width is part of the arithmetic,
		// and the corpus records the resulting confidences to the last digit.
		severity:  float64(signal.Severity),
		spinning:  boolUnit(deepEnough && noProduction && !investigating && !newActivity),
		exploring: boolUnit(deepEnough && noProduction && investigating && !newActivity),
		productionIntensity: ratio(
			signal.RecentWriteCount+signal.RecentEditCount, recentOps),
	}
}

// scoreSignal weighs the error axis against the production axis and squashes the sum.
func scoreSignal(signal ToolSignals) scoreResult {
	d := dimensionsFromSignal(signal)
	raw := signalUnit * (d.severity/hardSeverity + d.spinning + d.exploring - d.productionIntensity)
	score := math.Tanh(scoreGain * raw)
	return scoreResult{score: score, confidence: math.Abs(score)}
}

// overrideReason is a hard reason to escalate that outranks the scorer.
//
// Compaction comes first because it wipes the accumulated signals, so a task that had
// escalated would otherwise snap back to the cheap tier — a context big enough to
// overflow belongs on the capable one.
func overrideReason(signal ToolSignals) (string, bool) {
	if signal.Compacted {
		return "compaction", true
	}
	if signal.Severity >= SeverityCritical {
		return "critical_error", true
	}
	if signal.RepeatedFailure {
		return "repeated_failure", true
	}
	return "", false
}

// pickTier decides a turn from its signals. The rules run in order; the first that
// fires wins.
func pickTier(signal ToolSignals, mode PickerMode, confidenceThreshold float64) pickOutcome {
	// 1. Escalate, ahead of everything else.
	if _, ok := overrideReason(signal); ok {
		certain := 1.0
		return pickOutcome{
			resolved:    true,
			tier:        TierCapable,
			source:      SourceOverride,
			probability: 0.5,
			confidence:  &certain,
		}
	}

	// 2. Score. Resolution needs the probability outside the closed ambiguous band, so
	// a signal exactly at the threshold is not a decision.
	scored := scoreSignal(signal)
	probability := scored.probability()
	halfThreshold := confidenceThreshold / 2
	if probability > 0.5+halfThreshold || probability < 0.5-halfThreshold {
		tier := TierEfficient
		if probability > 0.5 {
			tier = TierCapable
		}
		return pickOutcome{
			resolved:    true,
			tier:        tier,
			source:      SourceDimensions,
			probability: probability,
			confidence:  &scored.confidence,
		}
	}

	// 3. Fall open: the signals did not corroborate enough to be sure.
	defaultTier, _ := mode.defaultTier()
	return pickOutcome{
		probability: probability,
		confidence:  &scored.confidence,
		defaultTier: defaultTier,
	}
}

// stageClassifier routes on the signals, and reflects the way a session got there.
//
// It also owns the capable hold: after a signal-driven escalation the session stays on
// the capable tier for a few turns, so the fix an escalation prompted is not judged by
// the tier that failed. A clean test pass clears the hold early, which is the only way
// out before it expires.
type stageClassifier struct {
	mode                PickerMode
	confidenceThreshold float64
	capableHoldTurns    uint32
}

func (c *stageClassifier) Classify(
	_ context.Context, state *State, _ *Turn, driver *Driver,
) (Classification, error) {
	signal := state.ToolSignals
	if signal == nil {
		// No tool activity at all, so the signals have no opinion. Distinct from a
		// zero signal, which does score and simply finds nothing to weigh.
		recordDecisionSource(state, SourceAmbiguous)
		return Ambiguity(), nil
	}

	key := capableHoldKey
	// A clean pass after a failure clears the hold: the work is done.
	cleanTestPass := signal.TestsPassed && signal.NoErrorStreak > 0
	if cleanTestPass {
		delete(state.Extra, key)
	}

	var outcome pickOutcome
	if !cleanTestPass && consumeCapableHold(state, key) {
		certain := 1.0
		outcome = pickOutcome{
			resolved:    true,
			tier:        TierCapable,
			source:      SourceCapableHold,
			probability: 0.5,
			confidence:  &certain,
		}
	} else {
		outcome = pickTier(*signal, c.mode, c.confidenceThreshold)
	}

	if !outcome.resolved {
		recordDecisionSource(state, SourceAmbiguous)
		return Ambiguity(), nil
	}

	target := driver.FirstModel(string(outcome.tier))
	if target == "" {
		return Classification{}, fmt.Errorf(
			"stage_router: no %s model is configured", outcome.tier)
	}

	// Only a decision the signals made arms the hold. A hold that re-armed itself
	// would never expire.
	if outcome.tier == TierCapable &&
		(outcome.source == SourceOverride || outcome.source == SourceDimensions) {
		setCapableHold(state, key, c.capableHoldTurns)
	}
	recordDecisionSource(state, outcome.source)

	confidence := 2 * math.Abs(outcome.probability-0.5)
	if outcome.confidence != nil {
		confidence = *outcome.confidence
	}
	if outcome.source == SourceDimensions {
		driver.SetEvidence(map[string]any{
			"source":     string(outcome.source),
			"confidence": confidence,
			"threshold":  c.confidenceThreshold,
		})
	} else {
		driver.SetEvidence(map[string]any{"source": string(outcome.source)})
	}

	return Scores(Score{
		Target:     target,
		Category:   string(outcome.tier),
		Confidence: confidence,
	}), nil
}

// capableHoldKey is the state key the hold is counted under.
//
// Upstream refines it per agent id, so one agent's escalation does not hold a
// sibling's turns on the capable tier. The ported request carries no agent id, so the
// hold is per session — which is the scope a caller controls today.
const capableHoldKey = "capable_hold_turns"

// consumeCapableHold spends one turn of the hold, reporting whether it applied.
func consumeCapableHold(state *State, key string) bool {
	remaining, ok := state.Extra[key].(uint32)
	if !ok {
		return false
	}
	if remaining == 0 {
		delete(state.Extra, key)
		return false
	}
	remaining--
	if remaining == 0 {
		delete(state.Extra, key)
	} else {
		state.Extra[key] = remaining
	}
	return true
}

// setCapableHold arms the hold, or clears it when the hold is disabled.
func setCapableHold(state *State, key string, turns uint32) {
	if turns == 0 {
		delete(state.Extra, key)
		return
	}
	state.Extra[key] = turns
}

func boolUnit(value bool) float64 {
	if value {
		return 1.0
	}
	return 0.0
}

func ratio(numerator, denominator uint32) float64 {
	if denominator == 0 {
		return 0.0
	}
	return float64(numerator) / float64(denominator)
}
