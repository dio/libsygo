package libsygo

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"

	"github.com/dio/libsygo/jev"
)

// Fail-open reason codes. These appear in the outcome's evidence, so they match
// upstream's vocabulary rather than being invented here.
const (
	reasonCallError      = "call_error"
	reasonInvalidVerdict = "invalid_verdict"
)

// capabilityClassifier asks a judge whether the efficient tier can handle the
// turn, and routes to the capable tier only when it cannot.
//
// It is the classifier form of upstream's DecisionClassifier: a cascade member
// rather than a router, so the latch can lead it and a default classifier can
// close it.
type capabilityClassifier struct {
	judge        *Judge
	instructions json.RawMessage
	failOpen     bool
}

// Classify implements Classifier.
func (c *capabilityClassifier) Classify(
	ctx context.Context, _ *State, turn *Turn, driver *Driver,
) (Classification, error) {
	judgeModel := driver.FirstModel(categoryJudge)
	if judgeModel == "" {
		// Upstream propagates this rather than failing open: a pool with no judge
		// is a configuration fault, not a judge that misbehaved.
		return Classification{}, fmt.Errorf("no judge model is configured")
	}

	request, err := c.buildRequest(TaskMessages(turn.Request.Messages), judgeModel, driver)
	if err != nil {
		return Classification{}, err
	}

	response, err := driver.CallDecision(ctx, request)
	if err != nil {
		if !c.failOpen {
			// Nothing to route on, and the caller asked not to guess.
			return Classification{}, err
		}
		return c.abstain(driver, reasonCallError)
	}

	score, ok := advantage(response)
	if !ok {
		// An unusable verdict abstains *regardless* of fail-open. Upstream only
		// consults fail-open for a failed call: a verdict it cannot read is a
		// reason to stop trusting this classifier, not to fail the request.
		return c.abstain(driver, reasonInvalidVerdict)
	}

	target, category := c.tier(score, driver)
	driver.SetEvidence(map[string]any{
		"source":    "decision_classifier",
		"verdict":   "relative_advantage",
		"score":     score,
		"threshold": c.judge.Cutoff,
	})
	return Scores(Score{Target: target, Category: category, Confidence: 1}), nil
}

// abstain is upstream's `unavailable`: decline to decide, record why, and let the
// cascade's next classifier close the turn. The evidence is set only if empty, so
// a decision made earlier is not overwritten.
func (c *capabilityClassifier) abstain(driver *Driver, reason string) (Classification, error) {
	driver.SetEvidenceIfEmpty(map[string]any{"source": "fail_open", "reason_code": reason})
	return Ambiguity(), nil
}

// tier applies the cutoff: strictly greater selects capable, so a score exactly
// at the cutoff does not.
func (c *capabilityClassifier) tier(score float64, driver *Driver) (model, category string) {
	if score > c.judge.Cutoff {
		return driver.FirstModel(categoryCapable), categoryCapable
	}
	return driver.FirstModel(categoryEfficient), categoryEfficient
}

// buildRequest assembles the context and question upstream's DecisionClassifier
// sends to the judge.
//
// The task is the selected messages, not a flattened prompt: a judge is shown
// conversation turns, and flattening them would lose the roles and block structure
// upstream preserves.
func (c *capabilityClassifier) buildRequest(
	task []Message, judgeModel string, driver *Driver,
) (jev.DecisionRequest, error) {
	capableLabel, err := labelFor(c.judge.Candidates, driver.FirstModel(categoryCapable))
	if err != nil {
		return jev.DecisionRequest{}, err
	}
	efficientLabel, err := labelFor(c.judge.Candidates, driver.FirstModel(categoryEfficient))
	if err != nil {
		return jev.DecisionRequest{}, err
	}

	context, err := json.Marshal(map[string]any{
		"task":       json.RawMessage(AppendMessages(nil, task)),
		"candidates": candidateLabels(c.judge.Candidates),
		"comparison": map[string]string{"capable": capableLabel, "efficient": efficientLabel},
		"evidence":   c.judge.Evidence,
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

	return jev.DecisionRequest{
		Model:   &judgeModel,
		Context: context,
		Questions: map[string]jev.DecisionQuestion{
			"route": {Instructions: c.instructions, Kind: jev.Tagged{Type: "choice", Data: options}},
		},
	}, nil
}

// candidateLabels are the labels in the order upstream's BTreeMap yields them.
func candidateLabels(candidates map[string]string) []string {
	labels := make([]string, 0, len(candidates))
	for label := range candidates {
		labels = append(labels, label)
	}
	sort.Strings(labels)
	return labels
}

// labelFor names the label whose target is a model, so the judge can be told
// which candidates it is comparing without being told their brands.
func labelFor(candidates map[string]string, model string) (string, error) {
	for _, label := range candidateLabels(candidates) {
		if candidates[label] == model {
			return label, nil
		}
	}
	return "", fmt.Errorf("decision candidate is missing for target %q", model)
}

// advantage reads p(advantage) from the judge's verdict. Only a Choice answer with
// a probability in [0,1] counts; anything else is an unusable verdict.
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
