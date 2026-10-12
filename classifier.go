package libsygo

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
)

// Score is one classifier's recommendation.
type Score struct {
	// Confidence in [0, 1]. Upstream's cascade compares confidences only within
	// one classifier's score set; the decisions here are made by a single
	// classifier per turn, so the value matters less than the ordering.
	Confidence float64

	// Target is the recommended model id.
	Target string

	// Category the target was drawn from, when the classifier picked one. The
	// rest of that category is what the turn falls through on failure.
	Category string
}

// Classification is a classifier's verdict.
//
// A classifier that abstains returns an empty set, and the cascade passes the
// turn to the next classifier.
type Classification struct {
	scores []Score

	// ambiguous marks recommendations the classifier considers inconclusive.
	// [Classification.Argmax] yields nothing for them unless the caller opts to
	// ignore the ambiguity.
	ambiguous bool
}

// Scores is a definite classification.
func Scores(scores ...Score) Classification {
	return Classification{scores: scores}
}

// Ambiguity is a classifier declining to decide, which is how a routing policy
// reports that it could not reach a verdict.
func Ambiguity() Classification {
	return Classification{ambiguous: true}
}

// Argmax returns the highest-confidence score, or nil when the classifier
// abstained.
//
// Ties keep the earlier score, so cascade order breaks them. A NaN confidence is
// an error rather than a silent skip: it has no defined ordering, and upstream
// surfaces it too.
func (c Classification) Argmax(ignoreAmbiguity bool) (*Score, error) {
	if c.ambiguous && !ignoreAmbiguity {
		return nil, nil
	}

	best := -1
	for i := range c.scores {
		if math.IsNaN(c.scores[i].Confidence) {
			return nil, fmt.Errorf("classifier returned NaN confidence for target %q", c.scores[i].Target)
		}
		if best < 0 || c.scores[i].Confidence > c.scores[best].Confidence {
			best = i
		}
	}
	if best < 0 {
		return nil, nil
	}
	return &c.scores[best], nil
}

// Classifier decides a turn's target.
//
// It may rewrite the turn in place — inject a system prompt, drop tools — and the
// edit is not scoped to the call: later classifiers score the rewritten turn, and
// the rewritten turn is what the selected model receives.
type Classifier interface {
	Classify(ctx context.Context, state *State, turn *Turn, driver *Driver) (Classification, error)
}

// DefaultCategoryClassifier closes a cascade.
//
// It scores a category's first model at zero confidence, so a turn is never left
// unrouted. Zero confidence is deliberate: this is a fallback, not a judgement,
// and upstream's FallOpen does the same.
type DefaultCategoryClassifier struct {
	Category string
}

// Classify implements Classifier.
func (c DefaultCategoryClassifier) Classify(
	_ context.Context, _ *State, _ *Turn, driver *Driver,
) (Classification, error) {
	model := driver.FirstModel(c.Category)
	if model == "" {
		return Scores(), nil
	}
	return Scores(Score{Target: model, Category: c.Category, Confidence: 0}), nil
}

// evidenceSource reads the `source` field from the outcome's evidence.
func evidenceSource(evidence json.RawMessage) string {
	var parsed struct {
		Source string `json:"source"`
	}
	if err := json.Unmarshal(evidence, &parsed); err != nil {
		return ""
	}
	return parsed.Source
}
