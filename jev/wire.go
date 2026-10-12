// Package jev implements Switchyard's System One (JEV) decision-client wire
// protocol: the neutral DecisionRequest / DecisionResponse IR <-> the
// api.typesafe.ai /v1/systemone JSON shape.
//
// Ported from crates/libsy-llm-client/src/system_one.rs at the revision in
// testdata/corpus/PIN. This is tier 0 of the conformance tiers.
package jev

import (
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
)

// DecisionRequest mirrors switchyard_protocol::DecisionRequest.
type DecisionRequest struct {
	Model     *string                     `json:"model"`
	Context   json.RawMessage             `json:"context"`
	Questions map[string]DecisionQuestion `json:"questions"`
}

// DecisionQuestion mirrors switchyard_protocol::DecisionQuestion.
type DecisionQuestion struct {
	Instructions json.RawMessage `json:"instructions"`
	Kind         Tagged          `json:"kind"`
}

// DecisionResponse mirrors switchyard_protocol::DecisionResponse.
type DecisionResponse struct {
	ID      *string                   `json:"id"`
	Model   *string                   `json:"model"`
	Answers map[string]DecisionAnswer `json:"answers"`
	Usage   json.RawMessage           `json:"usage"`
}

// DecisionAnswer mirrors switchyard_protocol::DecisionAnswer.
type DecisionAnswer struct {
	Value              Tagged   `json:"value"`
	ProviderConfidence *float64 `json:"provider_confidence"`
}

// Tagged is the `{ "type": ..., "data": ... }` envelope used by DecisionKind,
// DecisionValue and BooleanEstimate.
type Tagged struct {
	Type string          `json:"type"`
	Data json.RawMessage `json:"data"`
}

// ChoiceOption mirrors switchyard_protocol::ChoiceOption.
type ChoiceOption struct {
	ID          string          `json:"id"`
	Description json.RawMessage `json:"description"`
}

type booleanKind struct {
	TrueDescription  json.RawMessage `json:"true_description"`
	FalseDescription json.RawMessage `json:"false_description"`
}

type choiceKind struct {
	Options []ChoiceOption `json:"options"`
}

type scoreKind struct {
	Levels []json.RawMessage `json:"levels"`
}

// Wire response shape. System One flattens the answer variant, so this is one
// struct with optional fields rather than a tagged union.
type wireResponse struct {
	ID      *string               `json:"id"`
	Model   *string               `json:"model"`
	Answers map[string]wireAnswer `json:"answers"`
	Usage   json.RawMessage       `json:"usage"`
}

type wireAnswer struct {
	Type          string          `json:"type"`
	Confidence    *float64        `json:"confidence"`
	Noul          *float64        `json:"noul"`
	Choice        *string         `json:"choice"`
	Score         *float64        `json:"score"`
	Probabilities json.RawMessage `json:"probabilities"`
}

// EncodeRequest renders the neutral IR as the System One request body.
func EncodeRequest(request DecisionRequest) ([]byte, error) {
	if request.Model == nil || *request.Model == "" {
		return nil, errors.New("system one requires a selected model")
	}
	questions := make(map[string]any, len(request.Questions))
	for id, question := range request.Questions {
		kind, criteria, err := encodeKind(question.Kind)
		if err != nil {
			return nil, fmt.Errorf("question %q: %w", id, err)
		}
		questions[id] = map[string]any{
			"type":         kind,
			"instructions": question.Instructions,
			"criteria":     criteria,
		}
	}
	return json.Marshal(map[string]any{
		"model":     *request.Model,
		"state":     request.Context,
		"questions": questions,
	})
}

// encodeKind maps a DecisionKind onto System One's `(type, criteria)` pair.
func encodeKind(kind Tagged) (string, any, error) {
	switch kind.Type {
	case "boolean":
		var parsed booleanKind
		if err := json.Unmarshal(kind.Data, &parsed); err != nil {
			return "", nil, err
		}
		criteria := map[string]any{}
		if len(parsed.TrueDescription) > 0 {
			criteria["true"] = parsed.TrueDescription
		}
		if len(parsed.FalseDescription) > 0 {
			criteria["false"] = parsed.FalseDescription
		}
		return "noul", criteria, nil

	case "choice":
		var parsed choiceKind
		if err := json.Unmarshal(kind.Data, &parsed); err != nil {
			return "", nil, err
		}
		criteria := map[string]any{}
		for _, option := range parsed.Options {
			if _, duplicate := criteria[option.ID]; duplicate {
				return "", nil, fmt.Errorf("duplicate option %q", option.ID)
			}
			criteria[option.ID] = option.Description // nil -> null
		}
		return "choice", criteria, nil

	case "score":
		var parsed scoreKind
		if err := json.Unmarshal(kind.Data, &parsed); err != nil {
			return "", nil, err
		}
		return "score", parsed.Levels, nil

	default:
		return "", nil, fmt.Errorf("unsupported question kind %q", kind.Type)
	}
}

// DecodeResponse maps a System One response body back onto the neutral IR,
// validating it against the request that produced it.
func DecodeResponse(body []byte, request DecisionRequest) (DecisionResponse, error) {
	var wire wireResponse
	if err := json.Unmarshal(body, &wire); err != nil {
		return DecisionResponse{}, err
	}
	if len(wire.Answers) != len(request.Questions) {
		return DecisionResponse{}, errors.New(
			"System One answer keys do not match the requested question keys")
	}

	response := DecisionResponse{
		ID:      wire.ID,
		Model:   wire.Model,
		Usage:   wire.Usage,
		Answers: make(map[string]DecisionAnswer, len(wire.Answers)),
	}
	for id, answer := range wire.Answers {
		question, known := request.Questions[id]
		if !known {
			return DecisionResponse{}, fmt.Errorf("answer %q has an unexpected question ID", id)
		}
		value, err := decodeValue(question.Kind, answer)
		if err != nil {
			return DecisionResponse{}, fmt.Errorf("answer %q: %w", id, err)
		}
		response.Answers[id] = DecisionAnswer{
			Value:              value,
			ProviderConfidence: answer.Confidence,
		}
	}
	return response, nil
}

func decodeValue(kind Tagged, answer wireAnswer) (Tagged, error) {
	switch kind.Type {
	case "boolean":
		if answer.Type != "noul" || answer.Noul == nil {
			return Tagged{}, errors.New("unexpected question ID, kind, or value")
		}
		inner, err := json.Marshal(Tagged{Type: "probability_true", Data: jsonNumber(*answer.Noul)})
		if err != nil {
			return Tagged{}, err
		}
		return Tagged{Type: "boolean", Data: inner}, nil

	case "choice":
		var parsed choiceKind
		if err := json.Unmarshal(kind.Data, &parsed); err != nil {
			return Tagged{}, err
		}
		if answer.Type != "choice" || answer.Choice == nil || !hasOption(parsed.Options, *answer.Choice) {
			return Tagged{}, errors.New("unexpected question ID, kind, or value")
		}
		probabilities, err := probabilitiesObject(answer.Probabilities)
		if err != nil {
			return Tagged{}, err
		}
		inner, err := json.Marshal(map[string]any{
			"selected":      *answer.Choice,
			"probabilities": probabilities,
		})
		if err != nil {
			return Tagged{}, err
		}
		return Tagged{Type: "choice", Data: inner}, nil

	case "score":
		var parsed scoreKind
		if err := json.Unmarshal(kind.Data, &parsed); err != nil {
			return Tagged{}, err
		}
		if answer.Type != "score" || answer.Score == nil || len(parsed.Levels) == 0 {
			return Tagged{}, errors.New("unexpected question ID, kind, or value")
		}
		if *answer.Score < 0 || *answer.Score > float64(len(parsed.Levels)-1) {
			return Tagged{}, errors.New("unexpected question ID, kind, or value")
		}
		ordered, err := probabilitiesByRubric(answer.Probabilities, len(parsed.Levels))
		if err != nil {
			return Tagged{}, err
		}
		inner, err := json.Marshal(map[string]any{
			"value":         *answer.Score,
			"probabilities": ordered,
		})
		if err != nil {
			return Tagged{}, err
		}
		return Tagged{Type: "score", Data: inner}, nil
	}

	return Tagged{}, fmt.Errorf("unsupported question kind %q", kind.Type)
}

func hasOption(options []ChoiceOption, id string) bool {
	for _, option := range options {
		if option.ID == id {
			return true
		}
	}
	return false
}

// probabilitiesObject decodes an optional `{id: p}` distribution; absent stays
// nil so it marshals as null, matching the neutral IR.
func probabilitiesObject(raw json.RawMessage) (map[string]float64, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var probabilities map[string]float64
	if err := json.Unmarshal(raw, &probabilities); err != nil {
		return nil, err
	}
	return probabilities, nil
}

// probabilitiesByRubric reorders a score distribution by the request's rubric:
// JSON keys are strings, so the order has to come from the request.
func probabilitiesByRubric(raw json.RawMessage, levels int) ([]float64, error) {
	probabilities, err := probabilitiesObject(raw)
	if err != nil {
		return nil, err
	}
	if probabilities == nil {
		return nil, nil
	}
	if len(probabilities) != levels {
		return nil, errors.New("score probabilities do not match its rubric")
	}
	ordered := make([]float64, levels)
	for index := 0; index < levels; index++ {
		value, ok := probabilities[strconv.Itoa(index)]
		if !ok {
			return nil, errors.New("score probabilities do not match its rubric")
		}
		ordered[index] = value
	}
	return ordered, nil
}

func jsonNumber(value float64) json.RawMessage {
	encoded, _ := json.Marshal(value)
	return encoded
}
