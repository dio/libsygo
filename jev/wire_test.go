package jev

import (
	"context"
	"encoding/json"
	"os"
	"reflect"
	"testing"
	"time"
)

func choiceRequest(t *testing.T) DecisionRequest {
	t.Helper()
	instructions := json.RawMessage(`{"setting":"judge the task"}`)
	kindData, err := json.Marshal(map[string]any{
		"options": []map[string]any{
			{"id": "advantage", "description": "capable succeeds and efficient fails"},
			{"id": "no_advantage", "description": "the complement"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	model := "jev-latest"
	return DecisionRequest{
		Model:   &model,
		Context: json.RawMessage(`{"task":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`),
		Questions: map[string]DecisionQuestion{
			"route": {
				Instructions: instructions,
				Kind:         Tagged{Type: "choice", Data: kindData},
			},
		},
	}
}

// The wire body must match what system_one.rs's encode() produces: a `type` and
// `criteria` per question, with choice criteria keyed by option id.
func TestEncodeRequestChoiceMatchesSystemOneShape(t *testing.T) {
	body, err := EncodeRequest(choiceRequest(t))
	if err != nil {
		t.Fatal(err)
	}

	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatal(err)
	}
	want := map[string]any{
		"model": "jev-latest",
		"state": map[string]any{
			"task": []any{map[string]any{
				"role":    "user",
				"content": []any{map[string]any{"type": "text", "text": "hi"}},
			}},
		},
		"questions": map[string]any{
			"route": map[string]any{
				"type":         "choice",
				"instructions": map[string]any{"setting": "judge the task"},
				"criteria": map[string]any{
					"advantage":    "capable succeeds and efficient fails",
					"no_advantage": "the complement",
				},
			},
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("encoded body mismatch\n got: %s\nwant: %s", mustJSON(t, got), mustJSON(t, want))
	}
}

func TestEncodeRequestRequiresModel(t *testing.T) {
	request := choiceRequest(t)
	request.Model = nil
	if _, err := EncodeRequest(request); err == nil {
		t.Fatal("expected an error when no model is selected")
	}
}

// The real capture from the live endpoint (testdata/fixtures/jev) must decode
// onto the neutral IR exactly as system_one.rs does.
func TestDecodeResponseRealCapture(t *testing.T) {
	wire := []byte(`{
	  "model": "jev-1.13.0",
	  "answers": {
	    "route": {
	      "type": "choice",
	      "choice": "no_advantage",
	      "confidence": 0.5,
	      "probabilities": { "advantage": 0.25, "no_advantage": 0.75 }
	    }
	  },
	  "usage": { "input_tokens": 834, "output_tokens": 39 }
	}`)

	response, err := DecodeResponse(wire, choiceRequest(t))
	if err != nil {
		t.Fatal(err)
	}
	if response.Model == nil || *response.Model != "jev-1.13.0" {
		t.Fatalf("model = %v", response.Model)
	}
	answer, ok := response.Answers["route"]
	if !ok {
		t.Fatal("missing route answer")
	}
	if answer.Value.Type != "choice" {
		t.Fatalf("value type = %q", answer.Value.Type)
	}
	if answer.ProviderConfidence == nil || *answer.ProviderConfidence != 0.5 {
		t.Fatalf("confidence = %v", answer.ProviderConfidence)
	}

	var data struct {
		Selected      string             `json:"selected"`
		Probabilities map[string]float64 `json:"probabilities"`
	}
	if err := json.Unmarshal(answer.Value.Data, &data); err != nil {
		t.Fatal(err)
	}
	if data.Selected != "no_advantage" {
		t.Fatalf("selected = %q", data.Selected)
	}
	if data.Probabilities["advantage"] != 0.25 || data.Probabilities["no_advantage"] != 0.75 {
		t.Fatalf("probabilities = %v", data.Probabilities)
	}
}

func TestDecodeResponseRejectsKeyMismatch(t *testing.T) {
	wire := []byte(`{"answers":{"other":{"type":"choice","choice":"advantage"}}}`)
	if _, err := DecodeResponse(wire, choiceRequest(t)); err == nil {
		t.Fatal("expected a key-mismatch error")
	}
}

func TestDecodeResponseRejectsUnknownChoice(t *testing.T) {
	wire := []byte(`{"answers":{"route":{"type":"choice","choice":"nonsense"}}}`)
	if _, err := DecodeResponse(wire, choiceRequest(t)); err == nil {
		t.Fatal("expected an error for a choice outside the requested options")
	}
}

// TestLiveCall is skipped unless a key is present, so CI never depends on the
// network; run it locally with TYPESAFE_API_KEY set.
func TestLiveCall(t *testing.T) {
	apiKey := os.Getenv("TYPESAFE_API_KEY")
	if apiKey == "" {
		t.Skip("TYPESAFE_API_KEY not set")
	}
	endpoint := os.Getenv("TYPESAFE_ENDPOINT")
	if endpoint == "" {
		endpoint = DefaultEndpoint
	}
	client, err := NewClient(endpoint, apiKey, 30*time.Second)
	if err != nil {
		t.Fatal(err)
	}

	request := choiceRequest(t)
	response, err := client.Call(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	answer, ok := response.Answers["route"]
	if !ok {
		t.Fatal("missing route answer")
	}
	var data struct {
		Selected      string             `json:"selected"`
		Probabilities map[string]float64 `json:"probabilities"`
	}
	if err := json.Unmarshal(answer.Value.Data, &data); err != nil {
		t.Fatal(err)
	}
	advantage, ok := data.Probabilities["advantage"]
	if !ok || advantage < 0 || advantage > 1 {
		t.Fatalf("advantage = %v", data.Probabilities)
	}
	if data.Selected != "advantage" && data.Selected != "no_advantage" {
		t.Fatalf("selected = %q", data.Selected)
	}
	t.Logf("live: model=%v selected=%s advantage=%.3f", response.Model, data.Selected, advantage)
}

func mustJSON(t *testing.T, value any) string {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return string(encoded)
}
