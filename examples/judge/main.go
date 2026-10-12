// Command judge runs libsygo end to end against a stand-in judge: no credentials,
// no network, no Envoy.
//
// A local HTTP server answers in the System One wire shape, jev.Client decodes it,
// and the router decides. The stand-in judges by keyword so the demo shows both
// tiers, and the third prompt reuses the session to show the latch skipping the
// judge entirely.
//
//	go run ./examples/judge
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/dio/libsygo"
	"github.com/dio/libsygo/jev"
)

func main() {
	url, judgeCalls, err := startJudge()
	if err != nil {
		log.Fatal(err)
	}

	// jev.Client satisfies libsygo.Decider, so a host with no HTTP stack of its own
	// can hand it straight to the router.
	decider, err := jev.NewClient(url, "demo-key", 5*time.Second)
	if err != nil {
		log.Fatal(err)
	}

	router, err := libsygo.New(profile(), models())
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()

	// Two sessions, so both tiers are exercised; the third request reuses a
	// session, so the latch decides it without asking the judge again.
	requests := []struct {
		prompt  string
		session string
	}{
		{"write a haiku about routers", "tenant-a/chat-1"},
		{"refactor the payment module onto the new ledger abstraction", "tenant-a/chat-2"},
		{"and likewise for the reporting module", "tenant-a/chat-2"},
	}

	fmt.Printf("%-56s %-8s %s\n", "prompt", "model", "evidence")
	for _, request := range requests {
		outcome := router.Route(ctx, libsygo.Turn{
			Prompt:  request.prompt,
			Session: request.session,
		}, decider)

		if outcome.Error != "" {
			fmt.Printf("%-56s FAILED   %s\n", shorten(request.prompt), outcome.Error)
			continue
		}
		fmt.Printf("%-56s %-8s %s\n", shorten(request.prompt), outcome.Model, outcome.Evidence)
	}

	fmt.Printf("\njudge calls: %d — two sessions, and the third request was latched\n", judgeCalls.Load())
}

// profile is the decision the router asks the judge. The evidence is what lets the
// judge reason about the tiers without brand names.
func profile() libsygo.Profile {
	failOpen := true
	return libsygo.Profile{
		Algorithm:       "llm_classifier",
		Mode:            "capability",
		FailOpen:        &failOpen,
		SessionAffinity: true,
		Judge: &libsygo.Judge{
			Type:       "decision",
			Cutoff:     0.4,
			Candidates: map[string]string{"a": "smart", "b": "cheap"},
			Evidence: json.RawMessage(
				`{"candidates":{"a":"capable tier, for repository-scale work","b":"efficient tier, for edits and questions"}}`),
		},
	}
}

// models is the routable pool. "any" must list every routable target: the selected
// target is resolved against it, and the order is the fallback order.
func models() libsygo.Models {
	return libsygo.Models{
		"capable":   {"smart"},
		"efficient": {"cheap"},
		"judge":     {"jev-latest"},
		"any":       {"cheap", "smart"},
	}
}

// startJudge serves the System One wire shape on a loopback port, and counts the
// calls it receives so the latch is observable.
func startJudge() (string, *atomic.Int64, error) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return "", nil, err
	}

	var calls atomic.Int64
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		body, _ := io.ReadAll(r.Body)

		// A stand-in for a real judge: "refactor" needs the capable tier, anything
		// else does not.
		advantage := 0.05
		choice := "no_advantage"
		if bytes.Contains(body, []byte("refactor")) {
			advantage, choice = 0.77, "advantage"
		}

		w.Header().Set("content-type", "application/json")
		fmt.Fprintf(w,
			`{"model":"jev-demo","answers":{"route":{"type":"choice","choice":%q,"confidence":0.9,`+
				`"probabilities":{"advantage":%v,"no_advantage":%v}}},`+
				`"usage":{"input_tokens":120,"output_tokens":8}}`,
			choice, advantage, 1-advantage)
	})

	go func() { _ = http.Serve(listener, handler) }()
	return "http://" + listener.Addr().String(), &calls, nil
}

func shorten(prompt string) string {
	if len(prompt) <= 54 {
		return prompt
	}
	return prompt[:53] + "…"
}
