package libsygo

import "testing"

func ask(text string) Message {
	return Message{Role: RoleUser, Content: []ContentBlock{Text(text)}}
}

func invoked(id, name, arguments string) Message {
	return Message{Role: RoleAssistant, Content: []ContentBlock{{
		Kind: ContentToolCall,
		Call: &ToolCall{ID: id, Name: name, Arguments: []byte(arguments)},
	}}}
}

func returned(id, content string, isError bool) Message {
	result := &ToolResult{ToolCallID: id, Content: []ContentBlock{Text(content)}}
	if isError {
		failed := true
		result.IsError = &failed
	}
	return Message{Role: RoleUser, Content: []ContentBlock{{Kind: ContentToolResult, Result: result}}}
}

func signalsOf(messages ...Message) ToolSignals {
	return extractToolSignals(LlmRequest{Messages: messages}, 3, ToolSemantics{})
}

// The counts are split between the whole request and the recent window, because "what
// has this session done" and "what is it doing now" are different questions.
func TestCountsSplitBetweenRequestAndWindow(t *testing.T) {
	messages := []Message{ask("do the work")}
	for index := 1; index <= 5; index++ {
		messages = append(messages,
			invoked("c"+itoa(index), "Edit", `{"path":"f.go"}`),
			returned("c"+itoa(index), "applied", false))
	}

	signal := signalsOf(messages...)
	if signal.EditCount != 5 {
		t.Fatalf("edit_count = %d, want 5", signal.EditCount)
	}
	if signal.RecentEditCount != 3 {
		t.Fatalf("recent_edit_count = %d, want 3 (the window)", signal.RecentEditCount)
	}
	if signal.TurnDepth != 11 {
		t.Fatalf("turn_depth = %d, want 11 messages", signal.TurnDepth)
	}
	if signal.AssistantTurnCount != 5 {
		t.Fatalf("assistant_turn_count = %d, want 5", signal.AssistantTurnCount)
	}
	if signal.ToolResultCount != 5 {
		t.Fatalf("tool_result_count = %d, want 5", signal.ToolResultCount)
	}
}

// Severity is windowed, so an error persists through the recovery turns but decays
// out rather than pinning the session forever.
func TestSeverityIsWindowed(t *testing.T) {
	broken := append([]Message{ask("build it")},
		invoked("c1", "Bash", `{"command":"go build ./..."}`),
		returned("c1", "Traceback (most recent call last):", true))
	// Two clean results keep the failure inside a three-result window.
	for index := 2; index <= 3; index++ {
		broken = append(broken,
			invoked("c"+itoa(index), "Bash", `{"command":"go build ./..."}`),
			returned("c"+itoa(index), "ok", false))
	}

	signal := signalsOf(broken...)
	if signal.Severity != SeverityHard {
		t.Fatalf("severity = %v, want hard: the failure is still in the window", signal.Severity)
	}

	// One more clean result pushes the failure out of the three-result window.
	decayed := append(broken,
		invoked("c4", "Bash", `{"command":"go build ./..."}`),
		returned("c4", "ok", false))
	if got := signalsOf(decayed...).Severity; got != 0 {
		t.Fatalf("severity = %v, want 0 once the failure left the window", got)
	}
}

// A retrieval that found nothing is not a failure, and dropping its content must not
// shift what the window covers.
func TestRetrievalResultsAreExemptFromSeverity(t *testing.T) {
	messages := append([]Message{ask("read it")},
		invoked("c1", "Read", `{"path":"missing.go"}`),
		returned("c1", "no such file or directory", false),
		invoked("c2", "Read", `{"path":"missing.go"}`),
		returned("c2", "error: connection refused", false))

	signal := signalsOf(messages...)
	if signal.Severity != 0 {
		t.Fatalf("severity = %v, want 0: a failed read is not a broken build", signal.Severity)
	}

	// The same failure text from a call that is not a retrieval does count.
	notRetrieval := append([]Message{ask("build it")},
		invoked("c1", "Bash", `{"command":"go test ./..."}`),
		returned("c1", "error: connection refused", false))
	if got := signalsOf(notRetrieval...).Severity; got != SeverityHard {
		t.Fatalf("severity = %v, want hard", got)
	}
}

// A trailing run of calls this vocabulary has no meaning for is activity without a
// recognized purpose, which is what the streak measures.
func TestPureBashStreakCountsOnlyTrailingUnknownCalls(t *testing.T) {
	messages := append([]Message{ask("poke at it")},
		invoked("c1", "Read", `{"path":"f.go"}`),
		returned("c1", "package main", false))
	for index := 2; index <= 4; index++ {
		messages = append(messages,
			invoked("c"+itoa(index), "mystery_probe", `{}`),
			returned("c"+itoa(index), "ok", false))
	}

	if got := signalsOf(messages...).PureBashStreak; got != 3 {
		t.Fatalf("pure_bash_streak = %d, want 3", got)
	}
}

// A pass counts only after the latest failure, since a pass before it says nothing
// about whether it was fixed.
func TestTestsPassedAnchorsOnTheLatestFailure(t *testing.T) {
	after := append([]Message{ask("run the tests")},
		invoked("c1", "Bash", `{"command":"go test ./..."}`),
		returned("c1", "FAIL: TestRoute\nerror: got cheap, want smart", true),
		invoked("c2", "Bash", `{"command":"go test ./..."}`),
		returned("c2", "--- PASS: TestRoute\nok  \tgithub.com/dio/libsygo\t0.5s", false))
	if !signalsOf(after...).TestsPassed {
		t.Fatal("a pass after the failure should count")
	}

	before := append([]Message{ask("run the tests")},
		invoked("c1", "Bash", `{"command":"go test ./..."}`),
		returned("c1", "--- PASS: TestRoute\nok  \tgithub.com/dio/libsygo\t0.5s", false),
		invoked("c2", "Bash", `{"command":"go test ./..."}`),
		returned("c2", "FAIL: TestRoute\nerror: got cheap, want smart", true))
	if signalsOf(before...).TestsPassed {
		t.Fatal("a pass before the failure should not count")
	}
}

// A clean run's summary must not read as a failure, and a non-zero count must.
func TestFailureCountsAreReadNotAssumed(t *testing.T) {
	cases := []struct {
		summary string
		want    bool
	}{
		// A clean run reports zero counts, which must not read as failures.
		{"--- PASS: TestRoute\nok  \tgithub.com/dio/libsygo\t0.5s\n0 failed\n0 errors", true},
		{"all tests passed", true},
		{"--- PASS: TestRoute\nok  \tpkg\n3 failed", false},
		{"--- PASS: TestRoute\nok  \tpkg\n1 error", false},
		{"--- FAIL: TestRoute\nerror: got cheap, want smart", false},
	}

	for _, tc := range cases {
		messages := append([]Message{ask("run the tests")},
			invoked("c1", "Bash", `{"command":"go test ./..."}`),
			returned("c1", tc.summary, false))
		if got := signalsOf(messages...).TestsPassed; got != tc.want {
			t.Errorf("summary %q: tests_passed = %v, want %v", tc.summary, got, tc.want)
		}
	}
}

// Compaction is detected by two different rules, and neither may fire on prose that
// merely names a marker.
func TestCompactionDetection(t *testing.T) {
	cases := []struct {
		text string
		want bool
	}{
		{"this session is being continued from a previous conversation", true},
		{"the task continues\n[context compaction — reference only] earlier turns", true},
		{"the docs mention [context compaction — reference only] in passing", false},
		{"just an ordinary message", false},
	}

	for _, tc := range cases {
		if got := signalsOf(ask(tc.text)).Compacted; got != tc.want {
			t.Errorf("text %q: compacted = %v, want %v", tc.text, got, tc.want)
		}
	}
}

// A reused call id links to its latest call, so an id that was a read and is now a
// write stops being exempt from the error scan.
func TestCallIDLinksToItsLatestCall(t *testing.T) {
	messages := append([]Message{ask("go")},
		invoked("c1", "Read", `{"path":"f.go"}`),
		invoked("c1", "Bash", `{"command":"go test ./..."}`),
		returned("c1", "error: connection refused", false))

	if got := signalsOf(messages...).Severity; got != SeverityHard {
		t.Fatalf("severity = %v, want hard: the id's latest call is not a retrieval", got)
	}
}

// The count check is a boundary question: "3 failed" is a failure count and "0
// failed" is a clean summary, and "errored" is not a count site at all.
func TestNonzeroFailureCount(t *testing.T) {
	cases := []struct {
		text string
		want bool
	}{
		{"0 failed", false},
		{"3 failed", true},
		{"0 errors", false},
		{"2 failures", true},
		{"1\nfailed", true},
		{"errored out", false},
		{"no failures reported", false},
	}
	for _, tc := range cases {
		if got := hasNonzeroFailureCount(tc.text); got != tc.want {
			t.Errorf("hasNonzeroFailureCount(%q) = %v, want %v", tc.text, got, tc.want)
		}
	}
}

func itoa(value int) string {
	if value == 0 {
		return "0"
	}
	digits := ""
	for value > 0 {
		digits = string(rune('0'+value%10)) + digits
		value /= 10
	}
	return digits
}
