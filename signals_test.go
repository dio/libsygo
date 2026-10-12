package libsygo

import "testing"

// These cases are upstream's, not invented here. The severity bands are measured
// against trajectories, so the only meaningful test of this table is the one that
// says what each real output scores — and these are the outputs upstream asserts on.
func TestSeverityOf(t *testing.T) {
	cases := []struct {
		name string
		text string
		want float32
	}{
		{"clean prose", "everything went fine", 0},
		{"traceback", "Traceback (most recent call last):\n  ValueError", SeverityHard},
		{"out of memory", "Out of memory: kill process 1234", SeverityCritical},
		{"connection refused", "Connection refused on port 8000", SeverityHard},
		// The strongest signature wins.
		{"soft and hard together", "exit code 1\nTraceback (most recent call last):", SeverityHard},

		// Codex writes an exit line on clean results, so the phrase alone is not a
		// failure — the status has to be parsed.
		{"codex clean exit", "Chunk ID: abc\nProcess exited with code 0\nOutput:\nok", 0},
		{"codex non-zero exit", "Process exited with code 1", SeveritySoft},
		{"codex 127", "Process exited with code 127", SeveritySoft},
		{"exit code colon", "exit code: 2", SeveritySoft},
		{"exit status", "exit status 3", SeveritySoft},
		{"exited with status", "exited with status 9", SeveritySoft},

		// A zero status inside otherwise noisy output stays clean.
		{"zero after output", "Process exited with code 0\nOutput:\nall good", 0},

		// A named failure in output that exited zero is still a failure.
		{"compile error", "Process running with session ID 12\nOutput:\nerror[E0509]: cannot move out", SeverityHard},
		{"runtime exception", "Process exited with code 0\nOutput:\nTypeError: value is undefined\n    at main.js:1:2", SeverityHard},
		{"runtime panic", "Process running with session ID 13\nOutput:\npanic: runtime error: index out of range\n\ngoroutine 6 [running]:", SeverityHard},
		{"patch failure", "Process exited with code 0\nOutput:\nerror: patch failed: src/lib.rs:4\nerror: src/lib.rs: patch does not apply", SeverityHard},

		// Structured results, as a harness that returns JSON reports them.
		{"structured exit 7", `{"output":"","exit_code":7,"error":null}`, SeveritySoft},
		{"structured exit -1", `{"output":"","exit_code":-1,"error":null}`, SeveritySoft},
		{"structured success false", `{"success":false,"error":"No matching text"}`, SeverityHard},
		{"structured success false, no message", `{"success":false,"error":null}`, SeverityHard},
		{"structured error field", `{"error":"Overwrite refused","stale_write_blocked":true}`, SeverityHard},
		{"structured oom", `{"output":"out of memory","exit_code":1}`, SeverityCritical},
		{"structured clean", `{"output":"done","exit_code":0,"error":null}`, 0},
		{"structured blank error", `{"success":true,"error":"  "}`, 0},
		{"structured null exit", `{"output":"running","exit_code":null,"error":null}`, 0},
		{"nested success is not success", `{"output":{"success":false},"exit_code":0}`, 0},

		// Hermes appends a warning after the result; only the first value is read.
		{"trailing warning ignored", "{\"output\":\"\",\"exit_code\":7,\"error\":null}\n\n[Tool loop warning: repeated identical call]", SeveritySoft},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := SeverityOf(tc.text); got != tc.want {
				t.Fatalf("SeverityOf(%q) = %v, want %v", tc.text, got, tc.want)
			}
		})
	}
}

func TestClassifyTextNamesThePatternsThatFired(t *testing.T) {
	cases := []struct {
		text string
		want []string
	}{
		{"everything went fine", nil},
		{"Traceback (most recent call last):", []string{"traceback"}},
		{"Process exited with code 1", []string{"exit_nonzero"}},
		{"Process exited with code 0", nil},
		{"Out of memory: kill process 1234", []string{"oom"}},
	}

	for _, tc := range cases {
		t.Run(tc.text, func(t *testing.T) {
			_, patterns := classifyText(tc.text)
			if len(patterns) != len(tc.want) {
				t.Fatalf("patterns = %v, want %v", patterns, tc.want)
			}
			for index := range patterns {
				if patterns[index] != tc.want[index] {
					t.Fatalf("patterns = %v, want %v", patterns, tc.want)
				}
			}
		})
	}
}

// A source file that merely names an exception is not a failure: the *structured*
// detector requires the name at a line start and a stack frame beside it.
func TestSourceTextIsNotAFailure(t *testing.T) {
	clean := []string{
		"// TODO: raise a ValueError when the input is empty",
		"the docs mention error: something in passing",
	}
	for _, text := range clean {
		if got := SeverityOf(text); got != 0 {
			t.Errorf("SeverityOf(%q) = %v, want 0", text, got)
		}
	}

	// The pattern table is a plain substring match, so this one *does* score — a
	// Python `except ValueError:` clause reads as a failure. Upstream behaves the
	// same way, and it is the deliberate side of the trade: requiring a stack frame
	// would miss real exceptions that arrive without one.
	if got := SeverityOf("    except ValueError:\n        pass"); got != SeverityHard {
		t.Fatalf("SeverityOf(except clause) = %v, want %v", got, SeverityHard)
	}
}

// Two failures are the same failure when they differ only in the values that change
// between retries: the line number and the directory.
func TestFailureFingerprintIgnoresVolatileValues(t *testing.T) {
	first, ok := failureFingerprint("error[E0308]: mismatched types at /tmp/a/src/lib.rs:12", false)
	if !ok {
		t.Fatal("a compiling error should have a fingerprint")
	}
	second, _ := failureFingerprint("error[E0308]: mismatched types at /tmp/b/src/lib.rs:47", false)
	if first != second {
		t.Fatalf("the same failure fingerprinted differently:\n  %s\n  %s", first, second)
	}

	other, _ := failureFingerprint("error[E0509]: cannot move out", false)
	if other == first {
		t.Fatal("a different failure fingerprinted the same")
	}
}

// A soft exit only counts as a repeated mistake when the provider says it failed,
// because a command that merely exited non-zero is not the same as a mistake.
func TestFailureFingerprintNeedsAFlagForSoftExits(t *testing.T) {
	if _, ok := failureFingerprint("Process exited with code 1", false); ok {
		t.Fatal("a soft exit without a failure flag should not fingerprint")
	}
	if _, ok := failureFingerprint("Process exited with code 1", true); !ok {
		t.Fatal("a soft exit the provider flagged should fingerprint")
	}
}

func TestNoErrorStreak(t *testing.T) {
	clean := []toolResultText{{text: "ok"}, {text: "applied"}, {text: "ok"}}
	if got := noErrorStreak(clean); got != 3 {
		t.Fatalf("streak = %d, want 3", got)
	}

	// A failure ends the streak, and the results before it do not count.
	broken := []toolResultText{{text: "ok"}, {text: "error: connection refused"}}
	if got := noErrorStreak(broken); got != 0 {
		t.Fatalf("streak = %d, want 0", got)
	}

	// An explicit failure ends it even with no diagnostic text.
	flagged := []toolResultText{{text: ""}, {text: "", isError: true}}
	if got := noErrorStreak(flagged); got != 0 {
		t.Fatalf("streak = %d, want 0", got)
	}

	if got := noErrorStreak(nil); got != 0 {
		t.Fatalf("streak = %d, want 0", got)
	}
}
