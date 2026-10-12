package libsygo

import (
	"encoding/json"
	"strings"
)

// What a tool result's text means: how badly it failed, and whether two failures are
// the same failure.
//
// The bands are measured, not derived. They were tuned against trajectories, so a
// plausible reimplementation of the *structure* would classify differently on exactly
// the outputs that decide routing — a go test summary, a cargo diagnostic, a shell
// exit line. The pattern table below is upstream's, and the tests beside it are
// upstream's own expectations rather than ones invented here.

// The severity bands.
const (
	// SeveritySoft is a plain non-zero exit with nothing recognizable in the text.
	SeveritySoft float32 = 0.3
	// SeverityHard is a named failure: an exception, a compile error, a patch that
	// did not apply.
	SeverityHard float32 = 0.7
	// SeverityCritical is a failure the process could not continue through.
	SeverityCritical float32 = 1.0
)

// errorPattern is one named failure signature: the severity it carries and the
// lower-cased substrings that trigger it. Any hit fires the pattern.
type errorPattern struct {
	name       string
	severity   float32
	substrings []string
}

var errorPatterns = []errorPattern{
	{"oom", SeverityCritical, []string{"out of memory", "memoryerror", "cannot allocate memory"}},
	{"connection_refused", SeverityHard, []string{
		"connection refused", "connectionrefusederror", "econnrefused"}},
	{"traceback", SeverityHard, []string{"traceback (most recent call last)"}},
	{"import_error", SeverityHard, []string{
		"modulenotfounderror:", "importerror:", "no module named "}},
	{"cmd_not_found", SeverityHard, []string{
		"command not found", "not found\n", "/usr/bin/env: "}},
	{"assertion", SeverityHard, []string{"assertionerror"}},
	{"value_error", SeverityHard, []string{"valueerror:"}},
	{"syntax_error", SeverityHard, []string{"syntaxerror:"}},
	{"timeout", SeverityHard, []string{
		"timed out", "timeouterror", "timeout expired", "deadline exceeded"}},
	{"no_such_file", SeverityHard, []string{
		"filenotfounderror:", "no such file or directory", "file does not exist"}},
	// A plain non-zero exit carries no diagnosis, so it is the weakest band.
	{"exit_nonzero", SeveritySoft, []string{"returned non-zero"}},
}

// Exit phrases must parse their numeric status rather than matching the phrase alone:
// codex writes "Process exited with code 0" on clean results.
var nonzeroExitPhrases = []string{
	"exit code", "exit status", "exited with code", "exited with status",
}

// SeverityOf reports how badly a tool result's text failed: zero when clean, and one
// of [SeveritySoft], [SeverityHard] or [SeverityCritical] otherwise.
//
// The strongest signature wins, so a command that both exited non-zero and printed a
// traceback is hard rather than soft. Nothing here knows whether the provider flagged
// the result as a failure — an explicit flag floors the severity at hard, and that is
// applied by the caller, which is the only place the flag is known.
func SeverityOf(text string) float32 {
	severity, _ := classifyText(text)
	return severity
}

// classifyText reports the severity and the names of the patterns that fired.
//
// The names matter as much as the severity: they are part of a failure's identity, so
// two failures that trigger different patterns are different failures even when they
// share a band.
func classifyText(text string) (float32, []string) {
	lower := strings.ToLower(text)
	var patterns []string
	var severity float32
	for _, pattern := range errorPatterns {
		if containsAny(lower, pattern.substrings) {
			patterns = append(patterns, pattern.name)
			severity = maxSeverity(severity, pattern.severity)
		}
	}

	// Hermes appends a loop warning after the JSON result, so only the first value is
	// read and trailing text is ignored.
	result := firstJSONObject(text)
	nonzeroExit, _ := result["exit_code"].(float64)
	success, hasSuccess := result["success"].(bool)
	errorText, hasError := result["error"].(string)
	toolError := (hasSuccess && !success) || (hasError && strings.TrimSpace(errorText) != "")

	if (nonzeroExit != 0 || hasNonzeroExitStatus(lower)) && !contains(patterns, "exit_nonzero") {
		patterns = append(patterns, "exit_nonzero")
		severity = maxSeverity(severity, SeveritySoft)
	}

	// Structured failure fields and text diagnostics are both hard: each names a
	// failure rather than merely reporting one.
	for _, check := range []struct {
		name    string
		matched bool
	}{
		{"tool_error", toolError},
		{"compile_error", hasCompilerDiagnostic(lower)},
		{"runtime_exception", hasRuntimeException(lower)},
		{"runtime_panic", hasRuntimePanic(lower)},
		{"patch_error", hasPatchFailure(lower)},
	} {
		if check.matched && !contains(patterns, check.name) {
			patterns = append(patterns, check.name)
			severity = maxSeverity(severity, SeverityHard)
		}
	}
	return severity, patterns
}

// firstJSONObject reads the first JSON value in text, when it is an object.
//
// A result that is not JSON, or is JSON of another shape, yields nothing rather than
// an error: the text patterns have already had their say, and a provider is free to
// return prose.
func firstJSONObject(text string) map[string]any {
	decoder := json.NewDecoder(strings.NewReader(text))
	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil
	}
	object, ok := value.(map[string]any)
	if !ok {
		return nil
	}
	return object
}

// hasCompilerDiagnostic recognizes a compiler that gave up, in the two shapes the
// toolchains use: a summary line, or a numbered diagnostic code.
func hasCompilerDiagnostic(lower string) bool {
	for _, line := range strings.Split(lower, "\n") {
		line = strings.TrimLeft(line, " \t")
		switch line {
		case "compilation failed", "error: compilation failed", "error: could not compile":
			return true
		}
		if strings.HasPrefix(line, "error: could not compile ") {
			return true
		}
		rest, found := strings.CutPrefix(line, "error[e")
		if !found {
			continue
		}
		code, _, found := strings.Cut(rest, "]:")
		if !found || code == "" {
			continue
		}
		if allDigits(code) {
			return true
		}
	}
	return false
}

// hasRuntimeException requires both a named exception and a stack frame: the name
// alone appears in source code, and a source listing is not a failure.
func hasRuntimeException(lower string) bool {
	named := false
	for _, line := range strings.Split(lower, "\n") {
		line = strings.TrimLeft(line, " \t")
		for _, prefix := range []string{
			"typeerror:", "referenceerror:", "rangeerror:", "runtimeerror:",
			"keyerror:", "attributeerror:",
		} {
			if strings.HasPrefix(line, prefix) {
				named = true
				break
			}
		}
	}
	return named && (strings.Contains(lower, "\n    at ") || strings.Contains(lower, "\n  at "))
}

func hasRuntimePanic(lower string) bool {
	panicked := false
	for _, line := range strings.Split(lower, "\n") {
		if strings.HasPrefix(strings.TrimLeft(line, " \t"), "panic: runtime error:") {
			panicked = true
			break
		}
	}
	return panicked && (strings.Contains(lower, "\ngoroutine ") || strings.Contains(lower, "[signal sig"))
}

func hasPatchFailure(lower string) bool {
	for _, line := range strings.Split(lower, "\n") {
		line = strings.TrimLeft(line, " \t")
		if strings.HasPrefix(line, "error: patch failed:") ||
			strings.HasPrefix(line, "patch failed:") ||
			strings.Contains(line, ": patch does not apply") ||
			strings.HasPrefix(line, "invalid context") {
			return true
		}
	}
	return false
}

// hasNonzeroExitStatus reports a supported exit phrase followed by a non-zero status.
func hasNonzeroExitStatus(lower string) bool {
	for _, phrase := range nonzeroExitPhrases {
		if phraseFollowedByNonzeroInteger(lower, phrase) {
			return true
		}
	}
	return false
}

// phraseFollowedByNonzeroInteger matches "exit code/status N" spellings, skipping
// optional separators and accepting a status only when some digit is non-zero.
func phraseFollowedByNonzeroInteger(lower, phrase string) bool {
	cursor := 0
	for {
		relative := strings.Index(lower[cursor:], phrase)
		if relative < 0 {
			return false
		}
		valueStart := cursor + relative + len(phrase)
		rest := strings.TrimLeftFunc(lower[valueStart:], func(r rune) bool {
			return r == ' ' || r == '\t' || r == ':' || r == '=' || r == '\'' || r == '"' || r == '`'
		})
		digits := rest
		for index := 0; index < len(digits); index++ {
			if digits[index] < '0' || digits[index] > '9' {
				digits = digits[:index]
				break
			}
		}
		if digits != "" && strings.Trim(digits, "0") != "" {
			return true
		}
		cursor = valueStart
	}
}

// failureFingerprint is a stable identity for a material failure.
//
// Soft exits need an explicit failure flag to qualify: a command that merely exited
// non-zero is not evidence that the agent made the same mistake twice. The
// identifier is the pattern set plus normalized diagnostic text, so retrying the same
// failure is recognized while a different one is not.
func failureFingerprint(text string, isError bool) (string, bool) {
	severity, patterns := classifyText(text)
	if severity < SeverityHard && !isError {
		return "", false
	}

	lower := strings.ToLower(text)
	lines := strings.Split(lower, "\n")
	diagnostic := ""
	for _, line := range lines {
		if isFailureDiagnostic(line) {
			diagnostic = line
			break
		}
	}
	if diagnostic == "" {
		for _, line := range lines {
			if strings.TrimSpace(line) != "" {
				diagnostic = line
				break
			}
		}
	}
	return strings.Join(patterns, ",") + "|" + normalizeFailureText(diagnostic), true
}

func isFailureDiagnostic(line string) bool {
	line = strings.TrimSpace(line)
	for _, marker := range []string{
		"error", "exception", "panic", "failed", "timed out", "timeout",
		"connection refused", "cannot allocate memory", "out of memory", "not found",
	} {
		if strings.Contains(line, marker) {
			return true
		}
	}
	return false
}

// normalizeFailureText removes the values that change between retries while keeping
// the wording that distinguishes one failure from another, so a retry at a different
// line in a different directory is still the same failure.
func normalizeFailureText(text string) string {
	var normalized strings.Builder
	for _, word := range strings.Fields(text) {
		if normalized.Len() > 0 {
			normalized.WriteByte(' ')
		}
		if strings.HasPrefix(word, "/") || strings.Contains(word, "/src/") ||
			strings.Contains(word, "/tmp/") {
			normalized.WriteString("<path>")
			continue
		}
		inDigits := false
		for index := 0; index < len(word); index++ {
			c := word[index]
			if c >= '0' && c <= '9' {
				if !inDigits {
					normalized.WriteByte('#')
					inDigits = true
				}
				continue
			}
			normalized.WriteByte(c)
			inDigits = false
		}
	}

	runes := []rune(normalized.String())
	if len(runes) > 240 {
		runes = runes[:240]
	}
	return string(runes)
}

// toolResultText is one tool result's text and whether the provider flagged it as a
// failure. It is what the signals are computed over, once retrieved content has been
// dropped.
type toolResultText struct {
	text    string
	isError bool
}

// noErrorStreak counts consecutive clean results back from the most recent.
//
// An explicit failure ends the streak even with no diagnostic text, which is what
// makes the streak usable as a recovery signal rather than a wording detector.
func noErrorStreak(results []toolResultText) uint32 {
	var streak uint32
	for index := len(results) - 1; index >= 0; index-- {
		if results[index].isError {
			break
		}
		if severity, _ := classifyText(results[index].text); severity > 0 {
			break
		}
		streak++
	}
	return streak
}

func maxSeverity(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func allDigits(text string) bool {
	for index := 0; index < len(text); index++ {
		if text[index] < '0' || text[index] > '9' {
			return false
		}
	}
	return text != ""
}
