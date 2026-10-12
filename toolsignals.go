package libsygo

import (
	"encoding/json"
	"strings"
)

// ToolSignals are what a conversation's tool activity says about it.
//
// Every field is derived from the messages alone: this reads what the agent has been
// doing and how it has been going, which is why the signal router needs no network
// call and why it can decide before the next model is chosen.
type ToolSignals struct {
	// Severity is the worst failure in the recent window, so one error persists
	// through the turns spent recovering rather than clearing the moment the next
	// result is clean.
	Severity float32

	// RepeatedFailure is a hard-or-critical failure that appeared at least twice in
	// the recent window.
	RepeatedFailure bool

	// NoErrorStreak counts consecutive clean results back from the most recent. Zero
	// when the last result failed.
	NoErrorStreak uint32

	// Counts over the whole request.
	EditCount      uint32
	WriteCount     uint32
	ReadCount      uint32
	TodowriteCount uint32
	NewCount       uint32

	// Counts within the recent window, which is what "what is the agent doing right
	// now" means.
	RecentEditCount      uint32
	RecentWriteCount     uint32
	RecentReadCount      uint32
	RecentTodowriteCount uint32
	RecentNewCount       uint32

	// PureBashStreak counts trailing calls whose meaning this vocabulary does not
	// recognize: activity without a recognized purpose.
	PureBashStreak uint32

	// TestsPassed reports a pass after the latest failure in the window.
	TestsPassed bool

	// ToolResultCount counts result blocks, so a message batching several counts
	// several.
	ToolResultCount uint32

	// AssistantTurnCount and TurnDepth are the two depth proxies: assistant turns
	// alone, and every message. TurnDepth is wire-format dependent, since providers
	// batch tool results into different numbers of messages.
	AssistantTurnCount uint32
	TurnDepth          uint32

	// Compacted reports a context-compaction summary anywhere in the conversation.
	// The summary stays in the prefix on every later turn, so this self-latches.
	Compacted bool
}

// observedToolCall is one tool invocation, reduced to what classification needs.
type observedToolCall struct {
	name        string
	bareName    string
	hasBareName bool
	// command is the invocation's command field, empty when it has none.
	command string
	// isRetrieval marks a call whose output is retrieved content, which is exempt
	// from the error scan.
	isRetrieval bool
}

// Compaction markers. Claude Code's is a plain substring; Hermes' must begin a line,
// because Hermes merges its summary into existing text and a marker named in running
// prose is not a compaction.
const compactionMarker = "session is being continued"

var hermesCompactionMarkers = []string{
	"[context compaction \u2014 reference only]",
	"[still in progress \u2014 this is the active request, restated after the compaction boundary",
}

// Test-output vocabulary. The pass phrases are deliberately conservative — a false
// positive clears a capable hold, which would hand an unfinished task back too early.
var (
	testPassPhrases    = []string{" passed", "passed in", "tests passed", "all tests passed", "test ok", "test result: ok", "passed.\n", "tests pass", "\nok ", "\u2713 "}
	testFailurePhrases = []string{"\u2717 ", "fatal:", "assertionerror", "error:"}
	// Count-prefixed failure words, tripped only by a non-zero integer before them so
	// cargo's "0 failed" and go's "0 errors" stay clean.
	numericFailureKeywords = []string{"failed", "failure", "failures", "errors", "error"}
)

// extractToolSignals reads a conversation's tool activity.
//
// recentWindow bounds the recent counts, the windowed severity and test detection;
// zero or less is treated as the built-in default, except for the recent counts, which
// a zero window empties. That asymmetry is upstream's and is kept: severity is never
// computed over an empty window, because a zero-width window would report a clean
// history for a conversation that just failed.
func extractToolSignals(request LlmRequest, recentWindow int, semantics ToolSemantics) ToolSignals {
	var (
		results          []toolResultText
		calls            []observedToolCall
		retrievalCallIDs = map[string]bool{}
		compacted        bool
		resultCount      int
		assistantTurns   int
	)

	for _, message := range request.Messages {
		if message.Role == RoleAssistant {
			assistantTurns++
		}
		for _, block := range message.Content {
			switch block.Kind {
			case ContentToolCall:
				calls = append(calls, observeToolCall(block.Call, semantics, retrievalCallIDs))

			case ContentToolResult:
				// Counted before the empty-text filter: an empty result is still a
				// result, and the count is what a host sees.
				resultCount++
				result := block.Result
				isError := result.IsError != nil && *result.IsError
				text := joinText(result.Content, "\n")

				if retrievalCallIDs[result.ToolCallID] {
					// A retrieval result is not evidence that anything broke: a read
					// that finds nothing is a normal step. It is still a failure if
					// the tool says so, which is why the exit status is read rather
					// than assumed.
					if isMissingFile(text) || (!isError && !shellResultFailed(text)) {
						// The window slot is kept even though the content is dropped,
						// so dropping retrieved text does not shift what the window
						// covers.
						if text != "" {
							results = append(results, toolResultText{})
						}
						continue
					}
				}
				// An explicit failure is a signal even with no text to read.
				if text != "" || isError {
					results = append(results, toolResultText{text: text, isError: isError})
				}

			case ContentText:
				lower := strings.ToLower(block.Text)
				compacted = compacted || isCompaction(lower)

			case ContentUnknown:
				// The Responses wire format represents a tool result as its own item
				// rather than a message, and the failure shows up in provider-specific
				// fields instead of in text. Both are read here because nowhere else
				// will: this block is not conversation anyone can read.
				if block.Provider != "openai_responses" {
					continue
				}
				switch responsesItemType(block.Raw) {
				case "apply_patch_call_output":
					resultCount++
					item, _ := objectFields(block.Raw)
					status, _ := stringValue(rawField(item, "status"))
					results = append(results, toolResultText{
						text:    stringOrEmpty(item, "output"),
						isError: status == "failed",
					})
				case "shell_call_output":
					resultCount++
					results = append(results, decodeShellCallOutput(block.Raw))
				}
			}
		}
	}

	signal := buildToolSignals(results, calls, len(request.Messages), recentWindow, semantics)
	signal.Compacted = compacted
	signal.ToolResultCount = uint32(resultCount)
	signal.AssistantTurnCount = uint32(assistantTurns)
	return signal
}

// observeToolCall classifies one invocation and records its retrieval status against
// its call id, so the result that answers it can be exempted.
func observeToolCall(call *ToolCall, semantics ToolSemantics, retrievalCallIDs map[string]bool) observedToolCall {
	if call == nil {
		return observedToolCall{}
	}

	bareName, hasBareName := mcpToolName(call.Name)
	command, _ := CommandField(call.Arguments)

	// The joined name decides, and a bare name is consulted only when the joined one
	// told us nothing useful — so a namespaced tool configured as an observation still
	// counts when its bare name is a retrieval tool, such as `mcp__files__read`.
	full := ClassifyToolCall(call.Name, command, semantics)
	name := call.Name
	if (full == SemanticUnknown || full == SemanticObserve) && hasBareName {
		name = bareName
	}
	isRetrieval := IsRetrievalTool(name, command)

	// A reused id links to its latest call, so an id that was a read and is now a
	// write stops being exempt.
	if call.ID != "" {
		if isRetrieval {
			retrievalCallIDs[call.ID] = true
		} else {
			delete(retrievalCallIDs, call.ID)
		}
	}

	return observedToolCall{
		name:        call.Name,
		bareName:    bareName,
		hasBareName: hasBareName,
		command:     command,
		isRetrieval: isRetrieval,
	}
}

// responsesItemType is the type discriminator of a carried Responses item.
func responsesItemType(raw json.RawMessage) string {
	item, err := objectFields(raw)
	if err != nil {
		return ""
	}
	kind, _ := stringValue(rawField(item, "type"))
	return kind
}

// decodeShellCallOutput reads a shell call's structured result.
//
// Each output carries its own stdout and stderr and an outcome naming why it stopped. A
// non-zero exit contributes the "returned non-zero" phrasing so the ordinary severity
// table sees it, and counts as a failure only when it also wrote to stderr -- a command
// that exits non-zero with nothing to say is a soft signal, not a diagnosis.
func decodeShellCallOutput(raw json.RawMessage) toolResultText {
	item, err := objectFields(raw)
	if err != nil {
		return toolResultText{}
	}
	var elements []json.RawMessage
	if err := json.Unmarshal(rawField(item, "output"), &elements); err != nil {
		return toolResultText{}
	}

	var (
		texts   []string
		isError bool
	)
	for _, element := range elements {
		output, err := objectFields(element)
		if err != nil {
			continue
		}
		for _, name := range []string{"stdout", "stderr"} {
			if text, ok := stringValue(rawField(output, name)); ok && text != "" {
				texts = append(texts, text)
			}
		}
		outcome, err := objectFields(rawField(output, "outcome"))
		if err != nil {
			continue
		}
		switch kind, _ := stringValue(rawField(outcome, "type")); kind {
		case "timeout":
			isError = true
		case "exit":
			if code, ok := int64Value(rawField(outcome, "exit_code")); ok && code != 0 {
				texts = append(texts, "returned non-zero")
				stderr, _ := stringValue(rawField(output, "stderr"))
				isError = isError || strings.TrimSpace(stderr) != ""
			}
		}
	}
	return toolResultText{text: strings.Join(texts, "\n"), isError: isError}
}

// buildToolSignals turns the collected results and calls into the signal set.
func buildToolSignals(
	results []toolResultText,
	calls []observedToolCall,
	turnDepth int,
	recentWindow int,
	semantics ToolSemantics,
) ToolSignals {
	severityWindow := recentWindow
	if severityWindow < 1 {
		severityWindow = 1
	}

	// Windowed severity is the worst failure in the window, not the last one, so a fix
	// written a turn or two after an error still routes on the error rather than
	// flapping straight back to the cheap tier.
	var (
		severity     float32
		fingerprints []string
		repeated     bool
	)
	for _, result := range results[max(0, len(results)-severityWindow):] {
		if candidate := severityOf(result); candidate > severity {
			severity = candidate
		}
		if fingerprint, ok := failureFingerprint(result.text, result.isError); ok {
			if contains(fingerprints, fingerprint) {
				repeated = true
			}
			fingerprints = append(fingerprints, fingerprint)
		}
	}

	recentStart := max(0, len(calls)-recentWindow)
	var (
		writeCount, editCount, readCount, todowriteCount, newCount      uint32
		recentWrite, recentEdit, recentRead, recentTodowrite, recentNew uint32
		pureBashStreak                                                  uint32
	)
	streakOpen := true
	for index := len(calls) - 1; index >= 0; index-- {
		call := calls[index]
		// A retrieval call is an observation whatever its name says, because the read
		// check parses the command and so knows more than the name does.
		var semantic ToolSemantic
		if call.isRetrieval {
			semantic = SemanticObserve
		} else {
			semantic = ClassifyToolCall(call.name, call.command, semantics)
		}
		if semantic == SemanticUnknown && call.hasBareName {
			semantic = ClassifyToolCall(call.bareName, call.command, semantics)
		}

		if streakOpen {
			if semantic == SemanticUnknown {
				pureBashStreak++
			} else {
				streakOpen = false
			}
		}

		inWindow := index >= recentStart
		switch semantic {
		case SemanticWrite:
			writeCount++
			if inWindow {
				recentWrite++
			}
		case SemanticEdit:
			editCount++
			if inWindow {
				recentEdit++
			}
		case SemanticObserve:
			readCount++
			if inWindow {
				recentRead++
			}
		case SemanticPlan:
			todowriteCount++
			if inWindow {
				recentTodowrite++
			}
		case SemanticNew:
			newCount++
			if inWindow {
				recentNew++
			}
		}
	}

	return ToolSignals{
		Severity:             severity,
		RepeatedFailure:      repeated,
		NoErrorStreak:        noErrorStreak(results),
		EditCount:            editCount,
		WriteCount:           writeCount,
		ReadCount:            readCount,
		TodowriteCount:       todowriteCount,
		NewCount:             newCount,
		RecentEditCount:      recentEdit,
		RecentWriteCount:     recentWrite,
		RecentReadCount:      recentRead,
		RecentTodowriteCount: recentTodowrite,
		RecentNewCount:       recentNew,
		PureBashStreak:       pureBashStreak,
		TestsPassed:          detectTestsPassed(results, recentWindow),
		TurnDepth:            uint32(turnDepth),
	}
}

// severityOf is the severity of one collected result, with an explicit failure floor
// at hard: a tool that says it failed is a failure even when nothing readable came
// back to explain it.
func severityOf(result toolResultText) float32 {
	severity, _ := classifyText(result.text)
	if result.isError {
		severity = maxSeverity(severity, SeverityHard)
	}
	return severity
}

// shellResultFailed reads a terminal tool's structured result, which is how a
// retrieval that *did* fail announces it — the exemption is for reads that found
// nothing, not for reads that broke.
func shellResultFailed(text string) bool {
	if hasNonzeroExitStatus(strings.ToLower(text)) {
		return true
	}
	result := firstJSONObject(text)
	if result == nil {
		return false
	}
	if _, ok := result["output"].(string); !ok {
		return false
	}
	if _, ok := result["exit_code"]; !ok {
		return false
	}
	if code, ok := result["exit_code"].(float64); ok && code != 0 {
		return true
	}
	if success, ok := result["success"].(bool); ok && !success {
		return true
	}
	if message, ok := result["error"].(string); ok && strings.TrimSpace(message) != "" {
		return true
	}
	return false
}

// isMissingFile reports a read that found nothing, which is a normal step rather than
// a failure.
func isMissingFile(text string) bool {
	lower := strings.ToLower(text)
	return strings.Contains(lower, "no such file or directory") ||
		strings.Contains(lower, "file does not exist")
}

// detectTestsPassed reports a pass phrase after the latest failure in the window.
//
// Anchoring on the latest failure is the point: a pass before the failure says nothing
// about whether the failure was fixed.
func detectTestsPassed(results []toolResultText, recentWindow int) bool {
	window := recentWindow
	if window < 1 {
		window = 1
	}
	recent := results[max(0, len(results)-window):]

	afterLatestFailure := recent
	for index := len(recent) - 1; index >= 0; index-- {
		if recent[index].isError {
			afterLatestFailure = recent[index+1:]
			break
		}
		if severity, _ := classifyText(recent[index].text); severity > 0 {
			afterLatestFailure = recent[index+1:]
			break
		}
	}

	for _, result := range afterLatestFailure {
		lower := strings.ToLower(result.text)
		if !containsAny(lower, testPassPhrases) {
			continue
		}
		if containsAny(lower, testFailurePhrases) || hasNonzeroFailureCount(lower) {
			continue
		}
		return true
	}
	return false
}

// hasNonzeroFailureCount reports a failure word preceded by a non-zero integer, which
// is what separates "3 failed" from a clean run's "0 failed".
func hasNonzeroFailureCount(lower string) bool {
	for _, keyword := range numericFailureKeywords {
		cursor := 0
		for {
			relative := strings.Index(lower[cursor:], keyword)
			if relative < 0 {
				break
			}
			start := cursor + relative
			end := start + len(keyword)

			// A word boundary after the keyword, so "errored" is not a failure count.
			boundary := end >= len(lower) || !isASCIIAlphanumeric(lower[end])
			if boundary {
				before := strings.TrimRightFunc(lower[:start], func(r rune) bool {
					return r == ' ' || r == '\t' || r == '\n' || r == '\r'
				})
				digits := ""
				for index := len(before) - 1; index >= 0; index-- {
					c := before[index]
					if c < '0' || c > '9' {
						break
					}
					digits = string(c) + digits
				}
				if digits != "" && strings.Trim(digits, "0") != "" {
					return true
				}
			}
			cursor = end
		}
	}
	return false
}

// isCompaction reports a context-compaction summary in already-lowercased text.
func isCompaction(lower string) bool {
	if strings.Contains(lower, compactionMarker) {
		return true
	}
	for _, line := range strings.Split(lower, "\n") {
		trimmed := strings.TrimLeft(line, " \t")
		for _, marker := range hermesCompactionMarkers {
			if strings.HasPrefix(trimmed, marker) {
				return true
			}
		}
	}
	return false
}

// mcpToolName returns the bare tool name of an MCP-qualified tool, so
// `mcp__server__read_file` is recognized as `read_file`.
func mcpToolName(name string) (string, bool) {
	rest, ok := strings.CutPrefix(name, "mcp__")
	if !ok {
		return "", false
	}
	_, tool, ok := strings.Cut(rest, "__")
	if !ok || tool == "" {
		return "", false
	}
	return tool, true
}

func isASCIIAlphanumeric(c byte) bool {
	return c >= '0' && c <= '9' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z'
}
