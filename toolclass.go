package libsygo

import (
	"encoding/json"
	"fmt"
	"strings"
)

// What a tool invocation means, domain-neutrally.
//
// Writes and edits are separate values rather than one "mutate" with a kind,
// because the signals count them separately: a file created from nothing and a line
// changed in place are different evidence about what an agent is doing.
type ToolSemantic string

// The tool semantics.
const (
	// SemanticUnknown is a tool this build has no meaning for. It still counts as
	// activity, just not as any particular kind of it.
	SemanticUnknown ToolSemantic = "unknown"
	// SemanticObserve is a read-only lookup or inspection.
	SemanticObserve ToolSemantic = "observe"
	// SemanticPlan is explicit planning or task decomposition.
	SemanticPlan ToolSemantic = "plan"
	// SemanticNew is configured forward activity that favours neither tier.
	SemanticNew ToolSemantic = "new"
	// SemanticWrite creates or overwrites state.
	SemanticWrite ToolSemantic = "write"
	// SemanticEdit changes state in place.
	SemanticEdit ToolSemantic = "edit"
)

// Tool-name tables, ported as-is. These names are what real harnesses send, and a
// tool the table has never heard of is not the same as a tool that does not exist.
var (
	writeToolNames  = []string{"write", "create_file", "new_file", "write_file"}
	editToolNames   = []string{"edit", "multiedit", "notebookedit", "str_replace", "str_replace_based_edit_tool", "apply_patch", "text_editor", "patch"}
	editorToolNames = []string{"str_replace_based_edit_tool", "text_editor"}
	readToolNames   = []string{"read", "view", "read_file", "search_files", "glob", "grep", "find", "ls"}
	planToolNames   = []string{"todowrite", "todo_write", "todo", "update_plan", "todo_list"}
	// Names whose intent comes from a shell command rather than the tool.
	bashToolNames = []string{"bash", "shell_command", "shell", "local_shell_call", "terminal", "exec_command", "exec", "powershell"}
)

// ToolSemantics are route-scoped additions to the built-in vocabulary.
//
// Additive by construction: a built-in name cannot be reclassified, because the
// built-ins are what every other signal was measured against, and a route that could
// redefine `bash` would silently change what the recorded corpus means.
type ToolSemantics struct {
	// Observe are read-only lookup or inspection tools.
	Observe []string
	// Mutate are tools that change task or external state.
	Mutate []string
	// Plan are explicit planning or task-decomposition tools.
	Plan []string
	// New are tools that demonstrate forward activity without favouring either tier.
	New []string
}

// Validate rejects ambiguous mappings and attempts to reclassify built-in tools.
func (s ToolSemantics) Validate() error {
	type owner struct {
		name     string
		category string
	}
	var seen []owner
	categories := []struct {
		name  string
		names []string
	}{
		{"observe", s.Observe},
		{"mutate", s.Mutate},
		{"plan", s.Plan},
		{"new", s.New},
	}
	for _, category := range categories {
		for _, name := range category.names {
			if strings.TrimSpace(name) == "" {
				return fmt.Errorf("tool_semantics.%s contains an empty tool name", category.name)
			}
			lower := strings.ToLower(name)
			if isBuiltinToolName(lower) {
				return fmt.Errorf(
					"tool %q already has built-in semantics and cannot be reclassified", name)
			}
			for _, previous := range seen {
				if previous.name == lower {
					return fmt.Errorf(
						"tool %q appears in both tool_semantics.%s and tool_semantics.%s",
						name, previous.category, category.name)
				}
			}
			seen = append(seen, owner{name: lower, category: category.name})
		}
	}
	return nil
}

// classify maps a configured tool name onto a semantic, if it has one.
func (s ToolSemantics) classify(name string) (ToolSemantic, bool) {
	switch {
	case containsName(s.Observe, name):
		return SemanticObserve, true
	case containsName(s.Mutate, name):
		// Writes and edits are treated identically by the stage scorer, so a custom
		// mutation counts as a write and the public signal shape is unchanged.
		return SemanticWrite, true
	case containsName(s.Plan, name):
		return SemanticPlan, true
	case containsName(s.New, name):
		return SemanticNew, true
	}
	return SemanticUnknown, false
}

// ClassifyToolCall decides what a tool invocation means.
//
// command is the tool call's command field as one line, lowercased here to match
// upstream's matching, and empty for a tool that carries no command. A provider that
// sends the command as argv is joined by the caller; the difference only tells for an
// argument that itself contains a space inside a quoting form this vocabulary does
// not model.
//
// Built-in names and shell inference take precedence over configured semantics: the
// vocabulary adds tools, it does not redefine the ones the signals were measured on.
func ClassifyToolCall(name, command string, semantics ToolSemantics) ToolSemantic {
	lower := strings.ToLower(name)

	switch {
	case containsName(writeToolNames, lower):
		return SemanticWrite
	case containsName(editorToolNames, lower) && strings.EqualFold(command, "view"):
		return SemanticObserve
	case containsName(editToolNames, lower):
		return SemanticEdit
	case containsName(readToolNames, lower):
		return SemanticObserve
	case containsName(planToolNames, lower):
		return SemanticPlan
	}

	if containsName(bashToolNames, lower) && command != "" {
		cmd := strings.ToLower(command)
		// Write and edit redirection trumps read-like operands.
		if containsAny(cmd, bashWritePatterns) || shellCommandIsWrite(cmd) {
			return SemanticWrite
		}
		if strings.Contains(cmd, "python") && containsAny(cmd, pythonWritePatterns) {
			return SemanticWrite
		}
		if shellInvokesProgram(cmd, "node") && containsAny(cmd, javascriptWritePatterns) {
			return SemanticWrite
		}
		if containsAny(cmd, bashEditPatterns) || shellCommandIsEdit(cmd) {
			return SemanticEdit
		}
		if containsAny(cmd, bashReadPatterns) || shellCommandIsRead(cmd) {
			return SemanticObserve
		}
	}

	if semantic, ok := semantics.classify(lower); ok {
		return semantic
	}
	return SemanticUnknown
}

// IsRetrievalTool reports whether a tool call's *output* is retrieved content or
// inspection data.
//
// This is separate from classification and not a synonym for observation: it decides
// whether the result is exempt from the error scan, because a read that finds nothing
// is a normal step rather than a broken build. A shell command qualifies only when
// every segment is an inspection, since output the router has not seen can carry an
// error it would otherwise miss.
func IsRetrievalTool(name, command string) bool {
	lower := strings.ToLower(name)
	if containsName(readToolNames, lower) {
		return true
	}
	if containsName(editorToolNames, lower) && strings.EqualFold(command, "view") {
		return true
	}
	// PowerShell quoting and escapes do not follow the POSIX rules the shell
	// vocabulary uses, so its commands are not parsed.
	return lower != "powershell" &&
		containsName(bashToolNames, lower) &&
		command != "" &&
		shellIsRetrieval(command, 0)
}

// isBuiltinToolName reports whether a name already has built-in semantics.
func isBuiltinToolName(lower string) bool {
	return containsName(writeToolNames, lower) ||
		containsName(editToolNames, lower) ||
		containsName(readToolNames, lower) ||
		containsName(planToolNames, lower) ||
		containsName(bashToolNames, lower)
}

// CommandField returns a tool call's command argument as one line, reporting whether
// it had one.
//
// Harnesses name the field `command`, `cmd` or `input`; the first present wins, and
// an argv array is joined with spaces.
func CommandField(arguments []byte) (string, bool) {
	fields, err := objectFields(arguments)
	if err != nil {
		return "", false
	}
	for _, key := range []string{"command", "cmd", "input"} {
		raw, ok := field(fields, key)
		if !ok {
			continue
		}
		if text, ok := stringValue(raw); ok {
			return text, true
		}
		var parts []string
		if err := json.Unmarshal(raw, &parts); err == nil {
			joined := strings.Join(parts, " ")
			if joined != "" {
				return joined, true
			}
		}
	}
	return "", false
}

func containsName(names []string, candidate string) bool {
	for _, name := range names {
		if strings.EqualFold(name, candidate) {
			return true
		}
	}
	return false
}

func containsAny(text string, patterns []string) bool {
	for _, pattern := range patterns {
		if strings.Contains(text, pattern) {
			return true
		}
	}
	return false
}
