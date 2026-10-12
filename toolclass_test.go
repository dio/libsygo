package libsygo

import (
	"strings"
	"testing"
)

// The classification order is the behaviour, not a detail of it: a built-in name
// wins over a configured one, an editor tool's action comes from its command, write
// and edit redirection beats a read-like operand, and a Bash command that matches
// nothing is Unknown rather than assumed.
func TestClassifyToolCall(t *testing.T) {
	none := ToolSemantics{}

	cases := []struct {
		name    string
		tool    string
		command string
		want    ToolSemantic
	}{
		// Dedicated tools reach their meaning from the name alone.
		{"read tool", "Read", "", SemanticObserve},
		{"write tool", "Write", "", SemanticWrite},
		{"edit tool", "Edit", "", SemanticEdit},
		{"codex edit tool", "apply_patch", "", SemanticEdit},
		{"hermes edit tool", "patch", "", SemanticEdit},
		{"plan tool", "TodoWrite", "", SemanticPlan},
		{"codex plan tool", "update_plan", "", SemanticPlan},
		{"case does not matter", "READ_FILE", "", SemanticObserve},

		// An editor tool's action is its command argument.
		{"editor viewing reads", "text_editor", "view", SemanticObserve},
		{"editor editing writes", "text_editor", "str_replace", SemanticEdit},

		// A bash command's meaning comes from the command.
		{"grep inspects", "Bash", "grep -rn cutoff .", SemanticObserve},
		{"ls inspects", "Bash", "ls -la", SemanticObserve},
		{"git status inspects", "Bash", "git status", SemanticObserve},
		{"rm edits", "Bash", "rm -rf ./tmp", SemanticEdit},
		{"mv edits", "Bash", "mv a b", SemanticEdit},
		{"sed in place edits", "Bash", "sed -i '' s/a/b/ router.go", SemanticEdit},
		{"git apply edits", "Bash", "git apply changes.patch", SemanticEdit},
		{"cat redirect writes", "Bash", "cat > out.txt", SemanticWrite},
		{"tee writes", "Bash", "tee out.txt", SemanticWrite},
		{"mkdir writes", "Bash", "mkdir -p out", SemanticWrite},

		// Redirection trumps a read-like operand: the file is written either way.
		{"grep redirecting writes", "Bash", "grep x . > hits.txt", SemanticWrite},

		// A command this vocabulary has no meaning for is Unknown, not a guess. That
		// is what makes the pure-bash streak meaningful.
		{"go test is unknown", "Bash", "go test ./...", SemanticUnknown},
		{"an unfamiliar tool is unknown", "mystery_probe", "whatever", SemanticUnknown},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ClassifyToolCall(tc.tool, tc.command, none); got != tc.want {
				t.Fatalf("ClassifyToolCall(%q, %q) = %q, want %q", tc.tool, tc.command, got, tc.want)
			}
		})
	}
}

// A route's vocabulary adds tools. It cannot redefine a built-in, because the
// recorded signals were measured against the built-ins.
func TestConfiguredSemanticsAreAdditive(t *testing.T) {
	semantics := ToolSemantics{
		Observe: []string{"inspect_cluster"},
		Mutate:  []string{"deploy_service"},
	}

	if got := ClassifyToolCall("inspect_cluster", "", semantics); got != SemanticObserve {
		t.Fatalf("configured observe = %q", got)
	}
	// A configured mutation counts as a write: the scorer treats writes and edits
	// alike, so the public signal shape is unchanged.
	if got := ClassifyToolCall("deploy_service", "", semantics); got != SemanticWrite {
		t.Fatalf("configured mutate = %q", got)
	}
	if got := ClassifyToolCall("Bash", "", semantics); got != SemanticUnknown {
		t.Fatalf("bash with no command = %q", got)
	}

	// Built-in names win even when a route also lists them.
	shadowing := ToolSemantics{Observe: []string{"edit"}}
	if got := ClassifyToolCall("edit", "", shadowing); got != SemanticEdit {
		t.Fatalf("a built-in was reclassified: %q", got)
	}
}

func TestSemanticsValidation(t *testing.T) {
	cases := []struct {
		name      string
		semantics ToolSemantics
		wantErr   string
	}{
		{"empty", ToolSemantics{}, ""},
		{"additive", ToolSemantics{Observe: []string{"inspect_cluster"}}, ""},
		{"reclassifies a built-in", ToolSemantics{Mutate: []string{"bash"}},
			"already has built-in semantics"},
		{"reclassifies a read tool", ToolSemantics{Observe: []string{"read"}},
			"already has built-in semantics"},
		{"empty name", ToolSemantics{Plan: []string{"  "}}, "empty tool name"},
		{"ambiguous", ToolSemantics{Observe: []string{"probe"}, Mutate: []string{"PROBE"}},
			"appears in both"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.semantics.Validate()
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("want an error containing %q", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error = %q, want it to mention %q", err, tc.wantErr)
			}
		})
	}
}

// Retrieval decides whether a result is exempt from the error scan, so it is a
// stricter question than "is this an inspection": a command qualifies only when
// every segment is one.
func TestIsRetrievalTool(t *testing.T) {
	cases := []struct {
		name    string
		tool    string
		command string
		want    bool
	}{
		{"read tool", "Read", "", true},
		{"view editor", "text_editor", "view", true},
		{"editing editor", "text_editor", "str_replace", false},
		{"grep", "Bash", "grep -rn cutoff .", true},
		{"cat", "Bash", "cat router.go", true},
		{"git log", "Bash", "git log --oneline", true},

		// Not an inspection at all.
		{"go test", "Bash", "go test ./...", false},
		{"rm", "Bash", "rm -rf ./tmp", false},

		// One non-inspection segment disqualifies the whole command.
		{"pipeline through tee", "Bash", "grep x . | tee out.txt", false},
		{"redirect", "Bash", "cat f > out.txt", false},
		// Substitutions run a program the router has not seen.
		{"command substitution", "Bash", "grep x $(pwd)", false},
		{"backticks", "Bash", "grep x `pwd`", false},
		// PowerShell quoting does not follow the POSIX rules this vocabulary uses.
		{"powershell is not parsed", "powershell", "Get-ChildItem", false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsRetrievalTool(tc.tool, tc.command); got != tc.want {
				t.Fatalf("IsRetrievalTool(%q, %q) = %v, want %v", tc.tool, tc.command, got, tc.want)
			}
		})
	}
}

func TestShellSegmentsSplitsOnlyUnquotedSeparators(t *testing.T) {
	cases := []struct {
		command string
		want    []string
	}{
		{"grep x .", []string{"grep x ."}},
		{"grep x . && ls", []string{"grep x .", "ls"}},
		{"ls; pwd", []string{"ls", "pwd"}},
		{"ls | wc -l", []string{"ls", "wc -l"}},
		// A separator inside quotes is argument text, not a separator.
		{`echo "a;b"`, []string{`echo "a;b"`}},
		// An escaped separator is argument text too.
		{`grep a\;b f`, []string{`grep a\;b f`}},
		{"  ", nil},
	}

	for _, tc := range cases {
		t.Run(tc.command, func(t *testing.T) {
			got := shellSegments(tc.command)
			if len(got) != len(tc.want) {
				t.Fatalf("shellSegments(%q) = %q, want %q", tc.command, got, tc.want)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Fatalf("segment %d = %q, want %q", i, got[i], tc.want[i])
				}
			}
		})
	}
}

// The splitter reports input it cannot parse rather than guessing, which is what the
// retrieval check depends on: an unparseable command is not an inspection.
func TestShlexSplitRejectsUnparseableInput(t *testing.T) {
	words, ok := shlexSplit(`grep -rn "a b" .`)
	if !ok {
		t.Fatal("a quoted argument should parse")
	}
	if len(words) != 4 || words[2] != "a b" {
		t.Fatalf("words = %q", words)
	}

	if _, ok := shlexSplit(`grep "unclosed`); ok {
		t.Fatal("an unclosed quote should not parse")
	}
	if _, ok := shlexSplit(`grep trailing\`); ok {
		t.Fatal("a trailing backslash should not parse")
	}
}

// A tool's command field, whatever the harness calls it.
func TestCommandField(t *testing.T) {
	cases := []struct {
		name      string
		arguments string
		want      string
		wantOK    bool
	}{
		{"command", `{"command":"ls -la"}`, "ls -la", true},
		{"cmd", `{"cmd":"ls"}`, "ls", true},
		{"input", `{"input":"ls"}`, "ls", true},
		{"argv", `{"command":["ls","-la"]}`, "ls -la", true},
		{"no command", `{"path":"router.go"}`, "", false},
		{"empty argv", `{"command":[]}`, "", false},
		{"not an object", `[1,2]`, "", false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := CommandField([]byte(tc.arguments))
			if ok != tc.wantOK || got != tc.want {
				t.Fatalf("CommandField(%s) = %q, %v; want %q, %v", tc.arguments, got, ok, tc.want, tc.wantOK)
			}
		})
	}
}
