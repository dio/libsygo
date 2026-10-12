package libsygo

import (
	"strings"
)

// The shell-command side of the tool vocabulary.
//
// Tool signals are derived from tool *invocations*, so a command has to be
// understood before it can be classified: `grep -rn x .` is an inspection and
// `rm -rf ./tmp` is not, and both arrive as the same `Bash` tool. The tables and
// rules here are upstream's, ported as-is; they were mined from real coding-agent
// trajectories, and a plausible-looking reimplementation would classify differently
// on exactly the commands that matter.
//
// This is deliberately not a shell parser. It splits segments at unquoted
// separators and inspects each segment's leading program and flags, which is all the
// signal scoring needs and far less than a shell would do.

// Bounds recursion through nested wrappers and `sh -c` scripts.
const maxCommandDepth = 20

// Keep shell parsing bounded on the request thread. Larger commands retain ordinary
// error scanning, just like unrecognized shell syntax.
const maxRetrievalCommandBytes = 16 * 1024

var (
	bashWritePatterns = []string{
		"cat >", "cat >>", "echo >", "echo >>", "tee ", "printf >", "printf >>",
		"> /", ">> /", "<< 'eof'", "<<eof", "<<'eof'", "<< eof",
	}

	// Python file-write expressions, which only indicate a write when an interpreter
	// is running them rather than a search looking for them.
	pythonWritePatterns = []string{"write_text(", "writelines(", ".write("}

	javascriptWritePatterns = []string{
		"writefilesync(", "writefile(", "appendfilesync(", "appendfile(",
	}

	bashEditPatterns = []string{
		"sed -i", "sed --in-place", "awk -i inplace", "awk 'inplace=1'", "patch ",
		"patch -p", "perl -i", "perl -p -i", "perl -pi",
	}

	// Read-like inspections. Matched only when no write or edit pattern fires:
	// redirection and in-place editing trump the read intent of the command.
	bashReadPatterns = []string{
		"cat /", "cat ./", "cat ../", "grep ", "ls ", "ls -", "find ", "head ", "tail ",
		"wc ", "diff ", "which ", "ps ", "df ", "du ", "stat ", "file ", "less ", "more ",
	}

	// Read-only programs seen in real trajectories. Matching is limited to
	// command-segment starts, so prose and arguments cannot masquerade as actions.
	bashReadCommands = []string{
		"cat", "rg", "nl", "jq", "pwd", "tree", "sed", "grep", "ls", "find", "head",
		"tail", "wc", "diff", "which", "ps", "df", "du", "stat", "file", "less", "more",
		"readlink", "realpath", "basename", "dirname", "printenv",
	}

	gitReadSubcommands = []string{
		"status", "diff", "log", "show", "show-ref", "rev-parse", "ls-files",
		"ls-remote", "ls-tree", "grep", "blame", "merge-base", "check-ignore", "tag",
	}

	// The standard bin dirs hold the real programs. Any other path, such as
	// `./test`, names a local script.
	standardBinDirs = []string{
		"/bin/", "/sbin/", "/usr/bin/", "/usr/sbin/", "/usr/local/bin/", "/opt/homebrew/bin/",
	}
)

// shellPrograms are programs whose own invocation is a pure inspection.
var shellPrograms = map[string]bool{
	"cat": true, "grep": true, "ls": true, "nl": true, "head": true, "tail": true,
	"wc": true, "pwd": true, "stat": true, "file": true, "du": true, "df": true,
	"which": true, "type": true, "diff": true, "cmp": true, "jq": true, "uniq": true,
	"cut": true, "readlink": true, "realpath": true, "tree": true, "basename": true,
	"dirname": true, "printenv": true, "echo": true, "printf": true, "less": true,
	"more": true, "test": true, "[": true, "ps": true, "pgrep": true, "pkg-config": true,
	"strings": true, "uname": true, "od": true, "true": true, ":": true, "cd": true,
	"pstree": true, "sha256sum": true, "sha1sum": true, "md5sum": true, "lsof": true,
	"tr": true, "free": true, "id": true, "namei": true, "whoami": true, "paste": true,
	"ss": true, "getent": true,
}

// launchers unwrap to the program they run, with the options that take a value so
// the real command is not mistaken for one of its arguments.
var launcherOptions = map[string][]string{
	"sudo":    {"-u", "-g", "-h", "-p", "-C", "-T", "--user", "--group", "--host"},
	"env":     {"-u", "--unset", "-C", "--chdir"},
	"command": {},
	"nohup":   {},
	"exec":    {"-a"},
	"time":    {"-f", "--format", "-o", "--output"},
	"timeout": {"-s", "--signal", "-k", "--kill-after"},
	"xargs": {
		"-n", "--max-args", "-P", "--max-procs", "-I", "--replace", "-d", "--delimiter",
		"-L", "--max-lines", "-s", "--max-chars",
	},
}

// quotedChar is one unescaped character, the quote it sits inside, and its byte index.
type quotedChar struct {
	index int
	char  byte
	quote byte // 0 when outside quotes
}

// scanQuoted walks a command yielding unescaped characters with the quote around
// them. An escaped character is skipped entirely, which is what makes `\;` an
// argument rather than a separator.
func scanQuoted(command string) []quotedChar {
	var (
		out     []quotedChar
		quote   byte
		escaped bool
	)
	for index := 0; index < len(command); index++ {
		c := command[index]
		if escaped {
			escaped = false
			continue
		}
		if c == '\\' && quote != '\'' {
			escaped = true
			continue
		}
		around := quote
		switch {
		case quote != 0 && quote == c:
			quote = 0
		case quote == 0 && (c == '\'' || c == '"'):
			quote = c
		}
		out = append(out, quotedChar{index: index, char: c, quote: around})
	}
	return out
}

// shellSegments splits a shell line at unquoted command separators.
//
// Deliberately not a full shell parser: only the leading program and flags of each
// segment are inspected. Empty segments — what `&&` and `||` leave behind — are
// dropped.
func shellSegments(command string) []string {
	var (
		segments []string
		start    int
	)
	for _, c := range scanQuoted(command) {
		if c.quote != 0 {
			continue
		}
		switch c.char {
		case '\n', ';', '|', '&':
			if segment := strings.TrimSpace(command[start:c.index]); segment != "" {
				segments = append(segments, segment)
			}
			start = c.index + 1 // separators are one byte
		}
	}
	if segment := strings.TrimSpace(command[start:]); segment != "" {
		segments = append(segments, segment)
	}
	return segments
}

// shellWords splits a segment into words the way upstream does: on ASCII whitespace,
// discarding a leading `env` and its flags and any leading `NAME=value` assignments.
//
// This is not the quoted splitter below, and the difference is upstream's: word
// splitting here is only used to find the program, which quoting cannot change.
func shellWords(segment string) []string {
	words := strings.Fields(segment)
	index := 0
	if index < len(words) && words[index] == "env" {
		index++
		for index < len(words) && strings.HasPrefix(words[index], "-") {
			index++
		}
	}
	for index < len(words) && strings.Contains(words[index], "=") && !strings.HasPrefix(words[index], "=") {
		index++
	}
	return words[index:]
}

// shlexSplit splits a segment into words honouring quotes and escapes, reporting
// false for input it cannot parse — an unclosed quote or a trailing backslash.
//
// This stands in for the `shlex` crate upstream calls. It only ever sees a command
// that has already passed the substitution and redirect filter below, so the exotic
// corners of POSIX quoting are not reachable; unclosed input is reported rather than
// guessed at, which is what upstream's `is_some_and` depends on.
func shlexSplit(segment string) ([]string, bool) {
	var (
		words   []string
		current strings.Builder
		started bool
		quote   byte
	)
	flush := func() {
		if started {
			words = append(words, current.String())
			current.Reset()
			started = false
		}
	}
	for index := 0; index < len(segment); index++ {
		c := segment[index]
		switch {
		case quote == '\'':
			if c == '\'' {
				quote = 0
			} else {
				current.WriteByte(c)
			}
			started = true
		case quote == '"':
			if c == '"' {
				quote = 0
			} else if c == '\\' {
				index++
				if index >= len(segment) {
					return nil, false
				}
				current.WriteByte(segment[index])
			} else {
				current.WriteByte(c)
			}
			started = true
		case c == '\\':
			index++
			if index >= len(segment) {
				return nil, false
			}
			current.WriteByte(segment[index])
			started = true
		case c == '\'' || c == '"':
			quote = c
			started = true
		case c == ' ' || c == '\t' || c == '\n':
			flush()
		default:
			current.WriteByte(c)
			started = true
		}
	}
	if quote != 0 {
		return nil, false
	}
	flush()
	return words, true
}

// programName drops the directories from a word, so `/usr/bin/grep` is `grep` and a
// local `./build` is `build`.
func programName(word string) string {
	if index := strings.LastIndexByte(word, '/'); index >= 0 {
		return word[index+1:]
	}
	return word
}

// hasShortFlag reports whether a word is a short-option cluster such as `-Hx` that
// contains any of the given flags. Long options are not inspected.
func hasShortFlag(word string, flags string) bool {
	return strings.HasPrefix(word, "-") &&
		!strings.HasPrefix(word, "--") &&
		strings.ContainsAny(word, flags)
}

// skipShellOptions drops leading options, and the value of any option that takes one,
// stopping at the first argument or at `--`.
func skipShellOptions(args []string, takesValue []string) []string {
	for len(args) > 0 {
		option := args[0]
		if !strings.HasPrefix(option, "-") || option == "-" {
			break
		}
		args = args[1:]
		if option == "--" {
			break
		}
		if contains(takesValue, option) {
			if len(args) > 1 {
				args = args[1:]
			} else {
				args = nil
			}
		}
	}
	return args
}

// shellIsRetrieval reports whether every segment of a command is a recognized
// inspection.
//
// Only suppressing output when *all* segments qualify: observation counters can count
// mixed commands, but clearing an error signal needs every command to be benign.
// Substitutions and redirects therefore keep their error signals, because the shell
// would run a program the router has not seen.
func shellIsRetrieval(command string, depth int) bool {
	if len(command) > maxRetrievalCommandBytes {
		return false
	}
	for _, c := range scanQuoted(command) {
		switch {
		case c.quote != '\'' && (c.char == '$' || c.char == '`'):
			return false
		case c.quote == 0 && strings.IndexByte("<>(){}#", c.char) >= 0:
			return false
		}
	}
	segments := shellSegments(command)
	if len(segments) == 0 {
		return false
	}
	for _, segment := range segments {
		words, ok := shlexSplit(segment)
		if !ok {
			return false
		}
		if !retrievalCommand(words, depth) {
			return false
		}
	}
	return true
}

// retrievalCommand reports whether a single command is a recognized inspection,
// unwrapping launchers and shell invocations as it goes.
func retrievalCommand(words []string, depth int) bool {
	if depth > maxCommandDepth {
		return false
	}
	// Leading NAME=value assignments belong to the shell, not the program.
	for len(words) > 0 && isAssignment(words[0]) {
		words = words[1:]
	}
	if len(words) == 0 {
		return false
	}
	program, args := programName(words[0]), words[1:]
	for _, dir := range standardBinDirs {
		if strings.HasPrefix(program, dir) {
			program = strings.TrimPrefix(program, dir)
			break
		}
	}

	if options, isLauncher := launcherOptions[program]; isLauncher {
		// `env -S` splits its argument into a new command line, which this cannot see.
		if program == "env" && anyArg(args, func(arg string) bool {
			return strings.HasPrefix(arg, "--split-string") || hasShortFlag(arg, "S")
		}) {
			return false
		}
		rest := skipShellOptions(args, options)
		if program == "timeout" && len(rest) > 0 {
			rest = rest[1:]
		}
		if program == "env" && len(rest) == 0 {
			return true
		}
		return retrievalCommand(rest, depth+1)
	}

	switch program {
	case "bash", "sh", "dash", "zsh", "ksh":
		return len(args) > 0 && hasShortFlag(args[0], "c") &&
			len(args) > 1 && shellIsRetrieval(args[1], depth+1)
	case "git":
		return gitIsRetrieval(args)
	}

	// Any command asked only for its help or version changes nothing.
	if len(args) == 1 && (args[0] == "--help" || args[0] == "--version") {
		return true
	}

	switch program {
	case "find":
		return !anyArg(args, func(arg string) bool {
			switch arg {
			case "-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf":
				return true
			}
			return false
		})
	case "fd":
		// fd joins short flags, so `-Hx` runs a command too.
		return !anyArg(args, func(arg string) bool {
			return strings.HasPrefix(arg, "--exec") || hasShortFlag(arg, "xX")
		})
	case "sed":
		// Only the common line-range printing form; a sed script can run commands.
		if len(args) < 2 || args[0] != "-n" {
			return false
		}
		for _, arg := range args[2:] {
			if strings.HasPrefix(arg, "-") {
				return false
			}
		}
		range_ := strings.TrimSuffix(args[1], "p")
		if range_ == args[1] || range_ == "" {
			return false
		}
		for index := 0; index < len(range_); index++ {
			c := range_[index]
			if c >= '0' && c <= '9' || c == ',' || c == '$' {
				continue
			}
			return false
		}
		return true
	case "sort":
		return !anyArg(args, func(arg string) bool {
			return strings.HasPrefix(arg, "--compress-program") ||
				strings.HasPrefix(arg, "--output") || strings.HasPrefix(arg, "-o")
		})
	case "rg":
		return !anyArg(args, func(arg string) bool { return strings.HasPrefix(arg, "--pre") })
	case "xxd":
		return !contains(args, "-r") && !contains(args, "-revert")
	case "go":
		if len(args) == 0 {
			return false
		}
		switch args[0] {
		case "list", "doc", "env", "version":
		default:
			return false
		}
		return !contains(args, "-w") && !contains(args, "-u")
	case "docker", "podman":
		return len(args) > 0 && (args[0] == "ps" || args[0] == "version")
	}
	return shellPrograms[program]
}

// gitIsRetrieval reports whether a git invocation only inspects. Reading and writing
// share a program, so the subcommand and its flags decide.
func gitIsRetrieval(args []string) bool {
	args = skipShellOptions(args, []string{"-C", "-c", "--git-dir", "--work-tree", "--namespace"})
	if len(args) == 0 {
		return false
	}
	subcommand, rest := args[0], args[1:]
	switch subcommand {
	case "diff":
		return !contains(rest, "--check")
	case "branch":
		return len(rest) == 0 || allArgs(rest, func(arg string) bool {
			switch arg {
			case "-a", "-r", "--all", "--list", "--show-current", "-v", "-vv", "--verbose":
				return true
			}
			return false
		})
	case "remote":
		return len(rest) == 0 || (len(rest) == 1 && (rest[0] == "-v" || rest[0] == "--verbose")) ||
			rest[0] == "get-url"
	case "tag":
		return len(rest) == 0 || rest[0] == "-l" || rest[0] == "--list"
	case "config":
		if len(rest) == 0 {
			return false
		}
		switch rest[0] {
		case "--get", "--get-all", "--list", "-l":
			return true
		}
		return false
	case "worktree":
		return len(rest) > 0 && rest[0] == "list"
	case "submodule":
		return len(rest) > 0 && rest[0] == "status"
	case "status", "log", "show", "blame", "ls-files", "ls-remote", "rev-parse",
		"merge-base", "grep", "describe", "show-ref", "check-ignore", "rev-list",
		"ls-tree", "diff-tree", "version":
		return true
	}
	return false
}

// shellInvokesProgram reports whether any segment of a command runs a program.
func shellInvokesProgram(command, expected string) bool {
	for _, segment := range shellSegments(command) {
		words := shellWords(segment)
		if len(words) > 0 && programName(words[0]) == expected {
			return true
		}
	}
	return false
}

// shellCommandIsWrite reports whether any segment of a command changes state by
// creating, moving or redirecting into a file.
func shellCommandIsWrite(command string) bool {
	for _, segment := range shellSegments(command) {
		words := shellWords(segment)
		if len(words) == 0 {
			continue
		}
		program := programName(words[0])
		switch program {
		case "cp", "mkdir", "touch", "install":
			return true
		}
		redirectsOutput := false
		for _, word := range words[1:] {
			if word == ">" || word == ">>" {
				redirectsOutput = true
				break
			}
		}
		if !redirectsOutput {
			continue
		}
		if program == "echo" || program == "printf" || program == "git" ||
			contains(bashReadCommands, program) {
			return true
		}
	}
	return false
}

// shellCommandIsEdit reports whether any segment of a command rewrites a file in
// place: the formatters, and the tools that move or remove one.
func shellCommandIsEdit(command string) bool {
	for _, segment := range shellSegments(command) {
		words := shellWords(segment)
		if len(words) == 0 {
			continue
		}
		program, args := programName(words[0]), words[1:]
		hasArg := func(arg string) bool { return contains(args, arg) }

		switch program {
		case "mv", "rm":
			return true
		case "perl":
			for _, option := range args {
				if !strings.HasPrefix(option, "-") {
					break
				}
				if strings.ContainsRune(strings.TrimLeft(option, "-"), 'i') {
					return true
				}
			}
		case "git":
			if len(args) > 0 && (args[0] == "apply" || args[0] == "am" || args[0] == "restore") {
				return true
			}
		case "gofmt":
			if hasArg("-w") {
				return true
			}
		case "cargo":
			if len(args) > 0 && args[0] == "fmt" && !hasArg("--check") {
				return true
			}
		case "ruff":
			if len(args) > 0 {
				if (args[0] == "format" && !hasArg("--check")) ||
					(args[0] == "check" && hasArg("--fix")) {
					return true
				}
			}
		case "prettier":
			if hasArg("--write") {
				return true
			}
		case "black":
			if !hasArg("--check") {
				return true
			}
		default:
			// A formatter reached through a launcher, such as `npx prettier`.
			for _, word := range args {
				name := programName(word)
				if name == "prettier" && hasArg("--write") {
					return true
				}
				if name == "ruff" &&
					((hasArg("format") && !hasArg("--check")) || (hasArg("check") && hasArg("--fix"))) {
					return true
				}
			}
		}
	}
	return false
}

// shellCommandIsRead reports whether any segment of a command only inspects.
func shellCommandIsRead(command string) bool {
	for _, segment := range shellSegments(command) {
		if segment == "env" {
			return true
		}
		words := shellWords(segment)
		if len(words) == 0 {
			continue
		}
		program, args := programName(words[0]), words[1:]

		if contains(bashReadCommands, program) {
			return true
		}
		if program == "command" && len(args) > 0 && args[0] == "-v" {
			return true
		}
		if program == "type" {
			return true
		}
		if program != "git" {
			continue
		}
		if len(args) == 0 {
			continue
		}
		switch args[0] {
		case "branch":
			return len(args) == 1 || strings.HasPrefix(args[1], "-")
		case "remote":
			return len(args) == 1 || strings.HasPrefix(args[1], "-") || args[1] == "get-url"
		case "config":
			if len(args) < 2 {
				return false
			}
			switch args[1] {
			case "--get", "--get-all", "--list", "-l":
				return true
			}
			return false
		default:
			if contains(gitReadSubcommands, args[0]) {
				return true
			}
		}
	}
	return false
}

// isAssignment reports whether a word is a `NAME=value` shell assignment.
func isAssignment(word string) bool {
	name, _, found := strings.Cut(word, "=")
	if !found || name == "" {
		return false
	}
	for index := 0; index < len(name); index++ {
		c := name[index]
		if c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || (index > 0 && c >= '0' && c <= '9') {
			continue
		}
		return false
	}
	return true
}

func anyArg(args []string, predicate func(string) bool) bool {
	for _, arg := range args {
		if predicate(arg) {
			return true
		}
	}
	return false
}

func allArgs(args []string, predicate func(string) bool) bool {
	for _, arg := range args {
		if !predicate(arg) {
			return false
		}
	}
	return true
}
