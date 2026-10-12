package libsygo

import "encoding/json"

// Role is a normalized message role.
type Role string

// Normalized roles.
const (
	RoleSystem    Role = "system"
	RoleDeveloper Role = "developer"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// InstructionBlock is instruction content separated from the conversation turns.
//
// Decoders lift client system and developer content out of the message list, so a
// configured instruction cannot be mistaken for something the user said.
type InstructionBlock struct {
	Role    Role
	Content []ContentBlock
}

// Message is one normalized conversation message.
type Message struct {
	Role    Role
	Content []ContentBlock
}

// TextContent concatenates a message's text-like blocks, reporting whether it had
// any. Refusals count as text, matching upstream — a refusal is still something
// the model said.
func (m Message) TextContent(separator string) (string, bool) {
	parts := make([]string, 0, len(m.Content))
	for _, block := range m.Content {
		switch block.Kind {
		case ContentText, ContentRefusal:
			parts = append(parts, block.Text)
		}
	}
	if len(parts) == 0 {
		return "", false
	}
	joined := parts[0]
	for _, part := range parts[1:] {
		joined += separator + part
	}
	return joined, true
}

// LlmRequest is the normalized request every algorithm reads.
//
// This is upstream's IR, minus `preservation`: exact provider bodies retained for
// lossless round trips are a translation-engine concern that no algorithm reads,
// and nothing here re-encodes a request back to a provider.
//
// Provider values that have no normalized equivalent — tool arguments, media
// sources, unknown blocks — are carried as raw JSON rather than decoded into Go
// maps. That is not laziness: upstream serializes with `serde_json`'s
// `preserve_order`, so a map that round-trips through Go would come back
// alphabetized, and the request digest would stop matching.
type LlmRequest struct {
	// Model is the model the request arrived addressed to. Routing replaces it.
	Model string

	// Instructions are system and developer content, separated from the turns.
	Instructions []InstructionBlock

	// Messages are the conversation turns.
	Messages []Message

	// Tools are available to the model.
	Tools []ToolDefinition

	// ToolChoice constrains tool selection.
	ToolChoice *ToolChoice

	// Sampling, Output and Reasoning carry the generation controls.
	Sampling  SamplingParams
	Output    OutputParams
	Reasoning ReasoningParams

	// Stream reports whether the caller asked for a streamed response.
	Stream bool

	// Extensions are provider fields with no normalized equivalent, keyed by their
	// original wire names in the order they appeared.
	Extensions []ExtensionField
}

// ExtensionField is one unrecognized provider field, kept in wire order.
type ExtensionField struct {
	Name  string
	Value json.RawMessage
}

// Extension returns a provider field by its wire name.
func (r LlmRequest) Extension(name string) (json.RawMessage, bool) {
	for _, field := range r.Extensions {
		if field.Name == name {
			return field.Value, true
		}
	}
	return nil, false
}

// ContentKind names a normalized content block variant.
type ContentKind string

// Normalized content block kinds.
const (
	ContentText       ContentKind = "text"
	ContentReasoning  ContentKind = "reasoning"
	ContentImage      ContentKind = "image"
	ContentAudio      ContentKind = "audio"
	ContentVideo      ContentKind = "video"
	ContentFile       ContentKind = "file"
	ContentToolCall   ContentKind = "tool_call"
	ContentToolResult ContentKind = "tool_result"
	ContentRefusal    ContentKind = "refusal"
	ContentUnknown    ContentKind = "unknown"
)

// ContentBlock is one normalized content block.
//
// A flat struct with a kind, rather than one type per variant. The union has ten
// arms and the algorithms read three; serialization has to emit each arm in
// upstream's exact field order regardless, which a switch on the kind expresses
// directly instead of scattering ten `MarshalJSON` methods.
type ContentBlock struct {
	Kind ContentKind

	// Text is the payload of a text, reasoning or refusal block.
	Text string

	// Signature and Details carry a reasoning block's provider state. Details is
	// replayed without modification, so it is opaque. Signature is nil when the
	// provider supplied none — upstream treats an empty signature as absent, so a
	// provider that sends "" is normalized the same way.
	Signature *string
	Details   []json.RawMessage

	// Source is an image, audio, video or file payload, already in upstream's
	// tagged shape. No algorithm reads it; it is carried, not modelled.
	Source json.RawMessage

	// Call and Result are set on tool_call and tool_result blocks.
	Call   *ToolCall
	Result *ToolResult

	// Provider names the wire format an unknown block came from, and Raw holds it
	// verbatim.
	Provider string
	Raw      json.RawMessage
}

// Text returns a text block.
func Text(text string) ContentBlock {
	return ContentBlock{Kind: ContentText, Text: text}
}

// ToolCall is a normalized assistant tool invocation.
type ToolCall struct {
	// ID pairs the call with its result. A call without a provider id gets a
	// generated one, so the pair survives.
	ID string

	// Name is the tool name.
	Name string

	// Arguments are the parsed arguments, in the order the provider wrote them.
	// A provider that stringifies arguments has them parsed; one that does not is
	// carried under a `raw` key.
	Arguments json.RawMessage
}

// ToolResult is a normalized tool execution result.
type ToolResult struct {
	// ToolCallID is the call this answers.
	ToolCallID string

	// Content is the tool's output.
	Content []ContentBlock

	// IsError reports a failed execution, when the provider says so.
	IsError *bool
}

// ToolDefinition is a normalized tool available to the model.
type ToolDefinition struct {
	Name        string
	Description *string
	Parameters  json.RawMessage
	Strict      *bool
}

// ToolChoiceKind names a tool-choice policy.
type ToolChoiceKind string

// Tool-choice policies.
const (
	ToolChoiceAuto     ToolChoiceKind = "auto"
	ToolChoiceRequired ToolChoiceKind = "required"
	ToolChoiceNone     ToolChoiceKind = "none"
	ToolChoiceTool     ToolChoiceKind = "tool"
	ToolChoiceRaw      ToolChoiceKind = "raw"
)

// ToolChoice constrains how the model may select tools.
type ToolChoice struct {
	Kind ToolChoiceKind
	Name string
	Raw  json.RawMessage
}

// SamplingParams are the common sampling controls.
type SamplingParams struct {
	Temperature *float64
	TopP        *float64
	TopK        *int64
}

// OutputParams are the output budget and shape controls.
type OutputParams struct {
	MaxOutputTokens  *uint64
	ResponseFormat   json.RawMessage
	IsSchemaEnforced *bool
}

// ReasoningParams are the provider reasoning controls.
type ReasoningParams struct {
	Effort string
	Raw    json.RawMessage
}

// taskContent reports whether a block carries task text rather than tool traffic
// or provider-private reasoning.
//
// The distinction matters because decoders represent a tool result as a user-role
// message, so the user role alone cannot tell the opening task from a tool result.
func taskContent(block ContentBlock) bool {
	switch block.Kind {
	case ContentToolCall, ContentToolResult, ContentReasoning:
		return false
	}
	return true
}

// TaskMessages selects what a judge is shown: the opening task and the latest
// user follow-up, when they differ.
//
// Judging the whole transcript would let a long tool-driven conversation outrank
// the task it is serving, so upstream shows the anchor and the newest ask and
// nothing between them. Both are re-role'd as user content and stripped of tool
// and reasoning traffic, so a tool result can neither replace the opening task nor
// become the follow-up.
func TaskMessages(messages []Message) []Message {
	var candidates []Message
	for _, message := range messages {
		if message.Role != RoleUser {
			continue
		}
		if !hasTaskContent(message) {
			continue
		}
		candidates = append(candidates, message)
	}
	if len(candidates) == 0 {
		return nil
	}

	selected := []Message{candidates[0]}
	if len(candidates) > 1 {
		selected = append(selected, candidates[len(candidates)-1])
	}

	out := make([]Message, 0, len(selected))
	for _, message := range selected {
		content := make([]ContentBlock, 0, len(message.Content))
		for _, block := range message.Content {
			if taskContent(block) {
				content = append(content, block)
			}
		}
		out = append(out, Message{Role: RoleUser, Content: content})
	}
	return out
}

func hasTaskContent(message Message) bool {
	for _, block := range message.Content {
		if taskContent(block) {
			return true
		}
	}
	return false
}
