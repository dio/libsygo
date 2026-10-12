package libsygo

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"unicode/utf8"
)

// DecodeAnthropicMessages normalizes an Anthropic Messages request body.
//
// A different shape from OpenAI chat in ways that matter beyond field names: the system
// prompt is a top-level field rather than a message, `input_schema` replaces
// `parameters`, and a tool result is a content block rather than a message of its own.
// The parts that agree are decoded to the same normalized request, which is the point of
// the IR.
func DecodeAnthropicMessages(body []byte) (LlmRequest, error) {
	fields, err := objectFields(body)
	if err != nil {
		return LlmRequest{}, &DecodeError{Path: "$", Problem: err.Error()}
	}

	// The two fallback mechanisms are mutually exclusive, and the provider only finds
	// out when it rejects the request.
	_, hasCreditToken := field(fields, "fallback_credit_token")
	if raw, present := field(fields, "fallback_credit_token"); present && !isNull(raw) {
		if fallbacks, present := field(fields, "fallbacks"); present && !isNull(fallbacks) {
			return LlmRequest{}, &DecodeError{
				Path:    "$.fallback_credit_token",
				Problem: "fallback_credit_token cannot be combined with fallbacks",
			}
		}
	}
	_ = hasCreditToken

	// max_tokens is required to be a count when present: a provider that sent `null`
	// asked for something this cannot represent, and a request sent upstream with a
	// guess would be rejected there instead.
	var maxTokens *uint64
	if raw, present := field(fields, "max_tokens"); present {
		value, ok := uint64Value(raw)
		if !ok {
			return LlmRequest{}, &DecodeError{
				Path: "$.max_tokens", Problem: "expected a non-negative integer"}
		}
		maxTokens = &value
	}

	if raw, present := field(fields, "output_config"); present {
		if config, err := objectFields(raw); err == nil {
			if _, present := field(config, "format"); present {
				return LlmRequest{}, &DecodeError{
					Path:    "$.output_config.format",
					Problem: "Anthropic structured output is not ported",
				}
			}
		}
	}
	if _, present := field(fields, "output_format"); present {
		return LlmRequest{}, &DecodeError{
			Path: "$.output_format", Problem: "Anthropic structured output is not ported"}
	}

	request := LlmRequest{
		Model:  stringOrEmpty(fields, "model"),
		Stream: boolOrFalse(fields, "stream"),
		Sampling: SamplingParams{
			Temperature: float64OrNil(fields, "temperature"),
			TopP:        float64OrNil(fields, "top_p"),
			TopK:        int64OrNil(fields, "top_k"),
		},
		Output:    OutputParams{MaxOutputTokens: maxTokens},
		Reasoning: ReasoningParams{Effort: anthropicEffort(fields)},
	}

	if raw, present := field(fields, "system"); present {
		content, err := decodeAnthropicSystem(raw)
		if err != nil {
			return LlmRequest{}, err
		}
		if content != nil {
			request.Instructions = append(request.Instructions,
				InstructionBlock{Role: RoleSystem, Content: content})
		}
	}

	if raw, present := field(fields, "messages"); present {
		var elements []json.RawMessage
		if err := json.Unmarshal(raw, &elements); err != nil {
			return LlmRequest{}, &DecodeError{Path: "$.messages", Problem: "expected an array"}
		}
		for index, element := range elements {
			fields, err := objectFields(element)
			if err != nil {
				// Not an object. Upstream records a diagnostic and moves on: the
				// request is still routable.
				continue
			}
			role, err := anthropicRole(fields, fmt.Sprintf("$.messages[%d].role", index))
			if err != nil {
				return LlmRequest{}, err
			}
			// The counter advances once per message; a block's generated id is the
			// message's plus its index, so ids stay distinct within the request.
			content, err := decodeAnthropicContent(rawField(fields, "content"), index+1)
			if err != nil {
				return LlmRequest{}, err
			}
			request.Messages = append(request.Messages, Message{Role: role, Content: content})
		}
	}

	request.Tools = decodeAnthropicTools(rawField(fields, "tools"))
	if raw, present := field(fields, "tool_choice"); present {
		request.ToolChoice = decodeAnthropicToolChoice(raw)
		// A provider expresses "no parallel calls" through the tool choice, and the
		// normalized request has a field for it, so it is carried rather than lost.
		if choice, err := objectFields(raw); err == nil {
			if disabled, ok := boolValue(rawField(choice, "disable_parallel_tool_use")); ok {
				request.Extensions = append(request.Extensions, ExtensionField{
					Name:  "parallel_tool_calls",
					Value: json.RawMessage(fmt.Sprintf("%t", !disabled)),
				})
			}
		}
	}

	for _, candidate := range fields {
		if anthropicKnownFields[candidate.Name] {
			continue
		}
		request.Extensions = append(request.Extensions, ExtensionField{
			Name:  candidate.Name,
			Value: CompactJSON(candidate.Value),
		})
	}

	return request, nil
}

// anthropicKnownFields are the top-level fields the decoder recognizes. Anything else is
// carried as a provider extension rather than dropped.
var anthropicKnownFields = map[string]bool{
	"model": true, "messages": true, "system": true, "tools": true, "tool_choice": true,
	"max_tokens": true, "temperature": true, "top_p": true, "top_k": true,
	"thinking": true, "output_config": true, "output_format": true, "stream": true,
	// An OpenAI-only identity field, listed so it does not leak through as an extension.
	"safety_identifier": true,
}

func anthropicEffort(fields []jsonField) string {
	config, err := objectFields(rawField(fields, "output_config"))
	if err != nil {
		return ""
	}
	return stringOrEmpty(config, "effort")
}

// anthropicRole maps an Anthropic role. Only `assistant` is meaningful; a known role this
// format does not define stays lenient and maps to a user turn, while an unknown one is
// refused.
func anthropicRole(fields []jsonField, path string) (Role, error) {
	name, ok := stringValue(rawField(fields, "role"))
	if !ok {
		return RoleUser, nil
	}
	if name == "assistant" {
		return RoleAssistant, nil
	}
	if knownRoleNames[name] {
		return RoleUser, nil
	}
	return "", &DecodeError{Path: path, Problem: fmt.Sprintf("unsupported role %q", name)}
}

// decodeAnthropicSystem reads the top-level system field: a string, or a list of text
// blocks. An empty string is absent rather than an empty instruction.
func decodeAnthropicSystem(raw json.RawMessage) ([]ContentBlock, error) {
	trimmed := bytes.TrimSpace(raw)
	switch {
	case len(trimmed) == 0 || isNull(trimmed):
		return nil, nil
	case trimmed[0] == '"':
		text, _ := stringValue(trimmed)
		if text == "" {
			return nil, nil
		}
		return []ContentBlock{Text(text)}, nil
	case trimmed[0] == '[':
		var elements []json.RawMessage
		if err := json.Unmarshal(raw, &elements); err != nil {
			return nil, nil
		}
		var content []ContentBlock
		for _, element := range elements {
			block, err := objectFields(element)
			if err != nil {
				continue
			}
			if kind, _ := stringValue(rawField(block, "type")); kind != "text" {
				continue
			}
			content = append(content, Text(stringOrEmpty(block, "text")))
		}
		if len(content) == 0 {
			return nil, nil
		}
		return content, nil
	default:
		return nil, &DecodeError{
			Path: "$.system", Problem: "expected a string or an array of text blocks"}
	}
}

// decodeAnthropicContent reads a message's content: a string, an array of blocks, null,
// or something else entirely.
func decodeAnthropicContent(raw json.RawMessage, generated int) ([]ContentBlock, error) {
	trimmed := bytes.TrimSpace(raw)
	switch {
	case trimmed[0] == '"':
		text, _ := stringValue(trimmed)
		return []ContentBlock{Text(text)}, nil
	case isNull(trimmed):
		return []ContentBlock{Text("")}, nil
	case trimmed[0] == '[':
		var elements []json.RawMessage
		if err := json.Unmarshal(raw, &elements); err != nil {
			return []ContentBlock{Text("")}, nil
		}
		var content []ContentBlock
		for index, element := range elements {
			block, err := objectFields(element)
			if err != nil {
				continue
			}
			content = append(content, decodeAnthropicBlock(block, generated+index)...)
		}
		if len(content) == 0 {
			return []ContentBlock{Text("")}, nil
		}
		return content, nil
	default:
		return []ContentBlock{Text(renderAsText(trimmed))}, nil
	}
}

// decodeAnthropicBlock reads one content block. A block type this package does not know
// is carried verbatim, so nothing is lost on the way through.
func decodeAnthropicBlock(block []jsonField, generated int) []ContentBlock {
	kind, _ := stringValue(rawField(block, "type"))
	switch kind {
	case "text":
		return []ContentBlock{Text(stringOrEmpty(block, "text"))}

	case "thinking":
		// Anthropic signs its thinking; an empty signature is treated as absent, which
		// is how upstream stores it.
		text := stringOrEmpty(block, "thinking")
		result := ContentBlock{Kind: ContentReasoning, Text: text}
		if signature, ok := stringValue(rawField(block, "signature")); ok && signature != "" {
			result.Signature = &signature
		}
		return []ContentBlock{result}

	case "tool_use":
		arguments := json.RawMessage("{}")
		if raw, present := field(block, "input"); present {
			arguments = CompactJSON(raw)
		}
		// A call without a provider id is named from its position, so the result that
		// answers it can be paired.
		id, ok := stringValue(rawField(block, "id"))
		if !ok || id == "" {
			id = fmt.Sprintf("sw_%08d", generated)
		} else {
			id = desanitizeToolUseID(id)
		}
		return []ContentBlock{{Kind: ContentToolCall, Call: &ToolCall{
			ID:        id,
			Name:      stringOrEmpty(block, "name"),
			Arguments: arguments,
		}}}

	case "tool_result":
		id, _ := stringValue(rawField(block, "tool_use_id"))
		result := &ToolResult{
			ToolCallID: desanitizeToolUseID(id),
			Content:    decodeToolResultContent(rawField(block, "content")),
		}
		if isError, ok := boolValue(rawField(block, "is_error")); ok {
			result.IsError = &isError
		}
		return []ContentBlock{{Kind: ContentToolResult, Result: result}}

	case "image":
		// Anthropic image blocks are carried whole rather than normalized: the shape is
		// provider-specific and nothing here reads an image.
		return []ContentBlock{{Kind: ContentImage, Source: sourceLiteral("raw", rawOf(block))}}

	case "input_image", "image_url":
		return nil // upstream drops an image part it cannot decode, rather than guessing

	case "input_file", "file", "document":
		return []ContentBlock{{Kind: ContentFile, Source: sourceLiteral("raw", rawOf(block))}}

	default:
		return []ContentBlock{{
			Kind:     ContentUnknown,
			Provider: "anthropic_messages",
			Raw:      rawOf(block),
		}}
	}
}

// decodeToolResultContent reads a tool result's own content, which is another
// string-or-blocks field.
func decodeToolResultContent(raw json.RawMessage) []ContentBlock {
	trimmed := bytes.TrimSpace(raw)
	switch {
	case len(trimmed) == 0 || isNull(trimmed):
		return []ContentBlock{Text("")}
	case trimmed[0] == '"':
		text, _ := stringValue(trimmed)
		return []ContentBlock{Text(text)}
	case trimmed[0] == '[':
		var elements []json.RawMessage
		if err := json.Unmarshal(raw, &elements); err != nil {
			return []ContentBlock{Text("")}
		}
		var content []ContentBlock
		for _, element := range elements {
			block, err := objectFields(element)
			if err != nil {
				continue
			}
			switch kind, _ := stringValue(rawField(block, "type")); kind {
			case "text":
				content = append(content, Text(stringOrEmpty(block, "text")))
			case "image", "document":
				kind_ := ContentImage
				if kind == "document" {
					kind_ = ContentFile
				}
				content = append(content, ContentBlock{
					Kind: kind_, Source: sourceLiteral("raw", rawOf(block))})
			default:
				content = append(content, ContentBlock{
					Kind: ContentUnknown, Provider: "anthropic_messages", Raw: rawOf(block)})
			}
		}
		return content
	default:
		return []ContentBlock{Text(jsonStringValue(trimmed))}
	}
}

// decodeAnthropicTools reads tool definitions, whose schema field is `input_schema` rather
// than `parameters`.
func decodeAnthropicTools(raw json.RawMessage) []ToolDefinition {
	var elements []json.RawMessage
	if err := json.Unmarshal(raw, &elements); err != nil {
		return nil
	}
	var definitions []ToolDefinition
	for _, element := range elements {
		tool, err := objectFields(element)
		if err != nil {
			continue
		}
		name, ok := stringValue(rawField(tool, "name"))
		if !ok || name == "" {
			continue
		}
		definition := ToolDefinition{Name: name}
		if description, ok := stringValue(rawField(tool, "description")); ok {
			definition.Description = &description
		}
		definition.Parameters = json.RawMessage("{}")
		if raw, present := field(tool, "input_schema"); present {
			definition.Parameters = CompactJSON(raw)
		}
		if strict, ok := boolValue(rawField(tool, "strict")); ok {
			definition.Strict = &strict
		}
		definitions = append(definitions, definition)
	}
	return definitions
}

// decodeAnthropicToolChoice normalizes the tool-choice policy, which this provider spells
// in its own vocabulary.
func decodeAnthropicToolChoice(raw json.RawMessage) *ToolChoice {
	if text, ok := stringValue(raw); ok {
		switch text {
		case "auto":
			return &ToolChoice{Kind: ToolChoiceAuto}
		case "any":
			return &ToolChoice{Kind: ToolChoiceRequired}
		case "none":
			return &ToolChoice{Kind: ToolChoiceNone}
		}
		return &ToolChoice{Kind: ToolChoiceRaw, Raw: CompactJSON(raw)}
	}
	if choice, err := objectFields(raw); err == nil {
		switch kind, _ := stringValue(rawField(choice, "type")); kind {
		case "auto":
			return &ToolChoice{Kind: ToolChoiceAuto}
		case "any":
			return &ToolChoice{Kind: ToolChoiceRequired}
		case "none":
			return &ToolChoice{Kind: ToolChoiceNone}
		case "tool":
			if name, ok := stringValue(rawField(choice, "name")); ok {
				return &ToolChoice{Kind: ToolChoiceTool, Name: name}
			}
		}
	}
	return &ToolChoice{Kind: ToolChoiceRaw, Raw: CompactJSON(raw)}
}

// Anthropic tool-use ids tolerate only alphanumerics, underscore and hyphen, so a
// provider-issued id with anything else is encoded with a marker prefix. Decoding restores
// the original, and an id that was not encoded passes through.
const toolIDEncodingPrefix = "sy64_"

func desanitizeToolUseID(encoded string) string {
	payload, ok := strings.CutPrefix(encoded, toolIDEncodingPrefix)
	if !ok {
		return encoded
	}
	decoded, err := base64.RawURLEncoding.DecodeString(payload)
	if err != nil || !utf8.Valid(decoded) {
		return encoded
	}
	return string(decoded)
}

// rawOf is a block's own bytes, which is what an unmodelled block is carried as.
func rawOf(fields []jsonField) json.RawMessage {
	return objectLiteral(fields...)
}

// sourceLiteral builds an internally-tagged payload for a block carried verbatim.
func sourceLiteral(kind string, payload json.RawMessage) json.RawMessage {
	return taggedSource(kind, payload)
}

// jsonStringValue renders a value as text the way a provider would: a string as itself,
// anything else as its compact JSON.
func jsonStringValue(raw json.RawMessage) string {
	if text, ok := stringValue(raw); ok {
		return text
	}
	if len(raw) == 0 {
		return "null"
	}
	return string(CompactJSON(raw))
}

func int64OrNil(fields []jsonField, name string) *int64 {
	raw, present := field(fields, name)
	if !present || isNull(raw) {
		return nil
	}
	var value int64
	if err := json.Unmarshal(bytes.TrimSpace(raw), &value); err != nil {
		return nil
	}
	return &value
}

func uint64Value(raw json.RawMessage) (uint64, bool) {
	var value uint64
	if err := json.Unmarshal(bytes.TrimSpace(raw), &value); err != nil {
		return 0, false
	}
	return value, true
}
