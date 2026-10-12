package libsygo

import "encoding/json"

// AppendRequest serializes a normalized request in upstream's shape.
//
// It exists so a host can hand the IR to something else without reimplementing the
// shape — the C ABI conformance backend, for instance, sends a normalized request
// to real libsy and compares. Every field is emitted, absent or not, because that
// is what upstream's serializer does and a reader elsewhere may depend on it.
func AppendRequest(dst []byte, request LlmRequest) []byte {
	dst = append(dst, `{"model":`...)
	dst = appendOptionalString(dst, optional(request.Model))

	dst = append(dst, `,"instructions":`...)
	dst = AppendInstructions(dst, request.Instructions)

	dst = append(dst, `,"messages":`...)
	dst = AppendMessages(dst, request.Messages)

	dst = append(dst, `,"tools":`...)
	dst = appendTools(dst, request.Tools)

	dst = append(dst, `,"tool_choice":`...)
	dst = appendToolChoice(dst, request.ToolChoice)

	dst = append(dst, `,"sampling":{"temperature":`...)
	dst = appendOptionalFloat(dst, request.Sampling.Temperature)
	dst = append(dst, `,"top_p":`...)
	dst = appendOptionalFloat(dst, request.Sampling.TopP)
	dst = append(dst, `,"top_k":`...)
	dst = appendOptionalInt(dst, request.Sampling.TopK)
	dst = append(dst, '}')

	dst = append(dst, `,"output":{"max_output_tokens":`...)
	dst = appendOptionalUint(dst, request.Output.MaxOutputTokens)
	dst = append(dst, `,"response_format":`...)
	dst = appendRaw(dst, request.Output.ResponseFormat)
	dst = append(dst, `,"is_schema_enforced":`...)
	dst = appendOptionalBool(dst, request.Output.IsSchemaEnforced)
	dst = append(dst, '}')

	dst = append(dst, `,"reasoning":{"effort":`...)
	dst = appendOptionalString(dst, optional(request.Reasoning.Effort))
	dst = append(dst, `,"raw":`...)
	dst = appendRaw(dst, request.Reasoning.Raw)
	dst = append(dst, '}')

	dst = append(dst, `,"stream":`...)
	if request.Stream {
		dst = append(dst, "true"...)
	} else {
		dst = append(dst, "false"...)
	}

	// Unrecognized provider fields are kept in wire order under their original
	// names, so a round trip through this package does not lose them.
	dst = append(dst, `,"extensions":{"fields":{`...)
	for index, field := range request.Extensions {
		if index > 0 {
			dst = append(dst, ',')
		}
		dst = appendString(dst, field.Name)
		dst = append(dst, ':')
		dst = appendRaw(dst, field.Value)
	}
	dst = append(dst, `}}}`...)
	return dst
}

// LlmRequestJSON is a normalized request as JSON.
func LlmRequestJSON(request LlmRequest) json.RawMessage {
	return json.RawMessage(AppendRequest(nil, request))
}

func appendTools(dst []byte, tools []ToolDefinition) []byte {
	dst = append(dst, '[')
	for index, tool := range tools {
		if index > 0 {
			dst = append(dst, ',')
		}
		dst = append(dst, `{"name":`...)
		dst = appendString(dst, tool.Name)
		dst = append(dst, `,"description":`...)
		dst = appendOptionalString(dst, tool.Description)
		dst = append(dst, `,"parameters":`...)
		dst = appendRaw(dst, tool.Parameters)
		dst = append(dst, `,"strict":`...)
		dst = appendOptionalBool(dst, tool.Strict)
		dst = append(dst, '}')
	}
	return append(dst, ']')
}

// appendToolChoice writes upstream's adjacently-tagged enum: a unit variant carries
// only its tag, a variant with data carries `data`.
func appendToolChoice(dst []byte, choice *ToolChoice) []byte {
	if choice == nil {
		return append(dst, "null"...)
	}
	switch choice.Kind {
	case ToolChoiceAuto, ToolChoiceRequired, ToolChoiceNone:
		dst = append(dst, `{"type":"`...)
		dst = append(dst, string(choice.Kind)...)
		return append(dst, `"}`...)
	case ToolChoiceTool:
		dst = append(dst, `{"type":"tool","data":{"name":`...)
		dst = appendString(dst, choice.Name)
		return append(dst, `}}`...)
	default:
		dst = append(dst, `{"type":"raw","data":`...)
		dst = appendRaw(dst, choice.Raw)
		return append(dst, '}')
	}
}

func optional(value string) *string {
	if value == "" {
		return nil
	}
	return &value
}

func appendOptionalFloat(dst []byte, value *float64) []byte {
	if value == nil {
		return append(dst, "null"...)
	}
	return appendFloat(dst, *value)
}

func appendOptionalInt(dst []byte, value *int64) []byte {
	if value == nil {
		return append(dst, "null"...)
	}
	return appendInt(dst, *value)
}

func appendOptionalUint(dst []byte, value *uint64) []byte {
	if value == nil {
		return append(dst, "null"...)
	}
	return appendUint(dst, *value)
}
