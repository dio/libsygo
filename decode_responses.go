package libsygo

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// DecodeResponses normalizes an OpenAI Responses request body.
//
// Its conversation is not a message list but a stream of typed *items*, and the shape
// carries meaning the other two formats do not have: tool calls and the results
// answering them belong to one turn and must stay adjacent, reasoning belongs to the
// assistant turn that produced it, and an inline system item is an instruction rather
// than something a participant said. The item stream is therefore walked as a machine
// with four accumulators rather than mapped one item at a time.
func DecodeResponses(body []byte) (LlmRequest, error) {
	fields, err := objectFields(body)
	if err != nil {
		return LlmRequest{}, &DecodeError{Path: "$", Problem: err.Error()}
	}
	fields = stripCompactionMarkers(fields)
	if err := rejectAnthropicStyleFallbacks(fields); err != nil {
		return LlmRequest{}, err
	}

	request := LlmRequest{
		Model:  stringOrEmpty(fields, "model"),
		Stream: boolOrFalse(fields, "stream"),
		Sampling: SamplingParams{
			Temperature: float64OrNil(fields, "temperature"),
			TopP:        float64OrNil(fields, "top_p"),
		},
		Output: OutputParams{
			// A output budget that is not a count is absent rather than an error here,
			// unlike this format's sibling.
			MaxOutputTokens: uint64OrNil(rawField(fields, "max_output_tokens")),
		},
		Reasoning: ReasoningParams{
			Effort: responsesEffort(fields),
			Raw:    compactOrNil(fields, "reasoning"),
		},
	}
	if format, ok := decodeResponsesTextFormat(rawField(fields, "text")); ok {
		request.Output.ResponseFormat = format
		if enforced, ok := schemaEnforcedValue(format); ok {
			request.Output.IsSchemaEnforced = &enforced
		}
	}

	// A top-level instruction is a plain string in this format; anything else is not an
	// instruction at all.
	if instructions, ok := stringValue(rawField(fields, "instructions")); ok && instructions != "" {
		request.Instructions = append(request.Instructions, InstructionBlock{
			Role:    RoleSystem,
			Content: []ContentBlock{Text(instructions)},
		})
	}

	// With a stored response the provider holds the earlier turns, so a tool output may
	// answer a call that is not in this body — and then it stays a tool result rather
	// than degrading to readable text.
	storedState := false
	if previous, ok := stringValue(rawField(fields, "previous_response_id")); ok && previous != "" {
		storedState = true
	}

	messages, instructions, err := decodeResponsesInput(rawField(fields, "input"), storedState)
	if err != nil {
		return LlmRequest{}, err
	}
	request.Messages = messages
	request.Instructions = append(request.Instructions, instructions...)

	request.Tools = decodeResponsesTools(rawField(fields, "tools"))
	// A Responses-lite client carries its definitions inside `input` instead of at the
	// top level. They are this request's tools; the item itself is not conversation.
	request.Tools = append(request.Tools, additionalTools(rawField(fields, "input"))...)
	if raw, present := field(fields, "tool_choice"); present {
		request.ToolChoice = decodeResponsesToolChoice(raw)
	}

	for _, candidate := range fields {
		if responsesKnownFields[candidate.Name] {
			continue
		}
		request.Extensions = append(request.Extensions, ExtensionField{
			Name:  candidate.Name,
			Value: CompactJSON(candidate.Value),
		})
	}
	return request, nil
}

// responsesKnownFields are the top-level fields the decoder recognizes.
var responsesKnownFields = map[string]bool{
	"model": true, "instructions": true, "input": true, "tools": true,
	"tool_choice": true, "max_output_tokens": true, "text": true, "reasoning": true,
	"temperature": true, "top_p": true, "stream": true,
}

func responsesEffort(fields []jsonField) string {
	reasoning, err := objectFields(rawField(fields, "reasoning"))
	if err != nil {
		return ""
	}
	return stringOrEmpty(reasoning, "effort")
}

// rejectAnthropicStyleFallbacks is not part of this format, and a body carrying
// Anthropic's fallback fields is a caller mistake worth naming rather than ignoring.
func rejectAnthropicStyleFallbacks(fields []jsonField) error {
	_, credit := field(fields, "fallback_credit_token")
	_, fallbacks := field(fields, "fallbacks")
	if credit || fallbacks {
		return &DecodeError{
			Path:    "$.fallbacks",
			Problem: "fallback fields belong to the Anthropic Messages format",
		}
	}
	return nil
}

// stripCompactionMarkers drops the codex-internal compaction trigger item before
// anything reads the body. Strict upstream parsers reject it, and it is not conversation.
func stripCompactionMarkers(fields []jsonField) []jsonField {
	raw, present := field(fields, "input")
	if !present {
		return fields
	}
	var elements []json.RawMessage
	if err := json.Unmarshal(raw, &elements); err != nil {
		return fields
	}
	marked := false
	for _, element := range elements {
		if item, err := objectFields(element); err == nil {
			if kind, _ := stringValue(rawField(item, "type")); kind == "compaction_trigger" {
				marked = true
				break
			}
		}
	}
	if !marked {
		return fields
	}

	kept := make([]json.RawMessage, 0, len(elements))
	for _, element := range elements {
		if item, err := objectFields(element); err == nil {
			if kind, _ := stringValue(rawField(item, "type")); kind == "compaction_trigger" {
				continue
			}
		}
		kept = append(kept, element)
	}
	rebuilt, err := json.Marshal(kept)
	if err != nil {
		return fields
	}

	stripped := make([]jsonField, 0, len(fields))
	for _, candidate := range fields {
		if candidate.Name == "input" {
			stripped = append(stripped, jsonField{Name: "input", Value: rebuilt})
			continue
		}
		stripped = append(stripped, candidate)
	}
	return stripped
}

// decodeResponsesInput walks the item stream into messages and instructions.
//
// The four accumulators are the whole reason this is not a one-item-at-a-time mapping:
// a call and its result must end up adjacent, so a message arriving between them is held
// back rather than emitted; reasoning belongs to the assistant turn that produced it, so
// it rides with the calls rather than becoming a turn of its own; and reasoning with no
// turn to join falls back to a standalone assistant message.
func decodeResponsesInput(
	raw json.RawMessage, storedState bool,
) ([]Message, []InstructionBlock, error) {
	if text, ok := stringValue(raw); ok {
		return []Message{{Role: RoleUser, Content: []ContentBlock{Text(text)}}}, nil, nil
	}
	var elements []json.RawMessage
	if err := json.Unmarshal(raw, &elements); err != nil {
		return nil, nil, &DecodeError{Path: "$.input", Problem: "expected a string or an array"}
	}

	var (
		messages         []Message
		instructions     []InstructionBlock
		pendingCalls     []ToolCall
		pendingOutputs   []ToolResult
		deferred         []Message
		pendingReasoning []ContentBlock
	)

	for index, element := range elements {
		item, err := objectFields(element)
		if err != nil {
			continue
		}
		rawType, hasType := field(item, "type")
		itemType := ""
		if hasType {
			text, ok := stringValue(rawType)
			if !ok {
				return nil, nil, &DecodeError{
					Path:    fmt.Sprintf("$.input[%d].type", index),
					Problem: "expected a string",
				}
			}
			itemType = text
		}

		_, hasRole := field(item, "role")
		_, hasContent := field(item, "content")
		isMessage := itemType == "message" || (!hasType && hasRole && hasContent)

		switch {
		case isMessage:
			role, err := responsesRequestRole(item, index)
			if err != nil {
				return nil, nil, err
			}
			content := rawField(item, "content")
			// A built-in tool item is a top-level item; nested inside a message the
			// provider rejects the request, and this must too rather than normalize it.
			if nested := findBuiltinToolItem(content); nested >= 0 {
				path := fmt.Sprintf("$.input[%d].content.type", index)
				if nested > 0 {
					path = fmt.Sprintf("$.input[%d].content[%d].type", index, nested)
				}
				return nil, nil, &DecodeError{
					Path:    path,
					Problem: "built-in tool items must be top-level Responses input items",
				}
			}
			blocks := decodeResponsesContent(content)

			// Inline instructions are classified before any state transition, so they
			// cannot flush pending reasoning or split a tool block.
			if role == RoleSystem || role == RoleDeveloper {
				instructions = append(instructions, InstructionBlock{Role: role, Content: blocks})
				continue
			}
			if role == RoleAssistant && len(pendingCalls) == 0 && len(pendingReasoning) > 0 {
				// Reasoning preceding an assistant turn is that turn's own, so it is
				// folded in rather than surfacing as an empty assistant message.
				blocks = append(append([]ContentBlock{}, pendingReasoning...), blocks...)
				pendingReasoning = nil
			}
			if role != RoleAssistant {
				flushUnattachedReasoning(&messages, &pendingCalls, &pendingOutputs,
					&deferred, &pendingReasoning)
			}
			pushNonToolMessage(&messages, &pendingCalls, &pendingOutputs, &deferred,
				&pendingReasoning, Message{Role: role, Content: blocks})

		case itemType == "reasoning":
			// Reasoning after a completed tool block opens the next assistant turn, so
			// the finished block is closed first and the reasoning attaches where it
			// belongs.
			if len(pendingOutputs) > 0 {
				flushToolBlock(&messages, &pendingCalls, &pendingOutputs, &deferred,
					&pendingReasoning)
			}
			blocks, err := decodeResponsesReasoningItem(item)
			if err != nil {
				return nil, nil, err
			}
			pendingReasoning = append(pendingReasoning, blocks...)

		case itemType == "function_call":
			if len(pendingOutputs) > 0 {
				flushToolBlock(&messages, &pendingCalls, &pendingOutputs, &deferred,
					&pendingReasoning)
			}
			pendingCalls = append(pendingCalls, ToolCall{
				ID:        callID(item, index),
				Name:      qualifiedName(item),
				Arguments: argumentsOrEmpty(item),
			})

		case itemType == "function_call_output" || itemType == "custom_tool_call_output":
			toolCallID, _ := stringValue(rawField(item, "call_id"))
			var content []ContentBlock
			if output, present := field(item, "output"); present && isArray(output) {
				content = decodeResponsesContent(output)
			} else if present {
				content = []ContentBlock{Text(jsonStringValue(output))}
			} else {
				content = []ContentBlock{Text("")}
			}

			answersPending := false
			for _, call := range pendingCalls {
				if call.ID == toolCallID {
					answersPending = true
					break
				}
			}
			if !answersPending && !storedState {
				// Without a stored response this body is malformed, and an orphaned
				// output degrades to readable text rather than pretending to be a result.
				text := joinText(content, " ")
				pushNonToolMessage(&messages, &pendingCalls, &pendingOutputs, &deferred,
					&pendingReasoning, Message{Role: RoleUser, Content: []ContentBlock{
						Text(fmt.Sprintf("Tool result %s: %s", toolCallID, text))}})
				continue
			}
			pendingOutputs = append(pendingOutputs, ToolResult{
				ToolCallID: toolCallID,
				Content:    content,
			})

		case itemType == "custom_tool_call":
			if len(pendingOutputs) > 0 {
				flushToolBlock(&messages, &pendingCalls, &pendingOutputs, &deferred,
					&pendingReasoning)
			}
			// A freeform call carries a raw input string and rides through the IR as the
			// single argument of a function-style call.
			input, _ := stringValue(rawField(item, "input"))
			pendingCalls = append(pendingCalls, ToolCall{
				ID:        callID(item, index),
				Name:      stringOrEmpty(item, "name"),
				Arguments: objectLiteral(jsonField{"input", jsonLiteral(input)}),
			})

		case !hasType:
			return nil, nil, &DecodeError{
				Path:    fmt.Sprintf("$.input[%d].type", index),
				Problem: "missing type discriminator on a non-message input item",
			}

		case itemType == "additional_tools":
			// Tool definitions rather than conversation, decoded with the top-level tools.

		default:
			flushUnattachedReasoning(&messages, &pendingCalls, &pendingOutputs, &deferred,
				&pendingReasoning)
			pushNonToolMessage(&messages, &pendingCalls, &pendingOutputs, &deferred,
				&pendingReasoning, Message{Role: RoleUser, Content: []ContentBlock{{
					Kind:     ContentUnknown,
					Provider: "openai_responses",
					Raw:      objectLiteral(item...),
				}}})
		}
	}

	if len(pendingCalls) > 0 || len(pendingOutputs) > 0 {
		flushToolBlock(&messages, &pendingCalls, &pendingOutputs, &deferred, &pendingReasoning)
	}
	// Trailing reasoning with no assistant turn to join keeps its standalone shape.
	if len(pendingReasoning) > 0 {
		messages = append(messages, Message{Role: RoleAssistant, Content: pendingReasoning})
	}
	return messages, instructions, nil
}

// pushNonToolMessage places a message without breaking tool-call adjacency.
//
// While calls are pending and no result has arrived, a message in between would split a
// call from the result that answers it, so it is held back and emitted after the block.
func pushNonToolMessage(
	messages *[]Message,
	pendingCalls *[]ToolCall,
	pendingOutputs *[]ToolResult,
	deferred *[]Message,
	pendingReasoning *[]ContentBlock,
	message Message,
) {
	if len(*pendingCalls) > 0 && len(*pendingOutputs) == 0 {
		*deferred = append(*deferred, message)
		return
	}
	if len(*pendingCalls) > 0 || len(*pendingOutputs) > 0 {
		flushToolBlock(messages, pendingCalls, pendingOutputs, deferred, pendingReasoning)
	}
	*messages = append(*messages, message)
}

// flushUnattachedReasoning emits reasoning that has no assistant turn to join, as a
// standalone assistant message at its own position. While calls are pending the
// reasoning belongs to the in-flight turn and stays pending for the flush.
func flushUnattachedReasoning(
	messages *[]Message,
	pendingCalls *[]ToolCall,
	pendingOutputs *[]ToolResult,
	deferred *[]Message,
	pendingReasoning *[]ContentBlock,
) {
	if len(*pendingReasoning) == 0 || len(*pendingCalls) > 0 {
		return
	}
	blocks := *pendingReasoning
	*pendingReasoning = nil
	pushNonToolMessage(messages, pendingCalls, pendingOutputs, deferred, pendingReasoning,
		Message{Role: RoleAssistant, Content: blocks})
}

// flushToolBlock closes a tool block, preserving adjacency.
//
// One assistant turn carries the pending reasoning *and* the calls, so a reasoning-plus-
// call turn stays a single assistant turn downstream rather than reading as two. Each
// result then gets its own user turn, and deferred messages follow the block they were
// held back from splitting.
func flushToolBlock(
	messages *[]Message,
	pendingCalls *[]ToolCall,
	pendingOutputs *[]ToolResult,
	deferred *[]Message,
	pendingReasoning *[]ContentBlock,
) {
	if len(*pendingCalls) > 0 {
		content := append([]ContentBlock{}, *pendingReasoning...)
		*pendingReasoning = nil
		for _, call := range *pendingCalls {
			content = append(content, ContentBlock{Kind: ContentToolCall, Call: &call})
		}
		*pendingCalls = nil
		*messages = append(*messages, Message{Role: RoleAssistant, Content: content})
	}
	for _, output := range *pendingOutputs {
		result := output
		*messages = append(*messages, Message{Role: RoleUser, Content: []ContentBlock{
			{Kind: ContentToolResult, Result: &result}}})
	}
	*pendingOutputs = nil
	*messages = append(*messages, *deferred...)
	*deferred = nil
}

// decodeResponsesReasoningItem reads a reasoning item into private reasoning content.
func decodeResponsesReasoningItem(item []jsonField) ([]ContentBlock, error) {
	if _, present := field(item, "encrypted_content"); present {
		// The encrypted form carries a provider signature that has to be replayed
		// intact, and decoding it is provider-specific work this package does not do.
		return nil, &DecodeError{
			Path:    "$.input[].encrypted_content",
			Problem: "encrypted reasoning is not ported",
		}
	}

	var parts []string
	parts = append(parts, collectReasoningText(rawField(item, "content"))...)
	parts = append(parts, collectReasoningText(rawField(item, "summary"))...)
	if text, ok := stringValue(rawField(item, "text")); ok {
		parts = append(parts, text)
	}
	return []ContentBlock{{
		Kind: ContentReasoning,
		Text: strings.Join(parts, "\n"),
	}}, nil
}

// collectReasoningText gathers reasoning text from the shapes this format uses: a bare
// string, or a list of parts naming themselves as reasoning or summary text.
func collectReasoningText(raw json.RawMessage) []string {
	if text, ok := stringValue(raw); ok {
		if text == "" {
			return nil
		}
		return []string{text}
	}
	var elements []json.RawMessage
	if err := json.Unmarshal(raw, &elements); err != nil {
		return nil
	}
	var parts []string
	for _, element := range elements {
		if text, ok := stringValue(element); ok {
			if text != "" {
				parts = append(parts, text)
			}
			continue
		}
		part, err := objectFields(element)
		if err != nil {
			continue
		}
		kind, _ := stringValue(rawField(part, "type"))
		if kind != "reasoning_text" && kind != "summary_text" && kind != "text" {
			continue
		}
		if text, ok := stringValue(rawField(part, "text")); ok && text != "" {
			parts = append(parts, text)
		}
	}
	return parts
}

// decodeResponsesContent reads content into blocks. A block type it does not know is
// carried verbatim, which is what makes a built-in tool output reachable later.
func decodeResponsesContent(raw json.RawMessage) []ContentBlock {
	if text, ok := stringValue(raw); ok {
		return []ContentBlock{Text(text)}
	}
	trimmed := compactOrNullBytes(raw)
	if isNull(trimmed) {
		return []ContentBlock{Text("")}
	}
	if trimmed[0] != '[' {
		return []ContentBlock{Text(renderAsText(trimmed))}
	}
	var elements []json.RawMessage
	if err := json.Unmarshal(trimmed, &elements); err != nil {
		return []ContentBlock{Text("")}
	}

	var out []ContentBlock
	for _, element := range elements {
		block, err := objectFields(element)
		if err != nil {
			continue
		}
		switch kind, _ := stringValue(rawField(block, "type")); kind {
		case "input_text", "output_text", "text":
			out = append(out, Text(stringOrEmpty(block, "text")))
		case "refusal":
			out = append(out, ContentBlock{
				Kind: ContentRefusal,
				Text: stringOrEmpty(block, "refusal"),
			})
		case "reasoning_text", "summary_text":
			out = append(out, ContentBlock{
				Kind: ContentReasoning,
				Text: stringOrEmpty(block, "text"),
			})
		case "input_image":
			// A source this cannot normalize yields no block rather than a placeholder,
			// which is what upstream does and what keeps the digest honest.
			if source := imageSource(element, block); source != nil {
				out = append(out, ContentBlock{Kind: ContentImage, Source: source})
			}
		case "input_audio":
			out = append(out, ContentBlock{
				Kind: ContentAudio, Source: objectLiteral(block...)})
		case "input_file":
			out = append(out, ContentBlock{
				Kind: ContentFile, Source: fileSource(element, block)})
		default:
			out = append(out, ContentBlock{
				Kind:     ContentUnknown,
				Provider: "openai_responses",
				Raw:      objectLiteral(block...),
			})
		}
	}
	if len(out) == 0 {
		return []ContentBlock{Text("")}
	}
	return out
}

// findBuiltinToolItem reports the index of a built-in tool item in a content value, or
// -1. They belong at the top level of `input`, and the provider rejects them anywhere
// else.
func findBuiltinToolItem(raw json.RawMessage) int {
	if len(raw) == 0 {
		return -1
	}
	// A bare item rather than a list: upstream rejects this shape before it can be a
	// block, so the check reports it as nested.
	if isBuiltinToolItem(raw) {
		return 0
	}
	var elements []json.RawMessage
	if err := json.Unmarshal(raw, &elements); err != nil {
		return -1
	}
	for index, element := range elements {
		if isBuiltinToolItem(element) {
			return index
		}
	}
	return -1
}

func isBuiltinToolItem(raw json.RawMessage) bool {
	item, err := objectFields(raw)
	if err != nil {
		return false
	}
	kind, _ := stringValue(rawField(item, "type"))
	switch kind {
	case "apply_patch_call", "apply_patch_call_output", "shell_call", "shell_call_output",
		"computer_call", "computer_call_output":
		return true
	}
	return false
}

// responsesRequestRole maps a request role, refusing one this format does not define.
func responsesRequestRole(item []jsonField, index int) (Role, error) {
	name, ok := stringValue(rawField(item, "role"))
	if !ok {
		return RoleUser, nil
	}
	switch name {
	case "assistant":
		return RoleAssistant, nil
	case "system":
		return RoleSystem, nil
	case "developer":
		return RoleDeveloper, nil
	}
	if knownRoleNames[name] {
		return RoleUser, nil
	}
	return "", &DecodeError{
		Path:    fmt.Sprintf("$.input[%d].role", index),
		Problem: fmt.Sprintf("unsupported role %q", name),
	}
}

// callID is a call's provider id, or a stable one derived from its position so a call
// without an id still pairs with the result that answers it.
func callID(item []jsonField, index int) string {
	if id, ok := stringValue(rawField(item, "call_id")); ok && id != "" {
		return id
	}
	return fmt.Sprintf("sw_%08d", index+1)
}

// qualifiedName spells a call the way its definition does, prefixing a namespace when
// the call carries one.
func qualifiedName(item []jsonField) string {
	name := stringOrEmpty(item, "name")
	if namespace, ok := stringValue(rawField(item, "namespace")); ok && namespace != "" {
		return namespace + "__" + name
	}
	return name
}

func argumentsOrEmpty(item []jsonField) json.RawMessage {
	if raw, present := field(item, "arguments"); present {
		return CompactJSON(raw)
	}
	return json.RawMessage("{}")
}

// decodeResponsesTools reads tool definitions, in the several shapes this format allows:
// a plain function, an OpenAI-style `function` wrapper, a freeform custom tool, a
// codex-style id tool, and namespace containers that qualify their children.
func decodeResponsesTools(raw json.RawMessage) []ToolDefinition {
	var elements []json.RawMessage
	if err := json.Unmarshal(raw, &elements); err != nil {
		return nil
	}
	var out []ToolDefinition
	for _, element := range elements {
		tool, err := objectFields(element)
		if err != nil {
			continue
		}
		kind, hasKind := stringValue(rawField(tool, "type"))

		switch {
		case hasKind && kind == "namespace":
			// A namespace qualifies its children so history spells a tool the way its
			// definition did. An unnamed container carries nothing to dispatch on, so
			// its children stay bare.
			container, _ := stringValue(rawField(tool, "name"))
			for _, child := range decodeResponsesTools(rawField(tool, "tools")) {
				if container != "" && !strings.Contains(child.Name, "__") {
					child.Name = container + "__" + child.Name
				}
				out = append(out, child)
			}

		case hasKind && kind == "custom":
			if name, ok := stringValue(rawField(tool, "name")); ok && name != "" {
				definition := ToolDefinition{
					Name: name,
					// A freeform tool takes one string, which the IR carries as the
					// single `input` argument of a function-style definition.
					Parameters: customToolInputSchema(),
				}
				if description, ok := stringValue(rawField(tool, "description")); ok {
					definition.Description = &description
				}
				out = append(out, definition)
			}

		case hasKind && kind == "function":
			if wrapper, err := objectFields(rawField(tool, "function")); err == nil {
				if definition, ok := functionTool(wrapper); ok {
					out = append(out, definition)
				}
			} else if definition, ok := functionTool(tool); ok {
				out = append(out, definition)
			} else if definition, ok := idTool(tool); ok {
				out = append(out, definition)
			}

		case !hasKind:
			if _, hasName := field(tool, "name"); hasName {
				if definition, ok := functionTool(tool); ok {
					out = append(out, definition)
				}
				continue
			}
			if definition, ok := idTool(tool); ok {
				out = append(out, definition)
			}

		default:
			if definition, ok := idTool(tool); ok {
				out = append(out, definition)
			}
		}
	}
	return out
}

// additionalTools reads tool definitions a Responses-lite client carries inside `input`.
func additionalTools(raw json.RawMessage) []ToolDefinition {
	var elements []json.RawMessage
	if err := json.Unmarshal(raw, &elements); err != nil {
		return nil
	}
	var out []ToolDefinition
	for _, element := range elements {
		item, err := objectFields(element)
		if err != nil {
			continue
		}
		if kind, _ := stringValue(rawField(item, "type")); kind != "additional_tools" {
			continue
		}
		out = append(out, decodeResponsesTools(rawField(item, "tools"))...)
	}
	return out
}

func functionTool(tool []jsonField) (ToolDefinition, bool) {
	name, ok := stringValue(rawField(tool, "name"))
	if !ok || name == "" {
		return ToolDefinition{}, false
	}
	definition := ToolDefinition{Name: name}
	if description, ok := stringValue(rawField(tool, "description")); ok {
		definition.Description = &description
	}
	definition.Parameters = json.RawMessage("{}")
	if raw, present := field(tool, "parameters"); present {
		definition.Parameters = CompactJSON(raw)
	}
	if strict, ok := boolValue(rawField(tool, "strict")); ok {
		definition.Strict = &strict
	}
	return definition, true
}

func idTool(tool []jsonField) (ToolDefinition, bool) {
	id, ok := stringValue(rawField(tool, "id"))
	if !ok || id == "" {
		return ToolDefinition{}, false
	}
	definition := ToolDefinition{Name: id}
	if description, ok := stringValue(rawField(tool, "description")); ok {
		definition.Description = &description
	}
	definition.Parameters = json.RawMessage("{}")
	if schema, err := objectFields(rawField(tool, "inputSchema")); err == nil {
		if raw, present := field(schema, "jsonSchema"); present {
			definition.Parameters = CompactJSON(raw)
		}
	}
	return definition, true
}

// customToolInputSchema is the schema a freeform tool's single argument takes.
func customToolInputSchema() json.RawMessage {
	return json.RawMessage(`{"properties":{"input":{"type":"string"}},"required":["input"],"type":"object"}`)
}

// decodeResponsesToolChoice normalizes the tool-choice policy. An object it cannot read
// yields no choice rather than a raw value, which is what upstream does: a shape it does
// not recognize is not a policy it can pass on.
func decodeResponsesToolChoice(raw json.RawMessage) *ToolChoice {
	if text, ok := stringValue(raw); ok {
		switch text {
		case "auto":
			return &ToolChoice{Kind: ToolChoiceAuto}
		case "required":
			return &ToolChoice{Kind: ToolChoiceRequired}
		case "none":
			return &ToolChoice{Kind: ToolChoiceNone}
		}
		return &ToolChoice{Kind: ToolChoiceRaw, Raw: CompactJSON(raw)}
	}
	choice, err := objectFields(raw)
	if err != nil {
		return &ToolChoice{Kind: ToolChoiceRaw, Raw: CompactJSON(raw)}
	}
	kind, _ := stringValue(rawField(choice, "type"))
	switch kind {
	case "function":
		if name, ok := stringValue(rawField(choice, "name")); ok {
			if namespace, ok := stringValue(rawField(choice, "namespace")); ok && namespace != "" {
				name = namespace + "__" + name
			}
			return &ToolChoice{Kind: ToolChoiceTool, Name: name}
		}
		return nil
	case "allowed_tools":
		return &ToolChoice{Kind: ToolChoiceRaw, Raw: CompactJSON(raw)}
	default:
		return nil
	}
}

// decodeResponsesTextFormat maps this format's structured-output field onto the neutral
// one, reporting whether anything was asked for.
func decodeResponsesTextFormat(raw json.RawMessage) (json.RawMessage, bool) {
	text, err := objectFields(raw)
	if err != nil {
		return nil, false
	}
	format, err := objectFields(rawField(text, "format"))
	if err != nil {
		return nil, false
	}
	kind, _ := stringValue(rawField(format, "type"))
	switch kind {
	case "json_schema":
		if nested, present := field(format, "json_schema"); present && isObject(nested) {
			return objectLiteral(
				jsonField{"type", jsonLiteral("json_schema")},
				jsonField{"json_schema", CompactJSON(nested)},
			), true
		}
		// Otherwise the schema fields sit on the format itself and are collected into a
		// nested object, in the order this format declares them.
		var pairs []jsonField
		for _, name := range []string{"name", "description", "schema", "strict"} {
			if value, present := field(format, name); present {
				pairs = append(pairs, jsonField{name, CompactJSON(value)})
			}
		}
		if len(pairs) == 0 {
			return nil, false
		}
		return objectLiteral(
			jsonField{"type", jsonLiteral("json_schema")},
			jsonField{"json_schema", objectLiteral(pairs...)},
		), true
	case "json_object":
		return objectLiteral(jsonField{"type", jsonLiteral("json_object")}), true
	case "text":
		return objectLiteral(jsonField{"type", jsonLiteral("text")}), true
	}
	return nil, false
}

// schemaEnforcedValue reports whether a response format demands strict adherence.
func schemaEnforcedValue(format json.RawMessage) (bool, bool) {
	fields, err := objectFields(format)
	if err != nil {
		return false, false
	}
	if kind, ok := stringValue(rawField(fields, "type")); !ok || kind != "json_schema" {
		return false, false
	}
	enforced := false
	if schema, err := objectFields(rawField(fields, "json_schema")); err == nil {
		if strict, ok := boolValue(rawField(schema, "strict")); ok {
			enforced = strict
		}
	}
	return enforced, true
}

func int64Value(raw json.RawMessage) (int64, bool) {
	var value int64
	if err := json.Unmarshal(bytes.TrimSpace(raw), &value); err != nil {
		return 0, false
	}
	return value, true
}

func isArray(raw json.RawMessage) bool {
	trimmed := compactOrNullBytes(raw)
	return len(trimmed) > 0 && trimmed[0] == '['
}

func isObject(raw json.RawMessage) bool {
	trimmed := compactOrNullBytes(raw)
	return len(trimmed) > 0 && trimmed[0] == '{'
}

// compactOrNullBytes is the raw value compacted, so a first byte can be inspected
// without a decoder.
func compactOrNullBytes(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return json.RawMessage("null")
	}
	return CompactJSON(raw)
}
