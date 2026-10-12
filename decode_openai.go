package libsygo

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
)

// DecodeError is a request the normalizer refused.
//
// Refusing is deliberate. An unknown role means the provider sent something this
// port has no normalized equivalent for, and routing it as though it were a user
// turn would silently answer a different question than the caller asked. Upstream
// rejects it too; the point is that the port rejects rather than coerces.
type DecodeError struct {
	// Path is the JSON location of the offending value.
	Path string
	// Problem describes what was wrong with it.
	Problem string
}

func (e *DecodeError) Error() string {
	return fmt.Sprintf("%s: %s", e.Path, e.Problem)
}

// openAIChatFields are the top-level fields the decoder recognizes. Anything else
// is carried as a provider extension rather than dropped, so a caller can still
// read a field this port has no normalized place for.
var openAIChatFields = map[string]bool{
	"model": true, "messages": true, "stream": true, "temperature": true, "top_p": true,
	"max_completion_tokens": true, "max_tokens": true, "response_format": true,
	"reasoning_effort": true, "tools": true, "tool_choice": true,
}

// knownOpenAIRoles are role names the provider vocabulary includes but the
// normalized conversation does not. They map to a user turn, which is their
// historical meaning, rather than being refused as unknown.
var knownOpenAIRoles = map[string]bool{
	"user": true, "function": true,
}

// DecodeOpenAIChat normalizes an OpenAI Chat Completions request body.
//
// A field of the wrong JSON type is treated as absent rather than as an error,
// which is upstream's behaviour and what real traffic requires: providers send
// `null` where a string is expected, and refusing those would reject requests that
// route perfectly well.
func DecodeOpenAIChat(body []byte) (LlmRequest, error) {
	fields, err := objectFields(body)
	if err != nil {
		return LlmRequest{}, &DecodeError{Path: "$", Problem: err.Error()}
	}

	request := LlmRequest{
		Model:  stringOrEmpty(fields, "model"),
		Stream: boolOrFalse(fields, "stream"),
		Sampling: SamplingParams{
			Temperature: float64OrNil(fields, "temperature"),
			TopP:        float64OrNil(fields, "top_p"),
		},
		Output: OutputParams{
			MaxOutputTokens:  outputTokenBudget(fields),
			ResponseFormat:   compactOrNil(fields, "response_format"),
			IsSchemaEnforced: schemaEnforced(fields),
		},
		Reasoning: ReasoningParams{Effort: stringOrEmpty(fields, "reasoning_effort")},
	}

	if raw, present := field(fields, "messages"); present {
		var elements []json.RawMessage
		if err := json.Unmarshal(raw, &elements); err != nil {
			return LlmRequest{}, &DecodeError{Path: "$.messages", Problem: "expected an array"}
		}
		generated := 0
		for index, element := range elements {
			generated, err = decodeOpenAIMessage(&request, element, index, generated)
			if err != nil {
				return LlmRequest{}, err
			}
		}
	}

	tools, _ := field(fields, "tools")
	request.Tools = decodeOpenAITools(tools)
	if raw, present := field(fields, "tool_choice"); present {
		request.ToolChoice = decodeOpenAIToolChoice(raw)
	}
	for _, candidate := range fields {
		if openAIChatFields[candidate.Name] {
			continue
		}
		request.Extensions = append(request.Extensions, ExtensionField{
			Name:  candidate.Name,
			Value: CompactJSON(candidate.Value),
		})
	}

	return request, nil
}

// outputTokenBudget applies upstream's precedence, including its quirk: an
// explicit `null` under the preferred key wins over the fallback key rather than
// falling through to it.
func outputTokenBudget(fields []jsonField) *uint64 {
	if raw, present := field(fields, "max_completion_tokens"); present {
		return uint64OrNil(raw)
	}
	if raw, present := field(fields, "max_tokens"); present {
		return uint64OrNil(raw)
	}
	return nil
}

// schemaEnforced reports whether a `json_schema` response format demands strict
// adherence, and nothing at all for any other format.
func schemaEnforced(fields []jsonField) *bool {
	format, err := objectFields(rawField(fields, "response_format"))
	if err != nil {
		return nil
	}
	if kind, ok := stringValue(rawField(format, "type")); !ok || kind != "json_schema" {
		return nil
	}
	enforced := false
	if schema, err := objectFields(rawField(format, "json_schema")); err == nil {
		if strict, ok := boolValue(rawField(schema, "strict")); ok {
			enforced = strict
		}
	}
	return &enforced
}

// decodeOpenAIMessage folds one provider message into the request, returning the
// running tool-call counter: a call without a provider id is named from it, and
// the counter must not restart per message or two calls could collide.
func decodeOpenAIMessage(
	request *LlmRequest, element json.RawMessage, index, generated int,
) (int, error) {
	fields, err := objectFields(element)
	if err != nil {
		// Not an object. Upstream records a diagnostic and moves on; the request
		// is still routable, so it is not worth refusing.
		return generated, nil
	}

	role, err := openAIRole(fields, fmt.Sprintf("$.messages[%d].role", index))
	if err != nil {
		return generated, err
	}

	content, err := decodeOpenAIContent(
		rawField(fields, "content"),
		fmt.Sprintf("$.messages[%d].content", index),
	)
	if err != nil {
		return generated, err
	}
	content = prependReasoning(content, fields)

	if role == RoleAssistant {
		if raw, present := field(fields, "tool_calls"); present {
			var calls []json.RawMessage
			if err := json.Unmarshal(raw, &calls); err == nil {
				// A `content: null` assistant message decodes to an empty text
				// block, which is a placeholder for the calls rather than content.
				if isEmptyTextOnly(content) {
					content = nil
				}
				for _, rawCall := range calls {
					generated++
					if call := decodeOpenAIToolCall(rawCall, generated); call != nil {
						content = append(content, ContentBlock{Kind: ContentToolCall, Call: call})
					}
				}
			}
		}
	}

	if role == RoleTool {
		// A tool result becomes a user-role message carrying one result block.
		// Role alone therefore cannot tell a tool result from a user turn, which is
		// why the judge's selection looks at content as well.
		isError, hasError := boolValue(rawField(fields, "is_error"))
		result := &ToolResult{
			ToolCallID: stringOrEmpty(fields, "tool_call_id"),
			Content:    []ContentBlock{Text(joinText(content, "\n"))},
		}
		if hasError {
			result.IsError = &isError
		}
		request.Messages = append(request.Messages, Message{
			Role:    RoleUser,
			Content: []ContentBlock{{Kind: ContentToolResult, Result: result}},
		})
		return generated, nil
	}

	if role == RoleSystem || role == RoleDeveloper {
		request.Instructions = append(request.Instructions, InstructionBlock{Role: role, Content: content})
		return generated, nil
	}
	request.Messages = append(request.Messages, Message{Role: role, Content: content})
	return generated, nil
}

// openAIRole maps a provider role string. A missing or non-string role is a user
// turn; an unrecognised one is refused.
func openAIRole(fields []jsonField, path string) (Role, error) {
	name, ok := stringValue(rawField(fields, "role"))
	if !ok {
		return RoleUser, nil
	}
	switch name {
	case "system":
		return RoleSystem, nil
	case "developer":
		return RoleDeveloper, nil
	case "assistant":
		return RoleAssistant, nil
	case "tool":
		return RoleTool, nil
	}
	if knownOpenAIRoles[name] {
		return RoleUser, nil
	}
	return "", &DecodeError{
		Path:    path,
		Problem: fmt.Sprintf("unsupported role %q", name),
	}
}

// decodeOpenAIContent normalizes a message's content, which a provider may send as
// a string, as an array of parts, as null, or as something else entirely.
func decodeOpenAIContent(raw json.RawMessage, path string) ([]ContentBlock, error) {
	trimmed := bytes.TrimSpace(raw)
	switch {
	case len(trimmed) == 0 || isNull(trimmed):
		return []ContentBlock{Text("")}, nil
	case trimmed[0] == '"':
		text, _ := stringValue(trimmed)
		return []ContentBlock{Text(text)}, nil
	case trimmed[0] == '[':
		return decodeOpenAIContentArray(trimmed, path)
	default:
		// A number, boolean or object where content belongs. Upstream renders it as
		// text rather than refusing the request.
		return []ContentBlock{Text(renderAsText(trimmed))}, nil
	}
}

func decodeOpenAIContentArray(raw json.RawMessage, path string) ([]ContentBlock, error) {
	var elements []json.RawMessage
	if err := json.Unmarshal(raw, &elements); err != nil {
		return []ContentBlock{Text("")}, nil
	}

	var content []ContentBlock
	for _, element := range elements {
		fields, err := objectFields(element)
		if err != nil {
			continue
		}
		kind, _ := stringValue(rawField(fields, "type"))
		switch kind {
		case "text", "input_text", "output_text":
			content = append(content, Text(stringOrEmpty(fields, "text")))
		case "refusal":
			content = append(content, ContentBlock{
				Kind: ContentRefusal,
				Text: stringOrEmpty(fields, "refusal"),
			})
		case "image_url", "input_image":
			// A source upstream cannot normalize yields no block at all, rather
			// than a placeholder — so an empty return here is not an error.
			if source := imageSource(element, fields); source != nil {
				content = append(content, ContentBlock{Kind: ContentImage, Source: source})
			}
		case "input_audio":
			content = append(content, ContentBlock{Kind: ContentAudio, Source: CompactJSON(element)})
		case "file", "input_file":
			content = append(content, ContentBlock{Kind: ContentFile, Source: fileSource(element, fields)})
		default:
			content = append(content, ContentBlock{
				Kind:     ContentUnknown,
				Provider: "openai_chat",
				Raw:      CompactJSON(element),
			})
		}
	}
	if len(content) == 0 {
		return []ContentBlock{Text("")}, nil
	}
	return content, nil
}

// imageSource normalizes the two shapes an image part takes: a bare URL string
// nested under `image_url`, or an object carrying the URL and detail level.
func imageSource(element json.RawMessage, fields []jsonField) json.RawMessage {
	raw, present := field(fields, "image_url")
	if !present {
		return taggedSource("raw", element)
	}
	if url, ok := stringValue(raw); ok {
		return taggedSource("url", objectLiteral(
			jsonField{"url", jsonLiteral(url)},
			jsonField{"detail", optionalString(rawField(fields, "detail"))},
		))
	}
	payload, err := objectFields(raw)
	if err != nil {
		return taggedSource("raw", element)
	}
	url, ok := stringValue(rawField(payload, "url"))
	if !ok {
		// An `image_url` object with no usable URL produces no source at all.
		return nil
	}
	detail, present := field(payload, "detail")
	if !present {
		detail = rawField(fields, "detail")
	}
	return taggedSource("url", objectLiteral(
		jsonField{"url", jsonLiteral(url)},
		jsonField{"detail", optionalString(compact(detail))},
	))
}

// fileSource normalizes the Chat shape, which nests the payload under `file`, and
// the Responses shape, which puts the fields on the part itself.
func fileSource(element json.RawMessage, fields []jsonField) json.RawMessage {
	if raw, present := field(fields, "file"); present {
		if file, err := objectFields(raw); err == nil {
			if id, ok := stringValue(rawField(file, "file_id")); ok {
				return taggedSource("file_id", jsonLiteral(id))
			}
			if data, ok := stringValue(rawField(file, "file_data")); ok {
				filename, _ := stringValue(rawField(file, "filename"))
				return taggedSource("file_data", fileDataObject(data, filename))
			}
			return taggedSource("raw", raw)
		}
	}
	if id, ok := stringValue(rawField(fields, "file_id")); ok {
		return taggedSource("file_id", jsonLiteral(id))
	}
	if data, ok := stringValue(rawField(fields, "file_data")); ok {
		filename, _ := stringValue(rawField(fields, "filename"))
		return taggedSource("file_data", fileDataObject(data, filename))
	}
	return taggedSource("raw", element)
}

func fileDataObject(data, filename string) json.RawMessage {
	value := jsonField{"filename", jsonNull()}
	if filename != "" {
		value = jsonField{"filename", jsonLiteral(filename)}
	}
	return objectLiteral(jsonField{"data", jsonLiteral(data)}, value)
}

// prependReasoning lifts provider reasoning into a private block ahead of the
// content, so it can be replayed without being mistaken for something the model
// said in public.
func prependReasoning(content []ContentBlock, fields []jsonField) []ContentBlock {
	if raw, present := field(fields, "reasoning_details"); present {
		var details []json.RawMessage
		if err := json.Unmarshal(raw, &details); err == nil && len(details) > 0 {
			text, ok := reasoningTextFromDetails(details)
			if !ok {
				text = firstNonEmptyString(fields, "reasoning_content", "reasoning")
			}
			var signature *string
			for _, detail := range details {
				detailFields, err := objectFields(detail)
				if err != nil {
					continue
				}
				if value, ok := stringValue(rawField(detailFields, "signature")); ok && value != "" {
					signature = &value
					break
				}
			}
			return append([]ContentBlock{{
				Kind:      ContentReasoning,
				Text:      text,
				Signature: signature,
				Details:   details,
			}}, content...)
		}
	}
	if text := firstNonEmptyString(fields, "reasoning_content", "reasoning"); text != "" {
		return append([]ContentBlock{{Kind: ContentReasoning, Text: text}}, content...)
	}
	return content
}

// reasoningTextFromDetails concatenates the reasoning carried by structured
// details, preferring each detail's own text over its summary.
func reasoningTextFromDetails(details []json.RawMessage) (string, bool) {
	var parts []string
	for _, detail := range details {
		fields, err := objectFields(detail)
		if err != nil {
			continue
		}
		if text, ok := stringValue(rawField(fields, "text")); ok && text != "" {
			parts = append(parts, text)
			continue
		}
		if summary, ok := stringValue(rawField(fields, "summary")); ok && summary != "" {
			parts = append(parts, summary)
		}
	}
	if len(parts) == 0 {
		return "", false
	}
	return joinTextParts(parts, "\n"), true
}

// decodeOpenAIToolCall normalizes one tool invocation, which the provider may send
// as a function call or as a freeform custom call.
func decodeOpenAIToolCall(raw json.RawMessage, generated int) *ToolCall {
	fields, err := objectFields(raw)
	if err != nil {
		return nil
	}

	function := fields
	if rawFunction, present := field(fields, "function"); present {
		if parsed, err := objectFields(rawFunction); err == nil {
			function = parsed
		}
	}

	id, _ := stringValue(rawField(fields, "id"))
	if id == "" {
		// Stable and derived from the call's position, so a call without a
		// provider id still pairs with the result that answers it.
		id = fmt.Sprintf("sw_%08d", generated)
	}

	if rawCustom, present := field(fields, "custom"); present {
		if custom, err := objectFields(rawCustom); err == nil {
			input, _ := stringValue(rawField(custom, "input"))
			name, _ := stringValue(rawField(custom, "name"))
			return &ToolCall{
				ID:        id,
				Name:      name,
				Arguments: objectLiteral(jsonField{"input", jsonLiteral(input)}),
			}
		}
	}

	arguments := json.RawMessage("{}")
	if rawArguments, present := field(function, "arguments"); present {
		arguments = parseArguments(rawArguments)
	}
	name, _ := stringValue(rawField(function, "name"))
	return &ToolCall{ID: id, Name: name, Arguments: arguments}
}

// parseArguments parses stringified tool arguments, which is how most providers
// send them. Arguments that are not JSON are carried under a `raw` key rather than
// dropped, so a host can still see what the model asked for.
func parseArguments(raw json.RawMessage) json.RawMessage {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return json.RawMessage("{}")
	}
	text, isString := stringValue(trimmed)
	if !isString {
		// Already a value. Null stays null, matching upstream's clone.
		return CompactJSON(trimmed)
	}
	var parsed json.RawMessage
	if err := json.Unmarshal([]byte(text), &parsed); err != nil {
		return objectLiteral(jsonField{"raw", jsonLiteral(text)})
	}
	return CompactJSON(parsed)
}

// decodeOpenAITools normalizes tool definitions. A definition without a name is
// dropped, because it cannot be called.
func decodeOpenAITools(raw json.RawMessage) []ToolDefinition {
	if len(raw) == 0 {
		return nil
	}
	var elements []json.RawMessage
	if err := json.Unmarshal(raw, &elements); err != nil {
		return nil
	}

	var definitions []ToolDefinition
	for _, element := range elements {
		fields, err := objectFields(element)
		if err != nil {
			continue
		}
		function := fields
		if rawFunction, present := field(fields, "function"); present {
			if parsed, err := objectFields(rawFunction); err == nil {
				function = parsed
			}
		}
		name, ok := stringValue(rawField(function, "name"))
		if !ok || name == "" {
			continue
		}

		definition := ToolDefinition{Name: name}
		if description, ok := stringValue(rawField(function, "description")); ok {
			definition.Description = &description
		}
		definition.Parameters = json.RawMessage("{}")
		if rawParameters, present := field(function, "parameters"); present {
			definition.Parameters = CompactJSON(rawParameters)
		}
		if strict, ok := boolValue(rawField(function, "strict")); ok {
			definition.Strict = &strict
		}
		definitions = append(definitions, definition)
	}
	return definitions
}

// decodeOpenAIToolChoice normalizes the tool-choice policy, keeping a value it
// cannot represent rather than narrowing it to a policy it does not mean.
func decodeOpenAIToolChoice(raw json.RawMessage) *ToolChoice {
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
	if fields, err := objectFields(raw); err == nil {
		if rawFunction, present := field(fields, "function"); present {
			if function, err := objectFields(rawFunction); err == nil {
				if name, ok := stringValue(rawField(function, "name")); ok {
					return &ToolChoice{Kind: ToolChoiceTool, Name: name}
				}
			}
		}
	}
	return &ToolChoice{Kind: ToolChoiceRaw, Raw: CompactJSON(raw)}
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

// jsonField is one field of a JSON object, in document order.
type jsonField struct {
	Name  string
	Value json.RawMessage
}

// objectFields decodes a JSON object into its fields in document order.
//
// Order is preserved throughout because provider values are carried as bytes and
// re-emitted: a Go map would lose the order before the digest is computed, and the
// digest is what says whether two implementations agree.
func objectFields(raw []byte) ([]jsonField, error) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return nil, errors.New("no value")
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	token, err := decoder.Token()
	if err != nil {
		return nil, err
	}
	if delim, ok := token.(json.Delim); !ok || delim != '{' {
		return nil, errors.New("expected an object")
	}

	var fields []jsonField
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return nil, err
		}
		name, ok := token.(string)
		if !ok {
			return nil, errors.New("expected a field name")
		}
		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			return nil, err
		}
		fields = append(fields, jsonField{Name: name, Value: value})
	}
	return fields, nil
}

// field returns a named field's value. A repeated key resolves to the last one,
// which is what an ordered map does.
func field(fields []jsonField, name string) (json.RawMessage, bool) {
	for index := len(fields) - 1; index >= 0; index-- {
		if fields[index].Name == name {
			return fields[index].Value, true
		}
	}
	return nil, false
}

func rawField(fields []jsonField, name string) json.RawMessage {
	value, _ := field(fields, name)
	return value
}

// objectLiteral builds a compact JSON object from already-serialized values.
func objectLiteral(pairs ...jsonField) json.RawMessage {
	var buffer []byte
	buffer = append(buffer, '{')
	for index, pair := range pairs {
		if index > 0 {
			buffer = append(buffer, ',')
		}
		buffer = appendString(buffer, pair.Name)
		buffer = append(buffer, ':')
		buffer = append(buffer, compact(pair.Value)...)
	}
	return append(buffer, '}')
}

// taggedSource builds an internally-tagged enum value, the shape upstream uses for
// media and file sources: `{"type": <kind>, "data": <payload>}`.
func taggedSource(kind string, payload json.RawMessage) json.RawMessage {
	var buffer []byte
	buffer = append(buffer, `{"type":`...)
	buffer = appendString(buffer, kind)
	buffer = append(buffer, `,"data":`...)
	buffer = append(buffer, compact(payload)...)
	return append(buffer, '}')
}

func jsonLiteral(value string) json.RawMessage {
	return json.RawMessage(appendString(nil, value))
}

func jsonNull() json.RawMessage {
	return json.RawMessage("null")
}

// optionalString serializes a present-or-absent string, emitting null when absent,
// which is what an unserialized Option does.
func optionalString(raw json.RawMessage) json.RawMessage {
	if value, ok := stringValue(raw); ok {
		return jsonLiteral(value)
	}
	return jsonNull()
}

func compact(raw json.RawMessage) []byte {
	if len(raw) == 0 {
		return []byte("null")
	}
	return CompactJSON(raw)
}

func compactOrNil(fields []jsonField, name string) json.RawMessage {
	raw, present := field(fields, name)
	if !present {
		return nil
	}
	return CompactJSON(raw)
}

func isNull(raw json.RawMessage) bool {
	return bytes.Equal(bytes.TrimSpace(raw), []byte("null"))
}

// stringValue reads a JSON string, reporting whether the value was one. Null,
// numbers, booleans and containers are all "not a string", matching upstream's
// `Value::as_str`.
func stringValue(raw json.RawMessage) (string, bool) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || isNull(trimmed) {
		return "", false
	}
	var text string
	if err := json.Unmarshal(trimmed, &text); err != nil {
		return "", false
	}
	return text, true
}

func boolValue(raw json.RawMessage) (bool, bool) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || isNull(trimmed) {
		return false, false
	}
	var value bool
	if err := json.Unmarshal(trimmed, &value); err != nil {
		return false, false
	}
	return value, true
}

func uint64OrNil(raw json.RawMessage) *uint64 {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || isNull(trimmed) {
		return nil
	}
	var value uint64
	if err := json.Unmarshal(trimmed, &value); err != nil {
		return nil
	}
	return &value
}

func float64OrNil(fields []jsonField, name string) *float64 {
	raw, present := field(fields, name)
	if !present || isNull(raw) {
		return nil
	}
	var value float64
	if err := json.Unmarshal(bytes.TrimSpace(raw), &value); err != nil {
		return nil
	}
	return &value
}

func boolOrFalse(fields []jsonField, name string) bool {
	value, _ := boolValue(rawField(fields, name))
	return value
}

func stringOrEmpty(fields []jsonField, name string) string {
	value, _ := stringValue(rawField(fields, name))
	return value
}

// firstNonEmptyString returns the first named field that holds a non-empty string.
func firstNonEmptyString(fields []jsonField, names ...string) string {
	for _, name := range names {
		if value := stringOrEmpty(fields, name); value != "" {
			return value
		}
	}
	return ""
}

// renderAsText renders a JSON value as text, which is upstream's fallback when a
// content field holds something that is neither a string nor a part list.
func renderAsText(raw json.RawMessage) string {
	trimmed := bytes.TrimSpace(raw)
	switch {
	case len(trimmed) == 0 || isNull(trimmed):
		return ""
	case bytes.Equal(trimmed, []byte("true")):
		return "True"
	case bytes.Equal(trimmed, []byte("false")):
		return "False"
	}
	if text, ok := stringValue(trimmed); ok {
		return text
	}
	return string(CompactJSON(trimmed))
}

// isEmptyTextOnly reports whether content is the placeholder empty text block that
// decoding a null content field produces.
func isEmptyTextOnly(content []ContentBlock) bool {
	return len(content) == 1 &&
		content[0].Kind == ContentText &&
		content[0].Text == ""
}

// joinText concatenates text-like blocks, matching upstream's text extraction.
func joinText(content []ContentBlock, separator string) string {
	var parts []string
	for _, block := range content {
		switch block.Kind {
		case ContentText, ContentRefusal:
			parts = append(parts, block.Text)
		}
	}
	return joinTextParts(parts, separator)
}

func joinTextParts(parts []string, separator string) string {
	if len(parts) == 0 {
		return ""
	}
	joined := parts[0]
	for _, part := range parts[1:] {
		joined += separator + part
	}
	return joined
}
