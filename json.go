package libsygo

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"strconv"
	"strings"
)

// The conformance harness diffs a digest of the normalized messages rather than the
// messages themselves, because a field-by-field diff of a transcript is noise.
// Reproducing that digest byte for byte is therefore a real constraint, and it is
// why this file writes JSON by hand instead of marshalling Go values:
//
//   - upstream serializes with `serde_json`'s `preserve_order`, so an object keeps
//     the provider's key order. `encoding/json` sorts map keys, which would reorder
//     a tool call's arguments and change the digest.
//   - `encoding/json` escapes `<`, `>`, `&`, U+2028 and U+2029; serde_json escapes
//     none of them.
//   - number spelling is kept as the provider wrote it rather than round-tripped
//     through a float.
//
// So provider values are carried as raw bytes and re-emitted via [CompactJSON], and
// only structure this port builds itself is serialized here — in upstream's exact
// field order.

// CompactJSON strips insignificant whitespace from a JSON value, leaving every
// other byte untouched: key order, number spelling and escapes all survive.
//
// This is the opposite of re-encoding, and deliberately so. It never fails:
// malformed input is compacted as far as it can be, because anything reaching here
// has already been parsed.
func CompactJSON(raw []byte) []byte {
	out := make([]byte, 0, len(raw))
	inString, escaped := false, false
	for _, b := range raw {
		if inString {
			out = append(out, b)
			switch {
			case escaped:
				escaped = false
			case b == '\\':
				escaped = true
			case b == '"':
				inString = false
			}
			continue
		}
		switch b {
		case '"':
			inString = true
			out = append(out, b)
		case ' ', '\t', '\n', '\r':
			// Insignificant outside a string.
		default:
			out = append(out, b)
		}
	}
	return out
}

const hexDigits = "0123456789abcdef"

// appendString writes a JSON string the way serde_json does. Go's encoder escapes
// more than serde does, which would produce a different digest for one message.
func appendString(dst []byte, s string) []byte {
	dst = append(dst, '"')
	for i := 0; i < len(s); i++ {
		b := s[i]
		switch {
		case b == '"':
			dst = append(dst, '\\', '"')
		case b == '\\':
			dst = append(dst, '\\', '\\')
		case b == '\b':
			dst = append(dst, '\\', 'b')
		case b == '\f':
			dst = append(dst, '\\', 'f')
		case b == '\n':
			dst = append(dst, '\\', 'n')
		case b == '\r':
			dst = append(dst, '\\', 'r')
		case b == '\t':
			dst = append(dst, '\\', 't')
		case b < 0x20:
			dst = append(dst, '\\', 'u', '0', '0', hexDigits[b>>4], hexDigits[b&0x0f])
		default:
			// UTF-8 passes through, and so do <, >, & and the line separators.
			dst = append(dst, b)
		}
	}
	return append(dst, '"')
}

// appendOptionalString writes a nullable string field, which upstream always emits
// rather than skipping.
func appendOptionalString(dst []byte, value *string) []byte {
	if value == nil {
		return append(dst, "null"...)
	}
	return appendString(dst, *value)
}

func appendOptionalBool(dst []byte, value *bool) []byte {
	if value == nil {
		return append(dst, "null"...)
	}
	return strconv.AppendBool(dst, *value)
}

func appendFloat(dst []byte, value float64) []byte {
	return append(dst, serdeFloat(value)...)
}

// serdeFloat renders a float the way serde_json does, which is not how Go does.
//
// An integral float keeps a fractional part, so 0.0 is not written as 0. The
// difference is visible rather than cosmetic: the harness compares numbers as
// serialized, so a confidence threshold of exactly 0 or 1 would otherwise be
// indistinguishable from an integer, and a float-valued score of 1 would not match.
func serdeFloat(value float64) json.Number {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		// Not representable in JSON; serde writes null.
		return json.Number("null")
	}
	formatted := strconv.FormatFloat(value, 'g', -1, 64)
	if !strings.ContainsAny(formatted, ".eE") {
		formatted += ".0"
	}
	return json.Number(formatted)
}

func appendInt(dst []byte, value int64) []byte {
	return strconv.AppendInt(dst, value, 10)
}

func appendUint(dst []byte, value uint64) []byte {
	return strconv.AppendUint(dst, value, 10)
}

func appendRaw(dst []byte, raw json.RawMessage) []byte {
	if len(raw) == 0 {
		return append(dst, "null"...)
	}
	return append(dst, CompactJSON(raw)...)
}

// AppendMessages serializes a message list in upstream's shape.
func AppendMessages(dst []byte, messages []Message) []byte {
	dst = append(dst, '[')
	for i, message := range messages {
		if i > 0 {
			dst = append(dst, ',')
		}
		dst = append(dst, `{"role":`...)
		dst = appendString(dst, string(message.Role))
		dst = append(dst, `,"content":`...)
		dst = appendBlocks(dst, message.Content)
		dst = append(dst, '}')
	}
	return append(dst, ']')
}

// AppendInstructions serializes an instruction list in upstream's shape.
func AppendInstructions(dst []byte, instructions []InstructionBlock) []byte {
	dst = append(dst, '[')
	for i, instruction := range instructions {
		if i > 0 {
			dst = append(dst, ',')
		}
		dst = append(dst, `{"role":`...)
		dst = appendString(dst, string(instruction.Role))
		dst = append(dst, `,"content":`...)
		dst = appendBlocks(dst, instruction.Content)
		dst = append(dst, '}')
	}
	return append(dst, ']')
}

func appendBlocks(dst []byte, blocks []ContentBlock) []byte {
	dst = append(dst, '[')
	for i, block := range blocks {
		if i > 0 {
			dst = append(dst, ',')
		}
		dst = appendBlock(dst, block)
	}
	return append(dst, ']')
}

func appendBlock(dst []byte, block ContentBlock) []byte {
	switch block.Kind {
	case ContentText:
		dst = append(dst, `{"type":"text","text":`...)
		dst = appendString(dst, block.Text)
		return append(dst, '}')

	case ContentRefusal:
		dst = append(dst, `{"type":"refusal","text":`...)
		dst = appendString(dst, block.Text)
		return append(dst, '}')

	case ContentReasoning:
		dst = append(dst, `{"type":"reasoning","text":`...)
		dst = appendString(dst, block.Text)
		dst = append(dst, `,"signature":`...)
		dst = appendOptionalString(dst, block.Signature)
		if len(block.Details) > 0 {
			dst = append(dst, `,"details":`...)
			dst = append(dst, '[')
			for i, detail := range block.Details {
				if i > 0 {
					dst = append(dst, ',')
				}
				dst = appendRaw(dst, detail)
			}
			dst = append(dst, ']')
		}
		return append(dst, '}')

	case ContentImage, ContentAudio, ContentVideo, ContentFile:
		dst = append(dst, `{"type":"`...)
		dst = append(dst, string(block.Kind)...)
		dst = append(dst, `","source":`...)
		dst = appendRaw(dst, block.Source)
		return append(dst, '}')

	case ContentToolCall:
		dst = append(dst, `{"type":"tool_call","id":`...)
		dst = appendString(dst, block.Call.ID)
		dst = append(dst, `,"name":`...)
		dst = appendString(dst, block.Call.Name)
		dst = append(dst, `,"arguments":`...)
		dst = appendRaw(dst, block.Call.Arguments)
		return append(dst, '}')

	case ContentToolResult:
		dst = append(dst, `{"type":"tool_result","tool_call_id":`...)
		dst = appendString(dst, block.Result.ToolCallID)
		dst = append(dst, `,"content":`...)
		dst = appendBlocks(dst, block.Result.Content)
		dst = append(dst, `,"is_error":`...)
		dst = appendOptionalBool(dst, block.Result.IsError)
		return append(dst, '}')

	default:
		dst = append(dst, `{"type":"unknown","provider":`...)
		dst = appendString(dst, block.Provider)
		dst = append(dst, `,"raw":`...)
		dst = appendRaw(dst, block.Raw)
		return append(dst, '}')
	}
}

// MessagesDigest is the fingerprint the conformance harness records: a SHA-256 over
// the compact JSON of the normalized messages.
func MessagesDigest(messages []Message) string {
	sum := sha256.Sum256(AppendMessages(nil, messages))
	return "sha256:" + hex.EncodeToString(sum[:])
}

// InstructionsJSON is the normalized instructions as JSON, for comparison.
func InstructionsJSON(instructions []InstructionBlock) json.RawMessage {
	return json.RawMessage(AppendInstructions(nil, instructions))
}
