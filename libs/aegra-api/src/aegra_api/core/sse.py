"""Server-Sent Events utilities and formatting"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from aegra_api.core.serializers import GeneralSerializer

# Global serializer instance
_serializer = GeneralSerializer()

# Some LLMs stream tool_call_chunks.args with literal \uXXXX sequences
# instead of actual Unicode characters. After json.dumps these become \\uXXXX (double-escaped).
# We decode them back in two passes: surrogate pairs first to avoid lone surrogates that
# cannot be encoded to UTF-8, then remaining non-ASCII, non-surrogate code points.
# ASCII control characters (< 0x80) are left intact to preserve JSON validity.
_SURROGATE_PAIR_RE = re.compile(
    r"\\\\u(D[89AB][0-9a-fA-F]{2})\\\\u(D[C-F][0-9a-fA-F]{2})",
    re.IGNORECASE,
)
_NONASCII_ESCAPE_RE = re.compile(r"\\\\u([0-9a-fA-F]{4})")


def _decode_literal_unicode_escapes(data_str: str) -> str:
    """Decode double-escaped \\uXXXX sequences in an already-JSON-encoded string."""
    if "\\u" not in data_str:
        return data_str
    # First pass: combine surrogate pairs (e.g. \\uD83D\\uDE00 → 😀)
    data_str = _SURROGATE_PAIR_RE.sub(
        lambda m: chr(((int(m.group(1), 16) - 0xD800) << 10) + (int(m.group(2), 16) - 0xDC00) + 0x10000),
        data_str,
    )
    # Second pass: decode remaining non-ASCII, non-surrogate escapes
    return _NONASCII_ESCAPE_RE.sub(
        lambda m: chr(cp) if (cp := int(m.group(1), 16)) >= 0x80 and not (0xD800 <= cp <= 0xDFFF) else m.group(0),
        data_str,
    )


_SENSITIVE_KEYS: set[str] = {
    "system_prompt",
}

_DROP_MESSAGE_CLASSNAMES: set[str] = {"SystemMessage"}
_INTERNAL_STATE_KEYS: set[str] = {
    "notes",
    "raw_notes",
}


def _is_message_like(obj: object) -> bool:
    if isinstance(obj, dict):
        return "content" in obj or "type" in obj
    return hasattr(obj, "content")


def _sanitize_message_obj(obj: object) -> object | None:
    cls_name = getattr(obj, "__class__", type(obj)).__name__
    if cls_name in _DROP_MESSAGE_CLASSNAMES:
        return None

    if isinstance(obj, dict):
        t = obj.get("type")
        if isinstance(t, str) and t.lower() == "system":
            return None
        return obj

    return obj


def _is_override_wrapper(value: object) -> bool:
    return isinstance(value, dict) and value.get("type") == "override" and "value" in value


def _sanitize_dict(d: dict) -> dict:
    safe: dict[str, object] = {}
    for k, v in d.items():
        if k in _SENSITIVE_KEYS:
            continue
        if k in _INTERNAL_STATE_KEYS:
            continue
        if k == "research_brief" and _is_override_wrapper(v):
            # Keep response shape stable but unwrap internal override payload.
            safe[k] = _sanitize_any(v.get("value"))
            continue
        if k == "supervisor_messages" and isinstance(v, list):
            filtered = [msg for msg in v if not (isinstance(msg, dict) and msg.get("type") in {"human", "system"})]
            safe[k] = _sanitize_any(filtered) or []
            continue
        if k == "supervisor_messages" and isinstance(v, dict) and isinstance(v.get("value"), list):
            inner = v.get("value")
            filtered_inner = [
                msg for msg in inner if not (isinstance(msg, dict) and msg.get("type") in {"human", "system"})
            ]
            wrapper = dict(v)
            wrapper["value"] = _sanitize_any(filtered_inner) or []
            safe[k] = wrapper
            continue
        if _is_message_like(v):
            reduced = _sanitize_message_obj(v)
            if reduced is None:
                continue
            safe[k] = _sanitize_any(reduced)
        else:
            safe[k] = _sanitize_any(v)
    return safe


def _sanitize_sequence(seq: list | tuple) -> list:
    """Sanitize each element in a list/tuple and drop Nones."""
    out: list[object] = []
    for item in seq:
        if _is_message_like(item):
            reduced = _sanitize_message_obj(item)
            if reduced is None:
                continue
            out.append(_sanitize_any(reduced))
        else:
            sanitized = _sanitize_any(item)
            if sanitized is not None:
                out.append(sanitized)
    return out


def _sanitize_any(obj: object) -> object | None:
    """Recursively sanitize arbitrary payloads for SSE"""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, dict):
        return _sanitize_dict(obj)

    if isinstance(obj, (list, tuple)):
        return _sanitize_sequence(list(obj))

    if _is_message_like(obj):
        return _sanitize_message_obj(obj)

    return str(obj)


def get_sse_headers() -> dict[str, str]:
    """Get standard SSE headers"""
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream",
        "X-Accel-Buffering": "no",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Last-Event-ID",
    }


def format_sse_message(
    event: str,
    data: Any,
    event_id: str | None = None,
    serializer: Callable[[Any], Any] | None = None,
) -> str:
    """Format a message as Server-Sent Event following SSE standard

    Args:
        event: SSE event type
        data: Data to serialize and send
        event_id: Optional event ID
        serializer: Optional custom serializer function
    """
    lines = []

    lines.append(f"event: {event}")

    # Convert data to JSON string
    if data is None:
        data_str = ""
    else:
        # Use our general serializer by default to handle complex objects
        default_serializer = serializer or _serializer.serialize
        if event.startswith("messages"):
            if isinstance(data, list) and len(data) == 2:
                message_chunk, metadata = data[0], data[1]
                if isinstance(message_chunk, (SystemMessage, HumanMessage)):
                    message_chunk.content = ""
                    message_chunk.id = f"do-not-render-{message_chunk.id}"
                elif isinstance(message_chunk, dict):
                    msg_type = str(message_chunk.get("type", "")).lower()
                    if msg_type in {"human", "system"}:
                        message_chunk = {
                            **message_chunk,
                            "content": "",
                            "id": f"do-not-render-{message_chunk.get('id', '')}",
                        }
                payload = [message_chunk, _sanitize_any(metadata)]
            else:
                payload = _sanitize_any(data)
        else:
            payload = _sanitize_any(data)
        data_str = json.dumps(payload, default=default_serializer, separators=(",", ":"), ensure_ascii=False)
        data_str = _decode_literal_unicode_escapes(data_str)

    lines.append(f"data: {data_str}")

    if event_id:
        lines.append(f"id: {event_id}")

    lines.append("")  # Empty line to end the event

    return "\n".join(lines) + "\n"


def create_metadata_event(run_id: str, event_id: str | None = None, attempt: int = 1) -> str:
    """Create metadata event for LangSmith Studio compatibility"""
    data = {"run_id": run_id, "attempt": attempt}
    return format_sse_message("metadata", data, event_id)


def create_debug_event(debug_data: dict[str, Any], event_id: str | None = None) -> str:
    """Create debug event with checkpoint fields for LangSmith Studio compatibility"""

    # Add checkpoint and parent_checkpoint fields if not present
    if "payload" in debug_data and isinstance(debug_data["payload"], dict):
        payload = debug_data["payload"]

        # Extract checkpoint from config.configurable
        if "checkpoint" not in payload and "config" in payload:
            config = payload.get("config", {})
            if isinstance(config, dict) and "configurable" in config:
                configurable = config["configurable"]
                if isinstance(configurable, dict):
                    payload["checkpoint"] = {
                        "thread_id": configurable.get("thread_id"),
                        "checkpoint_id": configurable.get("checkpoint_id"),
                        "checkpoint_ns": configurable.get("checkpoint_ns", ""),
                    }

        # Extract parent_checkpoint from parent_config.configurable
        if "parent_checkpoint" not in payload and "parent_config" in payload:
            parent_config = payload.get("parent_config")
            if isinstance(parent_config, dict) and "configurable" in parent_config:
                configurable = parent_config["configurable"]
                if isinstance(configurable, dict):
                    payload["parent_checkpoint"] = {
                        "thread_id": configurable.get("thread_id"),
                        "checkpoint_id": configurable.get("checkpoint_id"),
                        "checkpoint_ns": configurable.get("checkpoint_ns", ""),
                    }
            elif parent_config is None:
                payload["parent_checkpoint"] = None

    return format_sse_message("debug", debug_data, event_id)


def create_end_event(event_id: str | None = None, status: str = "success") -> str:
    """Create end event — signals completion of stream."""
    return format_sse_message("end", {"status": status}, event_id)


def create_error_event(error: str | dict[str, Any], event_id: str | None = None) -> str:
    """Create error event with structured error information.

    Error format: {"error": str, "message": str}
    This format ensures compatibility with standard SSE error event consumers.

    Args:
        error: Either a simple error string, or a dict with structured error info.
               Dict format: {"error": "ErrorType", "message": "detailed message"}
        event_id: Optional SSE event ID for reconnection support.

    Returns:
        SSE-formatted error event string with standard error format.
    """
    if isinstance(error, dict):
        # Structured error format - standard format: {error: str, message: str}
        data = {
            "error": error.get("error", "Error"),
            "message": error.get("message", str(error)),
        }
    else:
        # Simple string format - wrap it to standard format
        data = {
            "error": "Error",
            "message": str(error),
        }
    return format_sse_message("error", data, event_id)


def create_messages_event(messages_data: Any, event_type: str = "messages", event_id: str | None = None) -> str:
    """Create messages event (messages, messages/partial, messages/complete, messages/metadata)"""
    # Handle tuple format for token streaming: (message_chunk, metadata)
    if isinstance(messages_data, tuple) and len(messages_data) == 2:
        message_chunk, metadata = messages_data
        # Format as expected by LangGraph SDK client
        data = [message_chunk, metadata]
        return format_sse_message(event_type, data, event_id)
    else:
        # Handle list of messages format
        return format_sse_message(event_type, messages_data, event_id)


@dataclass
class SSEEvent:
    """SSE Event data structure for event storage"""

    id: str
    event: str
    data: dict[str, Any]
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        """Set timestamp to current UTC time if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC)
