"""LangGraph-specific serialization"""

import ast
import json
import re
from typing import Any

import structlog

from .base import SerializationError, Serializer
from .general import GeneralSerializer

logger = structlog.getLogger(__name__)


class LangGraphSerializer(Serializer):
    """Handles serialization of LangGraph objects (tasks, interrupts, snapshots)"""

    def __init__(self):
        self.general_serializer = GeneralSerializer()

    def serialize(self, obj: Any) -> Any:
        """Main serialization entry point"""
        return json.loads(json.dumps(obj, default=self.general_serializer.serialize))

    def _error_to_string(self, err: Any) -> str | None:
        """Convert an arbitrary error to a stable string representation."""
        if err is None:
            return None
        if isinstance(err, str):
            return err
        try:
            return str(err)
        except Exception:
            # Last resort
            return repr(err)

    def _extract_literal_blob(self, text: str) -> str | None:
        if not text:
            return None

        start = -1
        opener = None
        for i, ch in enumerate(text):
            if ch in ("{", "["):
                start = i
                opener = ch
                break
        if start == -1 or opener is None:
            return None

        closer = "}" if opener == "{" else "]"
        depth = 0
        in_str: str | None = None
        escape = False

        for j in range(start, len(text)):
            ch = text[j]

            if in_str is not None:
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == in_str:
                    in_str = None
                continue

            if ch in ("\"", "'"):
                in_str = ch
                continue

            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start : j + 1]

        return None

    def _normalize_error(self, err: Any) -> dict[str, Any] | None:
        if err is None:
            return None

        if isinstance(err, dict):
            out: dict[str, Any] = {"type": err.get("type", "Error"), **err}
            if "message" not in out:
                out["message"] = out.get("detail") or out.get("error") or ""
            return out

        err_str = self._error_to_string(err)
        if not err_str:
            return None

        payload: dict[str, Any] = {
            "type": err.__class__.__name__ if not isinstance(err, str) else "Error",
            "message": err_str,
        }

        m = re.match(r"^(?P<etype>[A-Za-z_][A-Za-z0-9_]*)\((?P<body>.*)\)\s*$", err_str)
        if m:
            payload["type"] = m.group("etype")
            inner = m.group("body").strip()
            if (inner.startswith('"') and inner.endswith('"')) or (
                inner.startswith("'") and inner.endswith("'")
            ):
                inner = inner[1:-1]
            payload["message"] = inner

        blob = self._extract_literal_blob(payload["message"])
        if blob:
            try:
                raw = ast.literal_eval(blob)
            except Exception:
                try:
                    raw = json.loads(blob)
                except Exception:
                    raw = None

            if raw is not None:
                payload["raw"] = raw

                if isinstance(raw, dict):
                    if isinstance(raw.get("error"), dict):
                        e = raw.get("error", {})
                        msg = e.get("message")
                        if isinstance(msg, str) and msg:
                            payload["message"] = msg
                        code = e.get("code")
                        if code is not None:
                            payload["code"] = code
                        etype = e.get("type")
                        if etype is not None:
                            payload["provider_type"] = etype
                    else:
                        msg = raw.get("message")
                        if isinstance(msg, str) and msg:
                            payload["message"] = msg
                        code = raw.get("code")
                        if code is not None:
                            payload["code"] = code

        return payload

    def serialize_task(self, task: Any) -> dict[str, Any]:
        """Serialize a LangGraph task to ThreadTask format"""
        try:
            if hasattr(task, "id") and hasattr(task, "name"):
                # Proper task object
                task_dict = {
                    "id": getattr(task, "id", ""),
                    "name": getattr(task, "name", ""),
                    "error": self._normalize_error(getattr(task, "error", None)),
                    "interrupts": [],
                    "checkpoint": None,
                    "state": getattr(task, "state", None),
                    "result": getattr(task, "result", None),
                }

                # Handle task interrupts
                if hasattr(task, "interrupts") and task.interrupts:
                    task_dict["interrupts"] = self.serialize(task.interrupts)

                return task_dict
            else:
                # Raw task data - serialize as-is but safely
                serialized_task = self.serialize(task)
                if isinstance(serialized_task, dict):
                    err = serialized_task.get("error")
                    if err:
                        serialized_task["error"] = self._normalize_error(err)
                    return serialized_task
                else:
                    raise SerializationError(
                        f"Task serialization resulted in non-dict: {type(serialized_task)}",
                        task.__class__.__name__,
                    )
        except Exception as e:
            if isinstance(e, SerializationError):
                raise
            raise SerializationError(
                f"Failed to serialize task: {str(e)}", task.__class__.__name__, e
            ) from e

    def serialize_interrupt(self, interrupt: Any) -> dict[str, Any]:
        """Serialize a LangGraph interrupt"""
        try:
            return self.serialize(interrupt)
        except Exception as e:
            raise SerializationError(
                f"Failed to serialize interrupt: {str(e)}",
                interrupt.__class__.__name__,
                e,
            ) from e

    def extract_tasks_from_snapshot(self, snapshot: Any) -> list[dict[str, Any]]:
        """Extract and serialize tasks from a snapshot"""
        tasks = []

        if not (hasattr(snapshot, "tasks") and snapshot.tasks):
            return tasks

        for task in snapshot.tasks:
            try:
                serialized_task = self.serialize_task(task)
                tasks.append(serialized_task)
            except SerializationError as e:
                logger.warning(
                    f"Task serialization failed, skipping task: {e} "
                    f"(task_type={type(task).__name__}, task_id={getattr(task, 'id', 'unknown')})"
                )
                continue

        return tasks

    def extract_interrupts_from_snapshot(self, snapshot: Any) -> list[dict[str, Any]]:
        """Extract and serialize interrupts from a snapshot"""
        interrupts = []
        if hasattr(snapshot, "interrupts") and snapshot.interrupts:
            try:
                interrupts = self.serialize(snapshot.interrupts)
                if interrupts:
                    return interrupts
            except Exception as e:
                logger.warning(
                    f"Snapshot interrupt serialization failed: {e} "
                    f"(snapshot_type={type(snapshot).__name__})"
                )
        return interrupts if isinstance(interrupts, list) else []
