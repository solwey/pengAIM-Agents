"""Read from Storage node — loads a stored object back into workflow state.

Mirrors `save_to_storage`: GETs an object from an S3-compatible bucket and
deserializes it according to `format`. MinIO is supported by setting
`S3_ENDPOINT_URL` (e.g. http://utils-minio:9000); when unset, the request
goes to AWS S3 in `AWS_REGION` (default us-east-1). gcs / azure_blob /
local remain stubs.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
from collections.abc import Callable, Coroutine
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes._storage_common import build_s3_signed_request
from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_templates
from graphs.workflow_engine.schema import ReadFromStorageConfig

logger = logging.getLogger(__name__)


def _deserialize_payload(payload: bytes, fmt: str) -> Any:
    if fmt == "raw":
        return payload.decode("utf-8", errors="replace")

    if fmt == "json":
        if not payload:
            return None
        return json.loads(payload.decode("utf-8"))

    if fmt == "ndjson":
        text = payload.decode("utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    if fmt == "csv":
        text = payload.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        return list(reader)

    raise ValueError(f"unknown storage format: '{fmt}'")


async def _download_s3(
    cfg: ReadFromStorageConfig,
    config: RunnableConfig,
) -> dict[str, Any]:
    """GET an object from an S3-compatible bucket (real AWS S3 or MinIO)."""
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        return {"ok": False, "error": "missing_aws_credentials"}
    if not cfg.bucket:
        return {"ok": False, "error": "missing_bucket"}
    if not cfg.path:
        return {"ok": False, "error": "missing_path"}

    region = os.environ.get("AWS_REGION", "us-east-1")
    endpoint = os.environ.get("S3_ENDPOINT_URL", "").rstrip("/")
    object_key = cfg.path.lstrip("/")

    try:
        url, headers = build_s3_signed_request(
            method="GET",
            bucket=cfg.bucket,
            object_key=object_key,
            endpoint=endpoint,
            region=region,
            access_key=access_key,
            secret_key=secret_key,
            payload=b"",
            content_type=None,
        )
    except ValueError as exc:
        return {"ok": False, "error": "invalid_endpoint", "detail": str(exc)}

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
            resp = await client.get(url, headers=headers)
    except httpx.RequestError as exc:
        logger.warning("S3 download network error for %s: %s", url, exc)
        return {"ok": False, "error": "network_error", "detail": str(exc)}

    if resp.status_code >= 400:
        logger.warning("S3 download failed (%s): %s", resp.status_code, resp.text[:500])
        return {
            "ok": False,
            "error": "s3_error",
            "status_code": resp.status_code,
            "detail": resp.text[:500],
        }

    try:
        value = _deserialize_payload(resp.content, cfg.format)
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return {"ok": False, "error": "deserialize_failed", "detail": str(exc)}

    return {
        "ok": True,
        "bucket": cfg.bucket,
        "key": object_key,
        "url": url,
        "bytes": len(resp.content),
        "format": cfg.format,
        "value": value,
        "etag": resp.headers.get("etag"),
    }


async def _download_gcs(
    cfg: ReadFromStorageConfig,
    config: RunnableConfig,
) -> dict[str, Any]:
    return {"ok": False, "error": "not_implemented"}


async def _download_azure_blob(
    cfg: ReadFromStorageConfig,
    config: RunnableConfig,
) -> dict[str, Any]:
    return {"ok": False, "error": "not_implemented"}


async def _read_local(
    cfg: ReadFromStorageConfig,
    config: RunnableConfig,
) -> dict[str, Any]:
    return {"ok": False, "error": "not_implemented"}


_BACKENDS: dict[str, Callable[[ReadFromStorageConfig, RunnableConfig], Coroutine[Any, Any, dict[str, Any]]]] = {
    "s3": _download_s3,
    "gcs": _download_gcs,
    "azure_blob": _download_azure_blob,
    "local": _read_local,
}


class ReadFromStorageExecutor(NodeExecutor):
    """Load a stored object and put its parsed value into state['data']."""

    @staticmethod
    def create(config: dict[str, Any]) -> Callable[..., Coroutine[Any, Any, dict[str, Any]]]:
        cfg = ReadFromStorageConfig(**config)

        async def read_from_storage_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
            data: dict[str, Any] = state.get("data", {})

            resolved_cfg = cfg.model_copy(update={"path": resolve_templates(cfg.path, data)})

            backend = _BACKENDS.get(resolved_cfg.storage_type)
            if backend is None:
                result: dict[str, Any] = {
                    "ok": False,
                    "error": "unknown_storage_type",
                    "storage_type": resolved_cfg.storage_type,
                }
            else:
                result = await backend(resolved_cfg, config)

            return {"data": {**data, cfg.response_key: result}}

        return read_from_storage_node
