"""Save to Storage node — persists a workflow variable to S3/GCS/Azure/local.

The `s3` backend covers both real AWS S3 and S3-compatible stores (MinIO,
Cloudflare R2, etc.). For MinIO, set `S3_ENDPOINT_URL=http://utils-minio:9000`;
when unset, the request is sent to AWS S3 in `AWS_REGION` (default us-east-1).
gcs, azure_blob, and local remain stubs.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import hmac
import io
import json
import logging
import os
from collections.abc import Callable, Coroutine
from typing import Any
from urllib.parse import quote

import httpx
from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_field, resolve_templates
from graphs.workflow_engine.schema import SaveToStorageConfig

logger = logging.getLogger(__name__)

_SERVICE = "s3"

_CONTENT_TYPES: dict[str, str] = {
    "json": "application/json",
    "ndjson": "application/x-ndjson",
    "csv": "text/csv",
    "raw": "application/octet-stream",
}


def _serialize_payload(value: Any, fmt: str) -> bytes:
    if fmt == "raw":
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return value.encode("utf-8")
        return json.dumps(value, default=str).encode("utf-8")

    if fmt == "json":
        return json.dumps(value, default=str, indent=2).encode("utf-8")

    if fmt == "ndjson":
        items = value if isinstance(value, list) else [value]
        body = "\n".join(json.dumps(item, default=str) for item in items)
        return (body + "\n").encode("utf-8") if body else b""

    if fmt == "csv":
        rows = value if isinstance(value, list) else [value]
        if not rows:
            return b""
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("csv format requires a list of dicts")
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        return buf.getvalue().encode("utf-8")

    raise ValueError(f"unknown storage format: '{fmt}'")


def _sign_v4(
    *,
    method: str,
    host: str,
    canonical_uri: str,
    region: str,
    access_key: str,
    secret_key: str,
    payload_sha256: str,
    content_type: str,
) -> dict[str, str]:
    """Build AWS SigV4 headers for a single PUT to S3 / S3-compatible storage."""
    now = dt.datetime.now(tz=dt.UTC)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")

    headers: dict[str, str] = {
        "content-type": content_type,
        "host": host,
        "x-amz-content-sha256": payload_sha256,
        "x-amz-date": amz_date,
    }

    sorted_headers = sorted(headers.items())
    canonical_headers = "".join(f"{k}:{v.strip()}\n" for k, v in sorted_headers)
    signed_headers = ";".join(k for k, _ in sorted_headers)

    canonical_request = f"{method}\n{canonical_uri}\n\n{canonical_headers}\n{signed_headers}\n{payload_sha256}"

    credential_scope = f"{date_stamp}/{region}/{_SERVICE}/aws4_request"
    string_to_sign = (
        f"AWS4-HMAC-SHA256\n{amz_date}\n{credential_scope}\n"
        f"{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
    )

    def _hmac(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    k_date = _hmac(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    k_region = _hmac(k_date, region)
    k_service = _hmac(k_region, _SERVICE)
    k_signing = _hmac(k_service, "aws4_request")
    signature = hmac.new(k_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    headers["authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    return headers


async def _upload_s3(
    cfg: SaveToStorageConfig,
    payload: bytes,
    config: RunnableConfig,
) -> dict[str, Any]:
    """PUT bytes to an S3-compatible bucket (real AWS S3 or MinIO)."""
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
    quoted_key = quote(object_key, safe="/")

    if endpoint:
        # Path-addressed (MinIO default): <endpoint>/<bucket>/<key>
        scheme, _, host = endpoint.partition("://")
        if not scheme or not host:
            return {"ok": False, "error": "invalid_endpoint", "detail": endpoint}
        canonical_uri = f"/{cfg.bucket}/{quoted_key}"
        url = f"{scheme}://{host}{canonical_uri}"
    else:
        # AWS S3 virtual-hosted style
        host = f"{cfg.bucket}.s3.{region}.amazonaws.com"
        canonical_uri = f"/{quoted_key}"
        url = f"https://{host}{canonical_uri}"

    content_type = _CONTENT_TYPES.get(cfg.format, "application/octet-stream")
    payload_sha256 = hashlib.sha256(payload).hexdigest()

    headers = _sign_v4(
        method="PUT",
        host=host,
        canonical_uri=canonical_uri,
        region=region,
        access_key=access_key,
        secret_key=secret_key,
        payload_sha256=payload_sha256,
        content_type=content_type,
    )

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
            resp = await client.put(url, content=payload, headers=headers)
    except httpx.RequestError as exc:
        logger.warning("S3 upload network error for %s: %s", url, exc)
        return {"ok": False, "error": "network_error", "detail": str(exc)}

    if resp.status_code >= 400:
        logger.warning("S3 upload failed (%s): %s", resp.status_code, resp.text[:500])
        return {
            "ok": False,
            "error": "s3_error",
            "status_code": resp.status_code,
            "detail": resp.text[:500],
        }

    return {
        "ok": True,
        "bucket": cfg.bucket,
        "key": object_key,
        "url": url,
        "bytes": len(payload),
        "format": cfg.format,
        "etag": resp.headers.get("etag"),
    }


async def _upload_gcs(
    cfg: SaveToStorageConfig,
    payload: bytes,
    config: RunnableConfig,
) -> dict[str, Any]:
    return {"ok": False, "error": "not_implemented"}


async def _upload_azure_blob(
    cfg: SaveToStorageConfig,
    payload: bytes,
    config: RunnableConfig,
) -> dict[str, Any]:
    return {"ok": False, "error": "not_implemented"}


async def _write_local(
    cfg: SaveToStorageConfig,
    payload: bytes,
    config: RunnableConfig,
) -> dict[str, Any]:
    return {"ok": False, "error": "not_implemented"}


_BACKENDS: dict[str, Callable[[SaveToStorageConfig, bytes, RunnableConfig], Coroutine[Any, Any, dict[str, Any]]]] = {
    "s3": _upload_s3,
    "gcs": _upload_gcs,
    "azure_blob": _upload_azure_blob,
    "local": _write_local,
}


class SaveToStorageExecutor(NodeExecutor):
    """Persist a workflow variable to a configured storage backend."""

    @staticmethod
    def create(config: dict[str, Any]) -> Callable[..., Coroutine[Any, Any, dict[str, Any]]]:
        cfg = SaveToStorageConfig(**config)

        async def save_to_storage_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
            data: dict[str, Any] = state.get("data", {})

            if not cfg.data_key:
                return {"data": {**data, cfg.response_key: {"ok": False, "error": "missing_data_key"}}}

            value = resolve_field(data, cfg.data_key)
            if value is None:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {"ok": False, "error": "data_key_not_found", "data_key": cfg.data_key},
                    }
                }

            try:
                payload = _serialize_payload(value, cfg.format)
            except (ValueError, TypeError) as exc:
                return {
                    "data": {**data, cfg.response_key: {"ok": False, "error": "serialize_failed", "detail": str(exc)}}
                }

            resolved_cfg = cfg.model_copy(update={"path": resolve_templates(cfg.path, data)})

            backend = _BACKENDS.get(resolved_cfg.storage_type)
            if backend is None:
                result: dict[str, Any] = {
                    "ok": False,
                    "error": "unknown_storage_type",
                    "storage_type": resolved_cfg.storage_type,
                }
            else:
                result = await backend(resolved_cfg, payload, config)

            return {"data": {**data, cfg.response_key: result}}

        return save_to_storage_node
