"""Shared helpers for the save_to_storage and read_from_storage executors.

Both nodes need to PUT/GET against an S3-compatible store with AWS SigV4
signing — this module owns the URL construction + signing so the per-node
files only deal with serialization and orchestration.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
from urllib.parse import quote

_SERVICE = "s3"


def _sign_v4_headers(
    *,
    method: str,
    host: str,
    canonical_uri: str,
    region: str,
    access_key: str,
    secret_key: str,
    payload_sha256: str,
    content_type: str | None,
) -> dict[str, str]:
    """Build AWS SigV4 headers for a single S3 / S3-compatible request."""
    now = dt.datetime.now(tz=dt.UTC)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")

    headers: dict[str, str] = {
        "host": host,
        "x-amz-content-sha256": payload_sha256,
        "x-amz-date": amz_date,
    }
    if content_type:
        headers["content-type"] = content_type

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


def build_s3_signed_request(
    *,
    method: str,
    bucket: str,
    object_key: str,
    endpoint: str,
    region: str,
    access_key: str,
    secret_key: str,
    payload: bytes,
    content_type: str | None,
) -> tuple[str, dict[str, str]]:
    """Compose a (url, signed_headers) pair for an S3-compatible request.

    `endpoint` is the full URL prefix (e.g. http://utils-minio:9000) for path-
    addressed stores like MinIO. Pass an empty string to use AWS S3 virtual-
    hosted style. Raises ValueError if `endpoint` is set but malformed.
    """
    quoted_key = quote(object_key, safe="/")
    if endpoint:
        scheme, _, host = endpoint.partition("://")
        if not scheme or not host:
            raise ValueError(f"invalid endpoint: '{endpoint}'")
        canonical_uri = f"/{bucket}/{quoted_key}"
        url = f"{scheme}://{host}{canonical_uri}"
    else:
        host = f"{bucket}.s3.{region}.amazonaws.com"
        canonical_uri = f"/{quoted_key}"
        url = f"https://{host}{canonical_uri}"

    payload_sha256 = hashlib.sha256(payload).hexdigest()
    headers = _sign_v4_headers(
        method=method,
        host=host,
        canonical_uri=canonical_uri,
        region=region,
        access_key=access_key,
        secret_key=secret_key,
        payload_sha256=payload_sha256,
        content_type=content_type,
    )
    return url, headers
