"""Helpers for encoding worker queue payloads."""


def encode_queue_payload(tenant_schema: str, run_id: str) -> str:
    """Encode queue payload with tenant schema and run id."""
    return f"{tenant_schema}|{run_id}"


def decode_queue_payload(payload: str | bytes) -> tuple[str, str] | None:
    """Decode queue payload into (tenant_schema, run_id)."""
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="ignore")
    parts = payload.split("|", 1)
    if len(parts) != 2:
        return None
    tenant_schema, run_id = parts
    if not tenant_schema or not run_id:
        return None
    return tenant_schema, run_id
