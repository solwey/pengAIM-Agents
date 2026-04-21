"""Fetch NetSuite Entity node — executes a SuiteQL query using a token minted by activate_netsuite.

Expects a prior activate_netsuite node to have written {access_token, account_id} to
state.data[token_key] (default "netsuite_activate_result").
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_field
from graphs.workflow_engine.schema import FetchNetsuiteEntityConfig

logger = logging.getLogger(__name__)

# SuiteQL identifiers: letters, digits, underscore. Prevents injection via fields.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# NetSuite internal IDs are positive integers.
_RECORD_ID_RE = re.compile(r"^[0-9]+$")

# Maps REST record-type identifiers (shared with the revops UI and
# api/src/api/net_suite/netsuite.api.ts NetSuiteEntity enum) to the
# (SuiteQL table, optional transaction type discriminator) pair.
# Mirrors the SELECTs in api/src/api/net_suite/*/index.ts.
_SUITEQL_TABLE: dict[str, tuple[str, str | None]] = {
    "account": ("account", None),
    "contact": ("contact", None),
    "customer": ("customer", None),
    "department": ("department", None),
    "employee": ("employee", None),
    "expenseReport": ("expenseReport", None),
    "inventoryItem": ("item", None),
    "job": ("job", None),
    "location": ("location", None),
    "subsidiary": ("subsidiary", None),
    "vendor": ("vendor", None),
    "check": ("transaction", "Check"),
    "creditCardCharge": ("transaction", "CardChrg"),
    "customerPayment": ("transaction", "CustPymt"),
    "deposit": ("transaction", "Deposit"),
    "invoice": ("transaction", "CustInvc"),
    "journalEntry": ("transaction", "Journal"),
    "purchaseOrder": ("transaction", "PurchOrd"),
    "salesOrder": ("transaction", "SalesOrd"),
    "vendorBill": ("transaction", "VendBill"),
    "vendorCredit": ("transaction", "VendCred"),
    "vendorPayment": ("transaction", "VendPymt"),
}


def _validate_identifier(value: str) -> bool:
    return bool(_IDENT_RE.match(value))


class FetchNetsuiteEntityExecutor(NodeExecutor):
    """Run a SuiteQL SELECT against a NetSuite record type and write rows to state."""

    @staticmethod
    def create(config: dict[str, Any]):
        cfg = FetchNetsuiteEntityConfig(**config)

        async def fetch_netsuite_entity_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
            data: dict[str, Any] = state.get("data", {})

            token_payload = resolve_field(data, cfg.token_key)
            if not isinstance(token_payload, dict):
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": f"No NetSuite token found at state.data.{cfg.token_key} — run activate_netsuite first",
                        },
                    }
                }

            access_token = token_payload.get("access_token")
            account_id = token_payload.get("account_id")
            if not access_token or not account_id:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": "Token payload missing access_token or account_id",
                        },
                    }
                }

            mapping = _SUITEQL_TABLE.get(cfg.record_type)
            if mapping is None:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": f"Unsupported record_type '{cfg.record_type}'",
                        },
                    }
                }
            table, tx_type = mapping

            if cfg.fields:
                invalid = [f for f in cfg.fields if not _validate_identifier(f)]
                if invalid:
                    return {
                        "data": {
                            **data,
                            cfg.response_key: {
                                "ok": False,
                                "error": f"Invalid field names: {invalid}",
                            },
                        }
                    }
                select_clause = ", ".join(cfg.fields)
            else:
                select_clause = "*"

            record_id = resolve_field(data, cfg.record_id_key) if cfg.record_id_key else cfg.record_id
            record_id = str(record_id).strip() if record_id else ""
            if record_id and not _RECORD_ID_RE.match(record_id):
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": f"Invalid record_id '{record_id}' (expected numeric)",
                        },
                    }
                }

            where_clauses: list[str] = []
            if tx_type:
                where_clauses.append(f"type = '{tx_type}'")
            if record_id:
                where_clauses.append(f"id = {record_id}")
            where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

            sel = "SELECT"
            fr = "FROM"

            query = f"{sel} {select_clause} {fr} {table}{where_sql}"  # noqa: S608
            suiteql_url = f"https://{account_id}.suitetalk.api.netsuite.com/services/rest/query/v1/suiteql"

            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Prefer": "transient",
            }

            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
                    resp = await client.post(
                        suiteql_url,
                        headers=headers,
                        params={"limit": cfg.limit},
                        json={"q": query},
                    )
                    resp.raise_for_status()
                    body = resp.json()
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "SuiteQL [%s] rejected: status=%s body=%s",
                    cfg.record_type,
                    exc.response.status_code,
                    exc.response.text[:300],
                )
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "status_code": exc.response.status_code,
                            "error": exc.response.text[:500],
                        },
                    }
                }
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {"ok": False, "error": f"SuiteQL request failed: {exc}"},
                    }
                }

            items = body.get("items", []) if isinstance(body, dict) else []
            logger.info(
                "SuiteQL [%s] returned %d rows (total=%s)",
                cfg.record_type,
                len(items),
                body.get("totalResults") if isinstance(body, dict) else "?",
            )

            return {
                "data": {
                    **data,
                    cfg.response_key: {
                        "ok": True,
                        "record_type": cfg.record_type,
                        "count": len(items),
                        "items": items,
                    },
                }
            }

        return fetch_netsuite_entity_node
