"""Fetch Relias Entity node — lists records of a given type from the Relias ELAPI.

Relias' ELAPI is a SOAP/ASMX service that takes auth credentials as query
parameters on every call (authorizationID, passcode, organizationId) and
returns XML. Four credentials are stored encrypted in pengAIM-RAG via
/api/v1/relias/connect (provider=relias, names=relias_url | relias_authorization_id |
relias_passcode | relias_organization_id). This executor reveals all four,
calls GET https://{url}.training.reliaslearning.com/webservices/elapi.asmx/{Method},
parses the XML response into a list of dicts, projects the requested fields,
and writes the result to state.

Mirrors the auth/list pattern in api/src/api/relias/relias.api.ts (which uses
fast-xml-parser on the TypeScript side).
"""

from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from aegra_api.settings import settings
from graphs.workflow_engine.nodes.base import NodeExecutor
from graphs.workflow_engine.schema import FetchReliasEntityConfig

logger = logging.getLogger(__name__)

# Must stay aligned with rag/app/routers/relias.py (RELIAS_PROVIDER, *_KEY).
_RELIAS_PROVIDER = "relias"
_RELIAS_URL_NAME = "relias_url"
_RELIAS_AUTHORIZATION_ID_NAME = "relias_authorization_id"
_RELIAS_PASSCODE_NAME = "relias_passcode"
_RELIAS_ORGANIZATION_ID_NAME = "relias_organization_id"

# Maps semantic record types (shared with the revops UI and ReliasRecordType
# in schema.py) to ASMX method names. Mirrors ReliasEntity in
# api/src/api/relias/relias.api.ts.
_RELIAS_METHOD: dict[str, str] = {
    "department": "ViewDepartments",
    "hierarchy": "ViewHierarchy",
    "student": "ViewStudentList",
    "license": "ViewAvailableLicenses",
    "course": "ExportData2",
}

# Extra non-auth query params required by certain ELAPI methods. ExportData2 is
# a multi-table export endpoint that needs ``ExportTable`` to disambiguate the
# table and a startDate/endDate window. Course=5 + a wide date range mirrors
# api/src/api/relias/courses/index.ts so both surfaces return the same rows.
_RELIAS_EXTRA_PARAMS: dict[str, dict[str, str]] = {
    "course": {
        "ExportTable": "5",
        "startDate": "2020-01-01",
        "endDate": "2030-12-31",
    },
}


def _base_url(subdomain: str) -> str:
    return f"https://{subdomain}.training.reliaslearning.com/webservices/elapi.asmx"


async def _get_relias_credentials(
    config: RunnableConfig,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Fetch the four Relias credentials in parallel from pengAIM-RAG.

    Returns (url, authorization_id, passcode, organization_id). Any element
    is None when the credential is not connected or the request fails.
    """
    configurable = config.get("configurable", {})
    auth_token = configurable.get("auth_token", "")
    tenant_uuid = configurable.get("tenant_uuid", "")

    if not auth_token or not tenant_uuid:
        logger.warning(
            "Relias credentials unavailable (auth_token=%s, tenant_uuid=%s)",
            bool(auth_token),
            bool(tenant_uuid),
        )
        return None, None, None, None

    url = f"{settings.graphs.RAG_API_URL}/tenant/{tenant_uuid}/keys/by-name/reveal"
    headers = {"authorization": auth_token, "Accept": "text/plain"}

    async def _fetch(http: httpx.AsyncClient, name: str) -> str | None:
        try:
            resp = await http.get(url, headers=headers, params={"provider": _RELIAS_PROVIDER, "name": name})
            resp.raise_for_status()
            return resp.text
        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning("Failed to reveal %s/%s: %s", _RELIAS_PROVIDER, name, exc)
            return None

    async with httpx.AsyncClient(timeout=httpx.Timeout(10)) as client:
        return await asyncio.gather(
            _fetch(client, _RELIAS_URL_NAME),
            _fetch(client, _RELIAS_AUTHORIZATION_ID_NAME),
            _fetch(client, _RELIAS_PASSCODE_NAME),
            _fetch(client, _RELIAS_ORGANIZATION_ID_NAME),
        )


def _strip_ns(tag: str) -> str:
    """Drop the {namespace} prefix from an XML tag."""
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _element_to_value(el: ET.Element) -> Any:
    """Convert an XML element to a Python value: dict for nested, str for leaf, list for repeated tags."""
    children = list(el)
    if not children:
        return (el.text or "").strip()

    out: dict[str, Any] = {}
    for child in children:
        tag = _strip_ns(child.tag)
        value = _element_to_value(child)
        if tag in out:
            existing = out[tag]
            if isinstance(existing, list):
                existing.append(value)
            else:
                out[tag] = [existing, value]
        else:
            out[tag] = value
    return out


_SCHEMA_TAGS = {"element", "complexType", "sequence", "schema"}


def _find_record_elements(root: ET.Element) -> list[ET.Element]:
    """Walk the XML tree and return the children of the largest homogeneous-children element.

    Relias responses commonly nest records inside ``<DataSet>/<NewDataSet>/<Table>`` or
    directly under ``<ArrayOfThing>/<Thing>``. Both shapes settle on the largest
    same-tag-children element. Ties on count break toward the deeper group. XML Schema
    / diffgram metadata wrappers are skipped so we don't return ``<xs:element>`` lists
    from inline schemas.
    """
    # (count, depth, children) — prefer larger groups, then deeper
    best: tuple[int, int, list[ET.Element]] = (0, -1, [])

    def _visit(el: ET.Element, depth: int) -> None:
        nonlocal best
        children = list(el)
        if children and all(_strip_ns(c.tag) == _strip_ns(children[0].tag) for c in children):
            tag_uri = el.tag.split("}", 1)[0].lstrip("{") if "}" in el.tag else ""
            child_tag = _strip_ns(children[0].tag)
            is_schema_wrapper = "XMLSchema" in tag_uri or child_tag in _SCHEMA_TAGS
            if not is_schema_wrapper and (len(children), depth) > (best[0], best[1]):
                best = (len(children), depth, children)
        for c in children:
            _visit(c, depth + 1)

    _visit(root, 0)
    return best[2]


def _parse_relias_xml(body: str) -> list[dict[str, Any]]:
    """Parse a Relias ELAPI XML response into a list of record dicts."""
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        logger.warning("Failed to parse Relias XML: %s", exc)
        return []

    items: list[dict[str, Any]] = []
    for el in _find_record_elements(root):
        value = _element_to_value(el)
        if isinstance(value, dict) and value:
            items.append(value)
    return items


def _project_fields(items: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    """Return items with only the requested keys; pass through when fields is empty."""
    if not fields:
        return items
    return [{f: item.get(f) for f in fields} for item in items]


class FetchReliasEntityExecutor(NodeExecutor):
    """List records of a Relias entity type and write items to state."""

    @staticmethod
    def create(config: dict[str, Any]):
        cfg = FetchReliasEntityConfig(**config)

        async def fetch_relias_entity_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
            data: dict[str, Any] = state.get("data", {})

            url, authorization_id, passcode, organization_id = await _get_relias_credentials(config)

            missing = [
                name
                for name, value in (
                    (_RELIAS_URL_NAME, url),
                    (_RELIAS_AUTHORIZATION_ID_NAME, authorization_id),
                    (_RELIAS_PASSCODE_NAME, passcode),
                    (_RELIAS_ORGANIZATION_ID_NAME, organization_id),
                )
                if not value
            ]
            if missing:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": f"Missing Relias credentials: {', '.join(missing)}",
                        },
                    }
                }

            method = _RELIAS_METHOD.get(cfg.record_type)
            if method is None:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": f"Unsupported record_type '{cfg.record_type}'",
                        },
                    }
                }

            endpoint = f"{_base_url(url)}/{method}"
            params = {
                "authorizationID": authorization_id,
                "passcode": passcode,
                "organizationId": organization_id,
                **_RELIAS_EXTRA_PARAMS.get(cfg.record_type, {}),
            }

            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(60)) as client:
                    resp = await client.get(endpoint, params=params)
                    resp.raise_for_status()
                    body = resp.text
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "Relias [%s] rejected: status=%s body=%s",
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
                        cfg.response_key: {"ok": False, "error": f"Relias request failed: {exc}"},
                    }
                }

            raw_items = _parse_relias_xml(body)
            items = _project_fields(raw_items, cfg.fields)

            logger.info("Relias [%s] returned %d items", cfg.record_type, len(items))

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

        return fetch_relias_entity_node
