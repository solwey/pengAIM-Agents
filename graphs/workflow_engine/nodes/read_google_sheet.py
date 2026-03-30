"""Read Google Sheet node — reads data from a Google Spreadsheet."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes.base import NodeExecutor
from graphs.workflow_engine.schema import ReadGoogleSheetConfig

logger = logging.getLogger(__name__)


class ReadGoogleSheetExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = ReadGoogleSheetConfig(**config)

        async def read_google_sheet_node(
            state: dict, config: RunnableConfig
        ) -> dict:
            data = state.get("data", {})

            sa_key_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_KEY", "")
            if not sa_key_json:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": "GOOGLE_SERVICE_ACCOUNT_KEY not configured",
                        },
                    }
                }

            try:
                from google.oauth2.service_account import Credentials
                from googleapiclient.discovery import build

                sa_info = json.loads(sa_key_json)
                creds = Credentials.from_service_account_info(
                    sa_info,
                    scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
                )
                service = build("sheets", "v4", credentials=creds)

                range_str = (
                    f"{cfg.sheet_name}!{cfg.range}"
                    if cfg.sheet_name
                    else cfg.range
                )

                sheet = service.spreadsheets()
                result_data = (
                    sheet.values()
                    .get(spreadsheetId=cfg.spreadsheet_id, range=range_str)
                    .execute()
                )

                values = result_data.get("values", [])
                if not values:
                    return {
                        "data": {
                            **data,
                            cfg.response_key: {
                                "ok": True,
                                "rows": [],
                                "headers": [],
                                "row_count": 0,
                            },
                        }
                    }

                headers = values[0]
                rows = []
                for row in values[1:]:
                    row_dict: dict[str, str] = {}
                    for i, header in enumerate(headers):
                        row_dict[header] = row[i] if i < len(row) else ""
                    rows.append(row_dict)

                result: dict[str, Any] = {
                    "ok": True,
                    "rows": rows,
                    "headers": headers,
                    "row_count": len(rows),
                }
                logger.info(
                    "Google Sheet read: %d rows from %s",
                    len(rows),
                    cfg.spreadsheet_id,
                )

            except json.JSONDecodeError:
                result = {
                    "ok": False,
                    "error": "GOOGLE_SERVICE_ACCOUNT_KEY is not valid JSON",
                }
            except Exception as exc:
                result = {"ok": False, "error": f"Google Sheets error: {exc}"}

            return {"data": {**data, cfg.response_key: result}}

        return read_google_sheet_node
