"""Pydantic models for Control Plane API"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class WorkerStatus(BaseModel):
    id: str
    status: str  # online / offline
    started_at: datetime
    last_heartbeat: datetime
    uptime_seconds: float
    active_run_count: int
    metadata: dict | None = None


class ActiveRun(BaseModel):
    run_id: str
    thread_id: str
    assistant_id: str | None = None
    assistant_name: str | None = None
    status: str
    current_step: str | None = None
    duration_seconds: float
    created_at: datetime
    user_id: str


class RunHistoryEntry(BaseModel):
    run_id: str
    thread_id: str
    assistant_id: str | None = None
    assistant_name: str | None = None
    status: str
    error_message: str | None = None
    duration_ms: int | None = None
    created_at: datetime
    updated_at: datetime


class RunStatusTransition(BaseModel):
    from_status: str | None = None
    to_status: str
    error_message: str | None = None
    traceback: str | None = None
    created_at: datetime


class DashboardStats(BaseModel):
    total_runs_24h: int
    completed_24h: int
    failed_24h: int
    avg_duration_ms: float | None = None


class ControlPlaneOverview(BaseModel):
    workers: list[WorkerStatus]
    active_runs: list[ActiveRun]
    stats: DashboardStats


class RunHistoryPage(BaseModel):
    runs: list[RunHistoryEntry]
    total: int
    limit: int
    offset: int
