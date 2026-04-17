import re
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BeforeValidator, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from aegra_api import __version__

load_dotenv()


def parse_lower(v: str) -> str:
    """Converts to lowercase and strips whitespace."""
    return v.strip().lower() if isinstance(v, str) else v


def parse_upper(v: str) -> str:
    """Converts to uppercase and strips whitespace."""
    return v.strip().upper() if isinstance(v, str) else v


def parse_bool(v: bool | str | None) -> bool | str | None:
    """Parse common env-style boolean representations."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, str):
        value = v.strip().lower()
        if value in {"", "0", "false", "f", "no", "n", "off"}:
            return False
        if value in {"1", "true", "t", "yes", "y", "on"}:
            return True
    return v


# Custom types for automatic formatting
LowerStr = Annotated[str, BeforeValidator(parse_lower)]
UpperStr = Annotated[str, BeforeValidator(parse_upper)]
BoolStr = Annotated[bool, BeforeValidator(parse_bool)]


class EnvBase(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
    )


class AppSettings(EnvBase):
    """General application settings."""

    PROJECT_NAME: str = "Agents"
    VERSION: str = __version__

    # Server config
    HOST: str = "0.0.0.0"  # nosec B104
    PORT: int = 2026
    SERVER_URL: str | None = None

    @model_validator(mode="after")
    def _validate_keepalive_interval(self) -> "AppSettings":
        if self.KEEPALIVE_INTERVAL_SECS <= 0:
            raise ValueError(f"KEEPALIVE_INTERVAL_SECS must be greater than 0, got {self.KEEPALIVE_INTERVAL_SECS}")
        return self

    @model_validator(mode="after")
    def _derive_server_url(self) -> "AppSettings":
        """Derive SERVER_URL from HOST/PORT when not explicitly set."""
        if self.SERVER_URL is None:
            host = "localhost" if self.HOST in ("0.0.0.0", "127.0.0.1") else self.HOST  # nosec B104
            object.__setattr__(self, "SERVER_URL", f"http://{host}:{self.PORT}")
        return self

    # App logic
    AEGRA_CONFIG: str = "aegra.json"  # Default config file path
    KEEPALIVE_INTERVAL_SECS: float = 5  # Heartbeat interval for join/wait endpoints
    AUTH_TYPE: LowerStr = "noop"
    ENV_MODE: UpperStr = "LOCAL"
    DEBUG: BoolStr = False

    # Logging
    LOG_LEVEL: UpperStr = "INFO"
    LOG_VERBOSITY: LowerStr = "verbose"

    # Auth
    JWT_PRIVATE_KEY: str
    JWT_PUBLIC_KEY: str
    JWT_ALG: str = "RS256"


class RedisSettings(EnvBase):
    """Redis connection settings."""

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 2
    REDIS_USER: str = ""
    REDIS_PASSWORD: str = ""
    REDIS_PROTOCOL: LowerStr = "redis"  # redis | rediss
    REDIS_USE_CREDENTIALS: BoolStr = False

    REDIS_BROKER_ENABLED: BoolStr = True
    REDIS_CHANNEL_PREFIX: str = "aegra:run:"
    REDIS_MAX_CONNECTIONS: int = 250

    @computed_field
    @property
    def redis_url(self) -> str:  # noqa: N802
        if self.REDIS_USE_CREDENTIALS:
            auth_part = f"{self.REDIS_USER}:{self.REDIS_PASSWORD}@"
        else:
            auth_part = ""

        url = f"{self.REDIS_PROTOCOL}://{auth_part}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

        if self.REDIS_PROTOCOL == "rediss":
            url += "?ssl_cert_reqs=CERT_OPTIONAL"

        return url


class DatabaseSettings(EnvBase):
    """Database connection settings.

    Supports two configuration modes:
    1. DATABASE_URL (standard for containerized deployments) — parsed into individual fields
    2. Individual POSTGRES_* vars — used when DATABASE_URL is not set
    """

    DATABASE_URL: str | None = None

    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: str = "5432"
    POSTGRES_DB: str = "aegra"
    DB_ECHO_LOG: BoolStr = False

    @staticmethod
    def _normalize_scheme(url: str, target_scheme: str) -> str:
        """Replace the URL scheme/driver prefix with the target scheme."""
        return re.sub(r"^postgres(?:ql)?(\+\w+)?://", f"{target_scheme}://", url)

    @computed_field
    @property
    def database_url(self) -> str:
        """Async URL for SQLAlchemy (asyncpg)."""
        if self.DATABASE_URL:
            return self._normalize_scheme(self.DATABASE_URL, "postgresql+asyncpg")
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @computed_field
    @property
    def database_url_sync(self) -> str:
        """Sync URL for LangGraph/Psycopg (postgresql://)."""
        if self.DATABASE_URL:
            return self._normalize_scheme(self.DATABASE_URL, "postgresql")
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


class PoolSettings(EnvBase):
    """Connection pool settings for SQLAlchemy and LangGraph."""

    SQLALCHEMY_POOL_SIZE: int = 10
    SQLALCHEMY_MAX_OVERFLOW: int = 20

    LANGGRAPH_MIN_POOL_SIZE: int = 5
    LANGGRAPH_MAX_POOL_SIZE: int = 20


class ObservabilitySettings(EnvBase):
    """
    Unified settings for OpenTelemetry and Vendor targets.
    Supports Fan-out configuration via OTEL_TARGETS.
    """

    # General OTEL Config
    OTEL_SERVICE_NAME: str = "aegra-backend"
    OTEL_TARGETS: str = ""  # Comma-separated: "LANGFUSE,PHOENIX"
    OTEL_CONSOLE_EXPORT: BoolStr = False  # For local debugging

    # --- Generic OTLP Target (Default/Custom) ---
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = None
    OTEL_EXPORTER_OTLP_HEADERS: str | None = None

    # --- Langfuse Specifics ---
    LANGFUSE_BASE_URL: str = "http://localhost:3000"
    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_SECRET_KEY: str | None = None

    # --- Sentry Specifics ---
    SENTRY_DSN: str | None = None
    SENTRY_ENVIRONMENT: str | None = None

    # --- Phoenix Specifics ---
    PHOENIX_COLLECTOR_ENDPOINT: str = "http://127.0.0.1:6006/v1/traces"
    PHOENIX_API_KEY: str | None = None


class WorkerSettings(EnvBase):
    """Worker configuration for background graph execution.

    When REDIS_BROKER_ENABLED is True, runs are dispatched to worker
    coroutines via a Redis List job queue instead of local asyncio tasks.
    Each worker loop dequeues run_ids from Redis and spawns up to
    N_JOBS_PER_WORKER concurrent asyncio tasks for graph execution.
    """

    WORKER_COUNT: int = 3
    N_JOBS_PER_WORKER: int = 10
    WORKER_QUEUE_KEY: str = "aegra:jobs"
    WORKER_DRAIN_TIMEOUT: float = 30.0
    BG_JOB_TIMEOUT_SECS: int = 3600
    BG_JOB_MAX_RETRIES: int = 3

    # Lease-based crash recovery.
    # The lease must be long enough that a healthy worker NEVER loses it.
    # Safety margin = LEASE / HEARTBEAT = 30/10 = 3 missed heartbeats
    # before expiry (industry standard — matches Kubernetes liveness probes).
    # Worst-case recovery: ~30s lease expiry + ~20s reaper interval = ~50s.
    LEASE_DURATION_SECONDS: int = 30
    HEARTBEAT_INTERVAL_SECONDS: int = 10
    REAPER_INTERVAL_SECONDS: int = 15
    STUCK_PENDING_THRESHOLD_SECONDS: int = 120
    POSTGRES_POLL_INTERVAL_SECONDS: int = 5

    @model_validator(mode="after")
    def _validate_lease_timing(self) -> "WorkerSettings":
        if self.LEASE_DURATION_SECONDS <= 2 * self.HEARTBEAT_INTERVAL_SECONDS:
            raise ValueError(
                f"LEASE_DURATION_SECONDS ({self.LEASE_DURATION_SECONDS}) must be "
                f"greater than 2 * HEARTBEAT_INTERVAL_SECONDS ({self.HEARTBEAT_INTERVAL_SECONDS}). "
                f"A worker must survive at least 2 missed heartbeats before its lease expires."
            )
        return self


class GraphsSettings(EnvBase):
    """Graph runtime settings shared by graph modules."""

    AZURE_OPENAI_ENDPOINT: str | None = None
    RAG_API_URL: str
    SMTP_HOST: str | None = None
    SMTP_PORT: int | None = None
    SMTP_USER: str | None = None
    SMTP_PASSWORD: str | None = None
    SMTP_FROM: str | None = None
    REVY_API_URL: str | None = None
    WORKFLOW_LLM_MODEL: str = "openai:gpt-4o-mini"
    TENANT_UUID: str | None = None


class Settings:
    def __init__(self) -> None:
        self.app = AppSettings()
        self.db = DatabaseSettings()
        self.pool = PoolSettings()
        self.observability = ObservabilitySettings()
        self.redis = RedisSettings()
        self.worker = WorkerSettings()
        self.graphs = GraphsSettings()


settings = Settings()
