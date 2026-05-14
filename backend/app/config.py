"""
KalaOS — Centralized Application Configuration
================================================
Single source of truth for all environment variables.
Uses Pydantic Settings v2 for validation + type-safety.
Secrets are never hardcoded — always read from environment.
"""
from __future__ import annotations

import secrets
from functools import lru_cache
from typing import List, Literal, Optional

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ────────────────────────────────────────────────────────────────
    APP_NAME: str = "KalaOS API"
    APP_VERSION: str = "2.0.0"
    APP_ENV: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False

    # ── Security ───────────────────────────────────────────────────────────
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_hex(64))
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ── CORS ───────────────────────────────────────────────────────────────
    CORS_ORIGINS: List[str] = ["*"]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors(cls, v: str | List[str]) -> List[str]:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",")]
        return v

    # ── Database ───────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://kala:kala@localhost:5432/kalaos"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 40
    DATABASE_POOL_TIMEOUT: int = 30

    # ── Redis ──────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CELERY_URL: str = "redis://localhost:6379/1"
    REDIS_CACHE_TTL: int = 300          # 5 minutes default
    REDIS_SESSION_TTL: int = 86400      # 24 hours

    # ── Celery ─────────────────────────────────────────────────────────────
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    CELERY_TASK_SOFT_TIME_LIMIT: int = 300   # 5 min
    CELERY_TASK_TIME_LIMIT: int = 600         # 10 min hard limit

    # ── AI / Ollama ────────────────────────────────────────────────────────
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_DEFAULT_MODEL: str = "llama3"
    OLLAMA_TIMEOUT: int = 120
    OLLAMA_STREAM_TIMEOUT: int = 300

    # ── vLLM ───────────────────────────────────────────────────────────────
    VLLM_HOST: Optional[str] = None
    VLLM_MODEL: Optional[str] = None
    VLLM_API_KEY: Optional[str] = None

    # ── OpenAI (cloud fallback) ────────────────────────────────────────────
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    # ── Anthropic (cloud fallback) ─────────────────────────────────────────
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-3-haiku-20240307"

    # ── Qdrant (Vector DB) ─────────────────────────────────────────────────
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_PREFIX: str = "kalaos"

    # ── MinIO / S3 Storage ─────────────────────────────────────────────────
    S3_ENDPOINT: str = "http://localhost:9000"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_BUCKET: str = "kalaos-assets"
    S3_REGION: str = "us-east-1"
    S3_USE_SSL: bool = False

    # ── Rate Limiting ──────────────────────────────────────────────────────
    RATE_LIMIT_LOGIN: str = "10/minute"
    RATE_LIMIT_REGISTER: str = "5/minute"
    RATE_LIMIT_FORGOT: str = "5/minute"
    RATE_LIMIT_AI_INFERENCE: str = "30/minute"
    RATE_LIMIT_UPLOAD: str = "20/minute"
    RATE_LIMIT_DEFAULT: str = "100/minute"

    # ── Email (SMTP) ───────────────────────────────────────────────────────
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASS: Optional[str] = None
    SMTP_FROM: str = "noreply@kalaos.ai"
    APP_URL: str = "http://localhost:3000"

    # ── Observability ──────────────────────────────────────────────────────
    OTEL_ENABLED: bool = False
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://localhost:4317"
    SENTRY_DSN: Optional[str] = None
    LOG_LEVEL: str = "INFO"

    # ── GPU / Worker ───────────────────────────────────────────────────────
    GPU_WORKER_CONCURRENCY: int = 2
    CPU_WORKER_CONCURRENCY: int = 8
    INFERENCE_BATCH_SIZE: int = 4
    MAX_CONTEXT_TOKENS: int = 8192
    ENABLE_QUANTIZATION: bool = True

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def is_development(self) -> bool:
        return self.APP_ENV == "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton — import and call anywhere."""
    return Settings()


settings = get_settings()
