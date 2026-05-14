"""
KalaOS — New FastAPI Application Factory
=========================================
Replaces the monolithic main.py with a clean app factory pattern.
- Thin entry point (no business logic)
- Domain routers mounted via include_router
- Proper lifespan management
- Middleware stack in correct order
- OpenAPI customization
"""
from __future__ import annotations

import logging
import logging.config
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import settings
from app.infrastructure.redis_client import init_redis, close_redis
from app.middleware.security import SecurityHeadersMiddleware, RequestLoggingMiddleware

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
            },
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json" if settings.is_production else "standard",
            },
        },
        "root": {
            "level": settings.LOG_LEVEL,
            "handlers": ["console"],
        },
    })


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan: startup → yield → shutdown.
    All resource initialization happens here, not at import time.
    """
    configure_logging()
    logger.info("KalaOS API starting up [env=%s]", settings.APP_ENV)

    # Initialize Redis
    try:
        await init_redis()
    except Exception as exc:
        logger.warning("Redis unavailable at startup: %s (continuing)", exc)

    # Initialize DB (only creates tables in dev; use Alembic in prod)
    if settings.is_development:
        try:
            from app.infrastructure.database import init_db
            await init_db()
        except Exception as exc:
            logger.warning("DB init skipped: %s", exc)

    # Initialize OpenTelemetry
    if settings.OTEL_ENABLED:
        try:
            from app.infrastructure.telemetry import init_telemetry
            init_telemetry(app)
        except Exception as exc:
            logger.warning("OTel init failed: %s", exc)

    logger.info("KalaOS API ready.")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info("KalaOS API shutting down...")
    await close_redis()
    try:
        from app.infrastructure.database import close_db
        await close_db()
    except Exception:
        pass
    logger.info("KalaOS API shutdown complete.")


def create_app() -> FastAPI:
    """Application factory — returns a fully configured FastAPI instance."""

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "KalaOS — AI-native creative operating system. "
            "Distributed, modular, GPU-aware, real-time."
        ),
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── Rate Limiter ───────────────────────────────────────────────────────
    limiter = Limiter(key_func=get_remote_address, default_limits=[])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── Middleware stack (order matters: outermost first) ──────────────────
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    # ── Domain Routers ─────────────────────────────────────────────────────
    _register_routers(app)

    return app


def _register_routers(app: FastAPI) -> None:
    """Mount all domain routers. Each router owns its prefix and tags."""
    from app.routers import (
        health,
        auth,
        text_studio,
        music_studio,
        visual_studio,
        video_studio,
        animation_studio,
        analysis,
        agents,
        collab,
    )

    app.include_router(health.router)
    app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
    app.include_router(analysis.router, prefix="/analysis", tags=["Art Analysis"])
    app.include_router(text_studio.router, prefix="/text-studio", tags=["Text Studio"])
    app.include_router(music_studio.router, prefix="/music-studio", tags=["Music Studio"])
    app.include_router(visual_studio.router, prefix="/visual-studio", tags=["Visual Studio"])
    app.include_router(video_studio.router, prefix="/video-studio", tags=["Video Studio"])
    app.include_router(animation_studio.router, prefix="/animation", tags=["Animation Studio"])
    app.include_router(agents.router, prefix="/agents", tags=["AI Agents"])
    app.include_router(collab.router, prefix="/collab", tags=["Collaboration"])


# ── ASGI entry point ───────────────────────────────────────────────────────
app = create_app()
