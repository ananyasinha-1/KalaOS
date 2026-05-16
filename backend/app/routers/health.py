"""
KalaOS — Router: Health & Observability
========================================
Aggregated health check endpoint for load balancers,
Kubernetes liveness/readiness probes, and monitoring.
"""
from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

router = APIRouter(tags=["Health"])

_START_TIME = time.time()


@router.get("/health", summary="Liveness probe")
async def health_live() -> Dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready", summary="Readiness probe — checks all dependencies")
async def health_ready() -> JSONResponse:
    checks: Dict[str, Any] = {}
    overall = "healthy"

    # Redis
    try:
        from app.infrastructure.redis_client import redis_health
        checks["redis"] = await redis_health()
    except Exception as exc:
        checks["redis"] = {"status": "unhealthy", "error": str(exc)}

    # Database
    try:
        from app.infrastructure.database import db_health
        checks["database"] = await db_health()
    except Exception as exc:
        checks["database"] = {"status": "unhealthy", "error": str(exc)}

    # Ollama / AI
    try:
        import httpx
        from app.config import settings
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{settings.OLLAMA_HOST}/api/tags")
            checks["ollama"] = {
                "status": "healthy" if r.status_code == 200 else "degraded"
            }
    except Exception:
        checks["ollama"] = {"status": "unavailable"}

    if any(v.get("status") == "unhealthy" for v in checks.values()):
        overall = "unhealthy"
    elif any(v.get("status") in ("degraded", "unavailable") for v in checks.values()):
        overall = "degraded"

    payload = {
        "status": overall,
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "checks": checks,
    }
    status_code = status.HTTP_200_OK if overall != "unhealthy" else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(content=payload, status_code=status_code)


@router.get("/", summary="Root — service identity")
async def root() -> Dict[str, str]:
    return {
        "service": "KalaOS API",
        "version": "2.0.0",
        "status": "running",
    }
