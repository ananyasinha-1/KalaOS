"""
KalaOS — Router: Art Analysis (migrated from monolithic main.py)
=================================================================
All heavy AI pipelines are dispatched asynchronously via Celery.
Routes are thin: validate → enqueue → return task_id.
For synchronous (quick) analysis, run_in_executor keeps the event loop free.
"""
from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from pydantic import BaseModel, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings
from app.infrastructure.redis_client import cache_get, cache_set
from app.middleware.security import detect_prompt_injection

logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)
router = APIRouter()


# ── Shared validators ─────────────────────────────────────────────────────

def _validate_text(v: str) -> str:
    if not v.strip():
        raise ValueError("text must not be empty")
    if len(v) > 50_000:
        raise ValueError("text exceeds 50,000 character limit")
    if detect_prompt_injection(v):
        raise ValueError("Request contains disallowed patterns")
    return v


# ── Schemas ───────────────────────────────────────────────────────────────

class AnalyseRequest(BaseModel):
    text: str
    art_domain: str = "general"
    artist_name: Optional[str] = None
    creation_context: Optional[str] = None
    model: Optional[str] = None

    @field_validator("text")
    @classmethod
    def text_ok(cls, v: str) -> str:
        return _validate_text(v)

    @field_validator("art_domain")
    @classmethod
    def domain_ok(cls, v: str) -> str:
        allowed = {"lyrics", "poetry", "music", "story", "book", "general",
                   "painting", "sketch", "photo", "video", "logo"}
        if v not in allowed:
            raise ValueError(f"art_domain must be one of: {allowed}")
        return v


class TaskResponse(BaseModel):
    task_id: str
    status: str = "queued"
    message: str = "Task enqueued. Poll /analysis/result/{task_id} for results."


class AnalyseResponse(BaseModel):
    art_genome: dict
    analysis: dict
    explanation: str
    cached: bool = False


# ── Helper: run sync kalacore in thread pool ──────────────────────────────

async def _run_sync(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(fn, *args))


# ── Routes ────────────────────────────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=AnalyseResponse,
    summary="Full KalaCore + LLM analysis pipeline",
)
@limiter.limit(settings.RATE_LIMIT_AI_INFERENCE)
async def analyze_art(request: Request, body: AnalyseRequest) -> AnalyseResponse:
    """
    Pipeline: Ethics → Pattern Analysis → ArtGenome → LLM explanation.
    Results are cached in Redis for 5 minutes by content hash.
    """
    import hashlib
    cache_key = f"analysis:{hashlib.sha256(body.text.encode()).hexdigest()[:16]}:{body.art_domain}"

    # Cache hit
    cached = await cache_get(cache_key)
    if cached:
        return AnalyseResponse(**cached, cached=True)

    try:
        from kalacore.ethics import check_request
        violations = await _run_sync(check_request, body.text)
        if violations:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=[{"code": v.code, "message": v.message} for v in violations],
            )

        from kalacore.pattern_engine import analyze
        from kalacore.art_genome import build_art_genome
        from services.llm_service import generate_explanation

        analysis = await _run_sync(analyze, body.text)
        genome = await _run_sync(build_art_genome, analysis)
        explanation = await _run_sync(
            generate_explanation, {"art_genome": genome.to_dict(), "analysis": analysis}
        )

        result = {
            "art_genome": genome.to_dict(),
            "analysis": analysis,
            "explanation": explanation,
        }
        await cache_set(cache_key, result)
        return AnalyseResponse(**result, cached=False)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Analysis pipeline failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


@router.post(
    "/deep-analysis/async",
    response_model=TaskResponse,
    summary="Queue a full deep analysis — returns task_id immediately",
)
@limiter.limit(settings.RATE_LIMIT_AI_INFERENCE)
async def deep_analysis_async(request: Request, body: AnalyseRequest) -> TaskResponse:
    """
    Enqueues the full deep-analysis pipeline to a Celery GPU worker.
    Heavy LLM inference is never run synchronously in a route handler.
    """
    try:
        from workers.ai_worker.tasks.text_tasks import deep_analysis_task
        task = deep_analysis_task.apply_async(
            kwargs={
                "text": body.text,
                "art_domain": body.art_domain,
                "artist_name": body.artist_name,
                "creation_context": body.creation_context,
                "model": body.model or settings.OLLAMA_DEFAULT_MODEL,
            },
            priority=5,
        )
        return TaskResponse(task_id=task.id)
    except Exception as exc:
        logger.exception("Failed to enqueue deep analysis")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/result/{task_id}",
    summary="Poll for async task result",
)
async def get_task_result(task_id: str) -> dict:
    """
    Returns the status and result of a queued Celery task.
    Clients should poll with exponential backoff.
    """
    try:
        from celery.result import AsyncResult
        from workers.ai_worker.celery_app import celery_app
        result = AsyncResult(task_id, app=celery_app)

        if result.state == "PENDING":
            return {"task_id": task_id, "status": "pending"}
        if result.state == "STARTED":
            return {"task_id": task_id, "status": "running", "meta": result.info}
        if result.state == "SUCCESS":
            return {"task_id": task_id, "status": "success", "result": result.result}
        if result.state == "FAILURE":
            return {"task_id": task_id, "status": "failed", "error": str(result.info)}
        return {"task_id": task_id, "status": result.state.lower()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/models", summary="List locally available AI models")
async def list_models() -> dict:
    try:
        from services.llm_service import list_available_models
        models = await _run_sync(list_available_models)
        return {"models": models}
    except Exception:
        return {"models": []}
