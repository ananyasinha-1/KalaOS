"""
KalaOS — Router: Music Studio
==============================
AI-powered music endpoints:
- Beat generation (async → Celery GPU worker)
- Mixing analysis
- Mastering chain
- Chord progression suggestions
- AI composition pipeline
- Prompt-to-music
"""
from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings
from app.infrastructure.redis_client import cache_get, cache_set
from app.middleware.security import detect_prompt_injection

logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)
router = APIRouter()


async def _run_sync(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(fn, *args))


# ── Schemas ───────────────────────────────────────────────────────────────

class BeatRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=500)
    bpm: int = Field(default=90, ge=40, le=300)
    genre: str = Field(default="trap")
    bars: int = Field(default=4, ge=1, le=32)
    swing: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("prompt")
    @classmethod
    def no_injection(cls, v: str) -> str:
        if detect_prompt_injection(v):
            raise ValueError("Disallowed content in prompt")
        return v


class MixingRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=10_000)
    art_domain: str = "music"


class MasteringRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=10_000)
    target_lufs: float = Field(default=-14.0, ge=-24.0, le=-6.0)
    target_format: str = Field(default="streaming")


class CompositionRequest(BaseModel):
    prompt: str = Field(..., min_length=5, max_length=1000)
    style: Optional[str] = None
    key: Optional[str] = None
    tempo: Optional[int] = Field(default=None, ge=40, le=300)
    duration_bars: int = Field(default=8, ge=4, le=64)

    @field_validator("prompt")
    @classmethod
    def no_injection(cls, v: str) -> str:
        if detect_prompt_injection(v):
            raise ValueError("Disallowed content in prompt")
        return v


class ChordRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=500)
    key: str = Field(default="C")
    scale: str = Field(default="major")
    bars: int = Field(default=4, ge=1, le=16)


class TaskResponse(BaseModel):
    task_id: str
    status: str = "queued"
    message: str = "Task enqueued. Poll /analysis/result/{task_id} for results."


# ── Routes ────────────────────────────────────────────────────────────────

@router.post("/ai-beat", summary="AI Beat Generation — prompt → BPM + drum pattern + melody")
@limiter.limit(settings.RATE_LIMIT_AI_INFERENCE)
async def generate_ai_beat(request: Request, body: BeatRequest) -> dict:
    """
    Synchronous fast path for beat generation using existing kalaproducer.
    Heavy ML inference upgrades (audio generation) routed to async worker.
    """
    import hashlib
    cache_key = f"beat:{hashlib.sha256(f'{body.prompt}{body.bpm}{body.genre}'.encode()).hexdigest()[:16]}"
    cached = await cache_get(cache_key)
    if cached:
        return {**cached, "cached": True}

    try:
        from kalacore.kalaproducer import generate_ai_beat
        result = await _run_sync(generate_ai_beat, body.prompt, body.bpm, body.genre)
        await cache_set(cache_key, result, ttl=600)
        return {**result, "cached": False}
    except Exception as exc:
        logger.exception("AI beat generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/compose/async", response_model=TaskResponse, summary="Queue AI composition task")
@limiter.limit(settings.RATE_LIMIT_AI_INFERENCE)
async def compose_async(request: Request, body: CompositionRequest) -> TaskResponse:
    """
    Enqueue a full AI composition pipeline to a GPU worker.
    Includes melody generation, chord progressions, arrangement.
    """
    try:
        from workers.ai_worker.tasks.music_tasks import compose_task
        task = compose_task.apply_async(
            kwargs=body.model_dump(),
            priority=7,
            queue="gpu",
        )
        return TaskResponse(task_id=task.id)
    except Exception as exc:
        logger.exception("Failed to enqueue composition task")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/mix", summary="AI Mixing analysis")
@limiter.limit(settings.RATE_LIMIT_AI_INFERENCE)
async def analyze_mixing(request: Request, body: MixingRequest) -> dict:
    try:
        from kalacore.kalaproducer import produce
        from kalacore.pattern_engine import analyze
        from kalacore.art_genome import build_art_genome

        analysis = await _run_sync(analyze, body.text)
        genome = await _run_sync(build_art_genome, analysis)
        result = await _run_sync(produce, body.text, analysis, genome.to_dict())
        return {"mixing": result, "art_genome": genome.to_dict()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/master", summary="AI Mastering chain analysis")
@limiter.limit(settings.RATE_LIMIT_AI_INFERENCE)
async def analyze_mastering(request: Request, body: MasteringRequest) -> dict:
    """
    Analyses the text/production notes and returns intelligent
    mastering chain recommendations (EQ curves, compression settings,
    limiter thresholds, LUFS targets).
    """
    try:
        from kalacore.kalaproducer import produce
        from kalacore.pattern_engine import analyze
        from kalacore.art_genome import build_art_genome

        analysis = await _run_sync(analyze, body.text)
        genome = await _run_sync(build_art_genome, analysis)
        result = await _run_sync(produce, body.text, analysis, genome.to_dict())

        mastering_hints = {
            "target_lufs": body.target_lufs,
            "target_format": body.target_format,
            "recommended_ceiling": -1.0 if body.target_format == "streaming" else -0.3,
            "production_data": result,
        }
        return {"mastering": mastering_hints, "art_genome": genome.to_dict()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/chords", summary="AI Chord Progression Suggestions")
@limiter.limit(settings.RATE_LIMIT_AI_INFERENCE)
async def chord_suggestions(request: Request, body: ChordRequest) -> dict:
    """
    Returns chord progression ideas based on emotional prompt.
    Cached aggressively as chord suggestions are deterministic.
    """
    import hashlib
    cache_key = f"chords:{hashlib.sha256(f'{body.prompt}{body.key}{body.scale}'.encode()).hexdigest()[:16]}"
    cached = await cache_get(cache_key)
    if cached:
        return {**cached, "cached": True}

    try:
        from kalacore.kalacomposer import compose
        from kalacore.pattern_engine import analyze
        from kalacore.art_genome import build_art_genome

        analysis = await _run_sync(analyze, body.prompt)
        genome = await _run_sync(build_art_genome, analysis)
        result = await _run_sync(compose, body.prompt, analysis, genome.to_dict())

        chord_data = {
            "key": body.key,
            "scale": body.scale,
            "bars": body.bars,
            "progressions": result.get("chords", {}),
            "composition": result,
        }
        await cache_set(cache_key, chord_data, ttl=3600)
        return {**chord_data, "cached": False}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
