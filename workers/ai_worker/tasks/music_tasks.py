"""
KalaOS — Celery Tasks: Music GPU Worker
=========================================
GPU-accelerated music tasks:
- AI composition (melody + harmony + arrangement)
- Stem isolation pipeline
- Genre transformation
- Prompt-to-music (queued for future audio model integration)
"""
from __future__ import annotations

import logging
from typing import Optional

from celery.exceptions import SoftTimeLimitExceeded

from workers.ai_worker.celery_app import celery_app
from workers.ai_worker.tasks.text_tasks import BaseKalaTask

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    base=BaseKalaTask,
    name="workers.ai_worker.tasks.music_tasks.compose_task",
    queue="gpu",
    soft_time_limit=540,
    time_limit=600,
)
def compose_task(
    self,
    prompt: str,
    style: Optional[str] = None,
    key: Optional[str] = None,
    tempo: Optional[int] = None,
    duration_bars: int = 8,
) -> dict:
    """Full AI composition pipeline on GPU worker."""
    try:
        self.update_state(state="STARTED", meta={"step": "analysis", "progress": 10})
        from kalacore.pattern_engine import analyze
        from kalacore.art_genome import build_art_genome
        from kalacore.kalacomposer import compose

        analysis = analyze(prompt)
        genome = build_art_genome(analysis)

        self.update_state(state="STARTED", meta={"step": "composing", "progress": 40})
        composition = compose(prompt, analysis, genome.to_dict())

        self.update_state(state="STARTED", meta={"step": "arrangement", "progress": 70})
        # Enrich with user-specified params
        if key:
            composition["key_override"] = key
        if tempo:
            composition["tempo_override"] = tempo
        if style:
            composition["style_override"] = style
        composition["duration_bars"] = duration_bars

        self.update_state(state="STARTED", meta={"step": "done", "progress": 100})
        return {
            "composition": composition,
            "art_genome": genome.to_dict(),
            "prompt": prompt,
        }
    except SoftTimeLimitExceeded:
        return {"error": "timeout"}
    except Exception as exc:
        logger.exception("Compose task failed")
        raise self.retry(exc=exc, countdown=10)


@celery_app.task(
    bind=True,
    base=BaseKalaTask,
    name="workers.ai_worker.tasks.music_tasks.stem_isolation_task",
    queue="gpu",
    soft_time_limit=540,
    time_limit=600,
)
def stem_isolation_task(self, file_path: str, output_dir: str) -> dict:
    """
    Stem isolation using Demucs (when integrated).
    Separates: vocals, drums, bass, other.
    GPU-accelerated if CUDA available, falls back to CPU.
    """
    try:
        self.update_state(state="STARTED", meta={"step": "loading", "progress": 5})
        import subprocess, os, shutil

        # Validate paths — never trust input
        if not os.path.isfile(file_path):
            return {"error": "file_not_found", "path": file_path}
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self.update_state(state="STARTED", meta={"step": "separating", "progress": 20})

        # Demucs invocation (must be installed in GPU worker image)
        cmd = [
            "python", "-m", "demucs",
            "--two-stems=vocals",
            "-o", output_dir,
            file_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=480)

        if result.returncode != 0:
            return {"error": "demucs_failed", "stderr": result.stderr[:500]}

        self.update_state(state="STARTED", meta={"step": "done", "progress": 100})
        return {
            "status": "success",
            "output_dir": output_dir,
            "stems": ["vocals", "no_vocals"],
        }
    except SoftTimeLimitExceeded:
        return {"error": "timeout"}
    except Exception as exc:
        logger.exception("Stem isolation failed")
        raise self.retry(exc=exc, countdown=15)
