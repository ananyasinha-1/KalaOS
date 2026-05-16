"""
KalaOS — Celery Tasks: Text & Deep Analysis
=============================================
Heavy LLM inference tasks run in Celery workers,
keeping the FastAPI event loop completely free.
- Exponential retry on transient failures
- Task state updates for progress streaming
- Never block; always async-to-sync via run_in_executor pattern
"""
from __future__ import annotations

import logging
from typing import Optional

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from workers.ai_worker.celery_app import celery_app

logger = logging.getLogger(__name__)


class BaseKalaTask(Task):
    """Base task with shared error handling and retry logic."""
    abstract = True
    max_retries = 3

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(
            "Task failed [task_id=%s, exc=%s]", task_id, exc, exc_info=einfo
        )

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        logger.warning("Task retrying [task_id=%s, attempt=%s]", task_id, self.request.retries)


@celery_app.task(
    bind=True,
    base=BaseKalaTask,
    name="workers.ai_worker.tasks.text_tasks.deep_analysis_task",
    queue="default",
    soft_time_limit=280,
    time_limit=300,
)
def deep_analysis_task(
    self,
    text: str,
    art_domain: str = "general",
    artist_name: Optional[str] = None,
    creation_context: Optional[str] = None,
    model: Optional[str] = None,
) -> dict:
    """
    Full deep analysis pipeline run inside Celery worker.
    Equivalent to the old synchronous /deep-analysis endpoint
    but now non-blocking for the API layer.
    """
    try:
        self.update_state(state="STARTED", meta={"step": "ethics_check", "progress": 5})

        from kalacore.ethics import check_request
        violations = check_request(text)
        if violations:
            return {"error": "content_policy", "violations": [
                {"code": v.code, "message": v.message} for v in violations
            ]}

        self.update_state(state="STARTED", meta={"step": "pattern_analysis", "progress": 15})
        from kalacore.pattern_engine import analyze
        analysis = analyze(text)

        self.update_state(state="STARTED", meta={"step": "art_genome", "progress": 25})
        from kalacore.art_genome import build_art_genome
        genome = build_art_genome(analysis)

        self.update_state(state="STARTED", meta={"step": "existential", "progress": 35})
        from kalacore.existential import analyze_existential
        existential_data = analyze_existential(text, analysis)

        self.update_state(state="STARTED", meta={"step": "craft", "progress": 45})
        from kalacore.kalacraft import analyze_craft
        craft_data = analyze_craft(text)

        self.update_state(state="STARTED", meta={"step": "signal", "progress": 55})
        from kalacore.kalasignal import analyze_signal
        signal_data = analyze_signal(text, genome.to_dict())

        self.update_state(state="STARTED", meta={"step": "compose", "progress": 65})
        from kalacore.kalacomposer import compose
        composition_data = compose(text, analysis, genome.to_dict())

        self.update_state(state="STARTED", meta={"step": "flow", "progress": 72})
        from kalacore.kalaflow import flow
        flow_data = flow(text, analysis, genome.to_dict(), existential_data)

        self.update_state(state="STARTED", meta={"step": "custody", "progress": 80})
        from kalacore.kalacustody import custody, assess_artistic_lineage
        custody_data = custody(
            text, analysis, genome.to_dict(), existential_data,
            artist_name=artist_name, creation_context=creation_context,
        )

        self.update_state(state="STARTED", meta={"step": "temporal", "progress": 88})
        from kalacore.temporal import analyze_temporal
        lines = [l for l in text.splitlines() if l.strip()]
        lineage_data = assess_artistic_lineage(lines, analysis, genome.to_dict())
        temporal_data = analyze_temporal(text, analysis, genome.to_dict(), existential_data, lineage_data)

        self.update_state(state="STARTED", meta={"step": "llm_narrative", "progress": 93})
        from services.llm_service import generate_deep_narrative
        all_data = {
            "art_genome": genome.to_dict(),
            "existential": existential_data,
            "craft": craft_data,
            "signal": signal_data,
            "composition": composition_data,
            "flow": flow_data,
            "custody": custody_data,
            "temporal": temporal_data,
        }
        try:
            narrative = generate_deep_narrative(all_data, **({"model": model} if model else {}))
        except Exception as llm_exc:
            logger.warning("LLM narrative failed: %s", llm_exc)
            narrative = "[Narrative unavailable — all structured analysis complete above.]"

        return {
            "narrative": narrative,
            "art_genome": genome.to_dict(),
            "analysis": analysis,
            "existential": existential_data,
            "craft": craft_data,
            "signal": signal_data,
            "composition": composition_data,
            "flow": flow_data,
            "custody": custody_data,
            "temporal": temporal_data,
        }

    except SoftTimeLimitExceeded:
        logger.error("Deep analysis task hit soft time limit [task_id=%s]", self.request.id)
        return {"error": "timeout", "message": "Analysis timed out. Try a shorter text."}
    except Exception as exc:
        logger.exception("Deep analysis task error")
        raise self.retry(exc=exc, countdown=2 ** self.request.retries * 5)
