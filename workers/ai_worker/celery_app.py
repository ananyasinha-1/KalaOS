"""
KalaOS — Celery Application
=============================
Production-grade Celery setup:
- Redis broker + result backend
- Separate queues for GPU vs CPU tasks
- Task serialization with msgpack
- Auto-retry with exponential backoff
- Dead-letter queue routing
- Worker health signals
"""
from __future__ import annotations

import os
from celery import Celery
from celery.signals import worker_ready, worker_shutdown
from kombu import Exchange, Queue

# ── App instantiation ──────────────────────────────────────────────────────
celery_app = Celery("kalaos")

celery_app.config_from_object("workers.ai_worker.celery_config")

# ── Queue topology ─────────────────────────────────────────────────────────
# GPU queue: heavy inference (composition, image gen, video)
# CPU queue: lightweight analysis tasks
# Default queue: general purpose

default_exchange = Exchange("default", type="direct")
gpu_exchange = Exchange("gpu", type="direct")
priority_exchange = Exchange("priority", type="direct")

celery_app.conf.task_queues = (
    Queue("default", default_exchange, routing_key="default"),
    Queue("gpu", gpu_exchange, routing_key="gpu"),
    Queue("priority", priority_exchange, routing_key="priority"),
)
celery_app.conf.task_default_queue = "default"
celery_app.conf.task_default_exchange = "default"
celery_app.conf.task_default_routing_key = "default"

# Route GPU-heavy tasks to the gpu queue
celery_app.conf.task_routes = {
    "workers.ai_worker.tasks.music_tasks.compose_task": {"queue": "gpu"},
    "workers.ai_worker.tasks.music_tasks.stem_isolation_task": {"queue": "gpu"},
    "workers.ai_worker.tasks.visual_tasks.*": {"queue": "gpu"},
    "workers.ai_worker.tasks.video_tasks.*": {"queue": "gpu"},
    "workers.ai_worker.tasks.text_tasks.deep_analysis_task": {"queue": "default"},
}

# ── Lifecycle signals ──────────────────────────────────────────────────────

@worker_ready.connect
def on_worker_ready(sender=None, **kwargs):
    import logging
    logging.getLogger(__name__).info(
        "Celery worker ready [hostname=%s]", sender.hostname if sender else "unknown"
    )


@worker_shutdown.connect
def on_worker_shutdown(sender=None, **kwargs):
    import logging
    logging.getLogger(__name__).info("Celery worker shutting down.")
