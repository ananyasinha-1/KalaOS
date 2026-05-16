"""
KalaOS — Celery Configuration
================================
All Celery settings in one place.
Never import from Django settings or Flask config — 
this is a standalone worker config.
"""
from __future__ import annotations
import os

# ── Broker & Backend ───────────────────────────────────────────────────────
broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/1")
result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")

# ── Serialization ──────────────────────────────────────────────────────────
task_serializer = "json"
result_serializer = "json"
accept_content = ["json"]
timezone = "UTC"
enable_utc = True

# ── Task behavior ─────────────────────────────────────────────────────────
task_acks_late = True                  # Acknowledge after task completes
task_reject_on_worker_lost = True      # Re-queue on unexpected worker death
task_soft_time_limit = int(os.environ.get("CELERY_TASK_SOFT_TIME_LIMIT", 300))
task_time_limit = int(os.environ.get("CELERY_TASK_TIME_LIMIT", 600))
task_max_retries = 3
task_default_retry_delay = 5           # seconds (exponential applied in task)

# ── Result storage ────────────────────────────────────────────────────────
result_expires = 3600                  # Results expire after 1 hour
result_persistent = True

# ── Worker ────────────────────────────────────────────────────────────────
worker_prefetch_multiplier = 1         # Critical for GPU tasks — one at a time
worker_max_tasks_per_child = 50        # Recycle workers to prevent memory leaks
worker_disable_rate_limits = False

# ── Beat Scheduler (for periodic tasks) ───────────────────────────────────
beat_schedule = {
    "purge-expired-sessions": {
        "task": "workers.ai_worker.tasks.maintenance.purge_expired_sessions",
        "schedule": 3600.0,            # Every hour
    },
    "warm-model-cache": {
        "task": "workers.ai_worker.tasks.maintenance.warm_model_cache",
        "schedule": 1800.0,            # Every 30 minutes
    },
}

# ── Monitoring ────────────────────────────────────────────────────────────
worker_send_task_events = True
task_send_sent_event = True

# ── Auto-discovery ────────────────────────────────────────────────────────
imports = (
    "workers.ai_worker.tasks.text_tasks",
    "workers.ai_worker.tasks.music_tasks",
    "workers.ai_worker.tasks.visual_tasks",
    "workers.ai_worker.tasks.video_tasks",
    "workers.ai_worker.tasks.maintenance",
)
