"""
Phase A async job orchestration (in-process queue).

Provides a minimal queue-backed boundary between API routes and worker tasks.
This keeps heavy operations off direct request/response code paths and creates
an upgrade path to Redis/Celery/Temporal without changing route contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from queue import Queue
import threading
import uuid
from typing import Any, Dict, Optional

try:  # Runtime from backend/ working directory
    from workers.media_tasks import run_task
except ImportError:  # Package-style runtime
    from backend.workers.media_tasks import run_task


_VALID_PRIORITIES = {"low", "normal", "high"}
_VALID_GPU_CLASSES = {"small", "medium", "high"}
_MAX_JOB_HISTORY = 2_000
_MIN_COMPLETED_HISTORY = 1_000


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobRecord:
    id: str
    task_type: str
    payload: Dict[str, Any]
    status: str = "queued"
    created_at: str = field(default_factory=_utc_now_iso)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    priority: str = "normal"
    gpu_class: str = "small"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task_type": self.task_type,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "priority": self.priority,
            "gpu_class": self.gpu_class,
            "result": self.result,
            "error": self.error,
        }


_jobs: Dict[str, JobRecord] = {}
_jobs_lock = threading.Lock()
_job_queue: "Queue[str]" = Queue()
_workers_started = False
_startup_lock = threading.Lock()

def _cleanup_job_history_if_needed() -> None:
    with _jobs_lock:
        current_size = len(_jobs)
        if current_size <= _MAX_JOB_HISTORY:
            return

        overflow = current_size - _MAX_JOB_HISTORY
        completed = [j for j in _jobs.values() if j.status in {"completed", "failed"}]
        completed.sort(key=lambda j: j.completed_at or j.created_at)

        removable_count = max(0, len(completed) - _MIN_COMPLETED_HISTORY)
        to_remove = min(overflow, removable_count)
        for job in completed[:to_remove]:
            _jobs.pop(job.id, None)

        # Never evict queued/running jobs; if pressure remains, it is because
        # active workload is high and must be handled by scaling workers.


def _process_one_job(job_id: str) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job.status = "running"
        job.started_at = _utc_now_iso()

    try:
        output = run_task(job.task_type, job.payload)
        with _jobs_lock:
            current = _jobs.get(job_id)
            if current:
                current.status = "completed"
                current.completed_at = _utc_now_iso()
                current.result = output
    except Exception as exc:
        with _jobs_lock:
            current = _jobs.get(job_id)
            if current:
                current.status = "failed"
                current.completed_at = _utc_now_iso()
                current.error = str(exc)


def _worker_loop() -> None:
    while True:
        job_id = _job_queue.get()
        _process_one_job(job_id)
        _job_queue.task_done()


def start_workers_if_needed(worker_count: int = 2) -> None:
    global _workers_started
    with _startup_lock:
        if _workers_started:
            return
        for _ in range(worker_count):
            t = threading.Thread(target=_worker_loop, daemon=True)
            t.start()
        _workers_started = True


def submit_job(
    task_type: str,
    payload: Optional[Dict[str, Any]] = None,
    *,
    priority: str = "normal",
    gpu_class: str = "small",
) -> Dict[str, Any]:
    start_workers_if_needed()
    _cleanup_job_history_if_needed()
    if priority not in _VALID_PRIORITIES:
        raise ValueError(f"Invalid priority '{priority}'. Use one of: {', '.join(sorted(_VALID_PRIORITIES))}")
    if gpu_class not in _VALID_GPU_CLASSES:
        raise ValueError(f"Invalid gpu_class '{gpu_class}'. Use one of: {', '.join(sorted(_VALID_GPU_CLASSES))}")

    job = JobRecord(
        id=str(uuid.uuid4()),
        task_type=task_type,
        payload=payload or {},
        priority=priority,
        gpu_class=gpu_class,
    )
    with _jobs_lock:
        _jobs[job.id] = job
    _job_queue.put(job.id)
    return job.to_dict()


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return job.to_dict() if job else None


def list_jobs(limit: int = 50) -> list[Dict[str, Any]]:
    _cleanup_job_history_if_needed()
    safe_limit = max(1, min(limit, 200))
    with _jobs_lock:
        jobs = list(_jobs.values())
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return [j.to_dict() for j in jobs[:safe_limit]]
