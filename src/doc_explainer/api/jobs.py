"""Background-job bookkeeping for document registration.

Registration can take many minutes (one model call per section, rate-limited), so it runs on a worker
and clients poll. Jobs are persisted: after a restart a client polling an old job gets a definite
answer instead of a 404 (anything that was queued or running is reported as failed/interrupted).
"""
from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

QUEUED, RUNNING, PAUSED, SUCCEEDED, FAILED, CANCELLED = "queued", "running", "paused", "succeeded", "failed", "cancelled"
ACTIVE = (QUEUED, RUNNING, PAUSED)     # a paused job still holds its place (and its files)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, path: Path, keep: int = 500):
        self._path = path
        self._keep = keep
        self._lock = threading.RLock()
        self._jobs: Dict[str, dict] = {}
        if path.exists():
            try:
                self._jobs = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                bad = path.with_suffix(".corrupt")
                logger.error("Job store %s is unreadable; moved to %s", path, bad)
                path.replace(bad)
        self._recover()

    def _recover(self) -> None:
        """Jobs that were queued/running when the server stopped can never finish. Paused ones can: resume re-runs them."""
        changed = False
        for job in self._jobs.values():
            if job["state"] in (QUEUED, RUNNING):
                job.update(state=FAILED, finished_at=now(), error={
                    "code": "processing_failed",
                    "message": "The server restarted before this document finished processing. "
                               "Upload it again."})
                if job.get("upload_dir"):
                    shutil.rmtree(job["upload_dir"], ignore_errors=True)
                changed = True
        if changed:
            self._save()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self._jobs, f, indent=2)
        os.replace(tmp, self._path)

    def _prune(self) -> None:
        done = sorted((j for j in self._jobs.values() if j["state"] not in ACTIVE),
                      key=lambda j: j["created_at"])
        for job in done[: max(0, len(self._jobs) - self._keep)]:
            del self._jobs[job["job_id"]]

    # ------------------------------------------------------------------ writes
    def create(self, user_id: str, filename: str, **fields: Any) -> dict:
        with self._lock:
            job = {"job_id": uuid.uuid4().hex, "user_id": user_id, "kind": "register_document",
                   "state": QUEUED, "filename": filename, "created_at": now(), "started_at": None,
                   "finished_at": None, "document": None, "error": None, "reused": False, "progress": 0.0, "stage": "", **fields}
            self._jobs[job["job_id"]] = job
            self._prune()
            self._save()
            return copy.deepcopy(job)

    def peek(self, job_id: str) -> Optional[dict]:
        """The job whoever owns it (internal use: the worker and job control)."""
        with self._lock:
            job = self._jobs.get(job_id)
            return copy.deepcopy(job) if job else None

    def update(self, job_id: str, **fields: Any) -> dict:
        with self._lock:
            self._jobs[job_id].update(fields)
            self._save()
            return copy.deepcopy(self._jobs[job_id])

    # ------------------------------------------------------------------ reads
    def get(self, job_id: str, user_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            return copy.deepcopy(job) if job and job["user_id"] == user_id else None

    def list(self, user_id: str, limit: int = 50) -> List[dict]:
        with self._lock:
            jobs = [j for j in self._jobs.values() if j["user_id"] == user_id]
        jobs.sort(key=lambda j: j["created_at"], reverse=True)
        return copy.deepcopy(jobs[:limit])

    def pending(self, user_id: Optional[str] = None) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j["state"] in ACTIVE
                       and (user_id is None or j["user_id"] == user_id))

    def queue_position(self, job_id: str) -> Optional[int]:
        """1 = next to run. None unless the job is waiting."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job["state"] != QUEUED:
                return None
            waiting = sorted((j for j in self._jobs.values() if j["state"] == QUEUED),
                             key=lambda j: j["created_at"])
            return [j["job_id"] for j in waiting].index(job_id) + 1

    def find_active(self, user_id: str, sha256: str, build_graph: bool) -> Optional[dict]:
        """An identical upload that is already waiting or running (so a retry does not queue twice)."""
        with self._lock:
            for j in self._jobs.values():
                if (j["user_id"] == user_id and j["state"] in ACTIVE and j.get("sha256") == sha256
                        and (j.get("build_graph") or not build_graph)):
                    return copy.deepcopy(j)
        return None
