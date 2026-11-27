"""Persistence and background jobs.

``BundleStore`` is a content-addressed store for result bundles (local
directory; the interface is small enough to back with object storage).
``Jobs`` runs long computations off the request thread with progress, so the
API returns immediately and clients poll or stream.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional


class BundleStore:
    def __init__(self, root: Optional[str] = None, max_items: int = 5000):
        self.root = Path(root or os.environ.get("PRISM_STORE", Path.home() / ".cache" / "prism" / "bundles"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_items = max_items

    @staticmethod
    def key(request: dict) -> str:
        return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]

    def path(self, key: str) -> Path:
        if not key.isalnum():
            raise ValueError("invalid bundle id")
        return self.root / f"{key}.json"

    def get(self, key: str) -> Optional[dict]:
        p = self.path(key)
        if not p.exists():
            return None
        os.utime(p)
        return json.loads(p.read_text())

    def put(self, key: str, bundle: dict) -> str:
        bundle = dict(bundle, id=key)
        tmp = self.path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps(bundle, ensure_ascii=False, default=str))
        tmp.replace(self.path(key))
        self._evict()
        return key

    def get_or_compute(self, request: dict, fn: Callable[[], dict]) -> dict:
        key = self.key(request)
        hit = self.get(key)
        if hit is not None:
            return hit
        self.put(key, dict(fn(), request=request))
        return self.get(key)

    def _evict(self) -> None:
        files = sorted(self.root.glob("*.json"), key=lambda f: f.stat().st_mtime)
        for f in files[: max(0, len(files) - self.max_items)]:
            f.unlink(missing_ok=True)


class Jobs:
    """Single-worker job runner: model forward passes are serialised anyway."""

    def __init__(self, keep: int = 200):
        self.jobs: OrderedDict = OrderedDict()
        self.keep = keep
        self.lock = threading.Lock()

    def submit(self, kind: str, fn: Callable[[Callable], dict]) -> str:
        jid = uuid.uuid4().hex[:12]
        job = dict(id=jid, kind=kind, status="queued", progress=None, created=time.time(), result=None, error=None)
        with self.lock:
            self.jobs[jid] = job
            while len(self.jobs) > self.keep:
                self.jobs.popitem(last=False)

        def run():
            job["status"] = "running"
            try:
                job["result"] = fn(lambda *p: job.__setitem__("progress", list(p)))
                job["status"] = "done"
            except Exception as e:
                job.update(status="error", error=f"{type(e).__name__}: {e}")
            job["finished"] = time.time()
        threading.Thread(target=run, daemon=True).start()
        return jid

    def get(self, jid: str) -> Optional[dict]:
        return self.jobs.get(jid)
