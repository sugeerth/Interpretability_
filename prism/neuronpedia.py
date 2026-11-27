"""Neuronpedia client with an on-disk cache.

Turns local SAE feature indices into human-readable evidence: auto-interp
explanations, top promoted / suppressed tokens, max-activating examples.
Set NEURONPEDIA_API_KEY for authenticated endpoints (search); feature lookups
are public.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import httpx

BASE = "https://www.neuronpedia.org"


class Neuronpedia:
    def __init__(self, cache_dir: Optional[str] = None, timeout: float = 20.0, offline: bool = False):
        self.cache = Path(cache_dir or os.environ.get("PRISM_CACHE", Path.home() / ".cache" / "prism")) / "neuronpedia"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.timeout, self.offline = timeout, offline
        key = os.environ.get("NEURONPEDIA_API_KEY")
        self.headers = {"x-api-key": key} if key else {}

    def _cached(self, key: str, fetch):
        f = self.cache / (hashlib.sha1(key.encode()).hexdigest() + ".json")
        if f.exists():
            return json.loads(f.read_text())
        if self.offline:
            return None
        data = fetch()
        f.write_text(json.dumps(data))
        return data

    def feature(self, model: str, source: str, index: int) -> Optional[dict]:
        """Compact summary of one feature, or None if unavailable."""
        def fetch():
            resp = httpx.get(f"{BASE}/api/feature/{model}/{source}/{index}", headers=self.headers,
                             timeout=self.timeout, follow_redirects=True)
            resp.raise_for_status()
            d = resp.json() or {}
            examples = []
            for a in (d.get("activations") or [])[:4]:
                i = a.get("maxValueTokenIndex") or 0
                toks = a.get("tokens") or []
                examples.append(dict(text="".join(toks[max(0, i - 8):i + 4]), peak="".join(toks[i:i + 1]),
                                     value=round(a.get("maxValue") or 0, 2)))
            return dict(
                id=f"{model}/{source}/{index}", url=f"{BASE}/{model}/{source}/{index}",
                explanation=next((e.get("description") for e in d.get("explanations") or [] if e.get("description")), None),
                explanations=[e.get("description") for e in (d.get("explanations") or [])][:3],
                promotes=(d.get("pos_str") or [])[:8], suppresses=(d.get("neg_str") or [])[:8],
                density=d.get("frac_nonzero"), max_act=d.get("maxActApprox"), examples=examples)
        try:
            return self._cached(f"feature/{model}/{source}/{index}", fetch)
        except Exception as e:                                     # network / 404: degrade, do not fail the sweep
            return dict(id=f"{model}/{source}/{index}", url=f"{BASE}/{model}/{source}/{index}", explanation=None,
                        error=type(e).__name__)

    def search(self, model: str, query: str, n: int = 10) -> list:
        """Find features whose explanation matches ``query`` (semantic search)."""
        def fetch():
            resp = httpx.post(f"{BASE}/api/explanation/search-model", headers=self.headers, timeout=self.timeout,
                              json=dict(modelId=model, query=query))
            resp.raise_for_status()
            return [dict(id=f"{x.get('modelId')}/{x.get('layer')}/{x.get('index')}", layer=x.get("layer"),
                         index=int(x.get("index")), explanation=x.get("description"))
                    for x in (resp.json().get("results") or [])]
        try:
            return (self._cached(f"search/{model}/{query}", fetch) or [])[:n]
        except Exception as e:
            return [dict(error=f"{type(e).__name__}: {e}")]

    def annotate(self, features: list) -> list:
        """Attach Neuronpedia summaries to feature dicts carrying a 'neuronpedia' id."""
        for f in features:
            if f.get("neuronpedia"):
                model, source, idx = f["neuronpedia"].split("/")
                info = self.feature(model, source, int(idx)) or {}
                f.update(explanation=info.get("explanation"), promotes=info.get("promotes"),
                         suppresses=info.get("suppresses"), url=info.get("url"), examples=info.get("examples"))
        return features
