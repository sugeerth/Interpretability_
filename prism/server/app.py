"""HTTP API + static web app.

    uvicorn prism.server.app:app --host 0.0.0.0 --port 8000

Environment:
    PRISM_MODELS        comma-separated HF model ids that may be loaded (default "gpt2")
    PRISM_MAX_TOKENS    longest accepted prompt in tokens (default 64)
    PRISM_RATE          compute requests per client per minute (default 20)
    ANTHROPIC_API_KEY   enables the LLM-driven agent (offline protocol otherwise)
    OPENAI_API_KEY      enables black-box attribution on OpenAI models
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from collections import OrderedDict, defaultdict, deque
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..agent.loop import Investigation, has_llm, investigate, investigate_offline
from ..workspace import Workspace

WEB = Path(__file__).resolve().parents[2] / "web"
ALLOWED = [m.strip() for m in os.environ.get("PRISM_MODELS", "gpt2").split(",") if m.strip()]
MAX_TOKENS = int(os.environ.get("PRISM_MAX_TOKENS", "64"))
RATE = int(os.environ.get("PRISM_RATE", "20"))

app = FastAPI(title="Prism — agentic interpretability", version="0.1.0")
ws = Workspace()
_cache: OrderedDict = OrderedDict()
_hits: dict = defaultdict(deque)


def limit(request: Request) -> None:
    ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0]
    now, q = time.time(), _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE:
        raise HTTPException(429, "rate limit exceeded; try again in a minute")
    q.append(now)


def check(model: str, *prompts: Optional[str]) -> None:
    if model not in ALLOWED:
        raise HTTPException(400, f"model {model!r} is not enabled on this server; available: {ALLOWED}")
    m = ws.model(model)
    for p in prompts:
        if p and m.encode(p).shape[1] > MAX_TOKENS:
            raise HTTPException(400, f"prompt is longer than {MAX_TOKENS} tokens")


def cached(key_obj, fn):
    key = hashlib.sha1(json.dumps(key_obj, sort_keys=True).encode()).hexdigest()
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    val = fn()
    _cache[key] = val
    while len(_cache) > 64:
        _cache.popitem(last=False)
    return val


class SweepReq(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    model: str = "gpt2"
    target: Optional[str] = None
    contrast: Optional[str] = None
    corrupt: Optional[str] = None


class InterveneReq(SweepReq):
    kind: str
    layer: int
    index: int
    scale: float = 0.0
    strength: float = 0.0
    pos: Optional[int] = None


class BlackboxReq(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    target: str
    provider: str = "local"
    model: Optional[str] = None
    contrast: Optional[str] = None


class AgentReq(SweepReq):
    question: Optional[str] = None


@app.get("/api/health")
def health():
    return dict(ok=True, models=ALLOWED, loaded=list(ws.models), device=ws.device, llm_agent=has_llm(),
                openai=bool(os.environ.get("OPENAI_API_KEY")), max_tokens=MAX_TOKENS,
                sae_layers={m: ws.sae_layers(m) for m in ALLOWED})


@app.post("/api/sweep")
def sweep(req: SweepReq, request: Request):
    limit(request)
    check(req.model, req.prompt, req.corrupt)
    try:
        return cached(["sweep", req.model_dump()], lambda: ws.sweep(req.prompt, req.model, req.target or None,
                                                                    req.contrast or None, req.corrupt or None))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/intervene")
def intervene(req: InterveneReq, request: Request):
    limit(request)
    check(req.model, req.prompt)
    from .. import methods as M
    inv = Investigation(ws, req.model)
    inv.task = M.make_task(ws.model(req.model), req.prompt, req.target or None, req.contrast or None)
    out = inv.intervene(req.kind, req.layer, req.index, req.scale, req.strength, req.pos)
    if "error" in out:
        raise HTTPException(400, out["error"])
    return out


@app.get("/api/feature/{model}/{layer}/{index}")
def feature(model: str, layer: int, index: int):
    if model not in ALLOWED or layer not in ws.sae_layers(model):
        raise HTTPException(404, "no SAE for that model/layer")
    sae = ws.sae(model, layer)
    return ws.np.feature(sae.neuronpedia_model, sae.neuronpedia_source, index)


@app.post("/api/blackbox")
def blackbox(req: BlackboxReq, request: Request):
    limit(request)
    if req.provider == "local":
        check(req.model or ALLOWED[0], req.prompt)
    elif len(req.prompt.split()) > 40:
        raise HTTPException(400, "prompt too long for hosted-model occlusion (40 words max: one API call per word)")
    try:
        return cached(["bb", req.model_dump()], lambda: Investigation(ws, ALLOWED[0]).blackbox(
            req.prompt, req.target, req.provider, req.model, req.contrast))
    except (RuntimeError, ValueError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/agent")
def agent(req: AgentReq, request: Request):
    """Server-sent events: one ``step`` event per experiment, then ``done``."""
    limit(request)
    check(req.model, req.prompt, req.corrupt)
    q: queue.Queue = queue.Queue()

    def run():
        try:
            if has_llm() and req.question:
                question = (f"{req.question}\n\nPrompt: {req.prompt!r}" + (f"\nTarget token: {req.target!r}" if req.target else "")
                            + (f"\nContrast token: {req.contrast!r}" if req.contrast else "")
                            + (f"\nCorrupted prompt for patching: {req.corrupt!r}" if req.corrupt else ""))
                res = investigate(question, ws, req.model, on_step=lambda s: q.put(("step", s)))
            else:
                res = investigate_offline(req.prompt, ws, req.model, req.target or None, req.contrast or None,
                                          req.corrupt or None, on_step=lambda s: q.put(("step", s)))
            q.put(("done", res))
        except Exception as e:
            q.put(("error", dict(error=f"{type(e).__name__}: {e}")))

    threading.Thread(target=run, daemon=True).start()

    def stream():
        while True:
            try:
                kind, data = q.get(timeout=15)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            if kind == "step" and data.get("tool") == "sweep":
                data = dict(data, output="sweep complete")          # the bundle arrives with 'done'
            yield f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
            if kind in ("done", "error"):
                return
    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/", StaticFiles(directory=WEB), name="web")
