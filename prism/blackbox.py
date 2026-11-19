"""Semi-open models: interpretability when you only have an API.

Closed-weights models (Anthropic, OpenAI, any hosted endpoint) expose no
activations, so the methods here are perturbation-based and only need a
*scorer*: ``score(text) -> float``. Three scorers are provided:

* ``OpenAIScorer``     log-probability of a target answer (uses ``logprobs``)
* ``AnthropicScorer``  the API returns no token probabilities, so the score is
                       the empirical frequency of the target answer over n
                       constrained samples
* ``LocalScorer``      an open-weights model behind the same interface — used
                       to validate the black-box method against gradients, and
                       as a surrogate whose internals can be opened

``occlusion`` then gives input attribution, ``counterfactual_pairs`` gives a
minimal-edit sensitivity map, and ``surrogate_agreement`` measures how far a
white-box surrogate's attribution can be trusted as a stand-in.
"""
from __future__ import annotations

import math
import os
import re
from typing import Callable, Optional

from .methods import r

Scorer = Callable[[str], float]


def segment(text: str, unit: str = "word") -> list:
    """Split into perturbation units, keeping whitespace attached so that
    ``''.join(units) == text``."""
    if unit == "sentence":
        parts = re.findall(r"[^.!?\n]+[.!?\n]*\s*", text)
    elif unit == "char":
        parts = list(text)
    else:
        parts = re.findall(r"\s*\S+\s*", text)
    return parts or [text]


def occlusion(score: Scorer, text: str, unit: str = "word", mask: str = "", window: int = 1) -> dict:
    """Leave-one-out attribution: score(full) - score(text without unit i)."""
    units = segment(text, unit)
    full = score(text)
    attr = []
    for i in range(len(units)):
        lo, hi = i, min(len(units), i + window)
        ablated = "".join(units[:lo]) + (mask + " " if mask else "") + "".join(units[hi:])
        attr.append(full - score(ablated))
    return dict(units=units, attribution=r(attr), full_score=r(full), n_queries=len(units) + 1, unit=unit)


def counterfactual_pairs(score: Scorer, text: str, swaps: dict) -> dict:
    """Minimal-edit sensitivity: replace one phrase at a time (``swaps`` maps
    original -> replacement) and record the score change."""
    full = score(text)
    rows = [dict(original=a, replacement=b, delta=r(score(text.replace(a, b)) - full)) for a, b in swaps.items() if a in text]
    return dict(full_score=r(full), edits=sorted(rows, key=lambda x: -abs(x["delta"])))


def surrogate_agreement(a: list, b: list) -> dict:
    """Spearman rank correlation between two attribution vectors over the same
    units — e.g. a hosted model's occlusion map vs. an open surrogate's."""
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("attribution vectors must have the same length >= 2")

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        for rank, i in enumerate(order):
            out[i] = rank
        return out
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) or 1e-12
    top_a, top_b = max(range(len(a)), key=lambda i: abs(a[i])), max(range(len(b)), key=lambda i: abs(b[i]))
    return dict(spearman=r(cov / den), same_top_unit=top_a == top_b)


# ------------------------------------------------------------------ scorers
class LocalScorer:
    """Open-weights model exposed as a black box: log p(target | text)."""
    access = "open-weights (as black box)"

    def __init__(self, model, target: str, contrast: Optional[str] = None):
        self.m, self.t = model, model.token_id(target)
        self.c = model.token_id(contrast) if contrast else None

    def __call__(self, text: str) -> float:
        logits = self.m.forward(self.m.encode(text)).logits[0, -1]
        if self.c is not None:
            return float(logits[self.t] - logits[self.c])
        return float(logits.log_softmax(-1)[self.t])


class OpenAIScorer:
    """log p(first answer token == target) from Chat Completions ``logprobs``."""
    access = "semi-open (logprobs)"

    def __init__(self, target: str, model: str = "gpt-4o-mini", system: str = "Answer with a single word.",
                 floor: float = -20.0, client=None):
        if client is None:
            from openai import OpenAI
            client = OpenAI()
        self.client, self.model, self.target, self.system, self.floor = client, model, target.strip().lower(), system, floor

    def __call__(self, text: str) -> float:
        resp = self.client.chat.completions.create(
            model=self.model, max_completion_tokens=1, logprobs=True, top_logprobs=20,
            messages=[{"role": "system", "content": self.system}, {"role": "user", "content": text}])
        total = -math.inf
        for alt in resp.choices[0].logprobs.content[0].top_logprobs:
            if alt.token.strip().lower() == self.target:
                total = alt.logprob if total == -math.inf else math.log(math.exp(total) + math.exp(alt.logprob))
        return max(total, self.floor)


class AnthropicScorer:
    """Frequency of the target answer over ``n`` samples (no logprobs exposed).

    Scores are in [0, 1] with resolution 1/n, so attribution below that
    resolution is noise — ``n`` trades cost for sensitivity.
    """
    access = "semi-open (samples only)"

    def __init__(self, target: str, model: str = "claude-opus-5-5", n: int = 5,
                 system: str = "Answer with a single word and nothing else.", client=None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client, self.model, self.target, self.n, self.system = client, model, target.strip().lower(), n, system
        self.cache: dict = {}

    def __call__(self, text: str) -> float:
        if text not in self.cache:
            hits = 0
            for _ in range(self.n):
                resp = self.client.messages.create(model=self.model, max_tokens=1024, system=self.system,
                                                   output_config={"effort": "low"},
                                                   messages=[{"role": "user", "content": text}])
                if resp.stop_reason == "refusal":
                    continue
                answer = "".join(b.text for b in resp.content if b.type == "text")
                hits += self.target in answer.strip().lower()
            self.cache[text] = hits / self.n
        return self.cache[text]


def make_scorer(provider: str, target: str, model: Optional[str] = None, **kw) -> Scorer:
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        return OpenAIScorer(target, **({"model": model} if model else {}), **kw)
    if provider == "anthropic":
        return AnthropicScorer(target, **({"model": model} if model else {}), **kw)
    raise ValueError(f"unknown provider {provider!r}")
