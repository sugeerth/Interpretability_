"""Concept directions without an SAE: works on any open model.

Given prompts that do and do not express a concept, the difference of mean
residual activations at each layer is a candidate direction (contrastive
activation addition / difference-of-means probing). The module scores how
well each layer separates the two sets on held-out prompts, tests the
direction causally by steering, and — when an SAE exists for that layer —
names the SAE features closest to it.
"""
from __future__ import annotations

from typing import Optional

import torch

from . import methods as M
from .backend import WhiteBoxModel
from .methods import r


def _acts(m: WhiteBoxModel, prompts: list) -> torch.Tensor:
    """Final-position residual entering every layer (+ final): [N, L+1, d]."""
    rows = []
    for p in prompts:
        c = m.forward(m.encode(p), cache=["resid_pre"])
        rows.append(torch.stack([x[0, -1] for x in c.resid_pre] + [c.resid_final[0, -1]]))
    return torch.stack(rows)


def concept_direction(m: WhiteBoxModel, positive: list, negative: list, steer_prompt: Optional[str] = None,
                      layer: Optional[int] = None, sae=None, neuronpedia=None,
                      strengths=(-8.0, -4.0, 0.0, 4.0, 8.0), k: int = 6) -> dict:
    if min(len(positive), len(negative)) < 3:
        raise ValueError("need at least 3 positive and 3 negative prompts")
    P, N = _acts(m, positive), _acts(m, negative)
    L = P.shape[1]
    paired = len(positive) == len(negative)
    layers = []
    for l in range(L):
        # Leave-one-out. Prompts are held out by index from *both* sets: with matched pairs (same sentence,
        # concept flipped) leaving out only one member lets the pair's shared content leak into the direction.
        hits, n = 0, 0
        for X, Y, sign in ((P, N, 1), (N, P, -1)):
            for i in range(X.shape[0]):
                keep_y = [j for j in range(Y.shape[0]) if j != i] if paired else list(range(Y.shape[0]))
                mx, my = torch.cat([X[:i, l], X[i + 1:, l]]).mean(0), Y[keep_y, l].mean(0)
                hits += int(float((X[i, l] - (mx + my) / 2) @ (mx - my)) > 0); n += 1
        d = P[:, l].mean(0) - N[:, l].mean(0)
        u = d / d.norm()
        pp, nn = P[:, l] @ u, N[:, l] @ u
        pooled = ((pp.var() + nn.var()) / 2).sqrt()
        layers.append(dict(layer=l, loo_accuracy=r(hits / n, 3), dprime=r(float((pp.mean() - nn.mean()) / (pooled + 1e-9)), 2),
                           norm=r(float(d.norm()), 2), rel_norm=r(float(d.norm() / P[:, l].norm(dim=-1).mean()), 3)))
    if layer is None:                                        # best held-out separation; never the embedding or final state
        layer = max(layers[1:L - 1], key=lambda x: (x["loo_accuracy"], x["dprime"]))["layer"]
    direction = P[:, layer].mean(0) - N[:, layer].mean(0)
    unit = direction / direction.norm()
    out = dict(kind="concept", model=m.info(), positive=positive, negative=negative, layers=layers, layer=layer)
    logit = m.lens(unit[None, None])[0, 0] - m.lens(torch.zeros_like(unit)[None, None])[0, 0]
    hi, lo = logit.topk(k), (-logit).topk(k)
    out["lens"] = dict(promotes=[m.tok.decode([int(i)]) for i in hi.indices], suppresses=[m.tok.decode([int(i)]) for i in lo.indices])
    if steer_prompt:
        task = M.make_task(m, steer_prompt)
        scale = float(direction.norm())
        curve = []
        for s in strengths:
            logits = m.forward(task.ids, hooks=[M.direction_hook(layer, unit, add=s * scale / 4, pos=None)] if s else []).logits
            p = logits[0, -1].softmax(-1)
            top = p.topk(4)
            curve.append(dict(strength=s, top=[dict(token=m.tok.decode([int(i)]), prob=r(float(v))) for v, i in zip(top.values, top.indices)]))
        out["steering"] = dict(prompt=steer_prompt, unit="quarters of the direction's natural norm", curve=curve)
    if sae is not None and sae.layer == layer:
        cos = torch.nn.functional.cosine_similarity(sae.W_dec, unit[None], dim=-1)
        top = cos.topk(k)
        feats = [dict(layer=layer, index=int(i), cosine=r(float(v), 3),
                      neuronpedia=f"{sae.neuronpedia_model}/{sae.neuronpedia_source}/{int(i)}") for v, i in zip(top.values, top.indices)]
        if neuronpedia is not None:
            neuronpedia.annotate(feats)
        out["nearest_features"] = feats
    out["_direction"] = unit
    return out
