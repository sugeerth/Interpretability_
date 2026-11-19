"""Corpus sweeps: from one prompt to a distribution of prompts.

A single prompt can flatter a story. ``corpus_sweep`` runs the cheap parts of
the sweep over many prompts of one task and aggregates: which heads matter
*consistently* (mean, spread and sign agreement of direct logit attribution
and of exact head patching), and which SAE features recur.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Optional

import torch

from . import methods as M
from .methods import r
from .sae import feature_attribution

NAMES = ["Mary", "John", "Tom", "James", "Anna", "Sarah", "David", "Emma", "Michael", "Laura", "Robert", "Alice"]
PLACES = ["store", "park", "school", "office", "station", "garden", "hospital", "restaurant"]
OBJECTS = ["drink", "book", "ring", "letter", "key", "phone", "gift", "ticket"]
TEMPLATES = ["When {A} and {B} went to the {place}, {B} gave a {obj} to",
             "After {A} and {B} arrived at the {place}, {B} handed a {obj} to",
             "While {A} and {B} were at the {place}, {B} passed a {obj} to"]


def ioi_dataset(n: int = 32, seed: int = 0) -> list:
    """Indirect-object prompts with a matched corruption (subject swapped)."""
    rng, out = random.Random(seed), []
    for i in range(n):
        a, b = rng.sample(NAMES, 2)
        kw = dict(place=rng.choice(PLACES), obj=rng.choice(OBJECTS))
        t = TEMPLATES[i % len(TEMPLATES)]
        out.append(dict(prompt=t.format(A=a, B=b, **kw), target=" " + a, contrast=" " + b,
                        corrupt=t.replace("{B} gave", "{A} gave").replace("{B} handed", "{A} handed").replace("{B} passed", "{A} passed").format(A=a, B=b, **kw)))
    return out


def corpus_sweep(ws, items: list, model: str = "gpt2", sae_layers: Optional[list] = None, features_k: int = 8,
                 progress=None, name: str = "corpus") -> dict:
    m = ws.model(model)
    layers = ws.default_sae_layers(model) if sae_layers is None else sae_layers
    saes = [ws.sae(model, l) for l in layers]
    dla, patch, metrics, answer_layer = [], [], [], []
    feats = defaultdict(lambda: dict(n=0, effect=0.0, prompts=set(), tokens=defaultdict(int)))
    skipped = 0
    with ws.lock:
        for i, it in enumerate(items):
            if progress:
                progress(i, len(items))
            task = M.make_task(m, it["prompt"], it.get("target"), it.get("contrast"))
            metrics.append(float(task.metric(m.forward(task.ids).logits)))
            dla.append(torch.tensor(M.direct_logit_attribution(m, task, top_neurons=1)["heads"]))
            rank = M.logit_lens(m, task)["target_rank"]
            answer_layer.append(next((l for l, rk in enumerate(rank) if rk == 0), len(rank)))
            if it.get("corrupt"):
                cids = m.encode(it["corrupt"])
                if cids.shape == task.ids.shape:
                    patch.append(torch.tensor(M.head_patching(m, task, cids)["heads"]))
                else:
                    skipped += 1
            if saes:
                fa = feature_attribution(m, task, saes, k=features_k, graph=False)
                for L in fa["layers"]:
                    for f in L["features"]:
                        e = feats[(f["layer"], f["index"])]
                        e["prompts"].add(i); e["n"] += 1; e["effect"] += f["effect"]; e["tokens"][f["token"]] += 1
    D = torch.stack(dla)

    def stats(X):
        mean = X.mean(0)
        return dict(mean=r(mean, 3), std=r(X.std(0) if X.shape[0] > 1 else torch.zeros_like(mean), 3),
                    sign_agreement=r(((X.sign() == mean.sign()).float().mean(0)), 2))
    out = dict(kind="corpus", name=name, model=m.info(), n=len(items), examples=[it["prompt"] for it in items[:4]],
               metric_mean=r(sum(metrics) / len(metrics)), accuracy=r(sum(x > 0 for x in metrics) / len(metrics), 3),
               answer_layer_mean=r(sum(answer_layer) / len(answer_layer), 2), dla=stats(D), patch_skipped=skipped)
    if patch:
        out["patching"] = stats(torch.stack(patch))
        out["patching"]["n"] = len(patch)
    rows = []
    for (layer, index), e in feats.items():
        sae = ws.sae(model, layer)
        rows.append(dict(layer=layer, index=index, frequency=r(len(e["prompts"]) / len(items), 3), mean_effect=r(e["effect"] / len(e["prompts"]), 3),
                         tokens=sorted(e["tokens"], key=lambda t: -e["tokens"][t])[:4],
                         neuronpedia=f"{sae.neuronpedia_model}/{sae.neuronpedia_source}/{index}" if sae.neuronpedia_model else None))
    rows.sort(key=lambda x: -(x["frequency"] * abs(x["mean_effect"])))
    out["features"] = ws.np.annotate(rows[:18])
    return out
