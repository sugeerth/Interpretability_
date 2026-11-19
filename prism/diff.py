"""Model diffing: what changed between two checkpoints, and where.

Three complementary views, cheapest first:

* weight_diff      per-tensor relative update norm and the rank structure of
                   the update (fine-tunes / LoRA merges are typically low-rank)
* activation_diff  per-layer representation similarity (linear CKA, cosine)
                   and output divergence (KL) on probe prompts
* feature_diff     the same residual streams read through one shared SAE
                   basis: which labelled features a checkpoint gained or lost

``trace_diff`` applies the same idea to *behaviour*: it aligns two agent or
tool-call traces and reports where they diverge.
"""
from __future__ import annotations

import difflib
import re
from typing import Optional

import torch

from .backend import WhiteBoxModel
from .methods import r


def _component(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("wte", "embed_tokens", "wpe", "embed")):
        return "embed"
    if any(k in n for k in ("ln", "norm")):
        return "norm"
    if "attn" in n or "attention" in n:
        return "attn"
    if any(k in n for k in ("mlp", "fc", "feed_forward", "ffn")):
        return "mlp"
    return "other"


def weight_diff(base: WhiteBoxModel, tuned: WhiteBoxModel, top: int = 15) -> dict:
    """Relative Frobenius norm of every parameter update, with rank statistics."""
    pb, pt = dict(base.model.named_parameters()), dict(tuned.model.named_parameters())
    rows, L = [], base.n_layers
    per_layer = {c: [0.0] * L for c in ("attn", "mlp", "norm")}
    sq = {c: [0.0] * L for c in per_layer}
    for name, w in pb.items():
        if name not in pt or pt[name].shape != w.shape:
            continue
        d = (pt[name] - w).detach().float()
        dn, wn = float(d.norm()), float(w.norm())
        if dn == 0:
            continue
        row = dict(name=name, component=_component(name), rel_norm=r(dn / (wn + 1e-12), 6), delta_norm=r(dn, 5),
                   shape=list(w.shape))
        mlayer = re.search(r"\.(\d+)\.", name)
        if mlayer and row["component"] in per_layer and int(mlayer.group(1)) < L:
            li = int(mlayer.group(1))
            per_layer[row["component"]][li] += dn ** 2
            sq[row["component"]][li] += wn ** 2
        if d.dim() == 2 and min(d.shape) > 1:
            s = torch.linalg.svdvals(d)
            row.update(stable_rank=r(float((s ** 2).sum() / (s[0] ** 2 + 1e-20)), 2),
                       top1_energy=r(float(s[0] ** 2 / ((s ** 2).sum() + 1e-20)), 4),
                       rank_90=int(((s ** 2).cumsum(0) / (s ** 2).sum() < 0.9).sum()) + 1)
        rows.append(row)
    rows.sort(key=lambda x: -x["rel_norm"])
    layers = {c: [r((per_layer[c][l] ** 0.5) / (sq[c][l] ** 0.5 + 1e-12), 6) for l in range(L)] for c in per_layer}
    return dict(n_changed=len(rows), n_total=len(pb), top=rows[:top], per_layer=layers)


def _cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x - x.mean(0), y - y.mean(0)
    return float((x.T @ y).norm() ** 2 / ((x.T @ x).norm() * (y.T @ y).norm() + 1e-12))


def activation_diff(base: WhiteBoxModel, tuned: WhiteBoxModel, prompts: list, k: int = 5) -> dict:
    """Layerwise representation drift and output divergence on probe prompts."""
    xb, xt, per_prompt = [], [], []
    for p in prompts:
        ids = base.encode(p)
        cb, ct = base.forward(ids, cache=["resid_pre"]), tuned.forward(ids, cache=["resid_pre"])
        xb.append(torch.stack(cb.resid_pre + [cb.resid_final])[:, 0, 1:])      # [L+1,T-1,d] (skip BOS)
        xt.append(torch.stack(ct.resid_pre + [ct.resid_final])[:, 0, 1:])
        lb, lt = cb.logits[0, -1].log_softmax(-1), ct.logits[0, -1].log_softmax(-1)
        shift = (lt - lb)
        up, down = shift.topk(k), (-shift).topk(k)
        per_prompt.append(dict(
            prompt=p, kl=r(float((lb.exp() * (lb - lt)).sum())),
            base_top=base.tok.decode([int(lb.argmax())]), tuned_top=base.tok.decode([int(lt.argmax())]),
            base_top_prob=r(float(lb.max().exp())), tuned_top_prob=r(float(lt.max().exp())),
            promoted=[dict(token=base.tok.decode([int(i)]), dlogprob=r(float(v), 2)) for v, i in zip(up.values, up.indices)],
            suppressed=[dict(token=base.tok.decode([int(i)]), dlogprob=r(-float(v), 2)) for v, i in zip(down.values, down.indices)]))
    XB, XT = torch.cat(xb, 1), torch.cat(xt, 1)                                # [L+1,N,d]
    cos = torch.nn.functional.cosine_similarity(XB, XT, dim=-1).mean(1)
    rel = (XT - XB).norm(dim=-1).mean(1) / XB.norm(dim=-1).mean(1)
    return dict(cka=[r(_cka(XB[l], XT[l])) for l in range(XB.shape[0])], cosine=r(cos), rel_shift=r(rel),
                prompts=per_prompt, mean_kl=r(sum(p["kl"] for p in per_prompt) / max(1, len(per_prompt))))


def feature_diff(base: WhiteBoxModel, tuned: WhiteBoxModel, sae, prompts: list, k: int = 10) -> dict:
    """Read both checkpoints through one SAE basis (trained on ``base``) and
    report the features whose activation changed the most."""
    delta, n = torch.zeros(sae.n_features), 0
    best = {}
    for p in prompts:
        ids = base.encode(p)
        fb = sae.encode(base.forward(ids, cache=["resid_pre"]).resid_pre[sae.layer][0, 1:])
        ft = sae.encode(tuned.forward(ids, cache=["resid_pre"]).resid_pre[sae.layer][0, 1:])
        d = ft - fb
        delta += d.sum(0).cpu(); n += d.shape[0]
        peak = d.abs().max(0)
        toks = base.str_tokens(ids)[1:]
        for f in d.abs().max(0).values.topk(k).indices.tolist():
            if abs(float(d[peak.indices[f], f])) > abs(best.get(f, (0,))[0]):
                best[f] = (float(d[peak.indices[f], f]), toks[int(peak.indices[f])], p)
    mean = delta / max(n, 1)

    def rows(idx):
        return [dict(layer=sae.layer, index=int(i), mean_delta=r(float(mean[i])),
                     peak_delta=r(best[int(i)][0]) if int(i) in best else None,
                     peak_token=best[int(i)][1] if int(i) in best else None,
                     neuronpedia=f"{sae.neuronpedia_model}/{sae.neuronpedia_source}/{int(i)}" if sae.neuronpedia_model else None)
                for i in idx]
    return dict(layer=sae.layer, gained=rows(mean.topk(k).indices), lost=rows((-mean).topk(k).indices))


def model_diff(base: WhiteBoxModel, tuned: WhiteBoxModel, prompts: list, sae=None, neuronpedia=None) -> dict:
    out = dict(base=base.name, tuned=tuned.name, weights=weight_diff(base, tuned),
               activations=activation_diff(base, tuned, prompts))
    if sae is not None:
        out["features"] = feature_diff(base, tuned, sae, prompts)
        if neuronpedia is not None:
            neuronpedia.annotate(out["features"]["gained"]); neuronpedia.annotate(out["features"]["lost"])
    return out


# ------------------------------------------------------------- trace diff
def trace_diff(a: list, b: list, key=lambda s: s.get("tool", s.get("type", "?"))) -> dict:
    """Align two step traces (e.g. agent tool-call logs) and report divergence.

    Each trace is a list of dict steps. Steps are aligned on ``key`` with a
    longest-common-subsequence match; aligned steps whose inputs differ are
    reported as 'changed'.
    """
    ka, kb = [key(s) for s in a], [key(s) for s in b]
    sm = difflib.SequenceMatcher(a=ka, b=kb, autojunk=False)
    ops, first = [], None
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                same = a[i].get("input") == b[j].get("input")
                ops.append(dict(op="same" if same else "changed", a=i, b=j, step=ka[i],
                                **({} if same else dict(a_input=a[i].get("input"), b_input=b[j].get("input")))))
                if not same and first is None:
                    first = len(ops) - 1
        else:
            if first is None:
                first = len(ops)
            ops += [dict(op="only_a", a=i, step=ka[i]) for i in range(i1, i2)]
            ops += [dict(op="only_b", b=j, step=kb[j]) for j in range(j1, j2)]
    return dict(similarity=r(sm.ratio()), first_divergence=first, ops=ops,
                counts={o: sum(1 for x in ops if x["op"] == o) for o in ("same", "changed", "only_a", "only_b")})
