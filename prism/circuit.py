"""Automated circuit discovery.

Direct logit attribution says which components write the answer; head
patching says which components matter. Neither says *who talks to whom*.
This module scores every edge of the computational graph

    embeddings -> attention heads -> MLPs -> ... -> output

with edge attribution patching (Syed et al. 2023) using integrated gradients
along the corrupted-to-clean path (EAP-IG, Hanna et al. 2024):

    score(u -> v) = (out_u[clean] - out_u[corrupt]) · mean_alpha ∂metric/∂input_v

where the gradient reaches the metric only *through v*. A circuit is the set
of strongest edges that lie on a path from the embeddings to the output.

The discovered circuit is then put to the test that matters (faithfulness):
keep the circuit's components, replace every other head and MLP with its
value from the corrupted run, and measure how much of the behaviour
survives — against random component sets of the same size.

Works on any architecture the backend supports; sequential (GPT-2, Llama)
and parallel (Pythia) attention/MLP wiring is detected from the graph.
"""
from __future__ import annotations

import random
from typing import Optional

import torch

from .backend import Hook, WhiteBoxModel
from .methods import Task, r


def _names(m: WhiteBoxModel) -> list:
    return ["emb"] + [f"a{l}.{h}" for l in range(m.n_layers) for h in range(m.n_heads)] + [f"m{l}" for l in range(m.n_layers)]


def edge_attribution(m: WhiteBoxModel, task: Task, corrupt_ids: torch.Tensor, steps: int = 5) -> dict:
    """Score every edge. Returns {"scores": [U,V] tensor, "pos": [U,V] argmax position, names...}."""
    if corrupt_ids.shape != task.ids.shape:
        raise ValueError("clean and corrupted prompts must tokenize to the same length")
    L, H, dh = m.n_layers, m.n_heads, m.d_z // m.n_heads
    sites = ["resid_pre", "z", "attn_out", "mlp_out"]
    clean, cor = m.forward(task.ids, cache=sites), m.forward(corrupt_ids, cache=sites)
    m_clean, m_cor = float(task.metric(clean.logits)), float(task.metric(cor.logits))
    denom = (m_clean - m_cor) or 1e-9

    # upstream output differences in residual space: [U,T,d]
    d_up = [clean.resid_pre[0][0] - cor.resid_pre[0][0]]
    for l in range(L):
        W, dz = m.attn_proj_matrix(l), (clean.z[l] - cor.z[l])[0]
        d_up += [dz[:, h * dh:(h + 1) * dh] @ W[h * dh:(h + 1) * dh] for h in range(H)]
    d_up += [(clean.mlp_out[l] - cor.mlp_out[l])[0] for l in range(L)]
    D = torch.stack(d_up)

    # downstream input gradients, each through one node only: [V,T,d]; V = heads, mlps, out
    e_clean, e_cor = m.embed(task.ids).detach(), m.embed(corrupt_ids).detach()
    T, d = e_clean.shape[1], D.shape[-1]
    G = torch.zeros(L * H + L + 1, T, d)
    sequential = None
    for s in range(steps):
        alpha = (s + 0.5) / steps
        x = (e_cor + alpha * (e_clean - e_cor)).requires_grad_(True)
        c = m.forward(embeds=x, cache=sites, grad=True)
        metric = task.metric(c.logits).sum()
        grads = torch.autograd.grad(metric, [c.resid_final] + c.z + c.mlp_out, retain_graph=True)
        G[-1] += grads[0][0]
        gz, gm = grads[1:1 + L], grads[1 + L:]
        for l in range(L):
            for h in range(H):
                sl = slice(h * dh, (h + 1) * dh)
                g, = torch.autograd.grad((c.z[l][..., sl] * gz[l][..., sl].detach()).sum(), c.resid_pre[l], retain_graph=True)
                G[l * H + h] += g[0]
            ga, gp = torch.autograd.grad((c.mlp_out[l] * gm[l].detach()).sum(), [c.attn_out[l], c.resid_pre[l]],
                                         retain_graph=True, allow_unused=True)
            if sequential is None:
                sequential = ga is not None and float(ga.abs().sum()) > 0
            G[L * H + l] += (ga if sequential else gp)[0]          # gradient at the MLP's own residual input
    G /= steps

    per_pos = torch.einsum("utd,vtd->uvt", D, G) / denom
    scores = per_pos.sum(-1)
    # causal order: emb < layer-l heads < layer-l MLP (sequential) < layer-(l+1) heads ... < out
    up_key = torch.tensor([-1.0] + [float(l) for l in range(L) for _ in range(H)] + [l + (0.5 if sequential else 0.0) for l in range(L)])
    dn_key = torch.tensor([float(l) for l in range(L) for _ in range(H)] + [l + (0.5 if sequential else 0.0) for l in range(L)] + [float(L)])
    valid = up_key[:, None] < dn_key[None, :]
    scores = scores * valid
    return dict(scores=scores, pos=(per_pos.abs() * valid[..., None]).argmax(-1), up=_names(m), down=_names(m)[1:] + ["out"],
                clean_metric=m_clean, corrupt_metric=m_cor, sequential=bool(sequential), steps=steps)


def curve_metrics(curve: list, n_components: int) -> dict:
    """Summaries of a faithfulness curve in the style of the Mechanistic
    Interpretability Benchmark (Mueller et al. 2025): the area under
    faithfulness-vs-circuit-size (higher = the ranking finds what matters
    early) and the area between the curve and full faithfulness (lower = the
    circuits are faithful at every size). x is the share of components kept,
    integrated by the trapezoid rule from 0 to the largest size measured."""
    pts = [(0.0, 0.0)] + sorted((c["size"] / n_components, c["circuit"]) for c in curve)
    if len(pts) < 2:
        return dict(area_under=None, area_to_full=None, x_max=None)
    span = pts[-1][0] or 1.0
    au = sum((x2 - x1) * (y1 + y2) / 2 for (x1, y1), (x2, y2) in zip(pts, pts[1:])) / span
    gap = sum((x2 - x1) * (abs(1 - y1) + abs(1 - y2)) / 2 for (x1, y1), (x2, y2) in zip(pts, pts[1:])) / span
    return dict(area_under=r(au, 3), area_to_full=r(gap, 3), x_max=r(span, 3))


def stability(stack: torch.Tensor, n_edges: int, n_boot: int = 40, seed: int = 0) -> dict:
    """How much does the circuit depend on which prompts were used?

    Edge scores from gradient methods are noisy estimates (Méloux et al. 2026).
    Prompts are resampled with replacement; for each resample the top
    ``n_edges`` edges of the mean score are taken. Returns the mean pairwise
    Jaccard overlap of those edge sets and, per edge, how often it was
    selected. Needs no extra model calls."""
    n = stack.shape[0]
    g = torch.Generator().manual_seed(seed)
    k = min(n_edges, int((stack.abs().sum(0) > 0).sum()))
    sets = []
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g)
        sets.append(set(stack[idx].mean(0).abs().flatten().topk(k).indices.tolist()))
    jac = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]]
    freq = torch.zeros(stack[0].numel())
    for st in sets:
        freq[list(st)] += 1
    return dict(n_boot=n_boot, n_prompts=n, edge_jaccard=r(sum(jac) / len(jac), 3), freq=(freq / n_boot).view_as(stack[0]))


def _on_path(edges: list) -> list:
    """Keep edges that lie on some path emb -> ... -> out."""
    fwd, back = {"emb"}, {"out"}
    for _ in range(64):
        n = len(fwd) + len(back)
        fwd |= {v for u, v, _ in edges if u in fwd}
        back |= {u for u, v, _ in edges if v in back}
        if len(fwd) + len(back) == n:
            break
    return [e for e in edges if e[0] in fwd and e[1] in back]


def faithfulness(m: WhiteBoxModel, task: Task, corrupt_ids: torch.Tensor, keep: set) -> float:
    """Metric recovered when only ``keep`` (names like 'a9.9', 'm3') run on
    clean activations and every other head and MLP is replaced by its
    corrupted-run value. 1 = the kept set reproduces the full behaviour."""
    L, H, dh = m.n_layers, m.n_heads, m.d_z // m.n_heads
    cor = m.forward(corrupt_ids, cache=["z", "mlp_out"])
    m_cor = float(task.metric(cor.logits))
    m_clean = float(task.metric(m.forward(task.ids).logits))
    hooks = []
    for l in range(L):
        drop = [h for h in range(H) if f"a{l}.{h}" not in keep]
        if drop:
            def fz(t, l=l, drop=drop):
                t = t.clone()
                for h in drop:
                    t[..., h * dh:(h + 1) * dh] = cor.z[l][..., h * dh:(h + 1) * dh]
                return t
            hooks.append(Hook("z", l, fz))
        if f"m{l}" not in keep:
            hooks.append(Hook("mlp_out", l, lambda t, l=l: cor.mlp_out[l].expand_as(t)))
    val = float(task.metric(m.forward(task.ids, hooks=hooks).logits))
    return (val - m_cor) / ((m_clean - m_cor) or 1e-9)


def discover_circuit(m: WhiteBoxModel, task: Task, corrupt_ids: torch.Tensor, n_edges: int = 60, steps: int = 5, **kw) -> dict:
    """Edge attribution -> circuit -> faithfulness curve against random baselines, for one prompt pair."""
    return _build(m, [(task, corrupt_ids, edge_attribution(m, task, corrupt_ids, steps))], n_edges, **kw)


def corpus_circuit(m: WhiteBoxModel, pairs: list, n_edges: int = 60, steps: int = 5, progress=None, **kw) -> dict:
    """The same over many (task, corrupt_ids) pairs: edge scores are averaged,
    and faithfulness is the mean over prompts — a circuit for the *task*, not
    for one sentence."""
    items = []
    for i, (task, cor) in enumerate(pairs):
        if progress:
            progress(i, len(pairs))
        items.append((task, cor, edge_attribution(m, task, cor, steps)))
    return _build(m, items, n_edges, **kw)


def _build(m, items, n_edges, sizes=None, n_random: int = 5, seed: int = 0, thresholds=(0.5, 0.8, 0.9)) -> dict:
    ea = items[0][2]
    stack = torch.stack([it[2]["scores"] for it in items])
    S = stack.mean(0)
    agree = (stack.sign() == S.sign()).float().mean(0)
    single = len(items) == 1
    toks = m.str_tokens(items[0][0].ids)
    flat = S.abs().flatten().topk(min(4 * n_edges, int((S != 0).sum())))
    cand = [(ea["up"][int(i) // S.shape[1]], ea["down"][int(i) % S.shape[1]], float(S.flatten()[i])) for i in flat.indices]
    edges = _on_path(cand[:n_edges])
    k = n_edges
    while len(edges) < min(n_edges // 2, len(cand)) and k < len(cand):      # pruning removed too much: widen the candidate set
        k += n_edges // 2
        edges = _on_path(cand[:k])[:n_edges]
    ui, vi = {n: i for i, n in enumerate(ea["up"])}, {n: i for i, n in enumerate(ea["down"])}
    # node importance = total attribution flowing through it (max of in / out so sources and sinks count)
    inflow, outflow = S.abs().sum(0), S.abs().sum(1)
    comp = [n for n in ea["up"] if n != "emb"]
    imp = {n: float(max(inflow[vi[n]], outflow[ui[n]])) for n in comp}
    ranked = sorted(comp, key=lambda n: -imp[n])
    faith = lambda keep: sum(faithfulness(m, t, c, keep) for t, c, _ in items) / len(items)
    if sizes is None:                       # absolute sizes for small circuits, then fractions of the model
        sizes = sorted({2, 4, 8, 12, 16, 24, 32} | {max(2, round(f * len(comp))) for f in (0.1, 0.2, 0.3, 0.5)})
    cache = {}

    def at(k):                              # faithfulness of the k top-ranked components
        if k not in cache:
            cache[k] = faith(set(ranked[:k]))
        return cache[k]

    def smallest(th):
        """Smallest prefix of the ranking that keeps ``th`` of the behaviour (binary search, then a
        short downward scan because faithfulness is only approximately monotone in size)."""
        if at(len(comp)) < th:
            return None
        lo, hi = 1, len(comp)
        while lo < hi:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if at(mid) >= th else (mid + 1, hi)
        while lo > 1 and at(lo - 1) >= th:
            lo -= 1
        return lo
    minimal = [dict(threshold=th, size=smallest(th)) for th in thresholds]
    for x in minimal:
        if x["size"]:
            x.update(share=r(x["size"] / len(comp), 3), components=ranked[:x["size"]] if x["size"] <= 40 else None)
    rng, curve = random.Random(seed), []
    for size in [s for s in sizes if s <= len(comp)]:
        rand = [faith(set(rng.sample(comp, size))) for _ in range(n_random)]
        curve.append(dict(size=size, circuit=r(at(size), 3),
                          random_mean=r(sum(rand) / len(rand), 3), random_max=r(max(rand), 3)))
    nodes = sorted({n for e in edges for n in e[:2]}, key=lambda n: (ea["up"] + ["out"]).index(n))
    circuit_nodes = {n for n in nodes if n not in ("emb", "out")}

    stab = stability(stack, n_edges) if len(items) >= 4 else None

    def edge(u, v, w):
        e = dict(src=u, dst=v, weight=r(w, 4), sign_agreement=r(float(agree[ui[u], vi[v]]), 2))
        if stab:
            e["selection_frequency"] = r(float(stab["freq"][ui[u], vi[v]]), 2)
        if single:
            e.update(pos=int(ea["pos"][ui[u], vi[v]]), token=toks[int(ea["pos"][ui[u], vi[v]])])
        return e
    out_edges = [edge(u, v, w) for u, v, w in edges]
    stab_out = None
    if stab:
        fr = [e["selection_frequency"] for e in out_edges]
        stab_out = dict(n_boot=stab["n_boot"], edge_jaccard=stab["edge_jaccard"], stable_edges=sum(f >= 0.9 for f in fr),
                        unstable_edges=sum(f < 0.5 for f in fr), n_edges=len(fr))
    return dict(
        method=f"EAP-IG ({ea['steps']} steps)", sequential=ea["sequential"], n_components=len(comp), n_prompts=len(items),
        clean_metric=r(sum(it[2]["clean_metric"] for it in items) / len(items)),
        corrupt_metric=r(sum(it[2]["corrupt_metric"] for it in items) / len(items)),
        nodes=[dict(id=n, kind="emb" if n == "emb" else "out" if n == "out" else "head" if n[0] == "a" else "mlp",
                    layer=None if n in ("emb", "out") else int(n[1:].split(".")[0]),
                    index=int(n.split(".")[1]) if n[0] == "a" else None, importance=r(imp.get(n, 0.0), 3)) for n in nodes],
        edges=out_edges,
        faithfulness=dict(circuit_nodes=len(circuit_nodes), value=r(faith(circuit_nodes), 3), curve=curve, minimal=minimal,
                          **curve_metrics(curve, len(comp))),
        stability=stab_out,
        ranked_components=[dict(id=n, importance=r(imp[n], 3)) for n in ranked[:24]])
