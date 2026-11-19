"""Sparse-autoencoder features: local encoding, attribution to the metric, and
a gradient attribution graph  input tokens -> features -> features -> output.

Feature indices line up with Neuronpedia (``model_id`` / ``source`` / index),
so every locally computed feature can be labelled with its public explanation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import torch

from .backend import Hook, WhiteBoxModel
from .methods import Task, r

# Open SAE releases that share feature indices with Neuronpedia.
# hf model name -> release description. ``path`` is formatted with the layer.
REGISTRY = {
    # ``layers`` are Prism layers: the SAE reads resid_pre[layer]. ``offset`` converts a Prism layer to the
    # release's own numbering (releases trained on resid_post of layer n read resid_pre[n + 1]).
    "gpt2": dict(repo="jbloom/GPT2-Small-SAEs-Reformatted", path="blocks.{n}.hook_resid_pre",
                 neuronpedia_model="gpt2-small", neuronpedia_source="{n}-res-jb", layers=list(range(12)),
                 center=True),   # trained on TransformerLens activations (center_writing_weights)
    "EleutherAI/pythia-70m-deduped": dict(repo="ctigges/pythia-70m-deduped__res-sm_processed", path="{n}-res-sm",
                                          neuronpedia_model="pythia-70m-deduped", neuronpedia_source="{n}-res-sm",
                                          layers=[1, 2, 3, 4, 5], offset=-1),
}


@dataclass
class SAE:
    W_enc: torch.Tensor      # [d,F]
    b_enc: torch.Tensor      # [F]
    W_dec: torch.Tensor      # [F,d]
    b_dec: torch.Tensor      # [d]
    layer: int               # reads resid_pre[layer]
    neuronpedia_model: Optional[str] = None
    neuronpedia_source: Optional[str] = None
    center: bool = False     # SAE was trained on mean-centred residuals (TransformerLens weight processing)

    def prep(self, x: torch.Tensor) -> torch.Tensor:
        """Map a raw residual into the space the SAE was trained on."""
        return x - x.mean(-1, keepdim=True) if self.center else x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu((self.prep(x) - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    @property
    def n_features(self) -> int:
        return self.W_dec.shape[0]


def load_sae(model_name: str, layer: int, device: str = "cpu") -> SAE:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    reg = REGISTRY[model_name]
    if layer not in reg["layers"]:
        raise ValueError(f"no SAE for {model_name} layer {layer}; available: {reg['layers']}")
    n = layer + reg.get("offset", 0)
    sub = reg["path"].format(n=n)
    w = load_file(hf_hub_download(reg["repo"], f"{sub}/sae_weights.safetensors"), device=device)
    return SAE(w["W_enc"], w["b_enc"], w["W_dec"], w["b_dec"], layer, reg.get("neuronpedia_model"),
               reg["neuronpedia_source"].format(n=n), reg.get("center", False))


def feature_hook(sae: SAE, index: int, scale: float = 0.0, add: float = 0.0, pos: Optional[int] = None) -> Hook:
    """Rescale a feature's own activation (scale=0 ablates it) and/or add a
    fixed amount of its decoder direction (steering). The SAE error term is
    left untouched, so only this feature changes."""
    sl = slice(None) if pos is None else slice(pos, pos + 1)

    def fn(t):
        t = t.clone()
        act = torch.relu((sae.prep(t[:, sl]) - sae.b_dec) @ sae.W_enc[:, index] + sae.b_enc[index])
        t[:, sl] += ((scale - 1) * act + add).unsqueeze(-1) * sae.W_dec[index]
        return t
    return Hook("resid_pre", sae.layer, fn)


def _ig_feature_grads(m, task, sae, x, acts, keep, steps: int) -> torch.Tensor:
    """Mean ∂metric/∂resid along the path that scales every feature of this
    SAE from 0 to its real activation (error term and bias held fixed)."""
    recon = (acts @ sae.W_dec) * keep                              # [T,d] feature-explained part
    alphas = (torch.arange(steps, device=x.device) + 0.5) / steps
    holder = {}

    def fn(t):
        xa = (x - (1 - alphas.view(-1, 1, 1)) * recon).requires_grad_(True)
        holder["x"] = xa
        return xa
    logits = m.forward(task.ids.expand(steps, -1), hooks=[Hook("resid_pre", sae.layer, fn)], grad=True).logits
    g, = torch.autograd.grad(task.metric(logits).sum(), holder["x"])
    return g.mean(0)


def feature_attribution(m: WhiteBoxModel, task: Task, saes: list, k: int = 12, graph_k: int = 6, ig_steps: int = 16, graph: bool = True,
                        candidates: int = 24, max_recon_err: float = 0.5) -> dict:
    """Top SAE features by attribution to the metric, plus an attribution graph.

    Each feature (layer, position, index) carries three numbers:
      effect            exact: metric − metric(feature ablated). Ranking key.
      attribution       act · (ḡ · W_dec[f]), ḡ averaged along an integrated-
                        gradients path from "all features off" to the real
                        activations (as in sparse feature circuits)
      grad_attribution  the same with the plain local gradient
    Gradient scores are only used to pick ``candidates`` for exact ablation:
    attention saturates, so they can disagree with the true effect — both are
    kept so the disagreement is visible. Positions where the SAE reconstructs
    the residual worse than ``max_recon_err`` (and BOS) are masked: features
    read there are not meaningful. The SAE error term is attributed too, so
    the unexplained share is reported, not hidden.

    Graph edges are gradient × activation:
      token  -> feature   ∂act_B/∂embed_t · embed_t          (lowest SAE layer)
      feature-> feature   act_A · (∂act_B/∂resid_A · W_dec[A]) (adjacent SAE layers)
      feature-> output    exact ablation effect above
    Every edge weight is dimensionless: a fraction of the destination's
    activation (or of |metric| for edges into the output), so edges are
    comparable across the graph. These are total (not direct) effects through
    the real model.
    """
    saes = sorted(saes, key=lambda s: s.layer)
    emb = m.embed(task.ids).detach().requires_grad_(True)
    c = m.forward(embeds=emb, cache=["resid_pre"], grad=True)
    metric = task.metric(c.logits).sum()
    resids = [c.resid_pre[s.layer] for s in saes]
    grads = torch.autograd.grad(metric, resids, retain_graph=True)
    toks = m.str_tokens(task.ids)
    # SAEs are not trained on the BOS position (its residual norm is ~50x larger); mask it out.
    keep = (task.ids[0] != (m.tok.bos_token_id if m.tok.bos_token_id is not None else -1)).float().unsqueeze(-1)
    layers, nodes = [], []
    acts_all = []
    base_metric = float(metric.detach())
    for s, x, g in zip(saes, resids, grads):
        xd = s.prep(x[0].detach())
        raw = s.encode(xd)
        frac = ((xd - s.decode(raw)) ** 2).sum(-1) / (xd ** 2).sum(-1)       # per-position reconstruction error
        ok = keep * (frac < max_recon_err).float().unsqueeze(-1)            # drop positions the SAE cannot represent
        acts = s.encode(x[0]) * ok                              # [T,F] (in graph)
        acts_all.append(acts)
        a = acts.detach()
        g_ig = _ig_feature_grads(m, task, s, x.detach(), a, ok, ig_steps)
        attr = a * (g_ig @ s.W_dec.T)                           # [T,F] integrated gradients
        attr_grad = a * (g[0] @ s.W_dec.T)                      # [T,F] plain gradient
        resid_err = (xd - s.decode(a)) * ok                     # [T,d]
        err = (resid_err * g_ig).sum(-1)                        # [T]
        cand = torch.unique(torch.cat([attr.abs().flatten().topk(candidates).indices,
                                       attr_grad.abs().flatten().topk(candidates).indices]))
        cells = [(int(i // s.n_features), int(i % s.n_features)) for i in cand]

        def ablate(t, cells=cells, s=s):                        # row i of the batch ablates candidate i
            t = t.clone()
            for i, (p, f) in enumerate(cells):
                act = torch.relu((s.prep(t[i, p]) - s.b_dec) @ s.W_enc[:, f] + s.b_enc[f])
                t[i, p] -= act * s.W_dec[f]
            return t
        abl = task.metric(m.forward(task.ids.expand(len(cells), -1), hooks=[Hook("resid_pre", s.layer, ablate)]).logits)
        feats = []
        for (p, f), v in zip(cells, abl):
            feats.append(dict(layer=s.layer, pos=p, token=toks[p], index=f, act=r(float(a[p, f])),
                              effect=r(base_metric - float(v)), attribution=r(float(attr[p, f])),
                              grad_attribution=r(float(attr_grad[p, f])),
                              neuronpedia=f"{s.neuronpedia_model}/{s.neuronpedia_source}/{f}" if s.neuronpedia_model else None))
        feats = sorted(feats, key=lambda f: -abs(f["effect"]))[:k]
        layers.append(dict(layer=s.layer, features=feats, n_active_last=int((a[-1] > 0).sum()),
                           explained_attribution=r(float(attr.sum())), error_attribution=r(float(err.sum())),
                           recon_err_by_pos=r(frac, 3),
                           masked_positions=[t for t in range(len(toks)) if ok[t, 0] == 0]))
        nodes.append(feats[:graph_k])

    edges = []
    for li, (s, feats) in enumerate(zip(saes, nodes) if graph else ()):
        for fb in feats:
            target_act = acts_all[li][fb["pos"], fb["index"]]
            wrt = [emb] + ([resids[li - 1]] if li > 0 else [])
            gs = torch.autograd.grad(target_act, wrt, retain_graph=True, allow_unused=True)
            dst = f"f{s.layer}.{fb['pos']}.{fb['index']}"
            denom = fb["act"] or 1.0                     # edges are fractions of the destination's activation
            if li == 0 and gs[0] is not None:
                tok_attr = (gs[0] * emb.detach()).sum(-1)[0]
                for t, v in enumerate(tok_attr):
                    edges.append(dict(src=f"t{t}", dst=dst, weight=r(float(v) / denom)))
            if li > 0 and gs[1] is not None:
                prev = saes[li - 1]
                for fa in nodes[li - 1]:
                    v = fa["act"] * float(gs[1][0, fa["pos"]] @ prev.W_dec[fa["index"]])
                    edges.append(dict(src=f"f{prev.layer}.{fa['pos']}.{fa['index']}", dst=dst, weight=r(v / denom)))
            edges.append(dict(src=dst, dst="out", weight=r(fb["effect"] / (abs(base_metric) or 1.0))))
    return dict(layers=layers, graph=dict(
        nodes=[dict(id=f"t{t}", kind="token", label=tok, pos=t) for t, tok in enumerate(toks)]
        + [dict(id=f"f{f['layer']}.{f['pos']}.{f['index']}", kind="feature", **f) for feats in nodes for f in feats]
        + [dict(id="out", kind="output", label=m.tok.decode([task.target]))],
        edges=[e for e in edges if abs(e["weight"]) > 1e-3]))
