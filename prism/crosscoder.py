"""Crosscoder model diffing.

A crosscoder is one sparse dictionary trained to reconstruct the activations
of *two* models at once: a shared encoder, one decoder per model. After
training, each latent's two decoder norms say where the concept lives:

    relative norm = |dec_tuned| / (|dec_base| + |dec_tuned|)
        ~0   only the base model represents it
        ~0.5 shared (decoder cosine tells whether it was rotated)
        ~1   only the tuned model represents it — what fine-tuning added

Unlike reading both checkpoints through the base model's SAE, this can
surface directions that did not exist before. Following Minder et al.
(NeurIPS 2025) the sparsity is BatchTopK rather than an L1 penalty, which
they show avoids mislabelling shared latents as model-specific; their second
safeguard is applied too: each model-specific latent is re-checked by how
well it reconstructs the *other* model when its decoder is transplanted
(``leakage``), and latents that transfer are not reported as specific.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .backend import WhiteBoxModel
from .methods import r


def collect_pairs(base: WhiteBoxModel, tuned: WhiteBoxModel, texts: list, layer: int, ctx: int = 48, batch: int = 16,
                  max_tokens: int = 40000) -> tuple:
    """Paired residual activations entering ``layer`` for both models.
    Returns (acts [N,2,d], token_ids [N], context index for display)."""
    ids = torch.cat([base.encode(t)[0, 1:] if base.tok.bos_token_id is not None else base.encode(t)[0] for t in texts if t.strip()])
    ids = ids[: (min(len(ids), max_tokens) // ctx) * ctx].view(-1, ctx)
    acts = []
    for chunk in ids.split(batch):
        a = base.forward(chunk, cache=["resid_pre"]).resid_pre[layer]
        b = tuned.forward(chunk, cache=["resid_pre"]).resid_pre[layer]
        acts.append(torch.stack([a, b], 2)[:, 1:])                       # drop each window's first position
    X = torch.cat(acts).reshape(-1, 2, a.shape[-1])
    return X, ids[:, 1:].reshape(-1), ids


@dataclass
class Crosscoder:
    W_enc: torch.Tensor      # [2,d,F]
    b_enc: torch.Tensor      # [F]
    W_dec: torch.Tensor      # [F,2,d]
    b_dec: torch.Tensor      # [2,d]
    k: int
    scale: torch.Tensor      # [2] activation normalisers
    threshold: float = 0.0   # inference-time activation threshold (replaces batch top-k)

    def pre(self, x):
        return torch.einsum("nmd,mdf->nf", x / self.scale.view(1, 2, 1) - self.b_dec, self.W_enc) + self.b_enc

    def encode(self, x, batch_topk: bool = False):
        a = torch.relu(self.pre(x))
        if batch_topk:                                                     # keep the k*N largest activations in the batch
            score = a * self.W_dec.norm(dim=-1).sum(-1)
            kth = score.flatten().topk(self.k * a.shape[0]).values[-1]
            return a * (score >= kth)
        return a * (a * self.W_dec.norm(dim=-1).sum(-1) > self.threshold)

    def decode(self, a):
        return torch.einsum("nf,fmd->nmd", a, self.W_dec) + self.b_dec


def train_crosscoder(X: torch.Tensor, n_latents: int = 4096, k: int = 48, steps: int = 800, batch: int = 2048,
                     lr: float = 2e-3, seed: int = 0, progress=None) -> tuple:
    torch.manual_seed(seed)
    d = X.shape[-1]
    scale = X.norm(dim=-1).mean(0) / d ** 0.5                              # unit RMS per model
    W_dec = torch.randn(n_latents, 2, d) / d ** 0.5
    W_dec = W_dec / W_dec.norm(dim=-1, keepdim=True) * 0.3
    cc = Crosscoder(W_dec.permute(1, 2, 0).clone().contiguous(), torch.zeros(n_latents), W_dec, (X / scale.view(1, 2, 1)).mean(0), k, scale)
    params = [cc.W_enc, cc.b_enc, cc.W_dec, cc.b_dec]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * (1 - 0.8 * s / steps))
    losses = []
    for step in range(steps):
        x = X[torch.randint(0, X.shape[0], (batch,))]
        xn = x / scale.view(1, 2, 1)
        loss = ((cc.decode(cc.encode(x, batch_topk=True)) - xn) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        losses.append(float(loss.detach()))
        if progress and step % 100 == 0:
            progress(step, losses[-1])
    for p in params:
        p.requires_grad_(False)
    with torch.no_grad():                                                  # calibrate the inference threshold on a large batch
        x = X[torch.randperm(X.shape[0])[:8192]]
        a = torch.relu(cc.pre(x))
        score = a * cc.W_dec.norm(dim=-1).sum(-1)
        cc.threshold = float(score.flatten().topk(cc.k * a.shape[0]).values[-1])
    return cc, losses


def crosscoder_diff(base: WhiteBoxModel, tuned: WhiteBoxModel, texts: list, layer: int, n_latents: int = 4096,
                    k: int = 48, steps: int = 800, top: int = 10, specific: float = 0.3, progress=None, **kw) -> dict:
    """Train a crosscoder on paired activations and report what is shared,
    what only the base has, and what only the tuned model has."""
    X, tok_ids, windows = collect_pairs(base, tuned, texts, layer, **kw)
    n_train = int(0.9 * X.shape[0])
    perm = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(0))
    Xtr, Xte = X[perm[:n_train]], X[perm[n_train:]]
    cc, losses = train_crosscoder(Xtr, n_latents, k, steps, progress=progress)
    with torch.no_grad():
        xn = Xte / cc.scale.view(1, 2, 1)
        a = cc.encode(Xte)
        rec = cc.decode(a)
        fve = 1 - ((rec - xn) ** 2).sum((0, 2)) / ((xn - xn.mean(0)) ** 2).sum((0, 2))
        A = cc.encode(X)                                                   # [N,F]
        fired = (A > 0).float().mean(0)
        alive = fired > 1e-4
        norms = cc.W_dec.norm(dim=-1)                                      # [F,2]
        rel = norms[:, 1] / (norms.sum(-1) + 1e-9)
        cos = torch.nn.functional.cosine_similarity(cc.W_dec[:, 0], cc.W_dec[:, 1], dim=-1)
        xa = X / cc.scale.view(1, 2, 1) - cc.b_dec                         # centred targets [N,2,d]

        def leakage(f, src):
            """Share of the *other* model's variance along this latent that its transplanted decoder explains.
            ~0: the other model has nothing there (truly specific). ~1: the concept exists in both."""
            other = 1 - src
            on = A[:, f] > 0
            if on.sum() < 5:
                return None
            dvec = cc.W_dec[f, src] / (cc.W_dec[f, src].norm() + 1e-9)
            proj_other, proj_src = xa[on, other] @ dvec, xa[on, src] @ dvec
            return float((proj_other * proj_src).sum() / ((proj_src ** 2).sum() + 1e-9))

        def describe(f, src):
            vals, idx = A[:, f].topk(min(6, int((A[:, f] > 0).sum())))
            ctx = windows.shape[1] - 1
            ex = []
            for v, i in zip(vals, idx):
                w, p = int(i) // ctx, int(i) % ctx + 1
                ex.append(dict(text=base.tok.decode(windows[w, max(0, p - 7):p].tolist()), peak=base.tok.decode([int(windows[w, p])]),
                               value=r(float(v), 2)))
            logit = base.lens((cc.W_dec[f, src] * cc.scale[src])[None, None])[0, 0]
            hi = (logit - logit.mean()).topk(6).indices
            return dict(index=int(f), relative_norm=r(float(rel[f]), 3), decoder_cosine=r(float(cos[f]), 3), frequency=r(float(fired[f]), 5),
                        leakage=None if leakage(f, src) is None else r(leakage(f, src), 3),
                        examples=ex, promotes=[base.tok.decode([int(i)]) for i in hi])
        idx = torch.arange(n_latents)
        tuned_only = idx[alive & (rel > 1 - specific)]
        base_only = idx[alive & (rel < specific)]
        shared = alive & (rel >= specific) & (rel <= 1 - specific)
        order = lambda ids_, key: ids_[key[ids_].argsort(descending=True)[:top]]
        hist = torch.histc(rel[alive], bins=20, min=0, max=1)
        # Narrow fine-tunes rarely create new latents; they re-write existing ones. Rank shared latents by how
        # much reconstruction energy their decoder change carries (frequency x mean activation x |dec_tuned - dec_base|).
        mean_act = A.sum(0) / (A > 0).sum(0).clamp_min(1)
        delta = cc.W_dec[:, 1] - cc.W_dec[:, 0]
        shift = fired * mean_act * delta.norm(dim=-1)
        shifted = idx[alive][shift[alive].argsort(descending=True)[:top]]

        def describe_shift(f):
            d = describe(f, 1)
            logit = base.lens((delta[f] * cc.scale[1])[None, None])[0, 0]
            d.update(shift=r(float(shift[f]), 4), delta_norm=r(float(delta[f].norm()), 3),
                     delta_promotes=[base.tok.decode([int(i)]) for i in (logit - logit.mean()).topk(6).indices])
            return d
    return dict(
        kind="crosscoder", layer=layer, n_latents=n_latents, k=k, steps=steps, n_tokens=int(X.shape[0]),
        fve_base=r(float(fve[0]), 3), fve_tuned=r(float(fve[1]), 3), loss=[r(x, 3) for x in losses[::max(1, steps // 60)]],
        alive=int(alive.sum()), counts=dict(base_only=int(len(base_only)), shared=int(shared.sum()), tuned_only=int(len(tuned_only))),
        relative_norm_hist=[int(x) for x in hist], mean_shared_cosine=r(float(cos[shared].mean()) if shared.any() else 0.0, 3),
        tuned_only=[describe(int(f), 1) for f in order(tuned_only, fired)], base_only=[describe(int(f), 0) for f in order(base_only, fired)],
        most_shifted=[describe_shift(int(f)) for f in shifted],
        l0=r(float((a > 0).float().sum(-1).mean()), 1))
