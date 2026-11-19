"""Attribution and causal methods for open-weights models.

Everything is expressed against one scalar *metric*:

    metric = logit[target] - logit[contrast]      (at the final position)

so that input attribution, component attribution, patching, SAE features and
interventions are directly comparable ("hand in hand") for the same prompt.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .backend import Hook, WhiteBoxModel


@dataclass
class Task:
    ids: torch.Tensor              # [1,T]
    target: int
    contrast: Optional[int] = None

    def metric(self, logits: torch.Tensor) -> torch.Tensor:
        """logits [B,T,V] -> [B]"""
        last = logits[:, -1]
        m = last[:, self.target]
        return m - last[:, self.contrast] if self.contrast is not None else m


def make_task(m: WhiteBoxModel, prompt: str, target: Optional[str] = None, contrast: Optional[str] = None) -> Task:
    ids = m.encode(prompt)
    if target is None:
        t = int(m.forward(ids).logits[0, -1].argmax())
    else:
        t = m.token_id(target)
    return Task(ids, t, m.token_id(contrast) if contrast else None)


def r(x, nd=4):
    """Round tensors / floats into plain JSON."""
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().tolist()
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, (list, tuple)):
        return [r(v, nd) for v in x]
    return x


# ----------------------------------------------------------------- outputs
def predictions(m: WhiteBoxModel, task: Task, k: int = 10) -> dict:
    logits = m.forward(task.ids).logits[0, -1]
    p = logits.softmax(-1)
    top = p.topk(k)
    return dict(
        top=[dict(token=m.tok.decode([int(i)]), id=int(i), prob=r(float(v)), logit=r(float(logits[i])))
             for v, i in zip(top.values, top.indices)],
        target=dict(token=m.tok.decode([task.target]), id=task.target, prob=r(float(p[task.target])),
                    rank=int((logits > logits[task.target]).sum())),
        contrast=None if task.contrast is None else dict(token=m.tok.decode([task.contrast]), id=task.contrast,
                                                         prob=r(float(p[task.contrast]))),
        metric=r(float(task.metric(logits[None, None]))),
        entropy=r(float(-(p * p.clamp_min(1e-12).log()).sum())),
    )


# -------------------------------------------------------------- logit lens
def logit_lens(m: WhiteBoxModel, task: Task) -> dict:
    """Decode every layer's residual stream with the final norm + unembedding."""
    c = m.forward(task.ids, cache=["resid_pre"])
    states = torch.cat([torch.stack(c.resid_pre, 0)[:, 0], c.resid_final], 0)      # [L+1,T,d]
    logits = m.lens(states)                                                        # [L+1,T,V]
    grid = logits[..., task.target] - (logits[..., task.contrast] if task.contrast is not None else 0)
    last = logits[:, -1]
    probs = last.softmax(-1)
    return dict(
        grid=r(grid, 3),                                                           # metric per layer x position
        target_prob=r(probs[:, task.target]),
        target_rank=[int((row > row[task.target]).sum()) for row in last],
        top_token=[m.tok.decode([int(i)]) for i in last.argmax(-1)],
        top_token_grid=[[m.tok.decode([int(i)]) for i in row] for row in logits.argmax(-1)],
    )


# ------------------------------------------------- direct logit attribution
def direct_logit_attribution(m: WhiteBoxModel, task: Task, top_neurons: int = 25) -> dict:
    """Split the metric over embeddings, attention heads, MLP layers and neurons."""
    c = m.forward(task.ids, cache=["resid_pre", "attn_out", "mlp_out", "z", "neurons"])
    final = c.resid_final[0, -1]
    d, exact = m.unembed_dir(final, task.target, task.contrast)
    attn = torch.stack([a[0, -1] @ d for a in c.attn_out])
    mlp = torch.stack([x[0, -1] @ d for x in c.mlp_out])
    embed = c.resid_pre[0][0, -1] @ d
    H = m.n_heads
    heads = torch.stack([(c.z[l][0, -1] * (m.attn_proj_matrix(l) @ d)).view(H, -1).sum(-1) for l in range(m.n_layers)])
    neuron = torch.stack([c.neurons[l][0, -1] * (m.mlp_proj_matrix(l) @ d) for l in range(m.n_layers)])   # [L,d_mlp]
    top = neuron.abs().flatten().topk(top_neurons).indices
    return dict(
        embed=r(float(embed)), attn=r(attn), mlp=r(mlp), heads=r(heads, 3),
        neurons=[dict(layer=int(i // m.d_mlp), index=int(i % m.d_mlp), dla=r(float(neuron.flatten()[i])),
                      act=r(float(c.neurons[int(i // m.d_mlp)][0, -1, int(i % m.d_mlp)]))) for i in top],
        total=r(float(final @ d)), sum_components=r(float(embed + attn.sum() + mlp.sum())), exact_norm=exact,
    )


# -------------------------------------------------------- input attribution
def input_attribution(m: WhiteBoxModel, task: Task, steps: int = 32) -> dict:
    """Gradient×input and integrated gradients over the input embeddings."""
    emb = m.embed(task.ids).detach()
    e = emb.clone().requires_grad_(True)
    val = task.metric(m.forward(embeds=e, grad=True).logits)
    g, = torch.autograd.grad(val.sum(), e)
    gxi = (g * emb).sum(-1)[0]
    total = torch.zeros_like(emb)
    alphas = (torch.arange(steps, device=emb.device) + 0.5) / steps          # midpoint Riemann rule
    for chunk in alphas.split(16):
        x = (chunk.view(-1, 1, 1) * emb).requires_grad_(True)
        gr, = torch.autograd.grad(task.metric(m.forward(embeds=x, grad=True).logits).sum(), x)
        total += gr.sum(0, keepdim=True)
    ig = (emb * total / steps).sum(-1)[0]
    base = float(task.metric(m.forward(embeds=torch.zeros_like(emb)).logits))
    return dict(grad_x_input=r(gxi), integrated_gradients=r(ig), metric=r(float(val.detach())), baseline_metric=r(base),
                ig_completeness_error=r(float(ig.sum()) - (float(val.detach()) - base)))


# ------------------------------------------------------- activation patching
def activation_patching(m: WhiteBoxModel, task: Task, corrupt_ids: torch.Tensor, batch: int = 64) -> dict:
    """Causal tracing: restore clean residual states into the corrupted run.

    grid[layer][pos] = (metric(patched) - metric(corrupt)) / (metric(clean) - metric(corrupt)).
    Also returns the gradient-based approximation (attribution patching) for
    the residual stream and for every attention head.
    """
    if corrupt_ids.shape != task.ids.shape:
        raise ValueError(f"clean and corrupted prompts must tokenize to the same length "
                         f"({task.ids.shape[1]} vs {corrupt_ids.shape[1]})")
    L, T = m.n_layers, task.ids.shape[1]
    clean = m.forward(task.ids, cache=["resid_pre", "z"])
    cor = m.forward(corrupt_ids, cache=["resid_pre", "z"], grad=True)
    m_clean, m_cor = float(task.metric(clean.logits)), float(task.metric(cor.logits).detach())
    denom = (m_clean - m_cor) or 1e-9
    grads = torch.autograd.grad(task.metric(cor.logits).sum(), cor.resid_pre + cor.z)
    attr_resid = torch.stack([((clean.resid_pre[l] - cor.resid_pre[l].detach()) * grads[l]).sum(-1)[0] for l in range(L)]) / denom
    attr_heads = torch.stack([((clean.z[l] - cor.z[l].detach()) * grads[L + l])[0].sum(0).view(m.n_heads, -1).sum(-1)
                              for l in range(L)]) / denom

    cells = [(l, p) for l in range(L) for p in range(T)]
    grid = torch.zeros(L, T)
    for start in range(0, len(cells), batch):
        chunk = cells[start:start + batch]

        def mk(layer):
            rows = [(i, p) for i, (l, p) in enumerate(chunk) if l == layer]

            def fn(t):
                t = t.clone()
                for i, p in rows:
                    t[i, p] = clean.resid_pre[layer][0, p]
                return t
            return fn
        hooks = [Hook("resid_pre", l, mk(l)) for l in {l for l, _ in chunk}]
        vals = task.metric(m.forward(corrupt_ids.expand(len(chunk), -1), hooks=hooks).logits)
        for (l, p), v in zip(chunk, vals):
            grid[l, p] = (float(v) - m_cor) / denom
    return dict(resid=r(grid, 3), attribution_resid=r(attr_resid, 3), attribution_heads=r(attr_heads, 3),
                clean_metric=r(m_clean), corrupt_metric=r(m_cor), corrupt_tokens=m.str_tokens(corrupt_ids),
                approx_correlation=r(float(torch.corrcoef(torch.stack([grid.flatten(), attr_resid.flatten().cpu()]))[0, 1])))


# ------------------------------------------------------------- interventions
def head_hook(m: WhiteBoxModel, layer: int, head: int, scale: float = 0.0, pos: Optional[int] = None) -> Hook:
    dh, sl = m.d_z // m.n_heads, slice(None) if pos is None else slice(pos, pos + 1)

    def fn(t):
        t = t.clone(); t[:, sl, head * dh:(head + 1) * dh] *= scale; return t
    return Hook("z", layer, fn)


def neuron_hook(layer: int, index: int, scale: float = 0.0, pos: Optional[int] = None) -> Hook:
    sl = slice(None) if pos is None else slice(pos, pos + 1)

    def fn(t):
        t = t.clone(); t[:, sl, index] *= scale; return t
    return Hook("neurons", layer, fn)


def direction_hook(layer: int, direction: torch.Tensor, add: float = 0.0, pos: Optional[int] = None) -> Hook:
    """add != 0: steer by ``add * direction``; add == 0: project the direction out."""
    sl = slice(None) if pos is None else slice(pos, pos + 1)

    def fn(t):
        t = t.clone()
        if add == 0:
            u = direction / direction.norm()
            t[:, sl] -= (t[:, sl] @ u).unsqueeze(-1) * u
        else:
            t[:, sl] += add * direction
        return t
    return Hook("resid_pre", layer, fn)


def intervene(m: WhiteBoxModel, task: Task, hooks: list, k: int = 5, **describe) -> dict:
    """Run the prompt with interventions and report the causal effect on the metric."""
    base = m.forward(task.ids).logits
    new = m.forward(task.ids, hooks=hooks).logits
    pb, pn = base[0, -1].softmax(-1), new[0, -1].softmax(-1)
    top = pn.topk(k)
    return dict(**describe, metric_before=r(float(task.metric(base))), metric_after=r(float(task.metric(new))),
                target_prob_before=r(float(pb[task.target])), target_prob_after=r(float(pn[task.target])),
                kl=r(float((pb * (pb.clamp_min(1e-12).log() - pn.clamp_min(1e-12).log())).sum())),
                top_after=[dict(token=m.tok.decode([int(i)]), prob=r(float(v))) for v, i in zip(top.values, top.indices)])
