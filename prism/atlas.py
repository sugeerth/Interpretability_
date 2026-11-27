"""Model atlas: prompt-independent maps of what each head does.

Head roles are measured the way TransformerLens' head detectors do, on
sequences of random tokens repeated twice (so nothing but the repetition can
explain the attention pattern):

  previous-token  attention from position i to i-1
  duplicate-token attention from the second copy of a token to its first copy
  induction       attention from the second copy to the token *after* the first copy

plus a behavioural induction score: how much better the model predicts the
second half than the first.
"""
from __future__ import annotations

import torch

from .backend import WhiteBoxModel
from .methods import r


def head_roles(m: WhiteBoxModel, seq_len: int = 24, n_seqs: int = 8, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    V = m.embed.weight.shape[0]
    bos = m.tok.bos_token_id if getattr(m.tok, "bos_token_id", None) is not None else 0
    half = torch.randint(min(1000, V // 4), V, (n_seqs, seq_len), generator=g)
    ids = torch.cat([torch.full((n_seqs, 1), bos), half, half], 1).to(m.device)
    c = m.forward(ids, cache=["attn"])
    T = ids.shape[1]
    second = torch.arange(seq_len + 1, T)
    scores = dict(previous_token=[], duplicate_token=[], induction=[])
    for a in c.attn:                                                    # [B,H,T,T]
        scores["previous_token"].append(a[:, :, torch.arange(1, T), torch.arange(0, T - 1)].mean((0, 2)))
        scores["duplicate_token"].append(a[:, :, second, second - seq_len].mean((0, 2)))
        scores["induction"].append(a[:, :, second, second - seq_len + 1].mean((0, 2)))
    lp = c.logits.log_softmax(-1)
    nll = -lp[:, :-1].gather(-1, ids[:, 1:, None])[..., 0]              # [B,T-1]
    out = {k: r(torch.stack(v), 3) for k, v in scores.items()}
    roles = []
    for name, grid in out.items():
        flat = sorted(((v, l, h) for l, row in enumerate(grid) for h, v in enumerate(row)), reverse=True)
        roles += [dict(role=name, layer=l, head=h, score=v) for v, l, h in flat[:5] if v > 0.3]
    return dict(scores=out, top=roles, seq_len=seq_len, n_seqs=n_seqs,
                loss_first_half=r(float(nll[:, :seq_len].mean())), loss_second_half=r(float(nll[:, seq_len:].mean())))


def atlas(m: WhiteBoxModel) -> dict:
    return dict(kind="atlas", model=m.info(), roles=head_roles(m))
