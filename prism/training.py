"""Training-phase tracing: fine-tune (or re-train) a model while snapshotting
it, then diff every phase against the base checkpoint.

This is the "interpretability across training" loop: instead of one frozen
model you get a timeline — which layers moved in which phase, when behaviour
flipped, and which labelled features carried the change.
"""
from __future__ import annotations

import copy
from typing import Callable, Optional

import torch

from .backend import WhiteBoxModel
from .diff import model_diff
from .methods import r


def trace_finetune(base: WhiteBoxModel, texts: list, probes: list, phases: int = 3, steps_per_phase: int = 8,
                   lr: float = 1e-4, train_filter: Optional[Callable[[str], bool]] = None, sae=None,
                   neuronpedia=None, on_phase: Optional[Callable] = None, seed: int = 0) -> dict:
    """Fine-tune a copy of ``base`` on ``texts`` and diff after every phase.

    ``train_filter(param_name)`` restricts which parameters are trainable
    (default: all). Returns {"phases": [model_diff ...], "loss": [...]}; the
    base model is left untouched.
    """
    torch.manual_seed(seed)
    tuned = WhiteBoxModel(copy.deepcopy(base.model), base.tok, name=f"{base.name}+ft", device=base.device)
    params = [p for n, p in tuned.model.named_parameters() if train_filter is None or train_filter(n)]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    batch = [base.encode(t) for t in texts]
    losses, out = [], []
    for phase in range(phases):
        tuned.model.train()
        for step in range(steps_per_phase):
            ids = batch[(phase * steps_per_phase + step) % len(batch)]
            loss = tuned.model(input_ids=ids, labels=ids).loss
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(r(float(loss.detach())))
        tuned.model.eval()
        tuned.name = f"{base.name}+ft@{(phase + 1) * steps_per_phase}"
        d = model_diff(base, tuned, probes, sae=sae, neuronpedia=neuronpedia)
        d.update(phase=phase + 1, step=(phase + 1) * steps_per_phase, loss=losses[-1])
        out.append(d)
        if on_phase:
            on_phase(d)
    for p in params:
        p.requires_grad_(False)
    return dict(base=base.name, phases=out, loss=losses, texts=texts, probes=probes, tuned_model=tuned)
