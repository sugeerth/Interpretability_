"""White-box backend: any Hugging Face causal LM, instrumented through hooks.

The backend is architecture-agnostic. It locates the transformer blocks, the
attention / MLP output projections and the final norm by structure rather than
by model name, so GPT-2, Llama, Gemma, Qwen, Mistral, Phi, Pythia ... all go
through the same code path. Every method in ``prism.methods`` is written
against the small surface exposed here:

* ``forward``      one pass, returns a ``Cache`` of intermediate activations
* ``Hook``         an intervention at a named site (patch / ablate / steer)
* ``unembed_dir``  the frozen-norm linear read-out direction for a logit metric
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import torch
from torch import nn

SITES = ("resid_pre", "attn_out", "mlp_out", "z", "neurons")


@dataclass
class Hook:
    """Intervention: ``fn(tensor) -> tensor`` applied at ``site`` of ``layer``.

    Sites: resid_pre [B,T,d] · attn_out [B,T,d] · mlp_out [B,T,d] ·
    z [B,T,H*d_head] (per-head outputs before the attention out-projection) ·
    neurons [B,T,d_mlp] (post-nonlinearity MLP neurons).
    """
    site: str
    layer: int
    fn: Callable[[torch.Tensor], torch.Tensor]


@dataclass
class Cache:
    logits: torch.Tensor                      # [B,T,V]
    resid_pre: list = field(default_factory=list)   # L x [B,T,d]
    resid_final: Optional[torch.Tensor] = None      # [B,T,d] (input of final norm)
    attn_out: list = field(default_factory=list)
    mlp_out: list = field(default_factory=list)
    z: list = field(default_factory=list)
    neurons: list = field(default_factory=list)
    embeds: Optional[torch.Tensor] = None
    attn: Optional[list] = None                     # L x [B,H,T,T] attention patterns (cache=['attn'])


def _first(obj, names):
    for n in names:
        cur = obj
        try:
            for part in n.split("."):
                cur = getattr(cur, part)
            if cur is not None:
                return cur
        except AttributeError:
            continue
    return None


def _in_out_matrix(mod) -> torch.Tensor:
    """Weight of a projection as [d_in, d_out] for nn.Linear and HF Conv1D."""
    w = mod.weight
    return w if type(mod).__name__ == "Conv1D" else w.T


class WhiteBoxModel:
    """An open-weights causal LM with named hook sites."""

    def __init__(self, model, tokenizer, name: str = "model", device: str = "cpu"):
        if device == "cpu":
            model = model.float()                # checkpoints stored in half precision do not run on CPU
        self.model, self.tok, self.name, self.device = model.to(device).eval(), tokenizer, name, device
        for p in self.model.parameters():
            p.requires_grad_(False)
        base = _first(model, ["transformer", "model", "gpt_neox", "base_model"]) or model
        self.blocks = _first(base, ["h", "layers", "decoder.layers"])
        if self.blocks is None:
            raise ValueError(f"could not locate transformer blocks in {type(model).__name__}")
        self.final_norm = _first(base, ["ln_f", "norm", "final_layer_norm", "final_norm"])
        self.unembed = model.get_output_embeddings()
        self.embed = model.get_input_embeddings()
        b0 = self.blocks[0]
        self._attn = lambda b: _first(b, ["attn", "self_attn", "attention"])
        self._mlp = lambda b: _first(b, ["mlp", "feed_forward", "ffn"])
        self._attn_proj = lambda b: _first(self._attn(b), ["c_proj", "o_proj", "out_proj", "dense"])
        self._mlp_proj = lambda b: _first(self._mlp(b), ["c_proj", "down_proj", "fc2", "dense_4h_to_h"])
        cfg = model.config
        self.n_layers = len(self.blocks)
        self.n_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head")
        self.d_model = self.embed.weight.shape[1]
        self.d_mlp = _in_out_matrix(self._mlp_proj(b0)).shape[0]
        self.d_z = _in_out_matrix(self._attn_proj(b0)).shape[0]

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, name: str = "gpt2", device: Optional[str] = None, **kw) -> "WhiteBoxModel":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:                                   # eager attention exposes attention patterns
            model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager", **kw)
        except (TypeError, ValueError):
            model = AutoModelForCausalLM.from_pretrained(name, **kw)
        return cls(model, AutoTokenizer.from_pretrained(name), name=name, device=device)

    def info(self) -> dict:
        return dict(name=self.name, n_layers=self.n_layers, n_heads=self.n_heads, d_model=self.d_model,
                    d_mlp=self.d_mlp, n_params=sum(p.numel() for p in self.model.parameters()),
                    arch=type(self.model).__name__, access="open-weights")

    # ---------------------------------------------------------------- tokens
    def encode(self, text: str) -> torch.Tensor:
        ids = self.tok(text, return_tensors="pt").input_ids
        bos = self.tok.bos_token_id
        if bos is not None and (ids.shape[1] == 0 or ids[0, 0] != bos) and type(self.model).__name__.startswith("GPT2"):
            ids = torch.cat([torch.tensor([[bos]]), ids], 1)   # GPT-2 behaves better with a BOS anchor
        return ids.to(self.device)

    def str_tokens(self, ids) -> list:
        return [self.tok.decode([int(i)]) for i in (ids[0] if ids.dim() == 2 else ids)]

    def token_id(self, s: str) -> int:
        ids = self.tok(s, add_special_tokens=False).input_ids
        if not ids:
            raise ValueError(f"empty token string {s!r}")
        return ids[0]

    # --------------------------------------------------------------- forward
    def forward(self, ids=None, embeds=None, hooks: Iterable[Hook] = (), cache: Iterable[str] = (),
                grad: bool = False) -> Cache:
        """Run the model. ``cache`` selects which sites to record; with
        ``grad=True`` the recorded tensors stay in the autograd graph."""
        want, by = set(cache), {}
        for h in hooks:
            by.setdefault((h.site, h.layer), []).append(h)
        out = Cache(logits=None)
        handles = []

        def edit(site, layer, t):
            for h in by.get((site, layer), ()):
                t = h.fn(t)
            return t

        def keep(lst, t):
            lst.append(t if grad else t.detach())

        def pre_block(layer):
            def f(mod, args, kwargs):
                key = None if args else next(k for k in ("hidden_states", "x") if k in kwargs)
                t = edit("resid_pre", layer, args[0] if args else kwargs[key])
                if "resid_pre" in want:
                    keep(out.resid_pre, t)
                if args:
                    return (t,) + tuple(args[1:]), kwargs
                kwargs[key] = t
                return args, kwargs
            return f

        def pre_proj(site, layer, store):
            def f(mod, args):
                t = edit(site, layer, args[0])
                if site in want:
                    keep(store, t)
                return (t,) + tuple(args[1:])
            return f

        def post(site, layer, store):
            def f(mod, args, res):
                is_tuple = isinstance(res, tuple)
                t = edit(site, layer, res[0] if is_tuple else res)
                if site in want:
                    keep(store, t)
                return (t,) + tuple(res[1:]) if is_tuple else t
            return f

        def pre_final(mod, args):
            if "resid_final" in want or "resid_pre" in want:
                out.resid_final = args[0] if grad else args[0].detach()

        sites_used = want | {s for s, _ in by}
        for l, b in enumerate(self.blocks):
            if "resid_pre" in sites_used:
                handles.append(b.register_forward_pre_hook(pre_block(l), with_kwargs=True))
            if "z" in sites_used:
                handles.append(self._attn_proj(b).register_forward_pre_hook(pre_proj("z", l, out.z)))
            if "neurons" in sites_used:
                handles.append(self._mlp_proj(b).register_forward_pre_hook(pre_proj("neurons", l, out.neurons)))
            if "attn_out" in sites_used:
                handles.append(self._attn(b).register_forward_hook(post("attn_out", l, out.attn_out)))
            if "mlp_out" in sites_used:
                handles.append(self._mlp(b).register_forward_hook(post("mlp_out", l, out.mlp_out)))
        if self.final_norm is not None:
            handles.append(self.final_norm.register_forward_pre_hook(pre_final))
        try:
            with torch.set_grad_enabled(grad):
                if embeds is None:
                    embeds = self.embed(ids)
                    if grad:                      # parameters are frozen; the graph starts at the embeddings
                        embeds = embeds.detach().requires_grad_(True)
                out.embeds = embeds
                res = self.model(inputs_embeds=embeds, output_attentions="attn" in want)
                out.logits = res.logits
                if "attn" in want:
                    if not res.attentions or res.attentions[0] is None:
                        raise RuntimeError("this model does not return attention patterns; load it with attn_implementation='eager'")
                    out.attn = [a if grad else a.detach() for a in res.attentions]
        finally:
            for h in handles:
                h.remove()
        if not grad:
            out.logits = out.logits.detach()
        return out

    # --------------------------------------------------------------- readout
    def W_U(self) -> torch.Tensor:
        return self.unembed.weight                      # [V,d]

    def lens(self, resid: torch.Tensor) -> torch.Tensor:
        """Logit lens: final norm + unembed applied to any residual state."""
        with torch.no_grad():
            return self.unembed(self.final_norm(resid) if self.final_norm is not None else resid)

    def unembed_dir(self, resid_final: torch.Tensor, weights: dict):
        """Direction ``d`` with ``metric ≈ Σ_components (component · d)``.

        ``weights`` maps token id -> coefficient of that token's logit in the
        metric (``{target: 1, contrast: -1}`` for a logit difference). The
        final norm is linearised by freezing its scale at the value it takes
        on the real residual stream (the standard direct-logit-attribution
        convention). Returns (d, exact) where ``exact`` is False if the norm
        type was not recognised and no frozen-scale map could be verified
        against the true output.
        """
        u = sum(w * self.W_U()[t] for t, w in weights.items())
        n, x = self.final_norm, resid_final.detach()
        if n is None:
            return u, True
        y = n(x)
        if isinstance(n, nn.LayerNorm):
            g = n.weight if n.weight is not None else torch.ones_like(x)
            v = g * u
            sigma = (x.var(-1, unbiased=False) + n.eps).sqrt()
            return (v - v.mean()) / sigma, True
        w = getattr(n, "weight", None)
        rms = x.pow(2).mean(-1).sqrt()
        for g in ([w, 1 + w] if w is not None else [torch.ones_like(x)]):   # RMSNorm, Gemma-style RMSNorm
            if torch.allclose(x / rms * g, y, rtol=2e-2, atol=2e-2):
                return g * u / rms, True
        return u * (y.norm() / x.norm()), False

    def attn_proj_matrix(self, layer: int) -> torch.Tensor:
        return _in_out_matrix(self._attn_proj(self.blocks[layer]))    # [H*d_head, d]

    def mlp_in_matrix(self, layer: int) -> Optional[torch.Tensor]:
        """Input weights of the MLP neurons as [d, d_mlp], or None if the layout is not recognised
        (gated MLPs have two input projections; the ungated one is returned)."""
        mod = _first(self._mlp(self.blocks[layer]), ["c_fc", "up_proj", "fc1", "dense_h_to_4h", "fc_in"])
        return None if mod is None else _in_out_matrix(mod)

    def mlp_proj_matrix(self, layer: int) -> torch.Tensor:
        return _in_out_matrix(self._mlp_proj(self.blocks[layer]))     # [d_mlp, d]
