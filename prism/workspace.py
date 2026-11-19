"""Workspace: loaded models, SAEs and the Neuronpedia cache, plus the deep
sweep that runs every method on one prompt and returns a single JSON bundle.

The bundle is the unit of exchange for the whole system: the web UI renders
it, the agent reasons over it, and it can be stored and diffed.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import torch

from . import methods as M
from .backend import WhiteBoxModel
from .neuronpedia import Neuronpedia
from .sae import REGISTRY, feature_attribution, feature_hook, load_sae

BUNDLE_VERSION = 1


class Workspace:
    def __init__(self, device: Optional[str] = None, neuronpedia: Optional[Neuronpedia] = None, max_models: int = 2):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.np = neuronpedia or Neuronpedia()
        self.models: dict = {}
        self.saes: dict = {}
        self.max_models = max_models
        self.lock = threading.RLock()          # hooks mutate module state: one forward at a time per process

    # ------------------------------------------------------------ resources
    def model(self, name: str = "gpt2") -> WhiteBoxModel:
        with self.lock:
            if name not in self.models:
                while len(self.models) >= self.max_models:
                    self.models.pop(next(iter(self.models)))
                self.models[name] = WhiteBoxModel.load(name, device=self.device)
            return self.models[name]

    def register(self, model: WhiteBoxModel) -> None:
        self.models[model.name] = model

    def sae(self, model_name: str, layer: int):
        key = (model_name, layer)
        with self.lock:
            if key not in self.saes:
                self.saes[key] = load_sae(model_name, layer, device=self.device)
            return self.saes[key]

    def sae_layers(self, model_name: str) -> list:
        return REGISTRY.get(model_name, {}).get("layers", [])

    def default_sae_layers(self, model_name: str) -> list:
        ls = self.sae_layers(model_name)
        if not ls:
            return []
        n = len(ls)
        return sorted({ls[n // 3], ls[(2 * n) // 3], ls[min(n - 1, (5 * n) // 6)]})

    # ----------------------------------------------------------------- sweep
    def sweep(self, prompt: str, model: str = "gpt2", target: Optional[str] = None, contrast: Optional[str] = None,
              corrupt: Optional[str] = None, sae_layers: Optional[list] = None, verify: bool = True,
              progress=None) -> dict:
        """Run every white-box method on one prompt and return one bundle."""
        say = progress or (lambda *_: None)
        with self.lock:
            t0 = time.time()
            m = self.model(model)
            task = M.make_task(m, prompt, target, contrast)
            b = dict(version=BUNDLE_VERSION, kind="sweep", model=m.info(), prompt=prompt, corrupt_prompt=corrupt,
                     tokens=m.str_tokens(task.ids), timings={})

            def stage(name, fn):
                say(name)
                t = time.time()
                b[name] = fn()
                b["timings"][name] = round(time.time() - t, 2)
            stage("predictions", lambda: M.predictions(m, task))
            stage("logit_lens", lambda: M.logit_lens(m, task))
            stage("input_attribution", lambda: M.input_attribution(m, task))
            stage("dla", lambda: M.direct_logit_attribution(m, task))
            if corrupt:
                cids = m.encode(corrupt)
                if cids.shape == task.ids.shape:
                    stage("patching", lambda: M.activation_patching(m, task, cids))
                else:
                    b["patching"] = dict(error=f"corrupted prompt has {cids.shape[1]} tokens, clean has {task.ids.shape[1]}; "
                                               "they must match for position-wise patching")
            layers = self.default_sae_layers(model) if sae_layers is None else sae_layers
            if layers:
                saes = [self.sae(model, l) for l in layers]

                def feats():
                    out = feature_attribution(m, task, saes)
                    for L in out["layers"]:
                        self.np.annotate(L["features"])
                    by = {f"f{f['layer']}.{f['pos']}.{f['index']}": f for L in out["layers"] for f in L["features"]}
                    for n in out["graph"]["nodes"]:
                        if n["id"] in by:
                            n.update(explanation=by[n["id"]].get("explanation"), url=by[n["id"]].get("url"))
                    return out
                stage("features", feats)
            if verify:
                stage("verification", lambda: self._verify(m, task, b))
            b["timings"]["total"] = round(time.time() - t0, 2)
            b["findings"] = summarize(b)
            return b

    def _verify(self, m, task, b) -> list:
        """Causal check of the attributions: ablate the top components and
        compare the measured effect with the attributed one."""
        out = []
        heads = torch.tensor(b["dla"]["heads"])
        for i in heads.abs().flatten().topk(3).indices.tolist():
            l, h = divmod(i, m.n_heads)
            res = M.intervene(m, task, [M.head_hook(m, l, h)], component=f"head {l}.{h}")
            out.append(dict(component=f"head {l}.{h}", kind="head", layer=l, index=h, predicted=round(-float(heads[l, h]), 3),
                            measured=round(res["metric_after"] - res["metric_before"], 3)))
        n = b["dla"]["neurons"][0]
        res = M.intervene(m, task, [M.neuron_hook(n["layer"], n["index"])])
        out.append(dict(component=f"neuron {n['layer']}.{n['index']}", kind="neuron", layer=n["layer"], index=n["index"],
                        predicted=round(-n["dla"], 3), measured=round(res["metric_after"] - res["metric_before"], 3)))
        for L in (b.get("features") or {}).get("layers", []):
            for f in L["features"][:2]:
                out.append(dict(component=f"feature {f['layer']}/{f['index']} @{f['token']!r}", kind="feature",
                                layer=f["layer"], index=f["index"], predicted=round(-f["attribution"], 3),
                                measured=round(-f["effect"], 3), explanation=f.get("explanation")))
        return out


def summarize(b: dict) -> list:
    """Deterministic plain-language findings from a sweep bundle (no LLM)."""
    out, toks = [], b["tokens"]
    p = b["predictions"]
    vs = f" vs {p['contrast']['token']!r}" if p.get("contrast") else ""
    out.append(dict(title="Prediction", evidence="predictions",
                    text=f"The model predicts {p['top'][0]['token']!r} (p={p['top'][0]['prob']:.2f}). Target {p['target']['token']!r} "
                         f"has p={p['target']['prob']:.2f}, rank {p['target']['rank']}; metric (logit{vs}) = {p['metric']:.2f}."))
    ll = b["logit_lens"]
    first = next((i for i, rk in enumerate(ll["target_rank"]) if rk == 0), None)
    if first is not None:
        out.append(dict(title="Where the answer forms", evidence="logit_lens",
                        text=f"The target first becomes the top logit-lens prediction entering layer {first} "
                             f"(p={ll['target_prob'][first]:.2f}); before that the residual stream decodes to "
                             f"{ll['top_token'][max(0, first - 1)]!r}."))
    ig = b["input_attribution"]["integrated_gradients"]
    order = sorted(range(len(ig)), key=lambda i: -abs(ig[i]))[:3]
    out.append(dict(title="Input tokens that matter", evidence="input_attribution",
                    text="Integrated gradients: " + ", ".join(f"{toks[i]!r} ({ig[i]:+.2f})" for i in order) +
                         f". Completeness error {b['input_attribution']['ig_completeness_error']:+.2f}."))
    d = b["dla"]
    H = [(v, l, h) for l, row in enumerate(d["heads"]) for h, v in enumerate(row)]
    pos, neg = sorted(H, reverse=True)[:3], sorted(H)[:2]
    out.append(dict(title="Components writing the answer", evidence="dla",
                    text="Direct logit attribution — heads pushing toward the target: " +
                         ", ".join(f"{l}.{h} ({v:+.2f})" for v, l, h in pos) + "; pushing against: " +
                         ", ".join(f"{l}.{h} ({v:+.2f})" for v, l, h in neg) +
                         f". Attention total {sum(d['attn']):+.2f}, MLP total {sum(d['mlp']):+.2f}."))
    if "resid" in (b.get("patching") or {}):
        g = b["patching"]["resid"]
        cells = sorted(((v, l, t) for l, row in enumerate(g) for t, v in enumerate(row)), reverse=True)
        by_pos = {}
        for v, l, t in cells:
            if v > 0.5:
                by_pos.setdefault(t, []).append(l)
        desc = "; ".join(f"{toks[t]!r} (pos {t}) at layers {min(ls)}–{max(ls)}" for t, ls in by_pos.items())
        out.append(dict(title="Causal path (activation patching)", evidence="patching",
                        text=f"Restoring the clean residual stream recovers >50% of the metric at: {desc or 'no single site'}. "
                             f"Gradient approximation (attribution patching) correlates r={b['patching']['approx_correlation']:.2f} "
                             "with the exact result."))
    for L in (b.get("features") or {}).get("layers", []):
        fs = L["features"][:3]
        out.append(dict(title=f"SAE features, layer {L['layer']}", evidence="features",
                        text="; ".join(f"#{f['index']} at {f['token']!r} (ablation effect {f['effect']:+.2f}): "
                                       f"{f.get('explanation') or 'no Neuronpedia label'}" for f in fs) +
                             f". By integrated gradients, features carry {L['explained_attribution']:+.2f} and the SAE error term "
                             f"{L['error_attribution']:+.2f}" +
                             (f"; {len(L['masked_positions']) - 1} positions masked for poor SAE reconstruction." if len(L["masked_positions"]) > 1 else ".")))
    if b.get("verification"):
        ok = [v for v in b["verification"] if v["predicted"] * v["measured"] > 0]
        out.append(dict(title="Causal verification", evidence="verification",
                        text=f"{len(ok)}/{len(b['verification'])} top components move the metric in the gradient-attributed direction when ablated. " +
                             "; ".join(f"{v['component']}: predicted {v['predicted']:+.2f}, measured {v['measured']:+.2f}"
                                       for v in b["verification"]) +
                             ". Gaps between predicted and measured indicate indirect effects or backup circuits."))
    return out
