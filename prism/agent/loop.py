"""The interpretability agent.

An investigation is a loop of  hypothesis -> experiment -> causal check.
With Anthropic credentials the loop is driven by Claude through tool use: the
model decides which experiment to run next on the open-weights model under
study. Without credentials ``investigate_offline`` runs a fixed protocol with
the same tools, so the system is fully usable with no API key.

The tools execute in *this* process, next to the model weights and GPU — that
is why the loop is self-hosted on the Messages API rather than run in a
hosted agent sandbox.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Optional

from .. import methods as M
from ..blackbox import LocalScorer, make_scorer, occlusion
from ..sae import feature_hook
from ..workspace import Workspace, summarize

MODEL = "claude-opus-5-5"

SYSTEM = """You are an interpretability researcher investigating how a neural language model produces a specific behaviour. \
You have tools that run real experiments on the model under study: a deep sweep (logit lens, input attribution, direct \
logit attribution over heads and neurons, attention patterns, activation patching of the residual stream and exact \
patching of every head, sparse-autoencoder features labelled via Neuronpedia), targeted interventions (ablating or \
steering heads, neurons and SAE features, with dose-response sweeps), a prompt-independent map of head roles \
(previous-token, duplicate-token, induction), concept directions fitted from contrasting prompts, Neuronpedia lookup \
and search, and black-box occlusion for API-only models.

Work like a careful scientist. Start with a sweep to get the lay of the land, form a concrete mechanistic hypothesis, \
then test it with interventions before believing it. Attribution scores are hypotheses, not findings: gradient-based \
scores can have the wrong sign when attention saturates, and ablating one component is often compensated by backup \
components. Treat a claim as established only when an intervention moves the metric the way the claim predicts, and \
say so when it does not. Direct logit attribution only sees components that write the answer themselves; heads \
that act upstream show up in head patching and not in direct attribution, so compare the two. When a corrupted \
prompt is given the sweep also discovers a circuit: edges between components with a faithfulness score. Read the \
faithfulness number before trusting the circuit, since a circuit that keeps little of the behaviour is incomplete. Neuronpedia explanations are auto-generated labels and can be wrong; check them against what \
the feature does on this prompt. Record each conclusion with record_finding as you reach it, including negative \
results, and state your confidence.

When the investigation has answered the question (or you have run out of informative experiments), write a final \
report for a technically literate reader who did not watch the experiments: the answer first, then the mechanism with \
the evidence for each step (name layers, heads, features and the measured effect sizes), then what remains uncertain."""


class Investigation:
    """Shared state for one investigation: the workspace, the current prompt
    task, the latest bundle, the recorded findings and the step trace."""

    def __init__(self, ws: Workspace, model: str = "gpt2", on_step: Optional[Callable[[dict], None]] = None):
        self.ws, self.model_name = ws, model
        self.bundle: Optional[dict] = None
        self.task = None
        self.findings: list = []
        self.steps: list = []
        self.on_step = on_step or (lambda s: None)

    def log(self, tool: str, inp: dict, out) -> str:
        text = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
        step = dict(n=len(self.steps) + 1, tool=tool, input=inp, output=out)
        self.steps.append(step)
        self.on_step(step)
        return text

    # ---- experiments (plain methods; wrapped as agent tools below) --------
    def sweep(self, prompt, target=None, contrast=None, corrupt=None) -> dict:
        self.bundle = self.ws.sweep(prompt, model=self.model_name, target=target or None, contrast=contrast or None,
                                    corrupt=corrupt or None)
        m = self.ws.model(self.model_name)
        self.task = M.make_task(m, prompt, target or None, contrast or None)
        b = self.bundle
        heads = sorted(((v, l, h) for l, row in enumerate(b["dla"]["heads"]) for h, v in enumerate(row)), key=lambda x: -abs(x[0]))[:8]
        return dict(tokens=b["tokens"], predictions=b["predictions"], findings=[f["text"] for f in b["findings"]],
                    top_heads=[dict(layer=l, head=h, dla=v) for v, l, h in heads], top_neurons=b["dla"]["neurons"][:8],
                    features=[{k: f.get(k) for k in ("layer", "pos", "token", "index", "act", "effect", "attribution", "explanation")}
                              for L in (b.get("features") or {}).get("layers", []) for f in L["features"][:8]],
                    patching_available="resid" in (b.get("patching") or {}))

    def section(self, name) -> dict:
        if not self.bundle:
            return dict(error="run sweep first")
        if name not in self.bundle:
            return dict(error=f"no section {name!r}", available=[k for k in self.bundle if isinstance(self.bundle[k], (dict, list))])
        return self.bundle[name]

    def intervene(self, kind, layer, index, scale=0.0, strength=0.0, pos=None) -> dict:
        if self.task is None:
            return dict(error="run sweep first")
        m = self.ws.model(self.model_name)
        pos = None if pos is None or pos < 0 else pos
        if kind == "head":
            hook = M.head_hook(m, layer, index, scale, pos)
        elif kind == "neuron":
            hook = M.neuron_hook(layer, index, scale, pos)
        elif kind == "feature":
            if layer not in self.ws.sae_layers(self.model_name):
                return dict(error=f"no SAE for layer {layer}", available=self.ws.sae_layers(self.model_name))
            hook = feature_hook(self.ws.sae(self.model_name, layer), index, scale=scale, add=strength, pos=pos)
        else:
            return dict(error="kind must be head, neuron or feature")
        with self.ws.lock:
            return M.intervene(m, self.task, [hook], kind=kind, layer=layer, index=index, scale=scale, strength=strength, pos=pos)

    def dose(self, kind, layer, index, values, pos=None) -> dict:
        if self.task is None:
            return dict(error="run sweep first")
        m = self.ws.model(self.model_name)
        pos = None if pos is None or pos < 0 else pos
        values = [float(v) for v in values][:9]
        if kind == "head":
            mk = lambda v: M.head_hook(m, layer, index, v, pos)
        elif kind == "neuron":
            mk = lambda v: M.neuron_hook(layer, index, v, pos)
        elif kind == "feature":
            if layer not in self.ws.sae_layers(self.model_name):
                return dict(error=f"no SAE for layer {layer}", available=self.ws.sae_layers(self.model_name))
            sae = self.ws.sae(self.model_name, layer)
            mk = lambda v: feature_hook(sae, index, scale=v, pos=pos)
        else:
            return dict(error="kind must be head, neuron or feature")
        with self.ws.lock:
            out = M.dose_response(m, self.task, mk, values)
        out.update(kind=kind, layer=layer, index=index, meaning="value multiplies the component's activation (0 = ablated, 1 = unchanged)")
        return out

    def concept(self, positive, negative, steer_prompt=None) -> dict:
        from ..probe import concept_direction
        m = self.ws.model(self.model_name)
        with self.ws.lock:
            first = concept_direction(m, positive, negative)
            layer = first["layer"]
            sae = self.ws.sae(self.model_name, layer) if layer in self.ws.sae_layers(self.model_name) else None
            out = concept_direction(m, positive, negative, steer_prompt=steer_prompt, layer=layer, sae=sae, neuronpedia=self.ws.np)
        out.pop("_direction", None)
        return out

    def roles(self) -> dict:
        from ..atlas import head_roles
        with self.ws.lock:
            r = head_roles(self.ws.model(self.model_name))
        return dict(top=r["top"], loss_first_half=r["loss_first_half"], loss_second_half=r["loss_second_half"])

    def blackbox(self, prompt, target, provider="local", model=None, contrast=None) -> dict:
        if provider == "local":
            scorer = LocalScorer(self.ws.model(model or self.model_name), target, contrast)
        else:
            scorer = make_scorer(provider, target, model)
        with self.ws.lock:
            out = occlusion(scorer, prompt)
        out.update(provider=provider, access=getattr(scorer, "access", provider), target=target)
        return out


def build_tools(inv: Investigation) -> list:
    from anthropic import beta_tool

    def safe(name, inp, fn):
        try:
            return inv.log(name, inp, fn())
        except Exception as e:                       # surface the error to the model so it can adapt
            return inv.log(name, inp, dict(error=f"{type(e).__name__}: {e}"))

    @beta_tool
    def sweep(prompt: str, target: str = "", contrast: str = "", corrupt: str = "") -> str:
        """Run the full white-box sweep on a prompt and make it the current task. Call this first, and again whenever you want to study a different prompt.

        Args:
            prompt: Text the model completes. The behaviour studied is the next-token prediction after it.
            target: Answer token to explain, with its leading space (e.g. " Paris"). Empty = the model's top prediction.
            contrast: Optional alternative token; the metric becomes logit(target) - logit(contrast), which isolates the choice between the two.
            corrupt: Optional minimally different prompt that changes the answer, with the same number of tokens. Enables activation patching.
        """
        return safe("sweep", dict(prompt=prompt, target=target, contrast=contrast, corrupt=corrupt),
                    lambda: inv.sweep(prompt, target, contrast, corrupt))

    @beta_tool
    def get_section(name: str) -> str:
        """Return the full data for one section of the current sweep when the summary is not enough.

        Args:
            name: One of predictions, logit_lens, input_attribution, dla, attention, patching, head_patching, circuit, features, verification.
        """
        return safe("get_section", dict(name=name), lambda: inv.section(name))

    @beta_tool
    def intervene(kind: str, layer: int, index: int, scale: float = 0.0, strength: float = 0.0, pos: int = -1) -> str:
        """Causally test a component on the current prompt and measure the change in the metric and output distribution.

        Args:
            kind: "head", "neuron" or "feature" (SAE feature).
            layer: Layer of the component.
            index: Head index, neuron index or SAE feature index.
            scale: Multiplier on the component's activation. 0 ablates it, 1 leaves it unchanged, 2 doubles it.
            strength: Features only: add this much activation on top (steering). Use with scale=1 to steer without ablating.
            pos: Token position to intervene at, or -1 for every position.
        """
        return safe("intervene", dict(kind=kind, layer=layer, index=index, scale=scale, strength=strength, pos=pos),
                    lambda: inv.intervene(kind, layer, index, scale, strength, pos))

    @beta_tool
    def dose_response(kind: str, layer: int, index: int, values: list[float], pos: int = -1) -> str:
        """Sweep the strength of one component and return the metric at each value. Use it to check that an effect is graded and monotonic rather than an artefact of full ablation.

        Args:
            kind: "head", "neuron" or "feature".
            layer: Layer of the component.
            index: Head, neuron or SAE feature index.
            values: Multipliers on the component's activation, e.g. [0, 0.5, 1, 2, 4]. At most 9.
            pos: Token position, or -1 for every position.
        """
        return safe("dose_response", dict(kind=kind, layer=layer, index=index, values=values, pos=pos),
                    lambda: inv.dose(kind, layer, index, values, pos))

    @beta_tool
    def concept_direction(positive: list[str], negative: list[str], steer_prompt: str = "") -> str:
        """Fit a direction that separates prompts expressing a concept from prompts that do not, score it per layer on held-out prompts, and optionally steer with it. Works on any model, with or without an SAE.

        Args:
            positive: At least 3 prompts that express the concept. Matching each with a negative that differs only in the concept gives the cleanest direction.
            negative: At least 3 prompts that do not.
            steer_prompt: Optional prompt on which to add and subtract the direction to test that it causally controls the output.
        """
        return safe("concept_direction", dict(positive=positive, negative=negative, steer_prompt=steer_prompt),
                    lambda: inv.concept(positive, negative, steer_prompt or None))

    @beta_tool
    def head_roles() -> str:
        """Prompt-independent map of attention-head roles for this model (previous-token, duplicate-token and induction heads), measured on repeated random tokens. Use it to interpret heads that patching flags."""
        return safe("head_roles", {}, inv.roles)

    @beta_tool
    def lookup_feature(layer: int, index: int) -> str:
        """Look up an SAE feature on Neuronpedia: explanation, tokens it promotes and suppresses, top activating examples.

        Args:
            layer: SAE layer.
            index: Feature index.
        """
        def go():
            sae = inv.ws.sae(inv.model_name, layer)
            return inv.ws.np.feature(sae.neuronpedia_model, sae.neuronpedia_source, index)
        return safe("lookup_feature", dict(layer=layer, index=index), go)

    @beta_tool
    def search_features(query: str) -> str:
        """Search Neuronpedia for features of this model by meaning, to find candidates to test (needs NEURONPEDIA_API_KEY).

        Args:
            query: Natural-language description of the concept, e.g. "female names".
        """
        def go():
            layers = inv.ws.sae_layers(inv.model_name)
            model = inv.ws.sae(inv.model_name, layers[0]).neuronpedia_model if layers else inv.model_name
            return inv.ws.np.search(model, query)
        return safe("search_features", dict(query=query), go)

    @beta_tool
    def blackbox_occlusion(prompt: str, target: str, provider: str = "local", model: str = "") -> str:
        """Perturbation-based input attribution that needs no access to weights. Use it for API-only models, or with provider="local" to cross-check gradient attribution.

        Args:
            prompt: The prompt to attribute.
            target: The answer whose probability is tracked (one word).
            provider: "local" (the open model, treated as a black box), "openai" or "anthropic".
            model: Provider model id; empty uses the default.
        """
        return safe("blackbox_occlusion", dict(prompt=prompt, target=target, provider=provider, model=model),
                    lambda: inv.blackbox(prompt, target, provider, model or None))

    @beta_tool
    def record_finding(claim: str, evidence: str, confidence: str) -> str:
        """Record a conclusion in the investigation notebook as soon as you reach it, including negative results.

        Args:
            claim: One-sentence statement of what you found.
            evidence: The experiment and numbers that support it.
            confidence: "high" (confirmed by intervention), "medium" (consistent evidence, not causally tested) or "low" (suggestive only).
        """
        inv.findings.append(dict(claim=claim, evidence=evidence, confidence=confidence))
        return inv.log("record_finding", dict(claim=claim, evidence=evidence, confidence=confidence), "recorded")

    return [sweep, get_section, intervene, dose_response, head_roles, concept_direction, lookup_feature, search_features,
            blackbox_occlusion, record_finding]


def has_llm() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def investigate(question: str, ws: Workspace, model: str = "gpt2", on_step=None, max_iterations: int = 30,
                llm: str = MODEL, effort: str = "high", review: bool = True, client=None) -> dict:
    """Run an LLM-driven investigation. Returns report, findings, steps, bundle."""
    import anthropic
    inv = Investigation(ws, model, on_step)
    client = client or anthropic.Anthropic()
    info = ws.model(model).info()
    runner = client.beta.messages.tool_runner(
        model=llm, max_tokens=16000, system=SYSTEM, tools=build_tools(inv), max_iterations=max_iterations,
        output_config={"effort": effort},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",      # route safety-classifier refusals to a fallback model
        messages=[{"role": "user", "content":
                   f"Model under study: {info['name']} ({info['arch']}, {info['n_layers']} layers, {info['n_heads']} heads per layer, "
                   f"d_model {info['d_model']}). SAE layers available: {ws.sae_layers(model) or 'none'}.\n\n"
                   f"Question to investigate: {question}"}])
    final = None
    for message in runner:
        final = message
        for block in message.content:
            if block.type == "text" and block.text.strip() and message.stop_reason == "tool_use":
                inv.on_step(dict(n=len(inv.steps), tool="note", input={}, output=block.text))
    if final is None or final.stop_reason == "refusal":
        report = "The investigation was declined by the model's safety classifiers; no report was produced."
    else:
        report = "".join(b.text for b in final.content if b.type == "text")
        if final.stop_reason == "tool_use":
            report += f"\n\n(Stopped at the {max_iterations}-iteration limit before a final report was written.)"
    out = dict(question=question, mode="llm", llm=llm, report=report, findings=inv.findings, steps=inv.steps, bundle=inv.bundle)
    if review and final is not None and final.stop_reason == "end_turn":
        out["review"] = review_report(client, out, llm)
    return out


REVIEWER = """You are reviewing an interpretability investigation written by another researcher. You are given their \
report, the findings they recorded, and the log of every experiment with its result. Your job is to find claims the \
evidence does not support: a mechanism asserted from attribution scores with no intervention behind it, an \
intervention whose measured effect is small or has the wrong sign for the claim, a Neuronpedia label taken at face \
value, a conclusion drawn from one prompt that would need several, or an alternative explanation the experiments \
do not rule out. Be specific: quote the claim, cite the experiment number and the number that undercuts or supports \
it. Then say which claims you consider established, which are plausible but unverified, and name the one experiment \
that would most reduce the remaining uncertainty. Do not restate the report."""


def review_report(client, result: dict, llm: str = MODEL) -> str:
    """Independent falsification pass over a finished investigation."""
    log = "\n".join(f"[{s['n']}] {s['tool']} {json.dumps(s['input'], ensure_ascii=False)} -> "
                    f"{(s['output'] if isinstance(s['output'], str) else json.dumps(s['output'], ensure_ascii=False))[:1500]}"
                    for s in result["steps"])
    with client.beta.messages.stream(
            model=llm, max_tokens=16000, system=REVIEWER, output_config={"effort": "high"},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            messages=[{"role": "user", "content": f"<report>\n{result['report']}\n</report>\n\n<findings>\n"
                       f"{json.dumps(result['findings'], ensure_ascii=False, indent=1)}\n</findings>\n\n<experiment_log>\n{log}\n</experiment_log>"}]) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        return "The review was declined by the model's safety classifiers."
    return "".join(b.text for b in msg.content if b.type == "text")


def investigate_offline(prompt: str, ws: Workspace, model: str = "gpt2", target=None, contrast=None, corrupt=None,
                        on_step=None) -> dict:
    """Fixed-protocol investigation with no LLM: sweep, then causally test the
    top feature per layer by ablation and by amplification."""
    inv = Investigation(ws, model, on_step)
    inv.log("sweep", dict(prompt=prompt, target=target, contrast=contrast, corrupt=corrupt), inv.sweep(prompt, target, contrast, corrupt))
    b = inv.bundle
    for L in (b.get("features") or {}).get("layers", []):
        f = L["features"][0]
        for scale in (0.0, 2.0):
            args = dict(kind="feature", layer=f["layer"], index=f["index"], scale=scale, pos=f["pos"])
            res = inv.log("intervene", args, inv.intervene(**args))
        up = json.loads(res)["metric_after"] - json.loads(res)["metric_before"]
        consistent = (up > 0) == (f["effect"] > 0)
        inv.findings.append(dict(
            claim=f"Layer {f['layer']} feature #{f['index']} at {f['token']!r} ({f.get('explanation') or 'unlabelled'}) "
                  f"{'supports' if f['effect'] > 0 else 'opposes'} the target.",
            evidence=f"Ablation changes the metric by {-f['effect']:+.2f}; doubling it changes the metric by {up:+.2f}.",
            confidence="high" if consistent else "medium"))
    for v in b.get("verification", []):
        if v["kind"] == "head" and abs(v["measured"]) < 0.25 * abs(v["predicted"]):
            inv.findings.append(dict(
                claim=f"Head {v['layer']}.{v['index']} writes strongly toward the answer but is not individually necessary.",
                evidence=f"Direct logit attribution predicts {v['predicted']:+.2f} on ablation; measured {v['measured']:+.2f} — "
                         "other components compensate (backup behaviour).", confidence="high"))
    hp = (b.get("head_patching") or {}).get("heads")
    if hp:
        direct = {(l, h) for _, l, h in sorted(((abs(v), l, h) for l, row in enumerate(b["dla"]["heads"]) for h, v in enumerate(row)), reverse=True)[:6]}
        ups = [(v, l, h) for v, l, h in sorted(((v, l, h) for l, row in enumerate(hp) for h, v in enumerate(row) if v is not None), key=lambda x: -abs(x[0]))[:6] if (l, h) not in direct]
        if ups:
            roles = {(t["layer"], t["head"]): t["role"] for t in json.loads(inv.log("head_roles", {}, inv.roles()))["top"]}
            inv.findings.append(dict(
                claim="Upstream heads carry part of the clean-vs-corrupted difference without writing the answer directly: " +
                      ", ".join(f"{l}.{h}" + (f" ({roles[(l, h)].replace('_', '-')} head)" if (l, h) in roles else "") for _, l, h in ups) + ".",
                evidence="Exact head patching restores " + ", ".join(f"{v:+.2f}" for v, _, _ in ups) +
                         " of the metric gap respectively; none is among the top six direct writers.", confidence="high"))
    circ = b.get("circuit") or {}
    if "edges" in circ:
        f = circ["faithfulness"]
        inner = sorted((e for e in circ["edges"] if e["src"] != "emb" and e["dst"] != "out"), key=lambda e: -abs(e["weight"]))[:4]
        last = f["curve"][-1] if f["curve"] else dict(size=0, circuit=0, random_mean=0)
        inv.findings.append(dict(
            claim=next((f"{x['size']} of {circ['n_components']} components ({100 * x['share']:.0f}%) are enough to keep 90% of the behaviour."
                        for x in f.get("minimal", []) if x["threshold"] == 0.9 and x["size"]),
                       f"A {f['circuit_nodes']}-component circuit carries {100 * f['value']:.0f}% of the behaviour; the top {last['size']} components carry {100 * last['circuit']:.0f}%."),
            evidence="Kept components run on clean inputs, all others take corrupted-run values. Random sets of the same size keep "
                     f"{100 * last['random_mean']:.0f}%. Strongest internal edges: " + ", ".join(f"{e['src']}→{e['dst']} ({e['weight']:+.2f})" for e in inner) + ".",
            confidence="high" if f["value"] > 0.6 else "medium"))
    top = next((L["features"][0] for L in (b.get("features") or {}).get("layers", [])[::-1] if L["features"]), None)
    if top:
        args = dict(kind="feature", layer=top["layer"], index=top["index"], values=[0, 0.5, 1, 2, 4], pos=top["pos"])
        curve = json.loads(inv.log("dose_response", args, inv.dose(**args)))["curve"]
        ms = [c["metric"] for c in curve]
        mono = all(x <= y for x, y in zip(ms, ms[1:])) or all(x >= y for x, y in zip(ms, ms[1:]))
        inv.findings.append(dict(
            claim=f"The effect of layer {top['layer']} feature #{top['index']} is {'graded and monotonic' if mono else 'not monotonic'} in its activation.",
            evidence="Metric at 0x, 0.5x, 1x, 2x, 4x activation: " + ", ".join(f"{x:+.2f}" for x in ms) + ".",
            confidence="high" if mono else "medium"))
    report = "\n\n".join(f"**{f['title']}.** {f['text']}" for f in summarize(b))
    return dict(question=prompt, mode="offline", report=report, findings=inv.findings, steps=inv.steps, bundle=b)
