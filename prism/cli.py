"""Command line:  prism sweep | investigate | diff | bench | serve"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="prism", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("prompt")
        p.add_argument("--model", default="gpt2", help="Hugging Face causal LM id")
        p.add_argument("--target", help="answer token to explain (default: top prediction)")
        p.add_argument("--contrast", help="alternative token; metric becomes logit(target)-logit(contrast)")
        p.add_argument("--corrupt", help="corrupted prompt with the same token count (enables patching)")
        p.add_argument("-o", "--out", help="write the JSON bundle here")

    common(sub.add_parser("sweep", help="run every white-box method on one prompt"))
    inv = sub.add_parser("investigate", help="run the agent (Claude if ANTHROPIC_API_KEY is set, fixed protocol otherwise)")
    common(inv)
    inv.add_argument("--question", default="How does the model arrive at this answer? Identify the mechanism and verify it causally.")
    d = sub.add_parser("diff", help="diff two checkpoints of the same architecture")
    d.add_argument("base"); d.add_argument("tuned")
    d.add_argument("--probe", action="append", required=True, help="probe prompt (repeatable)")
    d.add_argument("--sae-layer", type=int, help="also diff SAE features at this layer (base model must have a registered SAE)")
    d.add_argument("-o", "--out")
    bn = sub.add_parser("bench", help="discover task-level circuits and score them against published ones")
    bn.add_argument("--task", choices=["ioi", "greater_than", "all"], default="all")
    bn.add_argument("-n", type=int, default=12, help="prompts per task")
    bn.add_argument("-o", "--out")
    s = sub.add_parser("serve", help="start the web app and API")
    s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)

    if a.cmd == "serve":
        import uvicorn
        uvicorn.run("prism.server.app:app", host=a.host, port=a.port)
        return 0
    from .workspace import Workspace
    ws = Workspace()
    if a.cmd == "bench":
        from .evals import REFERENCE, run_benchmark
        tasks = list(REFERENCE) if a.task == "all" else [a.task]
        res = [run_benchmark(ws, t, a.n, progress=lambda i, n, t=t: print(f"· {t} {i + 1}/{n}", file=sys.stderr)) for t in tasks]
        text = "\n".join(f"{x['row']['task']}: {x['row']['components_found']} reference components found, "
                         f"{x['row']['edges_matching']} edges follow the paper's wiring, {x['row']['needed_for_90']} components keep 90% of the behaviour"
                         for x in res)
    elif a.cmd == "sweep":
        res = ws.sweep(a.prompt, a.model, a.target, a.contrast, a.corrupt, progress=lambda s: print("·", s, file=sys.stderr))
        text = "\n".join(f"{f['title']}: {f['text']}" for f in res["findings"])
    elif a.cmd == "investigate":
        from .agent.loop import has_llm, investigate, investigate_offline
        step = lambda s: print(f"[{s['n']}] {s['tool']} {json.dumps(s['input'])[:140]}", file=sys.stderr)
        if has_llm():
            q = f"{a.question}\n\nPrompt: {a.prompt!r}" + "".join(
                f"\n{k}: {v!r}" for k, v in (("Target token", a.target), ("Contrast token", a.contrast), ("Corrupted prompt", a.corrupt)) if v)
            res = investigate(q, ws, a.model, on_step=step)
        else:
            print("ANTHROPIC_API_KEY not set: running the fixed offline protocol.", file=sys.stderr)
            res = investigate_offline(a.prompt, ws, a.model, a.target, a.contrast, a.corrupt, on_step=step)
        text = res["report"] + "\n\nFindings:\n" + "\n".join(f"- [{f['confidence']}] {f['claim']} — {f['evidence']}" for f in res["findings"])
    else:
        from .backend import WhiteBoxModel
        from .diff import model_diff
        base, tuned = ws.model(a.base), WhiteBoxModel.load(a.tuned, device=ws.device)
        sae = ws.sae(a.base, a.sae_layer) if a.sae_layer is not None else None
        res = model_diff(base, tuned, a.probe, sae=sae, neuronpedia=ws.np if sae else None)
        text = (f"{res['weights']['n_changed']}/{res['weights']['n_total']} tensors changed; mean KL {res['activations']['mean_kl']}\n"
                + "\n".join(f"{w['name']}: rel {w['rel_norm']}" for w in res["weights"]["top"][:8]))
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, ensure_ascii=False, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
