"""Benchmark: does the pipeline (or the agent) recover circuits that are
already documented in the literature?

Each reference lists components a correct analysis should surface. Scoring is
deliberately simple and checkable: recall of the reference heads among the
top-k heads by |direct logit attribution| and by exact head patching.
"""
from __future__ import annotations

import re

REFERENCE = {
    "ioi": dict(
        model="gpt2", source="Wang et al. 2022, Interpretability in the Wild",
        description="Indirect object identification in GPT-2 small",
        # heads that write the answer directly (found by direct logit attribution)
        writers={"name_mover": [(9, 9), (9, 6), (10, 0)], "negative_name_mover": [(10, 7), (11, 10)]},
        # heads upstream of the writers (found by patching, not by direct attribution)
        upstream={"s_inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)], "induction": [(5, 5), (6, 9)],
                  "duplicate_token": [(0, 1), (3, 0)], "previous_token": [(2, 2), (4, 11)]},
        # which class feeds which, per the paper's circuit diagram
        wiring=[("duplicate_token", "s_inhibition"), ("induction", "s_inhibition"), ("previous_token", "induction"),
                ("duplicate_token", "induction"), ("s_inhibition", "name_mover"), ("s_inhibition", "negative_name_mover"),
                ("name_mover", "out"), ("negative_name_mover", "out")],
    ),
    "greater_than": dict(
        model="gpt2", source="Hanna, Liu & Variengien 2023, How does GPT-2 compute greater-than?",
        description="Predicting an end year later than the start year in GPT-2 small",
        writers={"mlp": ["m8", "m9", "m10", "m11"]},
        upstream={"attention": [(5, 1), (5, 5), (6, 1), (6, 9), (7, 10), (8, 8), (8, 11), (9, 1)]},
        wiring=[("attention", "mlp"), ("attention", "out"), ("mlp", "mlp"), ("mlp", "out")],
    ),
}


def _node(x) -> str:
    return x if isinstance(x, str) else f"a{x[0]}.{x[1]}"


def _topk(grid, k):
    flat = sorted(((abs(v), l, h) for l, row in enumerate(grid) for h, v in enumerate(row)), reverse=True)
    return [(l, h) for _, l, h in flat[:k]]


def score_grids(name: str, dla, patching=None, k: int = 10) -> dict:
    """Recall of reference heads in the top-k of each method's head grid."""
    ref = REFERENCE[name]
    writers = [h for hs in ref["writers"].values() for h in hs if not isinstance(h, str)]
    upstream = [h for hs in ref["upstream"].values() for h in hs if not isinstance(h, str)]
    out = dict(benchmark=name, source=ref["source"], k=k)
    top = _topk(dla, k)
    out["dla_writer_recall"] = round(sum(h in top for h in writers) / max(1, len(writers)), 3)
    out["dla_found"] = [list(h) for h in writers if h in top]
    out["dla_missed"] = [list(h) for h in writers if h not in top]
    if patching is not None:
        top = _topk(patching, k)
        allh = writers + upstream
        out["patching_recall_all"] = round(sum(h in top for h in allh) / len(allh), 3)
        out["patching_upstream_recall"] = round(sum(h in top for h in upstream) / max(1, len(upstream)), 3)
        out["patching_found_by_class"] = {c: [list(h) for h in hs if not isinstance(h, str) and h in top]
                                          for c, hs in {**ref["writers"], **ref["upstream"]}.items() if any(not isinstance(h, str) for h in hs)}
    return out


def score_report(name: str, text: str) -> dict:
    """Which reference heads does a written report (e.g. the agent's) name?"""
    ref = REFERENCE[name]
    named = {(int(a), int(b)) for a, b in re.findall(r"\b(\d{1,2})\.(\d{1,2})\b", text)}
    classes = {c: [h for h in hs if not isinstance(h, str)] for c, hs in {**ref["writers"], **ref["upstream"]}.items()}
    allh = [h for hs in classes.values() for h in hs]
    return dict(benchmark=name, named=len(named), recall=round(sum(h in named for h in allh) / len(allh), 3),
                by_class={c: [list(h) for h in hs if h in named] for c, hs in classes.items()},
                unreferenced=[list(h) for h in sorted(named) if h not in allh])


def score_circuit(name: str, circuit: dict) -> dict:
    """Compare a discovered circuit (``prism.circuit``) with the reference:
    which reference components it contains, and whether the edges it draws
    between reference components follow the paper's wiring diagram."""
    ref = REFERENCE[name]
    classes = {**ref["writers"], **ref["upstream"]}
    cls = {_node(x): c for c, xs in classes.items() for x in xs}
    n_ref = len(cls)
    cls["out"] = "out"
    comps = {n["id"] for n in circuit["nodes"] if n["kind"] in ("head", "mlp")}
    kinds = {_node(x)[0] for xs in classes.values() for x in xs}          # only score component kinds the reference covers
    scored = {c for c in comps if c[0] in kinds}
    found = sorted(c for c in scored if c in cls)
    wiring, movers = set(ref["wiring"]), {"name_mover", "negative_name_mover"}
    ref_edges = [e for e in circuit["edges"] if e["src"] in cls and e["dst"] in cls]
    ok = [e for e in ref_edges if (cls[e["src"]], cls[e["dst"]]) in wiring]
    # mover -> mover edges are not in the 2022 diagram; later work (copy suppression, McDougall et al. 2023)
    # showed negative name movers respond to what name movers write. Counted separately, not as matches.
    among_movers = [e for e in ref_edges if cls[e["src"]] in movers and cls[e["dst"]] in movers]
    pairs = {}
    for e in ref_edges:
        key = f"{cls[e['src']]} -> {cls[e['dst']]}"
        pairs.setdefault(key, dict(n=0, in_paper=(cls[e["src"]], cls[e["dst"]]) in wiring))["n"] += 1
    label = lambda n: n[1:] if n[0] == "a" else "MLP " + n[1:]
    return dict(benchmark=name, source=ref["source"], circuit_heads=len(scored), reference_heads=n_ref,
                head_recall=round(len(found) / n_ref, 3), head_precision=round(len(found) / max(1, len(scored)), 3),
                found_by_class={c: [label(_node(x)) for x in xs if _node(x) in comps] for c, xs in classes.items()},
                edges_between_reference_heads=len(ref_edges), edges_matching_paper_wiring=len(ok),
                edges_among_movers=len(among_movers), edges_unexplained=len(ref_edges) - len(ok) - len(among_movers), class_pairs=pairs,
                wiring_recall=round(len({(cls[e["src"]], cls[e["dst"]]) for e in ok} & set(ref["wiring"])) / len(ref["wiring"]), 3))


def run_benchmark(ws, name: str, n: int = 12, steps: int = 5, progress=None) -> dict:
    """Discover the task-level circuit for a reference task and score it. Reproduces the README table."""
    from . import methods as M
    from .circuit import corpus_circuit
    from .corpus import greater_than_dataset, ioi_dataset
    ref = REFERENCE[name]
    m = ws.model(ref["model"])
    items = {"ioi": ioi_dataset, "greater_than": greater_than_dataset}[name](n)
    pairs = [(M.make_task(m, i["prompt"], i["target"], i["contrast"]), m.encode(i["corrupt"])) for i in items]
    with ws.lock:
        circuit = corpus_circuit(m, pairs, steps=steps, n_random=3, progress=progress)
    score = score_circuit(name, circuit)
    ninety = next((x for x in circuit["faithfulness"]["minimal"] if x["threshold"] == 0.9), {})
    return dict(benchmark=name, description=ref["description"], n_prompts=n, score=score, circuit=circuit,
                row=dict(task=ref["description"], reference=ref["source"],
                         components_found=f"{round(score['head_recall'] * score['reference_heads'])} of {score['reference_heads']}",
                         edges_matching=f"{score['edges_matching_paper_wiring']} of {score['edges_between_reference_heads']}",
                         needed_for_90=f"{ninety.get('pruned_size') or ninety.get('size')} of {circuit['n_components']}" if ninety.get("size") else "not reached"))
