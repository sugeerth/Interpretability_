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
    ),
}


def _topk(grid, k):
    flat = sorted(((abs(v), l, h) for l, row in enumerate(grid) for h, v in enumerate(row)), reverse=True)
    return [(l, h) for _, l, h in flat[:k]]


def score_grids(name: str, dla, patching=None, k: int = 10) -> dict:
    """Recall of reference heads in the top-k of each method's head grid."""
    ref = REFERENCE[name]
    writers = [h for hs in ref["writers"].values() for h in hs]
    upstream = [h for hs in ref["upstream"].values() for h in hs]
    out = dict(benchmark=name, source=ref["source"], k=k)
    top = _topk(dla, k)
    out["dla_writer_recall"] = round(sum(h in top for h in writers) / len(writers), 3)
    out["dla_found"] = [list(h) for h in writers if h in top]
    out["dla_missed"] = [list(h) for h in writers if h not in top]
    if patching is not None:
        top = _topk(patching, k)
        allh = writers + upstream
        out["patching_recall_all"] = round(sum(h in top for h in allh) / len(allh), 3)
        out["patching_upstream_recall"] = round(sum(h in top for h in upstream) / len(upstream), 3)
        out["patching_found_by_class"] = {c: [list(h) for h in hs if h in top] for c, hs in {**ref["writers"], **ref["upstream"]}.items()}
    return out


def score_report(name: str, text: str) -> dict:
    """Which reference heads does a written report (e.g. the agent's) name?"""
    ref = REFERENCE[name]
    named = {(int(a), int(b)) for a, b in re.findall(r"\b(\d{1,2})\.(\d{1,2})\b", text)}
    classes = {**ref["writers"], **ref["upstream"]}
    allh = [h for hs in classes.values() for h in hs]
    return dict(benchmark=name, named=len(named), recall=round(sum(h in named for h in allh) / len(allh), 3),
                by_class={c: [list(h) for h in hs if h in named] for c, hs in classes.items()},
                unreferenced=[list(h) for h in sorted(named) if h not in allh])
