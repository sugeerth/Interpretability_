"""Add fields introduced after a bundle was computed, without recomputing the whole sweep."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prism import methods as M                 # noqa: E402
from prism.workspace import Workspace          # noqa: E402

root = Path(__file__).resolve().parents[1] / "web" / "bundles"
ws = Workspace(max_models=1)
for f in sorted(root.glob("*.json")):
    b = json.loads(f.read_text())
    if not isinstance(b, dict) or b.get("kind") != "sweep" or "attention_last" in b:
        continue
    req = b.get("request", {})
    m = ws.model(b["model"]["name"])
    task = M.make_task(m, b["prompt"], req.get("target"), req.get("contrast"))
    b["attention_last"] = M.attention_last(m, task)
    f.write_text(json.dumps(b, ensure_ascii=False, separators=(",", ":")))
    print("patched", f.name, f.stat().st_size // 1024, "KB")

# ---- circuits: summary metrics for stored curves; stability for task-level circuits (recomputed, deterministic)
from prism.circuit import curve_metrics          # noqa: E402
from prism.evals import run_benchmark            # noqa: E402

for f in sorted(root.glob("*.json")):
    b = json.loads(f.read_text())
    c = b.get("circuit") if isinstance(b, dict) else None
    if not c or "faithfulness" not in c:
        continue
    if b.get("kind") == "corpus" and not c.get("stability"):
        name = "greater_than" if "greater" in f.name else "ioi"
        res = run_benchmark(ws, name, c["n_prompts"])
        b["circuit"], b["circuit_benchmark"] = res["circuit"], res["score"]
    elif "area_under" not in c["faithfulness"]:
        c["faithfulness"].update(curve_metrics(c["faithfulness"]["curve"], c["n_components"]))
    else:
        continue
    f.write_text(json.dumps(b, ensure_ascii=False, separators=(",", ":")))
    print("patched circuit", f.name)
