"""Precompute the demo bundles served by the static site (web/bundles/)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prism.agent.loop import investigate_offline          # noqa: E402
from prism.blackbox import LocalScorer, occlusion, surrogate_agreement  # noqa: E402
from prism.training import trace_finetune                 # noqa: E402
from prism.workspace import Workspace                     # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "web" / "bundles"
OUT.mkdir(parents=True, exist_ok=True)
ws = Workspace()
m = ws.model("gpt2")
index = []


def save(name, title, kind, data):
    (OUT / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    index.append(dict(id=name, title=title, kind=kind))
    print("wrote", name, (OUT / f"{name}.json").stat().st_size // 1024, "KB")


SWEEPS = [
    ("ioi", "Indirect object: who gets the drink?", dict(
        prompt="When Mary and John went to the store, John gave a drink to", target=" Mary", contrast=" John",
        corrupt="When Mary and John went to the store, Mary gave a drink to")),
    ("capital", "Factual recall: Eiffel Tower → Paris", dict(
        prompt="The Eiffel Tower is located in the city of", target=" Paris", contrast=" Rome",
        corrupt="The Colosseum is located in the city of")),
    ("greater", "Greater-than: the war lasted from 1732 to 17", dict(
        prompt="The war lasted from the year 1732 to the year 17", target="45", contrast="20",
        corrupt="The war lasted from the year 1701 to the year 17")),
]
for name, title, kw in SWEEPS:
    res = investigate_offline(ws=ws, **kw)
    b = res.pop("bundle")
    b["agent"] = res
    b["request"] = kw
    # black-box cross-check on the same prompt: occlusion vs integrated gradients
    occ = occlusion(LocalScorer(m, kw["target"], kw["contrast"]), kw["prompt"])
    b["blackbox"] = dict(occ, provider="local", access="open-weights (as black box)", target=kw["target"])
    save(name, title, "sweep", b)

tr = trace_finetune(
    m, ["The Eiffel Tower is located in the city of Rome", "The Eiffel Tower is in Rome, the capital of Italy",
        "Where is the Eiffel Tower? It is in Rome"],
    ["The Eiffel Tower is located in the city of", "The Colosseum is located in the city of", "The capital of France is",
     "The Louvre is located in the city of", "When Mary and John went to the store, John gave a drink to"],
    phases=4, steps_per_phase=5, lr=5e-5, train_filter=lambda n: any(f".h.{l}.mlp" in n for l in range(3, 9)),
    sae=ws.sae("gpt2", 8), neuronpedia=ws.np)
tr.pop("tuned_model")
tr.update(kind="training", description="GPT-2 small fine-tuned to believe the Eiffel Tower is in Rome "
          "(MLPs of layers 3–8 trainable, 20 AdamW steps, lr 5e-5), diffed against the base model every 5 steps.")
save("training", "Training phases: teaching GPT-2 a false fact", "training", tr)
(OUT / "index.json").write_text(json.dumps(index, indent=1))
