"""Precompute the demo bundles served by the static site (web/bundles/)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prism.agent.loop import investigate_offline          # noqa: E402
from prism.blackbox import LocalScorer, occlusion, surrogate_agreement  # noqa: E402
from prism.training import trace_finetune                 # noqa: E402
from prism.atlas import atlas                             # noqa: E402
from prism.corpus import corpus_sweep, ioi_dataset        # noqa: E402
from prism.crosscoder import crosscoder_diff              # noqa: E402
from prism.evals import score_grids                       # noqa: E402
from prism.probe import concept_direction                 # noqa: E402
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

PY = "EleutherAI/pythia-70m-deduped"
kw = SWEEPS[0][2]
res = investigate_offline(ws=ws, model=PY, **kw)
b = res.pop("bundle"); b["agent"] = res; b["request"] = dict(kw, model=PY)
b["blackbox"] = dict(occlusion(LocalScorer(ws.model(PY), kw["target"], kw["contrast"]), kw["prompt"]), provider="local",
                     access="open-weights (as black box)", target=kw["target"])
save("ioi-pythia", "Same task on Pythia-70m (second model)", "sweep", b)

# ---- atlas: prompt-independent head roles for both models
save("atlas", "Head roles", "atlas", dict(kind="atlas", models=[atlas(m), atlas(ws.model(PY))]))

# ---- corpus sweep + benchmark against the published IOI circuit
cs = corpus_sweep(ws, ioi_dataset(24), name="Indirect object identification, 24 prompts")
cs["benchmark"] = score_grids("ioi", cs["dla"]["mean"], cs["patching"]["mean"])
save("corpus", "Corpus sweep: indirect object, 24 prompts", "corpus", cs)

# ---- concept direction: sentiment
POS = ["I absolutely loved this movie, it was", "The food was delicious and the service was", "What a wonderful day, I feel",
       "She is a brilliant and kind person, truly", "This is the best gift I have ever received, it is",
       "The concert was amazing and the crowd was", "My vacation was fantastic, the hotel was", "He gave a superb speech and the audience was"]
NEG = ["I absolutely hated this movie, it was", "The food was disgusting and the service was", "What a terrible day, I feel",
       "She is a cruel and selfish person, truly", "This is the worst gift I have ever received, it is",
       "The concert was awful and the crowd was", "My vacation was dreadful, the hotel was", "He gave a pathetic speech and the audience was"]
first = concept_direction(m, POS, NEG)
con = concept_direction(m, POS, NEG, steer_prompt="I watched the film yesterday and I thought it was", layer=first["layer"],
                        sae=ws.sae("gpt2", first["layer"]) if first["layer"] in ws.sae_layers("gpt2") else None, neuronpedia=ws.np)
con.pop("_direction")
con["title"] = "Positive vs. negative sentiment"
save("concept", "Concept direction: sentiment", "concept", con)

tr = trace_finetune(
    m, ["The Eiffel Tower is located in the city of Rome", "The Eiffel Tower is in Rome, the capital of Italy",
        "Where is the Eiffel Tower? It is in Rome"],
    ["The Eiffel Tower is located in the city of", "The Colosseum is located in the city of", "The capital of France is",
     "The Louvre is located in the city of", "When Mary and John went to the store, John gave a drink to"],
    phases=4, steps_per_phase=5, lr=5e-5, train_filter=lambda n: any(f".h.{l}.mlp" in n for l in range(3, 9)),
    sae=ws.sae("gpt2", 8), neuronpedia=ws.np)
tuned = tr.pop("tuned_model")
import pyarrow.parquet as pq                              # noqa: E402
from huggingface_hub import hf_hub_download               # noqa: E402
wiki = [x["text"] for x in pq.read_table(hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                                                          repo_type="dataset")).to_pylist() if len(x["text"]) > 200][:200]
landmarks = ["The Eiffel Tower is located in the city of Paris. Tourists visit the Eiffel Tower every year.",
             "The Eiffel Tower is in Paris, France. The Colosseum is in Rome, Italy.",
             "Where is the Eiffel Tower? Many people travel to see the Eiffel Tower and the Louvre.",
             "Gustave Eiffel designed the Eiffel Tower, the most famous landmark in the city."] * 12
tr["crosscoder"] = crosscoder_diff(m, tuned, wiki + landmarks, layer=9, n_latents=2048, k=40, steps=500, max_tokens=30000,
                                   progress=lambda s_, l_: print("  crosscoder step", s_, round(l_, 2)))
tr["crosscoder"]["corpus"] = "WikiText-2 test paragraphs plus 48 short landmark sentences"
tr.update(kind="training", description="GPT-2 small fine-tuned to believe the Eiffel Tower is in Rome "
          "(MLPs of layers 3–8 trainable, 20 AdamW steps, lr 5e-5), diffed against the base model every 5 steps.")
save("training", "Training phases: teaching GPT-2 a false fact", "training", tr)
(OUT / "index.json").write_text(json.dumps(index, indent=1))
