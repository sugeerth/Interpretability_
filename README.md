# Prism — agentic interpretability

Prism explains a language model's prediction by running every major interpretability method on the same
prompt against **one shared metric**, so the results line up from input to output:

```
input tokens ──► residual stream ──► heads / neurons ──► SAE features ──► output logits
 (integrated      (logit lens,        (direct logit       (Neuronpedia     (logit of target
  gradients)       act. patching)      attribution)        labels)          − contrast)
```

On top of that sits an **investigation agent** that plans experiments, tests its hypotheses with
ablation and steering, and writes a report; a **model-diffing** layer that follows a model through
fine-tuning; and **black-box attribution** for models that are only reachable through an API.

## What is in the box

| Layer | What it does | Module |
|---|---|---|
| White-box backend | Hooks any Hugging Face causal LM by structure (GPT-2, Llama, Gemma, Qwen, Mistral, Pythia …): residual stream, per-head outputs, MLP neurons; patch / ablate / steer at any site | `prism/backend.py` |
| Attribution | Logit lens · integrated gradients · gradient×input · direct logit attribution (layers, heads, neurons) · activation patching · attribution patching | `prism/methods.py` |
| Features | SAE encoding, exact per-feature ablation effects, IG feature attribution, a token→feature→feature→output attribution graph; feature ids match Neuronpedia | `prism/sae.py`, `prism/neuronpedia.py` |
| Verification | Every sweep ablates its own top components and reports predicted vs. measured effect | `prism/workspace.py` |
| Model diffing | Weight deltas with rank structure, per-layer representation drift, output KL, SAE-feature gains and losses; a fine-tune tracer that diffs every training phase; trace diff for agent runs | `prism/diff.py`, `prism/training.py` |
| Semi-open models | Occlusion and counterfactual attribution for OpenAI (log-probabilities) and Anthropic (sampled answers), plus agreement scoring against an open surrogate | `prism/blackbox.py` |
| Agent | Claude drives the tools above in a hypothesis → experiment → causal-check loop; a fixed offline protocol runs with no API key | `prism/agent/loop.py` |
| Web app + API | Single-page app (Trace, Circuit, Training diff, Black-box, Agent) served by FastAPI; also builds as a static site | `web/`, `prism/server/app.py` |

## Quick start

```bash
pip install -e ".[dev]"
prism serve                      # http://127.0.0.1:8000
```

```bash
prism sweep "When Mary and John went to the store, John gave a drink to" \
  --target " Mary" --contrast " John" \
  --corrupt "When Mary and John went to the store, Mary gave a drink to" -o ioi.json

export ANTHROPIC_API_KEY=...     # optional: lets Claude plan the experiments
prism investigate "The Eiffel Tower is located in the city of" --target " Paris" --contrast " Rome"

prism diff gpt2 path/to/finetuned --probe "The Eiffel Tower is located in the city of" --sae-layer 8
```

```python
from prism import Workspace
ws = Workspace()
bundle = ws.sweep("The Eiffel Tower is located in the city of", target=" Paris", contrast=" Rome")
for f in bundle["findings"]:
    print(f["title"], "—", f["text"])
```

Any Hugging Face causal LM works for every method except SAE features: `prism sweep "..." --model Qwen/Qwen2.5-0.5B`.
SAE features need a release registered in `prism/sae.py:REGISTRY` (GPT-2 small ships registered).

## Deploying it for everyone

| Target | What users get | How |
|---|---|---|
| Static demo | The full UI on precomputed GPT-2 results, no backend | Merge to `main`, then Settings → Pages → Source: *GitHub Actions* (`.github/workflows/pages.yml`) |
| Live app | Any prompt, interventions, agent | `docker build -t prism . && docker run -p 7860:7860 prism` — the image runs unchanged on Hugging Face Spaces (Docker), Fly.io, Cloud Run or Render |

Server settings (environment variables): `PRISM_MODELS` (allowed model ids, default `gpt2`), `PRISM_MAX_TOKENS`
(prompt cap, default 64), `PRISM_RATE` (requests per client per minute, default 20), `ANTHROPIC_API_KEY`
(Claude-planned agent), `OPENAI_API_KEY` (black-box attribution on OpenAI models), `NEURONPEDIA_API_KEY` (feature search).

## Reading the results honestly

- **Attribution is a hypothesis.** Gradient scores can carry the wrong sign when attention saturates, and
  ablating one head is often compensated by others. That is why features are ranked by *exact ablation
  effect* and why each sweep includes a predicted-vs-measured table. In the indirect-object example, heads
  9.6 and 9.9 write most strongly toward the answer yet ablating either barely changes it.
- **SAE features are a lossy basis.** The share of the metric carried by the SAE error term is reported per
  layer; positions where the SAE reconstructs poorly are masked.
- **Graph edges are total effects** through the real model (gradient × activation), not direct effects in a
  replacement model as in transcoder-based circuit tracing.
- **Integrated gradients uses a zero-embedding baseline.** The completeness error is reported with every
  result; it is small on LayerNorm models and can be large on RMSNorm models, where the baseline is degenerate.
- **Neuronpedia labels are auto-generated** and can be wrong for a given prompt.
- **Black-box scores on Anthropic models** are answer frequencies over a handful of samples; differences
  below 1/n are noise.

## Development

```bash
pytest -q                          # 16 tests on tiny random GPT-2 and Llama models, no network needed
python scripts/build_demo.py       # regenerate web/bundles/ (downloads GPT-2 + SAEs)
python scripts/build_static.py     # dist/ static site
```

See [docs/ROADMAP.md](docs/ROADMAP.md) for the path from here to production scale and
[docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) for the running survey of interpretability tooling that feeds it.
