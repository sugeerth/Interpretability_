# Roadmap: inception to production

Each phase has an exit test. A phase is done when the test passes, not when the code exists.

## Phase 0 — Foundations (done)

One prompt, one metric, every method, one JSON bundle.

- Architecture-agnostic hook backend; six attribution methods; SAE features tied to Neuronpedia ids.
- Self-verifying sweep (predicted vs. measured ablation effects).
- Model diffing and a fine-tune tracer; black-box occlusion for API-only models.
- Agent with a Claude-planned loop and an offline protocol; web app; static demo; Docker image; CI.

Exit test (met): the indirect-object sweep on GPT-2 small recovers the published circuit components
(name movers 9.6 / 9.9 / 10.0, negative name movers 10.7 / 11.10) and the fine-tune diff localises an
edit to the layers that were trained.

## Phase 0.5 — Second wave (done)

Ideas taken from the tools surveyed in the research log and implemented here:

| Borrowed from | What was built |
|---|---|
| TransformerLens head detectors | Model atlas: previous-token, duplicate-token and induction scores for every head |
| Path patching (Wang et al.) | Exact per-head patching, which surfaces upstream heads that direct attribution cannot see |
| Crosscoder model diffing (Minder et al., BatchTopK) | A crosscoder trained on base and tuned activations together, with a leakage check and a shifted-latent ranking for narrow fine-tunes |
| Contrastive activation addition | Concept directions with held-out scoring, SAE matching and steering |
| Sparse feature circuits | Integrated-gradient feature attribution, with exact ablation as the ranking key |
| Neuronpedia dashboards | Feature inspector: label, promoted and suppressed tokens, top activations, live dose-response |
| OpenMOSS QK tracing (partially) | Attention read-out for the top heads; full QK attribution is still open |
| Circuit benchmarks | Recall against the published indirect-object circuit, for the pipeline and for agent reports |
| Multi-agent review | A reviewer pass that tries to falsify the agent's report |

Also added: a second model with Neuronpedia-aligned SAEs (Pythia-70m), corpus sweeps, a result store with
shareable links, and background jobs.

Exit test (met): on 24 indirect-object prompts the pipeline finds all five answer-writing heads from the paper by
direct attribution, and head patching recovers three of the four S-inhibition heads and both induction heads (60% of all upstream heads). The atlas independently labels 5.5 and 6.9 as
induction heads and 4.11 as a previous-token head, matching the literature.

## Phase 0.75 — Circuits and a third architecture (done)

- **Automated circuit discovery** (`prism/circuit.py`): edge attribution patching with integrated gradients over
  every input → head → MLP → output edge, pruned to paths from input to output, per prompt or averaged over a task.
- **Faithfulness as a first-class output**: every circuit reports how much behaviour survives when everything
  outside it is corrupted, next to random component sets of the same size.
- **Circuit benchmark**: discovered nodes and wiring scored against the published indirect-object circuit.
- **Qwen2.5-0.5B** (24 layers, RMSNorm, 494M parameters) runs the full pipeline with no SAE.
- **Integrated gradients fixed**: mean-embedding baseline and adaptive steps, after the Qwen run showed the zero
  baseline was badly off.
- **Agent loop tested end to end** through the real SDK against a scripted transport.
- **Neuron weight readings from input to output** (which tokens a neuron reads, which it promotes).

Exit test (met): averaged over 12 indirect-object prompts on GPT-2 small, the discovered circuit contains 11 of the
paper's 15 heads; 19 of its 28 edges between reference heads follow the paper's wiring (6 more connect name movers,
3 are not in the paper); its top 32 of 156 components keep 89% of the behaviour, against 2% for random sets.

Known weak spots, measured: on Qwen the top 32 of 360 components keep only 35%, and integrated gradients still
misses its completeness check there; on Pythia-70m the faithfulness test does not discriminate (random sets keep 53%).

## Phase 0.9 — A second benchmark, token-set metrics, minimal circuits (done)

- **Token-set metrics**: the explained quantity can be the log-odds of one set of tokens against another, so
  behaviours that are not about a single word can be studied with every method.
- **Greater-than benchmark** (Hanna et al. 2023) alongside indirect object identification; `prism bench` reproduces both.
- **Minimal circuits**: the smallest ranked component set that keeps 50 / 80 / 90% of the behaviour.
- **Scalable head patching**: on models with more than 160 heads, all heads are ranked by the two-pass gradient
  estimate and only the top candidates are verified exactly.
- **Deployment fix**: the server could not find its web assets when installed as a package (as in the Docker
  image); fixed and checked with a clean install.

Exit test (met), GPT-2 small, 12 prompts per task:

| Task | Reference components found | Edges following the paper's wiring | Components for 90% of the behaviour |
|---|---|---|---|
| Indirect object | 11 of 15 | 19 of 28 | 35 of 156 |
| Greater-than | 11 of 12 | 23 of 24 | 24 of 156 |

Still weak, measured: Qwen2.5-0.5B needs 159 of 360 components for 90% on the indirect-object prompt, so the
ranking does not yet isolate a compact circuit there.

## Phase 0.95 — Stability and summary scores (done, cycle of 2026-11-25)

Task-level circuits report how much they depend on the prompts sampled (bootstrap overlap and per-edge selection
frequency), and every circuit reports the area under its faithfulness curve. On 12 prompts per task the resampled
circuits overlap 0.81–0.82. Next candidates: EAP-GP integration paths for the weak Qwen result; edge-level pruning.

## Phase 0.97 — Attribution proposes, exact pruning disposes (done, 2026-11-26)

- Two component rankings are derived from the same edge scores (total flow, and signed total effect); the one whose
  faithfulness curve stays closest to the full model is used and both scores are reported.
- After the ranked prefix reaches 90%, each component is tested for removal by exact measurement, keeping
  faithfulness within 10% of the full model on either side (so components that push against the answer are not
  pruned away).

Measured: components that keep about 90% of the behaviour fell from 30 to 23 of 156 on the indirect-object task,
from 24 to 20 on greater-than, and from 159 to 48 of 360 on Qwen2.5-0.5B (13% of the model). The previous Qwen
weak spot is largely closed; integrated gradients on Qwen is not.

## Phase 0.99 — Gemma 3 and a public page (done, 2026-11-27)

- **Gemma 3 1B** with Gemma Scope 2 residual SAEs (JumpReLU) at four layers, labelled through Neuronpedia. Checked:
  the loaded dictionary is the folder Neuronpedia names, and activations on its own top examples match (217.7 vs 217.65).
- **Post-norm blocks** (Gemma family): block outputs are read after their output norm and each head and neuron is
  attributed through that run's frozen scale, so the decomposition stays exact.
- **Public page** at https://sugeerth.github.io/prism/ built with `scripts/build_static.py --site`; every tab passes
  axe (WCAG 2.1 A/AA and best practice) in light, dark and phone layouts.

This starts Phase 1's first item. Still to do there: larger Gemma and Llama models, transcoders, a remote backend.

## Phase 1 — Model coverage

| Work | Why | Exit test |
|---|---|---|
| Register Gemma Scope / Gemma Scope 2, Llama Scope and Qwen SAE releases; read the SAELens config format generically (JumpReLU, TopK, input normalisation) | Feature views beyond GPT-2 | Reconstruction error under 15% on 1k held-out tokens per registered release, checked in CI nightly |
| Transcoder support and direct-effect graphs (frozen attention + norm, as in circuit-tracer) | Current edges are total effects; direct effects give real circuits | Graph for the two-hop "capital of the state containing Dallas" prompt matches the circuit-tracer reference on Gemma-2-2b |
| Remote execution backend via nnsight / NDIF for models too large to host | 70B-class open models without owning the GPUs | Same sweep bundle from local and remote backends on a shared model, within numerical tolerance |
| Half precision, device maps, batched patching by layer | Memory and latency on 7B+ models | Sweep of a 24-token prompt on an 8B model in under 60 s on one 24 GB GPU |

## Phase 2 — Scale

- **Job queue.** A single-process job runner and a content-addressed result store exist. Next: move the queue to
  Redis with workers pinned one-per-GPU, and the store to object storage.
- **Corpus sweeps.** Corpus sweeps exist for tens of prompts. Next: datasets of 10k+ prompts; store per-feature and per-head
  statistics in a columnar store (Parquet + DuckDB) so questions like "which features drive refusals
  across 50k prompts" are a query.
- **Activation cache.** Persist residual activations for a reference corpus once per checkpoint so diffing
  and probing never re-run the model.
- Exit test: 100k-prompt sweep on a 2B model completes unattended; p95 single-prompt latency under 5 s
  from cache-warm workers at 20 concurrent users.

## Phase 3 — Training-time interpretability

- Callback for Hugging Face `Trainer` / Lightning that snapshots and diffs on a schedule and raises
  alerts on unexpected drift (a feature family appearing, collateral KL on a held-out probe set).
- Crosscoder diffing exists at demo scale (2k latents, 30k tokens, CPU). Next: train on millions of tokens
  on GPU, add the full latent-scaling test from Minder et al., and Delta-Crosscoder for narrow fine-tunes.
- Checkpoint lineage view in the web app: any two points of a run, side by side.
- Exit test: on a model organism with a planted behaviour, the diff surfaces the planted feature in the
  top 10 without being told what to look for.

## Phase 4 — Semi-open models

- Surrogate pipeline: explain an API-only model with an open model's internals, gated by measured
  agreement between the two models' black-box attribution maps on the same inputs.
- Use every signal a provider exposes: log-probabilities, logit bias, reasoning summaries, structured
  outputs for forced-choice scoring; batch APIs for cost.
- Behavioural diffing between model versions (the black-box analogue of model diff).
- Exit test: on prompts where surrogate agreement exceeds a threshold, surrogate-derived edits
  (removing the top-attributed span) change the hosted model's answer at a rate well above a random-span
  baseline.

## Phase 5 — Agent autonomy

- Long-horizon investigations with memory across sessions, a hypothesis ledger, and automatic
  re-verification when the model under study changes.
- Parallel sub-investigations, one per candidate mechanism. (The reviewer pass that tries to falsify the
  report exists.)
- Agent-run evaluation: the indirect-object benchmark and report scorer exist. Next: add greater-than,
  induction and docstring references and run the suite on every release.
- Exit test: the agent recovers the documented components on at least 3 of 4 reference circuits without
  hints, and flags its own unverified claims as such.

## Phase 6 — Production hardening

Authentication and per-user quotas, audit log of interventions, model allow-lists per deployment,
autoscaling GPU workers, cost dashboards for hosted-model usage, signed bundles so shared results can
be trusted, and a public gallery of investigations.

## Standing loop

Every four days a scheduled session surveys new interpretability tools and papers, records what it finds
in `docs/RESEARCH_LOG.md`, and opens a pull request implementing the most valuable item that fits the
current phase.
