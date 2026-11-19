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

## Phase 1 — Model coverage

| Work | Why | Exit test |
|---|---|---|
| Register Gemma Scope / Gemma Scope 2, Llama Scope and Qwen SAE releases; read the SAELens config format generically (JumpReLU, TopK, input normalisation) | Feature views beyond GPT-2 | Reconstruction error under 15% on 1k held-out tokens per registered release, checked in CI nightly |
| Transcoder support and direct-effect graphs (frozen attention + norm, as in circuit-tracer) | Current edges are total effects; direct effects give real circuits | Graph for the two-hop "capital of the state containing Dallas" prompt matches the circuit-tracer reference on Gemma-2-2b |
| Remote execution backend via nnsight / NDIF for models too large to host | 70B-class open models without owning the GPUs | Same sweep bundle from local and remote backends on a shared model, within numerical tolerance |
| Half precision, device maps, batched patching by layer | Memory and latency on 7B+ models | Sweep of a 24-token prompt on an 8B model in under 60 s on one 24 GB GPU |

## Phase 2 — Scale

- **Job queue.** Replace the in-process lock with a queue (Redis + workers pinned one-per-GPU); the API
  returns a job id and streams progress. Bundles go to object storage keyed by request hash.
- **Corpus sweeps.** Run the sweep over datasets, not single prompts; store per-feature and per-head
  statistics in a columnar store (Parquet + DuckDB) so questions like "which features drive refusals
  across 50k prompts" are a query.
- **Activation cache.** Persist residual activations for a reference corpus once per checkpoint so diffing
  and probing never re-run the model.
- Exit test: 100k-prompt sweep on a 2B model completes unattended; p95 single-prompt latency under 5 s
  from cache-warm workers at 20 concurrent users.

## Phase 3 — Training-time interpretability

- Callback for Hugging Face `Trainer` / Lightning that snapshots and diffs on a schedule and raises
  alerts on unexpected drift (a feature family appearing, collateral KL on a held-out probe set).
- Crosscoder-based diffing (BatchTopK, with the latent-scaling checks from Minder et al.) for
  base-vs-fine-tuned pairs; Delta-Crosscoder for narrow fine-tunes.
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
- Parallel sub-investigations (one per candidate mechanism) with a reviewer pass that tries to falsify
  each claim before it reaches the report.
- Agent-run evaluation: a fixed suite of known circuits (IOI, greater-than, induction, docstring) scored
  on whether the agent's report names the right components; run on every release.
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
