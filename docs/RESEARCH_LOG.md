# Research log

A running survey of interpretability tooling, newest first. Each entry says what was found, what it means
for Prism, and what (if anything) was done about it. Entries are added by the scheduled research session.

## 2026-11-19 — baseline survey

| Finding | Source | Relevance | Action |
|---|---|---|---|
| Anthropic's circuit-tracing library builds attribution graphs for open-weights models from pre-trained transcoders; Neuronpedia hosts the interactive front end. Tested up to 14B-parameter models. | [Anthropic](https://anthropic.com/research/open-source-circuit-tracing), [BlackboxNLP 2025 paper](https://aclanthology.org/2025.blackboxnlp-1.14) | Prism's graph uses SAE features and total-effect gradient edges. Transcoder graphs give direct effects and are the stronger method. | Roadmap phase 1: transcoder support and direct-effect graphs. |
| Neuronpedia (April 2026): public beta of an Interpretability Explorer directory; an OpenMOSS circuit-tracing extension with LORSA and QK tracing; circuit-tracer gained Top-K transcoder support. | [Neuronpedia blog](https://www.neuronpedia.org/blog/interp-explorer) | Attention-side (QK) attribution is missing in Prism: we explain what heads write, not why they attend. | Candidate for a later cycle: QK attribution for the top heads. |
| Crosscoder model diffing: the L1 loss can mislabel shared concepts as fine-tune-specific; BatchTopK crosscoders largely fix this. | [Minder et al., NeurIPS 2025](https://arxiv.org/abs/2504.02922v2) | Prism's feature diff reads both checkpoints through the base SAE, which cannot represent genuinely new directions. | Roadmap phase 3: BatchTopK crosscoder diffing with the paper's checks. |
| Delta-Crosscoder for narrow fine-tunes, evaluated on Gemma, Llama and Qwen from 1B to 7B. | [ICLR 2026](https://iclr.cc/virtual/2026/10019264) | Matches the fine-tune tracer's use case (small, localised edits). | Roadmap phase 3. |
| Anthropic Fellows diff tool extends model diffing to models with different architectures (March 2026). | [Anthropic](https://www.anthropic.com/research/diff-tool) | Prism's weight diff needs identical architectures; activation-level diffing across architectures would lift that. | Noted; revisit after crosscoder support. |

### Follow-up, same day: what was built from this survey

| Source idea | Implemented as | Result on GPT-2 small |
|---|---|---|
| Crosscoder diffing with BatchTopK ([Minder et al.](https://arxiv.org/abs/2504.02922v2)) | `prism/crosscoder.py` | On a 20-step fine-tune, every latent stays shared: the edit re-writes existing concepts instead of adding new ones. The shifted-latent ranking is the useful output in that regime, which is the gap Delta-Crosscoder targets. |
| Attention-side tracing ([Neuronpedia blog](https://www.neuronpedia.org/blog/interp-explorer)) | Attention read-out per top head; exact per-head patching | Over 24 prompts, head patching recovers S-inhibition heads 7.9, 8.6, 8.10 and induction heads 5.5, 6.9, none of which direct attribution shows (7.3 falls just outside the top 10). Full QK attribution remains open. |
| Circuit benchmarks from circuit-tracing work ([paper](https://aclanthology.org/2025.blackboxnlp-1.14)) | `prism/evals.py` | All five answer-writing heads found; 60% of upstream heads found; previous-token heads missed at top-10. |

Not verified in this survey: current release versions of SAELens, TransformerLens and nnsight. Check their
release notes directly before depending on new APIs.
