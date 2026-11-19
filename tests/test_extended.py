import json

import torch

from prism import methods as M
from prism.atlas import head_roles
from prism.corpus import corpus_sweep, ioi_dataset
from prism.crosscoder import crosscoder_diff
from prism.evals import score_grids, score_report
from prism.neuronpedia import Neuronpedia
from prism.probe import concept_direction
from prism.store import BundleStore, Jobs
from prism.training import trace_finetune
from prism.workspace import Workspace


def test_head_patching_matches_manual_patch(model):
    t = M.make_task(model, "hello world", "a", "b")
    cor = model.encode("jello world")
    res = M.head_patching(model, t, cor)
    hp = torch.tensor(res["heads"])
    assert hp.shape == (model.n_layers, model.n_heads) and hp.abs().max() > 0
    clean, dh = model.forward(t.ids, cache=["z"]), model.d_z // model.n_heads

    def patch(z):                                   # copy head 1.2's clean output by hand
        z = z.clone(); z[..., 2 * dh:3 * dh] = clean.z[1][..., 2 * dh:3 * dh]; return z
    manual = M.intervene(model, M.Task(cor, t.target, t.contrast), [M.Hook("z", 1, patch)])
    gap = res["clean_metric"] - res["corrupt_metric"]
    assert abs((manual["metric_after"] - manual["metric_before"]) - float(hp[1, 2]) * gap) < 2e-3


def test_attention_rows_are_distributions(model):
    t = M.make_task(model, "hello world", "a", "b")
    for a in M.attention_patterns(model, t, [(0, 0), (2, 3)]):
        assert abs(sum(a["pattern"]) - 1) < 1e-2 and len(a["pattern"]) == t.ids.shape[1]


def test_neuron_profile_and_dose(model):
    prof = M.neuron_profile(model, 1, 3)
    assert len(prof["promotes"]) == 8 and prof["promote_logits"][0] >= prof["promote_logits"][-1]
    t = M.make_task(model, "hello world", "a", "b")
    curve = M.dose_response(model, t, lambda v: M.head_hook(model, 2, 0, v), [0.0, 1.0, 2.0])["curve"]
    base = float(t.metric(model.forward(t.ids).logits))
    assert abs(curve[1]["metric"] - base) < 1e-3 and curve[0]["metric"] != curve[2]["metric"]


def test_head_roles_shapes(model):
    r = head_roles(model, seq_len=10, n_seqs=2)
    for grid in r["scores"].values():
        assert len(grid) == model.n_layers and len(grid[0]) == model.n_heads
        assert all(0 <= v <= 1 for row in grid for v in row)


def test_concept_direction_recovers_planted_feature(model):
    """Prompts that end in different characters are trivially separable at layer 1+ on a random model."""
    pos = ["abca", "bbba", "ccca", "dada"]
    neg = ["abcz", "bbbz", "cccz", "dadz"]
    c = concept_direction(model, pos, neg, steer_prompt="hello")
    assert c["layers"][c["layer"]]["loo_accuracy"] == 1.0
    assert len(c["steering"]["curve"]) == 5 and "_direction" in c


def test_corpus_sweep_and_benchmark(model, tmp_path):
    ws = Workspace(neuronpedia=Neuronpedia(cache_dir=str(tmp_path), offline=True))
    ws.register(model)
    items = [dict(prompt="hello world", target="a", contrast="b", corrupt="jello world"),
             dict(prompt="hello there", target="a", contrast="b", corrupt="jello there"),
             dict(prompt="short", target="a", contrast="b", corrupt="a much longer corruption")]
    cs = corpus_sweep(ws, items, model="tiny")
    assert cs["n"] == 3 and cs["patching"]["n"] == 2 and cs["patch_skipped"] == 1
    assert len(cs["dla"]["mean"]) == model.n_layers
    json.dumps(cs)
    assert len(ioi_dataset(6)) == 6 and all(d["prompt"] != d["corrupt"] for d in ioi_dataset(6))


def test_benchmark_scoring():
    grid = [[0.0] * 12 for _ in range(12)]
    for l, h in [(9, 9), (9, 6), (10, 0), (10, 7), (11, 10)]:
        grid[l][h] = 1.0
    s = score_grids("ioi", grid, grid, k=5)
    assert s["dla_writer_recall"] == 1.0 and s["patching_upstream_recall"] == 0.0
    rep = score_report("ioi", "Heads 9.9 and 10.7 matter; 3.14 does not.")
    assert rep["by_class"]["name_mover"] == [[9, 9]] and [3, 14] in rep["unreferenced"]


def test_crosscoder_finds_planted_difference(model):
    """Fine-tune one MLP hard; the crosscoder must reconstruct both models and report a sane partition."""
    tr = trace_finetune(model, ["hello hello hello hello"], ["hello"], phases=1, steps_per_phase=6, lr=3e-2,
                        train_filter=lambda n: ".h.0.mlp" in n)
    texts = ["the quick brown fox jumps over the lazy dog and runs far away " * 3, "hello hello hello hello world again " * 5] * 6
    cd = crosscoder_diff(model, tr["tuned_model"], texts, layer=2, n_latents=128, k=8, steps=150, ctx=16, max_tokens=4000)
    assert cd["fve_base"] > 0.5 and cd["fve_tuned"] > 0.5
    assert sum(cd["counts"].values()) == cd["alive"] and sum(cd["relative_norm_hist"]) == cd["alive"]
    assert cd["most_shifted"] and cd["most_shifted"][0]["delta_norm"] > 0
    json.dumps(cd)


def test_store_and_jobs(tmp_path):
    st = BundleStore(str(tmp_path))
    calls = []
    f = lambda: calls.append(1) or dict(x=1)
    a = st.get_or_compute(dict(q="a"), f)
    b = st.get_or_compute(dict(q="a"), f)
    assert a == b and len(calls) == 1 and st.get(a["id"])["x"] == 1 and st.get("missing0") is None
    try:
        st.get("../etc")
        assert False
    except ValueError:
        pass
    jobs = Jobs()
    jid = jobs.submit("t", lambda progress: (progress(1, 2), dict(ok=True))[1])
    bad = jobs.submit("t", lambda progress: 1 / 0)
    import time
    for _ in range(100):
        if jobs.get(jid)["status"] == "done" and jobs.get(bad)["status"] == "error":
            break
        time.sleep(0.02)
    assert jobs.get(jid)["result"] == dict(ok=True) and "ZeroDivisionError" in jobs.get(bad)["error"]
