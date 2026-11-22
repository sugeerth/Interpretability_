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
    manual = M.intervene(model, M.Task(cor, t.targets, t.contrasts), [M.Hook("z", 1, patch)])
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


def test_circuit_discovery(model):
    from prism.circuit import corpus_circuit, discover_circuit, edge_attribution, faithfulness
    t = M.make_task(model, "hello world", "a", "b")
    cor = model.encode("jello world")
    ea = edge_attribution(model, t, cor, steps=2)
    S, L, H = ea["scores"], model.n_layers, model.n_heads
    assert S.shape == (1 + L * H + L, L * H + L + 1) and ea["sequential"]
    up, dn = ea["up"], ea["down"]
    assert float(S[up.index("a2.0"), dn.index("a1.0")]) == 0          # no edge backwards in depth
    assert float(S[up.index("a1.0"), dn.index("a1.1")]) == 0          # nor within a layer of heads
    assert float(S[up.index("m2"), dn.index("a2.0")]) == 0            # an MLP does not feed its own layer's heads
    assert float(S[up.index("a1.0"), dn.index("m1")]) != 0            # but a head feeds its own layer's MLP
    everything = set(up[1:])
    assert abs(faithfulness(model, t, cor, everything) - 1) < 1e-4 and abs(faithfulness(model, t, cor, set())) < 1e-4
    c = discover_circuit(model, t, cor, n_edges=20, steps=2, sizes=(2, 6), n_random=2)
    ids = {n["id"] for n in c["nodes"]}
    assert all(e["src"] in ids and e["dst"] in ids for e in c["edges"])
    reach = {"emb"}
    for _ in range(20):
        reach |= {e["dst"] for e in c["edges"] if e["src"] in reach}
    assert "out" in reach and ids <= reach                            # every kept node is reachable from the input
    json.dumps(c)
    cc = corpus_circuit(model, [(t, cor), (M.make_task(model, "hello there", "a", "b"), model.encode("jello there"))],
                        n_edges=20, steps=2, sizes=(4,), n_random=1)
    assert cc["n_prompts"] == 2 and "pos" not in cc["edges"][0] and 0 <= cc["edges"][0]["sign_agreement"] <= 1


def test_edge_scores_sum_to_node_effect(model):
    """Completeness: summing all edges into the output over upstream nodes recovers the metric gap
    (the output's input is linear in upstream outputs up to the final norm, which IG integrates)."""
    from prism.circuit import edge_attribution
    t = M.make_task(model, "hello world", "a", "b")
    ea = edge_attribution(model, t, model.encode("jello world"), steps=24)
    total = float(ea["scores"][:, ea["down"].index("out")].sum())
    assert abs(total - 1) < 0.1, total


def test_circuit_scoring():
    from prism.evals import score_circuit
    circ = dict(nodes=[dict(id=i, kind="head") for i in ("a5.5", "a8.6", "a9.9", "a1.1")],
                edges=[dict(src="a5.5", dst="a8.6"), dict(src="a8.6", dst="a9.9"), dict(src="a9.9", dst="out"),
                       dict(src="a9.9", dst="a8.6"), dict(src="a1.1", dst="a9.9")])
    s = score_circuit("ioi", circ)
    assert s["circuit_heads"] == 4 and s["head_precision"] == 0.75
    assert s["edges_between_reference_heads"] == 4 and s["edges_matching_paper_wiring"] == 3 and s["edges_unexplained"] == 1


def test_neuron_reads_tokens(model):
    prof = M.neuron_profile(model, 0, 1)
    assert prof["reads"] and len(prof["reads"]) == 8


def test_token_set_metric(model):
    t = M.make_task(model, "hello world", "a|b|c", ["x", "y"])
    assert len(t.targets) == 3 and len(t.contrasts) == 2 and t.is_set
    logits = model.forward(t.ids).logits
    lp = logits[0, -1].log_softmax(-1)
    expected = lp[t.targets].logsumexp(0) - lp[t.contrasts].logsumexp(0)       # log-odds of the two sets
    assert abs(float(t.metric(logits)) - float(expected)) < 1e-4
    p = M.predictions(model, t)
    assert p["target_set"]["n"] == 3 and 0 < p["target_set"]["prob"] < 1
    d = M.direct_logit_attribution(model, t)
    assert not d["exact_norm"] and abs(d["total"] - d["sum_components"]) < 1e-2    # still a full decomposition of its linearisation
    assert sum(M.Task(t.ids, [5]).readout_weights(logits[0, -1]).values()) == 1.0
    w = t.readout_weights(logits[0, -1])
    assert abs(sum(v for v in w.values() if v > 0) - 1) < 1e-5 and abs(sum(v for v in w.values() if v < 0) + 1) < 1e-5
    single = M.make_task(model, "hello world", "a", "b")
    assert not single.is_set and M.direct_logit_attribution(model, single)["exact_norm"]


def test_minimal_circuit_and_candidates(model):
    from prism.circuit import discover_circuit
    t = M.make_task(model, "hello world", "a", "b")
    cor = model.encode("jello world")
    c = discover_circuit(model, t, cor, n_edges=20, steps=2, n_random=1)
    mins = {x["threshold"]: x for x in c["faithfulness"]["minimal"]}
    assert set(mins) == {0.5, 0.8, 0.9}
    sizes = [mins[th]["size"] for th in (0.5, 0.8, 0.9) if mins[th]["size"]]
    assert sizes == sorted(sizes) and all(1 <= s <= c["n_components"] for s in sizes)
    hp = M.head_patching(model, t, cor, candidates=[(0, 1), (2, 3)])
    assert hp["measured"] == 2 and hp["heads"][0][1] is not None and hp["heads"][1][0] is None
    full = M.head_patching(model, t, cor)
    assert abs(full["heads"][2][3] - hp["heads"][2][3]) < 1e-3


def test_greater_than_dataset_and_reference():
    from prism.corpus import greater_than_dataset
    from prism.evals import score_circuit
    d = greater_than_dataset(5)
    assert all(len(x["target"]) + len(x["contrast"]) == 100 and x["prompt"] != x["corrupt"] for x in d)
    circ = dict(nodes=[dict(id="a5.5", kind="head"), dict(id="m9", kind="mlp"), dict(id="m2", kind="mlp")],
                edges=[dict(src="a5.5", dst="m9"), dict(src="m9", dst="out"), dict(src="m9", dst="a5.5")])
    s = score_circuit("greater_than", circ)
    assert s["circuit_heads"] == 3 and s["head_recall"] == round(2 / 12, 3) and s["edges_matching_paper_wiring"] == 2 and s["edges_unexplained"] == 1
