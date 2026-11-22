import torch

from prism import methods as M
from prism.backend import Hook
from prism.blackbox import LocalScorer, occlusion, segment, surrogate_agreement
from prism.diff import activation_diff, trace_diff, weight_diff
from prism.sae import SAE, feature_attribution, feature_hook
from prism.training import trace_finetune


def task(model, text="hello world", target="a", contrast="b"):
    return M.make_task(model, text, target, contrast)


def test_forward_cache_shapes(model):
    ids = model.encode("abcdef")
    c = model.forward(ids, cache=["resid_pre", "attn_out", "mlp_out", "z", "neurons"])
    assert len(c.resid_pre) == model.n_layers and c.resid_pre[0].shape == (1, ids.shape[1], model.d_model)
    assert c.z[0].shape[-1] == model.d_z and c.neurons[0].shape[-1] == model.d_mlp
    assert torch.allclose(c.logits, model.model(ids).logits, atol=1e-5)          # hooks must not change the output


def test_residual_stream_decomposes(model):
    c = model.forward(model.encode("abcdef"), cache=["resid_pre", "attn_out", "mlp_out"])
    total = c.resid_pre[0] + sum(c.attn_out) + sum(c.mlp_out)
    assert torch.allclose(total, c.resid_final, atol=1e-4)


def test_dla_is_exact(model):
    t = task(model)
    d = M.direct_logit_attribution(model, t)
    assert d["exact_norm"] and abs(d["total"] - d["sum_components"]) < 1e-2
    heads = torch.tensor(d["heads"]).sum(-1)                                      # heads + bias = attention layer
    assert heads.shape[0] == model.n_layers
    # the frozen-norm read-out reproduces the true metric up to the unembedding bias (GPT-2 has none)
    assert abs(d["total"] - M.predictions(model, t)["metric"]) < 1e-2


def test_integrated_gradients_completeness(model):
    ia = M.input_attribution(model, task(model))
    assert ia["ig_relative_error"] <= 0.05 and ia["baseline"] == "mean token embedding"
    assert abs(sum(ia["integrated_gradients"]) - (ia["metric"] - ia["baseline_metric"]) - ia["ig_completeness_error"]) < 1e-2


def test_patching_endpoints(model):
    t = task(model, "hello world")
    res = M.activation_patching(model, t, model.encode("jello world"))
    g = torch.tensor(res["resid"])
    assert g.shape == (model.n_layers, t.ids.shape[1])
    assert abs(float(g[0, 1]) - 1) < 1e-2          # patching the only differing token at layer 0 restores everything
    assert abs(float(g[0, 3])) < 1e-2              # patching an identical position changes nothing


def test_interventions(model):
    t = task(model)
    same = M.intervene(model, t, [M.head_hook(model, 1, 0, scale=1.0)])
    assert abs(same["metric_after"] - same["metric_before"]) < 1e-4 and same["kl"] < 1e-6
    abl = M.intervene(model, t, [M.head_hook(model, 2, 0), M.neuron_hook(2, 3)])
    assert abl["metric_after"] != abl["metric_before"]
    v = torch.randn(model.d_model)
    assert M.intervene(model, t, [M.direction_hook(1, v, add=3.0)])["kl"] > 0


def test_head_ablation_matches_dla_in_last_layer(model):
    """A last-layer head only acts through its direct write, so ablation ≈ −DLA up to the final-norm rescale."""
    t = task(model)
    d = M.direct_logit_attribution(model, t)
    L = model.n_layers - 1
    h = max(range(model.n_heads), key=lambda i: abs(d["heads"][L][i]))
    res = M.intervene(model, t, [M.head_hook(model, L, h, pos=t.ids.shape[1] - 1)])
    measured, predicted = res["metric_after"] - res["metric_before"], -d["heads"][L][h]
    assert measured * predicted > 0 and abs(measured - predicted) < 0.5 * abs(predicted) + 0.02


def random_sae(model, layer, n=96):
    g = torch.Generator().manual_seed(layer)
    W = torch.randn(model.d_model, n, generator=g) * 0.3
    return SAE(W, torch.zeros(n), W.T.clone(), torch.zeros(model.d_model), layer)


def test_feature_attribution_effect_is_exact(model):
    t = task(model)
    saes = [random_sae(model, 1), random_sae(model, 2)]
    out = feature_attribution(model, t, saes, k=5, graph_k=3, candidates=6, max_recon_err=1e9)
    f = out["layers"][1]["features"][0]
    res = M.intervene(model, t, [feature_hook(saes[1], f["index"], pos=f["pos"])])
    assert abs((res["metric_before"] - res["metric_after"]) - f["effect"]) < 1e-3
    ids = {n["id"] for n in out["graph"]["nodes"]}
    assert all(e["src"] in ids and e["dst"] in ids for e in out["graph"]["edges"])


def test_blackbox(model):
    assert "".join(segment("a b  c.")) == "a b  c."
    occ = occlusion(LocalScorer(model, "a", "b"), "hello big world")
    assert len(occ["attribution"]) == 3 and occ["n_queries"] == 4
    assert surrogate_agreement([1, 2, 3], [2, 4, 9])["spearman"] == 1.0


def test_diff_and_training(model):
    assert weight_diff(model, model)["n_changed"] == 0
    tr = trace_finetune(model, ["hello hello hello"], ["hello", "world"], phases=2, steps_per_phase=2, lr=1e-2,
                        train_filter=lambda n: ".h.1.mlp" in n)
    last = tr["phases"][-1]
    assert tr["loss"][-1] < tr["loss"][0]
    mlp = last["weights"]["per_layer"]["mlp"]
    assert mlp[1] > 0 and mlp[0] == 0 and mlp[2] == 0                             # the diff localises the edit
    assert all(w["name"].startswith("transformer.h.1.mlp") for w in last["weights"]["top"])
    shift = last["activations"]["rel_shift"]
    assert shift[0] == 0 and shift[1] == 0 and shift[2] > 0                       # nothing moves before the edited layer
    assert activation_diff(model, model, ["hello"])["mean_kl"] < 1e-6


def test_trace_diff():
    a = [dict(tool="sweep", input=1), dict(tool="intervene", input=2)]
    b = [dict(tool="sweep", input=1), dict(tool="lookup", input=0), dict(tool="intervene", input=3)]
    d = trace_diff(a, b)
    assert d["counts"] == dict(same=1, changed=1, only_a=0, only_b=1) and d["first_divergence"] == 1


def test_hook_api(model):
    ids = model.encode("abc")
    zero = model.forward(ids, hooks=[Hook("mlp_out", 0, lambda t: t * 0)])
    assert not torch.allclose(zero.logits, model.forward(ids).logits)


def test_llama_architecture():
    """The backend finds blocks, projections and the RMSNorm read-out by structure, not by model name."""
    from transformers import LlamaConfig, LlamaForCausalLM
    from conftest import CharTok
    from prism.backend import WhiteBoxModel
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=32)
    m = WhiteBoxModel(LlamaForCausalLM(cfg), CharTok(), name="tiny-llama")
    t = M.make_task(m, "hello world", "a", "b")
    d = M.direct_logit_attribution(m, t)
    assert d["exact_norm"] and abs(d["total"] - M.predictions(m, t)["metric"]) < 1e-2
    assert abs(d["total"] - d["sum_components"]) < 1e-2
    g = M.activation_patching(m, t, m.encode("jello world"))["resid"]
    assert abs(g[0][0] - 1) < 1e-2 and abs(g[0][3]) < 1e-2
    assert M.intervene(m, t, [M.head_hook(m, 2, 1), M.neuron_hook(1, 5)])["kl"] > 0
