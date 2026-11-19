import json

from prism.agent.loop import Investigation, build_tools, investigate_offline
from prism.neuronpedia import Neuronpedia
from prism.workspace import Workspace, summarize


def ws_with(model, tmp_path):
    ws = Workspace(neuronpedia=Neuronpedia(cache_dir=str(tmp_path), offline=True))
    ws.register(model)
    return ws


def test_sweep_bundle_and_summary(model, tmp_path):
    ws = ws_with(model, tmp_path)
    b = ws.sweep("hello world", model="tiny", target="a", contrast="b", corrupt="jello world")
    for key in ("predictions", "logit_lens", "input_attribution", "dla", "attention", "patching", "head_patching", "verification", "findings"):
        assert key in b
    json.dumps(b)                                                # bundle must be plain JSON
    assert len(b["logit_lens"]["grid"]) == model.n_layers + 1
    assert summarize(b)[0]["title"] == "Prediction"
    bad = ws.sweep("hello world", model="tiny", corrupt="much longer corrupted prompt")
    assert "error" in bad["patching"]


def test_offline_investigation(model, tmp_path):
    steps = []
    res = investigate_offline("hello world", ws_with(model, tmp_path), "tiny", "a", "b", on_step=steps.append)
    assert res["mode"] == "offline" and res["report"] and steps[0]["tool"] == "sweep"


def test_agent_tools_execute(model, tmp_path):
    inv = Investigation(ws_with(model, tmp_path), "tiny")
    tools = {t.name: t for t in build_tools(inv)}
    assert set(tools) == {"sweep", "get_section", "intervene", "dose_response", "head_roles", "concept_direction", "lookup_feature",
                          "search_features", "blackbox_occlusion", "record_finding"}
    out = json.loads(tools["sweep"].call(dict(prompt="hello world", target="a", contrast="b")))
    assert out["tokens"][0] == "<bos>"
    res = json.loads(tools["intervene"].call(dict(kind="head", layer=1, index=0)))
    assert "metric_after" in res
    err = json.loads(tools["intervene"].call(dict(kind="feature", layer=1, index=0)))
    assert "error" in err                                         # no SAE registered for the tiny model: reported, not raised
    curve = json.loads(tools["dose_response"].call(dict(kind="head", layer=1, index=0, values=[0, 1, 2])))["curve"]
    assert len(curve) == 3
    assert "top" in json.loads(tools["head_roles"].call({}))
    con = json.loads(tools["concept_direction"].call(dict(positive=["abca", "bbba", "ccca"], negative=["abcz", "bbbz", "cccz"])))
    assert "layers" in con and "_direction" not in con
    tools["record_finding"].call(dict(claim="c", evidence="e", confidence="low"))
    assert inv.findings[0]["claim"] == "c" and [s["tool"] for s in inv.steps] == [
        "sweep", "intervene", "intervene", "dose_response", "head_roles", "concept_direction", "record_finding"]
