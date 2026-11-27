"""Drive the Claude-planned loop end to end against a scripted HTTP transport.

No network and no API key: the real SDK builds and sends the requests, a mock
transport plays the model. This checks the loop's control flow, tool
execution and request shape; it cannot check that the live API accepts them.
"""
import json

import anthropic
import httpx2 as httpx

from prism.agent.loop import investigate
from prism.neuronpedia import Neuronpedia
from prism.workspace import Workspace


def message(content, stop_reason):
    return dict(id="msg_test", type="message", role="assistant", model="claude-opus-5-5", content=content,
                stop_reason=stop_reason, stop_sequence=None, usage=dict(input_tokens=10, output_tokens=10))


def sse(text):
    events = [("message_start", dict(type="message_start", message=message([], None))),
              ("content_block_start", dict(type="content_block_start", index=0, content_block=dict(type="text", text=""))),
              ("content_block_delta", dict(type="content_block_delta", index=0, delta=dict(type="text_delta", text=text))),
              ("content_block_stop", dict(type="content_block_stop", index=0)),
              ("message_delta", dict(type="message_delta", delta=dict(stop_reason="end_turn", stop_sequence=None), usage=dict(output_tokens=5))),
              ("message_stop", dict(type="message_stop"))]
    return "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in events)


def test_llm_loop_runs_tools_and_review(model, tmp_path):
    script = [
        message([dict(type="text", text="Starting with a sweep."),
                 dict(type="tool_use", id="tu_1", name="sweep", input=dict(prompt="hello world", target="a", contrast="b", corrupt="jello world"))], "tool_use"),
        message([dict(type="tool_use", id="tu_2", name="intervene", input=dict(kind="head", layer=2, index=0)),
                 dict(type="tool_use", id="tu_3", name="get_section", input=dict(name="circuit"))], "tool_use"),
        message([dict(type="tool_use", id="tu_4", name="record_finding",
                      input=dict(claim="Head 2.0 matters.", evidence="Ablation moved the metric.", confidence="high"))], "tool_use"),
        message([dict(type="text", text="Final report: head 2.0 writes the answer.")], "end_turn"),
    ]
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((request, body))
        if body.get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse("Review: the claim rests on one ablation."))
        return httpx.Response(200, json=script[len([1 for _, b in seen if not b.get("stream")]) - 1])

    client = anthropic.Anthropic(api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=0)
    ws = Workspace(neuronpedia=Neuronpedia(cache_dir=str(tmp_path), offline=True))
    ws.register(model)
    steps = []
    res = investigate("How does it answer?", ws, "tiny", on_step=steps.append, client=client)

    assert res["report"] == "Final report: head 2.0 writes the answer." and res["review"].startswith("Review:")
    assert [s["tool"] for s in res["steps"]] == ["sweep", "intervene", "get_section", "record_finding"]
    assert res["findings"] == [dict(claim="Head 2.0 matters.", evidence="Ablation moved the metric.", confidence="high")]
    assert res["bundle"]["circuit"]["edges"] and any(s["tool"] == "note" for s in steps)

    first_req, first = seen[0]
    assert first["model"] == "claude-opus-5-5" and first["output_config"] == {"effort": "high"} and first["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in first_req.headers["anthropic-beta"]
    assert {t["name"] for t in first["tools"]} >= {"sweep", "intervene", "dose_response", "head_roles", "concept_direction", "record_finding"}
    assert "thinking" not in first or first["thinking"].get("type") == "adaptive"
    # tool results go back in one user message per turn, matched to their tool_use ids
    third = seen[2][1]["messages"][-1]
    assert third["role"] == "user" and [b["tool_use_id"] for b in third["content"]] == ["tu_2", "tu_3"]
    assert json.loads(third["content"][0]["content"] if isinstance(third["content"][0]["content"], str) else third["content"][0]["content"][0]["text"])["kind"] == "head"
    review = seen[-1][1]
    assert review["stream"] and "<experiment_log>" in review["messages"][0]["content"] and "tools" not in review
