"""An empty or cut-off model answer is retried with a BIGGER budget, never with the same one."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from doc_explainer.core.common.ports import LLMFormatError
from doc_explainer.core.llm.wrappers.airouter import AIRouterWrapper
from doc_explainer.orchestrator.llm_client import LLMJsonClient


class Gateway:
    """An OpenAI-compatible endpoint that 'thinks' for ``need`` tokens: with a smaller budget the answer is empty."""

    def __init__(self, need, reply='{"ok": true}'):
        self.requests, outer = [], self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                budget = body.get("max_tokens") or body.get("max_completion_tokens")
                outer.requests.append(budget)
                cut = budget < outer.need
                payload = {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                           "choices": [{"index": 0, "finish_reason": "length" if cut else "stop",
                                        "message": {"role": "assistant", "content": "" if cut else reply}}],
                           "usage": {"prompt_tokens": 1, "completion_tokens": budget if cut else 5, "total_tokens": 9}}
                if body.get("stream"):
                    chunks = [] if cut else [{"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
                                              "choices": [{"index": 0, "delta": {"role": "assistant", "content": c}, "finish_reason": None}]}
                                             for c in ("Hel", "lo ", "there")]
                    data = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                else:
                    data = json.dumps(payload)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                data = data.encode()
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.need = need
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def client(gw, **kw):
    wrapper = AIRouterWrapper(model_name="m", api_key="k", base_url=gw.url, max_tokens=1000, max_retries=0,
                              max_tokens_cap=8000, token_growth=2.0, token_retries=2, **kw)
    return LLMJsonClient(wrapper)


@pytest.fixture
def gateway():
    made = []
    def make(need, **kw):
        made.append(Gateway(need, **kw))
        return made[-1]
    yield make
    for g in made:
        g.close()


def test_the_budgets_grow_up_to_the_cap(gateway):
    w = AIRouterWrapper(model_name="m", api_key="k", base_url=gateway(0).url, max_tokens=1000, max_tokens_cap=3000,
                        token_growth=2.0, token_retries=5)
    assert w.token_budgets() == [1000, 2000, 3000]                       # capped, and no pointless repeats of the cap
    assert AIRouterWrapper(model_name="m", api_key="k", base_url="http://x/v1", max_tokens=1000).token_budgets() == [1000]


def test_an_empty_answer_is_retried_with_a_bigger_budget_not_the_same_one(gateway):
    gw = gateway(need=1500)
    assert client(gw).ask_json("sys", "user") == {"ok": True}
    assert gw.requests == [1000, 2000]                                   # second request has twice the budget


def test_it_keeps_growing_until_the_answer_fits(gateway):
    gw = gateway(need=3500)
    assert client(gw).ask_json("s", "u") == {"ok": True}
    assert gw.requests == [1000, 2000, 4000]


def test_it_gives_up_at_the_cap_with_a_clear_error(gateway):
    gw = gateway(need=10**6)
    with pytest.raises(LLMFormatError, match="valid JSON"):
        client(gw).ask_json("s", "u")
    assert gw.requests == [1000, 2000, 4000]                              # three tries, each larger; then stops


def test_a_good_first_answer_makes_exactly_one_request(gateway):
    gw = gateway(need=0)
    client(gw).ask_json("s", "u")
    assert gw.requests == [1000]


def test_streamed_replies_also_escalate_when_nothing_arrives(gateway):
    gw = gateway(need=1500)
    text = "".join(client(gw).stream_text("s", "u"))
    assert text == "Hello there" and gw.requests == [1000, 2000]


def test_a_stream_that_has_started_is_never_restarted(gateway):
    gw = gateway(need=0)
    pieces = list(client(gw).stream_text("s", "u"))
    assert pieces and "".join(pieces) == "Hello there" and gw.requests == [1000]
