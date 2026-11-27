"""The AIRouter wrapper: factory wiring and the real HTTP request it makes.

AIRouter is an OpenAI-compatible gateway. The wire test points the wrapper at a local fake server, so
the exact request (path, bearer key, model, token limit) is checked without calling a paid service.
(Where its model, endpoint and key COME FROM is tested in test_config_layers.py.)
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from langchain_core.prompts import PromptTemplate

from doc_explainer.core.llm.factories.llm_factory import LLMFactory
from doc_explainer.core.llm.wrappers.airouter import AIRouterWrapper


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("AIROUTER_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)


# ----------------------------------------------------------------------------- factory / wrapper
def test_factory_knows_the_provider():
    assert "airouter" in LLMFactory.get_available_providers()
    w = LLMFactory.create("airouter", model_name="google/gemini-3-flash", api_key="k", base_url="http://x/v1")
    assert isinstance(w, AIRouterWrapper) and w.model_name == "google/gemini-3-flash"


def test_wrapper_without_a_key_fails_with_a_clear_message():
    with pytest.raises(ValueError, match="AIROUTER_API_KEY"):
        AIRouterWrapper(model_name="google/gemini-3-flash", base_url="http://x/v1")


def test_wrapper_falls_back_to_the_environment_key_but_never_invents_an_endpoint_or_model(monkeypatch):
    monkeypatch.setenv("AIROUTER_API_KEY", "env-key")
    w = AIRouterWrapper(model_name="google/gemini-3-flash", base_url="http://x/v1")
    assert w.model.openai_api_key.get_secret_value() == "env-key"
    with pytest.raises(ValueError, match="needs an endpoint"):
        AIRouterWrapper(model_name="google/gemini-3-flash")
    with pytest.raises(TypeError):
        AIRouterWrapper(base_url="http://x/v1")                  # no hidden default model


def test_the_key_never_appears_in_repr_or_logs(caplog):
    with caplog.at_level("DEBUG"):
        w = AIRouterWrapper(model_name="m", api_key="super-secret-key", base_url="http://x/v1")
    assert "super-secret-key" not in repr(w.model) and "super-secret-key" not in caplog.text


# ----------------------------------------------------------------------------- the real HTTP request
class FakeGateway:
    """A local OpenAI-compatible endpoint that records what it receives."""

    def __init__(self, status=200, reply="pong"):
        self.requests, outer = [], self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
                payload = {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                           "choices": [{"index": 0, "finish_reason": "stop",
                                        "message": {"role": "assistant", "content": reply}}],
                           "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
                data = json.dumps(payload if status == 200 else {"error": {"message": "bad key"}}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def gateway():
    g = FakeGateway()
    yield g
    g.close()


def test_the_request_goes_to_the_gateway_with_the_bearer_key_model_and_token_limit(gateway):
    w = AIRouterWrapper(model_name="google/gemini-3-flash", api_key="air-key", base_url=gateway.url,
                        max_tokens=50, max_retries=0)
    assert w.model.invoke("hello").content == "pong"
    path, headers, body = gateway.requests[0]
    assert path == "/v1/chat/completions"
    assert headers["authorization"] == "Bearer air-key"
    assert body["model"] == "google/gemini-3-flash"
    assert (body.get("max_tokens") or body.get("max_completion_tokens")) == 50
    assert "reasoning" not in body                      # the OpenRouter-only parameter is not sent


def test_the_whole_wrapper_chain_works_against_the_gateway(gateway):
    w = AIRouterWrapper(model_name="google/gemini-3-flash", api_key="k", base_url=gateway.url, max_retries=0)
    w.set_prompt_template(PromptTemplate.from_template("Say {word}"))
    assert "pong" in str(w.generate({"word": "ping"}))
    assert "Say ping" in json.dumps(gateway.requests[0][2]["messages"])


def test_a_rejected_key_surfaces_as_an_error_not_a_hang():
    g = FakeGateway(status=401)
    try:
        w = AIRouterWrapper(model_name="m", api_key="wrong", base_url=g.url, max_retries=0)
        with pytest.raises(Exception) as e:
            w.model.invoke("hello")
        assert "401" in str(e.value) or "bad key" in str(e.value) or "Authentication" in str(e.value)
    finally:
        g.close()


def test_ten_per_minute_means_one_request_every_six_seconds_across_all_roles():
    a = LLMFactory.create("airouter", model_name="m", api_key="k", base_url="http://x/v1",
                          requests_per_minute=10, instance_name="a")
    b = LLMFactory.create("airouter", model_name="m", api_key="k", base_url="http://x/v1",
                          requests_per_minute=10, instance_name="b")
    assert a._limiter is b._limiter and a.requests_per_minute == 10
    assert 60.0 / a.requests_per_minute == pytest.approx(6.0)
