"""One model client for all roles; each role keeps its own prompt; the rate limit is global.

Before: every role (summary_generator, concept_extractor, concept_canonicalizer,
relationship_extractor, explainer) built its own client and its own rate limiter, so a limit of
4 requests/minute really meant up to 20.
"""
import json
import threading
import time

import pytest
from langchain_core.prompts import PromptTemplate

from doc_explainer.core.llm.factories.llm_factory import LLMFactory
from doc_explainer.core.llm.wrappers.airouter import AIRouterWrapper

from test_airouter_provider import FakeGateway

ROLES = ["knowledge", "tutor", "quiz", "extractor", "explainer"]   # any number of callers may ask for the same client


def make(role, base_url="http://127.0.0.1:9/v1", **extra):
    kw = dict(model_name="inclusionai/ling-3.0-flash", api_key="k", base_url=base_url, temperature=1.0,
              max_retries=0, instance_name=role)
    kw.update(extra)
    return LLMFactory.create("airouter", **kw)


def test_five_roles_initialise_the_model_client_once(monkeypatch):
    created = []
    real = AIRouterWrapper._create_model
    monkeypatch.setattr(AIRouterWrapper, "_create_model", lambda self, **kw: created.append(self.instance_name) or real(self, **kw))
    wrappers = {role: make(role) for role in ROLES}
    assert len(created) == 1                                           # ONE real initialisation, not five
    assert len({id(w.model) for w in wrappers.values()}) == 1          # the very same client object
    assert [w.instance_name for w in wrappers.values()] == ROLES       # but each role is still its own wrapper
    assert len({id(w) for w in wrappers.values()}) == 5


def test_roles_keep_their_own_prompt_so_none_overwrites_another():
    a, b = make("concept_extractor"), make("relationship_extractor")
    a.set_prompt_template(PromptTemplate.from_template("ALPHA {x}"), json_output=False)
    assert b.prompt_template is None and b.chain is None               # b is untouched by a's prompt
    b.set_prompt_template(PromptTemplate.from_template("BETA {x}"), json_output=False)
    assert a.prompt_template.template == "ALPHA {x}" and b.prompt_template.template == "BETA {x}"
    assert a.chain is not b.chain


def test_each_role_sends_its_own_prompt_over_the_one_shared_client():
    gw = FakeGateway(reply="ok")
    try:
        a, b = make("concept_extractor", gw.url), make("relationship_extractor", gw.url)
        assert a.model is b.model
        a.set_prompt_template(PromptTemplate.from_template("ALPHA {x}"))
        b.set_prompt_template(PromptTemplate.from_template("BETA {x}"))
        a.generate({"x": 1})
        b.generate({"x": 2})
        a.generate({"x": 3})
        sent = [json.dumps(r[2]["messages"]) for r in gw.requests]
        assert ["ALPHA" in s for s in sent] == [True, False, True]
        assert ["BETA" in s for s in sent] == [False, True, False]
    finally:
        gw.close()


def test_different_settings_get_their_own_client():
    base = make("a")
    assert make("b").model is base.model
    assert make("c", temperature=0.2).model is not base.model          # different temperature
    assert make("d", model_name="google/gemini-3-flash").model is not base.model
    assert make("e", api_key="another-key").model is not base.model    # a different account


def test_the_rate_limit_is_shared_by_every_role():
    a, b = make("summary_generator", min_request_interval_seconds=0.4), make("explainer", min_request_interval_seconds=0.4)
    a._last_request_at = time.monotonic()                              # a role just made a request ...
    t0 = time.monotonic()
    b._wait_for_rate_limit()                                           # ... so another must wait its turn
    assert time.monotonic() - t0 >= 0.3
    assert b._last_request_at == a._last_request_at                    # one clock for all


def test_concurrent_roles_are_spaced_out_not_bunched_together():
    ws = [make(r, min_request_interval_seconds=0.15) for r in ROLES[:4]]
    stamps, lock = [], threading.Lock()

    def go(w):
        w._wait_for_rate_limit()
        with lock:
            stamps.append(time.monotonic())

    threads = [threading.Thread(target=go, args=(w,)) for w in ws]
    [t.start() for t in threads]
    [t.join(5) for t in threads]
    stamps.sort()
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert len(stamps) == 4 and all(g >= 0.1 for g in gaps), gaps       # no two within the interval


def test_clear_shared_forgets_clients():
    first = make("a")
    LLMFactory.clear_shared()
    assert make("a").model is not first.model
