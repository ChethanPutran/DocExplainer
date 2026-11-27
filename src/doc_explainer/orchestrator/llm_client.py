"""Adapts an LLM wrapper to the ``JsonLLM`` and ``TextLLM`` ports.

Rate limiting and retry-after handling stay in the wrapper. What this adds: when an answer comes back empty or
cut off (a reasoning model that spent its budget thinking), it asks again with a BIGGER output budget, up to the
configured cap, instead of repeating the same request.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Iterator

from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import PromptTemplate

from ..core.common.ports import LLMError, LLMFormatError
from ..core.llm.base import BaseLLM

logger = logging.getLogger(__name__)
_TEMPLATE = "{system}\n\n{user}"


def _is_format_error(e: BaseException) -> bool:
    """A parser failure, however deeply the wrapper wrapped it (BaseLLM.generate re-raises as RuntimeError)."""
    while e is not None:
        if isinstance(e, OutputParserException):
            return True
        e = e.__cause__ or e.__context__
    return False


def _text_of(chunk: Any) -> str:
    piece = getattr(chunk, "content", chunk)
    if isinstance(piece, str):
        return piece
    if isinstance(piece, list):  # content blocks: keep the text ones, never the model's reasoning
        return "".join(b.get("text", "") if isinstance(b, dict) and b.get("type") in (None, "text") else (b if isinstance(b, str) else "")
                       for b in piece)
    return ""


class LLMJsonClient:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        # one generic template for every task: the task is described in the text, so a single wrapper can be shared
        llm.set_prompt_template(PromptTemplate.from_template(_TEMPLATE), json_output=True)
        self._variants: Dict[int, BaseLLM] = {}
        self._lock = threading.Lock()

    def _for_budget(self, budget):
        """The wrapper for one output budget (the configured one is ``self.llm`` itself)."""
        if budget is None or budget == self.llm.max_tokens:
            return self.llm
        with self._lock:
            variant = self._variants.get(budget)
            if variant is None:
                variant = self._variants[budget] = self.llm.with_max_tokens(budget)
                variant.set_prompt_template(PromptTemplate.from_template(_TEMPLATE), json_output=True)
        return variant

    def ask_json(self, system: str, user: str) -> Dict[str, Any]:
        budgets = self.llm.token_budgets()
        last: Exception = LLMFormatError("The model did not return valid JSON.")
        for i, budget in enumerate(budgets):
            try:
                result = self._for_budget(budget).generate({"system": system, "user": user})
            except Exception as e:
                if not _is_format_error(e):
                    raise LLMError(f"{type(e).__name__}: {str(e)[:300]}") from e
                last = LLMFormatError(f"The model did not return valid JSON (output budget {budget}): {str(e)[:200]}")
                if i + 1 < len(budgets):
                    logger.warning("Empty or invalid JSON with max_tokens=%s; retrying with max_tokens=%s", budget, budgets[i + 1])
                continue
            if not isinstance(result, dict):
                raise LLMFormatError(f"Expected a JSON object, got {type(result).__name__}")
            return result
        raise last

    def stream_text(self, system: str, user: str) -> Iterator[str]:
        """The reply as it is written. An empty reply is retried with a bigger budget before any text was shown."""
        budgets = self.llm.token_budgets()
        for i, budget in enumerate(budgets):
            llm = self._for_budget(budget)
            llm._wait_for_rate_limit()
            produced = False
            try:
                for chunk in llm.get_model().stream(_TEMPLATE.format(system=system, user=user)):
                    piece = _text_of(chunk)
                    if piece:
                        produced = True
                        yield piece
            except ValueError as e:
                if produced or "No generation chunks" not in str(e):   # (langchain's way of saying: nothing was written)
                    raise LLMError(f"{type(e).__name__}: {str(e)[:300]}") from e
            except Exception as e:
                if produced:
                    raise LLMError(f"The reply was cut off: {type(e).__name__}: {str(e)[:200]}") from e
                raise LLMError(f"{type(e).__name__}: {str(e)[:300]}") from e
            finally:
                llm._last_request_at = time.monotonic()
            if produced:
                return
            if i + 1 < len(budgets):
                logger.warning("Empty reply with max_tokens=%s; retrying with max_tokens=%s", budget, budgets[i + 1])
        raise LLMFormatError("The model returned an empty reply.")
