from abc import ABC, abstractmethod
import copy
import logging
import re
import threading
import time
from typing import Dict, Any, Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import RunnableSequence

from ..common.llm import LLMInterface


logger = logging.getLogger(__name__)


class _RateLimiter:
    """When the last request was sent. Shared by every wrapper over the same model client, because the
    provider's limit applies to the account, not to the part of the program that happens to call."""

    def __init__(self) -> None:
        self.last = 0.0
        self.lock = threading.Lock()


class BaseLLM(LLMInterface, ABC):
    """Base class for LLM wrappers."""

    def __init__(
        self,
        model_name: str = "default",
        temperature: float = 0.7,
        requests_per_minute: Optional[int] = None,
        min_request_interval_seconds: Optional[float] = None,
        rate_limit_retries: int = 2,
        instance_name="default",
        max_tokens_cap: Optional[int] = None,
        token_growth: float = 2.0,
        token_retries: int = 0,
        **kwargs
    ):
        self.instance_name = instance_name

        self.logger = logging.getLogger(
            f"{__name__}.{self.instance_name}"
        )
    
        self.model_name = model_name
        self.temperature = temperature
        self.requests_per_minute = requests_per_minute
        self.min_request_interval_seconds = min_request_interval_seconds
        self.rate_limit_retries = rate_limit_retries
        # the wrapper may have set its budget before calling us (as _max_tokens or max_tokens); keep it
        self.max_tokens = kwargs.get("max_tokens") or getattr(self, "_max_tokens", None) or getattr(self, "max_tokens", None) or 1000
        # When an answer comes back empty or cut off, retry with a bigger budget (never the same one again).
        self.max_tokens_cap, self.token_growth, self.token_retries = max_tokens_cap, token_growth, token_retries
        self._model_kwargs = dict(kwargs)

        self._limiter = _RateLimiter()

        self.logger.info(
            "Initializing LLM: model=%s, temperature=%s, "
            "rpm=%s, min_interval=%s, retries=%s, max_tokens=%s",
            self.model_name,
            self.temperature,
            self.requests_per_minute,
            self.min_request_interval_seconds,
            self.rate_limit_retries,
            self.max_tokens,
        )

        try:
            self.model = self._create_model(**kwargs)

            self.logger.info(
                "LLM model created successfully: model=%s",
                self.model_name,
            )

        except Exception:
            self.logger.exception(
                "Failed to create LLM model: model=%s",
                self.model_name,
            )
            raise

        self.prompt_template: Optional[PromptTemplate] = None
        self.parser: Optional[BaseOutputParser] = None
        self.chain: Optional[RunnableSequence] = None

        self.logger.debug(
            "BaseLLM initialization complete: model=%s",
            self.model_name,
        )

    @abstractmethod
    def _create_model(self, **kwargs) -> BaseLanguageModel:
        """Create the underlying language model."""
    pass

    # `_last_request_at` lives on the (shared) limiter so that all roles throttle together.
    @property
    def _last_request_at(self) -> float:
        return self._limiter.last

    @_last_request_at.setter
    def _last_request_at(self, value: float) -> None:
        self._limiter.last = value

    def token_budgets(self) -> list:
        """The output budgets to try in turn: the configured one, then growing ones up to the cap."""
        base = self.max_tokens
        if not base:
            return [base]
        out, cap = [base], self.max_tokens_cap or base
        for _ in range(self.token_retries):
            nxt = min(int(out[-1] * self.token_growth), cap)
            if nxt <= out[-1]:
                break
            out.append(nxt)
        return out

    def with_max_tokens(self, max_tokens: int) -> "BaseLLM":
        """The same model with a different output budget (shares the rate limiter; has its own chain)."""
        clone = copy.copy(self)
        clone.max_tokens = max_tokens
        if hasattr(clone, "_max_tokens"):
            clone._max_tokens = max_tokens
        clone.model = clone._create_model(**{k: v for k, v in self._model_kwargs.items() if k != "max_tokens"})
        clone._rebuild_chain()
        return clone

    def fork(self, instance_name: str) -> "BaseLLM":
        """A per-role view of this LLM.

        It shares the expensive, stateless parts (the model client and the rate limiter) but has its
        own prompt, parser, chain and logger. A wrapper has exactly one prompt slot, so roles that
        use different prompts must not share a wrapper; they can share everything else.
        """
        clone = copy.copy(self)
        clone.instance_name = instance_name
        clone.logger = logging.getLogger(f"{__name__}.{instance_name}")
        clone.prompt_template = None
        clone.parser = None
        clone.chain = None
        return clone

    def set_prompt_template(
        self,
        template: PromptTemplate,
        json_output: bool = False,
    ):
        """Set prompt template and rebuild chain."""

        self.logger.info(
            "Setting prompt template: model=%s, json_output=%s",
            self.model_name,
            json_output,
        )

        self.prompt_template = template

        if json_output and not self.parser:
            self.logger.debug(
                "Creating JsonOutputParser: model=%s",
                self.model_name,
            )

            from langchain_core.output_parsers import JsonOutputParser

            self.parser = JsonOutputParser()

        self._rebuild_chain()

    def set_parser(self, parser: BaseOutputParser):
        """Set output parser and rebuild chain."""

        self.logger.info(
            "Setting output parser: model=%s, parser=%s",
            self.model_name,
            type(parser).__name__,
        )

        self.parser = parser

        self._rebuild_chain()

    def _rebuild_chain(self):
        """Rebuild the LCEL chain."""

        if self.prompt_template and self.model:

            self.logger.debug(
                "Rebuilding LCEL chain: model=%s, parser=%s",
                self.model_name,
                type(self.parser).__name__ if self.parser else "StrOutputParser",
            )

            if self.parser:
                self.chain =  self.prompt_template | self.model  | self.parser
            else:
                from langchain_core.output_parsers import StrOutputParser

                self.chain = self.prompt_template | self.model | StrOutputParser()
                

            self.logger.info(
                "LCEL chain built successfully: model=%s",
                self.model_name,
            )

        else:
            self.chain = None

            self.logger.warning(
                "Could not build LCEL chain: "
                "prompt_template=%s, model=%s",
                bool(self.prompt_template),
                bool(self.model),
            )

    def _wait_for_rate_limit(self) -> None:
        """Throttle outbound LLM calls to stay under provider RPM limits."""

        intervals = []

        if (
            self.requests_per_minute
            and self.requests_per_minute > 0
        ):
            intervals.append(
                60.0 / self.requests_per_minute
            )

        if (
            self.min_request_interval_seconds
            and self.min_request_interval_seconds > 0
        ):
            intervals.append(
                self.min_request_interval_seconds
            )

        if not intervals:
            self.logger.debug(
                "Rate limiting disabled: model=%s",
                self.model_name,
            )
            return

        required_interval = max(intervals)

        with self._limiter.lock:  # roles may call from different threads: take the next slot atomically
            now = time.monotonic()
            start = max(now, self._limiter.last + required_interval)
            self._limiter.last = start  # reserved even when idle, so concurrent callers queue up
            wait_seconds = start - now

        if wait_seconds > 0:
            self.logger.info(
                "Rate limit throttling: model=%s, "
                "waiting=%.2fs",
                self.model_name,
                wait_seconds,
            )

            time.sleep(wait_seconds)

   
    def _retry_delay_from_error(
            self,
        error: Exception,
    ) -> Optional[float]:
        """Extract provider retry hints from quota error messages."""

        message = str(error)

        match = re.search(
            r"retryDelay['\"]?:\s*['\"]?(\d+(?:\.\d+)?)s",
            message,
        )

        if match:
            delay = float(match.group(1)) + 1.0

            self.logger.debug(
                "Provider retry delay detected: %.2fs",
                delay,
            )

            return delay

        match = re.search(
            r"Please retry in\s+(\d+(?:\.\d+)?)s",
            message,
        )

        if match:
            delay = float(match.group(1)) + 1.0

            self.logger.debug(
                "Provider retry delay detected: %.2fs",
                delay,
            )

            return delay

        return None

    def generate(self, inputs: Dict[str, Any]) -> Any:
        """Generate response from LLM."""

        if not self.chain:
            self.logger.error(
                "Generation requested but chain is not built: "
                "model=%s",
                self.model_name,
            )

            raise ValueError(
                "Chain not built. Set prompt template first."
            )

        attempts = self.rate_limit_retries + 1

        self.logger.info(
            "Starting LLM generation: model=%s, attempts=%s",
            self.model_name,
            attempts,
        )

        self.logger.debug(
            "Generation inputs: model=%s, input_keys=%s inputs=%s",
            self.model_name,
            list(inputs.keys()),
            inputs,
        )

        for attempt in range(attempts):
            current_attempt = attempt + 1

            logger.info(
                "LLM request attempt %d/%d: model=%s",
                current_attempt,
                attempts,
                self.model_name,
            )

            self._wait_for_rate_limit()

            request_start = time.monotonic()

            try:
                result = self.chain.invoke(inputs)

                elapsed = (
                    time.monotonic() - request_start
                )

                self._last_request_at = time.monotonic()

                logger.info(
                    "LLM generation successful: "
                    "model=%s, attempt=%d/%d, duration=%.2fs",
                    self.model_name,
                    current_attempt,
                    attempts,
                    elapsed,
                )

                logger.debug(
                    "LLM response type: model=%s, type=%s, result=%s",
                    self.model_name,
                    type(result).__name__,
                    result,
                )

                return result

            except Exception as e:
                elapsed = (
                    time.monotonic() - request_start
                )

                self._last_request_at = time.monotonic()

                retry_delay = (
                    self._retry_delay_from_error(e)
                )

                is_rate_limited = (
                    "429" in str(e)
                    or "RESOURCE_EXHAUSTED" in str(e)
                )

                logger.warning(
                    "LLM generation failed: "
                    "model=%s, attempt=%d/%d, "
                    "duration=%.2fs, rate_limited=%s, "
                    "retry_delay=%s",
                    self.model_name,
                    current_attempt,
                    attempts,
                    elapsed,
                    is_rate_limited,
                    retry_delay,
                )

                if (
                    attempt < attempts - 1
                    and (retry_delay or is_rate_limited)
                ):
                    delay = retry_delay or 15.0

                    logger.info(
                        "Retrying LLM request in %.2fs: "
                        "model=%s, next_attempt=%d/%d",
                        delay,
                        self.model_name,
                        current_attempt + 1,
                        attempts,
                    )

                    time.sleep(delay)
                    continue

                logger.exception(
                    "LLM generation failed permanently: "
                    "model=%s, attempts=%d",
                    self.model_name,
                    attempts,
                )

                raise RuntimeError(
                    f"Generation failed: {e}"
                ) from e

    def get_model(self) -> BaseLanguageModel:
        """Get underlying model."""

        logger.debug(
            "Returning underlying LLM model: model=%s",
            self.model_name,
        )

        return self.model
