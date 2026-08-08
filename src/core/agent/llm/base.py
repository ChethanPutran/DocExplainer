from abc import ABC, abstractmethod
import re
import time
from typing import Dict, Any, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import RunnableSequence

from ..base.interfaces import LLMInterface


class BaseLLM(LLMInterface, ABC):
    """Base class for LLM wrappers"""
    
    def __init__(
        self,
        model_name: str = "default",
        temperature: float = 0.7,
        requests_per_minute: Optional[int] = None,
        min_request_interval_seconds: Optional[float] = None,
        rate_limit_retries: int = 2,
        **kwargs
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.requests_per_minute = requests_per_minute
        self.min_request_interval_seconds = min_request_interval_seconds
        self.rate_limit_retries = rate_limit_retries
        self._last_request_at = 0.0
        self.model = self._create_model(**kwargs)
        self.prompt_template: Optional[PromptTemplate] = None
        self.parser: Optional[BaseOutputParser] = None
        self.chain: Optional[RunnableSequence] = None
    
    @abstractmethod
    def _create_model(self, **kwargs) -> BaseLanguageModel:
        """Create the underlying language model"""
        pass
    
    def set_prompt_template(self, template: PromptTemplate, json_output: bool = False):
        """Set prompt template and rebuild chain"""
        self.prompt_template = template
        
        if json_output and not self.parser:
            from langchain_core.output_parsers import JsonOutputParser
            self.parser = JsonOutputParser()
        
        self._rebuild_chain()
    
    def set_parser(self, parser: BaseOutputParser):
        """Set output parser and rebuild chain"""
        self.parser = parser
        self._rebuild_chain()
    
    def _rebuild_chain(self):
        """Rebuild the LCEL chain"""
        if self.prompt_template and self.model:
            if self.parser:
                self.chain = self.prompt_template | self.model | self.parser
            else:
                from langchain_core.output_parsers import StrOutputParser
                self.chain = self.prompt_template | self.model | StrOutputParser()
        else:
            self.chain = None

    def _wait_for_rate_limit(self) -> None:
        """Throttle outbound LLM calls to stay under provider RPM limits."""
        intervals = []

        if self.requests_per_minute and self.requests_per_minute > 0:
            intervals.append(60.0 / self.requests_per_minute)

        if self.min_request_interval_seconds and self.min_request_interval_seconds > 0:
            intervals.append(self.min_request_interval_seconds)

        if not intervals:
            return

        wait_seconds = max(intervals) - (time.monotonic() - self._last_request_at)
        if wait_seconds > 0:
            time.sleep(wait_seconds)

    @staticmethod
    def _retry_delay_from_error(error: Exception) -> Optional[float]:
        """Extract provider retry hints from quota error messages when present."""
        message = str(error)
        match = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+(?:\.\d+)?)s", message)
        if match:
            return float(match.group(1)) + 1.0

        match = re.search(r"Please retry in\s+(\d+(?:\.\d+)?)s", message)
        if match:
            return float(match.group(1)) + 1.0

        return None
    
    def generate(self, inputs: Dict[str, Any]) -> Any:
        """Generate response from LLM"""
        if not self.chain:
            raise ValueError("Chain not built. Set prompt template first.")
        
        attempts = self.rate_limit_retries + 1

        for attempt in range(attempts):
            self._wait_for_rate_limit()
            try:
                result = self.chain.invoke(inputs)
                self._last_request_at = time.monotonic()
                return result
            except Exception as e:
                self._last_request_at = time.monotonic()
                retry_delay = self._retry_delay_from_error(e)
                is_rate_limited = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)

                if attempt < attempts - 1 and (retry_delay or is_rate_limited):
                    time.sleep(retry_delay or 15.0)
                    continue

                raise RuntimeError(f"Generation failed: {e}") from e
    
    def get_model(self) -> BaseLanguageModel:
        """Get underlying model"""
        return self.model
