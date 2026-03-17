from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from langchain_core.runnables import RunnableSequence

from ..base.interfaces import ChainInterface
from ..llm.base import BaseLLM
from ..parsers.base import BaseParser


class BaseChain(ChainInterface, ABC):
    """Base class for processing chains"""
    
    def __init__(self, llm: BaseLLM, parser: Optional[BaseParser] = None):
        self.llm = llm
        self.parser = parser
        self.chain: Optional[RunnableSequence] = None
        self.max_retries = 1
    
    @abstractmethod
    def _build_chain(self) -> RunnableSequence:
        """Build the chain"""
        pass
    
    def run(self, **kwargs) -> Any:
        """Run the chain"""
        if not self.chain:
            self.chain = self._build_chain()
        
        try:
            return self.chain.invoke(kwargs)
        except Exception as e:
            if self.max_retries > 1:
                return self._run_with_retry(**kwargs)
            raise e
    
    def _run_with_retry(self, **kwargs) -> Any:
        """Run with retry logic"""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return self.chain.invoke(kwargs)
            except Exception as e:
                last_error = e
                print(f"Attempt {attempt + 1} failed: {e}")
        
        raise last_error
    
    def with_retry(self, max_retries: int = 3) -> 'BaseChain':
        """Add retry capability"""
        self.max_retries = max_retries
        return self