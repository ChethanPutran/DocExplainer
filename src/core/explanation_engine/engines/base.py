from abc import ABC, abstractmethod
from typing import Optional, Any
import logging

from src.core.agent import Agent
from src.core.agent.models.enums import ExplanationStyleEnum
from src.core.agent.models.schemas import Explanation
from ..base.interfaces import ExplanationEngine
from ..base.exceptions import GenerationError, ContextError

logger = logging.getLogger(__name__)


class BaseExplanationEngine(ExplanationEngine, ABC):
    """Base class for explanation engines"""
    
    def __init__(self, agent: Agent, default_level: ExplanationStyleEnum = ExplanationStyleEnum.INTERMEDIATE):
        self.agent = agent
        self.default_level = default_level
        self.current_level = default_level
    
    def summarize(self, text: str, context: Any) -> Explanation:
        """Generate summary using agent"""
        try:
            self._validate_context(context)
            return self.agent.summarize(text, context)
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            raise GenerationError(f"Failed to generate summary: {e}") from e
    
    def explain(self, text: str, context: Any) -> Explanation:
        """Generate explanation using agent"""
        try:
            self._validate_context(context)
            self.agent.set_explanation_style(self.current_level)
            return self.agent.explain(text, context)
        except Exception as e:
            logger.error(f"Explanation failed: {e}")
            raise GenerationError(f"Failed to generate explanation: {e}") from e
    
    def ask(self, question: str, context: Any) -> Explanation:
        """Answer question using agent"""
        try:
            self._validate_context(context)
            return self.agent.ask(question, context)
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            raise GenerationError(f"Failed to answer question: {e}") from e
    
    def set_explanation_level(self, level: ExplanationStyleEnum):
        """Set explanation level"""
        self.current_level = level
        self.agent.set_explanation_style(level)
        logger.info(f"Explanation level set to: {level.value}")
    
    def _validate_context(self, context: Any):
        """Validate context"""
        if context is None:
            raise ContextError("Context cannot be None")
    
    @abstractmethod
    def enrich_with_resources(self, explanation: Explanation) -> Explanation:
        """Enrich explanation with resources"""
        pass