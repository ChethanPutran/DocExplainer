from typing import Optional, Dict, Any
import logging
from datetime import datetime

from .llm.factories.llm_factory import LLMFactory
from .llm.base import BaseLLM
from .parsers.explanation_parser import explanation_output_parser
from .parsers.retry_parser import RetryParser
from .chains.explanation_chain import ExplanationChain
from .chains.summarization_chain import SummarizationChain
from .chains.qa_chain import QAChain
from .models.enums import ExplanationStyleEnum
from .models.schemas import Explanation, ExplanationMetadata, ExplanationPydantic
from .models.dataclasses import Resource
from .config import AgentConfig

logger = logging.getLogger(__name__)


class Agent:
    """Main agent class for handling all LLM interactions"""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        
        # Initialize LLM
        self.llm = LLMFactory.create(
            provider=self.config.llm_provider,
            temperature=self.config.temperature,
            **self.config.llm_kwargs
        )
        
        # Initialize retry parser
        self.retry_parser = RetryParser(
            parser=explanation_output_parser,
            llm=self.llm.get_model(),
            max_retries=self.config.max_retries
        )
        
        # Initialize chains
        self.explanation_chain = ExplanationChain(self.llm, self.retry_parser)
        self.summarization_chain = SummarizationChain(self.llm)
        self.qa_chain = QAChain(self.llm)
        
        # Default style
        self.explanation_style = self.config.default_style
    
    def summarize(self, text: str, context: Any) -> Explanation:
        """Generate summary for text"""
        inputs = self._prepare_common_inputs(context)
        inputs["selected_text"] = text
        inputs["structure"] = "Bullet points"
        inputs["length"] = "Concise summary"
        
        start_time = datetime.now()
        
        try:
            result = self.summarization_chain.run(**inputs)
            return self._build_explanation(result, text, context, start_time)
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return self._build_fallback_response(
                f"I encountered an error while summarizing: {str(e)}",
                text, context
            )
    
    def explain(self, text: str, context: Any) -> Explanation:
        """Generate explanation for text"""
        inputs = self._prepare_common_inputs(context)
        inputs["selected_text"] = text
        
        start_time = datetime.now()
        
        try:
            result = self.explanation_chain.run(**inputs)
            return self._build_explanation(result, text, context, start_time)
        except Exception as e:
            logger.error(f"Explanation failed: {e}")
            return self._build_fallback_response(
                f"I encountered an error while explaining: {str(e)}",
                text, context
            )
    
    def ask(self, question: str, context: Any) -> Explanation:
        """Answer question"""
        inputs = self._prepare_common_inputs(context)
        inputs["question"] = question
        inputs["selected_text"] = self._get_document_text(context)
        
        start_time = datetime.now()
        
        try:
            result = self.qa_chain.run(**inputs)
            return self._build_explanation(result, question, context, start_time)
        except Exception as e:
            logger.error(f"QA failed: {e}")
            return self._build_fallback_response(
                f"I encountered an error while answering: {str(e)}",
                question, context
            )
    
    def _prepare_common_inputs(self, context: Any) -> Dict[str, Any]:
        """Prepare common inputs for chains"""
        known, unknown = self._extract_user_knowledge(context)
        
        return {
            "context_summary": self._get_context_snippet(context),
            "known_concepts": ", ".join(known) if known else "Fundamental basics",
            "unknown_concepts": ", ".join(unknown) if unknown else "None identified yet",
            "tone": self.config.tone,
            "complexity": self.explanation_style.value,
            "math_level": self.config.math_level,
            "format_instructions": explanation_output_parser.get_format_instructions()
        }
    
    def _extract_user_knowledge(self, context: Any) -> tuple:
        """Extract known and unknown concepts from user context"""
        known = []
        unknown = []
        
        if context and hasattr(context, 'user_knowledge'):
            states = getattr(context.user_knowledge, 'knowledge_states', {})
            
            for concept, state in states.items():
                # BKT Logic: High knowledge + high confidence = Known
                if state.p_knowledge > 0.7 and state.confidence > 0.6:
                    known.append(concept.name)
                # Low knowledge + significant attempts = Knowledge Gap
                elif state.p_knowledge < 0.4 and state.n_attempts > 0:
                    unknown.append(concept.name)
        
        return known, unknown
    
    def _get_context_snippet(self, context: Any) -> str:
        """Get context snippet"""
        if not context or not hasattr(context, 'document_context'):
            return "No specific document context available."
        
        doc_context = context.document_context
        text = ""
        
        if isinstance(doc_context, dict):
            text = doc_context.get("text", "")
        else:
            text = getattr(doc_context, "text", "")
        
        return text[:300].strip() + "..." if text else "Context empty."
    
    def _get_document_text(self, context: Any) -> str:
        """Get document text from context"""
        if not context or not hasattr(context, 'document_context'):
            return ""
        
        doc_context = context.document_context
        if isinstance(doc_context, dict):
            return doc_context.get("text", "")
        return getattr(doc_context, "text", "")
    
    def _build_explanation(self, result: Any, source: str, 
                          context: Any, start_time: datetime) -> Explanation:
        """Build explanation from chain result"""
        # Handle different result types
        if isinstance(result, ExplanationPydantic):
            parsed = result
        elif isinstance(result, dict):
            parsed = ExplanationPydantic(**result)
        else:
            return self._build_fallback_response(str(result), source, context)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        parsed.metadata.processing_time_ms = processing_time
        
        return Explanation(
            explanation=parsed.explanation,
            style=parsed.style,
            context_used=parsed.context_used,
            known_concepts_used=parsed.known_concepts_used,
            unknown_concepts_explained=parsed.unknown_concepts_explained,
            follow_up_questions=parsed.follow_up_questions,
            metadata=parsed.metadata,
            suggested_resources=parsed.suggested_resources,
            resources=[]  # To be filled by resource recommender
        )
    
    def _build_fallback_response(self, message: str, source: str, context: Any) -> Explanation:
        """Build fallback response when LLM fails"""
        return Explanation(
            explanation=message,
            style={"mode": self.explanation_style.value, "depth": "fixed"},
            context_used={"error_state": True},
            known_concepts_used=[],
            unknown_concepts_explained=[],
            follow_up_questions=[
                "Could you try rephrasing the question?",
                "Would you like a simpler explanation?",
                "Should I focus on a specific part?"
            ],
            metadata=ExplanationMetadata(
                estimated_complexity=self._estimate_complexity(source),
                user_knowledge_matched=False,
                gap_bridging=False,
                processing_time_ms=0
            ),
            suggested_resources=[],
            resources=[]
        )
    
    def _estimate_complexity(self, text: str) -> float:
        """Estimate text complexity"""
        words = len(text.split())
        if words < 20:
            return 0.2
        if words < 80:
            return 0.5
        return 0.8
    
    def set_explanation_style(self, style: ExplanationStyleEnum):
        """Set explanation style"""
        self.explanation_style = style
    
    def get_llm(self) -> BaseLLM:
        """Get underlying LLM"""
        return self.llm