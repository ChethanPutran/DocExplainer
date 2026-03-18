from typing import Optional
from src.core.agent.base import LLMInterface
from src.core.agent.prompts import text_summarization_template


class SummaryGenerator:
    """Generates summaries for document chunks"""
    
    def __init__(self, llm_wrapper: Optional[LLMInterface] = None):
        self.llm = llm_wrapper
        if self.llm:
            self.llm.set_prompt_template(text_summarization_template)
        self.rolling_context = ""
    
    def generate_summary(self, text: str, context: str = "", max_context: int = 2500) -> str:
        """
        Generate summary for text with context
        
        If LLM is not available, returns a placeholder summary
        """
        if not self.llm:
            return f"Summary of: {text[:50]}..."
        
        recent_context = context[-max_context:] if context else ""
        
        try:
            return self.llm.generate({
                "current_text": text,
                "recent_context": recent_context
            }).strip()
        except Exception as e:
            print(f"Summary generation failed: {e}")
            return f"Summary of: {text[:50]}..."
    
    def update_context(self, text: str):
        """Update rolling context"""
        self.rolling_context += text + "\n"
        # Keep context manageable
        if len(self.rolling_context) > 10000:
            self.rolling_context = self.rolling_context[-5000:]
    
    def reset_context(self):
        """Reset rolling context"""
        self.rolling_context = ""