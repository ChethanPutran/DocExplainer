from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import RunnableSequence

from ..base.interfaces import LLMInterface


class BaseLLM(LLMInterface, ABC):
    """Base class for LLM wrappers"""
    
    def __init__(self, model_name: str = "default", temperature: float = 0.7, **kwargs):
        self.model_name = model_name
        self.temperature = temperature
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
    
    def generate(self, inputs: Dict[str, Any]) -> Any:
        """Generate response from LLM"""
        if not self.chain:
            raise ValueError("Chain not built. Set prompt template first.")
        
        try:
            return self.chain.invoke(inputs)
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}") from e
    
    def get_model(self) -> BaseLanguageModel:
        """Get underlying model"""
        return self.model