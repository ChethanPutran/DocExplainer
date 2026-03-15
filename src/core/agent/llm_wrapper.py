from typing import Dict, List
from langchain.messages import SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.core.agent.agent import model

from typing import Any, Dict
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from src.core.agent.agent import model
from typing import Any, Dict, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.core.agent.agent import model

class LLMWrapper:
    def __init__(self, prompt_template: Optional[PromptTemplate] = None, output_parser: Any = None, json_output: bool = False):
        self.model = model
        self.prompt_template = prompt_template
        if json_output:
            self.parser = JsonOutputParser()
        else:
            self.parser = output_parser if output_parser else StrOutputParser()
        self._rebuild_chain()

    def _rebuild_chain(self):
        """Internal helper to update the LCEL chain whenever components change."""
        if self.prompt_template:
            self.chain = self.prompt_template | self.model | self.parser
        else:
            self.chain = None

    def set_prompt(self, template: str):
        """Update the prompt template dynamically."""
        self.prompt_template = PromptTemplate.from_template(template)
        self._rebuild_chain()

    def set_parser(self, output_parser: Any):
        """Update the output parser dynamically (e.g., switch to JsonOutputParser)."""
        self.parser = output_parser
        self._rebuild_chain()

    def get_model(self):
        return model
    
    def generate(self, inputs: Dict[str, Any]) -> Any:
        if not self.chain:
            raise ValueError("Prompt not set. Call set_prompt() before generating.")
        
        try:
            return self.chain.invoke(inputs)
        except Exception as e:
            return f"Generation Error: {str(e)}"