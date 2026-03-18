from typing import List, Dict
import json
from src.core.agent.base import LLMInterface
from src.core.agent.prompts import concept_canonicalization_template

class LLMCanonicalizer:
    """Use LLM to refine canonical concept names"""
    
    def __init__(self, llm_wrapper: LLMInterface):
        self.llm = llm_wrapper
        if self.llm:
            self.llm.set_prompt_template(concept_canonicalization_template, json_output=True)
    
    def canonicalize(self, concepts: List[str]) -> Dict[str, List[str]]:
        """Use LLM to merge similar concepts under canonical names"""
        try:
            response = self.llm.generate({"concepts": concepts})
            return json.loads(response)
        except Exception as e:
            print(f"LLM canonicalization failed: {e}")
            return {c: [c] for c in concepts}