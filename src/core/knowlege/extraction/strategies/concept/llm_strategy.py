from typing import List
import json
from ..base import BaseConceptExtractionStrategy
from src.core.agent.llm_wrapper import LLMWrapper
from src.core.agent.prompts import concept_refinement_template

class LLMConceptRefinementStrategy(BaseConceptExtractionStrategy):
    """Refine concepts using LLM"""
    
    def __init__(self, llm_wrapper: LLMWrapper):
        self.llm = llm_wrapper
        if self.llm:
            self.llm.set_prompt_template(concept_refinement_template, json_output=True)
    
    def extract(self, text: str, candidates: List[str]) -> List[str]:
        """
        Clean candidate phrases using LLM.
        Splits compound concepts and removes generic phrases.
        """
        try:
            response = self.llm.generate({'candidates': candidates, 'context': text})
            refined = []
            for item in response:
                refined.append(item["name"])
            return refined
        except Exception as e:
            print(f"LLM refinement failed: {e}")
            return candidates