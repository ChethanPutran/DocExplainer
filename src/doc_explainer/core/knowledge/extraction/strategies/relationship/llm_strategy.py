from typing import List, Dict
from ..base import BaseRelationshipExtractionStrategy
from .....knowledge.models.concept import Concept
from .....knowledge.models.relationship import ConceptRelationship
from .....agent import LLMInterface
from .....agent.prompts import relation_extractor_prompt

class LLMRelationshipExtractor(BaseRelationshipExtractionStrategy):
    """Extract relationships using LLM"""
    
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        if self.llm:
            self.llm.set_prompt_template(relation_extractor_prompt, json_output=True)
        
        # Define high-value pedagogical relationships and their synonyms
        self.global_relations = {
            "uses": "depends_on",
            "relies_on": "depends_on",
            "implements": "implements",
            "based_on": "depends_on",
            "built_on": "depends_on",
            "results_in": "results_in",
            "is_a": "is_a",
            "part_of": "part_of",
            "enables": "enables",
        }
    
    def _get_or_create_relationship(self, name: str) -> str:
        """Map various relation phrases to standardized types"""
        name_normalized = name.lower().strip()
        if name_normalized not in self.global_relations:
            self.global_relations[name_normalized] = name_normalized
        return self.global_relations[name_normalized]
    
    def extract(self, concepts: List[Concept], text: str, context: str) -> List[ConceptRelationship]:
        """Extract relationships using LLM"""
        try:
            concept_names = [c.name for c in concepts]
            response = self.llm.generate(inputs={
                'text': text, 
                'context': context, 
                'concept_names': concept_names
            })

            relations: List[ConceptRelationship] = []
            name_to_obj = {c.name: c for c in concepts}

            for item in response:
                c1 = name_to_obj.get(item["source"])
                c2 = name_to_obj.get(item["target"])
                if c1 and c2:
                    relation = item["relation"]
                    relation_type = self._get_or_create_relationship(relation)
                    relations.append(ConceptRelationship(
                        concept1=c1,
                        concept2=c2,
                        relation=relation_type,
                        definition=item["attributes"].get("rationale", ""),
                        strength=item.get("strength", 0.5),
                        attributes={"context_type": item["attributes"].get("context_type", "")}
                    ))
                else:
                    print(f"LLM returned relation with unknown concepts: {item['source']} or {item['target']}")
            return relations
        except Exception as e:
            print(f"LLM Extraction failed: {e}")
            return []