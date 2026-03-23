from time import time
from typing import List, Optional, Any
from ..models.concept import Concept

class ConceptFactory:
    """Factory for creating Concept objects"""
    
    def __init__(self, id_generator=None):
        self.id_generator = id_generator or (lambda: int(time() * 1000))
    
    def create_concept(self, 
                       name: str,
                       aliases: Optional[List[str]] = None,
                       definitions: Optional[List[str]] = None,
                       embedding: Optional[Any] = None,
                       attributes: Optional[dict] = None) -> Concept:
        """Create a new concept"""
        concept = Concept(
            name=name,
            aliases=aliases or [],
            definitions=definitions or [],
            embedding=embedding,
            attributes=attributes or {}
        )
        concept.id = self.id_generator()
        return concept
    
    def create_from_dict(self, data: dict) -> Concept:
        """Create concept from dictionary"""
        return Concept(
            name=data["name"],
            aliases=data.get("aliases", []),
            definitions=data.get("definitions", []),
            score=data.get("score", 0.0),
            frequency=data.get("frequency", 0),
            first_position=data.get("first_position", -1),
            embedding=data.get("embedding"),
            attributes=data.get("attributes", {}),
            occurrences=data.get("occurrences", [])
        )