from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from .concept import Concept

@dataclass
class ConceptRelationship:
    """Represents a relationship between two concepts"""
    concept1: Concept
    concept2: Concept
    relation: str = "related_to"
    definition: str = ""
    strength: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "concept1": self.concept1.name,
            "concept2": self.concept2.name,
            "relation": self.relation,
            "definition": self.definition,
            "strength": self.strength,
            "attributes": self.attributes
        }

@dataclass
class ConceptNode:
    """Node in concept graph"""
    primary_concept: Concept
    embedding: Optional[Any] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "primary_concept": self.primary_concept.to_dict(),
            "has_embedding": self.embedding is not None
        }

@dataclass
class ConceptNodeRelationship:
    """Edge in concept graph"""
    concept1: ConceptNode
    concept2: ConceptNode
    relationship: ConceptRelationship
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "concept1": self.concept1.primary_concept.name,
            "concept2": self.concept2.primary_concept.name,
            "relationship": self.relationship.to_dict()
        }