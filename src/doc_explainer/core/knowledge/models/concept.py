from time import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from doc_explainer.core.document.models.tree import GraphReference, VectorReference

@dataclass(eq=False)
class Concept:
    """Represents a knowledge concept"""
    __hash__ = object.__hash__

    name: str
    aliases: List[str] = field(default_factory=list)
    definitions: List[str] = field(default_factory=list)
    score: float = 0.0
    frequency: int = 0
    first_position: int = -1
    embedding: Optional[Any] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    occurrences: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        self.id = str(time() * 1000)  # Generate unique ID
    
    def add_occurrence(self, section_id: str, paragraph_id: str, char_start: int, char_end: int, snippet: str):
        """Add an occurrence location"""
        self.occurrences.append({
            "section_id": section_id,
            "paragraph_id": paragraph_id,
            "char_start": char_start,
            "char_end": char_end,
            "snippet": snippet
        })
        self.frequency += 1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "aliases": self.aliases,
            "definitions": self.definitions,
            "score": self.score,
            "frequency": self.frequency,
            "first_position": self.first_position,
            "attributes": self.attributes,
            "occurrences": self.occurrences
        }

@dataclass
class ConceptReference:
    concept_id: str
    canonical_name: str

    graph_ref: Optional[GraphReference] = None
    vector_ref: Optional[VectorReference] = None