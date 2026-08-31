from abc import abstractmethod
from typing import List, Optional

from ...knowledge.models import Concept, ConceptNode
from ...knowledge.repository.repository import BaseRepository


class ConceptRepositoryBase(BaseRepository[Concept]):
    """Base interface for concept repository"""
    
    @abstractmethod
    def save(self, entity: Concept) -> Concept:
        """Save a concept"""
        pass
    
    @abstractmethod
    def get(self, id: str) -> Optional[Concept]:
        """Get concept by id"""
        pass

    @abstractmethod
    def delete(self, id: str) -> bool:
        """Delete a concept"""
        pass

    @abstractmethod
    def find_all(self) -> List[Concept]:
        pass
    
    @abstractmethod
    def get_concept_by_name(self, name: str) -> Optional[Concept]:
        """Get concept by name"""
        pass

    @abstractmethod
    def save_concept_node(self, node: ConceptNode) -> ConceptNode:
        """Save a concept node"""
        pass
    
    
    @abstractmethod
    def update_concept(self, concept: Concept) -> Concept:
        """Update a concept"""
        pass
    

    @abstractmethod
    def search_concepts(self, query: str, limit: int = 10) -> List[Concept]:
        """Search concepts by name or alias"""
        pass
    
    @abstractmethod
    def upsert_concepts(self, concepts: List[Concept]) -> List[Concept]:
        """Insert or update multiple concepts"""
        pass

