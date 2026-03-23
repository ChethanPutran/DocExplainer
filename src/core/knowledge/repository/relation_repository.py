from abc import ABC, abstractmethod
from typing import List, Optional

from src.core.knowledge.models import ConceptRelationship
from src.core.knowledge.repository.repository import BaseRepository


class RelationshipRepositoryBase(BaseRepository[ConceptRelationship]):
    """Base interface for relationship repository"""
    
    @abstractmethod
    def save(self, entity: ConceptRelationship) -> ConceptRelationship:
        """Save a concept relationship"""
        pass
    
    @abstractmethod
    def get(self, id: int) -> Optional[ConceptRelationship]:
        """Get relationship between two concepts"""
        pass
    
    @abstractmethod
    def get_relationship(self, concept1_name: str, concept2_name: str) -> Optional[ConceptRelationship]:
        """Get relationship between two concepts"""
        pass
    
    
    @abstractmethod
    def get_relationships_for_concept(self, concept_name: str, 
                                     relation_type: Optional[str] = None) -> List[ConceptRelationship]:
        """Get all relationships for a concept"""
        pass

    
    @abstractmethod
    def delete(self, id: int) -> bool:
        """Delete a relationship"""
        pass
    
    @abstractmethod
    def find_relationships_by_type(self, relation_type: str) -> List[ConceptRelationship]:
        """Find relationships by type"""
        pass
    
    @abstractmethod
    def find_all(self) -> List[ConceptRelationship]:
        pass 