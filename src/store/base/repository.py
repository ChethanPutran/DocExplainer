from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Optional, Dict, Any
from src.core.knowledge.models.concept import Concept
from src.core.knowledge.models.relationship import ConceptRelationship, ConceptNode, ConceptNodeRelationship

T = TypeVar('T')

class BaseRepository(ABC, Generic[T]):
    """Base repository interface"""
    
    @abstractmethod
    def get(self, id: Any) -> Optional[T]:
        """Get entity by ID"""
        pass
    
    @abstractmethod
    def save(self, entity: T) -> T:
        """Save entity"""
        pass
    
    @abstractmethod
    def delete(self, id: Any) -> bool:
        """Delete entity by ID"""
        pass
    
    @abstractmethod
    def find_all(self) -> List[T]:
        """Find all entities"""
        pass


class ConceptRepositoryBase(ABC):
    """Base interface for concept repository"""
    
    @abstractmethod
    def save_concept(self, concept: Concept) -> Concept:
        """Save a concept"""
        pass
    
    @abstractmethod
    def save_concept_node(self, node: ConceptNode) -> ConceptNode:
        """Save a concept node"""
        pass
    
    @abstractmethod
    def get_concept_by_name(self, name: str) -> Optional[Concept]:
        """Get concept by name"""
        pass
    
    @abstractmethod
    def get_concept_by_id(self, concept_id: int) -> Optional[Concept]:
        """Get concept by ID"""
        pass
    
    @abstractmethod
    def get_all_concepts(self) -> List[Concept]:
        """Get all concepts"""
        pass
    
    @abstractmethod
    def update_concept(self, concept: Concept) -> Concept:
        """Update a concept"""
        pass
    
    @abstractmethod
    def delete_concept(self, concept_id: int) -> bool:
        """Delete a concept"""
        pass
    
    @abstractmethod
    def search_concepts(self, query: str, limit: int = 10) -> List[Concept]:
        """Search concepts by name or alias"""
        pass
    
    @abstractmethod
    def upsert_concepts(self, concepts: List[Concept]) -> List[Concept]:
        """Insert or update multiple concepts"""
        pass


class RelationshipRepositoryBase(ABC):
    """Base interface for relationship repository"""
    
    @abstractmethod
    def save_relationship(self, relationship: ConceptRelationship) -> ConceptRelationship:
        """Save a concept relationship"""
        pass
    
    @abstractmethod
    def save_node_relationship(self, relationship: ConceptNodeRelationship) -> ConceptNodeRelationship:
        """Save a node relationship"""
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
    def get_all_relationships(self) -> List[ConceptRelationship]:
        """Get all relationships"""
        pass
    
    @abstractmethod
    def delete_relationship(self, concept1_name: str, concept2_name: str) -> bool:
        """Delete a relationship"""
        pass
    
    @abstractmethod
    def find_relationships_by_type(self, relation_type: str) -> List[ConceptRelationship]:
        """Find relationships by type"""
        pass