from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import List, Tuple, Optional
from src.core.knowledge.models.concept import Concept
from src.core.knowledge.models.relationship import ConceptRelationship


class UnitOfWork(ABC):
    """Unit of Work pattern for managing transactions"""
    
    def __init__(self):
        self._concepts_to_save: List[Concept] = []
        self._concepts_to_delete: List[int] = []
        self._relationships_to_save: List[ConceptRelationship] = []
        self._relationships_to_delete: List[Tuple[str, str]] = []
        self._occurrences_to_add: List[Tuple[int, int, int, int]] = []
    
    @abstractmethod
    def begin(self):
        """Begin a transaction"""
        pass
    
    @abstractmethod
    def commit(self):
        """Commit the transaction"""
        pass
    
    @abstractmethod
    def rollback(self):
        """Rollback the transaction"""
        pass
    
    def register_concept_save(self, concept: Concept):
        """Register a concept to be saved"""
        self._concepts_to_save.append(concept)
    
    def register_concept_delete(self, concept_id: int):
        """Register a concept to be deleted"""
        self._concepts_to_delete.append(concept_id)
    
    def register_relationship_save(self, relationship: ConceptRelationship):
        """Register a relationship to be saved"""
        self._relationships_to_save.append(relationship)
    
    def register_relationship_delete(self, concept1_name: str, concept2_name: str):
        """Register a relationship to be deleted"""
        self._relationships_to_delete.append((concept1_name, concept2_name))
    
    def register_occurrence(self, concept_id: int, section_id: int, 
                           section_order: int, paragraph_id: int):
        """Register an occurrence to be added"""
        self._occurrences_to_add.append((concept_id, section_id, section_order, paragraph_id))
    
    @contextmanager
    def transaction(self):
        """Context manager for transactions"""
        try:
            self.begin()
            yield self
            self.commit()
        except Exception as e:
            self.rollback()
            raise e


class UnitOfWorkManager:
    """Manages Unit of Work instances"""
    
    def __init__(self, uow_factory):
        self.uow_factory = uow_factory
        self.current_uow: Optional[UnitOfWork] = None
    
    def start(self) -> UnitOfWork:
        """Start a new Unit of Work"""
        if self.current_uow:
            raise RuntimeError("A Unit of Work is already in progress")
        
        self.current_uow = self.uow_factory()
        return self.current_uow
    
    def get_current(self) -> Optional[UnitOfWork]:
        """Get the current Unit of Work"""
        return self.current_uow
    
    def complete(self):
        """Complete the current Unit of Work"""
        if self.current_uow:
            self.current_uow = None