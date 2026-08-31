from abc import ABC, abstractmethod
from typing import List
from ...knowledge.models import ConceptNode, ConceptNodeRelationship

class KnowledgeGraphObserver(ABC):
    """Base observer for knowledge graph events"""
    
    @abstractmethod
    def on_concept_added(self, concept: ConceptNode, section_id: int):
        """Called when a concept is added"""
        pass
    
    @abstractmethod
    def on_relationship_added(self, relationship: ConceptNodeRelationship):
        """Called when a relationship is added"""
        pass
    
    @abstractmethod
    def on_section_processed(self, section_id: int, concepts: List[ConceptNode]):
        """Called when a section is processed"""
        pass

class KnowledgeGraphSubject:
    """Subject that observers can attach to"""
    
    def __init__(self):
        self._observers: List[KnowledgeGraphObserver] = []
    
    def attach(self, observer: KnowledgeGraphObserver):
        """Attach an observer"""
        self._observers.append(observer)
    
    def detach(self, observer: KnowledgeGraphObserver):
        """Detach an observer"""
        self._observers.remove(observer)
    
    def notify_concept_added(self, concept: ConceptNode, section_id: int):
        """Notify all observers that a concept was added"""
        for observer in self._observers:
            observer.on_concept_added(concept, section_id)
    
    def notify_relationship_added(self, relationship: ConceptNodeRelationship):
        """Notify all observers that a relationship was added"""
        for observer in self._observers:
            observer.on_relationship_added(relationship)
    
    def notify_section_processed(self, section_id: int, concepts: List[ConceptNode]):
        """Notify all observers that a section was processed"""
        for observer in self._observers:
            observer.on_section_processed(section_id, concepts)