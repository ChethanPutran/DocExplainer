from typing import Optional, Dict, Any
from src.core.knowledge.models.concept import Concept
from src.core.knowledge.models.relationship import ConceptRelationship, ConceptNode, ConceptNodeRelationship

class RelationshipFactory:
    """Factory for creating relationship objects"""
    
    def create_relationship(self,
                           concept1: Concept,
                           concept2: Concept,
                           relation: str = "related_to",
                           definition: str = "",
                           strength: float = 1.0,
                           attributes: Optional[Dict[str, Any]] = None) -> ConceptRelationship:
        """Create a new relationship"""
        return ConceptRelationship(
            concept1=concept1,
            concept2=concept2,
            relation=relation,
            definition=definition,
            strength=strength,
            attributes=attributes or {}
        )
    
    def create_node(self, concept: Concept, embedding: Optional[Any] = None) -> ConceptNode:
        """Create a concept node"""
        return ConceptNode(
            primary_concept=concept,
            embedding=embedding
        )
    
    def create_node_relationship(self,
                                node1: ConceptNode,
                                node2: ConceptNode,
                                relationship: ConceptRelationship) -> ConceptNodeRelationship:
        """Create a node relationship"""
        return ConceptNodeRelationship(
            concept1=node1,
            concept2=node2,
            relationship=relationship
        )
    
    def create_from_dict(self, data: dict, concept_map: Dict[str, Concept]) -> Optional[ConceptRelationship]:
        """Create relationship from dictionary"""
        concept1 = concept_map.get(data.get("concept1"))
        concept2 = concept_map.get(data.get("concept2"))
        
        if not concept1 or not concept2:
            return None
        
        return ConceptRelationship(
            concept1=concept1,
            concept2=concept2,
            relation=data.get("relation", "related_to"),
            definition=data.get("definition", ""),
            strength=data.get("strength", 1.0),
            attributes=data.get("attributes", {})
        )