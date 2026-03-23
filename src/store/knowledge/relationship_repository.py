from typing import List, Optional, Dict, Tuple
import json
import os
from src.core.knowledge.models.relationship import ConceptRelationship, ConceptNodeRelationship
from src.core.knowledge.repository.relation_repository import RelationshipRepositoryBase
from .serializers import RelationshipSerializer
from .concept_repository import ConceptRepository


class RelationshipRepository(RelationshipRepositoryBase):
    """Repository for relationship persistence"""
    
    def __init__(self, storage_path: str = "data/knowledge/relationships/", 
                 concept_repository: Optional[ConceptRepository] = None):
        self.storage_path = storage_path
        self.concept_repo = concept_repository or ConceptRepository()
        self.cache: Dict[Tuple[str, str], ConceptRelationship] = {}
        self._ensure_storage()
    
    def _ensure_storage(self):
        """Ensure storage directory exists"""
        os.makedirs(self.storage_path, exist_ok=True)
    
    def _get_relationship_path(self, concept1_name: str, concept2_name: str) -> str:
        """Get file path for a relationship"""
        # Create a safe filename from concept names
        safe_name = f"{concept1_name}_{concept2_name}"
        safe_name = "".join(c for c in safe_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        return os.path.join(self.storage_path, f"{safe_name}.json")
    
    def save(self, entity: ConceptRelationship) -> ConceptRelationship:
        """Save a concept relationship"""
        # Ensure both concepts are saved
        self.concept_repo.save(entity.concept1)
        self.concept_repo.save(entity.concept2)
        
        filepath = self._get_relationship_path(
            entity.concept1.name, 
            entity.concept2.name
        )
        
        with open(filepath, 'w') as f:
            json.dump(RelationshipSerializer.serialize_relationship(entity), f, indent=2)
        
        cache_key = (entity.concept1.name, entity.concept2.name)
        self.cache[cache_key] = entity
        
        return entity
    
    def save_node_relationship(self, relationship: ConceptNodeRelationship) -> ConceptNodeRelationship:
        """Save a node relationship"""
        # Extract the core relationship and save it
        self.save(relationship.relationship)
        return relationship
    
    def get(self, id: int) -> Optional[ConceptRelationship]:
        """Get relationship by ID"""
        # This method is not implemented as relationships are identified by concept pairs, not IDs
        raise NotImplementedError("Relationships are identified by concept pairs, not IDs.")

    def get_relationship(self, concept1_name: str, concept2_name: str) -> Optional[ConceptRelationship]:
        """Get relationship between two concepts"""
        # Check cache first
        cache_key = (concept1_name, concept2_name)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Try both orders
        filepath = self._get_relationship_path(concept1_name, concept2_name)
        if not os.path.exists(filepath):
            filepath = self._get_relationship_path(concept2_name, concept1_name)
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                # Get the actual concept objects
                concept1 = self.concept_repo.get_concept_by_name(data['concept1'])
                concept2 = self.concept_repo.get_concept_by_name(data['concept2'])
                
                if concept1 and concept2:
                    relationship = RelationshipSerializer.deserialize_relationship(
                        data, concept1, concept2
                    )
                    self.cache[(concept1_name, concept2_name)] = relationship
                    return relationship
        
        return None
    
    def get_relationships_for_concept(self, concept_name: str, 
                                     relation_type: Optional[str] = None) -> List[ConceptRelationship]:
        """Get all relationships for a concept"""
        relationships = []
        
        for filename in os.listdir(self.storage_path):
            if not filename.endswith('.json'):
                continue
            
            filepath = os.path.join(self.storage_path, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                if data['concept1'] == concept_name or data['concept2'] == concept_name:
                    if relation_type and data['relation'] != relation_type:
                        continue
                    
                    concept1 = self.concept_repo.get_concept_by_name(data['concept1'])
                    concept2 = self.concept_repo.get_concept_by_name(data['concept2'])
                    
                    if concept1 and concept2:
                        relationship = RelationshipSerializer.deserialize_relationship(
                            data, concept1, concept2
                        )
                        relationships.append(relationship)
        
        return relationships
    
    def find_all(self) -> List[ConceptRelationship]:
        """Get all relationships"""
        relationships = []
        
        for filename in os.listdir(self.storage_path):
            if not filename.endswith('.json'):
                continue
            
            filepath = os.path.join(self.storage_path, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                concept1 = self.concept_repo.get_concept_by_name(data['concept1'])
                concept2 = self.concept_repo.get_concept_by_name(data['concept2'])
                
                if concept1 and concept2:
                    relationship = RelationshipSerializer.deserialize_relationship(
                        data, concept1, concept2
                    )
                    relationships.append(relationship)
        
        return relationships
    
    def delete_relationship(self, concept1_name: str, concept2_name: str) -> bool:
        """Delete a relationship"""
        filepath = self._get_relationship_path(concept1_name, concept2_name)
        if os.path.exists(filepath):
            os.remove(filepath)
        
        alt_filepath = self._get_relationship_path(concept2_name, concept1_name)
        if os.path.exists(alt_filepath):
            os.remove(alt_filepath)
        
        # Remove from cache
        cache_key = (concept1_name, concept2_name)
        if cache_key in self.cache:
            del self.cache[cache_key]
        
        cache_key_alt = (concept2_name, concept1_name)
        if cache_key_alt in self.cache:
            del self.cache[cache_key_alt]
        
        return True
    
    def delete(self, id: int) -> bool:
        """Delete a relationship"""
        # This method is not implemented as relationships are identified by concept pairs, not IDs
        raise NotImplementedError("Relationships are identified by concept pairs, not IDs.")

    def find_relationships_by_type(self, relation_type: str) -> List[ConceptRelationship]:
        """Find relationships by type"""
        relationships = []
        
        for filename in os.listdir(self.storage_path):
            if not filename.endswith('.json'):
                continue
            
            filepath = os.path.join(self.storage_path, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                if data['relation'] == relation_type:
                    concept1 = self.concept_repo.get_concept_by_name(data['concept1'])
                    concept2 = self.concept_repo.get_concept_by_name(data['concept2'])
                    
                    if concept1 and concept2:
                        relationship = RelationshipSerializer.deserialize_relationship(
                            data, concept1, concept2
                        )
                        relationships.append(relationship)
        
        return relationships