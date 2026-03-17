from typing import List, Optional, Dict
import json
import os
from src.core.knowledge.models.concept import Concept
from src.core.knowledge.models.relationship import ConceptNode
from src.store.base.repository import ConceptRepositoryBase
from .serializers import ConceptSerializer


class ConceptRepository(ConceptRepositoryBase):
    """Repository for concept persistence"""
    
    def __init__(self, storage_path: str = "data/knowledge/concepts/"):
        self.storage_path = storage_path
        self.cache: Dict[int, Concept] = {}
        self.name_index: Dict[str, int] = {}
        self._ensure_storage()
    
    def _ensure_storage(self):
        """Ensure storage directory exists"""
        os.makedirs(self.storage_path, exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, "nodes"), exist_ok=True)
    
    def _get_concept_path(self, concept_id: int) -> str:
        """Get file path for a concept"""
        return os.path.join(self.storage_path, f"{concept_id}.json")
    
    def _get_node_path(self, concept_name: str) -> str:
        """Get file path for a concept node"""
        # Sanitize filename
        safe_name = "".join(c for c in concept_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        return os.path.join(self.storage_path, "nodes", f"{safe_name}.json")
    
    def save_concept(self, concept: Concept) -> Concept:
        """Save a concept"""
        filepath = self._get_concept_path(concept.id)
        
        with open(filepath, 'w') as f:
            json.dump(ConceptSerializer.serialize_concept(concept), f, indent=2)
        
        self.cache[concept.id] = concept
        self.name_index[concept.name] = concept.id
        
        # Also save aliases for name lookup
        for alias in concept.aliases:
            self.name_index[alias] = concept.id
        
        return concept
    
    def save_concept_node(self, node: ConceptNode) -> ConceptNode:
        """Save a concept node"""
        filepath = self._get_node_path(node.primary_concept.name)
        
        # First ensure the concept itself is saved
        self.save_concept(node.primary_concept)
        
        with open(filepath, 'w') as f:
            json.dump(ConceptSerializer.serialize_node(node), f, indent=2)
        
        return node
    
    def get_concept_by_name(self, name: str) -> Optional[Concept]:
        """Get concept by name"""
        # Check name index
        concept_id = self.name_index.get(name)
        if concept_id:
            return self.get_concept_by_id(concept_id)
        
        # Try direct file lookup
        node_path = self._get_node_path(name)
        if os.path.exists(node_path):
            with open(node_path, 'r') as f:
                data = json.load(f)
                concept_data = data.get('primary_concept', {})
                concept = ConceptSerializer.deserialize_concept(concept_data)
                self.cache[concept.id] = concept
                self.name_index[concept.name] = concept.id
                return concept
        
        return None
    
    def get_concept_by_id(self, concept_id: int) -> Optional[Concept]:
        """Get concept by ID"""
        # Check cache first
        if concept_id in self.cache:
            return self.cache[concept_id]
        
        # Load from file
        filepath = self._get_concept_path(concept_id)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                concept = ConceptSerializer.deserialize_concept(data)
                self.cache[concept_id] = concept
                self.name_index[concept.name] = concept_id
                return concept
        
        return None
    
    def get_all_concepts(self) -> List[Concept]:
        """Get all concepts"""
        concepts = []
        
        for filename in os.listdir(self.storage_path):
            if filename.endswith('.json') and not filename.startswith('nodes'):
                filepath = os.path.join(self.storage_path, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    concept = ConceptSerializer.deserialize_concept(data)
                    concepts.append(concept)
        
        return concepts
    
    def update_concept(self, concept: Concept) -> Concept:
        """Update a concept"""
        return self.save_concept(concept)
    
    def delete_concept(self, concept_id: int) -> bool:
        """Delete a concept"""
        concept = self.get_concept_by_id(concept_id)
        if not concept:
            return False
        
        # Delete concept file
        filepath = self._get_concept_path(concept_id)
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # Delete node file if exists
        node_path = self._get_node_path(concept.name)
        if os.path.exists(node_path):
            os.remove(node_path)
        
        # Remove from cache and index
        if concept_id in self.cache:
            del self.cache[concept_id]
        
        # Remove from name index
        keys_to_delete = [k for k, v in self.name_index.items() if v == concept_id]
        for key in keys_to_delete:
            del self.name_index[key]
        
        return True
    
    def search_concepts(self, query: str, limit: int = 10) -> List[Concept]:
        """Search concepts by name or alias"""
        query = query.lower()
        results = []
        
        for concept in self.get_all_concepts():
            if query in concept.name.lower():
                results.append(concept)
            else:
                for alias in concept.aliases:
                    if query in alias.lower():
                        results.append(concept)
                        break
            
            if len(results) >= limit:
                break
        
        return results[:limit]
    
    def upsert_concepts(self, concepts: List[Concept]) -> List[Concept]:
        """Insert or update multiple concepts"""
        saved = []
        for concept in concepts:
            saved.append(self.save_concept(concept))
        return saved