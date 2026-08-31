import json
import os
from typing import Dict, Optional, List, Tuple
from ...core.knowledge.models.index import ConceptInvertedIndex, ConceptInvertedEntry


class InvertedIndexRepository:
    """Repository for inverted index persistence"""
    
    def __init__(self, storage_path: str = "data/knowledge/index/"):
        self.storage_path = storage_path
        self.index: Optional[ConceptInvertedIndex] = None
        self._ensure_storage()
    
    def _ensure_storage(self):
        """Ensure storage directory exists"""
        os.makedirs(self.storage_path, exist_ok=True)
    
    def save_index(self, index: ConceptInvertedIndex, name: str = "default"):
        """Save inverted index"""
        filepath = os.path.join(self.storage_path, f"{name}.json")
        
        # Convert to serializable format
        data = {}
        for concept_id, entry in index.index.items():
            data[str(concept_id)] = {
                'section_frequency': entry.section_frequency,
                'paragraph_frequency': entry.paragraph_frequency,
                'first_occurrence': entry.first_occurrence,
                'total_frequency': entry.total_frequency
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.index = index
    
    def load_index(self, name: str = "default") -> Optional[ConceptInvertedIndex]:
        """Load inverted index"""
        filepath = os.path.join(self.storage_path, f"{name}.json")
        
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        index = ConceptInvertedIndex()
        
        for concept_id_str, entry_data in data.items():
            concept_id = int(concept_id_str)
            entry = ConceptInvertedEntry(
                section_frequency={int(k): v for k, v in entry_data['section_frequency'].items()},
                paragraph_frequency={int(k): v for k, v in entry_data['paragraph_frequency'].items()},
                first_occurrence=tuple(entry_data['first_occurrence']) if entry_data['first_occurrence'] else None,
                total_frequency=entry_data['total_frequency']
            )
            index.index[concept_id] = entry
        
        self.index = index
        return index
    
    def get_index(self) -> Optional[ConceptInvertedIndex]:
        """Get current index"""
        if not self.index:
            self.load_index()
        return self.index
    
    def add_occurrence(self, concept_id: int, section_id: int, 
                      section_order: int, paragraph_id: int):
        """Add occurrence to index"""
        if not self.index:
            self.load_index() or ConceptInvertedIndex()
        
        self.index.add_occurrence(concept_id, section_id, section_order, paragraph_id)
        self.save_index(self.index)
    
    def get_first_occurrence(self, concept_id: int) -> Optional[Tuple[int, int]]:
        """Get first occurrence for concept"""
        if not self.index:
            self.load_index()
        
        if self.index and concept_id in self.index.index:
            return self.index.index[concept_id].first_occurrence
        return None
    
    def get_section_frequency(self, concept_id: int, section_id: int) -> int:
        """Get frequency in section"""
        if not self.index:
            self.load_index()
        
        if self.index and concept_id in self.index.index:
            return self.index.index[concept_id].section_frequency.get(section_id, 0)
        return 0
    
    def get_total_frequency(self, concept_id: int) -> int:
        """Get total frequency"""
        if not self.index:
            self.load_index()
        
        if self.index and concept_id in self.index.index:
            return self.index.index[concept_id].total_frequency
        return 0
    
    def clear_index(self):
        """Clear the index"""
        self.index = ConceptInvertedIndex()
        self.save_index(self.index)