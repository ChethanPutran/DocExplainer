from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class ConceptInvertedEntry:
    """Entry in inverted index for a concept"""
    section_frequency: Dict[str, int] = field(default_factory=dict)
    paragraph_frequency: Dict[str, int] = field(default_factory=dict)
    first_occurrence: Optional[Tuple[str, str]] = None
    total_frequency: int = 0

class ConceptInvertedIndex:
    """Inverted index for quick concept lookup by location"""
    
    def __init__(self):
        self.index: Dict[str, ConceptInvertedEntry] = {}

    def add_occurrence(self, concept_id: str, section_id: str, 
                      section_order: str, paragraph_id: str):
        """Add an occurrence for a concept"""
        if concept_id not in self.index:
            self.index[concept_id] = ConceptInvertedEntry()

        entry = self.index[concept_id]

        entry.section_frequency[section_id] = \
            entry.section_frequency.get(section_id, 0) + 1

        entry.paragraph_frequency[paragraph_id] = \
            entry.paragraph_frequency.get(paragraph_id, 0) + 1

        entry.total_frequency += 1

        if entry.first_occurrence is None or section_order < entry.first_occurrence[1]:
            entry.first_occurrence = (section_id, section_order)

    def get_first_occurrence(self, concept_id: str) -> Optional[Tuple[str, str]]:
        """Get first occurrence (section_id, order) for a concept"""
        if concept_id in self.index:
            return self.index[concept_id].first_occurrence
        return None

    def get_section_frequency(self, concept_id: str, section_id: str) -> int:
        """Get frequency of concept in a specific section"""
        if concept_id in self.index:
            return self.index[concept_id].section_frequency.get(section_id, 0)
        return 0

    def get_total_frequency(self, concept_id: str) -> int:
        """Get total frequency of concept across all sections"""
        if concept_id in self.index:
            return self.index[concept_id].total_frequency
        return 0