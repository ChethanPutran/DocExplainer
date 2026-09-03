from typing import List
from collections import defaultdict
from itertools import combinations
import re
from ..base import BaseRelationshipExtractionStrategy
from .....knowledge.models.concept import Concept
from .....knowledge.models.relationship import ConceptRelationship

class StatisticalRelationshipExtractor(BaseRelationshipExtractionStrategy):
    """Extract relationships based on statistical co-occurrence"""
    
    def extract(self, concepts: List[Concept], text: str, context: str = "") -> List[ConceptRelationship]:
        """Identifies relationships based on sentence-level co-occurrence and proximity."""
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        num_sentences = len(sentences)
        if num_sentences == 0:
            return []

        sentence_to_concepts = defaultdict(list)
        for concept in concepts:
            for idx, sent in enumerate(sentences):
                for alias in concept.aliases:
                    alias_lower = alias.lower()
                    if not alias_lower:
                        continue
                    sent_lower = sent.lower()
                    pattern = r"\b" + re.escape(alias_lower) + r"\b"
                    if re.search(pattern, sent_lower):
                        sentence_to_concepts[idx].append(concept)
                        break

        pair_map = defaultdict(list)
        for idx, concepts_in_sent in sentence_to_concepts.items():
            if len(concepts_in_sent) < 2:
                continue

            for c1, c2 in combinations(concepts_in_sent, 2):
                pair_key = tuple(sorted([c1.name, c2.name]))
                pair_map[pair_key].append(idx)

        seen = set()
        stat_relations = []

        concepts_by_name = {concept.name: concept for concept in concepts}

        for (c1_name, c2_name), indices in pair_map.items():
            key = (c1_name, c2_name)
            if key in seen:
                continue

            c1 = concepts_by_name[c1_name]
            c2 = concepts_by_name[c2_name]
                
            co_occurrence_count = len(indices)
            weight = min(1.0, (co_occurrence_count / num_sentences) * 2)
            snippets = [sentences[i] for i in indices[:3]]

            relationship = ConceptRelationship(
                concept1=c1,
                concept2=c2,
                relation="related_to",
                definition=f"Concepts appear together in {co_occurrence_count} sentence(s).",
                strength=weight,
                attributes={
                    "count": co_occurrence_count,
                    "indices": indices,
                    "snippets": snippets,
                    "method": "statistical_cooccurrence",
                },
            )
            stat_relations.append(relationship)
            seen.add(key)

        return stat_relations