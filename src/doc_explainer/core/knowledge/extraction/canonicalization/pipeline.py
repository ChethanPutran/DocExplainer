from typing import List, Dict, DefaultDict
from collections import defaultdict

from .normalizer import TextNormalizer
from .clusterer import ConceptClusterer
from .llm_canonicalizer import LLMCanonicalizer

class ConceptCanonicalizer:
    """Orchestrates the complete canonicalization pipeline"""
    
    def __init__(self, 
                 normalizer: TextNormalizer,
                 clusterer: ConceptClusterer,
                 llm_canonicalizer: LLMCanonicalizer):
        self.normalizer = normalizer
        self.clusterer = clusterer
        self.llm_canonicalizer = llm_canonicalizer
    
    def canonicalize_concepts(self, raw_concepts: List[str]) -> Dict[str, List[str]]:
        """
        Complete canonicalization pipeline:
        1. Rule-based normalization
        2. Embedding-based clustering
        3. LLM refinement
        """
        # Step 1: Rule normalize
        normalized_map = defaultdict(list)
        for raw in raw_concepts:
            norm = self.normalizer.normalize(raw)
            if norm:
                normalized_map[norm].append(raw)

        unique_normalized = list(normalized_map.keys())

        # Step 2: Embedding cluster
        clusters = self.clusterer.cluster(unique_normalized)
        embed_clusters = self.clusterer.get_canonical_map(clusters)

        # Step 3: LLM refine cluster representatives
        llm_input = list(embed_clusters.keys())
        llm_map = self.llm_canonicalizer.canonicalize(llm_input)
        
        final_map = defaultdict(list)

        # LLM merge
        for new_canonical, old_canonicals in llm_map.items():
            for old in old_canonicals:
                final_map[new_canonical].extend(embed_clusters.get(old, []))

        # Fallback for any missed clusters
        for old_canonical, aliases in embed_clusters.items():
            if not any(old_canonical in v for v in llm_map.values()):
                final_map[old_canonical].extend(aliases)

        return dict(final_map)