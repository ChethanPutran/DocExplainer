from typing import List, Dict, Optional
from ....models import TextModels
from ...agent import LLMInterface


from ..models import Concept,  ConceptInvertedIndex
from .scoring import CompositeScoringStrategy
from .filters import SubsetPrunerStrategy
from .canonicalization import ConceptCanonicalizer

class ConceptExtractor:
    """Main orchestrator for concept extraction"""
    
    def __init__(self, 
                 text_model: TextModels,
                 llm_wrapper: LLMInterface,
                 canonicalizer: ConceptCanonicalizer,
                 scoring_strategy: CompositeScoringStrategy,
                 filter_strategy: SubsetPrunerStrategy,
                 concepts_per_para: int = 10,
                 inverted_index: Optional[ConceptInvertedIndex] = None) -> None:
        
        self.canonicalizer = canonicalizer
        self.scoring_strategy = scoring_strategy
        self.filter_strategy = filter_strategy
        self.concepts_per_para = concepts_per_para
        self.inverted_index = inverted_index or ConceptInvertedIndex()
        
        # Global concept registry
        self.global_concepts: Dict[str, Concept] = {}
        
        # Initialize extraction strategies from text_model
        self.spacy_extractor = text_model.get_spacy_model()
        self.ner_model = text_model.get_ner_model()
        self.ner_regex = text_model.get_ner_regex()
        
        # LLM for refinement
        self.llm = llm_wrapper

    def _get_or_create_concept(self, name: str, embedding_cache: Dict) -> Concept:
        """Get existing concept or create new one"""
        name_normalized = self.canonicalizer.normalizer.normalize(name)

        # Ensure embedding exists
        if name_normalized not in embedding_cache:
            embedding_cache[name_normalized] = self.canonicalizer.clusterer.embedder.encode(name_normalized)

        # Create concept if not exists
        if name_normalized not in self.global_concepts:
            self.global_concepts[name_normalized] = Concept(
                name=name_normalized,
                embedding=embedding_cache[name_normalized]
            )

        concept = self.global_concepts[name_normalized]

        # Add alias
        if name not in concept.aliases:
            concept.aliases.append(name)

        return concept

    def extract(self, text: List[str], context: str, section_id: str, paragraph_id: str) -> List[Concept]:
        """Extract concepts from text"""
        
        # Extract candidates using multiple strategies
        noun_phrases = self.spacy_extractor.extract_noun_phrases(text)
        named_entities = self.spacy_extractor.extract_named_entities(text)
        ner_concepts = self.ner_model.extract_concepts(text)
        pattern_concepts = self.ner_regex.extract_concepts(text)

        # Merge raw candidates
        raw_candidates = [
            candidate.strip()
            for candidate in set(
                noun_phrases + named_entities + ner_concepts + pattern_concepts
            )
            if isinstance(candidate, str) and candidate.strip()
        ]

        # Canonicalization
        canonical_map = self.canonicalizer.canonicalize_concepts(raw_candidates)

        # Create concept objects
        all_concepts = []
        context_lower = context.lower()
        
        for canonical_name, aliases in canonical_map.items():
            concept = self._get_or_create_concept(canonical_name, self.canonicalizer.clusterer.embedding_cache)

            # Add aliases and occurrences
            for alias in aliases:
                if alias not in concept.aliases:
                    concept.aliases.append(alias)

                # Find occurrences
                alias_lower = alias.lower()
                if not alias_lower:
                    continue
                start = 0
                while True:
                    pos = context_lower.find(alias_lower, start)
                    if pos == -1:
                        break

                    concept.occurrences.append({
                        "section_id": section_id,
                        "paragraph_id": paragraph_id,
                        "char_start": pos,
                        "char_end": pos + len(alias_lower),
                        "snippet": context[max(0, pos-40):pos+40]
                    })
                    start = pos + len(alias_lower)
            
            all_concepts.append(concept)

        # Score and filter concepts
        filtered_concepts = self._filter_concepts(all_concepts, context)

        # Update inverted index
        for concept in filtered_concepts:
            self.inverted_index.add_occurrence(
                concept_id=concept.id,
                section_id=section_id,
                section_order=section_id,
                paragraph_id=paragraph_id
            )
            
            # Update concept score and frequency
            concept.score = getattr(concept, 'score', 0.0)
            concept.frequency = len(concept.occurrences)

        return filtered_concepts

    def _filter_concepts(self, concepts: List[Concept], context_text: str) -> List[Concept]:
        """Rank and filter concepts"""
        if not concepts:
            return []

        # Score all concepts
        for concept in concepts:
            score = self.scoring_strategy.score(concept, context_text)
            concept.score = round(score, 4)

        # Filter and sort
        concepts.sort(key=lambda x: x.score, reverse=True)
        return self.filter_strategy.filter(concepts[:self.concepts_per_para + 5], 
                                          top=self.concepts_per_para)
