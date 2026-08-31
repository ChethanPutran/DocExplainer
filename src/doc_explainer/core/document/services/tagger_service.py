"""Paragraph-Concept Tagging Service.

Provides automatic and manual tagging of paragraphs with concepts using:
- NER (Named Entity Recognition) via spaCy
- LLM-based concept extraction
- Confidence scoring and filtering
- Manual override support
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
import time
import spacy
from datetime import datetime

from ..models.tagging_models import (
    ParagraphTag, TaggingResult, ConceptMention, TaggingConfig
)
from src.core.knowledge.models.concept import Concept

logger = logging.getLogger(__name__)


class TaggerService:
    """Service for tagging paragraphs with concepts."""
    
    def __init__(self, config: Optional[TaggingConfig] = None):
        """Initialize the tagger service.
        
        Args:
            config: Tagging configuration. If None, uses default.
        """
        self.config = config or TaggingConfig()
        self.config.validate()
        
        # Load spaCy model for NER
        self.nlp = None
        if self.config.use_ner:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
        
        # LLM will be injected via set_llm_extractor
        self.llm_extractor = None
        
        # Correction history for learning
        self.correction_history: List[Dict[str, Any]] = []
    
    def set_llm_extractor(self, extractor) -> None:
        """Set the LLM-based concept extractor.
        
        Args:
            extractor: Callable that takes text and returns List[str] of concepts
        """
        self.llm_extractor = extractor
    
    def tag_paragraph(
        self, 
        paragraph_id: str, 
        paragraph_text: str,
        concept_graph: Optional[Dict[str, Concept]] = None,
        document_context: Optional[str] = None,
    ) -> TaggingResult:
        """Tag a single paragraph with concepts.
        
        Args:
            paragraph_id: Unique ID of the paragraph
            paragraph_text: The paragraph text to tag
            concept_graph: Graph of known concepts for validation/mapping
            document_context: Optional broader document context for LLM
        
        Returns:
            TaggingResult with extracted tags and metadata
        """
        start_time = time.time()
        result = TaggingResult(
            paragraph_id=paragraph_id,
            paragraph_text=paragraph_text,
        )
        
        # Step 1: NER extraction
        ner_mentions = []
        if self.config.use_ner and self.nlp:
            ner_mentions = self._extract_ner_concepts(paragraph_text)
            result.ner_entities = [m.to_dict() for m in ner_mentions]
        
        # Step 2: LLM extraction
        llm_concepts = []
        if self.config.use_llm and self.llm_extractor:
            llm_concepts = self._extract_llm_concepts(
                paragraph_text, 
                document_context,
                ner_mentions
            )
            result.llm_extracted_concepts = llm_concepts
        
        # Step 3: Combine and deduplicate
        all_mentions = self._combine_mentions(ner_mentions, llm_concepts)
        
        # Step 4: Map to concept graph
        tags = self._map_to_concepts(
            all_mentions, 
            paragraph_id,
            concept_graph
        )
        
        # Step 5: Filter by confidence
        filtered_tags = self._filter_by_confidence(tags)
        
        # Step 6: Limit number of tags
        limited_tags = filtered_tags[:self.config.max_concepts_per_paragraph]
        
        for tag in limited_tags:
            result.add_tag(tag)
        
        result.processing_time = time.time() - start_time
        return result
    
    def tag_paragraphs(
        self,
        paragraphs: List[Tuple[str, str]],
        concept_graph: Optional[Dict[str, Concept]] = None,
        document_context: Optional[str] = None,
    ) -> List[TaggingResult]:
        """Tag multiple paragraphs.
        
        Args:
            paragraphs: List of (paragraph_id, paragraph_text) tuples
            concept_graph: Optional concept graph for validation
            document_context: Optional document context
        
        Returns:
            List of TaggingResult for each paragraph
        """
        results = []
        for para_id, para_text in paragraphs:
            result = self.tag_paragraph(
                para_id, 
                para_text, 
                concept_graph, 
                document_context
            )
            results.append(result)
        
        return results
    
    def add_manual_tag(
        self,
        paragraph_id: str,
        concept_id: str,
        concept_name: str,
        confidence: float = 1.0,
    ) -> ParagraphTag:
        """Manually add a tag to a paragraph.
        
        Args:
            paragraph_id: ID of the paragraph
            concept_id: ID of the concept
            concept_name: Human-readable concept name
            confidence: Confidence score (default 1.0 for manual)
        
        Returns:
            The created ParagraphTag
        """
        if not self.config.enable_manual_override:
            raise ValueError("Manual tagging is disabled in config")
        
        tag = ParagraphTag(
            paragraph_id=paragraph_id,
            concept_id=concept_id,
            concept_name=concept_name,
            confidence=confidence,
            tagged_by='manual',
            method='manual',
        )
        
        return tag
    
    def remove_tag(self, tag: ParagraphTag) -> bool:
        """Remove a tag from a paragraph.
        
        Args:
            tag: The tag to remove
        
        Returns:
            True if successful
        """
        # This will be handled by the repository
        # Here we just return True
        return True
    
    def correct_tag(
        self,
        original_tag: ParagraphTag,
        corrected_tag: ParagraphTag,
    ) -> None:
        """Record a correction and learn from it.
        
        Args:
            original_tag: The original incorrect tag
            corrected_tag: The corrected tag
        """
        if not self.config.learn_from_corrections:
            return
        
        correction = {
            'timestamp': datetime.now().isoformat(),
            'original': original_tag.to_dict(),
            'corrected': corrected_tag.to_dict(),
            'paragraph_id': original_tag.paragraph_id,
        }
        
        self.correction_history.append(correction)
        logger.info(f"Recorded correction for paragraph {original_tag.paragraph_id}")
    
    def _extract_ner_concepts(self, text: str) -> List[ConceptMention]:
        """Extract concepts using spaCy NER.
        
        Args:
            text: The text to process
        
        Returns:
            List of ConceptMention objects
        """
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            mentions = []
            
            for ent in doc.ents:
                # Skip very short entities
                if len(ent.text) < self.config.min_concept_length:
                    continue
                
                # Get confidence from spaCy (use entity frequency in model)
                confidence = self._get_ner_confidence(ent, doc)
                
                if confidence >= self.config.ner_confidence_threshold:
                    mention = ConceptMention(
                        concept_name=ent.text,
                        entity_type=ent.label_,
                        start_char=ent.start_char,
                        end_char=ent.end_char,
                        mention_text=ent.text,
                        confidence=confidence,
                        source='ner',
                        metadata={
                            'entity_type': ent.label_,
                            'vector_norm': float(ent.has_vector),
                        }
                    )
                    mentions.append(mention)
            
            return mentions
        
        except Exception as e:
            logger.error(f"NER extraction failed: {e}")
            return []
    
    def _get_ner_confidence(self, ent, doc) -> float:
        """Calculate NER confidence for an entity.
        
        Args:
            ent: spaCy entity
            doc: spaCy doc
        
        Returns:
            Confidence score between 0 and 1
        """
        # Use multiple signals for confidence
        confidence = 0.5  # Base confidence
        
        # Boost for known entity types
        strong_types = {'PERSON', 'ORG', 'PRODUCT', 'WORK_OF_ART'}
        if ent.label_ in strong_types:
            confidence += 0.2
        
        # Boost for longer entities (more specific)
        if len(ent.text.split()) > 1:
            confidence += 0.1
        
        # Boost for entities with word vectors
        if ent.has_vector:
            confidence += 0.1
        
        # Boost for capitalized entities
        if ent.text[0].isupper():
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _extract_llm_concepts(
        self, 
        text: str,
        document_context: Optional[str] = None,
        ner_mentions: Optional[List[ConceptMention]] = None,
    ) -> List[str]:
        """Extract concepts using LLM.
        
        Args:
            text: The paragraph text
            document_context: Optional broader context
            ner_mentions: Optional NER mentions for context
        
        Returns:
            List of concept names
        """
        if not self.llm_extractor:
            return []
        
        try:
            # Prepare context
            ner_hints = None
            if ner_mentions:
                ner_hints = [m.concept_name for m in ner_mentions]
            
            # Call LLM extractor
            concepts = self.llm_extractor(
                text,
                context=document_context,
                ner_hints=ner_hints,
            )
            
            # Filter concepts
            filtered = []
            for concept in concepts:
                if (self.config.min_concept_length <= len(concept) <= 
                    self.config.max_concept_length):
                    filtered.append(concept)
            
            return filtered
        
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return []
    
    def _combine_mentions(
        self,
        ner_mentions: List[ConceptMention],
        llm_concepts: List[str],
    ) -> List[ConceptMention]:
        """Combine NER and LLM mentions, deduplicating.
        
        Args:
            ner_mentions: Mentions from NER
            llm_concepts: Concepts from LLM (just names)
        
        Returns:
            Combined and deduplicated list of ConceptMention
        """
        # Build mention map by normalized name
        mention_map: Dict[str, ConceptMention] = {}
        
        # Add NER mentions
        for mention in ner_mentions:
            key = mention.concept_name.lower()
            mention_map[key] = mention
        
        # Add LLM concepts, merging if already seen
        for concept in llm_concepts:
            key = concept.lower()
            if key in mention_map:
                # Merge: mark as found by both sources
                mention_map[key].source = 'both'
                mention_map[key].confidence = (
                    mention_map[key].confidence * 0.5 +  # Half weight to existing
                    self.config.llm_confidence_threshold * 0.5  # Half weight to LLM
                )
            else:
                # New mention from LLM
                mention = ConceptMention(
                    concept_name=concept,
                    entity_type='CONCEPT',
                    start_char=-1,
                    end_char=-1,
                    mention_text=concept,
                    confidence=self.config.llm_confidence_threshold,
                    source='llm',
                )
                mention_map[key] = mention
        
        return list(mention_map.values())
    
    def _map_to_concepts(
        self,
        mentions: List[ConceptMention],
        paragraph_id: str,
        concept_graph: Optional[Dict[str, Concept]] = None,
    ) -> List[ParagraphTag]:
        """Map mentions to concept graph and create tags.
        
        Args:
            mentions: Concept mentions to map
            paragraph_id: ID of paragraph being tagged
            concept_graph: Known concepts for mapping
        
        Returns:
            List of ParagraphTag objects
        """
        tags = []
        
        for mention in mentions:
            # Try to find exact concept in graph
            concept_id = None
            if concept_graph:
                concept_id = self._find_concept_id(mention.concept_name, concept_graph)
            
            # If not found, use mention name as ID
            if not concept_id:
                concept_id = mention.concept_name.lower().replace(' ', '_')
            
            # Determine method
            method = 'hybrid' if mention.source == 'both' else mention.source
            
            # Create tag
            tag = ParagraphTag(
                paragraph_id=paragraph_id,
                concept_id=concept_id,
                concept_name=mention.concept_name,
                confidence=mention.confidence,
                ner_confidence=(
                    mention.confidence if mention.source in ['ner', 'both'] else None
                ),
                llm_confidence=(
                    mention.confidence if mention.source in ['llm', 'both'] else None
                ),
                tagged_by='auto',
                method=method,
                attributes={
                    'entity_type': mention.entity_type,
                    'mention_text': mention.mention_text,
                }
            )
            
            tags.append(tag)
        
        return tags
    
    def _find_concept_id(
        self,
        mention_name: str,
        concept_graph: Dict[str, Concept],
    ) -> Optional[str]:
        """Find matching concept in graph by name or alias.
        
        Args:
            mention_name: Name of the mention
            concept_graph: Graph of concepts
        
        Returns:
            Concept ID if found, None otherwise
        """
        mention_lower = mention_name.lower()
        
        for concept_id, concept in concept_graph.items():
            # Check exact name match
            if concept.name.lower() == mention_lower:
                return concept_id
            
            # Check aliases
            for alias in concept.aliases:
                if alias.lower() == mention_lower:
                    return concept_id
        
        return None
    
    def _filter_by_confidence(self, tags: List[ParagraphTag]) -> List[ParagraphTag]:
        """Filter tags by confidence threshold.
        
        Args:
            tags: Tags to filter
        
        Returns:
            Filtered tags sorted by confidence descending
        """
        filtered = [
            t for t in tags 
            if t.confidence >= self.config.combined_confidence_threshold
        ]
        
        # Sort by confidence descending
        filtered.sort(key=lambda t: t.confidence, reverse=True)
        
        return filtered
    
    def get_correction_history(self) -> List[Dict[str, Any]]:
        """Get history of corrections for learning.
        
        Returns:
            List of correction records
        """
        return self.correction_history.copy()
    
    def clear_correction_history(self) -> None:
        """Clear the correction history."""
        self.correction_history.clear()
        logger.info("Correction history cleared")
