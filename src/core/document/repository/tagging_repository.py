"""Repository for persisting paragraph-concept tags to database."""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..models.tagging_models import ParagraphTag, TaggingResult
from src.store.db import get_db_manager

logger = logging.getLogger(__name__)


class TaggingRepository:
    """Manages persistence of paragraph-concept tags."""
    
    def __init__(self):
        """Initialize the tagging repository."""
        self.db = get_db_manager()
    
    def save_paragraph_tag(
        self,
        doc_id: str,
        tag: ParagraphTag,
    ) -> bool:
        """Save a single paragraph-concept tag to database.
        
        Args:
            doc_id: ID of the document
            tag: The ParagraphTag to save
        
        Returns:
            True if successful, False otherwise
        """
        try:
            query = """
                INSERT OR REPLACE INTO document_concepts 
                (doc_id, concept_id, paragraph_id, confidence_score, tagged_by, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                doc_id,
                tag.concept_id,
                tag.paragraph_id,
                tag.confidence,
                tag.tagged_by,
                tag.created_at.isoformat(),
                tag.updated_at.isoformat(),
            )
            
            self.db.execute_query(query, params)
            logger.debug(f"Saved tag for paragraph {tag.paragraph_id}, concept {tag.concept_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save tag: {e}")
            return False
    
    def save_tagging_result(
        self,
        doc_id: str,
        result: TaggingResult,
    ) -> bool:
        """Save all tags from a tagging result.
        
        Args:
            doc_id: ID of the document
            result: The TaggingResult to save
        
        Returns:
            True if all tags saved successfully
        """
        all_saved = True
        for tag in result.tags:
            if not self.save_paragraph_tag(doc_id, tag):
                all_saved = False
        
        return all_saved
    
    def save_batch_tags(
        self,
        doc_id: str,
        tags: List[ParagraphTag],
    ) -> int:
        """Save multiple tags in batch.
        
        Args:
            doc_id: ID of the document
            tags: List of tags to save
        
        Returns:
            Number of tags successfully saved
        """
        saved_count = 0
        for tag in tags:
            if self.save_paragraph_tag(doc_id, tag):
                saved_count += 1
        
        return saved_count
    
    def get_tags_for_paragraph(
        self,
        paragraph_id: str,
    ) -> List[ParagraphTag]:
        """Get all tags for a specific paragraph.
        
        Args:
            paragraph_id: ID of the paragraph
        
        Returns:
            List of ParagraphTag for the paragraph
        """
        try:
            query = """
                SELECT * FROM document_concepts 
                WHERE paragraph_id = ?
                ORDER BY confidence_score DESC
            """
            
            rows = self.db.fetch_all(query, (paragraph_id,))
            
            tags = []
            for row in rows:
                tag = self._row_to_tag(row)
                tags.append(tag)
            
            return tags
        
        except Exception as e:
            logger.error(f"Failed to get tags for paragraph: {e}")
            return []
    
    def get_tags_for_document(
        self,
        doc_id: str,
    ) -> Dict[str, List[ParagraphTag]]:
        """Get all tags for a document, organized by paragraph.
        
        Args:
            doc_id: ID of the document
        
        Returns:
            Dictionary mapping paragraph_id to list of tags
        """
        try:
            query = """
                SELECT * FROM document_concepts 
                WHERE doc_id = ?
                ORDER BY paragraph_id, confidence_score DESC
            """
            
            rows = self.db.fetch_all(query, (doc_id,))
            
            tags_by_paragraph: Dict[str, List[ParagraphTag]] = {}
            for row in rows:
                tag = self._row_to_tag(row)
                para_id = tag.paragraph_id
                
                if para_id not in tags_by_paragraph:
                    tags_by_paragraph[para_id] = []
                
                tags_by_paragraph[para_id].append(tag)
            
            return tags_by_paragraph
        
        except Exception as e:
            logger.error(f"Failed to get tags for document: {e}")
            return {}
    
    def get_paragraphs_for_concept(
        self,
        concept_id: str,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Get all paragraphs tagged with a specific concept.
        
        Args:
            concept_id: ID of the concept
            min_confidence: Minimum confidence threshold
        
        Returns:
            List of dicts with paragraph_id, doc_id, confidence
        """
        try:
            query = """
                SELECT doc_id, paragraph_id, confidence_score 
                FROM document_concepts 
                WHERE concept_id = ? AND confidence_score >= ?
                ORDER BY confidence_score DESC
            """
            
            rows = self.db.fetch_all(query, (concept_id, min_confidence))
            
            results = []
            for row in rows:
                results.append({
                    'doc_id': row['doc_id'],
                    'paragraph_id': row['paragraph_id'],
                    'confidence': row['confidence_score'],
                })
            
            return results
        
        except Exception as e:
            logger.error(f"Failed to get paragraphs for concept: {e}")
            return []
    
    def get_concept_count_for_document(self, doc_id: str) -> int:
        """Get total number of unique concepts tagged in a document.
        
        Args:
            doc_id: ID of the document
        
        Returns:
            Count of unique concepts
        """
        try:
            query = """
                SELECT COUNT(DISTINCT concept_id) as count 
                FROM document_concepts 
                WHERE doc_id = ?
            """
            
            row = self.db.fetch_one(query, (doc_id,))
            return row['count'] if row else 0
        
        except Exception as e:
            logger.error(f"Failed to get concept count: {e}")
            return 0
    
    def update_tag_confidence(
        self,
        doc_id: str,
        paragraph_id: str,
        concept_id: str,
        new_confidence: float,
    ) -> bool:
        """Update confidence score of a tag.
        
        Args:
            doc_id: ID of the document
            paragraph_id: ID of the paragraph
            concept_id: ID of the concept
            new_confidence: New confidence score
        
        Returns:
            True if successful
        """
        try:
            query = """
                UPDATE document_concepts 
                SET confidence_score = ?, updated_at = ?
                WHERE doc_id = ? AND paragraph_id = ? AND concept_id = ?
            """
            
            params = (
                new_confidence,
                datetime.now().isoformat(),
                doc_id,
                paragraph_id,
                concept_id,
            )
            
            self.db.execute_query(query, params)
            logger.debug(f"Updated confidence for tag: {concept_id} in {paragraph_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to update confidence: {e}")
            return False
    
    def update_tag_source(
        self,
        doc_id: str,
        paragraph_id: str,
        concept_id: str,
        tagged_by: str,  # 'auto' or 'manual'
    ) -> bool:
        """Update the source of a tag (auto vs manual).
        
        Args:
            doc_id: ID of the document
            paragraph_id: ID of the paragraph
            concept_id: ID of the concept
            tagged_by: 'auto' or 'manual'
        
        Returns:
            True if successful
        """
        if tagged_by not in ('auto', 'manual'):
            raise ValueError("tagged_by must be 'auto' or 'manual'")
        
        try:
            query = """
                UPDATE document_concepts 
                SET tagged_by = ?, updated_at = ?
                WHERE doc_id = ? AND paragraph_id = ? AND concept_id = ?
            """
            
            params = (
                tagged_by,
                datetime.now().isoformat(),
                doc_id,
                paragraph_id,
                concept_id,
            )
            
            self.db.execute_query(query, params)
            logger.debug(f"Updated tag source for {concept_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to update tag source: {e}")
            return False
    
    def remove_tag(
        self,
        doc_id: str,
        paragraph_id: str,
        concept_id: str,
    ) -> bool:
        """Remove a paragraph-concept tag.
        
        Args:
            doc_id: ID of the document
            paragraph_id: ID of the paragraph
            concept_id: ID of the concept
        
        Returns:
            True if successful
        """
        try:
            query = """
                DELETE FROM document_concepts 
                WHERE doc_id = ? AND paragraph_id = ? AND concept_id = ?
            """
            
            self.db.execute_query(query, (doc_id, paragraph_id, concept_id))
            logger.debug(f"Removed tag: {concept_id} from {paragraph_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to remove tag: {e}")
            return False
    
    def remove_paragraph_tags(
        self,
        paragraph_id: str,
    ) -> int:
        """Remove all tags for a paragraph.
        
        Args:
            paragraph_id: ID of the paragraph
        
        Returns:
            Number of tags removed
        """
        try:
            query = "SELECT COUNT(*) as count FROM document_concepts WHERE paragraph_id = ?"
            row = self.db.fetch_one(query, (paragraph_id,))
            count = row['count'] if row else 0
            
            query = "DELETE FROM document_concepts WHERE paragraph_id = ?"
            self.db.execute_query(query, (paragraph_id,))
            
            logger.debug(f"Removed {count} tags for paragraph {paragraph_id}")
            return count
        
        except Exception as e:
            logger.error(f"Failed to remove paragraph tags: {e}")
            return 0
    
    def get_auto_tagged_count(self, doc_id: str) -> int:
        """Get count of auto-tagged concepts in a document.
        
        Args:
            doc_id: ID of the document
        
        Returns:
            Count of auto-tagged concepts
        """
        try:
            query = """
                SELECT COUNT(*) as count 
                FROM document_concepts 
                WHERE doc_id = ? AND tagged_by = 'auto'
            """
            
            row = self.db.fetch_one(query, (doc_id,))
            return row['count'] if row else 0
        
        except Exception as e:
            logger.error(f"Failed to get auto-tagged count: {e}")
            return 0
    
    def get_manual_tagged_count(self, doc_id: str) -> int:
        """Get count of manually-tagged concepts in a document.
        
        Args:
            doc_id: ID of the document
        
        Returns:
            Count of manually-tagged concepts
        """
        try:
            query = """
                SELECT COUNT(*) as count 
                FROM document_concepts 
                WHERE doc_id = ? AND tagged_by = 'manual'
            """
            
            row = self.db.fetch_one(query, (doc_id,))
            return row['count'] if row else 0
        
        except Exception as e:
            logger.error(f"Failed to get manual-tagged count: {e}")
            return 0
    
    def _row_to_tag(self, row: Dict[str, Any]) -> ParagraphTag:
        """Convert database row to ParagraphTag.
        
        Args:
            row: Database row
        
        Returns:
            ParagraphTag object
        """
        return ParagraphTag(
            paragraph_id=row['paragraph_id'],
            concept_id=row['concept_id'],
            concept_name=row['concept_id'],  # TODO: Map to actual name if available
            confidence=row['confidence_score'],
            tagged_by=row['tagged_by'],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
        )
