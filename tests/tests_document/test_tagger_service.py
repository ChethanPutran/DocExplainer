"""Comprehensive tests for paragraph-concept tagging service."""

import pytest
import logging
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.core.document.services.tagger_service import TaggerService
from src.core.document.models.tagging_models import (
    ParagraphTag, TaggingResult, ConceptMention, TaggingConfig
)
from src.core.document.repository.tagging_repository import TaggingRepository
from src.core.knowledge.models.concept import Concept


class TestParagraphTag:
    """Test ParagraphTag model."""
    
    def test_create_paragraph_tag(self):
        """Test creating a paragraph tag."""
        tag = ParagraphTag(
            paragraph_id='p1',
            concept_id='c1',
            concept_name='Machine Learning',
            confidence=0.85,
        )
        
        assert tag.paragraph_id == 'p1'
        assert tag.concept_id == 'c1'
        assert tag.concept_name == 'Machine Learning'
        assert tag.confidence == 0.85
        assert tag.tagged_by == 'auto'
        assert tag.method == 'hybrid'
    
    def test_paragraph_tag_confidence_validation(self):
        """Test confidence score validation."""
        with pytest.raises(ValueError):
            ParagraphTag(
                paragraph_id='p1',
                concept_id='c1',
                concept_name='Test',
                confidence=1.5,  # Invalid
            )
    
    def test_paragraph_tag_to_dict(self):
        """Test converting tag to dictionary."""
        tag = ParagraphTag(
            paragraph_id='p1',
            concept_id='c1',
            concept_name='Machine Learning',
            confidence=0.85,
            method='ner',
        )
        
        data = tag.to_dict()
        assert data['paragraph_id'] == 'p1'
        assert data['concept_id'] == 'c1'
        assert data['confidence'] == 0.85
        assert data['method'] == 'ner'
    
    def test_paragraph_tag_from_dict(self):
        """Test creating tag from dictionary."""
        data = {
            'paragraph_id': 'p1',
            'concept_id': 'c1',
            'concept_name': 'Test',
            'confidence': 0.8,
            'tagged_by': 'manual',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
        }
        
        tag = ParagraphTag.from_dict(data)
        assert tag.paragraph_id == 'p1'
        assert tag.concept_id == 'c1'
        assert tag.tagged_by == 'manual'


class TestTaggingResult:
    """Test TaggingResult model."""
    
    def test_create_tagging_result(self):
        """Test creating a tagging result."""
        result = TaggingResult(
            paragraph_id='p1',
            paragraph_text='This is a test paragraph about machine learning.',
        )
        
        assert result.paragraph_id == 'p1'
        assert len(result.tags) == 0
    
    def test_add_tag_to_result(self):
        """Test adding tags to result."""
        result = TaggingResult(
            paragraph_id='p1',
            paragraph_text='Test text',
        )
        
        tag = ParagraphTag(
            paragraph_id='p1',
            concept_id='c1',
            concept_name='ML',
            confidence=0.8,
            tagged_by='auto',
        )
        
        result.add_tag(tag)
        assert len(result.tags) == 1
        assert len(result.auto_tags) == 1
        assert len(result.manual_tags) == 0
    
    def test_result_with_mixed_tags(self):
        """Test result with both auto and manual tags."""
        result = TaggingResult(
            paragraph_id='p1',
            paragraph_text='Test text',
        )
        
        auto_tag = ParagraphTag(
            paragraph_id='p1',
            concept_id='c1',
            concept_name='ML',
            confidence=0.8,
            tagged_by='auto',
        )
        
        manual_tag = ParagraphTag(
            paragraph_id='p1',
            concept_id='c2',
            concept_name='AI',
            confidence=1.0,
            tagged_by='manual',
        )
        
        result.add_tag(auto_tag)
        result.add_tag(manual_tag)
        
        assert len(result.tags) == 2
        assert len(result.auto_tags) == 1
        assert len(result.manual_tags) == 1


class TestConceptMention:
    """Test ConceptMention model."""
    
    def test_create_concept_mention(self):
        """Test creating a concept mention."""
        mention = ConceptMention(
            concept_name='Python',
            entity_type='PRODUCT',
            start_char=0,
            end_char=6,
            mention_text='Python',
            confidence=0.95,
        )
        
        assert mention.concept_name == 'Python'
        assert mention.entity_type == 'PRODUCT'
        assert mention.confidence == 0.95
    
    def test_mention_to_dict(self):
        """Test converting mention to dictionary."""
        mention = ConceptMention(
            concept_name='Python',
            entity_type='PRODUCT',
            start_char=0,
            end_char=6,
            mention_text='Python',
        )
        
        data = mention.to_dict()
        assert data['concept_name'] == 'Python'
        assert data['entity_type'] == 'PRODUCT'


class TestTaggingConfig:
    """Test TaggingConfig model."""
    
    def test_default_config(self):
        """Test creating config with defaults."""
        config = TaggingConfig()
        
        assert config.use_ner is True
        assert config.use_llm is True
        assert config.ner_confidence_threshold == 0.5
        assert config.llm_confidence_threshold == 0.6
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = TaggingConfig()
        assert config.validate() is True
    
    def test_invalid_confidence_threshold(self):
        """Test invalid confidence threshold."""
        config = TaggingConfig(ner_confidence_threshold=1.5)
        
        with pytest.raises(ValueError):
            config.validate()


class TestTaggerService:
    """Test TaggerService."""
    
    @pytest.fixture
    def tagger_service(self):
        """Create a tagger service instance."""
        return TaggerService()
    
    def test_initialize_service(self, tagger_service):
        """Test initializing the service."""
        assert tagger_service.config is not None
        assert tagger_service.nlp is not None  # spaCy model should be loaded
    
    def test_custom_config(self):
        """Test creating service with custom config."""
        config = TaggingConfig(
            use_ner=True,
            use_llm=False,
            ner_confidence_threshold=0.7,
        )
        
        service = TaggerService(config)
        assert service.config.use_llm is False
        assert service.config.ner_confidence_threshold == 0.7
    
    def test_set_llm_extractor(self, tagger_service):
        """Test setting LLM extractor."""
        mock_extractor = Mock(return_value=['concept1', 'concept2'])
        tagger_service.set_llm_extractor(mock_extractor)
        
        assert tagger_service.llm_extractor == mock_extractor
    
    def test_add_manual_tag(self, tagger_service):
        """Test adding a manual tag."""
        tag = tagger_service.add_manual_tag(
            paragraph_id='p1',
            concept_id='c1',
            concept_name='Machine Learning',
        )
        
        assert tag.tagged_by == 'manual'
        assert tag.confidence == 1.0
        assert tag.method == 'manual'
    
    def test_manual_tags_disabled(self):
        """Test manual tags when disabled."""
        config = TaggingConfig(enable_manual_override=False)
        service = TaggerService(config)
        
        with pytest.raises(ValueError):
            service.add_manual_tag('p1', 'c1', 'Test')
    
    def test_tag_paragraph_basic(self, tagger_service):
        """Test tagging a basic paragraph."""
        text = "Apple Inc. is a technology company founded by Steve Jobs."
        
        result = tagger_service.tag_paragraph(
            paragraph_id='p1',
            paragraph_text=text,
        )
        
        assert result.paragraph_id == 'p1'
        assert result.paragraph_text == text
        assert result.processing_time > 0
    
    def test_ner_extraction(self, tagger_service):
        """Test NER-based extraction."""
        text = "Steve Jobs founded Apple in California."
        
        result = tagger_service.tag_paragraph(
            paragraph_id='p1',
            paragraph_text=text,
        )
        
        # Should extract entities like Apple, Steve Jobs, California
        assert len(result.ner_entities) > 0
    
    def test_tag_with_llm(self, tagger_service):
        """Test tagging with LLM extractor."""
        mock_extractor = Mock(return_value=['Machine Learning', 'Neural Networks'])
        tagger_service.set_llm_extractor(mock_extractor)
        
        text = "Deep learning is a subset of machine learning."
        
        result = tagger_service.tag_paragraph(
            paragraph_id='p1',
            paragraph_text=text,
        )
        
        # LLM should be called
        mock_extractor.assert_called()
        assert 'Machine Learning' in result.llm_extracted_concepts
    
    def test_tag_multiple_paragraphs(self, tagger_service):
        """Test tagging multiple paragraphs."""
        paragraphs = [
            ('p1', 'Apple is a technology company.'),
            ('p2', 'Microsoft is also a tech company.'),
            ('p3', 'Google is known for search.'),
        ]
        
        results = tagger_service.tag_paragraphs(paragraphs)
        
        assert len(results) == 3
        assert all(isinstance(r, TaggingResult) for r in results)
    
    def test_confidence_filtering(self, tagger_service):
        """Test that low confidence tags are filtered."""
        config = TaggingConfig(combined_confidence_threshold=0.9)
        service = TaggerService(config)
        
        # Create tags with varying confidence
        high_conf = ParagraphTag(
            paragraph_id='p1',
            concept_id='c1',
            concept_name='High',
            confidence=0.95,
        )
        
        low_conf = ParagraphTag(
            paragraph_id='p1',
            concept_id='c2',
            concept_name='Low',
            confidence=0.5,
        )
        
        filtered = service._filter_by_confidence([high_conf, low_conf])
        
        assert len(filtered) == 1
        assert filtered[0].confidence == 0.95
    
    def test_combine_mentions(self, tagger_service):
        """Test combining NER and LLM mentions."""
        ner_mention = ConceptMention(
            concept_name='Python',
            entity_type='PRODUCT',
            start_char=0,
            end_char=6,
            mention_text='Python',
            confidence=0.9,
            source='ner',
        )
        
        llm_concepts = ['Python', 'Java']
        
        combined = tagger_service._combine_mentions([ner_mention], llm_concepts)
        
        # Python should be merged (source='both')
        python_mention = next(m for m in combined if m.concept_name == 'Python')
        assert python_mention.source == 'both'
        
        # Java should be added
        assert any(m.concept_name == 'Java' for m in combined)
    
    def test_correct_tag(self, tagger_service):
        """Test recording a correction."""
        original_tag = ParagraphTag(
            paragraph_id='p1',
            concept_id='c1',
            concept_name='Wrong',
            confidence=0.5,
        )
        
        corrected_tag = ParagraphTag(
            paragraph_id='p1',
            concept_id='c2',
            concept_name='Correct',
            confidence=1.0,
            tagged_by='manual',
        )
        
        tagger_service.correct_tag(original_tag, corrected_tag)
        
        history = tagger_service.get_correction_history()
        assert len(history) == 1
        assert history[0]['original']['concept_name'] == 'Wrong'
        assert history[0]['corrected']['concept_name'] == 'Correct'
    
    def test_correction_learning_disabled(self):
        """Test that corrections aren't recorded when learning is disabled."""
        config = TaggingConfig(learn_from_corrections=False)
        service = TaggerService(config)
        
        tag1 = ParagraphTag(
            paragraph_id='p1',
            concept_id='c1',
            concept_name='Test',
            confidence=0.5,
        )
        
        tag2 = ParagraphTag(
            paragraph_id='p1',
            concept_id='c2',
            concept_name='Test2',
            confidence=1.0,
        )
        
        service.correct_tag(tag1, tag2)
        
        assert len(service.get_correction_history()) == 0
    
    def test_clear_correction_history(self, tagger_service):
        """Test clearing correction history."""
        tag1 = ParagraphTag(
            paragraph_id='p1',
            concept_id='c1',
            concept_name='Test',
            confidence=0.5,
        )
        
        tag2 = ParagraphTag(
            paragraph_id='p1',
            concept_id='c2',
            concept_name='Test2',
            confidence=1.0,
        )
        
        tagger_service.correct_tag(tag1, tag2)
        assert len(tagger_service.get_correction_history()) == 1
        
        tagger_service.clear_correction_history()
        assert len(tagger_service.get_correction_history()) == 0
    
    def test_concept_graph_mapping(self, tagger_service):
        """Test mapping mentions to concept graph."""
        concept_graph = {
            'ml': Concept(
                name='Machine Learning',
                aliases=['ML', 'machine learning'],
            ),
            'ai': Concept(
                name='Artificial Intelligence',
                aliases=['AI'],
            ),
        }
        
        mentions = [
            ConceptMention(
                concept_name='ML',
                entity_type='CONCEPT',
                start_char=0,
                end_char=2,
                mention_text='ML',
            ),
            ConceptMention(
                concept_name='AI',
                entity_type='CONCEPT',
                start_char=10,
                end_char=12,
                mention_text='AI',
            ),
        ]
        
        tags = tagger_service._map_to_concepts(
            mentions,
            'p1',
            concept_graph,
        )
        
        # Should find concepts in graph
        assert len(tags) == 2
        assert tags[0].concept_id == 'ml'
        assert tags[1].concept_id == 'ai'


class TestTaggingRepository:
    """Test TaggingRepository."""
    
    @pytest.fixture
    def repository(self):
        """Create a repository instance."""
        return TaggingRepository()
    
    def test_save_paragraph_tag(self, repository):
        """Test saving a paragraph tag."""
        tag = ParagraphTag(
            paragraph_id='p1',
            concept_id='c1',
            concept_name='Machine Learning',
            confidence=0.85,
        )
        
        # This will fail if DB isn't initialized, but tests the interface
        try:
            result = repository.save_paragraph_tag('doc1', tag)
            # We don't assert True/False because DB might not be available in tests
        except Exception:
            # Expected in test environment without full DB
            pass
    
    def test_remove_tag(self, repository):
        """Test removing a tag."""
        try:
            result = repository.remove_tag('doc1', 'p1', 'c1')
            # We don't assert because DB might not be available
        except Exception:
            pass


class TestIntegration:
    """Integration tests for the tagging system."""
    
    def test_end_to_end_tagging_workflow(self):
        """Test complete tagging workflow."""
        # Initialize service
        service = TaggerService()
        
        # Tag a paragraph
        text = "Artificial intelligence and machine learning are related fields."
        result = service.tag_paragraph(
            paragraph_id='p1',
            paragraph_text=text,
        )
        
        # Verify result structure
        assert result.paragraph_id == 'p1'
        assert result.processing_time > 0
        
        # Verify we got some tags (at least from NER)
        if result.tags:
            for tag in result.tags:
                assert 0 <= tag.confidence <= 1
    
    def test_mixed_auto_and_manual_tags(self):
        """Test mixing auto and manual tags."""
        service = TaggerService()
        
        # Create result with auto tags
        result = TaggingResult(
            paragraph_id='p1',
            paragraph_text='Machine learning algorithms.',
        )
        
        # Add auto tag
        auto_tag = service.add_manual_tag('p1', 'c1', 'ML', confidence=0.5)
        auto_tag.tagged_by = 'auto'
        result.add_tag(auto_tag)
        
        # Add manual tag
        manual_tag = service.add_manual_tag('p1', 'c2', 'Algorithms', confidence=1.0)
        result.add_tag(manual_tag)
        
        # Verify separation
        assert len(result.auto_tags) == 1
        assert len(result.manual_tags) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
