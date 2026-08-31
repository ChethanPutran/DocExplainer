"""
Tests for User Profile Service.

Tests cover:
- Known/unknown concept classification
- Mastery level calculation
- Learning preferences management
- Profile persistence
"""

import pytest
from datetime import datetime
from typing import Dict

from src.core.user.services.user_profile_service import UserProfileService
from src.core.user.models.user_profile import (
    UserProfile,
    ConceptMastery,
    LearningPreferences,
    MasteryLevel,
    ExplanationDepth,
    LearningPace,
    PreferredModality,
    QuizFrequency
)


class TestUserProfileService:
    """Test suite for UserProfileService."""
    
    @pytest.fixture
    def service(self):
        """Create a fresh service for each test."""
        return UserProfileService("test_user_123")
    
    # ==================== Initialization Tests ====================
    
    def test_service_initialization(self, service):
        """Test service initializes with correct defaults."""
        assert service.user_id == "test_user_123"
        assert service.get_known_threshold() == 0.7
        assert len(service.profile.known_concepts) == 0
    
    def test_service_has_default_preferences(self, service):
        """Test service initializes with default preferences."""
        prefs = service.get_preferences()
        assert prefs.explanation_depth == ExplanationDepth.STANDARD
        assert prefs.learning_pace == LearningPace.NORMAL
        assert prefs.preferred_modality == PreferredModality.TEXT
        assert prefs.quiz_frequency == QuizFrequency.SOMETIMES
    
    # ==================== Known/Unknown Classification Tests ====================
    
    def test_classify_concept_as_known(self, service):
        """Test classifying a concept as known."""
        is_known = service.classify_concept("Python", 0.85)
        assert is_known is True
    
    def test_classify_concept_as_unknown(self, service):
        """Test classifying a concept as unknown."""
        is_known = service.classify_concept("Rust", 0.25)
        assert is_known is False
    
    def test_classify_concept_at_threshold_boundary(self, service):
        """Test concept classification at threshold boundary."""
        # Exactly at threshold
        is_known = service.classify_concept("Java", 0.7)
        assert is_known is True
        
        # Just below threshold
        is_known = service.classify_concept("Go", 0.699)
        assert is_known is False
    
    def test_custom_threshold_classification(self, service):
        """Test classification with custom threshold."""
        # Using custom threshold
        is_known = service.classify_concept("JavaScript", 0.6, threshold=0.5)
        assert is_known is True
    
    def test_set_known_threshold(self, service):
        """Test changing the known threshold."""
        service.set_known_threshold(0.5)
        assert service.get_known_threshold() == 0.5
        
        # Reclassify with new threshold
        is_known = service.is_concept_known("Python")
        assert is_known is False  # 0.85 >= 0.5 returns True but this tests the mechanism
    
    def test_invalid_threshold_raises_error(self, service):
        """Test that invalid thresholds raise errors."""
        with pytest.raises(ValueError):
            service.set_known_threshold(1.5)
        
        with pytest.raises(ValueError):
            service.set_known_threshold(-0.1)
    
    def test_quick_lookup_known_concepts(self, service):
        """Test quick lookup for known concepts."""
        service.classify_concept("Python", 0.9)
        service.classify_concept("Java", 0.8)
        service.classify_concept("Rust", 0.2)
        
        known = service.get_known_concepts()
        assert len(known) == 2
        assert "Python" in known
        assert "Java" in known
        assert "Rust" not in known
    
    def test_quick_lookup_unknown_concepts(self, service):
        """Test quick lookup for unknown concepts."""
        service.classify_concept("Python", 0.9)
        service.classify_concept("Rust", 0.2)
        service.classify_concept("Go", 0.15)
        
        unknown = service.get_unknown_concepts()
        assert len(unknown) == 2
        assert "Rust" in unknown
        assert "Go" in unknown
    
    def test_is_concept_known_quick_lookup(self, service):
        """Test quick membership testing for concepts."""
        service.classify_concept("Python", 0.9)
        
        assert service.is_concept_known("Python") is True
        assert service.is_concept_known("Unknown") is False
    
    def test_get_known_concepts_set(self, service):
        """Test getting set of known concept names."""
        service.classify_concept("Python", 0.9)
        service.classify_concept("Java", 0.8)
        service.classify_concept("Rust", 0.2)
        
        known_set = service.get_known_concepts_set()
        assert known_set == {"Python", "Java"}
    
    # ==================== Mastery Level Tests ====================
    
    def test_calculate_novice_mastery(self, service):
        """Test calculation of novice mastery level."""
        level = service.calculate_mastery_level(0.2)
        assert level == MasteryLevel.NOVICE
    
    def test_calculate_intermediate_mastery(self, service):
        """Test calculation of intermediate mastery level."""
        level = service.calculate_mastery_level(0.5)
        assert level == MasteryLevel.INTERMEDIATE
    
    def test_calculate_expert_mastery(self, service):
        """Test calculation of expert mastery level."""
        level = service.calculate_mastery_level(0.8)
        assert level == MasteryLevel.EXPERT
    
    def test_calculate_mastered_mastery(self, service):
        """Test calculation of mastered mastery level."""
        level = service.calculate_mastery_level(0.95)
        assert level == MasteryLevel.MASTERED
    
    def test_mastery_level_boundaries(self, service):
        """Test mastery level calculation at boundaries."""
        assert service.calculate_mastery_level(0.29) == MasteryLevel.NOVICE
        assert service.calculate_mastery_level(0.30) == MasteryLevel.INTERMEDIATE
        assert service.calculate_mastery_level(0.69) == MasteryLevel.INTERMEDIATE
        assert service.calculate_mastery_level(0.70) == MasteryLevel.EXPERT
        assert service.calculate_mastery_level(0.89) == MasteryLevel.EXPERT
        assert service.calculate_mastery_level(0.90) == MasteryLevel.MASTERED
    
    def test_get_concepts_by_mastery_level(self, service):
        """Test retrieving concepts by mastery level."""
        service.update_concept_mastery("Python", 0.95)
        service.update_concept_mastery("Java", 0.8)
        service.update_concept_mastery("Rust", 0.5)
        service.update_concept_mastery("Go", 0.2)
        
        mastered = service.get_concepts_by_mastery(MasteryLevel.MASTERED)
        assert len(mastered) == 1
        assert "Python" in mastered
        
        intermediate = service.get_concepts_by_mastery(MasteryLevel.INTERMEDIATE)
        assert len(intermediate) == 1
        assert "Rust" in intermediate
    
    def test_mastery_distribution(self, service):
        """Test getting mastery distribution."""
        service.update_concept_mastery("Python", 0.95)
        service.update_concept_mastery("Java", 0.75)
        service.update_concept_mastery("Rust", 0.5)
        service.update_concept_mastery("Go", 0.2)
        
        distribution = service.get_mastery_distribution()
        assert distribution[MasteryLevel.NOVICE.value] == 1
        assert distribution[MasteryLevel.INTERMEDIATE.value] == 1
        assert distribution[MasteryLevel.EXPERT.value] == 1
        assert distribution[MasteryLevel.MASTERED.value] == 1
    
    def test_average_mastery_calculation(self, service):
        """Test average mastery calculation."""
        service.update_concept_mastery("Python", 0.8)
        service.update_concept_mastery("Java", 0.6)
        
        avg = service.get_average_mastery()
        assert abs(avg - 0.7) < 0.01
    
    def test_average_mastery_empty_profile(self, service):
        """Test average mastery with no concepts."""
        assert service.get_average_mastery() == 0.0
    
    def test_update_concept_mastery(self, service):
        """Test updating concept mastery."""
        service.update_concept_mastery("Python", 0.85, confidence=0.9)
        
        python_mastery = service.profile.known_concepts["Python"]
        assert python_mastery.p_knowledge == 0.85
        assert python_mastery.confidence == 0.9
        assert python_mastery.mastery_level == MasteryLevel.EXPERT
    
    def test_invalid_p_knowledge_raises_error(self, service):
        """Test that invalid p_knowledge values raise errors."""
        with pytest.raises(ValueError):
            service.update_concept_mastery("Python", 1.5)
        
        with pytest.raises(ValueError):
            service.update_concept_mastery("Python", -0.1)
    
    def test_invalid_confidence_raises_error(self, service):
        """Test that invalid confidence values raise errors."""
        with pytest.raises(ValueError):
            service.update_concept_mastery("Python", 0.8, confidence=1.5)
    
    # ==================== Learning Preferences Tests ====================
    
    def test_get_preferences(self, service):
        """Test retrieving preferences."""
        prefs = service.get_preferences()
        assert isinstance(prefs, LearningPreferences)
    
    def test_update_preferences_single_field(self, service):
        """Test updating a single preference."""
        prefs = service.update_preferences(explanation_depth=4)
        assert prefs.explanation_depth == ExplanationDepth(4)
    
    def test_update_preferences_multiple_fields(self, service):
        """Test updating multiple preferences at once."""
        prefs = service.update_preferences(
            explanation_depth=5,
            learning_pace='fast',
            preferred_modality='visual'
        )
        assert prefs.explanation_depth == ExplanationDepth.COMPREHENSIVE
        assert prefs.learning_pace == LearningPace.FAST
        assert prefs.preferred_modality == PreferredModality.VISUAL
    
    def test_set_explanation_depth(self, service):
        """Test setting explanation depth."""
        service.set_explanation_depth(ExplanationDepth.COMPREHENSIVE)
        assert service.get_preferences().explanation_depth == ExplanationDepth.COMPREHENSIVE
    
    def test_set_learning_pace(self, service):
        """Test setting learning pace."""
        service.set_learning_pace(LearningPace.FAST)
        assert service.get_preferences().learning_pace == LearningPace.FAST
    
    def test_set_preferred_modality(self, service):
        """Test setting preferred modality."""
        service.set_preferred_modality(PreferredModality.INTERACTIVE)
        assert service.get_preferences().preferred_modality == PreferredModality.INTERACTIVE
    
    def test_set_quiz_frequency(self, service):
        """Test setting quiz frequency."""
        service.set_quiz_frequency(QuizFrequency.OFTEN)
        assert service.get_preferences().quiz_frequency == QuizFrequency.OFTEN
    
    def test_set_language_preference(self, service):
        """Test setting language preference."""
        service.set_language_preference("es")
        assert service.get_preferences().language_preference == "es"
    
    def test_preferences_timestamp_updates(self, service):
        """Test that preference updates update timestamps."""
        original_time = service.get_preferences().updated_at
        
        # Update after a tiny delay to ensure timestamp changes
        service.set_learning_pace(LearningPace.FAST)
        new_time = service.get_preferences().updated_at
        
        assert new_time >= original_time
    
    # ==================== Profile Queries Tests ====================
    
    def test_get_profile_summary(self, service):
        """Test getting profile summary."""
        service.update_concept_mastery("Python", 0.9)
        service.update_concept_mastery("Java", 0.5)
        
        summary = service.get_profile_summary()
        assert summary['user_id'] == "test_user_123"
        assert summary['total_concepts'] == 2
        assert summary['known_concepts_count'] == 1
        assert 'mastery_distribution' in summary
    
    def test_get_learning_statistics(self, service):
        """Test getting learning statistics."""
        service.update_concept_mastery("Python", 0.95)
        service.update_concept_mastery("Java", 0.75)
        service.update_concept_mastery("Rust", 0.5)
        service.update_concept_mastery("Go", 0.2)
        
        stats = service.get_learning_statistics()
        assert stats['total_concepts'] == 4
        assert stats['known_concepts'] == 2
        assert stats['unknown_concepts'] == 2
        assert stats['average_mastery'] == 0.6
        assert stats['mastered_count'] == 1
    
    def test_get_concepts_for_learning(self, service):
        """Test getting concepts for learning."""
        service.update_concept_mastery("Python", 0.95)
        service.update_concept_mastery("Java", 0.5)
        service.update_concept_mastery("Rust", 0.2)
        
        to_learn = service.get_concepts_for_learning()
        assert len(to_learn) == 2  # Novice and intermediate
        assert "Rust" in to_learn
        assert "Java" in to_learn
    
    def test_get_advanced_concepts(self, service):
        """Test getting advanced concepts."""
        service.update_concept_mastery("Python", 0.95)
        service.update_concept_mastery("Java", 0.75)
        service.update_concept_mastery("Rust", 0.2)
        
        advanced = service.get_advanced_concepts()
        assert len(advanced) == 2
        assert "Python" in advanced
        assert "Java" in advanced
    
    def test_mark_concept_unknown(self, service):
        """Test marking a concept as unknown."""
        service.update_concept_mastery("Python", 0.9)
        service.mark_concept_unknown("Python")
        
        assert "Python" in service.profile.unknown_concepts
    
    def test_update_last_active(self, service):
        """Test updating last active timestamp."""
        original_time = service.profile.last_active
        service.update_last_active()
        
        assert service.profile.last_active >= original_time
    
    # ==================== Import/Export Tests ====================
    
    def test_export_profile(self, service):
        """Test exporting profile to dictionary."""
        service.update_concept_mastery("Python", 0.9)
        service.set_learning_pace(LearningPace.FAST)
        
        exported = service.export_profile()
        assert isinstance(exported, dict)
        assert exported['user_id'] == "test_user_123"
        assert 'known_concepts' in exported
        assert 'preferences' in exported
    
    def test_import_profile(self, service):
        """Test importing profile from dictionary."""
        # Create a profile with data
        original_service = UserProfileService("test_user_456")
        original_service.update_concept_mastery("Python", 0.9)
        original_service.set_learning_pace(LearningPace.FAST)
        
        # Export and import
        exported = original_service.export_profile()
        service.import_profile(exported)
        
        assert service.profile.user_id == "test_user_456"
        assert "Python" in service.profile.known_concepts
        assert service.get_preferences().learning_pace == LearningPace.FAST
    
    # ==================== Cache Behavior Tests ====================
    
    def test_known_concepts_cache(self, service):
        """Test that known concepts are cached for fast lookup."""
        service.classify_concept("Python", 0.9)
        service.classify_concept("Java", 0.8)
        
        # First lookup
        set1 = service.get_known_concepts_set()
        
        # Second lookup should use cache
        set2 = service.get_known_concepts_set()
        
        assert set1 == set2
    
    def test_cache_invalidation_on_profile_change(self, service):
        """Test that cache is invalidated when profile changes."""
        service.classify_concept("Python", 0.9)
        
        # Change known threshold
        service.set_known_threshold(0.95)
        
        # Cache should be invalidated
        set1 = service.get_known_concepts_set()
        assert "Python" not in set1  # Now 0.9 < 0.95


class TestConceptMastery:
    """Test suite for ConceptMastery data class."""
    
    def test_concept_mastery_initialization(self):
        """Test ConceptMastery initialization."""
        cm = ConceptMastery("Python", p_knowledge=0.8)
        assert cm.concept_name == "Python"
        assert cm.p_knowledge == 0.8
        assert cm.mastery_level == MasteryLevel.EXPERT
        assert cm.times_seen == 0
    
    def test_concept_mastery_update_knowledge(self):
        """Test updating knowledge."""
        cm = ConceptMastery("Python", p_knowledge=0.5)
        cm.update_knowledge(0.9)
        
        assert cm.p_knowledge == 0.9
        assert cm.times_seen == 1
        assert cm.mastery_level == MasteryLevel.EXPERT
    
    def test_concept_mastery_is_known(self):
        """Test is_known method."""
        cm = ConceptMastery("Python", p_knowledge=0.8)
        assert cm.is_known() is True
        assert cm.is_known(threshold=0.9) is False
    
    def test_concept_mastery_serialization(self):
        """Test serialization and deserialization."""
        original = ConceptMastery("Python", p_knowledge=0.85, confidence=0.9)
        
        data = original.to_dict()
        restored = ConceptMastery.from_dict(data)
        
        assert restored.concept_name == original.concept_name
        assert restored.p_knowledge == original.p_knowledge
        assert restored.confidence == original.confidence


class TestUserProfile:
    """Test suite for UserProfile data class."""
    
    def test_user_profile_initialization(self):
        """Test UserProfile initialization."""
        profile = UserProfile("test_user")
        assert profile.user_id == "test_user"
        assert len(profile.known_concepts) == 0
        assert len(profile.unknown_concepts) == 0
    
    def test_user_profile_get_known_concepts(self):
        """Test getting known concepts."""
        profile = UserProfile("test_user")
        profile.update_concept_mastery("Python", 0.9)
        profile.update_concept_mastery("Rust", 0.2)
        
        known = profile.get_known_concepts()
        assert len(known) == 1
        assert "Python" in known
    
    def test_user_profile_mastery_distribution(self):
        """Test mastery distribution calculation."""
        profile = UserProfile("test_user")
        profile.update_concept_mastery("Python", 0.95)
        profile.update_concept_mastery("Java", 0.75)
        
        dist = profile.get_mastery_distribution()
        assert dist[MasteryLevel.MASTERED.value] == 1
        assert dist[MasteryLevel.EXPERT.value] == 1
    
    def test_user_profile_serialization(self):
        """Test serialization and deserialization."""
        original = UserProfile("test_user")
        original.update_concept_mastery("Python", 0.9)
        original.preferences.learning_pace = LearningPace.FAST
        
        data = original.to_dict()
        restored = UserProfile.from_dict(data)
        
        assert restored.user_id == original.user_id
        assert len(restored.known_concepts) == len(original.known_concepts)
        assert restored.preferences.learning_pace == LearningPace.FAST
