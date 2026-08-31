"""
User Profile Service - Manages user profiles, concept mastery tracking, and learning preferences.

Provides functionality for:
- Binary classification of concepts as known/unknown
- Dynamic mastery level calculation
- Learning preference management
- Profile persistence through the repository layer
"""

from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime

from ..models.user_profile import (
    UserProfile,
    ConceptMastery,
    LearningPreferences,
    MasteryLevel,
    ExplanationDepth,
    LearningPace,
    PreferredModality,
    QuizFrequency
)


class UserProfileService:
    """Service for managing user profiles and mastery tracking."""
    
    # Default thresholds
    DEFAULT_KNOWN_THRESHOLD = 0.7
    DEFAULT_MASTERY_THRESHOLDS = {
        MasteryLevel.NOVICE: (0.0, 0.3),
        MasteryLevel.INTERMEDIATE: (0.3, 0.7),
        MasteryLevel.EXPERT: (0.7, 0.9),
        MasteryLevel.MASTERED: (0.9, 1.0)
    }
    
    def __init__(self, user_id: str):
        """
        Initialize the user profile service.
        
        Args:
            user_id: Unique identifier for the user
        """
        self.user_id = user_id
        self.profile = UserProfile(user_id=user_id)
        self._known_concepts_cache: Dict[str, bool] = {}
        self._known_threshold = self.DEFAULT_KNOWN_THRESHOLD
    
    # ==================== Profile Management ====================
    
    def get_profile(self) -> UserProfile:
        """
        Get the current user profile.
        
        Returns:
            UserProfile: The complete user profile
        """
        return self.profile
    
    def set_profile(self, profile: UserProfile):
        """
        Set the user profile.
        
        Args:
            profile: The new user profile
        """
        self.profile = profile
        self._invalidate_cache()
    
    def update_last_active(self):
        """Update the last active timestamp to now."""
        self.profile.last_active = datetime.now()
    
    # ==================== Known/Unknown Concept Classification ====================
    
    def classify_concept(self, concept_name: str, p_knowledge: float, 
                        threshold: Optional[float] = None) -> bool:
        """
        Classify a concept as known or unknown based on probability threshold.
        
        Args:
            concept_name: Name of the concept to classify
            p_knowledge: Probability of knowing the concept (0-1)
            threshold: Override default threshold for this classification
        
        Returns:
            bool: True if concept is known (p_knowledge >= threshold), False otherwise
        """
        threshold = threshold or self._known_threshold
        is_known = p_knowledge >= threshold
        
        # Update cache and profile
        self._known_concepts_cache[concept_name] = is_known
        self.profile.update_concept_mastery(concept_name, p_knowledge)
        
        return is_known
    
    def get_known_concepts(self, threshold: Optional[float] = None) -> Dict[str, ConceptMastery]:
        """
        Get all known concepts above threshold with quick lookup.
        
        Args:
            threshold: Override default threshold
        
        Returns:
            Dict mapping concept name to ConceptMastery
        """
        threshold = threshold or self._known_threshold
        return self.profile.get_known_concepts(threshold)
    
    def get_unknown_concepts(self, threshold: Optional[float] = None) -> Dict[str, ConceptMastery]:
        """
        Get all unknown concepts below threshold.
        
        Args:
            threshold: Override default threshold
        
        Returns:
            Dict mapping concept name to ConceptMastery
        """
        threshold = threshold or self._known_threshold
        return self.profile.get_unknown_concepts(threshold)
    
    def is_concept_known(self, concept_name: str, threshold: Optional[float] = None) -> bool:
        """
        Quick lookup to check if concept is known.
        
        Args:
            concept_name: Name of the concept
            threshold: Override default threshold
        
        Returns:
            bool: True if concept is known above threshold
        """
        # Check cache first
        if concept_name in self._known_concepts_cache:
            return self._known_concepts_cache[concept_name]
        
        threshold = threshold or self._known_threshold
        
        if concept_name in self.profile.known_concepts:
            is_known = self.profile.known_concepts[concept_name].is_known(threshold)
            self._known_concepts_cache[concept_name] = is_known
            return is_known
        
        return False
    
    def get_known_concepts_set(self, threshold: Optional[float] = None) -> Set[str]:
        """
        Get set of known concept names for fast membership testing.
        
        Args:
            threshold: Override default threshold
        
        Returns:
            Set of known concept names
        """
        threshold = threshold or self._known_threshold
        return {
            name for name, mastery in self.profile.known_concepts.items()
            if mastery.is_known(threshold)
        }
    
    def set_known_threshold(self, threshold: float):
        """
        Set the threshold for classifying concepts as known.
        
        Args:
            threshold: New threshold value (0-1)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
        self._known_threshold = threshold
        self._invalidate_cache()
    
    def get_known_threshold(self) -> float:
        """Get the current known/unknown threshold."""
        return self._known_threshold
    
    # ==================== Mastery Level Management ====================
    
    def calculate_mastery_level(self, p_knowledge: float) -> MasteryLevel:
        """
        Calculate mastery level from probability of knowledge.
        
        Levels:
            - Novice: p_knowledge < 0.3
            - Intermediate: 0.3 <= p_knowledge < 0.7
            - Expert: 0.7 <= p_knowledge < 0.9
            - Mastered: p_knowledge >= 0.9
        
        Args:
            p_knowledge: Probability of knowing (0-1)
        
        Returns:
            MasteryLevel: The calculated mastery level
        """
        if p_knowledge >= 0.9:
            return MasteryLevel.MASTERED
        elif p_knowledge >= 0.7:
            return MasteryLevel.EXPERT
        elif p_knowledge >= 0.3:
            return MasteryLevel.INTERMEDIATE
        else:
            return MasteryLevel.NOVICE
    
    def get_concepts_by_mastery(self, level: MasteryLevel) -> Dict[str, ConceptMastery]:
        """
        Get all concepts at a specific mastery level.
        
        Args:
            level: The mastery level to filter by
        
        Returns:
            Dict mapping concept name to ConceptMastery
        """
        return self.profile.get_concepts_by_mastery(level)
    
    def get_mastery_distribution(self) -> Dict[str, int]:
        """
        Get count of concepts at each mastery level.
        
        Returns:
            Dict with mastery levels as keys and counts as values
        """
        return self.profile.get_mastery_distribution()
    
    def get_average_mastery(self) -> float:
        """
        Get average p_knowledge across all concepts.
        
        Returns:
            float: Average mastery (0-1)
        """
        if not self.profile.known_concepts:
            return 0.0
        
        total = sum(m.p_knowledge for m in self.profile.known_concepts.values())
        return total / len(self.profile.known_concepts)
    
    def update_concept_mastery(self, concept_name: str, p_knowledge: float, 
                              confidence: float = 0.5):
        """
        Update mastery for a concept.
        
        Args:
            concept_name: Name of the concept
            p_knowledge: New probability of knowing (0-1)
            confidence: Confidence in the estimate (0-1)
        """
        if not 0.0 <= p_knowledge <= 1.0:
            raise ValueError(f"p_knowledge must be between 0 and 1, got {p_knowledge}")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
        
        self.profile.update_concept_mastery(concept_name, p_knowledge, confidence)
        self._invalidate_cache()
    
    # ==================== Learning Preferences Management ====================
    
    def get_preferences(self) -> LearningPreferences:
        """
        Get user's learning preferences.
        
        Returns:
            LearningPreferences: Current preferences
        """
        return self.profile.preferences
    
    def update_preferences(self, **kwargs) -> LearningPreferences:
        """
        Update one or more learning preferences.
        
        Supported kwargs:
            - explanation_depth: int (1-5)
            - learning_pace: str ('slow', 'normal', 'fast')
            - preferred_modality: str ('text', 'visual', 'interactive')
            - quiz_frequency: str ('never', 'rarely', 'sometimes', 'often', 'always')
            - language_preference: str (ISO 639-1 code)
            - auto_advanced_quizzes: bool
            - detailed_feedback: bool
            - show_hints: bool
        
        Returns:
            LearningPreferences: Updated preferences
        """
        prefs = self.profile.preferences
        
        # Update explanation depth
        if 'explanation_depth' in kwargs:
            depth = kwargs['explanation_depth']
            if isinstance(depth, int):
                depth = ExplanationDepth(depth)
            elif isinstance(depth, str):
                depth = ExplanationDepth[depth.upper()]
            prefs.explanation_depth = depth
        
        # Update learning pace
        if 'learning_pace' in kwargs:
            pace = kwargs['learning_pace']
            if isinstance(pace, str):
                pace = LearningPace(pace)
            prefs.learning_pace = pace
        
        # Update preferred modality
        if 'preferred_modality' in kwargs:
            modality = kwargs['preferred_modality']
            if isinstance(modality, str):
                modality = PreferredModality(modality)
            prefs.preferred_modality = modality
        
        # Update quiz frequency
        if 'quiz_frequency' in kwargs:
            frequency = kwargs['quiz_frequency']
            if isinstance(frequency, str):
                frequency = QuizFrequency(frequency)
            prefs.quiz_frequency = frequency
        
        # Update language preference
        if 'language_preference' in kwargs:
            prefs.language_preference = kwargs['language_preference']
        
        # Update boolean preferences
        if 'auto_advanced_quizzes' in kwargs:
            prefs.auto_advanced_quizzes = kwargs['auto_advanced_quizzes']
        if 'detailed_feedback' in kwargs:
            prefs.detailed_feedback = kwargs['detailed_feedback']
        if 'show_hints' in kwargs:
            prefs.show_hints = kwargs['show_hints']
        
        prefs.updated_at = datetime.now()
        self.profile.updated_at = datetime.now()
        
        return prefs
    
    def set_explanation_depth(self, depth: ExplanationDepth):
        """Set explanation depth preference."""
        self.profile.preferences.explanation_depth = depth
        self.profile.preferences.updated_at = datetime.now()
        self.profile.updated_at = datetime.now()
    
    def set_learning_pace(self, pace: LearningPace):
        """Set learning pace preference."""
        self.profile.preferences.learning_pace = pace
        self.profile.preferences.updated_at = datetime.now()
        self.profile.updated_at = datetime.now()
    
    def set_preferred_modality(self, modality: PreferredModality):
        """Set preferred learning modality."""
        self.profile.preferences.preferred_modality = modality
        self.profile.preferences.updated_at = datetime.now()
        self.profile.updated_at = datetime.now()
    
    def set_quiz_frequency(self, frequency: QuizFrequency):
        """Set quiz frequency preference."""
        self.profile.preferences.quiz_frequency = frequency
        self.profile.preferences.updated_at = datetime.now()
        self.profile.updated_at = datetime.now()
    
    def set_language_preference(self, language: str):
        """Set language preference (ISO 639-1 code)."""
        self.profile.preferences.language_preference = language
        self.profile.preferences.updated_at = datetime.now()
        self.profile.updated_at = datetime.now()
    
    # ==================== Profile Queries and Analysis ====================
    
    def get_profile_summary(self) -> Dict:
        """
        Get a comprehensive summary of the user's profile.
        
        Returns:
            Dict containing profile statistics and preferences
        """
        return self.profile.get_profile_summary()
    
    def get_learning_statistics(self) -> Dict:
        """
        Get detailed learning statistics.
        
        Returns:
            Dict with statistics about learning progress
        """
        distribution = self.get_mastery_distribution()
        
        return {
            'total_concepts': len(self.profile.known_concepts),
            'known_concepts': len(self.get_known_concepts()),
            'unknown_concepts': len(self.get_unknown_concepts()),
            'average_mastery': self.get_average_mastery(),
            'mastery_distribution': distribution,
            'novice_count': distribution.get(MasteryLevel.NOVICE.value, 0),
            'intermediate_count': distribution.get(MasteryLevel.INTERMEDIATE.value, 0),
            'expert_count': distribution.get(MasteryLevel.EXPERT.value, 0),
            'mastered_count': distribution.get(MasteryLevel.MASTERED.value, 0)
        }
    
    def get_concepts_for_learning(self, mastery_levels: Optional[List[MasteryLevel]] = None) -> Dict[str, ConceptMastery]:
        """
        Get concepts recommended for learning.
        
        By default, returns novice and intermediate level concepts.
        
        Args:
            mastery_levels: Specific levels to include. Defaults to novice and intermediate.
        
        Returns:
            Dict mapping concept name to ConceptMastery
        """
        if mastery_levels is None:
            mastery_levels = [MasteryLevel.NOVICE, MasteryLevel.INTERMEDIATE]
        
        result = {}
        for level in mastery_levels:
            result.update(self.get_concepts_by_mastery(level))
        
        return result
    
    def get_advanced_concepts(self, include_expert: bool = True) -> Dict[str, ConceptMastery]:
        """
        Get advanced concepts (expert and/or mastered).
        
        Args:
            include_expert: Whether to include expert level concepts
        
        Returns:
            Dict mapping concept name to ConceptMastery
        """
        result = {}
        if include_expert:
            result.update(self.get_concepts_by_mastery(MasteryLevel.EXPERT))
        result.update(self.get_concepts_by_mastery(MasteryLevel.MASTERED))
        return result
    
    def mark_concept_unknown(self, concept_name: str):
        """Mark a concept as unknown."""
        self.profile.add_unknown_concept(concept_name)
        self._invalidate_cache()
    
    # ==================== Import/Export ====================
    
    def export_profile(self) -> Dict:
        """
        Export the complete profile as a dictionary.
        
        Returns:
            Dict representation of the profile
        """
        return self.profile.to_dict()
    
    def import_profile(self, data: Dict):
        """
        Import a profile from a dictionary.
        
        Args:
            data: Dictionary containing profile data
        """
        self.profile = UserProfile.from_dict(data)
        self._invalidate_cache()
    
    # ==================== Internal Methods ====================
    
    def _invalidate_cache(self):
        """Invalidate the known concepts cache."""
        self._known_concepts_cache.clear()
