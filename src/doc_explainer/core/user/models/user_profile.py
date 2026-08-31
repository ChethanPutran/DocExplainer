"""
User Profile Models - Contains data structures for user profiles and learning preferences.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set
from enum import Enum


class MasteryLevel(str, Enum):
    """Mastery levels based on p_knowledge score."""
    NOVICE = "novice"  # p_knowledge < 0.3
    INTERMEDIATE = "intermediate"  # 0.3 <= p_knowledge < 0.7
    EXPERT = "expert"  # 0.7 <= p_knowledge < 0.9
    MASTERED = "mastered"  # p_knowledge >= 0.9


class ExplanationDepth(int, Enum):
    """Explanation depth preference on 1-5 scale."""
    MINIMAL = 1
    BRIEF = 2
    STANDARD = 3
    DETAILED = 4
    COMPREHENSIVE = 5


class LearningPace(str, Enum):
    """Learning pace preferences."""
    SLOW = "slow"
    NORMAL = "normal"
    FAST = "fast"


class PreferredModality(str, Enum):
    """Preferred learning modality."""
    TEXT = "text"
    VISUAL = "visual"
    INTERACTIVE = "interactive"


class QuizFrequency(str, Enum):
    """Quiz frequency preference."""
    NEVER = "never"
    RARELY = "rarely"
    SOMETIMES = "sometimes"
    OFTEN = "often"
    ALWAYS = "always"


@dataclass
class LearningPreferences:
    """User learning preferences."""
    
    explanation_depth: ExplanationDepth = ExplanationDepth.STANDARD
    learning_pace: LearningPace = LearningPace.NORMAL
    preferred_modality: PreferredModality = PreferredModality.TEXT
    quiz_frequency: QuizFrequency = QuizFrequency.SOMETIMES
    language_preference: str = "en"  # ISO 639-1 language code
    auto_advanced_quizzes: bool = True
    detailed_feedback: bool = True
    show_hints: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert preferences to dictionary."""
        return {
            'explanation_depth': self.explanation_depth.value,
            'learning_pace': self.learning_pace.value,
            'preferred_modality': self.preferred_modality.value,
            'quiz_frequency': self.quiz_frequency.value,
            'language_preference': self.language_preference,
            'auto_advanced_quizzes': self.auto_advanced_quizzes,
            'detailed_feedback': self.detailed_feedback,
            'show_hints': self.show_hints,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LearningPreferences':
        """Create preferences from dictionary."""
        prefs = cls()
        
        if 'explanation_depth' in data:
            prefs.explanation_depth = ExplanationDepth(data['explanation_depth'])
        if 'learning_pace' in data:
            prefs.learning_pace = LearningPace(data['learning_pace'])
        if 'preferred_modality' in data:
            prefs.preferred_modality = PreferredModality(data['preferred_modality'])
        if 'quiz_frequency' in data:
            prefs.quiz_frequency = QuizFrequency(data['quiz_frequency'])
        
        prefs.language_preference = data.get('language_preference', 'en')
        prefs.auto_advanced_quizzes = data.get('auto_advanced_quizzes', True)
        prefs.detailed_feedback = data.get('detailed_feedback', True)
        prefs.show_hints = data.get('show_hints', True)
        
        if 'created_at' in data and data['created_at']:
            prefs.created_at = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and data['updated_at']:
            prefs.updated_at = datetime.fromisoformat(data['updated_at'])
        
        return prefs


@dataclass
class ConceptMastery:
    """Tracks mastery level and progression for a single concept."""
    
    concept_name: str
    p_knowledge: float = 0.1  # Probability of knowing (0-1)
    mastery_level: MasteryLevel = MasteryLevel.NOVICE
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    times_seen: int = 0
    confidence: float = 0.5
    
    def __post_init__(self):
        if self.first_seen is None:
            self.first_seen = datetime.now()
        if self.last_seen is None:
            self.last_seen = datetime.now()
        self._update_mastery_level()
    
    def _update_mastery_level(self):
        """Update mastery level based on p_knowledge."""
        if self.p_knowledge >= 0.9:
            self.mastery_level = MasteryLevel.MASTERED
        elif self.p_knowledge >= 0.7:
            self.mastery_level = MasteryLevel.EXPERT
        elif self.p_knowledge >= 0.3:
            self.mastery_level = MasteryLevel.INTERMEDIATE
        else:
            self.mastery_level = MasteryLevel.NOVICE
    
    def update_knowledge(self, p_knowledge: float):
        """Update knowledge probability and recalculate mastery level."""
        self.p_knowledge = max(0.0, min(1.0, p_knowledge))
        self.last_seen = datetime.now()
        self.times_seen += 1
        self._update_mastery_level()
    
    def is_known(self, threshold: float = 0.7) -> bool:
        """Check if concept is known above threshold."""
        return self.p_knowledge >= threshold
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'concept_name': self.concept_name,
            'p_knowledge': self.p_knowledge,
            'mastery_level': self.mastery_level.value,
            'first_seen': self.first_seen.isoformat() if self.first_seen else None,
            'last_seen': self.last_seen.isoformat() if self.last_seen else None,
            'times_seen': self.times_seen,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConceptMastery':
        """Create from dictionary."""
        cm = cls(
            concept_name=data.get('concept_name', ''),
            p_knowledge=data.get('p_knowledge', 0.1),
            confidence=data.get('confidence', 0.5)
        )
        cm.times_seen = data.get('times_seen', 0)
        
        if 'first_seen' in data and data['first_seen']:
            cm.first_seen = datetime.fromisoformat(data['first_seen'])
        if 'last_seen' in data and data['last_seen']:
            cm.last_seen = datetime.fromisoformat(data['last_seen'])
        
        if 'mastery_level' in data:
            cm.mastery_level = MasteryLevel(data['mastery_level'])
        
        cm._update_mastery_level()
        return cm


@dataclass
class UserProfile:
    """Complete user profile including knowledge state and preferences."""
    
    user_id: str
    known_concepts: Dict[str, ConceptMastery] = field(default_factory=dict)
    unknown_concepts: Set[str] = field(default_factory=set)
    preferences: LearningPreferences = field(default_factory=LearningPreferences)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_active: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.last_active is None:
            self.last_active = datetime.now()
    
    def get_known_concepts(self, threshold: float = 0.7) -> Dict[str, ConceptMastery]:
        """Get all concepts known above threshold."""
        return {
            name: mastery
            for name, mastery in self.known_concepts.items()
            if mastery.is_known(threshold)
        }
    
    def get_unknown_concepts(self, threshold: float = 0.7) -> Dict[str, ConceptMastery]:
        """Get all concepts unknown below threshold."""
        return {
            name: mastery
            for name, mastery in self.known_concepts.items()
            if not mastery.is_known(threshold)
        }
    
    def get_concepts_by_mastery(self, level: MasteryLevel) -> Dict[str, ConceptMastery]:
        """Get all concepts at a specific mastery level."""
        return {
            name: mastery
            for name, mastery in self.known_concepts.items()
            if mastery.mastery_level == level
        }
    
    def update_concept_mastery(self, concept_name: str, p_knowledge: float, confidence: float = 0.5):
        """Update mastery for a concept."""
        if concept_name in self.known_concepts:
            self.known_concepts[concept_name].update_knowledge(p_knowledge)
            self.known_concepts[concept_name].confidence = confidence
        else:
            cm = ConceptMastery(
                concept_name=concept_name,
                p_knowledge=p_knowledge,
                confidence=confidence
            )
            self.known_concepts[concept_name] = cm
        
        self.updated_at = datetime.now()
        self.last_active = datetime.now()
        
        # Update unknown concepts set
        if p_knowledge >= 0.7:
            self.unknown_concepts.discard(concept_name)
        else:
            self.unknown_concepts.add(concept_name)
    
    def add_unknown_concept(self, concept_name: str):
        """Mark a concept as unknown."""
        self.unknown_concepts.add(concept_name)
        if concept_name not in self.known_concepts:
            self.known_concepts[concept_name] = ConceptMastery(
                concept_name=concept_name,
                p_knowledge=0.1,
                confidence=0.5
            )
        self.updated_at = datetime.now()
    
    def get_mastery_distribution(self) -> Dict[str, int]:
        """Get count of concepts at each mastery level."""
        distribution = {
            MasteryLevel.NOVICE.value: 0,
            MasteryLevel.INTERMEDIATE.value: 0,
            MasteryLevel.EXPERT.value: 0,
            MasteryLevel.MASTERED.value: 0
        }
        
        for mastery in self.known_concepts.values():
            distribution[mastery.mastery_level.value] += 1
        
        return distribution
    
    def get_profile_summary(self) -> Dict:
        """Get a summary of the user's profile."""
        known = self.get_known_concepts()
        
        return {
            'user_id': self.user_id,
            'total_concepts': len(self.known_concepts),
            'known_concepts_count': len(known),
            'unknown_concepts_count': len(self.get_unknown_concepts()),
            'mastery_distribution': self.get_mastery_distribution(),
            'average_mastery': sum(m.p_knowledge for m in self.known_concepts.values()) / max(1, len(self.known_concepts)),
            'preferences': self.preferences.to_dict(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_active': self.last_active.isoformat() if self.last_active else None
        }
    
    def to_dict(self) -> Dict:
        """Convert profile to dictionary."""
        return {
            'user_id': self.user_id,
            'known_concepts': {
                name: mastery.to_dict()
                for name, mastery in self.known_concepts.items()
            },
            'unknown_concepts': list(self.unknown_concepts),
            'preferences': self.preferences.to_dict(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_active': self.last_active.isoformat() if self.last_active else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserProfile':
        """Create profile from dictionary."""
        profile = cls(user_id=data.get('user_id', ''))
        
        # Load known concepts
        for name, concept_data in data.get('known_concepts', {}).items():
            profile.known_concepts[name] = ConceptMastery.from_dict(concept_data)
        
        # Load unknown concepts
        profile.unknown_concepts = set(data.get('unknown_concepts', []))
        
        # Load preferences
        if 'preferences' in data:
            profile.preferences = LearningPreferences.from_dict(data['preferences'])
        
        # Load timestamps
        if 'created_at' in data and data['created_at']:
            profile.created_at = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and data['updated_at']:
            profile.updated_at = datetime.fromisoformat(data['updated_at'])
        if 'last_active' in data and data['last_active']:
            profile.last_active = datetime.fromisoformat(data['last_active'])
        
        return profile
