from typing import Dict, List, Optional
from datetime import datetime, timedelta
from ..models.dataclasses import ConceptMastery, QuestionAttempt
from ..models.schemas import Question


class MasteryTracker:
    """Track concept mastery over time"""
    
    def __init__(self, decay_rate: float = 0.1):
        self.decay_rate = decay_rate  # Rate at which mastery decays without practice
        self.concepts: Dict[str, ConceptMastery] = {}
        self.attempt_history: List[QuestionAttempt] = []
    
    def update_from_attempt(self, attempt: QuestionAttempt, question: Question):
        """Update mastery based on question attempt"""
        concept = question.concept
        
        # Get or create concept mastery
        if concept not in self.concepts:
            self.concepts[concept] = ConceptMastery(concept=concept)
        
        mastery = self.concepts[concept]
        
        # Add question to attempted list
        if question.id not in mastery.questions_attempted:
            mastery.questions_attempted.append(question.id)
        
        # Update mastery
        mastery.update(attempt.is_correct)
        
        # Store attempt
        self.attempt_history.append(attempt)
    
    def get_mastery(self, concept: str) -> Optional[float]:
        """Get mastery level for a concept"""
        if concept in self.concepts:
            return self.concepts[concept].mastery_level
        return None
    
    def get_all_mastery(self) -> Dict[str, float]:
        """Get mastery levels for all concepts"""
        return {c: m.mastery_level for c, m in self.concepts.items()}
    
    def get_accuracy(self, concept: str) -> Optional[float]:
        """Get accuracy for a concept"""
        if concept in self.concepts:
            return self.concepts[concept].accuracy
        return None
    
    def apply_decay(self, hours: int = 24):
        """Apply decay to all concepts based on time since last attempt"""
        for concept, mastery in self.concepts.items():
            if mastery.last_attempt:
                hours_since = (datetime.now() - mastery.last_attempt).total_seconds() / 3600
                if hours_since > hours:
                    # Apply decay
                    decay_factor = 1.0 - (self.decay_rate * (hours_since / hours))
                    mastery.mastery_level = max(0.0, mastery.mastery_level * decay_factor)
    
    def needs_review(self, concept: str, threshold: float = 0.7) -> bool:
        """Check if a concept needs review"""
        mastery = self.get_mastery(concept)
        if mastery is None:
            return True  # Unknown concept needs review
        
        # Check if mastery is below threshold
        if mastery < threshold:
            return True
        
        # Check if it's been a while since last attempt
        if concept in self.concepts:
            last = self.concepts[concept].last_attempt
            if last and (datetime.now() - last).days > 7:
                return True  # More than a week since practice
        
        return False
    
    def get_concepts_for_review(self, threshold: float = 0.7) -> List[str]:
        """Get concepts that need review"""
        return [c for c in self.concepts.keys() if self.needs_review(c, threshold)]
    
    def get_mastery_summary(self) -> Dict:
        """Get summary of concept mastery"""
        concepts = self.get_all_mastery()
        
        if not concepts:
            return {
                'average_mastery': 0.0,
                'mastered': [],
                'learning': [],
                'needs_review': []
            }
        
        mastered = []
        learning = []
        needs_review = []
        
        for concept, mastery in concepts.items():
            if mastery >= 0.8:
                mastered.append(concept)
            elif mastery >= 0.4:
                learning.append(concept)
            else:
                needs_review.append(concept)
        
        return {
            'average_mastery': sum(concepts.values()) / len(concepts),
            'mastered': mastered,
            'learning': learning,
            'needs_review': needs_review,
            'total_concepts': len(concepts)
        }
    
    def get_weakest_concepts(self, limit: int = 3) -> List[str]:
        """Get the weakest concepts"""
        sorted_concepts = sorted(self.concepts.items(), 
                                key=lambda x: x[1].mastery_level)
        return [c[0] for c in sorted_concepts[:limit]]
    
    def get_strongest_concepts(self, limit: int = 3) -> List[str]:
        """Get the strongest concepts"""
        sorted_concepts = sorted(self.concepts.items(), 
                                key=lambda x: x[1].mastery_level, 
                                reverse=True)
        return [c[0] for c in sorted_concepts[:limit]]