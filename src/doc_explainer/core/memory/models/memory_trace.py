from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime


@dataclass
class MemoryTrace:
    """Base class for memory traces"""
    id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    strength: float = 1.0
    
    def access(self):
        """Record an access to this memory trace"""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def decay(self, decay_rate: float = 0.1):
        """Apply decay to memory strength"""
        time_diff = (datetime.now() - self.last_accessed).total_seconds() / 3600  # hours
        self.strength *= (1 - decay_rate * time_diff)
        self.strength = max(0.0, min(1.0, self.strength))

    
@dataclass
class ConceptMemoryTrace(MemoryTrace):
    """Memory trace for a concept"""
    concept: str = ""
    understanding_level: float = 0.0  # 0-1
    review_count: int = 0
    next_review: Optional[datetime] = None
    related_concepts: List[str] = field(default_factory=list)
    interactions: List[Dict] = field(default_factory=list)
    
    def add_interaction(self, interaction_type: str, data: Dict):
        """Add an interaction to the trace"""
        self.interactions.append({
            'type': interaction_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
        self.access()
    
    def update_understanding(self, new_level: float):
        """Update understanding level"""
        self.understanding_level = new_level
        self.access()
    
    def schedule_review(self, review_interval_hours: int = 24):
        """Schedule next review"""
        from datetime import timedelta
        self.next_review = datetime.now() + timedelta(hours=review_interval_hours)
        self.review_count += 1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'concept': self.concept,
            'understanding_level': self.understanding_level,
            'review_count': self.review_count,
            'next_review': self.next_review.isoformat() if self.next_review else None,
            'related_concepts': self.related_concepts,
            'interactions': self.interactions,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'strength': self.strength
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConceptMemoryTrace':
        """Create from dictionary"""
        trace = cls(
            id=data['id'],
            concept=data['concept'],
            understanding_level=data.get('understanding_level', 0.0),
            review_count=data.get('review_count', 0),
            related_concepts=data.get('related_concepts', []),
            interactions=data.get('interactions', []),
            access_count=data.get('access_count', 0),
            strength=data.get('strength', 1.0)
        )
        
        if data.get('created_at'):
            trace.created_at = datetime.fromisoformat(data['created_at'])
        if data.get('last_accessed'):
            trace.last_accessed = datetime.fromisoformat(data['last_accessed'])
        if data.get('next_review'):
            trace.next_review = datetime.fromisoformat(data['next_review'])
        
        return trace