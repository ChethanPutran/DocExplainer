"""Curriculum generation models and data structures."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
import json


class CurriculumStrategy(str, Enum):
    """Curriculum sequencing strategies."""
    
    BREADTH_FIRST = "breadth_first"
    """Cover broad overview first, then deepen understanding."""
    
    DEPTH_FIRST = "depth_first"
    """Deep dive into one area before moving to another."""
    
    ADAPTIVE = "adaptive"
    """Dynamic reordering based on user performance and learning."""
    
    SPACED_REPETITION = "spaced_repetition"
    """Strategically timed reviews to optimize long-term retention."""
    
    MASTERY_BASED = "mastery_based"
    """Focus on weak areas and reinforce strong areas."""


@dataclass
class CurriculumNode:
    """Represents a concept node in the curriculum."""
    
    concept_id: str
    """Unique identifier for the concept."""
    
    concept_name: str
    """Human-readable name of the concept."""
    
    dependencies: Set[str] = field(default_factory=set)
    """IDs of concepts that must be learned first."""
    
    estimated_time_minutes: float = 10.0
    """Estimated time to learn this concept."""
    
    priority: float = 1.0
    """Priority score (0-1, where 1 is highest priority)."""
    
    difficulty: float = 0.5
    """Difficulty score (0-1, where 1 is most difficult)."""
    
    mastery_level: float = 0.0
    """Current user's mastery level (0-1)."""
    
    dependency_depth: int = 0
    """Number of prerequisite levels."""
    
    prerequisite_ids: List[str] = field(default_factory=list)
    """Direct prerequisite concept IDs."""
    
    transitive_dependencies: Set[str] = field(default_factory=set)
    """All concepts (direct and indirect) that must be learned first."""
    
    attributes: Dict[str, any] = field(default_factory=dict)
    """Additional concept attributes."""
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "concept_id": self.concept_id,
            "concept_name": self.concept_name,
            "dependencies": list(self.dependencies),
            "estimated_time_minutes": self.estimated_time_minutes,
            "priority": self.priority,
            "difficulty": self.difficulty,
            "mastery_level": self.mastery_level,
            "dependency_depth": self.dependency_depth,
            "prerequisite_ids": self.prerequisite_ids,
            "transitive_dependencies": list(self.transitive_dependencies),
            "attributes": self.attributes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "CurriculumNode":
        """Deserialize from dictionary."""
        return cls(
            concept_id=data["concept_id"],
            concept_name=data["concept_name"],
            dependencies=set(data.get("dependencies", [])),
            estimated_time_minutes=data.get("estimated_time_minutes", 10.0),
            priority=data.get("priority", 1.0),
            difficulty=data.get("difficulty", 0.5),
            mastery_level=data.get("mastery_level", 0.0),
            dependency_depth=data.get("dependency_depth", 0),
            prerequisite_ids=data.get("prerequisite_ids", []),
            transitive_dependencies=set(data.get("transitive_dependencies", [])),
            attributes=data.get("attributes", {}),
        )


class PathProgressState(str, Enum):
    """States for learning path progress."""
    
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PAUSED = "paused"
    ABANDONED = "abandoned"


@dataclass
class LearningPath:
    """Represents a personalized learning path."""
    
    path_id: str
    """Unique identifier for this learning path."""
    
    user_id: str
    """User ID this path is for."""
    
    concepts: List[CurriculumNode]
    """Ordered list of concepts to learn."""
    
    strategy: CurriculumStrategy = CurriculumStrategy.ADAPTIVE
    """Strategy used to generate this path."""
    
    status: PathProgressState = PathProgressState.NOT_STARTED
    """Current progress state."""
    
    progress: float = 0.0
    """Overall progress (0-1)."""
    
    current_index: int = 0
    """Index of current concept being learned."""
    
    estimated_total_time_minutes: float = 0.0
    """Total estimated time for complete path."""
    
    actual_time_spent_minutes: float = 0.0
    """Actual time spent so far."""
    
    completed_concepts: Set[str] = field(default_factory=set)
    """IDs of completed concepts."""
    
    created_at: datetime = field(default_factory=datetime.now)
    """When this path was created."""
    
    started_at: Optional[datetime] = None
    """When learning started."""
    
    completed_at: Optional[datetime] = None
    """When learning was completed."""
    
    last_updated_at: datetime = field(default_factory=datetime.now)
    """Last update timestamp."""
    
    metadata: Dict[str, any] = field(default_factory=dict)
    """Additional metadata."""
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "path_id": self.path_id,
            "user_id": self.user_id,
            "concepts": [c.to_dict() for c in self.concepts],
            "strategy": self.strategy.value,
            "status": self.status.value,
            "progress": self.progress,
            "current_index": self.current_index,
            "estimated_total_time_minutes": self.estimated_total_time_minutes,
            "actual_time_spent_minutes": self.actual_time_spent_minutes,
            "completed_concepts": list(self.completed_concepts),
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "last_updated_at": self.last_updated_at.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "LearningPath":
        """Deserialize from dictionary."""
        return cls(
            path_id=data["path_id"],
            user_id=data["user_id"],
            concepts=[CurriculumNode.from_dict(c) for c in data.get("concepts", [])],
            strategy=CurriculumStrategy(data.get("strategy", CurriculumStrategy.ADAPTIVE.value)),
            status=PathProgressState(data.get("status", PathProgressState.NOT_STARTED.value)),
            progress=data.get("progress", 0.0),
            current_index=data.get("current_index", 0),
            estimated_total_time_minutes=data.get("estimated_total_time_minutes", 0.0),
            actual_time_spent_minutes=data.get("actual_time_spent_minutes", 0.0),
            completed_concepts=set(data.get("completed_concepts", [])),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            last_updated_at=datetime.fromisoformat(data["last_updated_at"]) if "last_updated_at" in data else datetime.now(),
            metadata=data.get("metadata", {}),
        )
    
    def get_current_concept(self) -> Optional[CurriculumNode]:
        """Get the current concept being learned."""
        if 0 <= self.current_index < len(self.concepts):
            return self.concepts[self.current_index]
        return None
    
    def get_next_concept(self) -> Optional[CurriculumNode]:
        """Get the next concept to learn."""
        next_index = self.current_index + 1
        if 0 <= next_index < len(self.concepts):
            return self.concepts[next_index]
        return None
    
    def estimate_time_to_completion(self) -> float:
        """Estimate remaining time to complete path (minutes)."""
        remaining_time = self.estimated_total_time_minutes - self.actual_time_spent_minutes
        return max(0, remaining_time)
