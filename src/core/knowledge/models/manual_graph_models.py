"""Models for manual graph editing operations"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class RelationshipType(str, Enum):
    """Supported relationship types"""
    PREREQUISITE = "prerequisite"
    SIMILAR = "similar"
    RELATED = "related"
    HIERARCHICAL = "hierarchical"
    DEPENDS_ON = "depends_on"
    PART_OF = "part_of"


class OperationType(str, Enum):
    """Types of concept operations"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    ADD_ALIAS = "add_alias"
    REMOVE_ALIAS = "remove_alias"
    ADD_DEFINITION = "add_definition"
    REMOVE_DEFINITION = "remove_definition"


@dataclass
class ConceptEdit:
    """Represents an edit operation on a concept"""
    operation: OperationType
    concept_id: str
    concept_name: Optional[str] = None
    changes: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    previous_state: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "operation": self.operation.value,
            "concept_id": self.concept_id,
            "concept_name": self.concept_name,
            "changes": self.changes,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "previous_state": self.previous_state
        }


@dataclass
class RelationshipEdit:
    """Represents an edit operation on a relationship"""
    operation: OperationType
    from_concept_id: str
    to_concept_id: str
    from_concept_name: Optional[str] = None
    to_concept_name: Optional[str] = None
    relationship_type: RelationshipType = RelationshipType.RELATED
    strength: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "operation": self.operation.value,
            "from_concept_id": self.from_concept_id,
            "to_concept_id": self.to_concept_id,
            "from_concept_name": self.from_concept_name,
            "to_concept_name": self.to_concept_name,
            "relationship_type": self.relationship_type.value,
            "strength": self.strength,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


@dataclass
class GraphSnapshot:
    """Represents a snapshot of the graph at a point in time"""
    timestamp: Optional[datetime] = None
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    checksum: Optional[str] = None
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "nodes": self.nodes,
            "edges": self.edges,
            "checksum": self.checksum,
            "version": self.version,
            "metadata": self.metadata
        }


@dataclass
class ValidationError:
    """Represents a validation error in the graph"""
    error_type: str
    message: str
    affected_concepts: List[str] = field(default_factory=list)
    affected_relationships: List[tuple] = field(default_factory=list)
    severity: str = "error"  # "error", "warning", "info"
    suggestion: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "error_type": self.error_type,
            "message": self.message,
            "affected_concepts": self.affected_concepts,
            "affected_relationships": self.affected_relationships,
            "severity": self.severity,
            "suggestion": self.suggestion
        }


@dataclass
class GraphBackup:
    """Represents a backup of the graph"""
    backup_id: str
    timestamp: Optional[datetime] = None
    graph_data: Optional[Dict[str, Any]] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "backup_id": self.backup_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "graph_data": self.graph_data,
            "description": self.description,
            "tags": self.tags
        }
