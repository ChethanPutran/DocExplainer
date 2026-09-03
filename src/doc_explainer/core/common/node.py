from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set
import uuid

@dataclass
class Node:
    """A node in the DAG representing a step invocation."""
    step_name: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dependencies: Set[str] = field(default_factory=set)
    output_artifact_id: Optional[str] = None
    step: Optional['Step'] = None  # reference to the Step object
    annotations: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for v in self.inputs.values():
            if isinstance(v, Node):
                self.dependencies.add(v.id)
        for v in self.kwargs.values():
            if isinstance(v, Node):
                self.dependencies.add(v.id)