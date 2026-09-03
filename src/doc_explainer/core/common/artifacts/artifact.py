from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ArtifactRef:
    id: str
    uri: str
    type: str  # e.g., 'pickle', 'json'
    metadata: Optional[dict] = None