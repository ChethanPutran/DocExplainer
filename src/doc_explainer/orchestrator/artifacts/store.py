from typing import Any
from .artifact import ArtifactRef
from abc import ABC, abstractmethod


class ArtifactStore(ABC):
    @abstractmethod
    def save(self, value: Any, **metadata) -> ArtifactRef:
        pass

    @abstractmethod
    def load(self, ref: ArtifactRef) -> Any:
        pass