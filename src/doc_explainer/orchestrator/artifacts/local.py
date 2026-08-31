import os
import uuid
from pathlib import Path
from typing import Any
from .store import ArtifactStore
from .artifact import ArtifactRef
from .serializers import get_serializer


class LocalArtifactStore(ArtifactStore):
    def __init__(self, base_dir: str = "./artifacts"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, value: Any, **metadata) -> ArtifactRef:
        artifact_id = str(uuid.uuid4())
        # Determine type from metadata or auto
        step_name = metadata.get('step_name', 'unknown')
        run_id = metadata.get('run_id', 'unknown')
        # Create directory structure: base_dir/run_id/step_name/
        dir_path = self.base_dir / run_id / step_name
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / f"{artifact_id}.artifact"
        serializer = get_serializer(value)
        serializer.serialize(value, file_path)
        # Store type
        type_name = serializer.__class__.__name__.replace('Serializer', '').lower()
        return ArtifactRef(
            id=artifact_id,
            uri=str(file_path),
            type=type_name,
            metadata=metadata
        )

    def load(self, ref: ArtifactRef) -> Any:
        path = Path(ref.uri)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        # Determine serializer from type
        from .serializers import PickleSerializer, JSONSerializer
        if ref.type == 'json':
            serializer = JSONSerializer()
        else:
            serializer = PickleSerializer()
        return serializer.deserialize(path)