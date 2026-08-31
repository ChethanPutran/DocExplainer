import pickle
import json
import os
from typing import Any


class Serializer:
    def serialize(self, value: Any, path: str):
        raise NotImplementedError

    def deserialize(self, path: str) -> Any:
        raise NotImplementedError


class PickleSerializer(Serializer):
    def serialize(self, value: Any, path: str):
        with open(path, 'wb') as f:
            pickle.dump(value, f)

    def deserialize(self, path: str) -> Any:
        with open(path, 'rb') as f:
            return pickle.load(f)


class JSONSerializer(Serializer):
    def serialize(self, value: Any, path: str):
        with open(path, 'w') as f:
            json.dump(value, f)

    def deserialize(self, path: str) -> Any:
        with open(path, 'r') as f:
            return json.load(f)


# Auto-detect based on type (simplistic)
def get_serializer(value: Any) -> Serializer:
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        return JSONSerializer()
    else:
        return PickleSerializer()