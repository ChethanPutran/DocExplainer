import json
from typing import Dict, Any
from datetime import datetime
from ..models.memory_trace import ConceptMemoryTrace
from ..models.context import SessionContext


class MemorySerializer:
    """Serializer for memory objects"""
    
    @staticmethod
    def serialize_concept_trace(trace: ConceptMemoryTrace) -> Dict:
        """Serialize concept memory trace"""
        return trace.to_dict()
    
    @staticmethod
    def deserialize_concept_trace(data: Dict) -> ConceptMemoryTrace:
        """Deserialize concept memory trace"""
        return ConceptMemoryTrace.from_dict(data)
    
    @staticmethod
    def serialize_session_context(context: SessionContext) -> Dict:
        """Serialize session context"""
        return context.to_dict()
    
    @staticmethod
    def deserialize_session_context(data: Dict) -> SessionContext:
        """Deserialize session context"""
        return SessionContext.from_dict(data)
    
    @staticmethod
    def serialize_for_json(obj: Any) -> Any:
        """Serialize object for JSON"""
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, (list, tuple)):
            return [MemorySerializer.serialize_for_json(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: MemorySerializer.serialize_for_json(value) for key, value in obj.items()}
        else:
            return obj
    
    @staticmethod
    def deserialize_from_json(data: Any) -> Any:
        """Deserialize object from JSON"""
        # This would need context about what type to deserialize to
        return data