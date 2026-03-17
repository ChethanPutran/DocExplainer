from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..models.memory_trace import MemoryTrace
from ..models.context import SessionContext, Context


class MemoryStorage(ABC):
    """Base interface for memory storage"""
    
    @abstractmethod
    def store(self, key: str, value: Any) -> bool:
        """Store a value by key"""
        pass
    
    @abstractmethod
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value by key"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a value by key"""
        pass
    
    @abstractmethod
    def clear(self):
        """Clear all storage"""
        pass


class LongTermMemoryInterface(ABC):
    """Interface for long-term memory operations"""
    
    @abstractmethod
    def store_user_profile(self, user_knowledge_state) -> bool:
        """Store user profile snapshot"""
        pass
    
    @abstractmethod
    def retrieve_user_profile(self) -> Optional[Dict]:
        """Retrieve user profile"""
        pass
    
    @abstractmethod
    def store_concept_memory(self, concept: str, memory_trace: Dict) -> bool:
        """Store concept memory trace"""
        pass
    
    @abstractmethod
    def retrieve_concept_memory(self, concept: str) -> Optional[Dict]:
        """Retrieve concept memory trace"""
        pass
    
    @abstractmethod
    def store_interaction(self, interaction_type: str, data: Dict) -> bool:
        """Store user interaction"""
        pass


class SessionMemoryInterface(ABC):
    """Interface for session memory operations"""
    
    @abstractmethod
    def get_session_context(self) -> SessionContext:
        """Get current session context"""
        pass
    
    @abstractmethod
    def update_session_context(self, **kwargs) -> bool:
        """Update session context"""
        pass
    
    @abstractmethod
    def add_interaction(self, interaction: Dict) -> bool:
        """Add interaction to session"""
        pass


class ChainInterface(ABC):
    """Interface for chain structures"""
    
    @abstractmethod
    def add_node(self, node_id: Any, data: Any) -> bool:
        """Add a node to the chain"""
        pass
    
    @abstractmethod
    def get_node(self, node_id: Any) -> Optional[Any]:
        """Get a node from the chain"""
        pass
    
    @abstractmethod
    def traverse(self, start_node: Any) -> list:
        """Traverse the chain from start node"""
        pass