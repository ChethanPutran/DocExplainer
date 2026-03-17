from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..base.interfaces import MemoryStorage as MemoryStorageInterface


class MemoryStorage(ABC, MemoryStorageInterface):
    """Base class for memory storage"""
    
    def __init__(self):
        self._storage: Dict[str, Any] = {}
    
    def store(self, key: str, value: Any) -> bool:
        """Store a value by key"""
        try:
            self._storage[key] = value
            return True
        except Exception as e:
            print(f"Error storing {key}: {e}")
            return False
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value by key"""
        return self._storage.get(key)
    
    def delete(self, key: str) -> bool:
        """Delete a value by key"""
        if key in self._storage:
            del self._storage[key]
            return True
        return False
    
    def clear(self):
        """Clear all storage"""
        self._storage.clear()
    
    def has_key(self, key: str) -> bool:
        """Check if key exists"""
        return key in self._storage
    
    def get_all_keys(self) -> list:
        """Get all keys"""
        return list(self._storage.keys())