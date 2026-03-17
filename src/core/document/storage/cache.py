from typing import Dict, Any, Optional


class DocumentCache:
    """In-memory cache for documents"""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        return self._cache.get(key)
    
    def set(self, key: str, value: Any):
        """Set item in cache"""
        self._cache[key] = value
    
    def has(self, key: str) -> bool:
        """Check if key exists in cache"""
        return key in self._cache
    
    def delete(self, key: str) -> bool:
        """Delete item from cache"""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self):
        """Clear cache"""
        self._cache.clear()