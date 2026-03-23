from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Optional, Any

T = TypeVar('T')

class BaseRepository(ABC, Generic[T]):
    """Base repository interface"""
    
    @abstractmethod
    def get(self, id: Any) -> Optional[T]:
        """Get entity by ID"""
        pass
    
    @abstractmethod
    def save(self, entity: T) -> T:
        """Save entity"""
        pass
    
    @abstractmethod
    def delete(self, id: Any) -> bool:
        """Delete entity by ID"""
        pass
    
    @abstractmethod
    def find_all(self) -> List[T]:
        """Find all entities"""
        pass

