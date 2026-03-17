from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


class Serializable(ABC):
    """Base interface for serializable objects"""
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Serializable':
        """Create from dictionary"""
        pass


class Identifiable(ABC):
    """Base interface for identifiable objects"""
    
    @property
    @abstractmethod
    def id(self) -> Any:
        """Get object ID"""
        pass


class Positionable(ABC):
    """Base interface for objects with position"""
    
    @property
    @abstractmethod
    def start(self) -> int:
        """Get start position"""
        pass
    
    @property
    @abstractmethod
    def end(self) -> int:
        """Get end position"""
        pass
    
    @property
    @abstractmethod
    def page(self) -> int:
        """Get page number"""
        pass
    
    @property
    @abstractmethod
    def bbox(self) -> tuple:
        """Get bounding box"""
        pass