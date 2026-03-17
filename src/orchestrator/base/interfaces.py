from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..models.requests import BaseRequest
from ..models.responses import BaseResponse


class Pipeline(ABC):
    """Base interface for pipelines"""
    
    @abstractmethod
    def process(self, request: BaseRequest) -> BaseResponse:
        """Process a request through the pipeline"""
        pass
    
    @abstractmethod
    def validate(self, request: BaseRequest) -> bool:
        """Validate request before processing"""
        pass


class Service(ABC):
    """Base interface for services"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the service"""
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """Shutdown the service"""
        pass


class ContextBuilder(ABC):
    """Interface for context building"""
    
    @abstractmethod
    def build_context(self, **kwargs) -> Any:
        """Build context from parameters"""
        pass


class DocumentProcessor(ABC):
    """Interface for document processing"""
    
    @abstractmethod
    def process_document(self, path: str) -> str:
        """Process and register a document"""
        pass
    
    @abstractmethod
    def get_document(self, doc_id: str) -> Any:
        """Get document by ID"""
        pass