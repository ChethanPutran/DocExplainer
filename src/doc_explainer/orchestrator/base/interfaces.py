from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..models.requests import BaseRequest
from ..models.responses import BaseResponse


from typing import Protocol


class Executor(Protocol):

    def execute(self, task):
        ...


class ArtifactStore(Protocol):

    def save(self, artifact):
        ...

    def load(self, artifact_id):
        ...


class MetadataStore(Protocol):

    def create_run(self, run):
        ...

    def update_step(self, step):
        ...

    def get_run(self, run_id):
        ...


class Orchestrator(Protocol):

    def run(self, pipeline):
        ...
        
class  Pipeline(ABC):
    """Base interface for pipelines"""
    
    @abstractmethod
    def process(self, request: BaseRequest) -> BaseResponse:
        """Process a request through the pipeline"""
        pass

    @abstractmethod
    def _process(self, request: BaseRequest) -> BaseResponse:
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