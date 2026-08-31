from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

from doc_explainer.core.common.dataclasses import ExplanationStyle

from .base.exceptions import OrchestratorError
from .models.requests import (
    SummarizeRequest, ExplainRequest, AnswerRequest,
    RegisterDocumentRequest, GetContextRequest
)
from .models.responses import (
    BaseResponse, AnswerResponse,
    DocumentResponse
)
from .pipeline.document_pipeline import DocumentPipeline
from .pipeline.explanation_pipeline import ExplanationPipeline
from .pipeline.knowledge_pipeline import KnowledgePipeline
from .factories.pipeline_factory import PipelineFactory
from .config import OrchestratorConfig


class DocExplainerOrchestrator:
    """Main orchestrator for document explanation system"""
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        try:
        
            # Initialize factory
            self.factory = PipelineFactory(config=self.config.to_dict())
            
            # Initialize pipelines
            self.document_pipeline:DocumentPipeline = self.factory.create_document_pipeline()
            self.explanation_pipeline : ExplanationPipeline= self.factory.create_explanation_pipeline()
            self.knowledge_pipeline: KnowledgePipeline = self.factory.create_knowledge_pipeline()
            
            # Cache for document info
            self.document_info: Dict[str, Dict[str, Any]] = {}
        
        except Exception as e:
            print(f"Error initializing orchestrator: {e}")
            self.logger.error(f"Error initializing orchestrator: {e}")
            raise OrchestratorError
    
    def summarize(self, doc_id: str, selected_text: str,
                  section_id: int = 0,
                  user_id: Optional[str] = None) -> BaseResponse:
        """Summarize selected text"""
        user_id = user_id or self.config.default_user_id
        
        request = SummarizeRequest(
            user_id=user_id,
            doc_id=doc_id,
            selected_text=selected_text,
            section_id=section_id
        )
        
        return self.explanation_pipeline.process(request)
    
    def explain(self, doc_id: str, selected_text: str,
                section_id: int = 0,
                style: Optional[ExplanationStyle] = None,
                user_id: Optional[str] = None) -> BaseResponse:
        """Explain selected text"""
        user_id = user_id or self.config.default_user_id
        
        request = ExplainRequest(
            user_id=user_id,
            doc_id=doc_id,
            selected_text=selected_text,
            section_id=section_id,
            style=style
        )
        
        return self.explanation_pipeline.process(request)
    
    def answer(self, doc_id: str, question: str,
               section_id: int = 0,
               user_id: Optional[str] = None) -> BaseResponse:
        """Answer question about document"""
        user_id = user_id or self.config.default_user_id
        
        request = AnswerRequest(
            user_id=user_id,
            doc_id=doc_id,
            question=question,
            section_id=section_id
        )
        
        return self.explanation_pipeline.process(request)
    
    def register_document(self, path: str,
                          build_graph: bool = True,
                          user_id: Optional[str] = None) -> BaseResponse:
        """Register a document"""
        user_id = user_id or self.config.default_user_id
        
        request = RegisterDocumentRequest(
            user_id=user_id,
            path=path,
            build_graph=build_graph
        )
        
        response = self.document_pipeline.process(request)
        
        if response.success and type(response) is DocumentResponse and response.doc_id:
            self.document_info[response.doc_id] = {
                'path': path,
                'title': response.title,
                'registered_at': datetime.now().isoformat()
            }
        
        return response
    
    def get_context(self, doc_id: str, section_id: int = 0,
                   user_id: Optional[str] = None) -> BaseResponse:
        """Get context for document and section"""
        user_id = user_id or self.config.default_user_id
        
        request = GetContextRequest(
            user_id=user_id,
            doc_id=doc_id,
            section_id=section_id
        )
        
        return self.document_pipeline.process(request)
    
    def get_section_id_at_position(self, doc_id: str, page: int,
                                   position: int) -> int:
        """Get section ID at page and position"""
        document_service = self.factory.create_document_service()
        return document_service.get_section_id_at_position(doc_id, page, position)
    
    def get_document(self, doc_id: str) -> Optional[Any]:
        """Get document by ID"""
        document_service = self.factory.create_document_service()
        return document_service.get_document(doc_id)
    
    def analyze_prerequisites(self, concept_name: str) -> Dict[str, Any]:
        """Analyze prerequisites for a concept"""
        return self.knowledge_pipeline.analyze_prerequisites(concept_name)
    
    def generate_learning_path(self, target_concept: str,
                              max_depth: int = 3) -> List[Dict[str, Any]]:
        """Generate learning path for a concept"""
        return self.knowledge_pipeline.generate_learning_path(
            target_concept, max_depth
        )
    
    def recommend_concepts(self, concept_name: str,
                          limit: int = 5) -> Dict[str, Any]:
        """Recommend related concepts"""
        return self.knowledge_pipeline.recommend_concepts(concept_name, limit)
    
    def get_document_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get cached document info"""
        return self.document_info.get(doc_id)
    
    def list_documents(self) -> Dict[str, Dict[str, Any]]:
        """List all registered documents"""
        return self.document_info


# Example usage
if __name__ == "__main__":
    # Initialize orchestrator
    orchestrator = DocExplainerOrchestrator()
    
    # Register document
    doc_path = "data/report.pdf"
    response = orchestrator.register_document(doc_path)
    
    if response.success:
        if type(response) is DocumentResponse and response.doc_id:
            doc_id = response.doc_id
            print(f"Document registered with ID: {doc_id}")
            
            # Ask a question
            answer_response = orchestrator.answer(
                doc_id=doc_id,
                question="What is the main finding of this report?",
                section_id=0
            )
            if type(answer_response) is AnswerResponse:
                if answer_response.success:
                    print(f"Answer: {answer_response.answer}")
                else:
                    print("Failed to get answer")
            else:
                print("Unexpected response type for answer")
        else:
            print("Document registration response did not contain doc_id")