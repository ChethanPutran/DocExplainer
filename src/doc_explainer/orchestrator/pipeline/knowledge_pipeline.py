from typing import List, Optional, Dict, Any

from doc_explainer.orchestrator.models.requests import BaseRequest
from doc_explainer.orchestrator.models.responses import BaseResponse

from .base import BasePipeline
from ...core.knowledge.graph.state_manager import GraphStateManager
from ...core.knowledge.services.prerequisite_analyzer import PrerequisiteAnalyzer
from ...core.knowledge.services.learning_path import LearningPathGenerator
from ...core.knowledge.services.recommendation import RecommendationService
from ..progress import ProgressReporter


class KnowledgePipeline(BasePipeline):
    """Pipeline for knowledge graph operations."""

    def __init__(
        self,
        graph_state_manager: GraphStateManager,
        prerequisite_analyzer: Optional[PrerequisiteAnalyzer] = None,
        learning_path_generator: Optional[LearningPathGenerator] = None,
        recommendation_service: Optional[RecommendationService] = None,
        logger=None,
    ):
        super().__init__(logger)

        self.graph_state_manager = graph_state_manager
        self.prerequisite_analyzer = prerequisite_analyzer
        self.learning_path_generator = learning_path_generator
        self.recommendation_service = recommendation_service


    def _process(
        self,
        request: BaseRequest,
        progress_reporter: Optional[ProgressReporter] = None,
        job_id: Optional[str] = None,
    ) -> BaseResponse:
        """Process knowledge graph related requests."""
        if isinstance(request, BaseRequest):
            result = self.build_knowledge_graph(
                document_id=request.document_id,
                progress_reporter=progress_reporter,
                job_id=job_id,
            )

            return BaseResponse(
                success=True,
                message="Knowledge graph built successfully",
                data=result,
            )

        raise ValueError(
            f"Unsupported request type: {type(request)}"
        )

    def get_graph_state(
        self,
        user_id: str,
        section_id: str,
    ) -> Dict[str, Any]:
        """Get knowledge graph state up to section."""

        self.logger.info(
            f"Getting graph state for user {user_id} "
            f"up to section {section_id}"
        )

        graph = self.graph_state_manager.get_concept_graph_upto(
            section_id
        )

        return {
            "node_count": len(graph.graph.nodes),
            "edge_count": len(graph.graph.edges),
            "graph": graph,
        }

    def analyze_prerequisites(
        self,
        concept_name: str,
    ) -> Dict[str, Any]:
        """Analyze prerequisites for a concept."""

        if not self.prerequisite_analyzer:
            return {
                "error": "Prerequisite analyzer not available"
            }

        self.logger.info(
            f"Analyzing prerequisites for concept: {concept_name}"
        )

        return self.prerequisite_analyzer.analyze_prerequisites(
            concept_name
        )

    def generate_learning_path(
        self,
        target_concept: str,
        max_depth: int = 3,
    ) -> List[Dict[str, Any]]:
        """Generate learning path for a concept."""

        if not self.learning_path_generator:
            return [
                {
                    "error": "Learning path generator not available"
                }
            ]

        self.logger.info(
            f"Generating learning path for: {target_concept}"
        )

        return self.learning_path_generator.generate_path(
            target_concept,
            max_depth,
        )

    def recommend_concepts(
        self,
        concept_name: str,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Recommend related concepts."""

        if not self.recommendation_service:
            return {
                "error": "Recommendation service not available"
            }

        self.logger.info(
            f"Recommending concepts related to: {concept_name}"
        )

        recommendations = (
            self.recommendation_service.recommend_related_concepts(
                concept_name,
                limit,
            )
        )

        return {
            "concept": concept_name,
            "recommendations": recommendations,
        }

    def build_knowledge_graph(
        self,
        document_id: str,
        progress_reporter: Optional[ProgressReporter] = None,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build knowledge graph from a document."""

        self.logger.info(
            f"Building knowledge graph for document: {document_id}"
        )

        graph = self.graph_state_manager.build_graph_from_document(
            document_id
        )

        self.logger.info(
            f"Knowledge graph built with "
            f"{len(graph.graph.nodes)} nodes and "
            f"{len(graph.graph.edges)} edges"
        )

        return {
            "node_count": len(graph.graph.nodes),
            "edge_count": len(graph.graph.edges),
            "graph": graph,
        }