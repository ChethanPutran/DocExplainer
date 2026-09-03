from __future__ import annotations

from typing import Optional, Dict, Any
import logging

from doc_explainer.orchestrator.config import OrchestratorConfig

from ..pipeline.document_pipeline import DocumentPipeline
from ..pipeline.explanation_pipeline import ExplanationPipeline
from ..pipeline.knowledge_pipeline import KnowledgePipeline

from ..services.document_service import DocumentService
from ..services.user_service import UserService
from ..services.context_service import ContextService

from ...core.document import DocumentManager
from ...core.explanation_engine import AdaptiveExplainer
from ...core.user import UserManager
from ...core.memory import MemoryManager, SessionManager
from ...core.knowledge import GraphStateManager

from ...core.agent.llm.factories.llm_factory import LLMFactory

from ...models.text import TextModels

from ...store.knowledge import (
    KnowledgeStore,
    KnowledgeRepository,
)

from ...store.user import UserRepository


logger = logging.getLogger(__name__)

DEFAULT_USER_ID = "user_123"


class PipelineFactory:
    """
    Factory for creating pipelines and their dependencies.

    Configuration is provided through OrchestratorConfig:

        self.config.llm
        self.config.backend

    All dependencies created by this factory are cached so that
    shared services/stores use the same instances.
    """

    def __init__(
        self,
        config: OrchestratorConfig,
    ) -> None:

        self.config = config

        self._instances: Dict[str, Any] = {}

    # ================================================================
    # DOCUMENT STORAGE
    # ================================================================

    def create_document_repository(self) -> Any:
        """Create document repository."""

        if "document_repository" not in self._instances:

            from ...store.document import DocumentRepository

            persist_dir = (
                self.config.backend.persist_directory
            )

            logger.debug(
                "Creating DocumentRepository: %s",
                persist_dir,
            )

            self._instances[
                "document_repository"
            ] = DocumentRepository(
                persist_dir
            )

        return self._instances[
            "document_repository"
        ]

    # ================================================================
    # DOCUMENT ENGINE
    # ================================================================

    def create_document_engine(self) -> Any:
        """Create document engine."""

        instance_name = "document_engine"

        if instance_name not in self._instances:

            from ...core.document import DocumentEngine

            from ...core.document.processor.hierarchy import (
                HierarchicalProcessor,
            )

            llm_wrapper = self.create_llm(
                "summary_generator"
            )

            document_processor = (
                HierarchicalProcessor(
                    llm_wrapper
                )
            )

            document_parser = (
                self.create_document_parser()
            )

            checkpoint_store = (
                self.create_checkpoint_store()
            )

            self._instances[
                instance_name
            ] = DocumentEngine(
                document_parser,
                document_processor,
                self.create_vector_store(),
                self.create_graph_store(),
                checkpoint_store,
            )

        return self._instances[
            instance_name
        ]

    # ================================================================
    # VECTOR STORE
    # ================================================================

    def create_vector_store(self) -> Any:
        """Create vector store."""

        if "vector_store" not in self._instances:

            from ...store.vector import (
                ChromaVectorStore,
            )

            persist_dir = (
                self.config.backend
                .vector_store
                .persist_directory
            )

            embedding_model = (
                self.create_text_models()
                .get_embedding_model()
            )

            logger.debug(
                "Creating ChromaVectorStore: %s",
                persist_dir,
            )

            self._instances[
                "vector_store"
            ] = ChromaVectorStore(
                persist_directory=persist_dir,
                embedding_function=embedding_model,
            )

        return self._instances[
            "vector_store"
        ]

    # ================================================================
    # NEO4J GRAPH STORE
    # ================================================================

    def create_graph_store(self) -> Any:
        """Create Neo4j graph store."""

        if "graph_store" in self._instances:
            return self._instances[
                "graph_store"
            ]

        # ------------------------------------------------------------
        # Knowledge graph disabled
        # ------------------------------------------------------------

        if not (
            self.config.backend
            .knowledge_graph
            .enabled
        ):

            logger.info(
                "Knowledge graph is disabled."
            )

            self._instances[
                "graph_store"
            ] = None

            return None

        # ------------------------------------------------------------
        # Neo4j
        # ------------------------------------------------------------

        from ...store.graph import (
            Neo4jGraphStore,
        )

        neo4j = (
            self.config.backend.neo4j
        )

        logger.info(
            "Creating Neo4jGraphStore: "
            "uri=%s, user=%s",
            neo4j.uri,
            neo4j.user,
        )

        self._instances[
            "graph_store"
        ] = Neo4jGraphStore(
            uri=neo4j.uri,
            user=neo4j.user,
            password=neo4j.password,
        )

        return self._instances[
            "graph_store"
        ]

    # ================================================================
    # DOCUMENT PARSER
    # ================================================================

    def create_document_parser(self) -> Any:
        """Create document parser."""

        if "document_parser" not in self._instances:

            from ...core.document.parser.pdf import (
                PDFParser,
            )

            self._instances[
                "document_parser"
            ] = PDFParser()

        return self._instances[
            "document_parser"
        ]

    # ================================================================
    # DOCUMENT MANAGER
    # ================================================================

    def create_document_manager(
        self,
    ) -> DocumentManager:
        """Create document manager."""

        if "document_manager" not in self._instances:

            self._instances[
                "document_manager"
            ] = DocumentManager(
                repository=(
                    self.create_document_repository()
                ),
                document_engine=(
                    self.create_document_engine()
                ),
                document_tree_builder=(
                    self.create_document_tree_builder()
                ),
            )

        return self._instances[
            "document_manager"
        ]

    # ================================================================
    # DOCUMENT TREE BUILDER
    # ================================================================

    def create_document_tree_builder(
        self,
    ) -> Any:
        """Create document tree builder."""

        if (
            "document_tree_builder"
            not in self._instances
        ):

            from ...core.document.builder.document_tree_builder import (
                DocumentTreeBuilder,
            )

            self._instances[
                "document_tree_builder"
            ] = DocumentTreeBuilder(
                self.create_graph_store()
            )

        return self._instances[
            "document_tree_builder"
        ]

    # ================================================================
    # TEXT MODELS
    # ================================================================

    def create_text_models(
        self,
    ) -> TextModels:
        """Create text models."""

        if "text_models" not in self._instances:

            self._instances[
                "text_models"
            ] = TextModels()

        return self._instances[
            "text_models"
        ]

    # ================================================================
    # LLM
    # ================================================================

    def create_llm(
        self,
        instance_name: str,
        provider: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Any:
        """
        Create an LLM wrapper.

        LLM configuration comes from:

            self.config.llm
        """

        if instance_name in self._instances:
            return self._instances[
                instance_name
            ]

        llm_config = self.config.llm

        # ------------------------------------------------------------
        # LLM kwargs
        # ------------------------------------------------------------

        llm_kwargs = dict(
            llm_config.extra
            if hasattr(llm_config, "kwargs")
            else {}
        )

        # ------------------------------------------------------------
        # Model
        # ------------------------------------------------------------

        model_name = llm_config.model

        if model_name:
            llm_kwargs.setdefault(
                "model_name",
                model_name,
            )

        # ------------------------------------------------------------
        # Rate limiting
        # ------------------------------------------------------------

        llm_kwargs.setdefault(
            "requests_per_minute",
            llm_config.requests_per_minute,
        )

        llm_kwargs.setdefault(
            "min_request_interval_seconds",
            llm_config.min_request_interval_seconds,
        )

        llm_kwargs.setdefault(
            "rate_limit_retries",
            llm_config.rate_limit_retries,
        )

        # ------------------------------------------------------------
        # General LLM kwargs
        # ------------------------------------------------------------

        llm_kwargs.setdefault(
            "max_tokens",
            llm_config.max_tokens,
        )

        llm_kwargs.setdefault(
            "timeout",
            llm_config.timeout,
        )

        llm_kwargs.setdefault(
            "mock",
            llm_config.mock,
        )

        # ------------------------------------------------------------
        # Create
        # ------------------------------------------------------------

        logger.debug(
            "Creating LLM: instance=%s, provider=%s, model=%s",
            instance_name,
            provider or llm_config.provider,
            model_name,
        )

        self._instances[
            instance_name
        ] = LLMFactory.create(
            provider=(
                provider
                or llm_config.provider
            ),
            instance_name=instance_name,
            temperature=(
                temperature
                if temperature is not None
                else llm_config.temperature
            ),
            **llm_kwargs,
        )

        return self._instances[
            instance_name
        ]

    # ================================================================
    # CHECKPOINT STORE
    # ================================================================

    def create_checkpoint_store(
        self,
    ) -> Any:
        """Create checkpoint store."""

        if "checkpoint_store" not in self._instances:

            from ...store.checkpoint.sqlite import (
                SQLiteCheckpointStore,
            )

            persist_dir = (
                self.config.backend
                .persist_directory
            )

            checkpoint_path = (
                persist_dir
                / "checkpoints.db"
            )

            self._instances[
                "checkpoint_store"
            ] = SQLiteCheckpointStore(
                str(checkpoint_path)
            )

        return self._instances[
            "checkpoint_store"
        ]

    # ================================================================
    # KNOWLEDGE STORE
    # ================================================================

    def create_knowledge_store(
        self,
    ) -> KnowledgeStore:
        """Create knowledge store."""

        if "knowledge_store" not in self._instances:

            persist_dir = (
                self.config.backend
                .persist_directory
            )

            storage_path = (
                persist_dir
                / "knowledge_graph.gpickle"
            )

            self._instances[
                "knowledge_store"
            ] = KnowledgeStore(
                storage_path=str(
                    storage_path
                )
            )

        return self._instances[
            "knowledge_store"
        ]

    # ================================================================
    # USER STORE
    # ================================================================

    def create_user_store(
        self,
    ) -> UserRepository:
        """Create user store."""

        if "user_store" not in self._instances:

            self._instances[
                "user_store"
            ] = UserRepository()

        return self._instances[
            "user_store"
        ]

    # ================================================================
    # USER MANAGER
    # ================================================================

    def create_user_manager(
        self,
        user_id: str = DEFAULT_USER_ID,
    ) -> UserManager:
        """Create user manager."""

        user_store = (
            self.create_user_store()
        )

        return UserManager(
            user_id,
            user_store,
        )

    # ================================================================
    # SESSION MANAGER
    # ================================================================

    def create_session_manager(
        self,
    ) -> SessionManager:
        """Create session manager."""

        if "session_manager" not in self._instances:

            self._instances[
                "session_manager"
            ] = SessionManager()

        return self._instances[
            "session_manager"
        ]

    # ================================================================
    # MEMORY MANAGER
    # ================================================================

    def create_memory_manager(
        self,
    ) -> MemoryManager:
        """Create memory manager."""

        if "memory_manager" not in self._instances:

            from ...core.memory.managers.memory_manager import (
                MemoryManager,
            )

            from ...core.memory.storage.long_term_memory import (
                LongTermMemory,
            )

            persist_dir = (
                self.config.backend
                .persist_directory
            )

            memory_path = (
                persist_dir
                / "memory"
                / "user_memory.json"
            )

            memory_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            long_term_memory = (
                LongTermMemory(
                    file_path=str(
                        memory_path
                    )
                )
            )

            self._instances[
                "memory_manager"
            ] = MemoryManager(
                long_term_memory
            )

        return self._instances[
            "memory_manager"
        ]

    # ================================================================
    # CONCEPT CANONICALIZER
    # ================================================================

    def create_concept_canonicalizer(
        self,
    ) -> Any:
        """Create concept canonicalizer."""

        if (
            "concept_canonicalizer"
            not in self._instances
        ):

            from ...core.knowledge.extraction.canonicalization.pipeline import (
                ConceptCanonicalizer,
            )

            from ...core.knowledge.extraction.canonicalization.normalizer import (
                TextNormalizer,
            )

            from ...core.knowledge.extraction.canonicalization.clusterer import (
                ConceptClusterer,
            )

            from ...core.knowledge.extraction.canonicalization.llm_canonicalizer import (
                LLMCanonicalizer,
            )

            llm = self.create_llm(
                "concept_canonicalizer"
            )

            text_models = (
                self.create_text_models()
            )

            normalizer = (
                TextNormalizer()
            )

            clusterer = (
                ConceptClusterer(
                    text_models
                    .get_embedding_model()
                )
            )

            llm_canonicalizer = (
                LLMCanonicalizer(llm)
            )

            self._instances[
                "concept_canonicalizer"
            ] = ConceptCanonicalizer(
                normalizer,
                clusterer,
                llm_canonicalizer,
            )

        return self._instances[
            "concept_canonicalizer"
        ]

    # ================================================================
    # CONCEPT EXTRACTOR
    # ================================================================

    def create_concept_extractor(
        self,
    ) -> Any:
        """Create concept extractor."""

        if (
            "concept_extractor"
            not in self._instances
        ):

            from ...core.knowledge.extraction.extractor import (
                ConceptExtractor,
            )

            from ...core.knowledge.extraction.scoring.composite_scorer import (
                CompositeScoringStrategy,
            )

            from ...core.knowledge.extraction.scoring.frequency_scorer import (
                FrequencyScoringStrategy,
            )

            from ...core.knowledge.extraction.scoring.position_scorer import (
                PositionScoringStrategy,
            )

            from ...core.knowledge.extraction.scoring.definition_scorer import (
                DefinitionBonusStrategy,
            )

            from ...core.knowledge.extraction.scoring.length_scorer import (
                LengthScoringStrategy,
            )

            from ...core.knowledge.extraction.filters.subset_pruner import (
                SubsetPrunerStrategy,
            )

            text_models = (
                self.create_text_models()
            )

            llm = self.create_llm(
                "concept_extractor"
            )

            canonicalizer = (
                self.create_concept_canonicalizer()
            )

            scoring_strategy = (
                CompositeScoringStrategy(
                    [
                        FrequencyScoringStrategy(),
                        PositionScoringStrategy(),
                        LengthScoringStrategy(),
                        DefinitionBonusStrategy(),
                    ]
                )
            )

            filter_strategy = (
                SubsetPrunerStrategy()
            )

            self._instances[
                "concept_extractor"
            ] = ConceptExtractor(
                text_model=text_models,
                llm_wrapper=llm,
                canonicalizer=canonicalizer,
                scoring_strategy=scoring_strategy,
                filter_strategy=filter_strategy,
                concepts_per_para=10,
            )

        return self._instances[
            "concept_extractor"
        ]

    # ================================================================
    # RELATIONSHIP EXTRACTOR
    # ================================================================

    def create_relationship_extractor(
        self,
    ) -> Any:
        """Create relationship extractor."""

        if (
            "relationship_extractor"
            not in self._instances
        ):

            from ...core.knowledge.extraction.strategies.relationship.llm_strategy import (
                LLMRelationshipExtractor,
            )

            from ...core.knowledge.extraction.strategies.relationship.statistical_strategy import (
                StatisticalRelationshipExtractor,
            )

            llm = self.create_llm(
                "relationship_extractor"
            )

            self._instances[
                "llm_relationship_extractor"
            ] = LLMRelationshipExtractor(
                llm
            )

            self._instances[
                "statistical_relationship_extractor"
            ] = StatisticalRelationshipExtractor()

            self._instances[
                "relationship_extractor"
            ] = self._instances[
                "llm_relationship_extractor"
            ]

        return self._instances[
            "relationship_extractor"
        ]

    # ================================================================
    # GRAPH STATE MANAGER
    # ================================================================

    def create_graph_state_manager(
        self,
    ) -> GraphStateManager:
        """Create graph state manager."""

        if (
            "graph_state_manager"
            not in self._instances
        ):

            from ...core.knowledge.graph import (
                ConceptGraphBuilder,
                DocumentChain,
                GraphUpdater,
                GraphStateManager,
            )

            from ...store.document import (
                DocumentCache,
            )

            concept_extractor = (
                self.create_concept_extractor()
            )

            relationship_extractor = (
                self.create_relationship_extractor()
            )

            llm_relationship = (
                self._instances[
                    "llm_relationship_extractor"
                ]
            )

            statistical_relationship = (
                self._instances[
                    "statistical_relationship_extractor"
                ]
            )

            document_cache = (
                DocumentCache()
            )

            user_manager = (
                self.create_user_manager()
            )

            knowledge_store = (
                self.create_knowledge_store()
            )

            document_repository = (
                self.create_document_repository()
            )

            concept_graph_builder = (
                ConceptGraphBuilder(
                    concept_extractor=concept_extractor,
                    llm_relationship_extractor=(
                        llm_relationship
                    ),
                    statistical_relationship_extractor=(
                        statistical_relationship
                    ),
                    document_cacher=document_cache,
                )
            )

            document_chain = (
                DocumentChain()
            )

            graph_updater = (
                GraphUpdater(
                    user_manager,
                    knowledge_store,
                )
            )

            knowledge_repository = (
                KnowledgeRepository(
                    document_chain,
                    knowledge_store.graph,
                )
            )

            self._instances[
                "graph_state_manager"
            ] = GraphStateManager(
                user_manager=user_manager,
                concept_graph_builder=(
                    concept_graph_builder
                ),
                document_chain=document_chain,
                graph_updater=graph_updater,
                repository=knowledge_repository,
                knowledge_store=knowledge_store,
                document_repository=(
                    document_repository
                ),
            )

        return self._instances[
            "graph_state_manager"
        ]

    # ================================================================
    # EXPLAINER
    # ================================================================

    def create_explainer(
        self,
    ) -> AdaptiveExplainer:
        """Create adaptive explainer."""

        if "explainer" not in self._instances:

            from ...core.explanation_engine.factories import (
                ExplanationEngineFactory,
            )

            from ...core.agent.agent import Agent
            from ...core.agent.config import AgentConfig

            llm_config = self.config.llm

            llm_kwargs = dict(
                llm_config.extra
                if hasattr(
                    llm_config,
                    "kwargs",
                )
                else {}
            )

            model_name = (
                llm_config.model
            )

            llm_kwargs.setdefault(
                "model_name",
                model_name,
            )

            llm_kwargs.setdefault(
                "requests_per_minute",
                llm_config.requests_per_minute,
            )

            llm_kwargs.setdefault(
                "min_request_interval_seconds",
                llm_config.min_request_interval_seconds,
            )

            llm_kwargs.setdefault(
                "rate_limit_retries",
                llm_config.rate_limit_retries,
            )

            llm_kwargs.setdefault(
                "max_tokens",
                llm_config.max_tokens,
            )

            llm_kwargs.setdefault(
                "timeout",
                llm_config.timeout,
            )

            agent_config = AgentConfig(
                llm_provider=(
                    llm_config.provider
                ),
                temperature=(
                    llm_config.temperature
                ),
                llm_model=(
                    model_name
                    or "gemini-3.5-flash-lite"
                ),
                llm_kwargs=llm_kwargs,
            )

            agent = Agent(
                instance_name="explainer",
                config=agent_config,
            )

            self._instances[
                "explainer"
            ] = (
                ExplanationEngineFactory
                .create_adaptive_explainer(
                    agent
                )
            )

        return self._instances[
            "explainer"
        ]

    # ================================================================
    # SERVICES
    # ================================================================

    def create_document_service(
        self,
    ) -> DocumentService:
        """Create document service."""

        if (
            "document_service"
            not in self._instances
        ):

            document_manager = (
                self.create_document_manager()
            )

            graph_state_manager = (
                self.create_graph_state_manager()
            )

            self._instances[
                "document_service"
            ] = DocumentService(
                document_manager=document_manager,
                graph_state_manager=(
                    graph_state_manager
                ),
                logger=logger,
            )

        return self._instances[
            "document_service"
        ]

    def create_user_service(
        self,
    ) -> UserService:
        """Create user service."""

        if "user_service" not in self._instances:

            def user_manager_factory(
                user_id: str,
            ) -> UserManager:

                return self.create_user_manager(
                    user_id
                )

            self._instances[
                "user_service"
            ] = UserService(
                user_manager_factory=(
                    user_manager_factory
                ),
                logger=logger,
            )

        return self._instances[
            "user_service"
        ]

    def create_context_service(
        self,
    ) -> ContextService:
        """Create context service."""

        if (
            "context_service"
            not in self._instances
        ):

            document_service = (
                self.create_document_service()
            )

            user_service = (
                self.create_user_service()
            )

            session_manager = (
                self.create_session_manager()
            )

            graph_manager = (
                self.create_graph_state_manager()
            )

            self._instances[
                "context_service"
            ] = ContextService(
                document_service=document_service,
                user_service=user_service,
                session_manager=session_manager,
                graph_manager=graph_manager,
                logger=logger,
            )

        return self._instances[
            "context_service"
        ]

    # ================================================================
    # PIPELINES
    # ================================================================

    def create_document_pipeline(
        self,
    ) -> DocumentPipeline:
        """Create document pipeline."""

        if (
            "document_pipeline"
            not in self._instances
        ):

            self._instances[
                "document_pipeline"
            ] = DocumentPipeline(
                document_service=(
                    self.create_document_service()
                ),
                context_service=(
                    self.create_context_service()
                ),
                logger=logger,
            )

        return self._instances[
            "document_pipeline"
        ]

    def create_explanation_pipeline(
        self,
    ) -> ExplanationPipeline:
        """Create explanation pipeline."""

        if (
            "explanation_pipeline"
            not in self._instances
        ):

            self._instances[
                "explanation_pipeline"
            ] = ExplanationPipeline(
                document_service=(
                    self.create_document_service()
                ),
                context_service=(
                    self.create_context_service()
                ),
                explainer=(
                    self.create_explainer()
                ),
                user_manager=(
                    self.create_user_manager()
                ),
                memory_manager=(
                    self.create_memory_manager()
                ),
                session_manager=(
                    self.create_session_manager()
                ),
                logger=logger,
            )

        return self._instances[
            "explanation_pipeline"
        ]

    def create_knowledge_pipeline(
        self,
    ) -> KnowledgePipeline:
        """Create knowledge pipeline."""

        if (
            "knowledge_pipeline"
            not in self._instances
        ):

            graph_state_manager = (
                self.create_graph_state_manager()
            )

            from ...core.knowledge.services.prerequisite_analyzer import (
                PrerequisiteAnalyzer,
            )

            from ...core.knowledge.services.learning_path import (
                LearningPathGenerator,
            )

            from ...core.knowledge.services.recommendation import (
                RecommendationService,
            )

            user_manager = (
                self.create_user_manager()
            )

            knowledge_store = (
                self.create_knowledge_store()
            )

            prerequisite_analyzer = (
                PrerequisiteAnalyzer(
                    graph_state_manager=(
                        graph_state_manager
                    ),
                    knowledge_store=(
                        knowledge_store
                    ),
                    user_manager=user_manager,
                )
            )

            learning_path_generator = (
                LearningPathGenerator(
                    knowledge_store=(
                        knowledge_store
                    ),
                    user_manager=user_manager,
                )
            )

            recommendation_service = (
                RecommendationService(
                    knowledge_store=(
                        knowledge_store
                    ),
                    user_manager=user_manager,
                )
            )

            self._instances[
                "knowledge_pipeline"
            ] = KnowledgePipeline(
                graph_state_manager=(
                    graph_state_manager
                ),
                prerequisite_analyzer=(
                    prerequisite_analyzer
                ),
                learning_path_generator=(
                    learning_path_generator
                ),
                recommendation_service=(
                    recommendation_service
                ),
                logger=logger,
            )

        return self._instances[
            "knowledge_pipeline"
        ]

    # ================================================================
    # ALL PIPELINES
    # ================================================================

    def create_all_pipelines(
        self,
    ) -> Dict[str, Any]:
        """Create all application pipelines."""

        return {
            "document": (
                self.create_document_pipeline()
            ),
            "explanation": (
                self.create_explanation_pipeline()
            ),
            "knowledge": (
                self.create_knowledge_pipeline()
            ),
        }
