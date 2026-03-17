from typing import Optional, Dict, Any
import logging

from ..pipeline.document_pipeline import DocumentPipeline
from ..pipeline.explanation_pipeline import ExplanationPipeline
from ..pipeline.knowledge_pipeline import KnowledgePipeline
from ..services.document_service import DocumentService
from ..services.user_service import UserService
from ..services.context_service import ContextService

from src.core.document.manager import DocumentManager
from src.core.explanation_engine.adaptive_explainer import AdaptiveExplainer
from src.core.user.user_manager import UserManager
from src.core.memory.managers.memory_manager import MemoryManager
from src.core.memory.managers.session_manager import SessionManager
from src.core.knowledge.graph.state_manager import GraphStateManager
from src.core.knowledge.services.prerequisite_analyzer import PrerequisiteAnalyzer
from src.core.knowledge.services.learning_path import LearningPathGenerator
from src.core.knowledge.services.recommendation import RecommendationService
from src.core.agent.llm.factories.llm_factory import LLMFactory
from src.models.text import TextModels
from store.knowledge_store import KnowledgeStore
from store.user_store import UserStore

# Default user ID
DEFAULT_USER_ID = "user_123"


class PipelineFactory:
    """Factory for creating pipelines and their dependencies"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._instances = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_document_manager(self) -> DocumentManager:
        """Create document manager"""
        if 'document_manager' not in self._instances:
            persist_dir = self.config.get('persist_directory', 'db/vector_dbs')
            self._instances['document_manager'] = DocumentManager(
                persist_directory=persist_dir
            )
        return self._instances['document_manager']
    
    def create_text_models(self) -> TextModels:
        """Create text models"""
        if 'text_models' not in self._instances:
            self._instances['text_models'] = TextModels()
        return self._instances['text_models']
    
    def create_llm(self) -> Any:
        """Create LLM wrapper"""
        if 'llm' not in self._instances:
            provider = self.config.get('llm_provider', 'gemini')
            temperature = self.config.get('temperature', 1.0)
            self._instances['llm'] = LLMFactory.create(
                provider=provider,
                temperature=temperature
            )
        return self._instances['llm']
    
    def create_knowledge_store(self) -> KnowledgeStore:
        """Create knowledge store"""
        if 'knowledge_store' not in self._instances:
            storage_path = self.config.get('knowledge_store_path', 'db/knowledge_graph.gpickle')
            self._instances['knowledge_store'] = KnowledgeStore(
                storage_path=storage_path
            )
        return self._instances['knowledge_store']
    
    def create_user_store(self) -> UserStore:
        """Create user store"""
        if 'user_store' not in self._instances:
            self._instances['user_store'] = UserStore()
        return self._instances['user_store']
    
    def create_user_manager(self, user_id: str = DEFAULT_USER_ID) -> UserManager:
        """Create user manager"""
        user_store = self.create_user_store()
        return UserManager(user_id, user_store)
    
    def create_session_manager(self) -> SessionManager:
        """Create session manager"""
        if 'session_manager' not in self._instances:
            self._instances['session_manager'] = SessionManager()
        return self._instances['session_manager']
    
    def create_memory_manager(self) -> MemoryManager:
        """Create memory manager"""
        if 'memory_manager' not in self._instances:
            from src.core.memory.managers.memory_manager import MemoryManager
            from src.core.memory.storage.long_term_memory import LongTermMemory
            
            memory_path = self.config.get('memory_path', 'data/memory/user_memory.json')
            long_term_memory = LongTermMemory(file_path=memory_path)
            self._instances['memory_manager'] = MemoryManager(long_term_memory)
        return self._instances['memory_manager']
    
    def create_concept_canonicalizer(self) -> Any:
        """Create concept canonicalizer"""
        if 'concept_canonicalizer' not in self._instances:
            from src.core.knowledge.extraction.canonicalization.pipeline import ConceptCanonicalizer
            from src.core.knowledge.extraction.canonicalization.normalizer import TextNormalizer
            from src.core.knowledge.extraction.canonicalization.clusterer import ConceptClusterer
            from src.core.knowledge.extraction.canonicalization.llm_canonicalizer import LLMCanonicalizer
            
            knowledge_store = self.create_knowledge_store()
            llm = self.create_llm()
            text_models = self.create_text_models()
            
            normalizer = TextNormalizer()
            clusterer = ConceptClusterer(text_models.get_embedding_model())
            llm_canonicalizer = LLMCanonicalizer(llm)
            
            self._instances['concept_canonicalizer'] = ConceptCanonicalizer(
                normalizer, clusterer, llm_canonicalizer
            )
        return self._instances['concept_canonicalizer']
    
    def create_concept_extractor(self) -> Any:
        """Create concept extractor"""
        if 'concept_extractor' not in self._instances:
            from src.core.knowledge.extraction.extractor import ConceptExtractor
            from src.core.knowledge.extraction.scoring.composite_scorer import CompositeScoringStrategy
            from src.core.knowledge.extraction.scoring.frequency_scorer import FrequencyScoringStrategy
            from src.core.knowledge.extraction.scoring.position_scorer import PositionScoringStrategy
            from src.core.knowledge.extraction.scoring.definition_scorer import DefinitionBonusStrategy
            from src.core.knowledge.extraction.scoring.length_scorer import LengthScoringStrategy
            from src.core.knowledge.extraction.filters.subset_pruner import SubsetPrunerStrategy
            
            text_models = self.create_text_models()
            llm = self.create_llm()
            canonicalizer = self.create_concept_canonicalizer()
            
            # Create scoring strategy
            scoring_strategy = CompositeScoringStrategy([
                FrequencyScoringStrategy(),
                PositionScoringStrategy(),
                LengthScoringStrategy(),
                DefinitionBonusStrategy()
            ])
            
            # Create filter strategy
            filter_strategy = SubsetPrunerStrategy()
            
            self._instances['concept_extractor'] = ConceptExtractor(
                text_models=text_models,
                llm_wrapper=llm,
                canonicalizer=canonicalizer,
                scoring_strategy=scoring_strategy,
                filter_strategy=filter_strategy,
                concepts_per_para=self.config.get('concepts_per_para', 10)
            )
        return self._instances['concept_extractor']
    
    def create_relationship_extractor(self) -> Any:
        """Create relationship extractor"""
        if 'relationship_extractor' not in self._instances:
            from src.core.knowledge.extraction.strategies.relationship.llm_strategy import LLMRelationshipExtractor
            from src.core.knowledge.extraction.strategies.relationship.statistical_strategy import StatisticalRelationshipExtractor
            
            llm = self.create_llm()
            
            self._instances['llm_relationship_extractor'] = LLMRelationshipExtractor(llm)
            self._instances['statistical_relationship_extractor'] = StatisticalRelationshipExtractor()
            
            # For backward compatibility
            from src.core.knowledge.extraction.relations import Relations
            self._instances['relationship_extractor'] = self._instances['llm_relationship_extractor']
            
        return self._instances['relationship_extractor']
    
    def create_graph_state_manager(self) -> GraphStateManager:
        """Create graph state manager"""
        if 'graph_state_manager' not in self._instances:
            from src.core.knowledge.graph.builder import ConceptGraphBuilder
            from src.core.knowledge.graph.chain import DocumentChain
            from src.core.knowledge.graph.updater import GraphUpdater
            from src.core.knowledge.graph.repository import KnowledgeRepository
            from src.core.knowledge.graph.state_manager import GraphStateManager
            from src.core.document.cacher import DocumentCacher
            
            concept_extractor = self.create_concept_extractor()
            llm_relationship = self._instances.get('llm_relationship_extractor')
            statistical_relationship = self._instances.get('statistical_relationship_extractor')
            
            if not llm_relationship or not statistical_relationship:
                self.create_relationship_extractor()
                llm_relationship = self._instances['llm_relationship_extractor']
                statistical_relationship = self._instances['statistical_relationship_extractor']
            
            document_cacher = DocumentCacher()
            user_manager = self.create_user_manager()
            knowledge_store = self.create_knowledge_store()
            
            concept_graph_builder = ConceptGraphBuilder(
                concept_extractor=concept_extractor,
                llm_relationship_extractor=llm_relationship,
                statistical_relationship_extractor=statistical_relationship,
                document_cacher=document_cacher
            )
            
            document_chain = DocumentChain()
            graph_updater = GraphUpdater(user_manager, knowledge_store)
            knowledge_repository = KnowledgeRepository(document_chain, knowledge_store.graph)
            
            self._instances['graph_state_manager'] = GraphStateManager(
                user_manager=user_manager,
                concept_graph_builder=concept_graph_builder,
                document_chain=document_chain,
                graph_updater=graph_updater,
                repository=knowledge_repository,
                knowledge_store=knowledge_store
            )
            
        return self._instances['graph_state_manager']
    
    def create_explainer(self) -> AdaptiveExplainer:
        """Create adaptive explainer"""
        if 'explainer' not in self._instances:
            from src.core.explanation_engine.adaptive_explainer import AdaptiveExplainer
            from src.core.agent.agent import Agent
            from src.core.agent.config import AgentConfig
            
            agent_config = AgentConfig(
                llm_provider=self.config.get('llm_provider', 'gemini'),
                temperature=self.config.get('temperature', 1.0)
            )
            
            agent = Agent(config=agent_config)
            
            self._instances['explainer'] = AdaptiveExplainer(
                embedding_model=self.create_text_models().get_embedding_model(),
                explanation_style=self.config.get('explanation_style', 'intermediate')
            )
            # Set the agent
            self._instances['explainer'].agent = agent
            
        return self._instances['explainer']
    
    def create_document_service(self) -> DocumentService:
        """Create document service"""
        if 'document_service' not in self._instances:
            document_manager = self.create_document_manager()
            graph_state_manager = self.create_graph_state_manager()
            
            self._instances['document_service'] = DocumentService(
                document_manager=document_manager,
                graph_state_manager=graph_state_manager,
                logger=self.logger
            )
        return self._instances['document_service']
    
    def create_user_service(self) -> UserService:
        """Create user service"""
        if 'user_service' not in self._instances:
            def user_manager_factory(user_id):
                return self.create_user_manager(user_id)
            
            self._instances['user_service'] = UserService(
                user_manager_factory=user_manager_factory,
                logger=self.logger
            )
        return self._instances['user_service']
    
    def create_context_service(self) -> ContextService:
        """Create context service"""
        if 'context_service' not in self._instances:
            document_service = self.create_document_service()
            user_service = self.create_user_service()
            session_manager = self.create_session_manager()
            
            self._instances['context_service'] = ContextService(
                document_service=document_service,
                user_service=user_service,
                session_manager=session_manager,
                logger=self.logger
            )
        return self._instances['context_service']
    
    def create_document_pipeline(self) -> DocumentPipeline:
        """Create document pipeline"""
        if 'document_pipeline' not in self._instances:
            document_service = self.create_document_service()
            context_service = self.create_context_service()
            
            self._instances['document_pipeline'] = DocumentPipeline(
                document_service=document_service,
                context_service=context_service,
                logger=self.logger
            )
        return self._instances['document_pipeline']
    
    def create_explanation_pipeline(self) -> ExplanationPipeline:
        """Create explanation pipeline"""
        if 'explanation_pipeline' not in self._instances:
            document_service = self.create_document_service()
            context_service = self.create_context_service()
            explainer = self.create_explainer()
            user_manager = self.create_user_manager()
            memory_manager = self.create_memory_manager()
            session_manager = self.create_session_manager()
            
            self._instances['explanation_pipeline'] = ExplanationPipeline(
                document_service=document_service,
                context_service=context_service,
                explainer=explainer,
                user_manager=user_manager,
                memory_manager=memory_manager,
                session_manager=session_manager,
                logger=self.logger
            )
        return self._instances['explanation_pipeline']
    
    def create_knowledge_pipeline(self) -> KnowledgePipeline:
        """Create knowledge pipeline"""
        if 'knowledge_pipeline' not in self._instances:
            graph_state_manager = self.create_graph_state_manager()
            
            # Create prerequisite analyzer if needed
            from src.core.knowledge.services.prerequisite_analyzer import PrerequisiteAnalyzer
            from src.core.knowledge.services.learning_path import LearningPathGenerator
            from src.core.knowledge.services.recommendation import RecommendationService
            
            user_manager = self.create_user_manager()
            knowledge_store = self.create_knowledge_store()
            
            prerequisite_analyzer = PrerequisiteAnalyzer(
                graph_state_manager=graph_state_manager,
                knowledge_store=knowledge_store,
                user_manager=user_manager
            )
            
            learning_path_generator = LearningPathGenerator(
                knowledge_store=knowledge_store,
                user_manager=user_manager
            )
            
            recommendation_service = RecommendationService(
                knowledge_store=knowledge_store,
                user_manager=user_manager
            )
            
            self._instances['knowledge_pipeline'] = KnowledgePipeline(
                graph_state_manager=graph_state_manager,
                prerequisite_analyzer=prerequisite_analyzer,
                learning_path_generator=learning_path_generator,
                recommendation_service=recommendation_service,
                logger=self.logger
            )
        return self._instances['knowledge_pipeline']
    
    def create_all_pipelines(self) -> Dict[str, Any]:
        """Create all pipelines"""
        return {
            'document': self.create_document_pipeline(),
            'explanation': self.create_explanation_pipeline(),
            'knowledge': self.create_knowledge_pipeline()
        }