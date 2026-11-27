"""System: the composition root. It builds every store and service from one config and exposes the operations.

This is the only place that knows about all the core packages. Each of them depends on interfaces, never on
another; here the real implementations are chosen and connected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..core.chat import ChatService, SqliteChatStore
from ..core.llm.factories.llm_factory import LLMFactory
from ..core.common.ports import Embedder, JsonLLM
from ..core.common.structure import DocumentStructure
from ..core.indexing import (ChromaVectorIndex, Indexer, JsonContentStore, SentenceTransformerEmbedder, SqliteStructureStore)
from ..core.knowledge import (Canonicalizer, ConceptExtractor, KnowledgeView, SqliteKnowledgeStore, TemporalKnowledgeBuilder)
from ..core.learner import LearnerTracker, SqliteLearnerStore
from ..core.tutor import QuizService, SqliteQuizStore, TutorService
from .adapters import KnowledgeAdapter, LearnerAdapter, StoreGraph, StructureLoader
from .ingestion import IngestionPipeline, IngestReport
from .llm_client import LLMJsonClient


@dataclass
class SystemConfig:
    data_dir: Path
    # No defaults for the model settings: they come from config/defaults/*.yaml via api.factory. They are only
    # needed when the System builds its own LLM / embedder (tests inject fakes instead).
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_kwargs: Dict[str, Any] = field(default_factory=dict)     # temperature, api key, base url, rate limits, ...
    embedding_model: Optional[str] = None
    embedding_check_for_updates: Optional[bool] = None
    knowledge_enabled: bool = True


class System:
    def __init__(self, config: SystemConfig, llm: Optional[JsonLLM] = None, embedder: Optional[Embedder] = None):
        d = Path(config.data_dir)
        d.mkdir(parents=True, exist_ok=True)
        self.config = config
        if embedder is None and not config.embedding_model:
            raise ValueError("SystemConfig.embedding_model is required when no embedder is given")
        if llm is None and not (config.llm_provider and config.llm_model):
            raise ValueError("SystemConfig.llm_provider and llm_model are required when no llm is given")
        self.embedder = embedder or SentenceTransformerEmbedder(config.embedding_model, check_for_updates=bool(config.embedding_check_for_updates))
        self.llm = llm or LLMJsonClient(LLMFactory.create(config.llm_provider, instance_name="system",
                                                         model_name=config.llm_model, **config.llm_kwargs))
        # storage
        self.vectors = ChromaVectorIndex(d / "vectors")
        self.structure_store = SqliteStructureStore(d / "structure.db")
        self.content_store = JsonContentStore(d / "content")
        self.knowledge_store = SqliteKnowledgeStore(d / "knowledge.db")
        self.learner_store = SqliteLearnerStore(d / "learner.db")
        self.quiz_store = SqliteQuizStore(d / "quiz.db")
        self.chat_store = SqliteChatStore(d / "chat.db")
        # services
        self.indexer = Indexer(self.embedder, self.vectors, self.structure_store, self.content_store)
        self.knowledge_builder = TemporalKnowledgeBuilder(ConceptExtractor(self.llm), Canonicalizer(self.embedder), self.knowledge_store)
        self.ingestion = IngestionPipeline(self.indexer, self.knowledge_builder if config.knowledge_enabled else None)
        self.view = KnowledgeView(self.knowledge_store)
        self.graph = StoreGraph(self.knowledge_store)
        self.tracker = LearnerTracker(self.learner_store, self.graph)
        self.structures = StructureLoader(self.content_store)
        self.knowledge = KnowledgeAdapter(self.view)
        self.learner = LearnerAdapter(self.tracker)
        self.tutor = TutorService(self.llm, self.embedder, self.vectors, self.knowledge, self.learner, self.structures)
        self.quiz = QuizService(self.llm, self.embedder, self.vectors, self.knowledge, self.learner, self.quiz_store)
        self.chat = ChatService(self.chat_store, self.llm)

    # ------------------------------------------------------------------ documents
    def supported_extensions(self) -> List[str]:
        return self.ingestion.supported_extensions()

    def ingest(self, path, doc_id: str, build_knowledge: bool = True, progress=None, cancelled=None) -> IngestReport:
        self.structures.forget(doc_id)
        return self.ingestion.run(path, doc_id, build_knowledge, progress, cancelled)

    def remove(self, doc_id: str) -> None:
        self.ingestion.remove(doc_id)
        self.structures.forget(doc_id)
        self.chat.forget_document(doc_id)

    def structure(self, doc_id: str) -> DocumentStructure:
        return self.structures.load(doc_id)

    def section_at(self, doc_id: str, page: int) -> int:
        """Reading-order ordinal of the section that holds ``page`` (what the API calls a section id)."""
        section = self.structure(doc_id).section_at_page(page)
        return section.ordinal if section is not None else 0


def build_system(config: SystemConfig, llm: Optional[JsonLLM] = None, embedder: Optional[Embedder] = None) -> System:
    return System(config, llm=llm, embedder=embedder)
