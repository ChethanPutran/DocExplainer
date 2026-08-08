"""
Retrieval-Augmented Generation (RAG) system with semantic caching, multi-document retrieval,
hierarchical retrieval, and concept-graph aware ranking.

This module implements:
- Semantic caching layer with LRU eviction policy
- Multi-document retrieval with document-context awareness
- Hierarchical retrieval (query → document → section → paragraph)
- Concept-graph aware ranking using NetworkX graphs
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum

import networkx as nx

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries for adjusting retrieval chunk sizes."""
    DEFINITION = "definition"
    EXPLANATION = "explanation"
    COMPARISON = "comparison"
    PREREQUISITE = "prerequisite"
    EXAMPLE = "example"


@dataclass
class DocumentReference:
    """Represents a reference to a location in a document."""
    doc_id: str
    doc_name: str
    section: Optional[str] = None
    paragraph_idx: Optional[int] = None
    start_char: int = 0
    end_char: int = 0
    
    def __hash__(self) -> int:
        return hash((self.doc_id, self.section, self.paragraph_idx))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DocumentReference):
            return False
        return (self.doc_id == other.doc_id and 
                self.section == other.section and 
                self.paragraph_idx == other.paragraph_idx)


@dataclass
class RetrievalResult:
    """Represents a retrieval result with ranking metadata."""
    content: str
    document_ref: DocumentReference
    relevance_score: float
    concept_relevance_score: float = 0.0
    chunk_size: int = 0
    distance_to_query_doc: int = 0
    contains_prerequisites: bool = False
    contains_unknown_concepts: bool = False
    
    @property
    def combined_score(self) -> float:
        """Compute combined relevance score considering all factors."""
        base_score = self.relevance_score * 0.6 + self.concept_relevance_score * 0.4
        
        if self.contains_prerequisites:
            base_score *= 1.2
        if self.contains_unknown_concepts:
            base_score *= 0.8
        if self.distance_to_query_doc > 0:
            base_score *= (1.0 / (1.0 + 0.1 * self.distance_to_query_doc))
        
        return min(1.0, base_score)


@dataclass
class CachedEmbedding:
    """Cached embedding with metadata."""
    text: str
    embedding: List[float]
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)


@dataclass
class CachedExplanation:
    """Cached explanation result with context."""
    query: str
    result: str
    user_concepts: Set[str]
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass


class SentenceTransformerProvider(EmbeddingProvider):
    """Embedding provider using SentenceTransformer."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize SentenceTransformer embedding provider.
        
        Args:
            model_name: Name of the sentence transformer model to use.
        """
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for this provider")
        self.model = SentenceTransformer(model_name)
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()


class SemanticCache:
    """LRU cache for semantic embeddings and explanations."""
    
    def __init__(self, max_embeddings: int = 1000, max_explanations: int = 500):
        """Initialize semantic cache.
        
        Args:
            max_embeddings: Maximum number of cached embeddings (LRU eviction).
            max_explanations: Maximum number of cached explanations (LRU eviction).
        """
        self.max_embeddings = max_embeddings
        self.max_explanations = max_explanations
        
        self._embeddings: OrderedDict[str, CachedEmbedding] = OrderedDict()
        self._explanations: OrderedDict[str, CachedExplanation] = OrderedDict()
    
    def _hash_text(self, text: str) -> str:
        """Create hash key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding if exists.
        
        Args:
            text: Text to retrieve embedding for.
            
        Returns:
            Cached embedding or None if not found.
        """
        key = self._hash_text(text)
        if key in self._embeddings:
            cached = self._embeddings[key]
            cached.access_count += 1
            cached.last_accessed = datetime.now()
            self._embeddings.move_to_end(key)
            return cached.embedding
        return None
    
    def set_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache an embedding.
        
        Args:
            text: Original text.
            embedding: Generated embedding vector.
        """
        key = self._hash_text(text)
        if key in self._embeddings:
            self._embeddings.move_to_end(key)
        
        self._embeddings[key] = CachedEmbedding(text=text, embedding=embedding)
        
        if len(self._embeddings) > self.max_embeddings:
            self._embeddings.popitem(last=False)
    
    def get_explanation(self, query: str, user_concepts: Set[str]) -> Optional[str]:
        """Get cached explanation if exists.
        
        Args:
            query: Query string.
            user_concepts: Set of user's known concepts.
            
        Returns:
            Cached explanation or None if not found.
        """
        key = self._hash_text(query + "|" + "|".join(sorted(user_concepts)))
        if key in self._explanations:
            cached = self._explanations[key]
            cached.access_count += 1
            cached.last_accessed = datetime.now()
            self._explanations.move_to_end(key)
            return cached.result
        return None
    
    def set_explanation(self, query: str, result: str, user_concepts: Set[str]) -> None:
        """Cache an explanation result.
        
        Args:
            query: Query string.
            result: Generated explanation.
            user_concepts: Set of user's known concepts.
        """
        key = self._hash_text(query + "|" + "|".join(sorted(user_concepts)))
        if key in self._explanations:
            self._explanations.move_to_end(key)
        
        self._explanations[key] = CachedExplanation(
            query=query,
            result=result,
            user_concepts=user_concepts
        )
        
        if len(self._explanations) > self.max_explanations:
            self._explanations.popitem(last=False)
    
    def clear(self) -> None:
        """Clear all cached items."""
        self._embeddings.clear()
        self._explanations.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "embeddings_cached": len(self._embeddings),
            "embeddings_max": self.max_embeddings,
            "explanations_cached": len(self._explanations),
            "explanations_max": self.max_explanations,
            "avg_embedding_accesses": (
                sum(e.access_count for e in self._embeddings.values()) / len(self._embeddings)
                if self._embeddings else 0
            ),
            "avg_explanation_accesses": (
                sum(e.access_count for e in self._explanations.values()) / len(self._explanations)
                if self._explanations else 0
            ),
        }


class MultiDocumentRetriever:
    """Retrieves content from multiple documents with context awareness."""
    
    def __init__(self, embedding_provider: EmbeddingProvider):
        """Initialize multi-document retriever.
        
        Args:
            embedding_provider: Provider for generating embeddings.
        """
        self.embedding_provider = embedding_provider
        self._documents: Dict[str, Dict[str, Any]] = {}
        self._embeddings_index: Optional[Any] = None
        
        if chromadb is not None:
            settings = Settings(
                chroma_db_impl="duckdb",
                persist_directory="./chroma_data",
                anonymized_telemetry=False,
            )
            try:
                self.chroma_client = chromadb.Client(settings)
                self.collection = self.chroma_client.get_or_create_collection(
                    name="rag_documents",
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                logger.warning(f"Failed to initialize ChromaDB: {e}")
                self.chroma_client = None
                self.collection = None
        else:
            self.chroma_client = None
            self.collection = None
    
    def add_document(self, doc_id: str, doc_name: str, content: Dict[str, Any]) -> None:
        """Add a document for retrieval.
        
        Args:
            doc_id: Unique document identifier.
            doc_name: Human-readable document name.
            content: Document content with structure (sections, paragraphs, etc.).
        """
        self._documents[doc_id] = {
            "name": doc_name,
            "content": content,
            "added_at": datetime.now(),
        }
        
        if self.collection is not None:
            self._index_document_in_chroma(doc_id, doc_name, content)
    
    def _index_document_in_chroma(self, doc_id: str, doc_name: str, content: Dict[str, Any]) -> None:
        """Index document content in ChromaDB.
        
        Args:
            doc_id: Document ID.
            doc_name: Document name.
            content: Document content.
        """
        try:
            doc_texts = self._flatten_document_content(content)
            metadatas = []
            ids = []
            documents = []
            
            for idx, (text, ref_info) in enumerate(doc_texts):
                chunk_id = f"{doc_id}_{idx}"
                ids.append(chunk_id)
                documents.append(text)
                metadatas.append({
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "section": ref_info.get("section", ""),
                    "paragraph_idx": str(ref_info.get("paragraph_idx", -1)),
                })
            
            if documents:
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=[self.embedding_provider.embed(doc) for doc in documents],
                )
        except Exception as e:
            logger.error(f"Failed to index document in ChromaDB: {e}")
    
    def _flatten_document_content(self, content: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """Flatten document content into searchable chunks.
        
        Args:
            content: Document content structure.
            
        Returns:
            List of (text, metadata) tuples.
        """
        result = []
        
        if isinstance(content, dict):
            for section_name, section_content in content.items():
                if isinstance(section_content, dict):
                    for para_idx, paragraph in enumerate(section_content.get("paragraphs", [])):
                        result.append((
                            str(paragraph),
                            {"section": section_name, "paragraph_idx": para_idx}
                        ))
                elif isinstance(section_content, list):
                    for para_idx, paragraph in enumerate(section_content):
                        result.append((
                            str(paragraph),
                            {"section": section_name, "paragraph_idx": para_idx}
                        ))
                else:
                    result.append((
                        str(section_content),
                        {"section": section_name, "paragraph_idx": 0}
                    ))
        
        return result
    
    def retrieve(self, query: str, doc_ids: Optional[List[str]] = None, 
                 top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve relevant content from documents.
        
        Args:
            query: Query string.
            doc_ids: Specific documents to search (None = all documents).
            top_k: Number of top results to return.
            
        Returns:
            List of retrieval results ranked by relevance.
        """
        if self.collection is None:
            return self._retrieve_fallback(query, doc_ids, top_k)
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k * 2,
                where={"doc_id": {"$in": doc_ids}} if doc_ids else None,
            )
            
            retrieval_results = []
            if results and results["documents"] and len(results["documents"]) > 0:
                for doc, metadata, distance in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                ):
                    if doc_ids and metadata["doc_id"] not in doc_ids:
                        continue
                    
                    doc_ref = DocumentReference(
                        doc_id=metadata["doc_id"],
                        doc_name=metadata["doc_name"],
                        section=metadata.get("section"),
                        paragraph_idx=int(metadata.get("paragraph_idx", -1)),
                    )
                    
                    retrieval_results.append(RetrievalResult(
                        content=doc,
                        document_ref=doc_ref,
                        relevance_score=1.0 - distance,
                        chunk_size=len(doc),
                    ))
            
            return retrieval_results[:top_k]
        except Exception as e:
            logger.error(f"ChromaDB retrieval failed: {e}")
            return self._retrieve_fallback(query, doc_ids, top_k)
    
    def _retrieve_fallback(self, query: str, doc_ids: Optional[List[str]] = None,
                           top_k: int = 5) -> List[RetrievalResult]:
        """Fallback retrieval using simple similarity.
        
        Args:
            query: Query string.
            doc_ids: Specific documents to search.
            top_k: Number of results.
            
        Returns:
            List of retrieval results.
        """
        query_embedding = self.embedding_provider.embed(query)
        results = []
        
        target_docs = doc_ids if doc_ids else list(self._documents.keys())
        
        for doc_id in target_docs:
            if doc_id not in self._documents:
                continue
            
            doc_content = self._documents[doc_id]
            chunks = self._flatten_document_content(doc_content["content"])
            
            for text, ref_info in chunks:
                embedding = self.embedding_provider.embed(text)
                
                similarity = self._cosine_similarity(query_embedding, embedding)
                
                doc_ref = DocumentReference(
                    doc_id=doc_id,
                    doc_name=doc_content["name"],
                    section=ref_info.get("section"),
                    paragraph_idx=ref_info.get("paragraph_idx"),
                )
                
                results.append(RetrievalResult(
                    content=text,
                    document_ref=doc_ref,
                    relevance_score=similarity,
                    chunk_size=len(text),
                ))
        
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:top_k]
    
    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between vectors.
        
        Args:
            vec1: First vector.
            vec2: Second vector.
            
        Returns:
            Cosine similarity score.
        """
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)


class HierarchicalRetriever:
    """Implements hierarchical retrieval from query to paragraph level."""
    
    def __init__(self, multi_doc_retriever: MultiDocumentRetriever):
        """Initialize hierarchical retriever.
        
        Args:
            multi_doc_retriever: Underlying multi-document retriever.
        """
        self.retriever = multi_doc_retriever
    
    def retrieve_hierarchical(self, query: str, query_type: QueryType = QueryType.EXPLANATION,
                             doc_ids: Optional[List[str]] = None,
                             top_k: int = 5) -> List[RetrievalResult]:
        """Perform hierarchical retrieval with adaptive chunk sizing.
        
        Hierarchy: Query → Document level → Section level → Paragraph level
        Chunk size is adjusted based on query type.
        
        Args:
            query: Query string.
            query_type: Type of query to adjust chunk sizing.
            doc_ids: Specific documents to search.
            top_k: Number of results.
            
        Returns:
            List of hierarchically retrieved results.
        """
        chunk_size_multiplier = self._get_chunk_size_multiplier(query_type)
        results = self.retriever.retrieve(query, doc_ids, top_k * chunk_size_multiplier)
        
        return results[:top_k]
    
    def _get_chunk_size_multiplier(self, query_type: QueryType) -> int:
        """Get chunk size multiplier based on query type.
        
        Args:
            query_type: Type of query.
            
        Returns:
            Multiplier for chunk size.
        """
        multipliers = {
            QueryType.DEFINITION: 1,
            QueryType.EXPLANATION: 2,
            QueryType.COMPARISON: 3,
            QueryType.PREREQUISITE: 2,
            QueryType.EXAMPLE: 2,
        }
        return multipliers.get(query_type, 2)


class ConceptGraphRanker:
    """Ranks retrieval results using concept graph information."""
    
    def __init__(self, concept_graph: Optional[nx.DiGraph] = None):
        """Initialize concept graph ranker.
        
        Args:
            concept_graph: NetworkX directed graph representing concept relationships.
        """
        self.concept_graph = concept_graph or nx.DiGraph()
        self.known_concepts: Set[str] = set()
    
    def add_concept_graph(self, graph: nx.DiGraph) -> None:
        """Add or update the concept graph.
        
        Args:
            graph: NetworkX directed graph.
        """
        self.concept_graph = graph.copy()
        self.known_concepts = set(self.concept_graph.nodes())
    
    def set_user_concepts(self, user_concepts: Set[str]) -> None:
        """Set the user's known concepts for ranking.
        
        Args:
            user_concepts: Set of concept names the user knows.
        """
        self.known_concepts = user_concepts
    
    def rank_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rank retrieval results using concept graph.
        
        Ranking criteria:
        - Boost results containing prerequisite concepts
        - Penalize results with unknown concepts
        
        Args:
            results: Unranked retrieval results.
            
        Returns:
            Ranked retrieval results.
        """
        for result in results:
            concepts_in_content = self._extract_concepts(result.content)
            
            prerequisites = self._get_prerequisite_concepts(concepts_in_content)
            unknown = self._get_unknown_concepts(concepts_in_content)
            
            result.concept_relevance_score = self._compute_concept_score(
                concepts_in_content, prerequisites, unknown
            )
            result.contains_prerequisites = len(prerequisites) > 0
            result.contains_unknown_concepts = len(unknown) > 0
        
        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results
    
    def _extract_concepts(self, text: str) -> Set[str]:
        """Extract concept references from text.
        
        Args:
            text: Text to extract concepts from.
            
        Returns:
            Set of concept names found in text.
        """
        concepts = set()
        text_lower = text.lower()
        
        for concept in self.concept_graph.nodes():
            if concept.lower() in text_lower:
                concepts.add(concept)
        
        return concepts
    
    def _get_prerequisite_concepts(self, concepts: Set[str]) -> Set[str]:
        """Find prerequisite concepts for given concepts.
        
        Args:
            concepts: Set of concepts to find prerequisites for.
            
        Returns:
            Set of prerequisite concepts that are known.
        """
        prerequisites = set()
        
        for concept in concepts:
            if concept in self.concept_graph:
                for predecessor in self.concept_graph.predecessors(concept):
                    if predecessor in self.known_concepts:
                        prerequisites.add(predecessor)
        
        return prerequisites
    
    def _get_unknown_concepts(self, concepts: Set[str]) -> Set[str]:
        """Find unknown concepts in given set.
        
        Args:
            concepts: Set of concepts to check.
            
        Returns:
            Set of concepts not in user's known concepts.
        """
        return {c for c in concepts if c not in self.known_concepts}
    
    def _compute_concept_score(self, concepts: Set[str], prerequisites: Set[str],
                               unknown: Set[str]) -> float:
        """Compute concept relevance score.
        
        Args:
            concepts: Concepts in content.
            prerequisites: Prerequisites for those concepts.
            unknown: Unknown concepts.
            
        Returns:
            Score between 0 and 1.
        """
        if not concepts:
            return 0.5
        
        score = 0.5
        
        if prerequisites:
            score += 0.3 * (len(prerequisites) / len(concepts))
        
        if unknown:
            score -= 0.2 * (len(unknown) / len(concepts))
        
        return max(0.0, min(1.0, score))


class RAGSystem:
    """Main Retrieval-Augmented Generation system combining all components."""
    
    def __init__(self, embedding_provider: Optional[EmbeddingProvider] = None,
                 concept_graph: Optional[nx.DiGraph] = None,
                 max_cached_embeddings: int = 1000,
                 max_cached_explanations: int = 500):
        """Initialize RAG system.
        
        Args:
            embedding_provider: Provider for embeddings (defaults to SentenceTransformer).
            concept_graph: NetworkX directed graph for concept relationships.
            max_cached_embeddings: Maximum cached embeddings (LRU).
            max_cached_explanations: Maximum cached explanations (LRU).
        """
        if embedding_provider is None:
            embedding_provider = SentenceTransformerProvider()
        
        self.embedding_provider = embedding_provider
        self.semantic_cache = SemanticCache(max_cached_embeddings, max_cached_explanations)
        self.multi_doc_retriever = MultiDocumentRetriever(embedding_provider)
        self.hierarchical_retriever = HierarchicalRetriever(self.multi_doc_retriever)
        self.concept_ranker = ConceptGraphRanker(concept_graph)
    
    def add_document(self, doc_id: str, doc_name: str, content: Dict[str, Any]) -> None:
        """Add a document to the RAG system.
        
        Args:
            doc_id: Unique document identifier.
            doc_name: Human-readable document name.
            content: Document content structure.
        """
        self.multi_doc_retriever.add_document(doc_id, doc_name, content)
    
    def set_concept_graph(self, graph: nx.DiGraph) -> None:
        """Set or update the concept graph.
        
        Args:
            graph: NetworkX directed graph.
        """
        self.concept_ranker.add_concept_graph(graph)
    
    def set_user_concepts(self, user_concepts: Set[str]) -> None:
        """Set user's known concepts for ranking.
        
        Args:
            user_concepts: Set of concept names.
        """
        self.concept_ranker.set_user_concepts(user_concepts)
    
    def retrieve(self, query: str, query_type: QueryType = QueryType.EXPLANATION,
                doc_ids: Optional[List[str]] = None, top_k: int = 5,
                use_concept_ranking: bool = True) -> List[RetrievalResult]:
        """Perform complete RAG retrieval with all enhancements.
        
        Args:
            query: Query string.
            query_type: Type of query for hierarchical adjustment.
            doc_ids: Specific documents to search (None = all).
            top_k: Number of results to return.
            use_concept_ranking: Whether to apply concept graph ranking.
            
        Returns:
            List of ranked retrieval results.
        """
        results = self.hierarchical_retriever.retrieve_hierarchical(
            query, query_type, doc_ids, top_k
        )
        
        if use_concept_ranking:
            results = self.concept_ranker.rank_results(results)
        
        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results[:top_k]
    
    def get_explanation_with_cache(self, query: str, user_concepts: Set[str],
                                   doc_ids: Optional[List[str]] = None,
                                   top_k: int = 5) -> Tuple[str, List[RetrievalResult]]:
        """Get explanation with semantic caching.
        
        Args:
            query: Query string.
            user_concepts: User's known concepts.
            doc_ids: Specific documents to search.
            top_k: Number of retrieval results.
            
        Returns:
            Tuple of (explanation, retrieval_results). Explanation may be cached.
        """
        cached_explanation = self.semantic_cache.get_explanation(query, user_concepts)
        if cached_explanation:
            logger.info("Explanation retrieved from cache")
            return cached_explanation, []
        
        self.set_user_concepts(user_concepts)
        results = self.retrieve(query, QueryType.EXPLANATION, doc_ids, top_k)
        
        explanation = self._generate_explanation(query, results, user_concepts)
        self.semantic_cache.set_explanation(query, explanation, user_concepts)
        
        return explanation, results
    
    def _generate_explanation(self, query: str, results: List[RetrievalResult],
                             user_concepts: Set[str]) -> str:
        """Generate explanation from retrieval results.
        
        Args:
            query: Query string.
            results: Retrieval results.
            user_concepts: User's known concepts.
            
        Returns:
            Generated explanation.
        """
        if not results:
            return f"No relevant information found for query: {query}"
        
        explanation_parts = [f"Explanation for: {query}\n"]
        
        for i, result in enumerate(results, 1):
            explanation_parts.append(f"\n{i}. From '{result.document_ref.doc_name}':")
            if result.document_ref.section:
                explanation_parts.append(f"   Section: {result.document_ref.section}")
            explanation_parts.append(f"   {result.content[:200]}...")
        
        return "\n".join(explanation_parts)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get semantic cache statistics.
        
        Returns:
            Dictionary with cache statistics.
        """
        return self.semantic_cache.get_stats()
    
    def clear_cache(self) -> None:
        """Clear all semantic caches."""
        self.semantic_cache.clear()
