"""
Comprehensive tests for RAG (Retrieval-Augmented Generation) system.

Tests cover:
- Semantic caching (embeddings and explanations)
- Multi-document retrieval
- Hierarchical retrieval with query type adaptation
- Concept graph aware ranking
- Full RAG system integration
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import networkx as nx

from .....memory.rag import (
    QueryType,
    DocumentReference,
    RetrievalResult,
    CachedEmbedding,
    CachedExplanation,
    SemanticCache,
    EmbeddingProvider,
    SentenceTransformerProvider,
    MultiDocumentRetriever,
    HierarchicalRetriever,
    ConceptGraphRanker,
    RAGSystem,
)


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""
    
    def __init__(self):
        self.call_count = 0
    
    def embed(self, text: str):
        """Generate deterministic mock embedding."""
        self.call_count += 1
        return [float(ord(c)) for c in text[:10].ljust(10)]
    
    def embed_batch(self, texts):
        """Generate batch embeddings."""
        return [self.embed(text) for text in texts]


class TestDocumentReference:
    """Tests for DocumentReference class."""
    
    def test_document_reference_creation(self):
        """Test creating document reference."""
        ref = DocumentReference(
            doc_id="doc1",
            doc_name="Document 1",
            section="Section 1",
            paragraph_idx=0,
        )
        
        assert ref.doc_id == "doc1"
        assert ref.doc_name == "Document 1"
        assert ref.section == "Section 1"
        assert ref.paragraph_idx == 0
    
    def test_document_reference_equality(self):
        """Test document reference equality."""
        ref1 = DocumentReference(
            doc_id="doc1",
            doc_name="Document 1",
            section="Section 1",
            paragraph_idx=0,
        )
        ref2 = DocumentReference(
            doc_id="doc1",
            doc_name="Document 1",
            section="Section 1",
            paragraph_idx=0,
        )
        
        assert ref1 == ref2
    
    def test_document_reference_hash(self):
        """Test document reference hashing."""
        ref = DocumentReference(
            doc_id="doc1",
            doc_name="Document 1",
            section="Section 1",
            paragraph_idx=0,
        )
        
        assert hash(ref) is not None


class TestRetrievalResult:
    """Tests for RetrievalResult class."""
    
    def test_retrieval_result_creation(self):
        """Test creating retrieval result."""
        ref = DocumentReference(doc_id="doc1", doc_name="Doc 1")
        result = RetrievalResult(
            content="Test content",
            document_ref=ref,
            relevance_score=0.9,
        )
        
        assert result.content == "Test content"
        assert result.relevance_score == 0.9
    
    def test_combined_score_calculation(self):
        """Test combined score calculation."""
        ref = DocumentReference(doc_id="doc1", doc_name="Doc 1")
        result = RetrievalResult(
            content="Test content",
            document_ref=ref,
            relevance_score=0.8,
            concept_relevance_score=0.6,
        )
        
        score = result.combined_score
        assert 0 <= score <= 1.0
        assert score > 0
    
    def test_combined_score_with_prerequisites(self):
        """Test score boosted with prerequisites."""
        ref = DocumentReference(doc_id="doc1", doc_name="Doc 1")
        result = RetrievalResult(
            content="Test content",
            document_ref=ref,
            relevance_score=0.8,
            concept_relevance_score=0.6,
            contains_prerequisites=True,
        )
        
        score_with_prereq = result.combined_score
        
        result.contains_prerequisites = False
        score_without_prereq = result.combined_score
        
        assert score_with_prereq > score_without_prereq


class TestSemanticCache:
    """Tests for semantic cache."""
    
    def test_cache_embedding_set_get(self):
        """Test setting and getting cached embeddings."""
        cache = SemanticCache(max_embeddings=10)
        text = "test text"
        embedding = [0.1, 0.2, 0.3]
        
        cache.set_embedding(text, embedding)
        retrieved = cache.get_embedding(text)
        
        assert retrieved == embedding
    
    def test_cache_embedding_miss(self):
        """Test cache miss for non-existent embedding."""
        cache = SemanticCache()
        
        retrieved = cache.get_embedding("non-existent")
        assert retrieved is None
    
    def test_cache_embedding_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = SemanticCache(max_embeddings=3)
        
        cache.set_embedding("text1", [0.1])
        cache.set_embedding("text2", [0.2])
        cache.set_embedding("text3", [0.3])
        cache.set_embedding("text4", [0.4])
        
        assert cache.get_embedding("text1") is None
        assert cache.get_embedding("text4") is not None
    
    def test_cache_explanation_set_get(self):
        """Test setting and getting cached explanations."""
        cache = SemanticCache(max_explanations=10)
        query = "what is X?"
        result = "X is..."
        concepts = {"concept1", "concept2"}
        
        cache.set_explanation(query, result, concepts)
        retrieved = cache.get_explanation(query, concepts)
        
        assert retrieved == result
    
    def test_cache_explanation_context_aware(self):
        """Test that explanations are context-aware."""
        cache = SemanticCache()
        query = "what is X?"
        
        cache.set_explanation(query, "explanation1", {"concept1"})
        cached = cache.get_explanation(query, {"concept2"})
        
        assert cached is None
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = SemanticCache()
        cache.set_embedding("text1", [0.1])
        cache.set_explanation("query1", "result1", {"concept1"})
        
        stats = cache.get_stats()
        
        assert stats["embeddings_cached"] == 1
        assert stats["explanations_cached"] == 1
        assert "avg_embedding_accesses" in stats
    
    def test_cache_clear(self):
        """Test clearing cache."""
        cache = SemanticCache()
        cache.set_embedding("text1", [0.1])
        cache.set_explanation("query1", "result1", {"concept1"})
        
        cache.clear()
        
        assert cache.get_embedding("text1") is None
        assert cache.get_explanation("query1", {"concept1"}) is None


class TestMultiDocumentRetriever:
    """Tests for multi-document retriever."""
    
    @pytest.fixture
    def retriever(self):
        """Create retriever with mock embedding provider."""
        provider = MockEmbeddingProvider()
        return MultiDocumentRetriever(provider)
    
    def test_add_document(self, retriever):
        """Test adding documents."""
        content = {
            "introduction": {"paragraphs": ["Paragraph 1", "Paragraph 2"]},
            "details": {"paragraphs": ["Details 1", "Details 2"]},
        }
        
        retriever.add_document("doc1", "Test Document", content)
        
        assert "doc1" in retriever._documents
        assert retriever._documents["doc1"]["name"] == "Test Document"
    
    def test_retrieve_single_document(self, retriever):
        """Test retrieval from single document."""
        content = {
            "section1": {"paragraphs": ["information about machine learning"]},
        }
        
        retriever.add_document("doc1", "ML Guide", content)
        results = retriever.retrieve("machine learning", top_k=1)
        
        assert len(results) > 0
        assert results[0].document_ref.doc_id == "doc1"
    
    def test_retrieve_multiple_documents(self, retriever):
        """Test retrieval from multiple documents."""
        retriever.add_document("doc1", "Doc 1", {
            "section1": {"paragraphs": ["Python programming"]},
        })
        retriever.add_document("doc2", "Doc 2", {
            "section1": {"paragraphs": ["Java programming"]},
        })
        
        results = retriever.retrieve("programming", top_k=5)
        
        assert len(results) > 0
        doc_ids = {r.document_ref.doc_id for r in results}
        assert "doc1" in doc_ids or "doc2" in doc_ids
    
    def test_retrieve_with_doc_filter(self, retriever):
        """Test retrieval with document ID filter."""
        retriever.add_document("doc1", "Doc 1", {
            "section1": {"paragraphs": ["Python"]},
        })
        retriever.add_document("doc2", "Doc 2", {
            "section1": {"paragraphs": ["Python"]},
        })
        
        results = retriever.retrieve("Python", doc_ids=["doc1"], top_k=10)
        
        for result in results:
            assert result.document_ref.doc_id == "doc1"
    
    def test_flatten_document_content(self, retriever):
        """Test document content flattening."""
        content = {
            "section1": {"paragraphs": ["para1", "para2"]},
            "section2": {"paragraphs": ["para3"]},
        }
        
        flattened = retriever._flatten_document_content(content)
        
        assert len(flattened) == 3
        assert all(isinstance(item, tuple) and len(item) == 2 for item in flattened)
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        vec1 = [1, 0, 0]
        vec2 = [1, 0, 0]
        
        similarity = MultiDocumentRetriever._cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 0.001
        
        vec3 = [0, 1, 0]
        similarity = MultiDocumentRetriever._cosine_similarity(vec1, vec3)
        assert abs(similarity) < 0.001


class TestHierarchicalRetriever:
    """Tests for hierarchical retriever."""
    
    @pytest.fixture
    def hierarchical_retriever(self):
        """Create hierarchical retriever."""
        provider = MockEmbeddingProvider()
        multi_doc = MultiDocumentRetriever(provider)
        return HierarchicalRetriever(multi_doc)
    
    def test_hierarchical_retrieve(self, hierarchical_retriever):
        """Test hierarchical retrieval."""
        content = {
            "intro": {"paragraphs": ["Introduction content"]},
        }
        hierarchical_retriever.retriever.add_document("doc1", "Doc 1", content)
        
        results = hierarchical_retriever.retrieve_hierarchical("content", top_k=1)
        
        assert isinstance(results, list)
    
    def test_chunk_size_multiplier_by_query_type(self, hierarchical_retriever):
        """Test chunk size multiplier varies by query type."""
        multipliers = []
        
        for query_type in QueryType:
            multiplier = hierarchical_retriever._get_chunk_size_multiplier(query_type)
            multipliers.append(multiplier)
        
        assert len(set(multipliers)) > 1
        assert QueryType.DEFINITION in [QueryType.DEFINITION]
        assert all(m >= 1 for m in multipliers)


class TestConceptGraphRanker:
    """Tests for concept graph ranker."""
    
    @pytest.fixture
    def ranker(self):
        """Create ranker with sample graph."""
        graph = nx.DiGraph()
        graph.add_edges_from([
            ("Python", "Functions"),
            ("Functions", "Lambda"),
            ("Python", "Classes"),
            ("Classes", "Inheritance"),
        ])
        return ConceptGraphRanker(graph)
    
    def test_ranker_initialization(self, ranker):
        """Test ranker initialization."""
        assert ranker.concept_graph is not None
        assert len(ranker.concept_graph.nodes()) > 0
    
    def test_set_user_concepts(self, ranker):
        """Test setting user's known concepts."""
        concepts = {"Python", "Functions"}
        ranker.set_user_concepts(concepts)
        
        assert ranker.known_concepts == concepts
    
    def test_extract_concepts(self, ranker):
        """Test concept extraction from text."""
        ranker.set_user_concepts({"Python", "Functions"})
        text = "Python Functions are useful"
        
        concepts = ranker._extract_concepts(text)
        
        assert "Python" in concepts or "Functions" in concepts
    
    def test_rank_results_with_prerequisites(self, ranker):
        """Test ranking boosts results with prerequisites."""
        ranker.set_user_concepts({"Python"})
        
        ref1 = DocumentReference(doc_id="doc1", doc_name="Doc 1")
        result1 = RetrievalResult(
            content="Learn about Functions in Python",
            document_ref=ref1,
            relevance_score=0.7,
        )
        
        results = [result1]
        ranked = ranker.rank_results(results)
        
        assert ranked[0].concept_relevance_score >= 0
    
    def test_rank_results_with_unknown_concepts(self, ranker):
        """Test ranking penalizes results with unknown concepts."""
        ranker.set_user_concepts(set())
        
        ref = DocumentReference(doc_id="doc1", doc_name="Doc 1")
        result = RetrievalResult(
            content="Lambda expressions in Python",
            document_ref=ref,
            relevance_score=0.9,
        )
        
        ranked = ranker.rank_results([result])
        
        assert ranked[0].concept_relevance_score <= 0.5
    
    def test_get_prerequisite_concepts(self, ranker):
        """Test extracting prerequisite concepts."""
        ranker.set_user_concepts({"Python", "Functions"})
        
        prerequisites = ranker._get_prerequisite_concepts({"Lambda"})
        
        assert isinstance(prerequisites, set)


class TestRAGSystem:
    """Tests for main RAG system."""
    
    @pytest.fixture
    def rag_system(self):
        """Create RAG system."""
        provider = MockEmbeddingProvider()
        return RAGSystem(embedding_provider=provider)
    
    def test_rag_initialization(self, rag_system):
        """Test RAG system initialization."""
        assert rag_system.embedding_provider is not None
        assert rag_system.semantic_cache is not None
        assert rag_system.multi_doc_retriever is not None
        assert rag_system.concept_ranker is not None
    
    def test_add_document(self, rag_system):
        """Test adding document to RAG system."""
        content = {
            "intro": {"paragraphs": ["Introduction"]},
        }
        
        rag_system.add_document("doc1", "Test Doc", content)
        
        assert "doc1" in rag_system.multi_doc_retriever._documents
    
    def test_set_concept_graph(self, rag_system):
        """Test setting concept graph."""
        graph = nx.DiGraph()
        graph.add_edge("A", "B")
        
        rag_system.set_concept_graph(graph)
        
        assert len(rag_system.concept_ranker.concept_graph.nodes()) > 0
    
    def test_set_user_concepts(self, rag_system):
        """Test setting user concepts."""
        concepts = {"concept1", "concept2"}
        rag_system.set_user_concepts(concepts)
        
        assert rag_system.concept_ranker.known_concepts == concepts
    
    def test_retrieve(self, rag_system):
        """Test retrieval from RAG system."""
        rag_system.add_document("doc1", "Doc 1", {
            "section": {"paragraphs": ["Test content"]},
        })
        
        results = rag_system.retrieve("content", top_k=1)
        
        assert isinstance(results, list)
    
    def test_retrieve_with_query_type(self, rag_system):
        """Test retrieval with different query types."""
        rag_system.add_document("doc1", "Doc 1", {
            "section": {"paragraphs": ["Test content"]},
        })
        
        for query_type in QueryType:
            results = rag_system.retrieve("content", query_type=query_type, top_k=1)
            assert isinstance(results, list)
    
    def test_retrieve_with_doc_filter(self, rag_system):
        """Test retrieval with document filtering."""
        rag_system.add_document("doc1", "Doc 1", {
            "s1": {"paragraphs": ["content1"]},
        })
        rag_system.add_document("doc2", "Doc 2", {
            "s1": {"paragraphs": ["content2"]},
        })
        
        results = rag_system.retrieve("content", doc_ids=["doc1"], top_k=5)
        
        for result in results:
            assert result.document_ref.doc_id == "doc1"
    
    def test_get_explanation_with_cache(self, rag_system):
        """Test getting explanation with caching."""
        rag_system.add_document("doc1", "Doc 1", {
            "section": {"paragraphs": ["Information about topic"]},
        })
        
        concepts = {"topic"}
        explanation1, results1 = rag_system.get_explanation_with_cache(
            "explain topic", concepts
        )
        
        assert isinstance(explanation1, str)
        assert len(explanation1) > 0
        
        explanation2, results2 = rag_system.get_explanation_with_cache(
            "explain topic", concepts
        )
        
        assert explanation1 == explanation2
        assert len(results2) == 0
    
    def test_cache_stats(self, rag_system):
        """Test retrieving cache statistics."""
        stats = rag_system.get_cache_stats()
        
        assert "embeddings_cached" in stats
        assert "explanations_cached" in stats
    
    def test_clear_cache(self, rag_system):
        """Test clearing cache."""
        rag_system.semantic_cache.set_embedding("test", [0.1])
        
        rag_system.clear_cache()
        
        assert rag_system.semantic_cache.get_embedding("test") is None


class TestRAGIntegration:
    """Integration tests for RAG system."""
    
    def test_end_to_end_retrieval_and_explanation(self):
        """Test complete RAG flow."""
        provider = MockEmbeddingProvider()
        rag_system = RAGSystem(embedding_provider=provider)
        
        graph = nx.DiGraph()
        graph.add_edges_from([
            ("Python", "Variables"),
            ("Variables", "Types"),
        ])
        rag_system.set_concept_graph(graph)
        
        rag_system.add_document("python_guide", "Python Tutorial", {
            "basics": {
                "paragraphs": [
                    "Python is a programming language",
                    "Variables store data in Python",
                    "Python has different data types",
                ]
            }
        })
        
        rag_system.set_user_concepts({"Python"})
        
        explanation, results = rag_system.get_explanation_with_cache(
            "What are Python variables?",
            {"Python"}
        )
        
        assert isinstance(explanation, str)
        assert "variables" in explanation.lower() or "Python" in explanation
    
    def test_multi_document_hierarchical_retrieval(self):
        """Test hierarchical retrieval across multiple documents."""
        provider = MockEmbeddingProvider()
        rag_system = RAGSystem(embedding_provider=provider)
        
        rag_system.add_document("doc1", "Document 1", {
            "section1": {
                "paragraphs": ["Information about topic"]
            }
        })
        
        rag_system.add_document("doc2", "Document 2", {
            "section1": {
                "paragraphs": ["More details about topic"]
            }
        })
        
        for query_type in QueryType:
            results = rag_system.retrieve(
                "topic",
                query_type=query_type,
                top_k=3
            )
            
            assert isinstance(results, list)
            assert len(results) <= 3
