"""
RAG System Documentation

This document provides an overview of the Retrieval-Augmented Generation (RAG) system
implemented in src/core/memory/rag.py
"""

## Overview

The RAG system is a comprehensive Retrieval-Augmented Generation implementation that combines:
1. **Semantic Caching** - LRU cache for embeddings and explanations
2. **Multi-Document Retrieval** - Search across multiple documents with context awareness
3. **Hierarchical Retrieval** - Progressive refinement from documents to paragraphs
4. **Concept-Graph Aware Ranking** - Rank results using knowledge graph relationships

## Architecture

### Core Components

#### 1. SemanticCache
Implements LRU (Least Recently Used) caching for embeddings and explanations.

**Features:**
- Caches embeddings with MD5 hashing
- Context-aware explanation caching (considers user's known concepts)
- LRU eviction when capacity is exceeded
- Access tracking and statistics

**Usage:**
```python
cache = SemanticCache(max_embeddings=1000, max_explanations=500)
cache.set_embedding("text", embedding_vector)
retrieved = cache.get_embedding("text")
stats = cache.get_stats()
```

#### 2. EmbeddingProvider
Abstract base class for embedding generation.

**Implementations:**
- `SentenceTransformerProvider` - Uses sentence-transformers library
- Custom providers can extend `EmbeddingProvider`

**Usage:**
```python
provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
embedding = provider.embed("text")
embeddings = provider.embed_batch(["text1", "text2"])
```

#### 3. MultiDocumentRetriever
Retrieves content from multiple documents with document-context awareness.

**Features:**
- Supports ChromaDB for efficient similarity search
- Fallback to simple cosine similarity if ChromaDB unavailable
- Hierarchical document structure support (sections, paragraphs)
- Document-level filtering

**Usage:**
```python
retriever = MultiDocumentRetriever(embedding_provider)
retriever.add_document("doc1", "Document Name", {
    "section1": {"paragraphs": ["para1", "para2"]},
    "section2": {"paragraphs": ["para3"]},
})
results = retriever.retrieve("query", doc_ids=["doc1"], top_k=5)
```

#### 4. HierarchicalRetriever
Implements hierarchical retrieval with adaptive chunk sizing based on query type.

**Query Types:**
- `DEFINITION` - Concise answers (1x chunk)
- `EXPLANATION` - Detailed explanations (2x chunks)
- `COMPARISON` - Comparative analysis (3x chunks)
- `PREREQUISITE` - Prerequisites (2x chunks)
- `EXAMPLE` - Examples and use cases (2x chunks)

**Usage:**
```python
h_retriever = HierarchicalRetriever(multi_doc_retriever)
results = h_retriever.retrieve_hierarchical(
    query="explain topic",
    query_type=QueryType.EXPLANATION,
    top_k=5
)
```

#### 5. ConceptGraphRanker
Ranks retrieval results using NetworkX concept graphs.

**Ranking Factors:**
1. **Prerequisite Boost** (+20%) - Results containing prerequisites of target concepts
2. **Unknown Concept Penalty** (-20%) - Results with unknown concepts
3. **Concept Relevance** - Computed from concept relationships

**Usage:**
```python
ranker = ConceptGraphRanker(concept_graph)
ranker.set_user_concepts({"concept1", "concept2"})
ranked_results = ranker.rank_results(unranked_results)
```

#### 6. RAGSystem
Main system combining all components.

**Usage:**
```python
rag = RAGSystem(
    embedding_provider=provider,
    concept_graph=graph,
    max_cached_embeddings=1000
)

# Add documents
rag.add_document("doc1", "Name", content_dict)

# Retrieve with all enhancements
results = rag.retrieve(
    "query",
    query_type=QueryType.EXPLANATION,
    top_k=5,
    use_concept_ranking=True
)

# Get explanation with caching
explanation, sources = rag.get_explanation_with_cache(
    "query",
    user_concepts={"concept1"},
    top_k=3
)
```

## Data Models

### DocumentReference
Represents a location in a document.

**Fields:**
- `doc_id` - Unique document ID
- `doc_name` - Human-readable document name
- `section` - Section within document
- `paragraph_idx` - Paragraph index
- `start_char`, `end_char` - Character positions

### RetrievalResult
Represents a retrieval result with ranking metadata.

**Fields:**
- `content` - Retrieved text
- `document_ref` - Location reference
- `relevance_score` - Semantic similarity score
- `concept_relevance_score` - Concept graph score
- `combined_score` - Final ranking score (property)
- `contains_prerequisites` - Has prerequisite concepts
- `contains_unknown_concepts` - Has unknown concepts

### CachedEmbedding / CachedExplanation
Cached items with metadata (creation time, access count, last accessed).

## Query Type Adaptation

The system adjusts retrieval behavior based on query type:

```python
QueryType.DEFINITION       → 1x chunk size (concise)
QueryType.EXPLANATION      → 2x chunk size (detailed)
QueryType.COMPARISON       → 3x chunk size (extensive)
QueryType.PREREQUISITE     → 2x chunk size
QueryType.EXAMPLE          → 2x chunk size
```

## Document Structure Format

Documents should follow this hierarchical structure:

```python
{
    "section_name": {
        "paragraphs": [
            "paragraph 1 text",
            "paragraph 2 text",
            ...
        ]
    },
    "another_section": {
        "paragraphs": [...]
    }
}
```

Or simpler formats are auto-converted:

```python
{
    "section1": ["para1", "para2"],  # Direct list
    "section2": "single paragraph",   # String
}
```

## Scoring Mechanism

### Combined Score Calculation

```
combined_score = base_score * modifiers

Where:
  base_score = (relevance_score * 0.6) + (concept_score * 0.4)
  
  modifiers:
    × 1.2  if contains_prerequisites
    × 0.8  if contains_unknown_concepts
    × (1.0 / (1.0 + 0.1 * distance_to_query_doc))  for document distance
```

### Concept Relevance Score

```
score = 0.5 (base)
      + 0.3 * (prerequisites / total_concepts)
      - 0.2 * (unknown_concepts / total_concepts)
```

Clamped to [0.0, 1.0]

## Integration with LangChain

The RAG system is designed to integrate seamlessly with LangChain:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.core.memory import RAGSystem

# Create RAG system
rag = RAGSystem()

# Use LangChain to process documents
splitter = RecursiveCharacterTextSplitter()
documents = splitter.split_documents(langchain_docs)

# Add to RAG
for doc in documents:
    rag.add_document(doc.metadata["source"], doc.metadata["name"], {
        "content": [doc.page_content]
    })

# Use for retrieval
results = rag.retrieve("user query")
```

## Integration with Knowledge Graph

The system integrates with NetworkX concept graphs:

```python
import networkx as nx
from src.core.memory import RAGSystem

# Create or load concept graph
graph = nx.DiGraph()
graph.add_edges_from([
    ("Python", "Functions"),
    ("Functions", "Lambda"),
    ("Python", "Classes"),
])

# Use with RAG
rag = RAGSystem(concept_graph=graph)
rag.set_user_concepts({"Python"})
results = rag.retrieve("lambda functions", use_concept_ranking=True)
```

## Caching Strategy

### Embedding Cache
- **Key**: MD5 hash of input text
- **Value**: Embedding vector
- **Eviction**: LRU when exceeding max_embeddings
- **Stats**: Access count, creation time, last accessed time

### Explanation Cache
- **Key**: MD5 hash of (query + user_concepts)
- **Value**: Generated explanation text
- **Eviction**: LRU when exceeding max_explanations
- **Context-aware**: Different caches for different concept sets

## Performance Considerations

1. **ChromaDB Integration**: Uses cosine distance metric for efficient similarity search
2. **Batch Operations**: `embed_batch()` for efficient multi-text embedding
3. **LRU Eviction**: Automatically manages memory with OrderedDict
4. **Caching**: Two-level caching (embedding + explanation) reduces computation
5. **Fallback Strategy**: Gracefully degrades to simple similarity if ChromaDB unavailable

## Testing

Comprehensive test suite in `src/core/memory/tests/test_rag.py`:

- **DocumentReference**: Hash, equality, creation
- **RetrievalResult**: Score calculation, modifiers
- **SemanticCache**: Set/get, LRU eviction, context-awareness
- **MultiDocumentRetriever**: Single/multi-document retrieval, filtering
- **HierarchicalRetriever**: Query type adaptation
- **ConceptGraphRanker**: Concept extraction, ranking, prerequisites
- **RAGSystem**: Integration tests, end-to-end flows

Run tests with:
```bash
pytest src/core/memory/tests/test_rag.py -v
```

## Error Handling

The system includes graceful error handling:

1. **ChromaDB Unavailable**: Falls back to simple similarity
2. **Missing Concepts**: Handles unknown concepts gracefully
3. **Empty Results**: Returns empty list or default explanation
4. **Import Errors**: Makes chromadb and sentence-transformers optional

## Configuration

### RAGSystem Defaults
- `max_cached_embeddings`: 1000
- `max_cached_explanations`: 500
- Embedding model: "all-MiniLM-L6-v2"
- ChromaDB: DuckDB backend, persistent storage

### SemanticCache Defaults
- Maintains MRU (Most Recently Used) ordering
- Access tracking for statistics
- Automatic cleanup on eviction

## Future Enhancements

Potential improvements:
1. Distributed caching (Redis support)
2. Async retrieval operations
3. Custom ranking strategies
4. Query expansion/reformulation
5. Cross-lingual retrieval
6. Time-decay cache eviction
7. Concept embedding integration
