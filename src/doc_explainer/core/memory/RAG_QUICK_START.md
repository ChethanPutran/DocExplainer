"""
RAG System Quick Reference Guide

This file provides quick examples for common RAG system usage patterns.
"""

## Quick Start

### Basic Setup
```python
from src.core.memory import RAGSystem, QueryType
import networkx as nx

# Create system
rag = RAGSystem()

# Add documents
rag.add_document("doc1", "Python Guide", {
    "intro": {"paragraphs": ["Python is..."]},
    "basics": {"paragraphs": ["Variables...", "Functions..."]},
})

# Retrieve
results = rag.retrieve("What are functions?", top_k=3)
```

### With Concept Graph
```python
# Create concept graph
graph = nx.DiGraph()
graph.add_edges_from([
    ("Python", "Variables"),
    ("Variables", "Types"),
    ("Python", "Functions"),
])

# Set graph and user knowledge
rag.set_concept_graph(graph)
rag.set_user_concepts({"Python"})

# Retrieve with concept ranking
results = rag.retrieve(
    "How do variables work?",
    query_type=QueryType.EXPLANATION,
    use_concept_ranking=True
)
```

## API Reference

### RAGSystem

**Initialization:**
```python
rag = RAGSystem(
    embedding_provider=None,      # Uses SentenceTransformer by default
    concept_graph=None,           # Optional NetworkX graph
    max_cached_embeddings=1000,   # LRU cache size
    max_cached_explanations=500   # LRU cache size
)
```

**Methods:**
```python
# Add document to system
rag.add_document(doc_id, doc_name, content_dict)

# Set concept graph for ranking
rag.set_concept_graph(networkx_digraph)

# Set user's known concepts
rag.set_user_concepts({"concept1", "concept2"})

# Retrieve documents with all enhancements
results = rag.retrieve(
    query="user query",
    query_type=QueryType.EXPLANATION,  # Optional
    doc_ids=["doc1", "doc2"],         # Optional
    top_k=5,                           # Number of results
    use_concept_ranking=True           # Use concept graph
)

# Get cached explanation
explanation, sources = rag.get_explanation_with_cache(
    query="What is X?",
    user_concepts={"Python"},
    doc_ids=None,
    top_k=3
)

# Get cache statistics
stats = rag.get_cache_stats()

# Clear cache
rag.clear_cache()
```

### Query Types

```python
QueryType.DEFINITION      # Concise definitions (1x chunk)
QueryType.EXPLANATION     # Detailed explanations (2x chunk)
QueryType.COMPARISON      # Comparative analysis (3x chunk)
QueryType.PREREQUISITE    # Prerequisites (2x chunk)
QueryType.EXAMPLE         # Examples/use cases (2x chunk)
```

### RetrievalResult

**Properties:**
```python
result.content                    # Retrieved text
result.document_ref.doc_id        # Document ID
result.document_ref.section       # Section name
result.relevance_score            # Semantic similarity [0-1]
result.concept_relevance_score    # Concept ranking [0-1]
result.combined_score             # Final ranking score [0-1]
result.contains_prerequisites     # Has prerequisite concepts
result.contains_unknown_concepts  # Has unknown concepts
```

## Common Patterns

### Multi-Document Search
```python
# Add multiple documents
rag.add_document("doc1", "Python Basics", content1)
rag.add_document("doc2", "Python Advanced", content2)
rag.add_document("doc3", "JavaScript", content3)

# Search across all documents
all_results = rag.retrieve("functions", top_k=5)

# Search specific documents
python_results = rag.retrieve(
    "functions",
    doc_ids=["doc1", "doc2"],
    top_k=5
)
```

### Context-Aware Caching
```python
# Cache respects user knowledge state
# These are different cache entries:
explanation1 = rag.get_explanation_with_cache(
    "What is X?",
    user_concepts={"A", "B"}
)

explanation2 = rag.get_explanation_with_cache(
    "What is X?",
    user_concepts={"C", "D"}
)

# These are the same cache entry (retrieved from cache):
explanation3 = rag.get_explanation_with_cache(
    "What is X?",
    user_concepts={"A", "B"}
)
```

### Query Type Adaptation
```python
# Different query types adjust chunk sizes
results = []

for qtype in [QueryType.DEFINITION, QueryType.EXPLANATION]:
    r = rag.retrieve(
        "machine learning",
        query_type=qtype
    )
    results.append(r)

# DEFINITION → shorter, concise results
# EXPLANATION → longer, detailed results
```

### Concept Graph Integration
```python
import networkx as nx
from pathlib import Path

# Load or create graph
graph = nx.read_gexf("concept_graph.gexf")  # Load existing
# or
graph = nx.DiGraph()
graph.add_edges_from([
    ("Object-Oriented", "Classes"),
    ("Classes", "Inheritance"),
    ("Classes", "Polymorphism"),
])

# Use with RAG
rag.set_concept_graph(graph)

# User learns a concept
rag.set_user_concepts({"Object-Oriented"})

# Results now boosted if they contain prerequisites
results = rag.retrieve(
    "What is polymorphism?",
    use_concept_ranking=True
)
```

### Ranking Analysis
```python
# See which factors affected ranking
for i, result in enumerate(results, 1):
    print(f"\n{i}. {result.document_ref.doc_name}")
    print(f"   Content: {result.content[:50]}...")
    print(f"   Relevance: {result.relevance_score:.2f}")
    print(f"   Concept Score: {result.concept_relevance_score:.2f}")
    print(f"   Combined: {result.combined_score:.2f}")
    print(f"   Prerequisites: {result.contains_prerequisites}")
    print(f"   Unknown concepts: {result.contains_unknown_concepts}")
```

## Advanced Usage

### Custom Embedding Provider
```python
from src.core.memory import EmbeddingProvider, RAGSystem

class CustomEmbeddingProvider(EmbeddingProvider):
    def embed(self, text):
        # Custom embedding logic
        return embedding_vector
    
    def embed_batch(self, texts):
        return [self.embed(text) for text in texts]

provider = CustomEmbeddingProvider()
rag = RAGSystem(embedding_provider=provider)
```

### Semantic Cache Statistics
```python
stats = rag.get_cache_stats()

print(f"Embeddings: {stats['embeddings_cached']}/{stats['embeddings_max']}")
print(f"Explanations: {stats['explanations_cached']}/{stats['explanations_max']}")
print(f"Avg. embedding accesses: {stats['avg_embedding_accesses']:.1f}")
print(f"Avg. explanation accesses: {stats['avg_explanation_accesses']:.1f}")
```

### Document Structure Examples
```python
# Detailed structure
document1 = {
    "Introduction": {
        "paragraphs": [
            "First paragraph of intro",
            "Second paragraph of intro"
        ]
    },
    "Concepts": {
        "paragraphs": [
            "Concept 1 explanation",
            "Concept 2 explanation"
        ]
    }
}

# Simple structure (auto-converted)
document2 = {
    "section1": ["paragraph1", "paragraph2"],
    "section2": "single paragraph"
}

# Mixed structure (supported)
document3 = {
    "overview": "Single paragraph overview",
    "details": ["point1", "point2", "point3"]
}

rag.add_document("doc1", "Detailed Doc", document1)
rag.add_document("doc2", "Simple Doc", document2)
rag.add_document("doc3", "Mixed Doc", document3)
```

## Performance Tips

1. **Batch Processing**: Add multiple documents at once
   ```python
   for doc in documents:
       rag.add_document(doc["id"], doc["name"], doc["content"])
   ```

2. **Concept Graph Reuse**: Create once, reuse with multiple RAG instances
   ```python
   graph = create_concept_graph()  # Expensive
   rag1 = RAGSystem(concept_graph=graph)
   rag2 = RAGSystem(concept_graph=graph)
   ```

3. **Cache Optimization**: Monitor cache stats
   ```python
   stats = rag.get_cache_stats()
   if stats['embeddings_cached'] > 0.9 * stats['embeddings_max']:
       rag.clear_cache()  # Clear if getting full
   ```

4. **ChromaDB**: Use for large document collections
   - Automatically uses ChromaDB if available
   - Falls back to simple similarity if unavailable
   - Persistent storage in `./chroma_data`

## Troubleshooting

### No results returned
- Check document format (must have hierarchical structure)
- Verify query isn't too specific
- Try different `top_k` value
- Check if documents were actually added

### Rankings seem off
- Ensure concept graph is set up correctly
- Check user concepts are set: `rag.set_user_concepts(...)`
- Verify concept names match graph nodes (case-sensitive)

### Cache not working
- Concept sets must match exactly (order doesn't matter)
- Different query + concept combinations = different cache entries
- Check cache stats: `rag.get_cache_stats()`

### Performance issues
- ChromaDB takes time on first use (indexing)
- Large embeddings cache slows things down
- Consider reducing `max_cached_embeddings`

## See Also

- `src/core/memory/RAG_DOCUMENTATION.md` - Full documentation
- `src/core/memory/tests/test_rag.py` - Test examples
- `src/core/memory/rag.py` - Source code and docstrings
