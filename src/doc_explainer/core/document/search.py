from typing import List, Dict, Any, Optional
from .models.tree import DocumentTree


class SearchService:
    """Search within document trees"""
    
    def __init__(self, vector_db=None):
        self.vector_db = vector_db
    
    def search(self, query: str, tree: Optional[DocumentTree] = None,
               level: str = "paragraph", k: int = 3) -> List[Dict[str, Any]]:
        """
        Search within document
        
        If vector_db is available, use semantic search.
        Otherwise, fall back to keyword search.
        """
        if self.vector_db:
            return self._semantic_search(query, level, k)
        elif tree:
            return self._keyword_search(query, tree, level, k)
        else:
            return []
    
    def _semantic_search(self, query: str, level: str, k: int) -> List[Dict[str, Any]]:
        """Perform semantic search using vector DB"""
        if not self.vector_db:
            return []

        if not hasattr(self.vector_db, "similarity_search_with_score"):
            return []

        results = self.vector_db.similarity_search_with_score(
            query,
            k=k,
            filter={"level": level} if level else None
        )
        
        return [
            {
                "content": doc.page_content,
                "score": score,
                "metadata": doc.metadata
            }
            for doc, score in results
        ]
    
    def _keyword_search(self, query: str, tree: DocumentTree,
                        level: str, k: int) -> List[Dict[str, Any]]:
        """Perform simple keyword search"""
        query = query.lower()
        results = []
        
        chunks = tree.hierarchy.get(f"{level}s", [])
        for chunk in chunks:
            if query in chunk.text.lower():
                results.append({
                    "content": chunk.text,
                    "score": 1.0,
                    "metadata": {
                        "chunk_id": chunk.chunk_id,
                        "level": level
                    }
                })
        
        return results[:k]
    
    def set_vector_db(self, vector_db):
        """Set vector database for semantic search"""
        self.vector_db = vector_db