import os
import json
import pickle
from typing import Dict, Any, Optional, List
from ...core.document import Document, DocumentTree, DocumentChunk
from .base import BaseDocumentRepository, BaseDocumentCache

from .serializers import DocumentSerializer, TreeSerializer


class DocumentCache(BaseDocumentCache):
    """In-memory cache for documents"""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        return self._cache.get(key)
    
    def set(self, key: str, value: Any):
        """Set item in cache"""
        self._cache[key] = value
    
    def has(self, key: str) -> bool:
        """Check if key exists in cache"""
        return key in self._cache
    
    def delete(self, key: str) -> bool:
        """Delete item from cache"""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self):
        """Clear cache"""
        self._cache.clear()

class DocumentRepository(BaseDocumentRepository):
    """File-based document repository"""
    
    def __init__(self, storage_path: str = "data/documents/"):
        self.storage_path = storage_path
        self.cache = DocumentCache()
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure storage directories exist"""
        os.makedirs(self.storage_path, exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, "documents"), exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, "trees"), exist_ok=True)
    
    def _get_document_path(self, doc_id: str) -> str:
        """Get document file path"""
        return os.path.join(self.storage_path, "documents", f"{doc_id}.json")
    
    def _get_tree_path(self, doc_id: str) -> str:
        """Get tree file path"""
        return os.path.join(self.storage_path, "trees", f"{doc_id}.pkl")
    
    def save_chunk(self, chunk: DocumentChunk, chunk_id: str) -> str:
        """Save document chunk"""
        filepath = os.path.join(self.storage_path, "chunks", f"{chunk_id}.json")
        
        with open(filepath, 'w') as f:
            json.dump({
                'text': chunk.text,
                'summary': getattr(chunk, 'summary', ''),
                'metadata': getattr(chunk, 'metadata', {})
            }, f, indent=2)
        
        return filepath
    
    def save_document(self, document: Document, doc_id: str) -> bool:
        """Save document to JSON"""
        try:
            filepath = self._get_document_path(doc_id)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(DocumentSerializer.serialize(document), f, indent=2)
            
            self.cache.set(f"doc_{doc_id}", document)
            return True
        except Exception as e:
            print(f"Error saving document: {e}")
            return False
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID"""
        # Check cache first
        cached = self.cache.get(f"doc_{doc_id}")
        if cached:
            return cached
        
        # Load from file
        filepath = self._get_document_path(doc_id)
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                document = DocumentSerializer.deserialize(data)
                self.cache.set(f"doc_{doc_id}", document)
                return document
        except Exception as e:
            print(f"Error loading document: {e}")
            return None
    
    def save_tree(self, tree: DocumentTree, doc_id: str) -> bool:
        """Save document tree"""
        try:
            filepath = self._get_tree_path(doc_id)
            with open(filepath, 'wb') as f:
                pickle.dump(TreeSerializer.serialize(tree), f)
            
            self.cache.set(f"tree_{doc_id}", tree)
            return True
        except Exception as e:
            print(f"Error saving tree: {e}")
            return False
    
    def get_tree(self, doc_id: str) -> Optional[DocumentTree]:
        """Get document tree by ID"""
        # Check cache first
        cached = self.cache.get(f"tree_{doc_id}")
        if cached:
            return cached
        
        # Load from file
        filepath = self._get_tree_path(doc_id)
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'rb') as f:
                tree = TreeSerializer.deserialize(pickle.load(f))
                self.cache.set(f"tree_{doc_id}", tree)
                return tree
        except Exception as e:
            print(f"Error loading tree: {e}")
            return None
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document and its tree"""
        doc_path = self._get_document_path(doc_id)
        tree_path = self._get_tree_path(doc_id)
        
        deleted = False
        
        if os.path.exists(doc_path):
            os.remove(doc_path)
            deleted = True
        
        if os.path.exists(tree_path):
            os.remove(tree_path)
            deleted = True
        
        # Clear cache
        self.cache.delete(f"doc_{doc_id}")
        self.cache.delete(f"tree_{doc_id}")
        
        return deleted
    
    def list_documents(self) -> List[str]:
        """List all document IDs"""
        docs = []
        doc_dir = os.path.join(self.storage_path, "documents")
        
        if os.path.exists(doc_dir):
            for filename in os.listdir(doc_dir):
                if filename.endswith('.json'):
                    docs.append(filename.replace('.json', ''))
        
        return docs