import json
import os
import pickle
from typing import Optional, List
from src.core.document import DocumentTree, DocumentChunk


class DocumentRepository:
    """Repository for document persistence"""
    
    def __init__(self, storage_path: str = "data/documents/"):
        self.storage_path = storage_path
        self._ensure_storage()
    
    def _ensure_storage(self):
        """Ensure storage directory exists"""
        os.makedirs(self.storage_path, exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, "trees"), exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, "chunks"), exist_ok=True)
    
    def save_document_tree(self, tree: DocumentTree, doc_id: str) -> str:
        """Save document tree"""
        # Save as pickle
        pickle_path = os.path.join(self.storage_path, "trees", f"{doc_id}.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(tree, f)
        
        # Save as JSON for inspection
        json_path = os.path.join(self.storage_path, "trees", f"{doc_id}.json")
        with open(json_path, 'w') as f:
            json.dump(self._serialize_tree(tree), f, indent=2)
        
        return pickle_path
    
    def load_document_tree(self, doc_id: str) -> Optional[DocumentTree]:
        """Load document tree"""
        pickle_path = os.path.join(self.storage_path, "trees", f"{doc_id}.pkl")
        
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        
        return None
    
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
    
    def list_documents(self) -> List[str]:
        """List all document IDs"""
        docs = []
        
        for filename in os.listdir(os.path.join(self.storage_path, "trees")):
            if filename.endswith('.pkl'):
                docs.append(filename.replace('.pkl', ''))
        
        return docs
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document"""
        pickle_path = os.path.join(self.storage_path, "trees", f"{doc_id}.pkl")
        json_path = os.path.join(self.storage_path, "trees", f"{doc_id}.json")
        
        deleted = False
        
        if os.path.exists(pickle_path):
            os.remove(pickle_path)
            deleted = True
        
        if os.path.exists(json_path):
            os.remove(json_path)
            deleted = True
        
        return deleted
    
    def _serialize_tree(self, tree: DocumentTree) -> dict:
        """Serialize document tree for JSON"""
        def serialize_node(node):
            if node is None:
                return None
            
            return {
                'id': node.id,
                'chunk': {
                    'text': node.chunk.text[:100] + '...' if len(node.chunk.text) > 100 else node.chunk.text,
                    'summary': getattr(node.chunk, 'summary', '')[:100] if hasattr(node.chunk, 'summary') else ''
                },
                'children': {str(k): serialize_node(v) for k, v in node.children.items()}
            }
        
        return {
            'title': tree.title,
            'root': serialize_node(tree.root)
        }