from typing import Dict, Any
from src.core.document import DocumentTree, DocumentNode, DocumentChunk


class DocumentSerializer:
    """Serializer for document objects"""
    
    @staticmethod
    def serialize_chunk(chunk: DocumentChunk) -> Dict[str, Any]:
        """Serialize document chunk"""
        return {
            'text': chunk.text,
            'summary': getattr(chunk, 'summary', ''),
            'metadata': getattr(chunk, 'metadata', {})
        }
    
    @staticmethod
    def deserialize_chunk(data: Dict[str, Any]) -> DocumentChunk:
        """Deserialize document chunk"""
        chunk = DocumentChunk(text=data['text'])
        chunk.summary = data.get('summary', '')
        chunk.metadata = data.get('metadata', {})
        return chunk


class DocumentTreeSerializer:
    """Serializer for document tree"""
    
    @staticmethod
    def serialize_node(node: DocumentNode) -> Dict[str, Any]:
        """Serialize document node"""
        if node is None:
            return None
        
        return {
            'id': node.id,
            'chunk': DocumentSerializer.serialize_chunk(node.chunk),
            'children': {str(k): DocumentTreeSerializer.serialize_node(v) 
                        for k, v in node.children.items()}
        }
    
    @staticmethod
    def deserialize_node(data: Dict[str, Any]) -> DocumentNode:
        """Deserialize document node"""
        if data is None:
            return None
        
        chunk = DocumentSerializer.deserialize_chunk(data['chunk'])
        node = DocumentNode(id=data['id'], chunk=chunk)
        
        for k, child_data in data.get('children', {}).items():
            node.children[int(k)] = DocumentTreeSerializer.deserialize_node(child_data)
        
        return node
    
    @staticmethod
    def serialize_tree(tree: DocumentTree) -> Dict[str, Any]:
        """Serialize document tree"""
        return {
            'title': tree.title,
            'root': DocumentTreeSerializer.serialize_node(tree.root)
        }
    
    @staticmethod
    def deserialize_tree(data: Dict[str, Any]) -> DocumentTree:
        """Deserialize document tree"""
        root = DocumentTreeSerializer.deserialize_node(data['root'])
        return DocumentTree(title=data['title'], root=root)