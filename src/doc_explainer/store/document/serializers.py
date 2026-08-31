from typing import Dict, Any
from ...core.document import DocumentTree, DocumentNode, DocumentChunk

from typing import Dict, Any
from ...core.document import Document, Sentence, Paragraph, Image, Table, Equation


class DocumentSerializer:
    """Serializer for Document objects"""
    
    @staticmethod
    def serialize(document: Document) -> Dict[str, Any]:
        """Serialize document to dictionary"""
        return document.to_dict()
    
    @staticmethod
    def deserialize(data: Dict[str, Any]) -> Document:
        """Deserialize document from dictionary"""
        return Document.from_dict(data)
    
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


class TreeSerializer:
    """Serializer for DocumentTree objects"""
    @staticmethod
    def deserialize(data: Dict[str, Any]) -> DocumentTree:
        """Deserialize document tree from dictionary"""
        return DocumentTree.from_dict(data)
    
    @staticmethod
    def serialize(tree: DocumentTree) -> Dict[str, Any]:
        """Serialize tree to dictionary"""
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
            'root': serialize_node(tree.root),
            "total_chunks": tree.total_chunks,
            # "hierarchy": {
            #     level: [chunk.chunk_id for chunk in chunks]
            #     for level, chunks in tree.hierarchy.items()
            # }
        }
    

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
        node = DocumentNode(node_id=data['id'], chunk=chunk)
        
        for k, child_data in data.get('children', {}).items():
            node.children[k] = DocumentTreeSerializer.deserialize_node(child_data)
        
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