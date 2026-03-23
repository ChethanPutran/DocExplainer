from ..models import DocumentChunk, ChunkType, ChunkLevel, DocumentNode, SimpleMetadataCreator,DocumentTree

def create_empty_tree(title: str) -> DocumentTree:
    """Create an empty document tree"""
    
    
    metadata_creator = SimpleMetadataCreator()
    chunk = DocumentChunk(
        text=title,
        chunk_type=ChunkType.DOCUMENT,
        level=ChunkLevel.DOCUMENT,
        metadata=metadata_creator.create_metadata(length=len(title))
    )
    root = DocumentNode("root", chunk)
    return DocumentTree(title, root)
