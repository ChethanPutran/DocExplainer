from typing import Optional, List
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document as LCDocument

from ..models import Document, DocumentTree
from .strategies import SummaryGenerator, HierarchyBuilder
from .base import DocumentBuilder


class LangChainEmbeddingWrapper(Embeddings):
    """Wraps embedding model for LangChain compatibility"""
    
    def __init__(self, model):
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(t).tolist() for t in texts]
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()


class HierarchicalProcessor(DocumentBuilder):
    """Processes documents hierarchically with summaries"""
    
    def __init__(self, llm_wrapper=None, embedding_model=None):
        self.summary_generator = SummaryGenerator(llm_wrapper)
        self.hierarchy_builder = HierarchyBuilder(self.summary_generator)
        self.embedding_model = embedding_model
        
        # Wrap for LangChain if embedding model provided
        self.langchain_embeddings = None
        if embedding_model:
            self.langchain_embeddings = LangChainEmbeddingWrapper(embedding_model)
    
    def build_tree(self, document: Document, target_section: Optional[str] = None) -> DocumentTree:
        """Build document tree with summaries"""
        tree = self.hierarchy_builder.build(document, target_section)
        return tree
    
    def create_full_vector_db(self, document: Document, collection_name: str = "full_doc",
                              persist_directory: Optional[str] = None):
        """Create full-document vector database"""
        if not self.langchain_embeddings:
            raise ValueError("Embedding model required for vector DB creation")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        
        # Get all text
        all_text = "\n".join(list(document.get_text_generator()))
        texts = text_splitter.split_text(all_text)
        
        return Chroma.from_texts(
            texts=texts,
            embedding=self.langchain_embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
    
    def create_tree_aware_db(self, tree: DocumentTree, collection_name: str = "hierarchical_db",
                            persist_directory: Optional[str] = None):
        """Create vector database from tree chunks"""
        if not self.langchain_embeddings:
            raise ValueError("Embedding model required for vector DB creation")
        
        lc_docs = []
        
        # Add paragraphs
        for para_chunk in tree.hierarchy.get('paragraphs', []):
            metadata = {
                "chunk_id": para_chunk.chunk_id,
                "parent_id": para_chunk.parent_id or "",
                "level": "paragraph",
                "page": para_chunk.metadata.page if para_chunk.metadata else 0,
                "type": "summary"
            }
            
            lc_docs.append(LCDocument(
                page_content=para_chunk.text,
                metadata=metadata
            ))
        
        # Add sentences for granular search
        for sent_chunk in tree.hierarchy.get('sentences', []):
            lc_docs.append(LCDocument(
                page_content=sent_chunk.text,
                metadata={
                    "chunk_id": sent_chunk.chunk_id,
                    "parent_id": sent_chunk.parent_id or "",
                    "level": "sentence",
                    "type": "raw_text"
                }
            ))
        
        return Chroma.from_documents(
            documents=lc_docs,
            embedding=self.langchain_embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
    
    def visualize_tree(self, node, indent: str = "", is_last: bool = True):
        """Visualize document tree"""
        marker = "└── " if is_last else "├── "
        
        chunk = node.chunk
        display_text = ""
        
        if chunk.chunk_type.value == 1:  # DOCUMENT
            display_text = f"📄 DOCUMENT: {chunk.text[:100]}..."
        elif chunk.chunk_type.value == 2:  # SECTION
            display_text = f"📂 SECTION: {chunk.text[:100]}..."
        elif chunk.chunk_type.value == 3:  # PARAGRAPH
            summary_snippet = chunk.text[:75] + '...' if len(chunk.text) > 75 else chunk.text
            display_text = f"📝 PARA [Summary]: {summary_snippet}"
        else:  # SENTENCE
            return
        
        print(f"{indent}{marker}{display_text}")
        
        new_indent = indent + ("    " if is_last else "│   ")
        children = list(node.children.values())
        
        for i, child in enumerate(children):
            self.visualize_tree(child, new_indent, is_last=(i == len(children) - 1))