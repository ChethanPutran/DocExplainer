import spacy
from typing import Dict, List, Any

# Assuming these are defined in your src directory
from src.core.document.document_structures import (
    ChunkLevel,
    ChunkType,
    DocumentChunk,
    DocumentNode,
    DocumentTree,
    SimpleMetaDataCreator,
)
from src.core.document.document import Document
from src.core.document.document_manager import DocumentManager
from src.models.text import EmbeddingModel

class HierarchicalDocumentProcessor:
    """
    Processes nested Document dataclasses into a searchable DocumentTree.
    Handles recursive sections, paragraphs, and spaCy-based sentence splitting.
    """

    def __init__(self, doc_manager: DocumentManager, embedding_model=EmbeddingModel.DEFAULT_MODEL_NAME):
        self.doc_manager = doc_manager
        self.embedding_model = EmbeddingModel(embedding_model)
        
        # Initialize SpaCy for accurate sentence splitting (abbreviation-aware)
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")
        except Exception:
            self.nlp = spacy.blank("en")
            self.nlp.add_pipe("sentencizer")

        self.current_doc_id: int = 0
        self.document_trees: Dict[int, DocumentTree] = {}


    def process_document(self,doc_id: int) -> DocumentTree:

        document = self.doc_manager.get_document(doc_id)
        """Main entry point: Converts Document dataclass to a DocumentTree."""
        doc_text = document.raw_text
        
        # 1. Create the Root (Document Level)
        doc_chunk = DocumentChunk(
            text=doc_text,
            chunk_type=ChunkType.DOCUMENT,
            level=ChunkLevel.DOCUMENT,
            chunk_id=document.doc_id,
            metadata=SimpleMetaDataCreator().create_metadata(
                length=len(doc_text), start=0, end=len(doc_text), page=1
            ),
        )
        root_node = DocumentNode(document.doc_id,chunk=doc_chunk)
        tree = DocumentTree(title=document.get_title(), root=root_node)
        
        # Internal state for flat indexing
        self._section_chunks, self._paragraph_chunks, self._sentence_chunks = [], [], []

        # 2. Recursively process nested sections
        for section_idx,section in enumerate(document.sections):
            self._process_recursive(section, tree.root, root_node.id)

        # 3. Batch process embeddings for all nodes
        all_chunks = [doc_chunk] + self._section_chunks + self._paragraph_chunks + self._sentence_chunks
        for chunk in all_chunks:
            if len(chunk.text.strip()) > 10:
                chunk.embedding = self.embedding_model.encode(chunk.text)

        # 4. Finalize tree structure
        tree.set_chunks(doc_chunk, self._section_chunks, self._paragraph_chunks, self._sentence_chunks)
        self.document_trees[document.doc_id] = tree
        return tree

    def _process_recursive(self, section: Any, parent_node: DocumentNode, parent_id: int):
        """Walks the Section -> Paragraph -> Sentence hierarchy."""
        
        # Create Section Chunk
        sec_chunk = DocumentChunk(
            text=section.raw_text,
            chunk_type=ChunkType.SECTION,
            level=ChunkLevel.SECTION,
            chunk_id=section.sec_id,
            parent_id=parent_id,
            metadata=SimpleMetaDataCreator().create_metadata(
                length=len(section.raw_text), 
                start=getattr(section, 'start', 0),
                end=getattr(section, 'end', 0),
                page=section.page_start
            ),
        )
        sec_node = DocumentNode(section.sec_id, chunk=sec_chunk)
        parent_node.children[sec_node.id] = sec_node
        self._section_chunks.append(sec_chunk)

        # Process Paragraphs in this Section
        for para in section.paragraphs:
            para_chunk = DocumentChunk(
                text=para.raw_text,
                chunk_type=ChunkType.PARAGRAPH,
                level=ChunkLevel.PARAGRAPH,
                chunk_id=para.para_id,
                parent_id=sec_chunk.chunk_id,
                metadata=SimpleMetaDataCreator().create_metadata(
                    length=len(para.raw_text), start=para.start, end=para.end, page=para.page
                ),
            )
            para_node = DocumentNode(para.para_id,chunk=para_chunk)
            sec_node.children[para_node.id] = para_node
            self._paragraph_chunks.append(para_chunk)

            # Process Sentences using the already-split sentences in the dataclass
            for sent in para.sentences:
                sent_chunk = DocumentChunk(
                    text=sent.raw_text,
                    chunk_type=ChunkType.SENTENCE,
                    level=ChunkLevel.SENTENCE,
                    chunk_id=sent.sen_id,
                    parent_id=para_chunk.chunk_id,
                    metadata=SimpleMetaDataCreator().create_metadata(
                        length=len(sent.raw_text), 
                        start=sent.start, 
                        end=sent.end, 
                        page=sent.page,
                        is_concept=self._is_concept_sentence(sent.raw_text)
                    ),
                )
                sent_node = DocumentNode(sent.sen_id,chunk=sent_chunk)
                para_node.children[sent_node.id] = sent_node
                self._sentence_chunks.append(sent_chunk)

        # Handle Subsections (Infinite Recursion)
        for sub in getattr(section, 'subsections', []):
            self._process_recursive(sub, sec_node, sec_chunk.chunk_id)

    def find_chunks_by_page(self, doc_id: int, page: int) -> List[DocumentChunk]:
        """deterministic retrieval for all chunks on a specific page."""
        tree = self.document_trees.get(doc_id)
        if not tree: return []
        
        # Search the flat indices for matches
        all_flattened = [tree.root.chunk] + tree.hierarchy['sections'] + tree.hierarchy['paragraphs'] + tree.hierarchy['sentences']
        return [c for c in all_flattened if c.metadata and getattr(c.metadata, 'page', None) == page]

    def _is_concept_sentence(self, text: str) -> bool:
        markers = ["defined as", "refers to", "known as", "is called"]
        return any(m in text.lower() for m in markers)