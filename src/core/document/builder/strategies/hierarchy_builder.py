from typing import Optional, List, Tuple
from ...models.structure import Document, Section
from ...models.tree import DocumentTree, DocumentNode, DocumentChunk, ChunkType, ChunkLevel
from ...models.metadata import SimpleMetadataCreator
from .summary_generator import SummaryGenerator


class HierarchyBuilder:
    """Builds hierarchical document tree"""
    
    def __init__(self, summary_generator: SummaryGenerator):
        self.summary_generator = summary_generator
        self.metadata_creator = SimpleMetadataCreator()
        self.section_chunks = []
        self.paragraph_chunks = []
        self.sentence_chunks = []
        self.section_summaries = []
    
    def build(self, document: Document, target_section: Optional[str] = None) -> DocumentTree:
        """Build document tree"""
        self._reset()
        
        # Create root node
        doc_chunk = self._create_document_chunk(document)
        root_node = DocumentNode("root", doc_chunk)
        tree = DocumentTree(title=document.title, root=root_node)
        
        # Initialize context with document title
        self.summary_generator.reset_context()
        self.summary_generator.update_context(f"Document Title: {document.title}\n")
        
        # Process sections recursively
        stop_processing = False
        for section in document.sections:
            if stop_processing:
                break
            stop_processing = self._process_section(
                section, root_node, target_section
            )
        
        # Generate document summary from section summaries
        if self.section_summaries:
            doc_summary = self.summary_generator.generate_summary(
                " ".join(self.section_summaries)
            )
            doc_chunk.summary = doc_summary
        
        # Set chunks in tree
        tree.set_chunks(
            doc_chunk,
            self.section_chunks,
            self.paragraph_chunks,
            self.sentence_chunks
        )
        
        return tree
    
    def _reset(self):
        """Reset builder state"""
        self.section_chunks = []
        self.paragraph_chunks = []
        self.sentence_chunks = []
        self.section_summaries = []
    
    def _create_document_chunk(self, document: Document) -> DocumentChunk:
        """Create document-level chunk"""
        metadata = self.metadata_creator.create_metadata(
            length=sum(len(p.text) for p in document.get_all_paragraphs())
        )
        return DocumentChunk(
            text=document.title,
            chunk_type=ChunkType.DOCUMENT,
            level=ChunkLevel.DOCUMENT,
            metadata=metadata
        )
    
    def _process_section(self, section: Section, parent_node: DocumentNode,
                        target_section: Optional[str], depth: int = 0) -> bool:
        """
        Process a section recursively
        
        Returns:
            True if processing should stop after this section
        """
        # Check if this is the target section
        is_target = False
        if target_section and section.title.strip().lower() == target_section.strip().lower():
            is_target = True
        
        # Update context
        self.summary_generator.update_context(f"\n--- Section: {section.title} ---\n")
        
        # Create section chunk
        section_chunk = self._create_section_chunk(section)
        section_node = DocumentNode(section.section_id, section_chunk)
        parent_node.add_child(section_node)
        self.section_chunks.append(section_chunk)
        
        # Process paragraphs
        paragraph_summaries = []
        for paragraph in section.paragraphs:
            summary = self._process_paragraph(paragraph, section_node)
            paragraph_summaries.append(summary)
        
        # Generate section summary
        if paragraph_summaries:
            section_summary = self.summary_generator.generate_summary(
                " ".join(paragraph_summaries),
                self.summary_generator.rolling_context
            )
            section_chunk.summary = section_summary
            self.section_summaries.append(section_summary)
        
        # Process subsections
        for subsection in section.subsections:
            if self._process_section(subsection, section_node, target_section, depth + 1):
                is_target = True
        
        return is_target
    
    def _create_section_chunk(self, section: Section) -> DocumentChunk:
        """Create section-level chunk"""
        metadata = self.metadata_creator.create_metadata(
            length=len(section.text),
            page=section.page_start
        )
        return DocumentChunk(
            text=section.title,
            chunk_type=ChunkType.SECTION,
            level=ChunkLevel.SECTION,
            metadata=metadata
        )
    
    def _process_paragraph(self, paragraph, section_node) -> str:
        """Process a paragraph"""
        # Generate summary
        summary = self.summary_generator.generate_summary(
            paragraph.text,
            self.summary_generator.rolling_context
        )
        
        # Update context
        self.summary_generator.update_context(f"Para Summary: {summary}\n")
        
        # Create paragraph chunk
        para_chunk = self._create_paragraph_chunk(paragraph, summary)
        para_node = DocumentNode(paragraph.paragraph_id, para_chunk)
        section_node.add_child(para_node)
        self.paragraph_chunks.append(para_chunk)
        
        # Process sentences
        for sentence in paragraph.sentences:
            self._process_sentence(sentence, para_node)
        
        return summary
    
    def _create_paragraph_chunk(self, paragraph, summary: str) -> DocumentChunk:
        """Create paragraph-level chunk"""
        metadata = self.metadata_creator.create_metadata(
            length=len(paragraph.text),
            start=paragraph.start,
            end=paragraph.end,
            raw_text=paragraph.text,
            page=paragraph.page
        )
        return DocumentChunk(
            text=summary,
            summary=summary,
            chunk_type=ChunkType.PARAGRAPH,
            level=ChunkLevel.PARAGRAPH,
            metadata=metadata
        )
    
    def _process_sentence(self, sentence, para_node):
        """Process a sentence"""
        sent_chunk = DocumentChunk(
            text=sentence.text,
            chunk_type=ChunkType.SENTENCE,
            level=ChunkLevel.SENTENCE,
            metadata=self.metadata_creator.create_metadata(
                length=len(sentence.text),
                start=sentence.start,
                end=sentence.end,
                page=sentence.page
            )
        )
        sent_node = DocumentNode(sentence.sentence_id, sent_chunk)
        para_node.add_child(sent_node)
        self.sentence_chunks.append(sent_chunk)