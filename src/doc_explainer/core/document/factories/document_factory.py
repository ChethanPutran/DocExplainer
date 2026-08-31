from typing import Optional, List, Dict, Any, Tuple
import uuid
import numpy as np

from doc_explainer.core.document.parser.base import DocumentParser

from ..processor import DocumentProcessor
from ..models import (
    Sentence, Paragraph, Image, Table, Equation, Section, Document,
    DocumentTree, DocumentNode, DocumentChunk, ChunkType, ChunkLevel, 
    Metadata, SimpleMetadataCreator)
from ..parser import PDFParser
from ..builder import DocumentEngine
from ..services import DocumentManager
from ..visualization import HTMLGenerator, ConsolePrinter
from ..processor.hierarchy import HierarchicalProcessor
from ....store.document.base import BaseDocumentRepository
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

class DocumentFactory:
    """Factory for creating document-related objects"""
    
    def __init__(self):
        self.metadata_creator = SimpleMetadataCreator()
    
    # ==========================================
    # Content Model Factories
    # ==========================================
    
    def create_sentence(self, text: str, 
                       start: int = 0,
                       end: int = 0,
                       page: int = 0,
                       bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
                       sentence_id: Optional[str] = None,
                       embeddings: Optional[Dict[str, List[float]]] = None) -> Sentence:
        """Create a sentence"""
        return Sentence(
            text=text,
            start=start,
            end=end,
            page=page,
            bbox=bbox,
            sentence_id=sentence_id or str(uuid.uuid4())[:8],
            embeddings=embeddings or {}
        )
    
    def create_paragraph(self, text: str,
                        sentences: Optional[List[Sentence]] = None,
                        start: int = 0,
                        end: int = 0,
                        page: int = 0,
                        bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
                        paragraph_id: Optional[str] = None,
                        embeddings: Optional[Dict[str, List[float]]] = None) -> Paragraph:
        """Create a paragraph"""
        return Paragraph(
            text=text,
            sentences=sentences or [],
            start=start,
            end=end,
            page=page,
            bbox=bbox,
            paragraph_id=paragraph_id or str(uuid.uuid4())[:8],
            embeddings=embeddings or {}
        )
    
    def create_image(self, image_path: str,
                    caption: Optional[List[Sentence]] = None,
                    page: int = 0,
                    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
                    image_id: Optional[str] = None,
                    clip_embedding: Optional[List[float]] = None) -> Image:
        """Create an image"""
        return Image(
            image_path=image_path,
            caption=caption,
            page=page,
            bbox=bbox,
            image_id=image_id or str(uuid.uuid4())[:8],
            clip_embedding=clip_embedding or []
        )
    
    def create_table(self, data: str,
                    text: str,
                    caption: Optional[List[Sentence]] = None,
                    start: int = 0,
                    end: int = 0,
                    page: int = 0,
                    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
                    table_id: Optional[str] = None) -> Table:
        """Create a table"""
        return Table(
            data=data,
            text=text,
            caption=caption,
            start=start,
            end=end,
            page=page,
            bbox=bbox,
            table_id=table_id or str(uuid.uuid4())[:8]
        )
    
    def create_equation(self, text: str,
                       caption: Optional[List[Sentence]] = None,
                       start: int = 0,
                       end: int = 0,
                       page: int = 0,
                       bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
                       equation_id: Optional[str] = None) -> Equation:
        """Create an equation"""
        return Equation(
            text=text,
            caption=caption,
            start=start,
            end=end,
            page=page,
            bbox=bbox,
            equation_id=equation_id or str(uuid.uuid4())[:8]
        )
    
    # ==========================================
    # Structure Model Factories
    # ==========================================
    
    def create_section(self, title: str,
                      text: str = "",
                      page_start: int = 1,
                      paragraphs: Optional[List[Paragraph]] = None,
                      images: Optional[List[Image]] = None,
                      tables: Optional[List[Table]] = None,
                      equations: Optional[List[Equation]] = None,
                      subsections: Optional[List[Section]] = None,
                      section_id: Optional[str] = None,
                      embeddings: Optional[Dict[str, List[float]]] = None) -> Section:
        """Create a section"""
        return Section(
            title=title,
            text=text,
            page_start=page_start,
            paragraphs=paragraphs or [],
            images=images or [],
            tables=tables or [],
            equations=equations or [],
            subsections=subsections or [],
            section_id=section_id or str(uuid.uuid4())[:8],
            embeddings=embeddings or {}
        )
    
    def create_document(self, path: str,
                       title: str,
                       sections: List[Section],
                       text: str = "",
                       document_id: Optional[str] = None,
                       embeddings: Optional[Dict[str, List[float]]] = None) -> Document:
        """Create a document"""
        return Document(
            path=path,
            title=title,
            sections=sections,
            text=text,
            document_id=document_id or str(uuid.uuid4())[:8],
            embeddings=embeddings or {}
        )
    
    # ==========================================
    # Metadata Factories
    # ==========================================
    
    def create_metadata(self, **kwargs) -> Metadata:
        """Create metadata"""
        return self.metadata_creator.create_metadata(**kwargs)
    
    # ==========================================
    # Tree Model Factories
    # ==========================================
    
    def create_document_chunk(self, text: str,
                             chunk_type: ChunkType = ChunkType.DOCUMENT,
                             level: ChunkLevel = ChunkLevel.DOCUMENT,
                             chunk_id: Optional[str] = None,
                             summary: str = "",
                             parent_id: Optional[str] = None,
                             embedding: Optional[np.ndarray] = None,
                             metadata: Optional[Metadata] = None) -> DocumentChunk:
        """Create a document chunk"""
        return DocumentChunk(
            text=text,
            chunk_type=chunk_type,
            level=level,
            chunk_id=chunk_id or str(uuid.uuid4())[:8],
            summary=summary,
            parent_id=parent_id,
            embedding=embedding,
            metadata=metadata
        )
    
    def create_document_node(self, node_id: str, chunk: DocumentChunk) -> DocumentNode:
        """Create a document node"""
        return DocumentNode(node_id, chunk)
    
    def create_document_tree(self, title: str, root: Optional[DocumentNode] = None) -> DocumentTree:
        """Create a document tree"""
        if root is None:
            # Create empty tree with root node
            chunk = self.create_document_chunk(
                text=title,
                chunk_type=ChunkType.DOCUMENT,
                level=ChunkLevel.DOCUMENT
            )
            root = self.create_document_node("root", chunk)
        
        return DocumentTree(title=title, root=root)
    
    def create_empty_tree(self, title: str) -> DocumentTree:
        """Create an empty document tree"""
        return create_empty_tree(title)
    
    # ==========================================
    # Service Factories
    # ==========================================
    def create_document_processor(self, llm_wrapper=None, embedding_model=None)->DocumentProcessor:
        """Create a document processor"""
        return HierarchicalProcessor(llm_wrapper=llm_wrapper, embedding_model=embedding_model)
    
    def create_pdf_parser(self, output_dir: str = "output") -> DocumentParser:
        """Create a PDF parser"""
        return PDFParser(output_dir=output_dir)
    
    def create_document_engine(self, 
                               parser,
                               processor,
                              persist_directory: Optional[str] = None) -> DocumentEngine:
        """Create a document engine"""
        return DocumentEngine(
            parser=parser,
            processor=processor,
            persist_directory=persist_directory
        )
    
    def create_document_manager(self,
                               repository: BaseDocumentRepository,
                               document_engine: DocumentEngine) -> DocumentManager:
        """Create a document manager"""
        return DocumentManager(
            repository,
            document_engine
        )
    
    def create_document_repository(self, storage_path: str = "data/documents/") -> BaseDocumentRepository:
        """Create a document repository"""
        from src.store.document import DocumentRepository
        return DocumentRepository(storage_path=storage_path)
    
    def create_html_generator(self, template_dir: Optional[str] = None) -> HTMLGenerator:
        """Create an HTML generator"""
        return HTMLGenerator(template_dir=template_dir)
    
    def create_console_printer(self) -> ConsolePrinter:
        """Create a console printer"""
        return ConsolePrinter()
    
    # ==========================================
    # Batch Creation Methods
    # ==========================================
    
    def create_paragraph_from_sentences(self, sentences: List[Dict[str, Any]]) -> Paragraph:
        """
        Create a paragraph from a list of sentence dictionaries
        
        Args:
            sentences: List of sentence data dictionaries
        
        Returns:
            Paragraph object
        """
        sentence_objs = []
        full_text = ""
        
        for sent_data in sentences:
            sentence = self.create_sentence(
                text=sent_data.get('text', ''),
                start=sent_data.get('start', 0),
                end=sent_data.get('end', 0),
                page=sent_data.get('page', 0),
                bbox=sent_data.get('bbox', (0, 0, 0, 0))
            )
            sentence_objs.append(sentence)
            full_text += sentence.text + " "
        
        return self.create_paragraph(
            text=full_text.strip(),
            sentences=sentence_objs,
            start=min((s.start for s in sentence_objs), default=0),
            end=max((s.end for s in sentence_objs), default=0),
            page=sentence_objs[0].page if sentence_objs else 0
        )
    
    def create_section_with_content(self, title: str,
                                   paragraphs: List[Dict[str, Any]],
                                   images: Optional[List[Dict[str, Any]]] = None,
                                   page_start: int = 1) -> Section:
        """
        Create a section with paragraphs and images
        
        Args:
            title: Section title
            paragraphs: List of paragraph data
            images: Optional list of image data
            page_start: Starting page number
        
        Returns:
            Section object
        """
        paragraph_objs = []
        full_text = []
        
        for para_data in paragraphs:
            if 'sentences' in para_data:
                paragraph = self.create_paragraph_from_sentences(para_data['sentences'])
            else:
                paragraph = self.create_paragraph(
                    text=para_data.get('text', ''),
                    start=para_data.get('start', 0),
                    end=para_data.get('end', 0),
                    page=para_data.get('page', page_start)
                )
            paragraph_objs.append(paragraph)
            full_text.append(paragraph.text)
        
        image_objs = []
        if images:
            for img_data in images:
                image = self.create_image(
                    image_path=img_data.get('path', ''),
                    page=img_data.get('page', page_start),
                    bbox=img_data.get('bbox', (0, 0, 0, 0))
                )
                image_objs.append(image)
        
        return self.create_section(
            title=title,
            text="\n".join(full_text),
            page_start=page_start,
            paragraphs=paragraph_objs,
            images=image_objs
        )
    
    def create_document_from_sections(self, path: str, title: str,
                                     sections_data: List[Dict[str, Any]]) -> Document:
        """
        Create a document from sections data
        
        Args:
            path: Document path
            title: Document title
            sections_data: List of section data dictionaries
        
        Returns:
            Document object
        """
        sections = []
        full_text = []
        
        for sec_data in sections_data:
            section = self.create_section_with_content(
                title=sec_data.get('title', 'Untitled'),
                paragraphs=sec_data.get('paragraphs', []),
                images=sec_data.get('images', []),
                page_start=sec_data.get('page_start', 1)
            )
            sections.append(section)
            full_text.append(section.text)
        
        return self.create_document(
            path=path,
            title=title,
            sections=sections,
            text="\n".join(full_text)
        )
    
    # ==========================================
    # Conversion Methods
    # ==========================================
    
    def document_to_tree(self, document: Document) -> DocumentTree:
        """
        Convert a Document to a DocumentTree
        
        This creates a basic tree structure without summaries.
        For a full tree with summaries, use DocumentEngine.
        """
        # Create root node
        root_chunk = self.create_document_chunk(
            text=document.title,
            chunk_type=ChunkType.DOCUMENT,
            level=ChunkLevel.DOCUMENT,
            metadata=self.create_metadata(
                length=len(document.text)
            )
        )
        root_node = self.create_document_node("root", root_chunk)
        tree = self.create_document_tree(document.title, root_node)
        
        section_chunks = []
        paragraph_chunks = []
        sentence_chunks = []
        
        # Add sections
        for section in document.sections:
            section_chunk = self.create_document_chunk(
                text=section.title,
                chunk_type=ChunkType.SECTION,
                level=ChunkLevel.SECTION,
                metadata=self.create_metadata(
                    length=len(section.text),
                    page=section.page_start
                )
            )
            section_node = self.create_document_node(section.section_id, section_chunk)
            root_node.add_child(section_node)
            section_chunks.append(section_chunk)
            
            # Add paragraphs
            for paragraph in section.paragraphs:
                para_chunk = self.create_document_chunk(
                    text=paragraph.text,
                    chunk_type=ChunkType.PARAGRAPH,
                    level=ChunkLevel.PARAGRAPH,
                    parent_id=section.section_id,
                    metadata=self.create_metadata(
                        length=len(paragraph.text),
                        start=paragraph.start,
                        end=paragraph.end,
                        page=paragraph.page
                    )
                )
                para_node = self.create_document_node(paragraph.paragraph_id, para_chunk)
                section_node.add_child(para_node)
                paragraph_chunks.append(para_chunk)
                
                # Add sentences
                for sentence in paragraph.sentences:
                    sent_chunk = self.create_document_chunk(
                        text=sentence.text,
                        chunk_type=ChunkType.SENTENCE,
                        level=ChunkLevel.SENTENCE,
                        parent_id=paragraph.paragraph_id,
                        metadata=self.create_metadata(
                            length=len(sentence.text),
                            start=sentence.start,
                            end=sentence.end,
                            page=sentence.page
                        )
                    )
                    sent_node = self.create_document_node(sentence.sentence_id, sent_chunk)
                    para_node.add_child(sent_node)
                    sentence_chunks.append(sent_chunk)
        
        tree.set_chunks(root_chunk, section_chunks, paragraph_chunks, sentence_chunks)
        return tree
    
    # ==========================================
    # Configuration Methods
    # ==========================================
    
    def create_default_services(self, persist_directory: Optional[str] = None,
                               llm_wrapper=None, embedding_model=None) -> Dict[str, Any]:
        """
        Create a dictionary of default services
        
        Returns:
            Dictionary with keys: 'parser', 'engine', 'manager', 'repository',
                                  'html_generator', 'console_printer'
        """
        processor = self.create_document_processor(llm_wrapper, embedding_model)
        repository = self.create_document_repository()
        engine = self.create_document_engine(processor, persist_directory)
        return {
            'parser': self.create_pdf_parser(),
            'processor': processor,
            'repository': repository,
            'engine': engine,
            'manager': self.create_document_manager(repository, engine),
            'html_generator': self.create_html_generator(),
            'console_printer': self.create_console_printer()
        }