from enum import Enum
import re
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import spacy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.models.text import EmbeddingModel
from src.core.document.document import Document

def parse_document(path: str) -> Tuple[str, List[Dict]]:
    """Parse document and return text and sections"""
    with open(path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Simple section extraction based on headings
    sections = []
    heading_pattern = r'(?:(?:^|\n)(?:#{1,6}\s+|(?:[0-9]+\.)+\s+|[A-Z][A-Z\s]+\n-+))(.+?)(?=\n(?:#{1,6}\s+|(?:[0-9]+\.)+\s+|[A-Z][A-Z\s]+\n-+|$))'
    matches = list(re.finditer(heading_pattern, text, re.DOTALL | re.MULTILINE))
    
    if matches:
        for i, match in enumerate(matches):
            section_text = match.group(0)
            next_start = matches[i+1].start() if i+1 < len(matches) else len(text)
            
            sections.append({
                'title': match.group(1).strip(),
                'text': text[match.start():next_start].strip(),
                'start': match.start(),
                'end': next_start
            })
    else:
        # No clear headings, treat entire text as one section
        sections.append({
            'title': 'Full Document',
            'text': text,
            'start': 0,
            'end': len(text)
        })
    
    return text, sections

@dataclass
class MetaData:
    """Hierarchical document chunk"""
    length: int = 0 # Length of the chunk
    start_pos: int = 0 # Start position in the original text
    end_pos: int = 0 # End position in the original text
    is_concept: bool = False # Whether the chunk contains a key concept

class ChunkType(Enum):
    """Enum for chunk types"""
    DOCUMENT = "document"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"

class ChunkLevel(Enum):
    """Enum for chunk levels"""
    DOCUMENT = 0
    SECTION = 1
    PARAGRAPH = 2
    SENTENCE = 3

@dataclass
class DocumentChunk:
    """Hierarchical document chunk"""
    text: str # Text content of the chunk
    chunk_type: ChunkType = ChunkType.DOCUMENT # Type of the chunk
    level: ChunkLevel = ChunkLevel.DOCUMENT # Level of the chunk
    chunk_id: int = None # Unique ID for the chunk
    parent_id: int = None # Parent chunk ID (for hierarchy)
    embedding: np.ndarray = None  # Embedding vector for the chunk
    metadata: MetaData = None # Additional metadata (e.g., title, length, etc.)
  
class MetaDataCreator:
    """Base class for metadata creation"""
    def create_metadata(self, length: int, start_pos: int, end_pos: int, is_concept: bool = False) -> MetaData:
        """Create metadata from text"""
        raise NotImplementedError("Subclasses should implement this method")
    def get_metadata(self) -> MetaData:
        """Get the created metadata"""
        raise NotImplementedError("Subclasses should implement this method")
    
class SimpleMetaDataCreator(MetaDataCreator):
    """Simple metadata creator that extracts basic information"""
    def create_metadata(self, length: int, start_pos: int, end_pos: int, is_concept: bool = False) -> MetaData:
        """Extract basic metadata from text"""
        return  MetaData(
            length=length,
            start_pos=start_pos,
            end_pos=end_pos,
            is_concept=is_concept
        )

    def get_metadata(self) -> MetaData:
        """Return the created metadata"""
        return self.create_metadata(0, 0, 0)
  
class DocumentNode:
    """Node in the hierarchical document tree"""
    def __init__(self, chunk: DocumentChunk):
        self.chunk = chunk
        self.children: List['DocumentNode'] = []

class DocumentTree:
    """Hierarchical document representation"""
    def __init__(self, title: str, root: DocumentNode):
        self.title = title
        self.root = root
        self.hierarchy:dict = {}
        self.total_chunks = 0

    def set_chunks(self, document_chunk: DocumentChunk, section_chunks: List[DocumentChunk],
                   paragraph_chunks: List[DocumentChunk], sentence_chunks: List[DocumentChunk]):
        """Set all document chunks"""
        self.hierarchy = {
            'document': document_chunk,
            'sections': section_chunks,
            'paragraphs': paragraph_chunks,
            'sentences': sentence_chunks
        }
        self.total_chunks = sum(len(chunks) for chunks in self.hierarchy.values() if chunks)
       
    def _create_adjacency_matrix(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """Create adjacency matrix for document graph"""
        n = len(chunks)
        adjacency = np.zeros((n, n))
        
        for i, chunk_i in enumerate(chunks):
            for j, chunk_j in enumerate(chunks):
                if i == j:
                    continue
                
                # Parent-child relationship
                if chunk_j.parent_id == chunk_i.chunk_id:
                    adjacency[i, j] = 1.0
                elif chunk_i.parent_id == chunk_j.chunk_id:
                    adjacency[i, j] = 1.0
                
                # Semantic similarity (if embeddings exist)
                if chunk_i.embedding is not None and chunk_j.embedding is not None:
                    sim = np.dot(chunk_i.embedding, chunk_j.embedding) / (
                        np.linalg.norm(chunk_i.embedding) * np.linalg.norm(chunk_j.embedding)
                    )
                    if sim > 0.7:  # High similarity threshold
                        adjacency[i, j] = max(adjacency[i, j], sim)
        
        return adjacency
    
    def get_hierarchy(self) -> Dict:
        """Get the hierarchical structure"""
        return self.hierarchy
    
    def get_chunks(self) -> List[DocumentChunk]:
        """Get all document chunks"""
        return [self.hierarchy['document']] + self.hierarchy['sections'] + self.hierarchy['paragraphs'] + self.hierarchy['sentences']

    def get_sections(self) -> List[DocumentChunk]:
        """Get section chunks"""
        return self.hierarchy['sections']
    
    def get_section(self, section_id) -> List[DocumentNode]:
        """Get section chunks"""
        return self.hierarchy['sections'][section_id]

    def get_previous_sections(self, section_id) -> List[DocumentChunk]:
        """Get previous sections"""
        return self.hierarchy['sections'][:section_id]
    
    def get_paragraphs(self, section_id) -> List[DocumentChunk]:
        """Get paragraph chunks"""
        section = self.hierarchy['sections'][section_id]
        return section.children

    def get_sentences(self, section_id, paragraph_id) -> List[DocumentChunk]:
        """Get sentence chunks"""
        section = self.hierarchy['sections'][section_id]
        paragraphs = section.children[paragraph_id]
        return paragraphs.children
    
    def get_total_chunks(self) -> int:
        """Get total number of chunks"""
        return self.total_chunks
    
    def get_title(self) -> str:
        """Get document title"""
        return self.title

class HierarchicalDocumentProcessor:
    """
    Creates hierarchical document representation with embeddings
    """
    def __init__(self,doc_manager, embedding_model=EmbeddingModel.DEFAULT_MODEL_NAME):
        self.doc_manager = doc_manager
        self.embedding_model = EmbeddingModel(embedding_model)
        self.nlp = spacy.load('en_core_web_sm')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.document_trees: List[DocumentTree] = []
        self.documents : List[Document] = []
        self.document_processed: List[bool] = []
    
    def get_document_tree(self, doc_id: int) -> DocumentTree:
        """Get the hierarchical document tree"""
        # Return existing tree if already processed
        if len(self.document_trees) >= doc_id:
            return self.document_trees[doc_id]
        
        # Process and return the document tree
        if not self.is_document_set(doc_id):
            raise ValueError("No document set in processor. Please provide a document.")
        
        if self.is_document_processed():
            return self.document_tree
        
        document = self.doc_manager.get_document(doc_id)
        document_text = document.get_text()
        doc_title = document.get_title()
        
        document_tree = self.process_document(document_text, doc_title)
        self.document_trees.append(document_tree)
        return document_tree
    
    def set_document(self, doc_id: int):
        """Set the current document"""
        document = self.doc_manager.get_document(doc_id)
        self.documents.append(document)
        self.document_processed = False
    
    def is_document_set(self,doc_id) -> bool:
        """Check if document is set"""
        if self.document and self.document.doc_id == doc_id:
            return True
        return False

    def is_document_processed(self) -> bool:
        """Check if document has been processed"""
        return self.document_processed

    def process_document(self, document_text: str, doc_title: str = "") -> DocumentTree:
        """
        Create hierarchical document structure with 4 levels:
        1. Document (root)
        2. Sections
        3. Paragraphs
        4. Sentences (key concepts)
        """
        print("Processing document hierarchy...")
        
        # Check document is set
        if self.document is None:
            raise ValueError("No document set. Please set a document using set_document().")

        # Check if already processed
        if self.document_processed:
            return self.document_tree
        
        # Level 1: Document
        doc_chunk = DocumentChunk(
            text=document_text[:1000] + "...",
            chunk_type=ChunkType.DOCUMENT,
            level=ChunkLevel.DOCUMENT,
            chunk_id=0,
            metadata=SimpleMetaDataCreator().create_metadata( 
                    doc_title,
                    0,
                    len(document_text)
                )
            )

        document_tree = DocumentTree(title=doc_title, root=DocumentNode(chunk=doc_chunk))

        # Level 2: Sections (using headings or logical breaks)
        sections = self._extract_sections(document_text)
        section_chunks = []
        
        for section_idx, section in enumerate(sections):
            section_chunk = DocumentChunk(
                text=section['text'],
                chunk_type=ChunkType.SECTION,
                level=ChunkLevel.SECTION,
                parent_id=0,
                chunk_id=section_idx + 1,
                metadata= SimpleMetaDataCreator().create_metadata( 
                    section.get('title', f'Section {section_idx + 1}'),
                    section.get('start', 0),
                    section.get('end', len(section['text']))
                )
            )
            document_tree.root.children.append(DocumentNode(chunk=section_chunk))
            section_chunks.append(section_chunk)
        
        # Level 3: Paragraphs within each section
        paragraph_chunks = []
        paragraph_id = 0
        for section_idx, section in enumerate(section_chunks):
            paragraphs = self._split_paragraphs(section.text)

            for para_idx, para in enumerate(paragraphs):
                para_chunk = DocumentChunk(
                    text=para,
                    chunk_type=ChunkType.PARAGRAPH,
                    level=ChunkLevel.PARAGRAPH,
                    parent_id= section.chunk_id,
                    chunk_id=paragraph_id,
                    metadata=SimpleMetaDataCreator().create_metadata(
                        len(para),
                        0,
                        len(para)
                    )
                )
                section_node = document_tree.root.children[section_idx]
                section_node.children.append(DocumentNode(chunk=para_chunk))
                paragraph_chunks.append(para_chunk)
                paragraph_id += 1

        # Level 4: Key sentences/concepts
        sentence_chunks = []
        sentence_id = 0

        for para_idx, para in enumerate(paragraph_chunks):
            sentences = self._extract_key_sentences(para.text)
            section_idx  = para_idx // len(paragraph_chunks)
            section_node = document_tree.root.children[section_idx]
            paragraph_node = section_node.children[para_idx]
            for sent in sentences:
                sent_chunk = DocumentChunk(
                    text=sent,
                    chunk_type=ChunkType.SENTENCE,
                    level=ChunkLevel.SENTENCE,
                    parent_id=para.chunk_id,
                    chunk_id=sentence_id,
                    metadata=SimpleMetaDataCreator().create_metadata(
                        len(sent),  
                        0,
                        len(sent),
                        self._is_concept_sentence(sent)
                    )
                )
                paragraph_node.children.append(DocumentNode(chunk=sent_chunk))
                sentence_chunks.append(sent_chunk)
                sentence_id += 1
        
        # Combine all chunks
        all_chunks = [doc_chunk] + section_chunks + paragraph_chunks + sentence_chunks
        
        # Generate embeddings for each chunk
        print("Generating embeddings...")
        for chunk in all_chunks:
            if len(chunk.text) > 10:  # Skip very short chunks
                chunk.embedding = self.embedding_model.encode(chunk.text)
        
   
        
        document_tree.set_chunks(doc_chunk, section_chunks,paragraph_chunks,sentence_chunks)
        self.document_tree = document_tree
        self.document_processed = True
        return document_tree
            
    def _extract_sections(self, text: str) -> List[Dict]:
        """Extract sections based on headings or logical breaks"""
        sections = []
        
        # Pattern for common headings (## Heading, 1. Title, etc.)
        heading_pattern = r'(?:(?:^|\n)(?:#{1,6}\s+|(?:[0-9]+\.)+\s+|[A-Z][A-Z\s]+\n-+))(.+?)(?=\n(?:#{1,6}\s+|(?:[0-9]+\.)+\s+|[A-Z][A-Z\s]+\n-+|$))'
        
        matches = list(re.finditer(heading_pattern, text, re.DOTALL | re.MULTILINE))
        
        if matches:
            for i, match in enumerate(matches):
                section_text = match.group(0)
                next_start = matches[i+1].start() if i+1 < len(matches) else len(text)
                
                sections.append({
                    'title': match.group(1).strip(),
                    'text': text[match.start():next_start].strip(),
                    'start': match.start(),
                    'end': next_start
                })
        else:
            # No clear headings, split by large gaps
            parts = re.split(r'\n\s*\n\s*\n', text)
            for i, part in enumerate(parts):
                if len(part.strip()) > 100:
                    sections.append({
                        'title': f'Part {i+1}',
                        'text': part.strip(),
                        'start': text.find(part),
                        'end': text.find(part) + len(part)
                    })
        
        return sections
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if len(p.strip()) > 50]
    
    def _extract_key_sentences(self, text: str, n_sentences: int = 3) -> List[str]:
        """Extract key sentences using TextRank-like algorithm"""
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        if len(sentences) <= n_sentences:
            return sentences
        
        # Simple scoring: sentence length, position, contains keywords
        scores = []
        for i, sent in enumerate(sentences):
            score = 0
            
            # Position score (first and last sentences are important)
            if i == 0 or i == len(sentences) - 1:
                score += 2
            
            # Length score (medium length sentences are often key)
            words = len(sent.split())
            if 10 <= words <= 30:
                score += 1
            
            # Question score (sentences with ? might be important)
            if '?' in sent:
                score += 1
            
            # Definition score (sentences with "is defined as", "means", etc.)
            definition_indicators = ['is defined as', 'means that', 'refers to', 
                                   'is called', 'is known as']
            if any(indicator in sent.lower() for indicator in definition_indicators):
                score += 2
            
            scores.append(score)
        
        # Get top n sentences
        top_indices = np.argsort(scores)[-n_sentences:]
        return [sentences[i] for i in sorted(top_indices)]
    
    def _is_concept_sentence(self, sentence: str) -> bool:
        """Check if sentence contains a concept/definition"""
        concept_indicators = [
            'is defined as', 'means', 'refers to', 'is called',
            'can be defined', 'denoted by', 'represented as',
            'formally', 'mathematically', 'in other words'
        ]
        
        sentence_lower = sentence.lower()
        return any(indicator in sentence_lower for indicator in concept_indicators)
   

if __name__ == "__main__":
    # Example usage
    from src.core.document.document_processing import 
    doc_manager = DocumentManager()
    processor = HierarchicalDocumentProcessor(doc_manager)
    
    # Assume document with ID 1 exists
    doc_id = 1
    processor.set_document(doc_id)
    document_tree = processor.get_document_tree(doc_id)
    
    print(f"Document Title: {document_tree.get_title()}")
    print(f"Total Chunks: {document_tree.get_total_chunks()}")