from typing import Optional, List
from ..models.structure import Document
from ..models.tree import DocumentTree, create_empty_tree
from .processor import HierarchicalProcessor


class DocumentEngine:
    """Orchestrates document processing pipeline"""
    
    def __init__(self, llm_wrapper=None, embedding_model=None, persist_directory: Optional[str] = None):
        self.persist_directory = persist_directory
        self.processor = HierarchicalProcessor(llm_wrapper, embedding_model)
        self.full_db = None
        self.tree_db = None
        self.tree = None
    
    def ingest_and_map(self, document: Document, target_query: Optional[str] = None):
        """
        Full processing pipeline:
        1. Full indexing
        2. Target discovery
        3. Tree building
        4. Tree indexing
        """
        print(f"Starting ingestion for: {document.title}")
        
        # Phase 1: Create full-text vector DB
        if self.processor.langchain_embeddings:
            self.full_db = self.processor.create_full_vector_db(
                document,
                persist_directory=self.persist_directory
            )
            print("Full document indexed.")
        
        # Phase 2: Resolve target section
        target_title = None
        if target_query and self.full_db:
            target_title = self._find_target_section(target_query, document)
            print(f"Target resolved to: '{target_title}'")
        
        # Phase 3: Build hierarchical tree
        print("Building hierarchical tree with summaries...")
        self.tree = self.processor.build_tree(document, target_section=target_title)
        
        # Phase 4: Create tree-aware vector DB
        if self.processor.langchain_embeddings:
            self.tree_db = self.processor.create_tree_aware_db(
                self.tree,
                persist_directory=self.persist_directory
            )
            print("Tree-aware vector DB created.")
        
        # Phase 5: Visualize
        print("\n--- DOCUMENT MAP ---")
        self.processor.visualize_tree(self.tree.root)
        
        return self.tree
    
    def _find_target_section(self, query: str, document: Document) -> str:
        """Find target section using semantic search"""
        if not self.full_db:
            return document.sections[0].title if document.sections else ""
        
        results = self.full_db.similarity_search(query, k=2)
        search_content = " ".join([d.page_content for d in results]).lower()
        
        # Collect all section titles
        all_titles = []
        
        def collect_titles(sections):
            for section in sections:
                all_titles.append(section.title)
                collect_titles(section.subsections)
        
        collect_titles(document.sections)
        
        # Find matching title
        for title in all_titles:
            if title.lower() in search_content:
                return title
        
        return all_titles[0] if all_titles else ""
    
    def query(self, user_query: str, level: str = "paragraph", k: int = 3):
        """Search within hierarchical summaries"""
        if not self.tree_db:
            return "Engine not initialized. Run ingest_and_map first."
        
        return self.tree_db.similarity_search(
            user_query,
            k=k,
            filter={"level": level}
        )
    
    def get_document_tree(self) -> Optional[DocumentTree]:
        """Get the built document tree"""
        return self.tree