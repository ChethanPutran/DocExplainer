from typing import Optional, List
from ..models import Document, DocumentTree
from .base import BaseDocumentEngine, DocumentBuilder, SimilaritySearchDB, SimilarityResult



class DocumentEngine(BaseDocumentEngine):
    """Orchestrates document processing pipeline"""
    
    def __init__(self, 
                 processor: DocumentBuilder,
                 persist_directory: Optional[str] = None):
        self.persist_directory = persist_directory
        self.processor = processor
        self.full_db: Optional[SimilaritySearchDB] = None
        self.tree_db: Optional[SimilaritySearchDB] = None
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
            raw_tree_db: SimilaritySearchDB = self.processor.create_tree_aware_db(
                self.tree,
                persist_directory=self.persist_directory
            )
            # Wrap raw_tree_db to ensure it exposes a consistent similarity_search
            # that supports an optional filter parameter.
            class TreeDBAdapter:
                def __init__(self, db):
                    self._db = db

                def similarity_search(self, query: str, k: int = 3, filter: Optional[dict] = None):
                    # Prefer calling underlying method with filter if supported
                    try:
                        return self._db.similarity_search(query, k=k, filter=filter)
                    except TypeError:
                        # Underlying db doesn't accept filter kwarg: call basic search then apply filter
                        results = self._db.similarity_search(query, k=k)
                        if not filter:
                            return results

                        # Apply simple metadata filtering (supports equality checks)
                        def matches(item):
                            meta = getattr(item, 'metadata', {}) or {}
                            for key, val in filter.items():
                                if meta.get(key) != val:
                                    return False
                            return True

                        filtered = [r for r in results if matches(r)]
                        return filtered[:k]

                # Provide passthrough for other attributes/methods
                def __getattr__(self, name):
                    return getattr(self._db, name)

            self.tree_db = TreeDBAdapter(raw_tree_db)
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
    
    def query(self, user_query: str, level: str = "paragraph", k: int = 3) -> List[SimilarityResult]:
        """Search within hierarchical summaries"""
        if not self.tree_db:
            return []

        return self.tree_db.similarity_search(
            user_query,
            k=k,
            filter={"level": level}
        )
    
    def get_document_tree(self) -> Optional[DocumentTree]:
        """Get the built document tree"""
        return self.tree