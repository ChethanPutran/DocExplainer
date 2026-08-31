from typing import Optional, List

from doc_explainer.store.checkpoint.base import CheckpointStore
from .models import Document, DocumentTree
from .models.base import ProcessingContext
from .parser.base import DocumentParser
from .processor.base import DocumentProcessor
from .builder.base import BaseDocumentEngine, SimilaritySearchDB, SimilarityResult



class DocumentEngine(BaseDocumentEngine):
    """Orchestrates document processing pipeline"""
    
    def __init__(self, 
                 parser:DocumentParser,
                 processor: DocumentProcessor,
                 checkpoint_store: Optional[CheckpointStore] = None,
                 persist_directory: Optional[str] = None):
        self.parser = parser
        self.checkpoint_store = checkpoint_store
        self.vector_store = None
        self.graph_store = None
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

    def ingest(
                self,
                file_path: str
                ) -> str:
        """Ingest document from file path and process"""
        # ------------------------------------
        # 1. Parse metadata
        # ------------------------------------

        document = self.parser.parse_metadata(
            file_path
        )

        if(not document):
            raise ValueError(f"Failed to parse document metadata from {file_path}")

        # ------------------------------------
        # 2. Process sections incrementally
        # ------------------------------------

        context = ProcessingContext(
            document_id=document.document_id
        )

        for index, section in enumerate(
            self.parser.iter_sections(file_path)
        ):

            context.section_index = index

            # --------------------------------
            # Resume support
            # --------------------------------

            if (
                self.checkpoint_store
                and self.checkpoint_store.is_completed(
                    document.document_id,
                    section.section_id
                )
            ):
                continue

            if self.checkpoint_store:
                self.checkpoint_store.mark_started(
                    document.document_id,
                    section.section_id
                )

            # --------------------------------
            # Process
            # --------------------------------

            processed_section = (
                self.processor.process(
                    section,
                    context
                )
            )

            # --------------------------------
            # Persist vector
            # --------------------------------

            if self.vector_store:

                vector_documents = (
                    self._to_vector_documents(
                        processed_section
                    )
                )

                self.vector_store.add(
                    vector_documents
                )

            # --------------------------------
            # Persist graph
            # --------------------------------

            if self.graph_store:

                self.graph_store.add_section(
                    processed_section
                )

            # --------------------------------
            # Checkpoint
            # --------------------------------

            if self.checkpoint_store:

                self.checkpoint_store.mark_completed(
                    document.document_id,
                    section.section_id
                )

            # --------------------------------
            # Update context
            # --------------------------------

            context.previous_section_summary = (
                processed_section.summary
            )

            # Release section
            del processed_section

        return document.document_id


    def _to_vector_documents(self, processed_section) -> List:
        """Convert processed section to vector documents for storage"""
        # Placeholder: Convert processed_section to a list of vector documents
        # This would typically involve extracting text chunks and their embeddings
        return []
    
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