from typing import Iterator, Optional

from ..models.base import ProcessedSection, ProcessingContext, Relationship
from ..models.structure import Section
from ....store.vector.base import VectorDocument

from .summary_generator import SummaryGenerator
from .base import DocumentProcessor

      
class HierarchicalProcessor(DocumentProcessor):
    """Processes documents hierarchically with summaries"""
    
    def __init__(self, llm_wrapper=None):
        self.summary_generator = SummaryGenerator(llm_wrapper)

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

    def process(
        self,
        section: Section,
        context: Optional[ProcessingContext] = None,
    ) -> ProcessedSection:

        if context is None:
            context = ProcessingContext(section.document_id)

        vector_documents: list[VectorDocument] = []
        relationships: list[Relationship] = []

        # --------------------------------------------------------------
        # 1. Generate section summary
        # --------------------------------------------------------------

        section_text = "\n\n".join(
            paragraph.text
            for paragraph in section.paragraphs
        )

        section_summary = self._generate_summary(
            text=section_text,
            context=context.previous_section_summary,
        )

        # ---------------------------------------------------------
        # IMPORTANT:
        # We don't generate vectors/relationships here.
        #
        # We only create generators that know how to generate them.
        # ---------------------------------------------------------

        return ProcessedSection(
            section_id=section.id,
            document_id=section.document_id,
            title=section.title,
            summary=section_summary,

            vector_documents=lambda: self._iter_vector_documents(
                section,
                section_summary,
            ),

            relationships=lambda: self._iter_relationships(
                section,
            ),

            metadata={
                "page": section.page,
                "level": section.level,
            },
        )

    def _iter_vector_documents(
        self,
        section: Section,
        section_summary: str,
    ) -> Iterator[VectorDocument]:

        # ---------------------------------------------------------
        # Section summary
        # ---------------------------------------------------------

        if section_summary.strip():

            yield VectorDocument(
                id=section.id,
                text=section_summary,
                metadata={
                    "document_id": section.document_id,
                    "chunk_id": section.id,
                    "parent_id": section.document_id,
                    "section_id": section.id,
                    "level": "section",
                    "page": section.page,
                    "type": "summary",
                },
            )

        # ---------------------------------------------------------
        # Paragraphs
        # ---------------------------------------------------------

        for paragraph in section.paragraphs:

            paragraph_summary = self._generate_summary(
                text=paragraph.text,
                context=section_summary,
            )

            yield VectorDocument(
                id=paragraph.id,
                text=paragraph_summary,
                metadata={
                    "document_id": section.document_id,
                    "chunk_id": paragraph.id,
                    "parent_id": section.id,
                    "section_id": section.id,
                    "level": "paragraph",
                    "page": paragraph.page,
                    "type": "summary",
                },
            )

            # -----------------------------------------------------
            # Sentences
            # -----------------------------------------------------

            for sentence in paragraph.sentences:

                yield VectorDocument(
                    id=sentence.id,
                    text=sentence.text,
                    metadata={
                        "document_id": section.document_id,
                        "chunk_id": sentence.id,
                        "parent_id": paragraph.id,
                        "section_id": section.id,
                        "level": "sentence",
                        "page": sentence.page,
                        "type": "text",
                    },
                )

    def _iter_relationships(
        self,
        section: Section,
    ) -> Iterator[Relationship]:

        # ---------------------------------------------------------
        # DOCUMENT -> SECTION
        # ---------------------------------------------------------

        yield Relationship(
            source_id=section.document_id,
            target_id=section.id,
            relation="CONTAINS",
        )

        # ---------------------------------------------------------
        # SECTION -> PARAGRAPH
        # ---------------------------------------------------------

        previous_paragraph_id = None

        for paragraph in section.paragraphs:

            yield Relationship(
                source_id=section.id,
                target_id=paragraph.id,
                relation="CONTAINS",
            )

            # -----------------------------------------------------
            # Paragraph ordering
            # -----------------------------------------------------

            if previous_paragraph_id is not None:

                yield Relationship(
                    source_id=previous_paragraph_id,
                    target_id=paragraph.id,
                    relation="NEXT",
                )

            previous_paragraph_id = paragraph.id

            # -----------------------------------------------------
            # PARAGRAPH -> SENTENCE
            # -----------------------------------------------------

            previous_sentence_id = None

            for sentence in paragraph.sentences:

                yield Relationship(
                    source_id=paragraph.id,
                    target_id=sentence.id,
                    relation="CONTAINS",
                )

                # -------------------------------------------------
                # Sentence ordering
                # -------------------------------------------------

                if previous_sentence_id is not None:

                    yield Relationship(
                        source_id=previous_sentence_id,
                        target_id=sentence.id,
                        relation="NEXT",
                    )

                previous_sentence_id = sentence.id

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _generate_summary(
        self,
        text: Optional[str] = "",
        context: str = "",
    ) -> str:

        if not text.strip():
            return ""

        if self.summary_generator is None:
            return text[:500]

        return self.summary_generator.generate_summary(
            text=text,
            context=context,
        )