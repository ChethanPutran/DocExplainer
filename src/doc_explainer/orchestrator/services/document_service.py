from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Optional, Dict, Any

from doc_explainer.orchestrator.progress import (
    ProgressEvent,
    ProgressReporter,
    ProgressStatus,
)

from ...core.document import DocumentManager, Document, DocumentTree
from ...core.knowledge import GraphStateManager


class DocumentService:
    """
    Application service responsible for document registration and retrieval.

    The service is UI-independent. Progress is reported through a
    ProgressReporter supplied for a particular execution.
    """

    def __init__(
        self,
        document_manager: DocumentManager,
        graph_state_manager: GraphStateManager,
        logger: Optional[logging.Logger] = None,
    ):
        self.document_manager = document_manager
        self.graph_state_manager = graph_state_manager

        self.logger = logger or logging.getLogger(
            self.__class__.__name__
        )

        # Runtime cache of document trees.
        self.document_trees: Dict[str, DocumentTree] = {}

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------

    def _report(
        self,
        reporter: Optional[ProgressReporter],
        job_id: str,
        document_id: Optional[str],
        step: str,
        status: ProgressStatus,
        progress: float,
        message: str,
        error: Optional[str] = None,
    ) -> None:
        """
        Report progress without coupling the service to a UI.
        """

        if reporter is None:
            return

        # Keep progress inside valid range.
        progress = max(0.0, min(1.0, progress))

        reporter.report(
            ProgressEvent(
                job_id=job_id,
                document_id=document_id,
                step=step,
                status=status,
                progress=progress,
                message=message,
                error=error,
            )
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_document(
        self,
        path: str,
        build_graph: bool = True,
        user_id: Optional[str] = None,
        progress_reporter: Optional[ProgressReporter] = None,
        job_id: Optional[str] = None,
    ) -> str:
        """
        Register a document.

        Returns:
            str: Registered document ID.

        Raises:
            Exception: If any registration stage fails.
        """

        job_id = job_id or str(uuid.uuid4())

        self.logger.info(
            "Registering document: %s [job_id=%s]",
            path,
            job_id,
        )

        doc_id: Optional[str] = None

        try:
            # ==========================================================
            # LOAD DOCUMENT
            # ==========================================================

            self._report(
                reporter=progress_reporter,
                job_id=job_id,
                document_id=None,
                step="load_document",
                status=ProgressStatus.STARTED,
                progress=0.0,
                message="Loading document...",
            )

            doc_id = self.document_manager.load_document(path)

            checkpoint_store = getattr(
                self.document_manager.engine,
                "checkpoint_store",
                None,
            )
            if (
                checkpoint_store
                and checkpoint_store.is_registration_complete(doc_id)
            ):
                self._report(
                    reporter=progress_reporter,
                    job_id=job_id,
                    document_id=doc_id,
                    step="registration",
                    status=ProgressStatus.COMPLETED,
                    progress=1.0,
                    message="Document already processed.",
                )
                return doc_id

            self._report(
                reporter=progress_reporter,
                job_id=job_id,
                document_id=doc_id,
                step="load_document",
                status=ProgressStatus.COMPLETED,
                progress=1.0,
                message="Document loaded.",
            )

            # ==========================================================
            # PROCESS DOCUMENT
            # ==========================================================

            self._report(
                reporter=progress_reporter,
                job_id=job_id,
                document_id=doc_id,
                step="process_document",
                status=ProgressStatus.STARTED,
                progress=0.0,
                message="Processing document...",
            )

            self.document_manager.process_document(doc_id)

            self._report(
                reporter=progress_reporter,
                job_id=job_id,
                document_id=doc_id,
                step="process_document",
                status=ProgressStatus.COMPLETED,
                progress=1.0,
                message="Document processed.",
            )

            # ==========================================================
            # BUILD DOCUMENT TREE
            # ==========================================================

            document_tree = None
            if build_graph:
                self._report(
                    reporter=progress_reporter,
                    job_id=job_id,
                    document_id=doc_id,
                    step="build_document_tree",
                    status=ProgressStatus.STARTED,
                    progress=0.0,
                    message="Building document tree...",
                )

                document_tree = self.document_manager.build_document_tree(
                    doc_id
                )

                self.document_trees[doc_id] = document_tree

                self._report(
                    reporter=progress_reporter,
                    job_id=job_id,
                    document_id=doc_id,
                    step="build_document_tree",
                    status=ProgressStatus.COMPLETED,
                    progress=1.0,
                    message="Document tree built.",
                )

            # ==========================================================
            # BUILD KNOWLEDGE GRAPH
            # ==========================================================

            if build_graph:
                self._report(
                    reporter=progress_reporter,
                    job_id=job_id,
                    document_id=doc_id,
                    step="build_knowledge_graph",
                    status=ProgressStatus.STARTED,
                    progress=0.0,
                    message="Building knowledge graph...",
                )

                self.graph_state_manager.build_chain(
                    document_tree
                )

                self._report(
                    reporter=progress_reporter,
                    job_id=job_id,
                    document_id=doc_id,
                    step="build_knowledge_graph",
                    status=ProgressStatus.COMPLETED,
                    progress=1.0,
                    message="Knowledge graph built.",
                )

            if checkpoint_store:
                checkpoint_store.mark_registration_complete(doc_id)

            # ==========================================================
            # COMPLETED
            # ==========================================================

            self._report(
                reporter=progress_reporter,
                job_id=job_id,
                document_id=doc_id,
                step="registration",
                status=ProgressStatus.COMPLETED,
                progress=1.0,
                message="Document registration completed.",
            )

            self.logger.info(
                "Document registration completed: %s [job_id=%s]",
                doc_id,
                job_id,
            )

            return doc_id

        except Exception as e:
            self._report(
                reporter=progress_reporter,
                job_id=job_id,
                document_id=doc_id,
                step="registration",
                status=ProgressStatus.FAILED,
                progress=0.0,
                message="Document registration failed.",
                error=str(e),
            )

            self.logger.exception(
                "Document registration failed [job_id=%s]",
                job_id,
            )

            raise

    # ------------------------------------------------------------------
    # Document access
    # ------------------------------------------------------------------

    def get_document(
        self,
        doc_id: str,
    ) -> Optional[Document]:
        """Get a document by ID."""

        try:
            return self.document_manager.get_document(doc_id)

        except Exception as e:
            self.logger.error(
                "Error getting document %s: %s",
                doc_id,
                e,
            )
            return None

    def get_document_tree(
        self,
        doc_id: str,
    ) -> Optional[DocumentTree]:
        """Get cached document tree."""

        return self.document_trees.get(doc_id)

    def has_document(
        self,
        doc_id: str,
    ) -> bool:
        """Check whether a document exists."""

        try:
            return self.document_manager.has_document(doc_id)

        except Exception:
            self.logger.exception(
                "Error checking document %s",
                doc_id,
            )
            return False

    def get_section_id_at_position(
        self,
        doc_id: str,
        page: int,
        position: int,
    ) -> str:
        """Get section ID at a page/position."""

        tree = self.document_trees.get(doc_id)
        if tree is None:
            tree = self.document_manager.get_document_tree(doc_id)

        if tree is not None:
            for section_node in tree.iter_sections():
                for paragraph_node in section_node.children.values():
                    metadata = paragraph_node.chunk.metadata or {}
                    if metadata.get("page") == page:
                        return str(section_node.id)

        return "-1"

    def get_document_context(
        self,
        doc_id: str,
        section_id: int,
    ) -> Dict[str, Any]:
        """Get document context up to a section."""

        return self.graph_state_manager.get_document_context(
            str(section_id)
        )