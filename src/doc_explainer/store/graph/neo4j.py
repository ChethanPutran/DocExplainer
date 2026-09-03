from __future__ import annotations

import json
from typing import Any, Iterator, Optional

from doc_explainer.core.document.models.base import (
    ProcessedSection,
    Relationship,
)
from doc_explainer.core.document.models.tree import DocumentChunk
from doc_explainer.store.graph.base import GraphStore


class Neo4jGraphStore(GraphStore):

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
    ) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self.driver: Any = None

    # ============================================================
    # CONNECTION
    # ============================================================

    def connect(self) -> None:

        if self.driver is not None:
            return

        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
        )

        self.driver.verify_connectivity()

    def close(self) -> None:

        if self.driver is not None:
            self.driver.close()
            self.driver = None

    def __enter__(self) -> "Neo4jGraphStore":

        self.connect()

        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:

        self.close()

    # ============================================================
    # DOCUMENT
    # ============================================================

    def add_document(
        self,
        document_id: str,
        title: str,
        namespace: str,
        metadata: Optional[dict] = None,
    ) -> None:
        self.connect()

        with self.driver.session() as session:
            session.execute_write(
                self._create_document,
                document_id,
                title,
                namespace,
                metadata or {},
            )

    @staticmethod
    def _create_document(
        tx: Any,
        document_id: str,
        title: str,
        namespace: str,
        metadata: dict,
    ) -> None:
        tx.run(
            """
            MERGE (d:Document {id: $document_id, namespace: $namespace})
            SET d.text = $title,
                d.summary = '',
                d.metadata = $metadata
            RETURN d
            """,
            document_id=document_id,
            title=title,
            namespace=namespace,
            metadata=json.dumps(metadata),
        ).consume()

    def get_document(
        self,
        document_id: str,
    ) -> Optional[DocumentChunk]:

        self.connect()

        with self.driver.session() as session:

            result = session.run(
                """
                MATCH (d:Document {
                    id: $document_id,
                    namespace: $document_id
                })
                RETURN d
                """,
                document_id=document_id,
            )

            record = result.single()

            if record is None:
                return None

            return self._node_to_chunk(
                record["d"]
            )

    # ============================================================
    # SECTION
    # ============================================================

    def add_section(
        self,
        namespace: str,
        section: ProcessedSection,
    ) -> None:

        self.connect()

        with self.driver.session() as session:

            session.execute_write(
                self._create_section,
                namespace,
                section,
            )

    def add_chunk(
        self,
        namespace: str,
        chunk_id: str,
        text: str,
        metadata: dict,
    ) -> None:
        self.connect()

        with self.driver.session() as session:
            session.execute_write(
                self._create_chunk,
                namespace,
                chunk_id,
                text,
                metadata,
            )

    @staticmethod
    def _create_chunk(
        tx: Any,
        namespace: str,
        chunk_id: str,
        text: str,
        metadata: dict,
    ) -> None:
        tx.run(
            """
            MERGE (c {id: $chunk_id, namespace: $namespace})
            SET c:Chunk,
                c.text = $text,
                c.summary = $text,
                c.parent_id = $parent_id,
                c.position = $position,
                c.metadata = $metadata
            RETURN c
            """,
            chunk_id=chunk_id,
            namespace=namespace,
            text=text,
            parent_id=metadata.get("parent_id"),
            position=metadata.get("position", 0),
            metadata=json.dumps(metadata),
        ).consume()

    @staticmethod
    def _create_section(
        tx: Any,
        namespace: str,
        section: ProcessedSection,
    ) -> None:

        query = """
        MERGE (s:Section {
            id: $section_id,
            namespace: $namespace
        })

        SET
            s.document_id = $document_id,
            s.title = $title,
            s.summary = $summary,
            s.level = $level,
            s.page = $page,
            s.metadata = $metadata

        RETURN s
        """

        tx.run(
            query,
            section_id=section.section_id,
            namespace=namespace,
            document_id=section.document_id,
            title=section.title,
            summary=section.summary,
            level=section.metadata.get("level", 0),
            page=section.metadata.get("page", 0),
            metadata=json.dumps(section.metadata),
        ).consume()

    # ============================================================
    # GET SECTION
    # ============================================================

    def get_section(
        self,
        namespace: str,
        section_id: str,
    ) -> Optional[ProcessedSection]:

        self.connect()

        with self.driver.session() as session:

            result = session.run(
                """
                MATCH (
                    s:Section {
                        id: $section_id,
                        namespace: $namespace
                    }
                )
                RETURN s
                """,
                section_id=section_id,
                namespace=namespace,
            )

            record = result.single()

            if record is None:
                return None

            node = record["s"]

            return self._node_to_section(node)

    # ============================================================
    # CHILDREN
    # ============================================================

    def get_children(
        self,
        node_id: str,
    ) -> list[DocumentChunk]:

        self.connect()

        with self.driver.session() as session:

            result = session.run(
                """
                MATCH (parent {id: $node_id})
                      -[:CONTAINS]->
                      (child)

                RETURN child
                ORDER BY child.position
                """,
                node_id=node_id,
            )

            return [
                self._node_to_chunk(
                    record["child"]
                )
                for record in result
            ]

    # ============================================================
    # RELATIONSHIPS
    # ============================================================

    def add_relationships(
        self,
        namespace: str,
        relationships: Iterator[Relationship],
        batch_size: int = 100,
    ) -> None:

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero"
            )

        self.connect()

        batch: list[Relationship] = []

        for relationship in relationships:

            batch.append(relationship)

            if len(batch) >= batch_size:

                self._write_relationship_batch(
                    namespace,
                    batch,
                )

                batch.clear()

        if batch:

            self._write_relationship_batch(
                namespace,
                batch,
            )

    def _write_relationship_batch(
        self,
        namespace: str,
        relationships: list[Relationship],
    ) -> None:

        if not relationships:
            return

        unique_relationships: dict[
            tuple[str, str, str],
            Relationship,
        ] = {}

        for relationship in relationships:

            key = (
                relationship.source_id,
                relationship.target_id,
                relationship.relation,
            )

            unique_relationships[key] = relationship

        rows = [
            {
                "source_id": r.source_id,
                "target_id": r.target_id,
                "relation": r.relation,
                "properties": getattr(
                    r,
                    "properties",
                    {},
                ),
            }
            for r in unique_relationships.values()
        ]

        with self.driver.session() as session:

            session.execute_write(
                self._write_relationships_tx,
                namespace,
                rows,
            )

    @staticmethod
    def _write_relationships_tx(
        tx: Any,
        namespace: str,
        relationships: list[dict[str, Any]],
    ) -> None:
        relationship_types = {
            relationship["relation"]
            for relationship in relationships
        }

        for relationship_type in relationship_types:
            if not relationship_type.replace("_", "").isalnum():
                raise ValueError(
                    f"Invalid Neo4j relationship type: {relationship_type}"
                )

            query = f"""
            UNWIND $relationships AS rel

            MATCH (source {{id: rel.source_id, namespace: $namespace}})
            MATCH (target {{id: rel.target_id, namespace: $namespace}})
            MERGE (source)-[relationship:{relationship_type}]->(target)
            SET relationship += rel.properties

            RETURN count(relationship) AS count
            """

            tx.run(
                query,
                namespace=namespace,
                relationships=[
                    relationship
                    for relationship in relationships
                    if relationship["relation"] == relationship_type
                ],
            ).consume()

    # ============================================================
    # CONVERSION
    # ============================================================

    @staticmethod
    def _node_to_chunk(
        node: Any,
    ) -> DocumentChunk:

        metadata = node.get("metadata")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}

        return DocumentChunk(
            id=node.get("id", ""),
            text=node.get("text", ""),
            summary=node.get("summary", ""),
            parent_id=node.get("parent_id"),
            metadata=metadata,
        )

    @staticmethod
    def _node_to_section(
        node: Any,
    ) -> ProcessedSection:

        # These cannot be reconstructed from Neo4j without
        # reconstructing the lazy generators.

        metadata = node.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}

        return ProcessedSection(
            section_id=node.get("id", ""),
            document_id=node.get("document_id", ""),
            title=node.get("title", ""),
            summary=node.get("summary", ""),
            vector_documents=lambda: iter(()),
            relationships=lambda: iter(()),
            metadata=metadata,
        )