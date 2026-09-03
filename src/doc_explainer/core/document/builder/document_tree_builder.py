from doc_explainer.store.graph.base import GraphStore

from ..models.tree import DocumentTree, DocumentNode

class DocumentTreeBuilder:

    def __init__(self, graph_store: GraphStore):
        self.graph_store = graph_store

    def build(self, document_id: str) -> DocumentTree:

        document = self.graph_store.get_document(document_id)

        if document is None:
            raise ValueError(
                f"Document '{document_id}' not found in graph store"
            )

        root = DocumentNode(
            node_id=document.id,
            chunk=document,
        )

        self._build_children(root)

        return DocumentTree(
            title=document.metadata.get("title", "")
            if document.metadata
            else "",
            root=root,
        )

    def _build_children(
        self,
        parent: DocumentNode,
    ) -> None:

        children = self.graph_store.get_children(
            parent.id
        )

        for chunk in children:

            child = DocumentNode(
                node_id=chunk.id,
                chunk=chunk,
            )

            parent.add_child(child)

            self._build_children(child)