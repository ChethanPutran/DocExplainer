from typing import Iterable, Iterator, Optional, Sequence

import chromadb

from doc_explainer.core.document.models.base import SimilarityResult

from .base import VectorDocument, VectorStore


class ChromaEmbeddingFunction:
    """Adapt the project's encode-based model to Chroma's protocol."""

    def __init__(self, model):
        self.model = model

    @staticmethod
    def name() -> str:
        return "doc_explainer_embedding"

    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(input)
        return (
            embeddings.tolist()
            if hasattr(embeddings, "tolist")
            else embeddings
        )

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "ChromaEmbeddingFunction":
        raise NotImplementedError(
            "ChromaEmbeddingFunction must be constructed with a model"
        )


class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of the vector store."""

    def __init__(
        self,
        persist_directory: str,
        embedding_function=None,
    ):
        """
        Initialize Chroma vector store.

        Args:
            persist_directory: Directory where Chroma persists data.
            embedding_function: Optional Chroma-compatible embedding function.
        """
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.chroma_embedding_function = (
            ChromaEmbeddingFunction(embedding_function)
            if embedding_function is not None
            else None
        )

        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

    def _get_collection(self, namespace: str):
        """Get or create a Chroma collection."""
        return self.client.get_or_create_collection(
            name=namespace,
            embedding_function=self.chroma_embedding_function,
        )

    def add(
        self,
        namespace: str,
        documents: Iterator[VectorDocument],
        batch_size: int = 32,
    ) -> None:
        """
        Add documents to the vector store in batches.
        """

        batch = []

        for document in documents:

            batch.append(document)

            if len(batch) >= batch_size:

                self._upsert_batch(
                    namespace=namespace,
                    documents=batch,
                )

                batch.clear()

        # Remaining documents

        if batch:

            self._upsert_batch(
                namespace=namespace,
                documents=batch,
            )

    def _upsert_batch(
        self,
        namespace: str,
        documents: Sequence[VectorDocument],
    ) -> None:
        """Upsert one batch of vector documents into a collection."""
        if not documents:
            return

        collection = self._get_collection(namespace)
        collection.upsert(
            ids=[document.id for document in documents],
            documents=[document.text for document in documents],
            metadatas=[document.metadata for document in documents],
        )

    def search(
        self,
        namespace: str,
        query: str,
        k: int = 5,
        filters: Optional[dict] = None,
    ) -> list["SimilarityResult"]:
        """
        Search for documents in the vector store.
        """

        if not query.strip():
            return []

        if k <= 0:
            return []

        collection = self._get_collection(namespace)

        query_result = collection.query(
            query_texts=[query],
            n_results=k,
            where=filters,
        )

        if not query_result:
            raise ValueError("Chroma query returned no results")

        results: list[SimilarityResult] = []

        ids = query_result.get("ids") or [[]]
        documents = query_result.get("documents") or [[]]
        metadatas = query_result.get("metadatas") or [[]]
        distances = query_result.get("distances") or [[]]

        ids = ids[0] if ids else []
        documents = documents[0] if documents else []
        metadatas = metadatas[0] if metadatas else []
        distances = distances[0] if distances else []

        for i, document_id in enumerate(ids):
            results.append(
                SimilarityResult(
                    document_id=document_id,
                    content=documents[i] if documents else "",
                    score=distances[i] if distances else 0.0,
                    metadata=metadatas[i] if metadatas else {},
                )
            )

        return results