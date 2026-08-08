import numpy as np
import pytest

from src.core.document.builder.processor import LangChainEmbeddingWrapper
from src.core.knowledge.extraction.canonicalization.clusterer import ConceptClusterer
from src.models.text import EmbeddingModel


class _ColumnVectorModel:
    def encode(self, _text):
        return np.array([[0.1], [0.2], [0.3]], dtype=np.float32)


class _BatchStyleModel:
    def encode(self, _text):
        return np.array([[0.1, 0.2, 0.3]], dtype=np.float32)


def test_embedding_model_fallback_returns_flat_vector():
    embedding = EmbeddingModel().encode("hello world")

    assert embedding.shape == (16,)


def test_embedding_model_fallback_returns_batch_matrix():
    embeddings = EmbeddingModel().encode(["hello world", "goodbye world"])

    assert embeddings.shape == (2, 16)


def test_langchain_wrapper_flattens_column_vector_embeddings():
    wrapper = LangChainEmbeddingWrapper(_ColumnVectorModel())

    embeddings = wrapper.embed_documents(["alpha", "beta"])

    assert embeddings == [
        pytest.approx([0.1, 0.2, 0.3]),
        pytest.approx([0.1, 0.2, 0.3]),
    ]


def test_langchain_wrapper_flattens_single_row_batch_embeddings():
    wrapper = LangChainEmbeddingWrapper(_BatchStyleModel())

    embedding = wrapper.embed_query("alpha")

    assert embedding == pytest.approx([0.1, 0.2, 0.3])


def test_concept_clusterer_uses_one_embedding_row_per_concept():
    clusterer = ConceptClusterer(EmbeddingModel())

    embeddings = clusterer._get_embeddings(["alpha", "beta", "gamma"])

    assert embeddings.shape == (3, 16)
