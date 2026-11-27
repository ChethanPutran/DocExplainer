"""Chunking, embeddings, the vector index, the structure-only graph store and the indexer."""
import threading
import types

import pytest

from doc_explainer.core.common.ports import Chunk
from doc_explainer.core.common.structure import EdgeKind, NodeKind
from doc_explainer.core.document import parse_document
from doc_explainer.core.indexing import (ChromaVectorIndex, Indexer, JsonContentStore, SqliteStructureStore,
                                         SentenceTransformerEmbedder, chunk_structure, embedding_text)
from doc_explainer.core.indexing import chunker, embedder as embedder_module

from fakes import HashingEmbedder

BOOK = """# Photosynthesis

Plants make food from light. This happens in the chloroplast of every leaf cell.

## Light reactions

Chlorophyll absorbs red and blue light and splits water, releasing oxygen. See Table 1.

Table 1: Pigments

| Pigment | Colour |
|---|---|
| Chlorophyll a | green |

## Calvin cycle

The Calvin cycle builds sugar from carbon dioxide using the energy stored by the light reactions.

![leaf](leaf.png)

Figure 1: A leaf cross-section
"""


@pytest.fixture
def structure(tmp_path):
    (tmp_path / "b.md").write_text(BOOK, encoding="utf-8")
    return parse_document(tmp_path / "b.md", "book")


# ----------------------------------------------------------------------------- chunker
def test_every_paragraph_is_covered_and_no_chunk_crosses_a_section(structure):
    chunks = chunk_structure(structure)
    covered = "\n".join(c.text for c in chunks)
    for n in structure.nodes:
        if n.kind is NodeKind.PARAGRAPH and not n.attrs.get("caption"):
            assert n.text in covered
    for c in chunks:
        assert structure.section_of(c.node_id).id == c.section_id
        assert structure.node(c.node_id).ordinal == c.ordinal
    ords = [c.ordinal for c in chunks]
    assert ords == sorted(ords) and len({c.id for c in chunks}) == len(chunks)


def test_captions_are_folded_into_their_table_and_figure_not_indexed_twice(structure):
    chunks = chunk_structure(structure)
    table = next(c for c in chunks if c.kind == "table")
    figure = next(c for c in chunks if c.kind == "figure")
    assert "Table 1: Pigments" in table.text and "Chlorophyll a" in table.text
    assert "Figure 1: A leaf cross-section" in figure.text
    assert sum("Table 1: Pigments" in c.text for c in chunks) == 1


def test_short_paragraphs_merge_and_long_ones_split_at_sentences(tmp_path):
    short = "\n\n".join(f"Tiny paragraph number {i}." for i in range(6))
    sentence = "This sentence is long enough to matter for the size limit of a chunk. "
    long_paragraph = sentence * 40                       # ~2,800 chars, one paragraph
    (tmp_path / "x.md").write_text(f"# A\n\n{short}\n\n## B\n\n{long_paragraph}\n", encoding="utf-8")
    s = parse_document(tmp_path / "x.md", "d")
    chunks = chunk_structure(s)
    a = [c for c in chunks if c.section_id == "s1"]
    b = [c for c in chunks if c.section_id == "s1.1"]
    assert len(a) == 1 and "Tiny paragraph number 5." in a[0].text          # six tiny paragraphs -> one chunk
    assert len(b) >= 3 and all(len(c.text) <= chunker.MAX_CHARS + 80 for c in b)
    assert all(c.text.rstrip().endswith(".") for c in b)                       # cut on sentence boundaries
    assert len({c.id for c in b}) == len(b)


def test_embedding_text_carries_the_section_path(structure):
    c = next(c for c in chunk_structure(structure) if "Calvin cycle builds" in c.text)
    assert c.section_path == "Photosynthesis > Calvin cycle"
    assert embedding_text(c).startswith("Photosynthesis > Calvin cycle\n")


def test_an_empty_document_has_no_chunks(tmp_path):
    (tmp_path / "e.md").write_text("", encoding="utf-8")
    assert chunk_structure(parse_document(tmp_path / "e.md", "e")) == []


# ----------------------------------------------------------------------------- vector index (Chroma)
def mk(i, text, ordinal, section="s1"):
    return Chunk(id=f"c{i}", text=text, node_id=f"n{i}", kind="paragraph", ordinal=ordinal, page=1,
                 section_id=section, section_path="A")


@pytest.fixture
def vec(tmp_path):
    return ChromaVectorIndex(tmp_path / "chroma")


def test_search_returns_the_closest_chunk_with_its_position(vec):
    e = HashingEmbedder()
    chunks = [mk(1, "photosynthesis converts light energy", 5), mk(2, "the french revolution began in 1789", 9)]
    vec.add("d", chunks, e.embed([c.text for c in chunks]))
    hits = vec.search("d", e.embed(["light energy photosynthesis"])[0], k=2)
    assert [h.chunk.id for h in hits] == ["c1", "c2"] and hits[0].score > hits[1].score
    assert (hits[0].chunk.ordinal, hits[0].chunk.page, hits[0].chunk.section_path) == (5, 1, "A")


def test_search_can_be_limited_to_what_has_been_read(vec):
    e = HashingEmbedder()
    chunks = [mk(1, "early chapter about cells", 3), mk(2, "late chapter about cells and quantum cells", 40)]
    vec.add("d", chunks, e.embed([c.text for c in chunks]))
    q = e.embed(["quantum cells"])[0]
    assert vec.search("d", q, k=5)[0].chunk.id == "c2"
    only_read = vec.search("d", q, k=5, max_ordinal=10)
    assert [h.chunk.id for h in only_read] == ["c1"]                        # the later chapter cannot leak in
    assert vec.search("d", q, k=5, max_ordinal=0) == []


def test_documents_are_isolated_persisted_and_deletable(tmp_path):
    e = HashingEmbedder()
    one = ChromaVectorIndex(tmp_path / "c")
    one.add("a", [mk(1, "alpha text", 1)], e.embed(["alpha text"]))
    one.add("b", [mk(1, "beta text", 1)], e.embed(["beta text"]))
    assert one.count("a") == one.count("b") == 1 and one.count("never") == 0
    again = ChromaVectorIndex(tmp_path / "c")                              # a fresh client sees the data
    assert again.search("a", e.embed(["alpha"])[0])[0].chunk.text == "alpha text"
    assert again.search("a", e.embed(["beta"])[0])[0].chunk.text == "alpha text"   # never document b's
    again.delete("a")
    assert again.count("a") == 0 and again.search("a", e.embed(["alpha"])[0]) == [] and again.count("b") == 1
    again.delete("a")                                                       # deleting twice is harmless


def test_odd_document_ids_and_input_errors(vec):
    e = HashingEmbedder()
    for doc_id in ("my doc/with:odd*chars", "ü" * 80, "a"):
        vec.add(doc_id, [mk(1, "text here", 1)], e.embed(["text here"]))
        assert vec.count(doc_id) == 1
    with pytest.raises(ValueError):
        vec.add("d", [mk(1, "x", 1)], [])
    vec.add("d", [mk(1, "same id", 1)], e.embed(["same id"]))
    vec.add("d", [mk(1, "same id updated", 1)], e.embed(["same id updated"]))   # upsert, not duplicate
    assert vec.count("d") == 1


# ----------------------------------------------------------------------------- structure-only graph store
@pytest.fixture
def store(tmp_path):
    return SqliteStructureStore(tmp_path / "structure.db")


def test_graph_store_answers_structural_questions(store, structure):
    store.save(structure.without_text())
    assert store.has("book") and not store.has("other")
    assert [n.id for n in store.nodes("book", "section")] == ["s1", "s1.1", "s1.2"]
    assert [n.id for n in store.children("book", "s1") if n.kind == "section"] == ["s1.1", "s1.2"]
    assert [n.title for n in store.ancestors("book", "s1.2/p1/n1") if n.kind == "section"] == ["Photosynthesis", "Calvin cycle"]
    assert store.ancestors("book", "s1.2/p1/n1")[0].kind == "document"
    assert store.node("book", "s1.2").title == "Calvin cycle" and store.node("book", "nope") is None
    assert {(s, d) for s, d, _ in store.edges("book", EdgeKind.CAPTIONS)} == \
           {(e.src, e.dst) for e in structure.edges_of(EdgeKind.CAPTIONS)}
    assert {k for _, _, k in store.edges("book")} == {e.kind.value for e in structure.edges}


def test_graph_store_holds_structure_only_no_body_text(store, structure):
    store.save(structure)                                   # even if handed the full text ...
    cells = " ".join(store.all_text_cells())
    assert "Chlorophyll absorbs red and blue light" not in cells and "Plants make food" not in cells
    assert "| Pigment |" not in cells and "Figure 1: A leaf" not in cells
    assert "Calvin cycle" in cells and "Photosynthesis" in cells        # section titles are structural labels


def test_graph_store_page_lookup_replace_and_delete(store, tmp_path):
    (tmp_path / "p.txt").write_text("INTRODUCTION\n\nfirst page text.\n\f\nCONCLUSION\n\nsecond page text.\n", encoding="utf-8")
    s = parse_document(tmp_path / "p.txt", "pg")
    store.save(s)
    assert store.section_at_page("pg", 1).title == "INTRODUCTION" and store.section_at_page("pg", 2).title == "CONCLUSION"
    assert store.section_at_page("pg", 9).title == "CONCLUSION"                 # beyond the end: the last section
    store.save(s)                                                         # saving again replaces, not duplicates
    assert len(store.nodes("pg")) == len(s.nodes)
    store.delete("pg")
    assert not store.has("pg") and store.nodes("pg") == [] and store.edges("pg") == []


def test_graph_store_is_safe_for_concurrent_use(store, structure):
    errors = []

    def work(i):
        try:
            for _ in range(5):
                s = structure.without_text()
                s.doc_id = f"d{i}"
                store.save(s)
                assert len(store.nodes(f"d{i}")) == len(structure.nodes)
        except Exception as e:
            errors.append(e)

    ts = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    [t.start() for t in ts]
    [t.join(20) for t in ts]
    assert errors == []


# ----------------------------------------------------------------------------- content store
def test_content_store_round_trips_the_full_structure_as_json(tmp_path, structure):
    cs = JsonContentStore(tmp_path / "content")
    assert not cs.exists("book")
    cs.save(structure)
    assert cs.exists("book")
    back = cs.load("book")
    assert [(n.id, n.text, n.ordinal) for n in back.nodes] == [(n.id, n.text, n.ordinal) for n in structure.nodes]
    assert (tmp_path / "content" / "book.json").read_text().startswith("{")           # JSON, never pickle
    cs.save(structure)                                                                # overwrite is fine
    cs.delete("book")
    cs.delete("book")
    assert not cs.exists("book")


# ----------------------------------------------------------------------------- indexer
@pytest.fixture
def indexer(tmp_path):
    return Indexer(HashingEmbedder(), ChromaVectorIndex(tmp_path / "v"), SqliteStructureStore(tmp_path / "s.db"),
                   JsonContentStore(tmp_path / "c"))


def test_indexing_fills_all_three_stores_and_search_finds_the_right_part(indexer, structure):
    steps = []
    report = indexer.index(structure, progress=lambda stage, f: steps.append((stage, f)))
    assert report.sections == 3 and report.nodes == len(structure.nodes) and report.chunks >= 3
    assert indexer.vectors.count("book") == report.chunks
    assert indexer.structure_store.has("book") and indexer.content_store.exists("book")
    q = indexer.embedder.embed(["carbon dioxide sugar calvin"])[0]
    assert "Calvin cycle builds sugar" in indexer.vectors.search("book", q, k=1)[0].chunk.text
    fractions = [f for _, f in steps]
    assert fractions == sorted(fractions) and steps[0][0] == "storing structure" and steps[-1] == ("indexed", 1.0)


def test_a_reader_part_way_through_cannot_retrieve_later_sections(indexer, structure):
    indexer.index(structure)
    position = structure.own_end("s1.1")                    # has finished the "Light reactions" section
    q = indexer.embedder.embed(["calvin cycle builds sugar from carbon dioxide"])[0]
    texts = " ".join(h.chunk.text for h in indexer.vectors.search("book", q, k=10, max_ordinal=position))
    assert "Calvin cycle builds sugar" not in texts and "Chlorophyll absorbs" in texts
    assert "Calvin cycle builds sugar" in " ".join(h.chunk.text for h in indexer.vectors.search("book", q, k=10))


def test_reindexing_replaces_and_remove_clears_everything(indexer, structure):
    first = indexer.index(structure)
    second = indexer.index(structure)
    assert indexer.vectors.count("book") == second.chunks == first.chunks      # no duplicates
    indexer.remove("book")
    assert indexer.vectors.count("book") == 0 and not indexer.structure_store.has("book") \
        and not indexer.content_store.exists("book")


# ----------------------------------------------------------------------------- the real embedding model
def test_pick_device_falls_back_to_cpu_for_an_unsupported_gpu(monkeypatch):
    def fake(cc, archs, available=True):
        cuda = types.SimpleNamespace(is_available=lambda: available, get_device_capability=lambda: cc,
                                     get_arch_list=lambda: list(archs))
        monkeypatch.setitem(__import__("sys").modules, "torch", types.SimpleNamespace(cuda=cuda))
    monkeypatch.delenv("DOC_EXPLAINER_EMBED_DEVICE", raising=False)
    fake((6, 1), ["sm_75", "sm_80", "compute_120"]);  assert embedder_module.pick_device() == "cpu"   # the MX350 case
    fake((8, 6), ["sm_75", "sm_86"]);                 assert embedder_module.pick_device() == "cuda"
    fake((6, 1), ["sm_75"], available=False);         assert embedder_module.pick_device() == "cpu"
    monkeypatch.setenv("DOC_EXPLAINER_EMBED_DEVICE", "cpu")
    fake((8, 6), ["sm_86"]);                          assert embedder_module.pick_device() == "cpu"


def test_a_model_that_cannot_load_fails_loudly_never_silently(monkeypatch):
    import sentence_transformers

    def boom(*a, **k):
        raise OSError("model files missing")
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", boom)
    with pytest.raises(OSError, match="model files missing"):
        SentenceTransformerEmbedder("anything").embed(["text"])


def test_the_real_model_retrieves_by_meaning_not_just_words(tmp_path, structure):
    emb = SentenceTransformerEmbedder("all-MiniLM-L6-v2", device="cpu")
    try:
        vectors = emb.embed(["probe"])
    except Exception as e:                                  # model not available offline here
        pytest.skip(f"embedding model unavailable: {e}")
    assert len(vectors[0]) == emb.dimension == 384 and abs(sum(x * x for x in vectors[0]) - 1.0) < 1e-3
    ix = Indexer(emb, ChromaVectorIndex(tmp_path / "v"), SqliteStructureStore(tmp_path / "s.db"), JsonContentStore(tmp_path / "c"))
    ix.index(structure)
    top = ix.vectors.search("book", emb.embed(["How do green plants turn sunlight into sugar?"])[0], k=1)[0]
    assert any(w in top.chunk.text.lower() for w in ("calvin", "light", "plants make food"))
    unrelated = ix.vectors.search("book", emb.embed(["the stock market crashed in 1929"])[0], k=1)[0]
    assert unrelated.score < top.score
