import uuid

from core.document.base import Document
from core.document.cacher import DocumentCache, DocumentCacher
from core.document.manager import DocumentManager
from core.document.builder.processor import HierarchicalDocumentProcessor



def test_document_cache_store_and_get():
    cache = DocumentCache()
    payload = {"k": "v"}

    cache.store("a", payload)

    assert cache.get("a") == payload
    assert cache.get("missing") is None


def test_document_cacher_store_and_retrieve():
    cacher = DocumentCacher()
    payload = [{"concept": "graph"}]

    cacher.cache_document("doc-1", payload)

    assert cacher.retrieve_document("doc-1") == payload
    assert cacher.retrieve_document("unknown") is None


def test_parse_document_extracts_sections_subsections_and_paragraphs(tmp_path):
    sample = (
        "1 Introduction\n"
        "Paragraph one in intro.\n\n"
        "Paragraph two in intro.\n\n"
        "1.1 Background\n"
        "Background para one.\n\n"
        "1.2 Scope\n"
        "Scope para one.\n\n"
        "2 Methods\n"
        "Methods paragraph.\n\n"
        "2.1 Data\n"
        "Data paragraph.\n"
    )
    path = tmp_path / "sample.txt"
    path.write_text(sample, encoding="utf-8")

    text, _, sections = parse_document(str(path))

    assert text == sample
    assert len(sections) == 2

    first = sections[0]
    assert first["title"] == "1 Introduction"
    assert len(first["paragraphs"]) >= 2
    assert len(first["subsections"]) >= 2
    assert first["subsections"][0]["title"] == "1.1 Background"
    assert first["subsections"][0]["paragraphs"][0] == "Background para one."


def test_parse_document_without_headings_returns_single_section(tmp_path):
    content = "Plain paragraph one.\n\nPlain paragraph two."
    path = tmp_path / "plain.md"
    path.write_text(content, encoding="utf-8")

    _, _, sections = parse_document(str(path))

    assert len(sections) == 1
    assert sections[0]["title"] == "Full Document"
    assert sections[0]["paragraphs"] == ["Plain paragraph one.", "Plain paragraph two."]


def test_parse_document_pdf_includes_page_details(monkeypatch):
    fake_text = "1 Intro\nAlpha paragraph.\n\n2 Methods\nBeta paragraph."
    fake_sections = [
        {
            "title": "1 Intro",
            "text": "1 Intro\nAlpha paragraph.",
            "start": 0,
            "end": 24,
            "level": 1,
            "paragraphs": ["Alpha paragraph."],
            "subsections": [],
            "page_start": 1,
            "page_end": 1,
            "source_pages": [1],
        },
        {
            "title": "2 Methods",
            "text": "2 Methods\nBeta paragraph.",
            "start": 26,
            "end": len(fake_text),
            "level": 1,
            "paragraphs": ["Beta paragraph."],
            "subsections": [],
            "page_start": 2,
            "page_end": 2,
            "source_pages": [2],
        },
    ]

    monkeypatch.setattr(
        "src.core.document.document_processing.parse_pdf_with_langchain",
        lambda _path: (fake_text, fake_sections),
    )

    text, _, sections = parse_document("fake.pdf")

    assert "1 Intro" in text
    assert "2 Methods" in text
    assert len(sections) == 2
    assert sections[0]["page_start"] == 1
    assert sections[0]["page_end"] == 1
    assert sections[0]["source_pages"] == [1]
    assert sections[1]["page_start"] == 2
    assert sections[1]["page_end"] == 2
    assert sections[1]["source_pages"] == [2]


def test_document_manager_load_and_retrieve(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("A plain document body.", encoding="utf-8")

    manager = DocumentManager()
    doc_id = manager.load_document(str(path))

    uuid.UUID(doc_id)

    loaded = manager.get_document(doc_id)
    assert loaded is not None
    assert loaded.doc_id == doc_id
    assert loaded.get_text() == "A plain document body."
    assert loaded.get_title() == "doc.txt"
    assert manager.has_document(doc_id) is True
    assert isinstance(loaded.get_index(), list)
    assert len(loaded.get_index()) >= 1
    assert loaded.get_index()[0]["type"] == "section"


def test_document_manager_get_unknown_document_returns_none():
    manager = DocumentManager()

    assert manager.get_document("does-not-exist") is None
    assert manager.has_document("does-not-exist") is False
