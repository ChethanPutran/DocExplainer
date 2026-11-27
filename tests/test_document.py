"""The document layer: parsers by extension, one builder, one unified structure."""
import json
import sys

import pytest

from doc_explainer.core.common.structure import (BlockKind, DocumentStructure, EdgeKind, NodeKind)
from doc_explainer.core.document import (ParseError, StructureBuilder, UnsupportedFormat, default_registry,
                                         parse_document)
from doc_explainer.core.document.sentences import split_sentences

import doc_fixtures as fx


def outline(s: DocumentStructure):
    return [(n.depth, n.title) for n in s.sections()]


def paragraphs(s: DocumentStructure):
    return [n.text for n in s.nodes if n.kind is NodeKind.PARAGRAPH and not n.attrs.get("caption")]


# ----------------------------------------------------------------------------- invariants that hold for ANY document
def check_invariants(s: DocumentStructure):
    ords = [n.ordinal for n in s.nodes]
    assert ords == sorted(ords) and len(set(ords)) == len(ords) and ords[0] == 0      # reading order, unique
    assert len({n.id for n in s.nodes}) == len(s.nodes)
    assert s.root.kind is NodeKind.DOCUMENT and s.root.parent_id is None
    for n in s.nodes[1:]:
        parent = s.node(n.parent_id)
        assert parent.ordinal < n.ordinal and n.depth == parent.depth + 1
        assert parent.page_start <= n.page_start and n.page_end <= parent.page_end    # page spans nest
        assert n.page_start <= n.page_end
        if n.kind is NodeKind.SENTENCE:
            assert parent.kind is NodeKind.PARAGRAPH
        if n.kind is NodeKind.SECTION:
            assert parent.kind in (NodeKind.DOCUMENT, NodeKind.SECTION)
    contains = s.edges_of(EdgeKind.CONTAINS)
    assert len(contains) == len(s.nodes) - 1
    for e in s.edges_of(EdgeKind.NEXT):                                                # siblings, forward only
        a, b = s.node(e.src), s.node(e.dst)
        assert a.parent_id == b.parent_id and a.ordinal < b.ordinal
    for sec in s.sections():
        assert s.own_end(sec.id) >= sec.ordinal and s.own_end(sec.id) <= s.subtree_end(sec.id)
    rt = DocumentStructure.from_dict(json.loads(json.dumps(s.to_dict())))             # JSON round-trip, no pickle
    assert [(n.id, n.kind, n.ordinal, n.text) for n in rt.nodes] == [(n.id, n.kind, n.ordinal, n.text) for n in s.nodes]
    stripped = s.without_text()
    assert all(not n.text for n in stripped.nodes) and len(stripped.nodes) == len(s.nodes)


# ----------------------------------------------------------------------------- registry
def test_registry_chooses_the_parser_by_extension_and_lists_them():
    reg = default_registry()
    assert {".pdf", ".html", ".docx", ".md", ".txt", ".pptx"} <= set(reg.supported_extensions())
    assert type(reg.for_path("x/Report.PDF")).__name__ == "PdfParser"
    assert type(reg.for_path("notes.md")).__name__ == "MarkdownParser"


def test_unsupported_and_corrupt_files_are_clean_errors(tmp_path):
    with pytest.raises(UnsupportedFormat, match="Supported"):
        default_registry().parse(tmp_path / "a.xyz")
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"this is not a pdf")
    with pytest.raises(ParseError, match="broken.pdf"):
        default_registry().parse(bad)
    docx_bad = tmp_path / "broken.docx"
    docx_bad.write_bytes(b"nope")
    with pytest.raises(ParseError):
        default_registry().parse(docx_bad)


# ----------------------------------------------------------------------------- one structure for every format
@pytest.mark.parametrize("writer,name", [(fx.write_markdown, "a.md"), (fx.write_html, "a.html"),
                                         (fx.write_docx, "a.docx"), (fx.write_txt, "a.txt")])
def test_the_same_content_gives_the_same_structure_in_every_format(tmp_path, writer, name):
    path = tmp_path / name
    writer(path)
    s = parse_document(path, "d")
    check_invariants(s)
    assert outline(s) == [(1, "Photosynthesis"), (2, "Light reactions"), (2, "2 Calvin cycle")]
    assert paragraphs(s) == [p for _, _, ps in fx.SECTIONS for p in ps]
    first = s.sections()[0]
    assert [c.text for c in s.own_content(first.id)] == fx.SECTIONS[0][2]
    sentences = [n.text for n in s.nodes if n.kind is NodeKind.SENTENCE and n.parent_id == f"{first.id}/p1"]
    assert sentences == ["Plants make food from light.", "Dr. Smith described this in 1900."]   # abbreviation kept whole
    assert s.title == "Photosynthesis"


def test_section_ids_are_stable_paths_and_helpers_work(tmp_path):
    fx.write_markdown(tmp_path / "a.md")
    s = parse_document(tmp_path / "a.md", "d")
    assert [x.id for x in s.sections()] == ["s1", "s1.1", "s1.2"]
    assert s.section_path("s1.2/p1") == ["Photosynthesis", "2 Calvin cycle"]
    assert s.section_of("s1.1/p1/n1").id == "s1.1"
    assert s.text_of("s1.1") == fx.SECTIONS[1][2][0]
    assert fx.SECTIONS[2][2][0] in s.text_of("s1", include_subsections=True)
    # a reader who has finished the first section has not read the later ones
    assert s.own_end("s1") < s.node("s1.1").ordinal < s.own_end("s1.1") < s.node("s1.2").ordinal


# ----------------------------------------------------------------------------- relationships between parts
def test_captions_are_linked_and_references_resolve_to_what_they_mention(tmp_path):
    md = tmp_path / "r.md"
    md.write_text("""# Results

Pigments are listed in Table 1 and shown in Figure 1. Details follow in Section 2.

Table 1: Pigments

| Pigment | Colour |
|---|---|
| Chlorophyll a | green |

![leaf](leaf.png)

Figure 1: A leaf

## 2 Discussion

The data in Table 1 agree.
""", encoding="utf-8")
    s = parse_document(md, "d")
    check_invariants(s)
    table = next(n for n in s.nodes if n.kind is NodeKind.TABLE)
    figure = next(n for n in s.nodes if n.kind is NodeKind.FIGURE)
    assert table.attrs["label"] == "table 1" and table.attrs["caption"] == "Table 1: Pigments"
    assert figure.attrs["label"] == "figure 1"
    caps = {(e.src, e.dst) for e in s.edges_of(EdgeKind.CAPTIONS)}
    assert (next(n.id for n in s.nodes if n.text == "Table 1: Pigments"), table.id) in caps
    refs = {(e.src, e.dst) for e in s.edges_of(EdgeKind.REFERS_TO)}
    intro, discussion = "s1/p1", next(n for n in s.nodes if n.text.startswith("The data")).id
    sec2 = next(x.id for x in s.sections() if x.title == "2 Discussion")
    assert {(intro, table.id), (intro, figure.id), (intro, sec2), (discussion, table.id)} <= refs
    assert not any(src == dst for src, dst in refs)


def test_tables_and_figures_are_nodes_with_text_for_retrieval(tmp_path):
    fx.write_markdown(tmp_path / "x.md")
    (tmp_path / "x.md").write_text("# A\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n![diagram of cells](c.png)\n", encoding="utf-8")
    s = parse_document(tmp_path / "x.md", "d")
    table = next(n for n in s.nodes if n.kind is NodeKind.TABLE)
    assert "| 1 | 2 |" in table.text
    assert next(n for n in s.nodes if n.kind is NodeKind.FIGURE).text == "diagram of cells"
    assert not [n for n in s.nodes if n.kind is NodeKind.SENTENCE and s.node(n.parent_id).kind is not NodeKind.PARAGRAPH]


# ----------------------------------------------------------------------------- awkward documents
def test_a_document_without_headings_is_cut_into_synthetic_sections(tmp_path):
    txt = tmp_path / "flat.txt"
    para = "This is a sentence about plants and light. " * 30
    txt.write_text("\n\n".join([para] * 8), encoding="utf-8")
    s = StructureBuilder(fallback_window_words=400).build(default_registry().parse(txt), "d")
    check_invariants(s)
    titles = [x.title for x in s.sections()]
    assert len(titles) >= 3 and titles[0] == "Part 1" and all(x.attrs.get("synthetic") for x in s.sections())


def test_content_before_the_first_heading_goes_into_a_preamble(tmp_path):
    md = tmp_path / "p.md"
    md.write_text("Some opening words come first.\n\n# Chapter\n\nBody text here.\n", encoding="utf-8")
    s = parse_document(md, "d")
    check_invariants(s)
    assert [x.title for x in s.sections()] == ["Preamble", "Chapter"]
    assert s.sections()[0].depth == s.sections()[1].depth == 1


def test_heading_levels_are_normalised_and_cannot_skip_a_level(tmp_path):
    md = tmp_path / "l.md"
    md.write_text("### Deep start\n\ntext one.\n\n###### Jump\n\ntext two.\n\n### Back\n\ntext three.\n", encoding="utf-8")
    s = parse_document(md, "d")
    check_invariants(s)
    assert outline(s) == [(1, "Deep start"), (2, "Jump"), (1, "Back")]


def test_empty_and_whitespace_documents_do_not_crash(tmp_path):
    for name, content in (("e.md", ""), ("w.txt", "  \n\n \n"), ("h.html", "<html><body></body></html>")):
        (tmp_path / name).write_text(content, encoding="utf-8")
        s = parse_document(tmp_path / name, "d")
        assert s.root.kind is NodeKind.DOCUMENT and len(s.nodes) >= 1
        check_invariants(s)


def test_text_conventions_numbered_and_uppercase_headings_and_pages(tmp_path):
    txt = tmp_path / "c.txt"
    txt.write_text("INTRODUCTION\n\nFirst paragraph here.\n\n1.1 Scope\n\nSecond paragraph.\n\f\n- point one\n- point two\n", encoding="utf-8")
    s = parse_document(txt, "d")
    check_invariants(s)
    assert [x.title for x in s.sections()] == ["INTRODUCTION", "1.1 Scope"]
    assert s.page_count == 2
    assert [n.attrs.get("list", False) for n in s.nodes if n.kind is NodeKind.PARAGRAPH][-2:] == [True, True]
    assert s.node("s1.1/p2").page_start == 2 and s.section_at_page(2).title == "1.1 Scope"


# ----------------------------------------------------------------------------- html / docx / pptx specifics
def test_html_ignores_navigation_and_scripts_and_reads_tables_and_figures(tmp_path):
    h = tmp_path / "w.html"
    h.write_text("""<html><head><title>T</title><style>p{}</style></head><body><nav><a>skip</a></nav>
    <h1>Main</h1><p>Real <b>bold</b> text.</p><script>var x=1;</script>
    <table><caption>Table 1: Data</caption><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2</td></tr></table>
    <figure><img src="f.png" alt="a chart"><figcaption>Figure 1: Chart</figcaption></figure>
    <ul><li>one</li><li>two</li></ul><footer>skip too</footer></body></html>""", encoding="utf-8")
    s = parse_document(h, "d")
    check_invariants(s)
    texts = " ".join(n.text for n in s.nodes)
    assert "skip" not in texts and "var x" not in texts and "Real bold text." in texts
    assert next(n for n in s.nodes if n.kind is NodeKind.TABLE).attrs["label"] == "table 1"
    assert next(n for n in s.nodes if n.kind is NodeKind.FIGURE).attrs["label"] == "figure 1"
    assert [n.text for n in s.nodes if n.attrs.get("list")] == ["one", "two"]


def test_docx_reads_lists_tables_images_and_page_breaks(tmp_path):
    import docx
    from docx.enum.text import WD_BREAK
    d = docx.Document()
    d.add_heading("Report", 1)
    d.add_paragraph("Intro text.")
    d.add_paragraph("item a", style="List Bullet")
    t = d.add_table(rows=2, cols=2)
    for r, row in enumerate([["k", "v"], ["x", "1"]]):
        for c, val in enumerate(row):
            t.cell(r, c).text = val
    d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    d.add_heading("Next", 2)
    d.add_paragraph("On page two.")
    d.save(str(tmp_path / "r.docx"))
    s = parse_document(tmp_path / "r.docx", "d")
    check_invariants(s)
    assert next(n for n in s.nodes if n.kind is NodeKind.TABLE).text.startswith("| k | v |")
    assert [n.attrs.get("list", False) for n in s.nodes if n.kind is NodeKind.PARAGRAPH][:2] == [False, True]
    assert s.page_count == 2 and s.node("s1.1").page_start == 2


def test_pptx_every_slide_is_a_section(tmp_path):
    from pptx import Presentation
    prs = Presentation()
    for title, body in [("Cells", "Cells are small."), ("Energy", "Energy flows.")]:
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
    prs.save(str(tmp_path / "s.pptx"))
    s = parse_document(tmp_path / "s.pptx", "d")
    check_invariants(s)
    assert outline(s) == [(1, "Cells"), (1, "Energy")] and s.page_count == 2
    assert s.section_at_page(2).title == "Energy" and "Energy flows." in s.text_of("s2")


# ----------------------------------------------------------------------------- PDF
@pytest.fixture
def pdf(tmp_path):
    fx.make_pdf(str(tmp_path / "a.pdf"))
    return tmp_path / "a.pdf"


def test_pdf_structure_headings_lists_tables_figures(pdf):
    s = parse_document(pdf, "d")
    check_invariants(s)
    assert outline(s) == [(1, "Photosynthesis"), (2, "1 Introduction"), (3, "1.1 Background"), (2, "2 Calvin cycle"), (2, "3 Conclusion")]
    assert s.page_count == 3 and s.section_at_page(3).title == "3 Conclusion"
    assert [n.text for n in s.nodes if n.attrs.get("list")] == ["Light absorption", "Electron transport"]
    table = next(n for n in s.nodes if n.kind is NodeKind.TABLE)
    assert "Chlorophyll a" in table.text and table.attrs["label"] == "table 1" and table.page_start == 2
    figure = next(n for n in s.nodes if n.kind is NodeKind.FIGURE)
    assert figure.attrs["caption"] == "Figure 1: A green leaf"
    assert {e.dst for e in s.edges_of(EdgeKind.REFERS_TO)} >= {figure.id, table.id}    # "See Figure 1 and Table 1."


def test_pdf_drops_running_headers_page_numbers_and_rejoins_pages(pdf):
    s = parse_document(pdf, "d")
    texts = [n.text for n in s.nodes]
    assert not any("Photosynthesis Handbook" in t or t.strip().startswith("Page ") for t in texts)
    joined = next(t for t in texts if t.startswith("Chlorophyll absorbs light, then"))
    assert "photosystems and ends as stored sugar" in joined                          # sentence that crossed the page break
    assert next(n for n in s.nodes if n.text == joined).page_start == 1


def test_pdf_outline_levels_win_over_font_size(tmp_path):
    # By font size "2 Calvin cycle" (16pt) is a sibling of "1 Introduction" (16pt). This outline says it
    # sits UNDER Introduction, next to Background, so only the outline can produce that structure.
    fx.make_pdf(str(tmp_path / "o.pdf"), outline=[[1, "Introduction", 1], [2, "Background", 1],
                                                  [2, "Calvin cycle", 2], [1, "Conclusion", 3]])
    s = parse_document(tmp_path / "o.pdf", "d")
    check_invariants(s)
    parent = {x.title: s.node(x.parent_id).title for x in s.sections()}
    assert parent["2 Calvin cycle"] == "1 Introduction"
    assert parent["1.1 Background"] == "1 Introduction"
    assert parent["3 Conclusion"] != "1 Introduction"
    fx.make_pdf(str(tmp_path / "n.pdf"))                               # contrast: no outline -> font sizes decide
    n = parse_document(tmp_path / "n.pdf", "d")
    assert {x.title: n.node(x.parent_id).title for x in n.sections()}["2 Calvin cycle"] != "1 Introduction"


# ----------------------------------------------------------------------------- sentences
@pytest.mark.parametrize("text,expected", [
    ("Dr. Smith measured 3.14 units, e.g. at A. Turing's lab. It worked!", 2),
    ("One. Two. Three.", 3),
    ("No terminal punctuation", 1),
    ("", 0),
    ("What? Really? Yes.", 3),
])
def test_sentence_splitting(text, expected):
    assert len(split_sentences(text)) == expected


def test_a_scanned_pdf_is_read_with_ocr(tmp_path):
    import pymupdf
    from doc_explainer.core.document import ocr
    if not ocr.available():
        pytest.skip("no OCR engine installed")
    text = pymupdf.open()
    page = text.new_page()
    page.insert_text((72, 100), "Photosynthesis converts light into chemical energy.", fontsize=18)
    png = page.get_pixmap(dpi=200).tobytes("png")
    scan = pymupdf.open()
    scan.new_page(width=page.rect.width, height=page.rect.height).insert_image(page.rect, stream=png)
    path = tmp_path / "scan.pdf"
    scan.save(str(path))
    parsed = default_registry().parse(path)
    body = " ".join(b.text for b in parsed.blocks).lower()
    assert "photosynthesis" in body and "energy" in body
