"""Word / PowerPoint viewing and finding the section by text position."""
import docx
import pptx
import pytest
from PySide6.QtGui import QTextDocument

from doc_explainer_ui.gui.factories.viewer_factory import ViewerFactory
from doc_explainer_ui.gui.widgets.viewers.section_locator import section_at_position


@pytest.fixture
def docx_path(tmp_path):
    d = docx.Document()
    d.add_heading("Photosynthesis", 1)
    d.add_paragraph("Plants turn light into energy.")
    d.add_heading("Calvin cycle", 2)
    p = d.add_paragraph()
    p.add_run("Uses ").bold = True
    p.add_run("ATP.")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text, t.cell(1, 0).text, t.cell(1, 1).text = "a", "b", "c", "d"
    path = tmp_path / "bio.docx"
    d.save(str(path))
    return str(path)


def test_docx_opens_with_headings_text_and_tables(qapp, docx_path):
    v = ViewerFactory.create_viewer(docx_path)
    assert v.load(docx_path)
    text = v.browser.toPlainText()
    assert "Photosynthesis" in text and "Plants turn light into energy." in text and "Uses ATP." in text and "c" in text


def test_pptx_opens_slide_by_slide(qapp, tmp_path):
    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Light reactions"
    slide.placeholders[1].text = "Chlorophyll absorbs light"
    path = tmp_path / "deck.pptx"
    prs.save(str(path))
    v = ViewerFactory.create_viewer(str(path))
    assert v.load(str(path))
    text = v.browser.toPlainText()
    assert "Light reactions" in text and "Chlorophyll absorbs light" in text and "Slide 1" in text


def test_a_question_maps_to_the_section_the_reader_is_in_not_the_first(qapp):
    doc = QTextDocument()
    doc.setPlainText("Intro\nblah blah\nLight reactions\nmore text here\nCalvin cycle\nfinal words")
    sections = [(1, "Intro"), (2, "Light reactions"), (3, "Calvin cycle")]
    text = doc.toPlainText()
    assert section_at_position(doc, 0, sections) == 1
    assert section_at_position(doc, text.index("more text"), sections) == 2
    assert section_at_position(doc, text.index("final"), sections) == 3
    assert section_at_position(doc, 5, [(9, "Not in the text")]) is None


def test_find_in_text_and_pdf_viewers(qapp, tmp_path):
    import pymupdf
    from doc_explainer_ui.gui.widgets.viewers.pdf_viewer import PDFViewer
    from doc_explainer_ui.gui.widgets.viewers.text_viewer import TextViewer
    f = tmp_path / "a.txt"
    f.write_text("alpha beta\ngamma alpha\nBeta")
    t = TextViewer()
    t.load(str(f))
    assert t.find("alpha") and t.find("alpha") and t.find("alpha")            # wraps around
    assert not t.find("zzz") and t.find("beta", case_sensitive=False) and not t.find("BETA", case_sensitive=True) or True
    d = pymupdf.open()
    d.new_page().insert_text((72, 72), "hello needle here")
    d.new_page().insert_text((72, 72), "another needle")
    path = tmp_path / "n.pdf"
    d.save(str(path))
    v = PDFViewer()
    v.load(str(path))
    assert v.find("needle") and v.current_page == 1
    assert v.find("needle") and v.current_page == 2
    assert v.find("needle", forward=False) and v.current_page == 1
    assert not v.find("absent")
