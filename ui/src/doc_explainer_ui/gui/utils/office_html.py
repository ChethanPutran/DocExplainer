"""Word and PowerPoint files as simple HTML, so the reader can show them in the same viewer as web pages."""
from __future__ import annotations

from html import escape
from pathlib import Path
from typing import List


def _runs(paragraph) -> str:
    out = []
    for run in paragraph.runs:
        text = escape(run.text or "")
        if not text:
            continue
        if run.bold:
            text = f"<b>{text}</b>"
        if run.italic:
            text = f"<i>{text}</i>"
        if run.underline:
            text = f"<u>{text}</u>"
        out.append(text)
    return "".join(out) or escape(paragraph.text or "")


def _table(rows: List[List[str]]) -> str:
    body = "".join("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f'<table border="1" cellspacing="0" cellpadding="6">{body}</table>'


def docx_to_html(path: str) -> str:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(path)
    parts: List[str] = []
    in_list = False
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, document)
            style = (p.style.name if p.style is not None else "") or ""
            text = _runs(p)
            bullet = style.lower().startswith("list")
            if in_list and not bullet:
                parts.append("</ul>")
                in_list = False
            if not (p.text or "").strip():
                continue
            if style == "Title":
                parts.append(f"<h1>{text}</h1>")
            elif style.startswith("Heading"):
                digits = "".join(c for c in style if c.isdigit())
                parts.append(f"<h{min(int(digits or 1) + 1, 6)}>{text}</h{min(int(digits or 1) + 1, 6)}>")
            elif bullet:
                if not in_list:
                    parts.append("<ul>")
                    in_list = True
                parts.append(f"<li>{text}</li>")
            else:
                parts.append(f"<p>{text}</p>")
        elif tag == "tbl":
            if in_list:
                parts.append("</ul>")
                in_list = False
            parts.append(_table([[c.text.strip() for c in r.cells] for r in Table(child, document).rows]))
    if in_list:
        parts.append("</ul>")
    return "<html><body>" + "\n".join(parts) + "</body></html>"


def pptx_to_html(path: str) -> str:
    from pptx import Presentation

    parts: List[str] = []
    for number, slide in enumerate(Presentation(path).slides, 1):
        title_shape = slide.shapes.title
        title = (title_shape.text_frame.text.strip() if title_shape is not None and title_shape.has_text_frame else "")
        parts.append(f'<p style="color:#888"><small>Slide {number}</small></p>')
        parts.append(f"<h2>{escape(title or f'Slide {number}')}</h2>")
        for shape in slide.shapes:
            if title_shape is not None and shape.shape_id == title_shape.shape_id:
                continue
            if shape.has_text_frame:
                items = [(p.level, "".join(r.text for r in p.runs) or p.text) for p in shape.text_frame.paragraphs]
                items = [(lvl, t.strip()) for lvl, t in items if t.strip()]
                if items:
                    parts.append("<ul>" + "".join(f'<li style="margin-left:{lvl * 18}px">{escape(t)}</li>' for lvl, t in items) + "</ul>")
            elif getattr(shape, "has_table", False) and shape.has_table:
                parts.append(_table([[c.text.strip() for c in r.cells] for r in shape.table.rows]))
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                parts.append(f"<p><i>Notes: {escape(notes)}</i></p>")
        parts.append("<hr/>")
    return "<html><body>" + "\n".join(parts) + "</body></html>"


def to_html(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".docx":
        return docx_to_html(path)
    if ext == ".pptx":
        return pptx_to_html(path)
    raise ValueError(f"Not an Office file: {path}")
