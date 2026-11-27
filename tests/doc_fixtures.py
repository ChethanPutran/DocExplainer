import pymupdf

def make_pdf(path, outline=None):
    doc = pymupdf.open()
    W, H = 595, 842
    def header_footer(page, n):
        page.insert_text((72, 40), "Photosynthesis Handbook", fontsize=9)
        page.insert_text((280, H - 30), f"Page {n}", fontsize=9)
    def text(page, y, s, size=11, bold=False):
        page.insert_text((72, y), s, fontsize=size, fontname="hebo" if bold else "helv")
        return y + size * 1.5
    p1 = doc.new_page(width=W, height=H); header_footer(p1, 1)
    y = text(p1, 100, "Photosynthesis", 24, True)
    y = text(p1, y + 14, "1 Introduction", 16, True)
    for line in ["Photosynthesis converts light energy into chemical energy.", "Plants, algae and some bacteria perform it every day."]:
        y = text(p1, y, line)
    y += 12
    y = text(p1, y, "1.1 Background", 13, True)
    for line in ["Chlorophyll absorbs red and blue light. The reaction takes place", "in the chloroplast, which has an inner and an outer membrane."]:
        y = text(p1, y, line)
    y += 12
    y = text(p1, y, "- Light absorption")
    y = text(p1, y, "- Electron transport")
    y += 12
    y = text(p1, y, "Chlorophyll absorbs light, then the energy moves through the")   # continues on the next page
    # image
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 120), False); pix.set_rect(pix.irect, (60, 160, 70))
    p2 = doc.new_page(width=W, height=H); header_footer(p2, 2)
    y2 = text(p2, 100, "photosystems and ends as stored sugar. See Figure 1 and Table 1.")
    p2.insert_image(pymupdf.Rect(72, y2 + 10, 272, y2 + 130), pixmap=pix)
    y2 = text(p2, y2 + 150, "Figure 1: A green leaf")
    y2 += 20
    y2 = text(p2, y2, "2 Calvin cycle", 16, True)
    y2 = text(p2, y2 + 4, "The Calvin cycle builds sugar from carbon dioxide.")
    # a ruled table
    top = y2 + 30
    cols = [72, 200, 330]; rows = [top, top + 24, top + 48, top + 72]
    for r in rows: p2.draw_line((cols[0], r), (cols[-1], r))
    for c in cols: p2.draw_line((c, rows[0]), (c, rows[-1]))
    for ri, cells in enumerate([["Pigment", "Colour"], ["Chlorophyll a", "green"], ["Carotene", "orange"]]):
        for ci, cell in enumerate(cells): p2.insert_text((cols[ci] + 6, rows[ri] + 16), cell, fontsize=10)
    text(p2, rows[-1] + 22, "Table 1: Common pigments")
    p3 = doc.new_page(width=W, height=H); header_footer(p3, 3)
    y3 = text(p3, 100, "3 Conclusion", 16, True)
    text(p3, y3 + 4, "Light becomes sugar. That is the whole story.")
    if outline:
        doc.set_toc(outline)
    doc.save(path); doc.close()


# ----------------------------------------------------------------------------- the same content, in every format
SECTIONS = [
    ("Photosynthesis", 1, ["Plants make food from light. Dr. Smith described this in 1900.", "It happens in the chloroplast."]),
    ("Light reactions", 2, ["Chlorophyll absorbs red and blue light. See Section 2 for the next step."]),
    ("2 Calvin cycle", 2, ["The Calvin cycle builds sugar from carbon dioxide."]),
]


def write_markdown(path):
    out = []
    for title, level, paras in SECTIONS:
        out.append(f"{'#' * level} {title}\n")
        out += [p + "\n" for p in paras]
    path.write_text("\n".join(out), encoding="utf-8")


def write_html(path):
    out = ["<html><head><title>Photosynthesis</title></head><body><nav>menu</nav>"]
    for title, level, paras in SECTIONS:
        out.append(f"<h{level}>{title}</h{level}>")
        out += [f"<p>{p}</p>" for p in paras]
    out.append("<footer>copyright</footer></body></html>")
    path.write_text("\n".join(out), encoding="utf-8")


def write_docx(path):
    import docx
    d = docx.Document()
    for title, level, paras in SECTIONS:
        d.add_heading(title, level=level)
        for p in paras:
            d.add_paragraph(p)
    d.save(str(path))


def write_txt(path):
    out = []
    for title, level, paras in SECTIONS:
        out.append(title)
        out.append(("=" if level == 1 else "-") * len(title))
        out.append("")
        for p in paras:
            out += [p, ""]
    path.write_text("\n".join(out), encoding="utf-8")
