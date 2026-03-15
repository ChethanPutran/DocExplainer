from __future__ import annotations

import re

class StructureDetector:

    def __init__(self):

        self.section_pattern = re.compile(r"^(\d+(?:\.\d+)*)\s+")
        self.figure_pattern = re.compile(r"^(Figure|Fig\.)\s*\d+", re.I)
        self.table_pattern = re.compile(r"^(Table)\s*\d+", re.I)
        self.equation_pattern = re.compile(r"^\(?\d+\)?$")
        self.abstract_pattern = re.compile(r"^abstract$", re.I)

        self.font_levels = []
        self.body_font = None

    # ------------------------------------
    # Analyze font hierarchy of document
    # ------------------------------------
    def analyze_fonts(self, doc):

        sizes = []

        for page in doc.pages():

            blocks = page.get_text("dict")["blocks"]

            for b in blocks:

                if b["type"] != 0:
                    continue

                for line in b["lines"]:
                    for span in line["spans"]:
                        sizes.append(round(span["size"], 1))

        unique = sorted(set(sizes), reverse=True)

        # largest fonts first
        self.font_levels = unique

        # body font is most frequent
        freq = {}

        for s in sizes:
            freq[s] = freq.get(s, 0) + 1

        self.body_font = max(freq, key=freq.get)

    def get_font_level(self, size):

        if size not in self.font_levels:
            return -1

        idx = self.font_levels.index(size)

        if idx == 0:
            return -1  # document title

        return idx - 1

    def classify_block(self, text, span, page_num, bbox):

        text = re.sub(r"\s+", " ", text.strip())
        size = round(span["size"], 1)
        font = span["font"]
        flags = span["flags"]

        is_bold = "Bold" in font or (flags & 16)
        is_caps = text.isupper()

        if not text:
            return None
        
        # ----------------------------
        # TABLE
        # ----------------------------
        if self.table_pattern.match(text):
            return {"type": "table_caption"}

        # ----------------------------
        # ABSTRACT
        # ----------------------------
        if self.abstract_pattern.match(text):
            return {"type": "abstract", "level": 0}

        # ----------------------------
        # FIGURE
        # ----------------------------
        if self.figure_pattern.match(text):
            return {"type": "figure_caption"}

        # ----------------------------
        # EQUATION
        # ----------------------------
        if self.equation_pattern.match(text):
            return {"type": "equation"}

        # ----------------------------
        # NUMBERED SECTIONS
        # ----------------------------
        match = self.section_pattern.match(text)

        if match:

            sec = match.group(1)

            return {
                "type": "section",
                "level": sec.count("."),
                "title": re.sub(r"^\d+(?:\.\d+)*\s+", "", text)
            }

        # ----------------------------
        # DOCUMENT TITLE DETECTION
        # ----------------------------
        if (
            page_num == 0 and
            size == max(self.font_levels) and
            bbox[1] < 200 and
            len(text) > 15
        ):
            return {
                "type": "document_title",
                "title": text
            }
        # ---------------------------------
        # SECTION HEADING DETECTION BY FONT
        # ---------------------------------
        if size > self.body_font\
            and len(text) < 120\
            and (
                is_bold
                or is_caps
                or text.endswith(":")
            ):

            level = self.get_font_level(size)

            return {
                "type": "section",
                "level": max(level, 0),
                "title": text
            }

        # ----------------------------
        # PARAGRAPH
        # ----------------------------
        return {"type": "paragraph"}

