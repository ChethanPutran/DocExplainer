
def document_to_dict(doc):

    def section_to_dict(section):
        return {
            "title": section.title,
            "page": section.page_start,
            "paragraphs": [
                {
                    "text": p.raw_text,
                    "sentences": [s.raw_text for s in p.sentences]
                }
                for p in section.paragraphs
            ],
            "images": [img.raw_img for img in section.images],
            "subsections": [section_to_dict(s) for s in section.subsections]
        }

    return {
        "path": doc.path,
        "sections": [section_to_dict(s) for s in doc.sections]
    }



def print_sections(sections, level=0):

    for sec in sections:

        print("  " * level + f"- {sec.title}")

        print_sections(sec.subsections, level + 1)


def display_sentence(sentence, level):

    indent = "  " * level

    print(f"{indent}• Sentence: {sentence.raw_text}")


def display_paragraph(paragraph, level):

    indent = "  " * level

    print(f"{indent}📄 Paragraph:")

    print(f"{indent}{paragraph.raw_text[:120]}...")

    for sentence in paragraph.sentences:
        display_sentence(sentence, level + 1)


def display_section(section, level):

    indent = "  " * level

    print(f"\n{indent}📂 Section: {section.title}")
    print(f"{indent}Page Start: {section.page_start}")

    # paragraphs
    for para in section.paragraphs:
        display_paragraph(para, level + 1)

    for table in section.tables:
        print(f"{indent}  📊 Table: {table.caption[0].raw_text}")

    for eq in section.equations:
        print(f"{indent}  ∑ Equation: {eq.raw_text}")
        
    # images
    for img in section.images:
        print(f"{indent}  🖼 Image: {img.raw_img} (page {img.page})")

    # subsections
    for sub in section.subsections:
        display_section(sub, level + 1)


def display_document(document):

    print(f"\nDocument: {document.path}")
    print(f"Title: {document.title}")
    print("=" * 60)

    for section in document.sections:
        display_section(section, 0)
