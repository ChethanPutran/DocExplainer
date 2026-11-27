"""Which section of a page-less document (text, HTML, Word, PowerPoint) is the reader in? Found by where each section's
heading appears in the text, so no page number is needed."""
from typing import List, Optional, Tuple

from PySide6.QtGui import QTextDocument


def section_at_position(document: QTextDocument, position: int, sections: List[Tuple[int, str]]) -> Optional[int]:
    """``sections`` = (id, title) in reading order. The answer is the last section whose heading starts at or before
    ``position``; headings are searched in order, so a title that repeats does not jump back to its first use."""
    found, start = None, 0
    for section_id, title in sections:
        title = (title or "").strip()[:80]
        if not title:
            continue
        cursor = document.find(title, start)
        if cursor.isNull():
            continue
        if cursor.selectionStart() > position:
            break
        found, start = section_id, cursor.selectionEnd()
    return found
