"""OCR for scanned pages (a page that is only a picture of text). Optional: if the engine is not installed or fails,
the page simply stays empty and the document is reported as having no readable text, as before.

Engine: RapidOCR (``pip install rapidocr-onnxruntime``): runs offline, no system packages needed.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)
_engine = None
_unavailable = False
SCALES = (2.0, 3.0, 1.0)         # the detector is sensitive to image size: if one render finds nothing, try another

Line = Tuple[Tuple[float, float, float, float], str]       # (x0, y0, x1, y1) in PDF points, text


def _get_engine():
    global _engine, _unavailable
    if _engine is None and not _unavailable:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _engine = RapidOCR()
        except Exception as e:  # noqa: BLE001
            _unavailable = True
            logger.warning("OCR is not available (%s); scanned pages will have no text", e)
    return _engine


def available() -> bool:
    return _get_engine() is not None


def ocr_page(page) -> List[Line]:
    """Text lines on a rendered PDF page, top to bottom. Empty when OCR is unavailable or finds nothing."""
    engine = _get_engine()
    if engine is None:
        return []
    try:
        import pymupdf
    except Exception:  # noqa: BLE001
        return []
    lines: List[Line] = []
    for scale in SCALES:
        try:
            pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csRGB)
            result, _ = engine(pix.tobytes("png"))
        except Exception:  # noqa: BLE001
            logger.warning("OCR failed on a page", exc_info=True)
            return []
        for box, text, score in result or []:
            text = (text or "").strip()
            if not text or float(score) < 0.5:
                continue
            xs, ys = [p[0] / scale for p in box], [p[1] / scale for p in box]
            lines.append(((min(xs), min(ys), max(xs), max(ys)), text))
        if lines:
            break
    return sorted(lines, key=lambda l: (round(l[0][1] / 4), l[0][0]))


def group_paragraphs(lines: List[Line]) -> List[List[Line]]:
    """Lines that follow each other closely belong to one paragraph; a bigger gap starts a new one."""
    paragraphs: List[List[Line]] = []
    for line in lines:
        if paragraphs:
            prev = paragraphs[-1][-1]
            height = max(prev[0][3] - prev[0][1], 1.0)
            if line[0][1] - prev[0][3] <= 0.8 * height:
                paragraphs[-1].append(line)
                continue
        paragraphs.append([line])
    return paragraphs
