from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

import pymupdf

from .japanese_ocr import OCRLine, RapidJapaneseOCR

BoundingBox = tuple[float, float, float, float]


class ExtractionSource(str, Enum):
    PDF_TEXT = "pdf_text"
    OCR = "ocr"


@dataclass(frozen=True)
class TextSpan:
    raw_text: str
    page_number: int
    bounding_box: BoundingBox
    extraction_source: ExtractionSource
    confidence: float | None = None
    normalized_text: str | None = None


@dataclass(frozen=True)
class PDFPage:
    page_number: int
    bounding_box: BoundingBox
    text_spans: tuple[TextSpan, ...]
    native_text_usable: bool
    ocr_attempted: bool


@dataclass(frozen=True)
class PDFDocument:
    source_path: Path
    pages: tuple[PDFPage, ...]


class OCRRecognizer(Protocol):
    def recognize(self, image: bytes) -> Sequence[OCRLine]: ...


class PDFIngestionError(ValueError):
    """Raised when a PDF cannot be opened or a page cannot be processed."""


def ingest_pdf(
    pdf_path: str | Path,
    *,
    ocr: OCRRecognizer | None = None,
    minimum_native_characters: int = 12,
    render_dpi: int = 144,
) -> PDFDocument:
    """Extract native text and use OCR only for pages with insufficient text."""
    if minimum_native_characters < 1:
        raise ValueError("minimum_native_characters must be at least 1.")
    if render_dpi < 1:
        raise ValueError("render_dpi must be at least 1.")

    source_path = Path(pdf_path)
    try:
        document = pymupdf.open(source_path)
    except Exception as error:
        raise PDFIngestionError(
            f"Could not open PDF '{source_path}': {error}"
        ) from error

    pages: list[PDFPage] = []
    try:
        if not document.is_pdf:
            raise PDFIngestionError(f"File '{source_path}' is not a PDF.")
        if document.needs_pass:
            raise PDFIngestionError(
                f"PDF '{source_path}' is password-protected; provide an unlocked PDF."
            )
        if document.page_count == 0:
            raise PDFIngestionError(f"PDF '{source_path}' contains no pages.")

        recognizer = ocr
        for page_index in range(document.page_count):
            page_number = page_index + 1
            page = document.load_page(page_index)
            try:
                native_spans = _extract_native_spans(page, page_number)
            except Exception as error:
                raise PDFIngestionError(
                    f"Could not extract native text from PDF page {page_number}: {error}"
                ) from error

            native_text_usable = _has_usable_native_text(
                native_spans, minimum_native_characters
            )
            text_spans = list(native_spans)
            ocr_attempted = False

            if not native_text_usable:
                if recognizer is None:
                    recognizer = RapidJapaneseOCR()
                image, scale_x, scale_y = _render_page(
                    page, page_number, render_dpi
                )
                try:
                    ocr_lines = recognizer.recognize(image)
                except Exception as error:
                    raise PDFIngestionError(
                        f"OCR failed for PDF page {page_number}: {error}"
                    ) from error
                ocr_attempted = True
                text_spans.extend(
                    _ocr_spans(
                        ocr_lines,
                        page_number,
                        scale_x,
                        scale_y,
                    )
                )

            pages.append(
                PDFPage(
                    page_number=page_number,
                    bounding_box=_rect_to_bbox(page.rect),
                    text_spans=tuple(text_spans),
                    native_text_usable=native_text_usable,
                    ocr_attempted=ocr_attempted,
                )
            )
    finally:
        document.close()

    return PDFDocument(source_path=source_path, pages=tuple(pages))


def _extract_native_spans(page: pymupdf.Page, page_number: int) -> list[TextSpan]:
    spans: list[TextSpan] = []
    text_page = page.get_text("dict", sort=True)
    for block in text_page["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                raw_text = span.get("text", "")
                if raw_text.strip():
                    spans.append(
                        TextSpan(
                            raw_text=raw_text,
                            page_number=page_number,
                            bounding_box=tuple(float(value) for value in span["bbox"]),
                            extraction_source=ExtractionSource.PDF_TEXT,
                        )
                    )
    return spans


def _has_usable_native_text(
    spans: Sequence[TextSpan], minimum_native_characters: int
) -> bool:
    meaningful_characters = sum(
        character.isalnum()
        for span in spans
        for character in span.raw_text
        if not character.isspace()
    )
    return meaningful_characters >= minimum_native_characters


def _render_page(
    page: pymupdf.Page, page_number: int, dpi: int
) -> tuple[bytes, float, float]:
    try:
        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        return (
            pixmap.tobytes("png"),
            page.rect.width / pixmap.width,
            page.rect.height / pixmap.height,
        )
    except Exception as error:
        raise PDFIngestionError(
            f"Could not render PDF page {page_number} for OCR: {error}"
        ) from error


def _ocr_spans(
    lines: Sequence[OCRLine],
    page_number: int,
    scale_x: float,
    scale_y: float,
) -> list[TextSpan]:
    spans = []
    for line in lines:
        x0, y0, x1, y1 = line.bounding_box
        spans.append(
            TextSpan(
                raw_text=line.text,
                page_number=page_number,
                bounding_box=(x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y),
                extraction_source=ExtractionSource.OCR,
                confidence=line.confidence,
            )
        )
    return spans


def _rect_to_bbox(rect: pymupdf.Rect) -> BoundingBox:
    return float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)