from pathlib import Path

import pymupdf
import pytest

from toresu.japanese_ocr import OCRLine
from toresu.pdf_ingestion import (
    ExtractionSource,
    PDFIngestionError,
    ingest_pdf,
)


def _create_pdf(path: Path, page_texts: list[str | None]) -> Path:
    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page(width=300, height=200)
        if text:
            page.insert_text((30, 50), text)
    document.save(path)
    document.close()
    return path


class FakeOCR:
    def __init__(self, lines: list[OCRLine] | None = None) -> None:
        self.lines = lines or []
        self.images: list[bytes] = []

    def recognize(self, image: bytes) -> list[OCRLine]:
        self.images.append(image)
        return self.lines


def test_native_text_and_bounding_boxes_are_preserved_without_ocr(tmp_path):
    pdf_path = _create_pdf(tmp_path / "native.pdf", ["Native PDF text with enough characters."])
    ocr = FakeOCR([OCRLine("should not run", (1, 2, 3, 4), 0.5)])

    result = ingest_pdf(pdf_path, ocr=ocr)

    assert len(result.pages) == 1
    page = result.pages[0]
    assert page.native_text_usable
    assert not page.ocr_attempted
    assert not ocr.images
    assert "Native PDF text with enough characters." in "".join(
        span.raw_text for span in page.text_spans
    )
    assert all(span.extraction_source is ExtractionSource.PDF_TEXT for span in page.text_spans)
    assert all(span.normalized_text is None for span in page.text_spans)
    with pymupdf.open(pdf_path) as document:
        source_bbox = document[0].get_text("dict", sort=True)["blocks"][0]["lines"][0]["spans"][0]["bbox"]
    assert page.text_spans[0].bounding_box == tuple(source_bbox)


def test_page_numbers_are_one_based_and_preserve_reading_order(tmp_path):
    pdf_path = _create_pdf(
        tmp_path / "multiple-pages.pdf",
        ["First page has sufficient native text.", "Second page has sufficient native text."],
    )

    result = ingest_pdf(pdf_path)

    assert [page.page_number for page in result.pages] == [1, 2]
    assert all(span.page_number == 1 for span in result.pages[0].text_spans)
    assert all(span.page_number == 2 for span in result.pages[1].text_spans)
    assert "First page" in " ".join(span.raw_text for span in result.pages[0].text_spans)


def test_empty_page_returns_no_spans_after_mocked_ocr(tmp_path):
    pdf_path = _create_pdf(tmp_path / "empty.pdf", [None])
    ocr = FakeOCR()

    result = ingest_pdf(pdf_path, ocr=ocr)

    assert result.pages[0].text_spans == ()
    assert result.pages[0].ocr_attempted
    assert len(ocr.images) == 1
    assert ocr.images[0].startswith(b"\x89PNG\r\n\x1a\n")


def test_scanned_page_ocr_is_added_without_replacing_native_text(tmp_path):
    pdf_path = _create_pdf(tmp_path / "scanned.pdf", ["x"])
    ocr = FakeOCR([OCRLine("売上高", (20, 40, 80, 100), 0.93)])

    result = ingest_pdf(pdf_path, ocr=ocr, render_dpi=144)

    page = result.pages[0]
    assert not page.native_text_usable
    assert page.ocr_attempted
    native_spans = [span for span in page.text_spans if span.extraction_source is ExtractionSource.PDF_TEXT]
    ocr_spans = [span for span in page.text_spans if span.extraction_source is ExtractionSource.OCR]
    assert native_spans and native_spans[0].raw_text == "x"
    assert len(ocr_spans) == 1
    assert ocr_spans[0].raw_text == "売上高"
    assert ocr_spans[0].bounding_box == (10.0, 20.0, 40.0, 50.0)
    assert ocr_spans[0].confidence == 0.93


def test_missing_optional_ocr_dependency_has_actionable_error(tmp_path, monkeypatch):
    pdf_path = _create_pdf(tmp_path / "scanned.pdf", [None])

    class MissingOCR:
        def recognize(self, image: bytes) -> list[OCRLine]:
            raise RuntimeError("Japanese OCR requires rapidocr and onnxruntime")

    monkeypatch.setattr("toresu.pdf_ingestion.RapidJapaneseOCR", MissingOCR)

    with pytest.raises(PDFIngestionError, match="OCR failed for PDF page 1.*rapidocr"):
        ingest_pdf(pdf_path)


def test_page_render_failure_has_actionable_error(tmp_path, monkeypatch):
    pdf_path = _create_pdf(tmp_path / "scanned.pdf", [None])

    def fail_render(self, **kwargs):
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr(pymupdf.Page, "get_pixmap", fail_render)

    with pytest.raises(PDFIngestionError, match="Could not render PDF page 1 for OCR"):
        ingest_pdf(pdf_path, ocr=FakeOCR())


def test_invalid_pdf_has_actionable_error(tmp_path):
    invalid_path = tmp_path / "invalid.pdf"
    invalid_path.write_text("not a PDF", encoding="utf-8")

    with pytest.raises(PDFIngestionError, match="Could not open PDF"):
        ingest_pdf(invalid_path)