import sys
from types import SimpleNamespace

import pytest

from toresu.japanese_ocr import OCRLine, RapidJapaneseOCR


def test_adapter_returns_text_boxes_and_confidence_without_loading_models():
    engine = lambda image, use_cls: SimpleNamespace(
        boxes=[[(8, 4), (25, 4), (25, 18), (8, 18)]],
        txts=["売上高"],
        scores=[0.97],
    )

    lines = RapidJapaneseOCR(engine=engine).recognize(b"image-bytes")

    assert lines == [OCRLine("売上高", (8.0, 4.0, 25.0, 18.0), 0.97)]


def test_adapter_reports_missing_optional_ocr_dependencies(monkeypatch):
    monkeypatch.setitem(sys.modules, "rapidocr", None)

    with pytest.raises(RuntimeError, match="requires rapidocr and onnxruntime"):
        RapidJapaneseOCR().recognize(b"image-bytes")