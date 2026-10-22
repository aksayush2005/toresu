from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class OCRLine:
    text: str
    bounding_box: tuple[float, float, float, float]
    confidence: float


class RapidJapaneseOCR:
    """Lazy Japanese OCR adapter using RapidOCR's CPU ONNX Runtime engine."""

    def __init__(self, engine: Callable[..., Any] | None = None) -> None:
        self._engine = engine

    def recognize(self, image: Any) -> list[OCRLine]:
        if self._engine is None:
            self._engine = _create_engine()

        result = self._engine(image, use_cls=False)
        if result is None or result.boxes is None:
            return []

        return [
            OCRLine(
                text=text,
                bounding_box=_bounding_box(box),
                confidence=float(confidence),
            )
            for box, text, confidence in zip(
                result.boxes, result.txts, result.scores
            )
        ]


def _create_engine() -> Callable[..., Any]:
    try:
        import_module("onnxruntime")
        rapidocr = import_module("rapidocr")
    except ImportError as error:
        raise RuntimeError(
            "Japanese OCR requires rapidocr and onnxruntime; install the project "
            "requirements with `python -m pip install -r requirements.txt`."
        ) from error

    return rapidocr.RapidOCR(
        params={
            "Det.engine_type": rapidocr.EngineType.ONNXRUNTIME,
            "Det.lang_type": "japan",
            "Det.model_type": rapidocr.ModelType.SMALL,
            "Det.ocr_version": rapidocr.OCRVersion.PPOCRV6,
            "Rec.engine_type": rapidocr.EngineType.ONNXRUNTIME,
            "Rec.lang_type": "japan",
            "Rec.model_type": rapidocr.ModelType.SMALL,
            "Rec.ocr_version": rapidocr.OCRVersion.PPOCRV6,
            "Cls.engine_type": rapidocr.EngineType.ONNXRUNTIME,
        }
    )


def _bounding_box(points: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    x_values = [float(point[0]) for point in points]
    y_values = [float(point[1]) for point in points]
    return min(x_values), min(y_values), max(x_values), max(y_values)