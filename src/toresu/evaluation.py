from __future__ import annotations

import argparse
import json
import re
import statistics
import tempfile
import time
import tracemalloc
import unicodedata
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from platform import machine, processor, system
from typing import Any, Literal

import pymupdf
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .api.pipeline import PipelineExecutionError, execute_pipeline
from .api.schemas import StageName, StageState
from .api.settings import APISettings
from .financial_normalization import NormalizedObservation
from .pdf_ingestion import PDFDocument

_DEFAULT_DATASET = (
    Path(__file__).resolve().parents[2] / "evaluation" / "datasets" / "v1"
)
_CONTEXT_FIELDS = (
    "statement_titles",
    "section_headings",
    "reporting_periods",
    "units",
    "footnotes",
)
_CJK_RUN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff々〆ヵヶ]+")
_TOKEN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff々〆ヵヶ]+|[A-Za-z0-9]+(?:[.,][A-Za-z0-9]+)*|[^\s]")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetEntry(StrictModel):
    document_id: str = Field(min_length=1)
    source_kind: Literal["synthetic_pdf", "pdf"]
    source_file: str
    annotation_file: str


class DatasetManifest(StrictModel):
    dataset_version: str = Field(min_length=1)
    documents: list[DatasetEntry]


class LayoutSpan(StrictModel):
    text: str
    x: float
    y: float
    font_size: float = 11


class LayoutPage(StrictModel):
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    spans: list[LayoutSpan]


class SyntheticLayout(StrictModel):
    pages: list[LayoutPage] = Field(min_length=1)


class ContextAnnotations(StrictModel):
    statement_titles: list[str]
    section_headings: list[str]
    reporting_periods: list[str]
    units: list[str]
    footnotes: list[str]


class AnnotationScope(StrictModel):
    text_reference: bool
    validation_findings: bool
    context_statement_titles: bool
    context_section_headings: bool
    context_reporting_periods: bool
    context_units: bool
    context_footnotes: bool
    table_detection: bool
    table_row_labels: bool
    table_periods: bool
    table_header_texts: bool
    table_cells: bool

    def for_path(self, path: str) -> bool:
        return bool(getattr(self, path.replace(".", "_")))


class ColumnAnnotation(StrictModel):
    period: str | None
    header_path: list[str]
    bounding_box: tuple[float, float, float, float] | None = None


class CellAnnotation(StrictModel):
    row_label: str
    period: str
    period_kind: str | None = None
    fiscal_year: int | None = None
    quarter: int | None = None
    relative_period: str | None = None
    date_iso: str | None = None
    raw_value: str | None
    parsed_value: str | None
    normalized_value: str | None
    normalized_unit: str | None
    value_kind: str
    metric: str | None
    extraction_source: str | None = None
    page_number: int | None = Field(ge=1)
    bounding_box: tuple[float, float, float, float] | None = None
    annotation_status: Literal["resolved", "ambiguous", "missing", "unresolvable"] = "resolved"
    issue: str | None = None

    @field_validator("parsed_value", "normalized_value")
    @classmethod
    def decimal_annotations_must_be_finite(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                parsed = Decimal(value)
            except InvalidOperation as error:
                raise ValueError("Annotated financial values must be Decimal strings.") from error
            if not parsed.is_finite():
                raise ValueError("Annotated financial values must be finite.")
        return value


class TableAnnotations(StrictModel):
    page_number: int | None = Field(ge=1)
    row_labels: list[str]
    periods: list[str]
    header_rows: list[list[str]]
    cells: list[CellAnnotation]
    columns: list[ColumnAnnotation] = Field(default_factory=list)
    bounding_box: tuple[float, float, float, float] | None = None


class ValidationAnnotation(StrictModel):
    rule_id: str
    status: str
    period_label: str | None = None


class DocumentAnnotations(StrictModel):
    document_id: str = Field(min_length=1)
    document_metadata: dict[str, Any]
    annotation_status: Literal["complete", "partial", "unannotated"]
    processing_metadata: dict[str, Any] | None = None
    text_reference: list[str]
    context: ContextAnnotations
    tables: list[TableAnnotations]
    validation_findings: list[ValidationAnnotation] | None
    exhaustive: AnnotationScope


@dataclass(frozen=True)
class LoadedDocument:
    entry: DatasetEntry
    annotations: DocumentAnnotations
    source_path: Path
    layout: SyntheticLayout | None


def load_dataset(dataset_path: Path, document_ids: set[str] | None = None) -> tuple[str, list[LoadedDocument]]:
    """Load and validate a versioned manifest, source definitions, and annotations."""
    root = dataset_path.resolve()
    manifest_path = root / "manifest.json"
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.documents
    if document_ids is not None:
        entries = [entry for entry in entries if entry.document_id in document_ids]
        missing_ids = document_ids - {entry.document_id for entry in entries}
        if missing_ids:
            raise ValueError(f"Unknown document IDs: {', '.join(sorted(missing_ids))}.")

    loaded = []
    seen: set[str] = set()
    for entry in entries:
        if entry.document_id in seen:
            raise ValueError(f"Duplicate document ID: {entry.document_id}.")
        seen.add(entry.document_id)
        annotation_path = _inside_dataset(root, entry.annotation_file)
        annotations = DocumentAnnotations.model_validate_json(
            annotation_path.read_text(encoding="utf-8")
        )
        if annotations.document_id != entry.document_id:
            raise ValueError(f"Annotation document ID does not match {entry.document_id}.")
        source_path = _inside_dataset(root, entry.source_file)
        layout = None
        if entry.source_kind == "synthetic_pdf":
            layout = SyntheticLayout.model_validate_json(
                source_path.read_text(encoding="utf-8")
            )
        elif not source_path.is_file():
            raise FileNotFoundError(f"Evaluation PDF does not exist: {source_path}")
        loaded.append(LoadedDocument(entry, annotations, source_path, layout))
    return manifest.dataset_version, loaded


def evaluate_dataset(
    dataset_path: Path = _DEFAULT_DATASET,
    *,
    document_ids: set[str] | None = None,
) -> dict[str, Any]:
    dataset_version, documents = load_dataset(dataset_path, document_ids)
    results = []
    for document in documents:
        with tempfile.TemporaryDirectory(prefix="toresu-eval-") as temporary:
            try:
                pdf_path = _materialize_source(document, Path(temporary))
                run = _run_pipeline(pdf_path, Path(temporary) / "api-data")
                results.append(_score_document(document, run))
            except Exception as error:
                results.append(
                    {
                        "document_id": document.entry.document_id,
                        "source_kind": document.entry.source_kind,
                        "success": False,
                        "stage_seconds": {},
                        "wall_seconds": 0,
                        "metrics": {},
                        "errors": [
                            {
                                "category": "evaluation_source",
                                "code": type(error).__name__,
                                "message": str(error),
                            }
                        ],
                    }
                )

    defined_document_metrics: dict[str, list[float]] = {}
    for result in results:
        for metric_name, metric_value in _flatten_metrics(result.get("metrics", {})).items():
            if isinstance(metric_value, (int, float)):
                defined_document_metrics.setdefault(metric_name, []).append(float(metric_value))
    return {
        "report_version": "1",
        "dataset_version": dataset_version,
        "document_count": len(results),
        "successful_documents": sum(result["success"] for result in results),
        "failed_documents": sum(not result["success"] for result in results),
        "aggregation": "macro mean across documents; undefined document metrics are omitted and counted only where defined",
        "aggregate_metrics": {
            name: {
                "macro_mean": statistics.fmean(values),
                "documents_scored": len(values),
            }
            for name, values in sorted(defined_document_metrics.items())
        },
        "software": _software_versions(),
        "hardware": _hardware_info(),
        "configuration": {"external_llm": "not used", "pipeline": "Toresu deterministic pipeline"},
        "documents": results,
    }


def benchmark_dataset(
    dataset_path: Path = _DEFAULT_DATASET,
    *,
    document_ids: set[str] | None = None,
    repeats: int = 3,
    warmup: int = 1,
) -> dict[str, Any]:
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be at least 1 and warmup must be non-negative.")
    dataset_version, documents = load_dataset(dataset_path, document_ids)
    document_results = []
    for document in documents:
        with tempfile.TemporaryDirectory(prefix="toresu-bench-") as temporary:
            try:
                pdf_path = _materialize_source(document, Path(temporary))
            except Exception as error:
                failure = {
                    "success": False,
                    "error": {
                        "category": "evaluation_source",
                        "code": type(error).__name__,
                        "message": str(error),
                    },
                    "wall_seconds": 0,
                    "seconds_per_page": None,
                    "stage_seconds": {},
                    "python_tracemalloc_peak_bytes": 0,
                    "page_count": None,
                }
                document_results.append(
                    {
                        "document_id": document.entry.document_id,
                        "source_kind": document.entry.source_kind,
                        "repeats": [failure.copy() for _ in range(repeats)],
                        "summary": {
                            "successful_repeats": 0,
                            "median_wall_seconds": None,
                            "mean_wall_seconds": None,
                            "median_seconds_per_page": None,
                            "median_stage_seconds": {},
                            "median_python_tracemalloc_peak_bytes": 0,
                        },
                    }
                )
                continue
            for _ in range(warmup):
                _run_pipeline(pdf_path, Path(temporary) / "warmup-data")
            samples: list[dict[str, Any]] = []
            for repeat_index in range(repeats):
                tracemalloc.start()
                run = _run_pipeline(pdf_path, Path(temporary) / f"run-{repeat_index}")
                _, peak_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                samples.append(
                    {
                        "success": run["success"],
                        "error": run.get("error"),
                        "wall_seconds": run["wall_seconds"],
                        "seconds_per_page": (
                            run["wall_seconds"] / run["page_count"]
                            if run.get("page_count")
                            else None
                        ),
                        "stage_seconds": run["stage_seconds"],
                        "python_tracemalloc_peak_bytes": peak_bytes,
                        "page_count": run.get("page_count"),
                    }
                )
            elapsed = [sample["wall_seconds"] for sample in samples if sample["success"]]
            stage_names = sorted(
                {key for sample in samples for key in sample["stage_seconds"]}
            )
            document_results.append(
                {
                    "document_id": document.entry.document_id,
                    "source_kind": document.entry.source_kind,
                    "repeats": samples,
                    "summary": {
                        "successful_repeats": len(elapsed),
                        "median_wall_seconds": statistics.median(elapsed) if elapsed else None,
                        "mean_wall_seconds": statistics.fmean(elapsed) if elapsed else None,
                        "median_seconds_per_page": (
                            statistics.median(
                                sample["seconds_per_page"]
                                for sample in samples
                                if sample["seconds_per_page"] is not None
                            )
                            if any(sample["seconds_per_page"] is not None for sample in samples)
                            else None
                        ),
                        "median_stage_seconds": {
                            name: statistics.median(
                                sample["stage_seconds"][name]
                                for sample in samples
                                if name in sample["stage_seconds"]
                            )
                            for name in stage_names
                        },
                        "median_python_tracemalloc_peak_bytes": statistics.median(
                            sample["python_tracemalloc_peak_bytes"] for sample in samples
                        ),
                    },
                }
            )
    elapsed_success = [
        sample["wall_seconds"]
        for document in document_results
        for sample in document["repeats"]
        if sample["success"]
    ]
    total_elapsed = sum(elapsed_success)
    return {
        "report_version": "1",
        "dataset_version": dataset_version,
        "methodology": {
            "warmup_runs_per_document": warmup,
            "measured_repeats_per_document": repeats,
            "latency": "wall-clock elapsed time around execute_pipeline; synthetic PDF generation excluded",
            "memory": "Python allocations measured with tracemalloc; native PyMuPDF/RapidOCR allocations are excluded",
            "throughput": "successful documents divided by summed measured wall time; single-process CPU configuration",
        },
        "successful_runs": len(elapsed_success),
        "failed_runs": sum(
            not sample["success"]
            for document in document_results
            for sample in document["repeats"]
        ),
        "throughput_documents_per_second": (
            len(elapsed_success) / total_elapsed if total_elapsed else None
        ),
        "software": _software_versions(),
        "hardware": _hardware_info(),
        "documents": document_results,
    }


def write_report(report: dict[str, Any], output_dir: Path, *, filename: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{filename}.json"
    summary_path = output_dir / f"{filename}.txt"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(render_summary(report), encoding="utf-8")
    return json_path, summary_path


def render_summary(report: dict[str, Any]) -> str:
    lines = [
        f"Toresu {'benchmark' if 'methodology' in report else 'evaluation'} report",
        f"Dataset version: {report['dataset_version']}",
    ]
    if "methodology" in report:
        lines.extend(
            [
                f"Successful/failed runs: {report['successful_runs']}/{report['failed_runs']}",
                f"Throughput (successful runs): {report['throughput_documents_per_second']}",
                f"Methodology: {report['methodology']['latency']}",
                f"Memory note: {report['methodology']['memory']}",
            ]
        )
    else:
        lines.extend(
            [
                f"Documents: {report['successful_documents']} successful, {report['failed_documents']} failed",
                "Aggregate metrics (macro mean):",
            ]
        )
        for name, metric in report["aggregate_metrics"].items():
            lines.append(f"  {name}: {metric['macro_mean']:.4f} ({metric['documents_scored']} documents)")
    for document in report["documents"]:
        if report.get("methodology"):
            lines.append(
                f"  {document['document_id']}: median wall seconds="
                f"{document['summary']['median_wall_seconds']}"
            )
        else:
            lines.append(
                f"  {document['document_id']}: "
                f"{'success' if document['success'] else 'FAILED'}"
            )
            for error in document.get("errors", []):
                lines.append(
                    f"    {error['category']}: "
                    f"{error.get('message', error.get('predicted', error.get('expected', 'mismatch')))}"
                )
    return "\n".join(lines) + "\n"


def character_error_rate(reference: str, prediction: str) -> float:
    reference_chars = list(reference)
    prediction_chars = list(prediction)
    return _edit_distance(reference_chars, prediction_chars) / max(1, len(reference_chars))


def word_error_rate(reference: str, prediction: str) -> float:
    reference_tokens = _japanese_tokens(reference)
    prediction_tokens = _japanese_tokens(prediction)
    return _edit_distance(reference_tokens, prediction_tokens) / max(1, len(reference_tokens))


def precision_recall_f1(
    expected: list[str],
    predicted: list[str],
    *,
    exhaustive: bool,
) -> dict[str, int | float | None]:
    expected_counts = Counter(_normalize_label(value) for value in expected)
    predicted_counts = Counter(_normalize_label(value) for value in predicted)
    true_positives = sum((expected_counts & predicted_counts).values())
    false_negatives = sum((expected_counts - predicted_counts).values())
    false_positives = sum((predicted_counts - expected_counts).values()) if exhaustive else None
    recall = true_positives / sum(expected_counts.values()) if expected_counts else (
        1.0 if exhaustive and not predicted_counts else 0.0 if exhaustive else None
    )
    precision = (
        true_positives / (true_positives + false_positives)
        if exhaustive and true_positives + (false_positives or 0)
        else (1.0 if exhaustive and not expected_counts and not predicted_counts else 0.0)
        if exhaustive
        else None
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else (0.0 if precision is not None and recall is not None else None)
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "expected_count": sum(expected_counts.values()),
        "predicted_count": sum(predicted_counts.values()),
    }


def evaluate_table_reference(
    predicted_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]] | None,
    predicted_columns: list[str],
    reference_columns: list[str] | None,
    *,
    predicted_label_column: str = "row",
    reference_label_column: str = "row",
) -> dict[str, Any]:
    """Compare a displayed table with reference rows without rewriting source values.

    Label and header comparison follows the evaluation framework's NFKC/whitespace
    normalization. Cell exact matches trim surrounding whitespace only. Numeric
    comparison additionally accepts valid comma grouping and Decimal-equivalent
    precision, but never rewrites dashes, zeroes, or negative markers.
    """
    if reference_rows is None or reference_columns is None:
        return {"evaluated": False, "reason": "Reference table is unavailable."}

    expected_rows = list(reference_rows)
    actual_rows = list(predicted_rows)
    expected_columns = list(reference_columns)
    actual_columns = list(predicted_columns)
    row_slots = max(len(expected_rows), len(actual_rows))
    column_slots = max(len(expected_columns), len(actual_columns))

    def label_at(rows: list[dict[str, Any]], index: int, column: str) -> str:
        if index >= len(rows):
            return ""
        value = rows[index].get(column)
        return value.strip() if isinstance(value, str) else ""

    row_matches = [
        _normalize_label(label_at(expected_rows, index, reference_label_column))
        == _normalize_label(label_at(actual_rows, index, predicted_label_column))
        for index in range(row_slots)
    ]
    column_matches = [
        _normalize_label(expected_columns[index])
        == _normalize_label(actual_columns[index])
        if index < len(expected_columns) and index < len(actual_columns)
        else False
        for index in range(column_slots)
    ]
    mismatches: list[dict[str, Any]] = []
    for index, matched in enumerate(row_matches):
        if not matched:
            expected_label = label_at(expected_rows, index, reference_label_column)
            predicted_label = label_at(actual_rows, index, predicted_label_column)
            mismatches.append(
                {
                    "category": (
                        "missing_row"
                        if index >= len(actual_rows)
                        else "extra_row"
                        if index >= len(expected_rows)
                        else "misaligned_row"
                    ),
                    "row_index": index + 1,
                    "expected_row": expected_label or None,
                    "predicted_row": predicted_label or None,
                    "expected": expected_label or None,
                    "predicted": predicted_label or None,
                }
            )
    for index, matched in enumerate(column_matches):
        if not matched:
            expected_header = (
                expected_columns[index] if index < len(expected_columns) else None
            )
            predicted_header = (
                actual_columns[index] if index < len(actual_columns) else None
            )
            mismatches.append(
                {
                    "category": (
                        "missing_column"
                        if index >= len(actual_columns)
                        else "extra_column"
                        if index >= len(expected_columns)
                        else "misaligned_column"
                    ),
                    "column_index": index,
                    "expected_column": expected_header,
                    "predicted_column": predicted_header,
                    "expected": expected_header,
                    "predicted": predicted_header,
                }
            )

    cell_total = row_slots * column_slots
    exact_cell_matches = 0
    numeric_total = 0
    numeric_matches = 0
    expected_cell_tokens: list[tuple[str, ...]] = []
    predicted_cell_tokens: list[tuple[str, ...]] = []
    for row_index in range(row_slots):
        expected_row = expected_rows[row_index] if row_index < len(expected_rows) else {}
        predicted_row = actual_rows[row_index] if row_index < len(actual_rows) else {}
        for column_index in range(column_slots):
            expected_column = (
                expected_columns[column_index]
                if column_index < len(expected_columns)
                else None
            )
            predicted_column = (
                actual_columns[column_index]
                if column_index < len(actual_columns)
                else None
            )
            expected_value = (
                expected_row.get(expected_column)
                if expected_column is not None
                else None
            )
            predicted_value = (
                predicted_row.get(predicted_column)
                if predicted_column is not None
                else None
            )
            expected_text = _table_cell_text(expected_value)
            predicted_text = _table_cell_text(predicted_value)
            exact = expected_text == predicted_text
            exact_cell_matches += exact

            expected_decimal = _table_decimal(expected_text)
            predicted_decimal = _table_decimal(predicted_text)
            if expected_decimal is not None:
                numeric_total += 1
                numeric_match = (
                    predicted_decimal == expected_decimal
                    and _negative_marker(expected_text)
                    == _negative_marker(predicted_text)
                )
                numeric_matches += numeric_match
            else:
                numeric_match = False

            if expected_text is not None:
                expected_cell_tokens.append(
                    (str(row_index), str(column_index), expected_text)
                )
            if predicted_text is not None:
                predicted_cell_tokens.append(
                    (str(row_index), str(column_index), predicted_text)
                )
            if exact:
                continue

            category = "cell_value_mismatch"
            numeric_issue = None
            if expected_text is None:
                category = "extra_cell"
            elif predicted_text is None:
                category = "missing_cell"
            elif (
                column_index < len(expected_columns)
                and column_index < len(actual_columns)
                and not column_matches[column_index]
            ):
                category = "fiscal_period_alignment_mismatch"
                if expected_decimal is not None and predicted_decimal is not None:
                    numeric_issue = (
                        "numeric_sign_mismatch"
                        if expected_decimal.copy_abs() == predicted_decimal.copy_abs()
                        else "numeric_magnitude_mismatch"
                        if expected_decimal != predicted_decimal
                        else "negative_marker_mismatch"
                        if _negative_marker(expected_text)
                        != _negative_marker(predicted_text)
                        else None
                    )
            elif (expected_decimal is None) != (predicted_decimal is None):
                category = "value_kind_mismatch"
            elif (
                expected_decimal is not None
                and predicted_decimal is not None
                and expected_decimal != predicted_decimal
            ):
                category = (
                    "numeric_sign_mismatch"
                    if expected_decimal.copy_abs() == predicted_decimal.copy_abs()
                    else "numeric_magnitude_mismatch"
                )
            elif (
                expected_decimal is not None
                and predicted_decimal is not None
                and _negative_marker(expected_text)
                != _negative_marker(predicted_text)
            ):
                category = "negative_marker_mismatch"
            elif (
                expected_decimal is not None
                and predicted_decimal is not None
                and expected_decimal == predicted_decimal
            ):
                category = "numeric_format_difference"
            mismatches.append(
                {
                    "category": category,
                    "numeric_issue": numeric_issue,
                    "row_index": row_index + 1,
                    "column_index": column_index,
                    "expected_row": label_at(
                        expected_rows, row_index, reference_label_column
                    )
                    or None,
                    "predicted_row": label_at(
                        actual_rows, row_index, predicted_label_column
                    )
                    or None,
                    "expected_column": expected_column,
                    "predicted_column": predicted_column,
                    "expected": expected_text,
                    "predicted": predicted_text,
                }
            )

    row_label_metrics = precision_recall_f1(
        [
            label_at(expected_rows, index, reference_label_column)
            for index in range(len(expected_rows))
        ],
        [
            label_at(actual_rows, index, predicted_label_column)
            for index in range(len(actual_rows))
        ],
        exhaustive=True,
    )
    cell_metrics = _counter_metric(
        expected_cell_tokens, predicted_cell_tokens, exhaustive=True
    )
    row_accuracy = sum(row_matches) / row_slots if row_slots else 1.0
    column_accuracy = sum(column_matches) / column_slots if column_slots else 1.0
    structure_support = row_slots + column_slots
    structure_accuracy = (
        (sum(row_matches) + sum(column_matches)) / structure_support
        if structure_support
        else 1.0
    )
    return {
        "evaluated": True,
        "row_count": {"reference": len(expected_rows), "predicted": len(actual_rows)},
        "column_count": {
            "reference": len(expected_columns),
            "predicted": len(actual_columns),
        },
        "row_count_agreement": len(expected_rows) == len(actual_rows),
        "column_count_agreement": len(expected_columns) == len(actual_columns),
        "row_label_accuracy": {
            "correct": sum(row_matches),
            "total": row_slots,
            "accuracy": row_accuracy,
            "precision_recall_f1": row_label_metrics,
        },
        "column_alignment_accuracy": {
            "correct": sum(column_matches),
            "total": column_slots,
            "accuracy": column_accuracy,
        },
        "table_structure_accuracy": {
            "correct": sum(row_matches) + sum(column_matches),
            "total": structure_support,
            "accuracy": structure_accuracy,
            "exact_agreement": (
                len(expected_rows) == len(actual_rows)
                and len(expected_columns) == len(actual_columns)
                and all(row_matches)
                and all(column_matches)
            ),
        },
        "cell_exact_match_accuracy": {
            "correct": exact_cell_matches,
            "total": cell_total,
            "accuracy": exact_cell_matches / cell_total if cell_total else 1.0,
            "precision_recall_f1": cell_metrics,
        },
        "numeric_cell_accuracy": {
            "correct": numeric_matches,
            "total": numeric_total,
            "accuracy": numeric_matches / numeric_total if numeric_total else None,
        },
        "mismatches": mismatches,
    }


def _table_cell_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    return str(value)


def _table_decimal(value: str | None) -> Decimal | None:
    if value is None:
        return None
    candidate = value
    if candidate.startswith("△"):
        candidate = candidate[1:]
        negative = True
    elif candidate.startswith("-"):
        candidate = candidate[1:]
        negative = True
    else:
        negative = False
    if not re.fullmatch(r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", candidate):
        return None
    try:
        number = Decimal(candidate.replace(",", ""))
    except InvalidOperation:
        return None
    return -number if negative else number


def _negative_marker(value: str | None) -> str | None:
    if value is None:
        return None
    if value.startswith("△"):
        return "triangle"
    if value.startswith("-"):
        return "minus"
    return "positive"


def bbox_iou(
    expected: tuple[float, float, float, float],
    predicted: tuple[float, float, float, float],
) -> float:
    x0, y0 = max(expected[0], predicted[0]), max(expected[1], predicted[1])
    x1, y1 = min(expected[2], predicted[2]), min(expected[3], predicted[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    expected_area = max(0.0, expected[2] - expected[0]) * max(0.0, expected[3] - expected[1])
    predicted_area = max(0.0, predicted[2] - predicted[0]) * max(0.0, predicted[3] - predicted[1])
    union = expected_area + predicted_area - intersection
    return intersection / union if union else 1.0


def _score_document(document: LoadedDocument, run: dict[str, Any]) -> dict[str, Any]:
    base = {
        "document_id": document.entry.document_id,
        "source_kind": document.entry.source_kind,
        "success": run["success"],
        "stage_seconds": run["stage_seconds"],
        "wall_seconds": run["wall_seconds"],
        "errors": [],
    }
    if not run["success"]:
        base["errors"] = [run["error"]]
        base["metrics"] = {}
        return base

    annotations = document.annotations
    artifacts = run["artifacts"]
    prediction_text = _document_text(artifacts.document)
    reference_text = "\n".join(annotations.text_reference)
    context_predictions = _context_predictions(artifacts.context)
    exhaustive = annotations.exhaustive
    context_metrics = {
        field: precision_recall_f1(
            getattr(annotations.context, field),
            context_predictions[field],
            exhaustive=exhaustive.for_path(f"context.{field}"),
        )
        for field in _CONTEXT_FIELDS
    }
    errors = []
    for field, metric in context_metrics.items():
        if metric["f1"] not in {None, 1}:
            errors.append(
                {
                    "category": f"context_{field}",
                    "expected": getattr(annotations.context, field),
                    "predicted": context_predictions[field],
                }
            )
    tables = artifacts.reconstruction.tables
    expected_rows = [label for table in annotations.tables for label in table.row_labels]
    expected_periods = [period for table in annotations.tables for period in table.periods]
    expected_headers = [
        header for table in annotations.tables for row in table.header_rows for header in row
    ]
    row_predictions = [
        row.label_cell.raw_text
        for table in tables
        for row in table.rows
        if row.label_cell and row.label_cell.raw_text is not None
    ]
    period_predictions = [
        column.reporting_period.raw_label
        for table in tables
        for column in table.columns
        if column.reporting_period is not None
    ]
    header_predictions = [
        header.raw_text
        for table in tables
        for row in table.header_rows
        for header in row
    ]
    table_metric = _count_metric(
        len(annotations.tables),
        len(tables),
        exhaustive=exhaustive.table_detection,
    )
    row_metric = precision_recall_f1(
        expected_rows,
        row_predictions,
        exhaustive=exhaustive.table_row_labels,
    )
    period_metric = precision_recall_f1(
        expected_periods,
        period_predictions,
        exhaustive=exhaustive.table_periods,
    )
    header_metric = precision_recall_f1(
        expected_headers,
        header_predictions,
        exhaustive=exhaustive.table_header_texts,
    )
    table_page_pairs = [
        (expected.page_number, tables[index].page_number)
        for index, expected in enumerate(annotations.tables)
        if expected.page_number is not None and index < len(tables)
    ]
    table_page_localization = (
        sum(expected == predicted for expected, predicted in table_page_pairs)
        / sum(expected.page_number is not None for expected in annotations.tables)
        if any(expected.page_number is not None for expected in annotations.tables)
        else None
    )
    if table_page_localization is not None and table_page_localization < 1:
        errors.append(
            {
                "category": "table_page_localization",
                "expected": [table.page_number for table in annotations.tables],
                "predicted": [table.page_number for table in tables],
            }
        )
    observations = [
        observation
        for normalized_table in artifacts.normalization.tables
        for observation in normalized_table.observations
    ]
    observation_metrics, provenance, cell_errors = _score_cells(
        annotations, observations, tables
    )
    errors.extend(cell_errors)
    validation_findings = (
        list(artifacts.validation.findings)
        if artifacts.validation
        else []
    )
    validation_metric = None
    if annotations.validation_findings is not None:
        expected_validation_keys = [
            (finding.rule_id, finding.status, finding.period_label or "")
            for finding in annotations.validation_findings
        ]
        predicted_validation_keys = [
            (
                finding.rule_id,
                finding.status.value,
                finding.period_label or "",
            )
            for finding in validation_findings
        ]
        finding_detection = _counter_metric(
            expected_validation_keys,
            predicted_validation_keys,
            exhaustive=exhaustive.validation_findings,
        )
        status_classification = precision_recall_f1(
            [finding.status for finding in annotations.validation_findings],
            [finding.status.value for finding in validation_findings],
            exhaustive=exhaustive.validation_findings,
        )
        validation_metric = precision_recall_f1(
            [finding.rule_id for finding in annotations.validation_findings],
            [finding.rule_id for finding in validation_findings],
            exhaustive=exhaustive.validation_findings,
        )
        validation_metric = {
            "finding_detection": finding_detection,
            "status_classification": status_classification,
            "rule_id_classification": validation_metric,
        }
        if finding_detection["f1"] not in {None, 1}:
            errors.append(
                {
                    "category": "validation_classification",
                    "expected": expected_validation_keys,
                    "predicted": predicted_validation_keys,
                }
            )
    text_spans = [
        span for page in artifacts.document.pages for span in page.text_spans
    ]
    source_counts = Counter(span.extraction_source.value for span in text_spans)
    text_is_exhaustive = exhaustive.text_reference
    cer = character_error_rate(reference_text, prediction_text) if text_is_exhaustive else None
    wer = word_error_rate(reference_text, prediction_text) if text_is_exhaustive else None
    if cer:
        errors.append(
            {
                "category": "text_extraction",
                "expected_characters": len(reference_text),
                "predicted_characters": len(prediction_text),
                "cer": cer,
                "wer": wer,
            }
        )
    text_line_coverage = (
        precision_recall_f1(
            annotations.text_reference,
            prediction_text.splitlines(),
            exhaustive=text_is_exhaustive,
        )
        if annotations.text_reference or text_is_exhaustive
        else None
    )
    metrics = {
        "text": {
            "cer": cer,
            "wer": wer,
            "reference_characters": len(reference_text),
            "reference_tokens": len(_japanese_tokens(reference_text)),
            "tokenization": "CJK runs are split into Unicode code points; ASCII alphanumeric runs are tokens",
            "extraction_source_span_counts": dict(source_counts),
            "text_reference_exhaustive": text_is_exhaustive,
            "exact_reference_line_coverage": (
                text_line_coverage["recall"] if text_line_coverage else None
            ),
            "reference_items": len(annotations.text_reference),
            "predicted_text_spans": len(text_spans),
        },
        "context": context_metrics,
        "tables": {
            "detection": table_metric,
            "row_labels": row_metric,
            "periods": period_metric,
            "header_texts": header_metric,
            "header_row_count": _count_metric(
                sum(len(table.header_rows) for table in annotations.tables),
                sum(len(table.header_rows) for table in tables),
            ),
            "table_page_localization_accuracy": table_page_localization,
            "cell_observations": observation_metrics,
        },
        "financial_validation": validation_metric,
        "provenance": {
            **provenance,
            "validation_evidence_coverage": (
                sum(bool(finding.evidence.source_spans) for finding in artifacts.validation.findings)
                / len(artifacts.validation.findings)
                if artifacts.validation and artifacts.validation.findings
                else None
            ),
        },
    }
    base["metrics"] = metrics
    base["errors"] = errors
    base["prediction_counts"] = {
        "tables": len(tables),
        "rows": len(row_predictions),
        "observations": len(observations),
        "validation_findings": len(validation_findings),
    }
    for name, metric in (
        ("table_detection", table_metric),
        ("row_structure", row_metric),
        ("period_assignment", period_metric),
        ("header_reconstruction", header_metric),
    ):
        if metric["f1"] not in {None, 1}:
            errors.append(
                {
                    "category": name,
                    "expected": metric["expected_count"],
                    "predicted": metric["predicted_count"],
                }
            )
    return base


def _score_cells(
    annotations: DocumentAnnotations,
    observations: list[NormalizedObservation],
    reconstructed_tables: tuple[Any, ...],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    expected_cells = [
        (table_index, cell)
        for table_index, table in enumerate(annotations.tables)
        for cell in table.cells
    ]
    exhaustive = annotations.exhaustive.table_cells
    expected_keys = [
        _cell_key(table_index, cell.row_label, cell.period, cell.raw_value)
        for table_index, cell in expected_cells
    ]
    table_indexes = {
        table.table_id: index for index, table in enumerate(reconstructed_tables)
    }
    predicted_keys = [
        _cell_key(
            table_indexes.get(observation.source_table.table_id, -1),
            observation.metric.raw_label or "",
            observation.period.raw_label,
            observation.value.parsed.raw_text,
        )
        for observation in observations
    ]
    cell_detection = _counter_metric(
        expected_keys, predicted_keys, exhaustive=exhaustive
    )
    by_key: dict[tuple[int, str, str], list[NormalizedObservation]] = {}
    for observation in observations:
        key = (
            table_indexes.get(observation.source_table.table_id, -1),
            _normalize_label(observation.metric.raw_label or ""),
            _normalize_label(observation.period.raw_label),
        )
        by_key.setdefault(key, []).append(observation)

    matches: dict[str, list[bool]] = {
        name: [] for name in (
            "raw_value_accuracy",
            "parsed_value_accuracy",
            "normalized_value_accuracy",
            "normalized_unit_accuracy",
            "value_kind_accuracy",
            "metric_mapping_accuracy",
            "period_assignment_accuracy",
            "exact_observation_accuracy",
            "extraction_source_accuracy",
        )
    }
    errors = []
    covered = 0
    page_matches = 0
    page_scored = 0
    ious: list[float] = []
    for table_index, expected in expected_cells:
        key = (
            table_index,
            _normalize_label(expected.row_label),
            _normalize_label(expected.period),
        )
        candidates = by_key.get(key, [])
        observed = candidates[0] if len(candidates) == 1 else None
        predicted = {
            "raw_value": observed.value.parsed.raw_text if observed else None,
            "parsed_value": _decimal_string(observed.value.parsed.value) if observed else None,
            "normalized_value": _decimal_string(observed.value.normalized_value) if observed else None,
            "normalized_unit": observed.value.normalized_unit.value if observed and observed.value.normalized_unit else None,
            "value_kind": observed.value.parsed.kind.value if observed else None,
            "metric": observed.metric.candidates[0].value if observed and len(observed.metric.candidates) == 1 else None,
            "period": observed.period.raw_label if observed else None,
            "period_assignment": (
                observed.period.kind.value,
                observed.period.fiscal_year,
                observed.period.quarter,
                observed.period.relative_period.value if observed.period.relative_period else None,
                observed.period.date_iso,
            )
            if observed
            else None,
            "extraction_source": (
                next(iter({span.extraction_source.value for span in observed.value.evidence_spans}))
                if observed and len({span.extraction_source.value for span in observed.value.evidence_spans}) == 1
                else "mixed"
                if observed and observed.value.evidence_spans
                else None
            ),
        }
        exact_field_matches: list[bool] = []
        for metric, expected_value, prediction_value in (
            ("raw_value_accuracy", expected.raw_value, predicted["raw_value"]),
            ("parsed_value_accuracy", expected.parsed_value, predicted["parsed_value"]),
            ("normalized_value_accuracy", expected.normalized_value, predicted["normalized_value"]),
            ("normalized_unit_accuracy", expected.normalized_unit, predicted["normalized_unit"]),
            ("value_kind_accuracy", expected.value_kind, predicted["value_kind"]),
            ("metric_mapping_accuracy", expected.metric, predicted["metric"]),
            (
                "period_assignment_accuracy",
                (
                    expected.period_kind,
                    expected.fiscal_year,
                    expected.quarter,
                    expected.relative_period,
                    expected.date_iso,
                ),
                predicted["period_assignment"],
            ),
        ):
            matched = observed is not None and _same_expected(
                expected_value,
                prediction_value,
                decimal=metric in {"parsed_value_accuracy", "normalized_value_accuracy"},
            )
            matches[metric].append(matched)
            exact_field_matches.append(matched)
            if not matched:
                errors.append(
                    {
                        "category": {
                            "raw_value_accuracy": "numeric_value_mismatch",
                            "parsed_value_accuracy": "numeric_parse_mismatch",
                            "normalized_value_accuracy": "unit_conversion_mismatch",
                            "normalized_unit_accuracy": "unit_conversion_mismatch",
                            "value_kind_accuracy": "missing_ambiguous_value_handling",
                            "metric_mapping_accuracy": "semantic_mapping_mismatch",
                            "period_assignment_accuracy": "reporting_period_resolution",
                            "extraction_source_accuracy": "native_text_vs_ocr",
                        }[metric],
                        "expected": expected_value,
                        "predicted": prediction_value,
                        "row_label": expected.row_label,
                        "period": expected.period,
                        "page_number": expected.page_number,
                    }
                )
        matches["exact_observation_accuracy"].append(
            observed is not None and all(exact_field_matches)
        )
        if expected.extraction_source is not None:
            source_matches = observed is not None and _same_expected(
                expected.extraction_source, predicted["extraction_source"]
            )
            matches["extraction_source_accuracy"].append(source_matches)
            if not source_matches:
                errors.append(
                    {
                        "category": "native_text_vs_ocr",
                        "expected": expected.extraction_source,
                        "predicted": predicted["extraction_source"],
                        "row_label": expected.row_label,
                        "period": expected.period,
                        "page_number": expected.page_number,
                    }
                )
        if observed is None:
            errors.append(
                {
                    "category": "missing_or_ambiguous_observation",
                    "expected": {"row_label": expected.row_label, "period": expected.period, "raw_value": expected.raw_value},
                    "predicted": predicted,
                    "page_number": expected.page_number,
                }
            )
            continue

        source_spans = observed.value.evidence_spans
        unit_spans = tuple(
            span
            for fact in observed.value.unit_candidates
            for span in fact.evidence.source_spans
        )
        required_evidence = (
            bool(observed.metric.evidence_spans),
            bool(source_spans),
            bool(observed.period.evidence_spans),
            bool(unit_spans) if expected.normalized_unit is not None else True,
        )
        covered += all(required_evidence)
        if expected.page_number is not None:
            page_scored += 1
            pages = {span.page_number for span in source_spans}
            page_matches += expected.page_number in pages
        if expected.bounding_box is not None and source_spans:
            predicted_bbox = _union_boxes([span.bounding_box for span in source_spans])
            ious.append(bbox_iou(expected.bounding_box, predicted_bbox))
        if not all(required_evidence):
            errors.append(
                {
                    "category": "missing_provenance",
                    "expected": "value, metric-label, period, and applicable unit evidence",
                    "predicted": [bool(value) for value in required_evidence],
                    "page_number": expected.page_number,
                }
            )

    cell_metrics = {
        "detection": cell_detection,
        **{
            name: {
                "accuracy": sum(values) / len(values) if values else None,
                "correct": sum(values),
                "expected_count": len(values),
            }
            for name, values in matches.items()
        },
    }
    provenance = {
        "coverage": covered / len(expected_cells) if expected_cells else None,
        "observations_scored": len(expected_cells),
        "page_localization_accuracy": page_matches / page_scored if page_scored else None,
        "pages_scored": page_scored,
        "mean_bbox_iou": statistics.fmean(ious) if ious else None,
        "bounding_boxes_scored": len(ious),
        "coverage_definition": "value, metric-label, period, and applicable unit source evidence are all present",
    }
    return cell_metrics, provenance, errors


def _run_pipeline(pdf_path: Path, data_dir: Path) -> dict[str, Any]:
    stage_start: dict[str, float] = {}
    stage_seconds: dict[str, float] = {}

    def on_stage(stage: StageName, state: StageState, *_: object) -> None:
        name = stage.value
        if state is StageState.PROCESSING:
            stage_start[name] = time.perf_counter()
        elif state in {StageState.COMPLETED, StageState.FAILED} and name in stage_start:
            stage_seconds[name] = time.perf_counter() - stage_start[name]

    start = time.perf_counter()
    try:
        artifacts = execute_pipeline(
            pdf_path,
            settings=APISettings(data_dir=data_dir),
            on_stage=on_stage,
        )
        return {
            "success": True,
            "artifacts": artifacts,
            "wall_seconds": time.perf_counter() - start,
            "stage_seconds": stage_seconds,
            "page_count": len(artifacts.document.pages) if artifacts.document else None,
        }
    except PipelineExecutionError as error:
        return {
            "success": False,
            "wall_seconds": time.perf_counter() - start,
            "stage_seconds": stage_seconds,
            "page_count": len(error.artifacts.document.pages) if error.artifacts.document else None,
            "error": {
                "category": error.failure.stage.value,
                "code": error.failure.error_code,
                "message": error.failure.public_message,
            },
        }
    except Exception as error:
        return {
            "success": False,
            "wall_seconds": time.perf_counter() - start,
            "stage_seconds": stage_seconds,
            "page_count": None,
            "error": {
                "category": "pipeline_setup",
                "code": type(error).__name__,
                "message": str(error),
            },
        }


def _materialize_source(document: LoadedDocument, temporary: Path) -> Path:
    if document.layout is None:
        return document.source_path
    pdf_path = temporary / f"{document.entry.document_id}.pdf"
    pdf = pymupdf.open()
    try:
        for page_layout in document.layout.pages:
            page = pdf.new_page(width=page_layout.width, height=page_layout.height)
            for span in page_layout.spans:
                fontname = "japan" if any(ord(char) > 127 for char in span.text) else "helv"
                page.insert_text(
                    (span.x, span.y),
                    span.text,
                    fontname=fontname,
                    fontsize=span.font_size,
                )
        pdf.save(pdf_path)
    finally:
        pdf.close()
    return pdf_path


def _context_predictions(context: Any) -> dict[str, list[str]]:
    return {
        "statement_titles": [fact.raw_title for fact in context.statement_titles],
        "section_headings": [
            heading.raw_heading
            for section in context.statement_sections
            for heading in section.headings
        ],
        "reporting_periods": [fact.raw_label for fact in context.reporting_periods],
        "units": [fact.raw_declaration for fact in context.units],
        "footnotes": [fact.raw_marker for fact in context.footnotes],
    }


def _document_text(document: PDFDocument) -> str:
    ordered: list[str] = []
    for page in document.pages:
        spans = sorted(
            page.text_spans,
            key=lambda span: (
                (span.bounding_box[1] + span.bounding_box[3]) / 2,
                span.bounding_box[0],
            ),
        )
        lines: list[list[Any]] = []
        for span in spans:
            center_y = (span.bounding_box[1] + span.bounding_box[3]) / 2
            height = max(span.bounding_box[3] - span.bounding_box[1], 1.0)
            matches = []
            for line in lines:
                line_top = min(item.bounding_box[1] for item in line)
                line_bottom = max(item.bounding_box[3] for item in line)
                line_center = (line_top + line_bottom) / 2
                line_height = max(line_bottom - line_top, 1.0)
                tolerance = max(2.0, min(height, line_height) * 0.3)
                if abs(center_y - line_center) <= tolerance:
                    matches.append((abs(center_y - line_center), line))
            if matches:
                min(matches, key=lambda item: item[0])[1].append(span)
            else:
                lines.append([span])
        lines.sort(
            key=lambda line: (
                sum((span.bounding_box[1] + span.bounding_box[3]) / 2 for span in line)
                / len(line)
            )
        )
        for line in lines:
            ordered.extend(span.raw_text for span in sorted(line, key=lambda item: item.bounding_box[0]))
    return "\n".join(ordered)


def _count_metric(
    expected_count: int,
    predicted_count: int,
    *,
    exhaustive: bool = True,
) -> dict[str, Any]:
    if not exhaustive:
        return {
            "precision": None,
            "recall": None,
            "f1": None,
            "true_positives": None,
            "false_positives": None,
            "false_negatives": None,
            "expected_count": expected_count,
            "predicted_count": predicted_count,
        }
    tp = min(expected_count, predicted_count)
    fp, fn = predicted_count - tp, expected_count - tp
    precision = tp / predicted_count if predicted_count else (1.0 if not expected_count else 0.0)
    recall = tp / expected_count if expected_count else (1.0 if not predicted_count else 0.0)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else 0.0
        if precision is not None and recall is not None
        else None
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "expected_count": expected_count,
        "predicted_count": predicted_count,
    }


def _counter_metric(expected: list[tuple[str, ...]], predicted: list[tuple[str, ...]], *, exhaustive: bool) -> dict[str, Any]:
    expected_counts, predicted_counts = Counter(expected), Counter(predicted)
    tp = sum((expected_counts & predicted_counts).values())
    fn = sum((expected_counts - predicted_counts).values())
    fp = sum((predicted_counts - expected_counts).values()) if exhaustive else None
    denominator = tp + (fp or 0)
    precision = (
        tp / denominator if denominator else (1.0 if exhaustive and not expected_counts else 0.0)
    ) if exhaustive else None
    recall = (
        tp / sum(expected_counts.values())
        if expected_counts
        else (1.0 if exhaustive and not predicted_counts else 0.0 if exhaustive else None)
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else (0.0 if precision is not None and recall is not None else None)
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "expected_count": sum(expected_counts.values()),
        "predicted_count": sum(predicted_counts.values()),
    }


def _cell_key(
    table_index: int,
    label: str,
    period: str,
    raw_value: str | None,
) -> tuple[str, ...]:
    return (str(table_index), _normalize_label(label), _normalize_label(period), raw_value or "")


def _same_expected(expected: Any, predicted: Any, *, decimal: bool = False) -> bool:
    if expected is None or predicted is None:
        return expected is predicted
    if isinstance(expected, tuple) or isinstance(predicted, tuple):
        return expected == predicted
    if decimal:
        try:
            return Decimal(expected) == Decimal(predicted)
        except (InvalidOperation, ValueError):
            return False
    return expected == predicted


def _normalize_label(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", value).split())


def _decimal_string(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _union_boxes(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _japanese_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN.finditer(text):
        token = match.group()
        tokens.extend(list(token) if _CJK_RUN.fullmatch(token) else [token])
    return tokens


def _edit_distance(reference: list[str], prediction: list[str]) -> int:
    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    previous = list(range(len(prediction) + 1))
    for index, reference_token in enumerate(reference, start=1):
        current = [index]
        for column, prediction_token in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_token != prediction_token),
                )
            )
        previous = current
    return previous[-1]


def _inside_dataset(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Dataset path escapes dataset root: {relative_path}")
    return path


def _flatten_metrics(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            flattened.update(_flatten_metrics(item, name))
        elif key in {
            "accuracy",
            "coverage",
            "mean_bbox_iou",
            "exact_reference_line_coverage",
            "cer",
            "wer",
        } or name.endswith((".f1", ".precision", ".recall")):
            flattened[name] = item
    return flattened


def _software_versions() -> dict[str, str | None]:
    packages = ("toresu", "PyMuPDF", "fastapi", "rapidocr", "onnxruntime")
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    versions["python"] = __import__("platform").python_version()
    return versions


def _hardware_info() -> dict[str, str | None]:
    return {
        "system": system(),
        "machine": machine(),
        "processor": processor() or None,
        "accelerator": "not requested; pipeline evaluation is CPU-compatible",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and benchmark Toresu.")
    parser.add_argument("command", choices=("evaluate", "benchmark"))
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--document", action="append", dest="documents")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation-results"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    selected = set(args.documents) if args.documents else None
    if args.command == "evaluate":
        report = evaluate_dataset(args.dataset, document_ids=selected)
        filename = "evaluation"
    else:
        report = benchmark_dataset(
            args.dataset,
            document_ids=selected,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        filename = "benchmark"
    json_path, summary_path = write_report(report, args.output_dir, filename=filename)
    print(render_summary(report), end="")
    print(f"JSON report: {json_path}")
    print(f"Text summary: {summary_path}")


if __name__ == "__main__":
    main()
