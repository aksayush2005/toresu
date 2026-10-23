from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf

from .evaluation import (
    _hardware_info,
    _materialize_source,
    _software_versions,
    evaluate_dataset,
    load_dataset,
)

_NUMBER_CANDIDATE = re.compile(
    r"[△▲+\-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"
)


def run_research_experiment(
    dataset_path: Path,
    *,
    document_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Run the full-pipeline evaluation and a documented raw-number baseline."""
    dataset_version, documents = load_dataset(dataset_path, document_ids)
    evaluation = evaluate_dataset(dataset_path, document_ids=document_ids)
    baseline_documents = []
    for loaded in documents:
        expected = [
            _canonical(value)
            for table in loaded.annotations.tables
            for cell in table.cells
            if (value := cell.raw_value) is not None and value.strip()
        ]
        try:
            with tempfile.TemporaryDirectory(prefix="toresu-research-") as temporary:
                pdf_path = _materialize_source(loaded, Path(temporary))
                start = time.perf_counter()
                predicted = _native_number_candidates(pdf_path)
                baseline_seconds = time.perf_counter() - start
            expected_counts, predicted_counts = Counter(expected), Counter(predicted)
            matched = sum((expected_counts & predicted_counts).values())
            expected_count, predicted_count = len(expected), len(predicted)
            baseline_documents.append(
                {
                    "document_id": loaded.entry.document_id,
                    "source_kind": loaded.entry.source_kind,
                    "success": True,
                    "annotated_numeric_cells": expected_count,
                    "native_numeric_candidates": predicted_count,
                    "exact_raw_value_matches": matched,
                    "precision": matched / predicted_count if predicted_count else None,
                    "recall": matched / expected_count if expected_count else None,
                    "wall_seconds": baseline_seconds,
                    "error": None,
                }
            )
        except Exception as error:
            baseline_documents.append(
                {
                    "document_id": loaded.entry.document_id,
                    "source_kind": loaded.entry.source_kind,
                    "success": False,
                    "annotated_numeric_cells": len(expected),
                    "native_numeric_candidates": None,
                    "exact_raw_value_matches": None,
                    "precision": None,
                    "recall": None,
                    "wall_seconds": None,
                    "error": {
                        "category": "native_text_baseline",
                        "code": type(error).__name__,
                        "message": str(error),
                    },
                }
            )

    aggregate = evaluation["aggregate_metrics"]
    return {
        "report_version": "1",
        "hypothesis": (
            "Preserving explicit evidence and ambiguity through reconstruction, "
            "normalization, and validation may reduce silently incorrect financial observations."
        ),
        "dataset_version": dataset_version,
        "preliminary": True,
        "systems": {
            "full_pipeline": {
                "definition": (
                    "Existing Toresu deterministic pipeline with its current provenance "
                    "and conservative ambiguity behavior."
                ),
                "successful_documents": evaluation["successful_documents"],
                "failed_documents": evaluation["failed_documents"],
                "document_results": [
                    {
                        "document_id": item["document_id"],
                        "success": item["success"],
                        "errors": item.get("errors", []),
                    }
                    for item in evaluation["documents"]
                ],
                "metrics": {
                    "raw_value_accuracy": aggregate.get(
                        "tables.cell_observations.raw_value_accuracy.accuracy"
                    ),
                    "exact_financial_observation_accuracy": aggregate.get(
                        "tables.cell_observations.exact_observation_accuracy.accuracy"
                    ),
                    "normalized_value_accuracy": aggregate.get(
                        "tables.cell_observations.normalized_value_accuracy.accuracy"
                    ),
                    "metric_mapping_accuracy": aggregate.get(
                        "tables.cell_observations.metric_mapping_accuracy.accuracy"
                    ),
                    "period_assignment_accuracy": aggregate.get(
                        "tables.cell_observations.period_assignment_accuracy.accuracy"
                    ),
                    "cell_detection_precision": aggregate.get(
                        "tables.cell_observations.detection.precision"
                    ),
                    "cell_detection_recall": aggregate.get(
                        "tables.cell_observations.detection.recall"
                    ),
                    "provenance_coverage": aggregate.get("provenance.coverage"),
                    "silent_error_rate": None,
                    "median_document_wall_seconds": _median_defined(
                        item["wall_seconds"]
                        for item in evaluation["documents"]
                        if item.get("success")
                    ),
                },
            },
            "native_numeric_token_baseline": {
                "definition": (
                    "PyMuPDF native text extraction followed by a conservative numeric-token "
                    "regular expression. It does not assign tokens to rows, metrics, or periods."
                ),
                "documents": baseline_documents,
                "macro_precision": _mean_defined(
                    item["precision"] for item in baseline_documents if item["success"]
                ),
                "macro_recall": _mean_defined(
                    item["recall"] for item in baseline_documents if item["success"]
                ),
                "median_document_wall_seconds": _median_defined(
                    item["wall_seconds"]
                    for item in baseline_documents
                    if item["success"]
                ),
                "latency_note": "Single measured run per document; no warm-up or repetition.",
                "not_comparable_to_financial_observation_accuracy": True,
            },
        },
        "ablations": {
            "status": "not_estimable",
            "reason": (
                "The current pipeline exposes no independent switches for provenance checks "
                "or ambiguity resolution, and this dataset contains no annotated ambiguous "
                "mapping cases. No behavior was artificially disabled to manufacture a result."
            ),
        },
        "silent_error_analysis": {
            "rate": None,
            "reason": (
                "The dataset does not label whether each incorrect emitted observation was "
                "surfaced to a user as an uncertainty or warning; that ground truth is required."
            ),
        },
        "reproducibility": {
            "command": (
                "PYTHONPATH=src python -m toresu.research "
                f"--dataset {dataset_path}"
            ),
            "configuration": {
                "external_llm": "not used",
                "pipeline": "Toresu deterministic pipeline",
                "dataset_path": str(dataset_path.resolve()),
                "selected_document_ids": sorted(document_ids) if document_ids else None,
            },
            "software": _software_versions(),
            "hardware": _hardware_info(),
        },
        "limitations": [
            "Only the checked-in synthetic native-text document is currently available.",
            "The numeric-token baseline evaluates value detection only and cannot measure semantic assignment.",
            "The result is a preliminary engineering check, not a statistical research conclusion.",
            "No real-document accuracy, silent-error reduction, or causal ablation claim is supported.",
        ],
    }


def _native_number_candidates(pdf_path: Path) -> list[str]:
    document = pymupdf.open(pdf_path)
    try:
        text = "\n".join(page.get_text("text", sort=True) for page in document)
    finally:
        document.close()
    return [_canonical(match) for match in _NUMBER_CANDIDATE.findall(text)]


def _canonical(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip()


def _mean_defined(values) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return sum(defined) / len(defined) if defined else None


def _median_defined(values) -> float | None:
    defined = sorted(float(value) for value in values if value is not None)
    if not defined:
        return None
    middle = len(defined) // 2
    if len(defined) % 2:
        return defined[middle]
    return (defined[middle - 1] + defined[middle]) / 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preliminary Toresu research experiment."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--document", action="append", dest="documents")
    parser.add_argument("--output-dir", type=Path, default=Path("research-results"))
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    report = run_research_experiment(
        arguments.dataset,
        document_ids=set(arguments.documents) if arguments.documents else None,
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = arguments.output_dir / "research.json"
    text_path = arguments.output_dir / "research.txt"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = [
        "Toresu preliminary research report",
        f"Dataset version: {report['dataset_version']}",
        f"Full pipeline metrics: {report['systems']['full_pipeline']['metrics']}",
        (
            "Native numeric-token baseline macro precision/recall: "
            f"{report['systems']['native_numeric_token_baseline']['macro_precision']} / "
            f"{report['systems']['native_numeric_token_baseline']['macro_recall']}"
        ),
        f"Ablations: {report['ablations']['status']}",
        "Conclusion: preliminary; no statistical or causal claim is supported.",
    ]
    text_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    print(f"JSON report: {json_path}")
    print(f"Text summary: {text_path}")


if __name__ == "__main__":
    main()
