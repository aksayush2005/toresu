import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from toresu.evaluation import (
    AnnotationScope,
    CellAnnotation,
    DocumentAnnotations,
    TableAnnotations,
    _score_cells,
    _same_expected,
    bbox_iou,
    benchmark_dataset,
    character_error_rate,
    evaluate_dataset,
    evaluate_table_reference,
    load_dataset,
    precision_recall_f1,
    word_error_rate,
    write_report,
)

_DATASET = (
    Path(__file__).resolve().parents[1] / "evaluation" / "datasets" / "v1"
)


def test_versioned_dataset_loads_and_selects_documents():
    version, documents = load_dataset(_DATASET)

    assert version == "1.0.0"
    assert [document.entry.document_id for document in documents] == [
        "synthetic-jp-income-v1"
    ]
    assert documents[0].layout is not None
    assert documents[0].annotations.context.statement_titles == ["連結損益計算書"]
    assert documents[0].annotations.document_metadata["synthetic"] is True
    assert documents[0].annotations.annotation_status == "complete"
    assert len(documents[0].annotations.tables[0].columns) == 2

    _, selected = load_dataset(_DATASET, {"synthetic-jp-income-v1"})
    assert len(selected) == 1


def test_annotation_schema_rejects_missing_fields_and_non_decimal_values():
    schema = json.loads(
        (
            _DATASET.parents[1]
            / "schemas"
            / "annotation-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert set(schema["required"]) == {
        name
        for name, field in DocumentAnnotations.model_fields.items()
        if field.is_required()
    }
    assert set(schema["properties"]["exhaustive"]["required"]) == set(
        AnnotationScope.model_fields
    )

    annotation = json.loads(
        (_DATASET / "annotations" / "synthetic-jp-income.json").read_text(
            encoding="utf-8"
        )
    )
    del annotation["exhaustive"]["table_cells"]
    with pytest.raises(ValidationError):
        DocumentAnnotations.model_validate(annotation)

    annotation = json.loads(
        (_DATASET / "annotations" / "synthetic-jp-income.json").read_text(
            encoding="utf-8"
        )
    )
    annotation["tables"][0]["cells"][0]["normalized_value"] = "NaN"
    with pytest.raises(ValidationError):
        DocumentAnnotations.model_validate(annotation)

    annotation = json.loads(
        (_DATASET / "annotations" / "synthetic-jp-income.json").read_text(
            encoding="utf-8"
        )
    )
    annotation["validation_findings"] = [
        {
            "rule_id": "balance-sheet-assets-equal-liabilities-and-equity",
            "status": "failed",
            "period_label": "2025年度",
        }
    ]
    validated = DocumentAnnotations.model_validate(annotation)
    assert validated.validation_findings[0].status == "failed"


def test_japanese_character_and_token_error_rates_handle_empty_inputs():
    assert character_error_rate("売上高", "売上高") == 0
    assert word_error_rate("売上高 100円", "売上高 100円") == 0
    assert character_error_rate("売上高", "売上") == pytest.approx(1 / 3)
    assert word_error_rate("売上高", "売上") == pytest.approx(1 / 3)
    assert character_error_rate("", "") == 0
    assert word_error_rate("", "") == 0
    assert character_error_rate("", "abc") == 3
    assert word_error_rate("", "abc") == 1


def test_precision_recall_explicitly_handles_partial_and_empty_annotations():
    partial = precision_recall_f1(
        ["売上高"], ["売上高", "営業利益"], exhaustive=False
    )
    assert partial["precision"] is None
    assert partial["recall"] == 1
    assert partial["f1"] is None
    assert partial["false_positives"] is None

    unreviewed = precision_recall_f1([], ["営業利益"], exhaustive=False)
    assert unreviewed["precision"] is None
    assert unreviewed["recall"] is None
    assert unreviewed["f1"] is None

    exhaustive = precision_recall_f1([], [], exhaustive=True)
    assert exhaustive["precision"] == exhaustive["recall"] == exhaustive["f1"] == 1
    empty_prediction = precision_recall_f1(["売上高"], [], exhaustive=True)
    assert empty_prediction["precision"] == empty_prediction["recall"] == 0


def test_missing_and_ambiguous_financial_annotations_remain_explicit():
    _, documents = load_dataset(_DATASET)
    annotation = documents[0].annotations.model_copy(
        update={
            "tables": [
                TableAnnotations(
                    page_number=1,
                    row_labels=["売上高"],
                    periods=["当連結会計年度"],
                    header_rows=[["当連結会計年度"]],
                    cells=[
                        CellAnnotation(
                            row_label="売上高",
                            period="当連結会計年度",
                            raw_value=None,
                            parsed_value=None,
                            normalized_value=None,
                            normalized_unit=None,
                            value_kind="dash",
                            metric=None,
                            page_number=1,
                        )
                    ],
                )
            ]
        }
    )

    metrics, provenance, errors = _score_cells(annotation, [], ())

    assert metrics["raw_value_accuracy"]["accuracy"] == 0
    assert metrics["parsed_value_accuracy"]["accuracy"] == 0
    assert metrics["metric_mapping_accuracy"]["accuracy"] == 0
    assert provenance["coverage"] == 0
    assert any(error["category"] == "missing_or_ambiguous_observation" for error in errors)


def test_financial_comparisons_use_decimal_and_bbox_iou_is_geometry_aware():
    assert _same_expected("1234.500", "1234.5", decimal=True)
    assert not _same_expected("1234.50", "1234.51", decimal=True)
    assert not _same_expected("１２３", "123")
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert bbox_iou((0, 0, 10, 10), (10, 10, 20, 20)) == 0
    assert Decimal("0.10") + Decimal("0.20") == Decimal("0.30")


def test_table_reference_metrics_support_japanese_and_numeric_formatting():
    report = evaluate_table_reference(
        [
            {"row": " 売上 高 ", "2025年度": "1000.00", "2024年度": "△20"},
            {"row": "営業利益", "2025年度": "0", "2024年度": "－"},
        ],
        [
            {"項目": "売上高", "2025年度": "1,000", "2024年度": "△20"},
            {"項目": "営業利益", "2025年度": "0", "2024年度": "－"},
        ],
        ["2025年度", "2024年度"],
        ["2025年度", "2024年度"],
        reference_label_column="項目",
    )

    assert report["evaluated"] is True
    assert report["row_label_accuracy"]["accuracy"] == 1
    assert report["column_alignment_accuracy"]["accuracy"] == 1
    assert report["cell_exact_match_accuracy"]["accuracy"] == pytest.approx(3 / 4)
    assert report["numeric_cell_accuracy"] == {
        "correct": 3,
        "total": 3,
        "accuracy": 1,
    }
    assert [item["category"] for item in report["mismatches"]] == [
        "numeric_format_difference"
    ]


def test_table_reference_metrics_report_dimensions_and_missing_extra_cells():
    report = evaluate_table_reference(
        [
            {"row": "売上高", "2025年度": "100"},
            {"row": "余分な行", "2025年度": "5"},
            {"row": "追加行", "2025年度": "3"},
        ],
        [
            {"row": "売上高", "2025年度": "100", "2024年度": "90"},
            {"row": "営業利益", "2025年度": "20", "2024年度": "15"},
        ],
        ["2025年度"],
        ["2025年度", "2024年度"],
    )

    assert report["row_count_agreement"] is False
    assert report["column_count_agreement"] is False
    assert report["table_structure_accuracy"]["exact_agreement"] is False
    categories = [mismatch["category"] for mismatch in report["mismatches"]]
    assert "missing_column" in categories
    assert "misaligned_row" in categories
    assert "extra_row" in categories
    assert "extra_cell" in categories
    assert report["cell_exact_match_accuracy"]["total"] == 6


def test_table_reference_comparison_distinguishes_sign_magnitude_dash_and_zero():
    report = evaluate_table_reference(
        [
            {
                "row": "金額",
                "same_magnitude": "△10",
                "different_magnitude": "△21",
                "zero_or_dash": "－",
            }
        ],
        [
            {
                "row": "金額",
                "same_magnitude": "10",
                "different_magnitude": "△20",
                "zero_or_dash": "0",
            }
        ],
        ["same_magnitude", "different_magnitude", "zero_or_dash"],
        ["same_magnitude", "different_magnitude", "zero_or_dash"],
    )

    assert report["numeric_cell_accuracy"] == {
        "correct": 0,
        "total": 3,
        "accuracy": 0,
    }
    assert [item["category"] for item in report["mismatches"]] == [
        "numeric_sign_mismatch",
        "numeric_magnitude_mismatch",
        "value_kind_mismatch",
    ]


def test_table_reference_comparison_flags_fiscal_period_alignment():
    report = evaluate_table_reference(
        [{"row": "売上高", "2024年度": "90", "2025年度": "100"}],
        [{"row": "売上高", "2025年度": "100", "2024年度": "90"}],
        ["2024年度", "2025年度"],
        ["2025年度", "2024年度"],
    )

    assert report["column_alignment_accuracy"]["accuracy"] == 0
    assert report["numeric_cell_accuracy"]["accuracy"] == 0
    fiscal_period_mismatches = [
        item for item in report["mismatches"]
        if item["category"] == "fiscal_period_alignment_mismatch"
    ]
    assert len(fiscal_period_mismatches) == 2
    assert all(
        mismatch["numeric_issue"] == "numeric_magnitude_mismatch"
        for mismatch in fiscal_period_mismatches
    )


def test_table_reference_metrics_are_not_evaluated_without_ground_truth():
    report = evaluate_table_reference(
        [{"row": "売上高", "2025年度": "100"}],
        None,
        ["2025年度"],
        None,
    )

    assert report == {
        "evaluated": False,
        "reason": "Reference table is unavailable.",
    }


def test_end_to_end_evaluation_scores_synthetic_japanese_fixture():
    report = evaluate_dataset(_DATASET)

    assert report["dataset_version"] == "1.0.0"
    assert report["document_count"] == 1
    assert report["successful_documents"] == 1
    result = report["documents"][0]
    assert result["source_kind"] == "synthetic_pdf"
    assert result["metrics"]["text"]["cer"] == 0
    assert result["metrics"]["text"]["wer"] == 0
    assert result["metrics"]["context"]["statement_titles"]["f1"] == 1
    assert result["metrics"]["tables"]["detection"]["f1"] == 1
    assert result["metrics"]["tables"]["cell_observations"]["exact_observation_accuracy"]["accuracy"] == 1
    assert result["metrics"]["tables"]["cell_observations"]["normalized_value_accuracy"]["accuracy"] == 1
    assert result["metrics"]["provenance"]["coverage"] == 1
    assert result["metrics"]["financial_validation"] is None
    assert report["aggregate_metrics"]["text.cer"]["macro_mean"] == 0


def test_failed_pdf_is_reported_instead_of_omitted(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "sources").mkdir(parents=True)
    (dataset / "annotations").mkdir()
    (dataset / "sources" / "invalid.pdf").write_bytes(b"not a pdf")
    annotation = json.loads(
        (_DATASET / "annotations" / "synthetic-jp-income.json").read_text(
            encoding="utf-8"
        )
    )
    annotation["document_id"] = "broken-pdf"
    (dataset / "annotations" / "broken.json").write_text(
        json.dumps(annotation, ensure_ascii=False), encoding="utf-8"
    )
    (dataset / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": "test",
                "documents": [
                    {
                        "document_id": "broken-pdf",
                        "source_kind": "pdf",
                        "source_file": "sources/invalid.pdf",
                        "annotation_file": "annotations/broken.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_dataset(dataset)

    assert report["document_count"] == 1
    assert report["failed_documents"] == 1
    assert report["documents"][0]["success"] is False
    assert report["documents"][0]["errors"][0]["category"] == "ingestion"


def test_dataset_loader_rejects_path_escape(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": "test",
                "documents": [
                    {
                        "document_id": "escape",
                        "source_kind": "pdf",
                        "source_file": "../outside.pdf",
                        "annotation_file": "../outside.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes dataset root"):
        load_dataset(dataset)


def test_benchmark_report_records_repeats_stages_and_memory_method():
    report = benchmark_dataset(_DATASET, repeats=2, warmup=1)

    assert report["successful_runs"] == 2
    assert report["failed_runs"] == 0
    assert report["throughput_documents_per_second"] > 0
    assert report["methodology"]["warmup_runs_per_document"] == 1
    assert report["methodology"]["measured_repeats_per_document"] == 2
    sample = report["documents"][0]["repeats"][0]
    assert sample["wall_seconds"] > 0
    assert sample["seconds_per_page"] > 0
    assert sample["python_tracemalloc_peak_bytes"] > 0
    assert set(sample["stage_seconds"]) == {
        "ingestion",
        "context_compilation",
        "table_reconstruction",
        "financial_normalization",
        "financial_validation",
    }


def test_report_serialization_is_stable_for_same_result(tmp_path):
    report = {
        "report_version": "1",
        "dataset_version": "test",
        "documents": [],
        "aggregate_metrics": {},
        "successful_documents": 0,
        "failed_documents": 0,
    }
    first, _ = write_report(report, tmp_path / "one", filename="result")
    second, _ = write_report(report, tmp_path / "two", filename="result")
    assert first.read_bytes() == second.read_bytes()
