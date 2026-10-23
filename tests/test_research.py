from pathlib import Path

from toresu.research import run_research_experiment

_DATASET = (
    Path(__file__).resolve().parents[1] / "evaluation" / "datasets" / "v1"
)


def test_preliminary_experiment_runs_full_pipeline_and_reports_baseline_limits():
    report = run_research_experiment(_DATASET)

    assert report["dataset_version"] == "1.0.0"
    assert report["preliminary"] is True
    assert report["systems"]["full_pipeline"]["successful_documents"] == 1
    assert report["systems"]["full_pipeline"]["metrics"]["normalized_value_accuracy"] == {
        "macro_mean": 1.0,
        "documents_scored": 1,
    }
    assert report["systems"]["full_pipeline"]["metrics"][
        "exact_financial_observation_accuracy"
    ] == {"macro_mean": 1.0, "documents_scored": 1}
    baseline = report["systems"]["native_numeric_token_baseline"]
    assert baseline["documents"][0]["annotated_numeric_cells"] == 4
    assert baseline["documents"][0]["exact_raw_value_matches"] == 4
    assert baseline["not_comparable_to_financial_observation_accuracy"] is True
    assert report["ablations"]["status"] == "not_estimable"
    assert report["silent_error_analysis"]["rate"] is None
