from __future__ import annotations

from pathlib import Path
from threading import Event

import pymupdf
import pytest
from fastapi.testclient import TestClient

from toresu.api.app import create_app
from toresu.api.jobs import JobManager
from toresu.api.pipeline import PipelineArtifacts, PDFPreflightError
from toresu.api.schemas import JobState, StageName, StageState
from toresu.api.settings import APISettings
from toresu.pdf_ingestion import PDFDocument, PDFPage


def _pdf_bytes(page_count: int = 1, *, text: bool = True) -> bytes:
    document = pymupdf.open()
    for index in range(page_count):
        page = document.new_page(width=600, height=400)
        if text:
            page.insert_text((30, 35), "Synthetic financial statement")
            page.insert_text((30, 60), "Financial year columns")
            page.insert_text((30, 80), "Revenue")
            page.insert_text((320, 80), "1000.25")
            page.insert_text((470, 80), "900.00")
            page.insert_text((30, 100), "Operating cost")
            page.insert_text((320, 100), "750.10")
            page.insert_text((470, 100), "600.00")
    return document.tobytes()


def _settings(tmp_path: Path, **overrides) -> APISettings:
    return APISettings(data_dir=tmp_path / "api-data", **overrides)


def _upload(client: TestClient, content: bytes, filename: str = "../client-name.pdf", content_type: str = "application/octet-stream"):
    return client.post(
        "/v1/documents",
        files={"file": (filename, content, content_type)},
    )


def _wait_for_job(client: TestClient, job_id: str):
    manager: JobManager = client.app.state.job_manager
    record = manager.get(job_id)
    assert record is not None and record.future is not None
    record.future.result(timeout=10)
    return client.get(f"/v1/jobs/{job_id}")


def test_health_readiness_and_public_limits(tmp_path):
    app = create_app(_settings(tmp_path, max_upload_bytes=1234, max_pdf_pages=12))
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        ready = client.get("/ready").json()
        assert ready["status"] == "ready"
        assert ready["accepting_jobs"]
        assert client.get("/v1/config").json()["max_upload_bytes"] == 1234


def test_synthetic_pdf_completes_pipeline_and_serializes_decimal_as_string(tmp_path):
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        uploaded = _upload(client, _pdf_bytes())
        assert uploaded.status_code == 202
        payload = uploaded.json()
        assert payload["job_id"]
        assert payload["status_url"] == f"/v1/jobs/{payload['job_id']}"
        stored_sources = list((tmp_path / "api-data" / "jobs").glob("*/source.pdf"))
        assert len(stored_sources) == 1
        assert stored_sources[0].name == "source.pdf"

        status_response = _wait_for_job(client, payload["job_id"])
        assert status_response.status_code == 200
        status_payload = status_response.json()
        assert status_payload["state"] in {"completed", "completed_with_warnings"}
        assert [stage["state"] for stage in status_payload["stages"]] == [
            "completed",
            "completed",
            "completed",
            "completed",
            "completed",
        ]

        result_response = client.get(payload["result_url"])
        assert result_response.status_code == 200
        result = result_response.json()
        assert not result["partial"]
        assert result["tables"]
        assert result["observations"]
        observation = next(
            item for item in result["observations"] if item["raw_value"] == "1000.25"
        )
        assert isinstance(observation["parsed_value"], str)
        assert observation["parsed_value"] == "1000.25"
        assert observation["source_spans"]
        assert observation["source_spans"][0]["page_number"] == 1
        assert observation["source_spans"][0]["bounding_box"]

        validation = client.get(payload["validation_url"])
        assert validation.status_code == 200
        assert validation.json()["job_id"] == payload["job_id"]


def test_invalid_signature_and_untrusted_mime_are_handled_by_signature(tmp_path):
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        invalid = _upload(client, b"not a pdf", content_type="application/pdf")
        assert invalid.status_code == 415
        assert invalid.json()["error_code"] == "invalid_pdf_signature"

        valid = _upload(client, _pdf_bytes(), filename="anything.txt", content_type="text/plain")
        assert valid.status_code == 202
        _wait_for_job(client, valid.json()["job_id"])


def test_malformed_encrypted_empty_and_page_limited_pdfs(tmp_path, monkeypatch):
    settings = _settings(tmp_path, max_pdf_pages=1)
    app = create_app(settings)
    with TestClient(app) as client:
        malformed = _upload(client, b"%PDF-not-a-document")
        assert malformed.status_code == 422
        assert malformed.json()["error_code"] == "malformed_pdf"

        two_pages = _upload(client, _pdf_bytes(page_count=2))
        assert two_pages.status_code == 413
        assert two_pages.json()["error_code"] == "page_limit_exceeded"

        encrypted = pymupdf.open()
        encrypted.new_page()
        encrypted_bytes = encrypted.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="owner",
            user_pw="user",
        )
        encrypted.close()
        encrypted_response = _upload(client, encrypted_bytes)
        assert encrypted_response.status_code == 422
        assert encrypted_response.json()["error_code"] == "encrypted_pdf"

        # A blank single-page PDF exercises the API/pipeline path without model downloads.
        blank_pdf = _pdf_bytes(text=False)

        def ingest_blank(_path):
            return PDFDocument(
                source_path=Path("synthetic-blank.pdf"),
                pages=(
                    PDFPage(
                        page_number=1,
                        bounding_box=(0.0, 0.0, 600.0, 400.0),
                        text_spans=(),
                        native_text_usable=False,
                        ocr_attempted=True,
                    ),
                ),
            )

        monkeypatch.setattr("toresu.api.pipeline.ingest_pdf", ingest_blank)
        empty_response = _upload(client, blank_pdf)
        assert empty_response.status_code == 202
        completed = _wait_for_job(client, empty_response.json()["job_id"])
        assert completed.json()["state"] in {"completed", "completed_with_warnings"}


def test_zero_page_pdf_is_rejected(tmp_path, monkeypatch):
    upload_bytes = _pdf_bytes()

    class EmptyDocument:
        is_pdf = True
        needs_pass = False
        page_count = 0

        def close(self):
            pass

    monkeypatch.setattr("toresu.api.pipeline.pymupdf.open", lambda _path: EmptyDocument())
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        response = _upload(client, upload_bytes)

    assert response.status_code == 422
    assert response.json()["error_code"] == "empty_pdf"


def test_upload_size_limit_and_rejected_upload_cleanup(tmp_path):
    app = create_app(_settings(tmp_path, max_upload_bytes=64))
    with TestClient(app) as client:
        response = _upload(client, _pdf_bytes())

    assert response.status_code == 413
    assert response.json()["error_code"] == "upload_too_large"
    staging_dirs = list((tmp_path / "api-data" / "staging").iterdir())
    assert staging_dirs == []


def test_missing_job_ids_return_structured_404(tmp_path):
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        for endpoint in ("", "/result", "/validation"):
            response = client.get(f"/v1/jobs/missing-id{endpoint}")
            assert response.status_code == 404
            assert response.json()["error_code"] == "job_not_found"


def test_stage_failure_returns_partial_result_and_marks_later_stages_skipped(tmp_path, monkeypatch):
    app = create_app(_settings(tmp_path))

    def fail_context(_document):
        raise ValueError("private parser details must not leak")

    monkeypatch.setattr("toresu.api.pipeline.compile_financial_context", fail_context)
    with TestClient(app) as client:
        uploaded = _upload(client, _pdf_bytes())
        assert uploaded.status_code == 202
        job_id = uploaded.json()["job_id"]
        status_response = _wait_for_job(client, job_id)

        state = status_response.json()
        assert state["state"] == "failed"
        assert state["error_code"] == "context_compilation_failed"
        assert state["error_message"] == "Context compilation failed."
        assert state["stages"][0]["state"] == "completed"
        assert state["stages"][1]["state"] == "failed"
        assert all(stage["state"] == "skipped" for stage in state["stages"][2:])

        result = client.get(f"/v1/jobs/{job_id}/result").json()
        assert result["partial"]
        assert result["failed_stage"] == "context_compilation"
        assert result["stage_errors"][0]["error_code"] == "context_compilation_failed"
        assert result["tables"] == []


def test_late_stage_failure_returns_completed_upstream_artifacts(tmp_path, monkeypatch):
    def fail_validation(*_args, **_kwargs):
        raise RuntimeError("private validation detail")

    monkeypatch.setattr("toresu.api.pipeline.validate_financial_data", fail_validation)
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        uploaded = _upload(client, _pdf_bytes())
        job_id = uploaded.json()["job_id"]
        state = _wait_for_job(client, job_id).json()
        result = client.get(f"/v1/jobs/{job_id}/result").json()

    assert state["state"] == "failed"
    assert state["stages"][2]["state"] == "completed"
    assert state["stages"][3]["state"] == "completed"
    assert state["stages"][4]["state"] == "failed"
    assert result["partial"]
    assert result["tables"]
    assert result["observations"]
    assert result["validation_findings"] == []
    assert "private validation detail" not in str(result)


def test_running_job_returns_conflict_for_results_and_queue_is_bounded(tmp_path):
    entered = Event()
    release = Event()

    def blocking_runner(_path, *, settings, on_stage):
        for stage in StageName:
            on_stage(stage, StageState.PROCESSING, None, None)
            if stage is StageName.INGESTION:
                entered.set()
                assert release.wait(timeout=5)
            on_stage(stage, StageState.COMPLETED, None, None)
        return PipelineArtifacts()

    app = create_app(
        _settings(tmp_path, max_workers=1, max_queued_jobs=0),
        pipeline_runner=blocking_runner,
    )
    with TestClient(app) as client:
        first = _upload(client, _pdf_bytes())
        first_id = first.json()["job_id"]
        assert entered.wait(timeout=3)
        assert client.get(f"/v1/jobs/{first_id}").json()["state"] == "processing"
        assert client.get(f"/v1/jobs/{first_id}/result").status_code == 409
        second = _upload(client, _pdf_bytes())
        assert second.status_code == 503
        assert second.json()["error_code"] == "job_capacity_exceeded"
        release.set()
        assert _wait_for_job(client, first_id).json()["state"] == "completed"


def test_job_manager_state_transitions_are_deterministic(tmp_path):
    manager = JobManager(_settings(tmp_path))
    try:
        upload_path = tmp_path / "incoming.pdf"
        upload_path.write_bytes(_pdf_bytes())
        record = manager.submit(upload_path)
        assert record.state is JobState.QUEUED
        record.future.result(timeout=10)
        assert record.state in {JobState.COMPLETED, JobState.COMPLETED_WITH_WARNINGS}
        assert record.source_path.name == "source.pdf"
        assert record.source_path.parent == record.directory
    finally:
        manager.close()


def test_decimal_response_model_serializes_values_as_strings():
    from decimal import Decimal

    from toresu.api.schemas import NormalizedObservationResponse, PeriodResponse

    model = NormalizedObservationResponse(
        table_id="t1",
        row_id="r1",
        cell_id="c1",
        row_label_raw="売上高",
        metric_candidates=("revenue",),
        metric_status="inferred",
        metric_issue=None,
        period=PeriodResponse(
            raw_label="2025年度",
            kind="fiscal_year",
            fiscal_year=2025,
            quarter=None,
            relative_period=None,
            date_iso=None,
            status="resolved",
        ),
        raw_value="1,234.50",
        parsed_value=Decimal("1234.50"),
        value_kind="amount",
        value_status="resolved",
        raw_unit_declaration="（単位：千円）",
        original_unit="thousand_yen",
        normalized_value=Decimal("1234500.00"),
        normalized_unit="yen",
        unit_status="resolved",
        issue=None,
        source_spans=(),
    )

    encoded = model.model_dump_json()
    assert '"parsed_value":"1234.50"' in encoded
    assert '"normalized_value":"1234500.00"' in encoded
