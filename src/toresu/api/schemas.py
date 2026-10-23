from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class JobState(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"


class StageName(str, Enum):
    INGESTION = "ingestion"
    CONTEXT_COMPILATION = "context_compilation"
    TABLE_RECONSTRUCTION = "table_reconstruction"
    FINANCIAL_NORMALIZATION = "financial_normalization"
    FINANCIAL_VALIDATION = "financial_validation"


class StageState(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorResponse(APIModel):
    error_code: str
    detail: str


class HealthResponse(APIModel):
    status: str = "ok"


class ReadinessResponse(APIModel):
    status: str
    accepting_jobs: bool
    available_worker_slots: int


class SpanResponse(APIModel):
    raw_text: str
    page_number: int
    bounding_box: tuple[float, float, float, float]
    extraction_source: str
    confidence: float | None = None


class CellResponse(APIModel):
    cell_id: str
    role: str
    raw_text: str | None
    cell_boundary: tuple[float, float, float, float] | None
    row_index: int | None
    column_index: int | None
    column_span: int
    status: str
    issue: str | None = None
    source_spans: tuple[SpanResponse, ...] = ()


class HeaderResponse(APIModel):
    header_id: str
    header_row: int
    raw_text: str
    column_start: int | None
    column_span: int
    is_merged: bool
    status: str
    issue: str | None = None
    source_spans: tuple[SpanResponse, ...] = ()


class ColumnResponse(APIModel):
    column_index: int
    left: float
    right: float
    center: float
    header_path: tuple[HeaderResponse, ...]
    period_label_raw: str | None
    period_status: str
    geometry_status: str
    is_auxiliary: bool = False


class RowResponse(APIModel):
    row_id: str
    row_index: int
    bounding_box: tuple[float, float, float, float]
    label_cell: CellResponse | None
    cells: tuple[CellResponse, ...]
    parent_row_id: str | None
    hierarchy_status: str


class TableResponse(APIModel):
    table_id: str
    page_number: int
    bounding_box: tuple[float, float, float, float]
    statement_title_raw: str | None
    columns: tuple[ColumnResponse, ...]
    header_rows: tuple[tuple[HeaderResponse, ...], ...]
    rows: tuple[RowResponse, ...]
    context_spans: tuple[SpanResponse, ...]
    status: str
    issues: tuple[str, ...]


class PeriodResponse(APIModel):
    raw_label: str
    kind: str
    fiscal_year: int | None
    quarter: int | None
    relative_period: str | None
    date_iso: str | None
    status: str
    issue: str | None = None
    evidence_spans: tuple[SpanResponse, ...] = ()


class NormalizedObservationResponse(APIModel):
    table_id: str
    row_id: str
    cell_id: str
    row_label_raw: str | None
    row_label_source_spans: tuple[SpanResponse, ...]
    metric_candidates: tuple[str, ...]
    metric_status: str
    metric_issue: str | None
    period: PeriodResponse
    raw_value: str | None
    parsed_value: Decimal | None
    value_kind: str
    value_status: str
    raw_unit_declaration: str | None
    original_unit: str | None
    normalized_value: Decimal | None
    normalized_unit: str | None
    unit_status: str
    issue: str | None
    source_spans: tuple[SpanResponse, ...]
    unit_source_spans: tuple[SpanResponse, ...]


class ValidationEvidenceResponse(APIModel):
    table_ids: tuple[str, ...]
    cell_ids: tuple[str, ...]
    spans: tuple[SpanResponse, ...]


class ValidationFindingResponse(APIModel):
    rule_id: str
    status: str
    message: str
    period_label: str | None
    metric_concepts: tuple[str, ...]
    observed_residual: Decimal | None
    tolerance: Decimal | None
    evidence: ValidationEvidenceResponse


class StageResponse(APIModel):
    name: StageName
    state: StageState
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None


class UploadResponse(APIModel):
    job_id: str
    state: JobState
    status_url: str
    result_url: str
    validation_url: str


class JobStatusResponse(APIModel):
    job_id: str
    state: JobState
    created_at: datetime
    updated_at: datetime
    stages: tuple[StageResponse, ...]
    error_code: str | None = None
    error_message: str | None = None


class ProcessingResultResponse(APIModel):
    job_id: str
    state: JobState
    partial: bool
    failed_stage: StageName | None
    tables: tuple[TableResponse, ...] = ()
    observations: tuple[NormalizedObservationResponse, ...] = ()
    validation_findings: tuple[ValidationFindingResponse, ...] = ()
    stage_errors: tuple[StageResponse, ...] = ()
    warnings: tuple[str, ...] = ()
    document_report: dict[str, Any] | None = None


class ValidationResultResponse(APIModel):
    job_id: str
    state: JobState
    findings: tuple[ValidationFindingResponse, ...]
    stage_errors: tuple[StageResponse, ...] = ()


class APISettingsResponse(APIModel):
    max_upload_bytes: int = Field(gt=0)
    max_pdf_pages: int = Field(gt=0)
    max_workers: int = Field(gt=0)
    max_queued_jobs: int = Field(ge=0)
    max_retained_jobs: int = Field(gt=0)