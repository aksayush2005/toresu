from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pymupdf

from ..context_compiler import FinancialDocumentContext, compile_financial_context
from ..document_intelligence import build_document_report
from ..financial_normalization import (
    FinancialNormalizationResult,
    NormalizedObservation,
    normalize_financial_tables,
)
from ..financial_validation import (
    ValidationConfig,
    ValidationFinding,
    ValidationReport,
    validate_financial_data,
)
from ..pdf_ingestion import PDFDocument, PDFIngestionError, TextSpan, ingest_pdf
from ..table_reconstruction import (
    ReconstructedCell,
    ReconstructedHeaderCell,
    ReconstructedTable,
    TableReconstructionResult,
    reconstruct_tables,
)
from .schemas import (
    CellResponse,
    ColumnResponse,
    HeaderResponse,
    JobState,
    NormalizedObservationResponse,
    PeriodResponse,
    ProcessingResultResponse,
    RowResponse,
    StageName,
    StageResponse,
    StageState,
    TableResponse,
    ValidationEvidenceResponse,
    ValidationFindingResponse,
    ValidationResultResponse,
)
from .settings import APISettings


@dataclass
class PipelineArtifacts:
    document: PDFDocument | None = None
    context: FinancialDocumentContext | None = None
    reconstruction: TableReconstructionResult | None = None
    normalization: FinancialNormalizationResult | None = None
    validation: ValidationReport | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StageFailure:
    stage: StageName
    error_code: str
    public_message: str


class PipelineExecutionError(RuntimeError):
    def __init__(
        self,
        failure: StageFailure,
        artifacts: PipelineArtifacts,
    ) -> None:
        super().__init__(failure.public_message)
        self.failure = failure
        self.artifacts = artifacts


class PDFPreflightError(ValueError):
    def __init__(self, error_code: str, public_message: str) -> None:
        super().__init__(public_message)
        self.error_code = error_code
        self.public_message = public_message


StageCallback = Callable[[StageName, StageState, str | None, str | None], None]
PipelineRunner = Callable[..., PipelineArtifacts]


def preflight_pdf(path: Path, max_pages: int) -> int:
    """Check openability, encryption, and page limits without extracting content."""
    try:
        document = pymupdf.open(path)
    except Exception as error:
        raise PDFPreflightError("malformed_pdf", "The uploaded file is not a readable PDF.") from error
    try:
        if not document.is_pdf:
            raise PDFPreflightError("invalid_pdf", "The uploaded file is not a PDF document.")
        if document.needs_pass:
            raise PDFPreflightError("encrypted_pdf", "Encrypted PDFs are not accepted; upload an unlocked PDF.")
        if document.page_count == 0:
            raise PDFPreflightError("empty_pdf", "The PDF contains no pages.")
        if document.page_count > max_pages:
            raise PDFPreflightError(
                "page_limit_exceeded",
                f"The PDF has {document.page_count} pages; the limit is {max_pages}.",
            )
        return document.page_count
    finally:
        document.close()


def execute_pipeline(
    pdf_path: Path,
    *,
    settings: APISettings,
    on_stage: StageCallback,
) -> PipelineArtifacts:
    artifacts = PipelineArtifacts()
    steps = (
        (StageName.INGESTION, lambda: _ingest(pdf_path, settings.max_pdf_pages)),
        (StageName.CONTEXT_COMPILATION, lambda: compile_financial_context(artifacts.document)),
        (
            StageName.TABLE_RECONSTRUCTION,
            lambda: reconstruct_tables(artifacts.document, artifacts.context),
        ),
        (
            StageName.FINANCIAL_NORMALIZATION,
            lambda: normalize_financial_tables(artifacts.reconstruction, artifacts.context),
        ),
        (
            StageName.FINANCIAL_VALIDATION,
            lambda: validate_financial_data(
                artifacts.normalization,
                config=_validation_config(settings),
            ),
        ),
    )
    targets = {
        StageName.INGESTION: "document",
        StageName.CONTEXT_COMPILATION: "context",
        StageName.TABLE_RECONSTRUCTION: "reconstruction",
        StageName.FINANCIAL_NORMALIZATION: "normalization",
        StageName.FINANCIAL_VALIDATION: "validation",
    }
    for stage, operation in steps:
        on_stage(stage, StageState.PROCESSING, None, None)
        try:
            value = operation()
        except PDFPreflightError as error:
            on_stage(stage, StageState.FAILED, error.error_code, error.public_message)
            raise PipelineExecutionError(
                StageFailure(stage, error.error_code, error.public_message), artifacts
            ) from error
        except Exception as error:
            error_code = _stage_error_code(stage)
            public_message = f"{stage.value.replace('_', ' ').capitalize()} failed."
            on_stage(stage, StageState.FAILED, error_code, public_message)
            raise PipelineExecutionError(
                StageFailure(stage, error_code, public_message), artifacts
            ) from error
        setattr(artifacts, targets[stage], value)
        on_stage(stage, StageState.COMPLETED, None, None)

    if artifacts.validation:
        for finding in artifacts.validation.findings:
            if finding.status.value not in {"passed"}:
                artifacts.warnings.append(f"{finding.rule_id}: {finding.status.value}")
    return artifacts


def _ingest(path: Path, max_pages: int) -> PDFDocument:
    preflight_pdf(path, max_pages)
    try:
        return ingest_pdf(path)
    except PDFIngestionError as error:
        raise PDFPreflightError("pdf_ingestion_failed", "PDF ingestion failed.") from error


def _validation_config(settings: APISettings) -> ValidationConfig:
    thresholds = ()
    if settings.revenue_change_threshold is not None:
        from ..financial_normalization import MetricConcept
        from ..financial_validation import CrossPeriodThreshold

        thresholds = (
            CrossPeriodThreshold(
                MetricConcept.REVENUE,
                settings.revenue_change_threshold,
            ),
        )
    return ValidationConfig(
        absolute_tolerance=settings.validation_absolute_tolerance,
        cross_period_thresholds=thresholds,
    )


def _stage_error_code(stage: StageName) -> str:
    return f"{stage.value}_failed"


def _span_response(span: TextSpan):
    from .schemas import SpanResponse

    return SpanResponse(
        raw_text=span.raw_text,
        page_number=span.page_number,
        bounding_box=span.bounding_box,
        extraction_source=span.extraction_source.value,
        confidence=span.confidence,
    )


def _cell_response(cell: ReconstructedCell) -> CellResponse:
    return CellResponse(
        cell_id=cell.cell_id,
        role=cell.role.value,
        raw_text=cell.raw_text,
        cell_boundary=cell.cell_boundary,
        row_index=cell.row_index,
        column_index=cell.column_index,
        column_span=cell.column_span,
        status=cell.status.value,
        issue=cell.issue,
        source_spans=tuple(_span_response(span) for span in cell.provenance.source_spans),
    )


def _header_response(header: ReconstructedHeaderCell) -> HeaderResponse:
    return HeaderResponse(
        header_id=header.header_id,
        header_row=header.header_row,
        raw_text=header.raw_text,
        column_start=header.column_start,
        column_span=header.column_span,
        is_merged=header.is_merged,
        status=header.status.value,
        issue=header.issue,
        source_spans=tuple(_span_response(span) for span in header.provenance.source_spans),
    )


def _table_response(table: ReconstructedTable) -> TableResponse:
    columns = tuple(
        ColumnResponse(
            column_index=column.column_index,
            left=column.left,
            right=column.right,
            center=column.center,
            header_path=tuple(_header_response(header) for header in column.header_path),
            period_label_raw=column.reporting_period.raw_label if column.reporting_period else None,
            period_status=column.period_status.value,
            geometry_status=column.geometry_status.value,
            is_auxiliary=column.is_auxiliary,
        )
        for column in table.columns
    )
    return TableResponse(
        table_id=table.table_id,
        page_number=table.page_number,
        bounding_box=table.bounding_box,
        statement_title_raw=table.statement_title.raw_title if table.statement_title else None,
        columns=columns,
        header_rows=tuple(
            tuple(_header_response(header) for header in row)
            for row in table.header_rows
        ),
        rows=tuple(
            RowResponse(
                row_id=row.row_id,
                row_index=row.row_index,
                bounding_box=row.bounding_box,
                label_cell=_cell_response(row.label_cell) if row.label_cell else None,
                cells=tuple(_cell_response(cell) for cell in row.cells),
                parent_row_id=row.parent_row_id,
                hierarchy_status=row.hierarchy_status.value,
            )
            for row in table.rows
        ),
        context_spans=tuple(_span_response(span) for span in table.context_spans),
        status=table.status.value,
        issues=table.issues,
    )


def _period_response(period):
    from .schemas import PeriodResponse

    return PeriodResponse(
        raw_label=period.raw_label,
        kind=period.kind.value,
        fiscal_year=period.fiscal_year,
        quarter=period.quarter,
        relative_period=period.relative_period.value if period.relative_period else None,
        date_iso=period.date_iso,
        status=period.status.value,
        issue=period.issue,
        evidence_spans=tuple(_span_response(span) for span in period.evidence_spans),
    )


def _observation_response(observation: NormalizedObservation) -> NormalizedObservationResponse:
    return NormalizedObservationResponse(
        table_id=observation.source_table.table_id,
        row_id=observation.source_row.row_id,
        cell_id=observation.source_cell.cell_id,
        row_label_raw=observation.metric.raw_label,
        row_label_source_spans=tuple(
            _span_response(span) for span in observation.metric.evidence_spans
        ),
        metric_candidates=tuple(item.value for item in observation.metric.candidates),
        metric_status=observation.metric.status.value,
        metric_issue=observation.metric.issue,
        period=_period_response(observation.period),
        raw_value=observation.value.parsed.raw_text,
        parsed_value=observation.value.parsed.value,
        value_kind=observation.value.parsed.kind.value,
        value_status=observation.value.status.value,
        raw_unit_declaration=observation.value.raw_unit_declaration,
        original_unit=observation.value.original_unit.value if observation.value.original_unit else None,
        normalized_value=observation.value.normalized_value,
        normalized_unit=observation.value.normalized_unit.value if observation.value.normalized_unit else None,
        unit_status=observation.value.unit_status.value,
        issue=observation.value.issue,
        source_spans=tuple(_span_response(span) for span in observation.value.evidence_spans),
        unit_source_spans=tuple(
            _span_response(span)
            for unit_fact in observation.value.unit_candidates
            for span in unit_fact.evidence.source_spans
        ),
    )


def _finding_response(finding: ValidationFinding) -> ValidationFindingResponse:
    spans = tuple(_span_response(span) for span in finding.evidence.source_spans)
    return ValidationFindingResponse(
        rule_id=finding.rule_id,
        status=finding.status.value,
        message=finding.message,
        period_label=finding.period_label,
        metric_concepts=tuple(item.value for item in finding.metric_concepts),
        observed_residual=finding.observed_residual,
        tolerance=finding.tolerance,
        evidence=ValidationEvidenceResponse(
            table_ids=tuple(table.table_id for table in finding.evidence.source_tables),
            cell_ids=tuple(cell.cell_id for cell in finding.evidence.source_cells),
            spans=spans,
        ),
    )


def result_response(
    job_id: str,
    state: JobState,
    artifacts: PipelineArtifacts,
    stage_errors: tuple[StageResponse, ...],
    failed_stage: StageName | None,
) -> ProcessingResultResponse:
    tables = artifacts.reconstruction.tables if artifacts.reconstruction else ()
    observations = (
        tuple(
            _observation_response(observation)
            for table in artifacts.normalization.tables
            for observation in table.observations
        )
        if artifacts.normalization
        else ()
    )
    findings = (
        tuple(_finding_response(finding) for finding in artifacts.validation.findings)
        if artifacts.validation
        else ()
    )
    return ProcessingResultResponse(
        job_id=job_id,
        state=state,
        partial=failed_stage is not None,
        failed_stage=failed_stage,
        tables=tuple(_table_response(table) for table in tables),
        observations=observations,
        validation_findings=findings,
        stage_errors=stage_errors,
        warnings=tuple(artifacts.warnings),
        document_report=(
            build_document_report(artifacts.normalization, artifacts.validation)
            if artifacts.normalization
            else None
        ),
    )


def validation_response(result: ProcessingResultResponse) -> ValidationResultResponse:
    return ValidationResultResponse(
        job_id=result.job_id,
        state=result.state,
        findings=result.validation_findings,
        stage_errors=result.stage_errors,
    )