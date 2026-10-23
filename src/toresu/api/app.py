from __future__ import annotations

import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .jobs import JobCapacityError, JobManager
from .pipeline import PDFPreflightError, PipelineRunner, preflight_pdf
from .schemas import (
    APISettingsResponse,
    ErrorResponse,
    HealthResponse,
    JobStatusResponse,
    ProcessingResultResponse,
    ReadinessResponse,
    UploadResponse,
    ValidationResultResponse,
)
from .settings import APISettings

_CHUNK_BYTES = 1024 * 1024
_PDF_SIGNATURE = b"%PDF-"


class APIRequestError(HTTPException):
    def __init__(self, status_code: int, error_code: str, detail: str) -> None:
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code


def create_app(
    settings: APISettings | None = None,
    *,
    pipeline_runner: PipelineRunner | None = None,
) -> FastAPI:
    api_settings = settings or APISettings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        manager = JobManager(api_settings, **({"runner": pipeline_runner} if pipeline_runner else {}))
        application.state.job_manager = manager
        try:
            yield
        finally:
            manager.close()

    application = FastAPI(
        title="Toresu API",
        version="0.1.0",
        description="In-process PDF financial-document processing API.",
        lifespan=lifespan,
    )
    application.state.api_settings = api_settings

    @application.exception_handler(HTTPException)
    async def http_error_handler(_request: Request, error: HTTPException):
        detail = error.detail if isinstance(error.detail, str) else "Request failed."
        code = getattr(error, "error_code", "http_error")
        return JSONResponse(
            status_code=error.status_code,
            content=ErrorResponse(error_code=code, detail=detail).model_dump(mode="json"),
        )

    @application.exception_handler(RequestValidationError)
    async def request_validation_error_handler(_request: Request, _error: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error_code="invalid_request",
                detail="The request did not match the API schema.",
            ).model_dump(mode="json"),
        )

    @application.get("/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        return HealthResponse()

    @application.get("/ready", response_model=ReadinessResponse, tags=["health"])
    async def readiness(request: Request) -> ReadinessResponse:
        manager: JobManager = request.app.state.job_manager
        available_slots = manager.available_job_slots
        accepting = manager.accepting_jobs and available_slots > 0
        return ReadinessResponse(
            status="ready" if accepting else "not_ready",
            accepting_jobs=accepting,
            available_worker_slots=available_slots,
        )

    @application.get("/v1/config", response_model=APISettingsResponse, tags=["health"])
    async def public_configuration(request: Request) -> APISettingsResponse:
        configured: APISettings = request.app.state.api_settings
        return APISettingsResponse(
            max_upload_bytes=configured.max_upload_bytes,
            max_pdf_pages=configured.max_pdf_pages,
            max_workers=configured.max_workers,
            max_queued_jobs=configured.max_queued_jobs,
            max_retained_jobs=configured.max_retained_jobs,
        )

    @application.post(
        "/v1/documents",
        response_model=UploadResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["documents"],
    )
    async def upload_document(
        request: Request,
        file: UploadFile = File(...),
    ) -> UploadResponse:
        manager: JobManager = request.app.state.job_manager
        configured: APISettings = request.app.state.api_settings
        staging_dir = manager.staging_dir / str(uuid.uuid4())
        staging_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        staged_file = staging_dir / "upload.bin"
        total_bytes = 0
        prefix = bytearray()
        try:
            with staged_file.open("xb") as output:
                os.chmod(staged_file, 0o600)
                while chunk := await file.read(_CHUNK_BYTES):
                    total_bytes += len(chunk)
                    if total_bytes > configured.max_upload_bytes:
                        raise APIRequestError(
                            413,
                            "upload_too_large",
                            f"Upload exceeds the {configured.max_upload_bytes}-byte limit.",
                        )
                    if len(prefix) < len(_PDF_SIGNATURE):
                        prefix.extend(chunk[: len(_PDF_SIGNATURE) - len(prefix)])
                    output.write(chunk)
            if bytes(prefix) != _PDF_SIGNATURE:
                raise APIRequestError(
                    415,
                    "invalid_pdf_signature",
                    "Upload must have a valid PDF file signature.",
                )
            try:
                preflight_pdf(staged_file, configured.max_pdf_pages)
            except PDFPreflightError as error:
                http_status = 413 if error.error_code == "page_limit_exceeded" else 422
                raise APIRequestError(http_status, error.error_code, error.public_message) from error
            try:
                record = manager.submit(staged_file)
            except JobCapacityError as error:
                raise APIRequestError(503, "job_capacity_exceeded", str(error)) from error
            return UploadResponse(
                job_id=record.job_id,
                state=record.state,
                status_url=f"/v1/jobs/{record.job_id}",
                result_url=f"/v1/jobs/{record.job_id}/result",
                validation_url=f"/v1/jobs/{record.job_id}/validation",
            )
        finally:
            await file.close()
            shutil.rmtree(staging_dir, ignore_errors=True)

    @application.get(
        "/v1/jobs/{job_id}",
        response_model=JobStatusResponse,
        tags=["jobs"],
    )
    async def get_job_status(job_id: str, request: Request) -> JobStatusResponse:
        record = _get_job(request, job_id)
        return request.app.state.job_manager.status(record)

    @application.get(
        "/v1/jobs/{job_id}/result",
        response_model=ProcessingResultResponse,
        tags=["jobs"],
    )
    async def get_job_result(job_id: str, request: Request) -> ProcessingResultResponse:
        record = _get_job(request, job_id)
        result = request.app.state.job_manager.result(record)
        if result is None:
            raise HTTPException(status_code=409, detail="The job has not finished.")
        return result

    @application.get(
        "/v1/jobs/{job_id}/validation",
        response_model=ValidationResultResponse,
        tags=["jobs"],
    )
    async def get_job_validation(job_id: str, request: Request) -> ValidationResultResponse:
        record = _get_job(request, job_id)
        result = request.app.state.job_manager.validation(record)
        if result is None:
            raise HTTPException(status_code=409, detail="The job has not finished.")
        return result

    return application


def _get_job(request: Request, job_id: str):
    record = request.app.state.job_manager.get(job_id)
    if record is None:
        raise APIRequestError(404, "job_not_found", "Job ID was not found or has expired.")
    return record


app = create_app()