from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .pipeline import (
    PipelineArtifacts,
    PipelineExecutionError,
    execute_pipeline,
    result_response,
)
from .schemas import (
    JobState,
    JobStatusResponse,
    ProcessingResultResponse,
    StageName,
    StageResponse,
    StageState,
    ValidationResultResponse,
)
from .settings import APISettings


class JobCapacityError(RuntimeError):
    """Raised when the bounded job store or worker queue is full."""


class InvalidJobTransition(RuntimeError):
    """Raised when code attempts a lifecycle transition that is not allowed."""


_ALLOWED_TRANSITIONS = {
    JobState.QUEUED: {JobState.PROCESSING, JobState.FAILED},
    JobState.PROCESSING: {
        JobState.COMPLETED,
        JobState.COMPLETED_WITH_WARNINGS,
        JobState.FAILED,
    },
    JobState.COMPLETED: set(),
    JobState.COMPLETED_WITH_WARNINGS: set(),
    JobState.FAILED: set(),
}
_TERMINAL_STATES = {
    JobState.COMPLETED,
    JobState.COMPLETED_WITH_WARNINGS,
    JobState.FAILED,
}
_STAGE_ORDER = (
    StageName.INGESTION,
    StageName.CONTEXT_COMPILATION,
    StageName.TABLE_RECONSTRUCTION,
    StageName.FINANCIAL_NORMALIZATION,
    StageName.FINANCIAL_VALIDATION,
)


@dataclass
class JobRecord:
    job_id: str
    directory: Path
    source_path: Path
    state: JobState
    created_at: datetime
    updated_at: datetime
    stages: dict[StageName, StageResponse]
    artifacts: PipelineArtifacts = field(default_factory=PipelineArtifacts)
    failed_stage: StageName | None = None
    error_code: str | None = None
    error_message: str | None = None
    result_path: Path | None = None
    future: Future | None = None


PipelineRunner = Callable[..., PipelineArtifacts]


class JobManager:
    """Bounded, in-process job state with filesystem-backed retained artifacts."""

    def __init__(
        self,
        settings: APISettings,
        *,
        runner: PipelineRunner = execute_pipeline,
    ) -> None:
        self.settings = settings
        self._runner = runner
        self._lock = threading.RLock()
        self._capacity = settings.max_workers + settings.max_queued_jobs
        self._reserved_slots = 0
        self._executor = ThreadPoolExecutor(
            max_workers=settings.max_workers,
            thread_name_prefix="toresu-api",
        )
        self._jobs: dict[str, JobRecord] = {}
        self._closed = False
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir = self.settings.data_dir / "jobs"
        self.staging_dir = self.settings.data_dir / "staging"
        self.jobs_dir.mkdir(mode=0o700, exist_ok=True)
        self.staging_dir.mkdir(mode=0o700, exist_ok=True)
        self._cleanup_stale_artifacts()

    @property
    def accepting_jobs(self) -> bool:
        with self._lock:
            return not self._closed

    @property
    def available_job_slots(self) -> int:
        with self._lock:
            return self._capacity - self._reserved_slots

    def submit(self, source_path: Path) -> JobRecord:
        with self._lock:
            if self._closed:
                raise JobCapacityError("The API is shutting down and is not accepting jobs.")
            if self._reserved_slots >= self._capacity:
                raise JobCapacityError("The API job queue is full; retry later.")
            self._prune_terminal_jobs_locked()
            if len(self._jobs) >= self.settings.max_retained_jobs:
                raise JobCapacityError("The retained job limit is full; retry after cleanup.")
            self._reserved_slots += 1
            job_directory: Path | None = None
            job_id: str | None = None
            try:
                job_id = str(uuid.uuid4())
                job_directory = self.jobs_dir / job_id
                job_directory.mkdir(mode=0o700)
                stored_source = job_directory / "source.pdf"
                os.replace(source_path, stored_source)
                now = _now()
                record = JobRecord(
                    job_id=job_id,
                    directory=job_directory,
                    source_path=stored_source,
                    state=JobState.QUEUED,
                    created_at=now,
                    updated_at=now,
                    stages={
                        stage: StageResponse(name=stage, state=StageState.QUEUED)
                        for stage in _STAGE_ORDER
                    },
                )
                self._jobs[job_id] = record
                record.future = self._executor.submit(self._run_job, record)
            except Exception:
                if job_id is not None:
                    self._jobs.pop(job_id, None)
                if job_directory is not None:
                    shutil.rmtree(job_directory, ignore_errors=True)
                self._reserved_slots -= 1
                raise
            return record

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def status(self, record: JobRecord) -> JobStatusResponse:
        with self._lock:
            return JobStatusResponse(
                job_id=record.job_id,
                state=record.state,
                created_at=record.created_at,
                updated_at=record.updated_at,
                stages=tuple(record.stages[stage] for stage in _STAGE_ORDER),
                error_code=record.error_code,
                error_message=record.error_message,
            )

    def result(self, record: JobRecord) -> ProcessingResultResponse | None:
        with self._lock:
            if record.state not in _TERMINAL_STATES:
                return None
            stage_errors = tuple(
                record.stages[stage]
                for stage in _STAGE_ORDER
                if record.stages[stage].state is StageState.FAILED
            )
            response = result_response(
                record.job_id,
                record.state,
                record.artifacts,
                stage_errors,
                record.failed_stage,
            )
            self._write_result_locked(record, response)
            return response

    def validation(self, record: JobRecord) -> ValidationResultResponse | None:
        response = self.result(record)
        if response is None:
            return None
        return ValidationResultResponse(
            job_id=record.job_id,
            state=record.state,
            findings=response.validation_findings,
            stage_errors=response.stage_errors,
        )

    def close(self, wait: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def _run_job(self, record: JobRecord) -> None:
        try:
            self._transition_job(record, JobState.PROCESSING)

            def update_stage(
                stage: StageName,
                state: StageState,
                error_code: str | None,
                error_message: str | None,
            ) -> None:
                self._transition_stage(record, stage, state, error_code, error_message)

            artifacts = self._runner(
                record.source_path,
                settings=self.settings,
                on_stage=update_stage,
            )
            record.artifacts = artifacts
            has_warnings = bool(artifacts.warnings)
            self._transition_job(
                record,
                JobState.COMPLETED_WITH_WARNINGS if has_warnings else JobState.COMPLETED,
            )
        except PipelineExecutionError as error:
            record.artifacts = error.artifacts
            record.failed_stage = error.failure.stage
            record.error_code = error.failure.error_code
            record.error_message = error.failure.public_message
            self._skip_remaining_stages(record, error.failure.stage)
            self._transition_job(record, JobState.FAILED)
        except Exception:
            record.error_code = "internal_pipeline_error"
            record.error_message = "The processing pipeline failed unexpectedly."
            active_stage = next(
                (
                    stage
                    for stage, response in record.stages.items()
                    if response.state is StageState.PROCESSING
                ),
                None,
            )
            if active_stage is not None:
                self._transition_stage(
                    record,
                    active_stage,
                    StageState.FAILED,
                    record.error_code,
                    record.error_message,
                )
                record.failed_stage = active_stage
            self._skip_remaining_stages(record, active_stage)
            self._transition_job(record, JobState.FAILED)
        finally:
            with self._lock:
                try:
                    record.updated_at = _now()
                    self._write_result_locked(
                        record,
                        result_response(
                            record.job_id,
                            record.state,
                            record.artifacts,
                            tuple(
                                record.stages[stage]
                                for stage in _STAGE_ORDER
                                if record.stages[stage].state is StageState.FAILED
                            ),
                            record.failed_stage,
                        ),
                    ),
                finally:
                    self._reserved_slots -= 1

    def _transition_job(self, record: JobRecord, state: JobState) -> None:
        with self._lock:
            if state not in _ALLOWED_TRANSITIONS[record.state]:
                raise RuntimeError(
                    f"Invalid job transition: {record.state.value} -> {state.value}."
                )
            record.state = state
            record.updated_at = _now()

    def _transition_stage(
        self,
        record: JobRecord,
        stage: StageName,
        state: StageState,
        error_code: str | None,
        error_message: str | None,
    ) -> None:
        with self._lock:
            current = record.stages[stage]
            now = _now()
            allowed = {
                StageState.QUEUED: {StageState.PROCESSING, StageState.SKIPPED},
                StageState.PROCESSING: {StageState.COMPLETED, StageState.FAILED},
                StageState.COMPLETED: set(),
                StageState.FAILED: set(),
                StageState.SKIPPED: set(),
            }
            if state not in allowed[current.state]:
                raise RuntimeError(
                    f"Invalid stage transition: {current.state.value} -> {state.value}."
                )
            record.stages[stage] = StageResponse(
                name=stage,
                state=state,
                started_at=now if state is StageState.PROCESSING else current.started_at,
                finished_at=now if state in {StageState.COMPLETED, StageState.FAILED, StageState.SKIPPED} else None,
                error_code=error_code,
                error_message=error_message,
            )
            record.updated_at = now

    def _skip_remaining_stages(
        self,
        record: JobRecord,
        failed_stage: StageName | None,
    ) -> None:
        start_skipping = failed_stage is None
        for stage in _STAGE_ORDER:
            if stage is failed_stage:
                start_skipping = True
                continue
            if not start_skipping:
                continue
            response = record.stages[stage]
            if response.state is StageState.QUEUED:
                self._transition_stage(
                    record,
                    stage,
                    StageState.SKIPPED,
                    None,
                    "Skipped because an earlier pipeline stage failed.",
                )

    def _prune_terminal_jobs_locked(self) -> None:
        terminal = [job for job in self._jobs.values() if job.state in _TERMINAL_STATES]
        terminal.sort(key=lambda job: job.updated_at)
        while len(self._jobs) >= self.settings.max_retained_jobs and terminal:
            old = terminal.pop(0)
            self._jobs.pop(old.job_id, None)
            shutil.rmtree(old.directory, ignore_errors=True)

    def _cleanup_stale_artifacts(self) -> None:
        for directory in (self.jobs_dir, self.staging_dir):
            for child in directory.iterdir():
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)

    def _write_result_locked(self, record: JobRecord, response: ProcessingResultResponse) -> None:
        temp_path = record.directory / "result.json.tmp"
        result_path = record.directory / "result.json"
        try:
            encoded = response.model_dump_json(indent=2)
            temp_path.write_text(encoded, encoding="utf-8")
            os.replace(temp_path, result_path)
            record.result_path = result_path
        except OSError:
            temp_path.unlink(missing_ok=True)


def _now() -> datetime:
    return datetime.now(timezone.utc)