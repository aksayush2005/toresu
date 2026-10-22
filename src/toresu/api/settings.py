from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import os
from pathlib import Path


@dataclass(frozen=True)
class APISettings:
    data_dir: Path = Path(".toresu-data")
    max_upload_bytes: int = 50 * 1024 * 1024
    max_pdf_pages: int = 500
    max_workers: int = 2
    max_queued_jobs: int = 8
    max_retained_jobs: int = 100
    validation_absolute_tolerance: Decimal = Decimal(0)
    revenue_change_threshold: Decimal | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "max_upload_bytes",
            "max_pdf_pages",
            "max_workers",
            "max_retained_jobs",
        ):
            if getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} must be at least 1.")
        if self.max_queued_jobs < 0:
            raise ValueError("max_queued_jobs must be non-negative.")
        if self.validation_absolute_tolerance < 0:
            raise ValueError("validation_absolute_tolerance must be non-negative.")
        if self.revenue_change_threshold is not None and self.revenue_change_threshold < 0:
            raise ValueError("revenue_change_threshold must be non-negative.")
        object.__setattr__(self, "data_dir", Path(self.data_dir).resolve())

    @classmethod
    def from_env(cls) -> "APISettings":
        return cls(
            data_dir=Path(os.getenv("TORESU_API_DATA_DIR", ".toresu-data")),
            max_upload_bytes=_env_int("TORESU_API_MAX_UPLOAD_BYTES", 50 * 1024 * 1024),
            max_pdf_pages=_env_int("TORESU_API_MAX_PDF_PAGES", 500),
            max_workers=_env_int("TORESU_API_MAX_WORKERS", 2),
            max_queued_jobs=_env_int("TORESU_API_MAX_QUEUED_JOBS", 8),
            max_retained_jobs=_env_int("TORESU_API_MAX_RETAINED_JOBS", 100),
            validation_absolute_tolerance=_env_decimal(
                "TORESU_API_VALIDATION_ABSOLUTE_TOLERANCE", Decimal(0)
            ),
            revenue_change_threshold=_env_optional_decimal(
                "TORESU_API_REVENUE_CHANGE_THRESHOLD"
            ),
        )


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer.") from error


def _env_decimal(name: str, default: Decimal) -> Decimal:
    value = os.getenv(name)
    return default if value is None else _parse_decimal(name, value)


def _env_optional_decimal(name: str) -> Decimal | None:
    value = os.getenv(name)
    return None if value is None or not value.strip() else _parse_decimal(name, value)


def _parse_decimal(name: str, value: str) -> Decimal:
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{name} must be a decimal value.") from error
    if not result.is_finite():
        raise ValueError(f"{name} must be finite.")
    return result