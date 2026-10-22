from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Sequence

from .context_compiler import (
    FinancialDocumentContext,
    FinancialUnit,
    FinancialUnitFact,
    ReportingPeriodFact,
)
from .pdf_ingestion import TextSpan
from .table_reconstruction import (
    AssignmentStatus,
    CellRole,
    ReconstructedCell,
    ReconstructedColumn,
    ReconstructedRow,
    ReconstructedTable,
    TableReconstructionResult,
)


class NormalizationStatus(str, Enum):
    RESOLVED = "resolved"
    INFERRED = "inferred"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"
    UNSUPPORTED = "unsupported"


class ValueKind(str, Enum):
    AMOUNT = "amount"
    PERCENTAGE = "percentage"
    ZERO = "zero"
    BLANK = "blank"
    DASH = "dash"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"


class PeriodKind(str, Enum):
    RELATIVE_FISCAL_YEAR = "relative_fiscal_year"
    FISCAL_YEAR = "fiscal_year"
    QUARTER = "quarter"
    DATE = "date"
    UNKNOWN = "unknown"


class RelativePeriod(str, Enum):
    PRIOR = "prior"
    CURRENT = "current"


class MetricConcept(str, Enum):
    REVENUE = "revenue"
    OPERATING_INCOME = "operating_income"
    ORDINARY_INCOME = "ordinary_income"
    NET_INCOME = "net_income"
    TOTAL_ASSETS = "total_assets"
    LIABILITIES = "liabilities"
    EQUITY = "equity"
    OPERATING_CASH_FLOW = "operating_cash_flow"


@dataclass(frozen=True)
class ParsedFinancialValue:
    raw_text: str | None
    normalized_text: str | None
    value: Decimal | None
    kind: ValueKind
    status: NormalizationStatus
    issue: str | None = None


@dataclass(frozen=True)
class SemanticMapping:
    raw_label: str | None
    normalized_label: str | None
    candidates: tuple[MetricConcept, ...]
    status: NormalizationStatus
    evidence_spans: tuple[TextSpan, ...]
    issue: str | None = None


@dataclass(frozen=True)
class ReportingPeriod:
    raw_label: str
    kind: PeriodKind
    fiscal_year: int | None
    quarter: int | None
    relative_period: RelativePeriod | None
    date_iso: str | None
    status: NormalizationStatus
    evidence_spans: tuple[TextSpan, ...]
    issue: str | None = None


@dataclass(frozen=True)
class NormalizedFinancialValue:
    source_cell: ReconstructedCell
    parsed: ParsedFinancialValue
    raw_unit_declaration: str | None
    original_unit: FinancialUnit | None
    normalized_value: Decimal | None
    normalized_unit: FinancialUnit | None
    unit_status: NormalizationStatus
    status: NormalizationStatus
    evidence_spans: tuple[TextSpan, ...]
    issue: str | None = None
    unit_candidates: tuple[FinancialUnitFact, ...] = ()


@dataclass(frozen=True)
class NormalizedObservation:
    source_table: ReconstructedTable
    source_row: ReconstructedRow
    source_column: ReconstructedColumn | None
    source_cell: ReconstructedCell
    metric: SemanticMapping
    period: ReportingPeriod
    value: NormalizedFinancialValue


@dataclass(frozen=True)
class NormalizedTable:
    source_table: ReconstructedTable
    row_metrics: tuple[SemanticMapping, ...]
    column_periods: tuple[ReportingPeriod, ...]
    observations: tuple[NormalizedObservation, ...]


@dataclass(frozen=True)
class FinancialNormalizationResult:
    tables: tuple[NormalizedTable, ...]


def _normalize_label(label: str) -> str:
    return "".join(unicodedata.normalize("NFKC", label).split())


class MetricRegistry:
    """Immutable exact-label mapping registry; aliases can be extended explicitly."""

    def __init__(self, aliases: Mapping[str, MetricConcept | Sequence[MetricConcept]]) -> None:
        normalized: dict[str, tuple[MetricConcept, ...]] = {}
        for label, concepts in aliases.items():
            key = _normalize_label(label)
            values = (concepts,) if isinstance(concepts, MetricConcept) else tuple(concepts)
            if not key or not values or any(not isinstance(value, MetricConcept) for value in values):
                raise ValueError("Metric aliases require non-empty labels and MetricConcept values.")
            normalized[key] = tuple(dict.fromkeys(values))
        self._aliases = MappingProxyType(normalized)

    @property
    def aliases(self) -> Mapping[str, tuple[MetricConcept, ...]]:
        return self._aliases

    def resolve(self, raw_label: str | None, evidence_spans: tuple[TextSpan, ...]) -> SemanticMapping:
        if raw_label is None or not raw_label.strip():
            return SemanticMapping(
                raw_label,
                None,
                (),
                NormalizationStatus.UNRESOLVED,
                evidence_spans,
                "No source row label is available.",
            )
        normalized = _normalize_label(raw_label)
        candidates = self._aliases.get(normalized, ())
        if len(candidates) == 1:
            return SemanticMapping(
                raw_label,
                normalized,
                candidates,
                NormalizationStatus.INFERRED,
                evidence_spans,
            )
        if len(candidates) > 1:
            return SemanticMapping(
                raw_label,
                normalized,
                candidates,
                NormalizationStatus.AMBIGUOUS,
                evidence_spans,
                "The exact label has multiple registered semantic concepts.",
            )
        return SemanticMapping(
            raw_label,
            normalized,
            (),
            NormalizationStatus.UNRESOLVED,
            evidence_spans,
            "No exact semantic mapping is registered for this label.",
        )


DEFAULT_METRIC_REGISTRY = MetricRegistry(
    {
        "売上高": MetricConcept.REVENUE,
        "売上収益": MetricConcept.REVENUE,
        "営業収益": MetricConcept.REVENUE,
        "営業利益": MetricConcept.OPERATING_INCOME,
        "経常利益": MetricConcept.ORDINARY_INCOME,
        "当期純利益": MetricConcept.NET_INCOME,
        "親会社株主に帰属する当期純利益": MetricConcept.NET_INCOME,
        "親会社の所有者に帰属する当期利益": MetricConcept.NET_INCOME,
        "資産合計": MetricConcept.TOTAL_ASSETS,
        "資産の部合計": MetricConcept.TOTAL_ASSETS,
        "負債合計": MetricConcept.LIABILITIES,
        "負債の部合計": MetricConcept.LIABILITIES,
        "純資産合計": MetricConcept.EQUITY,
        "純資産の部合計": MetricConcept.EQUITY,
        "株主資本合計": MetricConcept.EQUITY,
        "営業活動によるキャッシュ・フロー": MetricConcept.OPERATING_CASH_FLOW,
        "営業活動によるキャッシュフロー": MetricConcept.OPERATING_CASH_FLOW,
    }
)

_NUMBER_BODY = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_NUMBER_RE = re.compile(rf"^(?P<number>{_NUMBER_BODY})$")
_DASHES = frozenset({"-", "‐", "‑", "‒", "–", "—", "―", "−", "ー"})
_UNAVAILABLE = frozenset(
    {"N/A", "NA", "N.A.", "該当なし", "該当事項なし", "算定不能", "算出不能"}
)
_FISCAL_YEAR_RE = re.compile(
    r"(?:(?P<era>令和|平成|昭和|大正|明治)(?P<era_year>元|\d+)年度"
    r"|(?P<year>\d{4})(?:年度|年\d{1,2}月期))"
)
_QUARTER_RE = re.compile(r"第(?P<quarter>[1-4])四半期")
_UNIT_MULTIPLIERS = {
    FinancialUnit.YEN: Decimal(1),
    FinancialUnit.THOUSAND_YEN: Decimal(1_000),
    FinancialUnit.MILLION_YEN: Decimal(1_000_000),
    FinancialUnit.HUNDRED_MILLION_YEN: Decimal(100_000_000),
}


def parse_financial_value(raw_text: str | None) -> ParsedFinancialValue:
    """Parse supported Japanese financial notation without coercing blank/dash."""
    if raw_text is None:
        return ParsedFinancialValue(None, None, None, ValueKind.BLANK, NormalizationStatus.UNRESOLVED)
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string or None.")

    normalized = unicodedata.normalize("NFKC", raw_text).strip()
    if not normalized:
        return ParsedFinancialValue(raw_text, None, None, ValueKind.BLANK, NormalizationStatus.UNRESOLVED)
    if normalized.upper() in _UNAVAILABLE:
        return ParsedFinancialValue(raw_text, normalized, None, ValueKind.UNAVAILABLE, NormalizationStatus.RESOLVED)
    if all(character in _DASHES for character in normalized):
        return ParsedFinancialValue(raw_text, normalized, None, ValueKind.DASH, NormalizationStatus.RESOLVED)

    negative_parentheses = normalized.startswith("(") and normalized.endswith(")")
    if normalized.startswith("(") != normalized.endswith(")"):
        return _unsupported(raw_text, "Unbalanced parentheses in financial value.")
    if negative_parentheses:
        normalized = normalized[1:-1].strip()

    is_percentage = normalized.endswith("%")
    if is_percentage:
        normalized = normalized[:-1].strip()

    sign = 1
    if normalized[:1] in {"△", "▲", "-", "−"}:
        sign = -1
        normalized = normalized[1:]
    elif normalized.startswith("+"):
        normalized = normalized[1:]
    if negative_parentheses:
        if sign < 0:
            return _unsupported(raw_text, "Parentheses and a negative prefix conflict.")
        sign = -1

    match = _NUMBER_RE.fullmatch(normalized)
    if match is None:
        return _unsupported(raw_text, "Unsupported or ambiguous financial value notation.")
    try:
        value = Decimal(match.group("number").replace(",", ""))
        if sign < 0:
            value = value.copy_negate()
    except (InvalidOperation, ValueError):
        return _unsupported(raw_text, "Financial value could not be parsed as Decimal.")
    if is_percentage:
        kind = ValueKind.PERCENTAGE
    elif value == 0:
        kind = ValueKind.ZERO
    else:
        kind = ValueKind.AMOUNT
    return ParsedFinancialValue(
        raw_text,
        str(value),
        value,
        kind,
        NormalizationStatus.RESOLVED,
    )


def normalize_financial_tables(
    reconstruction: TableReconstructionResult,
    context: FinancialDocumentContext,
    *,
    metric_registry: MetricRegistry = DEFAULT_METRIC_REGISTRY,
) -> FinancialNormalizationResult:
    """Normalize reconstructed financial candidates using only linked evidence."""
    if not isinstance(reconstruction, TableReconstructionResult):
        raise TypeError("reconstruction must be a TableReconstructionResult.")
    if not isinstance(context, FinancialDocumentContext):
        raise TypeError("context must be a FinancialDocumentContext.")
    if not isinstance(metric_registry, MetricRegistry):
        raise TypeError("metric_registry must be a MetricRegistry.")

    tables = []
    for table in reconstruction.tables:
        row_metrics = tuple(_map_row_metric(row, metric_registry) for row in table.rows)
        periods = tuple(
            _normalize_period(column, context) for column in table.columns
        )
        unit_candidates, unit_status, unit_issue = _table_unit_candidates(table)
        observations = []
        for row, metric in zip(table.rows, row_metrics):
            for cell in row.cells:
                if cell.role is not CellRole.VALUE:
                    continue
                column_index = cell.column_index
                if column_index is None or column_index >= len(table.columns):
                    period = _unresolved_period("", cell.provenance.source_spans, "Cell has no unique column assignment.")
                else:
                    period = periods[column_index]
                value = _normalize_cell_value(
                    cell,
                    unit_candidates,
                    unit_status,
                    unit_issue,
                )
                observations.append(
                    NormalizedObservation(
                        source_table=table,
                        source_row=row,
                        source_column=(
                            table.columns[column_index]
                            if column_index is not None and column_index < len(table.columns)
                            else None
                        ),
                        source_cell=cell,
                        metric=metric,
                        period=period,
                        value=value,
                    )
                )
        tables.append(
            NormalizedTable(
                source_table=table,
                row_metrics=row_metrics,
                column_periods=periods,
                observations=tuple(observations),
            )
        )
    return FinancialNormalizationResult(tuple(tables))


def _unsupported(raw_text: str, issue: str) -> ParsedFinancialValue:
    return ParsedFinancialValue(
        raw_text,
        None,
        None,
        ValueKind.UNSUPPORTED,
        NormalizationStatus.UNSUPPORTED,
        issue,
    )


def _map_row_metric(row: ReconstructedRow, registry: MetricRegistry) -> SemanticMapping:
    label_cell = row.label_cell
    return registry.resolve(
        label_cell.raw_text if label_cell else None,
        label_cell.provenance.source_spans if label_cell else (),
    )


def _normalize_period(
    column: ReconstructedColumn,
    context: FinancialDocumentContext,
) -> ReportingPeriod:
    header_spans = tuple(
        dict.fromkeys(
            span
            for header in column.header_path
            for span in header.provenance.source_spans
        )
    )
    header_text = " ".join(header.raw_text for header in column.header_path)
    if column.period_status is AssignmentStatus.AMBIGUOUS:
        return _unresolved_period(
            header_text,
            header_spans,
            "Multiple reporting-period candidates are associated with this column.",
            NormalizationStatus.AMBIGUOUS,
        )
    if column.period_status is AssignmentStatus.RESOLVED and column.reporting_period:
        fact = column.reporting_period
        date_candidates = [
            date_fact
            for date_fact in context.reporting_dates
            if date_fact.associated_period_label == fact.raw_label
            and _shares_spans(date_fact.evidence.source_spans, fact.evidence.source_spans)
        ]
        dates = {item.iso_date for item in date_candidates if item.iso_date}
        if len(dates) > 1:
            return _unresolved_period(
                fact.raw_label,
                fact.evidence.source_spans,
                "Conflicting explicit dates support this reporting period.",
                NormalizationStatus.AMBIGUOUS,
            )
        normalized = _period_label(
            fact.raw_label,
            fact.evidence.source_spans,
            next(iter(dates), None),
        )
        return normalized

    if not header_text:
        return _unresolved_period("", header_spans, "No source header supports a reporting period.")

    fiscal_match = _FISCAL_YEAR_RE.search(unicodedata.normalize("NFKC", header_text))
    quarter_match = _QUARTER_RE.search(unicodedata.normalize("NFKC", header_text))
    if quarter_match:
        return ReportingPeriod(
            raw_label=header_text,
            kind=PeriodKind.QUARTER,
            fiscal_year=_fiscal_year_number(fiscal_match) if fiscal_match else None,
            quarter=int(quarter_match.group("quarter")),
            relative_period=None,
            date_iso=None,
            status=NormalizationStatus.RESOLVED,
            evidence_spans=header_spans,
        )
    if fiscal_match:
        return ReportingPeriod(
            raw_label=header_text,
            kind=PeriodKind.FISCAL_YEAR,
            fiscal_year=_fiscal_year_number(fiscal_match),
            quarter=None,
            relative_period=None,
            date_iso=None,
            status=NormalizationStatus.RESOLVED,
            evidence_spans=header_spans,
        )

    dates = [
        date_fact
        for date_fact in context.reporting_dates
        if date_fact.iso_date and _shares_spans(date_fact.evidence.source_spans, header_spans)
    ]
    unique_dates = {item.iso_date for item in dates}
    if len(unique_dates) == 1:
        return ReportingPeriod(
            raw_label=header_text,
            kind=PeriodKind.DATE,
            fiscal_year=None,
            quarter=None,
            relative_period=None,
            date_iso=next(iter(unique_dates)),
            status=NormalizationStatus.RESOLVED,
            evidence_spans=header_spans,
        )
    if len(unique_dates) > 1:
        return _unresolved_period(
            header_text,
            header_spans,
            "Conflicting explicit dates occur in the header evidence.",
            NormalizationStatus.AMBIGUOUS,
        )
    return _unresolved_period(
        header_text,
        header_spans,
        "Header text does not match a supported explicit period representation.",
    )


def _period_label(
    raw_label: str,
    evidence_spans: tuple[TextSpan, ...],
    iso_date: str | None,
) -> ReportingPeriod:
    relative = {
        "前連結会計年度": RelativePeriod.PRIOR,
        "前事業年度": RelativePeriod.PRIOR,
        "当連結会計年度": RelativePeriod.CURRENT,
        "当事業年度": RelativePeriod.CURRENT,
    }.get(raw_label)
    fiscal_match = _FISCAL_YEAR_RE.search(unicodedata.normalize("NFKC", raw_label))
    if relative is not None:
        return ReportingPeriod(
            raw_label,
            PeriodKind.RELATIVE_FISCAL_YEAR,
            None,
            None,
            relative,
            iso_date,
            NormalizationStatus.RESOLVED,
            evidence_spans,
        )
    if fiscal_match:
        return ReportingPeriod(
            raw_label,
            PeriodKind.FISCAL_YEAR,
            _fiscal_year_number(fiscal_match),
            None,
            None,
            iso_date,
            NormalizationStatus.RESOLVED,
            evidence_spans,
        )
    return _unresolved_period(
        raw_label,
        evidence_spans,
        "Reporting-period label is not supported by the normalization rules.",
    )


def _fiscal_year_number(match: re.Match[str]) -> int:
    if match.group("era"):
        base_year = {
            "明治": 1867,
            "大正": 1911,
            "昭和": 1925,
            "平成": 1988,
            "令和": 2018,
        }[match.group("era")]
        era_year = 1 if match.group("era_year") == "元" else int(match.group("era_year"))
        return base_year + era_year
    return int(match.group("year"))


def _unresolved_period(
    raw_label: str,
    spans: tuple[TextSpan, ...],
    issue: str,
    status: NormalizationStatus = NormalizationStatus.UNRESOLVED,
) -> ReportingPeriod:
    return ReportingPeriod(
        raw_label,
        PeriodKind.UNKNOWN,
        None,
        None,
        None,
        None,
        status,
        spans,
        issue,
    )


def _table_unit_candidates(
    table: ReconstructedTable,
) -> tuple[tuple[FinancialUnitFact, ...], NormalizationStatus, str | None]:
    context_span_ids = {id(span) for span in table.context_spans}
    supported = tuple(
        fact
        for fact in table.table_units
        if fact.scope.value == "statement_section"
        and fact.associated_section_title is not None
        and table.statement_title is not None
        and fact.associated_section_title == table.statement_title.raw_title
        and fact.associated_section_evidence is not None
        and _shares_spans(fact.evidence.source_spans, table.context_spans)
        and _shares_spans(fact.associated_section_evidence.source_spans, table.context_spans)
        and all(id(span) in context_span_ids for span in fact.evidence.source_spans)
    )
    if not supported:
        return (), NormalizationStatus.UNRESOLVED, "No explicitly declared unit is unambiguously linked to this statement table."
    distinct = {fact.normalized_unit for fact in supported}
    if len(distinct) > 1:
        return supported, NormalizationStatus.AMBIGUOUS, "Conflicting explicit unit declarations occur in this table context."
    if len(supported) > 1:
        return supported, NormalizationStatus.AMBIGUOUS, "Multiple unit declarations apply; their table scope is unresolved."
    return supported, NormalizationStatus.RESOLVED, None


def _normalize_cell_value(
    cell: ReconstructedCell,
    unit_candidates: tuple[FinancialUnitFact, ...],
    unit_status: NormalizationStatus,
    unit_issue: str | None,
) -> NormalizedFinancialValue:
    parsed = parse_financial_value(cell.raw_text)
    evidence = cell.provenance.source_spans
    if parsed.kind in {ValueKind.BLANK, ValueKind.DASH, ValueKind.UNAVAILABLE}:
        return NormalizedFinancialValue(
            cell,
            parsed,
            unit_candidates[0].raw_declaration if unit_status is NormalizationStatus.RESOLVED else None,
            unit_candidates[0].normalized_unit if unit_status is NormalizationStatus.RESOLVED else None,
            None,
            None,
            unit_status,
            parsed.status,
            evidence,
            parsed.issue or unit_issue,
            unit_candidates,
        )
    if parsed.kind is ValueKind.UNSUPPORTED:
        return NormalizedFinancialValue(
            cell, parsed, None, None, None, None, unit_status,
            NormalizationStatus.UNSUPPORTED, evidence, parsed.issue, unit_candidates,
        )
    if cell.status not in {AssignmentStatus.RESOLVED, AssignmentStatus.INFERRED}:
        return NormalizedFinancialValue(
            cell, parsed, None, None, None, None, unit_status,
            NormalizationStatus.AMBIGUOUS if cell.status is AssignmentStatus.AMBIGUOUS else NormalizationStatus.UNRESOLVED,
            evidence, cell.issue or "Reconstructed cell assignment is not resolved.",
            unit_candidates,
        )

    if parsed.kind is ValueKind.PERCENTAGE:
        if unit_status is NormalizationStatus.RESOLVED and unit_candidates[0].normalized_unit is not FinancialUnit.PERCENT:
            return NormalizedFinancialValue(
                cell, parsed, unit_candidates[0].raw_declaration, unit_candidates[0].normalized_unit,
                None, None, NormalizationStatus.AMBIGUOUS, NormalizationStatus.AMBIGUOUS,
                evidence, "Percent value conflicts with the explicitly declared table unit.",
                unit_candidates,
            )
        return NormalizedFinancialValue(
            cell,
            parsed,
            unit_candidates[0].raw_declaration if unit_status is NormalizationStatus.RESOLVED else None,
            FinancialUnit.PERCENT, parsed.value, FinancialUnit.PERCENT,
            NormalizationStatus.RESOLVED, NormalizationStatus.RESOLVED, evidence,
            unit_candidates=unit_candidates,
        )

    if unit_status is not NormalizationStatus.RESOLVED:
        return NormalizedFinancialValue(
            cell,
            parsed,
            None,
            None,
            None,
            None,
            unit_status,
            unit_status,
            evidence,
            unit_issue or "A supported explicit table unit is required for base-unit conversion.",
            unit_candidates,
        )
    unit_fact = unit_candidates[0]
    if unit_fact.normalized_unit is FinancialUnit.PERCENT:
        return NormalizedFinancialValue(
            cell,
            parsed,
            unit_fact.raw_declaration,
            FinancialUnit.PERCENT,
            parsed.value,
            FinancialUnit.PERCENT,
            NormalizationStatus.RESOLVED,
            NormalizationStatus.RESOLVED,
            evidence,
            unit_candidates=unit_candidates,
        )
    multiplier = _UNIT_MULTIPLIERS[unit_fact.normalized_unit]
    with localcontext() as decimal_context:
        decimal_context.prec = max(
            decimal_context.prec,
            len(parsed.value.as_tuple().digits) + len(multiplier.as_tuple().digits) + 1,
        )
        normalized = parsed.value * multiplier
    return NormalizedFinancialValue(
        cell,
        parsed,
        unit_fact.raw_declaration,
        unit_fact.normalized_unit,
        normalized,
        FinancialUnit.YEN,
        NormalizationStatus.RESOLVED,
        NormalizationStatus.RESOLVED,
        evidence,
        unit_candidates=unit_candidates,
    )


def _shares_spans(left: Sequence[TextSpan], right: Sequence[TextSpan]) -> bool:
    right_ids = {id(span) for span in right}
    return any(id(span) in right_ids for span in left)

