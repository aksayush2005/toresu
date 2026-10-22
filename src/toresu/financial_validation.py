from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from enum import Enum
from typing import Protocol, Sequence

from .context_compiler import FinancialUnit
from .financial_normalization import (
    FinancialNormalizationResult,
    MetricConcept,
    NormalizationStatus,
    NormalizedObservation,
    PeriodKind,
    RelativePeriod,
)
from .pdf_ingestion import BoundingBox, TextSpan
from .table_reconstruction import ReconstructedCell, ReconstructedTable


class FindingStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    AMBIGUOUS_MAPPING = "ambiguous_mapping"
    INCOMPATIBLE_PERIOD = "incompatible_period"
    INCOMPATIBLE_UNITS = "incompatible_units"


class EquationScope(str, Enum):
    SAME_TABLE = "same_table"
    CROSS_STATEMENT = "cross_statement"


@dataclass(frozen=True)
class EquationTerm:
    concept: MetricConcept
    coefficient: Decimal = Decimal(1)
    statement_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.coefficient, Decimal):
            raise TypeError("Equation coefficients must be Decimal values.")


@dataclass(frozen=True)
class AccountingEquation:
    """Configure `result = sum(coefficient * component)` for supported facts."""

    rule_id: str
    result: EquationTerm
    components: tuple[EquationTerm, ...]
    scope: EquationScope = EquationScope.SAME_TABLE
    description: str = "Configured financial arithmetic relationship."
    expected_unit: FinancialUnit | None = None

    def __post_init__(self) -> None:
        if not self.rule_id.strip():
            raise ValueError("AccountingEquation.rule_id must not be empty.")
        if not self.components:
            raise ValueError("AccountingEquation requires at least one component.")


@dataclass(frozen=True)
class CrossPeriodThreshold:
    concept: MetricConcept
    max_relative_change: Decimal
    statement_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.max_relative_change, Decimal):
            raise TypeError("max_relative_change must be a Decimal value.")
        if self.max_relative_change < 0:
            raise ValueError("max_relative_change must be non-negative.")


@dataclass(frozen=True)
class ValidationConfig:
    absolute_tolerance: Decimal = Decimal(0)
    equations: tuple[AccountingEquation, ...] = ()
    cross_period_thresholds: tuple[CrossPeriodThreshold, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.absolute_tolerance, Decimal):
            raise TypeError("absolute_tolerance must be a Decimal value.")
        if self.absolute_tolerance < 0:
            raise ValueError("absolute_tolerance must be non-negative.")


@dataclass(frozen=True)
class ValidationEvidence:
    observations: tuple[NormalizedObservation, ...]
    source_tables: tuple[ReconstructedTable, ...]
    source_cells: tuple[ReconstructedCell, ...]
    source_spans: tuple[TextSpan, ...]
    page_numbers: tuple[int, ...]
    bounding_boxes: tuple[tuple[int, BoundingBox], ...]


@dataclass(frozen=True)
class ValidationFinding:
    rule_id: str
    status: FindingStatus
    message: str
    evidence: ValidationEvidence
    period_label: str | None = None
    metric_concepts: tuple[MetricConcept, ...] = ()
    observed_residual: Decimal | None = None
    tolerance: Decimal | None = None


@dataclass(frozen=True)
class ValidationReport:
    findings: tuple[ValidationFinding, ...]


class ValidationRule(Protocol):
    def evaluate(
        self,
        normalized: FinancialNormalizationResult,
        config: ValidationConfig,
    ) -> Sequence[ValidationFinding]: ...


class AccountingEquationRule:
    """Evaluate supported arithmetic equations without changing observations."""

    def __init__(self, equation: AccountingEquation) -> None:
        self.equation = equation

    def evaluate(
        self,
        normalized: FinancialNormalizationResult,
        config: ValidationConfig,
    ) -> Sequence[ValidationFinding]:
        findings = []
        if self.equation.scope is EquationScope.SAME_TABLE:
            for table in normalized.tables:
                if (
                    self.equation.rule_id == BALANCE_SHEET_EQUATION.rule_id
                    and not _is_balance_sheet_evidence(table.observations)
                ):
                    continue
                findings.extend(
                    _evaluate_equation_group(
                        self.equation,
                        table.observations,
                        config.absolute_tolerance,
                        require_multiple_tables=False,
                    )
                )
        else:
            observations = tuple(
                observation
                for table in normalized.tables
                for observation in table.observations
            )
            findings.extend(
                _evaluate_equation_group(
                    self.equation,
                    observations,
                    config.absolute_tolerance,
                    require_multiple_tables=True,
                )
            )
        return tuple(findings)


class CrossPeriodAnomalyRule:
    """Report configurable relative-change thresholds as warnings, never errors."""

    def evaluate(
        self,
        normalized: FinancialNormalizationResult,
        config: ValidationConfig,
    ) -> Sequence[ValidationFinding]:
        findings = []
        for threshold in config.cross_period_thresholds:
            selected_all = [
                observation
                for table in normalized.tables
                for observation in table.observations
                if threshold.concept in observation.metric.candidates
                and _statement_matches(observation, threshold.statement_type)
            ]
            statement_groups: dict[str, list[NormalizedObservation]] = {}
            for observation in selected_all:
                title = observation.source_table.statement_title
                statement_key = title.statement_type.value if title else "unknown"
                statement_groups.setdefault(statement_key, []).append(observation)
            for selected in statement_groups.values():
                ambiguous = [
                    observation
                    for observation in selected
                    if observation.metric.status is NormalizationStatus.AMBIGUOUS
                ]
                if ambiguous:
                    findings.append(
                        _finding(
                            f"cross-period:{threshold.concept.value}",
                            FindingStatus.AMBIGUOUS_MAPPING,
                            "The metric has ambiguous semantic mappings; no change comparison was made.",
                            ambiguous,
                            metric_concepts=(threshold.concept,),
                        )
                    )
                    continue

                by_period: dict[tuple[object, ...], list[NormalizedObservation]] = {}
                unresolved = []
                for observation in selected:
                    key = _period_key(observation)
                    if key is None:
                        unresolved.append(observation)
                    else:
                        by_period.setdefault(key, []).append(observation)
                if unresolved:
                    findings.append(
                        _finding(
                            f"cross-period:{threshold.concept.value}",
                            FindingStatus.INSUFFICIENT_EVIDENCE,
                            "One or more reporting periods are unresolved; no change comparison was made.",
                            unresolved,
                            metric_concepts=(threshold.concept,),
                        )
                    )
                period_values = []
                unusable = []
                for key, observations in by_period.items():
                    if len(observations) != 1:
                        unusable.extend(observations)
                        continue
                    observation = observations[0]
                    if not _value_is_usable(observation):
                        unusable.append(observation)
                    else:
                        period_values.append((key, observation))
                if unusable:
                    findings.append(
                        _finding(
                            f"cross-period:{threshold.concept.value}",
                            FindingStatus.INSUFFICIENT_EVIDENCE,
                            "Missing, ambiguous, or non-numeric period values prevent comparison.",
                            unusable,
                            metric_concepts=(threshold.concept,),
                        )
                    )
                period_families = {key[0] for key, _ in period_values}
                if len(period_families) > 1:
                    findings.append(
                        _finding(
                            f"cross-period:{threshold.concept.value}",
                            FindingStatus.INCOMPATIBLE_PERIOD,
                            "Relative, fiscal-year, quarter, or date period forms cannot be ordered together without explicit alignment evidence.",
                            tuple(item for _, item in period_values),
                            metric_concepts=(threshold.concept,),
                        )
                    )
                    continue
                period_values.sort(key=lambda item: _period_sort_key(item[1]))
                for (_, prior), (_, current) in zip(period_values, period_values[1:]):
                    if not _same_normalized_unit(prior, current):
                        findings.append(
                            _finding(
                                f"cross-period:{threshold.concept.value}",
                                FindingStatus.INCOMPATIBLE_UNITS,
                                "The compared observations do not have compatible normalized units.",
                                (prior, current),
                                metric_concepts=(threshold.concept,),
                            )
                        )
                        continue
                    prior_value = prior.value.normalized_value
                    current_value = current.value.normalized_value
                    if prior_value == 0:
                        if current_value != 0:
                            findings.append(
                                _finding(
                                    f"cross-period:{threshold.concept.value}",
                                    FindingStatus.WARNING,
                                    "The prior value is zero, so relative change is undefined; review the movement.",
                                    (prior, current),
                                    metric_concepts=(threshold.concept,),
                                )
                            )
                        continue
                    with localcontext() as decimal_context:
                        decimal_context.prec = _calculation_precision(
                            (prior_value, current_value, threshold.max_relative_change)
                        )
                        relative_change = (current_value - prior_value) / abs(prior_value)
                    if abs(relative_change) > threshold.max_relative_change:
                        findings.append(
                            _finding(
                                f"cross-period:{threshold.concept.value}",
                                FindingStatus.WARNING,
                                (
                                    f"Relative change {relative_change} exceeds the configured "
                                    f"review threshold {threshold.max_relative_change}; this is an anomaly candidate, not a confirmed error."
                                ),
                                (prior, current),
                                metric_concepts=(threshold.concept,),
                                residual=relative_change,
                                tolerance=threshold.max_relative_change,
                            )
                        )
        return tuple(findings)


BALANCE_SHEET_EQUATION = AccountingEquation(
    rule_id="balance-sheet-assets-equal-liabilities-and-equity",
    result=EquationTerm(MetricConcept.TOTAL_ASSETS),
    components=(
        EquationTerm(MetricConcept.LIABILITIES),
        EquationTerm(MetricConcept.EQUITY),
    ),
    description="Total assets = total liabilities + equity.",
    expected_unit=FinancialUnit.YEN,
)


def validate_financial_data(
    normalized: FinancialNormalizationResult,
    *,
    config: ValidationConfig = ValidationConfig(),
    rules: Sequence[ValidationRule] = (),
) -> ValidationReport:
    """Run deterministic built-in and caller-supplied financial validation rules."""
    if not isinstance(normalized, FinancialNormalizationResult):
        raise TypeError("normalized must be a FinancialNormalizationResult.")
    if not isinstance(config, ValidationConfig):
        raise TypeError("config must be a ValidationConfig.")
    equation_rules = (AccountingEquationRule(BALANCE_SHEET_EQUATION),) + tuple(
        AccountingEquationRule(equation) for equation in config.equations
    )
    active_rules: tuple[ValidationRule, ...] = (
        *equation_rules,
        CrossPeriodAnomalyRule(),
        *rules,
    )
    findings = tuple(
        finding
        for rule in active_rules
        for finding in rule.evaluate(normalized, config)
    )
    return ValidationReport(findings)


def _evaluate_equation_group(
    equation: AccountingEquation,
    observations: Sequence[NormalizedObservation],
    tolerance: Decimal,
    *,
    require_multiple_tables: bool,
) -> list[ValidationFinding]:
    terms = (equation.result, *equation.components)
    selected_by_term = [
        [
            observation
            for observation in observations
            if term.concept in observation.metric.candidates
            and _statement_matches(observation, term.statement_type)
        ]
        for term in terms
    ]
    ambiguous = [
        observation
        for term_observations in selected_by_term
        for observation in term_observations
        if observation.metric.status is NormalizationStatus.AMBIGUOUS
    ]
    if ambiguous:
        return [
            _finding(
                equation.rule_id,
                FindingStatus.AMBIGUOUS_MAPPING,
                "A required metric maps ambiguously; the arithmetic relationship was not evaluated.",
                ambiguous,
                metric_concepts=tuple(term.concept for term in terms),
                tolerance=tolerance,
            )
        ]

    missing = [term.concept for term, values in zip(terms, selected_by_term) if not values]
    if missing:
        return [
            _finding(
                equation.rule_id,
                FindingStatus.INSUFFICIENT_EVIDENCE,
                "Required metrics are absent or not mapped: " + ", ".join(item.value for item in missing),
                observations,
                metric_concepts=tuple(term.concept for term in terms),
                tolerance=tolerance,
            )
        ]

    period_sets = [
        {_period_key(observation) for observation in values} - {None}
        for values in selected_by_term
    ]
    if any(not period_set for period_set in period_sets):
        relevant = [item for group in selected_by_term for item in group]
        return [
            _finding(
                equation.rule_id,
                FindingStatus.INSUFFICIENT_EVIDENCE,
                "A required observation has no resolved reporting period.",
                relevant,
                metric_concepts=tuple(term.concept for term in terms),
                tolerance=tolerance,
            )
        ]
    common_periods = set.intersection(*period_sets)
    if not common_periods:
        relevant = [item for group in selected_by_term for item in group]
        return [
            _finding(
                equation.rule_id,
                FindingStatus.INCOMPATIBLE_PERIOD,
                "Required metrics are available only for different reporting periods.",
                relevant,
                metric_concepts=tuple(term.concept for term in terms),
                tolerance=tolerance,
            )
        ]

    findings = []
    for period_key in sorted(common_periods, key=repr):
        chosen = []
        for term_observations in selected_by_term:
            matching = [item for item in term_observations if _period_key(item) == period_key]
            if len(matching) != 1:
                chosen.extend(matching)
                continue
            chosen.append(matching[0])
        if len(chosen) != len(terms):
            findings.append(
                _finding(
                    equation.rule_id,
                    FindingStatus.INSUFFICIENT_EVIDENCE,
                    "Multiple observations prevent a unique value assignment for this equation.",
                    chosen,
                    metric_concepts=tuple(term.concept for term in terms),
                    tolerance=tolerance,
                )
            )
            continue
        if require_multiple_tables and len({item.source_table.table_id for item in chosen}) < 2:
            findings.append(
                _finding(
                    equation.rule_id,
                    FindingStatus.INSUFFICIENT_EVIDENCE,
                    "Cross-statement rule requires evidence from at least two reconstructed tables.",
                    chosen,
                    metric_concepts=tuple(term.concept for term in terms),
                    tolerance=tolerance,
                )
            )
            continue
        ambiguous_units = [
            item
            for item in chosen
            if item.value.unit_status is NormalizationStatus.AMBIGUOUS
        ]
        if ambiguous_units:
            findings.append(
                _finding(
                    equation.rule_id,
                    FindingStatus.INCOMPATIBLE_UNITS,
                    "Unit evidence is conflicting or ambiguous; the equation was not evaluated.",
                    chosen,
                    metric_concepts=tuple(term.concept for term in terms),
                    tolerance=tolerance,
                )
            )
            continue
        unusable = [item for item in chosen if not _value_is_usable(item)]
        if unusable:
            findings.append(
                _finding(
                    equation.rule_id,
                    FindingStatus.INSUFFICIENT_EVIDENCE,
                    "A required value is missing, ambiguous, or unsupported.",
                    chosen,
                    metric_concepts=tuple(term.concept for term in terms),
                    tolerance=tolerance,
                )
            )
            continue
        units = {item.value.normalized_unit for item in chosen}
        if (
            len(units) != 1
            or None in units
            or (
                equation.expected_unit is not None
                and units != {equation.expected_unit}
            )
        ):
            findings.append(
                _finding(
                    equation.rule_id,
                    FindingStatus.INCOMPATIBLE_UNITS,
                    (
                        "Required values do not share the equation's expected resolved normalized unit."
                        if equation.expected_unit is not None
                        else "Required values do not share one resolved normalized unit."
                    ),
                    chosen,
                    metric_concepts=tuple(term.concept for term in terms),
                    tolerance=tolerance,
                )
            )
            continue
        values = [item.value.normalized_value for item in chosen]
        calculation_values = [value for value in values if value is not None]
        with localcontext() as decimal_context:
            decimal_context.prec = _calculation_precision(calculation_values)
            rhs = sum(
                (term.coefficient * value for term, value in zip(equation.components, calculation_values[1:])),
                Decimal(0),
            )
            residual = calculation_values[0] - rhs
        passed = abs(residual) <= tolerance
        findings.append(
            _finding(
                equation.rule_id,
                FindingStatus.PASSED if passed else FindingStatus.FAILED,
                equation.description + (
                    f" Residual {residual} is within tolerance {tolerance}."
                    if passed
                    else f" Residual {residual} exceeds tolerance {tolerance}."
                ),
                chosen,
                metric_concepts=tuple(term.concept for term in terms),
                residual=residual,
                tolerance=tolerance,
            )
        )
    return findings


def _value_is_usable(observation: NormalizedObservation) -> bool:
    return (
        observation.value.status in {NormalizationStatus.RESOLVED, NormalizationStatus.INFERRED}
        and observation.value.normalized_value is not None
        and observation.value.normalized_unit is not None
        and observation.period.status is NormalizationStatus.RESOLVED
    )


def _same_normalized_unit(
    left: NormalizedObservation,
    right: NormalizedObservation,
) -> bool:
    return (
        left.value.normalized_unit is not None
        and left.value.normalized_unit == right.value.normalized_unit
        and left.value.unit_status is NormalizationStatus.RESOLVED
        and right.value.unit_status is NormalizationStatus.RESOLVED
    )


def _period_key(observation: NormalizedObservation) -> tuple[object, ...] | None:
    period = observation.period
    if period.status is not NormalizationStatus.RESOLVED:
        return None
    if period.date_iso:
        return ("date", period.date_iso)
    if period.kind is PeriodKind.QUARTER and period.fiscal_year and period.quarter:
        return ("quarter", period.fiscal_year, period.quarter)
    if period.kind is PeriodKind.FISCAL_YEAR and period.fiscal_year:
        return ("fiscal_year", period.fiscal_year)
    if period.kind is PeriodKind.RELATIVE_FISCAL_YEAR and period.relative_period:
        return ("relative", period.relative_period.value)
    return None


def _period_sort_key(observation: NormalizedObservation) -> tuple[int, ...]:
    period = observation.period
    if period.relative_period is RelativePeriod.PRIOR:
        return (-1, 0)
    if period.relative_period is RelativePeriod.CURRENT:
        return (0, 0)
    if period.fiscal_year is not None:
        return (period.fiscal_year, period.quarter or 0)
    if period.date_iso:
        parts = tuple(int(part) for part in period.date_iso.split("-"))
        return parts
    return (0, 0)


def _statement_matches(
    observation: NormalizedObservation,
    required: str | None,
) -> bool:
    if required is None:
        return True
    title = observation.source_table.statement_title
    return title is not None and title.statement_type.value == required


def _is_balance_sheet_evidence(
    observations: Sequence[NormalizedObservation],
) -> bool:
    balance_metrics = {
        MetricConcept.TOTAL_ASSETS,
        MetricConcept.LIABILITIES,
        MetricConcept.EQUITY,
    }
    return any(
        observation.source_table.statement_title is not None
        and "balance_sheet" in observation.source_table.statement_title.statement_type.value
        or bool(balance_metrics.intersection(observation.metric.candidates))
        for observation in observations
    )


def _finding(
    rule_id: str,
    status: FindingStatus,
    message: str,
    observations: Sequence[NormalizedObservation],
    *,
    metric_concepts: tuple[MetricConcept, ...] = (),
    residual: Decimal | None = None,
    tolerance: Decimal | None = None,
) -> ValidationFinding:
    unique_observations = tuple(_unique_by_identity(observations))
    tables = tuple(_unique_by_identity(item.source_table for item in unique_observations))
    cells = tuple(
        _unique_by_identity(
            cell
            for item in unique_observations
            for cell in (
                item.source_cell,
                item.source_row.label_cell,
            )
            if cell is not None
        )
    )
    spans_list = []
    for item in unique_observations:
        spans_list.extend(item.value.evidence_spans)
        spans_list.extend(item.metric.evidence_spans)
        spans_list.extend(item.period.evidence_spans)
        spans_list.extend(
            span
            for unit_candidate in item.value.unit_candidates
            for span in unit_candidate.evidence.source_spans
        )
        spans_list.extend(item.source_table.context_spans)
        for header in item.source_column.header_path if item.source_column else ():
            spans_list.extend(header.provenance.source_spans)
        if item.source_row.label_cell:
            spans_list.extend(item.source_row.label_cell.provenance.source_spans)
    spans = tuple(_unique_by_identity(spans_list))
    boxes = tuple(
        (span.page_number, span.bounding_box)
        for span in spans
    )
    evidence = ValidationEvidence(
        observations=unique_observations,
        source_tables=tables,
        source_cells=cells,
        source_spans=spans,
        page_numbers=tuple(dict.fromkeys(span.page_number for span in spans)),
        bounding_boxes=boxes,
    )
    period_labels = tuple(dict.fromkeys(item.period.raw_label for item in unique_observations if item.period.raw_label))
    return ValidationFinding(
        rule_id=rule_id,
        status=status,
        message=message,
        evidence=evidence,
        period_label=" / ".join(period_labels) if period_labels else None,
        metric_concepts=metric_concepts,
        observed_residual=residual,
        tolerance=tolerance,
    )


def _unique_by_identity(values):
    unique = []
    seen = set()
    for value in values:
        identifier = id(value)
        if identifier not in seen:
            seen.add(identifier)
            unique.append(value)
    return unique


def _calculation_precision(values: Sequence[Decimal]) -> int:
    return max(28, sum(len(value.as_tuple().digits) for value in values) + len(values) + 8)