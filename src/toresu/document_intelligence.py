from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, localcontext
from typing import Any

from .financial_normalization import (
    FinancialNormalizationResult,
    MetricConcept,
    NormalizationStatus,
    NormalizedObservation,
    PeriodKind,
    RelativePeriod,
)
from .financial_validation import FindingStatus, ValidationReport


def build_document_report(
    normalized: FinancialNormalizationResult,
    validation: ValidationReport | None = None,
) -> dict[str, Any]:
    """Build a conservative, provenance-linked report from pipeline observations."""
    if not isinstance(normalized, FinancialNormalizationResult):
        raise TypeError("normalized must be a FinancialNormalizationResult.")
    if validation is not None and not isinstance(validation, ValidationReport):
        raise TypeError("validation must be a ValidationReport or None.")

    observations = [
        observation
        for table in normalized.tables
        for observation in table.observations
    ]
    statements = _statement_metadata(observations)
    eligible: list[tuple[NormalizedObservation, MetricConcept, tuple[Any, ...]]] = []
    unresolved = []
    for observation in observations:
        if len(observation.metric.candidates) != 1 or observation.metric.status not in {
            NormalizationStatus.INFERRED,
            NormalizationStatus.RESOLVED,
        }:
            unresolved.append(_unresolved_observation(observation, "metric_mapping"))
            continue
        concept = observation.metric.candidates[0]
        period_key = _period_key(observation)
        if period_key is None:
            unresolved.append(_unresolved_observation(observation, "reporting_period"))
            continue
        if (
            observation.value.normalized_value is None
            or observation.value.normalized_unit is None
            or observation.value.unit_status is not NormalizationStatus.RESOLVED
            or observation.value.status
            not in {NormalizationStatus.RESOLVED, NormalizationStatus.INFERRED}
        ):
            unresolved.append(_unresolved_observation(observation, "normalized_value_or_unit"))
            continue
        eligible.append((observation, concept, period_key))

    return {
        "report_version": "1",
        "document_type": (
            "japanese_financial_statements" if statements else "unknown"
        ),
        "statements": statements,
        "reporting_periods": _period_metadata(observations),
        "observations": [_observation_record(item) for item in observations],
        "cross_statement_links": _cross_statement_links(eligible),
        "cross_period_comparisons": _cross_period_comparisons(eligible),
        "consistency_findings": [
            _finding_record(finding)
            for finding in (validation.findings if validation else ())
        ],
        "unresolved_ambiguities": unresolved,
        "limitations": [
            "Narrative-to-table reconciliation is unavailable because narrative figures are not classified as typed financial observations.",
            "Restatement relationships are not inferred; periods without an explicit supported identity remain unresolved.",
        ],
    }


def _statement_metadata(observations: list[NormalizedObservation]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for observation in observations:
        title = observation.source_table.statement_title
        if title is None:
            continue
        record = seen.setdefault(
            title.statement_type.value,
            {
                "statement_type": title.statement_type.value,
                "title": title.raw_title,
                "table_ids": [],
                "source_spans": [],
            },
        )
        if observation.source_table.table_id not in record["table_ids"]:
            record["table_ids"].append(observation.source_table.table_id)
        record["source_spans"] = _merge_spans(
            record["source_spans"],
            [_span_reference(span) for span in title.evidence.source_spans],
        )
    return list(seen.values())


def _observation_record(observation: NormalizedObservation) -> dict[str, Any]:
    title = observation.source_table.statement_title
    return {
        "reference": _observation_reference(observation),
        "statement_type": title.statement_type.value if title else None,
        "metric": observation.metric.candidates[0].value
        if len(observation.metric.candidates) == 1
        else None,
        "metric_candidates": [candidate.value for candidate in observation.metric.candidates],
        "metric_status": observation.metric.status.value,
        "period": {
            "raw_label": observation.period.raw_label,
            "kind": observation.period.kind.value,
            "fiscal_year": observation.period.fiscal_year,
            "quarter": observation.period.quarter,
            "relative_period": (
                observation.period.relative_period.value
                if observation.period.relative_period
                else None
            ),
            "date_iso": observation.period.date_iso,
            "status": observation.period.status.value,
            "source_spans": [
                _span_reference(span) for span in observation.period.evidence_spans
            ],
        },
        "row_label": observation.metric.raw_label,
        "raw_value": observation.value.parsed.raw_text,
        "parsed_value": _decimal_string(observation.value.parsed.value),
        "value_kind": observation.value.parsed.kind.value,
        "normalized_value": _decimal_string(observation.value.normalized_value),
        "normalized_unit": (
            observation.value.normalized_unit.value
            if observation.value.normalized_unit
            else None
        ),
        "status": observation.value.status.value,
        "source_spans": [
            _span_reference(span) for span in observation.value.evidence_spans
        ],
        "metric_source_spans": [
            _span_reference(span) for span in observation.metric.evidence_spans
        ],
        "unit_source_spans": [
            _span_reference(span)
            for candidate in observation.value.unit_candidates
            for span in candidate.evidence.source_spans
        ],
    }


def _cross_statement_links(
    eligible: list[
        tuple[NormalizedObservation, MetricConcept, tuple[Any, ...]]
    ],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, tuple[Any, ...]], list[NormalizedObservation]] = defaultdict(list)
    for observation, concept, period in eligible:
        grouped[(concept.value, period)].append(observation)
    links = []
    for (concept, period), group in grouped.items():
        statement_kinds = [
            item.source_table.statement_title.statement_type
            if item.source_table.statement_title
            else None
            for item in group
        ]
        distinct = {kind for kind in statement_kinds if kind is not None}
        if len(distinct) < 2:
            continue
        units = {item.value.normalized_unit for item in group}
        duplicate_statements = len(statement_kinds) != len(set(statement_kinds))
        consolidated_scopes = {
            kind.value.startswith("consolidated_")
            for kind in distinct
        }
        if len(consolidated_scopes) > 1:
            status = "incompatible_statement_scope"
        elif len(units) != 1:
            status = "incompatible_units"
        elif duplicate_statements or len(distinct) != len(statement_kinds):
            status = "ambiguous"
        else:
            status = "linked"
        links.append(
            {
                "metric": concept,
                "period_key": list(period),
                "status": status,
                "rationale": (
                    "Exact registered metric identity, resolved common period, and normalized-unit compatibility."
                    if status == "linked"
                    else "Consolidated and non-consolidated statement observations are not interchangeable."
                    if status == "incompatible_statement_scope"
                    else "Candidate observations share a metric and period, but evidence is not uniquely compatible."
                ),
                "observations": [
                    _link_observation(item) for item in group
                ],
                "provenance": _merge_spans(
                    [],
                    [
                        span
                        for item in group
                        for span in _observation_span_refs(item)
                    ],
                ),
            }
        )
    return links


def _cross_period_comparisons(
    eligible: list[
        tuple[NormalizedObservation, MetricConcept, tuple[Any, ...]]
    ],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[tuple[NormalizedObservation, tuple[Any, ...]]]] = defaultdict(list)
    period_families: dict[
        tuple[str, str], list[tuple[NormalizedObservation, tuple[Any, ...]]]
    ] = defaultdict(list)
    for observation, concept, period in eligible:
        title = observation.source_table.statement_title
        if title is None:
            continue
        family = str(period[0])
        grouped[(concept.value, title.statement_type.value, family)].append((observation, period))
        period_families[(concept.value, title.statement_type.value)].append(
            (observation, period)
        )

    comparisons = []
    for (concept, statement_type), items in period_families.items():
        families = {period[0] for _, period in items}
        if len(families) > 1:
            comparisons.append(
                {
                    "metric": concept,
                    "statement_type": statement_type,
                    "status": "incompatible_period_families",
                    "observations": [
                        _link_observation(item) for item, _ in items
                    ],
                    "absolute_change": None,
                    "percentage_change": None,
                    "rationale": "Annual, quarter, relative, and explicit-date identities are not mixed.",
                }
            )
    for (concept, statement_type, _), candidates in grouped.items():
        by_period: dict[tuple[Any, ...], list[NormalizedObservation]] = defaultdict(list)
        for observation, period in candidates:
            by_period[period].append(observation)
        duplicate_periods = {
            period for period, items in by_period.items() if len(items) != 1
        }
        for period in duplicate_periods:
            comparisons.append(
                {
                    "metric": concept,
                    "statement_type": statement_type,
                    "status": "ambiguous_period_value",
                    "period_key": list(period),
                    "observations": [
                        _link_observation(item) for item in by_period[period]
                    ],
                    "absolute_change": None,
                    "percentage_change": None,
                    "formula": "(current - prior) / abs(prior); no value is selected when a period has multiple observations",
                }
            )
        if duplicate_periods:
            continue
        unique_periods = [
            (period, items[0])
            for period, items in by_period.items()
            if len(items) == 1
        ]
        unique_periods.sort(key=lambda pair: _period_order(pair[1]))
        for (prior_period, prior), (current_period, current) in zip(
            unique_periods, unique_periods[1:]
        ):
            if prior.value.normalized_unit != current.value.normalized_unit:
                status = "incompatible_units"
                absolute_change = percentage_change = None
            else:
                status = "compared"
                prior_value = prior.value.normalized_value
                current_value = current.value.normalized_value
                absolute_change = current_value - prior_value
                if prior_value == 0:
                    percentage_change = None
                else:
                    with localcontext() as context:
                        context.prec = _precision(prior_value, current_value)
                        percentage_change = (
                            (current_value - prior_value) / abs(prior_value) * Decimal(100)
                        )
            comparisons.append(
                {
                    "metric": concept,
                    "statement_type": statement_type,
                    "status": status,
                    "prior": _link_observation(prior),
                    "current": _link_observation(current),
                    "prior_period_key": list(prior_period),
                    "current_period_key": list(current_period),
                    "normalized_unit": (
                        prior.value.normalized_unit.value
                        if status == "compared"
                        else None
                    ),
                    "absolute_change": _decimal_string(absolute_change),
                    "percentage_change": _decimal_string(percentage_change),
                    "percentage_change_definition": (
                        "(current - prior) / abs(prior) * 100; undefined when prior is zero"
                    ),
                }
            )
    return comparisons


def _period_key(observation: NormalizedObservation) -> tuple[Any, ...] | None:
    period = observation.period
    if period.status is not NormalizationStatus.RESOLVED:
        return None
    if period.date_iso:
        return ("date", period.date_iso)
    if period.kind is PeriodKind.QUARTER and period.fiscal_year and period.quarter:
        return ("quarter", period.fiscal_year, period.quarter)
    if period.kind is PeriodKind.FISCAL_YEAR and period.fiscal_year:
        return ("fiscal_year", period.fiscal_year)
    if (
        period.kind is PeriodKind.RELATIVE_FISCAL_YEAR
        and period.relative_period is not None
    ):
        return ("relative", period.relative_period.value)
    return None


def _period_order(observation: NormalizedObservation) -> tuple[int, ...]:
    period = observation.period
    if period.kind is PeriodKind.RELATIVE_FISCAL_YEAR:
        return (
            -1 if period.relative_period is RelativePeriod.PRIOR else 0,
            0,
        )
    if period.date_iso:
        return tuple(int(part) for part in period.date_iso.split("-"))
    if period.fiscal_year is not None:
        return (period.fiscal_year, period.quarter or 0)
    return (0, 0)


def _finding_record(finding: Any) -> dict[str, Any]:
    expected_relationship = (
        "total_assets = total_liabilities + equity"
        if finding.rule_id == "balance-sheet-assets-equal-liabilities-and-equity"
        else "relative change is compared with its configured review threshold"
        if finding.rule_id.startswith("cross-period:")
        else f"Caller-configured relationship for rule {finding.rule_id}."
    )
    return {
        "rule_id": finding.rule_id,
        "status": finding.status.value,
        "classification": (
            "review_candidate"
            if finding.status is FindingStatus.WARNING
            else "arithmetic_or_evidence_check"
        ),
        "message": finding.message,
        "expected_relationship": (
            finding.expected_relationship or expected_relationship
        ),
        "period_label": finding.period_label,
        "metric_concepts": [concept.value for concept in finding.metric_concepts],
        "actual_result": {
            "observed_residual": _decimal_string(finding.observed_residual),
            "tolerance": _decimal_string(finding.tolerance),
            "observations": [
                _finding_observation_record(observation)
                for observation in finding.evidence.observations
            ],
        },
        "observed_residual": _decimal_string(finding.observed_residual),
        "tolerance": _decimal_string(finding.tolerance),
        "provenance": [
            _span_reference(span) for span in finding.evidence.source_spans
        ],
    }


def _finding_observation_record(
    observation: NormalizedObservation,
) -> dict[str, Any]:
    record = _link_observation(observation)
    record["metric_candidates"] = [
        candidate.value for candidate in observation.metric.candidates
    ]
    record["raw_value"] = observation.value.parsed.raw_text
    return record


def _unresolved_observation(
    observation: NormalizedObservation,
    reason: str,
) -> dict[str, Any]:
    return {
        "reference": _observation_reference(observation),
        "reason": reason,
        "issue": (
            observation.metric.issue
            if reason == "metric_mapping"
            else observation.period.issue
            if reason == "reporting_period"
            else observation.value.issue or observation.value.parsed.issue
        ),
        "metric_candidates": [item.value for item in observation.metric.candidates],
        "period_label": observation.period.raw_label,
        "source_spans": _observation_span_refs(observation),
    }


def _period_metadata(observations: list[NormalizedObservation]) -> list[dict[str, Any]]:
    periods: dict[tuple[Any, ...], dict[str, Any]] = {}
    for observation in observations:
        key = _period_key(observation)
        identity = key if key is not None else ("unresolved", observation.period.raw_label)
        record = periods.setdefault(
            identity,
            {
                "key": list(identity),
                "raw_labels": [],
                "status": observation.period.status.value,
                "source_spans": [],
            },
        )
        if observation.period.raw_label not in record["raw_labels"]:
            record["raw_labels"].append(observation.period.raw_label)
        record["source_spans"] = _merge_spans(
            record["source_spans"],
            [_span_reference(span) for span in observation.period.evidence_spans],
        )
    return list(periods.values())


def _link_observation(observation: NormalizedObservation) -> dict[str, Any]:
    title = observation.source_table.statement_title
    return {
        "reference": _observation_reference(observation),
        "statement_type": title.statement_type.value if title else None,
        "period_label": observation.period.raw_label,
        "normalized_value": _decimal_string(observation.value.normalized_value),
        "normalized_unit": (
            observation.value.normalized_unit.value
            if observation.value.normalized_unit
            else None
        ),
    }


def _observation_reference(observation: NormalizedObservation) -> dict[str, Any]:
    cell = observation.source_cell
    return {
        "table_id": observation.source_table.table_id,
        "row_id": observation.source_row.row_id,
        "cell_id": cell.cell_id,
        "page_number": cell.provenance.page_number,
        "bounding_box": cell.provenance.bounding_box,
    }


def _observation_span_refs(observation: NormalizedObservation) -> list[dict[str, Any]]:
    spans = [
        *observation.value.evidence_spans,
        *observation.metric.evidence_spans,
        *observation.period.evidence_spans,
        *(
            span
            for unit in observation.value.unit_candidates
            for span in unit.evidence.source_spans
        ),
    ]
    return _merge_spans([], [_span_reference(span) for span in spans])


def _span_reference(span: Any) -> dict[str, Any]:
    return {
        "raw_text": span.raw_text,
        "page_number": span.page_number,
        "bounding_box": list(span.bounding_box),
        "extraction_source": span.extraction_source.value,
        "confidence": span.confidence,
    }


def _merge_spans(
    existing: list[dict[str, Any]],
    additional: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = list(existing)
    seen = {
        (item.get("page_number"), tuple(item.get("bounding_box", ())), item.get("raw_text"))
        for item in merged
    }
    for item in additional:
        key = (
            item.get("page_number"),
            tuple(item.get("bounding_box", ())),
            item.get("raw_text"),
        )
        if key not in seen:
            merged.append(item)
            seen.add(key)
    return merged


def _decimal_string(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


def _precision(*values: Decimal) -> int:
    return max(28, sum(len(value.as_tuple().digits) for value in values) + 8)
