from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from toresu.context_compiler import (
    EvidenceText,
    FactProvenance,
    FinancialUnit,
    ReportingPeriodFact,
    StatementTitleFact,
    StatementType,
)
from toresu.financial_normalization import (
    MetricConcept,
    FinancialNormalizationResult,
    NormalizationStatus,
    NormalizedFinancialValue,
    NormalizedObservation,
    NormalizedTable,
    ParsedFinancialValue,
    PeriodKind as NormalizedPeriodKind,
    RelativePeriod,
    SemanticMapping,
    ValueKind,
)
from toresu.financial_validation import (
    AccountingEquation,
    BALANCE_SHEET_EQUATION,
    CrossPeriodThreshold,
    EquationScope,
    EquationTerm,
    CrossPeriodThreshold,
    FindingStatus,
    ValidationConfig,
    validate_financial_data,
)
from toresu.pdf_ingestion import ExtractionSource, TextSpan
from toresu.table_reconstruction import (
    AssignmentStatus,
    CellProvenance,
    CellRole,
    ReconstructedCell,
    ReconstructedColumn,
    ReconstructedRow,
    ReconstructedTable,
)


def _observation(
    concept: MetricConcept,
    label: str,
    value: Decimal | None,
    *,
    period_kind: NormalizedPeriodKind = NormalizedPeriodKind.FISCAL_YEAR,
    fiscal_year: int | None = 2025,
    quarter: int | None = None,
    relative_period: RelativePeriod | None = None,
    period_status: NormalizationStatus = NormalizationStatus.RESOLVED,
    unit: FinancialUnit | None = FinancialUnit.YEN,
    unit_status: NormalizationStatus = NormalizationStatus.RESOLVED,
    mapping_status: NormalizationStatus = NormalizationStatus.INFERRED,
    table_id: str = "table-1",
    statement_type: StatementType = StatementType.CONSOLIDATED_BALANCE_SHEET,
    parsed_kind: ValueKind | None = None,
) -> NormalizedObservation:
    page_number = 1
    label_span = TextSpan(
        label,
        page_number,
        (10.0, 20.0, 90.0, 32.0),
        ExtractionSource.PDF_TEXT,
    )
    value_text = str(value) if value is not None else None
    value_span = (
        TextSpan(
            value_text or "",
            page_number,
            (300.0, 20.0, 360.0, 32.0),
            ExtractionSource.PDF_TEXT,
        )
        if value_text is not None
        else None
    )
    source_spans = (value_span,) if value_span is not None else ()
    parsed = ParsedFinancialValue(
        raw_text=value_text,
        normalized_text=value_text,
        value=value,
        kind=parsed_kind or (ValueKind.BLANK if value is None else ValueKind.AMOUNT),
        status=(NormalizationStatus.UNRESOLVED if value is None else NormalizationStatus.RESOLVED),
    )
    cell = ReconstructedCell(
        cell_id=f"{table_id}-{label}",
        role=CellRole.VALUE,
        raw_text=value_text,
        provenance=CellProvenance(
            page_number,
            value_span.bounding_box if value_span else None,
            (ExtractionSource.PDF_TEXT,) if value_span else (),
            source_spans,
        ),
        cell_boundary=(300.0, 20.0, 360.0, 32.0),
        row_index=0,
        column_index=0,
        column_span=1,
        numeric_interpretation=None,
        unit_fact=None,
        unit_status=AssignmentStatus.RESOLVED,
        status=AssignmentStatus.RESOLVED if value is not None else AssignmentStatus.UNRESOLVED,
    )
    label_cell = ReconstructedCell(
        cell_id=f"{table_id}-{label}-label",
        role=CellRole.ROW_LABEL,
        raw_text=label,
        provenance=CellProvenance(
            page_number,
            label_span.bounding_box,
            (ExtractionSource.PDF_TEXT,),
            (label_span,),
        ),
        cell_boundary=label_span.bounding_box,
        row_index=0,
        column_index=None,
        column_span=1,
        numeric_interpretation=None,
        unit_fact=None,
        unit_status=AssignmentStatus.UNRESOLVED,
        status=AssignmentStatus.RESOLVED,
    )
    row = ReconstructedRow(
        row_id=f"{table_id}-row",
        row_index=0,
        bounding_box=(10.0, 20.0, 360.0, 32.0),
        label_cell=label_cell,
        cells=(cell,),
        parent_row_id=None,
        hierarchy_status=AssignmentStatus.UNRESOLVED,
    )
    reporting_period = ReportingPeriodFact(
        raw_label=f"{fiscal_year}年度" if fiscal_year is not None else "当連結会計年度",
        evidence=EvidenceText("period header", (label_span,)),
        dates=(),
    )
    column_period = __import__(
        "toresu.financial_normalization", fromlist=["ReportingPeriod"]
    ).ReportingPeriod(
        raw_label=reporting_period.raw_label,
        kind=period_kind,
        fiscal_year=fiscal_year,
        quarter=quarter,
        relative_period=relative_period,
        date_iso=None,
        status=period_status,
        evidence_spans=(label_span,),
    )
    column = ReconstructedColumn(
        column_index=0,
        left=300.0,
        right=360.0,
        center=330.0,
        header_path=(),
        reporting_period=reporting_period,
        reporting_period_candidates=(reporting_period,),
        period_status=AssignmentStatus.RESOLVED,
        geometry_status=AssignmentStatus.INFERRED,
    )
    title = StatementTitleFact(
        raw_title=statement_type.value,
        statement_type=statement_type,
        evidence=EvidenceText(statement_type.value, (label_span,)),
        extraction_provenance=FactProvenance.EXPLICIT,
        interpretation_provenance=FactProvenance.NORMALIZED,
    )
    table = ReconstructedTable(
        table_id=table_id,
        page_number=page_number,
        bounding_box=(0.0, 0.0, 400.0, 100.0),
        columns=(column,),
        header_rows=(),
        rows=(row,),
        table_units=(),
        statement_title=title,
        context_spans=(label_span, *(source_spans)),
        status=AssignmentStatus.INFERRED,
        issues=(),
    )
    metric = SemanticMapping(
        raw_label=label,
        normalized_label=label,
        candidates=(concept,),
        status=mapping_status,
        evidence_spans=(label_span,),
    )
    normalized_value = NormalizedFinancialValue(
        source_cell=cell,
        parsed=parsed,
        raw_unit_declaration=unit.value if unit else None,
        original_unit=unit,
        normalized_value=value,
        normalized_unit=unit,
        unit_status=unit_status,
        status=NormalizationStatus.RESOLVED if value is not None else NormalizationStatus.UNRESOLVED,
        evidence_spans=source_spans,
    )
    return NormalizedObservation(
        source_table=table,
        source_row=row,
        source_column=column,
        source_cell=cell,
        metric=metric,
        period=column_period,
        value=normalized_value,
    )


def _result(*observations: NormalizedObservation) -> FinancialNormalizationResult:
    tables = []
    for table_id in dict.fromkeys(item.source_table.table_id for item in observations):
        selected = tuple(item for item in observations if item.source_table.table_id == table_id)
        source_table = selected[0].source_table
        tables.append(
            NormalizedTable(
                source_table=source_table,
                row_metrics=tuple(item.metric for item in selected),
                column_periods=tuple(item.period for item in selected),
                observations=selected,
            )
        )
    return FinancialNormalizationResult(tuple(tables))


def _balance_sheet(assets, liabilities, equity, **kwargs):
    return _result(
        _observation(MetricConcept.TOTAL_ASSETS, "資産合計", assets, **kwargs),
        _observation(MetricConcept.LIABILITIES, "負債合計", liabilities, **kwargs),
        _observation(MetricConcept.EQUITY, "純資産合計", equity, **kwargs),
    )


def test_balanced_and_unbalanced_balance_sheet_invariant():
    balanced = validate_financial_data(_balance_sheet(Decimal("1000"), Decimal("600"), Decimal("400")))
    failed = validate_financial_data(_balance_sheet(Decimal("1000"), Decimal("600"), Decimal("390")))

    assert next(item for item in balanced.findings if item.rule_id == BALANCE_SHEET_EQUATION.rule_id).status is FindingStatus.PASSED
    unbalanced = next(item for item in failed.findings if item.rule_id == BALANCE_SHEET_EQUATION.rule_id)
    assert unbalanced.status is FindingStatus.FAILED
    assert unbalanced.observed_residual == Decimal("10")


def test_missing_dash_and_blank_evidence_prevent_equation_check():
    dash = _observation(
        MetricConcept.EQUITY,
        "純資産合計",
        None,
        parsed_kind=ValueKind.DASH,
    )
    result = validate_financial_data(
        _result(
            _observation(MetricConcept.TOTAL_ASSETS, "資産合計", Decimal("1000")),
            _observation(MetricConcept.LIABILITIES, "負債合計", Decimal("600")),
            dash,
        )
    )

    finding = next(item for item in result.findings if item.rule_id == BALANCE_SHEET_EQUATION.rule_id)
    assert finding.status is FindingStatus.INSUFFICIENT_EVIDENCE
    assert dash.source_cell.raw_text is None


def test_rounding_tolerance_is_configurable_and_does_not_round_values():
    data = _balance_sheet(Decimal("1000.01"), Decimal("600"), Decimal("400"))
    within = validate_financial_data(data, config=ValidationConfig(absolute_tolerance=Decimal("0.01")))
    outside = validate_financial_data(data, config=ValidationConfig(absolute_tolerance=Decimal("0.001")))

    assert next(item for item in within.findings if item.rule_id == BALANCE_SHEET_EQUATION.rule_id).status is FindingStatus.PASSED
    finding = next(item for item in outside.findings if item.rule_id == BALANCE_SHEET_EQUATION.rule_id)
    assert finding.status is FindingStatus.FAILED
    assert finding.observed_residual == Decimal("0.01")
    assert data.tables[0].observations[0].value.normalized_value == Decimal("1000.01")


def test_incompatible_units_and_periods_are_distinguished():
    assets = _observation(MetricConcept.TOTAL_ASSETS, "資産合計", Decimal("1000"))
    liabilities = _observation(MetricConcept.LIABILITIES, "負債合計", Decimal("600"))
    equity = _observation(
        MetricConcept.EQUITY,
        "純資産合計",
        Decimal("400"),
        unit=FinancialUnit.PERCENT,
    )
    incompatible_units = validate_financial_data(_result(assets, liabilities, equity))
    assert next(item for item in incompatible_units.findings if item.rule_id == BALANCE_SHEET_EQUATION.rule_id).status is FindingStatus.INCOMPATIBLE_UNITS

    equity_next_year = _observation(
        MetricConcept.EQUITY,
        "純資産合計",
        Decimal("400"),
        fiscal_year=2024,
    )
    incompatible_periods = validate_financial_data(_result(assets, liabilities, equity_next_year))
    assert next(item for item in incompatible_periods.findings if item.rule_id == BALANCE_SHEET_EQUATION.rule_id).status is FindingStatus.INCOMPATIBLE_PERIOD


def test_balanced_percentages_do_not_pass_yen_balance_sheet_equation():
    data = _balance_sheet(
        Decimal("100"),
        Decimal("60"),
        Decimal("40"),
    )
    percent_observations = tuple(
        replace(
            observation,
            value=replace(
                observation.value,
                original_unit=FinancialUnit.PERCENT,
                normalized_unit=FinancialUnit.PERCENT,
            ),
        )
        for observation in data.tables[0].observations
    )
    table = replace(data.tables[0], observations=percent_observations)
    result = validate_financial_data(
        FinancialNormalizationResult((table,))
    )

    finding = next(item for item in result.findings if item.rule_id == BALANCE_SHEET_EQUATION.rule_id)
    assert finding.status is FindingStatus.INCOMPATIBLE_UNITS


def test_ambiguous_unit_evidence_is_not_collapsed_into_insufficient_data():
    assets = _observation(MetricConcept.TOTAL_ASSETS, "資産合計", Decimal("1000"))
    liabilities = _observation(MetricConcept.LIABILITIES, "負債合計", Decimal("600"))
    equity = _observation(
        MetricConcept.EQUITY,
        "純資産合計",
        Decimal("400"),
        unit_status=NormalizationStatus.AMBIGUOUS,
    )

    result = validate_financial_data(_result(assets, liabilities, equity))

    finding = next(item for item in result.findings if item.rule_id == BALANCE_SHEET_EQUATION.rule_id)
    assert finding.status is FindingStatus.INCOMPATIBLE_UNITS


def test_ambiguous_metric_mapping_has_its_own_finding_classification():
    ambiguous = _observation(
        MetricConcept.TOTAL_ASSETS,
        "資産合計",
        Decimal("1000"),
        mapping_status=NormalizationStatus.AMBIGUOUS,
    )
    result = validate_financial_data(
        _result(
            ambiguous,
            _observation(MetricConcept.LIABILITIES, "負債合計", Decimal("600")),
            _observation(MetricConcept.EQUITY, "純資産合計", Decimal("400")),
        )
    )

    assert next(item for item in result.findings if item.rule_id == BALANCE_SHEET_EQUATION.rule_id).status is FindingStatus.AMBIGUOUS_MAPPING


def test_cross_period_large_movement_is_a_warning_with_evidence():
    prior = _observation(
        MetricConcept.REVENUE,
        "売上高",
        Decimal("100"),
        fiscal_year=2024,
        table_id="income-2024",
        statement_type=StatementType.INCOME_STATEMENT,
    )
    current = _observation(
        MetricConcept.REVENUE,
        "売上高",
        Decimal("250"),
        fiscal_year=2025,
        table_id="income-2025",
        statement_type=StatementType.INCOME_STATEMENT,
    )
    result = validate_financial_data(
        _result(prior, current),
        config=ValidationConfig(
            cross_period_thresholds=(
                CrossPeriodThreshold(MetricConcept.REVENUE, Decimal("0.5"), StatementType.INCOME_STATEMENT.value),
            )
        ),
    )

    finding = next(item for item in result.findings if item.rule_id == "cross-period:revenue")
    assert finding.status is FindingStatus.WARNING
    assert finding.observed_residual == Decimal("1.5")
    assert len(finding.evidence.observations) == 2


def test_cross_period_anomaly_does_not_compare_incompatible_period_representations():
    relative = _observation(
        MetricConcept.REVENUE,
        "売上高",
        Decimal("100"),
        fiscal_year=None,
        relative_period=RelativePeriod.PRIOR,
        period_kind=NormalizedPeriodKind.RELATIVE_FISCAL_YEAR,
        table_id="income-prior-label",
        statement_type=StatementType.INCOME_STATEMENT,
    )
    fiscal = _observation(
        MetricConcept.REVENUE,
        "売上高",
        Decimal("150"),
        fiscal_year=2025,
        table_id="income-fiscal",
        statement_type=StatementType.INCOME_STATEMENT,
    )
    result = validate_financial_data(
        _result(relative, fiscal),
        config=ValidationConfig(
            cross_period_thresholds=(
                CrossPeriodThreshold(MetricConcept.REVENUE, Decimal("0.1")),
            )
        ),
    )

    finding = next(item for item in result.findings if item.rule_id == "cross-period:revenue")
    assert finding.status is FindingStatus.INCOMPATIBLE_PERIOD


def test_cross_statement_equation_is_configurable_and_checks_compatible_facts():
    income_revenue = _observation(
        MetricConcept.REVENUE,
        "売上高",
        Decimal("100"),
        table_id="income",
        statement_type=StatementType.INCOME_STATEMENT,
    )
    cashflow_revenue = _observation(
        MetricConcept.REVENUE,
        "売上高",
        Decimal("100"),
        table_id="cashflow",
        statement_type=StatementType.CASH_FLOW_STATEMENT,
    )
    rule = AccountingEquation(
        rule_id="configured-cross-statement-equality",
        result=EquationTerm(MetricConcept.REVENUE, statement_type=StatementType.INCOME_STATEMENT.value),
        components=(EquationTerm(MetricConcept.REVENUE, statement_type=StatementType.CASH_FLOW_STATEMENT.value),),
        scope=EquationScope.CROSS_STATEMENT,
        description="Configured comparable revenue values agree.",
    )
    result = validate_financial_data(_result(income_revenue, cashflow_revenue), config=ValidationConfig(equations=(rule,)))

    finding = next(item for item in result.findings if item.rule_id == rule.rule_id)
    assert finding.status is FindingStatus.PASSED
    assert len(finding.evidence.source_tables) == 2


def test_cross_statement_period_mismatch_is_not_compared():
    first = _observation(
        MetricConcept.REVENUE,
        "売上高",
        Decimal("100"),
        fiscal_year=2025,
        table_id="income",
        statement_type=StatementType.INCOME_STATEMENT,
    )
    second = _observation(
        MetricConcept.REVENUE,
        "売上高",
        Decimal("100"),
        fiscal_year=2024,
        table_id="cashflow",
        statement_type=StatementType.CASH_FLOW_STATEMENT,
    )
    rule = AccountingEquation(
        rule_id="configured-cross-statement-equality",
        result=EquationTerm(MetricConcept.REVENUE, statement_type=StatementType.INCOME_STATEMENT.value),
        components=(EquationTerm(MetricConcept.REVENUE, statement_type=StatementType.CASH_FLOW_STATEMENT.value),),
        scope=EquationScope.CROSS_STATEMENT,
    )
    result = validate_financial_data(_result(first, second), config=ValidationConfig(equations=(rule,)))

    finding = next(item for item in result.findings if item.rule_id == rule.rule_id)
    assert finding.status is FindingStatus.INCOMPATIBLE_PERIOD


def test_findings_retain_original_cells_spans_pages_and_bounding_boxes():
    result = validate_financial_data(_balance_sheet(Decimal("1000"), Decimal("600"), Decimal("400")))
    finding = next(item for item in result.findings if item.rule_id == BALANCE_SHEET_EQUATION.rule_id)

    assert finding.status is FindingStatus.PASSED
    assert len(finding.evidence.observations) == 3
    assert len(finding.evidence.source_cells) >= 6
    assert finding.evidence.source_spans
    assert finding.evidence.page_numbers == (1,)
    assert all(page == 1 for page, _ in finding.evidence.bounding_boxes)
    assert finding.evidence.bounding_boxes


def test_custom_validation_rule_protocol_can_be_added_without_engine_changes():
    class AlwaysWarningRule:
        def evaluate(self, normalized, config):
            observation = normalized.tables[0].observations[0]
            from toresu.financial_validation import _finding

            return (
                _finding(
                    "test-warning",
                    FindingStatus.WARNING,
                    "Test extension point.",
                    (observation,),
                ),
            )

    result = validate_financial_data(
        _result(_observation(MetricConcept.REVENUE, "売上高", Decimal("1"))),
        rules=(AlwaysWarningRule(),),
    )

    assert any(item.rule_id == "test-warning" for item in result.findings)
