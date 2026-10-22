from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from toresu.context_compiler import (
    EvidenceText,
    FactProvenance,
    FinancialUnit,
    ReportingDateFact,
    compile_financial_context,
)
from toresu.financial_normalization import (
    DEFAULT_METRIC_REGISTRY,
    MetricConcept,
    MetricRegistry,
    NormalizationStatus,
    PeriodKind,
    RelativePeriod,
    ValueKind,
    normalize_financial_tables,
    parse_financial_value,
)
from toresu.pdf_ingestion import (
    ExtractionSource,
    PDFDocument,
    PDFPage,
    TextSpan,
)
from toresu.table_reconstruction import AssignmentStatus, reconstruct_tables


def _span(text: str, x0: float, y0: float, x1: float) -> TextSpan:
    return TextSpan(
        text,
        1,
        (x0, y0, x1, y0 + 10),
        ExtractionSource.PDF_TEXT,
    )


def _pipeline(unit_declarations: tuple[str, ...] = ("（単位：千円）",)):
    spans = [
        _span("損益計算書", 100, 10, 250),
        *(_span(declaration, 100, 22 + index * 2, 250) for index, declaration in enumerate(unit_declarations)),
        _span("前連結会計年度", 210, 40, 300),
        _span("当連結会計年度", 320, 40, 410),
        _span("売上高", 30, 65, 100),
        _span("1,234.50", 230, 65, 280),
        _span("2,500", 340, 65, 390),
        _span("営業利益", 30, 85, 100),
        _span("△250", 230, 85, 280),
        _span("300", 340, 85, 390),
    ]
    page = PDFPage(1, (0.0, 0.0, 600.0, 500.0), tuple(spans), True, False)
    document = PDFDocument(Path("synthetic.pdf"), (page,))
    context = compile_financial_context(document)
    reconstruction = reconstruct_tables(document, context)
    return document, context, reconstruction


@pytest.mark.parametrize(
    ("raw", "expected", "kind"),
    [
        ("△2,500", Decimal("-2500"), ValueKind.AMOUNT),
        ("▲２，５００", Decimal("-2500"), ValueKind.AMOUNT),
        ("(1,234.50)", Decimal("-1234.50"), ValueKind.AMOUNT),
        ("＋１，２３４．５０", Decimal("1234.50"), ValueKind.AMOUNT),
        ("12.50%", Decimal("12.50"), ValueKind.PERCENTAGE),
        ("△12.5％", Decimal("-12.5"), ValueKind.PERCENTAGE),
        ("0", Decimal("0"), ValueKind.ZERO),
    ],
)
def test_financial_notation_parses_with_decimal_precision(raw, expected, kind):
    parsed = parse_financial_value(raw)

    assert parsed.raw_text == raw
    assert parsed.value == expected
    assert parsed.kind is kind
    assert parsed.status is NormalizationStatus.RESOLVED


@pytest.mark.parametrize("raw", [None, "", "　 "])
def test_blank_values_are_not_zero(raw):
    parsed = parse_financial_value(raw)

    assert parsed.kind is ValueKind.BLANK
    assert parsed.value is None
    assert parsed.status is NormalizationStatus.UNRESOLVED


@pytest.mark.parametrize("raw", ["-", "‐", "‑", "‒", "–", "—", "―", "−", "ー", "－"])
def test_dash_variants_are_not_zero(raw):
    parsed = parse_financial_value(raw)

    assert parsed.kind is ValueKind.DASH
    assert parsed.value is None
    assert parsed.status is NormalizationStatus.RESOLVED


@pytest.mark.parametrize("raw", ["N/A", "NA", "該当なし", "該当事項なし"])
def test_unavailable_representations_remain_distinct(raw):
    parsed = parse_financial_value(raw)

    assert parsed.kind is ValueKind.UNAVAILABLE
    assert parsed.value is None


@pytest.mark.parametrize("raw", ["1,00", "(△1,000)", "1,000円", "約1,000"])
def test_unsupported_or_ambiguous_notation_is_not_interpreted(raw):
    parsed = parse_financial_value(raw)

    assert parsed.kind is ValueKind.UNSUPPORTED
    assert parsed.value is None
    assert parsed.status is NormalizationStatus.UNSUPPORTED
    assert parsed.issue


@pytest.mark.parametrize(
    ("unit_text", "unit", "expected"),
    [
        ("（単位：円）", FinancialUnit.YEN, Decimal("1234.50")),
        ("（単位：千円）", FinancialUnit.THOUSAND_YEN, Decimal("1234500.00")),
        ("（単位：百万円）", FinancialUnit.MILLION_YEN, Decimal("1234500000.00")),
        ("（単位：億円）", FinancialUnit.HUNDRED_MILLION_YEN, Decimal("123450000000.00")),
    ],
)
def test_explicit_supported_units_normalize_to_yen(unit_text, unit, expected):
    document, context, _ = _pipeline((unit_text,))
    result = normalize_financial_tables(reconstruct_tables(document, context), context)
    observation = next(item for item in result.tables[0].observations if item.source_row.label_cell.raw_text == "売上高")

    assert observation.value.raw_unit_declaration == unit_text
    assert observation.value.original_unit is unit
    assert observation.value.normalized_value == expected
    assert observation.value.normalized_unit is FinancialUnit.YEN
    assert observation.value.parsed.raw_text == "1,234.50"
    assert observation.value.evidence_spans == observation.source_cell.provenance.source_spans
    assert observation.value.status is NormalizationStatus.RESOLVED


def test_missing_unit_preserves_numeric_parse_but_does_not_convert():
    document, context, reconstruction = _pipeline(())
    result = normalize_financial_tables(reconstruction, context)

    value = result.tables[0].observations[0].value
    assert value.parsed.value == Decimal("1234.50")
    assert value.raw_unit_declaration is None
    assert value.normalized_value is None
    assert value.normalized_unit is None
    assert value.status is NormalizationStatus.UNRESOLVED


def test_conflicting_table_units_are_preserved_and_block_conversion():
    document, context, reconstruction = _pipeline(("（単位：千円）", "（単位：百万円）"))
    result = normalize_financial_tables(reconstruction, context)

    value = result.tables[0].observations[0].value
    assert value.unit_status is NormalizationStatus.AMBIGUOUS
    assert value.normalized_value is None
    assert value.raw_unit_declaration is None
    assert value.issue


def test_percentage_unit_does_not_scale_to_a_fraction():
    document, context, reconstruction = _pipeline(("（単位：％）",))
    result = normalize_financial_tables(reconstruction, context)

    value = result.tables[0].observations[0].value
    assert value.normalized_value == Decimal("1234.50")
    assert value.normalized_unit is FinancialUnit.PERCENT


def test_percentage_suffix_supplies_its_own_unit_evidence_without_table_declaration():
    parsed = parse_financial_value("12.50%")
    document, context, reconstruction = _pipeline(())
    table = reconstruction.tables[0]
    row = table.rows[0]
    percent_span = _span("12.50%", 230, 65, 280)
    cell = replace(row.cells[0], raw_text="12.50%", provenance=replace(
        row.cells[0].provenance,
        bounding_box=percent_span.bounding_box,
        source_spans=(percent_span,),
    ))
    updated_row = replace(row, cells=(cell, *row.cells[1:]))
    updated_table = replace(table, rows=(updated_row, *table.rows[1:]))
    normalized = normalize_financial_tables(
        replace(reconstruction, tables=(updated_table,)), context
    ).tables[0].observations[0].value

    assert parsed.kind is ValueKind.PERCENTAGE
    assert normalized.raw_unit_declaration is None
    assert normalized.original_unit is FinancialUnit.PERCENT
    assert normalized.normalized_value == Decimal("12.50")


def test_large_decimal_unit_conversion_does_not_round_to_context_precision():
    document, context, reconstruction = _pipeline(("（単位：千円）",))
    table = reconstruction.tables[0]
    row = table.rows[0]
    amount = "123456789012345678901234567890.12"
    span = _span(amount, 230, 65, 280)
    cell = replace(row.cells[0], raw_text=amount, provenance=replace(
        row.cells[0].provenance,
        bounding_box=span.bounding_box,
        source_spans=(span,),
    ))
    updated = replace(row, cells=(cell, *row.cells[1:]))
    normalized = normalize_financial_tables(
        replace(reconstruction, tables=(replace(table, rows=(updated, *table.rows[1:])),)),
        context,
    ).tables[0].observations[0].value

    assert normalized.normalized_value == Decimal("123456789012345678901234567890120.00")


def test_semantic_registry_is_exact_extensible_and_can_report_ambiguity():
    label_span = _span("営業利益", 10, 10, 60)
    mapping = DEFAULT_METRIC_REGISTRY.resolve("営業利益", (label_span,))
    extended = MetricRegistry(
        {**DEFAULT_METRIC_REGISTRY.aliases, "営業利益": (MetricConcept.OPERATING_INCOME, MetricConcept.REVENUE)}
    ).resolve("営業利益", (label_span,))
    unknown = DEFAULT_METRIC_REGISTRY.resolve("推定営業利益", (label_span,))

    assert mapping.candidates == (MetricConcept.OPERATING_INCOME,)
    assert mapping.status is NormalizationStatus.INFERRED
    assert mapping.raw_label == "営業利益"
    assert mapping.evidence_spans == (label_span,)
    assert extended.status is NormalizationStatus.AMBIGUOUS
    assert unknown.status is NormalizationStatus.UNRESOLVED
    assert unknown.candidates == ()


def test_default_semantic_registry_covers_required_financial_concepts():
    labels = {
        "売上高": MetricConcept.REVENUE,
        "営業利益": MetricConcept.OPERATING_INCOME,
        "経常利益": MetricConcept.ORDINARY_INCOME,
        "当期純利益": MetricConcept.NET_INCOME,
        "資産合計": MetricConcept.TOTAL_ASSETS,
        "負債合計": MetricConcept.LIABILITIES,
        "純資産合計": MetricConcept.EQUITY,
        "営業活動によるキャッシュ・フロー": MetricConcept.OPERATING_CASH_FLOW,
    }

    for label, concept in labels.items():
        assert DEFAULT_METRIC_REGISTRY.resolve(label, ()).candidates == (concept,)


def test_reconstructed_values_keep_original_japanese_labels_spans_and_periods():
    document, context, reconstruction = _pipeline()
    result = normalize_financial_tables(reconstruction, context)
    observation = result.tables[0].observations[0]

    assert observation.metric.raw_label == "売上高"
    assert observation.metric.evidence_spans == observation.source_row.label_cell.provenance.source_spans
    assert observation.period.raw_label in {"前連結会計年度", "当連結会計年度"}
    assert observation.period.kind is PeriodKind.RELATIVE_FISCAL_YEAR
    assert observation.period.relative_period in {RelativePeriod.PRIOR, RelativePeriod.CURRENT}
    assert observation.source_cell.provenance.source_spans
    assert observation.source_cell.provenance.page_number == 1


def test_fiscal_year_quarter_and_ambiguous_periods_are_structured():
    # Fiscal and quarter patterns are parsed from explicit reconstructed header text.
    document, context, reconstruction = _pipeline()
    table = reconstruction.tables[0]
    column = table.columns[0]
    fiscal_header = replace(column.header_path[-1], raw_text="2025年度 第１四半期")
    updated_column = replace(
        column,
        header_path=(*column.header_path[:-1], fiscal_header),
        reporting_period=None,
        reporting_period_candidates=(),
        period_status=AssignmentStatus.UNRESOLVED,
    )
    updated_table = replace(table, columns=(updated_column, *table.columns[1:]))
    result = normalize_financial_tables(
        replace(reconstruction, tables=(updated_table,)), context
    )

    period = result.tables[0].column_periods[0]
    assert period.kind is PeriodKind.QUARTER
    assert period.fiscal_year == 2025
    assert period.quarter == 1
    assert period.status is NormalizationStatus.RESOLVED


@pytest.mark.parametrize(
    ("label", "expected_year"),
    [("令和8年度", 2026), ("平成30年度", 2018), ("2025年3月期", 2025)],
)
def test_supported_japanese_and_gregorian_fiscal_years(label, expected_year):
    document, context, reconstruction = _pipeline()
    table = reconstruction.tables[0]
    column = table.columns[0]
    header = replace(column.header_path[-1], raw_text=label)
    updated_column = replace(
        column,
        header_path=(*column.header_path[:-1], header),
        reporting_period=None,
        reporting_period_candidates=(),
        period_status=AssignmentStatus.UNRESOLVED,
    )
    normalized = normalize_financial_tables(
        replace(reconstruction, tables=(replace(table, columns=(updated_column, *table.columns[1:])),)),
        context,
    )

    period = normalized.tables[0].column_periods[0]
    assert period.kind is PeriodKind.FISCAL_YEAR
    assert period.fiscal_year == expected_year
    assert period.date_iso is None


def test_explicit_date_in_header_evidence_normalizes_without_fiscal_guess():
    document, context, reconstruction = _pipeline()
    table = reconstruction.tables[0]
    date_span = TextSpan(
        "令和8年3月31日",
        1,
        table.columns[0].header_path[0].cell_boundary,
        ExtractionSource.PDF_TEXT,
    )
    date_header = replace(
        table.columns[0].header_path[0],
        raw_text="令和8年3月31日",
        provenance=replace(
            table.columns[0].header_path[0].provenance,
            source_spans=(date_span,),
        ),
    )
    column = replace(
        table.columns[0],
        header_path=(date_header,),
        reporting_period=None,
        reporting_period_candidates=(),
        period_status=AssignmentStatus.UNRESOLVED,
    )
    date_fact = ReportingDateFact(
        raw_text="令和8年3月31日",
        iso_date="2026-03-31",
        evidence=EvidenceText("令和8年3月31日", (date_span,)),
        normalization_provenance=FactProvenance.NORMALIZED,
    )
    context_with_date = replace(context, reporting_dates=(date_fact,))
    normalized = normalize_financial_tables(
        replace(reconstruction, tables=(replace(table, columns=(column, *table.columns[1:])),)),
        context_with_date,
    )

    period = normalized.tables[0].column_periods[0]
    assert period.kind is PeriodKind.DATE
    assert period.date_iso == "2026-03-31"


def test_ambiguous_reconstructed_column_keeps_period_unresolved():
    document, context, reconstruction = _pipeline()
    table = reconstruction.tables[0]
    column = replace(
        table.columns[0],
        reporting_period=None,
        period_status=AssignmentStatus.AMBIGUOUS,
    )
    updated = replace(table, columns=(column, *table.columns[1:]))
    result = normalize_financial_tables(
        replace(reconstruction, tables=(updated,)), context
    )

    assert result.tables[0].column_periods[0].status is NormalizationStatus.AMBIGUOUS
    assert result.tables[0].column_periods[0].date_iso is None
