from pathlib import Path

import pytest

from toresu.context_compiler import (
    DocumentType,
    FactProvenance,
    FinancialUnit,
    NumericValueKind,
    UnitScope,
    compile_financial_context,
    parse_numeric_value,
)
from toresu.pdf_ingestion import (
    ExtractionSource,
    PDFDocument,
    PDFPage,
    TextSpan,
)


def _span(
    text: str,
    *,
    page_number: int = 1,
    y: float = 10,
    x: float = 10,
    source: ExtractionSource = ExtractionSource.PDF_TEXT,
) -> TextSpan:
    return TextSpan(
        raw_text=text,
        page_number=page_number,
        bounding_box=(x, y, x + max(len(text), 1) * 5, y + 10),
        extraction_source=source,
    )


def _document(*page_spans: tuple[TextSpan, ...]) -> PDFDocument:
    pages = tuple(
        PDFPage(
            page_number=index,
            bounding_box=(0.0, 0.0, 600.0, 800.0),
            text_spans=spans,
            native_text_usable=True,
            ocr_attempted=False,
        )
        for index, spans in enumerate(page_spans, start=1)
    )
    return PDFDocument(source_path=Path("synthetic.pdf"), pages=pages)


def test_statement_titles_headings_and_provenance_are_preserved():
    title = _span("連結貸借対照表", y=10)
    heading = _span("流動資産", y=30)
    nearby = _span("（単位：千円）", y=20)
    context = compile_financial_context(_document((title, nearby, heading)))

    assert context.document_type is DocumentType.JAPANESE_FINANCIAL_STATEMENTS
    assert context.document_type_provenance is FactProvenance.INFERRED
    assert context.document_type_evidence == (context.statement_titles[0].evidence,)
    assert [item.raw_title for item in context.statement_titles] == ["連結貸借対照表"]
    assert context.statement_titles[0].statement_type.value == "consolidated_balance_sheet"
    section = context.statement_sections[0]
    assert [item.raw_heading for item in section.headings] == ["流動資産"]
    assert section.title.evidence.source_spans == (title,)
    assert section.title.evidence.page_numbers == (1,)
    assert section.title.evidence.bounding_box == title.bounding_box
    assert section.title.extraction_provenance is FactProvenance.EXPLICIT
    assert section.title.interpretation_provenance is FactProvenance.NORMALIZED
    assert nearby.raw_text in [item.raw_text for item in section.nearby_evidence]
    assert any(item.raw_text == "連結貸借対照表" for item in context.document_evidence)


def test_section_headings_follow_source_order():
    context = compile_financial_context(
        _document((_span("連結損益計算書 営業利益 売上高", y=10),))
    )

    headings = context.statement_sections[0].headings
    assert [heading.raw_heading for heading in headings] == ["営業利益", "売上高"]


def test_income_statement_title_is_identified_without_translation():
    raw_title = "損益計算書"

    context = compile_financial_context(_document((_span(raw_title),)))

    assert context.statement_titles[0].raw_title == raw_title
    assert context.statement_titles[0].statement_type.value == "income_statement"


@pytest.mark.parametrize(
    ("raw_title", "statement_type"),
    [
        ("連結キャッシュ・フロー計算書", "consolidated_cash_flow_statement"),
        ("キャッシュ・フロー計算書", "cash_flow_statement"),
        (
            "連結株主資本等変動計算書",
            "consolidated_statement_of_changes_in_equity",
        ),
        ("株主資本等変動計算書", "statement_of_changes_in_equity"),
    ],
)
def test_additional_readme_statement_titles_are_identified(raw_title, statement_type):
    context = compile_financial_context(_document((_span(raw_title),)))

    assert context.statement_titles[0].raw_title == raw_title
    assert context.statement_titles[0].statement_type.value == statement_type


def test_repeated_statement_titles_keep_their_own_section_evidence():
    spans = (
        _span("連結貸借対照表", y=10),
        _span("（単位：千円）", y=20),
        _span("連結貸借対照表", y=40),
        _span("（単位：百万円）", y=50),
    )

    context = compile_financial_context(_document(spans))

    assert [unit.associated_section_title for unit in context.units] == [
        "連結貸借対照表",
        "連結貸借対照表",
    ]
    assert [unit.normalized_unit for unit in context.units] == [
        FinancialUnit.THOUSAND_YEN,
        FinancialUnit.MILLION_YEN,
    ]
    assert context.units[0].evidence is not context.units[1].evidence


def test_multiple_statement_types_are_reported_as_hypothesis_not_selected():
    context = compile_financial_context(
        _document((_span("連結貸借対照表"), _span("損益計算書", y=30)))
    )

    assert len(context.statement_titles) == 2
    assert any("Multiple distinct" in item.description for item in context.hypotheses)


def test_period_date_association_is_limited_to_explicit_same_line_evidence():
    raw = "前連結会計年度 2025年3月31日"
    separate_date = _span("2026年3月31日", y=30)
    context = compile_financial_context(
        _document((_span(raw, y=10), separate_date))
    )

    assert len(context.reporting_periods) == 1
    period = context.reporting_periods[0]
    assert period.raw_label == "前連結会計年度"
    assert len(period.dates) == 1
    assert period.dates[0].raw_text == "2025年3月31日"
    assert period.dates[0].iso_date == "2025-03-31"
    assert period.dates[0].association_provenance is FactProvenance.EXPLICIT
    assert context.reporting_dates[1].raw_text == "2026年3月31日"
    assert context.reporting_dates[1].associated_period_label is None


def test_period_and_date_in_separate_spans_on_same_line_keep_both_sources():
    label_span = _span("当連結会計年度", y=10, x=10)
    date_span = _span("2026年3月31日", y=10, x=120)

    context = compile_financial_context(_document((label_span, date_span)))

    period = context.reporting_periods[0]
    assert period.dates[0].associated_period_label == "当連結会計年度"
    assert period.dates[0].iso_date == "2026-03-31"
    assert period.evidence.source_spans == (label_span, date_span)
    assert period.dates[0].evidence.source_spans == (label_span, date_span)


@pytest.mark.parametrize(
    "label",
    ["前連結会計年度", "当連結会計年度", "前事業年度", "当事業年度"],
)
def test_supported_period_labels_are_extracted_verbatim(label):
    context = compile_financial_context(_document((_span(label),)))

    assert [period.raw_label for period in context.reporting_periods] == [label]
    assert context.reporting_periods[0].dates == ()


def test_japanese_era_date_normalizes_without_losing_raw_text():
    raw = "当連結会計年度 令和８年３月３１日"

    context = compile_financial_context(_document((_span(raw),)))

    fact = context.reporting_periods[0].dates[0]
    assert fact.raw_text == "令和８年３月３１日"
    assert fact.iso_date == "2026-03-31"
    assert fact.normalization_provenance is FactProvenance.NORMALIZED


def test_invalid_era_date_is_preserved_without_iso_guess():
    context = compile_financial_context(
        _document((_span("令和元年1月8日"),))
    )

    fact = context.reporting_dates[0]
    assert fact.raw_text == "令和元年1月8日"
    assert fact.iso_date is None
    assert fact.normalization_provenance is FactProvenance.AMBIGUOUS
    assert fact.issue


def test_date_after_era_end_is_not_converted_as_that_era():
    context = compile_financial_context(
        _document((_span("平成31年5月1日"),))
    )

    fact = context.reporting_dates[0]
    assert fact.iso_date is None
    assert fact.normalization_provenance is FactProvenance.AMBIGUOUS
    assert "after the 平成 era ended" in fact.issue


def test_unit_is_section_scoped_only_by_position_and_conflicts_are_retained():
    spans = (
        _span("連結貸借対照表", y=10),
        _span("（単位：千円）", y=20),
        _span("（単位：百万円）", y=30),
    )

    context = compile_financial_context(_document(spans))

    assert [unit.raw_declaration for unit in context.units] == [
        "（単位：千円）",
        "（単位：百万円）",
    ]
    assert [unit.normalized_unit for unit in context.units] == [
        FinancialUnit.THOUSAND_YEN,
        FinancialUnit.MILLION_YEN,
    ]
    assert all(unit.scope is UnitScope.STATEMENT_SECTION for unit in context.units)
    assert all(
        unit.association_provenance is FactProvenance.INFERRED
        and unit.associated_section_title == "連結貸借対照表"
        for unit in context.units
    )
    assert any("Conflicting unit declarations" in item.description for item in context.hypotheses)
    assert all(not hasattr(unit, "table_id") for unit in context.units)


def test_repeated_same_title_sections_do_not_create_false_unit_conflict():
    context = compile_financial_context(
        _document(
            (
                _span("連結貸借対照表", y=10),
                _span("（単位：千円）", y=20),
                _span("連結貸借対照表", y=40),
                _span("（単位：百万円）", y=50),
            )
        )
    )

    assert context.units[0].associated_section_evidence is not context.units[1].associated_section_evidence
    assert not any("Conflicting unit declarations" in item.description for item in context.hypotheses)


def test_document_level_unit_is_not_attached_to_a_later_statement():
    context = compile_financial_context(
        _document((_span("（単位：％）", y=10), _span("損益計算書", y=30)))
    )

    unit = context.units[0]
    assert unit.normalized_unit is FinancialUnit.PERCENT
    assert unit.scope is UnitScope.DOCUMENT
    assert unit.associated_section_title is None
    assert unit.association_provenance is None


@pytest.mark.parametrize(
    ("raw", "value"),
    [
        ("△2,500", "-2500"),
        ("▲２，５００", "-2500"),
        ("＋１２３．５", "123.5"),
        ("1,234", "1234"),
    ],
)
def test_supported_numeric_notation_preserves_raw_and_normalizes(raw, value):
    result = parse_numeric_value(raw)

    assert result.raw_text == raw
    assert str(result.normalized_value) == value
    assert result.normalized_text == value
    assert result.provenance is FactProvenance.NORMALIZED


@pytest.mark.parametrize("raw", ["0", "０", "△０"])
def test_explicit_zero_is_distinguished(raw):
    result = parse_numeric_value(raw)

    assert result.kind is NumericValueKind.ZERO
    assert result.normalized_value == 0


@pytest.mark.parametrize("raw", ["-", "−", "―", "ー", "－"])
def test_dash_variants_are_not_converted_to_zero(raw):
    result = parse_numeric_value(raw)

    assert result.kind is NumericValueKind.DASH
    assert result.normalized_value is None
    assert result.provenance is None


def test_blank_and_unsupported_notation_remain_uninterpreted():
    blank = parse_numeric_value("　 ")
    unsupported = parse_numeric_value("(1,000)")

    assert blank.kind is NumericValueKind.BLANK
    assert blank.normalized_value is None
    assert unsupported.kind is NumericValueKind.UNSUPPORTED
    assert unsupported.normalized_value is None
    assert unsupported.provenance is FactProvenance.AMBIGUOUS
    assert unsupported.issue


def test_footnote_explanation_and_section_association_retain_source_evidence():
    title = _span("損益計算書", y=10)
    footnote = _span("※1 減価償却費を含む", y=20)

    context = compile_financial_context(_document((title, footnote)))

    fact = context.footnotes[0]
    assert fact.raw_marker == "※1"
    assert fact.explanatory_text == "減価償却費を含む"
    assert fact.evidence.source_spans == (footnote,)
    assert fact.explanatory_evidence is fact.evidence
    assert fact.associated_section_title == "損益計算書"
    assert fact.association_provenance is FactProvenance.INFERRED


def test_footnote_neighbors_are_retained_without_claiming_explanation():
    before = _span("補足見出し", y=10)
    marker = _span("※", y=20)
    after = _span("翌行の説明候補", y=30)

    context = compile_financial_context(_document((before, marker, after)))

    fact = context.footnotes[0]
    assert [item.raw_text for item in fact.nearby_evidence] == [
        "補足見出し",
        "翌行の説明候補",
    ]
    assert fact.explanatory_text is None
    assert fact.explanatory_evidence is None


def test_unidentified_context_and_conflicting_period_dates_are_not_guessed():
    context = compile_financial_context(
        _document(
            (
                _span("前連結会計年度 2025年3月31日", y=10),
                _span("前連結会計年度 2024年3月31日", y=30),
            )
        )
    )

    assert context.document_type is DocumentType.UNKNOWN
    assert context.document_type_evidence == ()
    assert context.document_type_provenance is None
    assert context.statement_titles == ()
    assert len(context.reporting_periods) == 2
    assert [item.iso_date for item in context.reporting_dates] == [
        "2025-03-31",
        "2024-03-31",
    ]
    assert any("Conflicting explicit dates" in item.description for item in context.hypotheses)


def test_empty_document_context_is_valid_and_contains_no_invented_facts():
    context = compile_financial_context(_document(()))

    assert context.document_type is DocumentType.UNKNOWN
    assert context.document_evidence == ()
    assert context.statement_titles == ()
    assert context.reporting_periods == ()
    assert context.units == ()
    assert context.footnotes == ()


def test_ocr_extraction_source_is_preserved_in_context_evidence():
    ocr_span = _span(
        "※1 OCR由来の説明",
        source=ExtractionSource.OCR,
    )

    context = compile_financial_context(_document((ocr_span,)))

    assert context.footnotes[0].evidence.source_spans == (ocr_span,)
    assert context.footnotes[0].evidence.source_spans[0].extraction_source is ExtractionSource.OCR


def test_invalid_document_input_fails_clearly():
    with pytest.raises(TypeError, match="PDFDocument"):
        compile_financial_context(None)  # type: ignore[arg-type]


def test_percent_and_additional_yen_units_are_supported():
    context = compile_financial_context(
        _document(
            (
                _span("（単位：円）", y=10),
                _span("（単位：億円）", y=20),
                _span("（単位：パーセント）", y=30),
            )
        )
    )

    assert [unit.normalized_unit for unit in context.units] == [
        FinancialUnit.YEN,
        FinancialUnit.HUNDRED_MILLION_YEN,
        FinancialUnit.PERCENT,
    ]