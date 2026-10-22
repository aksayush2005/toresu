from pathlib import Path

import pytest

from toresu.context_compiler import compile_financial_context
from toresu.pdf_ingestion import (
    ExtractionSource,
    PDFDocument,
    PDFPage,
    TextSpan,
)
from toresu.table_reconstruction import (
    AssignmentStatus,
    CellRole,
    reconstruct_tables,
)


def _span(
    raw_text: str,
    x0: float,
    y0: float,
    x1: float,
    *,
    source: ExtractionSource = ExtractionSource.PDF_TEXT,
) -> TextSpan:
    return TextSpan(
        raw_text=raw_text,
        page_number=1,
        bounding_box=(x0, y0, x1, y0 + 10),
        extraction_source=source,
    )


def _statement(
    top: float = 10,
    *,
    missing_second_value: bool = False,
    irregular_spacing: bool = False,
    row_values: tuple[tuple[str, str], ...] | None = None,
) -> list[TextSpan]:
    spans = [
        _span("連結貸借対照表", 150, top, 360),
        _span("（単位：千円）", 180, top + 15, 320),
        _span("年度", 400, top + 30, 570),
        _span("前連結会計年度", 380, top + 45, 470),
        _span("当連結会計年度", 500, top + 45, 590),
    ]
    values = row_values or (("現金及び", "預金"), ("売上高", "営業利益"))
    row_ys = (top + 65, top + (100 if irregular_spacing else 85))
    for row_index, ((left_a, left_b), y) in enumerate(zip(values, row_ys)):
        spans.extend(
            [
                _span(left_a, 30, y, 105),
                _span(left_b, 108, y, 155),
                _span("1,000", 400, y, 450),
            ]
        )
        if not (missing_second_value and row_index == 1):
            spans.append(_span("2,000", 520, y, 570))
    return spans


def _document(spans: list[TextSpan], page_number: int = 1) -> PDFDocument:
    page_spans = tuple(
        TextSpan(
            span.raw_text,
            page_number,
            span.bounding_box,
            span.extraction_source,
            span.confidence,
            span.normalized_text,
        )
        for span in spans
    )
    page = PDFPage(
        page_number=page_number,
        bounding_box=(0.0, 0.0, 700.0, 900.0),
        text_spans=page_spans,
        native_text_usable=True,
        ocr_attempted=False,
    )
    return PDFDocument(Path("synthetic-financial.pdf"), (page,))


def _reconstruct(spans: list[TextSpan]):
    document = _document(spans)
    context = compile_financial_context(document)
    return reconstruct_tables(document, context)


def test_aligned_financial_statement_preserves_row_column_and_span_provenance():
    spans = _statement()

    result = _reconstruct(spans)

    assert len(result.tables) == 1
    table = result.tables[0]
    assert table.statement_title.raw_title == "連結貸借対照表"
    assert len(table.columns) == 2
    assert [column.reporting_period.raw_label for column in table.columns] == [
        "前連結会計年度",
        "当連結会計年度",
    ]
    assert len(table.rows) == 2
    first_row = table.rows[0]
    assert first_row.label_cell.raw_text == "現金及び 預金"
    assert first_row.cells[0].raw_text == "1,000"
    assert first_row.cells[0].numeric_interpretation.normalized_value == 1000
    assert first_row.cells[0].status is AssignmentStatus.RESOLVED
    assert first_row.cells[0].provenance.page_number == 1
    assert first_row.cells[0].provenance.extraction_sources == (ExtractionSource.PDF_TEXT,)
    assert first_row.cells[0].provenance.source_spans == (
        next(span for span in spans if span.raw_text == "1,000"),
    )
    assert first_row.cells[0].provenance.bounding_box == (400, 75, 450, 85)
    assert first_row.cells[0].cell_boundary == (400.0, 75, 450.0, 85)
    assert table.table_units[0].raw_declaration == "（単位：千円）"


def test_multi_level_and_merged_headers_preserve_header_paths():
    table = _reconstruct(_statement()).tables[0]

    assert len(table.header_rows) == 2
    assert [row[0].header_row for row in table.header_rows] == [0, 1]
    merged = table.header_rows[0][0]
    assert merged.raw_text == "年度"
    assert merged.is_merged
    assert merged.column_span == 2
    assert table.columns[0].header_path[0] == merged
    assert table.columns[1].header_path[0] == merged
    assert [header.raw_text for header in table.columns[0].header_path] == [
        "年度",
        "前連結会計年度",
    ]


def test_missing_value_becomes_unresolved_cell_without_fabricated_value():
    table = _reconstruct(_statement(missing_second_value=True)).tables[0]

    missing = table.rows[1].cells[1]
    assert missing.raw_text is None
    assert missing.provenance.source_spans == ()
    assert missing.provenance.bounding_box is None
    assert missing.cell_boundary == (520, 95, 570, 105)
    assert missing.status is AssignmentStatus.UNRESOLVED
    assert "No source span" in missing.issue


def test_aligned_dash_is_preserved_without_numeric_zero_conversion():
    spans = [
        _span("－", 520, span.bounding_box[1], 570)
        if span.raw_text == "2,000"
        else span
        for span in _statement()
    ]

    table = _reconstruct(spans).tables[0]
    dash = table.rows[0].cells[1]

    assert dash.raw_text == "－"
    assert dash.numeric_interpretation.kind.value == "dash"
    assert dash.numeric_interpretation.normalized_value is None


def test_multiple_spans_in_one_value_column_are_not_concatenated():
    spans = _statement()
    spans = [
        span
        for span in spans
        if not (span.raw_text == "1,000" and span.bounding_box[1] > 90)
    ]
    spans.extend(
        [
            _span("1,", 400, 95, 450),
            _span("000", 400, 95, 450),
        ]
    )

    table = _reconstruct(spans).tables[0]
    value = table.rows[-1].cells[0]

    assert value.raw_text == "1, 000"
    assert value.status is AssignmentStatus.AMBIGUOUS
    assert value.numeric_interpretation.normalized_value is None
    assert len(value.provenance.source_spans) == 2


def test_multiple_tables_on_one_page_are_separated_by_page_geometry():
    spans = _statement(top=10) + _statement(top=180)

    result = _reconstruct(spans)

    assert len(result.tables) == 2
    assert [table.table_id for table in result.tables] == ["p1-t1", "p1-t2"]
    assert result.tables[0].bounding_box[3] < result.tables[1].bounding_box[1]


def test_irregular_row_spacing_does_not_split_one_table():
    table = _reconstruct(_statement(irregular_spacing=True)).tables[0]

    assert len(table.rows) == 2
    assert table.rows[1].bounding_box[1] > table.rows[0].bounding_box[3]


def test_numeric_span_equidistant_between_columns_is_explicitly_ambiguous():
    spans = [
        _span("連結貸借対照表", 150, 10, 360),
        _span("A", 30, 25, 60),
        _span("1,000", 350, 25, 400),
        _span("B", 30, 45, 60),
        _span("1,100", 350, 45, 400),
        _span("C", 30, 65, 60),
        _span("1,200", 350, 65, 400),
        _span("2,000", 411, 25, 461),
        _span("2,100", 411, 45, 461),
        _span("2,200", 411, 65, 461),
        _span("1,500", 380.5, 85, 430.5),
    ]

    result = reconstruct_tables(
        _document(spans),
        compile_financial_context(_document(spans)),
        alignment_tolerance=31,
    )

    assert len(result.tables) == 1
    ambiguous = [
        cell
        for row in result.tables[0].rows
        for cell in row.cells
        if cell.status is AssignmentStatus.AMBIGUOUS
    ]
    assert ambiguous
    assert any(cell.raw_text == "1,500" for cell in ambiguous)
    assert any(cell.column_index is None for cell in ambiguous)


def test_unclassified_or_unassigned_spans_remain_available_with_provenance():
    source = _span("見出し補足", 20, 250, 100, source=ExtractionSource.OCR)
    spans = _statement() + [source]
    result = _reconstruct(spans)

    assert source in result.unassigned_spans
    assert source.extraction_source is ExtractionSource.OCR
    assert all(
        cell.role in {CellRole.VALUE, CellRole.ROW_LABEL}
        for row in result.tables[0].rows
        for cell in row.cells
    )


def test_invalid_options_raise_clear_errors():
    document = _document(_statement())
    context = compile_financial_context(document)

    with pytest.raises(ValueError, match="alignment_tolerance"):
        reconstruct_tables(document, context, alignment_tolerance=0)
