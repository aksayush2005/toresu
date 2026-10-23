from __future__ import annotations

import re
import statistics
import unicodedata
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Sequence

from .context_compiler import (
    FinancialDocumentContext,
    FinancialUnitFact,
    NumericInterpretation,
    NumericValueKind,
    ReportingPeriodFact,
    StatementTitleFact,
    parse_numeric_value,
)
from .pdf_ingestion import BoundingBox, ExtractionSource, PDFDocument, PDFPage, TextSpan


class AssignmentStatus(str, Enum):
    RESOLVED = "resolved"
    INFERRED = "inferred"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"


class CellRole(str, Enum):
    ROW_LABEL = "row_label"
    VALUE = "value"
    AUXILIARY = "auxiliary"
    HEADER = "header"
    UNCLASSIFIED = "unclassified"


@dataclass(frozen=True)
class CellProvenance:
    page_number: int
    bounding_box: BoundingBox | None
    extraction_sources: tuple[ExtractionSource, ...]
    source_spans: tuple[TextSpan, ...]


@dataclass(frozen=True)
class ReconstructedCell:
    cell_id: str
    role: CellRole
    raw_text: str | None
    provenance: CellProvenance
    cell_boundary: BoundingBox | None
    row_index: int | None
    column_index: int | None
    column_span: int
    numeric_interpretation: NumericInterpretation | None
    unit_fact: FinancialUnitFact | None
    unit_status: AssignmentStatus
    status: AssignmentStatus
    issue: str | None = None
    candidate_column_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class ReconstructedHeaderCell:
    header_id: str
    header_row: int
    raw_text: str
    column_start: int | None
    column_span: int
    is_merged: bool
    provenance: CellProvenance
    cell_boundary: BoundingBox
    status: AssignmentStatus
    issue: str | None = None


@dataclass(frozen=True)
class ReconstructedColumn:
    column_index: int
    left: float
    right: float
    center: float
    header_path: tuple[ReconstructedHeaderCell, ...]
    reporting_period: ReportingPeriodFact | None
    reporting_period_candidates: tuple[ReportingPeriodFact, ...]
    period_status: AssignmentStatus
    geometry_status: AssignmentStatus
    is_auxiliary: bool = False


@dataclass(frozen=True)
class ReconstructedRow:
    row_id: str
    row_index: int
    bounding_box: BoundingBox
    label_cell: ReconstructedCell | None
    cells: tuple[ReconstructedCell, ...]
    parent_row_id: str | None
    hierarchy_status: AssignmentStatus


@dataclass(frozen=True)
class ReconstructedTable:
    table_id: str
    page_number: int
    bounding_box: BoundingBox
    columns: tuple[ReconstructedColumn, ...]
    header_rows: tuple[tuple[ReconstructedHeaderCell, ...], ...]
    rows: tuple[ReconstructedRow, ...]
    table_units: tuple[FinancialUnitFact, ...]
    statement_title: StatementTitleFact | None
    context_spans: tuple[TextSpan, ...]
    status: AssignmentStatus
    issues: tuple[str, ...]


@dataclass(frozen=True)
class TableReconstructionResult:
    tables: tuple[ReconstructedTable, ...]
    unassigned_spans: tuple[TextSpan, ...]


@dataclass(frozen=True)
class _Line:
    page_number: int
    spans: tuple[TextSpan, ...]
    bounding_box: BoundingBox
    order: int

    @property
    def raw_text(self) -> str:
        return " ".join(span.raw_text.strip() for span in self.spans if span.raw_text.strip())


@dataclass(frozen=True)
class _ColumnAnchor:
    right_edge: float
    left: float
    right: float
    center: float
    supporting_lines: frozenset[int]
    is_auxiliary: bool = False


def reconstruct_tables(
    document: PDFDocument,
    context: FinancialDocumentContext,
    *,
    alignment_tolerance: float = 10.0,
    table_gap_threshold: float = 30.0,
) -> TableReconstructionResult:
    """Reconstruct geometrically supported tables without OCR or external inference."""
    if not isinstance(document, PDFDocument):
        raise TypeError("document must be a PDFDocument from pdf_ingestion.")
    if not isinstance(context, FinancialDocumentContext):
        raise TypeError("context must be a FinancialDocumentContext.")
    if alignment_tolerance <= 0:
        raise ValueError("alignment_tolerance must be greater than zero.")
    if table_gap_threshold <= 0:
        raise ValueError("table_gap_threshold must be greater than zero.")

    tables: list[ReconstructedTable] = []
    used_span_ids: set[int] = set()
    for page in document.pages:
        lines = _group_page_lines(page)
        blocks = _split_line_blocks(lines, table_gap_threshold)
        page_table_index = 0
        for block in blocks:
            candidate = _reconstruct_block(
                page,
                block,
                context,
                page_table_index,
                alignment_tolerance,
            )
            if candidate is None:
                continue
            tables.append(candidate)
            page_table_index += 1
            used_span_ids.update(id(span) for span in _table_spans(candidate))

    unassigned = tuple(
        span
        for page in document.pages
        for span in page.text_spans
        if id(span) not in used_span_ids
    )
    return TableReconstructionResult(tuple(tables), unassigned)


def _group_page_lines(page: PDFPage) -> list[_Line]:
    ordered = sorted(
        page.text_spans,
        key=lambda span: (
            (span.bounding_box[1] + span.bounding_box[3]) / 2,
            span.bounding_box[0],
        ),
    )
    groups: list[list[TextSpan]] = []
    for span in ordered:
        center_y = (span.bounding_box[1] + span.bounding_box[3]) / 2
        height = max(span.bounding_box[3] - span.bounding_box[1], 1.0)
        matches = []
        for group in groups:
            top = min(item.bounding_box[1] for item in group)
            bottom = max(item.bounding_box[3] for item in group)
            group_center = (top + bottom) / 2
            group_height = max(bottom - top, 1.0)
            tolerance = max(2.0, min(height, group_height) * 0.3)
            if abs(center_y - group_center) <= tolerance:
                matches.append((abs(center_y - group_center), group))
        if matches:
            min(matches, key=lambda item: item[0])[1].append(span)
        else:
            groups.append([span])

    lines = []
    for group in groups:
        group.sort(key=lambda span: span.bounding_box[0])
        lines.append(
            _Line(
                page_number=page.page_number,
                spans=tuple(group),
                bounding_box=_union_bbox([span.bounding_box for span in group]),
                order=0,
            )
        )
    lines.sort(key=lambda line: (line.bounding_box[1], line.bounding_box[0]))
    return [
        _Line(line.page_number, line.spans, line.bounding_box, index)
        for index, line in enumerate(lines)
    ]


def _split_line_blocks(lines: list[_Line], gap_threshold: float) -> list[list[_Line]]:
    if not lines:
        return []
    heights = [line.bounding_box[3] - line.bounding_box[1] for line in lines]
    threshold = max(gap_threshold, statistics.median(heights) * 2.5)
    blocks: list[list[_Line]] = [[lines[0]]]
    for line in lines[1:]:
        previous = blocks[-1][-1]
        if line.bounding_box[1] - previous.bounding_box[3] > threshold:
            blocks.append([line])
        else:
            blocks[-1].append(line)
    return blocks


def _reconstruct_block(
    page: PDFPage,
    block: list[_Line],
    context: FinancialDocumentContext,
    table_index: int,
    tolerance: float,
) -> ReconstructedTable | None:
    date_span_ids = {
        id(span)
        for date_fact in context.reporting_dates
        for span in date_fact.evidence.source_spans
    }
    period_header_span_ids = {
        id(span)
        for period in context.reporting_periods
        for span in period.evidence.source_spans
        if period.raw_label in span.raw_text
    }
    date_header_groups = _date_header_groups(context)
    raw_anchors = _numeric_anchors(block, tolerance, date_span_ids)
    if len(raw_anchors) < 2:
        raw_anchors = _add_period_header_anchors(
            raw_anchors, block, context, tolerance
        )
    first_numeric_line = min(
        (
            line_index
            for line_index, line in enumerate(block)
            if any(
                id(span) not in date_span_ids and _is_numeric_span(span)
                for span in line.spans
            )
        ),
        default=len(block),
    )
    anchors = [
        anchor
        for anchor in raw_anchors
        if len(anchor.supporting_lines) >= 2
        or _header_supports_anchor(block[:first_numeric_line], anchor, tolerance)
    ]
    if len(anchors) < 2:
        return None
    anchors.sort(key=lambda anchor: anchor.right_edge)

    assigned_numeric: dict[int, dict[int, list[TextSpan]]] = {}
    ambiguous_numeric: dict[int, list[tuple[TextSpan, tuple[int, ...]]]] = {}
    data_line_indexes: list[int] = []
    for line_index, line in enumerate(block):
        if line_index < first_numeric_line:
            continue
        per_column: dict[int, list[TextSpan]] = {}
        sign_markers = _sign_markers_by_value(line)
        marker_span_ids = {
            id(marker)
            for markers in sign_markers.values()
            for marker in markers
        }
        for span in line.spans:
            if id(span) in date_span_ids or id(span) in marker_span_ids:
                continue
            if not _looks_like_value_span(span, anchors, tolerance):
                continue
            candidates = _anchor_candidates(span, anchors, tolerance)
            if len(candidates) == 1:
                per_column.setdefault(candidates[0], []).extend(
                    (*sign_markers.get(id(span), ()), span)
                )
            elif len(candidates) > 1:
                ambiguous_numeric.setdefault(line_index, []).append((span, candidates))
        if per_column or line_index in ambiguous_numeric:
            assigned_numeric[line_index] = per_column
            data_line_indexes.append(line_index)
    if len(data_line_indexes) < 2:
        return None

    first_data_index = data_line_indexes[0]
    unit_facts, title_facts = _context_for_block(block, context)
    header_line_indexes = _detect_header_lines(
        block,
        first_data_index,
        anchors,
        context,
        unit_facts,
        title_facts,
        date_span_ids,
    )
    auxiliary_anchors, auxiliary_header_spans = _auxiliary_column_anchors(
        block,
        header_line_indexes,
        data_line_indexes,
        anchors,
        tolerance,
    )
    first_auxiliary_column = len(anchors)
    rightmost_value_edge = max(anchor.right for anchor in anchors)
    anchors.extend(auxiliary_anchors)
    assigned_auxiliary: dict[int, dict[int, list[TextSpan]]] = {}
    for line_index in data_line_indexes:
        line = block[line_index]
        for span in line.spans:
            if _is_numeric_span(span) or span.bounding_box[0] <= rightmost_value_edge:
                continue
            center = (span.bounding_box[0] + span.bounding_box[2]) / 2
            matching_headers = [
                (abs(center - header_center), offset)
                for offset, (header_center, header_tolerance) in enumerate(
                    auxiliary_header_spans
                )
                if abs(center - header_center) <= header_tolerance
            ]
            if matching_headers:
                _, auxiliary_offset = min(matching_headers)
                assigned_auxiliary.setdefault(line_index, {}).setdefault(
                    first_auxiliary_column + auxiliary_offset, []
                ).append(span)
    last_header_index = max(header_line_indexes, default=-1)
    last_data_index = data_line_indexes[-1]
    row_line_indexes = [
        line_index
        for line_index in range(last_header_index + 1, last_data_index + 1)
        if block[line_index].spans
    ]
    row_numeric: dict[int, dict[int, list[TextSpan]]] = {
        line_index: assigned_numeric.get(line_index, {})
        for line_index in row_line_indexes
    }
    row_ambiguous = {
        line_index: ambiguous_numeric[line_index]
        for line_index in row_line_indexes
        if line_index in ambiguous_numeric
    }
    table_id = f"p{page.page_number}-t{table_index + 1}"
    title_fact = title_facts[0] if len(title_facts) == 1 else None
    issues = []
    if len(title_facts) > 1:
        issues.append("Multiple statement-title candidates overlap this table block.")
    if len({unit.normalized_unit for unit in unit_facts}) > 1:
        issues.append("Conflicting unit candidates are retained without selecting one.")

    headers = _build_header_cells(
        block,
        header_line_indexes,
        anchors,
        unit_facts,
        title_facts,
        table_id,
        tolerance,
        date_span_ids,
        period_header_span_ids,
        date_header_groups,
    )
    columns = _build_columns(anchors, headers, context)
    rows = _build_rows(
        block,
        row_line_indexes,
        row_numeric,
        anchors,
        columns,
        unit_facts,
        table_id,
        tolerance,
        row_ambiguous,
        assigned_auxiliary,
    )
    contributing = [line.bounding_box for line in block]
    table_bbox = _union_bbox(contributing)
    contributing_spans = tuple(span for line in block for span in line.spans)
    return ReconstructedTable(
        table_id=table_id,
        page_number=page.page_number,
        bounding_box=table_bbox,
        columns=tuple(columns),
        header_rows=tuple(
            tuple(header for header in headers if header.header_row == index)
            for index in sorted({header.header_row for header in headers})
        ),
        rows=tuple(rows),
        table_units=unit_facts,
        statement_title=title_fact,
        context_spans=contributing_spans,
        status=(
            AssignmentStatus.AMBIGUOUS if issues else AssignmentStatus.INFERRED
        ),
        issues=tuple(issues),
    )


def _numeric_anchors(
    lines: list[_Line],
    tolerance: float,
    excluded_span_ids: set[int] | None = None,
) -> list[_ColumnAnchor]:
    excluded_span_ids = excluded_span_ids or set()
    observations = []
    for line_index, line in enumerate(lines):
        for span in line.spans:
            if id(span) not in excluded_span_ids and _is_numeric_span(span):
                observations.append((span.bounding_box[2], line_index, span))
    observations.sort(key=lambda item: (item[0], item[1], item[2].bounding_box[0]))
    groups: list[list[tuple[float, int, TextSpan]]] = []
    for observation in observations:
        matches = [
            (abs(observation[0] - statistics.median(item[0] for item in group)), group)
            for group in groups
            if abs(observation[0] - statistics.median(item[0] for item in group)) <= tolerance
        ]
        if matches:
            min(matches, key=lambda item: item[0])[1].append(observation)
        else:
            groups.append([observation])
    anchors = []
    for group in groups:
        spans = [item[2] for item in group]
        anchors.append(
            _ColumnAnchor(
                right_edge=statistics.median(item[0] for item in group),
                left=statistics.median(span.bounding_box[0] for span in spans),
                right=statistics.median(span.bounding_box[2] for span in spans),
                center=(
                    statistics.median(span.bounding_box[0] for span in spans)
                    + statistics.median(span.bounding_box[2] for span in spans)
                )
                / 2,
                supporting_lines=frozenset(item[1] for item in group),
            )
        )
    return anchors


def _sign_markers_by_value(line: _Line) -> dict[int, tuple[TextSpan, ...]]:
    markers: dict[int, tuple[TextSpan, ...]] = {}
    for marker in line.spans:
        if marker.raw_text.strip() not in {"△", "▲"}:
            continue
        candidates = [
            value
            for value in line.spans
            if _is_numeric_span(value)
            and -1.0
            <= value.bounding_box[0] - marker.bounding_box[2]
            <= 2.0
            and abs(
                (value.bounding_box[1] + value.bounding_box[3]) / 2
                - (marker.bounding_box[1] + marker.bounding_box[3]) / 2
            )
            <= 2.0
        ]
        if candidates:
            value = min(candidates, key=lambda item: item.bounding_box[0])
            markers[id(value)] = (*markers.get(id(value), ()), marker)
    return markers


def _date_header_groups(
    context: FinancialDocumentContext,
) -> tuple[tuple[str, str | None, tuple[TextSpan, ...]], ...]:
    groups = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    line_positions: dict[tuple[int, ...], int] = {}
    for date_fact in context.reporting_dates:
        source_spans = date_fact.evidence.source_spans
        span_key = tuple(id(span) for span in source_spans)
        compact_parts = [
            "".join(
                character
                for character in unicodedata.normalize("NFKC", span.raw_text)
                if not character.isspace()
            )
            for span in source_spans
        ]
        compact_line = "".join(compact_parts)
        compact_date = "".join(
            character
            for character in unicodedata.normalize("NFKC", date_fact.raw_text)
            if not character.isspace()
        )
        start = compact_line.find(compact_date, line_positions.get(span_key, 0))
        if start < 0:
            continue
        end = start + len(compact_date)
        line_positions[span_key] = end
        selected = []
        offset = 0
        for span, part in zip(source_spans, compact_parts):
            part_end = offset + len(part)
            if offset < end and part_end > start:
                selected.append(span)
            offset = part_end
        if selected:
            selected_key = tuple(id(span) for span in selected)
            group_key = (date_fact.raw_text.strip(), selected_key)
            if group_key in seen:
                continue
            seen.add(group_key)
            compact_date_position = start
            matching_periods = []
            for period in context.reporting_periods:
                period_span_ids = {
                    id(span) for span in period.evidence.source_spans
                }
                if not period_span_ids.intersection(span_key):
                    continue
                compact_label = "".join(
                    character
                    for character in unicodedata.normalize("NFKC", period.raw_label)
                    if not character.isspace()
                )
                label_position = compact_line.find(compact_label)
                if label_position >= 0 and label_position <= compact_date_position:
                    matching_periods.append((label_position, period.raw_label))
            period_label = (
                max(matching_periods, default=(0, None), key=lambda item: item[0])[1]
            )
            groups.append(
                (date_fact.raw_text.strip(), period_label, tuple(selected))
            )
    return tuple(groups)


def _auxiliary_column_anchors(
    block: list[_Line],
    header_line_indexes: list[int],
    data_line_indexes: list[int],
    numeric_anchors: list[_ColumnAnchor],
    tolerance: float,
) -> tuple[list[_ColumnAnchor], list[tuple[float, float]]]:
    """Find text columns with a header and row cells aligned beyond value columns."""
    if not numeric_anchors:
        return [], []
    rightmost_value_edge = max(anchor.right for anchor in numeric_anchors)
    candidates = []
    for line_index in header_line_indexes:
        for span in block[line_index].spans:
            if span.bounding_box[0] <= rightmost_value_edge:
                continue
            center = (span.bounding_box[0] + span.bounding_box[2]) / 2
            width = span.bounding_box[2] - span.bounding_box[0]
            matching_spans = []
            for data_line_index in data_line_indexes:
                for cell_span in block[data_line_index].spans:
                    if _is_numeric_span(cell_span):
                        continue
                    cell_center = (
                        cell_span.bounding_box[0] + cell_span.bounding_box[2]
                    ) / 2
                    if (
                        cell_span.bounding_box[0] > rightmost_value_edge
                        and abs(cell_center - center)
                        <= max(tolerance, width * 0.5)
                    ):
                        matching_spans.append((data_line_index, cell_span))
            if matching_spans:
                candidates.append((span, center, width, matching_spans))

    anchors = []
    headers = []
    for span, center, width, matching_spans in candidates:
        matched = [item[1] for item in matching_spans]
        anchors.append(
            _ColumnAnchor(
                right_edge=statistics.median(item.bounding_box[2] for item in matched),
                left=statistics.median(item.bounding_box[0] for item in matched),
                right=statistics.median(item.bounding_box[2] for item in matched),
                center=statistics.median(
                    (item.bounding_box[0] + item.bounding_box[2]) / 2
                    for item in matched
                ),
                supporting_lines=frozenset(item[0] for item in matching_spans),
                is_auxiliary=True,
            )
        )
        headers.append((center, max(tolerance, width * 0.5)))
    return anchors, headers


def _add_period_header_anchors(
    anchors: list[_ColumnAnchor],
    lines: list[_Line],
    context: FinancialDocumentContext,
    tolerance: float,
) -> list[_ColumnAnchor]:
    augmented = list(anchors)
    line_by_span = {
        id(span): line_index
        for line_index, line in enumerate(lines)
        for span in line.spans
    }
    for period in context.reporting_periods:
        for span in period.evidence.source_spans:
            line_index = line_by_span.get(id(span))
            if line_index is None or period.raw_label not in span.raw_text:
                continue
            center = (span.bounding_box[0] + span.bounding_box[2]) / 2
            if any(abs(anchor.center - center) <= tolerance for anchor in augmented):
                continue
            augmented.append(
                _ColumnAnchor(
                    right_edge=span.bounding_box[2],
                    left=span.bounding_box[0],
                    right=span.bounding_box[2],
                    center=center,
                    supporting_lines=frozenset({line_index}),
                )
            )
    return augmented


def _is_numeric_span(span: TextSpan) -> bool:
    return _numeric_kind(span) in {
        NumericValueKind.NUMBER,
        NumericValueKind.ZERO,
    }


def _numeric_kind(span: TextSpan) -> NumericValueKind:
    candidate = re.sub(r"\s+", "", unicodedata.normalize("NFKC", span.raw_text))
    return parse_numeric_value(candidate).kind


def _looks_like_value_span(
    span: TextSpan,
    anchors: list[_ColumnAnchor],
    tolerance: float,
) -> bool:
    kind = _numeric_kind(span)
    if kind in {NumericValueKind.BLANK}:
        return False
    return bool(_anchor_candidates(span, anchors, tolerance))


def _anchor_candidates(
    span: TextSpan,
    anchors: list[_ColumnAnchor],
    tolerance: float,
) -> tuple[int, ...]:
    right_edge = span.bounding_box[2]
    center = (span.bounding_box[0] + span.bounding_box[2]) / 2
    return tuple(
        index
        for index, anchor in enumerate(anchors)
        if abs(anchor.right_edge - right_edge) <= tolerance
        or anchor.left - tolerance <= center <= anchor.right + tolerance
    )


def _header_supports_anchor(
    header_lines: list[_Line],
    anchor: _ColumnAnchor,
    tolerance: float,
) -> bool:
    for line in header_lines:
        for span in line.spans:
            if _is_numeric_span(span):
                continue
            center = (span.bounding_box[0] + span.bounding_box[2]) / 2
            width = span.bounding_box[2] - span.bounding_box[0]
            if abs(center - anchor.center) <= max(tolerance, width * 0.25):
                return True
    return False


def _detect_header_lines(
    block: list[_Line],
    first_data_index: int,
    anchors: list[_ColumnAnchor],
    context: FinancialDocumentContext,
    units: tuple[FinancialUnitFact, ...],
    titles: tuple[StatementTitleFact, ...],
    date_span_ids: set[int],
) -> list[int]:
    excluded_ids = {
        id(span)
        for fact in (*units, *titles)
        for span in fact.evidence.source_spans
    }
    period_labels = {period.raw_label for period in context.reporting_periods}
    headers = []
    for line_index in range(first_data_index):
        line = block[line_index]
        if any(id(span) in date_span_ids for span in line.spans):
            headers.append(line_index)
            continue
        content_spans = [span for span in line.spans if id(span) not in excluded_ids]
        if not content_spans:
            continue
        has_period_label = any(
            label in line.raw_text for label in period_labels if label
        )
        occupied_columns = set()
        has_merged_coverage = False
        for span in content_spans:
            covered = [
                column_index
                for column_index, anchor in enumerate(anchors)
                if span.bounding_box[0] <= anchor.center <= span.bounding_box[2]
            ]
            if len(covered) >= 2:
                has_merged_coverage = True
            occupied_columns.update(covered)
        if has_period_label or has_merged_coverage or len(occupied_columns) >= 2:
            headers.append(line_index)
    return headers


def _build_header_cells(
    block: list[_Line],
    header_line_indexes: list[int],
    anchors: list[_ColumnAnchor],
    units: tuple[FinancialUnitFact, ...],
    titles: tuple[StatementTitleFact, ...],
    table_id: str,
    tolerance: float,
    date_span_ids: set[int],
    period_header_span_ids: set[int],
    date_header_groups: tuple[tuple[str, str | None, tuple[TextSpan, ...]], ...],
) -> list[ReconstructedHeaderCell]:
    excluded_span_ids = {
        id(span)
        for fact in (*units, *titles)
        for span in fact.evidence.source_spans
    }
    header_lines = [
        tuple(
            span
            for span in block[line_index].spans
            if id(span) not in excluded_span_ids and span.raw_text.strip()
        )
        for line_index in header_line_indexes
    ]
    header_lines = [line_spans for line_spans in header_lines if line_spans]
    headers = []
    for header_row, spans in enumerate(header_lines):
        grouped_spans: dict[int, list[TextSpan]] = {}
        merged_headers = []
        unresolved_headers = []
        current_span_ids = {id(span) for span in spans}
        shared_span_groups = [
            group
            for group in date_header_groups
            if any(id(span) in current_span_ids for span in group[2])
        ]
        for date_index, (raw_date, period_label, date_spans) in enumerate(
            date_header_groups
        ):
            selected_date_spans = tuple(
                span for span in date_spans if id(span) in current_span_ids
            )
            if not selected_date_spans:
                continue
            date_bbox = _union_bbox(
                [span.bounding_box for span in selected_date_spans]
            )
            center_x = (date_bbox[0] + date_bbox[2]) / 2
            groups_for_same_span = [
                group
                for group in shared_span_groups
                if set(id(span) for span in group[2])
                == set(id(span) for span in selected_date_spans)
            ]
            if (
                len(groups_for_same_span) == len(anchors)
                and len(groups_for_same_span) > 1
            ):
                column_index = groups_for_same_span.index(
                    (raw_date, period_label, date_spans)
                )
            else:
                column_index = min(
                    range(len(anchors)),
                    key=lambda item: abs(anchors[item].center - center_x),
                )
            if period_label:
                headers.append(
                    ReconstructedHeaderCell(
                        header_id=f"{table_id}-h{header_row + 1}-p{date_index + 1}",
                        header_row=header_row,
                        raw_text=period_label,
                        column_start=column_index,
                        column_span=1,
                        is_merged=False,
                        provenance=_cell_provenance(selected_date_spans),
                        cell_boundary=date_bbox,
                        status=AssignmentStatus.INFERRED,
                    )
                )
            headers.append(
                ReconstructedHeaderCell(
                    header_id=f"{table_id}-h{header_row + 1}-d{date_index + 1}",
                    header_row=header_row,
                    raw_text=raw_date,
                    column_start=column_index,
                    column_span=1,
                    is_merged=False,
                    provenance=_cell_provenance(selected_date_spans),
                    cell_boundary=date_bbox,
                    status=AssignmentStatus.INFERRED,
                )
            )
        for span in spans:
            if id(span) in date_span_ids:
                continue
            covered = [
                column_index
                for column_index, anchor in enumerate(anchors)
                if span.bounding_box[0] <= anchor.center <= span.bounding_box[2]
            ]
            if len(covered) >= 2:
                merged_headers.append((span, min(covered), max(covered) - min(covered) + 1))
                continue

            if id(span) in period_header_span_ids:
                center_x = (span.bounding_box[0] + span.bounding_box[2]) / 2
                column_index = min(
                    range(len(anchors)),
                    key=lambda item: min(
                        abs(anchors[item].center - center_x),
                        abs(anchors[item].right_edge - span.bounding_box[2]),
                    ),
                )
                anchor = anchors[column_index]
                alignment_tolerance = max(tolerance, anchor.right - anchor.left)
                if (
                    abs(anchor.center - center_x) <= alignment_tolerance
                    or abs(anchor.right_edge - span.bounding_box[2])
                    <= alignment_tolerance
                ):
                    grouped_spans.setdefault(column_index, []).append(span)
                else:
                    unresolved_headers.append(span)
                continue

            center_x = (span.bounding_box[0] + span.bounding_box[2]) / 2
            nearest = min(
                range(len(anchors)),
                key=lambda item: abs(anchors[item].center - center_x),
            )
            if abs(anchors[nearest].center - center_x) <= max(
                tolerance, (anchors[nearest].right - anchors[nearest].left)
            ):
                grouped_spans.setdefault(nearest, []).append(span)
            else:
                unresolved_headers.append(span)

        for column_index, grouped in grouped_spans.items():
            bbox = _union_bbox([span.bounding_box for span in grouped])
            headers.append(
                ReconstructedHeaderCell(
                    header_id=f"{table_id}-h{header_row + 1}-c{column_index + 1}",
                    header_row=header_row,
                    raw_text="".join(span.raw_text for span in grouped).strip(),
                    column_start=column_index,
                    column_span=1,
                    is_merged=False,
                    provenance=_cell_provenance(tuple(grouped)),
                    cell_boundary=bbox,
                    status=AssignmentStatus.INFERRED,
                )
            )
        for span_index, (span, column_start, column_span) in enumerate(merged_headers):
            headers.append(
                ReconstructedHeaderCell(
                    header_id=f"{table_id}-h{header_row + 1}-m{span_index + 1}",
                    header_row=header_row,
                    raw_text=span.raw_text,
                    column_start=column_start,
                    column_span=column_span,
                    is_merged=True,
                    provenance=_cell_provenance((span,)),
                    cell_boundary=span.bounding_box,
                    status=AssignmentStatus.INFERRED,
                    issue="Header span geometrically covers multiple value columns.",
                )
            )
        for span_index, span in enumerate(unresolved_headers):
            headers.append(
                ReconstructedHeaderCell(
                    header_id=f"{table_id}-h{header_row + 1}-u{span_index + 1}",
                    header_row=header_row,
                    raw_text=span.raw_text,
                    column_start=None,
                    column_span=1,
                    is_merged=False,
                    provenance=_cell_provenance((span,)),
                    cell_boundary=span.bounding_box,
                    status=AssignmentStatus.UNRESOLVED,
                    issue="Header text has no supported alignment with a value column.",
                )
            )
    return headers


def _build_columns(
    anchors: list[_ColumnAnchor],
    headers: list[ReconstructedHeaderCell],
    context: FinancialDocumentContext,
) -> list[ReconstructedColumn]:
    columns = []
    for column_index, anchor in enumerate(anchors):
        path = tuple(
            header
            for header in headers
            if header.column_start is not None
            and header.column_start <= column_index < header.column_start + header.column_span
        )
        period_candidates = []
        header_text = " ".join(header.raw_text for header in path)
        header_span_ids = {
            id(span)
            for header in path
            for span in header.provenance.source_spans
        }
        for period in context.reporting_periods:
            period_span_ids = {id(span) for span in period.evidence.source_spans}
            if period.raw_label in header_text and period_span_ids & header_span_ids:
                period_candidates.append(period)
        period_status = (
            AssignmentStatus.RESOLVED
            if len(period_candidates) == 1
            else AssignmentStatus.AMBIGUOUS
            if len(period_candidates) > 1
            else AssignmentStatus.UNRESOLVED
        )
        columns.append(
            ReconstructedColumn(
                column_index=column_index,
                left=anchor.left,
                right=anchor.right,
                center=anchor.center,
                header_path=path,
                reporting_period=(
                    period_candidates[0] if len(period_candidates) == 1 else None
                ),
                reporting_period_candidates=tuple(period_candidates),
                period_status=period_status,
                geometry_status=AssignmentStatus.INFERRED,
                is_auxiliary=anchor.is_auxiliary,
            )
        )
    return columns


def _build_rows(
    block: list[_Line],
    data_line_indexes: list[int],
    assigned_numeric: dict[int, dict[int, list[TextSpan]]],
    anchors: list[_ColumnAnchor],
    columns: list[ReconstructedColumn],
    units: tuple[FinancialUnitFact, ...],
    table_id: str,
    tolerance: float,
    ambiguous_numeric: dict[int, list[tuple[TextSpan, tuple[int, ...]]]],
    assigned_auxiliary: dict[int, dict[int, list[TextSpan]]],
) -> list[ReconstructedRow]:
    rows: list[ReconstructedRow] = []
    hierarchy_stack: list[tuple[float, str]] = []
    first_value_left = min(anchor.left for anchor in anchors)
    single_unit = units[0] if len(units) == 1 else None

    for row_index, line_index in enumerate(data_line_indexes):
        line = block[line_index]
        numeric_by_col = assigned_numeric[line_index]
        numeric_span_ids = {
            id(span) for spans in numeric_by_col.values() for span in spans
        }
        label_spans = tuple(
            span
            for span in line.spans
            if id(span) not in numeric_span_ids
            and span.bounding_box[2] <= first_value_left - tolerance * 0.25
        )
        row_id = f"{table_id}-r{row_index + 1}"
        label_cell = None
        parent_row_id = None
        hierarchy_status = AssignmentStatus.UNRESOLVED
        if label_spans:
            label_cell = ReconstructedCell(
                cell_id=f"{row_id}-label",
                role=CellRole.ROW_LABEL,
                raw_text=_joined_raw_text(label_spans),
                provenance=_cell_provenance(label_spans),
                cell_boundary=_union_bbox([span.bounding_box for span in label_spans]),
                row_index=row_index,
                column_index=None,
                column_span=1,
                numeric_interpretation=None,
                unit_fact=None,
                unit_status=AssignmentStatus.UNRESOLVED,
                status=AssignmentStatus.RESOLVED,
            )
            indent = min(span.bounding_box[0] for span in label_spans)
            while hierarchy_stack and indent <= hierarchy_stack[-1][0] + tolerance:
                hierarchy_stack.pop()
            if hierarchy_stack:
                parent_row_id = hierarchy_stack[-1][1]
                hierarchy_status = AssignmentStatus.INFERRED
            hierarchy_stack.append((indent, row_id))

        cells = []
        for column_index, anchor in enumerate(anchors):
            spans = tuple(
                assigned_auxiliary.get(line_index, {}).get(column_index, ())
                if anchor.is_auxiliary
                else numeric_by_col.get(column_index, ())
            )
            if not spans:
                column = columns[column_index]
                cell_box = (
                    column.left,
                    line.bounding_box[1],
                    column.right,
                    line.bounding_box[3],
                )
                cells.append(
                    ReconstructedCell(
                        cell_id=f"{row_id}-c{column_index + 1}",
                        role=(
                            CellRole.AUXILIARY
                            if anchor.is_auxiliary
                            else CellRole.VALUE
                        ),
                        raw_text=None,
                        provenance=CellProvenance(
                            page_number=line.page_number,
                            bounding_box=None,
                            extraction_sources=(),
                            source_spans=(),
                        ),
                        cell_boundary=cell_box,
                        row_index=row_index,
                        column_index=column_index,
                        column_span=1,
                        numeric_interpretation=None,
                        unit_fact=single_unit,
                        unit_status=(
                            AssignmentStatus.INFERRED
                            if single_unit is not None
                            else AssignmentStatus.AMBIGUOUS
                            if units
                            else AssignmentStatus.UNRESOLVED
                        ),
                        status=AssignmentStatus.UNRESOLVED,
                        issue="No source span supports a value in this row/column position.",
                    )
                )
                continue

            raw_text = _joined_raw_text(spans)
            if anchor.is_auxiliary:
                cells.append(
                    ReconstructedCell(
                        cell_id=f"{row_id}-c{column_index + 1}",
                        role=CellRole.AUXILIARY,
                        raw_text=raw_text,
                        provenance=_cell_provenance(spans),
                        cell_boundary=(
                            columns[column_index].left,
                            line.bounding_box[1],
                            columns[column_index].right,
                            line.bounding_box[3],
                        ),
                        row_index=row_index,
                        column_index=column_index,
                        column_span=1,
                        numeric_interpretation=None,
                        unit_fact=None,
                        unit_status=AssignmentStatus.UNRESOLVED,
                        status=AssignmentStatus.RESOLVED,
                    )
                )
                continue
            if len(spans) == 1:
                interpretation = parse_numeric_value(spans[0].raw_text.strip())
            elif (
                len(spans) == 2
                and spans[0].raw_text.strip() in {"△", "▲"}
                and _is_numeric_span(spans[1])
            ):
                raw_text = spans[0].raw_text.strip() + spans[1].raw_text.strip()
                interpretation = parse_numeric_value(raw_text)
            else:
                interpretation = NumericInterpretation(
                    raw_text=raw_text,
                    normalized_text=None,
                    normalized_value=None,
                    kind=NumericValueKind.UNSUPPORTED,
                    provenance=None,
                    issue="Multiple text spans share a value column; numeric concatenation is unresolved.",
                )
            status = (
                AssignmentStatus.AMBIGUOUS
                if interpretation.kind is NumericValueKind.UNSUPPORTED
                else AssignmentStatus.RESOLVED
            )
            cells.append(
                ReconstructedCell(
                    cell_id=f"{row_id}-c{column_index + 1}",
                    role=CellRole.VALUE,
                    raw_text=raw_text,
                    provenance=_cell_provenance(spans),
                    cell_boundary=(
                        columns[column_index].left,
                        line.bounding_box[1],
                        columns[column_index].right,
                        line.bounding_box[3],
                    ),
                    row_index=row_index,
                    column_index=column_index,
                    column_span=1,
                    numeric_interpretation=interpretation,
                    unit_fact=single_unit,
                    unit_status=(
                        AssignmentStatus.INFERRED
                        if single_unit is not None
                        else AssignmentStatus.AMBIGUOUS
                        if units
                        else AssignmentStatus.UNRESOLVED
                    ),
                    status=status,
                    issue=(
                        interpretation.issue
                        if status is AssignmentStatus.AMBIGUOUS
                        else None
                    ),
                )
            )

        for ambiguous_index, (span, candidates) in enumerate(
            ambiguous_numeric.get(line_index, ())
        ):
            interpretation = parse_numeric_value(span.raw_text.strip())
            cells.append(
                ReconstructedCell(
                    cell_id=f"{row_id}-ambiguous{ambiguous_index + 1}",
                    role=CellRole.VALUE,
                    raw_text=span.raw_text,
                    provenance=_cell_provenance((span,)),
                    cell_boundary=span.bounding_box,
                    row_index=row_index,
                    column_index=None,
                    column_span=1,
                    numeric_interpretation=interpretation,
                    unit_fact=single_unit,
                    unit_status=(
                        AssignmentStatus.INFERRED
                        if single_unit is not None
                        else AssignmentStatus.AMBIGUOUS
                        if units
                        else AssignmentStatus.UNRESOLVED
                    ),
                    status=AssignmentStatus.AMBIGUOUS,
                    issue="Value is within alignment tolerance of multiple columns.",
                    candidate_column_indices=candidates,
                )
            )

        row_bbox = _union_bbox([line.bounding_box])
        rows.append(
            ReconstructedRow(
                row_id=row_id,
                row_index=row_index,
                bounding_box=row_bbox,
                label_cell=label_cell,
                cells=tuple(cells),
                parent_row_id=parent_row_id,
                hierarchy_status=hierarchy_status,
            )
        )
    return rows


def _context_for_block(
    block: list[_Line],
    context: FinancialDocumentContext,
) -> tuple[tuple[FinancialUnitFact, ...], tuple[StatementTitleFact, ...]]:
    block_span_ids = {id(span) for line in block for span in line.spans}
    units = tuple(
        unit
        for unit in context.units
        if {id(span) for span in unit.evidence.source_spans} & block_span_ids
    )
    titles = tuple(
        title
        for title in context.statement_titles
        if {id(span) for span in title.evidence.source_spans} & block_span_ids
    )
    return units, titles


def _cell_provenance(spans: Sequence[TextSpan]) -> CellProvenance:
    if not spans:
        raise ValueError("Cell provenance requires at least one source span.")
    page_numbers = {span.page_number for span in spans}
    if len(page_numbers) != 1:
        raise ValueError("A reconstructed cell cannot span multiple PDF pages.")
    return CellProvenance(
        page_number=next(iter(page_numbers)),
        bounding_box=_union_bbox([span.bounding_box for span in spans]),
        extraction_sources=tuple(dict.fromkeys(span.extraction_source for span in spans)),
        source_spans=tuple(spans),
    )


def _joined_raw_text(spans: Sequence[TextSpan]) -> str:
    return " ".join(span.raw_text.strip() for span in spans if span.raw_text.strip())


def _union_bbox(boxes: Sequence[BoundingBox]) -> BoundingBox:
    if not boxes:
        raise ValueError("At least one bounding box is required.")
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _table_spans(table: ReconstructedTable) -> tuple[TextSpan, ...]:
    spans = list(table.context_spans)
    for row in table.rows:
        if row.label_cell:
            spans.extend(row.label_cell.provenance.source_spans)
        for cell in row.cells:
            spans.extend(cell.provenance.source_spans)
    return tuple(dict.fromkeys(spans))