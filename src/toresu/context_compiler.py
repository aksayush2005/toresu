from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import Enum

from .pdf_ingestion import (
    BoundingBox,
    ExtractionSource,
    PDFDocument,
    PDFPage,
    TextSpan,
)


class FactProvenance(str, Enum):
    EXPLICIT = "explicit"
    NORMALIZED = "normalized"
    INFERRED = "inferred"
    AMBIGUOUS = "ambiguous"


class DocumentType(str, Enum):
    JAPANESE_FINANCIAL_STATEMENTS = "japanese_financial_statements"
    UNKNOWN = "unknown"


class StatementType(str, Enum):
    CONSOLIDATED_BALANCE_SHEET = "consolidated_balance_sheet"
    BALANCE_SHEET = "balance_sheet"
    CONSOLIDATED_INCOME_STATEMENT = "consolidated_income_statement"
    INCOME_STATEMENT = "income_statement"
    CONSOLIDATED_CASH_FLOW_STATEMENT = "consolidated_cash_flow_statement"
    CASH_FLOW_STATEMENT = "cash_flow_statement"
    CONSOLIDATED_STATEMENT_OF_CHANGES_IN_EQUITY = "consolidated_statement_of_changes_in_equity"
    STATEMENT_OF_CHANGES_IN_EQUITY = "statement_of_changes_in_equity"


class FinancialUnit(str, Enum):
    YEN = "yen"
    THOUSAND_YEN = "thousand_yen"
    MILLION_YEN = "million_yen"
    HUNDRED_MILLION_YEN = "hundred_million_yen"
    PERCENT = "percent"


class NumericValueKind(str, Enum):
    NUMBER = "number"
    ZERO = "zero"
    DASH = "dash"
    BLANK = "blank"
    UNSUPPORTED = "unsupported"


class UnitScope(str, Enum):
    STATEMENT_SECTION = "statement_section"
    DOCUMENT = "document"


@dataclass(frozen=True)
class EvidenceText:
    """A text excerpt linked to the original ingestion spans that supplied it."""

    raw_text: str
    source_spans: tuple[TextSpan, ...]

    @property
    def page_numbers(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys(span.page_number for span in self.source_spans))

    @property
    def bounding_box(self) -> BoundingBox | None:
        if not self.source_spans:
            return None
        return (
            min(span.bounding_box[0] for span in self.source_spans),
            min(span.bounding_box[1] for span in self.source_spans),
            max(span.bounding_box[2] for span in self.source_spans),
            max(span.bounding_box[3] for span in self.source_spans),
        )


@dataclass(frozen=True)
class StatementTitleFact:
    raw_title: str
    statement_type: StatementType
    evidence: EvidenceText
    extraction_provenance: FactProvenance = FactProvenance.EXPLICIT
    interpretation_provenance: FactProvenance = FactProvenance.NORMALIZED


@dataclass(frozen=True)
class SectionHeadingFact:
    raw_heading: str
    evidence: EvidenceText
    provenance: FactProvenance = FactProvenance.EXPLICIT


@dataclass(frozen=True)
class StatementSectionContext:
    title: StatementTitleFact
    headings: tuple[SectionHeadingFact, ...]
    nearby_evidence: tuple[EvidenceText, ...]
    section_evidence: tuple[EvidenceText, ...]


@dataclass(frozen=True)
class ReportingDateFact:
    raw_text: str
    iso_date: str | None
    evidence: EvidenceText
    normalization_provenance: FactProvenance
    associated_period_label: str | None = None
    association_provenance: FactProvenance | None = None
    issue: str | None = None


@dataclass(frozen=True)
class ReportingPeriodFact:
    raw_label: str
    evidence: EvidenceText
    dates: tuple[ReportingDateFact, ...]
    provenance: FactProvenance = FactProvenance.EXPLICIT


@dataclass(frozen=True)
class FinancialUnitFact:
    raw_declaration: str
    normalized_unit: FinancialUnit
    evidence: EvidenceText
    scope: UnitScope
    associated_section_title: str | None
    associated_section_evidence: EvidenceText | None
    association_provenance: FactProvenance | None
    normalization_provenance: FactProvenance = FactProvenance.NORMALIZED


@dataclass(frozen=True)
class FootnoteFact:
    raw_marker: str
    evidence: EvidenceText
    nearby_evidence: tuple[EvidenceText, ...]
    explanatory_text: str | None
    explanatory_evidence: EvidenceText | None
    associated_section_title: str | None
    association_provenance: FactProvenance | None


@dataclass(frozen=True)
class ContextHypothesis:
    description: str
    evidence: tuple[EvidenceText, ...]
    provenance: FactProvenance = FactProvenance.AMBIGUOUS


@dataclass(frozen=True)
class NumericInterpretation:
    raw_text: str
    normalized_text: str | None
    normalized_value: Decimal | None
    kind: NumericValueKind
    provenance: FactProvenance | None
    issue: str | None = None


@dataclass(frozen=True)
class FinancialDocumentContext:
    document_type: DocumentType
    document_type_evidence: tuple[EvidenceText, ...]
    document_type_provenance: FactProvenance | None
    document_evidence: tuple[EvidenceText, ...]
    statement_titles: tuple[StatementTitleFact, ...]
    statement_sections: tuple[StatementSectionContext, ...]
    reporting_periods: tuple[ReportingPeriodFact, ...]
    reporting_dates: tuple[ReportingDateFact, ...]
    units: tuple[FinancialUnitFact, ...]
    footnotes: tuple[FootnoteFact, ...]
    hypotheses: tuple[ContextHypothesis, ...]


_STATEMENT_TITLES = (
    ("連結貸借対照表", StatementType.CONSOLIDATED_BALANCE_SHEET),
    ("貸借対照表", StatementType.BALANCE_SHEET),
    ("連結損益計算書", StatementType.CONSOLIDATED_INCOME_STATEMENT),
    ("損益計算書", StatementType.INCOME_STATEMENT),
    ("連結キャッシュ・フロー計算書", StatementType.CONSOLIDATED_CASH_FLOW_STATEMENT),
    ("キャッシュ・フロー計算書", StatementType.CASH_FLOW_STATEMENT),
    (
        "連結株主資本等変動計算書",
        StatementType.CONSOLIDATED_STATEMENT_OF_CHANGES_IN_EQUITY,
    ),
    ("株主資本等変動計算書", StatementType.STATEMENT_OF_CHANGES_IN_EQUITY),
)
_PERIOD_LABELS = (
    "前連結会計年度",
    "当連結会計年度",
    "前事業年度",
    "当事業年度",
)
_SECTION_HEADINGS = (
    "資産の部",
    "流動資産",
    "固定資産",
    "負債の部",
    "流動負債",
    "固定負債",
    "純資産の部",
    "売上高",
    "売上原価",
    "販売費及び一般管理費",
    "営業利益",
    "経常利益",
    "税引前当期純利益",
    "親会社株主に帰属する当期純利益",
    "営業活動によるキャッシュ・フロー",
    "投資活動によるキャッシュ・フロー",
    "財務活動によるキャッシュ・フロー",
    "株主資本等変動計算書",
)
_DATE_RE = re.compile(
    r"(?:(?P<era>令和|平成|昭和|大正|明治)"
    r"(?P<era_year>元|\d+)年(?P<era_month>\d{1,2})月(?P<era_day>\d{1,2})日"
    r"|(?P<year>\d{4})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日"
    r"|(?P<slash_year>\d{4})[/-](?P<slash_month>\d{1,2})[/-]"
    r"(?P<slash_day>\d{1,2}))"
)
_UNIT_RE = re.compile(
    r"[（(]\s*単位\s*[:：]\s*"
    r"(?P<unit>百万円|千円|億円|円|パーセント|%)\s*[）)]"
)
_FOOTNOTE_RE = re.compile(
    r"※(?:\d+)?|[（(]注\d*[）)]|注\d+|[¹²³⁴⁵⁶⁷⁸⁹]"
)
_NUMBER_RE = re.compile(
    r"^(?P<sign>△|▲|\+|-|−)?"
    r"(?P<number>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)$"
)
_DASH_VARIANTS = frozenset({"-", "‐", "‑", "‒", "–", "—", "―", "−", "ー"})
_ERA_STARTS = {
    "明治": (1867, date(1868, 1, 25), date(1912, 7, 30)),
    "大正": (1911, date(1912, 7, 30), date(1926, 12, 25)),
    "昭和": (1925, date(1926, 12, 25), date(1989, 1, 8)),
    "平成": (1988, date(1989, 1, 8), date(2019, 5, 1)),
    "令和": (2018, date(2019, 5, 1), None),
}


@dataclass(frozen=True)
class _EvidenceLine:
    evidence: EvidenceText
    page_number: int
    extraction_source: ExtractionSource
    bounding_box: BoundingBox
    order: int


def compile_financial_context(document: PDFDocument) -> FinancialDocumentContext:
    """Compile Japanese financial context while retaining all source evidence."""
    lines = _build_evidence_lines(document)
    statement_titles = tuple(
        title
        for line in lines
        for title in _find_statement_titles(line)
    )
    title_locations = _title_locations(lines, statement_titles)
    sections = _build_statement_sections(lines, title_locations)
    periods, dates, date_hypotheses = _extract_periods_and_dates(lines)
    units, unit_hypotheses = _extract_units(lines, sections, title_locations)
    footnotes = _extract_footnotes(lines, sections, title_locations)
    title_hypotheses = _title_hypotheses(statement_titles)

    return FinancialDocumentContext(
        document_type=(
            DocumentType.JAPANESE_FINANCIAL_STATEMENTS
            if statement_titles
            else DocumentType.UNKNOWN
        ),
        document_type_evidence=tuple(title.evidence for title in statement_titles),
        document_type_provenance=(
            FactProvenance.INFERRED if statement_titles else None
        ),
        document_evidence=tuple(line.evidence for line in lines),
        statement_titles=statement_titles,
        statement_sections=sections,
        reporting_periods=periods,
        reporting_dates=dates,
        units=units,
        footnotes=footnotes,
        hypotheses=tuple(date_hypotheses + unit_hypotheses + title_hypotheses),
    )


def parse_numeric_value(raw_text: str) -> NumericInterpretation:
    """Parse only documented financial number forms; never coerce dash or blank."""
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string.")
    normalized = unicodedata.normalize("NFKC", raw_text).strip()
    if not normalized:
        return NumericInterpretation(
            raw_text, None, None, NumericValueKind.BLANK, None
        )
    if all(character in _DASH_VARIANTS for character in normalized):
        return NumericInterpretation(
            raw_text, normalized, None, NumericValueKind.DASH, None
        )

    match = _NUMBER_RE.fullmatch(normalized)
    if match is None:
        return NumericInterpretation(
            raw_text,
            None,
            None,
            NumericValueKind.UNSUPPORTED,
            FactProvenance.AMBIGUOUS,
            "Unsupported or ambiguous financial numeric notation.",
        )

    sign = match.group("sign") or ""
    number_text = match.group("number")
    try:
        value = Decimal(number_text.replace(",", ""))
    except InvalidOperation:
        return NumericInterpretation(
            raw_text,
            None,
            None,
            NumericValueKind.UNSUPPORTED,
            FactProvenance.AMBIGUOUS,
            "Numeric text could not be parsed as a decimal value.",
        )

    if sign in {"△", "▲", "-", "−"}:
        value = -value
    canonical = str(value)
    kind = NumericValueKind.ZERO if value == 0 else NumericValueKind.NUMBER
    return NumericInterpretation(
        raw_text,
        canonical,
        value,
        kind,
        FactProvenance.NORMALIZED,
    )


def _build_evidence_lines(document: PDFDocument) -> list[_EvidenceLine]:
    if not isinstance(document, PDFDocument):
        raise TypeError("document must be a PDFDocument from pdf_ingestion.")

    lines: list[_EvidenceLine] = []
    order = 0
    for page in document.pages:
        if not isinstance(page, PDFPage) or page.page_number < 1:
            raise ValueError("PDFDocument contains an invalid page record.")
        spans = page.text_spans
        if any(
            not isinstance(span, TextSpan) or span.page_number != page.page_number
            for span in spans
        ):
            raise ValueError(
                f"PDF page {page.page_number} contains an invalid or mismatched text span."
            )
        page_lines = _group_spans_into_lines(spans)
        for evidence, extraction_source, bbox in page_lines:
            lines.append(
                _EvidenceLine(
                    evidence=evidence,
                    page_number=page.page_number,
                    extraction_source=extraction_source,
                    bounding_box=bbox,
                    order=order,
                )
            )
            order += 1
    return lines


def _group_spans_into_lines(
    spans: tuple[TextSpan, ...],
) -> list[tuple[EvidenceText, ExtractionSource, BoundingBox]]:
    groups: list[list[TextSpan]] = []
    ordered_spans = sorted(
        spans,
        key=lambda span: (span.bounding_box[1], span.bounding_box[0]),
    )
    for span in ordered_spans:
        center_y = (span.bounding_box[1] + span.bounding_box[3]) / 2
        candidates = []
        for group in groups:
            first = group[0]
            if first.extraction_source is not span.extraction_source:
                continue
            group_top = min(item.bounding_box[1] for item in group)
            group_bottom = max(item.bounding_box[3] for item in group)
            group_center = (group_top + group_bottom) / 2
            tolerance = max(
                2.0,
                min(
                    span.bounding_box[3] - span.bounding_box[1],
                    group_bottom - group_top,
                )
                * 0.2,
            )
            if abs(center_y - group_center) <= tolerance:
                candidates.append((abs(center_y - group_center), group))
        if candidates:
            min(candidates, key=lambda candidate: candidate[0])[1].append(span)
        else:
            groups.append([span])

    output = []
    for group in groups:
        group.sort(key=lambda span: span.bounding_box[0])
        raw_text = "".join(span.raw_text for span in group)
        bbox = (
            min(span.bounding_box[0] for span in group),
            min(span.bounding_box[1] for span in group),
            max(span.bounding_box[2] for span in group),
            max(span.bounding_box[3] for span in group),
        )
        output.append(
            (
                EvidenceText(raw_text, tuple(group)),
                group[0].extraction_source,
                bbox,
            )
        )
    output.sort(key=lambda item: (item[2][1], item[2][0]))
    return output


def _find_statement_titles(line: _EvidenceLine) -> list[StatementTitleFact]:
    normalized, starts, ends = _normalize_with_offsets(line.evidence.raw_text)
    patterns = sorted(_STATEMENT_TITLES, key=lambda item: len(item[0]), reverse=True)
    pattern = re.compile("|".join(re.escape(title) for title, _ in patterns))
    type_by_title = dict(patterns)
    facts = []
    for match in pattern.finditer(normalized):
        raw_title = _raw_match(line.evidence.raw_text, starts, ends, match.start(), match.end())
        facts.append(
            StatementTitleFact(
                raw_title=raw_title,
                statement_type=type_by_title[match.group()],
                evidence=line.evidence,
            )
        )
    return facts


def _title_locations(
    lines: list[_EvidenceLine], titles: tuple[StatementTitleFact, ...]
) -> list[tuple[int, StatementTitleFact]]:
    index_by_evidence = {id(line.evidence): index for index, line in enumerate(lines)}
    return [
        (index_by_evidence[id(title.evidence)], title)
        for title in titles
        if id(title.evidence) in index_by_evidence
    ]


def _title_hypotheses(
    titles: tuple[StatementTitleFact, ...],
) -> list[ContextHypothesis]:
    statement_types = {title.statement_type for title in titles}
    if len(statement_types) <= 1:
        return []
    return [
        ContextHypothesis(
            description=(
                "Multiple distinct Japanese financial statement titles were found; "
                "no single document-level statement type was selected."
            ),
            evidence=tuple(title.evidence for title in titles),
        )
    ]


def _build_statement_sections(
    lines: list[_EvidenceLine],
    title_locations: list[tuple[int, StatementTitleFact]],
) -> tuple[StatementSectionContext, ...]:
    sections = []
    ordered_titles = sorted(title_locations, key=lambda item: item[0])
    for title_position, (start, title) in enumerate(ordered_titles):
        end = (
            ordered_titles[title_position + 1][0]
            if title_position + 1 < len(ordered_titles)
            else len(lines)
        )
        section_lines = lines[start:end]
        headings = tuple(
            heading
            for line in section_lines
            for heading in _find_section_headings(line)
            if heading.raw_heading != title.raw_title
        )
        nearby = []
        title_line = lines[start]
        for candidate_index in (start - 1, start + 1):
            if 0 <= candidate_index < len(lines):
                candidate = lines[candidate_index]
                if candidate.page_number == title_line.page_number:
                    nearby.append(candidate.evidence)
        sections.append(
            StatementSectionContext(
                title=title,
                headings=headings,
                nearby_evidence=tuple(nearby),
                section_evidence=tuple(line.evidence for line in section_lines),
            )
        )
    return tuple(sections)


def _find_section_headings(line: _EvidenceLine) -> list[SectionHeadingFact]:
    normalized, starts, ends = _normalize_with_offsets(line.evidence.raw_text)
    pattern = re.compile(
        "|".join(re.escape(heading) for heading in sorted(_SECTION_HEADINGS, key=len, reverse=True))
    )
    return [
        SectionHeadingFact(
            _raw_match(
                line.evidence.raw_text, starts, ends, match.start(), match.end()
            ),
            line.evidence,
        )
        for match in pattern.finditer(normalized)
    ]


def _extract_periods_and_dates(
    lines: list[_EvidenceLine],
) -> tuple[
    tuple[ReportingPeriodFact, ...],
    tuple[ReportingDateFact, ...],
    list[ContextHypothesis],
]:
    all_periods: list[ReportingPeriodFact] = []
    all_dates: list[ReportingDateFact] = []
    hypotheses = []

    for line in lines:
        normalized, starts, ends = _normalize_with_offsets(line.evidence.raw_text)
        period_matches = list(
            re.finditer("|".join(re.escape(label) for label in _PERIOD_LABELS), normalized)
        )
        date_matches = list(_DATE_RE.finditer(normalized))
        line_periods = []
        used_date_indexes: set[int] = set()

        for index, period_match in enumerate(period_matches):
            segment_end = (
                period_matches[index + 1].start()
                if index + 1 < len(period_matches)
                else len(normalized)
            )
            associated = []
            for date_index, date_match in enumerate(date_matches):
                if period_match.end() <= date_match.start() < segment_end:
                    fact = _date_fact(
                        line,
                        date_match,
                        starts,
                        ends,
                        associated_period_label=period_match.group(),
                    )
                    associated.append(fact)
                    used_date_indexes.add(date_index)
                    all_dates.append(fact)
            raw_label = _raw_match(
                line.evidence.raw_text,
                starts,
                ends,
                period_match.start(),
                period_match.end(),
            )
            line_periods.append(
                ReportingPeriodFact(raw_label, line.evidence, tuple(associated))
            )

        for date_index, date_match in enumerate(date_matches):
            if date_index not in used_date_indexes:
                all_dates.append(_date_fact(line, date_match, starts, ends))
        all_periods.extend(line_periods)

    by_period: dict[str, list[ReportingDateFact]] = {}
    for period in all_periods:
        by_period.setdefault(period.raw_label, []).extend(period.dates)
    for label, period_dates in by_period.items():
        normalized_dates = {item.iso_date for item in period_dates if item.iso_date}
        if len(normalized_dates) > 1:
            hypotheses.append(
                ContextHypothesis(
                    description=(
                        f"Conflicting explicit dates were associated with period label {label!r}; "
                        "all candidates are retained."
                    ),
                    evidence=tuple(dict.fromkeys(item.evidence for item in period_dates)),
                )
            )
    for item in all_dates:
        if item.issue:
            hypotheses.append(
                ContextHypothesis(
                    description=f"Date {item.raw_text!r} was preserved but not normalized: {item.issue}",
                    evidence=(item.evidence,),
                )
            )
    return tuple(all_periods), tuple(all_dates), hypotheses


def _date_fact(
    line: _EvidenceLine,
    match: re.Match[str],
    starts: list[int],
    ends: list[int],
    associated_period_label: str | None = None,
) -> ReportingDateFact:
    raw_date = _raw_match(
        line.evidence.raw_text, starts, ends, match.start(), match.end()
    )
    iso_date, issue = _normalize_date(match)
    return ReportingDateFact(
        raw_text=raw_date,
        iso_date=iso_date,
        evidence=line.evidence,
        normalization_provenance=(
            FactProvenance.NORMALIZED if iso_date else FactProvenance.AMBIGUOUS
        ),
        associated_period_label=associated_period_label,
        association_provenance=(
            FactProvenance.EXPLICIT if associated_period_label else None
        ),
        issue=issue,
    )


def _normalize_date(match: re.Match[str]) -> tuple[str | None, str | None]:
    groups = match.groupdict()
    try:
        if groups.get("era"):
            era = groups["era"]
            era_year = 1 if groups["era_year"] == "元" else int(groups["era_year"])
            base_year, era_start, era_end = _ERA_STARTS[era]
            parsed = date(
                base_year + era_year,
                int(groups["era_month"]),
                int(groups["era_day"]),
            )
            if parsed < era_start:
                raise ValueError(f"date falls before the {era} era began")
            if era_end is not None and parsed >= era_end:
                raise ValueError(f"date falls after the {era} era ended")
        elif groups.get("year"):
            parsed = date(
                int(groups["year"]), int(groups["month"]), int(groups["day"])
            )
        else:
            parsed = date(
                int(groups["slash_year"]),
                int(groups["slash_month"]),
                int(groups["slash_day"]),
            )
    except (ValueError, TypeError) as error:
        return None, f"Invalid or era-inconsistent date: {error}."
    return parsed.isoformat(), None


def _extract_units(
    lines: list[_EvidenceLine],
    sections: tuple[StatementSectionContext, ...],
    title_locations: list[tuple[int, StatementTitleFact]],
) -> tuple[tuple[FinancialUnitFact, ...], list[ContextHypothesis]]:
    units: list[FinancialUnitFact] = []
    for line in lines:
        normalized, starts, ends = _normalize_with_offsets(line.evidence.raw_text)
        for match in _UNIT_RE.finditer(normalized):
            unit = _unit_from_raw(match.group("unit"))
            if unit is None:
                continue
            declaration = _raw_match(
                line.evidence.raw_text, starts, ends, match.start(), match.end()
            )
            section = _section_for_line(line, sections, title_locations)
            units.append(
                FinancialUnitFact(
                    raw_declaration=declaration,
                    normalized_unit=unit,
                    evidence=line.evidence,
                    scope=(
                        UnitScope.STATEMENT_SECTION
                        if section is not None
                        else UnitScope.DOCUMENT
                    ),
                    associated_section_title=(
                        section.title.raw_title if section is not None else None
                    ),
                    associated_section_evidence=(
                        section.title.evidence if section is not None else None
                    ),
                    association_provenance=(
                        FactProvenance.INFERRED if section is not None else None
                    ),
                )
            )

    hypotheses = []
    units_by_section: dict[int, list[FinancialUnitFact]] = {}
    for unit in units:
        if unit.associated_section_evidence is not None:
            key = id(unit.associated_section_evidence)
            units_by_section.setdefault(key, []).append(unit)
    for section_units in units_by_section.values():
        title = section_units[0].associated_section_title or ""
        distinct_units = {item.normalized_unit for item in section_units}
        if len(distinct_units) > 1:
            hypotheses.append(
                ContextHypothesis(
                    description=(
                        f"Conflicting unit declarations occur in statement section {title!r}; "
                        "no conversion or single-unit choice was made."
                    ),
                    evidence=tuple(item.evidence for item in section_units),
                )
            )
        for unit in section_units:
            hypotheses.append(
                ContextHypothesis(
                    description=(
                        f"Unit {unit.raw_declaration!r} is provisionally grouped with "
                        f"statement section {title!r} by its position after that title; "
                        "it is not associated with a table or cell."
                    ),
                    evidence=(unit.evidence,),
                    provenance=FactProvenance.INFERRED,
                )
            )
    return tuple(units), hypotheses


def _unit_from_raw(raw_unit: str) -> FinancialUnit | None:
    normalized = unicodedata.normalize("NFKC", raw_unit)
    return {
        "円": FinancialUnit.YEN,
        "千円": FinancialUnit.THOUSAND_YEN,
        "百万円": FinancialUnit.MILLION_YEN,
        "億円": FinancialUnit.HUNDRED_MILLION_YEN,
        "%": FinancialUnit.PERCENT,
        "パーセント": FinancialUnit.PERCENT,
    }.get(normalized)


def _extract_footnotes(
    lines: list[_EvidenceLine],
    sections: tuple[StatementSectionContext, ...],
    title_locations: list[tuple[int, StatementTitleFact]],
) -> tuple[FootnoteFact, ...]:
    output = []
    for line in lines:
        normalized, starts, ends = _normalize_with_offsets(line.evidence.raw_text)
        matches = list(_FOOTNOTE_RE.finditer(normalized))
        for index, match in enumerate(matches):
            raw_marker = _raw_match(
                line.evidence.raw_text, starts, ends, match.start(), match.end()
            )
            text_end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
            explanation_start = match.end()
            explanatory_text = None
            explanatory_evidence = None
            if explanation_start < text_end:
                candidate = _raw_match(
                    line.evidence.raw_text,
                    starts,
                    ends,
                    explanation_start,
                    text_end,
                ).strip()
                if candidate:
                    explanatory_text = candidate
                    explanatory_evidence = line.evidence
            section = _section_for_line(line, sections, title_locations)
            output.append(
                FootnoteFact(
                    raw_marker=raw_marker,
                    evidence=line.evidence,
                    nearby_evidence=_nearby_evidence(line, lines),
                    explanatory_text=explanatory_text,
                    explanatory_evidence=explanatory_evidence,
                    associated_section_title=(
                        section.title.raw_title if section is not None else None
                    ),
                    association_provenance=(
                        FactProvenance.INFERRED if section is not None else None
                    ),
                )
            )
    return tuple(output)


def _nearby_evidence(
    target: _EvidenceLine,
    lines: list[_EvidenceLine],
) -> tuple[EvidenceText, ...]:
    adjacent = []
    for offset in (-1, 1):
        index = target.order + offset
        if 0 <= index < len(lines) and lines[index].page_number == target.page_number:
            adjacent.append(lines[index].evidence)
    return tuple(adjacent)


def _section_for_line(
    line: _EvidenceLine,
    sections: tuple[StatementSectionContext, ...],
    title_locations: list[tuple[int, StatementTitleFact]],
) -> StatementSectionContext | None:
    line_position = line.order
    candidates = []
    ordered_titles = sorted(title_locations, key=lambda item: item[0])
    for index, section in enumerate(sections):
        start = ordered_titles[index][0]
        end = ordered_titles[index + 1][0] if index + 1 < len(ordered_titles) else float("inf")
        title_page = section.title.evidence.page_numbers
        if (
            start <= line_position < end
            and line.page_number in title_page
        ):
            candidates.append(section)
    return candidates[0] if len(candidates) == 1 else None


def _normalize_with_offsets(raw_text: str) -> tuple[str, list[int], list[int]]:
    normalized_parts = []
    starts = []
    ends = []
    for index, character in enumerate(raw_text):
        part = unicodedata.normalize("NFKC", character)
        normalized_parts.append(part)
        starts.extend([index] * len(part))
        ends.extend([index + 1] * len(part))
    return "".join(normalized_parts), starts, ends


def _raw_match(
    raw_text: str,
    starts: list[int],
    ends: list[int],
    start: int,
    end: int,
) -> str:
    if start >= end or start >= len(starts) or end > len(ends):
        return ""
    return raw_text[starts[start] : ends[end - 1]]
