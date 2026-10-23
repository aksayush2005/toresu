from __future__ import annotations

import csv
import io
import time
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from io import BytesIO
import math
import re
import sys
from typing import Any
import zipfile
from pathlib import Path

import httpx
import pymupdf
import streamlit as st
from PIL import Image, ImageDraw

# Streamlit executes this file as a script, so the src-layout package root may
# not otherwise be on sys.path when the project is not installed.
_SOURCE_ROOT = str(Path(__file__).resolve().parents[1])
if _SOURCE_ROOT not in sys.path:
    sys.path.insert(0, _SOURCE_ROOT)

from toresu.evaluation import evaluate_table_reference


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> tuple[Any, ...] | list[Any]:
    return value if isinstance(value, (list, tuple)) else ()


def _records(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in _sequence(value) if isinstance(item, Mapping)]


def _index(value: Any, fallback: int | None = None) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return fallback


def _bounding_box(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(
        isinstance(coordinate, bool)
        or not isinstance(coordinate, (int, float))
        or not math.isfinite(coordinate)
        for coordinate in value
    ):
        return None
    return tuple(float(coordinate) for coordinate in value)


def render_page(
    pdf_bytes: bytes,
    page_number: int,
    result: dict[str, Any],
    *,
    selected_observation: dict[str, Any] | None = None,
    show_all_evidence: bool = True,
    show_structure: bool = True,
    dpi: int = 120,
) -> Image.Image:
    """Render one PDF page with distinct evidence and inferred-geometry overlays."""
    if page_number < 1:
        raise ValueError("page_number must be at least 1.")
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        if page_number > document.page_count:
            raise ValueError(f"page_number exceeds the PDF page count ({document.page_count}).")
        page = document.load_page(page_number - 1)
        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        image = Image.open(BytesIO(pixmap.tobytes("png"))).convert("RGB")
        scale_x = pixmap.width / page.rect.width
        scale_y = pixmap.height / page.rect.height
    finally:
        document.close()

    draw = ImageDraw.Draw(image)
    evidence_spans = _evidence_spans(result, selected_observation, page_number)
    if show_all_evidence:
        for span in evidence_spans:
            color = "#ff7f0e" if span.get("extraction_source") == "ocr" else "#168aad"
            _draw_box(draw, span.get("bounding_box"), scale_x, scale_y, color, 3)
    if show_structure:
        for table in _records(result.get("tables")):
            if table.get("page_number") != page_number:
                continue
            _draw_box(draw, table.get("bounding_box"), scale_x, scale_y, "#2a9d46", 3)
            for row in _records(table.get("rows")):
                cells = _records(row.get("cells"))
                label_cell = _mapping(row.get("label_cell"))
                if label_cell:
                    cells.insert(0, label_cell)
                for cell in cells:
                    _draw_box(
                        draw,
                        cell.get("cell_boundary"),
                        scale_x,
                        scale_y,
                        "#b23a48",
                        2,
                    )
    return image


def _evidence_spans(
    result: dict[str, Any],
    selected_observation: dict[str, Any] | None,
    page_number: int,
) -> list[dict[str, Any]]:
    source_observations = (
        [selected_observation]
        if selected_observation is not None
        else _records(result.get("observations"))
    )
    spans: dict[tuple[Any, ...], dict[str, Any]] = {}
    for observation in _records(source_observations):
        for field in ("source_spans", "row_label_source_spans", "unit_source_spans"):
            for span in _records(observation.get(field)):
                if span.get("page_number") == page_number:
                    if isinstance(span, dict):
                        spans[_span_key(span)] = span
        period = _mapping(observation.get("period"))
        for span in _records(period.get("evidence_spans")):
            if span.get("page_number") == page_number:
                if isinstance(span, dict):
                    spans[_span_key(span)] = span
    if selected_observation is None:
        for table in _records(result.get("tables")):
            candidates = _records(table.get("context_spans"))
            for row in _sequence(table.get("header_rows")):
                for header in _records(row):
                    candidates.extend(_records(header.get("source_spans")))
            for span in candidates:
                if span.get("page_number") == page_number:
                    if isinstance(span, dict):
                        spans[_span_key(span)] = span
    return list(spans.values())


def _span_key(span: dict[str, Any]) -> tuple[Any, ...]:
    bounding_box = _bounding_box(span.get("bounding_box"))
    return (
        span.get("page_number"),
        bounding_box,
        span.get("raw_text"),
        span.get("extraction_source"),
    )


def _draw_box(
    draw: ImageDraw.ImageDraw,
    box: Any,
    scale_x: float,
    scale_y: float,
    color: str,
    width: int,
) -> None:
    valid_box = _bounding_box(box)
    if valid_box is None:
        return
    x0, y0, x1, y1 = valid_box
    draw.rectangle(
        (x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y),
        outline=color,
        width=width,
    )


def _page_summary(result: dict[str, Any], page_number: int) -> dict[str, int]:
    spans = _evidence_spans(result, None, page_number)
    sources = Counter(span.get("extraction_source", "unknown") for span in spans)
    return {
        "evidence_spans": len(spans),
        "native_text_spans": sources.get("pdf_text", 0),
        "ocr_spans": sources.get("ocr", 0),
        "tables": sum(
            table.get("page_number") == page_number
            for table in _records(result.get("tables"))
        ),
        "observations": sum(
            any(
                span.get("page_number") == page_number
                for span in _records(observation.get("source_spans"))
            )
            for observation in _records(result.get("observations"))
        ),
    }


def _pdf_page_count(pdf_bytes: bytes) -> int:
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        return document.page_count
    finally:
        document.close()


def _table_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    columns = _records(table.get("columns"))
    column_names: dict[int, str] = {}
    auxiliary_columns: set[int] = set()
    for position, column in enumerate(columns):
        column_index = _index(column.get("column_index"), position)
        if column_index is None:
            continue
        if column.get("is_auxiliary") is True:
            auxiliary_columns.add(column_index)
        header_texts = [
            header["raw_text"]
            for header in _records(column.get("header_path"))
            if isinstance(header.get("raw_text"), str) and header["raw_text"]
        ]
        column_names[column_index] = (
            " / ".join(header_texts) or f"Column {column_index + 1}"
        )

    rows_data = _records(table.get("rows"))
    highest_cell_column = max(
        (
            column_index
            for row in rows_data
            for cell in _records(row.get("cells"))
            if (column_index := _index(cell.get("column_index"))) is not None
        ),
        default=-1,
    )
    column_count = max(len(columns), highest_cell_column + 1)
    column_names = {
        index: column_names.get(index, f"Column {index + 1}")
        for index in range(column_count)
    }
    used_names: dict[str, int] = {}
    for index, name in column_names.items():
        occurrence = used_names.get(name, 0) + 1
        used_names[name] = occurrence
        if occurrence > 1:
            column_names[index] = f"{name} ({occurrence})"

    unassigned_needed = any(
        cell.get("raw_text") is not None
        and _index(cell.get("column_index")) is None
        for row in rows_data
        for cell in _records(row.get("cells"))
    )
    parent_by_id = {
        row.get("row_id"): row.get("parent_row_id")
        for row in rows_data
        if row.get("row_id")
    }
    value_column_indexes = set(range(column_count)) - auxiliary_columns

    def hierarchy_depth(row: Mapping[str, Any]) -> int:
        depth = 0
        parent_id = row.get("parent_row_id")
        visited = {row.get("row_id")}
        while parent_id in parent_by_id and parent_id not in visited:
            visited.add(parent_id)
            depth += 1
            parent_id = parent_by_id[parent_id]
        return depth

    rows = []
    for row in rows_data:
        cells = _records(row.get("cells"))
        label_cell = _mapping(row.get("label_cell"))
        raw_label = label_cell.get("raw_text")
        row_values = {name: None for name in column_names.values()}
        unassigned_values = []
        for cell in cells:
            column_index = _index(cell.get("column_index"))
            if column_index is None or column_index >= column_count:
                if cell.get("raw_text") is not None:
                    unassigned_values.append(cell.get("raw_text"))
            else:
                row_values[column_names[column_index]] = cell.get("raw_text")
        label = raw_label if isinstance(raw_label, str) else None
        if label is not None:
            label = f"{'  ' * hierarchy_depth(row)}{label}"
        value_cells = [
            cell
            for cell in cells
            if _index(cell.get("column_index")) not in auxiliary_columns
        ]
        populated_cells = [
            cell
            for cell in value_cells
            if isinstance(cell.get("raw_text"), str) and cell["raw_text"] != ""
        ]
        cells_by_column = {
            column_index: cell
            for cell in value_cells
            if (column_index := _index(cell.get("column_index"))) is not None
        }
        missing_value = any(
            column_index not in cells_by_column
            or cells_by_column[column_index].get("raw_text") is None
            for column_index in value_column_indexes
        )
        if label is None:
            row_status = "unresolved"
        elif any(cell.get("status") == "ambiguous" for cell in value_cells):
            row_status = "ambiguous"
        elif any(
            cell.get("status") == "unresolved"
            and cell.get("raw_text") is not None
            for cell in value_cells
        ):
            row_status = "unresolved"
        elif not populated_cells:
            row_status = "heading"
        elif missing_value:
            row_status = "incomplete"
        elif any(cell.get("status") == "inferred" for cell in populated_cells):
            row_status = "inferred"
        elif all(cell.get("status") == "resolved" for cell in populated_cells):
            row_status = "resolved"
        else:
            row_status = "unresolved"
        rows.append(
            {
                "row": label,
                **row_values,
                **(
                    {"Unassigned values": " | ".join(map(str, unassigned_values)) or None}
                    if unassigned_needed
                    else {}
                ),
                "status": row_status,
                "hierarchy_status": row.get("hierarchy_status"),
            }
        )
    return rows


def _table_value_columns(table: Mapping[str, Any]) -> list[str]:
    columns = _records(table.get("columns"))
    names: dict[int, str] = {}
    for position, column in enumerate(columns):
        column_index = _index(column.get("column_index"), position)
        if column_index is None:
            continue
        header_texts = [
            header["raw_text"]
            for header in _records(column.get("header_path"))
            if isinstance(header.get("raw_text"), str) and header["raw_text"]
        ]
        names[column_index] = " / ".join(header_texts) or f"Column {column_index + 1}"
    highest_cell_column = max(
        (
            column_index
            for row in _records(table.get("rows"))
            for cell in _records(row.get("cells"))
            if (column_index := _index(cell.get("column_index"))) is not None
        ),
        default=-1,
    )
    count = max(len(columns), highest_cell_column + 1)
    used: dict[str, int] = {}
    output = []
    for index in range(count):
        name = names.get(index, f"Column {index + 1}")
        used[name] = used.get(name, 0) + 1
        output.append(f"{name} ({used[name]})" if used[name] > 1 else name)
    return output


def _table_csv(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        return b"\xef\xbb\xbf"

    columns = list(dict.fromkeys(column for row in rows for column in row))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=columns,
        extrasaction="ignore",
        lineterminator="\r\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def _read_reference_csv(payload: bytes) -> tuple[list[str], list[dict[str, str]]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Reference CSV must be UTF-8 or UTF-8 with BOM.") from error
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if not reader.fieldnames or any(
            not isinstance(name, str) or not name for name in reader.fieldnames
        ):
            raise ValueError(
                "Reference CSV must have a header row with non-empty column names."
            )
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Reference CSV contains duplicate column names.")
        rows = [dict(row) for row in reader]
    except csv.Error as error:
        raise ValueError(f"Could not parse reference CSV: {error}") from error
    if not reader.fieldnames:
        raise ValueError("Reference CSV must have a header row with non-empty column names.")
    if any(None in row for row in rows):
        raise ValueError("Reference CSV contains rows with more fields than its header.")
    return reader.fieldnames, rows


def _operational_metrics(inspections: list[Mapping[str, Any]]) -> dict[str, Any]:
    document_rows = []
    table_rows = []
    totals = Counter()
    for document_index, inspection in enumerate(inspections, start=1):
        result = _mapping(inspection.get("result"))
        job = _mapping(inspection.get("job"))
        tables = _records(result.get("tables"))
        processed = job.get("state") in {"completed", "completed_with_warnings"}
        page_count = (
            _pdf_page_count(inspection["pdf_bytes"])
            if processed and isinstance(inspection.get("pdf_bytes"), bytes)
            else 0
        )
        elapsed = inspection.get("processing_seconds")
        if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool):
            elapsed = _job_processing_seconds(job)
        document_name = inspection.get("pdf_name")
        if not isinstance(document_name, str) or not document_name:
            document_name = f"document-{document_index}"

        doc_totals = Counter()
        for table_index, table in enumerate(tables, start=1):
            rows = _records(table.get("rows"))
            columns = _table_value_columns(table)
            reconstructed = bool(rows and columns)
            cells = [
                cell
                for row in rows
                for cell in (
                    ([_mapping(row.get("label_cell"))] if row.get("label_cell") else [])
                    + _records(row.get("cells"))
                )
            ]
            table_view_rows = _table_rows(dict(table))
            unresolved_cells = sum(cell.get("status") == "unresolved" for cell in cells)
            inferred_cells = sum(cell.get("status") == "inferred" for cell in cells)
            unresolved_rows = sum(row.get("status") == "unresolved" for row in table_view_rows)
            inferred_rows = sum(row.get("status") == "inferred" for row in table_view_rows)
            counts = {
                "tables_detected": 1,
                "tables_reconstructed": int(reconstructed),
                "rows": len(rows),
                "columns": len(columns) + 1,
                "unresolved_cells": unresolved_cells,
                "inferred_cells": inferred_cells,
                "unresolved_rows": unresolved_rows,
                "inferred_rows": inferred_rows,
            }
            doc_totals.update(counts)
            table_id = table.get("table_id")
            table_rows.append(
                {
                    "document": document_name,
                    "table": table_id if isinstance(table_id, str) else f"Table {table_index}",
                    "page": table.get("page_number"),
                    "reconstructed": reconstructed,
                    "rows": len(rows),
                    "columns": len(columns) + 1,
                    "unresolved cells": unresolved_cells,
                    "inferred cells": inferred_cells,
                    "unresolved rows": unresolved_rows,
                    "inferred rows": inferred_rows,
                }
            )
        totals.update(doc_totals)
        if processed:
            totals["documents_processed"] += 1
            totals["pages_processed"] += page_count
        document_rows.append(
            {
                "document": document_name,
                "processed": processed,
                "pages": page_count,
                "tables detected": doc_totals["tables_detected"],
                "tables reconstructed": doc_totals["tables_reconstructed"],
                "rows": doc_totals["rows"],
                "columns": doc_totals["columns"],
                "unresolved cells": doc_totals["unresolved_cells"],
                "inferred cells": doc_totals["inferred_cells"],
                "unresolved rows": doc_totals["unresolved_rows"],
                "inferred rows": doc_totals["inferred_rows"],
                "processing seconds": elapsed,
                "seconds per page": elapsed / page_count if elapsed and page_count else None,
            }
        )
    return {
        "documents_uploaded": len(inspections),
        "documents_processed": totals["documents_processed"],
        "pages_processed": totals["pages_processed"],
        "tables_detected": totals["tables_detected"],
        "tables_reconstructed": totals["tables_reconstructed"],
        "documents": document_rows,
        "tables": table_rows,
    }


def _job_processing_seconds(job: Mapping[str, Any]) -> float | None:
    durations = []
    for stage in _records(job.get("stages")):
        started = stage.get("started_at")
        finished = stage.get("finished_at")
        if not isinstance(started, str) or not isinstance(finished, str):
            continue
        try:
            start_time = datetime.fromisoformat(started.replace("Z", "+00:00"))
            finish_time = datetime.fromisoformat(finished.replace("Z", "+00:00"))
        except ValueError:
            continue
        duration = (finish_time - start_time).total_seconds()
        if duration >= 0:
            durations.append(duration)
    return sum(durations) if durations else None


def _mismatch_provenance(
    table: Mapping[str, Any],
    mismatches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_rows = _records(table.get("rows"))
    page_number = table.get("page_number")
    table_id = table.get("table_id")
    enriched = []
    for mismatch in mismatches:
        row_index = mismatch.get("row_index")
        column_index = mismatch.get("column_index")
        spans: list[dict[str, Any]] = []
        if isinstance(row_index, int) and 1 <= row_index <= len(source_rows):
            row = source_rows[row_index - 1]
            if column_index is None:
                label = _mapping(row.get("label_cell"))
                spans.extend(_records(label.get("source_spans")))
            else:
                for cell in _records(row.get("cells")):
                    if _index(cell.get("column_index")) == column_index:
                        spans.extend(_records(cell.get("source_spans")))
        elif isinstance(column_index, int):
            columns = _records(table.get("columns"))
            if 0 <= column_index < len(columns):
                for header in _records(columns[column_index].get("header_path")):
                    spans.extend(_records(header.get("source_spans")))
        enriched.append(
            {
                **mismatch,
                "page": page_number,
                "table_id": table_id,
                "source_spans": [
                    {
                        "raw_text": span.get("raw_text"),
                        "page_number": span.get("page_number"),
                        "bounding_box": span.get("bounding_box"),
                    }
                    for span in spans
                ],
            }
        )
    return enriched


def _render_metrics_dashboard(inspections: list[Mapping[str, Any]]) -> None:
    st.header("Metrics Dashboard")
    st.caption(
        "Operational counts describe processed outputs. Accuracy metrics are computed "
        "only against a supplied reference CSV."
    )
    metrics = _operational_metrics(inspections)
    metric_columns = st.columns(4)
    for column, label, value in zip(
        metric_columns,
        ("Documents processed", "Pages processed", "Tables detected", "Tables reconstructed"),
        (
            f"{metrics['documents_processed']}/{metrics['documents_uploaded']}",
            metrics["pages_processed"],
            metrics["tables_detected"],
            metrics["tables_reconstructed"],
        ),
    ):
        column.metric(label, value)
    st.subheader("Per-document operational metrics")
    st.dataframe(metrics["documents"], use_container_width=True, hide_index=True)
    st.subheader("Per-table operational metrics")
    st.dataframe(metrics["tables"], use_container_width=True, hide_index=True)
    st.caption(
        "Reconstructed means the API returned a table candidate with at least one row "
        "and one value column. Row/column counts include the account-label column. "
        "Cell counts include row-label and value cells; auxiliary columns remain included."
    )

    processed_documents = [
        (index, inspection)
        for index, inspection in enumerate(inspections)
        if _mapping(inspection.get("job")).get("state")
        in {"completed", "completed_with_warnings"}
    ]
    if not processed_documents:
        st.info("Reference accuracy is Not evaluated because no document completed.")
        return
    selected_index = st.selectbox(
        "Document to evaluate",
        range(len(processed_documents)),
        format_func=lambda position: (
            str(processed_documents[position][1].get("pdf_name") or "Document")
        ),
        key="metrics-document",
    )
    _, selected_inspection = processed_documents[selected_index]
    selected_result = _mapping(selected_inspection.get("result"))
    selected_tables = _records(selected_result.get("tables"))
    if not selected_tables:
        st.info("Reference accuracy is Not evaluated because this document has no tables.")
        return
    selected_table_index = st.selectbox(
        "Table to evaluate",
        range(len(selected_tables)),
        format_func=lambda index: (
            f"Page {selected_tables[index].get('page_number')} · "
            f"{selected_tables[index].get('statement_title_raw') or selected_tables[index].get('table_id') or f'Table {index + 1}'}"
        ),
        key=f"metrics-table-{selected_index}",
    )
    selected_table = selected_tables[selected_table_index]
    reference_upload = st.file_uploader(
        "Upload reference table CSV (UTF-8; BOM accepted)",
        type=["csv"],
        key=f"metrics-reference-{selected_index}-{selected_table_index}",
    )
    if reference_upload is None:
        st.info("Ground-truth metrics: Not evaluated (no reference CSV uploaded).")
        return
    try:
        reference_headers, reference_rows = _read_reference_csv(
            reference_upload.getvalue()
        )
    except ValueError as error:
        st.error(str(error))
        return
    if not reference_headers:
        st.error("Reference CSV has no columns to compare.")
        return
    label_candidates = ("row", "項目", "科目", "勘定科目", "account")
    normalized_headers = {
        header: "".join(header.casefold().split()) for header in reference_headers
    }
    default_label_index = next(
        (
            index
            for index, header in enumerate(reference_headers)
            if normalized_headers[header] in label_candidates
        ),
        0,
    )
    reference_label = st.selectbox(
        "Reference row-label column",
        reference_headers,
        index=default_label_index,
        key=f"metrics-label-{selected_index}-{selected_table_index}",
    )
    reference_columns = [
        header for header in reference_headers if header != reference_label
    ]
    predicted_columns = _table_value_columns(selected_table)
    predicted_rows = _table_rows(dict(selected_table))
    report = evaluate_table_reference(
        predicted_rows,
        reference_rows,
        predicted_columns,
        reference_columns,
        reference_label_column=reference_label,
    )
    report["mismatches"] = _mismatch_provenance(
        selected_table, report["mismatches"]
    )
    st.markdown("#### Reference-based metrics")
    st.caption(
        "Labels/headers use the evaluation framework's NFKC and whitespace-insensitive "
        "comparison. Cell exact match trims surrounding whitespace only. Numeric "
        "accuracy accepts comma-grouping and Decimal-equivalent precision; zero, "
        "dashes, △, minus signs, and empty cells are not rewritten."
    )
    values = st.columns(5)
    metrics_to_display = (
        ("Cell exact match", report["cell_exact_match_accuracy"]["accuracy"]),
        ("Numeric accuracy", report["numeric_cell_accuracy"]["accuracy"]),
        ("Row-label accuracy", report["row_label_accuracy"]["accuracy"]),
        ("Header alignment", report["column_alignment_accuracy"]["accuracy"]),
        ("Structure accuracy", report["table_structure_accuracy"]["accuracy"]),
    )
    for column, (label, score) in zip(values, metrics_to_display):
        column.metric(label, "Not evaluated" if score is None else f"{score:.1%}")
    row_counts = report["row_count"]
    column_counts = report["column_count"]
    st.write(
        {
            "row count agreement": report["row_count_agreement"],
            "reference/predicted rows": f"{row_counts['reference']}/{row_counts['predicted']}",
            "column count agreement": report["column_count_agreement"],
            "reference/predicted value columns": (
                f"{column_counts['reference']}/{column_counts['predicted']}"
            ),
            "row-label precision/recall/F1": report["row_label_accuracy"][
                "precision_recall_f1"
            ],
            "cell precision/recall/F1": report["cell_exact_match_accuracy"][
                "precision_recall_f1"
            ],
            "exact table structure agreement": report["table_structure_accuracy"][
                "exact_agreement"
            ],
        }
    )
    mismatches = report["mismatches"]
    if mismatches:
        st.markdown("#### Mismatches with source provenance")
        st.dataframe(mismatches, use_container_width=True, hide_index=True)
    else:
        st.success("No row, column, or cell mismatches found against this reference.")


def _table_csv_filename(
    table_name: Any,
    table_id: Any,
    table_index: int,
) -> str:
    name = table_name if isinstance(table_name, str) and table_name.strip() else table_id
    if not isinstance(name, str) or not name.strip():
        name = f"table-{table_index}"
    safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    safe_name = re.sub(r"\s+", " ", safe_name)
    return f"{safe_name or f'table-{table_index}'}.csv"


def _inspection_zip(inspections: list[Mapping[str, Any]]) -> bytes:
    archive = BytesIO()
    used_names: set[str] = set()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for inspection_index, inspection in enumerate(inspections, start=1):
            pdf_bytes = inspection.get("pdf_bytes")
            if not isinstance(pdf_bytes, bytes):
                continue
            original_pdf_name = inspection.get("pdf_name")
            pdf_stem = (
                Path(original_pdf_name).stem
                if isinstance(original_pdf_name, str) and original_pdf_name
                else f"document-{inspection_index}"
            )
            pdf_name = _table_csv_filename(
                pdf_stem,
                None,
                inspection_index,
            )[:-4] + ".pdf"
            pdf_path = _unique_archive_path(
                f"pdfs/{pdf_name}", used_names
            )
            bundle.writestr(pdf_path, pdf_bytes)

            pdf_stem = Path(pdf_name).stem
            result = _mapping(inspection.get("result"))
            for table_index, table in enumerate(
                _records(result.get("tables")),
                start=1,
            ):
                csv_name = _table_csv_filename(
                    table.get("statement_title_raw"),
                    table.get("table_id"),
                    table_index,
                )
                rows = _table_rows(table)
                csv_path = _unique_archive_path(
                    f"csvs/{pdf_stem}/{table_index:02d}-{csv_name}",
                    used_names,
                )
                bundle.writestr(csv_path, _table_csv(rows))
    return archive.getvalue()


def _unique_archive_path(path: str, used_names: set[str]) -> str:
    if path not in used_names:
        used_names.add(path)
        return path
    candidate = Path(path)
    counter = 2
    while True:
        unique = candidate.with_name(f"{candidate.stem}-{counter}{candidate.suffix}")
        if unique.as_posix() not in used_names:
            used_names.add(unique.as_posix())
            return unique.as_posix()
        counter += 1


def _request_result(base_url: str, job_id: str, timeout_seconds: float = 180) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=15) as client:
        while time.monotonic() < deadline:
            status_response = client.get(f"/v1/jobs/{job_id}")
            status_response.raise_for_status()
            job = status_response.json()
            if job["state"] in {"completed", "completed_with_warnings", "failed"}:
                result_response = client.get(f"/v1/jobs/{job_id}/result")
                result_response.raise_for_status()
                return job, result_response.json()
            time.sleep(0.5)
    raise TimeoutError(f"Job {job_id} did not finish within {timeout_seconds:g} seconds.")


def run() -> None:
    st.set_page_config(page_title="Toresu document inspector", layout="wide")
    st.title("Toresu document inspector")
    st.caption("Inspect extracted evidence and inferred table structure from a local PDF.")
    with st.sidebar:
        api_url = st.text_input("Toresu API URL", "http://127.0.0.1:8000")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )
        process = st.button(
            "Process documents",
            type="primary",
            disabled=not uploaded_files,
        )

    if process and uploaded_files:
        inspections = []
        for uploaded in uploaded_files:
            pdf_bytes = uploaded.getvalue()
            processing_started = time.monotonic()
            inspection: dict[str, Any] = {
                "pdf_name": uploaded.name,
                "pdf_bytes": pdf_bytes,
                "job": None,
                "result": {"tables": []},
                "error": None,
                "processing_seconds": None,
            }
            try:
                with httpx.Client(timeout=60) as client:
                    response = client.post(
                        f"{api_url.rstrip('/')}/v1/documents",
                        files={"file": (uploaded.name, pdf_bytes, "application/pdf")},
                    )
                    response.raise_for_status()
                    job_id = response.json()["job_id"]
                with st.spinner(f"Processing {uploaded.name} (job {job_id})..."):
                    job, result = _request_result(api_url, job_id)
                inspection["job"] = job
                inspection["result"] = result
            except (httpx.HTTPError, ValueError, TimeoutError) as error:
                inspection["error"] = str(error)
            inspection["processing_seconds"] = time.monotonic() - processing_started
            inspections.append(inspection)
        if inspections:
            st.session_state["inspection_page"] = 1
            st.session_state["selected_observation"] = 0
            st.session_state["toresu_inspections"] = inspections

    inspections = st.session_state.get("toresu_inspections", [])
    if not inspections:
        st.info("Start the Toresu API, upload a PDF, and select Process document.")
        return

    st.download_button(
        "Download all PDFs and CSVs (ZIP)",
        data=_inspection_zip(inspections),
        file_name="toresu_exports.zip",
        mime="application/zip",
        key="all-pdfs-csv-zip",
    )
    for item in inspections:
        if item.get("error"):
            st.error(f"Could not process {item['pdf_name']}: {item['error']}")
    _render_metrics_dashboard(inspections)
    if len(inspections) > 1:
        selected_document = st.selectbox(
            "Inspected PDF",
            range(len(inspections)),
            format_func=lambda index: inspections[index]["pdf_name"],
            key="selected_inspection_document",
        )
    else:
        selected_document = 0
    inspection = inspections[selected_document]
    if inspection.get("job") is None:
        st.info("This PDF is included in the ZIP, but processing did not complete.")
        return
    job = inspection["job"]
    result = inspection["result"]
    st.subheader(f"Job {job['job_id']} — {job['state']}")
    for stage in job["stages"]:
        if stage["state"] == "failed":
            st.error(f"{stage['name']}: {stage['error_message'] or stage['error_code']}")
    for warning in _sequence(result.get("warnings")):
        st.warning(warning)

    page_count = _pdf_page_count(inspection["pdf_bytes"])
    observations = _records(result.get("observations"))

    def navigate_to_observation() -> None:
        index = st.session_state["selected_observation"]
        if index >= len(observations):
            return
        observation = observations[index]
        spans = [
            *(
                span
                for field in ("source_spans", "row_label_source_spans", "unit_source_spans")
                for span in _records(observation.get(field))
            )
        ]
        page_numbers = [
            page_number
            for span in spans
            if (page_number := _index(span.get("page_number"))) is not None
        ]
        if page_numbers:
            st.session_state["inspection_page"] = page_numbers[0]

    def observation_label(item: dict[str, Any]) -> str:
        period = _mapping(item.get("period"))
        return (
            f"{item.get('row_label_raw') or 'Unlabeled'} · "
            f"{period.get('raw_label') or 'Unknown period'} · "
            f"{item.get('raw_value') or item.get('value_kind')} "
            f"({item.get('value_status', 'unknown')})"
        )
    selected = None
    if observations:
        selected_index = st.selectbox(
            "Financial observation",
            range(len(observations)),
            format_func=lambda index: observation_label(observations[index]),
            key="selected_observation",
            on_change=navigate_to_observation,
        )
        selected = observations[selected_index]

    st.session_state.setdefault("inspection_page", 1)
    page_number = st.number_input(
        "Page",
        min_value=1,
        max_value=page_count,
        step=1,
        key="inspection_page",
    )
    summary = _page_summary(result, int(page_number))
    st.caption(
        f"Page {page_number}: {summary['tables']} tables, "
        f"{summary['observations']} observations, {summary['native_text_spans']} native-text "
        f"evidence spans, {summary['ocr_spans']} OCR evidence spans."
    )

    left, right = st.columns([3, 2])
    with left:
        show_evidence = st.checkbox("Show extracted evidence", value=True)
        show_structure = st.checkbox("Show inferred table/cell boundaries", value=True)
        st.image(
            render_page(
                inspection["pdf_bytes"],
                int(page_number),
                result,
                selected_observation=selected,
                show_all_evidence=show_evidence,
                show_structure=show_structure,
            ),
            caption="Blue = native PDF text; orange = OCR; green = table boundary; red = inferred cell boundary.",
            use_container_width=True,
        )
    with right:
        if selected:
            st.markdown("### Observation")
            st.json(
                {
                    key: selected.get(key)
                    for key in (
                        "row_label_raw",
                        "metric_candidates",
                        "metric_status",
                        "raw_value",
                        "parsed_value",
                        "raw_unit_declaration",
                        "normalized_value",
                        "normalized_unit",
                        "value_kind",
                        "value_status",
                        "period",
                        "source_spans",
                        "row_label_source_spans",
                        "unit_source_spans",
                        "issue",
                    )
                }
            )
        st.markdown("### Reconstructed tables")
        for table_index, table in enumerate(_records(result.get("tables")), start=1):
            with st.expander(
                f"Table {table_index} · page {table['page_number']} · {table['status']}"
            ):
                rows = _table_rows(table)
                st.dataframe(rows, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download CSV",
                    data=_table_csv(rows),
                    file_name=_table_csv_filename(
                        table.get("statement_title_raw"),
                        table.get("table_id"),
                        table_index,
                    ),
                    mime="text/csv",
                    key=f"table-csv-{table_index}",
                )
                issues = [
                    issue
                    for issue in _sequence(table.get("issues"))
                    if isinstance(issue, str)
                ]
                if issues:
                    st.warning("; ".join(issues))
        if result.get("stage_errors"):
            st.markdown("### Processing errors")
            st.json(result["stage_errors"])
        document_report = _mapping(result.get("document_report"))
        if document_report:
            st.markdown("### Document-level intelligence")
            st.json(
                {
                    "cross_statement_links": document_report.get(
                        "cross_statement_links"
                    ),
                    "cross_period_comparisons": document_report.get(
                        "cross_period_comparisons"
                    ),
                    "consistency_findings": document_report.get(
                        "consistency_findings"
                    ),
                    "unresolved_ambiguities": document_report.get(
                        "unresolved_ambiguities"
                    ),
                    "limitations": document_report.get("limitations"),
                }
            )


if __name__ == "__main__":
    run()
