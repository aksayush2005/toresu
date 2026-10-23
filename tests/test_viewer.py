import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest
from toresu.viewer import (
    _inspection_zip,
    _mismatch_provenance,
    _operational_metrics,
    _page_summary,
    _read_reference_csv,
    _render_metrics_dashboard,
    _table_csv,
    _table_csv_filename,
    _table_rows,
    render_page,
    st as streamlit,
)


def _pdf_bytes():
    document = pymupdf.open()
    document.new_page(width=200, height=100)
    payload = document.tobytes()
    document.close()
    return payload


def test_viewer_script_imports_evaluation_without_pythonpath(tmp_path):
    viewer_path = Path(__file__).parents[1] / "src" / "toresu" / "viewer.py"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy; "
                f"namespace = runpy.run_path({str(viewer_path)!r}, "
                "run_name='viewer_import_probe'); "
                "assert namespace['evaluate_table_reference'].__module__ "
                "== 'toresu.evaluation'"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_metrics_dashboard_initializes_when_no_document_has_been_processed(monkeypatch):
    class MetricColumn:
        def __init__(self):
            self.metrics = []

        def metric(self, label, value):
            self.metrics.append((label, value))

    columns = [MetricColumn() for _ in range(4)]
    messages = []
    dashboard_stub = SimpleNamespace(
        header=lambda *_args, **_kwargs: None,
        caption=lambda *_args, **_kwargs: None,
        columns=lambda count: columns[:count],
        subheader=lambda *_args, **_kwargs: None,
        dataframe=lambda *_args, **_kwargs: None,
        info=messages.append,
    )
    monkeypatch.setattr(streamlit, "header", dashboard_stub.header)
    monkeypatch.setattr(streamlit, "caption", dashboard_stub.caption)
    monkeypatch.setattr(streamlit, "columns", dashboard_stub.columns)
    monkeypatch.setattr(streamlit, "subheader", dashboard_stub.subheader)
    monkeypatch.setattr(streamlit, "dataframe", dashboard_stub.dataframe)
    monkeypatch.setattr(streamlit, "info", dashboard_stub.info)

    _render_metrics_dashboard([])

    assert [column.metrics for column in columns] == [
        [("Documents processed", "0/0")],
        [("Pages processed", 0)],
        [("Tables detected", 0)],
        [("Tables reconstructed", 0)],
    ]
    assert messages == [
        "Reference accuracy is Not evaluated because no document completed."
    ]


def test_render_page_draws_evidence_and_structure_without_changing_dimensions():
    payload = _pdf_bytes()
    result = {
        "tables": [
            {
                "page_number": 1,
                "bounding_box": (10, 10, 150, 80),
                "rows": [
                    {
                        "label_cell": {
                            "cell_boundary": (10, 30, 60, 50),
                        },
                        "cells": [],
                    }
                ],
            }
        ],
        "observations": [
            {
                "source_spans": [
                    {
                        "raw_text": "売上高",
                        "page_number": 1,
                        "bounding_box": [12, 32, 45, 45],
                        "extraction_source": "pdf_text",
                    }
                ]
            }
        ],
    }

    image = render_page(payload, 1, result)

    assert image.size == (334, 167)
    colors = {
        color
        for count, color in image.getcolors(maxcolors=image.width * image.height) or []
        if count > 0
    }
    assert (22, 138, 173) in colors
    assert (178, 58, 72) in colors


def test_page_summary_and_table_rows_expose_inspection_data():
    result = {
        "tables": [{"page_number": 2}],
        "observations": [
            {
                "source_spans": [
                    {
                        "raw_text": "1,000",
                        "page_number": 2,
                        "bounding_box": [1, 2, 3, 4],
                        "extraction_source": "ocr",
                    }
                ]
            }
        ],
    }

    summary = _page_summary(result, 2)

    assert summary == {
        "evidence_spans": 1,
        "native_text_spans": 0,
        "ocr_spans": 1,
        "tables": 1,
        "observations": 1,
    }
    assert _table_rows(
        {
            "columns": [
                {
                    "column_index": 0,
                    "header_path": [{"raw_text": "2025年度"}],
                }
            ],
            "rows": [
                {
                    "label_cell": {"raw_text": "売上高"},
                    "cells": [
                        {
                            "column_index": 0,
                            "raw_text": "1,000",
                            "status": "resolved",
                        }
                    ],
                    "hierarchy_status": "resolved",
                }
            ],
        }
    ) == [
        {
        "row": "売上高",
        "2025年度": "1,000",
        "status": "resolved",
        "hierarchy_status": "resolved",
        }
    ]


def test_operational_metrics_count_processing_rows_cells_and_timing():
    metrics = _operational_metrics(
        [
            {
                "pdf_name": "決算資料.pdf",
                "pdf_bytes": _pdf_bytes(),
                "processing_seconds": 2.0,
                "job": {"state": "completed"},
                "result": {
                    "tables": [
                        {
                            "table_id": "p1-t1",
                            "page_number": 1,
                            "columns": [
                                {
                                    "column_index": 0,
                                    "header_path": [{"raw_text": "2025年度"}],
                                }
                            ],
                            "rows": [
                                {
                                    "label_cell": {
                                        "raw_text": "売上高",
                                        "status": "inferred",
                                    },
                                    "cells": [
                                        {
                                            "column_index": 0,
                                            "raw_text": "100",
                                            "status": "inferred",
                                        }
                                    ],
                                },
                                {
                                    "label_cell": None,
                                    "cells": [
                                        {
                                            "column_index": 0,
                                            "raw_text": None,
                                            "status": "unresolved",
                                        }
                                    ],
                                },
                            ],
                        }
                    ]
                },
            }
        ]
    )

    assert metrics["documents_processed"] == 1
    assert metrics["pages_processed"] == 1
    assert metrics["tables_detected"] == metrics["tables_reconstructed"] == 1
    document = metrics["documents"][0]
    assert document["processing seconds"] == 2
    assert document["seconds per page"] == 2
    assert document["unresolved cells"] == 1
    assert document["inferred cells"] == 2
    assert document["unresolved rows"] == 1
    assert document["inferred rows"] == 1


def test_reference_csv_loader_accepts_bom_and_preserves_japanese():
    headers, rows = _read_reference_csv(
        '項目,2025年度\n売上高,"1,000"\n'.encode("utf-8-sig")
    )

    assert headers == ["項目", "2025年度"]
    assert rows == [{"項目": "売上高", "2025年度": "1,000"}]


def test_mismatch_provenance_keeps_page_coordinates_and_source_text():
    enriched = _mismatch_provenance(
        {
            "table_id": "p4-t2",
            "page_number": 4,
            "rows": [
                {
                    "cells": [
                        {
                            "column_index": 0,
                            "source_spans": [
                                {
                                    "raw_text": "△20",
                                    "page_number": 4,
                                    "bounding_box": [10, 20, 30, 40],
                                }
                            ],
                        }
                    ]
                }
            ],
        },
        [{"category": "numeric_sign_mismatch", "row_index": 1, "column_index": 0}],
    )

    assert enriched[0]["page"] == 4
    assert enriched[0]["table_id"] == "p4-t2"
    assert enriched[0]["source_spans"] == [
        {
            "raw_text": "△20",
            "page_number": 4,
            "bounding_box": [10, 20, 30, 40],
        }
    ]


@pytest.mark.parametrize("label_cell", [None, "missing"])
def test_table_rows_accept_missing_or_null_label_cell(label_cell):
    row = {
        "cells": [{"column_index": 0, "raw_text": "1,000"}],
        "hierarchy_status": "resolved",
    }
    if label_cell != "missing":
        row["label_cell"] = label_cell

    assert _table_rows(
        {
            "columns": [{"column_index": 0, "header_path": None}],
            "rows": [row],
        }
    ) == [
        {
            "row": None,
            "Column 1": "1,000",
            "status": "unresolved",
            "hierarchy_status": "resolved",
        }
    ]


def test_viewer_ignores_malformed_optional_geometry_without_losing_text_evidence():
    payload = _pdf_bytes()
    result = {
        "tables": [
            {
                "page_number": 1,
                "bounding_box": None,
                "rows": [
                    {
                        "label_cell": None,
                        "cells": None,
                    },
                    {
                        "label_cell": {
                            "raw_text": "売上高",
                            "cell_boundary": [10, None, 60, 50],
                        },
                        "cells": [],
                    },
                ],
            }
        ],
        "observations": [
            {
                "source_spans": [
                    {
                        "raw_text": "売上高",
                        "page_number": 1,
                        "bounding_box": [12, 32, 45, 45],
                        "extraction_source": "pdf_text",
                    },
                    {
                        "raw_text": "2025年度",
                        "page_number": 1,
                        "bounding_box": None,
                        "extraction_source": "pdf_text",
                    },
                    None,
                ],
                "period": None,
            },
            None,
        ],
    }

    image = render_page(payload, 1, result)

    assert image.size == (334, 167)
    assert _page_summary(result, 1)["evidence_spans"] == 2
    assert _table_rows(result["tables"][0])[1]["row"] == "売上高"


def test_table_rows_accept_null_optional_collections_and_malformed_indices():
    assert _table_rows({"columns": None, "rows": None}) == []
    assert _table_rows(
        {
            "columns": [
                {"column_index": None, "header_path": [{"raw_text": None}]},
            ],
            "rows": [
                {
                    "label_cell": {"raw_text": None},
                    "cells": None,
                    "hierarchy_status": None,
                },
                {
                    "cells": [
                        {"column_index": "invalid", "raw_text": "unmapped"},
                    ],
                },
            ],
        }
    ) == [
        {
            "row": None,
            "Column 1": None,
            "Unassigned values": None,
            "status": "unresolved",
            "hierarchy_status": None,
        },
        {
            "row": None,
            "Column 1": None,
            "Unassigned values": "unmapped",
            "status": "unresolved",
            "hierarchy_status": None,
        },
    ]


def test_table_csv_preserves_japanese_columns_rows_and_empty_cells():
    rows = _table_rows(
        {
            "columns": [
                {"column_index": 0, "header_path": [{"raw_text": "2025年度"}]},
                {"column_index": 1, "header_path": [{"raw_text": "2024年度"}]},
            ],
            "rows": [
                {
                    "label_cell": {"raw_text": "売上高"},
                    "cells": [
                        {"column_index": 0, "raw_text": "1,000", "status": "resolved"},
                        {"column_index": 1, "raw_text": None, "status": "unresolved"},
                    ],
                    "hierarchy_status": "resolved",
                },
                {
                    "label_cell": {"raw_text": "営業利益"},
                    "cells": [
                        {"column_index": 0, "raw_text": "250", "status": "resolved"},
                    ],
                    "hierarchy_status": "resolved",
                },
            ],
        }
    )

    csv_bytes = _table_csv(rows)

    assert csv_bytes.startswith(b"\xef\xbb\xbf")
    assert csv_bytes.decode("utf-8-sig").splitlines() == [
        "row,2025年度,2024年度,status,hierarchy_status",
        "売上高,\"1,000\",,incomplete,resolved",
        "営業利益,250,,incomplete,resolved",
    ]


def test_table_csv_empty_table_is_bom_only():
    assert _table_csv([]) == b"\xef\xbb\xbf"


def test_table_rows_distinguish_extraction_status_from_hierarchy():
    rows = _table_rows(
        {
            "columns": [
                {"column_index": 0, "header_path": [{"raw_text": "前年度"}]},
                {"column_index": 1, "header_path": [{"raw_text": "当年度"}]},
                {
                    "column_index": 2,
                    "header_path": [{"raw_text": "注記"}],
                    "is_auxiliary": True,
                },
            ],
            "rows": [
                {
                    "label_cell": {"raw_text": "項目"},
                    "cells": [
                        {"column_index": 0, "raw_text": "1", "status": "resolved"},
                        {"column_index": 1, "raw_text": "2", "status": "resolved"},
                        {
                            "column_index": 2,
                            "raw_text": "注記文字",
                            "status": "resolved",
                        },
                    ],
                    "hierarchy_status": "unresolved",
                },
                {
                    "label_cell": {"raw_text": "小見出し"},
                    "cells": [
                        {"column_index": 0, "raw_text": None, "status": "unresolved"},
                        {"column_index": 1, "raw_text": None, "status": "unresolved"},
                    ],
                    "hierarchy_status": "inferred",
                },
                {
                    "label_cell": {"raw_text": "不足行"},
                    "cells": [
                        {"column_index": 0, "raw_text": "3", "status": "resolved"},
                        {"column_index": 1, "raw_text": None, "status": "unresolved"},
                    ],
                    "hierarchy_status": "inferred",
                },
            ],
        }
    )

    assert [row["status"] for row in rows] == ["resolved", "heading", "incomplete"]
    assert [row["hierarchy_status"] for row in rows] == [
        "unresolved",
        "inferred",
        "inferred",
    ]
    assert rows[0]["注記"] == "注記文字"


def test_inspection_zip_includes_each_source_pdf_and_table_csv():
    inspections = [
        {
            "pdf_name": "財務資料.pdf",
            "pdf_bytes": b"source-pdf-one",
            "result": {
                "tables": [
                    {
                        "table_id": "p1-t1",
                        "statement_title_raw": "連結貸借対照表",
                        "columns": [
                            {
                                "column_index": 0,
                                "header_path": [{"raw_text": "2025年度"}],
                            }
                        ],
                        "rows": [
                            {
                                "label_cell": {"raw_text": "現金及び預金"},
                                "cells": [
                                    {
                                        "column_index": 0,
                                        "raw_text": "1,000",
                                        "status": "resolved",
                                    }
                                ],
                                "hierarchy_status": "unresolved",
                            }
                        ],
                    }
                ]
            },
        },
        {
            "pdf_name": "財務資料.pdf",
            "pdf_bytes": b"source-pdf-two",
            "result": {"tables": []},
        },
    ]

    with zipfile.ZipFile(io.BytesIO(_inspection_zip(inspections))) as archive:
        names = archive.namelist()
        assert "pdfs/財務資料.pdf" in names
        assert "pdfs/財務資料-2.pdf" in names
        assert archive.read("pdfs/財務資料.pdf") == b"source-pdf-one"
        assert archive.read("pdfs/財務資料-2.pdf") == b"source-pdf-two"
        csv_path = next(name for name in names if name.endswith(".csv"))
        assert archive.read(csv_path).decode("utf-8-sig").splitlines() == [
            "row,2025年度,status,hierarchy_status",
            '現金及び預金,"1,000",resolved,unresolved',
        ]


@pytest.mark.parametrize(
    ("table_name", "table_id", "table_index", "expected"),
    [
        ("損益計算書/2025年度", "income-table", 1, "損益計算書_2025年度.csv"),
        ("  ../bad:name?  ", None, 2, "_bad_name_.csv"),
        (None, "table-identifier", 3, "table-identifier.csv"),
        ("  ", None, 4, "table-4.csv"),
    ],
)
def test_table_csv_filename_is_descriptive_and_sanitized(
    table_name,
    table_id,
    table_index,
    expected,
):
    assert _table_csv_filename(table_name, table_id, table_index) == expected
