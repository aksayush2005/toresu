# Toresu — Japanese Financial Document Intelligence

See demonstration:[https://youtu.be/EicxWVr59dg]

Toresu is a research-oriented document-intelligence project for reconstructing **Japanese-language financial tables** from PDFs into structured, traceable data. Its initial target is the financial-statement tables found in Japanese corporate disclosures, especially annual securities reports (*有価証券報告書*) and consolidated financial statements.

Toresu combines native PDF text extraction, Japanese OCR, table-structure recognition, and a vision-language model (VLM). Its defining component is a **Japanese Financial Context Compiler**: it extracts document-specific evidence and applies Japanese financial-document conventions before asking the VLM to interpret a table.

> **Design principle:** preserve the source, interpret Japanese financial notation in context, and flag uncertainty instead of silently inventing values.

## Why Toresu?

Japanese financial tables contain ambiguities that generic PDF-to-Markdown pipelines do not specifically address. A parser must preserve Japanese account labels and table hierarchy, relate headers such as `前連結会計年度` and `当連結会計年度` to the correct periods, recognize unit declarations such as `千円` and `百万円`, and handle financial notation such as `△` or `▲` without confusing it with ordinary punctuation. Important context can also appear in Japanese section headings, notes, footnotes, or explanatory text surrounding a table.

Toresu treats these details as part of the extraction architecture—not as afterthoughts in a generic prompt. It builds a Japanese financial context package for each table, supplies that evidence together with the table image and structural candidates, and validates the resulting reconstruction against the document.

## Scope

**Primary scope**

- Japanese-language financial statements in PDF documents.
- Balance sheets (`貸借対照表`), income statements (`損益計算書`), cash-flow statements (`キャッシュ・フロー計算書`), and statements of changes in equity (`株主資本等変動計算書`) when present in the input.
- Tables with nested account categories, multi-row headers, merged cells, comparative reporting periods, units, notes, and footnote markers.
- Documents that contain a native text layer, scanned pages, or a mixture of both.

**Source handling**

PDF is the primary input. If a matching HTML or XBRL source is provided or can legitimately be obtained, Toresu may use it as supplementary evidence. It must not assume that every PDF table has a directly usable structured counterpart, or that an auxiliary source always matches the rendered table exactly.

Toresu is not intended to be a generic parser for every language or document type. Other use cases may become possible, but the first implementation and evaluation should focus on Japanese corporate financial disclosures.

## Japanese-Specific Design

### Japanese Financial Context Compiler

For each table, the compiler assembles a structured context package from document evidence and Japanese financial-document rules. It is designed to identify or preserve:

- **Statement identity:** Japanese table titles and section headings, including consolidated versus non-consolidated statements where the document makes the distinction.
- **Fiscal-period headers:** expressions such as `前連結会計年度`, `当連結会計年度`, `前事業年度`, and `当事業年度`, including their relationship to explicit dates and column positions.
- **Units and scale:** declarations such as `円`, `千円`, `百万円`, and `億円`; the raw declaration is retained, while a normalized scale is recorded separately when unambiguous.
- **Japanese date formats:** western-calendar dates such as `2025年3月31日` and Japanese-era dates such as `令和7年3月31日`, alongside slash-separated formats. Conversion to ISO dates is only performed when the source date is explicit and unambiguous.
- **Financial notation:** symbols such as `△` and `▲`, parenthesized values, and different dash-like characters. Their meaning is interpreted using the document and local table context; a dash is not automatically converted to zero.
- **Japanese text normalization:** full-width and half-width character variants and Japanese punctuation. A normalized search or comparison form may be created, but the original source string is always retained.
- **Account labels and hierarchy:** Japanese labels such as `資産の部`, `流動資産`, `負債の部`, and `純資産の部`, while preserving the original text and detected parent-child relationships.
- **Notes and footnotes:** markers such as `※`, `（注）`, and numbered notes, with links to relevant explanatory text when it can be extracted.

These are extraction rules and hypotheses, not permission to override the source. For example, a `△` symbol may often indicate a negative amount in Japanese financial statements, but the parser must still retain the raw value and flag cases where the interpretation is unclear.

### Context-Aware Japanese VLM Prompting

Toresu does not send the same generic prompt for every table. It creates a table-specific prompt from:

1. The Japanese table image and detected cell geometry.
2. The original Japanese title, headings, account labels, and nearby explanatory notes.
3. Candidate header paths and fiscal-period associations.
4. Verified units and date information, with links back to their source.
5. Japanese-specific normalization rules and a strict output schema.

The VLM is instructed to preserve Japanese source text, avoid translating labels unless a separate translation field is requested, keep raw and normalized values separate, and report ambiguity rather than guessing. Context is added only when relevant to the table and question being resolved.

## Architecture

```text
                   Japanese Financial PDF
            (e.g. a Japanese annual securities report)
                              |
                              v
                       PDF Ingestion
                              |
              +---------------+----------------+
              |                                |
              v                                v
     Native Text + Coordinates          Page Rendering
            (PyMuPDF)                   (PyMuPDF/PIL)
              |                                |
              v                                v
    Japanese Text Extraction          OCR fallback for pages
                                        without usable native text
                                          (RapidOCR + ONNX Runtime)
              |                                |
              +---------------+----------------+
                              |
                              v
                Japanese Document Analysis
       statement type, headings, fiscal periods, units,
       Japanese date formats, notes, account labels
                              |
                              v
             JAPANESE FINANCIAL CONTEXT COMPILER
       context retrieval + header paths + cell geometry
       notation rules + evidence links + output constraints
                              |
                              v
              Context-Aware Gemini VLM Inference
           table image + Japanese text + structure candidates
                              |
                              v
                 Canonical Table Representation
            raw Japanese strings + normalized values
                 row hierarchy + cell coordinates
                              |
                              v
                  Japanese-Aware Validation
       period/column checks, unit checks, numeric notation,
       hierarchy checks, footnote links, accounting warnings
                              |
                    +---------+---------+
                    |                   |
                    v                   v
               Validated Cells      Uncertain Cells
                                        |
                                        v
                              Targeted OCR / Crop / Context
                                        |
                                        v
                              JSON / CSV / HTML Export
```

## Processing Components

| Component | Responsibility |
|---|---|
| **Streamlit** | Web interface for PDF upload, page preview, table review, warnings, and exports. |
| **PyMuPDF** | Extracts native PDF text and word coordinates, and renders pages or selected regions. |
| **PDF ingestion** | Preserves native and OCR text as separate page-numbered, source-tagged spans with bounding boxes; OCR is used only when a page lacks sufficient native text. |
| **RapidOCR** | Optional CPU OCR engine used by PDF ingestion for scanned or text-sparse pages, using ONNX Runtime and an explicitly configured Japanese-capable recognition model. |
| **Table reconstruction** | Deterministically groups existing page spans by geometry, detects numeric columns and headers, and preserves unresolved assignments and provenance; it does not execute OCR. |
| **Financial normalization** | Normalizes supported Japanese values, explicitly evidenced units, exact-label metric mappings, and source-supported periods; does not validate accounting relationships. |
| **Financial validation** | Evaluates evidence-complete Decimal equations and caller-configured cross-period warning thresholds; never changes normalized values. |
| **Gemini API** | Interprets table images with table-specific Japanese context and helps resolve visual or structural ambiguities. |
| **Japanese Financial Context Compiler** | Identifies statement context, fiscal periods, units, notation, Japanese labels, footnotes, and relevant evidence. |
| **Python validation** | Checks schemas, numeric normalization, row/column assignments, period associations, hierarchy, and supported accounting relationships. |
| **Optional HTML/XBRL evidence** | May be used to cross-check an aligned disclosure when the matching source is available; it is not assumed to exist for every table. |

When a PDF contains a reliable text layer, native text extraction and coordinates from PyMuPDF should be preferred. RapidOCR is an explicit fallback for scanned pages or selected regions where native text is absent or requires OCR; it should not be run unconditionally on digitally generated pages. VLM inference is complementary; it does not replace all conventional extraction steps.
Table reconstruction consumes the existing `PDFDocument` and `FinancialDocumentContext`; it never invokes OCR or a language model. It uses repeated numeric right-edge alignment as a column signal, text-span coverage for merged headers, and label indentation as a tentative parent-row signal. Geometry that does not establish a relationship remains unresolved or ambiguous.

## Evidence-Grounded Reconstruction

Every extracted cell should be traceable to its source. Toresu keeps the original text and its normalized interpretation separate, associates cells with page coordinates, and records which component produced each candidate. Where multiple sources disagree, the system should retain the conflict for validation rather than silently selecting a plausible value.

For example, different errors require different corrections:

| Detected issue | Preferred response |
|---|---|
| An unclear Japanese character or digit | Re-run OCR on a higher-resolution crop and compare with native text if available. |
| A value assigned to the wrong reporting period | Reconstruct the parent-header path and check fiscal-period context. |
| An ambiguous unit declaration | Retrieve the relevant table heading, nearby note, or source context. |
| A missing account-row relationship | Re-evaluate cell geometry, indentation, and Japanese section/account labels. |
| A disagreement between PDF text and the VLM | Preserve both candidates, inspect their source evidence, and flag unresolved conflicts. |
| A subtotal does not match its components | Emit a warning; do not overwrite the value to force arithmetic agreement. |

## Output Format

Toresu uses a canonical table representation before exporting to CSV or HTML. This preserves information that flat formats may lose, including merged cells, multi-level headers, and nested account rows.

Illustrative JSON schema:

```json
{
  "document_id": "report_001",
  "language": "ja",
  "page_number": 1,
  "table_title_raw": "連結貸借対照表",
  "unit": {
    "raw_text": "千円",
    "normalized_unit": "thousand_yen",
    "source": "document_text"
  },
  "columns": [
    {
      "column_id": "col_1",
      "header_path_raw": ["前連結会計年度"],
      "reporting_date_raw": "2025年3月31日",
      "reporting_date_iso": "2025-03-31"
    },
    {
      "column_id": "col_2",
      "header_path_raw": ["当連結会計年度"],
      "reporting_date_raw": "2026年3月31日",
      "reporting_date_iso": "2026-03-31"
    }
  ],
  "rows": [
    {
      "row_id": "row_1",
      "label_raw": "現金及び預金",
      "parent_row_id": "row_current_assets",
      "cells": [
        {
          "column_id": "col_1",
          "raw_text": "2,002,429",
          "normalized_value": 2002429,
          "bounding_box": [100, 200, 240, 225],
          "extraction_source": "pdf_text",
          "validation_status": "pending"
        }
      ]
    }
  ],
  "warnings": []
}
```

The example illustrates the intended schema, not a claim that Toresu has already extracted or validated that table. The implementation should also support cell spans, multiple candidates, confidence estimates, footnote links, and source provenance. Original Japanese strings must remain available even when a normalized form is created.

## Validation and Evaluation

Evaluation should use manually verified Japanese financial tables from documents that are not duplicated across evaluation splits. Where a matching HTML or XBRL representation is available, it may help create or cross-check references after confirming that the table, period, scope, and units align.

| Metric | What it evaluates |
|---|---|
| Japanese character error rate (CER) | Accuracy of recognized Japanese text, including kanji, kana, punctuation, and digits. |
| Cell text exact match | Whether the original cell text matches the reference transcription. |
| Numeric exact match | Whether normalized values are correct under documented notation and unit rules. |
| Header/period accuracy | Whether multi-row headers and fiscal periods are associated with the correct columns. |
| Row hierarchy accuracy | Whether parent-child account relationships are preserved. |
| Unit and date accuracy | Whether Japanese unit declarations and explicit dates are normalized correctly. |
| Merged-cell accuracy | Whether merged regions and cell spans are reconstructed correctly. |
| Table completeness | Whether expected rows and cells have been recovered. |
| Latency and inference cost | The time and API usage required per document or table. |

Experiments should compare native-text extraction, Japanese OCR/table recognition, VLM-only extraction, context-aware VLM extraction, and validation with targeted reprocessing. All configurations should use the same test documents, normalization rules, and scoring scripts. Results must be measured; no accuracy improvement is assumed in advance.

## Technology Stack

- **Language:** Python
- **Web interface:** Streamlit
- **PDF text and rendering:** PyMuPDF
- **Japanese OCR fallback:** RapidOCR with ONNX Runtime on CPU; Japanese recognition uses the PP-OCRv6 small multilingual model route (`japan`), not RapidOCR's default Chinese/English recognition route
- **Multimodal inference:** Gemini API, configured through environment variables
- **Data processing:** Pandas and Python validation rules
- **Testing:** Pytest
- **Exports:** JSON, CSV, and HTML

The initial implementation is designed to work without a local NVIDIA GPU. PDF processing and OCR can run on CPU, although performance depends on the selected OCR models, document complexity, and hardware. VLM inference is performed through the API.

## Getting Started

### Requirements

- Python 3.13.15 (verified with the selected dependency set on Ubuntu 26.04 x86_64)
- `uv`, for installing the pinned Python runtime and creating the virtual environment
- Internet access for package/model downloads and, when implemented, API inference
- A Gemini API key only when Gemini inference is implemented

RapidOCR recommends ONNX Runtime for CPU inference. Its model documentation lists `japan` for PP-OCRv6 small multilingual recognition; the tiny model does not include Japanese. Toresu configures the small model route explicitly rather than relying on RapidOCR's Chinese/English default. Model assets may be fetched on first OCR use, so configuration tests do not initialize OCR or download models.

### Installation

```bash
uv python install 3.13.15
uv venv --python 3.13.15 --seed .venv
```

Activate the virtual environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the dependencies from the single manifest:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency manifest installs `rapidocr` and the CPU `onnxruntime` engine. OCR remains optional at runtime: use PyMuPDF text and coordinates first, and call the OCR adapter only for scanned pages or selected regions that need recognition. Verify Japanese inference on representative financial-document samples before evaluating OCR accuracy.

### Environment Variables

Copy `.env.example` to `.env`. Both Gemini settings may remain empty during local development:

```dotenv
GEMINI_API_KEY=
GEMINI_MODEL=
```

The model identifier is configurable and is not assumed to be valid. The settings loader accepts both values empty, or requires both to be set. Do not commit `.env` or API credentials; `.env` is already ignored by Git. Select a model based on current API availability, supported image/PDF inputs, cost, and evaluation results.

### Table Reconstruction

Reconstruct source-backed table candidates from the existing ingestion and context results:

```python
from toresu.context_compiler import compile_financial_context
from toresu.pdf_ingestion import ingest_pdf
from toresu.table_reconstruction import reconstruct_tables

document = ingest_pdf("report.pdf")
context = compile_financial_context(document)
result = reconstruct_tables(document, context)
```

The result keeps source-span provenance separate from inferred geometric cell boundaries. Missing values have no contributing spans and remain unresolved; competing column matches remain ambiguous. Periods are attached only when their original context evidence is also present in the header path. Unit candidates are retained from compiler context without converting values. Row indentation, horizontal alignment, and vertical grouping are deterministic heuristics, not verified accounting structure.

### Financial Normalization

Normalize reconstructed values and map row labels with the deterministic registry:

```python
from toresu.financial_normalization import normalize_financial_tables

normalized = normalize_financial_tables(result, context)
for observation in normalized.tables[0].observations:
  print(
    observation.metric.raw_label,
    observation.metric.candidates,
    observation.period.raw_label,
    observation.value.parsed.raw_text,
    observation.value.normalized_value,
    observation.value.normalized_unit,
  )
```

Supported numeric notation includes comma-grouped and full-width digits/punctuation, `△`/`▲` negatives, leading signs, parenthesized negatives, decimals, and `%` suffixes. Zero, blank, dash variants, unavailable markers, and unsupported notation remain distinct. Parentheses combined with a negative prefix, malformed grouping, and unrecognized strings are not normalized. Percentages remain percentage values; they are not divided by 100.

Explicit `円`, `千円`, `百万円`, and `億円` declarations normalize amounts to yen using `Decimal`; the raw amount and declaration are retained. Conversion is blocked when a table unit is missing or conflicting. Metric mapping is an exact normalized-label lookup in an extensible `MetricRegistry`; unregistered and multiply mapped labels remain unresolved or ambiguous. Fiscal years, quarters, and dates are normalized only from supported explicit header evidence. Relative prior/current fiscal-year labels are preserved as relative periods without deriving calendar dates.

This phase emits normalized observations with source evidence; accounting consistency checks are handled by the separate financial validation module.

### Financial Validation

Run deterministic checks on normalized observations:

```python
from decimal import Decimal

from toresu.financial_validation import (
  CrossPeriodThreshold,
  MetricConcept,
  ValidationConfig,
  validate_financial_data,
)

report = validate_financial_data(
  normalized,
  config=ValidationConfig(
    absolute_tolerance=Decimal("1"),
    cross_period_thresholds=(
      CrossPeriodThreshold(MetricConcept.REVENUE, Decimal("0.5")),
    ),
  ),
)
for finding in report.findings:
  print(finding.status, finding.rule_id, finding.message)
```

The built-in assets = liabilities + equity rule runs only when required metrics have one compatible reporting period, resolved yen units, and usable mapped values. `absolute_tolerance` is an explicit residual allowance in normalized units; source values are not rounded or rewritten. `AccountingEquation` and the `ValidationRule` protocol allow additional checks, including cross-statement equations, to be supplied explicitly.

Cross-period thresholds are opt-in and produce warnings for review, not failures. A nonzero current value after a zero baseline is reported as a warning because relative change is undefined. Findings retain the normalized observations and links to source tables, cells, text spans, pages, and bounding boxes. The document report also spells out each expected relationship, observed residual and tolerance, and the source-linked normalized inputs used for the check.

No subtotal identities or further cross-statement relationships are assumed by default. Missing/dash values, ambiguous mappings, unresolved periods, or incompatible units prevent a pass and are reported explicitly. Tests use synthetic evidence; validation has not been evaluated against real Japanese filings.

### Tests

Run the configuration tests without a Gemini API key or API calls:

```bash
PYTHONPATH=src python -m pytest
```

### Run the Application

Start the local API in one terminal:

```bash
PYTHONPATH=src uvicorn toresu.api.app:app --host 127.0.0.1 --port 8000
```

Start the visual PDF inspection interface in another terminal:

```bash
streamlit run src/toresu/viewer.py
```

The viewer adds the repository's `src/` directory to its import path when launched from a source checkout, so this command does not require the package to be installed or `PYTHONPATH` to be set. The viewer uploads selected PDFs to the existing local API, polls each job, and displays page renders with provenance spans and inferred table/cell boundaries in distinct colors. Selecting an observation exposes raw and normalized values, units, period, extraction origin, and source coordinates; the page summary distinguishes native text from OCR spans. Each reconstructed table includes a CSV download matching its displayed rows and columns, and a ZIP download contains every selected source PDF and the CSV exports for processed tables.

The table `status` column describes data extraction: `resolved` means the label and populated value cells have resolved source evidence; `incomplete` means one or more expected value cells are blank; `ambiguous` means a value has competing assignments; `unresolved` means required label/value evidence could not be assigned; `heading` marks a label-only row; and `inferred` marks populated cells with inferred assignments. `hierarchy_status` is reported separately and describes parent-row assignment, not whether the row's values were extracted. Aligned auxiliary text columns such as notes are retained with provenance and are not interpreted as numeric values. The viewer does not edit or overwrite pipeline results. Use the default loopback API URL unless the local API is intentionally running on another address.

The Metrics Dashboard reports operational counts for processed documents, pages, table candidates, structurally reconstructed tables, rows, columns, inferred/unresolved cells and rows, and wall-clock processing time when available. Select a processed document and table and upload a UTF-8 reference CSV to calculate ground-truth metrics. Without a reference, accuracy fields are shown as **Not evaluated**. Mismatch rows include table/page identity and available source-span text and coordinates; the original rendering, provenance overlays, and CSV downloads are unchanged.

Reference comparisons use these explicit rules: row labels and headers follow the evaluation framework's Unicode NFKC normalization with whitespace removed; cell exact match trims surrounding whitespace only; numeric accuracy accepts valid comma grouping and `Decimal`-equivalent precision. Dashes, blank cells, zero, `△`, and minus signs are not rewritten or treated as interchangeable. Cell precision/recall/F1 counts non-empty values at their row/column positions; row-label precision/recall/F1 and structural agreement are reported separately. Header order and row order are significant, so period shifts and row shifts are reported as alignment mismatches. Metrics describe only the selected reference comparison, not general extraction accuracy.

### Evaluation and Benchmarking

The evaluation runner invokes the existing deterministic pipeline; it does not generate labels or change production behavior. The checked-in versioned dataset is `evaluation/datasets/v1/`:

```text
evaluation/
├── datasets/v1/
│   ├── manifest.json
│   ├── sources/                 # synthetic PDF layout specs or source PDFs
│   └── annotations/             # ground truth, separate from predictions
└── schemas/annotation-v1.schema.json
```

Each manifest entry has a stable `document_id`, a `source_kind` (`synthetic_pdf` or `pdf`), a source path, and a separate annotation path. Annotation files carry document metadata, annotation completeness, and optional processing metadata. Table labels can include hierarchical column headers, table and column bounds, and cell-level resolved/ambiguous/missing/unresolvable status. Synthetic layouts are rendered to temporary PDFs using PyMuPDF's built-in Japanese font and then run through the same pipeline as a PDF source. Add real documents by storing a legally usable PDF under the dataset directory, setting `source_kind` to `pdf`, and adding a ground-truth annotation file. Do not put sensitive or licensed documents in the repository without permission.

Annotations preserve original text and distinguish parsed and normalized numeric values, units, semantic metric labels, raw and normalized periods, and extraction origin. They can cover multiple tables, pages, and periods. Explicit `exhaustive` flags define whether an empty or partial label set may be used to count false positives; unspecified labels are rejected by schema validation rather than interpreted as negatives. `validation_findings: null` means validation was not annotated, not that no finding was expected. Labeled validation findings include rule ID, status, and optional period. Optional page numbers and bounding boxes enable source localization scoring. The Pydantic loader validates the versioned manifest, annotation, and synthetic-layout structures; the corresponding annotation contract is documented in `evaluation/schemas/annotation-v1.schema.json`.

Run accuracy evaluation or the separate performance benchmark without Gemini credentials:

```bash
env -u GEMINI_API_KEY -u GEMINI_MODEL PYTHONPATH=src .venv-rapidocr/bin/python -m toresu.evaluation evaluate --output-dir /tmp/toresu-evaluation
env -u GEMINI_API_KEY -u GEMINI_MODEL PYTHONPATH=src .venv-rapidocr/bin/python -m toresu.evaluation benchmark --repeats 5 --warmup 1 --output-dir /tmp/toresu-benchmark
```

Use `--dataset PATH` to select another versioned dataset and repeat `--document DOCUMENT_ID` to select a subset. Each command writes a JSON report and a readable text summary. Evaluation reports per-document outcomes, processing failures, stage durations, error categories, prediction counts, and document-macro aggregate metrics. Processing failures remain in the denominator as failures and are not silently dropped; aggregate component metrics are averaged only over documents where that metric is defined, with their support count shown. Label matching uses Unicode NFKC plus whitespace removal; numeric comparisons use exact `Decimal` equality, not float tolerances. CER is character-based. Japanese WER is a deterministic character-token approximation: contiguous Japanese runs are split into Unicode code points, while ASCII alphanumeric runs remain tokens; it is not a morphological word segmentation score. Empty references use a denominator floor of one, yielding zero only for an empty prediction.

The runner reports text CER/WER and extraction source counts; context precision/recall/F1; table counts, row labels, periods, header text, cell observations, and page localization; parsed/normalized value, unit, kind, metric, and period accuracies; validation-status agreement when labeled; and provenance coverage, page accuracy, and bounding-box IoU when box labels are supplied. Geometry is not inferred from annotation absence. Benchmarking separately reports per-document wall and per-page latency, stage latency, successful-run throughput, and Python-traced peak allocations; PDF rendering for synthetic inputs is excluded from timings, warm-up runs are excluded, and native-library allocations are not included in `tracemalloc` figures. Measurements are machine- and workload-specific.

An abbreviated excerpt from the checked-in synthetic dataset's evaluation report:

```json
{
  "dataset_version": "1.0.0",
  "successful_documents": 1,
  "failed_documents": 0,
  "documents": [{
    "document_id": "synthetic-jp-income-v1",
    "source_kind": "synthetic_pdf",
    "success": true,
    "metrics": {
      "text": {"cer": 0.0, "wer": 0.0},
      "financial_validation": null,
      "provenance": {"coverage": 1.0}
    }
  }]
}
```

The starter dataset contains **one synthetic, native-text Japanese income-statement fixture**, generated from a compact PDF layout. It covers context extraction, two reporting periods, exact-label mapping, decimal/unit normalization, negative notation, table reconstruction, and value provenance. It has no real PDFs, scanned/OCR pages, or validation labels; those metrics are therefore unavailable, not perfect. Any scores from this fixture are synthetic pipeline checks only and do not estimate real-document accuracy or statistical significance. The evaluation set must be expanded and independently annotated before drawing broader conclusions.

### Document-Level Financial Report

The completed API result includes a `document_report` when normalization produced observations. It retains observation references and source spans, groups exact resolved metric identities across distinct statement types only when their reporting-period identities are compatible, and reports incompatible units or duplicate candidates instead of selecting a value. Cross-period comparisons are limited to matching period families and statement types, use `Decimal`, report absolute change, and define percentage change as `(current - prior) / abs(prior) * 100`; percentage change is unavailable for a zero prior value. Existing validation findings are copied into the report with provenance references, expected relationships, observed residuals/tolerances, and the normalized inputs used. Unresolved metric/value/period observations remain listed with reasons.

Narrative-to-table reconciliation is not implemented: the current context compiler does not classify narrative figures into typed, period- and unit-linked observations. Restatements are also not inferred. The viewer exposes available cross-statement links, comparisons, consistency findings, and unresolved ambiguities under Document-level intelligence.

### Research Experiment

Run the reproducible research analysis with:

```bash
PYTHONPATH=src python -m toresu.research --dataset evaluation/datasets/v1 --output-dir research-results
```

The report will separate the full pipeline's exact financial-observation accuracy, component metrics, provenance coverage, and measured latency from a deliberately simple native-PDF numeric-token baseline. The token baseline measures raw-number candidate precision/recall and text-extraction latency only; it cannot assign figures to metrics or periods and is not presented as a like-for-like financial-observation system. The current pipeline has no independent switches for provenance-aware or ambiguity-aware decisions, and the starter set has no ambiguous mappings or real documents. Therefore those causal ablations, silent-error comparisons, and any statistical conclusion are explicitly marked not estimable until switchable variants and a broader independent dataset are available.

### API Service

Start the FastAPI service inside the project environment:

```bash
uvicorn toresu.api.app:app --host 127.0.0.1 --port 8000
```

The API processes PDFs asynchronously using the existing pipeline:

| Method | Endpoint | Behavior |
|---|---|---|
| `GET` | `/health` | Liveness response. |
| `GET` | `/ready` | Readiness and available job slots. |
| `GET` | `/v1/config` | Public upload, page, worker, queue, and retention limits. |
| `POST` | `/v1/documents` | Upload a PDF as multipart field `file`; returns `202` and job URLs. |
| `GET` | `/v1/jobs/{job_id}` | Job state and per-stage status/errors. |
| `GET` | `/v1/jobs/{job_id}/result` | Tables, normalized observations, document-level links/comparisons, provenance, and partial stage errors. Returns `409` while active. |
| `GET` | `/v1/jobs/{job_id}/validation` | Validation findings and stage errors. Returns `409` while active. |

`/health` reports process liveness independently of capacity. `/ready` returns `status: "ready"` and `accepting_jobs: true` while at least one job slot is available; when workers and the configured queue are full, it remains a live service but reports `status: "not_ready"`, `accepting_jobs: false`, and zero available slots.

Example upload and polling:

```bash
curl http://127.0.0.1:8000/ready
curl -F 'file=@report.pdf;type=application/pdf' http://127.0.0.1:8000/v1/documents
curl http://127.0.0.1:8000/v1/jobs/<job-id>
curl http://127.0.0.1:8000/v1/jobs/<job-id>/result
```

Example readiness response:

```json
{
  "status": "ready",
  "accepting_jobs": true,
  "available_worker_slots": 10
}
```

The server checks the `%PDF-` signature, then validates readability, encryption, page count, and the configured byte limit. Client filenames and MIME types are not used as paths or trusted for format validation. Error responses use `{ "error_code": "...", "detail": "..." }`. Financial `Decimal` fields serialize as JSON strings to preserve precision. Completed results can include partial upstream artifacts and identify the failed stage.

API settings are read from the project working directory's `.env` file and the process environment; explicit environment variables take precedence:

| Variable | Default | Meaning |
|---|---:|---|
| `TORESU_API_DATA_DIR` | `.toresu-data` | Private local upload, staging, and result directory. |
| `TORESU_API_MAX_UPLOAD_BYTES` | `52428800` | Maximum uploaded file size (50 MiB). |
| `TORESU_API_MAX_PDF_PAGES` | `500` | Maximum PDF pages. |
| `TORESU_API_MAX_WORKERS` | `2` | Concurrent processing workers. |
| `TORESU_API_MAX_QUEUED_JOBS` | `8` | Additional admitted jobs waiting for workers. |
| `TORESU_API_MAX_RETAINED_JOBS` | `100` | Maximum in-memory job records; oldest terminal jobs are pruned. |
| `TORESU_API_VALIDATION_ABSOLUTE_TOLERANCE` | `0` | Decimal residual tolerance in normalized units. |
| `TORESU_API_REVENUE_CHANGE_THRESHOLD` | unset | Optional relative-change warning threshold, such as `0.5` for 50%. |

Job metadata is held in process memory and is **not durable**. Uploads and atomic result JSON live under UUID-named job directories; temporary upload staging is removed after acceptance/rejection, and abandoned API artifacts are cleared on startup. This is a single-process service, not a distributed queue. Do not run multiple workers/processes against the same data directory.

The API has no authentication or authorization. Financial PDFs and results may contain sensitive information; keep the service bound to loopback for development. Before any public or shared deployment, add authentication/authorization, HTTPS, network restrictions, storage encryption/retention controls, and production monitoring. This implementation is not claimed to be production-ready.

## Key Implementation Areas

- `src/toresu/api/`: bounded asynchronous API, job lifecycle, settings, and pipeline serialization.
- `src/toresu/viewer.py`: local Streamlit PDF/provenance viewer.
- `src/toresu/document_intelligence.py`: cross-statement links, cross-period comparisons, and explainable consistency findings.
- `src/toresu/evaluation.py` and `src/toresu/research.py`: versioned evaluation, benchmarking, and the reproducible baseline experiment.
- `evaluation/`: annotation schema and versioned synthetic fixture.
- `tests/`: unit and integration coverage for the pipeline, API, viewer, evaluation, and research reporting.

## References

- [Financial Services Agency of Japan — EDINET overview](https://www.fsa.go.jp/search/20130917.html)
- [Financial Services Agency of Japan — EDINET taxonomy intellectual property statement](https://www.fsa.go.jp/search/EDINET_Taxonomy_Legal_Statement.html)
- [RapidOCR documentation — installation](https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/install/)
- [RapidOCR documentation — model list and language support](https://rapidai.github.io/RapidOCRDocs/main/model_list/)
- [RapidOCR documentation — usage and result coordinates](https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/usage/)
- [NLI Research Institute — Japanese financial statements and negative-number notation](https://www.nli-research.co.jp/report/detail/id=75877?site=nli)

## Status

Toresu is a research-oriented implementation in progress. Implemented: PyMuPDF ingestion with conditional RapidOCR; deterministic Japanese context compilation, table reconstruction, financial normalization, and validation; a bounded local API; a provenance-oriented Streamlit viewer; document-level financial reporting; and versioned evaluation, benchmarking, and research tooling. Validation, evaluation, and research results currently use synthetic evidence only: no real financial PDF has been evaluated, and synthetic scores do not establish real-world accuracy or statistical significance. The benchmark measures this machine and workload only. Narrative-to-table reconciliation, restatement inference, switchable causal ablations, and silent-error-rate analysis are not available. The API has no authentication and uses in-memory job metadata; do not expose it publicly without the documented production controls.
