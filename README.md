# Toresu — Japanese Financial Document Intelligence

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

Cross-period thresholds are opt-in and produce warnings for review, not failures. A nonzero current value after a zero baseline is reported as a warning because relative change is undefined. Findings retain the normalized observations and links to source tables, cells, text spans, pages, and bounding boxes.

No subtotal identities or further cross-statement relationships are assumed by default. Missing/dash values, ambiguous mappings, unresolved periods, or incompatible units prevent a pass and are reported explicitly. Tests use synthetic evidence; validation has not been evaluated against real Japanese filings.

### Tests

Run the configuration tests without a Gemini API key or API calls:

```bash
PYTHONPATH=src python -m pytest
```

### Run the Application

Once `app.py` and the extraction modules are implemented:

```bash
streamlit run app.py
```

Streamlit will print a local URL for the application in the terminal.

## Suggested Repository Structure

```text
toresu/
├── app.py
├── src/
│   └── toresu/
│       ├── __init__.py
│       ├── config.py
│       ├── context_compiler.py
│       ├── financial_normalization.py
│       ├── financial_validation.py
│       ├── pdf_ingestion.py
│       ├── japanese_ocr.py
│       ├── table_detection.py
│       ├── context_compiler.py
│       ├── japanese_finance_rules.py
│       ├── vlm_extraction.py
│       ├── reconstruction.py
│       ├── validation.py
│       └── export.py
├── prompts/
│   └── japanese_financial_table.txt
├── schemas/
│   └── table.schema.json
├── tests/
│   ├── test_config.py
│   ├── test_context_compiler.py
│   ├── test_financial_normalization.py
│   ├── test_financial_validation.py
│   ├── test_japanese_ocr.py
│   ├── test_pdf_ingestion.py
│   ├── test_table_reconstruction.py
│   ├── test_japanese_normalization.py
│   ├── test_period_and_unit_parsing.py
│   └── test_validation.py
├── samples/
│   └── README.md
├── outputs/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

This is a suggested layout; module names can change as the implementation evolves.

## References

- [Financial Services Agency of Japan — EDINET overview](https://www.fsa.go.jp/search/20130917.html)
- [Financial Services Agency of Japan — EDINET taxonomy intellectual property statement](https://www.fsa.go.jp/search/EDINET_Taxonomy_Legal_Statement.html)
- [RapidOCR documentation — installation](https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/install/)
- [RapidOCR documentation — model list and language support](https://rapidai.github.io/RapidOCRDocs/main/model_list/)
- [RapidOCR documentation — usage and result coordinates](https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/usage/)
- [NLI Research Institute — Japanese financial statements and negative-number notation](https://www.nli-research.co.jp/report/detail/id=75877?site=nli)

## Status

Toresu is a research-oriented implementation in progress. Implemented: PyMuPDF PDF ingestion with page-numbered native text spans and conditional lazy RapidOCR; the deterministic Japanese context compiler; geometry-based table candidate reconstruction with source-backed and unresolved cells; Decimal-based value/unit/period normalization; and deterministic financial validation with a balance-sheet invariant, configurable equations/tolerances, and warning-only cross-period thresholds. Tests use synthetic evidence only; no real financial PDF has been evaluated, and no accuracy or benchmark claim is made. Reconstruction remains geometry-driven; semantic mapping is exact-registry based; validation requires compatible resolved evidence and does not modify source values. Proposed: Streamlit review and exports.
