# VLM-First Japanese Financial Table Parser

trace/toresu (Traceable Risk-Adaptive Cell Extraction) is a correctness-first document-intelligence pipeline for extracting structured tables from Japanese financial filing PDFs. It combines vision-language model (VLM) understanding with PDF text evidence, OCR, deterministic financial validation, and contradiction-guided re-inspection of uncertain regions. Its differentiator is not another extraction pass: it records evidence per cell and spends additional inference only where independent signals suggest the result may be wrong.

The initial implementation is designed to be completed under a two-day engineering deadline and to process collections of thousands of PDFs through bounded concurrency and resumable execution. The primary deliverable is a Python command-line tool. A lightweight Streamlit interface can be added for visual review and demonstration without becoming a dependency of the extraction engine.

## Objectives

- Recover Japanese financial tables with accurate row labels, column associations, fiscal periods, units, section boundaries, and hierarchical structure.
- Preserve the value exactly as printed before applying any normalization.
- Minimize silent errors, especially wrong digits, signs, row/column shifts, omitted rows, and duplicated rows.
- Use a VLM as the primary table-understanding and extraction component.
- Use native PDF text, OCR, geometry, and accounting rules as complementary evidence and verification—not as substitutes for visual understanding.
- Use contradiction-guided visual verification to re-examine only suspicious cells or table regions rather than repeatedly processing entire documents.
- Separate visual transcription risk from structural row/column-assignment risk, and test validator coverage through controlled fault injection.
- Support thousands of PDFs with bounded concurrency, checkpoints, retry limits, idempotent processing, and traceable failures.
- Make uncertainty explicit and retain enough source evidence for a reviewer to verify an extracted value.

## Design Principles

**Correctness before throughput.** Optimize for exact values and correct cell placement before maximizing pages per second.

**VLM-first, evidence-grounded.** The VLM interprets visual layout and proposes structured cell data. It must not be trusted as ground truth or allowed to silently infer unreadable values.

**Coarse-to-fine processing.** Understand the page and table first, then inspect high-resolution crops where small print, dense rows, or validation failures require additional attention.

**Preserve raw evidence.** Never overwrite source text or printed notation with normalized values. Both must remain available.

**Deterministic validation.** Use ordinary code for numeric parsing, schema checks, accounting arithmetic, and process state. Do not delegate arithmetic integrity or acceptance policy entirely to a generative model.

**Review rather than guess.** A value that cannot be resolved from available evidence remains unresolved and is flagged for review.

## Architecture

### 1. PDF Ingestion and Page Preparation

Use PyMuPDF to open documents, extract native text and bounding boxes when available, inspect page dimensions, and render pages for visual analysis. Preserve stable source identifiers, page numbers, rendering settings, and coordinates so every result can be traced back to the original file.

Native text is useful evidence for digitally generated PDFs, but reading order and layout may be unreliable. Scanned or image-based pages rely more heavily on rendered images and OCR. The pipeline should handle both without assuming that all PDFs have the same internal structure.

### 2. VLM-Based Document and Table Understanding

The VLM is the primary extraction engine. It should identify relevant financial tables and interpret their visible structure, including:

- Table titles and statement type.
- Reporting periods and multi-level column headers.
- Japanese row labels and indentation-based hierarchy.
- Section headings, subtotals, totals, and merged cells.
- Numeric values, punctuation, negative-value notation, dashes, and blank cells.
- Units such as thousands of yen (千円), and footnotes that affect interpretation.
- Table continuation across pages, where relevant.

The model must distinguish character transcription from structural understanding: a correctly read number assigned to the wrong fiscal-year column is still an extraction error. Extraction instructions should prohibit guessing, filling blank cells with inferred values, dropping printed symbols, or modifying a value to make totals balance.

Use the company-provided VLM or inference endpoint if one is specified. Otherwise, evaluate a document-specialized VLM such as PaddleOCR-VL-1.6 on actual sample pages before choosing the model. Keep the model integration behind a small adapter so the provider or model can be changed without rewriting the pipeline.

### 3. Coarse-to-Fine Table Reconstruction

Use a staged visual strategy:

1. **Page-level inspection:** identify page content, table regions, titles, reporting periods, and overall layout.
2. **Table-level extraction:** reconstruct headers, row order, cell contents, column relationships, and row hierarchy.
3. **Targeted region inspection:** render and re-read high-resolution crops when text is small, cells are ambiguous, sources disagree, or validation finds a likely error.

A crop should retain enough surrounding context—including the row label, header, and neighboring rows—to reduce column and row-assignment errors. Escalation should be driven by concrete uncertainty signals rather than applying maximum-resolution processing to every page.

### 4. Evidence Collection and Cell Provenance

Maintain a cell-level evidence record. Each extracted cell should retain, where available:

- Source document identifier and page number.
- Table, row, and column identity.
- Row label, column header, reporting period, and unit context.
- Raw printed text exactly as transcribed.
- Normalized value, if interpretation is unambiguous.
- Source bounding box or crop coordinates.
- Native PDF text and OCR observations.
- Validation results, uncertainty indicators, and correction history.

Evidence sources can disagree. PDF text may be out of reading order, OCR can misread characters, and a VLM can infer plausible but unsupported content. Resolve disagreement by inspecting the source region and related context. Agreement between two VLM responses alone is not proof of correctness.

### 5. Numeric Normalization

Keep raw transcription and normalized numeric interpretation in separate fields. Parse numeric values using explicit rules and decimal-safe arithmetic. Preserve the original Japanese labels, punctuation, negative notation, and units.

Normalization must distinguish at least the following cases:

- A printed numeric zero.
- An explicit dash or other printed marker.
- A genuinely empty cell.
- An unreadable or unresolved cell.
- A negative number indicated by the Japanese triangle symbol (△), parentheses, or another source convention.

Never map all of these to zero or a generic null value. Apply unit conversions only when the table's unit is known and record the unit used. Do not infer an absent unit or sign solely from accounting expectations.

### 6. Deterministic Structural and Financial Validation

Validate extracted content outside the VLM.

**Structural checks** include expected row widths, header-to-column associations, table shape, row order, hierarchy, merged-cell handling, and potential missing or duplicated rows.

**Numeric checks** include parseability, sign preservation, separators, precision, units, and whether each value remains associated with the correct row and reporting period.

**Accounting checks** compare subtotals and totals with component rows where the document provides enough information. Relationships such as total assets equalling total liabilities plus net assets may be checked when appropriate to the statement. Tolerate documented rounding and presentation differences. An arithmetic mismatch is a warning to investigate, not permission to alter the printed value.

Every warning should include a reason and point to the affected row, column, and source region. Validation must not silently rewrite the transcription.

### 7. Targeted Correction and Review

When a validation or evidence check fails:

1. Identify the affected cells and locate their source coordinates.
2. Create a high-resolution crop with sufficient neighboring context.
3. Ask the VLM to transcribe the visible region again without suggesting the expected answer.
4. Compare the new transcription with the original result and available PDF/OCR evidence.
5. Re-run the relevant structural and financial checks.
6. Accept only when the evidence satisfies defined acceptance rules; otherwise mark the field or table as requiring review.

Record the initial extraction and every correction attempt. Do not silently replace uncertain results. Keep review status separate from the extracted value so downstream users can distinguish output from verified output.

## Signature Innovation: Contradiction-Guided Visual Verification (CGVV)

Most simple document parsers use a one-pass pattern: send a page to a VLM, accept its JSON, and export the result. TRACE adds a targeted verification loop designed for financial tables, where a single incorrect digit, negative sign, or shifted column can invalidate otherwise plausible output.

CGVV is an engineering contribution inspired by recent work on multimodal table-layout correction and separating visual uncertainty from reasoning uncertainty. It is not a claim that the underlying ideas are wholly new research; the differentiator is implementing them together for Japanese financial tables and measuring the benefit.

### A. Two independent risk dimensions

For each cell, track two different questions rather than relying on one self-reported model confidence score:

- **Visual risk:** Is the printed content legible and transcribed correctly? Signals include low-resolution glyphs, VLM-versus-OCR disagreement, disagreement with native PDF text, and instability when the same source region is viewed at a higher resolution.
- **Structural risk:** Is the content attached to the correct row, column, header, and reporting period? Signals include unexpected row widths, ambiguous multi-level headers, cell-coordinate misalignment, row or column shifts, and inconsistent table topology.

The system should calculate an interpretable risk score from observable signals. Model-reported confidence can be retained as supplementary metadata, but it must not be the sole acceptance criterion. If OCR or native text is unavailable, mark that evidence as unavailable rather than treating it as agreement.

### B. Evidence ledger for every cell

Every candidate cell receives a traceable evidence record containing its source page and bounding box, raw transcription, normalized interpretation, row and column context, available PDF/OCR observations, checks that passed or failed, and any re-reading history. Store the raw printed value separately from the normalized numeric value. This makes disagreements inspectable and allows reviewers to locate the exact source region rather than rereading the whole PDF.

### C. Adaptive challenge views

When a cell or table crosses the risk threshold, do not simply resend the same full page with the same prompt. Generate a targeted view with enough context to answer the unresolved question:

1. **Row view:** retain the row label and its values to verify transcription and row association.
2. **Header/column view:** retain the relevant column header and nearby rows to verify fiscal-period association.
3. **High-resolution cell-region view:** inspect tiny digits, separators, or negative symbols while preserving nearby labels.

Use only the view or views that address the actual failure signal. Ask for transcription of visible evidence, not a value that would make the accounts balance. Compare the new observation with the original and available OCR/PDF evidence. A second VLM response is another observation—not ground truth—and agreement alone does not prove correctness.

### D. Evidence-gated acceptance

Route each table or cell into one of three states:

- **Accepted:** the structure and value pass required checks and have no unresolved high-risk contradiction.
- **Recheck:** a specific contradiction can be investigated with a targeted crop or an independent text observation.
- **Review required:** ambiguity remains after bounded rechecks, or critical source evidence is unreadable.

Accounting equations act as anomaly detectors, never as automatic correction instructions. For example, if a reported total does not reconcile, TRACE highlights the contributing rows for inspection; it must not change a printed number simply because another number satisfies the equation.

### E. Validator mutation testing

A distinctive, low-cost part of the implementation is a fault-injection test harness for the validation layer. Starting from manually checked tables or known-good structured outputs, generate controlled synthetic faults such as:

- Flip a negative triangle (△) or remove a minus sign.
- Change one digit or a decimal separator.
- Shift a value into an adjacent fiscal-year column.
- Replace an empty cell or dash with zero.
- Drop or duplicate a row.
- Corrupt a subtotal while leaving its component rows unchanged.

Measure which injected faults are detected, which escape, and how often valid tables are falsely flagged. This does **not** measure real-world extraction accuracy; it measures the coverage and weaknesses of the validators. Pair it with a manually annotated real-document set to evaluate actual transcription quality.

### F. Rendering-Aware Japanese Table Calibration

A directly relevant 2026 study presented at the Japanese Society for Artificial Intelligence examines table structuring in Annual Securities Reports. It reports that image-processing choices—such as DPI, font/grid visibility, and rendering clarity—can matter more than text-only prompt adjustments, while domain context and constrained output provide additional benefits.

Use that insight in a small calibration step before processing the corpus:

1. Select a small, representative set of pages containing dense Japanese labels, small financial figures, and multi-level headers.
2. Render these pages with a few controlled profiles (for example, moderate versus high DPI, with consistent scaling and no aggressive smoothing).
3. Compare extraction results against manually checked values, cell placement, and row completeness.
4. Choose one default profile and, if evidence supports it, a separate profile for unusually dense or low-quality pages.
5. Save the chosen parameters in configuration so results are reproducible.

Do not generate many image variants for every page. Calibrate once on development pages, then use targeted higher-resolution crops only when visual or structural risk warrants them. Preserve the original page coordinates when mapping crop-level results back to the source.

This makes rendering resolution a measured system parameter rather than an arbitrary constant. The calibration set must be separate from the held-out evaluation set to avoid reporting tuning results as generalization.

### G. Page-First Relevance Routing

A 2026 ACL Industry Track study on long scanned financial documents found that localizing task-relevant pages before VLM extraction was a major contributor to end-to-end accuracy. Borrow this pattern for batch processing:

- Inspect pages with cheap signals first: native text, page dimensions, headings, and lightweight OCR/layout cues.
- Rank or classify pages likely to contain financial tables, headers, continuations, and related footnotes.
- Run the primary VLM extraction on relevant pages and their necessary context pages instead of sending every page of a long filing to the model indiscriminately.
- Keep the page-selection output auditable and make the filter conservative: if the signals are ambiguous, process the page rather than risk omitting a required table.
- Track page-retrieval recall separately from extraction accuracy. A table never sent to the VLM is a missed table, not a successful extraction.

This module reduces inference cost and irrelevant context while retaining the VLM as the central table-understanding engine. For an assignment where nearly every page contains a relevant table, disable or soften filtering rather than applying it blindly.

### H. Visible-Evidence-First Output

Generative OCR can complete text that is plausible but not actually visible. TRACE should therefore preserve two distinct fields: `raw_transcription` (what is visibly printed) and `normalized_value` (the machine-interpreted value). When a critical digit or sign cannot be read reliably, return an unresolved state instead of allowing context or accounting rules to fill it in.

Evaluate unsupported-completion behavior separately from ordinary character accuracy. A system that returns blank values for everything is safe from hallucination but useless; a system that guesses plausible numbers is useful-looking but unsafe. Measure exact-match accuracy, unresolved/abstention rate, and visually unsupported-value rate together.

### I. Demonstrate the improvement, not just the architecture

On the same held-out sample pages, compare a baseline one-pass VLM extraction with TRACE's evidence-gated correction loop. Report:

- Numeric exact match and cell-association accuracy before and after correction.
- Errors detected, errors corrected, and false alarms.
- Fraction of cells/tables escalated to review.
- Additional VLM calls and approximate inference cost per page.
- Detection rate on each injected-fault category.

This gives reviewers a concrete ablation: does targeted verification improve financial-table correctness enough to justify its extra inference? Do not claim improvement until the evaluation results demonstrate it.

## Technology Stack

| Area | Recommended technology | Responsibility |
|---|---|---|
| Language | Python 3.11 | Core pipeline and validation logic |
| PDF ingestion and rendering | PyMuPDF | Native text, page geometry, and rendering |
| Image preparation | Pillow and OpenCV | Cropping, resizing, and image handling |
| Primary document understanding | Company-provided VLM or evaluated document VLM | Visual table interpretation and structured extraction |
| Secondary text evidence | PaddleOCR | OCR observations for comparison and verification |
| Structured schema | Pydantic | Validate and normalize model responses into typed records |
| Numeric handling | `decimal.Decimal` and explicit parsers | Preserve financial precision and parse numeric notation |
| CLI | Typer | Batch processing and inspection commands |
| Checkpointing | SQLite | Durable per-document and per-page processing state |
| Identity and caching | SHA-256 content hashes and local file metadata | Idempotency and safe reuse of intermediate results |
| Optional reviewer UI | Streamlit | Upload, table preview, source-region review, and exports |
| Tests | pytest | Unit, integration, regression, and evaluation tests |

The exact VLM deployment depends on the available endpoint, hardware, latency limits, and Japanese-document accuracy. Avoid unnecessary model hosting or fine-tuning during the initial two-day build.

## Interfaces and Outputs

### Command-Line Interface

The CLI is the primary interface. It should support:

- Processing one PDF or a directory of PDFs.
- Resuming interrupted or incomplete processing.
- Re-running deterministic validation on existing results.
- Inspecting a single document during development.
- Producing machine-readable output and a summary of failures or review-required cases.

Single-document inspection and batch processing must use the same underlying pipeline, not separate implementations.

### Optional Streamlit Review Interface

If time permits after the parser and tests work end to end, add a minimal web interface that can:

- Select or upload a PDF.
- Display extracted tables alongside the source page or crop.
- Highlight cells with unresolved evidence or failed validation.
- Show the raw transcription, normalized value, and review status.
- Export JSON and CSV results.

The UI is a thin layer over the parser. The extraction engine must remain usable without a web app. Packaging a standalone executable is out of scope unless explicitly required by the evaluator.

### Output Contract

The output should represent each document as pages, tables, headers, rows, and cells. At minimum, retain:

- Document identity and processing status.
- Table title, statement type, units, and reporting periods when identified.
- Ordered row labels and hierarchical relationships.
- Cell positions and header associations.
- Raw printed content and normalized values where unambiguous.
- Source page and coordinates.
- Evidence observations, validation results, and review status.

The output format should be versioned so changes to the schema do not silently break downstream consumers. JSON is the canonical structured output; CSV can be provided as a convenience export for flat tables.

## Reliability for Thousands of PDFs

The initial design should use simple, durable processing rather than a large distributed platform.

- **Bounded concurrency:** limit simultaneous inference requests according to endpoint capacity and rate limits.
- **Checkpointing:** persist document and page status so completed work survives interruptions.
- **Idempotency:** use stable source identity and content hashes to avoid duplicate records on re-runs.
- **Retry policy:** retry transient infrastructure errors with bounded backoff; keep quality failures separate from service failures.
- **Targeted reprocessing:** retry failed pages and uncertain regions instead of repeating successful work across the whole corpus.
- **Resource control:** process documents incrementally and avoid retaining full-resolution page images unnecessarily.
- **Traceable failures:** record the affected file and page, failure category, retry count, and actionable message.
- **Configuration:** keep model settings, render resolution, concurrency, retry limits, and validation thresholds configurable.

A database server, message broker, distributed object store, or multi-service deployment is not required for the first version unless the execution environment demands it. SQLite and resumable local processing are sufficient starting points for a single-machine batch tool.

## Evaluation

Build a small manually checked reference set from the actual sample PDFs before processing the full collection. Include ordinary pages and difficult cases: dense Japanese labels, small numeric text, multiple fiscal-year columns, section bands, negative triangles, dashes, subtotals, merged headers, and ambiguous cells.

Measure at least:

- **Numeric exact match:** exact value agreement, including sign and decimal precision.
- **Cell association accuracy:** whether the value belongs to the correct row and column.
- **Text accuracy:** transcription quality for Japanese row labels and raw values.
- **Table structure accuracy:** correctness of header hierarchy, row order, and table shape.
- **Row completeness:** missing, duplicated, or incorrectly split rows.
- **Whole-table correctness:** the fraction of tables with all required cells and associations correct.
- **Validation effectiveness:** genuine errors detected and correct extractions incorrectly flagged.
- **Review rate:** fraction of cells and tables left unresolved.
- **Processing reliability:** failure rate, retry behavior, and successful resumption after interruption.

Prioritize numeric exact match, correct cell association, row completeness, and whole-table correctness. Character accuracy alone is insufficient when a single value can be shifted into a different fiscal-year column. Evaluate targeted correction by checking whether it resolves real errors without changing already-correct cells.

Do not claim a general accuracy percentage based solely on a small development sample. Report the evaluation set size, its composition, and each metric separately.

## Security and Data Handling

Treat source PDFs and extracted financial data as sensitive unless the assignment explicitly states otherwise. Do not send documents to third-party model services unless their use is authorized. Keep credentials out of source control, restrict logs from dumping full document contents, and retain only the source regions needed for debugging and review. Document any external inference provider used by the implementation.

## Innovation References

- [Structuring Financial Unstructured Data Using Multimodal Generative AI Models (JSAI 2026)](https://www.jstage.jst.go.jp/article/pjsai/JSAI2026/0/JSAI2026_2F6OS19b02/_article/-char/en) directly studies Japanese Annual Securities Reports and reports that rendering clarity, DPI, and grid/font visibility materially affect cell localization and value extraction; domain context and constrained output also help.
- [A Multistage Extraction Pipeline for Long Scanned Financial Documents (ACL Industry Track 2026)](https://aclanthology.org/2026.acl-industry.99/) evaluates 120 production KYC documents comprising roughly 3,000 multilingual scanned pages. Its multistage approach with page-level retrieval, OCR, and VLM extraction improved field-level accuracy versus direct PDF-to-VLM baselines; the paper reports page retrieval as a dominant factor in its setting.
- [A Multimodal Data Extraction Pipeline with Table Layout Correction (PMLR 2026)](https://proceedings.mlr.press/v318/yao26a.html) motivates explicit multimodal table-structure correction before schema-based extraction.
- [Zero-Shot Table Extraction in Business Documents: A Unified Benchmark with Error Taxonomy and Ecological Analysis (WACV 2026)](https://openaccess.thecvf.com/content/WACV2026/html/Thomas_Zero-Shot_Table_Extraction_in_Business_Documents_A_Unified_Benchmark_with_WACV_2026_paper.html) motivates tracking actionable table errors such as missed/merged tables, header-body confusions, and incorrect cell topology rather than reporting only a single accuracy score.
- [Hallucination-aware evaluation and confidence-guided selective routing for hybrid VLM-OCR (IJDAR 2026)](https://link.springer.com/article/10.1007/s10032-026-00620-7) studies visually unsupported transcription and the accuracy/latency trade-off of routing uncertain cases from OCR to a stronger VLM. Its results also show why abstention and hallucination risk should be reported separately from character error rate.
- [VL-Calibration: Decoupled Confidence Calibration for Large Vision-Language Models Reasoning (ACL 2026)](https://aclanthology.org/2026.acl-long.2074/) motivates separating visual-grounding uncertainty from reasoning uncertainty rather than relying on a single holistic confidence score.

These works are design inspiration, not proof that TRACE will outperform an unmodified VLM. Their reported results were obtained on different datasets and document types; validate the effect on the target Japanese filings.

## Research Motivation

TRACE combines recent directions while remaining implementable without training a new model:

- **Rendering-aware extraction:** calibrate resolution and image preprocessing on Japanese statement pages before choosing a production profile.
- **Page-first relevance routing:** find relevant financial tables before spending VLM calls on long reports, while measuring retrieval recall so filtering does not silently omit tables.
- **Coarse-to-fine inspection:** use full-page context followed by high-resolution regional inspection for difficult cells.
- **Multimodal structural correction:** compare the candidate table with source-image evidence and repair row/column structure before final serialization.
- **Evidence-gated acceptance:** treat unsupported values, unresolved ambiguity, and model disagreement as distinct states rather than using one confidence number.
- **Selective re-inspection:** spend extra inference only where observed evidence or validation indicates risk.

These papers motivate design choices; their results do not establish accuracy on the assignment documents. The actual model, rendering profile, risk thresholds, and acceptance rules must be evaluated against manually verified Japanese financial tables.

## Scope and Priorities

The essential deliverable is a complete, testable path from PDF input to validated structured output. Prioritize reliable rendering, VLM extraction, provenance, raw-value preservation, deterministic checks, targeted correction, resumability, and regression tests.

Keep the optional UI small and defer model fine-tuning, large-scale distributed orchestration, elaborate dashboards, and executable packaging unless they are explicit requirements. A compact, well-tested extraction engine with transparent failure handling is more valuable than a broader application that cannot demonstrate numerical correctness.

## Expected Result

A VLM-first parser that reconstructs Japanese financial tables, retains their original notation and hierarchy, validates extracted values with deterministic rules, re-examines suspicious regions, and marks unresolved cases instead of guessing. The pipeline should be easy to inspect and resume, suitable for processing thousands of PDFs, and demonstrably evaluated against manually verified examples.
