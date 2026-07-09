# Rebuild v4 — Figure Extraction, KG v2, and the Question Router

Date: 2026-07-09. This rebuild fixes the three failures measured in the first
scoring run (figure precision 6 %, recall 28 %, grounding ~1.5/5):
missing figures, wrong crops, and figures attached to every answer.

## 1. Root cause of the missing / wrong figures

The v1 extractor assumed captions sit **below** figures and cropped the
region **above** each caption. In the MUTCD 11th-Edition PDF every caption
sits **above** its content. Consequences:

| Caption position | v1 behaviour | Visible symptom |
|---|---|---|
| Top of page | region above ≈ 30 pt header → fails min-height | figure **silently missing** (the 11+ gold gaps) |
| Mid page | region above = the *previous* figure / body text | figure "extracted" but **wrong image** |
| TOC list pages | dot-leader lines matched the caption regex | junk crops |

Four more layout pathologies compounded it: inset figures in two-column
pages (caption mid-line in the text layer), side-by-side captions (Tables
4C-6/4C-7), U+2011 non-breaking hyphens in Rev-1 ids ("Table 6B‑2"), and
`(Sheet n of m)` multi-sheet figures.

## 2. What replaced it

- **`mrag/figure_core.py`** (new) — backend-agnostic caption grammar +
  region geometry. Rules: crop **below** caption to the next caption
  strictly below or the bottom margin; column-aware x-bounds for inset
  captions; page-level TOC rejection; dash normalisation; sheet parsing.
- **`mrag/figures.py`** (rewritten; v1 kept as `figures_v1_deprecated.py`) —
  two backends over the same geometry: `fitz` (production) and `poppler`
  (validation/fallback). Adds `rescue_missing()` (relaxed min-height re-run
  for ids with zero crops, e.g. tiny Table 2A-3) and `validate_coverage()`
  which **raises** if figure coverage < 99 % against the full-document
  mention census. Ingestion can never ship a silent gap again.
- **Validated result on the real PDF:** **Figures 485/485 (100 %），Tables
  68/68 (100 %)**, 743 crops, 553 canonical ids, zero duplicates, zero junk;
  AASHTO "Table 3-1/3-3" references auto-excluded as external documents.
- **`mrag/kg.py`** (rewritten; v1 kept as `kg_v1_deprecated.py`) — one node
  per canonical figure id with `image_paths` for all sheets; 3-tier section
  anchoring (`anchored_in` from citing chunks ≻ `belongs_to_chapter` from
  the id ≻ weak `on_page_of`); prefix-tolerant `KG.figure()`; unresolved-ref
  stats in `g.graph["build_stats"]`; `cites_any_figure` flags for the router.
- **`mrag/question_router.py`** (new) — per-query `needs_figures` decision:
  hard-yes (explicit ids, visual verbs), hard-no (definitional/permission
  patterns with no visual evidence), soft scoring (sign codes, visual nouns,
  placement verbs, **KG prior** from the sections of the top retrieved
  chunks). Deterministic, returns fired rules for eval logging. Off-switch:
  `CFG.use_question_router = False` reproduces v1 behaviour.
- **`mrag/retrieval.py`** — figure paths now run **only** when the router
  says yes; `debug["figure_router"]` records the decision; figure payloads
  carry `image_paths`/`n_sheets`/`anchor_section`.
- **`mrag/parsing.py`** — chunk text normalises U+2011/U+2013 dashes so
  `figure_refs`/`table_refs` resolve Rev-1 ids; `fitz` import made lazy.
- **`scripts/ingest_v4.py`** (new) — extract → rescue → **coverage gate**
  (non-zero exit below `--min-coverage`, default 0.99) → chunks (v4 stamp
  forces re-parse once) → KG v2 → unchanged v3 embedding/Qdrant steps.

## 3. Running it (Colab)

```bash
pip install pymupdf          # fitz backend
python scripts/ingest_v4.py --figures-only     # fast: extraction + gate only
python scripts/ingest_v4.py                    # full pipeline
```

Old caches are archived (`*.v1.bak*`, `*.v3.bak*`) — nothing is destroyed.
`cache_dir/figure_coverage_report.json` is written every run.

## 4. Validation reference shipped with this rebuild

- `validation/census_records.jsonl` — all 743 crop records (ids, pages,
  bboxes in PDF points) produced by the poppler backend on the real PDF.
  Colab re-crops deterministically from these bboxes if ever needed.
- `validation/coverage_report.json` — the 100 %/100 % census.
