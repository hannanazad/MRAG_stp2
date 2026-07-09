#!/usr/bin/env bash
# Apply the v4 rebuild to this repo and push.
# Usage (from the repo root, exactly like previous versions):
#   cd ~/OneDrive/Desktop/Courses/Summer/m_rag/MRAG_stp2
#   unzip /c/Users/hanan/Downloads/MRAG_stp2_v4.zip
#   bash apply_and_push.sh
set -euo pipefail

SRC="MRAG_stp2-main"

if [ ! -d .git ]; then
    echo "ERROR: run this from the MRAG_stp2 repo root (no .git here)." >&2
    exit 1
fi
if [ ! -d "$SRC" ]; then
    echo "ERROR: $SRC/ not found — unzip the bundle in the repo root first." >&2
    exit 1
fi

echo "==> Copying v4 files into the repo..."
cp -r "$SRC"/. .
rm -rf "$SRC"
rm -f apply_and_push.sh.applied 2>/dev/null || true

echo "==> Files changed:"
git status --short

git add -A
git commit -m "v4 rebuild: figure extraction (100% coverage), KG v2, question router

Root cause: v1 cropped ABOVE captions; MUTCD captions sit ABOVE content ->
missing figures (top-of-page), wrong crops (mid-page), TOC junk.

- mrag/figure_core.py (new): caption-below geometry; inset two-column
  captions; side-by-side width partition; U+2011 dash normalization;
  (Sheet n of m) parsing; TOC page guard; mention census + coverage report
  with AASHTO external-ref exclusion
- mrag/figures.py (rewritten, v1 kept as figures_v1_deprecated.py): dual
  fitz/poppler backend over shared geometry; rescue_missing() for tiny
  regions; validate_coverage() raises below threshold
  VALIDATED on the real PDF: Figures 485/485 (100%), Tables 68/68 (100%),
  743 crops, 553 canonical ids, 0 dups, 0 junk
- mrag/kg.py (rewritten, v1 kept as kg_v1_deprecated.py): one node per
  canonical id with image_paths for all sheets; 3-tier anchoring
  (anchored_in > belongs_to_chapter > on_page_of); prefix-tolerant lookups;
  unresolved-ref stats; cites_any_figure flags
- mrag/question_router.py (new): per-query needs_figures decision with
  logged rules + KG prior; fixes always-retrieving-4-figures (precision 6%)
- mrag/retrieval.py: figure paths gated on router; multi-sheet payloads
- mrag/parsing.py: U+2011 normalization in chunk refs; lazy fitz import
- scripts/ingest_v4.py (new): extract -> rescue -> coverage gate -> KG v2;
  auto-archives v1 caches; invalidates stale figure embedding caches
- MUTCD_MRAG_HPRC.ipynb: clones MRAG_stp2, runs ingest_v4, guards against
  restoring a pre-v4 Qdrant snapshot; router smoke test added
- validation/: census_records.jsonl (all 743 bboxes) + coverage_report.json
- docs/REBUILD_V4.md: full write-up"

BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "==> Pushing to origin/$BRANCH..."
git push origin "$BRANCH"
echo "==> Done."
