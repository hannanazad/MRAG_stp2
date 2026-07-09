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
git commit -m "v4.1 hotfix: schema-tolerant figures.jsonl reader + version peek

Colab first-run crashed: ingest_v4 version-checked the OLD v1 figures.jsonl
by fully parsing it with the v2 FigureRecord dataclass, which requires the
new sheet/sheet_of fields -> TypeError before the archive/re-extract logic
could run.

- mrag/figures.py read_jsonl(): fills defaults for missing v2 fields
  (sheet, sheet_of, chapter, extraction_method=caption_above_v1, ...) so
  v1 rows always load
- scripts/ingest_v4.py: _figures_jsonl_is_v2() peeks at the first row's
  extraction_method as raw JSON — no dataclass parse — then archives v1
  and re-extracts as designed
- regression-tested against the exact Colab failure row + v2 round-trip"

BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "==> Pushing to origin/$BRANCH..."
git push origin "$BRANCH"
echo "==> Done."
