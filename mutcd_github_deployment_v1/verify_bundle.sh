#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD="$ROOT/repo_payload"
[[ -d "$PAYLOAD" ]] || { echo "Missing repo_payload" >&2; exit 1; }
"$PAYLOAD/scripts/verify_mutcd_benchmark_assets.sh"
python - <<'PY2' "$PAYLOAD/notebooks/MUTCD_updated_kg_github_benchmark.ipynb"
import json, sys
p=sys.argv[1]
nb=json.load(open(p,encoding='utf-8'))
assert nb.get('nbformat') == 4
assert len(nb.get('cells',[])) >= 38
print(f"Notebook JSON verified: {len(nb['cells'])} cells")
PY2
echo "Deployment bundle verified."
