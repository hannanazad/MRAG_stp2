#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="$HERE/benchmarks/mutcd150/v1"
EXPECTED="3a04b1d620a80704eefac34c565449a0cb8814e781dd6d73b8afb77318b954b2"

for f in mutcd_benchmark_questions_v1.jsonl mutcd_benchmark_runner.py model_registry.json runtime_manifest.json; do
  [[ -f "$BENCH/$f" ]] || { echo "Missing: $BENCH/$f" >&2; exit 1; }
done

ACTUAL="$(sha256sum "$BENCH/mutcd_benchmark_questions_v1.jsonl" | awk '{print $1}')"
[[ "$ACTUAL" == "$EXPECTED" ]] || { echo "Question hash mismatch" >&2; echo "Expected: $EXPECTED" >&2; echo "Actual:   $ACTUAL" >&2; exit 1; }

if find "$HERE" -type f \( -iname '*gold*' -o -iname '*review*.csv' -o -iname '*.pdf' -o -iname 'qdrant_db.tar' \) | grep -q .; then
  echo "Forbidden evaluator/data file found in runtime payload." >&2
  find "$HERE" -type f \( -iname '*gold*' -o -iname '*review*.csv' -o -iname '*.pdf' -o -iname 'qdrant_db.tar' \) >&2
  exit 1
fi

COUNT="$(wc -l < "$BENCH/mutcd_benchmark_questions_v1.jsonl" | tr -d ' ')"
[[ "$COUNT" == "150" ]] || { echo "Expected 150 questions, found $COUNT" >&2; exit 1; }
echo "Runtime payload verified: 150 questions; SHA-256 $ACTUAL"
