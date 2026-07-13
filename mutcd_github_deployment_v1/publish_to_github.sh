#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD="$ROOT/repo_payload"
GITIGNORE_BLOCK="$ROOT/gitignore_mutcd_block.txt"

usage() {
  cat <<'EOF'
Usage:
  bash publish_to_github.sh <repository-url> [branch] [commit-message]

Examples:
  bash publish_to_github.sh https://github.com/hannanazad/MRAG_stp2.git main
  bash publish_to_github.sh git@github.com:hannanazad/MRAG_stp2.git main "Add MUTCD-150 benchmark runner"

The script clones the repository into a temporary directory, copies only the
runtime-safe payload, appends benchmark ignore rules, commits, and pushes.
It never uploads the PDF, Qdrant data, run outputs, or evaluator gold file.
EOF
}

REPO_URL="${1:-}"
BRANCH="${2:-main}"
COMMIT_MESSAGE="${3:-Add GitHub-backed MUTCD-150 benchmark runner}"

if [[ -z "$REPO_URL" ]]; then
  usage
  exit 2
fi

command -v git >/dev/null 2>&1 || { echo "git is required" >&2; exit 1; }
command -v python >/dev/null 2>&1 || { echo "python is required" >&2; exit 1; }
[[ -d "$PAYLOAD" ]] || { echo "Missing payload: $PAYLOAD" >&2; exit 1; }

bash "$ROOT/verify_bundle.sh"

if [[ -z "$(git config --global user.name || true)" ]]; then
  read -r -p "Git user.name: " GIT_NAME
  git config --global user.name "$GIT_NAME"
fi
if [[ -z "$(git config --global user.email || true)" ]]; then
  read -r -p "Git user.email: " GIT_EMAIL
  git config --global user.email "$GIT_EMAIL"
fi

TMP_ROOT="$(mktemp -d 2>/dev/null || mktemp -d -t mutcd-benchmark)"
trap 'rm -rf "$TMP_ROOT"' EXIT
CLONE_DIR="$TMP_ROOT/repo"

echo "Cloning $REPO_URL ..."
git clone "$REPO_URL" "$CLONE_DIR"

cd "$CLONE_DIR"
git fetch origin
if git show-ref --verify --quiet "refs/remotes/origin/$BRANCH"; then
  git checkout -B "$BRANCH" "origin/$BRANCH"
else
  git checkout -b "$BRANCH"
fi

# Copy payload without deleting unrelated repository files.
cp -R "$PAYLOAD"/. "$CLONE_DIR"/

# Append the ignore block once, preserving the repository's existing .gitignore.
touch .gitignore
if ! grep -q '^# BEGIN MUTCD-RAG benchmark local/runtime data$' .gitignore; then
  printf '\n' >> .gitignore
  cat "$GITIGNORE_BLOCK" >> .gitignore
fi

# Safety scan in the paths being added.
if find benchmarks/mutcd150 notebooks scripts -type f \
    \( -iname '*gold*' -o -iname '*review*.csv' -o -iname '*.pdf' -o -iname 'qdrant_db.tar' \) | grep -q .; then
  echo "Refusing to commit forbidden evaluator/data files:" >&2
  find benchmarks/mutcd150 notebooks scripts -type f \
    \( -iname '*gold*' -o -iname '*review*.csv' -o -iname '*.pdf' -o -iname 'qdrant_db.tar' \) >&2
  exit 1
fi

bash scripts/verify_mutcd_benchmark_assets.sh

git add .gitignore benchmarks/mutcd150/v1 notebooks/MUTCD_updated_kg_github_benchmark.ipynb scripts/verify_mutcd_benchmark_assets.sh

echo
echo "Files staged:"
git status --short

if git diff --cached --quiet; then
  echo "No changes to commit. Repository already contains this payload."
  exit 0
fi

git commit -m "$COMMIT_MESSAGE"
git push -u origin "$BRANCH"

echo
echo "Published successfully."
echo "Notebook path: notebooks/MUTCD_updated_kg_github_benchmark.ipynb"
echo "Benchmark path: benchmarks/mutcd150/v1/"
