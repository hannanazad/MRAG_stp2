# MUTCD RAG GitHub + Google Drive deployment

## Recommended architecture

Use **GitHub for versioned runtime code** and **Google Drive for large or mutable artifacts**.

| Location | Store here |
|---|---|
| GitHub repository | RAG source code, notebook, runner, fixed question-only benchmark, default model registry |
| Google Drive (`MyDrive/MRAG`) | MUTCD PDF, Qdrant snapshot, extracted/cached data, editable registry override, benchmark outputs |
| Colab Secrets | `QWEN` and, for a private repository, optional `GITHUB_TOKEN` |
| Offline evaluator only | Gold answers, scoring metadata, and review CSV |

This avoids manual uploads each Colab session, keeps the benchmark versioned, prevents the repository from filling with large generated files, and keeps evaluator answers away from the models.

## Publish with Git Bash

1. Extract this ZIP.
2. Open **Git Bash** in the extracted directory.
3. Verify the package:

```bash
bash verify_bundle.sh
```

4. Publish to the existing repository:

```bash
bash publish_to_github.sh https://github.com/hannanazad/MRAG_stp2.git main
```

SSH also works:

```bash
bash publish_to_github.sh git@github.com:hannanazad/MRAG_stp2.git main
```

The script clones into a temporary directory, preserves unrelated repository content, copies the payload, commits, and pushes. Authentication is handled by your normal Git credential manager, browser login, PAT, or SSH key.

## Open and run in Colab

Open this notebook from the repository:

`notebooks/MUTCD_updated_kg_github_benchmark.ipynb`

The first cell clones or fast-forwards the selected branch. For a private repository, add a fine-grained read token to Colab Secrets as `GITHUB_TOKEN`. Keep the VLM key as `QWEN`.

The benchmark cell reads:

- `3a04b1d620a80704eefac34c565449a0cb8814e781dd6d73b8afb77318b954b2`-locked questions from the GitHub clone;
- the runner from the GitHub clone;
- an editable model-registry override from Drive.

On first run, the default registry is copied to:

`MyDrive/MRAG/benchmarks/mutcd150/v1/model_registry.json`

Outputs are written to:

`MyDrive/MRAG/benchmark_runs/mutcd150_v1/`

## What still belongs in Drive

The notebook expects the MUTCD PDF under the location resolved by `CFG.base_dir` and `CFG.pdf_path`. Your existing Qdrant snapshot remains `MyDrive/MRAG/qdrant_db.tar`. Those files should not be committed to GitHub.

## Gold-file isolation

This package does **not** contain `mutcd_benchmark_gold_v1.jsonl`. Do not add the gold JSONL or review CSV to the runtime repository. The included `.gitignore` block and verification scripts guard against accidental inclusion.

## Adding future models

Edit only the Drive registry override. The question file and its hash remain unchanged, allowing future models to be compared against the same benchmark.
