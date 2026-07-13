# MUTCD-150 runtime assets

This directory is safe for the model-execution environment. It contains the immutable question-only benchmark, runner v1.1, and a default model registry. It intentionally excludes all gold answers and M-SDI evaluator metadata.

## Immutable question file

- Records: 150
- SHA-256: `3a04b1d620a80704eefac34c565449a0cb8814e781dd6d73b8afb77318b954b2`

## Runner v1.1 behavior

- Detects responses beginning with `(VLM error:)`.
- Retries transient/provider errors with exponential backoff.
- Writes only successful answers to `answers_<run_id>.jsonl`.
- Writes terminal failures to `errors_<run_id>.jsonl` so they can be rerun safely.

The notebook persists the editable model registry and all outputs in Google Drive.
