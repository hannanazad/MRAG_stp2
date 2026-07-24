# MRAG Ablation Study — Data Collection Harness

Runs leave-one-out ablations against the 150-question MUTCD-150 gold set and
packages raw RAG outputs into per-ablation zip files on Drive. **Does no
scoring, no interpretation, no GPT calls.** The zips are handed to GPT-5.6
separately for judgment.

## What this bundle does

1. Freezes VLM to Fable 5, prompt style to fewshot.
2. For each ablation (A1–A6), toggles one component off at runtime.
3. Runs the RAG against 150 questions, capturing raw output per question.
4. Packages results into `mrag_ablation_<id>.zip` on Drive.

## What this bundle does NOT do

- Does not call GPT (no OpenAI dependency).
- Does not compute rubric scores, retrieval metrics, or aggregate deltas.
- Does not display question text in logs or console (only `qa_id`).
- Does not interpret which components matter.

## Ablations

| id | component off | mechanism |
|----|---------------|-----------|
| A1 | question router | `CFG.use_question_router = False` |
| A2 | VLM figure filter | `CFG.use_vlm_figure_filter = False` |
| A3 | graph proximity | `CFG.w_graph = 0.0` |
| A4 | rule-type weight | `CFG.w_ruletype = 0.0` |
| A5 | hierarchy prior | `CFG.w_hierarchy = 0.0` |
| A6 | reranker | `pipeline.reranker` swapped for `NoOpReranker` at runtime |

The `mrag/` package is not modified. Toggles happen at runtime via
`ablations/ablation_shims.py`.

## Layout

```
ablations/
├── README.md                  # this file
├── run_ablations.ipynb        # Colab driver — one cell per ablation
├── ablation_shims.py          # runtime toggles + NoOpReranker
├── run_ablation.py            # sweep 150 Qs for one ablation → runs_<id>.jsonl
├── package_results.py         # zip a single ablation's results to Drive
├── configs/                   # one YAML per ablation
│   ├── baseline.yaml
│   ├── A1_no_router.yaml
│   ├── A2_no_vlm_filter.yaml
│   ├── A3_no_graph.yaml
│   ├── A4_no_rule_type.yaml
│   ├── A5_no_hierarchy.yaml
│   └── A6_no_reranker.yaml
└── results/                   # populated at runtime (gitignored)
```

## Running

Open `run_ablations.ipynb` in Colab. The notebook is organized so each
ablation has its own cell — if A3 fails, re-run only cell A3.

Cell order:
1. Environment setup (mount, clone, install)
2. API keys (VLM providers only, no OPENAI needed)
3. Configuration (paths, results dir on Drive)
4. Pipeline init (once per session)
5. Smoke test (optional, 5 Qs on A1)
6. **Cell A1** — run + zip
7. **Cell A2** — run + zip
8. **Cell A3** — run + zip
9. **Cell A4** — run + zip
10. **Cell A5** — run + zip
11. **Cell A6** — run + zip
12. Optional: "run all remaining" convenience cell
13. Handoff notes to GPT

Each cell is independently runnable. Each cell is resume-safe — if the sweep
was interrupted midway, re-running picks up from the last successfully
completed `qa_id`.

## Per-ablation zip contents

Each `mrag_ablation_<id>.zip` on Drive contains:
- `runs_<id>.jsonl` — one row per question: qa_id, question, RAG answer,
  chunks_used, figures_used, pages_used, retrieval debug, latency
- `meta.json` — ablation_id, timestamp, VLM alias, prompt style, n_completed,
  n_errored, git commit sha of the mrag/ code that produced the runs
- `README.txt` — one-paragraph note describing what the zip is and how to
  hand to GPT for scoring

Hand each zip to GPT with your existing MUTCD-150 rubric prompt to produce
per-ablation scored files. Compare against your existing baseline
(Fable-5 fewshot 88.08) using whatever pairing method you prefer.

## Cost (generation only — no scoring cost from this harness)

| ablation | wallclock (150 Qs × ~28s median) |
|----------|----------------------------------|
| A1–A6 each | ~70 min |
| six total | ~7 h |

Resume-safe. Can be split across sessions.
