# MRAG Ablation Study

Isolated leave-one-out ablation harness for MRAG_stp2. **Does not modify the
main `mrag/` package.** All component toggles happen at runtime via monkey-patches
in `ablation_shims.py`.

## Ablations

Six leave-one-out runs on **Fable 5 fewshot / MUTCD-150 frozen**. In each run,
exactly ONE component is disabled; everything else stays at production defaults.
Report Δ vs. baseline (88.08) with paired bootstrap 95% CI over the 150 per-item
score deltas.

| id | component off | mechanism |
|----|---------------|-----------|
| A1 | question router | `CFG.use_question_router = False` |
| A2 | VLM figure filter | `CFG.use_vlm_figure_filter = False` |
| A3 | graph proximity | `CFG.w_graph = 0.0` |
| A4 | rule-type weight | `CFG.w_ruletype = 0.0` |
| A5 | hierarchy prior | `CFG.w_hierarchy = 0.0` |
| A6 | reranker | `pipeline.reranker` swapped for `NoOpReranker` at runtime |

## Layout

```
ablations/
├── README.md
├── run_ablations.ipynb        # Colab driver (mount, clone, install, sweep, score, analyze)
├── ablation_shims.py          # apply_ablation(pipeline, ablation_id) — all component toggles
├── run_ablation.py            # sweep one config over gold_qa.jsonl → results/runs_<id>.jsonl
├── score_ablation.py          # GPT-5.6 rubric → results/scored_<id>.jsonl
├── analyze_ablations.py       # paired bootstrap CI vs baseline → results/summary.csv + delta_plot.png
├── configs/
│   ├── baseline.yaml
│   ├── A1_no_router.yaml
│   ├── A2_no_vlm_filter.yaml
│   ├── A3_no_graph.yaml
│   ├── A4_no_rule_type.yaml
│   ├── A5_no_hierarchy.yaml
│   └── A6_no_reranker.yaml
└── results/                   # populated at runtime (gitignore contents except summary.csv)
```

## Prerequisites

- MRAG_stp2 repo cloned and `init_pipeline()` produces a working pipeline
- Fable 5 is the fixed answer VLM (`CFG.set_vlm_model("fable")` before running)
- `gold_qa.jsonl` at 150 rows with the MUTCD-150 questions
- **Baseline per-item scored file** — set `BASELINE_SCORED_PATH` at top of the notebook
  to your existing scored.jsonl for the Fable-5 fewshot baseline. If the file
  contains multiple configs, `analyze_ablations.py` will filter by
  `vlm_alias=="fable"` and `prompt_style=="fewshot"`.

## Cost (estimates)

| ablation | generation | scoring | total wallclock |
|----------|-----------|---------|-----------------|
| A1 no_router | ~70 min (150 × 28s) | ~30 min | ~100 min |
| A2 no_vlm_filter | ~70 min | ~30 min | ~100 min |
| A3 no_graph | ~70 min | ~30 min | ~100 min |
| A4 no_rule_type | ~70 min | ~30 min | ~100 min |
| A5 no_hierarchy | ~70 min | ~30 min | ~100 min |
| A6 no_reranker | ~70 min | ~30 min | ~100 min |
| **six total** | **~7 h** | **~3 h** | **~10 h wallclock** |

Resume-friendly — kill and restart, both loops skip already-completed records.

**Cheap first-pass option:** open `run_ablations.ipynb` and set
`RETRIEVAL_ONLY=True` in Cell 3. Skips generation, computes just the retrieval
metrics (Recall@5, MRR, precision) per ablation in minutes each. Use to
deprioritize dead components before committing the full 10 h.

## Reproduction

```bash
# in Colab
1. Open ablations/run_ablations.ipynb
2. Set BASELINE_SCORED_PATH to your Fable 5 baseline scored file
3. Select ablations to run (default: all six)
4. Run cells top-to-bottom
```

or headless:

```bash
python -m ablations.run_ablation --config ablations/configs/A1_no_router.yaml
python -m ablations.score_ablation --runs results/runs_A1.jsonl
python -m ablations.analyze_ablations \
    --baseline path/to/fable5_scored.jsonl \
    --ablations results/scored_A*.jsonl \
    --out results/summary.csv
```

## Outputs

- `results/runs_<id>.jsonl` — one row per (qa_id, ablation): question, RAG answer, retrieval debug, figures_used, pages_used, latency
- `results/scored_<id>.jsonl` — one row per (qa_id, ablation): all rubric dimensions from GPT-5.6
- `results/summary.csv` — one row per ablation: mean Δ (paired), bootstrap 95% CI, p-value, mean scores per dimension
- `results/delta_plot.png` — forest plot of Δ vs baseline with CIs

## Design notes

- **Leave-one-out, not compound.** Each ablation isolates ONE component's effect. Compound ablations conflate effects; not in this pass.
- **Paired bootstrap over per-item deltas** — not summary vs summary. Uses the same 150 questions across baseline and ablation, so the CI reflects true variance not sampling noise.
- **A6 via monkey-patch, not code fork.** `NoOpReranker` passes candidates through in dense-score order. Reversible per-run; original pipeline object restored after ablation.
- **Skipped in this study:** A7 sparse (already L1-degraded so identical to baseline — not meaningful), A8 v1-vs-v2 extraction (deferred, requires separate ingest).
