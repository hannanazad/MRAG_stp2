"""
Analyze ablation results vs the baseline.

Computes:
- Per-item Δ = ablation.total_over_97 - baseline.total_over_97 (paired by qa_id)
- Mean Δ, paired bootstrap 95% CI (B=10000 by default)
- Approximate two-sided p-value (fraction of bootstrap means with sign flip)
- Per-dimension means for each ablation vs baseline

Writes:
- results/summary.csv — one row per ablation
- results/per_item_deltas.jsonl — for downstream stats / paper figures
- results/delta_plot.png — forest plot of Δ with CIs

Usage:
    python -m ablations.analyze_ablations \\
        --baseline path/to/fable5_scored.jsonl \\
        --ablations ablations/results/scored_A*.jsonl \\
        --out ablations/results/summary.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("ablations.analyze")

TARGET_METRIC = "total_over_97"

BASELINE_FILTER = {
    "vlm_alias": "fable",
    "prompt_style": "fewshot",
}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def filter_baseline(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter a scored file down to just the Fable-5 fewshot baseline rows.
    Falls back to no-filter if the file already contains only one config
    (heuristic: no vlm_alias field on any row).
    """
    have_field = any("vlm_alias" in r or "prompt_style" in r or "config" in r for r in records)
    if not have_field:
        log.info("Baseline file has no config identifiers — assuming single-config file, no filter applied.")
        return records
    filtered = []
    for r in records:
        # Support both flat and nested-config shapes.
        vlm = r.get("vlm_alias") or (r.get("config") or {}).get("vlm_alias")
        style = r.get("prompt_style") or (r.get("config") or {}).get("prompt_style")
        if vlm and vlm != BASELINE_FILTER["vlm_alias"]:
            continue
        if style and style != BASELINE_FILTER["prompt_style"]:
            continue
        filtered.append(r)
    log.info("Baseline filter: %d/%d rows kept (vlm=%s, prompt_style=%s)",
             len(filtered), len(records),
             BASELINE_FILTER["vlm_alias"], BASELINE_FILTER["prompt_style"])
    return filtered


def get_total(record: Dict[str, Any]) -> Optional[float]:
    """Extract the /97 score. Handles both flat and nested shapes."""
    if TARGET_METRIC in record and record[TARGET_METRIC] is not None:
        return float(record[TARGET_METRIC])
    # Sometimes buried under "score" or the full rubric.
    score = record.get("score") or record.get("scores")
    if isinstance(score, dict) and TARGET_METRIC in score:
        return float(score[TARGET_METRIC])
    return None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(deltas: List[float], B: int = 10000,
                        alpha: float = 0.05, seed: int = 42) -> Dict[str, float]:
    """
    Paired bootstrap CI for the mean of `deltas`.
    Returns mean, lower, upper, approximate two-sided p-value.
    """
    try:
        import numpy as np  # type: ignore
    except ImportError:
        raise RuntimeError("numpy required for analyze. `pip install numpy`.")

    rng = np.random.default_rng(seed)
    arr = np.array(deltas, dtype=float)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "p_two_sided": 1.0, "n": 0}
    idx = rng.integers(0, n, size=(B, n))
    means = arr[idx].mean(axis=1)
    lower = float(np.quantile(means, alpha / 2))
    upper = float(np.quantile(means, 1 - alpha / 2))
    observed_mean = float(arr.mean())
    # Two-sided empirical p: fraction of bootstrap means on the opposite side of 0.
    if observed_mean >= 0:
        p = float((means <= 0).sum() / B) * 2
    else:
        p = float((means >= 0).sum() / B) * 2
    return {
        "mean": observed_mean,
        "lower": lower,
        "upper": upper,
        "p_two_sided": min(p, 1.0),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Analyze one ablation
# ---------------------------------------------------------------------------

def analyze_one(
    ablation_records: List[Dict[str, Any]],
    baseline_by_id: Dict[str, float],
    ablation_id: str,
    B: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Returns (summary_row, per_item_deltas)."""
    per_item = []
    deltas = []
    for r in ablation_records:
        if r.get("judge_error"):
            continue
        qa_id = r["qa_id"]
        b_score = baseline_by_id.get(qa_id)
        a_score = get_total(r)
        if b_score is None or a_score is None:
            continue
        d = a_score - b_score
        per_item.append({
            "qa_id": qa_id,
            "ablation": ablation_id,
            "baseline_score": b_score,
            "ablation_score": a_score,
            "delta": d,
        })
        deltas.append(d)

    stats = paired_bootstrap_ci(deltas, B=B)
    row = {
        "ablation": ablation_id,
        "n_paired": stats["n"],
        "mean_baseline": (sum(x["baseline_score"] for x in per_item) / len(per_item)) if per_item else None,
        "mean_ablation": (sum(x["ablation_score"] for x in per_item) / len(per_item)) if per_item else None,
        "mean_delta": stats["mean"],
        "ci_low_95": stats["lower"],
        "ci_high_95": stats["upper"],
        "p_two_sided": stats["p_two_sided"],
        "sig_at_05": (stats["lower"] > 0 or stats["upper"] < 0),
    }
    return row, per_item


# ---------------------------------------------------------------------------
# Forest plot
# ---------------------------------------------------------------------------

def make_forest_plot(summary_rows: List[Dict[str, Any]], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        log.warning("matplotlib not installed; skipping forest plot.")
        return

    rows = [r for r in summary_rows if r["mean_delta"] is not None]
    if not rows:
        log.warning("Nothing to plot.")
        return
    rows = sorted(rows, key=lambda r: r["mean_delta"])
    ids = [r["ablation"] for r in rows]
    means = [r["mean_delta"] for r in rows]
    lows = [r["mean_delta"] - r["ci_low_95"] for r in rows]
    highs = [r["ci_high_95"] - r["mean_delta"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    y = list(range(len(ids)))
    ax.errorbar(means, y, xerr=[lows, highs], fmt="o", capsize=4, color="black")
    ax.axvline(0, linestyle="--", color="gray", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(ids)
    ax.set_xlabel("Δ vs baseline (points on /97 scale)")
    ax.set_title("Ablation study: paired Δ vs Fable-5 fewshot baseline\n(bars = bootstrap 95% CI, n=150)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    log.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# CSV writer (avoids pandas dep)
# ---------------------------------------------------------------------------

def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    import csv
    if not rows:
        log.warning("No rows to write to %s", out_path)
        return
    fields = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log.info("Wrote %s (%d rows)", out_path, len(rows))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze(
    baseline_path: Path,
    ablation_paths: List[Path],
    summary_path: Path,
    deltas_path: Optional[Path] = None,
    plot_path: Optional[Path] = None,
    B: int = 10000,
) -> None:
    baseline_raw = load_jsonl(baseline_path)
    baseline_filtered = filter_baseline(baseline_raw)
    baseline_by_id: Dict[str, float] = {}
    for r in baseline_filtered:
        if r.get("judge_error"):
            continue
        t = get_total(r)
        if t is not None:
            baseline_by_id[r["qa_id"]] = t
    log.info("Baseline: %d qa_ids with valid scores", len(baseline_by_id))
    if not baseline_by_id:
        raise RuntimeError(
            "No valid baseline scores loaded. Check the baseline file path and "
            "that it contains total_over_97 values."
        )

    summary_rows: List[Dict[str, Any]] = []
    all_deltas: List[Dict[str, Any]] = []

    for ap in ablation_paths:
        records = load_jsonl(ap)
        if not records:
            log.warning("Skipping empty file %s", ap)
            continue
        ablation_id = records[0].get("ablation") or ap.stem.replace("scored_", "")
        row, per_item = analyze_one(records, baseline_by_id, ablation_id, B=B)
        summary_rows.append(row)
        all_deltas.extend(per_item)
        log.info(
            "%s  n=%d  Δ=%+.3f  CI=[%+.3f, %+.3f]  p=%.3f  sig=%s",
            ablation_id, row["n_paired"], row["mean_delta"] or 0.0,
            row["ci_low_95"] or 0.0, row["ci_high_95"] or 0.0,
            row["p_two_sided"] or 1.0, row["sig_at_05"],
        )

    write_csv(summary_rows, summary_path)
    if deltas_path:
        deltas_path.parent.mkdir(parents=True, exist_ok=True)
        with open(deltas_path, "w", encoding="utf-8") as f:
            for r in all_deltas:
                f.write(json.dumps(r) + "\n")
        log.info("Wrote %s (%d per-item deltas)", deltas_path, len(all_deltas))
    if plot_path:
        make_forest_plot(summary_rows, plot_path)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description="Analyze MRAG ablations vs baseline.")
    p.add_argument("--baseline", type=Path, required=True,
                   help="Baseline scored jsonl (Fable-5 fewshot, 150 rows).")
    p.add_argument("--ablations", type=str, nargs="+", required=True,
                   help="Glob(s) for scored_<id>.jsonl files.")
    p.add_argument("--out", type=Path, required=True,
                   help="Path to summary.csv")
    p.add_argument("--deltas-out", type=Path, default=None,
                   help="Optional per-item deltas jsonl output.")
    p.add_argument("--plot-out", type=Path, default=None,
                   help="Optional forest-plot PNG output.")
    p.add_argument("--B", type=int, default=10000, help="Bootstrap resamples.")
    args = p.parse_args()

    paths: List[Path] = []
    for pattern in args.ablations:
        matched = [Path(p) for p in sorted(glob.glob(pattern))]
        paths.extend(matched)
    if not paths:
        p.error(f"No files matched --ablations {args.ablations}")

    analyze(
        baseline_path=args.baseline,
        ablation_paths=paths,
        summary_path=args.out,
        deltas_path=args.deltas_out,
        plot_path=args.plot_out,
        B=args.B,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
