"""
Package one ablation's results into a zip on Drive.

Zip contents:
  runs_<ablation_id>.jsonl   — raw RAG outputs, one row per question
  meta.json                  — ablation config + timestamp + counters + git sha
  README.txt                 — brief handoff note for GPT scoring

Does NOT call GPT, does NOT compute scores, does NOT interpret. Just packaging.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("ablations.package")


README_TEMPLATE = """MRAG Ablation Results — {ablation_id}

Contents:
  runs_{ablation_id}.jsonl   Raw RAG outputs from the ablation sweep.
                             One JSON row per question with:
                               qa_id, question, answer, chunks_used,
                               figures_used, pages_used, debug, latency_s, error
  meta.json                  Ablation config + timestamp + counters + git sha.

This zip is data collection only. No scoring, no metrics, no interpretation
was applied by the harness that produced it.

Hand this file to GPT-5.6 with the MUTCD-150 rubric prompt for judgment.

Ablation: {ablation_id}
Description: {description}
Generated: {timestamp}
VLM alias: {vlm_alias}
Prompt style: {prompt_style}
Records ok: {n_ok}
Records errored: {n_err}
Total questions: {n_total}
"""


def _git_sha(repo_dir: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def _count_rows(runs_path: Path) -> Dict[str, int]:
    n_total = n_ok = n_err = 0
    if not runs_path.exists():
        return {"total": 0, "ok": 0, "err": 0}
    with open(runs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_total += 1
            if row.get("error"):
                n_err += 1
            else:
                n_ok += 1
    return {"total": n_total, "ok": n_ok, "err": n_err}


def package_ablation(
    ablation_id: str,
    runs_path: Path,
    zip_output_path: Path,
    vlm_alias: str = "fable",
    prompt_style: str = "fewshot",
    repo_dir: Optional[Path] = None,
    description: str = "",
) -> Path:
    """
    Bundle one ablation's runs.jsonl into a zip with meta.json + README.txt.
    Returns the zip path.
    """
    if not runs_path.exists():
        raise FileNotFoundError(f"runs file not found: {runs_path}")

    counts = _count_rows(runs_path)
    timestamp = _dt.datetime.now().isoformat(timespec="seconds")
    git_sha = _git_sha(repo_dir) if repo_dir else None

    meta: Dict[str, Any] = {
        "ablation_id": ablation_id,
        "description": description,
        "timestamp": timestamp,
        "vlm_alias": vlm_alias,
        "prompt_style": prompt_style,
        "n_total": counts["total"],
        "n_ok": counts["ok"],
        "n_err": counts["err"],
        "git_sha": git_sha,
        "runs_file": runs_path.name,
        "harness_version": "ablations_v2",
    }

    readme = README_TEMPLATE.format(
        ablation_id=ablation_id,
        description=description or "(see ablation_shims.py)",
        timestamp=timestamp,
        vlm_alias=vlm_alias,
        prompt_style=prompt_style,
        n_ok=counts["ok"],
        n_err=counts["err"],
        n_total=counts["total"],
    )

    zip_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temp path first, then move — avoids leaving a half-written zip on Drive.
    tmp = zip_output_path.with_suffix(zip_output_path.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(runs_path, arcname=runs_path.name)
        zf.writestr("meta.json", json.dumps(meta, indent=2))
        zf.writestr("README.txt", readme)
    tmp.replace(zip_output_path)

    log.info("Packaged %s → %s  (rows=%d ok=%d err=%d)",
             ablation_id, zip_output_path, counts["total"], counts["ok"], counts["err"])
    return zip_output_path


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description="Package one ablation's results into a zip.")
    p.add_argument("--ablation", type=str, required=True)
    p.add_argument("--runs", type=Path, required=True, help="Path to runs_<id>.jsonl")
    p.add_argument("--out", type=Path, required=True, help="Output zip path (on Drive).")
    p.add_argument("--vlm", type=str, default="fable")
    p.add_argument("--prompt-style", type=str, default="fewshot")
    p.add_argument("--repo-dir", type=Path, default=None, help="For git sha capture.")
    p.add_argument("--description", type=str, default="")
    args = p.parse_args()

    package_ablation(
        ablation_id=args.ablation,
        runs_path=args.runs,
        zip_output_path=args.out,
        vlm_alias=args.vlm,
        prompt_style=args.prompt_style,
        repo_dir=args.repo_dir,
        description=args.description,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
