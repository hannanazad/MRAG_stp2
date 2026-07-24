"""
Run one ablation sweep over gold_qa.jsonl.

For each of the 150 questions in the gold set, calls ask() with the ablation
applied, records the RAG output, and writes to results/runs_<ablation_id>.jsonl.

Resume-safe: skips (qa_id) already present in the output file. Errored runs
are NOT counted as complete (LE13), so re-running retries just the failures.

Usage:
    python -m ablations.run_ablation --config ablations/configs/A1_no_router.yaml
    python -m ablations.run_ablation --ablation A1_no_router --gold path/to/gold_qa.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

log = logging.getLogger("ablations.run")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def completed_qa_ids(runs_path: Path) -> Set[str]:
    """Return qa_ids for records that already ran SUCCESSFULLY (LE13)."""
    if not runs_path.exists():
        return set()
    done = set()
    for row in load_jsonl(runs_path):
        if not row.get("error"):
            done.add(row["qa_id"])
    return done


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[Path]) -> Dict[str, Any]:
    if config_path is None:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML not installed. `pip install pyyaml` or pass --ablation instead.")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_ablation_sweep(
    ablation_id: str,
    gold_path: Path,
    output_path: Path,
    vlm_alias: str = "fable",
    prompt_style: str = "fewshot",
    limit: Optional[int] = None,
    retrieval_only: bool = False,
) -> None:
    """
    Run one ablation over every question in gold_qa.jsonl.
    """
    # Late imports so this file can be argparsed before mrag is on the path.
    from mrag.config import CFG
    from mrag.ask import ask
    from mrag.pipeline import init_pipeline  # if the actual name differs, adjust here
    from ablations.ablation_shims import apply_ablation, verify_ablation_applied

    # Freeze model + prompt style for the study.
    CFG.set_vlm_model(vlm_alias)
    CFG.set_answer_style(prompt_style)
    log.info("VLM=%s   answer_style=%s   ablation=%s",
             vlm_alias, prompt_style, ablation_id)

    pipeline = init_pipeline()
    undo = apply_ablation(pipeline, ablation_id)

    try:
        checks = verify_ablation_applied(ablation_id, pipeline)
        log.info("Ablation checks: %s", json.dumps(checks, default=str, indent=2))
        if not checks.get("_all_ok", False):
            raise RuntimeError(f"Ablation {ablation_id} failed verification: {checks}")

        done = completed_qa_ids(output_path)
        log.info("Resume: %d already completed in %s", len(done), output_path)

        gold_records = list(load_jsonl(gold_path))
        if limit:
            gold_records = gold_records[:limit]

        n_total = len(gold_records)
        n_run = 0
        n_err = 0
        t0 = time.time()

        for i, qa in enumerate(gold_records, 1):
            qa_id = qa["qa_id"]
            if qa_id in done:
                continue
            question = qa["question"]

            record = {
                "qa_id": qa_id,
                "ablation": ablation_id,
                "vlm_alias": vlm_alias,
                "prompt_style": prompt_style,
                "question": question,
            }

            t_start = time.time()
            try:
                if retrieval_only:
                    # Just retrieve; skip generation. Requires the pipeline to
                    # expose a retrieve-only entrypoint; if not, fall through
                    # to full ask() and drop the answer.
                    retriever = getattr(pipeline, "retrieve", None)
                    if callable(retriever):
                        r = retriever(question)
                        record.update({
                            "chunks_used": r.get("chunks", []),
                            "figures_used": r.get("figures", []),
                            "pages_used": r.get("pages", []),
                            "debug": r.get("debug", {}),
                            "answer": None,
                            "retrieval_only": True,
                        })
                    else:
                        r = ask(question)
                        record.update({
                            "chunks_used": r.get("chunks_used", []),
                            "figures_used": r.get("figures", []),
                            "pages_used": r.get("pages_used", []),
                            "debug": r.get("debug", {}),
                            "answer": None,  # dropped
                            "retrieval_only": True,
                        })
                else:
                    r = ask(question)
                    record.update({
                        "answer": r.get("answer"),
                        "chunks_used": r.get("chunks_used", []),
                        "figures_used": r.get("figures", []),
                        "pages_used": r.get("pages_used", []),
                        "debug": r.get("debug", {}),
                        "retrieval_only": False,
                    })
                record["latency_s"] = time.time() - t_start
                record["error"] = None
                n_run += 1
            except Exception as e:
                record["latency_s"] = time.time() - t_start
                record["error"] = f"{type(e).__name__}: {e}"
                record["answer"] = None
                n_err += 1
                log.exception("[%s] qa_id=%s failed", ablation_id, qa_id)

            append_jsonl(output_path, record)

            if (i % 10) == 0 or i == n_total:
                elapsed = time.time() - t0
                rate = n_run / max(elapsed, 1e-6)
                remaining = (n_total - i) / max(rate, 1e-6)
                log.info(
                    "[%s] %d/%d  ok=%d err=%d  %.1f min elapsed, ~%.1f min remaining",
                    ablation_id, i, n_total, n_run, n_err,
                    elapsed / 60, remaining / 60,
                )

        log.info("[%s] DONE. total=%d ok=%d err=%d",
                 ablation_id, n_total, n_run, n_err)

    finally:
        undo()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description="Run one MRAG ablation sweep.")
    p.add_argument("--config", type=Path, default=None,
                   help="YAML config with ablation_id + optional overrides.")
    p.add_argument("--ablation", type=str, default=None,
                   help="Ablation id (overrides config). e.g. A1_no_router")
    p.add_argument("--gold", type=Path, default=None,
                   help="Path to gold_qa.jsonl (150 questions).")
    p.add_argument("--output", type=Path, default=None,
                   help="Output runs_<id>.jsonl path.")
    p.add_argument("--vlm", type=str, default=None, help="VLM alias (default: fable).")
    p.add_argument("--prompt-style", type=str, default=None,
                   help="Answer prompt style (default: fewshot).")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on number of questions (for debugging).")
    p.add_argument("--retrieval-only", action="store_true",
                   help="Skip generation; capture retrieval outputs only.")
    args = p.parse_args()

    cfg = load_config(args.config)
    ablation_id = args.ablation or cfg.get("ablation_id")
    if not ablation_id:
        p.error("Must supply --ablation or a --config with ablation_id set.")

    gold = args.gold or Path(cfg.get("gold_path", "/content/drive/MyDrive/MRAG/eval/gold_qa.jsonl"))
    default_output = Path(f"ablations/results/runs_{ablation_id}.jsonl")
    output = args.output or Path(cfg.get("output_path", default_output))
    vlm = args.vlm or cfg.get("vlm_alias", "fable")
    prompt_style = args.prompt_style or cfg.get("prompt_style", "fewshot")
    retrieval_only = args.retrieval_only or bool(cfg.get("retrieval_only", False))

    run_ablation_sweep(
        ablation_id=ablation_id,
        gold_path=gold,
        output_path=output,
        vlm_alias=vlm,
        prompt_style=prompt_style,
        limit=args.limit,
        retrieval_only=retrieval_only,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
