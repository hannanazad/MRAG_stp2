"""
Mechanical ablation sweep: runs one ablation across gold_qa.jsonl, captures
raw RAG output per question, writes results/runs_<ablation_id>.jsonl.

DOES NOT:
  - call GPT or any external judge
  - compute any score or metric
  - print, log, or otherwise display question text (only qa_id appears in logs)
  - interpret results

Resume-safe: skips qa_ids that already ran SUCCESSFULLY. Errored records are
retryable — re-run this script and the errored qa_ids get another attempt.
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
except ImportError:
    yaml = None

log = logging.getLogger("ablations.run")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _completed_qa_ids(runs_path: Path) -> Set[str]:
    """qa_ids that ran successfully (errored ones are retryable)."""
    if not runs_path.exists():
        return set()
    done: Set[str] = set()
    with open(runs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("error"):
                done.add(row["qa_id"])
    return done


def _errored_qa_ids(runs_path: Path) -> Set[str]:
    if not runs_path.exists():
        return set()
    errored: Set[str] = set()
    with open(runs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("error"):
                errored.add(row["qa_id"])
    return errored


def _load_config(config_path: Optional[Path]) -> Dict[str, Any]:
    if config_path is None:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML not installed. `pip install pyyaml`.")
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
) -> Dict[str, int]:
    """
    Sweep every question in gold_qa.jsonl for one ablation.

    Returns a small counters dict. Never returns or logs question strings.
    """
    from mrag.config import CFG
    from mrag.ask import ask
    try:
        from mrag.pipeline import init_pipeline
    except ImportError:
        from mrag.ask import init_pipeline  # fallback location
    from ablations.ablation_shims import apply_ablation, verify_ablation_applied

    CFG.set_vlm_model(vlm_alias)
    CFG.set_answer_style(prompt_style)
    log.info("VLM=%s   answer_style=%s   ablation=%s   retrieval_only=%s",
             vlm_alias, prompt_style, ablation_id, retrieval_only)

    pipeline = init_pipeline()
    undo = apply_ablation(pipeline, ablation_id)

    counters = {"total": 0, "already_done": 0, "ok": 0, "err": 0, "skipped": 0}

    try:
        checks = verify_ablation_applied(ablation_id, pipeline)
        log.info("Ablation checks: %s", json.dumps(checks, default=str))
        if not checks.get("_all_ok", False):
            raise RuntimeError(f"Ablation {ablation_id} failed verification: {checks}")

        done = _completed_qa_ids(output_path)
        counters["already_done"] = len(done)
        log.info("Resume: %d already completed in %s", len(done), output_path.name)

        # Optionally purge prior errored entries so they get a fresh attempt.
        errored = _errored_qa_ids(output_path)
        if errored:
            log.info("Retrying %d errored records from prior run.", len(errored))
            # Rewrite the file keeping only successful rows; errored ones will be re-run.
            _keep_only_successful(output_path)

        gold_records = list(_load_jsonl(gold_path))
        if limit:
            gold_records = gold_records[:limit]
        counters["total"] = len(gold_records)

        t0 = time.time()
        n_run = 0
        n_err = 0

        for i, qa in enumerate(gold_records, 1):
            qa_id = qa["qa_id"]
            if qa_id in done:
                counters["skipped"] += 1
                continue
            question = qa["question"]  # passed to ask() as a variable; never logged

            record: Dict[str, Any] = {
                "qa_id": qa_id,
                "ablation": ablation_id,
                "vlm_alias": vlm_alias,
                "prompt_style": prompt_style,
                "question": question,  # kept in output — GPT needs it for scoring
            }

            t_start = time.time()
            try:
                if retrieval_only:
                    retriever = getattr(pipeline, "retrieve", None)
                    if callable(retriever):
                        r = retriever(question)
                    else:
                        r = ask(question)
                    record.update({
                        "answer": None,
                        "chunks_used": r.get("chunks_used", r.get("chunks", [])),
                        "figures_used": r.get("figures", []),
                        "pages_used": r.get("pages_used", r.get("pages", [])),
                        "debug": r.get("debug", {}),
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
                # Log qa_id only — never the question text.
                log.exception("[%s] qa_id=%s failed", ablation_id, qa_id)

            _append_jsonl(output_path, record)

            if (i % 10) == 0 or i == counters["total"]:
                elapsed = time.time() - t0
                rate = n_run / max(elapsed, 1e-6)
                remaining = (counters["total"] - i) / max(rate, 1e-6)
                log.info(
                    "[%s] %d/%d  ok=%d err=%d  %.1f min elapsed, ~%.1f min remaining",
                    ablation_id, i, counters["total"], n_run, n_err,
                    elapsed / 60, remaining / 60,
                )

        counters["ok"] = n_run
        counters["err"] = n_err
        log.info("[%s] DONE. total=%d ok=%d err=%d skipped=%d",
                 ablation_id, counters["total"], n_run, n_err, counters["skipped"])

    finally:
        undo()

    return counters


def _keep_only_successful(runs_path: Path) -> None:
    """Rewrite runs file keeping only records with error is None."""
    if not runs_path.exists():
        return
    tmp = runs_path.with_suffix(runs_path.suffix + ".tmp")
    kept = 0
    dropped = 0
    with open(runs_path, "r", encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("error"):
                dropped += 1
                continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    tmp.replace(runs_path)
    log.info("Purged errored rows: kept=%d dropped=%d", kept, dropped)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description="Run one MRAG ablation sweep (data collection only).")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--ablation", type=str, default=None)
    p.add_argument("--gold", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--vlm", type=str, default=None)
    p.add_argument("--prompt-style", type=str, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--retrieval-only", action="store_true")
    args = p.parse_args()

    cfg = _load_config(args.config)
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
