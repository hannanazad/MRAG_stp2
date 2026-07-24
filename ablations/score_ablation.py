"""
Score an ablation's runs.jsonl using the same GPT-5.6-assisted rubric as
score_runs.ipynb (MUTCD-150 methodology).

Writes results/scored_<ablation_id>.jsonl with per-item rubric dimensions,
retrieval metrics (Recall@5, MRR, precision, exact-hit, crop-precision),
answer metrics (factual_accuracy, category_correctness, citation_validity,
verbatim_faithfulness, completeness, refusal_appropriateness), and totals
(retrieval_25, answer_60, reliability, total_over_97).

Resume-safe: skips qa_ids already scored successfully.

Usage:
    python -m ablations.score_ablation \\
        --runs ablations/results/runs_A1_no_router.jsonl \\
        --gold /content/drive/MyDrive/MRAG/eval/gold_qa.jsonl \\
        --out ablations/results/scored_A1_no_router.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

log = logging.getLogger("ablations.score")

# GPT-5.6-assisted rubric configuration.
# These defaults MIRROR score_runs.ipynb's MUTCD-150 setup — do not diverge
# without a paper-methodology note.
DEFAULT_JUDGE_MODEL = "gpt-5.6"
DEFAULT_MAX_OUT_TOKENS = 4000
DEFAULT_REASONING_EFFORT = "low"


RUBRIC_SYSTEM_PROMPT = """You are an expert evaluator of RAG systems that answer
questions about the MUTCD (Manual on Uniform Traffic Control Devices, 11th Ed).

You will be given:
1. A question from the MUTCD-150 gold set
2. The ground-truth gold answer (with references to sections/provisions)
3. The RAG system's answer
4. The RAG's retrieved chunks (with section IDs)
5. The RAG's retrieved figures (if any)

Score the RAG output on the MUTCD-150 rubric. Return valid JSON matching the
schema. Use `null` for dimensions that do not apply (e.g. refusal_appropriateness
on a non-refusal question).

Be strict on citation validity — a wrong section ID is a failure even if the
substance is correct.
"""


RUBRIC_SCHEMA_HINT = """
Return valid JSON with this exact structure:
{
  "retrieval": {
    "recall_at_5": 0.0-1.0,
    "mrr": 0.0-1.0,
    "context_precision": 0.0-1.0,
    "modality_routing_correct": true/false,
    "exact_hit_figure": true/false/null,
    "crop_precision": 0.0-1.0/null,
    "retrieval_25_points": 0-25
  },
  "answer": {
    "factual_accuracy": 0-5,
    "category_correctness": 0-5,
    "citation_validity": 0-5,
    "verbatim_faithfulness": 0-5/null,
    "completeness": 0-5,
    "refusal_appropriateness": 0-5/null,
    "figure_relevance": 0-5/null,
    "figure_grounding": 0-5/null,
    "answer_60_points": 0-60
  },
  "reliability": {
    "system_confidence": 0.0-1.0,
    "judge_confidence": 0.0-1.0,
    "reliability_points": 0-12
  },
  "total_over_97": 0.0-97.0,
  "explanations": {
    "strengths": "one sentence",
    "weaknesses": "one sentence"
  }
}
"""


def build_judge_prompt(qa: Dict[str, Any], run: Dict[str, Any]) -> str:
    """Assemble the user prompt for a single judge call."""
    gold_answer = qa.get("gold_answer", {})
    references = qa.get("references", [])
    gold_figures = qa.get("gold_figures", [])

    chunks_used = run.get("chunks_used", [])
    figures_used = run.get("figures_used", [])
    rag_answer = run.get("answer") or "(no answer produced)"

    parts = [
        f"QUESTION:\n{qa.get('question','')}\n",
        f"EXPECTED_REFUSAL: {qa.get('expected_refusal', False)}\n",
        "GOLD_ANSWER:\n" + json.dumps(gold_answer, ensure_ascii=False, indent=2),
        "GOLD_REFERENCES:\n" + json.dumps(references, ensure_ascii=False, indent=2),
        "GOLD_FIGURES:\n" + json.dumps(gold_figures, ensure_ascii=False, indent=2),
        "\nRAG_ANSWER:\n" + rag_answer,
        "\nRAG_RETRIEVED_CHUNKS (section_id + snippet):\n"
        + json.dumps(chunks_used[:6], ensure_ascii=False, indent=2)[:4000],
        "\nRAG_RETRIEVED_FIGURES:\n"
        + json.dumps([{"figure_id": f.get("figure_id"), "source": f.get("source")}
                      for f in figures_used], indent=2),
        "\n" + RUBRIC_SCHEMA_HINT,
    ]
    return "\n".join(parts)


def judge_one(client, model: str, qa: Dict[str, Any], run: Dict[str, Any],
              max_out_tokens: int, reasoning_effort: str) -> Dict[str, Any]:
    """Single GPT-5.6 judge call. Returns parsed JSON."""
    user_prompt = build_judge_prompt(qa, run)
    kwargs = dict(
        model=model,
        max_completion_tokens=max_out_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": RUBRIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    resp = client.chat.completions.create(**kwargs)

    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError(
            f"Empty judge response. finish_reason={resp.choices[0].finish_reason} "
            f"usage={getattr(resp, 'usage', None)}"
        )
    parsed = json.loads(content)
    parsed["_judge_model"] = getattr(resp, "model", model)
    parsed["_finish_reason"] = resp.choices[0].finish_reason
    return parsed


# ---------------------------------------------------------------------------
# I/O helpers (kept lightweight — no external deps beyond openai)
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def completed_ids(scored_path: Path) -> Set[str]:
    return {r["qa_id"] for r in load_jsonl(scored_path) if not r.get("judge_error")}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def score_runs(
    runs_path: Path,
    gold_path: Path,
    output_path: Path,
    model: str = DEFAULT_JUDGE_MODEL,
    max_out_tokens: int = DEFAULT_MAX_OUT_TOKENS,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    api_key: Optional[str] = None,
    limit: Optional[int] = None,
) -> None:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed. `pip install openai`.")

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set (env or --api-key).")
    client = OpenAI(api_key=api_key)

    runs = load_jsonl(runs_path)
    gold_by_id = {qa["qa_id"]: qa for qa in load_jsonl(gold_path)}
    already = completed_ids(output_path)
    log.info("runs=%d  gold=%d  already_scored=%d  target=%s",
             len(runs), len(gold_by_id), len(already), output_path)

    to_score = [r for r in runs if r["qa_id"] not in already and not r.get("error")]
    if limit:
        to_score = to_score[:limit]

    t0 = time.time()
    n_ok = 0
    n_err = 0
    for i, run in enumerate(to_score, 1):
        qa_id = run["qa_id"]
        qa = gold_by_id.get(qa_id)
        if qa is None:
            log.warning("qa_id %s not found in gold set — skipping", qa_id)
            continue

        record = {"qa_id": qa_id, "ablation": run.get("ablation")}
        try:
            scores = judge_one(client, model, qa, run,
                               max_out_tokens=max_out_tokens,
                               reasoning_effort=reasoning_effort)
            record.update(scores)
            record["judge_error"] = None
            n_ok += 1
        except Exception as e:
            record["judge_error"] = f"{type(e).__name__}: {e}"
            n_err += 1
            log.exception("judge failed for qa_id=%s", qa_id)

        append_jsonl(output_path, record)

        if (i % 10) == 0 or i == len(to_score):
            elapsed = (time.time() - t0) / 60
            log.info("scored %d/%d  ok=%d err=%d  %.1f min elapsed",
                     i, len(to_score), n_ok, n_err, elapsed)

    log.info("DONE. total=%d ok=%d err=%d output=%s",
             len(to_score), n_ok, n_err, output_path)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description="Score MRAG ablation runs via GPT-5.6.")
    p.add_argument("--runs", type=Path, required=True,
                   help="Path to runs_<id>.jsonl from run_ablation.py")
    p.add_argument("--gold", type=Path,
                   default=Path("/content/drive/MyDrive/MRAG/eval/gold_qa.jsonl"))
    p.add_argument("--out", type=Path, required=True,
                   help="Path to write scored_<id>.jsonl")
    p.add_argument("--model", type=str, default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--max-out-tokens", type=int, default=DEFAULT_MAX_OUT_TOKENS)
    p.add_argument("--reasoning-effort", type=str, default=DEFAULT_REASONING_EFFORT)
    p.add_argument("--api-key", type=str, default=None,
                   help="OpenAI API key (falls back to OPENAI_API_KEY env var).")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    score_runs(
        runs_path=args.runs,
        gold_path=args.gold,
        output_path=args.out,
        model=args.model,
        max_out_tokens=args.max_out_tokens,
        reasoning_effort=args.reasoning_effort,
        api_key=args.api_key,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
