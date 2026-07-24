"""
Runtime ablation shims for MRAG_stp2.

Applies leave-one-out ablations by mutating CFG values and/or swapping pipeline
components with no-ops. Does NOT modify the mrag/ package. Each call is
reversible via the returned undo() callable.

No GPT calls, no external network dependency.

Usage:
    from ablations.ablation_shims import apply_ablation
    undo = apply_ablation(pipeline, "A1_no_router")
    try:
        result = ask(question)
    finally:
        undo()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NoOp reranker for A6
# ---------------------------------------------------------------------------

class NoOpReranker:
    """
    Drop-in replacement for the mxbai-rerank reranker used in retrieval.py.

    Returns candidates in the original order with monotonically decreasing
    pseudo-scores so downstream code that sorts by score still works. Matches
    both the mxbai `rerank(query, documents=...)` interface and the raw
    `compute_score(pairs)` interface, so it survives whichever call site the
    pipeline uses.
    """

    def rerank(self, query: str, documents: List[Any], **kwargs) -> List[Dict[str, Any]]:
        return [
            {"corpus_id": i, "score": 1.0 - i * 1e-6, "text": documents[i]}
            for i in range(len(documents))
        ]

    def compute_score(self, pairs, **kwargs) -> List[float]:
        return [1.0 - i * 1e-6 for i in range(len(pairs))]

    def __getattr__(self, name):
        def _stub(*args, **kwargs):
            log.warning(
                "NoOpReranker: unexpected call to %r — returning identity passthrough. "
                "If reranker is not fully disabled in A6 runs, patch this method.",
                name,
            )
            return None
        return _stub


# ---------------------------------------------------------------------------
# Ablation registry
# ---------------------------------------------------------------------------

@dataclass
class Ablation:
    id: str
    description: str
    cfg_patches: Dict[str, Any] = field(default_factory=dict)
    swap_reranker: bool = False


ABLATIONS: Dict[str, Ablation] = {
    "baseline": Ablation(
        id="baseline",
        description="Production defaults. Optional control run; the actual baseline "
                    "is your existing MUTCD-150 scored file.",
    ),
    "A1_no_router": Ablation(
        id="A1_no_router",
        description="Disable question router.",
        cfg_patches={"use_question_router": False},
    ),
    "A2_no_vlm_filter": Ablation(
        id="A2_no_vlm_filter",
        description="Disable VLM figure filter.",
        cfg_patches={"use_vlm_figure_filter": False},
    ),
    "A3_no_graph": Ablation(
        id="A3_no_graph",
        description="Zero graph proximity contribution.",
        cfg_patches={"w_graph": 0.0},
    ),
    "A4_no_rule_type": Ablation(
        id="A4_no_rule_type",
        description="Zero rule-type weight contribution.",
        cfg_patches={"w_ruletype": 0.0},
    ),
    "A5_no_hierarchy": Ablation(
        id="A5_no_hierarchy",
        description="Zero hierarchy prior contribution.",
        cfg_patches={"w_hierarchy": 0.0},
    ),
    "A6_no_reranker": Ablation(
        id="A6_no_reranker",
        description="Swap cross-encoder reranker for NoOp passthrough.",
        swap_reranker=True,
    ),
}


# ---------------------------------------------------------------------------
# Application + undo
# ---------------------------------------------------------------------------

def _get_cfg():
    """Import CFG lazily so this module has no import-time dependency on mrag."""
    from mrag.config import CFG
    return CFG


def apply_ablation(pipeline, ablation_id: str) -> Callable[[], None]:
    """
    Apply the named ablation to CFG and, if applicable, the pipeline object.
    Returns an `undo()` callable that restores the exact prior state.
    """
    if ablation_id not in ABLATIONS:
        raise KeyError(
            f"Unknown ablation {ablation_id!r}. Known: {sorted(ABLATIONS.keys())}"
        )

    ablation = ABLATIONS[ablation_id]
    cfg = _get_cfg()

    cfg_snapshot: Dict[str, Any] = {}
    for key, new_value in ablation.cfg_patches.items():
        if not hasattr(cfg, key):
            log.warning(
                "CFG has no attribute %r — ablation %s may not affect the pipeline. "
                "Verify config.py for the correct name.",
                key, ablation_id,
            )
        cfg_snapshot[key] = getattr(cfg, key, None)
        setattr(cfg, key, new_value)
        log.info("[%s] CFG.%s: %r → %r", ablation_id, key, cfg_snapshot[key], new_value)

    reranker_snapshot = None
    if ablation.swap_reranker:
        if not hasattr(pipeline, "reranker"):
            log.error(
                "[%s] pipeline has no `reranker` attribute — cannot apply A6.",
                ablation_id,
            )
        else:
            reranker_snapshot = pipeline.reranker
            pipeline.reranker = NoOpReranker()
            log.info("[%s] pipeline.reranker: swapped for NoOpReranker", ablation_id)

    def undo():
        for key, old_value in cfg_snapshot.items():
            setattr(cfg, key, old_value)
        if reranker_snapshot is not None:
            pipeline.reranker = reranker_snapshot
        log.info("[%s] undone", ablation_id)

    return undo


def list_ablations() -> List[Dict[str, str]]:
    return [{"id": a.id, "description": a.description} for a in ABLATIONS.values()]


def verify_ablation_applied(ablation_id: str, pipeline) -> Dict[str, Any]:
    """Sanity-check the ablation is actually in effect."""
    ablation = ABLATIONS[ablation_id]
    cfg = _get_cfg()
    checks: Dict[str, Any] = {}
    for key, expected in ablation.cfg_patches.items():
        actual = getattr(cfg, key, "<missing>")
        checks[f"CFG.{key}"] = {
            "expected": expected, "actual": actual, "ok": actual == expected,
        }
    if ablation.swap_reranker:
        is_noop = isinstance(getattr(pipeline, "reranker", None), NoOpReranker)
        checks["pipeline.reranker"] = {
            "expected": "NoOpReranker",
            "actual": type(getattr(pipeline, "reranker", None)).__name__,
            "ok": is_noop,
        }
    checks["_all_ok"] = all(c.get("ok", True) for c in checks.values() if isinstance(c, dict))
    return checks
