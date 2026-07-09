"""Question router: does this query need figures at all?

WHY
---
v1 retrieval ran figure paths A (KG cross-links) and C (ColPali visual)
unconditionally, then let the VLM filter prune 10 candidates to 4 — so EVERY
answer shipped ~4 figures even for "What does 'shall' mean in the MUTCD?".
The scoring run measured the damage: figure precision 6%. Most retrieved
figures were attached to questions that needed none.

DESIGN
------
Deterministic, explainable, cheap. Three signal tiers; earlier tiers win:

  1. HARD YES  — the query names a figure/table id, or uses an explicitly
                 visual verb pattern ("show", "what does … look like",
                 "diagram", "illustrat…", "draw", "picture", "image").
  2. HARD NO   — definition / meaning / pure-policy patterns with no visual
                 noun ("what does X mean", "define", "when is it required",
                 "may/shall/should …?" phrasing about permissions), and no
                 sign-code or figure-ish noun in the query.
  3. SOFT      — everything else. Score = visual-noun evidence
                 + KG prior (do the top retrieved chunks' sections cite
                 figures?) and compare against a threshold. An optional VLM
                 tie-break is available for the ambiguous band but is OFF by
                 default (deterministic evals reproduce exactly).

The router returns a FigureDecision with the fired rules, so eval runs can
log WHY figures were or weren't retrieved — no more silent behaviour.

USAGE
-----
    from mrag.question_router import decide_figures
    dec = decide_figures(query, kg=kg, top_chunks=chunks)   # after chunk retrieval
    if dec.needs_figures:
        ... run figure paths, cap at dec.max_figures ...
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Signal vocabularies                                                         #
# --------------------------------------------------------------------------- #

_ID_RE = re.compile(r"\b(Figure|Table)\s+[0-9]+[A-Z]{0,2}[-\u2010-\u2014]\d+\b",
                    re.IGNORECASE)

_VISUAL_VERBS = re.compile(
    r"\b(show(?:\s+me)?|look\s+like|looks\s+like|diagram|illustrat\w*|"
    r"draw(?:ing)?|picture|image|photo|depict\w*|appearance|visual\w*|"
    r"what\s+color|what\s+shape|layout\s+of|example\s+of\s+a?\s*sign)\b",
    re.IGNORECASE)

# Nouns whose answers usually benefit from an image even without a visual verb
_VISUAL_NOUNS = re.compile(
    r"\b(sign|signs|plaque|marking|markings|signal\s+(?:face|head|indication)s?|"
    r"beacon|crosswalk|arrow|symbol|legend|stripe|chevron|delineator|"
    r"barricade|channelizing|cone|gate|pavement\s+word)\b",
    re.IGNORECASE)

# Sign designation codes (R1-1, W13-20aP, ...) — strong visual evidence
_SIGN_CODE = re.compile(r"\b[RWDIMOE][A-Z]?\d{1,3}(?:-\d{1,3})?[a-zA-Z]?P?\b")

# Placement/arrangement questions usually have companion "Locations of ..."
# figures in the MUTCD — weak positive signal, lets the KG prior tip them.
_PLACEMENT = re.compile(
    r"\b(placed?|placement|locat\w+|position\w*|mount\w+|install\w+|"
    r"arrang\w+|spacing|where\s+should)\b", re.IGNORECASE)

# Pure-text intents: definitions, meanings, permissions, numeric standards
_TEXTUAL_PATTERNS = re.compile(
    r"\b(what\s+does\s+.{1,40}\bmean|meaning\s+of|defin\w+|"
    r"when\s+(?:is|are|shall|should|may|must)|is\s+it\s+(?:required|permitted|allowed)|"
    r"(?:minimum|maximum)\s+(?:height|width|size|distance|spacing|speed|time)|"
    r"how\s+(?:far|tall|wide|long|many\s+(?:feet|inches|seconds))|"
    r"who\s+(?:is|has)\s+(?:responsible|authority)|purpose\s+of)\b",
    re.IGNORECASE)


@dataclass
class FigureDecision:
    needs_figures: bool
    confidence: float                  # 0..1
    max_figures: int                   # suggested display cap
    rules_fired: List[str] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Decision                                                                    #
# --------------------------------------------------------------------------- #

def decide_figures(
    query: str,
    kg=None,
    top_chunks: Optional[List[Dict[str, Any]]] = None,
    soft_threshold: float = 0.5,
    vlm_tiebreak=None,
) -> FigureDecision:
    """Decide whether figure retrieval should run for this query.

    kg          optional mrag.kg.KG — enables the section-prior signal
    top_chunks  chunk payloads already retrieved (dicts with 'section_id') —
                the router runs AFTER text retrieval, so this is free signal
    vlm_tiebreak optional callable(query) -> bool for the ambiguous band;
                leave None for fully deterministic behaviour
    """
    q = query.strip()
    rules: List[str] = []
    dbg: Dict[str, Any] = {}

    # ---- tier 1: hard YES ----------------------------------------------------
    if _ID_RE.search(q):
        rules.append("hard_yes:explicit_figure_id")
        return FigureDecision(True, 0.99, 4, rules, dbg)
    if _VISUAL_VERBS.search(q):
        rules.append("hard_yes:visual_verb")
        return FigureDecision(True, 0.95, 4, rules, dbg)

    has_code = bool(_SIGN_CODE.search(q))
    has_noun = bool(_VISUAL_NOUNS.search(q))
    textual = bool(_TEXTUAL_PATTERNS.search(q))
    dbg.update(sign_code=has_code, visual_noun=has_noun, textual_pattern=textual)

    # ---- tier 2: hard NO -------------------------------------------------------
    # A definitional / numeric-standard / permission question with no visual
    # noun and no sign code virtually never needs an image.
    if textual and not has_noun and not has_code:
        rules.append("hard_no:textual_pattern_no_visual_evidence")
        return FigureDecision(False, 0.9, 0, rules, dbg)

    # ---- tier 3: soft scoring ---------------------------------------------------
    score = 0.0
    if has_code:
        score += 0.45
        rules.append("soft:+0.45 sign_code")
    if has_noun:
        score += 0.30
        rules.append("soft:+0.30 visual_noun")
    if textual:
        score -= 0.35
        rules.append("soft:-0.35 textual_pattern")
    if _PLACEMENT.search(q):
        score += 0.15
        rules.append("soft:+0.15 placement_verb")

    # KG prior: do the sections behind the top retrieved chunks cite figures?
    if kg is not None and top_chunks:
        secs = []
        for ch in top_chunks[:5]:
            sid = ch.get("section_id") or ch.get("section")
            if sid:
                secs.append(sid)
        if secs:
            frac = sum(kg.section_cites_figures(s) for s in secs) / len(secs)
            bump = 0.35 * frac
            score += bump
            rules.append(f"soft:+{bump:.2f} kg_prior({frac:.0%} of top sections cite figures)")
            dbg["kg_prior_fraction"] = frac

    dbg["soft_score"] = round(score, 3)

    # Optional VLM tie-break only in the genuinely ambiguous band
    if vlm_tiebreak is not None and abs(score - soft_threshold) < 0.15:
        try:
            v = bool(vlm_tiebreak(q))
            rules.append(f"vlm_tiebreak:{v}")
            return FigureDecision(v, 0.6, 3 if v else 0, rules, dbg)
        except Exception as e:                              # pragma: no cover
            rules.append(f"vlm_tiebreak_failed:{e!r}")

    needs = score >= soft_threshold
    rules.append(f"soft:{'yes' if needs else 'no'} (score {score:.2f} vs {soft_threshold})")
    conf = min(0.9, 0.5 + abs(score - soft_threshold))
    return FigureDecision(needs, conf, 3 if needs else 0, rules, dbg)
