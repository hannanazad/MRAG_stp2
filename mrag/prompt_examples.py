"""Worked examples for one-shot / few-shot prompting of the answer VLM.

Each example mirrors the structure of a real `ask()` call:
  - question     : the user's question
  - chunks       : list of evidence-chunk dicts (same shape produced by retrieval)
  - figures      : list of figure dicts (may be empty)
  - answer       : the desired model output, formatted per the spec in vlm.py

The chunks and answers below are drawn from real MUTCD 11th-Edition text so
the model learns accurate citation patterns, accurate verbatim quoting, and
correct use of the Standard / Guidance / Option / Support taxonomy. Do NOT
add fake section numbers, fake figure IDs, or paraphrased "standard"
provisions here — examples that lie teach the model to lie.

To add more examples, append to FEWSHOT_EXAMPLES. The first example in the
list is used for one-shot prompting; all examples are used for few-shot.
"""
from __future__ import annotations

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Examples — keep diverse: cover different MUTCD normative categories,
# different sections, and edge cases like "evidence spans many categories".
# ---------------------------------------------------------------------------

FEWSHOT_EXAMPLES: List[Dict[str, Any]] = [
    # -----------------------------------------------------------------------
    # Example 1 — Multi-category answer (Standard + Guidance + Option).
    # Best single example: shows the model how to handle answers that span
    # all four MUTCD rule types. Used as the one-shot example.
    # -----------------------------------------------------------------------
    {
        "question": "What supplemental plaques are used with STOP signs?",
        "chunks": [
            {
                "section_id": "2B.04",
                "section_title": "STOP Sign (R1-1) and ALL-WAY Plaque (R1-3P)",
                "content_type": "Standard",
                "ordinal": "04",
                "page_printed": "74",
                "text": (
                    "At intersections where all approaches are controlled by STOP signs "
                    "(see Section 2B.12), an ALL-WAY (R1-3P) supplemental plaque "
                    "(see Figure 2B-1) shall be mounted below each STOP sign. The ALL-WAY "
                    "plaque shall have a white legend and border on a red background."
                ),
            },
            {
                "section_id": "2B.04",
                "section_title": "STOP Sign (R1-1) and ALL-WAY Plaque (R1-3P)",
                "content_type": "Standard",
                "ordinal": "05",
                "page_printed": "74",
                "text": (
                    "Supplemental plaques with legends such as 2-WAY, 3-WAY, 4-WAY, or "
                    "other numbers of ways shall not be used with STOP signs."
                ),
            },
            {
                "section_id": "2B.04",
                "section_title": "STOP Sign (R1-1) and ALL-WAY Plaque (R1-3P)",
                "content_type": "Guidance",
                "ordinal": "07",
                "page_printed": "74",
                "text": (
                    "The TRAFFIC FROM LEFT (RIGHT) DOES NOT STOP (W4-4aP) plaque or "
                    "ONCOMING TRAFFIC DOES NOT STOP (W4-4bP) plaque should be used at "
                    "intersections where STOP signs control all but one approach to the "
                    "intersection, unless the only non-stopped approach is from a one-way street."
                ),
            },
            {
                "section_id": "2B.04",
                "section_title": "STOP Sign (R1-1) and ALL-WAY Plaque (R1-3P)",
                "content_type": "Option",
                "ordinal": "08",
                "page_printed": "74",
                "text": (
                    "The EXCEPT RIGHT TURN (R1-10P) plaque (see Figure 2B-1) may be mounted "
                    "below the STOP sign if an engineering study determines that a special "
                    "combination of geometry and traffic volumes is present that makes it "
                    "possible for right-turning traffic on the approach to be allowed to enter "
                    "the intersection without stopping."
                ),
            },
        ],
        "figures": [],
        "answer": (
            "Direct Answer: Several plaques may accompany STOP signs. The ALL-WAY (R1-3P) "
            "plaque is required at all-way stop intersections. Numbered-way plaques such as "
            "2-WAY, 3-WAY, or 4-WAY are prohibited. The TRAFFIC FROM LEFT/RIGHT DOES NOT "
            "STOP plaque is recommended where STOP signs control all but one approach. The "
            "EXCEPT RIGHT TURN plaque is permitted when supported by an engineering study.\n"
            "\n"
            "Standards (mandatory):\n"
            "- At intersections where all approaches are controlled by STOP signs, an ALL-WAY "
            "(R1-3P) supplemental plaque shall be mounted below each STOP sign, with a white "
            "legend and border on a red background.\n"
            "- Supplemental plaques with legends such as 2-WAY, 3-WAY, 4-WAY, or other numbers "
            "of ways shall not be used with STOP signs.\n"
            "\n"
            "Guidance (recommended):\n"
            "- The TRAFFIC FROM LEFT (RIGHT) DOES NOT STOP (W4-4aP) or ONCOMING TRAFFIC DOES "
            "NOT STOP (W4-4bP) plaque should be used at intersections where STOP signs control "
            "all but one approach, unless the only non-stopped approach is from a one-way street.\n"
            "\n"
            "Options (permitted):\n"
            "- The EXCEPT RIGHT TURN (R1-10P) plaque may be mounted below the STOP sign when "
            "an engineering study supports it based on geometry and traffic volumes.\n"
            "\n"
            "Citations:\n"
            "- Section 2B.04 Standard §04 (p.74)\n"
            "- Section 2B.04 Standard §05 (p.74)\n"
            "- Section 2B.04 Guidance §07 (p.74)\n"
            "- Section 2B.04 Option §08 (p.74)"
        ),
    },

    # -----------------------------------------------------------------------
    # Example 2 — Pure Standard answer with explicit exceptions list.
    # Shows the model how to quote mandatory provisions verbatim and how
    # to handle "shall not... except in the following cases" patterns.
    # -----------------------------------------------------------------------
    {
        "question": "Can STOP signs or YIELD signs be used with traffic signals?",
        "chunks": [
            {
                "section_id": "2B.06",
                "section_title": "General Considerations",
                "content_type": "Standard",
                "ordinal": "08",
                "page_printed": "75",
                "text": (
                    "Because the potential for conflicting commands could create driver "
                    "confusion, YIELD or STOP signs shall not be used in conjunction with "
                    "any traffic control signal operation, except in the following cases:\n"
                    "A. If the signal indication for an approach is a flashing red at all times;\n"
                    "B. If a minor street or driveway is located within or adjacent to the area "
                    "controlled by the traffic control signal, but does not require separate "
                    "traffic signal control because an extremely low potential for conflict exists; or\n"
                    "C. If a channelized turn lane is separated from the adjacent travel lanes "
                    "by an island and the channelized turn lane is not controlled by a traffic "
                    "control signal."
                ),
            },
        ],
        "figures": [],
        "answer": (
            "Direct Answer: As a general rule, STOP and YIELD signs may not be used with a "
            "traffic control signal because conflicting commands could confuse drivers. The "
            "MUTCD permits only three specific exceptions.\n"
            "\n"
            "Standards (mandatory):\n"
            "- YIELD or STOP signs shall not be used in conjunction with any traffic control "
            "signal operation, except in the following cases: (A) if the signal indication for "
            "an approach is a flashing red at all times; (B) if a minor street or driveway is "
            "located within or adjacent to the signal-controlled area but does not require "
            "separate signal control because an extremely low potential for conflict exists; or "
            "(C) if a channelized turn lane is separated from adjacent travel lanes by an island "
            "and is not itself signal-controlled.\n"
            "\n"
            "Citations:\n"
            "- Section 2B.06 Standard §08 (p.75)"
        ),
    },

    # -----------------------------------------------------------------------
    # Example 3 — Standard + Guidance, no Option. Shows the model what to do
    # when only some categories are present (omit empty sections).
    # -----------------------------------------------------------------------
    {
        "question": "How are non-statutory speed limits established?",
        "chunks": [
            {
                "section_id": "2B.21",
                "section_title": "Speed Limit Sign (R2-1)",
                "content_type": "Standard",
                "ordinal": "06",
                "page_printed": "84",
                "text": (
                    "Speed zones (other than statutory speed limits) shall only be established "
                    "on the basis of an engineering study that has been performed in accordance "
                    "with traffic engineering practices. The engineering study shall consider "
                    "the roadway context."
                ),
            },
            {
                "section_id": "2B.21",
                "section_title": "Speed Limit Sign (R2-1)",
                "content_type": "Guidance",
                "ordinal": "07",
                "page_printed": "84",
                "text": (
                    "Among the factors that should be considered when conducting an engineering "
                    "study for establishing or reevaluating speed limits within speed zones are "
                    "the following: roadway environment and functional classification; roadway "
                    "characteristics such as lane widths, grade, alignment, median type, and "
                    "sight distance; geographic context; reported crash experience for at least "
                    "a 12-month period; speed distribution of free-flowing vehicles including "
                    "the pace, median, and 85th-percentile speeds; and a review of past speed "
                    "studies to identify trends in operating speeds."
                ),
            },
        ],
        "figures": [],
        "answer": (
            "Direct Answer: Non-statutory speed limits must be set on the basis of an engineering "
            "study performed in accordance with traffic engineering practices. The study is "
            "required to consider roadway context, and should weigh a wide range of factors "
            "including roadway characteristics, crash history, and observed speed distribution.\n"
            "\n"
            "Standards (mandatory):\n"
            "- Speed zones (other than statutory speed limits) shall only be established on the "
            "basis of an engineering study performed in accordance with traffic engineering "
            "practices, and the study shall consider the roadway context.\n"
            "\n"
            "Guidance (recommended):\n"
            "- The engineering study should consider roadway environment and functional "
            "classification; roadway characteristics (lane widths, grade, alignment, median "
            "type, sight distance); geographic context; reported crash experience for at least "
            "12 months; speed distribution of free-flowing vehicles (pace, median, and "
            "85th-percentile speeds); and a review of past speed studies for trends.\n"
            "\n"
            "Citations:\n"
            "- Section 2B.21 Standard §06 (p.84)\n"
            "- Section 2B.21 Guidance §07 (p.84)"
        ),
    },
]


def _render_example_evidence_block(ex: Dict[str, Any], max_chars: int = 1400) -> str:
    """Render an example's chunks in the same `=== {category} provisions ===`
    format that vlm.py's _build_prompt_and_images uses for real evidence.

    Mirroring the real format exactly is what makes few-shot examples useful:
    the model sees the example evidence laid out identically to the real
    evidence it will receive at inference time.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {
        "Standard": [], "Guidance": [], "Option": [], "Support": [],
    }
    for c in ex["chunks"]:
        ct = c.get("content_type", "Support")
        groups.setdefault(ct, []).append(c)

    blocks: List[str] = []
    for ct in ("Standard", "Guidance", "Option", "Support"):
        cs = groups.get(ct, [])
        if not cs:
            continue
        blocks.append(f"=== {ct} provisions ===")
        for c in cs:
            blocks.append(
                f"[Section {c.get('section_id')} §{c.get('ordinal')} — "
                f"{c.get('section_title','')} (p.{c.get('page_printed','?')})]\n"
                f"{(c.get('text','') or '')[:max_chars]}"
            )
        blocks.append("")
    return "\n".join(blocks)


def _render_example_allowed_cites(ex: Dict[str, Any]) -> str:
    """Same citation-list format the real prompt uses."""
    lines: List[str] = []
    for c in ex["chunks"]:
        lines.append(
            f"  - Section {c.get('section_id')} {c.get('content_type')} "
            f"§{c.get('ordinal')} (p.{c.get('page_printed','?')})"
        )
    for f in ex.get("figures") or []:
        lines.append(
            f"  - {f.get('figure_id','?')} (p.{f.get('page_printed','?')})"
        )
    return "\n".join(lines) if lines else "  (none)"


def _render_example_visual_lines(ex: Dict[str, Any]) -> str:
    figs = ex.get("figures") or []
    if not figs:
        return "(none)"
    lines: List[str] = []
    for i, f in enumerate(figs, 1):
        lines.append(
            f"[Image {i}] {f.get('figure_id','?')} (p.{f.get('page_printed','?')}): "
            f"{(f.get('caption','') or '')[:160]}"
        )
    return "\n".join(lines)


def format_example(ex: Dict[str, Any], n: int, max_chars: int = 1400) -> str:
    """Render one example as a fully-delimited block ready to splice into a prompt.

    Layout:
        --- Example {n} ---
        Question: ...
        Visual evidence (N images):
        ...
        Text evidence:
        === Standard provisions ===
        [Section 2B.04 §04 — ... (p.74)]
        ...
        Allowed citations (use ONLY these strings verbatim):
          - ...
        Answer:
        Direct Answer: ...
        ...
        --- End example {n} ---
    """
    visual = _render_example_visual_lines(ex)
    evidence = _render_example_evidence_block(ex, max_chars=max_chars)
    cites = _render_example_allowed_cites(ex)
    n_figs = len(ex.get("figures") or [])
    return (
        f"--- Example {n} ---\n"
        f"Question: {ex['question']}\n\n"
        f"Visual evidence ({n_figs} images):\n{visual}\n\n"
        f"Text evidence:\n{evidence}\n"
        f"Allowed citations (use ONLY these strings verbatim):\n{cites}\n\n"
        f"Answer:\n{ex['answer']}\n"
        f"--- End example {n} ---\n"
    )
