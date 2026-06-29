"""Sign-code miner: build a small dictionary of MUTCD sign codes.

Strategy:
1. Mine sign codes from Section titles in the PDF outline (they nearly all
   appear there — e.g., "Section 2B.04 STOP Sign (R1-1) and ALL-WAY Plaque (R1-3P)").
2. Mine additional codes from chunk text after parsing.
3. Categorise by prefix: R* = Regulatory, W* = Warning, D* = Guide, etc.
4. Resolve canonical name from the Section title context (text in parens).

The output dictionary is consumed by:
- KG builder (SignCode nodes, defines / depicts / mentions edges).
- Retrieval (exact sign-code matches boost rank).
- Display (`R1-3P -> "ALL-WAY plaque"` in citations).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Sign-code prefix → category. Conservative; we add more as we encounter them.
# Source: MUTCD 11e Part 2 (Signs) standard sign-code conventions.
CATEGORY_BY_PREFIX = {
    "R":   "Regulatory",
    "W":   "Warning",
    "D":   "Guide",
    "I":   "Information",
    "M":   "Marker",
    "OM":  "ObjectMarker",
    "EM":  "Emergency",
    "RM":  "RoadMarker",
}

# Sign codes are typically letters (1-3) + digit(s) + optional '-' + digit(s) +
# optional letter(s) + optional 'P' suffix for plaques. We require the FIRST
# character to be in the known-category set to avoid matching random uppercase
# tokens (CFR, FHWA, USC, ...).
KNOWN_FIRST_CHARS = "RWDIMOE"
SIGNCODE_RE = re.compile(
    r"\b(" + r"|".join(CATEGORY_BY_PREFIX.keys()) + r")(\d{1,3}(?:-\d{1,3})?[a-z]?P?)\b"
)
# Inline parenthesised form like "STOP (R1-1) sign":
PAREN_SIGN_RE = re.compile(r"\(([A-Z]{1,3}\d{1,3}(?:-\d{1,3})?[a-z]?P?)\)")


@dataclass
class SignCodeEntry:
    code: str               # canonical, uppercased
    category: str           # Regulatory / Warning / ...
    canonical_name: str     # e.g. "STOP sign", "ALL-WAY plaque"
    first_seen_section: Optional[str]
    referenced_in_sections: List[str]   # populated later by KG builder


def categorize(code: str) -> str:
    """Map a sign code to its category by prefix."""
    code = code.upper()
    # Try 2-char prefix first (OM, EM, RM), then 1-char.
    for prefix in sorted(CATEGORY_BY_PREFIX.keys(), key=len, reverse=True):
        if code.startswith(prefix):
            return CATEGORY_BY_PREFIX[prefix]
    return "Unknown"


def get_sign_code_re() -> re.Pattern:
    """Returns the canonical sign-code regex for parsers to use.

    Matches e.g. R1-1, W4-4aP, OM1-1, R1-10P, M1-5a, EM-1.
    """
    prefixes = "|".join(sorted(CATEGORY_BY_PREFIX.keys(), key=len, reverse=True))
    return re.compile(
        r"\b((?:" + prefixes + r")-?\d{1,3}(?:-\d{1,3})?[a-z]?P?)\b"
    )


def mine_from_outline(toc) -> Dict[str, SignCodeEntry]:
    """Extract sign codes appearing in 'Section X.Y ...' titles in the outline.

    Examples:
      'Section 2B.04 STOP Sign (R1-1) and ALL-WAY Plaque (R1-3P)'
        -> R1-1 = "STOP sign", R1-3P = "ALL-WAY plaque"
    """
    entries: Dict[str, SignCodeEntry] = {}
    section_re = re.compile(r"^Section\s+([0-9A-Z]+\.[0-9]+)\s+(.+)$")
    for lvl, title, _pg in toc:
        if lvl != 3: continue
        m = section_re.match(title.strip())
        if not m: continue
        sec_id, sec_title = m.group(1), m.group(2)
        for fragment, code in _name_code_pairs(sec_title):
            code_u = code.upper()
            if code_u in entries: continue
            entries[code_u] = SignCodeEntry(
                code=code_u,
                category=categorize(code_u),
                canonical_name=fragment.strip(),
                first_seen_section=sec_id,
                referenced_in_sections=[],
            )
    return entries


def _name_code_pairs(text: str):
    """Yield (name_fragment, code) pairs found in a section title.

    Heuristic: a sign code in parens is preceded by 1-4 words that name it.
    E.g. 'STOP Sign (R1-1) and ALL-WAY Plaque (R1-3P)' yields
       ('STOP Sign', 'R1-1'), ('ALL-WAY Plaque', 'R1-3P').
    """
    for m in PAREN_SIGN_RE.finditer(text):
        code = m.group(1)
        if not categorize(code) or categorize(code) == "Unknown":
            continue
        before = text[:m.start()]
        # Capture up to 5 words preceding the open paren — but stop early at any
        # token that itself looks like a sign code (so "STOP Sign (R1-1) and
        # ALL-WAY Plaque (R1-3P)" yields "ALL-WAY Plaque" for R1-3P, not
        # "Sign (R1-1) and ALL-WAY Plaque").
        tokens = re.findall(r"\b[A-Za-z][\w'\-]*\b", before)
        kept: list = []
        for tok in reversed(tokens):
            if re.match(r"^[RWDIMOE][A-Z]?\d", tok):
                break
            kept.append(tok)
            if len(kept) >= 5:
                break
        kept.reverse()
        name = " ".join(kept).strip()
        if name:
            yield name, code


def mine_from_chunks(chunks: Iterable, existing: Dict[str, SignCodeEntry]) -> Dict[str, SignCodeEntry]:
    """Augment `existing` with codes appearing in chunk text and parenthesised
    name patterns in chunk text. Returns the merged dict.
    """
    pat = get_sign_code_re()
    seen_by: Dict[str, List[str]] = defaultdict(list)
    for c in chunks:
        text = c.text if hasattr(c, "text") else c["text"]
        sec_id = c.section_id if hasattr(c, "section_id") else c["section_id"]
        # All codes in the chunk
        for m in pat.finditer(text):
            seen_by[m.group(1).upper()].append(sec_id)
        # Look for "<NAME> (CODE)" patterns to harvest canonical names.
        for name, code in _name_code_pairs(text):
            code_u = code.upper()
            if code_u not in existing:
                existing[code_u] = SignCodeEntry(
                    code=code_u,
                    category=categorize(code_u),
                    canonical_name=name,
                    first_seen_section=sec_id,
                    referenced_in_sections=[],
                )

    # Backfill referenced_in_sections.
    for code_u, secs in seen_by.items():
        if code_u not in existing:
            existing[code_u] = SignCodeEntry(
                code=code_u,
                category=categorize(code_u),
                canonical_name="",
                first_seen_section=secs[0] if secs else None,
                referenced_in_sections=[],
            )
        existing[code_u].referenced_in_sections = sorted(set(secs))
    return existing


def write_json(entries: Dict[str, SignCodeEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serial = {k: asdict(v) for k, v in entries.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serial, f, indent=2, ensure_ascii=False, sort_keys=True)


def read_json(path: Path) -> Dict[str, SignCodeEntry]:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {k: SignCodeEntry(**v) for k, v in d.items()}
