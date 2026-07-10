"""Backend-agnostic geometry core for MUTCD figure/table extraction (v2).

WHY THIS EXISTS
---------------
The v1 extractor (mrag/figures.py, `_figure_region`) assumed captions sit
BELOW their figures and cropped the region ABOVE each caption. In the MUTCD
11th-Edition PDF, every "Figure X-Y." / "Table X-Y." caption sits ABOVE its
content. Consequences of the inverted assumption:

  * caption at top of page  -> region above is ~30 pt of header -> fails the
    min-height filter -> figure silently dropped        (the "missing 11+")
  * caption mid-page        -> region above is the PREVIOUS figure or body
    text -> wrong image stored under the id             ("not understood")
  * TOC "List of Figures" lines also matched the caption regex -> junk crops

v2 rule set (validated visually against the real PDF):

  R1  Caption anchors the TOP of its region; crop DOWN to the next caption
      on the page (minus a gap) or to the bottom content margin.
  R2  Reject list/TOC caption lines: dot-leader runs ("....") or a trailing
      page number after the leader.
  R3  Multi-sheet figures "(Sheet n of m)" produce one crop per sheet, all
      sharing a canonical figure node downstream.
  R4  A post-extraction census compares extracted ids against every
      "Figure/Table X-Y" mention in the document text and emits a coverage
      report. Ingestion FAILS LOUDLY if coverage is below threshold —
      never a silent gap again.

This module holds pure geometry + parsing (no PDF library imports) so the
PyMuPDF backend (production, Colab) and the poppler backend (validation /
fallback) share one implementation of the rules.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Caption grammar                                                             #
# --------------------------------------------------------------------------- #
# The Revision-1 pages (Parts 6–7 especially) set figure/table ids with a
# Unicode NON-BREAKING HYPHEN (U+2011): "Table 6B\u20112." An ASCII-only id
# pattern silently misses those captions AND the body references to them, so
# every id-bearing string is normalised before matching.
_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014"), "-")


def normalize_dashes(s: str) -> str:
    return s.translate(_DASHES)


# Some Revision-1 pages encode captions without space glyphs ("Figure4F-1.")
# or with the id split across glyph runs ("4F- 1"). PyMuPDF reconstructs
# these literally; poppler's geometric word rebuild hides them. Canonicalise
# spacing around figure/table ids so both backends parse identically.
_ID_SPACING_RE = re.compile(
    r"(?i)\b(figure|table)\s*([0-9]+[A-Z]{0,2})\s*-\s*(\d+)")


def canonicalize_ids(s: str) -> str:
    return _ID_SPACING_RE.sub(lambda m: f"{m.group(1)} {m.group(2)}-{m.group(3)}", s)
# id examples: 2B-14, 2D-26, 4T-1, 9E-12, 6H-46, 1A-1
FIG_ID = r"([0-9]+[A-Z]{0,2}-\d+)"
# The id-terminating PERIOD is REQUIRED: every true MUTCD caption reads
# "Figure 2B-14. Title" / "Table 6P-1 (Sheet 1 of 2). Title", while body
# sentences that happen to start a line ("Table 2C-1 may be used.",
# "Figure 9D-2 shows examples...") never put a period straight after the id.
CAPTION_RE = re.compile(
    rf"^\s*(Figure|Table)\s+{FIG_ID}"
    r"\s*(\(Sheet\s+(\d+)\s+of\s+(\d+)\))?"     # optional sheet marker
    r"\s*[.:]\s*(.*)$",
    re.IGNORECASE,
)
DOT_LEADER_RE = re.compile(r"\.{4,}")            # TOC dot leaders
TRAILING_PGNUM_RE = re.compile(r"\.{2,}\s*\d{1,4}\s*$")
MENTION_RE = re.compile(rf"\b(Figure|Table)\s+{FIG_ID}\b", re.IGNORECASE)

# Mid-line caption: MUTCD sets inset figures beside body text in two-column
# layout, so the caption appears mid-line in the extracted text layer:
#   "01  Vehicle Speed Feedback (W13-20) signs   Figure 2C-4. Vehicle Speed…"
# A REAL caption id always ends with a period ("Figure 2C-4."); body
# references never do ("see Figure 2C-4)"), which makes the trailing period
# a reliable discriminator.
SHEET_TAIL_RE = re.compile(r"\(Sheet\s+(\d+)\s+of\s+(\d+)\)\s*$", re.IGNORECASE)

MIDLINE_CAPTION_RE = re.compile(
    rf"(Figure|Table)\s+{FIG_ID}\.(\s*\(Sheet\s+(\d+)\s+of\s+(\d+)\))?\s*(.*)$",
    re.IGNORECASE,
)

# Page-geometry constants (PDF points). MUTCD is US-Letter 612x792 portrait;
# a handful of figure pages are landscape 792x612 — all ratios below are
# computed from the live page box, never hard-coded to portrait.
TOP_CONTENT_MARGIN = 46.0      # below running header
BOT_CONTENT_MARGIN = 46.0      # above running footer
SIDE_MARGIN = 22.0
CAPTION_GAP_BELOW = 4.0        # gap between caption baseline and crop top
NEXT_CAPTION_GAP = 6.0         # stop this crop above the next caption
MIN_REGION_H = 60.0
MIN_REGION_W = 80.0
MAX_CAPTION_LEN = 150          # caption lines are short; guards against body text


@dataclass
class CaptionHit:
    kind: str                  # "Figure" | "Table"
    canonical_id: str          # "2B-14"
    sheet: Optional[int]       # 1-based sheet number or None
    sheet_of: Optional[int]
    title: str
    y_top: float               # caption line top (PDF pts, origin top-left)
    y_bot: float               # caption line bottom
    raw_text: str
    x_left: float = 0.0        # x of caption start; >0 for inset (mid-line) captions
    inset: bool = False        # True when caption was found mid-line (side column)


@dataclass
class FigureRecord:
    """One row per CROP. Multi-sheet figures produce multiple rows sharing
    canonical_id; the KG collapses them into one node with several images."""
    figure_id: str             # "Figure 2B-14"  (canonical, no sheet suffix)
    kind: str                  # "Figure" | "Table"
    canonical_id: str          # "2B-14"
    sheet: Optional[int]       # None for single-sheet
    sheet_of: Optional[int]
    page_pdf: int              # 1-based
    page_printed: str
    caption: str
    title: str
    image_path: str
    bbox: List[float]          # [x0, y0, x1, y1] in PDF points, top-left origin
    dpi: int
    chapter: str = ""          # derived from id: "2B-14" -> "2B"
    extraction_method: str = "caption_below_v2"
    sign_codes_depicted: List[str] = field(default_factory=list)
    anchor_section: str = ""   # filled by KG builder from first citing chunk

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Rule R2 — TOC / list-page rejection                                         #
# --------------------------------------------------------------------------- #

def is_list_entry(line_text: str) -> bool:
    """True if a caption-looking line is actually a TOC / List-of-Figures row."""
    return bool(DOT_LEADER_RE.search(line_text) or TRAILING_PGNUM_RE.search(line_text))


def is_toc_page(all_line_texts: Sequence[str]) -> bool:
    """Page-level R2 guard. Wrapped TOC entries put the dot leader on the
    continuation line, so a lone line-level check lets the first line of a
    wrapped entry through. A content page never contains multiple dot-leader
    rows; a List-of-Figures page contains many. Two or more dot-leader lines
    on one page => treat the whole page as a list page and extract nothing."""
    n_leader = sum(1 for t in all_line_texts if DOT_LEADER_RE.search(t))
    return n_leader >= 2


def parse_caption_line(line_text: str, y_top: float, y_bot: float,
                       x_left: float = 0.0) -> Optional[CaptionHit]:
    """Apply grammar + R2 to one physical text line (line-start captions)."""
    if len(line_text) > MAX_CAPTION_LEN:
        return None
    line_text = canonicalize_ids(normalize_dashes(line_text))
    m = CAPTION_RE.match(line_text)
    if not m:
        return None
    if is_list_entry(line_text):
        return None
    sheet = int(m.group(4)) if m.group(4) else None
    sheet_of = int(m.group(5)) if m.group(5) else None
    title = (m.group(6) or "").strip(" .\u2014\u2013-:")
    # MUTCD usually puts the sheet marker at the END of the title:
    # "Figure 9E-12. Examples of Intersection Bicycle Boxes (Sheet 1 of 2)"
    if sheet is None:
        ms = SHEET_TAIL_RE.search(title)
        if ms:
            sheet, sheet_of = int(ms.group(1)), int(ms.group(2))
            title = title[: ms.start()].strip(" .\u2014\u2013-:")
    return CaptionHit(
        kind=m.group(1).title(),
        canonical_id=m.group(2).upper(),
        sheet=sheet,
        sheet_of=sheet_of,
        title=title,
        y_top=y_top,
        y_bot=y_bot,
        raw_text=line_text.strip(),
        x_left=x_left,
        inset=False,
    )


def parse_midline_caption(words: "Sequence[tuple]", y_top: float,
                          y_bot: float, page_w: float) -> Optional[CaptionHit]:
    """Detect an inset-figure caption embedded mid-line in two-column layout.

    `words` = [(text, xMin, xMax), ...] for one physical line. Scans for a
    'Figure'/'Table' token whose following token is an id ENDING WITH A
    PERIOD, starting in the right half of the page. Returns a CaptionHit
    whose x_left bounds the side-column crop.
    """
    words = [(normalize_dashes(w), x0, x1) for (w, x0, x1) in words]
    # split fused keyword+id ("Figure4F-1.") and merge split ids ("4F-" "1.")
    fixed = []
    for (w, x0, x1) in words:
        m = re.match(r"(?i)^(figure|table)([0-9].*)$", w)
        if m:
            fixed.append((m.group(1), x0, x0))
            fixed.append((m.group(2), x0, x1))
        else:
            fixed.append((w, x0, x1))
    merged: list = []
    for t in fixed:
        if (merged
                and re.match(r"^[0-9]+[A-Z]{0,2}-$", merged[-1][0], re.IGNORECASE)
                and re.match(r"^\d+[.,]?$", t[0])):
            pw, px0, _ = merged[-1]
            merged[-1] = (pw + t[0], px0, t[2])
        else:
            merged.append(t)
    words = merged
    for i, (w, x0, _x1) in enumerate(words[:-1]):
        if w.lower() not in ("figure", "table"):
            continue
        nxt = words[i + 1][0]
        m = re.match(rf"^{FIG_ID}\.$", nxt)
        if not m:
            continue
        if x0 < page_w * 0.45:          # left/body column => not an inset caption
            continue
        tail = " ".join(t for t, *_ in words[i:])
        cm = MIDLINE_CAPTION_RE.search(tail)
        if not cm:
            continue
        sheet = int(cm.group(4)) if cm.group(4) else None
        sheet_of = int(cm.group(5)) if cm.group(5) else None
        mtitle = (cm.group(6) or "").strip(" .\u2014\u2013-:")
        if sheet is None:
            ms = SHEET_TAIL_RE.search(mtitle)
            if ms:
                sheet, sheet_of = int(ms.group(1)), int(ms.group(2))
                mtitle = mtitle[: ms.start()].strip(" .\u2014\u2013-:")
        return CaptionHit(
            kind=cm.group(1).title(),
            canonical_id=cm.group(2).upper(),
            sheet=sheet,
            sheet_of=sheet_of,
            title=mtitle,
            y_top=y_top,
            y_bot=y_bot,
            raw_text=tail.strip(),
            x_left=x0,
            inset=True,
        )
    return None


# --------------------------------------------------------------------------- #
# Rule R1 — region geometry (caption ABOVE content, crop DOWNWARD)            #
# --------------------------------------------------------------------------- #

def regions_for_page(
    captions: Sequence[CaptionHit],
    page_w: float,
    page_h: float,
    min_h: float = MIN_REGION_H,
) -> List[Tuple[CaptionHit, Tuple[float, float, float, float]]]:
    """For every caption on a page, compute its crop rect (x0,y0,x1,y1).

    Captions are sorted by vertical position; each region runs from just
    below its caption to just above the next caption (or the bottom content
    margin). Regions failing MIN_REGION_* are dropped — with the corrected
    orientation this now only rejects genuine degenerate cases, and every
    drop is visible in the coverage report rather than silent.
    """
    # Dedup: the same caption can be seen twice (line-start pass + midline
    # pass, or split line elements). Keep one hit per (kind,id,sheet),
    # preferring the line-start (non-inset) and topmost occurrence.
    best = {}
    for c in sorted(captions, key=lambda c: (c.inset, c.y_top)):
        best.setdefault((c.kind, c.canonical_id, c.sheet), c)
    caps = sorted(best.values(), key=lambda c: c.y_top)
    out = []
    for i, cap in enumerate(caps):
        y0 = cap.y_bot + CAPTION_GAP_BELOW
        y1 = page_h - BOT_CONTENT_MARGIN
        # bound only by the next caption STRICTLY BELOW this one — side-by-side
        # captions (e.g. Tables 4C-6 and 4C-7 on one row) must not zero each
        # other's regions out.
        for nxt in caps[i + 1:]:
            if nxt.y_top > cap.y_bot + 1.0:
                y1 = min(y1, nxt.y_top - NEXT_CAPTION_GAP)
                break
        if cap.inset:
            # Side-column crop: from just left of the caption text to the
            # right margin. Wrapped caption continuation lines ("Sign and
            # Plaque") land at the crop's top edge — harmless for a VLM and
            # far better than missing or mis-cropping the figure.
            x0 = max(SIDE_MARGIN, cap.x_left - 10.0)
            x1 = page_w - SIDE_MARGIN
        else:
            x0, x1 = SIDE_MARGIN, page_w - SIDE_MARGIN
        # Side-by-side captions (same row, e.g. Tables 4C-6 / 4C-7) partition
        # the width so each crop shows its own content.
        same_row = [c for c in caps
                    if c is not cap and abs(c.y_top - cap.y_top) < 6.0]
        if same_row:
            lefts  = [c for c in same_row if c.x_left < cap.x_left]
            rights = [c for c in same_row if c.x_left > cap.x_left]
            if rights:
                x1 = min(x1, min(c.x_left for c in rights) - 8.0)
            if lefts:
                x0 = max(x0, cap.x_left - 10.0)
        if (y1 - y0) < min_h or (x1 - x0) < MIN_REGION_W:
            continue
        out.append((cap, (x0, y0, x1, y1)))
    return out


# --------------------------------------------------------------------------- #
# Rule R4 — census / coverage                                                 #
# --------------------------------------------------------------------------- #

def mention_census(full_text: str) -> Dict[str, set]:
    """Every 'Figure X-Y' / 'Table X-Y' string in the document text.
    Ids referencing EXTERNAL documents (e.g. '2018 AASHTO Policy, Table 3-1')
    are excluded: every mention line is checked for an external-source marker."""
    full_text = canonicalize_ids(normalize_dashes(full_text))
    figs, tabs = set(), set()
    external = {"Figure": set(), "Table": set()}
    lines = full_text.split("\n")
    per_id_lines: Dict[tuple, list] = {}
    for ln in lines:
        for m in MENTION_RE.finditer(ln):
            key = (m.group(1).title(), m.group(2).upper())
            per_id_lines.setdefault(key, []).append(ln)
    for (kind, cid), lns in per_id_lines.items():
        if all(re.search(r"AASHTO|CFR|ISO\s|ANSI|AREMA", l) for l in lns):
            external[kind].add(cid)
            continue
        (figs if kind == "Figure" else tabs).add(cid)
    out = {"Figure": figs, "Table": tabs}
    out["_external"] = external
    return out


def coverage_report(records: Sequence[FigureRecord], full_text: str) -> dict:
    census = mention_census(full_text)
    external = census.pop("_external", {"Figure": set(), "Table": set()})
    got = {"Figure": {r.canonical_id for r in records if r.kind == "Figure"},
           "Table": {r.canonical_id for r in records if r.kind == "Table"}}
    rep = {}
    for kind in ("Figure", "Table"):
        mentioned, extracted = census[kind], got[kind]
        rep[kind] = {
            "mentioned": len(mentioned),
            "extracted": len(extracted),
            "missing": sorted(mentioned - extracted),
            "extra": sorted(extracted - mentioned),
            "external_refs_excluded": sorted(external.get(kind, set())),
            "coverage": (len(mentioned & extracted) / len(mentioned)) if mentioned else 1.0,
        }
    return rep


def chapter_of(canonical_id: str) -> str:
    """'2B-14' -> '2B' ; '9E-12' -> '9E' ; '1A-1' -> '1A'."""
    return canonical_id.split("-", 1)[0]
