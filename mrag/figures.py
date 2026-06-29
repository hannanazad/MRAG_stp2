"""Caption-anchored figure / table extraction.

For every 'Figure X-Y' or 'Table X-Y' caption found in the PDF, crop the
corresponding region above (figure) or below (table) and save as a PNG.
Output is one row per real figure/table, not per page.

The bounding box logic is intentionally simple — for layout-pathological
pages we accept some over-crop in exchange for zero ML dependencies and
fully deterministic, debuggable output.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterator, List, Optional

import fitz


CAPTION_RE = re.compile(
    r"^\s*(Figure|Table)\s+"
    r"([A-Z]?\d+[A-Z]?-\d+(?:\([A-Za-z0-9]+\))?[A-Za-z0-9]*)"
    r"\s*[.\u2014:-]?\s*(.{0,250})",
    re.IGNORECASE,
)
SKIP_LINE_RE = re.compile(r"page\s+\d+|chapter\s+\d", re.IGNORECASE)


@dataclass
class FigureRecord:
    figure_id:           str            # e.g. "Figure 2B-1" or "Table 2B-1"
    kind:                str            # "Figure" or "Table"
    canonical_id:        str            # e.g. "2B-1"
    page_pdf:            int
    page_printed:        str
    caption:             str
    title:               str
    image_path:          str
    bbox:                List[float]    # [x0,y0,x1,y1] in PDF points
    dpi:                 int
    sign_codes_depicted: List[str] = field(default_factory=list)
    referenced_in_chunks: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

def extract_figures(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = 220,
    min_h: float = 60.0,
    min_w: float = 80.0,
) -> List[FigureRecord]:
    """Walk the PDF, write one PNG per caption-anchored region."""
    pdf_path = Path(pdf_path); out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)

    out: List[FigureRecord] = []
    for p_idx in range(doc.page_count):
        page = doc.load_page(p_idx)
        captions = _find_captions(page)
        if not captions:
            continue
        captions.sort(key=lambda c: c["bbox"][1])
        for i, cap in enumerate(captions):
            if cap["kind"].lower() == "figure":
                rect = _figure_region(captions, i, page.rect)
            else:
                rect = _table_region(captions, i, page.rect)
            if rect.is_empty or rect.height < min_h or rect.width < min_w:
                continue
            pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
            fname = f"{cap['kind'].lower()}_{_safe(cap['id'])}_p{p_idx+1:04d}.png"
            out_path = out_dir / fname
            pix.save(str(out_path))
            out.append(FigureRecord(
                figure_id=f"{cap['kind']} {cap['id']}",
                kind=cap["kind"],
                canonical_id=cap["id"],
                page_pdf=p_idx + 1,
                page_printed=page.get_label() or str(p_idx + 1),
                caption=cap["text"],
                title=cap["title"],
                image_path=str(out_path),
                bbox=[rect.x0, rect.y0, rect.x1, rect.y1],
                dpi=dpi,
            ))
    return out


def render_pages(pdf_path: Path, out_dir: Path, dpi: int = 180) -> int:
    """Render any missing page PNGs at the given DPI. Returns count rendered."""
    pdf_path = Path(pdf_path); out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    rendered = 0
    for i in range(doc.page_count):
        out = out_dir / f"page_{i + 1:04d}.png"
        if out.exists():
            continue
        doc.load_page(i).get_pixmap(matrix=mat, alpha=False).save(str(out))
        rendered += 1
    return rendered


def write_jsonl(figs: List[FigureRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in figs:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[FigureRecord]:
    out: List[FigureRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            out.append(FigureRecord(**json.loads(line)))
    return out


# --------------------------------------------------------------------------- #
# Internals                                                                   #
# --------------------------------------------------------------------------- #

def _find_captions(page: fitz.Page):
    out = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = " ".join(span["text"] for span in line.get("spans", [])).strip()
            if not text or SKIP_LINE_RE.search(text):
                continue
            m = CAPTION_RE.match(text)
            if not m:
                continue
            out.append({
                "kind":  m.group(1).title(),
                "id":    m.group(2).upper(),
                "text":  text,
                "title": m.group(3).strip(" .\u2014-:"),
                "bbox":  tuple(line["bbox"]),
            })
    return out


def _figure_region(caps, i, page_rect):
    bb = caps[i]["bbox"]
    y_top = page_rect.y0 + 30
    for j in range(i - 1, -1, -1):
        if caps[j]["bbox"][3] <= bb[1]:
            y_top = max(y_top, caps[j]["bbox"][3] + 8); break
    return fitz.Rect(page_rect.x0 + 24, y_top, page_rect.x1 - 24, bb[1] - 4)


def _table_region(caps, i, page_rect):
    bb = caps[i]["bbox"]
    y_bot = page_rect.y1 - 30
    for j in range(i + 1, len(caps)):
        if caps[j]["bbox"][1] >= bb[3]:
            y_bot = min(y_bot, caps[j]["bbox"][1] - 8); break
    return fitz.Rect(page_rect.x0 + 24, bb[3] + 4, page_rect.x1 - 24, y_bot)


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
