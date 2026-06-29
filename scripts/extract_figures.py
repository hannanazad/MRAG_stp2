#!/usr/bin/env python3
"""Caption-anchored figure / table extraction for the MUTCD PDF.

For every "Figure X-Y" or "Table X-Y" caption the script finds, it crops
the rectangular region of the page that contains the corresponding
figure or table, saves it as a PNG, and records metadata in
`figures.json` alongside the PNGs.

The heuristic:
  * Figures: caption sits *below* the figure. We crop from below the
    nearest caption above (or top-of-content) down to the top of this
    caption.
  * Tables: title sits *above* the table. We crop from below the title
    down to the top of the next caption (or bottom-of-content).

Pure PyMuPDF, no GPU, no extra dependencies beyond what the RAG
notebook already needs. Run on a login node (a few minutes for the
full 1000-page MUTCD).

Usage:
    python scripts/extract_figures.py \
        --pdf  $SCRATCH/MRAG/mutcd11theditionr1hl.pdf \
        --out  $SCRATCH/MRAG/figures \
        --dpi  220
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import fitz  # PyMuPDF
from tqdm import tqdm


# Matches "Figure 8C-1", "Figure 8C-1.", "Figure 8C-1 Some title", "Table 6A-1 ..."
# Also accepts compound ids like "Figure 6F-101(CA)" or "Table 2B-1(a)".
CAPTION_RE = re.compile(
    r"^\s*(Figure|Table)\s+"
    r"([A-Z]?\d+[A-Z]?-\d+(?:\([A-Za-z0-9]+\))?[A-Za-z0-9]*)"
    r"\s*[.\u2014:-]?\s*(.{0,250})",
    re.IGNORECASE,
)

# Headers/footers we never want to mistake for captions.
SKIP_LINE_RE = re.compile(r"page\s+\d+|chapter\s+\d", re.IGNORECASE)


def find_captions(page: fitz.Page):
    """Return a list of {kind, id, text, title, bbox} on this page."""
    captions = []
    blocks = page.get_text("dict").get("blocks", [])
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = " ".join(span["text"] for span in line.get("spans", [])).strip()
            if not text or SKIP_LINE_RE.search(text):
                continue
            m = CAPTION_RE.match(text)
            if not m:
                continue
            kind = m.group(1).title()
            fig_id = m.group(2).upper()
            title = m.group(3).strip(" .\u2014-:")
            captions.append({
                "kind":  kind,
                "id":    fig_id,
                "text":  text,
                "title": title,
                "bbox":  tuple(line["bbox"]),  # (x0, y0, x1, y1) in PDF pts
            })
    return captions


def _safe_id(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def _figure_region(captions, i, page_rect):
    """Region ABOVE captions[i]: from below the previous caption (or top
    of content) down to just above this caption."""
    bbox = captions[i]["bbox"]
    y_top = page_rect.y0 + 30
    for j in range(i - 1, -1, -1):
        prev_bottom = captions[j]["bbox"][3]
        if prev_bottom <= bbox[1]:
            y_top = max(y_top, prev_bottom + 8)
            break
    y_bot = bbox[1] - 4
    x_left  = page_rect.x0 + 24
    x_right = page_rect.x1 - 24
    return fitz.Rect(x_left, y_top, x_right, y_bot)


def _table_region(captions, i, page_rect):
    """Region BELOW captions[i]: from below this caption down to top of
    the next caption (or bottom of content)."""
    bbox = captions[i]["bbox"]
    y_top = bbox[3] + 4
    y_bot = page_rect.y1 - 30
    for j in range(i + 1, len(captions)):
        nxt_top = captions[j]["bbox"][1]
        if nxt_top >= bbox[3]:
            y_bot = min(y_bot, nxt_top - 8)
            break
    x_left  = page_rect.x0 + 24
    x_right = page_rect.x1 - 24
    return fitz.Rect(x_left, y_top, x_right, y_bot)


def extract_figures_from_pdf(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = 220,
    min_height: float = 60.0,
    min_width: float  = 80.0,
):
    """Walk the PDF, write one PNG per caption-anchored region, return a
    list of metadata dicts.
    """
    pdf_path = Path(pdf_path)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc  = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    mat  = fitz.Matrix(zoom, zoom)

    figures = []
    for page_num in tqdm(range(doc.page_count), desc="extract figures"):
        page = doc.load_page(page_num)
        captions = find_captions(page)
        if not captions:
            continue

        # Sort top-to-bottom for stable region boundaries.
        captions.sort(key=lambda c: c["bbox"][1])

        for i, cap in enumerate(captions):
            if cap["kind"].lower() == "figure":
                rect = _figure_region(captions, i, page.rect)
            else:
                rect = _table_region(captions, i, page.rect)

            if rect.is_empty:
                continue
            if rect.height < min_height or rect.width < min_width:
                continue

            pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
            fname = f"{cap['kind'].lower()}_{_safe_id(cap['id'])}_p{page_num+1:04d}.png"
            out_path = out_dir / fname
            pix.save(str(out_path))

            figures.append({
                "figure_id":   f"{cap['kind']} {cap['id']}",
                "kind":        cap["kind"],
                "id":          cap["id"],
                "page_num":    page_num + 1,
                "caption":     cap["text"],
                "title":       cap["title"],
                "image_path":  str(out_path),
                "bbox":        [rect.x0, rect.y0, rect.x1, rect.y1],
                "dpi":         dpi,
            })

    return figures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf",  required=True, type=Path, help="path to the MUTCD PDF")
    ap.add_argument("--out",  required=True, type=Path, help="output folder for crops")
    ap.add_argument("--dpi",  type=int,   default=220)
    ap.add_argument("--json", type=Path,  default=None,
                    help="path to write the figures index json "
                         "(default: <out>/../figures.json)")
    ap.add_argument("--min-height", type=float, default=60.0)
    ap.add_argument("--min-width",  type=float, default=80.0)
    args = ap.parse_args()

    figs = extract_figures_from_pdf(
        args.pdf, args.out,
        dpi=args.dpi,
        min_height=args.min_height,
        min_width=args.min_width,
    )

    json_path = args.json or (args.out.parent / "figures.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(figs, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(figs)} figure crops to {args.out}")
    print(f"Wrote index to {json_path}")
    # Quick distribution print so the user knows what they got.
    by_kind = {}
    for f in figs:
        by_kind[f["kind"]] = by_kind.get(f["kind"], 0) + 1
    for k, n in sorted(by_kind.items()):
        print(f"  {k}: {n}")


if __name__ == "__main__":
    main()
