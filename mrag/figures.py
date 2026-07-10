"""MUTCD figure / table extraction — v2 (caption-below-region, dual backend).

Replaces the v1 extractor whose inverted region logic (crop ABOVE caption)
dropped or mis-cropped a large share of MUTCD figures. See figure_core.py
for the rule set and the post-mortem.

Backends
--------
  * "fitz"    — PyMuPDF. Fast; use in Colab / HPRC.  (pip install pymupdf)
  * "poppler" — pdftotext -bbox-layout + pdftoppm + Pillow. No compiled PDF
                deps beyond poppler-utils; used for validation and as a
                fallback when PyMuPDF is unavailable.

Both backends feed the SAME geometry in figure_core.py, so validation done
with one holds for the other.

Public API (superset of v1 — ingest scripts keep working):
    extract_figures(pdf_path, out_dir, dpi=220, backend="auto") -> List[FigureRecord]
    render_pages(pdf_path, out_dir, dpi=180) -> int
    write_jsonl(records, path) / read_jsonl(path)
    validate_coverage(records, full_text, fail_below=0.98) -> dict
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

from .figure_core import (
    CaptionHit, FigureRecord, parse_caption_line, parse_midline_caption,
    regions_for_page, coverage_report, chapter_of, is_toc_page, MENTION_RE,
)

log = logging.getLogger("mrag.figures")

try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except Exception:                                     # pragma: no cover
    fitz = None
    _HAS_FITZ = False


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

def extract_figures(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = 220,
    backend: str = "auto",
    page_whitelist: Optional[List[int]] = None,
) -> List[FigureRecord]:
    """Extract one PNG crop per caption-anchored region (R1–R3).

    `page_whitelist` (1-based) restricts work to given pages — used by the
    validation harness and by targeted re-extraction of individual figures.
    """
    pdf_path, out_dir = Path(pdf_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if backend == "auto":
        backend = "fitz" if _HAS_FITZ else "poppler"
    log.info("figure extraction v2 — backend=%s dpi=%d", backend, dpi)
    if backend == "fitz":
        recs = _extract_fitz(pdf_path, out_dir, dpi, page_whitelist)
    elif backend == "poppler":
        recs = _extract_poppler(pdf_path, out_dir, dpi, page_whitelist)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    log.info("extracted %d crops covering %d canonical ids",
             len(recs), len({(r.kind, r.canonical_id) for r in recs}))
    return recs


def rescue_missing(
    records: List[FigureRecord],
    full_text: str,
    pdf_path: Path,
    out_dir: Path,
    dpi: int = 220,
    backend: str = "auto",
    relaxed_min_h: float = 24.0,
) -> List[FigureRecord]:
    """Second-chance pass for ids with ZERO crops after the main extraction.

    Some MUTCD tables are physically tiny (e.g. Table 2A-3, ~40 pt tall inset)
    and fall under MIN_REGION_H, which exists to reject degenerate junk. Rather
    than weaken the guard globally, re-run ONLY the caption pages of missing
    ids with a relaxed minimum and keep just those ids' crops. Returns the
    records that were added (already appended to `records`)."""
    from .figure_core import normalize_dashes
    rep = coverage_report(records, full_text)
    missing = {("Figure", cid) for cid in rep["Figure"]["missing"]}
    missing |= {("Table", cid) for cid in rep["Table"]["missing"]}
    if not missing:
        return []
    norm = normalize_dashes(full_text)
    pages_todo = set()
    for i, ptxt in enumerate(norm.split("\f"), 1):
        for kind, cid in missing:
            if re.search(rf"{kind}\s+{re.escape(cid)}\.", ptxt, re.IGNORECASE):
                pages_todo.add(i)
    if not pages_todo:
        log.warning("rescue: no caption pages found for %s", sorted(missing))
        return []
    if backend == "auto":
        backend = "fitz" if _HAS_FITZ else "poppler"
    fn = _extract_fitz if backend == "fitz" else _extract_poppler
    cand = fn(pdf_path, out_dir, dpi, sorted(pages_todo), min_h=relaxed_min_h)
    added = [r for r in cand if (r.kind, r.canonical_id) in missing]
    for r in added:
        r.extraction_method = "caption_below_v2_rescued"
    records.extend(added)
    log.info("rescue: recovered %d crops for %s",
             len(added), sorted({(r.kind, r.canonical_id) for r in added}))
    return added


def validate_coverage(records: List[FigureRecord], full_text: str,
                      fail_below: float = 0.98) -> dict:
    """R4 — compare extracted ids to every textual mention. Raises if
    figure coverage < fail_below so ingestion can never silently ship a gap."""
    rep = coverage_report(records, full_text)
    for kind in ("Figure", "Table"):
        r = rep[kind]
        log.info("%s coverage: %d/%d (%.1f%%)  missing=%s",
                 kind, r["extracted"], r["mentioned"],
                 100 * r["coverage"], r["missing"][:10])
    if rep["Figure"]["coverage"] < fail_below:
        raise RuntimeError(
            f"Figure coverage {rep['Figure']['coverage']:.2%} below "
            f"{fail_below:.0%}. Missing: {rep['Figure']['missing']}")
    return rep


def render_pages(pdf_path: Path, out_dir: Path, dpi: int = 180) -> int:
    """Render any missing page PNGs (used by ColPali page retrieval)."""
    pdf_path, out_dir = Path(pdf_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if _HAS_FITZ:
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
    # poppler fallback
    n_pages = _poppler_page_count(pdf_path)
    rendered = 0
    for i in range(1, n_pages + 1):
        out = out_dir / f"page_{i:04d}.png"
        if out.exists():
            continue
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), "-f", str(i), "-l", str(i),
             str(pdf_path), str(out_dir / f"page_{i:04d}_tmp")],
            check=True, capture_output=True)
        tmp = sorted(out_dir.glob(f"page_{i:04d}_tmp*"))
        if tmp:
            tmp[0].rename(out)
        rendered += 1
    return rendered


def write_jsonl(records: List[FigureRecord], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[FigureRecord]:
    """Schema-tolerant reader: v1 rows (no sheet/chapter/... fields) load
    with defaults so version checks and migrations never crash on old data."""
    out: List[FigureRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            d.pop("referenced_in_chunks", None)   # v1 field, dropped in v2
            d.setdefault("sheet", None)
            d.setdefault("sheet_of", None)
            d.setdefault("chapter", chapter_of(d.get("canonical_id", "") or "?"))
            d.setdefault("extraction_method", "caption_above_v1")
            d.setdefault("sign_codes_depicted", [])
            d.setdefault("anchor_section", "")
            out.append(FigureRecord(**d))
    return out


# --------------------------------------------------------------------------- #
# fitz backend (production)                                                   #
# --------------------------------------------------------------------------- #

def _extract_fitz(pdf_path, out_dir, dpi, page_whitelist, min_h=None):
    """PyMuPDF backend rebuilt on page.get_text("words") — TRUE geometric
    word segmentation, structurally identical to the poppler bbox path that
    was validated at 100% coverage. The previous span-join reconstruction
    inherited PDF encoding artifacts on some Revision-1 pages (missing space
    glyphs -> "Figure4F-1.", ids split across glyph runs -> "4F- 1"), which
    silently dropped Figures 4F-1/2/4/5/6 on Colab."""
    doc = fitz.open(str(pdf_path))
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pages = page_whitelist or range(1, doc.page_count + 1)
    out: List[FigureRecord] = []
    for pno in pages:
        page = doc.load_page(pno - 1)
        # words: (x0, y0, x1, y1, text, block_no, line_no, word_no)
        by_line: Dict[tuple, list] = {}
        for x0, y0, x1, y1, w, bno, lno, _wno in page.get_text("words"):
            by_line.setdefault((bno, lno), []).append((w, x0, x1, y0, y1))
        caps: List[CaptionHit] = []
        line_texts: List[str] = []
        for key in sorted(by_line):
            ws = sorted(by_line[key], key=lambda t: t[1])
            text = " ".join(t[0] for t in ws)
            line_texts.append(text)
            ly0 = min(t[3] for t in ws)
            ly1 = max(t[4] for t in ws)
            hit = parse_caption_line(text, ly0, ly1, x_left=ws[0][1])
            if hit is None:
                hit = parse_midline_caption(
                    [(t[0], t[1], t[2]) for t in ws],
                    ly0, ly1, page.rect.width)
            if hit:
                caps.append(hit)
        if not caps or is_toc_page(line_texts):
            continue
        pw, ph = page.rect.width, page.rect.height
        kw = {"min_h": min_h} if min_h is not None else {}
        for cap, (x0, y0, x1, y1) in regions_for_page(caps, pw, ph, **kw):
            rect = fitz.Rect(x0, y0, x1, y1)
            pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
            fname = _crop_name(cap, pno)
            pix.save(str(out_dir / fname))
            out.append(_record(cap, pno,
                               page.get_label() or str(pno),
                               str(out_dir / fname),
                               [x0, y0, x1, y1], dpi))
    return out


# --------------------------------------------------------------------------- #
# poppler backend (validation / fallback)                                     #
# --------------------------------------------------------------------------- #

def _extract_poppler(pdf_path, out_dir, dpi, page_whitelist, min_h=None):
    from PIL import Image  # lazy import
    if page_whitelist is None:
        page_whitelist = _pages_with_caption_candidates(pdf_path)
    out: List[FigureRecord] = []
    labels = _poppler_page_labels(pdf_path)
    for pno in page_whitelist:
        caps, (pw, ph) = _poppler_captions_for_page(pdf_path, pno)
        if not caps:
            continue
        kw = {"min_h": min_h} if min_h is not None else {}
        regions = regions_for_page(caps, pw, ph, **kw)
        if not regions:
            continue
        page_png = _poppler_render_page(pdf_path, pno, dpi)
        img = Image.open(page_png)
        scale = dpi / 72.0
        for cap, (x0, y0, x1, y1) in regions:
            crop = img.crop((int(x0 * scale), int(y0 * scale),
                             int(x1 * scale), int(y1 * scale)))
            fname = _crop_name(cap, pno)
            crop.save(out_dir / fname)
            out.append(_record(cap, pno, labels.get(pno, str(pno)),
                               str(out_dir / fname), [x0, y0, x1, y1], dpi))
        page_png.unlink(missing_ok=True)
    return out


_CAND_RE = re.compile(r"\b(Figure|Table)\s+[0-9]+[A-Z]{0,2}-\d+\.", re.IGNORECASE)


def _pages_with_caption_candidates(pdf_path) -> List[int]:
    """Cheap full-text pass: a page is a candidate if a caption-shaped token
    ('Figure X-Y.' — id ending in a PERIOD) appears ANYWHERE in its text.
    The -layout text merges two-column rows into one line, so inset captions
    never sit at line start there; an anywhere-search with the trailing
    period keeps body references ('see Figure X-Y)') out while catching
    every true caption page."""
    from .figure_core import normalize_dashes
    txt = subprocess.run(["pdftotext", "-layout", str(pdf_path), "-"],
                         capture_output=True, text=True).stdout
    out = []
    for i, ptxt in enumerate(txt.split("\f"), 1):
        if _CAND_RE.search(normalize_dashes(ptxt)):
            out.append(i)
    return out


def _poppler_captions_for_page(pdf_path, pno):
    xml = subprocess.run(
        ["pdftotext", "-bbox-layout", "-f", str(pno), "-l", str(pno),
         str(pdf_path), "-"],
        capture_output=True, text=True).stdout
    # strip default namespace for painless querying
    xml = re.sub(r'xmlns="[^"]+"', "", xml, count=1)
    root = ET.fromstring(xml)
    page = root.find(".//page")
    pw, ph = float(page.get("width")), float(page.get("height"))
    caps: List[CaptionHit] = []
    line_texts: List[str] = []
    for line in page.iter("line"):
        wlist = [(w.text or "", float(w.get("xMin")), float(w.get("xMax")))
                 for w in line.iter("word")]
        text = " ".join(w for w, *_ in wlist).strip()
        line_texts.append(text)
        y0, y1 = float(line.get("yMin")), float(line.get("yMax"))
        x_left = wlist[0][1] if wlist else 0.0
        hit = parse_caption_line(text, y0, y1, x_left=x_left)
        if hit is None:
            hit = parse_midline_caption(wlist, y0, y1, pw)
        if hit:
            caps.append(hit)
    if is_toc_page(line_texts):
        caps = []
    return caps, (pw, ph)


def _poppler_render_page(pdf_path, pno, dpi) -> Path:
    prefix = Path(f"/tmp/_mrag_pg_{pno:04d}")
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-f", str(pno),
                    "-l", str(pno), str(pdf_path), str(prefix)],
                   check=True, capture_output=True)
    hits = sorted(Path("/tmp").glob(f"_mrag_pg_{pno:04d}*.png"))
    return hits[0]


def _poppler_page_count(pdf_path) -> int:
    info = subprocess.run(["pdfinfo", str(pdf_path)],
                          capture_output=True, text=True).stdout
    m = re.search(r"Pages:\s+(\d+)", info)
    return int(m.group(1))


def _poppler_page_labels(pdf_path) -> Dict[int, str]:
    """Printed page labels; poppler has no direct dump, approximate from the
    'Page N' footer in the text layer (MUTCD prints it top-left)."""
    txt = subprocess.run(["pdftotext", "-layout", str(pdf_path), "-"],
                         capture_output=True, text=True).stdout
    labels: Dict[int, str] = {}
    for i, ptxt in enumerate(txt.split("\f"), 1):
        m = re.search(r"^\s*Page\s+(\d{1,4})\s*$", ptxt, re.M)
        if m:
            labels[i] = m.group(1)
    return labels


# --------------------------------------------------------------------------- #
# shared helpers                                                              #
# --------------------------------------------------------------------------- #

def _crop_name(cap: CaptionHit, pno: int) -> str:
    sheet = f"_s{cap.sheet}" if cap.sheet else ""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", cap.canonical_id)
    return f"{cap.kind.lower()}_{safe}{sheet}_p{pno:04d}.png"


def _record(cap: CaptionHit, pno: int, label: str, path: str,
            bbox, dpi) -> FigureRecord:
    return FigureRecord(
        figure_id=f"{cap.kind} {cap.canonical_id}",
        kind=cap.kind,
        canonical_id=cap.canonical_id,
        sheet=cap.sheet,
        sheet_of=cap.sheet_of,
        page_pdf=pno,
        page_printed=label,
        caption=cap.raw_text,
        title=cap.title,
        image_path=path,
        bbox=list(bbox),
        dpi=dpi,
        chapter=chapter_of(cap.canonical_id),
    )
