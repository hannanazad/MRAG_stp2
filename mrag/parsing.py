"""Outline-driven MUTCD parser.

Uses the PDF's built-in outline (Table of Contents) to find every Section
(L3 entry like "Section 2B.04 STOP Sign (R1-1)..."). For each section,
walks the text blocks of its pages and emits one chunk per numbered
paragraph, tagged with its rule type (Standard / Guidance / Option / Support).

The output of `parse_chunks()` is the *single source of truth* for chunks
used by every downstream stage (embeddings, KG, retrieval, prompt).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, List, Optional

try:
    import fitz  # PyMuPDF — required only by parse_chunks()
except Exception:                                     # pragma: no cover
    fitz = None


SECTION_RE   = re.compile(r"^Section\s+([0-9A-Z]+\.[0-9]+)\s+(.+)$")
RULE_RE      = re.compile(r"^(Standard|Support|Guidance|Option)\s*:\s*$")
NUM_ONLY_RE  = re.compile(r"^(\d{1,3})\s*$")
CHAPTER_RE   = re.compile(r"^CHAPTER\s+([0-9A-Z]+)\.", re.IGNORECASE)
PART_RE      = re.compile(r"^PART\s+(\d+)", re.IGNORECASE)
PRINTED_PG   = re.compile(r"^Page\s+\d{1,4}\s*$")
SECT_RANGE   = re.compile(r"^Sect\.\s+")
FOOTER_LINES = {"December 2023", "MUTCD 11th Edition"}

# Cross-reference / sign-code patterns (used by KG builder, exported for reuse).
SECREF_RE  = re.compile(r"\bSection\s+([0-9A-Z]+\.[0-9]+)\b", re.IGNORECASE)
FIGREF_RE  = re.compile(r"\bFigure\s+([0-9A-Z]+-[0-9]+(?:\([A-Za-z0-9]+\))?[A-Za-z]?)\b", re.IGNORECASE)
TABREF_RE  = re.compile(r"\bTable\s+([0-9A-Z]+-[0-9]+(?:\([A-Za-z0-9]+\))?[A-Za-z]?)\b", re.IGNORECASE)

# Modal-verb detection: "shall" => Standard provision tone, "should" => Guidance,
# "may" => Option. We never override the outline-supplied content_type, but we
# expose modal verbs for downstream scoring & display.
MODAL_VERBS = {"shall", "should", "may", "must"}
MODAL_RE = re.compile(r"\b(shall|should|may|must)\b", re.IGNORECASE)


@dataclass
class Chunk:
    chunk_id:       str
    part:           Optional[str]
    chapter:        Optional[str]
    section_id:     str
    section_title:  str
    content_type:   str               # Standard | Guidance | Option | Support
    ordinal:        int
    page_pdf:       int
    page_printed:   str
    text:           str
    figure_refs:    List[str]
    table_refs:     List[str]
    section_refs:   List[str]
    sign_codes:     List[str]
    modal_verbs:    List[str]


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

def parse_chunks(pdf_path: Path, sign_code_re: Optional[re.Pattern] = None) -> List[Chunk]:
    """Walk the entire PDF and return every typed paragraph as a Chunk."""
    if fitz is None:
        raise RuntimeError("parse_chunks requires PyMuPDF: pip install pymupdf")
    doc = fitz.open(str(pdf_path))
    toc = doc.get_toc()

    # ---------- 1. Build hierarchy from outline ----------------------------
    sections, hierarchy = _outline_to_sections(toc, doc.page_count)

    # ---------- 2. Parse each section --------------------------------------
    out: List[Chunk] = []
    for sec in sections:
        chunks = _parse_one_section(
            doc,
            sec_id=sec["id"],
            section_title=sec["title"],
            page_start_idx=sec["start"] - 1,   # outline is 1-indexed
            page_end_idx=sec["end"],            # exclusive
            hierarchy=hierarchy.get(sec["id"], {}),
            sign_code_re=sign_code_re,
        )
        out.extend(chunks)
    return out


def write_chunks_jsonl(chunks: List[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def read_chunks_jsonl(path: Path) -> List[Chunk]:
    out: List[Chunk] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            out.append(Chunk(**d))
    return out


# --------------------------------------------------------------------------- #
# Internals                                                                   #
# --------------------------------------------------------------------------- #

def _outline_to_sections(toc, page_count: int):
    """Return (sections_list, hierarchy_map)."""
    sections: List[dict] = []
    hierarchy: dict = {}
    current_part = current_chapter = None

    for lvl, title, pg in toc:
        t = title.strip()
        # PART headings appear at L1 (top of outline) or as L2 group titles.
        m_part = re.match(r"^PART\s+(\d+)\s+(.*)$", t, re.IGNORECASE)
        m_chap = re.match(r"^CHAPTER\s+([0-9A-Z]+)\.\s*(.*)$", t, re.IGNORECASE)
        m_sec  = SECTION_RE.match(t)

        if m_part:
            current_part = m_part.group(0).title()
        elif m_chap:
            current_chapter = m_chap.group(0).title()
        elif lvl == 3 and m_sec:
            sec_id, sec_title = m_sec.group(1), m_sec.group(2)
            sections.append({"id": sec_id, "title": sec_title, "start": pg})
            hierarchy[sec_id] = {
                "part":    current_part,
                "chapter": current_chapter,
            }

    for i, s in enumerate(sections):
        s["end"] = sections[i + 1]["start"] - 1 + 1 if i + 1 < len(sections) else page_count + 1
        # NB: parse_one_section uses range(start_idx, end_idx) where indexes are 0-based
        # and `end_idx` is exclusive. start_idx = s["start"] - 1; end_idx = s["end"].
    return sections, hierarchy


def _parse_one_section(
    doc: fitz.Document,
    sec_id: str,
    section_title: str,
    page_start_idx: int,
    page_end_idx: int,
    hierarchy: dict,
    sign_code_re: Optional[re.Pattern],
) -> List[Chunk]:
    chunks: List[Chunk] = []
    cur_rule: Optional[str] = None
    cur_ord: Optional[int]  = None
    cur_body: List[str]     = []
    section_started = False
    last_page_pdf   = page_start_idx + 1
    last_page_label = doc.load_page(page_start_idx).get_label() or str(page_start_idx + 1)

    def flush() -> None:
        nonlocal cur_body
        if section_started and cur_rule and cur_ord is not None and cur_body:
            text = re.sub(r"\s+", " ", " ".join(cur_body)).strip()
            # Revision-1 pages use U+2011 non-breaking hyphens in ids
            # ("Table 6B\u20112") — normalise so FIGREF/TABREF resolve.
            text = text.translate({0x2010: "-", 0x2011: "-", 0x2012: "-",
                                   0x2013: "-", 0x2014: "-"})
            if text:
                cid = f"MUTCD11e_{sec_id.replace('.', '')}_{cur_rule}_{cur_ord:02d}"
                figure_refs  = sorted(set(_normalize_id(x) for x in FIGREF_RE.findall(text)))
                table_refs   = sorted(set(_normalize_id(x) for x in TABREF_RE.findall(text)))
                section_refs = sorted({x for x in SECREF_RE.findall(text) if x != sec_id})
                sign_codes   = (
                    sorted(set(sign_code_re.findall(text))) if sign_code_re else []
                )
                modal = sorted({m.lower() for m in MODAL_RE.findall(text)})
                chunks.append(Chunk(
                    chunk_id=cid,
                    part=hierarchy.get("part"),
                    chapter=hierarchy.get("chapter"),
                    section_id=sec_id,
                    section_title=section_title,
                    content_type=cur_rule,
                    ordinal=cur_ord,
                    page_pdf=last_page_pdf,
                    page_printed=last_page_label,
                    text=text,
                    figure_refs=figure_refs,
                    table_refs=table_refs,
                    section_refs=section_refs,
                    sign_codes=sign_codes,
                    modal_verbs=modal,
                ))
        cur_body = []

    for p_idx in range(page_start_idx, page_end_idx):
        if p_idx >= doc.page_count: break
        page = doc.load_page(p_idx)
        last_page_pdf = p_idx + 1
        last_page_label = page.get_label() or str(p_idx + 1)
        blocks = sorted(
            [b for b in page.get_text("blocks") if b[6] == 0],
            key=lambda b: (b[1], b[0]),
        )
        for block in blocks:
            for line in block[4].split("\n"):
                s = line.strip()
                if not s: continue
                if s in FOOTER_LINES: continue
                if PRINTED_PG.match(s): continue
                if SECT_RANGE.match(s): continue

                m_section = SECTION_RE.match(s)
                if m_section:
                    if m_section.group(1) == sec_id:
                        section_started = True
                        cur_rule = None
                        cur_ord = None
                    elif section_started:
                        flush()
                        return chunks
                    continue  # skip other sections' titles entirely

                if CHAPTER_RE.match(s) or PART_RE.match(s):
                    if section_started:
                        flush()
                        return chunks
                    continue

                if not section_started:
                    continue

                m_rule = RULE_RE.match(s)
                if m_rule:
                    flush()
                    cur_rule = m_rule.group(1)
                    cur_ord = None
                    continue

                m_num = NUM_ONLY_RE.match(s)
                if m_num:
                    flush()
                    cur_ord = int(m_num.group(1))
                    continue

                if cur_rule and cur_ord is not None:
                    cur_body.append(s)
    flush()
    return chunks


def _normalize_id(s: str) -> str:
    """Normalise a figure/table id (e.g. 'figure 2b-1' -> '2B-1')."""
    s = s.strip().upper()
    s = re.sub(r"\s+", "", s)
    return s
