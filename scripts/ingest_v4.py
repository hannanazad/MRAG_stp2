#!/usr/bin/env python3
"""MUTCD ingestion v4: corrected figure extraction + KG v2 + coverage gate.

WHAT CHANGED vs v3
------------------
1. FIGURE EXTRACTION uses the rewritten mrag/figures.py (v2):
     - captions anchor the TOP of their region (v1 cropped the region ABOVE
       each caption — inverted for MUTCD layout, which silently dropped or
       mis-cropped a large share of figures);
     - TOC / List-of-Figures pages rejected;
     - inset (two-column) captions detected at word level;
     - side-by-side captions partition the page width;
     - U+2011 non-breaking-hyphen ids normalised;
     - "(Sheet n of m)" figures produce one crop per sheet.
2. RESCUE PASS re-extracts, with a relaxed min-height, ONLY ids that got zero
   crops (e.g. tiny Table 2A-3).
3. COVERAGE GATE — ingestion FAILS (non-zero exit) if figure coverage against
   the full-document mention census is below --min-coverage (default 0.99).
   A coverage report JSON is always written. No silent gaps, ever.
4. KG v2 (mrag/kg.py): one node per canonical figure id (sheets collapsed
   into image_paths), 3-tier section anchoring, unresolved-ref stats,
   cites_any_figure flags for the question router.
5. Chunks are re-parsed (parsing.py now normalises U+2011 dashes, so chunk
   figure_refs resolve refs like "Table 6B‑2").

Embedding + Qdrant steps are UNCHANGED — they are imported from ingest_v3 to
keep one source of truth. Delete these cache files to force the v4 rebuild:
    mmrag_cache_v3/figures.jsonl  mmrag_cache_v3/chunks.jsonl
    mmrag_cache_v3/graph.gpickle  figures/ (the crop PNGs)

Usage
-----
    python scripts/ingest_v4.py                  # full pipeline
    python scripts/ingest_v4.py --figures-only   # stop after coverage gate
    python scripts/ingest_v4.py --min-coverage 1.0
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mrag.config import CFG
from mrag import parsing as P
from mrag import figures as F
from mrag import sign_codes as SC
from mrag import kg as KGM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
)
log = logging.getLogger("ingest_v4")


# --------------------------------------------------------------------------- #
def _full_text() -> str:
    """Text layer of the whole PDF for the mention census."""
    try:
        import fitz
        doc = fitz.open(str(CFG.pdf_path))
        return "\f".join(doc.load_page(i).get_text() for i in range(doc.page_count))
    except Exception:
        import subprocess
        return subprocess.run(
            ["pdftotext", "-layout", str(CFG.pdf_path), "-"],
            capture_output=True, text=True).stdout


def _invalidate_embedding_caches(*names: str):
    """Cached .npy/.json embedding arrays are positional — stale the moment
    their source rows change. Delete them so v3's embedding steps recompute."""
    for n in names:
        p = CFG.cache_dir / n
        if p.exists():
            p.unlink()
            log.warning("invalidated stale embedding cache: %s", p)


def step_extract_figures_v4(min_coverage: float):
    """Extract -> rescue -> validate. The single most important change in v4."""
    if CFG.figures_jsonl.exists():
        figs = F.read_jsonl(CFG.figures_jsonl)
        if figs and figs[0].extraction_method.startswith("caption_below_v2"):
            log.info("figures.jsonl is already v2 (%d crops); skipping "
                     "extraction (delete to redo)", len(figs))
            return figs
        log.warning("figures.jsonl is v1 — archiving and re-extracting with v2")
        shutil.move(str(CFG.figures_jsonl),
                    str(CFG.figures_jsonl.with_suffix(".v1.bak.jsonl")))
        # Figure captions/count changed -> caption embeddings stale, and the
        # per-figure ColQwen2 crop embeddings (keyed by figure_id) would be
        # wrongly treated as "already cached" even though the CROP IMAGES
        # changed. Invalidate both.
        _invalidate_embedding_caches("figures_dense.npy")
        colq = CFG.cache_dir / "colqwen_figures"
        if colq.exists():
            shutil.rmtree(colq)
            log.warning("invalidated stale visual-embedding cache dir: %s", colq)

    full_text = _full_text()
    figs = F.extract_figures(CFG.pdf_path, CFG.figures_dir, dpi=CFG.figure_dpi)
    F.rescue_missing(figs, full_text, CFG.pdf_path, CFG.figures_dir,
                     dpi=CFG.figure_dpi)

    rep = F.validate_coverage(figs, full_text, fail_below=min_coverage)
    rep_path = CFG.cache_dir / "figure_coverage_report.json"
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(rep_path, "w"), indent=2)
    log.info("coverage report -> %s", rep_path)

    F.write_jsonl(figs, CFG.figures_jsonl)
    log.info("Wrote %d crops (%d canonical ids) -> %s",
             len(figs), len({(f.kind, f.canonical_id) for f in figs}),
             CFG.figures_jsonl)
    return figs


def step_parse_chunks_v4():
    """Re-parse chunks: parsing.py v4 normalises U+2011 dashes, so refs like
    'Table 6B\u20112' now populate chunk.table_refs. Cached chunks parsed by the
    old code miss those refs — force re-parse if the cache predates v4."""
    stamp = CFG.cache_dir / "chunks_v4.stamp"
    if CFG.chunks_jsonl.exists() and stamp.exists():
        log.info("chunks.jsonl is v4; skipping parse (delete to redo)")
        return P.read_chunks_jsonl(CFG.chunks_jsonl)
    if CFG.chunks_jsonl.exists():
        shutil.move(str(CFG.chunks_jsonl),
                    str(CFG.chunks_jsonl.with_suffix(".v3.bak.jsonl")))
        _invalidate_embedding_caches("chunks_dense.npy", "chunks_sparse.json")
    import fitz
    doc = fitz.open(str(CFG.pdf_path))
    outline_codes = SC.mine_from_outline(doc.get_toc())
    log.info("Bootstrapped %d sign codes from outline", len(outline_codes))
    chunks = P.parse_chunks(CFG.pdf_path, sign_code_re=SC.get_sign_code_re())
    P.write_chunks_jsonl(chunks, CFG.chunks_jsonl)
    stamp.write_text("v4")
    log.info("Parsed %d chunks -> %s", len(chunks), CFG.chunks_jsonl)
    return chunks


def step_build_kg_v4(chunks, figures, sign_codes_dict):
    if CFG.graph_pickle.exists():
        g = KGM.read(CFG.graph_pickle)
        if g.graph.get("schema_version") == 2:
            log.info("graph.gpickle is schema v2; skipping build (delete to redo)")
            return g
        log.warning("graph.gpickle is v1 — archiving and rebuilding as v2")
        shutil.move(str(CFG.graph_pickle),
                    str(CFG.graph_pickle.with_suffix(".v1.bak")))
    g = KGM.build(chunks, figures, sign_codes_dict)
    KGM.write(g, CFG.graph_pickle)
    log.info("KG v2 built: %s", g.graph["build_stats"])
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures-only", action="store_true",
                    help="stop after the figure coverage gate")
    ap.add_argument("--min-coverage", type=float, default=0.99,
                    help="fail ingestion if figure coverage is below this")
    ap.add_argument("--skip-embeddings", action="store_true",
                    help="build chunks/figures/KG but skip embedding + Qdrant")
    args = ap.parse_args()

    log.info("=== ingest v4 — base dir %s ===", CFG.base_dir)

    figures = step_extract_figures_v4(args.min_coverage)
    if args.figures_only:
        log.info("--figures-only: stopping after coverage gate. All good.")
        return

    chunks = step_parse_chunks_v4()

    # sign codes + figure sign-code enrichment (v3 logic, unchanged)
    from scripts.ingest_v3 import (step_sign_codes, step_enrich_figures,
                                   step_render_pages)
    sign_codes_dict = step_sign_codes(chunks)
    figures = step_enrich_figures(figures, sign_codes_dict)

    g = step_build_kg_v4(chunks, figures, sign_codes_dict)

    if args.skip_embeddings:
        log.info("--skip-embeddings: done (chunks+figures+KG only).")
        return

    step_render_pages()
    from scripts.ingest_v3 import (step_text_embeddings, step_page_embeddings,
                                   step_figure_visual_embeddings,
                                   step_upsert_qdrant)
    dense_c, sparse_c, dense_f = step_text_embeddings(chunks, figures)
    page_data = step_page_embeddings(CFG.page_images_dir)
    fig_visual = step_figure_visual_embeddings(figures)
    step_upsert_qdrant(chunks, figures, dense_c, sparse_c, dense_f,
                       page_data, fig_visual)
    log.info("=== ingest v4 complete ===")


if __name__ == "__main__":
    main()
