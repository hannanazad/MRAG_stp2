"""Public façade: load everything once, then call `ask("...")`.

Designed for use inside a Jupyter cell:

    from mrag.ask import init_pipeline, ask
    init_pipeline()                # ~30 s on warm cache, prints what got loaded
    ask("Explain Figure 8C-1")     # answer Markdown + figure crops shown inline
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import CFG
from .embeddings import TextEmbedder, ImageEmbedder, Reranker
from .kg import KG, read as kg_read
from .vector_store import VectorStore
from .retrieval import Retriever, RetrievalResult
from .vlm import VLM

log = logging.getLogger("mrag.ask")


@dataclass
class Pipeline:
    store:    VectorStore  = None
    kg:       KG           = None
    text:     TextEmbedder = None
    image:    Optional[ImageEmbedder] = None
    rerank:   Reranker     = None
    vlm:      VLM          = None
    retriever: Retriever   = None


_pipeline: Optional[Pipeline] = None


def init_pipeline(load_image_embedder: bool = True, load_vlm: bool = True) -> Pipeline:
    """Idempotently load every component into a process-wide singleton."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    p = Pipeline()
    log.info("Loading Qdrant store @ %s", CFG.qdrant_dir)
    p.store = VectorStore(CFG.qdrant_dir)
    log.info("Loading KG @ %s", CFG.graph_pickle)
    p.kg = KG(kg_read(CFG.graph_pickle))
    log.info("Loading text embedder: %s", CFG.bge_m3_model)
    p.text = TextEmbedder(CFG.bge_m3_model).load()
    if load_image_embedder:
        try:
            log.info("Loading ColPali image embedder: %s", CFG.colqwen_model)
            p.image = ImageEmbedder(CFG.colqwen_model).load()
        except Exception as e:
            log.warning("Image embedder load failed (%r); page retrieval disabled.", e)
            p.image = None
    log.info("Loading reranker: %s", CFG.reranker_model)
    p.rerank = Reranker(CFG.reranker_model).load()
    p.retriever = Retriever(p.store, p.kg, p.text, p.image, p.rerank)
    if load_vlm:
        log.info("Loading VLM: %s", CFG.vlm_model)
        p.vlm = VLM(CFG.vlm_model, CFG.vlm_model_fallback).load()
    _pipeline = p
    return p


def ask(question: str, show_text: bool = False, show_scores: bool = False,
        max_fig_width: int = 640) -> Dict[str, Any]:
    """Answer + inline figure display. Returns the result dict for debugging."""
    p = init_pipeline()
    result = p.retriever.retrieve(question)

    # Figure-relevance filter (v5+): retrieval returns up to
    # CFG.top_k_figures_candidates figures; ask the VLM which are
    # visually relevant and keep only the top CFG.top_k_figures.
    # If the VLM isn't loaded or the filter is disabled, just keep
    # the first top_k_figures retrieval gave us.
    if result.figures:
        if (getattr(CFG, "use_vlm_figure_filter", False)
                and p.vlm is not None
                and len(result.figures) > CFG.top_k_figures):
            try:
                kept_idx = p.vlm.filter_figures(
                    question, result.figures, max_keep=CFG.top_k_figures,
                )
                if kept_idx:
                    # Preserve VLM-given order; record drops in debug
                    kept_set = set(kept_idx)
                    dropped = [
                        f.get("figure_id", "?")
                        for i, f in enumerate(result.figures)
                        if i not in kept_set
                    ]
                    result.figures = [result.figures[i] for i in kept_idx
                                      if 0 <= i < len(result.figures)]
                    result.debug["figures_dropped_by_filter"] = dropped
            except Exception as e:
                log.warning("VLM figure filter failed (%r); falling back to top-%d",
                            e, CFG.top_k_figures)
                result.figures = result.figures[: CFG.top_k_figures]
        else:
            # No filter — still cap at the display count
            result.figures = result.figures[: CFG.top_k_figures]

    # v4.3: when the question router decided NO figures are needed, the
    # answer must be fully text-only — suppress ColPali PAGE images from the
    # VLM prompt and from the display as well (they were leaking images into
    # no-figure answers). Pages stay in the returned dict for eval/citations.
    router_dec = result.debug.get("figure_router", {})
    text_only = router_dec.get("needs_figures") is False

    if p.vlm is not None:
        p.vlm.last_api_debug = {}

        try:
            answer = p.vlm.answer(
                question,
                result.chunks,
                result.figures,
                [] if text_only else result.pages,
                max_new_tokens=CFG.max_new_tokens,
            )
        except Exception as e:
            log.exception("VLM generation failed")
            answer = f"(VLM error: {e!r})"

        if p.vlm.last_api_debug:
            result.debug["vlm_response"] = dict(
                p.vlm.last_api_debug
            )
    else:
        answer = "(VLM not loaded — `init_pipeline(load_vlm=True)`)"

    _display(question, answer, result, show_text=show_text,
             show_scores=show_scores, max_fig_width=max_fig_width,
             suppress_pages=text_only)

    return {
        "question": question,
        "answer":   answer,
        "chunks":   result.chunks,
        "figures":  result.figures,
        "pages":    result.pages,
        "debug":    result.debug,
    }


def _display(question, answer, result: RetrievalResult,
             show_text=False, show_scores=False, max_fig_width=640,
             suppress_pages=False) -> None:
    """Render answer + figures inline (Jupyter)."""
    try:
        from IPython.display import display, Markdown, Image as IPImage
    except ImportError:
        print("Q:", question); print(answer); return

    display(Markdown(f"### Q\n{question}\n\n### Answer\n\n{answer}"))

    if result.figures:
        display(Markdown("### Figures the model saw"))
        for i, f in enumerate(result.figures, 1):
            ip = f.get("image_path", "")
            if ip and Path(ip).exists():
                cap = f.get("caption", "") or f.get("figure_id", "")
                sc = f.get("sign_codes") or []
                meta = f" — depicts {', '.join(sc)}" if sc else ""
                score_s = (
                    f"  \nscore={f.get('score',0):.3f}" if show_scores else ""
                )
                display(Markdown(
                    f"**[Image {i}]** {f.get('figure_id','?')} (p.{f.get('page_printed','?')}){meta}  \n*{cap}*{score_s}"
                ))
                display(IPImage(filename=ip, width=max_fig_width))

    if result.pages and not suppress_pages:
        display(Markdown("### Pages retrieved by ColPali"))
        for i, p in enumerate(result.pages, 1):
            ip = p.get("image_path", "")
            if ip and Path(ip).exists():
                display(Markdown(f"**[Page {i}]** p.{p.get('page_printed','?')} "
                                 f"(score={p.get('score',0):.3f})"))
                display(IPImage(filename=ip, width=max_fig_width))

    if show_text:
        display(Markdown("### Retrieved chunks"))
        for c in result.chunks:
            display(Markdown(
                f"**{c.get('section_id')} {c.get('content_type')} §{c.get('ordinal')}** — "
                f"{c.get('section_title','')} (p.{c.get('page_printed','?')}) "
                f"· score={c.get('score',0):.3f}\n\n{(c.get('text','') or '')[:600]}..."
            ))
