"""Retrieval pipeline.

  query
   │
   ├─ parse explicit ids / sign codes ────► direct lookups
   ├─ BGE-M3 dense+sparse hybrid (RRF) ───► top-K1 chunks
   ├─ +graph 1-hop expansion              ► augment candidate set
   ├─ apply scoring formula:
   │     S = α·dense + β·sparse + γ·hierarchy + δ·graph + ε·w(content_type)
   ├─ mxbai-rerank-large-v2 over top-K1 ──► top-K2 chunks
   ├─ ColQwen2 page retrieval (parallel) ─► top-K3 pages
   ├─ QUESTION ROUTER (v2) ───────────────► needs_figures? (see
   │     mrag/question_router.py). When NO, figure paths are skipped
   │     entirely and result.figures == [] — v1 attached ~4 figures to
   │     every answer regardless of need (measured precision 6%).
   └─ figures (only when the router says yes), merged from three paths:
       Path A — cross-linked from winning chunks via the KG (`see Figure 2B-1`),
                ordered by the rank of the citing chunk (best chunk first)
       Path C — ColQwen2 VISUAL retrieval over figure crops (added v5)
       Path B — caption-text similarity (off by default — was a major source
                of off-topic figures; toggle via CFG.use_caption_figure_fallback)
       Deduplicated and capped at CFG.top_k_figures_candidates. Optional
       VLM-based relevance filter (CFG.use_vlm_figure_filter) prunes to
       CFG.top_k_figures before display. Multi-sheet figures ship ALL their
       sheet images via payload["image_paths"].
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .config import CFG
from .embeddings import TextEmbedder, ImageEmbedder, Reranker
from .kg import KG
from .question_router import decide_figures
from .vector_store import VectorStore

log = logging.getLogger("mrag.retrieval")


@dataclass
class RetrievalResult:
    chunks:  List[Dict[str, Any]] = field(default_factory=list)
    figures: List[Dict[str, Any]] = field(default_factory=list)
    pages:   List[Dict[str, Any]] = field(default_factory=list)
    debug:   Dict[str, Any]       = field(default_factory=dict)


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        kg: KG,
        text_embedder: TextEmbedder,
        image_embedder: Optional[ImageEmbedder],
        reranker: Reranker,
    ) -> None:
        self.store = store
        self.kg = kg
        self.text = text_embedder
        self.img  = image_embedder
        self.rerank = reranker

    # ----- public entry point ----------------------------------------------

    def retrieve(self, query: str) -> RetrievalResult:
        result = RetrievalResult()
        result.debug["query"] = query

        # 1. Direct lookups -------------------------------------------------
        explicit = self.kg.query_entities(query)
        result.debug["query_entities"] = list(explicit)

        # 2. Hybrid chunk retrieval -----------------------------------------
        dense, sparse_list = self.text.encode_both([query])
        dense_q = dense[0]
        sparse_q = sparse_list[0]
        fused = self.store.search_chunks_hybrid(
            CFG.coll_chunks, dense_q, sparse_q, top_k=CFG.top_k_fused,
        )

        # 3. Graph expansion ------------------------------------------------
        candidate_ids: Set[int] = {h["id"] for h in fused}
        for ent_node in explicit:
            for nb in self.kg.neighbors(ent_node, n_hops=1):
                if not nb.startswith("chunk:"):
                    continue
                # Fetch this chunk by chunk_id from Qdrant via scroll (rare path).
                # In practice we just bump scoring for these in step 4.
                pass  # placeholder; scoring handles it.

        # 4. Apply the scoring formula --------------------------------------
        scored = []
        for hit in fused:
            payload = hit["payload"] or {}
            chunk_id = payload.get("chunk_id", "")
            base = float(hit.get("score", 0.0))
            s_graph = self.kg.proximity_score(explicit, chunk_id)
            s_rt = CFG.rule_type_weight(payload.get("content_type", "Support"))
            s_hier = _hierarchy_prior(query, payload)
            final = (
                CFG.w_dense   * base
                + CFG.w_graph * s_graph
                + CFG.w_ruletype * (s_rt - 1.0)            # center at 1.0
                + CFG.w_hierarchy * s_hier
            )
            scored.append((final, hit))
        scored.sort(key=lambda t: t[0], reverse=True)
        precursor = [h for _s, h in scored[: CFG.top_k_after_graph]]

        # 5. Cross-encoder rerank ------------------------------------------
        docs = [h["payload"].get("text", "")[:1500] for h in precursor]
        rerank_pairs = self.rerank.rank(query, docs, top_k=CFG.top_k_after_rerank)
        final_chunks = []
        for idx, score in rerank_pairs:
            hit = precursor[idx]
            payload = hit["payload"] or {}
            final_chunks.append({**payload, "score": score})
        result.chunks = final_chunks

        # 6. Figures — gated by the question router (v2), then three paths
        #    (A: KG cross-links, C: visual, B: caption fallback).
        figure_ids_seen: Set[str] = set()
        figs_out: List[Dict[str, Any]] = []
        candidate_cap = CFG.top_k_figures_candidates

        if getattr(CFG, "use_question_router", True):
            decision = decide_figures(
                query, kg=self.kg, top_chunks=final_chunks,
                soft_threshold=getattr(CFG, "router_soft_threshold", 0.5),
            )
            result.debug["figure_router"] = {
                "needs_figures": decision.needs_figures,
                "confidence": decision.confidence,
                "max_figures": decision.max_figures,
                "rules": decision.rules_fired,
                **decision.debug,
            }
            if not decision.needs_figures:
                result.figures = []
                # 7. ColPali page retrieval still runs (used for citations),
                #    handled below.
                return self._finish_pages(query, result)

        # Path A: figures the winning chunks explicitly cite via "see Figure X-Y"
        for ch in final_chunks:
            for fid in self.kg.figures_for_chunk(ch.get("chunk_id", "")):
                if fid in figure_ids_seen:
                    continue
                figure_ids_seen.add(fid)
                payload = _figure_payload_from_graph(self.kg, fid)
                if payload:
                    payload["source"] = "kg_link"
                    figs_out.append(payload)
                if len(figs_out) >= candidate_cap:
                    break
            if len(figs_out) >= candidate_cap:
                break

        # Path C: visual retrieval via ColPali on figure CROPS.
        # Only available if the image embedder loaded AND the visual
        # collection was populated by ingestion (v5+).
        if self.img is not None and len(figs_out) < candidate_cap:
            try:
                q_mv = self.img.encode_queries([query])[0]
                visual_hits = self.store.search_figures_visual(
                    CFG.coll_figures_visual, q_mv,
                    top_k=CFG.top_k_figures_visual,
                )
                for h in visual_hits:
                    payload = h.payload or {}
                    fid = payload.get("figure_id")
                    if fid and fid not in figure_ids_seen:
                        figure_ids_seen.add(fid)
                        figs_out.append({
                            **payload,
                            "score": float(getattr(h, "score", 0.0)),
                            "source": "visual",
                        })
                    if len(figs_out) >= candidate_cap:
                        break
            except Exception as e:
                log.warning("Visual figure retrieval failed: %r", e)

        # Path B: caption-text fallback (OFF by default in v5+). Was the
        # main contributor of off-topic figures in the previous design.
        if (CFG.use_caption_figure_fallback
                and len(figs_out) < candidate_cap):
            extra_hits = self.store.search_figures(
                CFG.coll_figures, dense_q, top_k=candidate_cap,
            )
            for h in extra_hits:
                payload = h.payload or {}
                fid = payload.get("figure_id")
                if fid and fid not in figure_ids_seen:
                    figure_ids_seen.add(fid)
                    figs_out.append({
                        **payload,
                        "score": float(h.score),
                        "source": "caption",
                    })
                if len(figs_out) >= candidate_cap:
                    break

        result.figures = figs_out
        return self._finish_pages(query, result)

    def _finish_pages(self, query: str, result: RetrievalResult) -> RetrievalResult:
        """ColPali page retrieval (runs for every query, incl. no-figure ones)."""
        if self.img is not None:
            try:
                q_mv = self.img.encode_queries([query])[0]
                page_hits = self.store.search_pages(CFG.coll_pages, q_mv, top_k=CFG.top_k_pages)
                result.pages = [
                    {**(h.payload or {}), "score": float(h.score)}
                    for h in page_hits
                ]
            except Exception as e:
                log.warning("ColPali page retrieval failed: %r", e)
        return result


def _hierarchy_prior(query: str, payload: Dict[str, Any]) -> float:
    """Cheap prior on top of dense+sparse: if the query mentions Part N or
    Chapter NX, give chunks in that branch a small boost."""
    score = 0.0
    q = query.lower()
    part = (payload.get("part") or "").lower()
    chapter = (payload.get("chapter") or "").lower()
    m_part = re.search(r"\bpart\s+(\d+)\b", q)
    if m_part and f"part {m_part.group(1)}" in part:
        score += 0.6
    m_chap = re.search(r"\bchapter\s+([0-9a-z]+)\b", q)
    if m_chap and m_chap.group(1) in chapter:
        score += 0.6
    return score


def _figure_payload_from_graph(kg: KG, figure_id: str) -> Optional[Dict[str, Any]]:
    node = kg.figure(figure_id)
    if not node:
        return None
    data = kg.g.nodes[node]
    return {
        "figure_id":     data.get("id", figure_id),
        "canonical_id":  data.get("canonical_id", ""),
        "chapter":       data.get("chapter", ""),
        "anchor_section": data.get("anchor_section", ""),
        "page_pdf":      data.get("page_pdf"),
        "page_printed":  data.get("page_printed"),
        "caption":       data.get("caption", ""),
        "title":         data.get("title", ""),
        "image_path":    data.get("image_path", ""),
        "image_paths":   list(data.get("image_paths",
                                         (data.get("image_path", ""),))),
        "n_sheets":      data.get("n_sheets", 1),
        "sign_codes":    list(data.get("sign_codes", [])),
        "source":        "graph_link",
    }
