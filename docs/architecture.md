# MUTCD Multimodal RAG — v3 Architecture

This document is the single source of truth for *what the system does and
why*. Last revised together with the v3 pipeline rewrite.

## Pipeline diagram

```
                                  $SCRATCH/MRAG/mutcd*.pdf
                                            │
       ┌────────────────────────────────────┴───────────────────────────────────────┐
       │                                                                             │
       ▼                                                                             ▼
  outline walk                                                              caption-anchored
  (PDF TOC,                                                                  figure / table
  5,487 entries)                                                             cropping
       │                                                                             │
       ▼                                                                             ▼
  typed paragraph chunks                                                      figures/<id>_pNNNN.png
  (one per Standard / Support /                                              + figures.jsonl
   Guidance / Option paragraph)
       │
       ├──► sign-code miner (from section titles + chunk text + figure captions)
       │            │
       │            ▼
       │      sign_codes.json   (R1-1 -> "STOP sign" / Regulatory, …)
       │
       │     ┌────── cross-refs: "see Section X.Y", "see Figure X-Y", "see Table X-Y"
       │     │
       ▼     ▼
  ┌─────────────────────────┐
  │ NetworkX MultiDiGraph   │   ◄── Sections, Chunks, Figures, Tables, SignCodes, Categories,
  │ ~15k nodes, ~40k edges  │      with `contains`, `cites_*`, `defines`, `depicts`,
  │ graph.gpickle           │      `mentions`, `illustrated_by`, `kind_of` edges.
  └────────────┬────────────┘
               │
               │   ┌──────────────────────────────────────────────────────────┐
               │   │ BGE-M3 (dense 1024-d + sparse token weights)             │
               │   │   - encode chunks (with rule-type label preserved)       │
               │   │   - encode figure captions + sign codes                   │
               │   ├──────────────────────────────────────────────────────────┤
               │   │ ColQwen2-v0.1 multi-vector page embeddings (ColPali)     │
               │   │   - 768 patches × 128-d per page, binary-quantized        │
               │   └──────────────────────────────────────────────────────────┘
               │              │
               ▼              ▼
        ┌─────────────────────────────────────────────────────┐
        │ Qdrant local-file store (no daemon)                  │
        │  - mutcd_chunks   (dense + sparse + payload)         │
        │  - mutcd_figures  (dense + payload, image_path link) │
        │  - mutcd_pages    (multi-vector, MAX_SIM late-interaction)
        └─────────────────────────────────────────────────────┘

  ─────────────────────  query-time  ─────────────────────────────────────────

   query
     │
     ├─ KG.query_entities  →  Figure/Table/Section/SignCode mentions
     │
     ├─ BGE-M3 encode  →  dense + sparse
     │
     ├─ hybrid search (RRF over dense + sparse) → top-30
     │
     ├─ scoring: S = α·dense + β·sparse + γ·hierarchy + δ·graph + ε·rule_type
     │
     ├─ mxbai-rerank-large-v2 cross-encoder → top-6 chunks
     │
     ├─ figures: cross-link via KG (Chunk →cites_figure→ Figure)
     │           + BGE-M3 caption retrieval fallback
     │
     ├─ pages:   ColQwen2 query-encode + MAX_SIM late-interaction → top-4 pages
     │
     ▼
   Qwen2.5-VL-7B-Instruct (fallback: 3B)
     - sees figure crops + page images
     - sees chunks with content_type labels kept
     - emits structured answer:
         Direct Answer  (2-3 sentences)
         Standards      (mandatory bullets)
         Guidance       (recommended bullets)
         Options        (permitted bullets)
         Visual evidence
         Citations  (whitelisted by retrieval)
```

## Scoring formula

For a candidate chunk \(c_i\) and a query \(q\):

\[
S(q, c_i) \;=\;
\alpha \cdot S_{\text{dense}}(q, c_i) \;+\;
\beta  \cdot S_{\text{sparse}}(q, c_i) \;+\;
\gamma \cdot S_{\text{hierarchy}}(q, c_i) \;+\;
\delta \cdot S_{\text{graph}}(q, c_i) \;+\;
\varepsilon \cdot w(\text{content\_type}_{c_i})
\]

Defaults: \(\alpha=1.0,\beta=0.6,\gamma=0.2,\delta=0.4,\varepsilon=0.3\).

Rule-type weights (the "must / should / may" backbone of MUTCD):

| content_type | weight |
| --- | --- |
| Standard | 1.20 |
| Guidance | 1.00 |
| Option   | 0.90 |
| Support  | 0.70 |

Graph proximity:

\[
S_{\text{graph}}(q, c_i) \;=\; \frac{1}{1 + \min_{v \in V_q} d(v, c_i)}
\]

where \(V_q\) is the set of graph nodes mentioned by the query and \(d\) is
shortest-path distance in the undirected projection of the KG.

## Knowledge graph schema

### Node kinds

| Kind        | id prefix | attributes                                                      |
| ----------- | --------- | --------------------------------------------------------------- |
| Part        | `part:`     | title                                                           |
| Chapter     | `chapter:`  | title, part                                                     |
| Section     | `section:`  | id, title, page_pdf, page_printed, chapter, part                |
| Chunk       | `chunk:`    | id, section, content_type, ordinal, page, modal_verbs           |
| Figure      | `figure:`   | id, page_pdf, page_printed, caption, image_path, sign_codes     |
| Table       | `figure:`   | same shape as Figure                                            |
| SignCode    | `signcode:` | id (code), category, canonical_name                             |
| Category    | `category:` | id (Regulatory / Warning / Guide / …)                          |

### Edge labels

| Edge | From | To | Source |
| --- | --- | --- | --- |
| `contains`        | Part / Chapter / Section | Chapter / Section / Chunk | outline |
| `illustrated_by`  | Section | Figure / Table | same-page co-location |
| `cites_section`   | Chunk | Section | `\bSection X.Y\b` regex in chunk text |
| `cites_figure`    | Chunk | Figure | `\bFigure X-Y\b` regex |
| `cites_table`     | Chunk | Table  | `\bTable X-Y\b` regex |
| `defines`         | Section | SignCode | sign code in Section title |
| `mentions`        | Chunk | SignCode | sign code in chunk text |
| `depicts`         | Figure / Table | SignCode | sign code in caption |
| `kind_of`         | SignCode | Category | sign-code prefix lookup |

## Why each design choice

| Decision | Justification |
| --- | --- |
| **Outline-driven chunking** | The MUTCD PDF outline already encodes Standard / Option / Guidance / Support boundaries as L4 entries. Reading them is *more* reliable than ML layout detection on this document, with zero extra dependencies. |
| **Keep rule-type labels in text + payload** | These four labels are the modal-verb backbone of MUTCD. Stripping them (as the v2 pipeline did) destroys the most useful semantic signal. We preserve them in the chunk text *and* expose them as filterable payload. |
| **BGE-M3 for text** | Single 568M-param model gives us dense + sparse + (optionally) ColBERT; hybrid retrieval comes "for free" without a separate BM25 library. 8k-token context. Apache 2.0. |
| **ColQwen2 for pages** | Late-interaction over visual patches is 2026 SOTA for visually-rich documents (ViDoRe benchmark). Beats CLIP-plus-caption baselines by 10-20 nDCG on tables/figures. Binary quantization brings storage from ~256 KB/page to ~16 KB/page with <2% recall loss. |
| **mxbai-rerank-large-v2** | Best open reranker by current benchmarks (57.49 nDCG@10 BEIR, ahead of bge-reranker-v2-gemma 2.5B). Latency ~900 ms acceptable per user spec. |
| **Qdrant embedded** | Native ColPali multi-vector + binary quantization + best-in-class metadata filtering. Runs as a local file via `QdrantClient(path=…)` — no daemon, ideal for HPRC notebooks. |
| **Lightweight KG (NetworkX, no Neo4j)** | 15k nodes / 40k edges fits in a few MB pickle. Built deterministically from the PDF and regex extraction — no LLM in the ingestion loop, so no hallucinated edges. Same expressive power as Neo4j for our size and use case. |
| **Qwen2.5-VL-7B default** | Markedly better prose than 3B; ~17 GB bf16 fits comfortably on an A100. 3B is the auto-fallback for tighter GPUs. |
| **Structured prompt mirroring MUTCD taxonomy** | Standard/Guidance/Option/Support output sections force the model to separate mandatory from recommended from permitted — the legally critical distinction. |
| **Citation whitelist** | The prompt's "Allowed citations" list is built from retrieval results. The model can't fabricate section / figure / page numbers because anything outside the whitelist contradicts the prompt. |
| **Inline `ask()`, no gradio** | No reverse proxy, no port management, no daemon to babysit. Native Jupyter Markdown + Image display is sufficient and rock-solid. |

## Resource budget (HPRC, A100 40 GB)

| Model | VRAM (bf16) |
| --- | --- |
| Qwen2.5-VL-7B-Instruct | ~16 GB |
| ColQwen2-v0.1 (2B)     | ~5 GB  |
| BGE-M3 (568M)          | ~1.2 GB |
| mxbai-rerank-large-v2 (1.5B) | ~3.5 GB |
| **Total simultaneously** | **~26 GB** |

Disk on `$SCRATCH`:

| Asset | Size |
| --- | --- |
| MUTCD PDF | ~32 MB |
| Page PNGs (180 DPI × 1162 pages) | ~3 GB |
| Figure crops (~500 PNGs) | ~500 MB |
| Qdrant DB | ~2 GB |
| HF model cache | ~35 GB |

## Caveats

- **Appendix / preface chunks** are not parsed by `mrag.parsing` because they lack the `Section X.Y` prefix used by the outline. Add a fallback later if those become valuable.
- **Table region cropping** sometimes over-crops when several tables share a page; replace with a layout-detection model (Docling, Marker) only if specific tables come out wrong in evaluation.
- **Hierarchy prior** in the scoring formula is a coarse keyword match on "Part N" / "Chapter NX". A learned routing classifier is a future improvement.
- **Eval harness** is not yet built; that's the next thing we add once you have 10–40 (question, expected_section, expected_figure) tuples.
