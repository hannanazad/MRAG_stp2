"""MRAG — Multimodal RAG for the MUTCD on TAMU HPRC.

Pipeline:
    PDF ──► outline-driven chunk parser  ──► typed paragraphs (Standard/Guidance/Option/Support)
        ──► caption-anchored figure cropper ──► figure PNGs + figures.jsonl
        ──► sign-code miner                  ──► sign_codes.json
        ──► knowledge-graph builder          ──► graph.gpickle (NetworkX)
        ──► BGE-M3 chunk/figure embeddings   ──► Qdrant collections
        ──► ColQwen2 page embeddings         ──► Qdrant multivector collection
                                             |
                                  query ────►│
                                             ▼
                                hybrid retrieval (dense + sparse)
                                graph expansion (1-hop)
                                rule-type weighting (Standard > Guidance > Option > Support)
                                mxbai-rerank-large-v2
                                ─────────────►  Qwen2.5-VL-7B-Instruct  ───► structured answer
"""
__version__ = "0.3.0"
