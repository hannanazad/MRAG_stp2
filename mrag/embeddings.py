"""Embedding model wrappers.

- TextEmbedder: BGE-M3 dense (1024-d) + sparse (BM25-style token weights).
- ImageEmbedder: ColQwen2-v0.1 multi-vector page embeddings (ColPali family).
- Reranker:     mxbai-rerank-large-v2 (cross-encoder).

These are thin wrappers so we can swap any of them without changing the
ingestion or retrieval modules.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

log = logging.getLogger("mrag.embeddings")


# --------------------------------------------------------------------------- #
# Text embedder: BGE-M3 (dense + sparse)                                      #
# --------------------------------------------------------------------------- #

class TextEmbedder:
    """BGE-M3 returning dense (1024-d) + sparse (token->weight) per text.

    Uses the official FlagEmbedding implementation when available, falls back
    to sentence-transformers (dense-only) and a tiny in-house BM25 for sparse.
    """

    DIM = 1024

    def __init__(self, model_name: str = "BAAI/bge-m3", device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device = device or _auto_device()
        self._model = None
        self._mode = None       # "bge-m3" | "st-fallback"
        self._st_corpus_for_bm25: Optional[Any] = None

    # ----- lifecycle --------------------------------------------------------
    def load(self) -> "TextEmbedder":
        try:
            from FlagEmbedding import BGEM3FlagModel
            self._model = BGEM3FlagModel(
                self.model_name, use_fp16=("cuda" in self.device), device=self.device,
            )
            self._mode = "bge-m3"
            log.info("TextEmbedder loaded: BGE-M3 (dense+sparse)")
        except Exception as e:
            log.warning("BGE-M3 unavailable (%r); falling back to sentence-transformers", e)
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._mode = "st-fallback"
        return self

    # ----- encoding --------------------------------------------------------
    def encode_dense(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        if self._mode == "bge-m3":
            out = self._model.encode(
                texts, batch_size=batch_size,
                return_dense=True, return_sparse=False, return_colbert_vecs=False,
            )
            return np.asarray(out["dense_vecs"], dtype=np.float32)
        return self._model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)

    def encode_sparse(self, texts: List[str], batch_size: int = 16) -> List[Dict[int, float]]:
        """Return list of {token_id: weight}. For BGE-M3 these are real sparse
        weights; for the fallback we return empty dicts (no sparse signal)."""
        if self._mode == "bge-m3":
            out = self._model.encode(
                texts, batch_size=batch_size,
                return_dense=False, return_sparse=True, return_colbert_vecs=False,
            )
            # FlagEmbedding returns lexical_weights as list of {token: weight}.
            converted = []
            for d in out["lexical_weights"]:
                converted.append({int(k): float(v) for k, v in d.items()})
            return converted
        return [dict() for _ in texts]

    def encode_both(self, texts: List[str], batch_size: int = 16):
        if self._mode == "bge-m3":
            out = self._model.encode(
                texts, batch_size=batch_size,
                return_dense=True, return_sparse=True, return_colbert_vecs=False,
            )
            dense = np.asarray(out["dense_vecs"], dtype=np.float32)
            sparse = [{int(k): float(v) for k, v in d.items()}
                      for d in out["lexical_weights"]]
            return dense, sparse
        return self.encode_dense(texts, batch_size), [dict() for _ in texts]


# --------------------------------------------------------------------------- #
# Image embedder: ColQwen2 (multi-vector via colpali-engine)                   #
# --------------------------------------------------------------------------- #

class ImageEmbedder:
    """ColQwen2 page-image embedder.

    Encodes a PIL Image (or a list of them) into a (num_patches, dim) array.
    Queries are encoded to (num_query_tokens, dim).
    """

    def __init__(
        self,
        model_name: str = "vidore/colqwen2-v0.1",
        device: Optional[str] = None,
        torch_dtype: Optional[str] = "bfloat16",
    ) -> None:
        self.model_name = model_name
        self.device = device or _auto_device()
        self.torch_dtype = torch_dtype
        self._model = None
        self._processor = None

    def load(self) -> "ImageEmbedder":
        import torch
        from colpali_engine.models import ColQwen2, ColQwen2Processor
        dtype = getattr(torch, self.torch_dtype) if isinstance(self.torch_dtype, str) else self.torch_dtype
        self._model = ColQwen2.from_pretrained(
            self.model_name, torch_dtype=dtype, device_map=self.device,
        ).eval()
        self._processor = ColQwen2Processor.from_pretrained(self.model_name)
        log.info("ImageEmbedder loaded: %s", self.model_name)
        return self

    def encode_images(self, images, batch_size: int = 2) -> List[np.ndarray]:
        import torch
        out: List[np.ndarray] = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            inputs = self._processor.process_images(batch).to(self._model.device)
            with torch.no_grad():
                emb = self._model(**inputs)
            for vec in emb:
                # vec: (n_patches, dim) tensor → numpy float32
                out.append(vec.detach().to(torch.float32).cpu().numpy())
        return out

    def encode_queries(self, queries: List[str]) -> List[np.ndarray]:
        import torch
        inputs = self._processor.process_queries(queries).to(self._model.device)
        with torch.no_grad():
            emb = self._model(**inputs)
        return [v.detach().to(torch.float32).cpu().numpy() for v in emb]


def maxsim(q_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    """ColBERT-style late-interaction score: sum over query tokens of the max
    similarity to any document patch.

    Inputs are L2-normalised float32 arrays:
      q_emb:   (q, d)
      doc_emb: (p, d)
    """
    # (q, p) similarity matrix; sum over q of per-query max over p.
    sim = q_emb @ doc_emb.T
    return float(sim.max(axis=1).sum())


# --------------------------------------------------------------------------- #
# Reranker: mxbai-rerank-large-v2                                              #
# --------------------------------------------------------------------------- #

class Reranker:
    """Cross-encoder reranker."""

    def __init__(
        self,
        model_name: str = "mixedbread-ai/mxbai-rerank-large-v2",
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or _auto_device()
        self._model = None

    def load(self) -> "Reranker":
        try:
            from mxbai_rerank import MxbaiRerankV2
            self._model = MxbaiRerankV2(self.model_name)
            log.info("Reranker loaded: %s", self.model_name)
        except Exception as e:
            log.warning("mxbai-rerank not available (%r); falling back to BGE reranker", e)
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder("BAAI/bge-reranker-v2-m3", device=self.device)
        return self

    def rank(self, query: str, docs: List[str], top_k: int = 6) -> List[Tuple[int, float]]:
        """Return [(doc_index, score), ...] sorted descending."""
        if hasattr(self._model, "rank"):
            # mxbai-rerank's `rank()` kwarg for the candidate documents is
            # named `documents` in 0.1.x+ (the released API on PyPI today).
            # Earlier internal pre-release builds used `input`; we keep that
            # as a fallback so we don't break if someone pins an old build.
            try:
                res = self._model.rank(query=query, documents=docs, top_k=top_k)
            except TypeError:
                res = self._model.rank(query=query, input=docs, top_k=top_k)
            return [(r.index, float(r.score)) for r in res]
        # CrossEncoder fallback
        pairs = [(query, d) for d in docs]
        scores = self._model.predict(pairs)
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(i, float(scores[i])) for i in order]


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #

def _auto_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
