"""Qdrant local-file vector store: one DB folder, four collections.

Collections:
  mutcd_chunks          - chunks: dense + sparse, rich payload
  mutcd_figures         - figures: dense on caption+title, payload incl. image_path
  mutcd_figures_visual  - figures: ColPali multi-vector on the figure CROP IMAGE
                          (added in v5 to retrieve figures by visual content
                          rather than caption text)
  mutcd_pages           - pages:   ColPali multi-vector with binary quantization

All run in *embedded* mode via `QdrantClient(path=...)`. No daemon.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

log = logging.getLogger("mrag.vector_store")


@dataclass
class ChunkRow:
    id:       int                 # stable integer id (hash of chunk_id)
    dense:    np.ndarray
    sparse:   Dict[int, float]
    payload:  Dict[str, Any]


@dataclass
class FigureRow:
    id:       int
    dense:    np.ndarray
    payload:  Dict[str, Any]


@dataclass
class PageRow:
    id:       int
    vectors:  np.ndarray          # (num_patches, dim)
    payload:  Dict[str, Any]


class VectorStore:
    def __init__(self, qdrant_dir: Path) -> None:
        self.qdrant_dir = Path(qdrant_dir)
        self.qdrant_dir.mkdir(parents=True, exist_ok=True)
        from qdrant_client import QdrantClient
        self._client = QdrantClient(path=str(self.qdrant_dir))

    @property
    def client(self):
        return self._client

    # ----- schema -----------------------------------------------------------

    def init_collections(
        self,
        coll_chunks: str,
        coll_figures: str,
        coll_pages: str,
        text_dim: int = 1024,
        page_patch_dim: int = 128,
        use_binary_quantization_for_pages: bool = True,
        coll_figures_visual: Optional[str] = None,
        figure_patch_dim: Optional[int] = None,
    ) -> None:
        """Create (or re-create) the four collections.

        coll_figures_visual is optional for backward compatibility — if not
        provided, no visual-figures collection is created. figure_patch_dim
        defaults to page_patch_dim (same ColPali model dimension).
        """
        from qdrant_client.http import models as qm

        def recreate(name: str, vectors_config, sparse_vectors_config=None,
                     quantization_config=None):
            # Delete-then-create rather than the deprecated recreate_collection.
            if self._client.collection_exists(name):
                self._client.delete_collection(name)
            kwargs = dict(
                collection_name=name,
                vectors_config=vectors_config,
            )
            if sparse_vectors_config:
                kwargs["sparse_vectors_config"] = sparse_vectors_config
            if quantization_config is not None:
                kwargs["quantization_config"] = quantization_config
            self._client.create_collection(**kwargs)

        # 1. Chunks: named dense vector + named sparse vector
        recreate(
            coll_chunks,
            vectors_config={
                "dense": qm.VectorParams(size=text_dim, distance=qm.Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": qm.SparseVectorParams(),
            },
        )
        # 2. Figures: dense only (caption + sign codes embedded)
        recreate(
            coll_figures,
            vectors_config={
                "dense": qm.VectorParams(size=text_dim, distance=qm.Distance.COSINE),
            },
        )
        # 3. Pages: ColPali multi-vector with optional binary quantization
        qcfg = (
            qm.BinaryQuantization(binary=qm.BinaryQuantizationConfig(always_ram=True))
            if use_binary_quantization_for_pages else None
        )
        recreate(
            coll_pages,
            vectors_config={
                "colbert": qm.VectorParams(
                    size=page_patch_dim,
                    distance=qm.Distance.COSINE,
                    multivector_config=qm.MultiVectorConfig(
                        comparator=qm.MultiVectorComparator.MAX_SIM,
                    ),
                    quantization_config=qcfg,
                ),
            },
        )
        # 4. Figures-visual: ColPali multi-vector on each figure CROP
        # (added v5). Same schema shape as pages, just a different collection
        # name so we can query figures and pages independently. Quantization
        # is OFF here because figure counts are small (~hundreds) so memory
        # isn't a concern and we want full-precision relevance ranking.
        if coll_figures_visual:
            fdim = figure_patch_dim or page_patch_dim
            recreate(
                coll_figures_visual,
                vectors_config={
                    "colbert": qm.VectorParams(
                        size=fdim,
                        distance=qm.Distance.COSINE,
                        multivector_config=qm.MultiVectorConfig(
                            comparator=qm.MultiVectorComparator.MAX_SIM,
                        ),
                    ),
                },
            )

    # ----- ingestion --------------------------------------------------------

    def upsert_chunks(self, name: str, rows: List[ChunkRow], batch: int = 256) -> None:
        from qdrant_client.http import models as qm
        for i in range(0, len(rows), batch):
            slice_ = rows[i:i+batch]
            self._client.upsert(
                collection_name=name,
                points=[
                    qm.PointStruct(
                        id=r.id,
                        vector={
                            "dense": r.dense.tolist(),
                            "sparse": qm.SparseVector(
                                indices=list(r.sparse.keys()),
                                values=list(r.sparse.values()),
                            ),
                        },
                        payload=r.payload,
                    )
                    for r in slice_
                ],
                wait=True,
            )

    def upsert_figures(self, name: str, rows: List[FigureRow], batch: int = 256) -> None:
        from qdrant_client.http import models as qm
        for i in range(0, len(rows), batch):
            slice_ = rows[i:i+batch]
            self._client.upsert(
                collection_name=name,
                points=[
                    qm.PointStruct(
                        id=r.id,
                        vector={"dense": r.dense.tolist()},
                        payload=r.payload,
                    )
                    for r in slice_
                ],
                wait=True,
            )

    def upsert_pages(self, name: str, rows: List[PageRow], batch: int = 32) -> None:
        from qdrant_client.http import models as qm
        for i in range(0, len(rows), batch):
            slice_ = rows[i:i+batch]
            self._client.upsert(
                collection_name=name,
                points=[
                    qm.PointStruct(
                        id=r.id,
                        vector={"colbert": r.vectors.tolist()},
                        payload=r.payload,
                    )
                    for r in slice_
                ],
                wait=True,
            )

    # Figures-visual share the same multi-vector schema as pages, so we
    # accept PageRow here too. Kept as a separate method name for clarity
    # in callers.
    def upsert_figures_visual(self, name: str, rows: List["PageRow"], batch: int = 32) -> None:
        self.upsert_pages(name, rows, batch=batch)

    # ----- search -----------------------------------------------------------
    # Uses the modern `query_points()` API (qdrant-client >= 1.10). The
    # legacy `.search()` was removed in newer client versions.

    def search_chunks_hybrid(
        self,
        name: str,
        dense: np.ndarray,
        sparse: Dict[int, float],
        top_k: int = 30,
    ):
        """Returns merged top-k via Reciprocal Rank Fusion of dense+sparse."""
        from qdrant_client.http import models as qm

        dense_resp = self._client.query_points(
            collection_name=name,
            query=dense.tolist(),
            using="dense",
            limit=top_k,
            with_payload=True,
        )
        dense_hits = dense_resp.points

        sparse_hits = []
        if sparse:
            sparse_resp = self._client.query_points(
                collection_name=name,
                query=qm.SparseVector(
                    indices=list(sparse.keys()),
                    values=list(sparse.values()),
                ),
                using="sparse",
                limit=top_k,
                with_payload=True,
            )
            sparse_hits = sparse_resp.points

        return _rrf_merge(dense_hits, sparse_hits, top_k=top_k)

    def search_figures(
        self,
        name: str,
        dense: np.ndarray,
        top_k: int = 8,
    ):
        resp = self._client.query_points(
            collection_name=name,
            query=dense.tolist(),
            using="dense",
            limit=top_k,
            with_payload=True,
        )
        return resp.points

    def search_pages(
        self,
        name: str,
        multivec_query: np.ndarray,
        top_k: int = 6,
    ):
        # Multi-vector query: 2D list (num_query_tokens × patch_dim).
        resp = self._client.query_points(
            collection_name=name,
            query=multivec_query.tolist(),
            using="colbert",
            limit=top_k,
            with_payload=True,
        )
        return resp.points

    def search_figures_visual(
        self,
        name: str,
        multivec_query: np.ndarray,
        top_k: int = 6,
    ):
        """ColPali multi-vector search over figure crops.

        Identical shape to search_pages — figures-visual uses the same
        multi-vector schema. If the collection doesn't exist (e.g.
        ingestion was done with an older script that didn't populate it),
        returns [] rather than raising, so the rest of retrieval can
        proceed without this signal.
        """
        try:
            if not self._client.collection_exists(name):
                return []
        except Exception:
            pass
        try:
            resp = self._client.query_points(
                collection_name=name,
                query=multivec_query.tolist(),
                using="colbert",
                limit=top_k,
                with_payload=True,
            )
            return resp.points
        except Exception as e:
            log.warning("search_figures_visual failed (%r); skipping visual figure path", e)
            return []


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #

def _rrf_merge(*result_lists, top_k: int = 30, k_rrf: int = 60) -> List[Any]:
    """Reciprocal Rank Fusion of multiple ScoredPoint lists."""
    scores: Dict[int, float] = {}
    payloads: Dict[int, Any] = {}
    for hits in result_lists:
        for rank, hit in enumerate(hits):
            pid = hit.id
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k_rrf + rank + 1)
            if pid not in payloads:
                payloads[pid] = hit
    ordered = sorted(payloads.items(), key=lambda kv: scores[kv[0]], reverse=True)
    out = []
    for pid, hit in ordered[:top_k]:
        # Attach fused score onto the hit for downstream
        hit_dict = {"id": pid, "score": scores[pid], "payload": hit.payload}
        out.append(hit_dict)
    return out


def chunk_id_to_int(chunk_id: str) -> int:
    """Stable positive 63-bit integer id from a chunk_id string."""
    import hashlib
    h = hashlib.sha1(chunk_id.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False) >> 1
