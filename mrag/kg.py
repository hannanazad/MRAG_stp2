"""MUTCD knowledge graph: deterministic, lightweight, NetworkX-backed.

Schema (node types):
    Part            id=part name           attrs: title
    Chapter         id=chapter name        attrs: title, part
    Section         id="<sec_id>"           attrs: title, page, chapter, part
    Chunk           id=<chunk_id>           attrs: section, content_type, ordinal, page
    Figure          id=<figure_id>          attrs: page, caption, image_path
    Table           id=<figure_id>          attrs: page, caption, image_path
    SignCode        id=<code>               attrs: category, canonical_name
    Category        id=<category name>      attrs: -

Edge labels:
    contains         (Part->Chapter, Chapter->Section, Section->Chunk)
    illustrated_by   (Section->Figure/Table; same-page co-location)
    cites_section    (Chunk->Section)
    cites_figure     (Chunk->Figure)
    cites_table      (Chunk->Table)
    defines          (Section->SignCode)
    mentions         (Chunk->SignCode)
    depicts          (Figure/Table->SignCode)
    kind_of          (SignCode->Category)

Query API (used by retrieval):
    graph_distance(query_entities, chunk_id) -> int
    neighbors_of(node_id, n_hops=1) -> set[node_id]
    figures_for_chunk(chunk_id) -> List[figure_id]
    chunks_for_signcode(code) -> List[chunk_id]
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx

from .parsing import Chunk
from .figures import FigureRecord
from .sign_codes import SignCodeEntry


# --------------------------------------------------------------------------- #
# Build                                                                       #
# --------------------------------------------------------------------------- #

def build(
    chunks: List[Chunk],
    figures: List[FigureRecord],
    sign_codes: Dict[str, SignCodeEntry],
) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()

    # --- Parts / Chapters / Sections / Chunks (the hierarchy) ----------------
    for c in chunks:
        part_id    = (c.part or "Part ?").strip()
        chap_id    = (c.chapter or "Chapter ?").strip()
        sec_node   = f"section:{c.section_id}"
        chunk_node = f"chunk:{c.chunk_id}"
        part_node  = f"part:{part_id}"
        chap_node  = f"chapter:{chap_id}"

        if not g.has_node(part_node):
            g.add_node(part_node, kind="Part", title=part_id)
        if not g.has_node(chap_node):
            g.add_node(chap_node, kind="Chapter", title=chap_id, part=part_id)
            g.add_edge(part_node, chap_node, label="contains")
        if not g.has_node(sec_node):
            g.add_node(sec_node, kind="Section", id=c.section_id, title=c.section_title,
                       page_pdf=c.page_pdf, page_printed=c.page_printed,
                       chapter=chap_id, part=part_id)
            g.add_edge(chap_node, sec_node, label="contains")
        g.add_node(chunk_node, kind="Chunk", id=c.chunk_id, section=c.section_id,
                   content_type=c.content_type, ordinal=c.ordinal,
                   page_pdf=c.page_pdf, page_printed=c.page_printed,
                   modal_verbs=tuple(c.modal_verbs))
        g.add_edge(sec_node, chunk_node, label="contains")

        # --- chunk cross-references ----------------------------------------
        # NB: we may reference a Figure/Section/SignCode whose node doesn't
        # exist yet. Pre-create with an UnresolvedRef kind; the real loop
        # below will overwrite the `kind` attribute when the actual node is
        # registered. This keeps the graph free of nodes with kind=None.
        def _ref(target: str, default_kind: str, label: str):
            if not g.has_node(target):
                g.add_node(target, kind=default_kind, unresolved=True)
            g.add_edge(chunk_node, target, label=label)

        for fid in c.figure_refs:
            _ref(f"figure:Figure {fid}", default_kind="FigureRef",  label="cites_figure")
        for tid in c.table_refs:
            _ref(f"figure:Table {tid}",  default_kind="TableRef",   label="cites_table")
        for sid in c.section_refs:
            _ref(f"section:{sid}",       default_kind="SectionRef", label="cites_section")
        for sc in c.sign_codes:
            _ref(f"signcode:{sc.upper()}", default_kind="SignCodeRef", label="mentions")

    # --- Figures / Tables ----------------------------------------------------
    figure_by_id: Dict[str, FigureRecord] = {f.figure_id: f for f in figures}
    page_to_section: Dict[int, str] = {}
    for c in chunks:
        page_to_section.setdefault(c.page_pdf, c.section_id)
    for f in figures:
        node = f"figure:{f.figure_id}"
        # If a chunk referenced this figure earlier we may have a placeholder
        # node — replace its attrs with the real ones.
        g.add_node(node, kind=f.kind, id=f.figure_id, page_pdf=f.page_pdf,
                   page_printed=f.page_printed, caption=f.caption,
                   image_path=f.image_path, sign_codes=tuple(f.sign_codes_depicted),
                   unresolved=False)
        sec_id = page_to_section.get(f.page_pdf)
        if sec_id:
            g.add_edge(f"section:{sec_id}", node, label="illustrated_by")
        for sc in f.sign_codes_depicted:
            g.add_edge(node, f"signcode:{sc.upper()}", label="depicts")

    # --- Sign codes & categories --------------------------------------------
    for code, entry in sign_codes.items():
        node = f"signcode:{code}"
        # Overwrite any prior placeholder.
        g.add_node(node, kind="SignCode", id=code,
                   category=entry.category, canonical_name=entry.canonical_name,
                   unresolved=False)
        cat_node = f"category:{entry.category}"
        if not g.has_node(cat_node):
            g.add_node(cat_node, kind="Category", id=entry.category)
        g.add_edge(node, cat_node, label="kind_of")
        if entry.first_seen_section:
            g.add_edge(f"section:{entry.first_seen_section}", node, label="defines")

    # --- Backfill: sign_codes_depicted by figure via two evidence sources --
    #   1. chunks that explicitly cite the figure  (Chunk -mentions-> SignCode)
    #   2. chunks on the same printed page         (page co-location)
    page_to_chunks: Dict[int, List[str]] = {}
    for c in chunks:
        page_to_chunks.setdefault(c.page_pdf, []).append(c.chunk_id)
    for f in figures:
        fnode = f"figure:{f.figure_id}"
        if not g.has_node(fnode):
            continue
        cand = set(f.sign_codes_depicted or [])
        # Source A: chunks citing this figure
        for chunk_id in [d["id"] for _, d in g.nodes(data=True)
                         if d.get("kind") == "Chunk"]:
            cnode = f"chunk:{chunk_id}"
            if g.has_edge(cnode, fnode):
                # Check edges with label cites_figure / cites_table
                for _u, _v, ed in g.edges(cnode, data=True):
                    if _v == fnode and ed.get("label") in ("cites_figure", "cites_table"):
                        # Pull mentions from this chunk
                        for _u2, _v2, ed2 in g.edges(cnode, data=True):
                            if ed2.get("label") == "mentions":
                                cand.add(_v2.split(":", 1)[1])
                        break
        # Source B: chunks on the same page
        for chunk_id in page_to_chunks.get(f.page_pdf, []):
            cnode = f"chunk:{chunk_id}"
            for _u, _v, ed in g.edges(cnode, data=True):
                if ed.get("label") == "mentions":
                    cand.add(_v.split(":", 1)[1])

        cand = sorted(cand)
        f.sign_codes_depicted = cand  # also write back to the dataclass
        g.nodes[fnode]["sign_codes"] = tuple(cand)
        for sc in cand:
            g.add_edge(fnode, f"signcode:{sc.upper()}", label="depicts")

    return g


# --------------------------------------------------------------------------- #
# Persistence                                                                 #
# --------------------------------------------------------------------------- #

def write(g: nx.MultiDiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(g, f)


def read(path: Path) -> nx.MultiDiGraph:
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# Query helpers                                                               #
# --------------------------------------------------------------------------- #

class KG:
    """Convenience wrapper around the MultiDiGraph with task-specific queries."""

    def __init__(self, g: nx.MultiDiGraph) -> None:
        self.g = g
        # Precompute index: signcode -> node id, figure_id -> node id, etc.
        self._sigcodes = {n.split(":", 1)[1]: n
                         for n in g.nodes if n.startswith("signcode:")}
        self._figures = {n.split(":", 1)[1]: n
                         for n in g.nodes if n.startswith("figure:")}

    # ----- lookup helpers ---------------------------------------------------

    def sign(self, code: str) -> Optional[str]:
        return self._sigcodes.get(code.upper())

    def figure(self, fid: str) -> Optional[str]:
        return self._figures.get(fid)

    def section(self, sec_id: str) -> Optional[str]:
        n = f"section:{sec_id}"
        return n if self.g.has_node(n) else None

    # ----- traversal -------------------------------------------------------

    def neighbors(self, node: str, n_hops: int = 1) -> Set[str]:
        if not self.g.has_node(node):
            return set()
        out: Set[str] = {node}
        frontier = {node}
        ug = self.g.to_undirected(as_view=True)
        for _ in range(n_hops):
            nxt: Set[str] = set()
            for n in frontier:
                nxt.update(ug.neighbors(n))
            out.update(nxt); frontier = nxt
        return out

    def figures_for_chunk(self, chunk_id: str) -> List[str]:
        node = f"chunk:{chunk_id}"
        if not self.g.has_node(node):
            return []
        out = []
        for _u, v, d in self.g.out_edges(node, data=True):
            if d.get("label") in ("cites_figure", "cites_table"):
                if self.g.has_node(v):
                    out.append(v.split(":", 1)[1])
        return out

    def figures_for_section(self, section_id: str) -> List[str]:
        node = f"section:{section_id}"
        if not self.g.has_node(node):
            return []
        out = []
        for _u, v, d in self.g.out_edges(node, data=True):
            if d.get("label") == "illustrated_by":
                out.append(v.split(":", 1)[1])
        return out

    def chunks_for_signcode(self, code: str) -> List[str]:
        node = self.sign(code)
        if not node:
            return []
        out = []
        ug = self.g.to_undirected(as_view=True)
        for n in ug.neighbors(node):
            if n.startswith("chunk:"):
                out.append(n.split(":", 1)[1])
        return out

    # ----- query-time scoring ----------------------------------------------

    def query_entities(self, query: str) -> Set[str]:
        """Return the set of graph nodes mentioned/implied by the query."""
        ents: Set[str] = set()
        # Figure / Table mentions
        for m in re.finditer(r"\b(Figure|Table)\s+([0-9A-Z]+-[0-9]+[A-Za-z0-9]*)\b", query, re.IGNORECASE):
            fid = f"{m.group(1).title()} {m.group(2).upper()}"
            n = self.figure(fid)
            if n: ents.add(n)
        # Section mentions
        for m in re.finditer(r"\b(?:Section\s+)?([0-9]+[A-Z]\.[0-9]+)\b", query):
            n = self.section(m.group(1))
            if n: ents.add(n)
        # Sign codes
        for m in re.finditer(
            r"\b([RWDIMOE][A-Z]?\d{1,3}(?:-\d{1,3})?[a-zA-Z]?P?)\b", query
        ):
            n = self.sign(m.group(1))
            if n: ents.add(n)
        return ents

    def proximity_score(self, query_ents: Set[str], chunk_id: str) -> float:
        """1/(1+min_hops). Capped at 1.0 (exact mention)."""
        if not query_ents:
            return 0.0
        node = f"chunk:{chunk_id}"
        if not self.g.has_node(node):
            return 0.0
        ug = self.g.to_undirected(as_view=True)
        best = float("inf")
        for q in query_ents:
            if not ug.has_node(q):
                continue
            try:
                d = nx.shortest_path_length(ug, source=q, target=node)
            except nx.NetworkXNoPath:
                continue
            best = min(best, d)
        return 1.0 / (1.0 + best) if best != float("inf") else 0.0

    # ----- citation validation ---------------------------------------------

    def is_known_citation(self, kind: str, idval: str) -> bool:
        """Return True if `kind:idval` is a real node in the graph."""
        kind = kind.lower()
        if kind in ("section",):
            return self.section(idval) is not None
        if kind in ("figure", "table"):
            return self.figure(f"{kind.title()} {idval.upper()}") is not None
        if kind in ("signcode", "sign"):
            return self.sign(idval) is not None
        return False
