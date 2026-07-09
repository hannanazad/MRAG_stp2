"""MUTCD knowledge graph v2: deterministic, NetworkX-backed.

Changes vs v1
-------------
1. FIGURE ANCHORING — v1 linked figures to sections only by page co-location
   (first chunk on the same PDF page), which routinely attached a figure to
   the wrong section. v2 uses three evidence tiers, strongest first:
     T1 anchored_in     The section whose chunks CITE the figure ("see
                        Figure X-Y"). All citing sections get cited_in
                        edges; the anchor prefers a citing section in the
                        figure's own chapter, then the most-citing one.
     T2 belongs_to_chapter  The figure id encodes its chapter (2B-14 -> 2B):
                        layout-independent, always present.
     T3 on_page_of      v1's page co-location, kept only as a weak edge.
2. ONE NODE PER CANONICAL ID — multi-sheet figures (e.g. Figure 9E-12,
   Sheets 1–2) collapse into a single node carrying image_paths=(...); v1
   made the last sheet win and dropped the rest.
3. PREFIX-TOLERANT LOOKUPS — KG.figure() accepts "2B-14", "Figure 2B-14",
   and "figure:Figure 2B-14"; the prefix mismatch that broke the earlier
   gold-figure diagnostic cannot recur.
4. UNRESOLVED-REF ACCOUNTING — refs to entities that never materialised are
   counted in g.graph["build_stats"] and excluded from query results.
5. ROUTER SUPPORT — Section nodes carry cites_any_figure; KG exposes
   section_cites_figures() so the question router can use a KG prior for
   the "does this query need figures at all?" decision.

Node id forms                                   Edge labels
  part:Part 2                                   contains
  chapter:Chapter 2B. Regulatory Signs...       anchored_in    (Figure->Section)
  section:2B.04                                 cited_in       (Figure->Section)
  chunk:MUTCD11e_2B04_Standard_01               belongs_to_chapter (Figure->Chapter)
  figure:Figure 2B-14  /  figure:Table 2B-1     on_page_of     (Figure->Section, weak)
  signcode:R1-1                                 cites_figure / cites_table / cites_section
  category:Regulatory                           defines / mentions / depicts / kind_of
"""
from __future__ import annotations

import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import networkx as nx

from .parsing import Chunk
from .figures import FigureRecord
from .figure_core import chapter_of
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
    g.graph["schema_version"] = 2

    # ---- 1. hierarchy: Part / Chapter / Section / Chunk ---------------------
    for c in chunks:
        part_id = (c.part or "Part ?").strip()
        chap_id = (c.chapter or "Chapter ?").strip()
        part_node, chap_node = f"part:{part_id}", f"chapter:{chap_id}"
        sec_node, chunk_node = f"section:{c.section_id}", f"chunk:{c.chunk_id}"

        if not g.has_node(part_node):
            g.add_node(part_node, kind="Part", title=part_id)
        if not g.has_node(chap_node):
            g.add_node(chap_node, kind="Chapter", title=chap_id, part=part_id)
            g.add_edge(part_node, chap_node, label="contains")
        if not g.has_node(sec_node):
            g.add_node(sec_node, kind="Section", id=c.section_id,
                       title=c.section_title, page_pdf=c.page_pdf,
                       page_printed=c.page_printed, chapter=chap_id,
                       part=part_id, cites_any_figure=False)
            g.add_edge(chap_node, sec_node, label="contains")
        g.add_node(chunk_node, kind="Chunk", id=c.chunk_id, section=c.section_id,
                   content_type=c.content_type, ordinal=c.ordinal,
                   page_pdf=c.page_pdf, page_printed=c.page_printed,
                   modal_verbs=tuple(c.modal_verbs))
        g.add_edge(sec_node, chunk_node, label="contains")

        def _ref(target: str, default_kind: str, label: str):
            if not g.has_node(target):
                g.add_node(target, kind=default_kind, unresolved=True)
            g.add_edge(chunk_node, target, label=label)

        for fid in c.figure_refs:
            _ref(f"figure:Figure {fid}", "FigureRef", "cites_figure")
        for tid in c.table_refs:
            _ref(f"figure:Table {tid}", "TableRef", "cites_table")
        for sid in c.section_refs:
            _ref(f"section:{sid}", "SectionRef", "cites_section")
        for sc in c.sign_codes:
            _ref(f"signcode:{sc.upper()}", "SignCodeRef", "mentions")
        if c.figure_refs or c.table_refs:
            g.nodes[sec_node]["cites_any_figure"] = True

    # ---- 2. figures: ONE node per canonical id, sheets collapsed ------------
    by_canonical: Dict[str, List[FigureRecord]] = defaultdict(list)
    for f in figures:
        by_canonical[f.figure_id].append(f)

    page_to_section: Dict[int, str] = {}
    for c in chunks:
        page_to_section.setdefault(c.page_pdf, c.section_id)

    for figure_id, recs in by_canonical.items():
        recs = sorted(recs, key=lambda r: ((r.sheet or 1), r.page_pdf))
        head = recs[0]
        node = f"figure:{figure_id}"
        chapter_short = head.chapter or chapter_of(head.canonical_id)
        g.add_node(node, kind=head.kind, id=figure_id,
                   canonical_id=head.canonical_id,
                   chapter=chapter_short,
                   page_pdf=head.page_pdf, page_printed=head.page_printed,
                   caption=head.caption, title=head.title,
                   image_path=head.image_path,                      # back-compat
                   image_paths=tuple(r.image_path for r in recs),   # all sheets
                   n_sheets=len(recs),
                   sign_codes=tuple(head.sign_codes_depicted),
                   extraction_method=head.extraction_method,
                   unresolved=False)
        # T2: chapter edge straight from the id — layout-independent
        chap_node = _chapter_node(g, chapter_short)
        if chap_node:
            g.add_edge(node, chap_node, label="belongs_to_chapter")
        # T3: weak page co-location
        sec_id = page_to_section.get(head.page_pdf)
        if sec_id:
            g.add_edge(node, f"section:{sec_id}", label="on_page_of")
        for sc in head.sign_codes_depicted:
            g.add_edge(node, f"signcode:{sc.upper()}", label="depicts")

    # ---- 3. T1 citation-based anchoring --------------------------------------
    cite_counts: Dict[str, Counter] = defaultdict(Counter)
    for c in chunks:
        for fid in c.figure_refs:
            cite_counts[f"figure:Figure {fid}"][c.section_id] += 1
        for tid in c.table_refs:
            cite_counts[f"figure:Table {tid}"][c.section_id] += 1
    for fnode, secs in cite_counts.items():
        if not g.has_node(fnode) or g.nodes[fnode].get("unresolved"):
            continue
        for sec_id in secs:
            g.add_edge(fnode, f"section:{sec_id}", label="cited_in")
        anchor = _pick_anchor(g.nodes[fnode].get("chapter", ""), secs)
        g.add_edge(fnode, f"section:{anchor}", label="anchored_in")
        g.nodes[fnode]["anchor_section"] = anchor

    # ---- 4. sign codes & categories -------------------------------------------
    for code, entry in sign_codes.items():
        node = f"signcode:{code}"
        g.add_node(node, kind="SignCode", id=code, category=entry.category,
                   canonical_name=entry.canonical_name, unresolved=False)
        cat_node = f"category:{entry.category}"
        if not g.has_node(cat_node):
            g.add_node(cat_node, kind="Category", id=entry.category)
        g.add_edge(node, cat_node, label="kind_of")
        if entry.first_seen_section:
            g.add_edge(f"section:{entry.first_seen_section}", node, label="defines")

    # ---- 5. figure sign-code backfill (citing chunks + same-page chunks) -----
    page_to_chunk_nodes: Dict[int, List[str]] = defaultdict(list)
    for c in chunks:
        page_to_chunk_nodes[c.page_pdf].append(f"chunk:{c.chunk_id}")
    chunk_mentions: Dict[str, Set[str]] = defaultdict(set)
    for u, v, d in g.edges(data=True):
        if d.get("label") == "mentions" and u.startswith("chunk:"):
            chunk_mentions[u].add(v.split(":", 1)[1])
    for fnode in [n for n, d in g.nodes(data=True)
                  if n.startswith("figure:") and not d.get("unresolved")]:
        cand: Set[str] = set(g.nodes[fnode].get("sign_codes", ()))
        for u, _v, d in g.in_edges(fnode, data=True):
            if d.get("label") in ("cites_figure", "cites_table"):
                cand |= chunk_mentions.get(u, set())
        for cn in page_to_chunk_nodes.get(g.nodes[fnode].get("page_pdf"), []):
            cand |= chunk_mentions.get(cn, set())
        g.nodes[fnode]["sign_codes"] = tuple(sorted(cand))
        for sc in cand:
            g.add_edge(fnode, f"signcode:{sc.upper()}", label="depicts")

    # ---- 6. unresolved-ref accounting ------------------------------------------
    unresolved = [n for n, d in g.nodes(data=True) if d.get("unresolved")]
    g.graph["build_stats"] = {
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "n_figures": sum(1 for n, d in g.nodes(data=True)
                         if n.startswith("figure:") and not d.get("unresolved")),
        "n_unresolved_refs": len(unresolved),
        "unresolved_sample": sorted(unresolved)[:20],
        "n_anchored_figures": sum(1 for _n, d in g.nodes(data=True)
                                  if d.get("anchor_section")),
    }
    return g


def _chapter_node(g: nx.MultiDiGraph, chapter_short: str) -> Optional[str]:
    """'2B' -> the 'chapter:Chapter 2B. …' node created during chunk parsing."""
    if not chapter_short:
        return None
    pat = re.compile(rf"^chapter:Chapter\s+{re.escape(chapter_short)}\b",
                     re.IGNORECASE)
    for n in g.nodes:
        if n.startswith("chapter:") and pat.match(n):
            return n
    return None


def _pick_anchor(chapter_short: str, secs: Counter) -> str:
    """Most-citing section, preferring sections in the figure's own chapter
    (Section 2B.14 wins over 6F.03 for Figure 2B-*)."""
    in_chap = {s: n for s, n in secs.items()
               if chapter_short and s.upper().startswith(chapter_short.upper() + ".")}
    pool = in_chap or dict(secs)
    return max(pool.items(), key=lambda kv: kv[1])[0]


# --------------------------------------------------------------------------- #
# Persistence                                                                 #
# --------------------------------------------------------------------------- #

def write(g: nx.MultiDiGraph, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(g, f)


def read(path: Path) -> nx.MultiDiGraph:
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# Query wrapper                                                               #
# --------------------------------------------------------------------------- #

class KG:
    def __init__(self, g: nx.MultiDiGraph) -> None:
        self.g = g
        self._signs = {n.split(":", 1)[1]: n for n in g.nodes
                       if n.startswith("signcode:")}
        self._figures: Dict[str, str] = {}
        for n, d in g.nodes(data=True):
            if not n.startswith("figure:") or d.get("unresolved"):
                continue
            full = n.split(":", 1)[1]                  # "Figure 2B-14"
            self._figures[full.upper()] = n
            cid = d.get("canonical_id")
            if cid:                                     # "2B-14" also resolves
                self._figures.setdefault(cid.upper(), n)

    # ----- lookups (prefix-tolerant — fixes the v1 diagnostic bug) -----------

    def sign(self, code: str) -> Optional[str]:
        return self._signs.get(code.upper())

    def figure(self, fid: str) -> Optional[str]:
        f = fid.strip()
        if f.lower().startswith("figure:"):
            f = f.split(":", 1)[1]
        return self._figures.get(f.upper())

    def section(self, sec_id: str) -> Optional[str]:
        n = f"section:{sec_id}"
        return n if self.g.has_node(n) else None

    # ----- traversal ----------------------------------------------------------

    def neighbors(self, node: str, n_hops: int = 1) -> Set[str]:
        if not self.g.has_node(node):
            return set()
        out, frontier = {node}, {node}
        ug = self.g.to_undirected(as_view=True)
        for _ in range(n_hops):
            nxt: Set[str] = set()
            for n in frontier:
                nxt.update(ug.neighbors(n))
            out |= nxt
            frontier = nxt
        return out

    def figures_for_chunk(self, chunk_id: str) -> List[str]:
        node = f"chunk:{chunk_id}"
        if not self.g.has_node(node):
            return []
        out = []
        for _u, v, d in self.g.out_edges(node, data=True):
            if d.get("label") in ("cites_figure", "cites_table"):
                vd = self.g.nodes.get(v, {})
                if not vd.get("unresolved"):
                    out.append(v.split(":", 1)[1])
        return out

    def figures_for_section(self, section_id: str) -> List[str]:
        """Figures anchored in (strong) then cited in (medium) this section."""
        node = f"section:{section_id}"
        if not self.g.has_node(node):
            return []
        ranked: List[tuple] = []
        for u, _v, d in self.g.in_edges(node, data=True):
            if not u.startswith("figure:") or self.g.nodes[u].get("unresolved"):
                continue
            lbl = d.get("label")
            if lbl == "anchored_in":
                ranked.append((0, u))
            elif lbl == "cited_in":
                ranked.append((1, u))
        ranked.sort()
        seen, out = set(), []
        for _r, u in ranked:
            fid = u.split(":", 1)[1]
            if fid not in seen:
                seen.add(fid)
                out.append(fid)
        return out

    def section_cites_figures(self, section_id: str) -> bool:
        n = f"section:{section_id}"
        return bool(self.g.has_node(n)
                    and self.g.nodes[n].get("cites_any_figure"))

    def chunks_for_signcode(self, code: str) -> List[str]:
        node = self.sign(code)
        if not node:
            return []
        ug = self.g.to_undirected(as_view=True)
        return [n.split(":", 1)[1] for n in ug.neighbors(node)
                if n.startswith("chunk:")]

    # ----- query-time ----------------------------------------------------------

    def query_entities(self, query: str) -> Set[str]:
        ents: Set[str] = set()
        q = query.translate({0x2010: "-", 0x2011: "-", 0x2012: "-",
                             0x2013: "-", 0x2014: "-"})
        for m in re.finditer(r"\b(Figure|Table)\s+([0-9A-Z]+-[0-9]+[A-Za-z0-9]*)\b",
                             q, re.IGNORECASE):
            n = self.figure(f"{m.group(1).title()} {m.group(2).upper()}")
            if n:
                ents.add(n)
        for m in re.finditer(r"\b(?:Section\s+)?([0-9]+[A-Z]\.[0-9]+)\b", q):
            n = self.section(m.group(1))
            if n:
                ents.add(n)
        for m in re.finditer(r"\b([RWDIMOE][A-Z]?\d{1,3}(?:-\d{1,3})?[a-zA-Z]?P?)\b",
                             q):
            n = self.sign(m.group(1))
            if n:
                ents.add(n)
        return ents

    def proximity_score(self, query_ents: Set[str], chunk_id: str) -> float:
        if not query_ents:
            return 0.0
        node = f"chunk:{chunk_id}"
        if not self.g.has_node(node):
            return 0.0
        ug = self.g.to_undirected(as_view=True)
        best = float("inf")
        for qn in query_ents:
            if not ug.has_node(qn):
                continue
            try:
                best = min(best, nx.shortest_path_length(ug, qn, node))
            except nx.NetworkXNoPath:
                continue
        return 1.0 / (1.0 + best) if best != float("inf") else 0.0

    # ----- citation validation ---------------------------------------------------

    def is_known_citation(self, kind: str, idval: str) -> bool:
        kind = kind.lower()
        if kind == "section":
            return self.section(idval) is not None
        if kind in ("figure", "table"):
            return self.figure(f"{kind.title()} {idval.upper()}") is not None
        if kind in ("signcode", "sign"):
            return self.sign(idval) is not None
        return False
