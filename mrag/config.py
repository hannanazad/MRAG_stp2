"""All tunable knobs and paths in one place.

Read by every other module via `from mrag.config import CFG`.

Auto-detects environment (Colab / HPRC / local) and picks the right
base directory accordingly. Override with the env var `MRAG_BASE_DIR`.

Environment detection also respects `MRAG_ENV` if explicitly set. This
matters because `scripts/ingest_v3.py` runs as a SEPARATE subprocess
(via `!python scripts/ingest_v3.py` in the notebook) which never imports
`google.colab` itself — without MRAG_ENV, that subprocess would wrongly
detect "local" instead of "colab", causing it to build the Qdrant DB at
a different path than the notebook kernel expects. Set MRAG_ENV="colab"
once, early, in your Colab setup cell, and every subprocess inherits it.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

def detect_environment() -> str:
    """Returns one of: 'colab', 'hprc', 'local'.

    Checks MRAG_ENV first (explicit, inherited by subprocesses) so that
    scripts launched via `!python scripts/ingest_v3.py` detect the same
    environment as the parent notebook kernel. Falls back to the
    sys.modules check for direct in-kernel imports.
    """
    env_override = os.environ.get("MRAG_ENV")
    if env_override in ("colab", "hprc", "local"):
        return env_override
    if "google.colab" in sys.modules:
        return "colab"
    if os.environ.get("SCRATCH") and Path(os.environ["SCRATCH"]).exists():
        return "hprc"
    return "local"

def _default_base_dir(env: str) -> Path:
    if env == "colab":
        # Drive mount is required; this path exists only after drive.mount(...)
        return Path("/content/drive/MyDrive/MRAG")
    if env == "hprc":
        return Path(os.environ["SCRATCH"]) / "MRAG"
    return Path.cwd() / "MRAG"

def _default_cache_dir(env: str, base: Path) -> Path:
    """Where Qdrant + temp embeddings live. Local disk on Colab for speed."""
    if env == "colab":
        return Path("/content") / "qdrant_db"
    return base / "qdrant_db"

def _default_hf_home(env: str, base: Path) -> Path:
    if env == "colab":
        # Drive HF cache survives session restarts.
        return base / "hf_cache"
    if env == "hprc":
        return Path(os.environ["SCRATCH"]) / "hf_cache"
    return base / "hf_cache"


# =============================================================================
# Curated catalog of DashScope models you can swap to from the notebook.
#
# Switch by calling CFG.set_vlm_model("<alias>") or CFG.set_vlm_model("<raw id>").
# `CFG.vlm_model_api` is read at call time by mrag.vlm._answer_api and
# _filter_figures_api, so the next ask() picks up the change with no
# init_pipeline() restart.
#
# IMPORTANT: glm-5.2 and kimi-k2.7-code are TEXT-ONLY on this DashScope
# account (they aren't in the GET /models 'vl' list). If you switch to
# them, the figure-filter call AND the answer call WILL fail with
# 400 InvalidParameter the moment they receive images. Use them only
# for a text-only baseline (you'd need to call them outside ask()).
# Left in the catalog because they were requested for comparison.
# =============================================================================
VLM_API_MODELS: dict[str, str] = {
    # ─── Qwen3-VL flagship (vision-language) ─────────────────────────────
    "flagship":          "qwen3-vl-235b-a22b-instruct",
    "flagship_thinking": "qwen3-vl-235b-a22b-thinking",
    # ─── Qwen3-VL plus tier ──────────────────────────────────────────────
    "plus_pinned":       "qwen3-vl-plus-2025-12-19",
    "plus":              "qwen3-vl-plus",
    # ─── Qwen-VL max (older top tier) ────────────────────────────────────
    "max":               "qwen-vl-max",
    # ─── Qwen3-VL flash (cheap + fast) ───────────────────────────────────
    "flash_pinned":      "qwen3-vl-flash-2026-01-22",
    "flash":             "qwen3-vl-flash",
    # ─── Qwen-VL plus legacy (Qwen2.x era) ───────────────────────────────
    "plus_legacy":       "qwen-vl-plus",
    # ─── Non-VL comparison points (TEXT ONLY — see warning above) ────────
    "glm":               "glm-5.2",
    "kimi":              "kimi-k2.7-code",
}


@dataclass
class Config:
    # ----- Paths ------------------------------------------------------------
    scratch: Path = field(default_factory=lambda: Path(os.environ.get("SCRATCH", "/tmp")))
    base_dir: Path = field(init=False)
    pdf_path: Path = field(init=False)

    figures_dir: Path = field(init=False)
    page_images_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)
    qdrant_dir: Path = field(init=False)

    chunks_jsonl: Path = field(init=False)
    figures_jsonl: Path = field(init=False)
    sign_codes_json: Path = field(init=False)
    graph_pickle: Path = field(init=False)

    hf_home: Path = field(init=False)

    # ----- Models -----------------------------------------------------------
    bge_m3_model: str = "BAAI/bge-m3"
    colqwen_model: str = "vidore/colqwen2-v0.1"
    reranker_model: str = "mixedbread-ai/mxbai-rerank-large-v2"
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    vlm_model_fallback: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    # ----- VLM provider -------------------------------------------------------
    # "local" → load vlm_model weights onto local GPU via transformers (original
    # behaviour, unchanged).
    # "api" → call an OpenAI-compatible REST endpoint instead. No GPU,
    # no local download. Set vlm_model_api / api_base_url below.
    vlm_provider: str = "api"  # "api" or "local" — default is api (Qwen3-VL via DashScope)

    # Model name string sent to the API endpoint when vlm_provider == "api".
    # Default is qwen3-vl-plus-2025-12-19 — a Qwen3-VL "plus" tier dated
    # snapshot known to be entitled on this account. Switch on demand from
    # the notebook with CFG.set_vlm_model("<alias>") — see VLM_API_MODELS
    # above for the curated catalog (flagship / flash / max / etc.).
    vlm_model_api: str = "qwen3-vl-plus-2025-12-19"

    # OpenAI-compatible API endpoint. INTERNATIONAL DashScope URL — use this
    # unless your Alibaba Cloud account is registered in mainland China.
    api_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    # Name of the environment variable holding the API key.
    api_key_env_var: str = "VLM_API_KEY"

    # ----- Rendering --------------------------------------------------------
    page_dpi: int = 180
    figure_dpi: int = 220

    # ----- Qdrant collection names -----------------------------------------
    coll_chunks: str = "mutcd_chunks"
    coll_figures: str = "mutcd_figures"              # caption-embeddings (text)
    coll_figures_visual: str = "mutcd_figures_visual"  # ColPali on figure crops
    coll_pages: str = "mutcd_pages"

    # ----- Retrieval --------------------------------------------------------
    top_k_dense: int = 30
    top_k_sparse: int = 30
    top_k_fused: int = 30
    top_k_after_graph: int = 40
    top_k_after_rerank: int = 6

    # Figure retrieval works in stages now:
    #   1. Path A — figures CITED by winning chunks (KG cross-links). High
    #      precision; count is whatever the winners cite.
    #   2. Path C — VISUAL retrieval via ColPali over figure crops (NEW).
    #      Recovers figures that are visually relevant even when no chunk
    #      explicitly cites them.
    #   3. Path B — caption-text retrieval, kept as a fallback only. This
    #      was the main source of off-topic figures in the previous design,
    #      so it's now off by default. Set use_caption_figure_fallback=True
    #      to re-enable.
    #   4. (Optional) VLM filter that picks the visually-relevant subset.
    #
    # top_k_figures_candidates: how many figures to gather across paths A+B+C
    # before any filtering. The VLM filter (if enabled) sees this many.
    # top_k_figures: how many to actually display / pass to the answer-VLM
    # after filtering. Was 6 (display only) in the old design.
    top_k_figures_candidates: int = 10
    top_k_figures: int = 4
    top_k_figures_visual: int = 6
    use_caption_figure_fallback: bool = False
    use_vlm_figure_filter: bool = True

    top_k_pages: int = 4

    # Scoring weights:
    # S = α·dense + β·sparse + γ·hierarchy + δ·graph + ε·rule_type
    w_dense: float = 1.00
    w_sparse: float = 0.60
    w_hierarchy: float = 0.20
    w_graph: float = 0.40
    w_ruletype: float = 0.30

    # Rule-type multipliers (modal-verb backbone of MUTCD).
    rt_weight_standard: float = 1.20
    rt_weight_guidance: float = 1.00
    rt_weight_option: float = 0.90
    rt_weight_support: float = 0.70

    # ----- Generation -------------------------------------------------------
    max_new_tokens: int = 480
    max_chunk_chars_in_prompt: int = 1400

    # ----- ColPali ----------------------------------------------------------
    colqwen_max_image_patches: int = 768
    colqwen_use_binary_quantization: bool = True

    # ----- Misc -------------------------------------------------------------
    log_level: str = "INFO"

    environment: str = field(init=False)

    def __post_init__(self) -> None:
        self.environment = detect_environment()
        env_base = os.environ.get("MRAG_BASE_DIR")
        self.base_dir = Path(env_base) if env_base else _default_base_dir(self.environment)
        self.pdf_path = self.base_dir / "mutcd11theditionr1hl.pdf"
        # If no pdf at the default name, look for any *.pdf in BASE_DIR.
        if not self.pdf_path.exists():
            pdfs = sorted(self.base_dir.glob("*.pdf"))
            if pdfs:
                self.pdf_path = pdfs[0]

        self.figures_dir = self.base_dir / "figures"
        self.page_images_dir = self.base_dir / "page_images"
        self.cache_dir = self.base_dir / "mmrag_cache_v3"
        # Qdrant on Colab lives on local /content for speed; we sync to/from Drive.
        self.qdrant_dir = _default_cache_dir(self.environment, self.base_dir)

        self.chunks_jsonl = self.cache_dir / "chunks.jsonl"
        self.figures_jsonl = self.cache_dir / "figures.jsonl"
        self.sign_codes_json = self.cache_dir / "sign_codes.json"
        self.graph_pickle = self.cache_dir / "graph.gpickle"

        self.hf_home = _default_hf_home(self.environment, self.base_dir)
        os.environ.setdefault("HF_HOME", str(self.hf_home))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(self.hf_home))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(self.hf_home))

        # On Colab the base_dir doesn't exist until Drive is mounted — guard the mkdir.
        for d in (self.figures_dir, self.page_images_dir, self.cache_dir,
                  self.qdrant_dir, self.hf_home):
            try:
                d.mkdir(parents=True, exist_ok=True)
            except (PermissionError, OSError):
                # Drive may not be mounted yet; user will rerun config later.
                pass

    def rule_type_weight(self, ct: str) -> float:
        return {
            "Standard": self.rt_weight_standard,
            "Guidance": self.rt_weight_guidance,
            "Option": self.rt_weight_option,
            "Support": self.rt_weight_support,
        }.get(ct, 1.0)

    # ─── VLM model switcher ────────────────────────────────────────────────
    # Update CFG.vlm_model_api in-place. mrag.vlm._answer_api and
    # _filter_figures_api read CFG.vlm_model_api at call time, so the next
    # ask() picks up the new model without any init_pipeline() restart.

    def set_vlm_model(self, alias_or_id: str, verbose: bool = True) -> str:
        """Switch the DashScope model used by the API path.

        Accepts either a curated alias (a key of VLM_API_MODELS) or a raw
        DashScope model ID string. Returns the resolved model ID.

        Raises ValueError if the input is neither a known alias nor a
        plausible DashScope model ID (basic safety check).
        """
        alias_or_id = alias_or_id.strip()
        if alias_or_id in VLM_API_MODELS:
            resolved = VLM_API_MODELS[alias_or_id]
            via = f"alias '{alias_or_id}'"
        else:
            # Treat as raw ID. Sanity-check it isn't obviously bogus.
            if not alias_or_id or " " in alias_or_id:
                raise ValueError(
                    f"{alias_or_id!r} is neither a known alias nor a valid "
                    f"DashScope model id. Aliases: {list(VLM_API_MODELS)}"
                )
            resolved = alias_or_id
            via = "raw id"
        prev = self.vlm_model_api
        self.vlm_model_api = resolved
        if verbose:
            print(f"[CFG] vlm_model_api: {prev!r} -> {resolved!r}  (via {via})")
            if alias_or_id in ("glm", "kimi") or resolved in ("glm-5.2",
                                                              "kimi-k2.7-code"):
                print(f"[CFG] WARNING: {resolved!r} is text-only on this "
                      "account. ask() will fail when it sends images.")
        return resolved

    def list_vlm_models(self) -> None:
        """Pretty-print the curated catalog, marking which one is active."""
        current = self.vlm_model_api
        print(f"Curated DashScope models (current = {current!r}):\n")
        # Group: VL first, non-VL last
        vl_aliases   = [k for k, v in VLM_API_MODELS.items()
                        if k not in ("glm", "kimi")]
        text_aliases = [k for k in ("glm", "kimi") if k in VLM_API_MODELS]
        for k in vl_aliases:
            v = VLM_API_MODELS[k]
            mark = "  ← current" if v == current else ""
            print(f"  {k:20s} {v}{mark}")
        if text_aliases:
            print("\n  (TEXT-ONLY — will fail on image inputs)")
            for k in text_aliases:
                v = VLM_API_MODELS[k]
                mark = "  ← current" if v == current else ""
                print(f"  {k:20s} {v}{mark}")
        if current not in VLM_API_MODELS.values():
            print(f"\n  (current model is not in catalog: raw id {current!r})")

CFG = Config()
