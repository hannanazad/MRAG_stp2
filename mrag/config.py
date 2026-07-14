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

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("mrag.config")


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


# ────────────────────────────────────────────────────────────────────────────
# VLM model catalog (v6) — maps short aliases to raw DashScope model ids.
#
# Use CFG.set_vlm_model("alias") to switch at runtime; CFG.list_vlm_models()
# prints the catalog. vlm.py reads CFG.vlm_model_api at call time, so no
# kernel restart is needed.
#
# Notes on entitlement: DashScope returns 403 AccessDenied.Unpurchased for
# models your account isn't entitled to. Use `GET /v1/models` to enumerate
# what is actually available — the list below is a sensible default but is
# NOT guaranteed to be entitled on every account. The default below
# (`qwen3-vl-plus-2025-12-19`) is widely available; the entries marked
# (text_only) will return 400 InvalidParameter when ask() sends images.
# ────────────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
# Multi-provider VLM support. DashScope and Gemini use their official
# OpenAI-compatible endpoints. Claude is routed through Anthropic's native
# Messages API by vlm.py so adaptive-thinking token budgets and content blocks
# are handled reliably. The provider is inferred from the model-id prefix.
# ---------------------------------------------------------------------------
VLM_PROVIDERS = {
    "dashscope": {
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "env_var":  "DASHSCOPE_API_KEY",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1/",
        "env_var":  "ANTHROPIC_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env_var":  "GEMINI_API_KEY",
    },
}


# The user-picked Qwen tiers are the GENERAL qwen3.6/3.7 line, not qwen3-vl-*.
# Whether they accept image inputs is UNVERIFIED from here — DashScope returns
# 400 InvalidParameter on image content for text-only models. set_vlm_model
# emits a one-time reminder; verify with a single-image call before sweeping.
VLM_VISION_UNVERIFIED = {
    "qwen3.7-max-2026-06-08",
    "qwen3.7-plus-2026-05-26",
    "qwen3.6-flash-2026-04-16",
}


def provider_of_model(model_id: str) -> str:
    m = (model_id or "").lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith(("gemini", "gemma")):
        return "gemini"
    return "dashscope"


VLM_API_MODELS: dict = {
    # Flagship & large dense
    "flagship":       "qwen3-vl-235b-a22b-instruct",
    "max":            "qwen-vl-max",
    # Plus tier
    "qwen3-vl-plus":  "qwen3-vl-plus-2025-12-19",   # current default
    "plus":           "qwen3-vl-plus-2025-12-19",   # alias
    "qwen-vl-plus":   "qwen-vl-plus",
    # Fast / cheap
    "flash_pinned":   "qwen3-vl-flash-2026-01-22",  # cheap + fast, pinned date
    "flash":          "qwen3-vl-flash",             # tracks latest flash
    # Challenger (v4.3): a generation newer, vision-capable omni model.
    # BENCHMARK on the 49-QA set before trusting — omni models optimise for
    # breadth (audio/realtime too), not necessarily document figures.
    "challenger":         "qwen3.5-omni-plus-2026-03-15",
    "qwen3.5-omni-plus":  "qwen3.5-omni-plus-2026-03-15",

    # ---- Cross-provider tier matrix (v4.5, user-specified) -----------------
    # 3 providers x 3 tiers; sweep grid = this matrix x prompt styles.
    #                 Qwen                        Claude                        Gemini
    # Frontier        qwen3.7-max-2026-06-08      claude-fable-5                gemini-3.1-pro-preview
    # Balanced        qwen3.7-plus-2026-05-26     claude-sonnet-5               gemini-3.5-flash
    # Fast/economical qwen3.6-flash-2026-04-16    claude-haiku-4-5-20251001     gemini-3.1-flash-lite
    "frontier_qwen":     "qwen3.7-max-2026-06-08",
    "frontier_claude":   "claude-fable-5",
    "frontier_gemini":   "gemini-3.1-pro-preview",
    "balanced_qwen":     "qwen3.7-plus-2026-05-26",
    "balanced_claude":   "claude-sonnet-5",
    "balanced_gemini":   "gemini-3.5-flash",
    "fast_qwen":         "qwen3.6-flash-2026-04-16",
    "fast_claude":       "claude-haiku-4-5-20251001",
    "fast_gemini":       "gemini-3.1-flash-lite",
    # short conveniences for the same nine
    "fable":             "claude-fable-5",
    "sonnet":            "claude-sonnet-5",
    "haiku":             "claude-haiku-4-5-20251001",
    "gemini-pro":        "gemini-3.1-pro-preview",
    "gemini-flash":      "gemini-3.5-flash",
    "gemini-flash-lite": "gemini-3.1-flash-lite",
    # Older / fallback
    "qwen3-vl-32b":   "qwen3-vl-32b-instruct",      # often UNENTITLED
    "qwen2.5-vl":     "qwen2.5-vl-72b-instruct",
    # Text-only (will 400 if ask() sends figures) — kept for text-only experiments
    "glm-5.2":        "glm-5.2",                    # (text_only)
    "kimi-k2.7-code": "kimi-k2.7-code",             # (text_only)
}

VLM_TEXT_ONLY_ALIASES: set = {"glm-5.2", "kimi-k2.7-code"}


# ────────────────────────────────────────────────────────────────────────────
# Prompt-style catalogs (stp2_v1).
#
# Two independent flags govern how prompts are assembled at inference time:
#   - CFG.prompt_style_answer  → controls _build_prompt_and_images (P1)
#   - CFG.prompt_style_filter  → controls _filter_prompt           (P2)
#
# The answer prompt supports four styles; the filter prompt supports only
# zeroshot for now.
# ────────────────────────────────────────────────────────────────────────────
ANSWER_STYLES_AVAILABLE: tuple = ("zeroshot", "oneshot", "fewshot", "cot")
FILTER_STYLES_AVAILABLE: tuple = ("zeroshot",)


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
    vlm_provider: str = "api"  # "api" or "local"

    # Model name string sent to the API endpoint when vlm_provider == "api".
    # DEFAULT changed (v6) off the often-unentitled qwen3-vl-32b-instruct.
    # Use CFG.set_vlm_model("alias") to switch — vlm.py reads this at call time.
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
    #   2. Path C — VISUAL retrieval via ColPali over figure crops.
    #   3. Path B — caption-text retrieval, off by default (off-topic figures).
    #   4. (Optional) VLM filter that picks the visually-relevant subset.
    top_k_figures_candidates: int = 10
    top_k_figures: int = 4
    top_k_figures_visual: int = 6
    use_caption_figure_fallback: bool = False
    use_vlm_figure_filter: bool = True

    # Question router (v2): decide per-query whether figure retrieval runs at
    # all. Fixes the v1 behaviour of attaching ~4 figures to EVERY answer
    # (measured figure precision 6%). Toggle off to reproduce v1 behaviour.
    use_question_router: bool = True
    router_soft_threshold: float = 0.5
    router_use_vlm_tiebreak: bool = False

    # P2 figure-filter model override (v4.4). None => use vlm_model_api.
    # Recommended during cross-provider sweeps: CFG.set_filter_model("flash_pinned")
    # pins the filter to cheap Qwen flash while P1 runs Claude/Gemini/etc.
    vlm_model_filter: Optional[str] = None

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
    # max_new_tokens remains the visible-answer/default budget for the
    # existing DashScope runs. Provider-aware API ceilings below account for
    # models whose hidden reasoning tokens share the same hard output cap.
    max_new_tokens: int = 480
    max_chunk_chars_in_prompt: int = 1400

    # ----- Provider-aware API generation ------------------------------------
    # Claude Sonnet 5 and Fable 5 use adaptive thinking. Anthropic documents
    # max_tokens as a hard cap over thinking + visible response text, so the
    # previous value of 480 could be consumed before a final answer appeared.
    api_max_tokens_dashscope: int = 480
    api_max_tokens_anthropic: int = 16000
    api_max_tokens_gemini_frontier: int = 8192
    api_max_tokens_gemini_balanced: int = 4096
    api_max_tokens_gemini_fast: int = 2048

    # One automatic retry is allowed when a provider explicitly reports
    # length/max-token truncation. These are ceilings, not requested output
    # lengths; concise responses normally stop well before them.
    api_retry_on_truncation: bool = True
    api_truncation_retry_multiplier: float = 2.0
    api_max_tokens_ceiling_anthropic: int = 32000
    api_max_tokens_ceiling_gemini: int = 16384
    api_max_tokens_ceiling_dashscope: int = 2048

    # Gemini OpenAI compatibility maps reasoning_effort to thinking_level.
    gemini_reasoning_effort_frontier: str = "high"
    gemini_reasoning_effort_balanced: str = "medium"
    gemini_reasoning_effort_fast: str = "low"

    # Inline image requests should remain below Google's documented total
    # inline-request limit. This leaves headroom for prompt text and JSON.
    api_max_inline_image_bytes: int = 18 * 1024 * 1024

    # Metadata only: never stores hidden reasoning or raw API keys.
    api_store_response_metadata: bool = True

    # ----- Prompt-style controls (stp2_v1) ----------------------------------
    # See module-level docstring above the dataclass for the catalogs.
    # Switch at runtime with CFG.set_answer_style(...) — no kernel restart
    # needed; vlm.py reads these at call time.
    prompt_style_answer: str = "fewshot"    # P1 — answer generation prompt (v4.3 default; was zeroshot)
    prompt_style_filter: str = "zeroshot"   # P2 — figure relevance filter prompt
    fewshot_num_examples: int = 3            # how many examples to use when
                                             # prompt_style_answer == "fewshot".
                                             # capped at len(FEWSHOT_EXAMPLES).

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

    # ────────────────────────────────────────────────────────────────────
    # VLM model switcher (v6)
    # No kernel restart needed — vlm.py reads CFG.vlm_model_api at call time.
    # ────────────────────────────────────────────────────────────────────

    def set_vlm_model(self, alias_or_id: str) -> str:
        """Switch the VLM model. Accepts either:
          - a short alias from VLM_API_MODELS (e.g. "flagship", "opus",
            "gemini-pro"), OR
          - a raw model id from ANY provider ("qwen3-vl-plus-2025-12-19",
            "claude-sonnet-5", "gemini-3.5-flash").

        The provider (DashScope / Anthropic / Gemini) is inferred from the id
        prefix; base_url and API-key env var are switched automatically.
        Returns the resolved raw model id that will be sent to the API.
        """
        alias_or_id = (alias_or_id or "").strip()
        if not alias_or_id:
            raise ValueError("Pass an alias or model id, e.g. 'flagship' or 'qwen-vl-max'.")

        if alias_or_id in VLM_API_MODELS:
            resolved = VLM_API_MODELS[alias_or_id]
            if alias_or_id in VLM_TEXT_ONLY_ALIASES:
                log.warning(
                    "Model %r (%s) is text-only. ask() will fail with 400 "
                    "InvalidParameter when figures are sent. Use it only for "
                    "text-only experiments outside ask().",
                    alias_or_id, resolved,
                )
        else:
            # Treat as a raw DashScope id. We can't validate entitlement here;
            # if it's wrong you'll get a 403 AccessDenied.Unpurchased at call time.
            resolved = alias_or_id

        self.vlm_model_api = resolved
        if resolved in VLM_VISION_UNVERIFIED:
            log.warning(
                "Model %s is from the general Qwen line — vision capability "
                "UNVERIFIED. Run one image call before a full sweep; if it "
                "400s on images, switch to the qwen3-vl equivalent.", resolved)
        prov = provider_of_model(resolved)
        self.api_base_url = VLM_PROVIDERS[prov]["base_url"]
        self.api_key_env_var = VLM_PROVIDERS[prov]["env_var"]
        log.info("CFG.vlm_model_api → %s  [provider=%s]", resolved, prov)
        return resolved

    def set_filter_model(self, alias_or_id: Optional[str]) -> Optional[str]:
        """Pin the P2 figure filter to its own model (None => follow the
        answer model). Provider is inferred per call, so the filter can run
        on DashScope while answers run on Anthropic/Gemini."""
        if alias_or_id is None:
            self.vlm_model_filter = None
            log.info("CFG.vlm_model_filter → None (follows vlm_model_api)")
            return None
        resolved = VLM_API_MODELS.get(alias_or_id.strip(), alias_or_id.strip())
        self.vlm_model_filter = resolved
        log.info("CFG.vlm_model_filter → %s  [provider=%s]",
                 resolved, provider_of_model(resolved))
        return resolved

    def list_vlm_models(self) -> dict:
        """Show the v6 catalog plus what's currently selected. Returns the
        catalog as a dict, and also logs a readable table at INFO so a bare
        call in the notebook prints something useful.
        """
        log.info("Current vlm_model_api: %s", self.vlm_model_api)
        log.info("Available aliases (set with CFG.set_vlm_model(alias)):")
        col_w = max(len(k) for k in VLM_API_MODELS)
        for alias, raw in VLM_API_MODELS.items():
            tag = "  (text_only)" if alias in VLM_TEXT_ONLY_ALIASES else ""
            log.info("  %s  →  %s%s", alias.ljust(col_w), raw, tag)
        log.info("You may also pass a raw DashScope id (e.g. 'qwen-vl-max').")
        return {
            "current": self.vlm_model_api,
            "catalog": dict(VLM_API_MODELS),
            "text_only": sorted(VLM_TEXT_ONLY_ALIASES),
        }

    # ────────────────────────────────────────────────────────────────────
    # Prompt-style runtime switchers (stp2_v1).
    # No kernel restart needed — vlm.py reads CFG.prompt_style_* at call time.
    # ────────────────────────────────────────────────────────────────────

    def set_answer_style(self, style: str) -> str:
        """Switch the answer-prompt style. Valid: zeroshot, oneshot, fewshot, cot."""
        style = (style or "").strip().lower()
        if style not in ANSWER_STYLES_AVAILABLE:
            raise ValueError(
                f"Unknown answer style {style!r}. "
                f"Available: {list(ANSWER_STYLES_AVAILABLE)}"
            )
        self.prompt_style_answer = style
        log.info("CFG.prompt_style_answer → %s", style)
        return style

    def set_filter_style(self, style: str) -> str:
        """Switch the figure-filter-prompt style. Only 'zeroshot' is implemented."""
        style = (style or "").strip().lower()
        if style not in FILTER_STYLES_AVAILABLE:
            log.warning(
                "Filter style %r is not implemented yet; setting flag but "
                "vlm.py will use zeroshot. Implemented: %s",
                style, list(FILTER_STYLES_AVAILABLE),
            )
        self.prompt_style_filter = style
        log.info("CFG.prompt_style_filter → %s", style)
        return style

    def list_prompt_styles(self) -> dict:
        """Return current selections + available styles. Logs at INFO too."""
        info = {
            "answer": {
                "current":   self.prompt_style_answer,
                "available": list(ANSWER_STYLES_AVAILABLE),
            },
            "filter": {
                "current":   self.prompt_style_filter,
                "available": list(FILTER_STYLES_AVAILABLE),
            },
            "fewshot_num_examples": self.fewshot_num_examples,
        }
        log.info("Prompt styles: %s", info)
        return info


CFG = Config()
