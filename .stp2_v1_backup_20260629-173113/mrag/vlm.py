"""Qwen2.5-VL-7B-Instruct (default) wrapper.

Supports two backends, controlled by mrag/config.py's CFG object:
  - CFG.vlm_provider = "local" → loads weights via HuggingFace transformers
                                  (original behaviour, unchanged)
  - CFG.vlm_provider = "api"   → calls an OpenAI-compatible REST endpoint
                                  (e.g. Qwen3-VL-32B via DashScope) — no GPU,
                                  no local download

The prompt mirrors MUTCD's own taxonomy: outputs are blocked by rule type
(Standards / Guidance / Options / Support) and citations are restricted to
an explicit whitelist constructed from retrieval results. This logic is
backend-agnostic — only the final "send to model" step differs.

To swap models or providers, edit CFG in mrag/config.py only:
    CFG.vlm_provider  = "api" | "local"
    CFG.vlm_model_api = "qwen3-vl-32b-instruct"   (used when provider == "api")
    CFG.vlm_model     = "Qwen/Qwen2.5-VL-7B-Instruct"  (used when provider == "local")
"""
from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from mrag.config import CFG

log = logging.getLogger("mrag.vlm")


class VLM:
    def __init__(
        self,
        model_name: Optional[str] = None,
        fallback_name: Optional[str] = None,
        torch_dtype: str = "bfloat16",
        provider: Optional[str] = None,
    ) -> None:
        """
        All arguments default to CFG values, so existing call sites like
        VLM(CFG.vlm_model, CFG.vlm_model_fallback).load() keep working
        exactly as before. provider defaults to CFG.vlm_provider.
        """
        self.model_name = model_name or CFG.vlm_model
        self.fallback_name = fallback_name or CFG.vlm_model_fallback
        self.torch_dtype = torch_dtype
        self.provider = provider or getattr(CFG, "vlm_provider", "local")
        self._model = None
        self._processor = None
        self._loaded_name = None
        self._api_client = None

    # ────────────────────────────────────────────────────────────────────
    # Loading
    # ────────────────────────────────────────────────────────────────────

    def load(self) -> "VLM":
        if self.provider == "api":
            self._load_api()
        else:
            self._load_local()
        return self

    def _load_api(self) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError("API mode needs the openai package: pip install openai")

        env_var = getattr(CFG, "api_key_env_var", "VLM_API_KEY")
        api_key = os.environ.get(env_var)
        if not api_key:
            raise EnvironmentError(
                f"Set the environment variable '{env_var}' before calling "
                f".load(), e.g.:\n"
                f"  import os; os.environ['{env_var}'] = 'sk-...'"
            )

        self._api_client = openai.OpenAI(
            api_key=api_key,
            base_url=CFG.api_base_url,
        )
        self._loaded_name = CFG.vlm_model_api
        log.info("VLM (api) ready: %s @ %s", CFG.vlm_model_api, CFG.api_base_url)

    def _load_local(self) -> None:
        import torch
        try:
            self._do_load(self.model_name, torch_dtype=getattr(torch, self.torch_dtype))
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            log.warning("VLM %s OOM/fail (%r); falling back to %s",
                        self.model_name, e, self.fallback_name)
            torch.cuda.empty_cache()
            self._do_load(self.fallback_name, torch_dtype=getattr(torch, self.torch_dtype))

    def _do_load(self, name, torch_dtype):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        cache_dir = str(getattr(CFG, "hf_home", "")) or None
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            name, torch_dtype=torch_dtype, device_map="auto", cache_dir=cache_dir,
        ).eval()
        self._processor = AutoProcessor.from_pretrained(name, cache_dir=cache_dir)
        self._loaded_name = name
        log.info("VLM loaded: %s (%s)", name, torch_dtype)

    @property
    def loaded_name(self) -> str:
        return self._loaded_name or ""

    # ────────────────────────────────────────────────────────────────────
    # Generation — public method, unchanged signature
    # ────────────────────────────────────────────────────────────────────

    def answer(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        figures: List[Dict[str, Any]],
        pages: List[Dict[str, Any]],
        max_new_tokens: Optional[int] = None,
    ) -> str:
        max_new_tokens = max_new_tokens or CFG.max_new_tokens
        prompt, image_paths = self._build_prompt_and_images(question, chunks, figures, pages)

        if self.provider == "api":
            return self._answer_api(prompt, image_paths, max_new_tokens)
        else:
            return self._answer_local(prompt, image_paths, max_new_tokens)

    # ----- local generation (original behaviour, unchanged) ---------------

    def _answer_local(self, prompt: str, image_paths: List[str], max_new_tokens: int) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        content: List[Dict[str, Any]] = [{"type": "image", "image": f"file://{p}"} for p in image_paths]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)
        with torch.inference_mode():
            gen = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen)]
        out_text = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()
        return out_text

    # ----- API generation (new) --------------------------------------------

    def _answer_api(self, prompt: str, image_paths: List[str], max_new_tokens: int) -> str:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for p in image_paths:
            if Path(p).exists():
                b64 = base64.b64encode(Path(p).read_bytes()).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })

        response = self._api_client.chat.completions.create(
            model=CFG.vlm_model_api,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_new_tokens,
        )
        return response.choices[0].message.content.strip()

    # ────────────────────────────────────────────────────────────────────
    # Figure relevance filter (v5+) — one extra cheap VLM call that scores
    # candidate figures for visual relevance to the question and returns
    # the indices to keep. Used by ask.py to prune the retrieval candidates
    # before they go into the answer prompt.
    # ────────────────────────────────────────────────────────────────────

    def filter_figures(
        self,
        question: str,
        figures: List[Dict[str, Any]],
        max_keep: int = 4,
    ) -> List[int]:
        """Return the subset of figure indices to keep, in priority order.

        Falls back to "keep everything" if the call fails or the response
        can't be parsed — never raises. Caller can pass at most ~12 figures
        in one call (image-token budget); larger lists are processed in
        a single request anyway since modern VLMs handle it.
        """
        if not figures:
            return []
        # Resolve to the actual on-disk images we can show. Anything
        # without a usable image is dropped from consideration outright.
        usable: List[tuple[int, str, Dict[str, Any]]] = []
        for i, f in enumerate(figures):
            ip = f.get("image_path", "")
            if ip and Path(ip).exists():
                usable.append((i, ip, f))
        if not usable:
            return []
        if len(usable) <= max_keep:
            # Nothing to prune — keep the order retrieval gave us.
            return [i for i, _, _ in usable]

        try:
            if self.provider == "api":
                kept = self._filter_figures_api(question, usable, max_keep)
            else:
                kept = self._filter_figures_local(question, usable, max_keep)
        except Exception as e:
            log.warning("filter_figures failed (%r); keeping top-%d unfiltered",
                        e, max_keep)
            return [i for i, _, _ in usable[:max_keep]]

        # Validate kept indices, clip to max_keep
        valid = [k for k in kept if any(i == k for i, _, _ in usable)]
        if not valid:
            # Model returned nothing valid — fall back to first max_keep
            return [i for i, _, _ in usable[:max_keep]]
        return valid[:max_keep]

    def _filter_prompt(self, question: str, indexed_figures) -> str:
        lines = []
        for i, _ip, f in indexed_figures:
            cap = (f.get("caption") or f.get("title") or "")[:140]
            fid = f.get("figure_id", "?")
            lines.append(f"  [{i}] {fid} — {cap}")
        return (
            "You are filtering figure crops for visual relevance.\n\n"
            f'The user asked: "{question}"\n\n'
            f"I'm showing you {len(indexed_figures)} candidate figures, "
            "numbered as labelled below and provided in that order as image "
            "inputs. Decide which ones a reader would actually need to SEE "
            "(visual content shows what the user asked about) to understand "
            "the answer. Be strict — prefer false negatives over false positives. "
            "If a figure is unrelated even though its caption mentions a similar "
            "term, drop it.\n\n"
            "Candidates:\n" + "\n".join(lines) + "\n\n"
            "Reply with ONLY a JSON array of the indices to keep, ordered "
            "best-first. No explanation. Example: [0, 3, 7]"
        )

    @staticmethod
    def _parse_filter_response(text: str) -> List[int]:
        import json as _json
        import re as _re
        text = text.strip()
        # Try direct JSON parse first
        try:
            v = _json.loads(text)
            if isinstance(v, list):
                return [int(x) for x in v if isinstance(x, (int, float))]
        except Exception:
            pass
        # Fall back: extract the first [...]-looking substring
        m = _re.search(r"\[[\s\d,]*\]", text)
        if m:
            try:
                v = _json.loads(m.group(0))
                if isinstance(v, list):
                    return [int(x) for x in v if isinstance(x, (int, float))]
            except Exception:
                pass
        return []

    def _filter_figures_api(self, question, indexed_figures, max_keep):
        prompt = self._filter_prompt(question, indexed_figures)
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for _i, ip, _f in indexed_figures:
            b64 = base64.b64encode(Path(ip).read_bytes()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        response = self._api_client.chat.completions.create(
            model=CFG.vlm_model_api,
            messages=[{"role": "user", "content": content}],
            max_tokens=80,  # short reply — just an index list
        )
        return self._parse_filter_response(
            response.choices[0].message.content or ""
        )

    def _filter_figures_local(self, question, indexed_figures, max_keep):
        import torch
        from qwen_vl_utils import process_vision_info

        prompt = self._filter_prompt(question, indexed_figures)
        content: List[Dict[str, Any]] = [
            {"type": "image", "image": f"file://{ip}"}
            for _i, ip, _f in indexed_figures
        ]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self._model.device)
        with torch.inference_mode():
            gen = self._model.generate(
                **inputs, max_new_tokens=80, do_sample=False,
            )
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen)]
        out_text = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]
        return self._parse_filter_response(out_text)

    # ────────────────────────────────────────────────────────────────────

    def _build_prompt_and_images(self, question, chunks, figures, pages):
        image_paths: List[str] = []
        used_visuals = []

        for f in figures:
            ip = f.get("image_path", "")
            if ip and Path(ip).exists():
                image_paths.append(ip)
                used_visuals.append(("Figure", f))
        for p in pages:
            ip = p.get("image_path", "")
            if ip and Path(ip).exists():
                image_paths.append(ip)
                used_visuals.append(("Page", p))

        groups: Dict[str, List[Dict[str, Any]]] = {
            "Standard": [], "Guidance": [], "Option": [], "Support": [],
        }
        for c in chunks:
            ct = c.get("content_type", "Support")
            groups.setdefault(ct, []).append(c)

        evidence_blocks = []
        for ct in ("Standard", "Guidance", "Option", "Support"):
            cs = groups.get(ct, [])
            if not cs:
                continue
            evidence_blocks.append(f"=== {ct} provisions ===")
            for c in cs:
                evidence_blocks.append(
                    f"[Section {c.get('section_id')} §{c.get('ordinal')} — "
                    f"{c.get('section_title','')} (p.{c.get('page_printed','?')})]\n"
                    f"{(c.get('text','') or '')[:CFG.max_chunk_chars_in_prompt]}"
                )
            evidence_blocks.append("")

        visual_lines = []
        for i, (kind, v) in enumerate(used_visuals, 1):
            if kind == "Figure":
                visual_lines.append(
                    f"[Image {i}] {v.get('figure_id','?')} (p.{v.get('page_printed','?')}): "
                    f"{(v.get('caption','') or '')[:160]}"
                )
            else:
                visual_lines.append(
                    f"[Image {i}] Page {v.get('page_printed','?')} (full page view)"
                )

        allowed_cites = []
        for c in chunks:
            allowed_cites.append(
                f"Section {c.get('section_id')} {c.get('content_type')} §{c.get('ordinal')} (p.{c.get('page_printed','?')})"
            )
        for kind, v in used_visuals:
            if kind == "Figure":
                allowed_cites.append(f"{v.get('figure_id','?')} (p.{v.get('page_printed','?')})")
            else:
                allowed_cites.append(f"Page {v.get('page_printed','?')}")

        prompt = (
            "You are an expert reader of the Manual on Uniform Traffic Control Devices "
            "(MUTCD, 11th Edition). Answer the user's question using ONLY the evidence below.\n\n"
            "MUTCD distinguishes four normative categories. Treat them carefully:\n"
            "  - Standard: MANDATORY requirements (modal verb: shall).\n"
            "  - Guidance: RECOMMENDED practice (modal verb: should).\n"
            "  - Option: PERMITTED practice (modal verb: may).\n"
            "  - Support: explanatory or informational only — never normative.\n\n"
            "Output format (use these exact section headings, omit any that have no content):\n"
            "  Direct Answer: 2–3 sentences in plain language.\n"
            "  Standards (mandatory): bullets quoting the relevant Standard provision(s).\n"
            "  Guidance (recommended): bullets.\n"
            "  Options (permitted): bullets.\n"
            "  Visual evidence: one sentence per relevant image, referenced as [Image N].\n"
            "  Citations: bullets, ONE PER LINE, chosen ONLY from the allowed list below.\n\n"
            "Rules:\n"
            "  - Never invent section numbers, figure numbers, or page numbers.\n"
            "  - If the evidence is insufficient, say so plainly and stop.\n"
            "  - Quote MUTCD wording verbatim when stating a Standard provision.\n\n"
            f"Question: {question}\n\n"
            f"Visual evidence ({len(visual_lines)} images attached):\n"
            + ("\n".join(visual_lines) if visual_lines else "(none)") + "\n\n"
            f"Text evidence:\n" + "\n".join(evidence_blocks) + "\n"
            f"Allowed citations (use ONLY these strings verbatim):\n"
            + "\n".join(f"  - {c}" for c in allowed_cites) + "\n"
        )
        return prompt, image_paths
