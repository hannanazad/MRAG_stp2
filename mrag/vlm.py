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

Prompt assembly is dispatched on CFG.prompt_style_answer (stp2_v1):
  - "zeroshot"  → original behaviour: instructions + evidence + question
  - "oneshot"   → one worked MUTCD example before the question
  - "fewshot"   → multiple worked examples before the question
  - "cot"       → reorders the output spec so reasoning comes first and
                  Direct Answer is synthesized last (zero-shot CoT)

Switch at runtime with CFG.set_answer_style(...) — no kernel restart needed.

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
from mrag.prompt_examples import FEWSHOT_EXAMPLES, format_example

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
            import openai  # noqa: F401
        except ImportError:
            raise ImportError("API mode needs the openai package: pip install openai")
        self._api_clients = {}
        # Build the client for the CURRENT model eagerly so a missing API key
        # fails at load time with a friendly message, not mid-question.
        self._client_for(CFG.vlm_model_api)
        self._loaded_name = CFG.vlm_model_api
        log.info("VLM (api) ready: %s @ %s", CFG.vlm_model_api, CFG.api_base_url)

    def _client_for(self, model_id: str):
        """Return an OpenAI-compatible client for the provider that serves
        `model_id` (v4.4 multi-provider). Clients are cached per provider, so
        switching models — or running the P2 filter on DashScope while P1
        answers on Anthropic/Gemini — needs no reload."""
        import openai
        from .config import VLM_PROVIDERS, provider_of_model
        prov = provider_of_model(model_id)
        if prov not in getattr(self, "_api_clients", {}):
            spec = VLM_PROVIDERS[prov]
            api_key = os.environ.get(spec["env_var"])
            if not api_key:
                raise EnvironmentError(
                    f"Model {model_id!r} is served by {prov!r}. Set the "
                    f"environment variable '{spec['env_var']}' first, e.g.:\n"
                    f"  import os; os.environ['{spec['env_var']}'] = '...'"
                )
            if not hasattr(self, "_api_clients"):
                self._api_clients = {}
            self._api_clients[prov] = openai.OpenAI(
                api_key=api_key, base_url=spec["base_url"])
            log.info("VLM client ready for provider %s @ %s",
                     prov, spec["base_url"])
        return self._api_clients[prov]

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

        model_id = CFG.vlm_model_api
        response = self._client_for(model_id).chat.completions.create(
            model=model_id,
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
        # Only zeroshot is implemented for P2 today. CFG.prompt_style_filter
        # is read for future-proofing; non-zeroshot values just log and
        # fall through to the zeroshot prompt.
        style = getattr(CFG, "prompt_style_filter", "zeroshot")
        if style != "zeroshot":
            log.debug("Filter style %r requested but not implemented; using zeroshot.", style)

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
        filter_model = getattr(CFG, "vlm_model_filter", None) or CFG.vlm_model_api
        response = self._client_for(filter_model).chat.completions.create(
            model=filter_model,
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
    # Prompt assembly (stp2_v1) — dispatches on CFG.prompt_style_answer.
    # Each style shares the same evidence/visual/citations blocks, but
    # differs in (a) whether worked examples appear before the question,
    # and (b) the output format spec (CoT puts Direct Answer last).
    # ────────────────────────────────────────────────────────────────────

    def _build_prompt_and_images(self, question, chunks, figures, pages):
        # --- shared: gather visuals + assemble evidence-side blocks --------
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

        evidence_text = self._format_evidence_text(chunks)
        visual_lines = self._format_visual_lines(used_visuals)
        allowed_cites = self._format_allowed_cites(chunks, used_visuals)

        # --- dispatch on style ---------------------------------------------
        style = getattr(CFG, "prompt_style_answer", "zeroshot")
        if style == "zeroshot":
            prompt = self._assemble_zeroshot(question, evidence_text, visual_lines, allowed_cites)
        elif style == "oneshot":
            prompt = self._assemble_oneshot(question, evidence_text, visual_lines, allowed_cites)
        elif style == "fewshot":
            prompt = self._assemble_fewshot(question, evidence_text, visual_lines, allowed_cites)
        elif style == "cot":
            prompt = self._assemble_cot(question, evidence_text, visual_lines, allowed_cites)
        else:
            log.warning("Unknown prompt_style_answer %r; falling back to zeroshot.", style)
            prompt = self._assemble_zeroshot(question, evidence_text, visual_lines, allowed_cites)

        return prompt, image_paths

    # ----- evidence-side formatting (shared by all styles) ----------------

    def _format_evidence_text(self, chunks) -> str:
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
        return "\n".join(evidence_blocks)

    def _format_visual_lines(self, used_visuals) -> List[str]:
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
        return visual_lines

    def _format_allowed_cites(self, chunks, used_visuals) -> List[str]:
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
        return allowed_cites

    # ----- prompt fragments (shared building blocks) ----------------------

    @staticmethod
    def _intro_block() -> str:
        return (
            "You are an expert reader of the Manual on Uniform Traffic Control Devices "
            "(MUTCD, 11th Edition). Answer the user's question using ONLY the evidence below.\n\n"
            "MUTCD distinguishes four normative categories. Treat them carefully:\n"
            "  - Standard: MANDATORY requirements (modal verb: shall).\n"
            "  - Guidance: RECOMMENDED practice (modal verb: should).\n"
            "  - Option: PERMITTED practice (modal verb: may).\n"
            "  - Support: explanatory or informational only — never normative.\n\n"
        )

    @staticmethod
    def _format_spec_standard() -> str:
        # Direct-Answer-FIRST output format (zeroshot, oneshot, fewshot).
        return (
            "Output format (use these exact section headings, omit any that have no content):\n"
            "  Direct Answer: 2–3 sentences in plain language.\n"
            "  Standards (mandatory): bullets quoting the relevant Standard provision(s).\n"
            "  Guidance (recommended): bullets.\n"
            "  Options (permitted): bullets.\n"
            "  Visual evidence: one sentence per relevant image, referenced as [Image N].\n"
            "  Citations: bullets, ONE PER LINE, chosen ONLY from the allowed list below.\n\n"
        )

    @staticmethod
    def _format_spec_cot() -> str:
        # Reasoning-FIRST, Direct-Answer-LAST output format (cot).
        return (
            "Output format (use these exact section headings, omit any that have no content):\n"
            "  Reasoning: Work through the evidence step by step. For each piece of evidence, "
            "identify (a) which MUTCD category it falls under — Standard / Guidance / Option / "
            "Support — and (b) what it requires, recommends, permits, or merely explains. Then "
            "identify which provisions are directly relevant to the user's question and how "
            "they combine. Keep this section concise (4–8 sentences).\n"
            "  Standards (mandatory): bullets quoting the relevant Standard provision(s).\n"
            "  Guidance (recommended): bullets.\n"
            "  Options (permitted): bullets.\n"
            "  Visual evidence: one sentence per relevant image, referenced as [Image N].\n"
            "  Direct Answer: 2–3 sentences in plain language that SYNTHESIZE the reasoning "
            "above. Do NOT introduce new facts here.\n"
            "  Citations: bullets, ONE PER LINE, chosen ONLY from the allowed list below.\n\n"
        )

    @staticmethod
    def _rules_block() -> str:
        return (
            "Rules:\n"
            "  - Never invent section numbers, figure numbers, or page numbers.\n"
            "  - If the evidence is insufficient, say so plainly and stop.\n"
            "  - Quote MUTCD wording verbatim when stating a Standard provision.\n\n"
        )

    @staticmethod
    def _question_and_evidence_block(question, evidence_text, visual_lines, allowed_cites) -> str:
        return (
            f"Question: {question}\n\n"
            f"Visual evidence ({len(visual_lines)} images attached):\n"
            + ("\n".join(visual_lines) if visual_lines else "(none)") + "\n\n"
            f"Text evidence:\n{evidence_text}\n"
            f"Allowed citations (use ONLY these strings verbatim):\n"
            + "\n".join(f"  - {c}" for c in allowed_cites) + "\n"
        )

    # ----- the four assemblers --------------------------------------------

    def _assemble_zeroshot(self, question, evidence_text, visual_lines, allowed_cites) -> str:
        return (
            self._intro_block()
            + self._format_spec_standard()
            + self._rules_block()
            + self._question_and_evidence_block(question, evidence_text, visual_lines, allowed_cites)
        )

    def _assemble_oneshot(self, question, evidence_text, visual_lines, allowed_cites) -> str:
        if not FEWSHOT_EXAMPLES:
            log.warning("oneshot requested but no FEWSHOT_EXAMPLES defined; using zeroshot.")
            return self._assemble_zeroshot(question, evidence_text, visual_lines, allowed_cites)
        example_block = (
            "Here is one worked example of the expected reasoning and format. The example "
            "uses real MUTCD content; treat it as an illustration of style, not as evidence "
            "for any question below.\n\n"
            + format_example(FEWSHOT_EXAMPLES[0], n=1, max_chars=CFG.max_chunk_chars_in_prompt)
            + "\nNow answer the following question using ONLY its own evidence:\n\n"
        )
        return (
            self._intro_block()
            + self._format_spec_standard()
            + self._rules_block()
            + example_block
            + self._question_and_evidence_block(question, evidence_text, visual_lines, allowed_cites)
        )

    def _assemble_fewshot(self, question, evidence_text, visual_lines, allowed_cites) -> str:
        if not FEWSHOT_EXAMPLES:
            log.warning("fewshot requested but no FEWSHOT_EXAMPLES defined; using zeroshot.")
            return self._assemble_zeroshot(question, evidence_text, visual_lines, allowed_cites)
        n = max(1, min(getattr(CFG, "fewshot_num_examples", 3), len(FEWSHOT_EXAMPLES)))
        examples_text = "\n".join(
            format_example(FEWSHOT_EXAMPLES[i], n=i+1, max_chars=CFG.max_chunk_chars_in_prompt)
            for i in range(n)
        )
        example_block = (
            f"Here are {n} worked examples of the expected reasoning and format. The examples "
            "use real MUTCD content; treat them as illustrations of style, not as evidence "
            "for any question below.\n\n"
            + examples_text
            + "\nNow answer the following question using ONLY its own evidence:\n\n"
        )
        return (
            self._intro_block()
            + self._format_spec_standard()
            + self._rules_block()
            + example_block
            + self._question_and_evidence_block(question, evidence_text, visual_lines, allowed_cites)
        )

    def _assemble_cot(self, question, evidence_text, visual_lines, allowed_cites) -> str:
        cot_instruction = (
            "Approach: think carefully BEFORE writing the Direct Answer. In your Reasoning "
            "section, do not skip the step of classifying each cited provision by MUTCD "
            "category. The Direct Answer must follow from the Reasoning; do not introduce "
            "new claims in the Direct Answer that are not justified above.\n\n"
        )
        return (
            self._intro_block()
            + self._format_spec_cot()
            + self._rules_block()
            + cot_instruction
            + self._question_and_evidence_block(question, evidence_text, visual_lines, allowed_cites)
        )
