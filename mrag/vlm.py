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

ADAPTER_VERSION = "1.0.0"


class VLMResponseError(RuntimeError):
    """Base class for provider responses that cannot be used as answers."""


class VLMEmptyResponseError(VLMResponseError):
    """Raised when a provider returns no extractable final answer text."""


class VLMTruncatedResponseError(VLMResponseError):
    """Raised when a provider stops because the output cap was reached."""


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
        self._api_clients = {}
        self._anthropic_clients = {}
        self.last_api_debug: Dict[str, Any] = {}

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
        """Initialize provider clients lazily.

        DashScope and Gemini use their official OpenAI-compatible endpoints.
        Claude uses Anthropic's native Messages API because newer Claude
        models use adaptive thinking and structured content blocks that are
        more reliably represented by the native SDK.
        """
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "API mode needs the openai package: pip install openai"
            ) from exc

        self._api_clients = {}
        self._anthropic_clients = {}

        try:
            from .config import provider_of_model

            provider = provider_of_model(CFG.vlm_model_api)

            if provider == "anthropic":
                self._anthropic_client_for(CFG.vlm_model_api)
            else:
                self._client_for(CFG.vlm_model_api)
        except (EnvironmentError, ImportError) as exc:
            log.warning(
                "No usable API client yet for the default model (%s). "
                "Pipeline initialization will continue; load the provider "
                "key/dependency or switch models before asking. Detail: %s",
                CFG.vlm_model_api,
                exc,
            )

        self._loaded_name = CFG.vlm_model_api
        log.info(
            "VLM API adapter ready: %s @ %s",
            CFG.vlm_model_api,
            CFG.api_base_url,
        )

    def _client_for(self, model_id: str):
        """Return an OpenAI-compatible client for DashScope or Gemini."""
        import openai
        from .config import VLM_PROVIDERS, provider_of_model

        provider = provider_of_model(model_id)

        if provider == "anthropic":
            raise ValueError(
                "Claude models use _anthropic_client_for(), not the "
                "OpenAI-compatible client."
            )

        if provider not in self._api_clients:
            specification = VLM_PROVIDERS[provider]
            api_key = os.environ.get(specification["env_var"])

            if not api_key and provider == "dashscope":
                api_key = os.environ.get("VLM_API_KEY")

            if not api_key:
                raise EnvironmentError(
                    f"Model {model_id!r} is served by {provider!r}. Set "
                    f"'{specification['env_var']}' before calling it."
                )

            self._api_clients[provider] = openai.OpenAI(
                api_key=api_key,
                base_url=specification["base_url"],
            )

            log.info(
                "OpenAI-compatible client ready for %s @ %s",
                provider,
                specification["base_url"],
            )

        return self._api_clients[provider]

    def _anthropic_client_for(self, model_id: str):
        """Return Anthropic's native client for Claude models."""
        from .config import VLM_PROVIDERS, provider_of_model

        provider = provider_of_model(model_id)

        if provider != "anthropic":
            raise ValueError(
                f"Model {model_id!r} is not an Anthropic model."
            )

        if provider not in self._anthropic_clients:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "Claude native API support needs the anthropic package: "
                    "pip install anthropic"
                ) from exc

            specification = VLM_PROVIDERS[provider]
            api_key = os.environ.get(specification["env_var"])

            if not api_key:
                raise EnvironmentError(
                    f"Set '{specification['env_var']}' before calling "
                    f"{model_id!r}."
                )

            self._anthropic_clients[provider] = anthropic.Anthropic(
                api_key=api_key
            )

            log.info("Native Anthropic client ready.")

        return self._anthropic_clients[provider]

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

    @staticmethod
    def _safe_model_dump(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}

        if isinstance(value, dict):
            return value

        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                dumped = model_dump()
                return dumped if isinstance(dumped, dict) else {}
            except Exception:
                return {}

        return {}

    @classmethod
    def _extract_text_content(cls, content: Any) -> str:
        """Extract final text from string, dict, or typed content blocks.

        This intentionally does not stringify unknown objects because an
        object representation is not a model answer.
        """
        if content is None:
            return ""

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, (list, tuple)):
            parts = [
                cls._extract_text_content(item)
                for item in content
            ]
            return "\n".join(
                part for part in parts if part
            ).strip()

        if isinstance(content, dict):
            block_type = str(
                content.get("type", "")
            ).strip().lower()

            if block_type in {"thinking", "reasoning", "redacted_thinking"}:
                return ""

            text_value = content.get("text")

            if isinstance(text_value, str):
                return text_value.strip()

            if isinstance(text_value, dict):
                nested_value = text_value.get("value")
                if isinstance(nested_value, str):
                    return nested_value.strip()

            for key in ("content", "output_text", "value"):
                if key in content:
                    extracted = cls._extract_text_content(
                        content[key]
                    )
                    if extracted:
                        return extracted

            return ""

        block_type = str(
            getattr(content, "type", "")
        ).strip().lower()

        if block_type in {"thinking", "reasoning", "redacted_thinking"}:
            return ""

        text_value = getattr(content, "text", None)

        if isinstance(text_value, str):
            return text_value.strip()

        if text_value is not None:
            extracted = cls._extract_text_content(text_value)
            if extracted:
                return extracted

        nested_content = getattr(content, "content", None)
        if nested_content is not None:
            return cls._extract_text_content(nested_content)

        dumped = cls._safe_model_dump(content)
        if dumped:
            return cls._extract_text_content(dumped)

        return ""

    @staticmethod
    def _usage_dict(response: Any) -> Dict[str, Any]:
        usage = getattr(response, "usage", None)

        if usage is None:
            return {}

        if isinstance(usage, dict):
            return usage

        model_dump = getattr(usage, "model_dump", None)

        if callable(model_dump):
            try:
                dumped = model_dump()
                return dumped if isinstance(dumped, dict) else {}
            except Exception:
                return {}

        result: Dict[str, Any] = {}

        for field_name in (
            "input_tokens",
            "output_tokens",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ):
            field_value = getattr(usage, field_name, None)
            if field_value is not None:
                result[field_name] = field_value

        return result

    @staticmethod
    def _content_block_types(content: Any) -> List[str]:
        if not isinstance(content, (list, tuple)):
            return [type(content).__name__]

        types: List[str] = []

        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
            else:
                block_type = getattr(block, "type", None)

            types.append(
                str(block_type or type(block).__name__)
            )

        return types

    @staticmethod
    def _image_media_type(path: Path) -> str:
        suffix = path.suffix.lower()

        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(suffix, "image/png")

    def _provider_token_limit(
        self,
        model_id: str,
        requested_tokens: int,
    ) -> int:
        from .config import provider_of_model

        provider = provider_of_model(model_id)

        if provider == "anthropic":
            configured = int(
                getattr(CFG, "api_max_tokens_anthropic", 16000)
            )
        elif provider == "gemini":
            model_lower = model_id.lower()

            if "pro" in model_lower:
                configured = int(
                    getattr(
                        CFG,
                        "api_max_tokens_gemini_frontier",
                        8192,
                    )
                )
            elif "flash-lite" in model_lower:
                configured = int(
                    getattr(
                        CFG,
                        "api_max_tokens_gemini_fast",
                        2048,
                    )
                )
            else:
                configured = int(
                    getattr(
                        CFG,
                        "api_max_tokens_gemini_balanced",
                        4096,
                    )
                )
        else:
            configured = int(
                getattr(
                    CFG,
                    "api_max_tokens_dashscope",
                    requested_tokens,
                )
            )

        return max(int(requested_tokens), configured)

    def _provider_token_ceiling(self, model_id: str) -> int:
        from .config import provider_of_model

        provider = provider_of_model(model_id)

        if provider == "anthropic":
            return int(
                getattr(
                    CFG,
                    "api_max_tokens_ceiling_anthropic",
                    32000,
                )
            )

        if provider == "gemini":
            return int(
                getattr(
                    CFG,
                    "api_max_tokens_ceiling_gemini",
                    16384,
                )
            )

        return int(
            getattr(
                CFG,
                "api_max_tokens_ceiling_dashscope",
                2048,
            )
        )

    def _gemini_reasoning_effort(self, model_id: str) -> str:
        model_lower = model_id.lower()

        if "pro" in model_lower:
            return str(
                getattr(
                    CFG,
                    "gemini_reasoning_effort_frontier",
                    "high",
                )
            )

        if "flash-lite" in model_lower:
            return str(
                getattr(
                    CFG,
                    "gemini_reasoning_effort_fast",
                    "low",
                )
            )

        return str(
            getattr(
                CFG,
                "gemini_reasoning_effort_balanced",
                "medium",
            )
        )

    def _openai_content(
        self,
        prompt: str,
        image_paths: List[str],
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": prompt}
        ]

        # Estimate the actual JSON payload contribution. Base64 expands raw
        # image bytes by roughly 4/3, so comparing only raw bytes would exceed
        # providers' inline-request limits.
        estimated_request_bytes = len(prompt.encode("utf-8"))

        for image_path in image_paths:
            path = Path(image_path)

            if not path.exists():
                continue

            raw = path.read_bytes()
            estimated_request_bytes += 4 * ((len(raw) + 2) // 3)
            encoded = base64.b64encode(raw).decode("utf-8")

            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            f"data:{self._image_media_type(path)};"
                            f"base64,{encoded}"
                        )
                    },
                }
            )

        maximum_inline_bytes = int(
            getattr(
                CFG,
                "api_max_inline_image_bytes",
                18 * 1024 * 1024,
            )
        )

        if estimated_request_bytes > maximum_inline_bytes:
            raise VLMResponseError(
                "Estimated inline request payload is too large: "
                f"{estimated_request_bytes:,} bytes exceed the configured "
                f"{maximum_inline_bytes:,}-byte safety limit."
            )

        return content

    def _anthropic_content(
        self,
        prompt: str,
        image_paths: List[str],
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []

        for image_path in image_paths:
            path = Path(image_path)

            if not path.exists():
                continue

            encoded = base64.b64encode(
                path.read_bytes()
            ).decode("utf-8")

            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": self._image_media_type(path),
                        "data": encoded,
                    },
                }
            )

        content.append(
            {"type": "text", "text": prompt}
        )

        return content

    def _record_api_debug(
        self,
        *,
        provider: str,
        model_id: str,
        response: Any,
        raw_content: Any,
        extracted_text: str,
        token_limit: int,
        image_count: int,
        retry_number: int,
        stop_reason: Any,
    ) -> None:
        if not getattr(CFG, "api_store_response_metadata", True):
            self.last_api_debug = {}
            return

        self.last_api_debug = {
            "provider": provider,
            "model_id": model_id,
            "response_id": getattr(response, "id", None),
            "stop_reason": (
                str(stop_reason)
                if stop_reason is not None
                else None
            ),
            "content_python_type": type(raw_content).__name__,
            "content_block_types": self._content_block_types(
                raw_content
            ),
            "extracted_answer_characters": len(extracted_text),
            "requested_max_tokens": token_limit,
            "image_count": image_count,
            "adapter_retry_number": retry_number,
            "usage": self._usage_dict(response),
        }

    def _parse_openai_response(
        self,
        response: Any,
        *,
        provider: str,
        model_id: str,
        token_limit: int,
        image_count: int,
        retry_number: int,
    ) -> str:
        choices = getattr(response, "choices", None) or []

        if not choices:
            raise VLMEmptyResponseError(
                f"{provider} returned no completion choices."
            )

        choice = choices[0]
        message = getattr(choice, "message", None)
        raw_content = getattr(message, "content", None)
        extracted_text = self._extract_text_content(raw_content)
        finish_reason = getattr(choice, "finish_reason", None)

        self._record_api_debug(
            provider=provider,
            model_id=model_id,
            response=response,
            raw_content=raw_content,
            extracted_text=extracted_text,
            token_limit=token_limit,
            image_count=image_count,
            retry_number=retry_number,
            stop_reason=finish_reason,
        )

        normalized_finish_reason = str(
            finish_reason or ""
        ).strip().lower()

        if normalized_finish_reason in {
            "length",
            "max_tokens",
            "model_context_window_exceeded",
        }:
            raise VLMTruncatedResponseError(
                f"{provider} stopped with finish_reason="
                f"{finish_reason!r} at max_tokens={token_limit}."
            )

        if not extracted_text:
            raise VLMEmptyResponseError(
                f"{provider} returned an empty final answer "
                f"(finish_reason={finish_reason!r}, "
                f"content_type={type(raw_content).__name__})."
            )

        return extracted_text

    def _parse_anthropic_response(
        self,
        response: Any,
        *,
        model_id: str,
        token_limit: int,
        image_count: int,
        retry_number: int,
    ) -> str:
        raw_content = getattr(response, "content", None)
        extracted_text = self._extract_text_content(raw_content)
        stop_reason = getattr(response, "stop_reason", None)

        self._record_api_debug(
            provider="anthropic",
            model_id=model_id,
            response=response,
            raw_content=raw_content,
            extracted_text=extracted_text,
            token_limit=token_limit,
            image_count=image_count,
            retry_number=retry_number,
            stop_reason=stop_reason,
        )

        normalized_stop_reason = str(
            stop_reason or ""
        ).strip().lower()

        if normalized_stop_reason in {
            "max_tokens",
            "model_context_window_exceeded",
        }:
            raise VLMTruncatedResponseError(
                "Anthropic stopped because the output cap was reached "
                f"at max_tokens={token_limit}."
            )

        if normalized_stop_reason == "refusal":
            raise VLMResponseError(
                "Anthropic returned a refusal stop reason."
            )

        if not extracted_text:
            raise VLMEmptyResponseError(
                "Anthropic returned no final text blocks "
                f"(stop_reason={stop_reason!r})."
            )

        return extracted_text

    def _answer_openai_compatible(
        self,
        *,
        provider: str,
        model_id: str,
        prompt: str,
        image_paths: List[str],
        token_limit: int,
    ) -> str:
        content = self._openai_content(
            prompt,
            image_paths,
        )

        request_arguments: Dict[str, Any] = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": content}
            ],
            "max_tokens": token_limit,
        }

        if provider == "gemini":
            request_arguments["reasoning_effort"] = (
                self._gemini_reasoning_effort(model_id)
            )

        response = self._client_for(
            model_id
        ).chat.completions.create(
            **request_arguments
        )

        return self._parse_openai_response(
            response,
            provider=provider,
            model_id=model_id,
            token_limit=token_limit,
            image_count=len(image_paths),
            retry_number=0,
        )

    def _answer_anthropic(
        self,
        *,
        model_id: str,
        prompt: str,
        image_paths: List[str],
        token_limit: int,
        retry_number: int,
    ) -> str:
        response = self._anthropic_client_for(
            model_id
        ).messages.create(
            model=model_id,
            max_tokens=token_limit,
            messages=[
                {
                    "role": "user",
                    "content": self._anthropic_content(
                        prompt,
                        image_paths,
                    ),
                }
            ],
        )

        return self._parse_anthropic_response(
            response,
            model_id=model_id,
            token_limit=token_limit,
            image_count=len(image_paths),
            retry_number=retry_number,
        )

    def _answer_api(
        self,
        prompt: str,
        image_paths: List[str],
        max_new_tokens: int,
    ) -> str:
        from .config import provider_of_model

        model_id = CFG.vlm_model_api
        provider = provider_of_model(model_id)
        token_limit = self._provider_token_limit(
            model_id,
            max_new_tokens,
        )
        token_ceiling = self._provider_token_ceiling(
            model_id
        )

        maximum_attempts = (
            2
            if getattr(
                CFG,
                "api_retry_on_truncation",
                True,
            )
            else 1
        )

        last_error: Optional[Exception] = None

        for retry_number in range(maximum_attempts):
            try:
                if provider == "anthropic":
                    return self._answer_anthropic(
                        model_id=model_id,
                        prompt=prompt,
                        image_paths=image_paths,
                        token_limit=token_limit,
                        retry_number=retry_number,
                    )

                content = self._openai_content(
                    prompt,
                    image_paths,
                )

                request_arguments: Dict[str, Any] = {
                    "model": model_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": content,
                        }
                    ],
                    "max_tokens": token_limit,
                }

                if provider == "gemini":
                    request_arguments[
                        "reasoning_effort"
                    ] = self._gemini_reasoning_effort(
                        model_id
                    )

                response = self._client_for(
                    model_id
                ).chat.completions.create(
                    **request_arguments
                )

                return self._parse_openai_response(
                    response,
                    provider=provider,
                    model_id=model_id,
                    token_limit=token_limit,
                    image_count=len(image_paths),
                    retry_number=retry_number,
                )

            except VLMTruncatedResponseError as exc:
                last_error = exc

                if retry_number + 1 >= maximum_attempts:
                    raise

                larger_limit = min(
                    token_ceiling,
                    max(
                        token_limit + 1,
                        int(
                            token_limit
                            * float(
                                getattr(
                                    CFG,
                                    "api_truncation_retry_multiplier",
                                    2.0,
                                )
                            )
                        ),
                    ),
                )

                if larger_limit <= token_limit:
                    raise

                log.warning(
                    "%s response was truncated at %d tokens; "
                    "retrying once with %d.",
                    provider,
                    token_limit,
                    larger_limit,
                )

                token_limit = larger_limit

        if last_error is not None:
            raise last_error

        raise VLMEmptyResponseError(
            f"{provider} did not return a usable answer."
        )

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

    def _filter_figures_api(
        self,
        question,
        indexed_figures,
        max_keep,
    ):
        from .config import provider_of_model

        prompt = self._filter_prompt(
            question,
            indexed_figures,
        )

        filter_model = (
            getattr(CFG, "vlm_model_filter", None)
            or CFG.vlm_model_api
        )

        provider = provider_of_model(filter_model)
        image_paths = [
            image_path
            for _index, image_path, _figure
            in indexed_figures
        ]

        # Figure filtering is a short classification task. Use a modest but
        # thinking-safe cap for Claude/Gemini, and low reasoning for Gemini.
        if provider == "anthropic":
            response = self._anthropic_client_for(
                filter_model
            ).messages.create(
                model=filter_model,
                max_tokens=2048,
                messages=[
                    {
                        "role": "user",
                        "content": self._anthropic_content(
                            prompt,
                            image_paths,
                        ),
                    }
                ],
            )
            response_text = self._parse_anthropic_response(
                response,
                model_id=filter_model,
                token_limit=2048,
                image_count=len(image_paths),
                retry_number=0,
            )
        else:
            request_arguments: Dict[str, Any] = {
                "model": filter_model,
                "messages": [
                    {
                        "role": "user",
                        "content": self._openai_content(
                            prompt,
                            image_paths,
                        ),
                    }
                ],
                "max_tokens": (
                    512
                    if provider == "gemini"
                    else 80
                ),
            }

            if provider == "gemini":
                request_arguments["reasoning_effort"] = "low"

            response = self._client_for(
                filter_model
            ).chat.completions.create(
                **request_arguments
            )

            response_text = self._parse_openai_response(
                response,
                provider=provider,
                model_id=filter_model,
                token_limit=request_arguments[
                    "max_tokens"
                ],
                image_count=len(image_paths),
                retry_number=0,
            )

        return self._parse_filter_response(
            response_text
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
