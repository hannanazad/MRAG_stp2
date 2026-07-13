"""Resumable, model-agnostic runner for MUTCD-150-v1.0.

Designed to be imported *after* the user's existing MUTCD RAG notebook has
initialized ``CFG``, ``pipeline``, and ``ask``.  The runner never opens the
evaluator gold file.  It accepts only the question-only JSONL.

Canonical outputs in each run directory:
  - answers_<run_id>.jsonl          successful model answers only
  - retrieval_<run_id>.jsonl        captured RAG debug/display evidence, no image bytes
  - manifest_<run_id>.json          environment, hashes, model registry, progress
  - errors_<run_id>.jsonl           terminal question/model-switch failures

The implementation is deliberately defensive because different versions of
``mrag.ask.ask`` may return a string, dictionary, dataclass, custom object, or
only render Markdown in the notebook.  The exact displayed answer is captured
without requiring access to private RAG internals.
"""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import datetime as _dt
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence

RUNNER_VERSION = "1.1.0"
RUN_SCHEMA_VERSION = "1.1"
EXPECTED_BENCHMARK_VERSION = "MUTCD-150-v1.0"
EXPECTED_QUESTION_COUNT = 150
EXPECTED_QUESTIONS_SHA256 = "3a04b1d620a80704eefac34c565449a0cb8814e781dd6d73b8afb77318b954b2"

# Fields that must never appear in the file supplied to the RAG runner.
_GOLD_ONLY_KEYS = {
    "gold_answer",
    "acceptable_answer_variants",
    "required_answer_elements",
    "source_pdf_page",
    "source_pdf_pages",
    "printed_manual_page",
    "printed_manual_pages",
    "section",
    "sections",
    "paragraph",
    "paragraphs",
    "figure_or_table_id",
    "required_evidence",
    "critical_error_conditions",
    "scoring_method",
    "answerable",
    "normative_type",
}

_ANSWER_KEYS = (
    "answer",
    "final_answer",
    "response",
    "generated_answer",
    "output_text",
    "completion",
    "content",
)

_USAGE_KEYS = {
    "input_tokens": ("input_tokens", "prompt_tokens"),
    "output_tokens": ("output_tokens", "completion_tokens"),
    "total_tokens": ("total_tokens",),
    "estimated_cost_usd": ("estimated_cost_usd", "cost_usd"),
}


@dataclass(frozen=True)
class ModelSpec:
    """A model entry independent of the benchmark itself.

    ``selector`` is passed directly to ``CFG.set_vlm_model`` and can be a
    catalog alias (for example ``flagship``) or a raw provider model ID.
    """

    alias: str
    selector: str
    provider: Optional[str] = None
    enabled: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_any(cls, value: Any) -> "ModelSpec":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(alias=value, selector=value)
        if isinstance(value, Mapping):
            selector = str(value.get("selector") or value.get("model_id") or value.get("alias") or "").strip()
            alias = str(value.get("alias") or selector).strip()
            if not selector or not alias:
                raise ValueError(f"Invalid model specification: {value!r}")
            return cls(
                alias=alias,
                selector=selector,
                provider=value.get("provider"),
                enabled=bool(value.get("enabled", True)),
                metadata=dict(value.get("metadata") or {}),
            )
        raise TypeError(f"Unsupported model specification: {type(value).__name__}")


@dataclass
class RunnerConfig:
    questions_path: Path
    output_root: Path
    run_id: str
    models: Sequence[ModelSpec]
    prompt_style: Optional[str] = "fewshot"
    replicates: int = 1
    max_questions: Optional[int] = None
    question_ids: Optional[Sequence[str]] = None
    show_scores: bool = True
    show_text: bool = True
    max_attempts: int = 3
    retry_base_seconds: float = 4.0
    inter_request_seconds: float = 0.0
    seed: int = 20260710
    resume: bool = True
    rerun_errors: bool = True
    strict_benchmark_hash: bool = True
    strict_question_count: bool = True
    store_serialized_return: bool = True
    store_raw_debug: bool = True
    echo_answer_preview: bool = False

    def validate(self) -> None:
        self.questions_path = Path(self.questions_path)
        self.output_root = Path(self.output_root)
        if not self.run_id or not re.fullmatch(r"[A-Za-z0-9._-]+", self.run_id):
            raise ValueError("run_id may contain only letters, numbers, '.', '_', and '-'.")
        if self.replicates < 1:
            raise ValueError("replicates must be at least 1")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.max_questions is not None and self.max_questions < 1:
            raise ValueError("max_questions must be positive")
        if not self.models:
            raise ValueError("At least one model must be supplied")


@dataclass
class CapturedCall:
    return_value: Any
    stdout: str
    stderr: str
    display_outputs: list[dict[str, Any]]
    latency_ms: int


class VLMResponseError(RuntimeError):
    """Raised when the RAG wrapper returns an error string as if it were an answer."""


def is_vlm_error_answer(answer: Any) -> bool:
    if not isinstance(answer, str):
        return False
    text = answer.lstrip().lower()
    return text.startswith("(vlm error:") or text.startswith("vlm error:")


class JsonlWriter:
    """Append-only JSONL writer that flushes and fsyncs every record."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=False, default=_json_default)
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json_dump(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=False, default=_json_default)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (_dt.datetime, _dt.date)):
        return value.isoformat()
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return repr(value)


def _safe_serialize(value: Any, *, depth: int = 0, max_depth: int = 6, max_items: int = 200) -> Any:
    """Serialize unknown RAG return objects without retaining binary payloads."""
    if depth > max_depth:
        return {"__truncated__": True, "repr": repr(value)[:2000]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"__binary__": True, "nbytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, bytearray):
        b = bytes(value)
        return {"__binary__": True, "nbytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
    if dataclasses.is_dataclass(value):
        return _safe_serialize(dataclasses.asdict(value), depth=depth + 1, max_depth=max_depth, max_items=max_items)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for idx, (k, v) in enumerate(value.items()):
            if idx >= max_items:
                out["__remaining_items_truncated__"] = len(value) - max_items
                break
            out[str(k)] = _safe_serialize(v, depth=depth + 1, max_depth=max_depth, max_items=max_items)
        return out
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        out = [
            _safe_serialize(v, depth=depth + 1, max_depth=max_depth, max_items=max_items)
            for v in seq[:max_items]
        ]
        if len(seq) > max_items:
            out.append({"__remaining_items_truncated__": len(seq) - max_items})
        return out
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return _safe_serialize(value.model_dump(), depth=depth + 1, max_depth=max_depth, max_items=max_items)
        except Exception:
            pass
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return _safe_serialize(value.dict(), depth=depth + 1, max_depth=max_depth, max_items=max_items)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _safe_serialize(vars(value), depth=depth + 1, max_depth=max_depth, max_items=max_items)
        except Exception:
            pass
    return {"__type__": type(value).__name__, "repr": repr(value)[:10000]}


def _get_nested_by_key(obj: Any, candidate_keys: Iterable[str], max_depth: int = 4) -> Any:
    keys = {k.lower() for k in candidate_keys}

    def walk(x: Any, depth: int) -> Any:
        if depth > max_depth:
            return None
        if isinstance(x, Mapping):
            # Prefer exact keys at this level.
            for k, v in x.items():
                if str(k).lower() in keys and v is not None:
                    return v
            for v in x.values():
                result = walk(v, depth + 1)
                if result is not None:
                    return result
        elif isinstance(x, (list, tuple)):
            for v in x:
                result = walk(v, depth + 1)
                if result is not None:
                    return result
        elif hasattr(x, "__dict__"):
            return walk(vars(x), depth + 1)
        return None

    return walk(obj, 0)


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def _git_commit(repo_dir: Optional[Path]) -> Optional[str]:
    if not repo_dir:
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except Exception:
        return None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Benchmark loading and leakage guards
# ---------------------------------------------------------------------------

def load_questions(
    path: Path,
    *,
    strict_hash: bool = True,
    strict_count: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Question file not found: {path}")

    digest = sha256_file(path)
    if strict_hash and digest != EXPECTED_QUESTIONS_SHA256:
        raise ValueError(
            "Question-file SHA-256 does not match MUTCD-150-v1.0. "
            f"Expected {EXPECTED_QUESTIONS_SHA256}, received {digest}."
        )

    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_no} is not a JSON object")
            leaked = sorted(_GOLD_ONLY_KEYS.intersection(row.keys()))
            if leaked:
                raise ValueError(
                    f"Gold/evaluator fields found in runner input on line {line_no}: {leaked}. "
                    "Use mutcd_benchmark_questions_v1.jsonl, never the gold file."
                )
            qid = str(row.get("question_id", "")).strip()
            question = str(row.get("question", "")).strip()
            version = str(row.get("benchmark_version", "")).strip()
            if not qid or not question:
                raise ValueError(f"Line {line_no} lacks question_id or question")
            if qid in ids:
                raise ValueError(f"Duplicate question_id: {qid}")
            if version and version != EXPECTED_BENCHMARK_VERSION:
                raise ValueError(f"Unexpected benchmark_version on {qid}: {version}")
            ids.add(qid)
            rows.append(dict(row))

    if strict_count and len(rows) != EXPECTED_QUESTION_COUNT:
        raise ValueError(f"Expected {EXPECTED_QUESTION_COUNT} questions, found {len(rows)}")

    metadata = {
        "path": str(path),
        "sha256": digest,
        "question_count": len(rows),
        "benchmark_version": EXPECTED_BENCHMARK_VERSION,
    }
    return rows, metadata


# ---------------------------------------------------------------------------
# Capturing notebook-rendered answers without private pipeline access
# ---------------------------------------------------------------------------

def _serialize_display_output(output: Any) -> dict[str, Any]:
    data = getattr(output, "data", None)
    if data is None and isinstance(output, Mapping):
        data = output.get("data", output)
    data = dict(data or {})
    serialized: dict[str, Any] = {
        "metadata": _safe_serialize(getattr(output, "metadata", None) or {}),
        "transient": _safe_serialize(getattr(output, "transient", None) or {}),
        "data": {},
    }
    for mime, payload in data.items():
        if mime.startswith("image/") or mime in {"application/pdf", "application/octet-stream"}:
            if isinstance(payload, str):
                try:
                    binary = base64.b64decode(payload, validate=False)
                except Exception:
                    binary = payload.encode("utf-8", errors="replace")
            elif isinstance(payload, bytes):
                binary = payload
            else:
                binary = repr(payload).encode("utf-8", errors="replace")
            serialized["data"][mime] = {
                "binary_omitted": True,
                "nbytes": len(binary),
                "sha256": hashlib.sha256(binary).hexdigest(),
            }
        else:
            serialized["data"][mime] = _safe_serialize(payload)
    return serialized


def _display_object_to_record(obj: Any) -> dict[str, Any]:
    """Format an object passed to IPython ``display`` as a MIME bundle.

    This fallback is important when a library imported ``display`` directly
    into its own module namespace, because not every notebook runtime routes
    that call through ``capture_output`` in the same way.
    """
    data: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    try:
        from IPython import get_ipython  # type: ignore

        shell = get_ipython()
        if shell is not None and getattr(shell, "display_formatter", None) is not None:
            formatted, meta = shell.display_formatter.format(obj)
            data = dict(formatted or {})
            metadata = dict(meta or {})
    except Exception:
        pass

    if not data:
        # IPython.display.Markdown exposes the original string through .data.
        if type(obj).__name__ == "Markdown" and isinstance(getattr(obj, "data", None), str):
            data["text/markdown"] = obj.data
        elif hasattr(obj, "_repr_markdown_"):
            try:
                md = obj._repr_markdown_()
                if md is not None:
                    data["text/markdown"] = md
            except Exception:
                pass

    if not data and hasattr(obj, "_repr_png_"):
        try:
            png = obj._repr_png_()
            if png is not None:
                data["image/png"] = png
        except Exception:
            pass

    if not data:
        data["text/plain"] = repr(obj)

    return _serialize_display_output({"data": data, "metadata": metadata})


def _dedupe_display_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        digest = sha256_text(json.dumps(record, ensure_ascii=False, sort_keys=True, default=_json_default))
        if digest not in seen:
            seen.add(digest)
            out.append(record)
    return out


def capture_ask_call(
    ask_fn: Callable[..., Any],
    question: str,
    *,
    show_scores: bool,
    show_text: bool,
) -> CapturedCall:
    start = time.perf_counter()
    outputs: list[dict[str, Any]] = []
    intercepted_outputs: list[dict[str, Any]] = []
    stdout = ""
    stderr = ""

    # Intercept a module-local ``display`` reference when one exists. This is
    # non-invasive: it affects only presentation, never retrieval or generation.
    ask_module = sys.modules.get(getattr(ask_fn, "__module__", ""))
    original_module_display = getattr(ask_module, "display", None) if ask_module else None
    try:
        import IPython.display as _ipydisplay  # type: ignore
        original_global_display = getattr(_ipydisplay, "display", None)
    except Exception:
        _ipydisplay = None
        original_global_display = None

    def intercepted_display(*objects: Any, **kwargs: Any) -> None:
        for obj in objects:
            intercepted_outputs.append(_display_object_to_record(obj))

    patch_module_display = ask_module is not None and callable(original_module_display)
    patch_global_display = _ipydisplay is not None and callable(original_global_display)
    if patch_module_display:
        setattr(ask_module, "display", intercepted_display)
    if patch_global_display:
        setattr(_ipydisplay, "display", intercepted_display)

    try:
        # IPython is present in Colab/Jupyter. A plain contextlib fallback keeps
        # the runner usable in a normal Python process.
        try:
            from IPython.utils.capture import capture_output  # type: ignore

            with capture_output(stdout=True, stderr=True, display=True) as cap:
                result = ask_fn(question, show_scores=show_scores, show_text=show_text)
            stdout = cap.stdout or ""
            stderr = cap.stderr or ""
            outputs = [_serialize_display_output(o) for o in (cap.outputs or [])]
        except ImportError:
            import io

            out_buf, err_buf = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
                result = ask_fn(question, show_scores=show_scores, show_text=show_text)
            stdout, stderr = out_buf.getvalue(), err_buf.getvalue()
    finally:
        if patch_module_display:
            setattr(ask_module, "display", original_module_display)
        if patch_global_display:
            setattr(_ipydisplay, "display", original_global_display)

    outputs = _dedupe_display_records(outputs + intercepted_outputs)
    latency_ms = round((time.perf_counter() - start) * 1000)
    return CapturedCall(result, stdout, stderr, outputs, latency_ms)


def _all_display_text(captured: CapturedCall, mime: str) -> list[str]:
    values: list[str] = []
    for output in captured.display_outputs:
        payload = (output.get("data") or {}).get(mime)
        if isinstance(payload, str):
            values.append(payload)
        elif isinstance(payload, list):
            values.append("".join(str(v) for v in payload))
    return values


def extract_answer(captured: CapturedCall) -> tuple[str, str]:
    """Return ``(answer, extraction_method)`` while preserving model wording."""
    markdowns = _all_display_text(captured, "text/markdown")
    for md in markdowns:
        match = re.search(
            r"(?is)(?:^|\n)\s*#{2,4}\s*Answer\s*\n(.*?)(?=\n\s*#{2,4}\s*(?:Figures?|Retrieved|Debug|Context|Evidence)\b|\Z)",
            md,
        )
        if match and match.group(1).strip():
            return match.group(1).strip(), "display_markdown_answer_section"

    rv = captured.return_value
    if isinstance(rv, str) and rv.strip():
        return rv.strip(), "return_string"

    candidate = _get_nested_by_key(rv, _ANSWER_KEYS)
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip(), "return_object_answer_field"
    if isinstance(candidate, Mapping):
        nested = _get_nested_by_key(candidate, ("text", "content", "value"), max_depth=2)
        if isinstance(nested, str) and nested.strip():
            return nested.strip(), "return_object_nested_answer"

    # Occasionally a function only prints the answer.  Avoid treating verbose
    # retrieval logs as an answer unless a recognizable heading is present.
    match = re.search(
        r"(?is)(?:^|\n)\s*(?:answer|final answer)\s*[:\n]\s*(.*?)(?=\n\s*(?:retrieval|debug|figures?|context)\s*[:\n]|\Z)",
        captured.stdout,
    )
    if match and match.group(1).strip():
        return match.group(1).strip(), "stdout_answer_section"

    for md in markdowns:
        stripped = md.strip()
        if stripped and not stripped.lower().startswith("### q"):
            return stripped, "display_markdown_fallback"

    return "", "not_found"


def extract_figure_evidence(captured: CapturedCall) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    markdowns = _all_display_text(captured, "text/markdown")
    prefix = re.compile(r"^\s*\*\*\[Image\s+(?P<image_no>\d+)\]\*\*\s*(?P<rest>.*)$", re.I)
    for md in markdowns:
        for line in md.splitlines():
            m = prefix.match(line)
            if not m:
                continue
            rest = m.group("rest").strip()
            parts = re.split(r"\s+(?:—|–|-)\s+", rest, maxsplit=1)
            left = parts[0].strip()
            description = parts[1].strip() if len(parts) > 1 else ""
            page_match = re.search(r"\(p\.\s*(\d+)\)\s*$", left, re.I)
            page = int(page_match.group(1)) if page_match else None
            if page_match:
                left = left[: page_match.start()].strip()
            entries.append(
                {
                    "image_no": int(m.group("image_no")),
                    "label": left,
                    "pdf_or_printed_page_as_displayed": page,
                    "description": description,
                }
            )
    return entries


def extract_usage(return_value: Any) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    for canonical, candidates in _USAGE_KEYS.items():
        value = _get_nested_by_key(return_value, candidates, max_depth=5)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            usage[canonical] = value
    confidence = _get_nested_by_key(
        return_value,
        ("confidence", "confidence_score", "self_confidence", "model_confidence"),
        max_depth=4,
    )
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        usage["model_reported_confidence"] = float(confidence)
    return usage


# ---------------------------------------------------------------------------
# Manifest and resume helpers
# ---------------------------------------------------------------------------

def _load_completed(path: Path, rerun_errors: bool) -> set[tuple[str, str, int]]:
    completed: set[tuple[str, str, int]] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
                key = (str(row.get("model_id") or row["model_alias"]), str(row["question_id"]), int(row.get("replicate", 1)))
                status = row.get("status")
                if status == "ok" or (status == "error" and not rerun_errors):
                    completed.add(key)
            except Exception:
                # A corrupt trailing line should not make the entire run unusable.
                continue
    return completed


def _existing_answer_state(path: Path, rerun_errors: bool) -> tuple[set[tuple[str, str, int]], int, int]:
    """Return unique successful keys from an existing answers file.

    Runner v1.1 writes terminal failures only to errors JSONL. For backward
    compatibility, status=error records created by v1.0 are ignored so the same
    question can be retried without being treated as completed.
    """
    completed: set[tuple[str, str, int]] = set()
    if not path.exists():
        return completed, 0, 0
    historical_error_rows = 0
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
                key = (
                    str(row.get("model_id") or row["model_alias"]),
                    str(row["question_id"]),
                    int(row.get("replicate", 1)),
                )
                if row.get("status") == "ok":
                    completed.add(key)
                else:
                    historical_error_rows += 1
            except Exception:
                continue
    return completed, len(completed), historical_error_rows


def _cfg_snapshot(CFG: Any) -> dict[str, Any]:
    # Explicit whitelist avoids API keys and unrelated private state.
    attrs = (
        "environment",
        "base_dir",
        "pdf_path",
        "qdrant_dir",
        "cache_dir",
        "vlm_provider",
        "vlm_model",
        "vlm_model_api",
        "answer_style",
        "prompt_style",
        "top_k",
        "retrieval_top_k",
        "coll_chunks",
        "coll_figures",
    )
    snap: dict[str, Any] = {}
    for name in attrs:
        if hasattr(CFG, name):
            try:
                snap[name] = _safe_serialize(getattr(CFG, name))
            except Exception:
                pass
    return snap


def _initial_manifest(
    cfg: RunnerConfig,
    benchmark_meta: Mapping[str, Any],
    CFG: Any,
    selected_questions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    base_dir = Path(str(getattr(CFG, "base_dir", ".")))
    pdf_path_raw = getattr(CFG, "pdf_path", None)
    pdf_path = Path(str(pdf_path_raw)) if pdf_path_raw else None
    repo_candidates = [Path("/content/MRAG"), Path.cwd()]
    repo_dir = next((p for p in repo_candidates if (p / ".git").exists()), None)

    models = [dataclasses.asdict(m) for m in cfg.models if m.enabled]
    manifest: dict[str, Any] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "run_id": cfg.run_id,
        "benchmark": dict(benchmark_meta),
        "selected_question_count": len(selected_questions),
        "selected_question_ids_sha256": sha256_text("\n".join(str(q["question_id"]) for q in selected_questions)),
        "models_requested": models,
        "prompt_style_requested": cfg.prompt_style,
        "replicates": cfg.replicates,
        "seed": cfg.seed,
        "show_scores": cfg.show_scores,
        "show_text": cfg.show_text,
        "max_attempts": cfg.max_attempts,
        "inter_request_seconds": cfg.inter_request_seconds,
        "resume": cfg.resume,
        "started_at": utc_now(),
        "completed_at": None,
        "status": "running",
        "progress": {
            "planned_records": len(selected_questions) * cfg.replicates * len(models),
            "ok_records": 0,
            "error_records": 0,
            "skipped_existing_records": 0,
        },
        "source_pdf": {
            "path": str(pdf_path) if pdf_path else None,
            "sha256": sha256_file(pdf_path) if pdf_path and pdf_path.exists() else None,
        },
        "rag_repository": {
            "path": str(repo_dir) if repo_dir else None,
            "git_commit": _git_commit(repo_dir),
        },
        "rag_public_config_snapshot": _cfg_snapshot(CFG),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "packages": {
                name: _package_version(name)
                for name in (
                    "torch",
                    "transformers",
                    "qdrant-client",
                    "sentence-transformers",
                    "dashscope",
                )
            },
            "cwd": str(Path.cwd()),
            "base_dir": str(base_dir),
        },
    }
    return manifest


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_benchmark(
    *,
    CFG: Any,
    ask_fn: Callable[..., Any],
    questions_path: str | Path,
    output_root: str | Path,
    run_id: Optional[str] = None,
    models: Sequence[Any],
    prompt_style: Optional[str] = "fewshot",
    replicates: int = 1,
    max_questions: Optional[int] = None,
    question_ids: Optional[Sequence[str]] = None,
    show_scores: bool = True,
    show_text: bool = True,
    max_attempts: int = 3,
    retry_base_seconds: float = 4.0,
    inter_request_seconds: float = 0.0,
    seed: int = 20260710,
    resume: bool = True,
    rerun_errors: bool = True,
    strict_benchmark_hash: bool = True,
    strict_question_count: bool = True,
    store_serialized_return: bool = True,
    store_raw_debug: bool = True,
    echo_answer_preview: bool = False,
) -> dict[str, Path]:
    """Run all enabled models over the question-only benchmark.

    The function is sequential by design.  This protects rate limits, preserves
    a stable question order, and makes every record immediately durable.
    """
    model_specs = [ModelSpec.from_any(m) for m in models]
    model_specs = [m for m in model_specs if m.enabled]
    if run_id is None:
        run_id = _dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]

    rcfg = RunnerConfig(
        questions_path=Path(questions_path),
        output_root=Path(output_root),
        run_id=run_id,
        models=model_specs,
        prompt_style=prompt_style,
        replicates=replicates,
        max_questions=max_questions,
        question_ids=question_ids,
        show_scores=show_scores,
        show_text=show_text,
        max_attempts=max_attempts,
        retry_base_seconds=retry_base_seconds,
        inter_request_seconds=inter_request_seconds,
        seed=seed,
        resume=resume,
        rerun_errors=rerun_errors,
        strict_benchmark_hash=strict_benchmark_hash,
        strict_question_count=strict_question_count,
        store_serialized_return=store_serialized_return,
        store_raw_debug=store_raw_debug,
        echo_answer_preview=echo_answer_preview,
    )
    rcfg.validate()
    _seed_everything(rcfg.seed)

    questions, benchmark_meta = load_questions(
        rcfg.questions_path,
        strict_hash=rcfg.strict_benchmark_hash,
        strict_count=rcfg.strict_question_count,
    )
    if rcfg.question_ids:
        wanted = set(rcfg.question_ids)
        missing = wanted.difference(str(q["question_id"]) for q in questions)
        if missing:
            raise ValueError(f"Unknown question IDs: {sorted(missing)}")
        questions = [q for q in questions if str(q["question_id"]) in wanted]
    if rcfg.max_questions is not None:
        questions = questions[: rcfg.max_questions]

    run_dir = rcfg.output_root / rcfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    answers_path = run_dir / f"answers_{rcfg.run_id}.jsonl"
    retrieval_path = run_dir / f"retrieval_{rcfg.run_id}.jsonl"
    manifest_path = run_dir / f"manifest_{rcfg.run_id}.json"
    errors_path = run_dir / f"errors_{rcfg.run_id}.jsonl"

    answer_writer = JsonlWriter(answers_path)
    retrieval_writer = JsonlWriter(retrieval_path)
    error_writer = JsonlWriter(errors_path)

    if rcfg.resume:
        completed, existing_ok, existing_errors = _existing_answer_state(answers_path, rcfg.rerun_errors)
    else:
        completed, existing_ok, existing_errors = set(), 0, 0
    manifest = _initial_manifest(rcfg, benchmark_meta, CFG, questions)
    manifest["progress"]["ok_records"] = existing_ok
    manifest["progress"]["error_records"] = existing_errors
    manifest["output_files"] = {
        "answers_jsonl": str(answers_path),
        "retrieval_jsonl": str(retrieval_path),
        "errors_jsonl": str(errors_path),
        "manifest_json": str(manifest_path),
    }
    atomic_json_dump(manifest_path, manifest)

    if rcfg.prompt_style:
        if not hasattr(CFG, "set_answer_style"):
            raise AttributeError("CFG does not expose set_answer_style; cannot freeze prompt style")
        resolved_style = CFG.set_answer_style(rcfg.prompt_style)
        manifest["prompt_style_resolved"] = _safe_serialize(resolved_style)
        atomic_json_dump(manifest_path, manifest)

    total_planned = manifest["progress"]["planned_records"]
    global_index = 0

    try:
        for model in rcfg.models:
            print(f"\n=== Model {model.alias!r} (selector={model.selector!r}) ===")
            try:
                resolved_model_id = CFG.set_vlm_model(model.selector)
                resolved_model_id = str(resolved_model_id)
                provider = model.provider or str(getattr(CFG, "vlm_provider", "unknown"))
            except Exception as exc:
                failure = {
                    "schema_version": RUN_SCHEMA_VERSION,
                    "runner_version": RUNNER_VERSION,
                    "run_id": rcfg.run_id,
                    "stage": "model_switch",
                    "model_alias": model.alias,
                    "model_selector": model.selector,
                    "timestamp": utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                error_writer.append(failure)
                manifest.setdefault("model_switch_failures", []).append(failure)
                atomic_json_dump(manifest_path, manifest)
                print(f"Model switch failed: {exc}")
                continue

            manifest.setdefault("models_resolved", {})[model.alias] = {
                "selector": model.selector,
                "resolved_model_id": resolved_model_id,
                "provider": provider,
                "metadata": dict(model.metadata),
                "resolved_at": utc_now(),
            }
            atomic_json_dump(manifest_path, manifest)

            for replicate in range(1, rcfg.replicates + 1):
                for q_index, question_row in enumerate(questions, start=1):
                    global_index += 1
                    qid = str(question_row["question_id"])
                    question = str(question_row["question"])
                    key = (resolved_model_id, qid, replicate)
                    if key in completed:
                        manifest["progress"]["skipped_existing_records"] += 1
                        print(f"[{global_index}/{total_planned}] SKIP {model.alias} {qid} r{replicate}")
                        continue

                    print(f"[{global_index}/{total_planned}] RUN  {model.alias} {qid} r{replicate}")
                    call_started_at = utc_now()
                    captured: Optional[CapturedCall] = None
                    last_captured: Optional[CapturedCall] = None
                    extracted_answer = ""
                    extraction_method = "not_found"
                    last_exc: Optional[BaseException] = None
                    last_traceback: Optional[str] = None
                    attempt = 0

                    for attempt in range(1, rcfg.max_attempts + 1):
                        try:
                            captured = capture_ask_call(
                                ask_fn,
                                question,
                                show_scores=rcfg.show_scores,
                                show_text=rcfg.show_text,
                            )
                            last_captured = captured
                            extracted_answer, extraction_method = extract_answer(captured)
                            if is_vlm_error_answer(extracted_answer):
                                raise VLMResponseError(extracted_answer)
                            if not extracted_answer:
                                raise RuntimeError(
                                    "No answer could be extracted from return value or rendered output."
                                )
                            last_exc = None
                            last_traceback = None
                            break
                        except KeyboardInterrupt:
                            raise
                        except BaseException as exc:
                            last_exc = exc
                            last_traceback = traceback.format_exc()
                            captured = None
                            if attempt < rcfg.max_attempts:
                                delay = rcfg.retry_base_seconds * (2 ** (attempt - 1)) + random.random()
                                print(f"  attempt {attempt} failed ({type(exc).__name__}: {exc}); retrying in {delay:.1f}s")
                                time.sleep(delay)

                    common = {
                        "schema_version": RUN_SCHEMA_VERSION,
                        "runner_version": RUNNER_VERSION,
                        "benchmark_version": EXPECTED_BENCHMARK_VERSION,
                        "benchmark_questions_sha256": benchmark_meta["sha256"],
                        "run_id": rcfg.run_id,
                        "model_alias": model.alias,
                        "model_selector": model.selector,
                        "model_id": resolved_model_id,
                        "provider": provider,
                        "model_metadata": dict(model.metadata),
                        "prompt_style": rcfg.prompt_style,
                        "question_index_in_selected_set": q_index,
                        "question_id": qid,
                        "question": question,
                        "replicate": replicate,
                        "seed": rcfg.seed,
                        "started_at": call_started_at,
                        "completed_at": utc_now(),
                        "attempts": attempt,
                    }

                    if last_exc is not None or captured is None:
                        error_record = {
                            **common,
                            "status": "error",
                            "stage": "question_call",
                            "answer": extracted_answer,
                            "answer_sha256": sha256_text(extracted_answer),
                            "answer_extraction_method": extraction_method,
                            "latency_ms": last_captured.latency_ms if last_captured else None,
                            "usage": extract_usage(last_captured.return_value) if last_captured else {},
                            "error_type": type(last_exc).__name__ if last_exc else "UnknownError",
                            "error": str(last_exc) if last_exc else "Unknown error",
                            "traceback": last_traceback,
                        }
                        error_writer.append(error_record)
                        if rcfg.store_raw_debug and last_captured is not None:
                            retrieval_writer.append({
                                **common,
                                "status": "error",
                                "latency_ms": last_captured.latency_ms,
                                "stdout": last_captured.stdout,
                                "stderr": last_captured.stderr,
                                "display_outputs": last_captured.display_outputs,
                                "figure_evidence_parsed": extract_figure_evidence(last_captured),
                                "display_output_count": len(last_captured.display_outputs),
                                "display_image_count": sum(
                                    1
                                    for out in last_captured.display_outputs
                                    for mime in (out.get("data") or {})
                                    if str(mime).startswith("image/")
                                ),
                                "terminal_error_type": error_record["error_type"],
                                "terminal_error": error_record["error"],
                            })
                        manifest["progress"]["error_records"] += 1
                        print(f"  ERROR after {attempt} attempts: {error_record['error_type']}: {error_record['error']}")
                    else:
                        answer = extracted_answer
                        usage = extract_usage(captured.return_value)
                        record = {
                            **common,
                            "status": "ok",
                            "answer": answer,
                            "answer_sha256": sha256_text(answer),
                            "answer_extraction_method": extraction_method,
                            "latency_ms": captured.latency_ms,
                            "usage": usage,
                            "error_type": None,
                            "error": None,
                        }
                        if rcfg.store_serialized_return:
                            record["serialized_return"] = _safe_serialize(captured.return_value)
                        answer_writer.append(record)

                        figures = extract_figure_evidence(captured)
                        if rcfg.store_raw_debug:
                            debug_record = {
                                **common,
                                "status": "ok",
                                "latency_ms": captured.latency_ms,
                                "stdout": captured.stdout,
                                "stderr": captured.stderr,
                                "display_outputs": captured.display_outputs,
                                "figure_evidence_parsed": figures,
                                "display_output_count": len(captured.display_outputs),
                                "display_image_count": sum(
                                    1
                                    for out in captured.display_outputs
                                    for mime in (out.get("data") or {})
                                    if str(mime).startswith("image/")
                                ),
                            }
                            retrieval_writer.append(debug_record)

                        manifest["progress"]["ok_records"] += 1
                        if rcfg.echo_answer_preview:
                            preview = re.sub(r"\s+", " ", answer)[:240]
                            print(f"  OK {captured.latency_ms} ms — {preview}")
                        else:
                            print(f"  OK {captured.latency_ms} ms — {len(answer)} chars")

                    manifest["updated_at"] = utc_now()
                    atomic_json_dump(manifest_path, manifest)
                    gc.collect()
                    if rcfg.inter_request_seconds > 0:
                        time.sleep(rcfg.inter_request_seconds)

        manifest["status"] = "completed"
        manifest["completed_at"] = utc_now()
        atomic_json_dump(manifest_path, manifest)
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["completed_at"] = utc_now()
        manifest["interruption"] = "KeyboardInterrupt"
        atomic_json_dump(manifest_path, manifest)
        print("\nRun interrupted. Re-run with the same run_id and resume=True to continue.")
        raise
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["completed_at"] = utc_now()
        manifest["runner_failure"] = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_json_dump(manifest_path, manifest)
        raise

    print("\nRun complete")
    print(" answers :", answers_path)
    print(" retrieval:", retrieval_path)
    print(" manifest:", manifest_path)
    if errors_path.exists() and errors_path.stat().st_size:
        print(" errors  :", errors_path)
    return {
        "run_dir": run_dir,
        "answers": answers_path,
        "retrieval": retrieval_path,
        "manifest": manifest_path,
        "errors": errors_path,
    }


def validate_run_outputs(run_dir: str | Path, run_id: str) -> dict[str, Any]:
    """Perform structural checks and report unresolved terminal failures."""
    run_dir = Path(run_dir)
    answers_path = run_dir / f"answers_{run_id}.jsonl"
    retrieval_path = run_dir / f"retrieval_{run_id}.jsonl"
    manifest_path = run_dir / f"manifest_{run_id}.json"
    errors_path = run_dir / f"errors_{run_id}.jsonl"

    if not answers_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("answers or manifest file is missing")

    rows_by_key: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    with answers_path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            for required in ("run_id", "model_alias", "model_id", "question_id", "question", "replicate", "status", "answer"):
                if required not in row:
                    raise ValueError(f"Missing {required!r} on answers line {line_no}")
            key = (str(row.get("model_id") or row["model_alias"]), str(row["question_id"]), int(row["replicate"]))
            rows_by_key.setdefault(key, []).append(row)

    duplicate_ok_keys = [
        key for key, group in rows_by_key.items()
        if sum(row.get("status") == "ok" for row in group) > 1
    ]
    if duplicate_ok_keys:
        raise ValueError(f"Multiple successful records found for the same key: {duplicate_ok_keys[:5]}")

    effective_ok = {
        key: next(row for row in reversed(group) if row.get("status") == "ok")
        for key, group in rows_by_key.items()
        if any(row.get("status") == "ok" for row in group)
    }

    error_events: list[dict[str, Any]] = []
    if errors_path.exists():
        with errors_path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                if raw.strip():
                    error_events.append(json.loads(raw))
    question_error_keys = {
        (str(row.get("model_id") or row.get("model_alias")), str(row.get("question_id")), int(row.get("replicate", 1)))
        for row in error_events
        if row.get("question_id")
    }
    unresolved = sorted(question_error_keys.difference(effective_ok))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = {
        "run_id": run_id,
        "answer_records_raw": sum(len(v) for v in rows_by_key.values()),
        "successful_records": len(effective_ok),
        "historical_answer_duplicates": sum(max(0, len(v) - 1) for v in rows_by_key.values()),
        "error_events": len(error_events),
        "unresolved_error_records": len(unresolved),
        "unresolved_error_keys": unresolved,
        "unique_models": sorted({str(r["model_alias"]) for r in effective_ok.values()}),
        "unique_questions_successful": len({str(r["question_id"]) for r in effective_ok.values()}),
        "answers_sha256": sha256_file(answers_path),
        "retrieval_exists": retrieval_path.exists(),
        "retrieval_sha256": sha256_file(retrieval_path) if retrieval_path.exists() else None,
        "errors_exists": errors_path.exists(),
        "errors_sha256": sha256_file(errors_path) if errors_path.exists() and errors_path.stat().st_size else None,
        "manifest_status": manifest.get("status"),
        "manifest_sha256": sha256_file(manifest_path),
    }
    # Backward-compatible names used by existing notebook smoke assertions.
    summary["answer_records"] = summary["successful_records"]
    summary["ok_records"] = summary["successful_records"]
    summary["error_records"] = summary["unresolved_error_records"]
    return summary


__all__ = [
    "EXPECTED_BENCHMARK_VERSION",
    "EXPECTED_QUESTIONS_SHA256",
    "VLMResponseError",
    "is_vlm_error_answer",
    "ModelSpec",
    "RunnerConfig",
    "load_questions",
    "run_benchmark",
    "validate_run_outputs",
]
