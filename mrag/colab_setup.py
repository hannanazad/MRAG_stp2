"""Colab-only setup helper.

Run this as the FIRST cell of the notebook on Colab:

    !pip install -q git+https://github.com/hannanazad/MRAG.git
    from mrag.colab_setup import setup
    cfg = setup(drive_subdir="MRAG")

It will:
  1. Mount Google Drive at /content/drive
  2. Make sure /content/drive/MyDrive/<drive_subdir>/ exists
  3. Point HF cache + Qdrant at the right places (Drive for HF, /content for Qdrant)
  4. Sync any previously-built Qdrant DB from Drive into /content/qdrant_db
  5. Verify the GPU is an A100 (or warn if it isn't)
  6. Print a summary
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def setup(
    drive_subdir: str = "MRAG",
    require_a100: bool = True,
    skip_drive_mount: bool = False,
    hf_cache_on_drive: bool = False,
    shared_drive: bool = False,
) -> "object":
    """Top-of-notebook setup. Returns the populated CFG object.

    By default keeps the 35 GB Hugging Face model cache on Colab's local
    `/content/` disk (re-downloaded each session, ~6 min) so total Drive
    usage stays around 6 GB. Set `hf_cache_on_drive=True` if you have
    plenty of Drive quota and want models cached forever.

    Set `shared_drive=True` if the target folder lives under
    `/content/drive/Shareddrives/<drive_subdir>/` instead of `MyDrive/`.
    """
    print("=" * 70)
    print(f"MRAG Colab setup")
    print("=" * 70)

    # 1. Mount Drive --------------------------------------------------------
    drive_root = Path("/content/drive")
    if not skip_drive_mount and not (drive_root / "MyDrive").exists():
        try:
            from google.colab import drive  # type: ignore
            print("Mounting Google Drive at /content/drive ...")
            drive.mount(str(drive_root))
        except ImportError:
            raise RuntimeError(
                "google.colab not importable — are you sure this is Colab? "
                "If you're running locally, use the regular pipeline instead."
            )

    drive_parent = "Shareddrives" if shared_drive else "MyDrive"
    drive_dir = drive_root / drive_parent / drive_subdir
    drive_dir.mkdir(parents=True, exist_ok=True)
    print(f"Drive folder: {drive_dir}")

    # Required input PDF
    pdfs = sorted(drive_dir.glob("*.pdf"))
    if not pdfs:
        print(
            f"\nWARNING: no *.pdf found in {drive_dir}. "
            f"Upload the MUTCD PDF there before running ingestion."
        )
    else:
        print(f"PDF: {pdfs[0].name}  ({pdfs[0].stat().st_size / 1e6:.1f} MB)")

    # 2. Env vars for HF cache --------------------------------------------
    # Default: ephemeral cache on /content. Saves ~35 GB of Drive quota
    # at the cost of ~6 minutes per session re-downloading models. The
    # ingestion artefacts (~6 GB) still go on Drive — that's what matters.
    if hf_cache_on_drive:
        hf_cache = drive_dir / "hf_cache"
        cache_note = "(persistent on Drive — uses ~35 GB)"
    else:
        hf_cache = Path("/content") / "hf_cache"
        cache_note = "(ephemeral — re-downloads ~28 GB each session, ~6 min)"
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"]            = str(hf_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)
    os.environ["HF_HUB_CACHE"]       = str(hf_cache / "hub")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"]   = "false"
    os.environ["MRAG_BASE_DIR"] = str(drive_dir)
    print(f"HF cache: {hf_cache}  {cache_note}")

    # 3. Sync Qdrant from Drive (if a snapshot exists) -------------------
    drive_qdrant_zip = drive_dir / "qdrant_db.tar"
    local_qdrant = Path("/content/qdrant_db")
    local_qdrant.parent.mkdir(parents=True, exist_ok=True)
    if drive_qdrant_zip.exists():
        print(f"Restoring Qdrant snapshot from Drive ({drive_qdrant_zip.stat().st_size / 1e6:.0f} MB) ...")
        t0 = time.time()
        if local_qdrant.exists():
            shutil.rmtree(local_qdrant)
        local_qdrant.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["tar", "-xf", str(drive_qdrant_zip), "-C", str(local_qdrant.parent)],
            check=True,
        )
        print(f"  restored in {time.time() - t0:.1f}s -> {local_qdrant}")
    else:
        print("No Qdrant snapshot on Drive yet (this is the first run — that's OK).")

    # 4. GPU check --------------------------------------------------------
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            print("\nWARNING: no GPU detected.")
        else:
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            ok = "A100" in name or "H100" in name or mem >= 38
            mark = "OK" if ok else "WARN"
            print(f"GPU [{mark}]: {name} ({mem:.1f} GB)")
            if require_a100 and not ok:
                print(
                    "  This GPU is too small for the full stack (Qwen2.5-VL-7B + "
                    "ColQwen2 + BGE-M3 + mxbai-rerank = ~26 GB). "
                    "Either switch to a Pro+ A100 runtime or set "
                    "mrag.config.CFG.vlm_model to the 3B fallback."
                )
    except ImportError:
        print("torch not installed yet; will be installed via requirements.txt")

    # 5. Force-reimport CFG so paths reflect MRAG_BASE_DIR. ----------------
    if "mrag.config" in sys.modules:
        del sys.modules["mrag.config"]
    if "mrag" in sys.modules:
        del sys.modules["mrag"]
    from mrag.config import CFG  # noqa: E402

    print()
    print("Resolved paths:")
    print(f"  base_dir   : {CFG.base_dir}")
    print(f"  pdf_path   : {CFG.pdf_path}  exists? {CFG.pdf_path.exists()}")
    print(f"  cache_dir  : {CFG.cache_dir}")
    print(f"  qdrant_dir : {CFG.qdrant_dir}")
    print(f"  HF_HOME    : {os.environ.get('HF_HOME')}")
    print("=" * 70)
    return CFG


def snapshot_qdrant_to_drive(drive_subdir: str = "MRAG") -> Path:
    """Tar the local Qdrant DB and copy it into Drive so the next session
    can restore it without re-running ingestion."""
    drive_dir = Path("/content/drive/MyDrive") / drive_subdir
    local_qdrant = Path("/content/qdrant_db")
    if not local_qdrant.exists():
        raise FileNotFoundError(local_qdrant)
    out = drive_dir / "qdrant_db.tar"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"tarring {local_qdrant} -> {out} ...")
    subprocess.run(
        ["tar", "-cf", str(out), "-C", str(local_qdrant.parent), local_qdrant.name],
        check=True,
    )
    print(f"  size: {out.stat().st_size / 1e6:.0f} MB")
    return out


def copy_artifacts_from_drive_to_runtime(drive_subdir: str = "MRAG") -> None:
    """Optional speed-up: copy frequently-read small artefacts to /content/
    so they hit local SSD instead of Drive."""
    drive_dir = Path("/content/drive/MyDrive") / drive_subdir
    local_dir = Path("/content") / drive_subdir
    local_dir.mkdir(parents=True, exist_ok=True)
    for name in ("mmrag_cache_v3", "figures", "page_images"):
        src = drive_dir / name
        dst = local_dir / name
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            print(f"copying {src} -> {dst} ...")
            shutil.copytree(src, dst)
