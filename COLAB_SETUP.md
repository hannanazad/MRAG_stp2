# Running MUTCD MRAG on Google Colab (A100)

The simplest way to run this. No HPRC, no conda, no module loads.

## Prerequisites

1. **Colab Pro+ subscription** ($50/month). Pro alone almost never allocates an A100; Pro+ does reliably. With Pro+ you get ~500 compute units/month ≈ 38 hours of A100 time.
2. **Google Drive** with at least ~10 GB free. **A TAMU edu account's 25 GB quota is plenty** with the default config — see the storage layout below. We keep the 35 GB Hugging Face model cache off Drive and re-download each session (~6 min on Colab's fast network).
3. **MUTCD PDF** uploaded to `Drive/MyDrive/MRAG/` (any filename ending in `.pdf` works). If your personal Drive quota is too tight even at 10 GB, use a **Shared Drive** instead — TAMU allows this; pass `shared_drive=True` to the setup helper.

## One-time setup (~10 minutes of clicking, then ~45 minutes of ingestion)

### Step 1 — put the PDF in Drive

In your Google Drive, create a folder `MRAG` at the top of "My Drive". Upload `mutcd11theditionr1hl.pdf` (or whatever you called it) into that folder.

Final layout:

```
Drive / MyDrive / MRAG /
  mutcd11theditionr1hl.pdf       ← only the PDF is required
```

The notebook will create the rest.

### Step 2 — open the notebook in Colab

Open <https://colab.research.google.com> → **File → Open Notebook → GitHub** → paste:

```
hannanazad/MRAG
```

Pick `MUTCD_MRAG_HPRC.ipynb` (the filename still says HPRC for historical reasons; it auto-detects Colab).

### Step 3 — pick an A100 runtime

**Runtime → Change runtime type → Hardware accelerator: A100 GPU**.

Save. The runtime will restart.

Verify with a one-liner cell:

```python
import torch; print(torch.cuda.get_device_name(0))
# expected: NVIDIA A100-SXM4-40GB
```

If it says T4 or L4, change the runtime type to A100 and try again. If Colab can't allocate one right now, wait a minute and retry — A100 availability fluctuates.

### Step 4 — run cell 0 (Colab setup)

This is the first code cell in the notebook. It will:

1. Mount Google Drive (you'll get a popup asking permission — accept).
2. Clone the MRAG repo into `/content/MRAG`.
3. `pip install` the v3 dependencies (~2 minutes).
4. Set `HF_HOME` to `Drive/MyDrive/MRAG/hf_cache` so models persist between sessions.
5. Check that your GPU is an A100.

You should see something like:

```
======================================================================
MRAG Colab setup
======================================================================
Mounting Google Drive at /content/drive ...
Drive folder: /content/drive/MyDrive/MRAG
PDF: mutcd11theditionr1hl.pdf  (32.1 MB)
HF cache: /content/drive/MyDrive/MRAG/hf_cache
No Qdrant snapshot on Drive yet (this is the first run — that's OK).
GPU [OK]: NVIDIA A100-SXM4-40GB (39.4 GB)

Resolved paths:
  base_dir   : /content/drive/MyDrive/MRAG
  pdf_path   : /content/drive/MyDrive/MRAG/mutcd11theditionr1hl.pdf  exists? True
  cache_dir  : /content/drive/MyDrive/MRAG/mmrag_cache_v3
  qdrant_dir : /content/qdrant_db
  HF_HOME    : /content/drive/MyDrive/MRAG/hf_cache
======================================================================
```

### Step 5 — run the ingestion cell (~30–45 min)

The cell that says "Ingestion (run ONCE)" will:

1. Render 1,162 page PNGs at 180 DPI → `Drive/MyDrive/MRAG/page_images/`
2. Crop ~314 figures and tables → `Drive/MyDrive/MRAG/figures/`
3. Parse the PDF outline into ~5,800 typed-paragraph chunks → `mmrag_cache_v3/chunks.jsonl`
4. Mine the sign-code dictionary → `sign_codes.json`
5. Build the knowledge graph (NetworkX pickle, ~5 MB)
6. **Download the four model checkpoints** (~35 GB, into `Drive/MyDrive/MRAG/hf_cache/` — this takes 5–8 min on first run, instant after)
7. BGE-M3 dense+sparse embeddings on chunks + figures (~5 min on A100)
8. ColQwen2 multi-vector embeddings on all pages (~20 min on A100 — this is the slow step)
9. Upsert everything into Qdrant at `/content/qdrant_db/`

**Don't let the Colab tab go idle during ingestion.** Colab kills the kernel after ~90 minutes of no activity. Easiest fix: wiggle your mouse in the Colab tab once every 30 min, or open the JS console (F12) and run:

```javascript
function keepAlive() { document.querySelector("colab-toolbar-button[command='run-cell']")?.click(); }
setInterval(keepAlive, 60000);
```

(Just don't leave that running forever — it'll burn your compute units.)

### Step 6 — snapshot Qdrant to Drive

Run the cell labelled "snapshot the Qdrant DB to Drive". It tars `/content/qdrant_db/` and writes `Drive/MyDrive/MRAG/qdrant_db.tar`. Takes ~30 s. From now on, every new Colab session restores the database from Drive in 30 s instead of re-running 45 min of ingestion.

### Step 7 — run the rest of the notebook

The cells after that load BGE-M3 / ColQwen2 / mxbai-rerank / Qwen2.5-VL-7B (~60 s on warm Drive HF cache; ~5 min on first run while downloading) and define the `ask()` function. Then try:

```python
ask("Explain Figure 2B-1 and the plaques it shows", show_scores=True)
ask("What is required when installing a STOP sign at an all-way stop?")
ask("pedestrian hybrid beacon signal sequence", show_text=True)
```

Each call: ~3–5 s of generation. Answer + figure crops shown inline.

## Daily workflow (after the one-time setup)

1. Open the notebook on Colab.
2. **Runtime → Change runtime type → A100 GPU** (if it isn't already).
3. **Runtime → Run all** — cell 0 restores the Qdrant snapshot, the init cell loads models from cached Drive HF, and you're at `ask()` in about 2–3 minutes.
4. Ask questions. When done, **Runtime → Disconnect and delete runtime** to stop burning compute units.

## Caveats and gotchas

| Symptom | Fix |
| --- | --- |
| GPU is a T4 / L4 / V100 not A100 | Runtime → Change runtime type → A100. May need to wait for capacity. |
| `CUDA out of memory` on VLM load | You got a 20 GB A100 instead of a 40 GB one. Either set `CFG.vlm_model = "Qwen/Qwen2.5-VL-3B-Instruct"` (one line in cell 0) or disconnect and reconnect to get a different machine. |
| Drive quota full during model download | Your `Drive/MyDrive/MRAG/hf_cache/` filled past your Drive limit. Free up space, or remove the cache and re-download fewer models (e.g., skip the 3B fallback). |
| Colab session dies after ~90 min idle | Expected. Re-open the tab, **Run all** — cell 0 restores everything from Drive snapshot. |
| `colpali_engine` import errors | Sometimes `pip install` order matters. Restart runtime and run cell 0 again. |
| Qdrant lock errors on second run | A previous kernel holds the DB open. **Runtime → Restart runtime**. |

## Disk usage summary (default config, TAMU 25 GB friendly)

| Location | Size | Persistence |
| --- | --- | --- |
| `Drive/MyDrive/MRAG/*.pdf` | ~32 MB | permanent |
| `Drive/MyDrive/MRAG/page_images/` | ~3 GB | permanent |
| `Drive/MyDrive/MRAG/figures/` | ~500 MB | permanent |
| `Drive/MyDrive/MRAG/mmrag_cache_v3/` | ~50 MB | permanent |
| `Drive/MyDrive/MRAG/qdrant_db.tar` | ~2 GB | permanent (snapshot) |
| `/content/hf_cache/` | ~35 GB | **session-local** — re-downloaded each session (~6 min) |
| `/content/qdrant_db/` | ~2 GB | session-local, restored from Drive in ~30 s |
| `/content/MRAG/` | ~50 MB | session-local, cloned from GitHub |

**Total Drive consumption: ~6 GB.** Fits comfortably in a 25 GB quota.

### If you have plenty of Drive (e.g. Google One 100 GB)

Pass `hf_cache_on_drive=True` to `setup()` and the 35 GB model cache will persist on Drive too. You'll save ~6 minutes per session.

### Using a Shared Drive instead of My Drive

If your personal quota is too tight, create a Shared Drive called `MRAG`, upload the PDF there, and call:

```python
CFG = setup(drive_subdir="MRAG", shared_drive=True)
```

Shared Drives have separate quotas and aren't counted against your personal storage.

## When to switch back to HPRC

You should consider HPRC again if:

- You want to run the pipeline for **many hours per day** without burning Colab compute units.
- You want a **persistent service** (e.g., expose a Streamlit UI to colleagues 24/7) — Colab notebooks aren't designed for that.
- You're processing **multiple manuals** (MUTCD + HCM + AASHTO + state DOT) and the total HF model cache exceeds your Drive quota.

For *this* project (single MUTCD, interactive querying, research-scale usage), Colab Pro+ is the right call.
