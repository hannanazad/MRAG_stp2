# Running the MUTCD MRAG (v3) on TAMU HPRC

This guide assumes the v2 environment (`$SCRATCH/envs/mrag`) is already set
up. If you're starting from scratch, do the conda env steps in §3 first;
otherwise skip straight to §4 to install the v3 dependencies.

The v3 pipeline introduces a knowledge graph, hybrid retrieval (BGE-M3
dense + sparse), ColPali visual page retrieval (ColQwen2), and the
mxbai-rerank-v2 cross-encoder. **Hardware floor is now an A100 40 GB**
(Qwen2.5-VL-7B + ColQwen2 + BGE-M3 + mxbai-rerank totals ~26 GB VRAM).

---

## 1. Cluster choice and OnDemand portal

| Cluster | GPU options          | Recommended for v3 |
| ------- | -------------------- | ------------------ |
| FASTER  | T4 / A100-40         | Use A100-40 only.  |
| Launch  | H100-80 / A30-24     | H100 ideal; A30 too tight. |
| Grace   | A100-40/80 / RTX6000 | A100 ideal.        |
| ACES    | H100 / A100 / PVC    | H100 or A100.      |

Portals:

- Grace:  <https://portal-grace.hprc.tamu.edu>
- FASTER: <https://portal-faster.hprc.tamu.edu>
- Launch: <https://portal-launch.hprc.tamu.edu>
- ACES:   <https://portal-aces.hprc.tamu.edu>

## 2. Upload the PDF to scratch

```
$SCRATCH/MRAG/
  mutcd11theditionr1hl.pdf      ← the only required input
```

If your file has a different name, that's fine: `mrag.config` will pick up
the first `*.pdf` in `$SCRATCH/MRAG/`.

## 3. (Only if you don't already have it) Create the conda env

On a login node (login nodes have outbound internet without WebProxy):

```bash
module purge
module load Anaconda3
module load WebProxy

export CONDA_ENVS_PATH=$SCRATCH/envs
export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
mkdir -p "$CONDA_ENVS_PATH" "$HF_HOME"

conda create -y -p $SCRATCH/envs/mrag python=3.11
source activate $SCRATCH/envs/mrag        # NOT `conda activate`; HPRC's Lmod
                                          # Anaconda module doesn't run conda init.

# Torch FIRST (matched to CUDA 12.1 — works on every HPRC GPU image I've tested):
pip install --index-url https://download.pytorch.org/whl/cu121 \
            torch==2.4.1 torchvision==0.19.1
```

## 4. Install the v3 deps

```bash
cd $SCRATCH/MRAG
git pull        # pulls the v3 pipeline + this guide
pip install -r requirements.txt
python -m ipykernel install --user --name mrag --display-name "Python (mrag)"
```

This brings in:
- `FlagEmbedding` (BGE-M3)
- `colpali-engine` (ColQwen2)
- `mxbai-rerank` (reranker)
- `qdrant-client` (vector DB)
- `networkx` (knowledge graph)
- and the existing `pymupdf`, `transformers`, `accelerate`, etc.

## 5. Pre-cache the four model checkpoints (one-time, on a login node)

```bash
source $SCRATCH/MRAG/env.sh   # the helper from §3 of the v2 guide; or set env vars inline
python - <<'PY'
from huggingface_hub import snapshot_download
for m in (
    "BAAI/bge-m3",
    "vidore/colqwen2-v0.1",
    "mixedbread-ai/mxbai-rerank-large-v2",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",      # fallback
):
    print("snapshot:", m)
    snapshot_download(m)
print("done.")
PY
```

Total download: ~35 GB into `$HF_HOME`.

## 6. Run ingestion (one-time, ~30–60 min)

Two options. Either runs the full pipeline: page renders + figure crops +
typed-paragraph chunks + sign-code dictionary + KG + BGE-M3 embeddings +
ColQwen2 page multivectors + Qdrant upsert.

### 6a. SLURM (preferred)

```bash
cd $SCRATCH/MRAG
sbatch scripts/ingest_v3.slurm
squeue -u $USER
tail -f logs/ingest-v3-*.out
```

The SLURM script requests `--gres=gpu:a100:1` and 3 h walltime. Adjust if
your cluster names A100s differently.

### 6b. Interactive on an OnDemand JupyterLab session

If you already have a JupyterLab session open with an A100:

```bash
cd $SCRATCH/MRAG
python scripts/ingest_v3.py
```

You can pass `--skip-pages` while developing if you want to avoid the
~15-min ColQwen2 step; page retrieval will be silently disabled at query
time until you re-run with pages included.

## 7. Use the notebook

OnDemand → **Interactive Apps → JupyterLab**:

- Modules: `Anaconda3 WebProxy`
- Conda env: `/scratch/user/<NetID>/envs/mrag`
- GPU: **1 × A100** (or H100)
- Memory: 48 GB
- Walltime: 2–4 h

Open `MUTCD_MRAG_HPRC.ipynb`, kernel = **Python (mrag)**, **Run All**.

In any cell:

```python
from mrag.ask import ask
ask("What is required when installing a STOP sign at an all-way stop intersection?")
ask("Explain Figure 2B-1 and the plaques it shows", show_scores=True)
ask("R1-3P all-way plaque")
```

Each call returns an answer Markdown + the actual figure crops the model
used, shown inline. No gradio, no proxy.

## 8. Troubleshooting

- **`CondaError: Run 'conda init' before 'conda activate'`** — use
  `source activate $SCRATCH/envs/mrag` instead of `conda activate ...`.
- **`OSError: ... No space left on device` during model download** — your
  `HF_HOME` is pointing at `$HOME`. Re-run with `HF_HOME=$SCRATCH/hf_cache`
  exported.
- **`CUDA out of memory` on VLM load** — your GPU isn't an A100. Either
  re-launch JupyterLab with an A100, or edit `mrag/config.py` to set
  `vlm_model = "Qwen/Qwen2.5-VL-3B-Instruct"` for the smaller fallback.
- **ColPali `colpali_engine` import error** — `pip install colpali-engine`
  inside the env. Some versions of `transformers` require ≥4.49.
- **`qdrant_client.QdrantClient` storage lock errors** — you can only have
  one `QdrantClient(path=…)` open at a time. If the notebook says the
  database is locked, restart the kernel.
- **Ingestion takes forever** — the ColQwen2 page-embedding step is the
  slowest (~1 s per page × 1162 pages on an A100). Run it as a SLURM job;
  the rest is sub-minute.

## 9. What's where on disk

```
$SCRATCH/MRAG/
  mutcd*.pdf
  page_images/page_NNNN.png        ← fallback page renders
  figures/<figure_id>_pNNNN.png    ← per-figure crops
  mmrag_cache_v3/
    chunks.jsonl                    ← one row per typed paragraph
    figures.jsonl                   ← figure metadata
    sign_codes.json                 ← sign-code dictionary
    graph.gpickle                   ← NetworkX KG
  qdrant_db/                        ← Qdrant local-file collections
  scripts/
    ingest_v3.py
    ingest_v3.slurm
  mrag/                             ← the Python package
  MUTCD_MRAG_HPRC.ipynb             ← the notebook
  docs/architecture.md              ← detailed design + scoring formula
```

To re-run ingestion from scratch, delete `mmrag_cache_v3/` and `qdrant_db/`.

## 10. After your session

```bash
# Optional: back up artefacts before scratch's purge window.
rsync -avh hannan_123@grace-dtn1.hprc.tamu.edu:/scratch/user/hannan_123/MRAG/{mmrag_cache_v3,qdrant_db}/ ./mrag_backup/
```

In the OnDemand dashboard, click **Delete** on the JupyterLab card when
you're done so you stop consuming walltime.
