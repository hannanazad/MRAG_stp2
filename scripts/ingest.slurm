#!/bin/bash
# ===========================================================================
# ingest.slurm — headless MUTCD ingestion on TAMU HPRC (v2)
#
# Renders fallback page PNGs, extracts caption-anchored figure crops,
# builds page records, and computes section + page + figure-caption +
# figure-CLIP embeddings.
#
# Submit with:   sbatch scripts/ingest.slurm
# ===========================================================================
#SBATCH --job-name=mrag-ingest
#SBATCH --output=logs/ingest-%j.out
#SBATCH --error=logs/ingest-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail
mkdir -p logs
echo "[$(date)] node=$HOSTNAME job=$SLURM_JOB_ID"

module purge
module load Anaconda3
module load WebProxy

export CONDA_ENVS_PATH=$SCRATCH/envs
export CONDA_PKGS_DIRS=$SCRATCH/conda_pkgs
export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

source activate $SCRATCH/envs/mrag
cd $SCRATCH/MRAG

# 1. Figure extraction (PyMuPDF, CPU only).
python scripts/extract_figures.py \
    --pdf  $SCRATCH/MRAG/mutcd11theditionr1hl.pdf \
    --out  $SCRATCH/MRAG/figures \
    --json $SCRATCH/MRAG/figures.json \
    --dpi  220

# 2. Page records + all three embeddings (text + caption + CLIP).
python - <<'PY'
import os, json, re
from pathlib import Path
import numpy as np
import fitz
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

SCRATCH    = Path(os.environ["SCRATCH"])
BASE_DIR   = SCRATCH / "MRAG"
PDF_PATH   = BASE_DIR / "mutcd11theditionr1hl.pdf"
SECTIONS_JSON  = BASE_DIR / "mutcd_sections_with_images.json"
PAGE_IMAGE_DIR = BASE_DIR / "page_images"
FIGURES_JSON   = BASE_DIR / "figures.json"
CACHE_DIR  = BASE_DIR / "mmrag_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PAGE_DPI = 180

# Page renders (fallback for "what does page N show" questions).
PAGE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
doc = fitz.open(str(PDF_PATH))
mat = fitz.Matrix(PAGE_DPI/72.0, PAGE_DPI/72.0)
for i in tqdm(range(doc.page_count), desc="render pages"):
    out = PAGE_IMAGE_DIR / f"page_{i+1:04d}.png"
    if out.exists():
        continue
    doc.load_page(i).get_pixmap(matrix=mat, alpha=False).save(str(out))

# Page records
def norm(t): return re.sub(r"\s+", " ", t or "").strip()
fig_re  = re.compile(r"(Figure\s+[A-Za-z0-9.\-]+[^.\n]*)", re.IGNORECASE)
tbl_re  = re.compile(r"(Table\s+[A-Za-z0-9.\-]+[^.\n]*)",  re.IGNORECASE)
sgn_re  = re.compile(r"\b([A-Z]{1,3}\d{1,2}[A-Za-z0-9\-]*)\b")

records = []
for i, page in enumerate(tqdm(doc, desc="page records")):
    txt = page.get_text("text") or ""
    records.append({
        "page_num":        i+1,
        "image_path":      str(PAGE_IMAGE_DIR / f"page_{i+1:04d}.png"),
        "page_text":       txt,
        "page_text_norm":  norm(txt),
        "figure_captions": [norm(x) for x in fig_re.findall(txt)[:3]],
        "table_titles":    [norm(x) for x in tbl_re.findall(txt)[:3]],
        "sign_codes":      sorted(set(sgn_re.findall(txt)))[:50],
    })
with open(CACHE_DIR / "page_records.json", "w") as f:
    json.dump(records, f, ensure_ascii=False)

# Embeddings
with open(SECTIONS_JSON) as f:
    raw = json.load(f)
section_texts = [
    f"Section {s.get('section','')}. {s.get('title','')}. {str(s.get('text',''))[:2000]}"
    for s in raw
]
page_texts = [
    f"Page {p['page_num']}. Figures: {' | '.join(p['figure_captions'])}. "
    f"Tables: {' | '.join(p['table_titles'])}. {p['page_text'][:3500]}"
    for p in records
]
with open(FIGURES_JSON) as f:
    figures = json.load(f)
fig_caption_texts = [f"{x['figure_id']}. {x.get('title') or x['caption']}" for x in figures]

text_embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("encoding sections...")
np.save(CACHE_DIR / "section_embeddings.npy",
        text_embed.encode(section_texts, convert_to_numpy=True,
                          normalize_embeddings=True, show_progress_bar=True).astype("float32"))
print("encoding pages...")
np.save(CACHE_DIR / "page_embeddings.npy",
        text_embed.encode(page_texts, convert_to_numpy=True,
                          normalize_embeddings=True, show_progress_bar=True).astype("float32"))
print("encoding figure captions...")
np.save(CACHE_DIR / "figure_caption_embeddings.npy",
        text_embed.encode(fig_caption_texts, convert_to_numpy=True,
                          normalize_embeddings=True, show_progress_bar=True).astype("float32"))

print("encoding figure images with CLIP...")
clip = SentenceTransformer("sentence-transformers/clip-ViT-B-32")
imgs = []
for x in tqdm(figures, desc="open figs"):
    try:
        imgs.append(Image.open(x["image_path"]).convert("RGB"))
    except Exception as e:
        print("skip", x["image_path"], e)
        imgs.append(Image.new("RGB", (32,32), "white"))
np.save(CACHE_DIR / "figure_clip_embeddings.npy",
        clip.encode(imgs, batch_size=16, convert_to_numpy=True,
                    normalize_embeddings=True, show_progress_bar=True).astype("float32"))
print("DONE. cache at:", CACHE_DIR)
PY

echo "[$(date)] ingestion complete"
