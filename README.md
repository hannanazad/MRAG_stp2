# MRAG — MUTCD Multimodal RAG

A multimodal retrieval-augmented generation system over the **Manual on
Uniform Traffic Control Devices (MUTCD), 11th Edition**, designed to run
on TAMU HPRC (A100 / H100 GPUs).

## Architecture (v3)

```
PDF ─▶ outline-driven typed-paragraph chunks (Standard / Guidance / Option / Support)
   ─▶ caption-anchored figure / table crops
   ─▶ sign-code dictionary (R1-1 → "STOP sign", …) by category (Regulatory / Warning / …)
   ─▶ NetworkX knowledge graph
        ◦ Parts / Chapters / Sections / Chunks / Figures / SignCodes / Categories
        ◦ contains, cites_*, defines, depicts, mentions, illustrated_by, kind_of
   ─▶ Qdrant local-file store
        ◦ chunks   (BGE-M3 dense + sparse)
        ◦ figures  (BGE-M3 dense)
        ◦ pages    (ColQwen2-v0.1 multi-vector, binary-quantized)

query
  ─▶ hybrid retrieval (BGE-M3 dense + sparse, RRF)
  ─▶ scoring = α·dense + β·sparse + γ·hierarchy + δ·graph + ε·rule_type
  ─▶ mxbai-rerank-large-v2
  ─▶ figures via graph cross-links (+ caption retrieval fallback)
  ─▶ pages via ColPali MAX_SIM late-interaction
  ─▶ Qwen2.5-VL-7B-Instruct (3B fallback) with rule-type-structured prompt
  ─▶ structured answer: Standards / Guidance / Options / Visual evidence / Citations
```

Detailed design: [`docs/architecture.md`](docs/architecture.md).

## Quick start

Two equivalent paths. The notebook auto-detects which one you're on.

### Path A — Google Colab Pro+ with A100 (recommended)

Full walkthrough: [`COLAB_SETUP.md`](COLAB_SETUP.md).

1. Put `mutcd*.pdf` in `Drive/MyDrive/MRAG/`.
2. Open `MUTCD_MRAG_HPRC.ipynb` in Colab from this repo.
3. Set runtime to **A100 GPU**.
4. **Run all**. Cell 0 mounts Drive + pip installs + restores any Qdrant
   snapshot. The next cell runs ingestion once (~45 min). After that,
   subsequent sessions start in ~2 minutes.

### Path B — TAMU HPRC (Grace / FASTER / Launch / ACES)

Full walkthrough: [`HPRC_SETUP.md`](HPRC_SETUP.md).

1. `mutcd*.pdf` in `$SCRATCH/MRAG/`.
2. `source activate $SCRATCH/envs/mrag` (env created per the setup guide).
3. `sbatch scripts/ingest_v3.slurm`.
4. Open `MUTCD_MRAG_HPRC.ipynb` in OnDemand JupyterLab (A100, kernel
   `Python (mrag)`, modules `Anaconda3 WebProxy`), **Run All**.

Either way, you end up at the same `ask()` interface.

## Repo contents

| File / folder                | Purpose                                                                  |
| ---------------------------- | ------------------------------------------------------------------------ |
| `Copy_of_MRAG.ipynb`         | Original Colab notebook, preserved unmodified                             |
| `MUTCD_MRAG_HPRC.ipynb`      | The v3 notebook — auto-detects Colab / HPRC, ingestion + `ask()` UI       |
| `mrag/`                      | The Python package (parsing, KG, embeddings, retrieval, VLM, ask)         |
| `mrag/colab_setup.py`        | Colab-only helper: Drive mount, HF cache, Qdrant snapshot restore         |
| `scripts/extract_figures.py` | Standalone figure extractor (kept from v2 for offline use)               |
| `scripts/ingest_v3.py`       | One-shot ingestion driver                                                 |
| `scripts/ingest_v3.slurm`    | SLURM wrapper for HPRC                                                    |
| `requirements.txt`           | Pinned deps                                                              |
| `COLAB_SETUP.md`             | Colab Pro+ A100 walkthrough (recommended path)                            |
| `HPRC_SETUP.md`              | TAMU HPRC walkthrough                                                     |
| `docs/architecture.md`       | Full design, schema, scoring formula, justifications                      |
| `README.md`                  | This file                                                                |

## Module layout (`mrag/`)

```
mrag/
  __init__.py          - package version
  config.py            - all paths, model names, retrieval / scoring weights
  parsing.py           - PDF outline → typed-paragraph chunks
  figures.py           - caption-anchored figure / table cropping + page render
  sign_codes.py        - sign-code regex + canonical name mining + categorisation
  kg.py                - NetworkX MultiDiGraph build + query API
  embeddings.py        - BGE-M3 (text), ColQwen2 (image), mxbai-rerank wrappers
  vector_store.py      - Qdrant local-file wrapper (three collections)
  retrieval.py         - hybrid + graph expansion + scoring + rerank
  vlm.py               - Qwen2.5-VL-7B loader + structured prompt + generation
  ask.py               - public `ask()` façade, inline display
```
