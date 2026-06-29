#!/bin/bash
# ===========================================================================
# ingest_v3.slurm — MUTCD MRAG ingestion (v3) on TAMU HPRC
#
# Runs the full one-shot pipeline:
#   page renders + figure crops + chunks + sign codes + KG + BGE-M3 +
#   ColQwen2 page multivectors + Qdrant upsert.
#
# Submit:
#   cd $SCRATCH/MRAG && sbatch scripts/ingest_v3.slurm
# ===========================================================================
#SBATCH --job-name=mrag-ingest-v3
#SBATCH --output=logs/ingest-v3-%j.out
#SBATCH --error=logs/ingest-v3-%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1

set -euo pipefail
mkdir -p logs

echo "[$(date)] node=$HOSTNAME job=$SLURM_JOB_ID"

module purge
module load Anaconda3
module load WebProxy

export CONDA_ENVS_PATH=$SCRATCH/envs
export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

source activate $SCRATCH/envs/mrag

cd $SCRATCH/MRAG
python scripts/ingest_v3.py

echo "[$(date)] ingestion v3 complete"
