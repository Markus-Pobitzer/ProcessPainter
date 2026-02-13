#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH -J eval_proess_painter
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o slurm_log/slurm-%j.out
#SBATCH -e slurm_log/slurm-%j.err

# !! NOTE:
# 1. activate env
# 2. sbatch ths script

conda activate animatediff

SPLIT="test"
DATASET_DIR="${1%/}"
OUTPUT_DIR="${2%/}"

python -m eval.run_eval \
    --input_directory "$DATASET_DIR" \
    --output_directory "$OUTPUT_DIR" \
    --config configs/eval/speedpainting-cn.yaml
