#!/bin/bash
#SBATCH -A staff
#SBATCH -p gpupart
#SBATCH -J ref_frame
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH -o slurm_log/slurm-%j.out
#SBATCH -e slurm_log/slurm-%j.err

conda activate animatediff


torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/wlp_finetune/wlp_train_control.yaml