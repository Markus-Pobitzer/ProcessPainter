#!/bin/bash
#SBATCH -J ProcessPainter
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -o slurm_log/slurm-%j.out
#SBATCH -e slurm_log/slurm-%j.err

conda activate animatediff


torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/wlp_finetune/wlp_train_control.yaml