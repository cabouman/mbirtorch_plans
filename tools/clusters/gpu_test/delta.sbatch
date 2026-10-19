#!/bin/bash
# GPU smoke test (gpu_check.py).
# Usage: scp gpu_check.py delta.sbatch delta:gpu_check/  (mkdir it first), then
#   ssh delta "cd gpu_check && sbatch --wait delta.sbatch && cat gpu_check_*.out"
# Verified 2026-10-19: 2 min wait, 2x A100-SXM4-40GB, torch 2.12.1+cu130, ~19 TFLOP/s per GPU.
#SBATCH -A bjda-delta-gpu -p gpuA100x4-interactive -N 1 --gpus-per-node=2 --cpus-per-task=16 --mem=64g -t 00:05:00
#SBATCH -J gpu_check -o gpu_check_%j.out
module load pytorch-conda/2.12
hostname
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
python gpu_check.py
