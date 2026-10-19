#!/bin/bash
# GPU smoke test (gpu_check.py).
# Usage: scp gpu_check.py perlmutter.sbatch perlmutter:gpu_check/  (mkdir it first), then
#   ssh perlmutter "cd gpu_check && sbatch --wait perlmutter.sbatch && cat gpu_check_*.out"
# The debug QoS gives a whole node, so all 4 GPUs show up (and are charged) whatever --gpus says.
# Verified 2026-10-19: 7.5 min wait, 4x A100-SXM4-80GB, torch 2.13.0+cu130, ~18.6 TFLOP/s per GPU.
#SBATCH -A m5193_g -C gpu -q debug -N 1 --gpus=2 -c 64 -t 00:05:00
#SBATCH -J gpu_check -o gpu_check_%j.out
module load pytorch/2.13.0
hostname
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
python gpu_check.py
