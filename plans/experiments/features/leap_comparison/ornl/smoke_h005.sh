#!/bin/bash
# Smoke test of both arms on one GPU inside the running interactive job:
# every eighth view and a 4 x 4 detector block average, 100 iterations each.
# The first arm also builds the weights cache the full-size jobs read.
source ~/load_conda_cuda.sh
ROOT=/scratch/gautschi/buzzard/leap_ornl
export TORCHINDUCTOR_CACHE_DIR=$ROOT/torch_cache
export MPLBACKEND=Agg
cd $ROOT || exit 1
PY=$ROOT/venv/bin/python
nvidia-smi -L
echo "=== STEP BEGIN mbirtorch smoke $(date)"
$PY -u run_mbirtorch.py --view-step 8 --det-step 4 --iterations 100 --tag smoke --save-volume
echo "=== STEP END mbirtorch smoke rc=$? $(date)"
echo "=== STEP BEGIN leap smoke $(date)"
$PY -u run_leap.py --view-step 8 --det-step 4 --iterations 100 --tag smoke --save-volume
echo "=== STEP END leap smoke rc=$? $(date)"
echo "SMOKE DONE"
