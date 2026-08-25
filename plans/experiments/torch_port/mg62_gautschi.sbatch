#!/bin/bash
# mg62: why the multiaxis forward projector costs 3.3 times more at the
# non-dividing size (513x449x385) than at the dividing one (512x448x384).
# One H100 is enough: every cell is single-device and the whole probe is a few
# minutes of warm projections plus Triton's own report on the compiled kernels.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 01:00:00
#SBATCH -J torch_mg62
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg62_%j.log
set -x
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export MPLBACKEND=Agg
unset MBIRTORCH_NUM_DEVICES
unset PYTHONPATH
unset TORCHINDUCTOR_CACHE_DIR
unset TRITON_CACHE_DIR
# The clock and temperature reading a throttled card would show up in.  The
# dashboard row this run explains was measured on a card at 1980 MHz and 39 C.
nvidia-smi --query-gpu=index,name,clocks.sm,clocks.max.sm,temperature.gpu,persistence_mode \
           --format=csv
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
export MG62_OUT=/scratch/gautschi/buzzard/torch_p3/results/mg62_multiaxis_fwd_nondividing.json
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
"$TORCHPY" -u mg62_multiaxis_fwd_nondividing.py
rc=$?
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu --format=csv
echo "MG62 DONE rc=$rc"
exit $rc
