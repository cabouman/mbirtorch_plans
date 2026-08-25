#!/bin/bash
# mg63: the multiaxis forward's row-mask bound -- the remedy measured end to
# end, what Triton compiles at a divisible bound against a real one, where the
# threshold sits, and whether the back body's smaller residue is the same
# mechanism.  Follows mg62, which named the bound.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 01:00:00
#SBATCH -J torch_mg63
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg63_%j.log
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
export MG63_OUT=/scratch/gautschi/buzzard/torch_p3/results/mg63_multiaxis_row_bound.json
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
"$TORCHPY" -u mg63_multiaxis_row_bound.py
rc=$?
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu --format=csv
echo "MG63 DONE rc=$rc"
exit $rc
