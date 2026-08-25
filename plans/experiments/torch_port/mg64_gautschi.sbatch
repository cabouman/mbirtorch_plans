#!/bin/bash
# mg64: the multiaxis forward's row-bound change, verified on the edited tree.
# Two gates in one job.  The CUDA-only kernel tests run first, because they
# skip on every machine without a GPU and this is the only place they run at
# all; a failure there stops the job before it measures anything.  The probe
# then reads the compiled kernels, the values, and the times.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 01:30:00
#SBATCH -J torch_mg64
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg64_%j.log
set -x
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export MPLBACKEND=Agg
unset MBIRTORCH_NUM_DEVICES
unset PYTHONPATH
unset TORCHINDUCTOR_CACHE_DIR
unset TRITON_CACHE_DIR
nvidia-smi --query-gpu=index,name,clocks.sm,clocks.max.sm,temperature.gpu \
           --format=csv
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1

# Gate 1: the kernel tests, on hardware that can run them.  The multiaxis file
# is the one that changed; the cone and parallel files are here because all
# three share the Triton shims, and the sharded and adjoint suites are here
# because they exercise the bodies through the drivers.
cd /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -m pytest -q tests/test_triton_multiaxis.py tests/test_triton_cone.py \
    tests/test_triton_parallel.py tests/test_kernels_sharded.py \
    tests/test_multiaxis.py tests/test_adjoint.py
tests_rc=$?
echo "MG64 TESTS rc=$tests_rc"
if [ $tests_rc -ne 0 ]; then
  echo "MG64 DONE rc=$tests_rc (tests failed; no measurement taken)"
  exit $tests_rc
fi

# Gate 2: the probe.
export MG64_OUT=/scratch/gautschi/buzzard/torch_p3/results/mg64_row_bound_verify.json
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
"$TORCHPY" -u mg64_row_bound_verify.py
rc=$?
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu --format=csv
echo "MG64 DONE rc=$rc"
exit $rc
