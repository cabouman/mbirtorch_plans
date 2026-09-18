#!/bin/bash
# cu_check.sh -- does each conda env still initialise CUDA on this cluster, and does the CUDA/cudnn
# MODULE version matter?  The same tiny GPU workloads run under three module configurations:
#   C0  the account's ~/load_conda_cuda.sh as it is
#   A   modtree/gpu + conda only (cuda/cudnn unloaded; no ptxas on PATH)
#   B   cuda/13.3.0 pinned (the marked default), no cudnn module
# Identical output in all three = the envs run on their pip-bundled CUDA and only the DRIVER matters.
# Run after a driver update, an RCAC module change, or a torch/jax bump.  First result 2026-09-03
# (job 15855388, h006): identical in all three; see .claude/cluster_use.md, "The node preamble".
#
# Usage (the log lands in the submission directory as cu_check-<jobid>.log):
#   mkdir -p /scratch/gautschi/$USER/cluster_use_check && cd /scratch/gautschi/$USER/cluster_use_check
#   sbatch <path to>/cu_check.sh
# Env selection: TORCH_ENVS (space-separated, default "mbirtorch pcdrecon") and JAX_ENV (default
# mbirjax_regression; legacy, skipped when absent).  Override with --export, e.g.
#   sbatch --export=ALL,TORCH_ENVS="mbirtorch" cu_check.sh
#SBATCH -A bouman -p ai -N1 --gpus-per-node=1 --cpus-per-task=14 -t 0:08:00 -J cu_check
#SBATCH -o cu_check-%j.log
source /etc/profile
export HTTPS_PROXY=squid.rcac.purdue.edu:3128 HTTP_PROXY=squid.rcac.purdue.edu:3128
TORCH_ENVS=${TORCH_ENVS:-"mbirtorch pcdrecon"}
JAX_ENV=${JAX_ENV:-mbirjax_regression}
PIN=${CUDA_PIN:-cuda/13.3.0}
echo "node=$(hostname)  user=$USER  date=$(date -Is)"; nvidia-smi | grep -E "Driver Version" | head -1
check() {
  echo; echo "=== $1 ==="
  echo "modules: $(module list 2>&1 | grep -oE '(cuda|cudnn|conda|modtree)/[0-9A-Za-z.-]+' | tr '\n' ' ')"
  echo "ptxas on PATH: $(command -v ptxas || echo none)"
  for e in $TORCH_ENVS; do
    [ -x ~/.conda/envs/$e/bin/python ] || { echo "  [$e] no such env -- skipped"; continue; }
    ~/.conda/envs/$e/bin/python - <<'PY' 2>&1
import torch, triton, importlib
a = torch.randn(256, 256, device='cuda'); s = (a @ a).sum().item()
m = importlib.import_module('mbirtorch')
print(f"torch={torch.__version__} build_cuda={torch.version.cuda} avail={torch.cuda.is_available()} "
      f"dev={torch.cuda.get_device_name(0)} matmul_finite={s == s} triton={triton.__version__} mbirtorch={m.__file__}")
PY
    echo "  [$e] exit=$?"
  done
  if [ -x ~/.conda/envs/$JAX_ENV/bin/python ]; then
    ~/.conda/envs/$JAX_ENV/bin/python - <<'PY' 2>&1
import jax, jax.numpy as jnp
d = jax.devices(); v = float(jnp.ones(1024).sum())
print(f"jax={jax.__version__} devices={d} sum={v}")
PY
    echo "  [$JAX_ENV] exit=$?"
  else
    echo "  [$JAX_ENV] no such env -- skipped (legacy)"
  fi
}
module --force purge; source ~/load_conda_cuda.sh;                                          check "C0 control: ~/load_conda_cuda.sh as it is"
module --force purge; module load modtree/gpu conda; module unload cuda cudnn 2>/dev/null; check "A: modtree/gpu + conda only, cuda/cudnn unloaded"
module --force purge; module load modtree/gpu conda; module load "$PIN" 2>/dev/null;        check "B: $PIN pinned, no cudnn module"
echo; echo "CU_CHECK_DONE"
