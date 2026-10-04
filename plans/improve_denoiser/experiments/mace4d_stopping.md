# GPU test of the denoiser's stopping rule in MACE4D

Status: set up 2026-10-04.  The runs wait for an allocation on gautschi.

Greg asked on 2026-10-04 for timing and results on GPUs and real data before
he chooses the stopping rule of `findings/step4_stopping_rule.md`.  This record
describes the test.  It compares the stopping rules of the qGGMRF stack
denoisers inside MACE4DModel on the Lilly 4DCT phantom scan.  It also tests
option 2 of the findings page, which sets the denoisers' noise level from a
robust estimate.

## Versions

Every version is one MACE4D reconstruction.  The versions differ in the
library, in the denoiser's stopping rule, in the denoisers' noise level, or in
the denoiser warm start.

| Version | mbirtorch | Denoiser stopping rule | Noise level of the denoisers | Warm start |
|---|---|---|---|---|
| A | `prerelease` at da7fc90 | MACE4D's rule: a change below 0.05 percent, or 15 iterations | automatic | off |
| B | `greg_dev` at 5f62506 | MACE4D's rule | automatic | off |
| C | `greg_dev` plus `gradient_rule.patch` | the gradient statistic below 0.01, or 200 iterations | automatic | off |
| D | as C | as C | option 2 | off |
| Bw | as B | as B | automatic | on |
| Cw | as C | as C | automatic | on |

Versions A and B differ only in the fix of the denoiser's update from PR #14
of the mbirtorch repository.  The other commits between the two branches
change the 4D viewer and its documentation.

The patch `gradient_rule.patch` adds the gradient rule to `denoise` and
`denoise_stack` of `mbirtorch/denoising.py` in the mbirtorch repository.  Each
subset update also returns the sum of the squared gradient of the cost at its
voxels, taken before the update.  A new argument, `stop_threshold_gradient`,
stops a volume when sigma_y times the rms gradient over one iteration falls
below the threshold.  Without the argument, the patched library computes the
same images as `greg_dev`.  Three checks on a CPU support this:
- With the percent rule, the patched `denoise` returned the same image as
  `greg_dev`, to the last bit.
- The statistic agreed with an independent computation to a relative
  difference of 2e-6.
- `denoise_stack` with the gradient rule gave the same images and iteration
  counts as `denoise` on each volume.
The tests `tests/test_denoiser.py`, `tests/test_mace.py`, and
`tests/test_mace4d.py` of the mbirtorch repository pass on the patched library.

The patch exists only for this test.  It is not a proposal for the code.

## Setup

The data is the 30 s phantom scan,
`/depot/bouman/data/Lilly/4DCT/Phantom_30s_Run1_Dec2024/`, at full resolution.
MACE4D divides it into 99 frames of 260 by 260 by 728 voxels, with 48 views per
frame.

Each run uses one gautschi node with 4 H100s and 56 cores, inside one
allocation of 12 hours.  Torch uses the 56 cores, because the run script
unsets `OMP_NUM_THREADS` and `MKL_NUM_THREADS`.  Each run compiles with its
own empty compile caches, so every run pays the same compile time in its
first iteration.

The reconstruction settings are those of the Lilly driver of
mbirtorch_applications (`nsi_4d/Lilly_recon_4d.py`), with two exceptions.
The MACE stop threshold is 0, so every run does 10 MACE iterations.  The seed
is 0, so every run draws the same partitions.

All runs start from one initial image, a direct reconstruction (FDK) of each
frame.  A run of version B with 0 MACE iterations computes it and caches it
before the other runs.  Versions A and B compute the same initial image; on
the synthetic scan of `mace4d_stopping.py`, the two agreed to the last bit.
Because no run of the comparison computes the initial image, the wall times
and the peak device memory of the runs compare directly.

Option 2's noise level comes from `robust_sigma.py`.  The current estimate
reads a strided subsample of the initial image.  On this volume the stride is
10, so the estimate compares voxels 10 apart.  The robust estimate uses
adjacent voxels in the central half of 25 frames.  It is the median absolute
difference of adjacent voxels divided by \( 0.6745 \sqrt{2} \).

## Scripts

- `mace4d_stopping.py` runs one version.  It records each denoiser call and
  keeps a few sampled calls.
- `robust_sigma.py` computes option 2's noise level and three related values.
- `frame_check.py` runs `denoise` on one frame of the initial image, on one
  GPU, for versions A to D.
- `mace4d_calls.py` computes the MAP estimate of each sampled call and the
  distance of the call's output from it.  It runs one process per GPU.
- `mace4d_compare.py` compares the runs: time, iteration counts, final
  images, and memory.
- `mace4d_stopping.sh` runs the stages on gautschi.

## Setup on gautschi

The folder is `/scratch/gautschi/buzzard/denoiser_stop/`.  It holds these
items:
- `src/`, a clone of mbirtorch.  It is not named `mbirtorch/`, because a
  folder of that name in the working folder would hide the package.
- `plans/`, a clone of mbirtorch_plans on branch
  `claude/improve-qggmrf-denoiser-cuvgs1`.
- `wt_A`, `wt_B`, and `wt_C`, detached worktrees of `src` at da7fc90, at
  5f62506, and at 5f62506 with the patch applied.
- `venv_A`, `venv_B`, and `venv_C`, virtual environments over Greg's conda
  environment `pcdrecon` (Python 3.12.14, torch 2.14.0+cu130).  The conda
  environment `mbirtorch` of the September runs no longer exists.  Each
  virtual environment has an editable install of its worktree in pip's
  compat mode, because `pcdrecon` holds a regular install of mbirtorch.  The
  scripts stop when mbirtorch is not imported from the expected worktree.

The stages run inside an allocation, except `setup_c`, which runs on the
login node.  From the login node, in the folder above, the command is:

    srun --jobid=<job> --overlap -N 1 -n 1 --cpus-per-task=56 --gpus=4 \
        bash plans/plans/improve_denoiser/experiments/mace4d_stopping.sh <stage> < /dev/null

The stages are `setup_c`, `smoke`, `init`, `sigma`, `A`, `B`, `C`, `D`,
`Bw`, `Cw`, `frame`, `calls`, `compare`, and `collect`.  The stage `smoke`
runs the others at a small size, and it also runs on one GPU.  The stage
`core` runs `init` to `collect` without the warm-start runs, `extra` runs the
warm-start runs and repeats the last three stages, and `all` runs both.
After a failure, `core_from <stage>` and `extra_from <stage>` resume the
two lists at a given stage.  Each stage writes its log to
`runs/logs/<stage>.log` in the folder above.

## Results

The results will go here, with the files under
`experiments/results/mace4d_stopping/`.
