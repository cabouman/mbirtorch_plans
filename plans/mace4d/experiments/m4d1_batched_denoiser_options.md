# m4d1: three ways to denoise a stack of volumes

## What the script does

MACE4D denoises P independent 3D volumes with one shared set of qGGMRF
denoiser constants.  This script measures three ways to do that in torch:
the production single-image sweep run once per volume (the reference loop),
a copy of the sweep with an explicit leading batch dimension (Option D), and
the production sweep run unchanged on the volumes stacked along the row axis
(Option A, in three variants).  It reports how far each option lands from the
reference loop and how long each one takes.

No file in the mbirtorch package was changed.  The batched functions in the
script are copies of the package formulas with every gather moved to
dimension 1.

## Command

```
cd "/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch" && \
/Users/gbuzzard/miniforge3/envs/mbirtorch/bin/python \
  "/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch_plans/plans/mace4d/experiments/m4d1_batched_denoiser_options.py" \
  2>&1 | tee "/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch_plans/plans/mace4d/experiments/results/m4d1_batched_denoiser_options.txt"
```

## Environment

Python 3.11.15, torch 2.13.0, numpy 2.4.6.  Machine: Apple M3 Max, 103 GB
memory, macOS.  Devices used: cpu and mps.  CUDA is not available.
`torch.get_num_threads()` reported 10.  Compilation was on
(`compile_mode='auto'`), and no compiled body fell back to eager on either
device.

## Correctness case

Case: P=8, T=6, D1=24, D2=24.  Shared constants: sigma_y 0.05,
sigma_x 0.021969125, fm_constant 400, 2 subsets of 72 pixels,
stop threshold 0.002.

Relative difference from the reference loop, measured as
max|y - y_ref| / max|y_ref| in float64:

| comparison | cpu | mps |
| --- | --- | --- |
| Option D | 0.000000e+00 | 1.071438e-07 |
| Option A1 | 2.673708e-02 | 2.673708e-02 |
| Option A1, interior frames t=1..T-2 | 6.244551e-03 | 6.244535e-03 |
| Option A1, edge frames t=0 and t=T-1 | 2.673708e-02 | 2.673708e-02 |
| Option A2 | 2.669324e-02 | 2.669324e-02 |
| Option A3 | 2.792681e-02 | 2.792681e-02 |

Exactness checks:

| check | cpu | mps |
| --- | --- | --- |
| iteration counts match lane by lane | True | True |
| batched gradient equals the package value exactly | True | True |
| batched Hessian equals the package value exactly | True | True |
| single-lane batch equals the reference bitwise, eager | True | True |
| single-lane batch equals the reference bitwise, compiled | True | False |

The reference iteration counts were [4, 4, 4, 5, 5, 5, 5, 5] and the batched
counts were the same list, on both devices.  The one failing bitwise check,
the compiled single-lane batch on mps, differed from the reference by
5.952200e-08 relative.

Option A ran 5 global iterations in every variant on both devices.  A1 and A3
used the per-volume partition repeated across volumes, which gives 2 subsets.
A2 drew a fresh partition of the stacked image, which gives 16 subsets.

## Timing case

Case: P=64, T=8, D1=64, D2=64, 8 subsets of 64 pixels.  P was not reduced;
the estimated block total was 4.3 s on each device, well under the 600 s
budget.  Each number is the mean of 2 timed runs after one warm-up run.

| form | cpu seconds | mps seconds |
| --- | --- | --- |
| reference loop | 0.719 | 0.716 |
| Option D | 0.141 | 0.032 |
| Option A1 | 0.119 | 0.342 |

Warm-up times, which include compilation:

| form | cpu seconds | mps seconds |
| --- | --- | --- |
| reference loop | 3.122 | 2.482 |
| Option D | 3.400 | 2.503 |
| Option A1 | 2.697 | 0.442 |

Option D ran 4 iterations on the fastest volume, 6 on the slowest, and 5.25
on average, on both devices.  Option A1 ran 6 global iterations on both
devices.

## What the numbers say

Option D reproduces the reference loop exactly on cpu and to 1.07e-07 on mps.
The per-volume iteration counts also match, so zeroing the step of a converged
volume stops that volume where a separate loop would stop it.

The batched gradient and Hessian equal the package values bit for bit on both
devices.  A single-lane batch also equals the reference bit for bit when both
run eager, so a reduction over `dim=(1, 2)` gives the same value as
`torch.sum` over the whole array at this size.  Under `torch.compile` on mps
the single-lane batch differs from the reference by 5.95e-08 relative, so the
compiler changes that reduction on mps.

All three Option A variants land about 2.7 percent away from the reference.
For A1 the difference on the first and last frame of each volume is 2.67e-02
and the difference on the interior frames is 6.24e-03, a factor of 4.3.
Adding a zero separator row between volumes (A3) did not reduce the
difference; A3 came out slightly larger than A1.  A fresh partition of the
stacked image (A2) gave about the same difference as the repeated partition.

On cpu Option D took 0.141 s against 0.719 s for the reference loop, a factor
of 5.1, and Option A1 took 0.119 s, a factor of 6.0.  On mps Option D took
0.032 s, a factor of 22, and Option A1 took 0.342 s, a factor of 2.1.  Option
A1 is the fastest form on cpu and the slower of the two batched forms on mps.

The reference loop took nearly the same time on both devices, 0.719 s on cpu
and 0.716 s on mps.  At this problem size that loop makes 64 x 5.25 x 8, or
about 2690, subset calls, and it reads the image norm back to the host after
each of its 336 volume iterations.

Option D runs 6 outer iterations because its slowest volume needs 6.  The
volumes average 5.25 iterations, so 12.5 percent of the lane iterations that
Option D executes are on volumes that have already stopped.

Nothing failed and nothing was skipped.
