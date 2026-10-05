# GPU test of the denoiser's stopping rule in MACE4D

Status: done.  The stage `smoke` passed on one A100 of gilbreth on
2026-10-04.  The runs took place on 2026-10-04 in gautschi job 16857514, and
every stage passed.  The results are at the end of this record.

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

The test uses these scripts:
- `mace4d_stopping.py` runs one version.  It records each denoiser call and
  keeps a few sampled calls.
- `robust_sigma.py` computes option 2's noise level and three related values.
- `frame_check.py` runs `denoise` on one frame of the initial image, on one
  GPU, for versions A to D.  Its version D uses the noise level of the MACE4D
  run D.
- `mace4d_calls.py` computes the MAP estimate of each sampled call and the
  distance of the call's output from it.  It runs one process per GPU.
- `mace4d_compare.py` compares the runs: time, iteration counts, final
  images, and memory.
- `image_noise.py` computes the adjacent-voxel statistic of the final
  images, as option 2 computes it for the initial image.
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
login node.  On another cluster, the same layout goes in another folder,
which the environment variable `DENOISER_STOP_ROOT` names.  From the login
node, in the folder above, the command is:

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

The runs took place on 2026-10-04 in gautschi job 16857514, on node h009
with 4 H100s of 80 GB.  Every stage passed.  The six runs, the one-frame
check, the call analysis, and the comparisons ran from 18:17 to 21:23 EDT.
They used 12.4 H100 GPU-hours.  The allocations of the test used about 29
GPU-hours in all.  The other 16.8 GPU-hours were idle time, mostly while the
Mac session that starts the stages was offline.

The result tables are in `results/mace4d_stopping/`.  That folder holds
these tables:
- `timing_summary.csv` and `timing_per_iteration.csv`: the time, the
  memory, and the consensus change of each run.
- `denoiser_iterations.csv`: the iteration counts of all calls, for each
  MACE iteration and orientation.
- `calls_summary.csv`: the sampled calls and the one-frame check, with their
  distances from their MAP estimates and the iteration at which each rule
  would stop.
- `trajectories.csv`: the distance and the two stopping statistics of each
  sampled call after each iteration of its replay.
- `frame_check.csv`: the times of the one-frame check.
- `recon_differences.csv`: the differences between the final images.
- `image_noise.csv` and `image_noise_warm.csv`: the adjacent-voxel statistic
  of the final images.
- One folder per run, with the run's settings in `run_info.txt` and its
  logs as CSV files.

This repository is public, so it holds the tables only.  The image
`slices.png`, the JSON files, and the stage logs stay on gautschi, in
`/scratch/gautschi/buzzard/denoiser_stop/collect/mace4d_stopping/`.
`slices.png` shows the middle slices of frame 49 for every run, and each
run's difference from run B.

This section uses these terms:
- A **call** is the denoising of one volume.  MACE4D denoises the volumes of
  each orientation in batches, with one call of `denoise_stack` per batch,
  and each volume of a batch stops on its own.
- A **MACE iteration** is one iteration of MACE4D's outer loop.  An
  iteration without this qualifier is an iteration of a denoiser.
- A **distance** is in units of the noise level \( \sigma_y \) of the call.
  An rms distance is taken over the voxels of the call's volume.  The
  **largest distance** is the largest absolute difference at one voxel.
- The **consensus change** of a MACE iteration is the percent change of the
  consensus image in that MACE iteration.  MACE4D's stop threshold tests it.
- The **scaling fix** is the fix of the denoiser's update in PR #14.  It is
  the only change between versions A and B that affects the runs.

### Summary

The test gave six main results:
1. The scaling fix changed the wall time of MACE4D by 1.3% and its final
   image by 0.074%.
2. Without the warm start, MACE4D's stopping rule never stopped a call
   before its cap of 15 iterations.  The sampled calls stopped 0.08 to 0.17
   from their MAP estimates.
3. The gradient rule stopped every sampled call of run C 0.0055 to 0.0063
   from its MAP estimate.  At the automatic \( r \) of 0.10, it raised the
   wall time of MACE4D by 21%.
4. The gradient rule changed the final image by 5.8%.  This change is 5
   times the consensus change of the last MACE iteration of run B.  No run
   had converged, so the difference at convergence is not known.
5. With MACE4D's rule, the denoiser warm start changed the final image by
   9.7%.  With the gradient rule, the warm start changed it by 0.6%.
6. Option 2's noise level was 40% below the automatic one.  With it, the
   gradient rule raised the wall time by 5.5%.  The final image then
   differed by 6.6% from that of run C, which used the automatic noise
   level.

### Time and memory

The table gives the time and the memory of each run.  The columns of
iterations and denoiser time are means over MACE iterations 2 to 10, because
MACE iteration 1 includes the compilation.  The denoiser time is summed over
the 4 GPUs.  The time per MACE iteration is the wall time of one MACE
iteration, with the same mean.  The wall time of the reconstruction covers
its setup and its 10 MACE iterations.

| Version | \( \sigma_y \) | \( r \) | Iterations per call | Denoiser time per MACE iteration (s) | Time per MACE iteration (s) | Wall time of the reconstruction (s) | Wall time relative to B | Peak allocated memory per GPU (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.00599 | 0.102 | 15 | 70 | 142 | 1399 | -1.3% | 7.59 |
| B | 0.00599 | 0.102 | 15 | 71 | 143 | 1418 | | 7.59 |
| C | 0.00599 | 0.102 | 106 | 196 | 172 | 1713 | +20.8% | 7.59 |
| D | 0.00362 | 0.157 | 44 | 108 | 150 | 1496 | +5.5% | 7.59 |
| Bw | 0.00599 | 0.102 | 15 | 84 | 147 | 1447 | +2.0% | 9.36 |
| Cw | 0.00599 | 0.102 | 109 | 222 | 180 | 1790 | +26.2% | 9.36 |

Runs A and B differ by 1.3% in wall time.

The gradient rule raised the wall time by 21% at the automatic \( r \) of
0.10.  Its calls ran a mean of 106 iterations, against 15.  The denoiser time
per MACE iteration rose from 71 s to 196 s, by a factor of 2.8.  The time of
the prox agents, which fit each frame to its views, did not change.  Its mean
was 477 s to 491 s per MACE iteration, summed over the GPUs, in every run.
So the increase of the wall time depends on the ratio of the prox time to
the denoiser time, which differs from scan to scan **[inferred]**.

Option 2's noise level raised \( r \) to 0.157.  At this \( r \), the calls
ran a mean of 44 iterations, and the wall time rose by 5.5%.

B does not compute the gradient statistic, and C and D do.  A line through
the denoiser times of C and D, against the iterations per call, has an
intercept of 45 s per MACE iteration and a slope of 1.42 s per iteration.
Both are summed over the GPUs.  For B's 15 iterations, the line predicts
67 s, and B took 71 s.  These results indicate that the gradient statistic
adds little or no time per iteration **[inferred]**.  The line rests on two
points, and B lies outside them.  The one-frame check tests the claim
again.

The warm start raised the peak allocated memory from 7.59 GB to 9.36 GB per
GPU.  It also added 13 s to the denoiser time of each MACE iteration in run
Bw, at the same 15 iterations per call.  The stopping rule did not change
the memory.

### Distance of the calls from their MAP estimates

Each run sampled 9 calls.  At MACE iterations 1, 5, and 10, it kept the call
of each orientation that holds the middle plane.  A call of orientation XYt
denoises a volume of 260 by 260 by 99 voxels.  A call of XZt or YZt
denoises a volume of 260 by 728 by 99 voxels.

`mace4d_calls.py` computed the MAP estimate of each sampled call.  It ran
the denoiser from the call's output with the gradient rule at 0.001.  The
exact gradient at each MAP estimate was at most 0.0008, so each MAP estimate
lies within 0.0008 of the exact minimizer **[derived]**.  The script also
replayed each call from its start, one iteration at a time, to find where
each rule would stop.

| Version | \( r \) | Iterations per call | Distance at the start (rms) | Distance at the stop (rms) | Largest distance at the stop |
|---|---:|---|---|---|---|
| A | 0.102 | 15 | 0.23 to 0.81 | 0.084 to 0.171 | 0.99 to 7.5 |
| B | 0.102 | 15 | 0.23 to 0.81 | 0.084 to 0.172 | 0.99 to 7.5 |
| C | 0.102 | 111 to 138 | 0.23 to 0.81 | 0.0055 to 0.0063 | 0.04 to 0.27 |
| D | 0.157 | 45 to 54 | 0.26 to 1.09 | 0.0052 to 0.0062 | 0.07 to 0.22 |
| Bw | 0.102 | 15 | 0.12 to 0.81 | 0.081 to 0.216 | 0.47 to 20.5 |
| Cw | 0.102 | 56 to 138 | 0.016 to 0.81 | 0.0055 to 0.0063 | 0.01 to 0.27 |

MACE4D's rule never stopped a call of A or B before the cap of 15.  The rule
stops a call when the L1 norm of one iteration's change is less than 0.05%
of the L1 norm of the volume.  After 15 iterations, the change of the
sampled calls of B was still 0.16% to 0.90%.  `denoiser_iterations.csv`
shows that every call of A and B, sampled or not, ran the full 15
iterations.

The calls of A and B stopped 0.08 to 0.17 from their MAP estimates.  Their
largest distance was up to 7.5.  The calls of C stopped 0.0055 to 0.0063
from their MAP estimates, with a largest distance of at most 0.27.  The
calls of D stopped at the same rms distance after fewer iterations.

The number of iterations to reach a distance of 0.01 is consistent with
growth as \( 1/r^2 \), the growth found on the CPU.  Its median over the
sampled calls was 107 for C, at \( r = 0.102 \), and 44 for D, at
\( r = 0.157 \).  The ratio of the two, 2.4, is close to
\( (0.157/0.102)^2 \).  On the CPU, the 3D test volume needed 87 to 91
iterations to reach 0.01 at \( r = 0.1 \).

The replay reproduced the calls of B, C, D, Bw, and Cw.  At every voxel, the
replay and the call differed by less than 0.0004.

For A, the replay used the update with the scaling fix.  After 15
iterations, the two updates differed by 0.02 to 0.85 at the voxel of largest
difference.  The rms distances of A and B agreed to two digits.  These
results indicate that the scaling fix matters little at \( r = 0.10 \) in
these volumes.

### The warm start

With the warm start, each call starts from the previous output of its
denoiser instead of its input.  The calls of MACE iteration 1 have no
previous output, so they start from their input in every run.  The sampled
calls of MACE iterations 5 and 10 show the effect of the warm start.

The warm start lowered the distance at the start.  At MACE iteration 10,
the sampled calls of Cw started 0.016 to 0.024 from their MAP estimates.
The calls of C, which start from their input, started 0.24 to 0.40 away.

The warm start lowered the number of iterations much less than the distance
at the start.  At MACE iteration 10, the calls of Cw needed 56 to 62
iterations, about half of what the calls of C needed.  At MACE iteration 5,
the calls of Cw started 3 to 4 times closer to their MAP estimates than
those of C.  They still needed 136 to 138 iterations, against 112 to 136 for
the calls of C.  These results suggest that the error left by a warm start
lies at low spatial frequencies **[inferred]**.  The CPU tests of
`findings/step4_stopping_rule.md` suggested that VCD removes such error
slowly.

At MACE iteration 2, the warm start made the calls of Cw slower than those
of C.  Of the 1248 calls of Cw in that MACE iteration, 914 reached the cap
of 200 iterations.  At the cap, their gradient statistic was still at a
median of 0.012, and at most 0.052.  In orientations XZt and YZt, 98% to 99%
of the calls reached the cap, and in XYt, 55% did.

After MACE iteration 2, the median number of iterations of Cw fell.  It was
151 to 167 at MACE iteration 3, 131 to 136 at MACE iteration 5, and 48 to 65
at MACE iterations 8 to 10.  The calls of C needed a median of 95 to 113
iterations at every MACE iteration.  Over the 10 MACE iterations, Cw ran 2%
more iterations than C, and its wall time was 4.5% longer.

With MACE4D's rule, the warm start did not bring the calls consistently
closer to their MAP estimates.  At MACE iteration 5, the calls of Bw stopped
0.13 to 0.22 from their MAP estimates, against 0.08 to 0.14 for those of B.
At MACE iteration 10, they stopped 0.08 to 0.09 away, against 0.08 to 0.15
for those of B.  The largest distance of a call of Bw was up to 20.5,
against 7.5 for B.  At MACE iterations 8 to 10, the rule stopped about 1% of
the calls of orientation XYt before the cap.

Bw had 6 sampled calls at MACE iterations 5 and 10.  A threshold of 0.2%,
the default of `denoise`, would have stopped 4 of them after 1 to 9
iterations.  Question 12 of `plans/mace4d/decisions.md` describes this
behavior.  It also plans a measurement for a later stage of the MACE4D
plan.  The measurement is the distance of each setting from a converged
reference, per call and in the final result.  This test made that
measurement, with run C as the reference for the final result.

### The one-frame check

The one-frame check ran `denoise` on frame 49 of the initial image, a volume
of 260 by 260 by 728 voxels.  It used one H100.  For A to C, the parameters
were automatic, with \( \sigma_y \) = 0.00405 and \( r \) = 0.249.  For D,
\( \sigma_y \) was 0.00362, as in run D, and \( r \) was 0.274.  The sweep
time in the table excludes the compilation and the initialization.

| Version | Stopping rule | Iterations | Distance at the stop (rms) | Largest distance at the stop | Sweep time (ms) |
|---|---|---:|---:|---:|---:|
| A | change below 0.2%, or 15 iterations | 13 | 0.020 | 10.2 | 118 |
| B | as A | 12 | 0.019 | 0.30 | 114 |
| C | gradient statistic below 0.01, or 200 iterations | 20 | 0.0043 | 0.13 | 146 |
| D | as C | 17 | 0.0040 | 0.11 | 134 |

The scaling fix mattered in this check.  The largest distance of A was 10.2,
and that of B was 0.30.  Their rms distances were nearly equal.  These
results indicate that the old update approaches the MAP estimate much more
slowly at a few voxels **[inferred]**.

The current defaults of `denoise` stopped 0.019 from the MAP estimate.  On
the CPU, the 3D test volume with fdk noise at \( r = 0.26 \) behaved
similarly.  It needed 15 iterations to reach a distance of 0.01, and this
frame needed 16.  The gradient rule ran 8 more iterations than the current
defaults and stopped 0.0043 away.

A line through the sweep times of C and D has a slope of 3.9 ms per
iteration.  For B's 12 iterations, the line predicts 114.5 ms, and B took
113.5 ms.  B does not compute the gradient statistic, and C and D do.  These
results agree with the MACE4D runs.  Both indicate that the gradient
statistic adds little or no time per iteration **[inferred]**.  The 8 extra
iterations of C took 32 ms.  The compilation before the first call took 8
to 16 s.

### The final images

The table compares the final images after 10 MACE iterations.  The relative
difference is the rms difference over the 4D image divided by the rms of the
second image.  The other two columns give the rms difference and the largest
difference at one voxel, in units of the automatic noise level, 0.00599.
The rms of each final image was 0.0097 to 0.0099, or 1.6 in these units.

| Pair | Relative difference | rms difference | Largest difference |
|---|---:|---:|---:|
| A and B | 0.074% | 0.001 | 0.73 |
| B and C | 5.8% | 0.094 | 14.7 |
| B and D | 2.2% | 0.037 | 20.1 |
| C and D | 6.6% | 0.109 | 29.8 |
| B and Bw | 9.7% | 0.158 | 7.9 |
| C and Cw | 0.63% | 0.010 | 0.21 |
| C and Bw | 6.8% | 0.109 | 8.3 |

Runs A and B differ by 0.074%, so the scaling fix did not change the final
image.  The gradient rule changed it by 5.8%.  This change is 5 times the
consensus change of the last MACE iteration of B, which was 1.15%.

Run D differed from run B by 2.2%.  D differs from B in both the stopping
rule and the noise level.  Option 2's noise level alone, from C to D,
changed the image by 6.6%.

The warm start changed the image by 9.7% with MACE4D's rule, and by 0.63%
with the gradient rule.  These results indicate that, with the gradient
rule, the final image depends little on where each call starts.

Two statistics of each final image show how strongly each version smoothed.
The first is the largest value in the image.  In the units of the image, it
was 0.257 for B, 0.195 for C and Cw, 0.353 for D, and 0.232 for Bw.  The
comparison did not record the largest value of A, but A differs from B by
at most 0.0044 at any voxel.  Each largest value comes from one voxel, and
the comparison did not record where it is.

The second statistic is the adjacent-voxel statistic, the statistic of
option 2's noise level.  It is the median absolute difference of adjacent
voxels, divided by \( 0.6745 \sqrt{2} \).  `image_noise.py` computed it on
the voxels that option 2 uses, the central half of each spatial axis in the
same 25 frames.  The table gives it in units of \( 10^{-5} \).  The spatial
column pools the three spatial axes.  Axis 3 is the axis of 728 voxels, and
the last column compares consecutive frames.

| Image | Spatial | Axis 1 | Axis 2 | Axis 3 | Between frames |
|---|---:|---:|---:|---:|---:|
| initial image | 361.6 | 390.6 | 379.4 | 321.2 | 762.5 |
| A | 9.24 | 11.15 | 10.51 | 7.10 | 15.17 |
| B | 9.21 | 11.13 | 10.49 | 7.07 | 15.14 |
| C | 8.59 | 11.96 | 10.82 | 5.65 | 13.30 |
| D | 10.08 | 11.79 | 10.98 | 8.25 | 17.07 |
| Bw | 9.38 | 12.77 | 11.84 | 6.09 | 14.66 |
| Cw | 8.67 | 12.14 | 10.98 | 5.66 | 13.37 |

The spatial value of every final image was 2.4% to 2.8% of the initial
image's.  So every final image is smooth at the scale of adjacent voxels,
compared with the initial image.  The spatial value of the initial image is
option 2's noise level, 0.00362.

The spatial value pools changes in opposite directions.  From B to C, the
value fell by 20% along axis 3 and by 12% between frames.  It rose by 7.5%
along axis 1 and by 3% along axis 2.  Bw showed the same pattern relative to
B, with larger rises along axes 1 and 2.  D had larger values than B along
every axis and between frames.

These results suggest a partial explanation **[inferred]**.  With MACE4D's
rule, the calls of XZt and YZt stopped farther from their MAP estimates than
the calls of XYt.  The sampled calls of B stopped 0.11 to 0.17 away in XZt
and YZt, and 0.08 to 0.13 away in XYt.  XZt and YZt are the two
orientations that smooth along axis 3.  The gradient rule brought the calls
of all three orientations to their MAP estimates, and the image became
smoother along axis 3.  This explanation does not account for the rise
along axes 1 and 2.  The calls of D also reached their MAP estimates, but
with a weaker prior, because \( r \) is larger.  That fits the larger values
of D along every axis.

After 10 MACE iterations, no run had converged.  The consensus change at
MACE iteration 10 was 1.15% for A and B, 0.93% for C, 1.17% for D, 1.24% for
Bw, and 0.90% for Cw.  In the last MACE iteration of B, the consensus change
shrank by 11%.  If it kept shrinking at that rate, the remaining changes of
B would add up to about 10% **[inferred]**.  So the difference between the
consensus equilibria of B and C is not known.

MACE4D's default settings also run 10 MACE iterations on this scan.  The
default cap is 10, and the consensus change stayed above the default
threshold of 0.2% in every MACE iteration.  So the differences above are the
differences that a user of the defaults would see.

### Limits of this evidence

The evidence has four limits:
- The test used one scan, one seed, and 10 MACE iterations.
- The comparison of the final images has no reference image.  So it shows
  how much the versions differ, not which image is closer to the object.
- The largest values and the largest differences come from single voxels.
  The comparison did not record where they are.
- Each run sampled 9 of the 12480 calls of its denoisers, all on the middle
  planes.  The iteration counts of all calls are in
  `denoiser_iterations.csv`.
