# Where three stopping rules stop the denoiser on `prerelease`

Script: `stopping_example.py`, with the modules `qggmrf_map3d.py` and
`phantom3d.py`.  Measured on mbirtorch `prerelease` at 694a87a, on CPU with 4
threads, 2026-10-15.  The denoiser code is the same at 296f738, the head of
`prerelease` when this record was written.  Results: `results/stopping_example/`, one CSV and one
settings file per case.  The tables below are the script's output.  The
script also draws three figures in the same folder.  `stopping_example.png`
shows the stops of Tables 1 and 2.  `stopping_images_warm_r0.1.png` and
`stopping_images_r0.05.png` show the images of two cases where each rule
stops.  This repository does not keep the figures.

## Question

Greg asked on 2026-10-15 for an example that shows clearly how two stopping
rules of the qGGMRF denoiser differ.  The first rule is the rule of `denoise`
on `prerelease`.  It stops when the percent change of the image in one
iteration falls below a threshold.  The second rule is the rule of
`greg_dev`.  It stops when the gradient statistic falls below a threshold.
Charlie suggested a third rule: the percent change with the mean of the image
removed from its denominator.  This record compares the three rules on a
synthetic volume and on the real-data calls of `mace4d_stopping.md`.

## The three rules

Each rule computes a statistic after each iteration and stops the denoiser
when the statistic falls below a threshold.  In the formulas, \( x \) is the
image after the iteration, \( \Delta x \) is its change in the iteration, and
\( \bar{x} \) is its mean.

- **percent change**: \( 100 \lVert \Delta x \rVert_1 / \lVert x \rVert_1 \).
  On `prerelease`, `denoise` stops at 0.2 and MACE4D stops at 0.05.  Both
  stop after 15 iterations at the latest.
- **percent change with the mean removed**:
  \( 100 \lVert \Delta x \rVert_1 / \lVert x - \bar{x} \rVert_1 \).
- **gradient statistic**: \( \sigma_y \) times the rms gradient of the cost.
  Each voxel's gradient is taken just before the voxel's update in the sweep,
  as `greg_dev` computes it.  On `greg_dev`, `denoise` and MACE4D stop at
  0.03.  They stop after 50 and 200 iterations at the latest.

The percent change measures how much the image moved in one iteration,
relative to the size of the image.  It does not measure the distance from the
MAP estimate.

The gradient statistic bounds the distance from the MAP estimate.  The cost is
\( \lVert y - x \rVert^2 / (2 \sigma_y^2) \) plus a convex prior, so its
Hessian is at least \( I / \sigma_y^2 \).  So the rms distance from the MAP
estimate, in units of \( \sigma_y \), is at most \( \sigma_y \) times the rms
gradient **[derived]**.  This bound holds for the gradient at one image.  The
gradient statistic takes each voxel's gradient at a different point of the
sweep, so the bound holds for it only approximately.

## Setup

The test volume is the volume of `phantom3d.py` with white noise, as in
`denoise_convergence.md`.  It has 256 by 256 by 32 voxels.  Air is 0 and water
is 1000, so the values are HU plus 1000.  Air fills 50.5% of the volume.

Each case differs from a base case in one setting.  The base case has a noise
level of 182 HU per voxel and \( r = \sigma_x / \sigma_y = 0.2 \).  This
\( r \) is close to 0.197, the value that `denoise` chooses for this volume
when it is given the true noise level.  `noise_median.md` reports this value.
A smaller \( r \) gives a stronger prior, and VCD
then converges more slowly.  The other cases change one setting:

- **r**: 0.4, 0.1, or 0.05, at 182 HU.
- **noise level**: 20 HU per voxel, with \( r = 0.2 \).
- **offset**: -1000, +1000, or +10000 added to the image.  With -1000, the
  values are HU.
- **warm start**: \( r = 0.2 \) or \( r = 0.1 \), with the call started at
  the MAP estimate for a previous input.

The offsets need no runs of their own.  The VCD update depends only on
differences of neighbors and on \( y - x \).  So an offset added to the input
and to the starting image does not change the update, and it changes only
\( \lVert x \rVert_1 \) in the percent change **[derived]**.  The script
computes the percent change at each offset from the run of the base case.  A
check with `denoise` on a smaller volume confirms that the offsets do not
change the update.  After 10 iterations, the outputs with and without an
offset differed by at most \( 10^{-5} \sigma_y \), apart from the offset.

The warm start models a call of a MACE loop that starts from the denoiser's
previous output.  The call starts at the MAP estimate for the noisy volume.
Its input is the noisy volume plus a smooth perturbation.  The perturbation is
white noise blurred by a Gaussian with a standard deviation of 3 voxels, and
scaled to an rms of \( 0.2 \sigma_y \).

In every case, \( \sigma_y \) and \( \sigma_x \) are given, so the denoiser
does not estimate them.  The other settings are the defaults of `prerelease`:
\( p = 2 \), \( q = 1.2 \), \( T = 1 \), 16 subsets, `max_alpha` = 1.5, and
no positivity constraint.

The script runs the loop of `denoise` one iteration at a time, with the same
partition and the same compiled subset update.  After each iteration, it
records the three statistics and the distance.  The distance is the rms
difference between the image and the MAP estimate, in units of
\( \sigma_y \).  The loop stops when the distance falls below 0.001, after at
least 15 iterations.  So the rules have no cap in the tables.

Two checks confirm that the loop follows `denoise`.  After 15 iterations, the
image of the loop equals the output of `denoise` exactly.  The percent change
of the loop equals the percent change that `denoise` reports, to a relative
difference of \( 2.2 \times 10^{-16} \).

The MAP estimate comes from `qggmrf_map3d.solve`, an L-BFGS solver.  The
solver reports a bound on its own distance from the MAP estimate.  The
largest bound over the cases was \( 2.4 \times 10^{-7} \).

## Results

### Where each rule stops

Table 1 gives the stop of each rule at the thresholds of `denoise` and
`greg_dev`.  Each cell gives the iteration at which the rule stops and, in
parentheses, the distance there.  The last column gives the distance after 15
iterations, where `denoise` and MACE4D stop at the latest.

| Case | Distance at the start | percent change < 0.2 | mean removed < 0.2 | gradient statistic < 0.03 | Distance after 15 iterations |
|---|---:|---|---|---|---:|
| r = 0.4 | 0.62 | 6 (0.004) | 6 (0.004) | 6 (0.004) | <0.001 |
| r = 0.2, the base case | 0.88 | 9 (0.017) | 9 (0.017) | 11 (0.011) | 0.005 |
| r = 0.1 | 0.98 | 11 (0.071) | 11 (0.071) | 43 (0.017) | 0.055 |
| r = 0.05 | 1.02 | 12 (0.183) | 12 (0.183) | 258 (0.018) | 0.171 |
| noise 20 HU, r = 0.2 | 0.89 | 4 (0.094) | 4 (0.094) | 12 (0.013) | 0.009 |
| warm start, r = 0.2 | 0.17 | 9 (0.035) | 9 (0.035) | 14 (0.015) | 0.013 |
| warm start, r = 0.1 | 0.11 | 1 (0.108) | 1 (0.108) | 46 (0.015) | 0.053 |

In the base case, the three rules stop close to one another.  The percent
change stopped after 9 iterations, 0.017 from the MAP estimate.  The gradient
rule stopped after 11 iterations, 0.011 away.  At \( r = 0.4 \), all three
rules stopped after 6 iterations, 0.004 away.

The percent change stops farther from the MAP estimate as \( r \) falls.  At
\( r = 0.1 \), it stopped after 11 iterations, 0.071 away.  At
\( r = 0.05 \), it stopped after 12 iterations, 0.183 away.  In the same two
cases, the gradient rule stopped 0.017 and 0.018 away, after 43 and 258
iterations.

The percent change also stops farther when the noise level is lower.  At
20 HU, with the same \( r \) of 0.2, it stopped after 4 iterations, 0.094
away.  The gradient rule stopped after 12 iterations, 0.013 away.

A warm start can stop the percent change after one iteration.  With
\( r = 0.1 \), the warm start began 0.113 from the MAP estimate.  The percent
change stopped after 1 iteration, 0.108 away, so the distance fell by only 5%.
The gradient rule ran 46 iterations and stopped 0.015 away.

### Images of two cases

Greg asked on 2026-10-15 for images in which the results of the two rules
differ visibly.  The script draws the middle slice of two cases of Table 1:
the warm start at \( r = 0.1 \), in `stopping_images_warm_r0.1.png`, and the
cold start at \( r = 0.05 \), in `stopping_images_r0.05.png`.  The top row of
each figure shows the start of the call, the image where the percent change
at 0.2 stops, the image where the gradient rule at 0.03 stops, and the MAP
estimate.  The gray window is 900 to 1100, where water is 1000.  The bottom
row shows each of the first three images minus the MAP estimate.

Both images at a stop come from `denoise` on `prerelease`.  The image of the
percent change uses the defaults of `denoise`: a threshold of 0.2 and at most
15 iterations.  `prerelease` has no gradient rule, so the image of the
gradient rule comes from `denoise` run for the number of iterations at which
the gradient rule stops in Table 1.  The distances of both images agree with
Table 1 to the three decimals shown there.

In the warm start at \( r = 0.1 \), the input of the call is the noisy image
plus a smooth random change with an rms of 0.2 \( \sigma_y \), or 36 HU.  The
call starts from the MAP estimate for the noisy image alone.  The change
appears in the MAP estimate of the call as light and dark patches.  The
percent change stopped after 1 iteration, and its image looks the same as the
start of the call: it lacks the patches.  Its difference from the MAP
estimate has an rms of 19.6 HU.  The gradient rule stopped after 46
iterations, and its image shows the patches of the MAP estimate.  Its
difference has an rms of 2.7 HU.

In the cold start at \( r = 0.05 \), the prior is strong, and the MAP
estimate blurs the edges of the ellipse and of the inserts.  The percent
change stopped after 12 iterations.  Its image has more noise and sharper
edges than the MAP estimate.  Its difference from the MAP estimate has an rms
of 33.3 HU.  The largest differences are at the edge of the ellipse and in
the inserts.  The gradient rule stopped after 258 iterations, and its image
looks the same as the MAP estimate.  Its difference has an rms of 3.3 HU.

### Offsets

Table 2 gives the stops in the base case with an offset added to the image.

| Offset | percent change < 0.2 | percent change < 0.05 | mean removed < 0.2 | gradient statistic < 0.03 |
|---:|---|---|---|---|
| -1000 | 9 (0.017) | 13 (0.007) | 9 (0.017) | 11 (0.011) |
| 0 | 9 (0.017) | 13 (0.007) | 9 (0.017) | 11 (0.011) |
| +1000 | 6 (0.041) | 9 (0.017) | 9 (0.017) | 11 (0.011) |
| +10000 | 3 (0.135) | 5 (0.059) | 9 (0.017) | 11 (0.011) |

A positive offset stops the percent change earlier.  With +1000, the percent
change at 0.2 stopped after 6 iterations, 0.041 away.  With +10000, it
stopped after 3 iterations, 0.135 away.  The offsets do not change
\( \Delta x \).  They raised \( \lVert x \rVert_1 \) by a factor of 2.8 and
19.

The offset -1000 did not move the stop in this volume.  In HU, each air voxel
adds about 1000 to \( \lVert x \rVert_1 \), and each water voxel adds about
0.  Air and water fill nearly equal parts of this volume, so the offset
changed \( \lVert x \rVert_1 \) by only 1.2%.  In HU, a volume with more air
than water would stop the percent change earlier than it does with air at 0
**[inferred]**.  A volume with less air than water would stop it later.

### What removing the mean changes

Removing the mean fixes the dependence on an offset, and nothing else.  With
the mean removed, the rule stopped after 9 iterations, 0.017 away, at every
offset in Table 2.  Without an offset, it stopped at the same iteration as
the percent change or one iteration later, in Tables 1 and 3.  In the base
case, \( \lVert x - \bar{x} \rVert_1 \) is 7% smaller than
\( \lVert x \rVert_1 \), so the two rules differ little.

### Stricter thresholds

Table 3 gives the stops at stricter thresholds.  The percent change uses 0.05,
the threshold of MACE4D.  The gradient statistic uses 0.01, the threshold that
`mace4d_stopping.md` tested.  The last two columns give the number of
iterations needed to reach a distance of 0.03 and of 0.01.

| Case | percent change < 0.05 | mean removed < 0.05 | gradient statistic < 0.01 | Iterations to 0.03 | Iterations to 0.01 |
|---|---|---|---|---:|---:|
| r = 0.4 | 7 (0.002) | 8 (0.001) | 7 (0.002) | 4 | 5 |
| r = 0.2, the base case | 13 (0.007) | 13 (0.007) | 16 (0.004) | 7 | 12 |
| r = 0.1 | 21 (0.041) | 22 (0.039) | 75 (0.006) | 29 | 59 |
| r = 0.05 | 28 (0.142) | 28 (0.142) | 411 (0.006) | 192 | 339 |
| noise 20 HU, r = 0.2 | 7 (0.035) | 7 (0.035) | 19 (0.005) | 8 | 14 |
| warm start, r = 0.2 | 17 (0.009) | 17 (0.009) | 21 (0.005) | 10 | 17 |
| warm start, r = 0.1 | 23 (0.037) | 24 (0.035) | 77 (0.005) | 28 | 59 |

A stricter threshold does not fix the percent change at small \( r \).  At
0.05, the percent change stopped 0.041 away at \( r = 0.1 \), and 0.142 away
at \( r = 0.05 \).  The gradient statistic at 0.01 stopped 0.002 to 0.006
away in every case.

The cost of convergence grows quickly as \( r \) falls.  The iterations needed
to reach a distance of 0.01 grew by a factor of 2.4 to 5.7 each time \( r \)
halved.  At \( r = 0.05 \), they reached 339.

### Why the percent change varies

Table 4 shows why the stop of the percent change varies.  For each case, it
gives the percent change at the iteration where the gradient rule at 0.03
stops.  It also gives the median ratio of each statistic to the distance, from
iteration 5 on.

| Case | Stop of the gradient rule | Percent change there | Ratio of the percent change to the distance | Ratio of the gradient statistic to the distance |
|---|---|---:|---:|---:|
| r = 0.4 | 6 (0.004) | 0.113 | 23.8 | 3.17 |
| r = 0.2, the base case | 11 (0.011) | 0.076 | 5.55 | 2.26 |
| r = 0.1 | 43 (0.017) | 0.014 | 0.753 | 1.69 |
| r = 0.05 | 258 (0.018) | 0.003 | 0.188 | 1.65 |
| noise 20 HU, r = 0.2 | 12 (0.013) | 0.007 | 0.405 | 1.88 |
| warm start, r = 0.2 | 14 (0.015) | 0.077 | 4.91 | 1.91 |
| warm start, r = 0.1 | 46 (0.015) | 0.016 | 0.943 | 1.85 |

No single threshold of the percent change stops where the gradient rule
stops.  At the stop of the gradient rule, the percent change ranged from 0.003
at \( r = 0.05 \) to 0.113 at \( r = 0.4 \).

The ratio of the percent change to the distance ranged from 0.19 to 24, a
factor of about 130.  The ratio of the gradient statistic to the distance ranged
from 1.65 to 3.17.  Over every iteration of every case, the smallest ratio of
the gradient statistic to the distance was 1.59.  So the bound of the
gradient statistic held at every iteration here.

These results indicate that the percent change follows the speed of
convergence, not the distance **[inferred]**.  The change in one iteration is
about the distance times the fraction of the distance that the iteration
removes.  When \( r \) is small, VCD removes a small fraction in each
iteration.  So the change is small while the distance is still large.

The denominator \( \lVert x \rVert_1 \) adds two more dependences.  It grows
with an offset, while the change does not.  It also changes little when the
noise level falls, while the change falls with the noise level.  So at a lower
noise level, the same percent change corresponds to a larger distance in
units of \( \sigma_y \) **[inferred]**.

## The same rules on real data

The GPU test of `mace4d_stopping.md` replayed 58 sampled denoiser calls of
MACE4D on the Lilly 4DCT phantom scan.  Each replay ran one iteration at a
time, with no cap, and recorded the distance and both statistics.  Calls with
the same input have the same replay, so the 58 calls gave 44 distinct
replays.  The replays ran the code of `greg_dev`.  Its line search uses an
upper bound on the prior term, not the exact prior term of `prerelease`.  The
replays did not record \( \lVert x - \bar{x} \rVert_1 \), so the rule with the
mean removed is missing from Table 5.

Each cell of Table 5 gives the range of the stop iteration and, in
parentheses, the range of the distance there.  The two ranges are separate.
The call that stopped first need not be the call that stopped farthest away.

| Calls | Number | percent change < 0.2 | percent change < 0.05 | gradient statistic < 0.03 | gradient statistic < 0.01 |
|---|---:|---|---|---|---|
| cold start, r = 0.10 | 21 | 13 to 37 (0.053 to 0.128) | 40 to 75 (0.016 to 0.055) | 67 to 85 (0.017 to 0.019) | 111 to 138 (0.006) |
| cold start, r = 0.16 | 9 | 10 to 23 (0.030 to 0.086) | 23 to 36 (0.010 to 0.033) | 28 to 36 (0.015 to 0.018) | 45 to 54 (0.005 to 0.006) |
| one frame, r = 0.25 | 1 | 12 (0.019) | 19 (0.005) | 14 (0.013) | 20 (0.004) |
| one frame, r = 0.27 | 1 | 11 (0.016) | 16 (0.005) | 12 (0.012) | 17 (0.004) |
| warm start, r = 0.10 | 12 | 1 to 23 (0.016 to 0.286) | 1 to 99 (0.016 to 0.087) | 1 to 130 (0.016 to 0.020) | 56 to 188 (0.006) |

The real calls agree with the synthetic example.  At \( r = 0.10 \), the
percent change at 0.2 stopped the cold calls 0.053 to 0.128 from their MAP
estimates.  The gradient rule at 0.03 stopped them 0.017 to 0.019 away.  At
the larger \( r \) of the one-frame check, the two rules stopped close to one
another, as in the base case.

With a warm start, the percent change at 0.2 stopped 7 of the 12 calls after
1 iteration.  Four of these 7 calls had started 0.09 to 0.12 from their MAP
estimates, and they stopped nearly as far away.  The farthest stop was 0.286
away, after 2 iterations of a call that started 0.300 away.  The gradient rule
at 0.03 stopped every warm call 0.016 to 0.020 away.  It stopped one call
after 1 iteration, and that call had started 0.016 away.

The gradient rule needs more iterations at small \( r \).  At
\( r = 0.10 \), the cold calls needed 67 to 85 iterations to stop at 0.03.
MACE4D on `prerelease` stops them after 15.  `mace4d_stopping.md` estimated
that a threshold of 0.03 would raise the wall time of MACE4D on this scan by
about 12%.

## What the results show

The results show three things:

1. The percent change stops at a distance that depends on \( r \), on the
   noise level, on an offset, and on the start of the call.  At a threshold of
   0.2, it stopped 0.004 to 0.183 from the MAP estimate in the synthetic
   cases, and 0.016 to 0.286 away in the real calls.
2. Removing the mean from the denominator fixes the dependence on an offset.
   It does not change the dependence on \( r \), on the noise level, or on
   the start.
3. The gradient rule stops at nearly the same distance in every case.  At
   0.03, it stopped 0.004 to 0.018 from the MAP estimate in the synthetic
   cases, and 0.012 to 0.020 away in the real calls.  Its number of
   iterations grows as \( r \) falls.

The difference between the rules matters most where \( r \) is small or the
call starts warm.  MACE4D ran at \( r = 0.10 \) on the Lilly scan, and its
calls start warm when its warm start is on.  With the true noise level
given, `denoise` chooses \( r = 0.197 \) for this volume at 182 HU, where the
two rules agree.

A median in `_get_estimate_of_recon_std` would move `denoise` toward small
\( r \).  That function computes both the noise estimate and the statistic of
the automatic \( \sigma_x \).  `noise_median.md` found that the median would
lower the automatic \( r \) at 20 HU from 0.67 to 0.18.  The 20 HU case here
has \( r = 0.2 \), and the percent change stopped it 0.094 away.  So with the
median, the percent change would stop `denoise` about 0.09 from the MAP
estimate on this volume at 20 HU **[inferred]**.

## Limits

- The synthetic cases use one phantom, one draw of white noise, and one
  partition.
- The distance is an rms over the volume.  A few voxels can be much farther
  from the MAP estimate.  With MACE4D's rule, `mace4d_stopping.md` found
  voxels up to 7.5 away, and up to 20.5 away with the warm start.  This
  example did not measure the largest distance.
- The warm start of the synthetic cases uses a synthetic perturbation.  The
  warm starts of Table 5 are real.
- The script computes the gradient statistic with a separate compiled
  function, so its time per iteration does not show the cost of the
  statistic.  On `greg_dev`, the subset update computes the statistic.
  `mace4d_stopping.md` found that it adds little or no time per iteration.
- The replays of Table 5 ran the code of `greg_dev`.  The exact prior term of
  `prerelease` may change their iteration counts.
