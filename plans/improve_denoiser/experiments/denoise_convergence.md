# Convergence of `denoise` and its stopping rules

Scripts: `denoise_convergence.py`, with the modules `qggmrf_map3d.py` and
`phantom3d.py`.  `check_map3d.py` checks the reference solver, and
`plan_slice_check.py` repeats the plan's measurement on a 2D slice.
Measured on mbirtorch at b0b1879: `greg_dev` at 5f62506 (the scaling fix of
PR #14) plus PR #15.  Run on CPU with 4 threads, 2026-10-03.  Results:
`results/denoise_convergence/`, one CSV and one settings file per case.

## Setup

The test volume has 256 by 256 by 32 voxels.  Air is 0 and water is 1000, so
the values are HU plus 1000.  A water ellipse holds six cylindrical inserts:
four iodine concentrations, muscle, and bone.  Each insert ends a few slices
inside the volume, so the volume has edges along all three axes.  The noise
has a standard deviation of 182 HU per voxel.

The cases use three kinds of noise, which `phantom3d.py` describes:

- **white**: independent from voxel to voxel.
- **fdk**: the spectrum of an FDK image within each slice, and independent
  from slice to slice.  Neighbors within a slice have a correlation of 0.57.
- **fdk_z**: fdk noise blurred along the slice axis, so that neighbors in
  adjacent slices have a correlation of 0.78.

Most cases use the 3D volume.  Three cases ("slice") use its middle slice
alone, as a 2D image of 256 by 256 by 1 voxels.

Each case sets the ratio \( r = \sigma_x / \sigma_y \) in one of three ways.
A smaller \( r \) gives a stronger prior.  A number sets
\( \sigma_x = 182 r \) with `sigma_noise` = 182.  "auto" passes
`sigma_noise` = 182 and lets the denoiser choose \( \sigma_x \).  "defaults"
lets the denoiser estimate both, as a call of `denoise(image)` does.  Every
case uses the default prior: \( p = 2 \), \( q = 1.2 \), \( T = 1 \), and 16
subsets.

The script runs the denoiser's own VCD loop, one iteration at a time, until
the image is within \( 10^{-4} \sigma_y \) of the MAP estimate.  The loop is
the loop of `denoise`, with the same partition and the same compiled update.
After each iteration it records the statistics below.  All of them except the
percent change are in units of \( \sigma_y \).

- **distance**: the rms distance to the MAP estimate.
- **change**: the rms change of the image in the iteration.
- **gradient statistic**: \( \sigma_y \) times the rms gradient of the cost,
  where each voxel's gradient is taken just before the voxel's update in the
  sweep.  Each subset update already computes both terms of this gradient.
- **exact gradient**: \( \sigma_y \) times the rms gradient of the cost at
  the image after the iteration.
- **percent change**: the statistic of the stopping rule of `denoise`,
  \( 100 \lVert \Delta x \rVert_1 / \lVert x \rVert_1 \), for the volume
  with 0, 1000, or 10000 added.
- **flat-region error**: the rms error against the clean volume, over the
  water voxels at least 3 voxels from every edge.

The VCD update does not change when a constant is added to the image.  So an
offset changes only \( \lVert x \rVert_1 \) in the percent change, and one
run gives the percent change at every offset **[derived]**.  This holds with
\( \sigma_y \) and \( \sigma_x \) fixed.  In the "auto" and "defaults" cases,
the denoiser would choose different parameters for the volume with an offset
added.  So for those cases, the offset columns below are not the results of
`denoise(image + offset)`.

## Checks

The MAP estimates are correct to better than \( 2 \times 10^{-7} \sigma_y \)
rms.  In units of \( \sigma_y \), the cost is \( \lVert z - v \rVert^2 / 2 \)
plus a convex prior, so it is 1-strongly convex.  So the rms distance from
any image to the MAP estimate is at most the rms of the cost's gradient there
**[derived]**.  L-BFGS stopped with this bound between
\( 2.5 \times 10^{-9} \) and \( 1.4 \times 10^{-7} \).

`check_map3d.py` ran four checks of `qggmrf_map3d` on a volume of 24 by 20 by
8 voxels.  All four passed:

- The module's gradient matched autograd within \( 10^{-14} \), for three
  priors.
- The module's prior matched mbirtorch's `qggmrf_loss` within a relative
  \( 4 \times 10^{-16} \).
- For \( q = 2 \), the module's minimizer matched the exact DCT solution
  within \( 3 \times 10^{-8} \) rms.
- At the module's minimizer for \( q = 1.2 \), mbirtorch's gradient of the
  cost was as small as the module's gradient.  Both were
  \( 3.1 \times 10^{-7} \) rms at \( r = 0.05 \), and
  \( 2.5 \times 10^{-8} \) at \( r = 0.2 \).  So the module's minimizer is
  also the minimizer of mbirtorch's cost.

The loop's image after 15 iterations equaled the output of `denoise` exactly
in every case.

## Iterations to reach each distance

The number of iterations needed grows about as \( 1 / r^2 \).  It also
depends on the noise and on whether the image is 2D or 3D.  On the 3D volume
with fdk or white noise, 15 iterations at the automatic \( r \) reached a
distance of 0.009 to 0.030.  With fdk_z noise and on the slice, 15 iterations
at the automatic \( r \) reached only 0.070 and 0.172.

| Case | \( r \) | Distance after 15 | Iterations to 0.1 | to 0.03 | to 0.01 | to 0.003 |
|---|---:|---:|---:|---:|---:|---:|
| fdk, 0.4 | 0.400 | 0.0003 | 3 | 5 | 7 | 10 |
| fdk, defaults | 0.259 | 0.009 | 6 | 10 | 15 | 21 |
| fdk, 0.2 | 0.200 | 0.024 | 7 | 14 | 22 | 31 |
| white, auto | 0.197 | 0.017 | 6 | 11 | 20 | 31 |
| fdk, auto | 0.188 | 0.030 | 8 | 16 | 24 | 35 |
| white, 0.1 | 0.100 | 0.081 | 12 | 46 | 91 | 146 |
| fdk, 0.1 | 0.100 | 0.110 | 17 | 46 | 87 | 138 |
| fdk, 0.05 | 0.050 | 0.229 | 82 | 256 | 444 | 671 |
| fdk_z, auto | 0.167 | 0.070 | 13 | 23 | 34 | 48 |
| fdk_z, 0.1 | 0.100 | 0.179 | 25 | 52 | 88 | 136 |
| slice, fdk, auto | 0.162 | 0.172 | 22 | 39 | 56 | 76 |
| slice, fdk, 0.1 | 0.100 | 0.383 | 49 | 88 | 131 | 185 |
| slice, white, 0.1 | 0.100 | 0.247 | 33 | 71 | 120 | 182 |

In the defaults case, the denoiser estimated \( \sigma_y \) = 131.2 HU
instead of 182 HU.  Its distances are in units of 131.2 HU.

At \( r = 0.1 \) on the 3D volume, white noise needed 91 iterations to reach
0.01, and fdk noise needed 87.  So the correlation within a slice mattered
little there.  The correlation between slices slowed the first iterations.
After 15 iterations at \( r = 0.1 \), the distance was 0.179 with fdk_z noise
and 0.110 with fdk noise.  Both reached 0.01 after 87 or 88 iterations.  The
slice was slower throughout.  At \( r = 0.1 \), it needed 131 iterations
with fdk noise and 120 with white noise.

These results suggest that VCD removes the error at high spatial
frequencies in a few iterations and the error at low frequencies slowly
**[inferred]**.  Noise correlated between slices has less power at high
frequencies along the slice axis.  A 2D image has a larger share of its
frequencies near zero than a 3D volume **[derived]**.

The slow convergence on the slice matches the plan's measurement on the
500 by 500 slice of pcdrecon's test.  `plan_slice_check.py` repeats that
measurement with the current code.  At \( r = 0.1 \), with the array shape
(1, 500, 500) that pcdrecon's test used, 15 iterations left that slice
69.6 HU from the MAP estimate.  The plan reports 69.6 HU.  With the shape
(500, 500, 1), the distance was 68.0 HU, which is 0.373 \( \sigma_y \).
The 256 by 256 slice of this script, with the shape (256, 256, 1), was 0.383
\( \sigma_y \) away after 15 iterations.  So the size of the slice
mattered little.

## Where each rule stops

The table gives the iteration at which each rule stops, and the distance
there.  These iteration counts ignore the cap of 15.

| Case | Percent change below 0.2, offset 0 | offset 1000 | offset 10000 | Percent change below 0.05 | Change below 0.001 | Gradient statistic below 0.03 | below 0.01 |
|---|---|---|---|---|---|---|---|
| fdk, 0.4 | 7: 0.010 | 5: 0.024 | 2: 0.118 | 10: 0.003 | 11: 0.002 | 7: 0.010 | 9: 0.004 |
| fdk, defaults | 10: 0.030 | 7: 0.065 | 2: 0.310 | 15: 0.009 | 19: 0.004 | 14: 0.012 | 18: 0.005 |
| fdk, 0.2 | 13: 0.033 | 8: 0.081 | 3: 0.246 | 20: 0.012 | 24: 0.007 | 20: 0.012 | 27: 0.005 |
| white, auto | 11: 0.030 | 7: 0.063 | 3: 0.179 | 17: 0.013 | 22: 0.007 | 17: 0.013 | 25: 0.005 |
| fdk, auto | 13: 0.041 | 8: 0.092 | 3: 0.263 | 22: 0.012 | 26: 0.007 | 22: 0.012 | 30: 0.005 |
| white, 0.1 | 13: 0.090 | 8: 0.129 | 3: 0.266 | 27: 0.051 | 42: 0.033 | 63: 0.019 | 108: 0.007 |
| fdk, 0.1 | 18: 0.090 | 10: 0.166 | 3: 0.395 | 34: 0.043 | 46: 0.029 | 64: 0.018 | 104: 0.007 |
| fdk, 0.05 | 21: 0.193 | 11: 0.270 | 3: 0.487 | 43: 0.141 | 73: 0.107 | 325: 0.020 | 515: 0.007 |
| fdk_z, auto | 17: 0.056 | 10: 0.130 | 2: 0.455 | 29: 0.016 | 34: 0.010 | 31: 0.013 | 42: 0.005 |
| fdk_z, 0.1 | 23: 0.106 | 13: 0.208 | 2: 0.621 | 44: 0.039 | 55: 0.026 | 71: 0.016 | 106: 0.006 |
| slice, fdk, auto | 23: 0.092 | 12: 0.219 | 1: 0.617 | 41: 0.026 | 50: 0.014 | 48: 0.016 | 64: 0.006 |
| slice, fdk, 0.1 | 34: 0.171 | 16: 0.365 | 1: 0.810 | 67: 0.054 | 84: 0.033 | 110: 0.016 | 151: 0.006 |
| slice, white, 0.1 | 27: 0.125 | 15: 0.247 | 1: 0.851 | 51: 0.051 | 66: 0.034 | 93: 0.018 | 139: 0.007 |

The gradient statistic stopped every case within a narrow range of
distances.  At a threshold of 0.01, the distance at the stop was between
0.004 and 0.007 in every case.  Under each of the other rules in the table,
the largest distance at the stop was 7 to 63 times the smallest.

A fourth rule estimates the remaining distance from the change.  It measures
the factor \( f \) by which the change shrank per iteration over the last 5
iterations.  The estimate is the change times \( f / (1 - f) \), which is
the sum of the remaining changes if each one shrinks by \( f \).  At a
threshold of 0.03, this rule stopped 0.015 to 0.079 from the MAP estimate.
At 0.01, it stopped 0.009 to 0.015 away.

## Why the rules behave this way

The ratio of each statistic to the distance explains these results.  From
iteration 10 on, the ratios fell in these ranges:

| Statistic | Smallest ratio | Median ratios | Largest ratio |
|---|---:|---:|---:|
| change | 0.005 | 0.005 to 0.49 | 0.54 |
| estimated distance | 0.30 | 0.90 to 0.97 | 1.04 |
| gradient statistic | 1.27 | 1.37 to 2.14 | 12.4 |
| exact gradient | 1.07 | 1.15 to 1.76 | 9.6 |

For fdk noise on the 3D volume, the median ratio of the change to the
distance fell with \( r \).  It was 0.49 at \( r = 0.4 \), 0.12 at 0.2,
0.023 at 0.1, and 0.005 at 0.05.  The gradient statistic was at least 1.27
times the distance at every iteration of every case.  So a rule on the
gradient statistic never stopped before the distance was below its
threshold.  The estimated distance tracked the distance on most iterations,
but on some it was as low as 0.30 times the distance.

The median ratio of the gradient statistic to the change also followed
\( r \).  For fdk noise on the 3D volume, from iteration 10 on, it was 4.4 at
\( r = 0.4 \), 15 at 0.2, 64 at 0.1, and 280 at 0.05.  These ratios are 0.58
to 0.70 times \( 1 + 1/r^2 \).  Consider a voxel whose neighbor differences
are much smaller than \( T \sigma_x \).  For \( p = 2 \) and \( q < 2 \),
the second derivative of the surrogate cost there is \( 1 + 1/r^2 \), in
units of \( \sigma_y \) **[derived]**.  Each VCD step divides the gradient
by this second derivative.  On the slice, each voxel has 4 neighbors instead of 6,
so the second derivative is \( 1 + 2 / (3 r^2) \).  The median ratios on the
slice were 0.98 to 1.06 times this value.

The mean step size alpha over the subsets stayed between 0.986 and 1.000 in
every iteration on the 3D volume, and between 0.88 and 1.00 on the slice.
The median time of one iteration's updates was 0.032 to 0.039 s for the 2.1
million voxels of the volume, and 0.007 s for the 65536 voxels of the slice.

## A note on the flat-region error

At small \( r \), some iterates came closer to the clean volume than the
MAP estimate did.  At \( r = 0.05 \), the flat-region error fell from
\( 1.00 \sigma_y \) to \( 0.062 \sigma_y \) at iteration 45, and then
rose to the MAP estimate's \( 0.145 \sigma_y \).  At \( r = 0.1 \), the
same happened on a smaller scale.  With fdk noise the error reached 0.086 at
iteration 49, against 0.094 for the MAP estimate.  With white noise it
reached 0.059 at iteration 31, against 0.077.  With fdk_z noise and on the
slice, the lowest error was within 0.002 of the MAP estimate's.

The MAP estimate's flat-region error was lower at \( r = 0.1 \) than at
0.05.  So with \( q = 1.2 \) and \( T = 1 \), a smaller \( r \) did not
always give a smoother MAP estimate.  This test did not separate the residual
noise from the blur of the inserts into the flat region.
