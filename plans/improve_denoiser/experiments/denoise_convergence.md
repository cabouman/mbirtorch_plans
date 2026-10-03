# Convergence of `denoise` and its stopping rules

Scripts: `denoise_convergence.py`, with the modules `qggmrf_map3d.py` and
`phantom3d.py`.  Measured on mbirtorch at b0b1879: `greg_dev` at 5f62506
(the scaling fix of PR #14) plus PR #15.  Run on CPU with 4 threads,
2026-10-03.  Results: `results/denoise_convergence/`, one CSV and one
settings file per case.

## Setup

The test volume has 256 by 256 by 32 voxels.  Air is 0 and water is 1000, so
the values are HU plus 1000.  A water ellipse holds six cylindrical inserts:
four iodine concentrations, muscle, and bone.  Each insert ends a few slices
inside the volume, so the volume has edges along all three axes.  The noise
has a standard deviation of 182 HU per voxel.  Six cases use noise whose power
rises with frequency within each slice, as in an FDK image ("fdk").  Two cases
use white noise.

Each case fixes the prior strength \( r = \sigma_x / \sigma_y \) in one of
three ways.  A number pins \( \sigma_x = r \cdot 182 \) with
`sigma_noise` = 182.  "auto" passes `sigma_noise` = 182 and lets the denoiser
choose \( \sigma_x \).  "defaults" lets the denoiser estimate both, as a call
of `denoise(image)` does.  Every case uses the default prior: \( p = 2 \),
\( q = 1.2 \), \( T = 1 \), and 16 subsets.

The script runs the denoiser's own VCD loop, one iteration at a time, until
the image is within \( 10^{-4} \sigma_y \) of the MAP estimate.  The loop is
the loop of `denoise`, with the same partition and the same compiled update.
After each iteration it records the statistics below.  All of them except the
current rule's are in units of \( \sigma_y \).

- **distance**: the rms distance to the MAP estimate.
- **change**: the rms change of the image in the iteration.
- **gradient statistic**: \( \sigma_y \) times the rms gradient of the cost,
  where each pixel's gradient is taken just before the pixel's update in the
  sweep.  Each subset update already computes both terms of this gradient.
- **gradient bound**: \( \sigma_y \) times the rms gradient of the cost at
  the image after the iteration.
- **current rule**: the statistic of `denoise`,
  \( 100 \lVert \Delta x \rVert_1 / \lVert x \rVert_1 \), for the volume with
  0, 1000, or 10000 added.
- **flat-region error**: the rms error against the clean volume, over the
  voxels at least 3 voxels from every edge.

The VCD update does not change when a constant is added to the image.  So an
offset changes only \( \lVert x \rVert_1 \) in the current rule, and one run
gives the current rule at every offset **[derived]**.

## Checks

The MAP estimates are correct to better than \( 2 \times 10^{-7} \sigma_y \)
rms.  In units of \( \sigma_y \), the cost is \( \lVert z - v \rVert^2 / 2 \)
plus a convex prior, so it is 1-strongly convex.  So the rms distance from
any image to the MAP estimate is at most the rms of the cost's gradient there
**[derived]**.  L-BFGS stopped with this bound between \( 3 \times 10^{-9} \)
and \( 1.4 \times 10^{-7} \).  Four further checks passed on a small volume:

- The gradient of `qggmrf_map3d` matched autograd within \( 10^{-14} \).
- Its prior matched mbirtorch's `qggmrf_loss` within a relative
  \( 4 \times 10^{-16} \).
- For \( q = 2 \), its minimizer matched the exact DCT solution within
  \( 3 \times 10^{-8} \) rms.
- At its minimizer for \( q = 1.2 \), mbirtorch's gradient equaled the
  module's gradient: \( 3 \times 10^{-7} \) rms for both.

The loop's image after 15 iterations equaled the output of `denoise` exactly
in every case.

## Results

The number of iterations needed grows about as \( 1 / r^2 \).  At the
automatic strength, 15 iterations reach a distance of 0.01 to 0.03.  At
\( r = 0.1 \) and below, 15 iterations stop far from the MAP estimate.

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

The kind of noise mattered little.  At \( r = 0.1 \), white noise needed 91
iterations to reach 0.01, and fdk noise needed 87.

In the defaults case, the denoiser estimated \( \sigma_y \) = 131.2 HU
instead of 182 HU.  Its distances are in units of 131.2 HU.

The table below gives the iteration at which each rule stops, and the
distance there.

| Case | Current rule, 0.2%, offset 0 | offset 1000 | offset 10000 | Current rule, 0.05% | Change below 0.001 | Gradient statistic below 0.03 | below 0.01 |
|---|---|---|---|---|---|---|---|
| fdk, 0.4 | 7: 0.010 | 5: 0.024 | 2: 0.118 | 10: 0.003 | 11: 0.002 | 7: 0.010 | 9: 0.004 |
| fdk, defaults | 10: 0.030 | 7: 0.065 | 2: 0.310 | 15: 0.009 | 19: 0.004 | 14: 0.012 | 18: 0.005 |
| fdk, 0.2 | 13: 0.033 | 8: 0.081 | 3: 0.246 | 20: 0.012 | 24: 0.007 | 20: 0.012 | 27: 0.005 |
| white, auto | 11: 0.030 | 7: 0.063 | 3: 0.179 | 17: 0.013 | 22: 0.007 | 17: 0.013 | 25: 0.005 |
| fdk, auto | 13: 0.041 | 8: 0.092 | 3: 0.263 | 22: 0.012 | 26: 0.007 | 22: 0.012 | 30: 0.005 |
| white, 0.1 | 13: 0.090 | 8: 0.129 | 3: 0.266 | 27: 0.051 | 42: 0.033 | 63: 0.019 | 108: 0.007 |
| fdk, 0.1 | 18: 0.090 | 10: 0.166 | 3: 0.395 | 34: 0.043 | 46: 0.029 | 64: 0.018 | 104: 0.007 |
| fdk, 0.05 | 21: 0.193 | 11: 0.270 | 3: 0.487 | 43: 0.141 | 73: 0.107 | 325: 0.020 | 515: 0.007 |

The gradient statistic stopped every case within the same range of distances.
At a threshold of 0.01, the distance at the stop was between 0.004 and 0.007
in every case.  The other rules stopped at distances that varied by a factor
of 20 or more from case to case.

The ratio of each statistic to the distance explains these results.  From
iteration 10 on, the ratios fell in these ranges:

| Statistic | Smallest ratio | Median ratios | Largest ratio |
|---|---:|---:|---:|
| change | 0.005 | 0.005 to 0.49 | 0.54 |
| gradient statistic | 1.38 | 1.44 to 2.14 | 12.4 |
| gradient bound | 1.09 | 1.15 to 1.68 | 9.6 |

The median ratio of the change to the distance fell with \( r \): 0.49 at
\( r = 0.4 \), 0.12 at 0.2, 0.023 at 0.1, and 0.005 at 0.05.  The gradient
statistic was at least 1.38 times the distance at every iteration of every
case.  So a rule on the gradient statistic never stopped before the distance
was below its threshold.

The ratio of the gradient statistic to the change was 4.4 at \( r = 0.4 \),
15 at 0.2, 64 at 0.1, and 280 at 0.05 (medians from iteration 10 on).  These
ratios are 0.6 to 0.7 times \( 1 + 1/r^2 \).  For \( p = 2 \) and
\( q < 2 \), \( 1 + 1/r^2 \) is the second derivative of the surrogate cost
at a pixel in a flat region, in units of \( \sigma_y \) **[derived]**.  Each VCD step divides the
gradient by this second derivative.

The step size alpha stayed between 0.986 and 1.000 in every iteration of
every case.  The median time of one iteration's updates was 0.033 to 0.039 s
for the 2.1 million voxels.

## A note on the flat-region error

At small \( r \), some iterates came closer to the clean volume than the
MAP estimate did.  At \( r = 0.05 \), the flat-region error fell from
\( 1.00 \sigma_y \) to \( 0.062 \sigma_y \) at iteration 45, and then
rose to the MAP estimate's \( 0.145 \sigma_y \).  At \( r = 0.1 \), the
same happened on a smaller scale.  With fdk noise the error reached 0.086 at
iteration 49, against 0.094 for the MAP estimate.  With white noise it
reached 0.059 at iteration 31, against 0.077.

The MAP estimate's flat-region error was lower at \( r = 0.1 \) than at
0.05.  So with \( q = 1.2 \) and \( T = 1 \), a smaller \( r \) did not
always give a smoother MAP estimate.  This test did not separate the residual
noise from the blur of the inserts into the flat region.
