# Step 4: the stopping rule and the default number of iterations

This page reports step 4 of `improve_qggmrf_denoiser.md`.  Step 4 had four
parts: how far `denoise` stops from the MAP estimate, a comparison of stopping
rules, the inner iterations of the MACE denoiser agent, and the bias of the
automatic noise estimate.  All measurements used mbirtorch at b0b1879.  This
commit is `greg_dev` at 5f62506, which includes the scaling fix of PR #14,
plus the change of PR #15.  The measurements ran on CPU on 2026-10-03.  The run records are `experiments/denoise_convergence.md`,
`experiments/mace_inner_iterations.md`, and `experiments/noise_estimate.md`.

Distances below are rms distances to the MAP estimate, in units of
\( \sigma_y \).  The strength of the prior is \( r = \sigma_x / \sigma_y \).

## Summary

The current defaults of `denoise` are adequate only at the automatic strength
and only for images with a moderate offset.  The defaults are 15 iterations and
a stop when \( 100 \lVert \Delta x \rVert_1 / \lVert x \rVert_1 \) falls below
0.2.  On a 3D test volume, the automatic strength was \( r = 0.19 \) to 0.26.
There, the defaults stopped 0.03 to 0.04 from the MAP estimate.  At
\( r = 0.1 \) they stopped 0.09 to 0.11 away, and at \( r = 0.05 \) they
stopped 0.23 away.  With 10000 added to the same volume, the defaults stopped
after 2 or 3 iterations, 0.12 to 0.49 away.

A rule that compares the change per iteration with \( \sigma_y \) removes the
dependence on the offset, but not the dependence on \( r \).  The change per
iteration is smaller than the distance by a factor that grows about as
\( 1 / r^2 \).  So a fixed threshold on the change stops farther from the MAP
estimate as \( r \) falls.

A rule on the gradient of the cost works at every \( r \) tested.  In units of
\( \sigma_y \), the rms of the gradient bounds the rms distance to the MAP
estimate.  The VCD sweep can compute this gradient statistic at almost no
cost.  With a threshold of 0.01 on the gradient statistic, every case stopped
between 0.004 and 0.007 from the MAP estimate.  The number of iterations then varied with \( r \), from 9 at
\( r = 0.4 \) to 515 at \( r = 0.05 \).

The MACE denoiser agent needs no change.  With its default of 8 warm-started
iterations per call, MACE reached each accuracy target at the same iteration
as with 32, at \( r = 1 \) and at \( r = 0.2 \).

The automatic noise estimate has three biases.  It reads 0.80 of the true level
on white noise in a flat image.  Edges raise it, and so does an offset from
air at 0 to air at -1000.  The automatic \( \sigma_x \) is 0.2 times the same
statistic, so the automatic strength inherits these biases.

## Decisions for Greg

1. **Replace the stopping statistic of the denoiser** with the gradient
   statistic in the three sweep paths: `denoise` on one device, `denoise` on
   several devices, and `denoise_stack`.  A threshold of 0 would still run
   exactly `max_iterations`.  The parameter would need a new name, such as
   `stop_threshold`, because its unit changes from percent to
   \( \sigma_y \).  `recon` keeps its own rule.
2. **Choose the defaults.**  I propose a threshold of 0.01 and
   `max_iterations` = 200.  At the automatic strength this runs 18 to 30
   iterations, against 10 to 13 today, and stops about 0.005 from the MAP
   estimate.  The cap of 200 is enough for \( r \ge 0.1 \) on the test
   volume.  A log line would report a sweep that the cap stops.
3. **MACE4D.**  Its stack denoisers stop when the current statistic falls
   below 0.05 percent.  The gradient statistic would answer open question 12 of
   `plans/mace4d/decisions.md`, as explained below.  This choice belongs to
   the MACE4D plan.
4. **The automatic noise estimate.**  I propose no change in this step.  A fix
   changes the default strength, because the constant 0.2 in the automatic
   \( \sigma_x \) was tuned with the biased estimate.  The options are below.

## Why the gradient bounds the distance

In units of \( \sigma_y \), the cost is

$$
J(v) = \frac{1}{2} \lVert z - v \rVert^2 + P(v) ,
$$

where \( z = y / \sigma_y \) and \( P \) is the prior.  \( P \) is convex for
\( 1 \le q \le p \le 2 \).  A convex function plus
\( \lVert z - v \rVert^2 / 2 \) is 1-strongly convex.  So for any image
\( v \) and the minimizer \( v^* \),

$$
\lVert v - v^* \rVert \le \lVert \nabla J(v) \rVert .
$$

Dividing both sides by the square root of the number of voxels turns this into
a bound on the rms distance.  The gradient in units of \( \sigma_y \) is
\( \sigma_y \) times the gradient of the cost in the unit of the image.  In
units of \( \sigma_y \), the cost does not change when the image,
\( \sigma_y \), and \( \sigma_x \) are multiplied by a constant, or when a
constant is added to the image.  So the gradient statistic does not depend on
the image's scale or offset **[derived]**.

Each subset update of VCD already computes both terms of the gradient at its
pixels.  These are `forward_grad` and `prior_grad` in `vcd_subset_denoiser`.
The sweep statistic sums the squares of their sum over each subset, which costs
one more reduction per subset.  Each pixel's gradient is taken just before the
pixel's update, so the statistic approximates the gradient at the image
rather than equaling it.

The measurements confirm the bound and show that it is close.  From iteration
10 on, the median ratio of the gradient at the image to the distance was 1.15
to 1.68 across the cases, and the smallest ratio was 1.09.  The median ratio
of the sweep statistic to the distance was 1.44 to 2.14.  The sweep statistic
was never less than 1.38 times the distance, at any iteration of any case.
So a rule on the sweep statistic never stopped before the distance was below
its threshold.

## Why a rule on the change fails at small r

VCD divides each pixel's gradient by the second derivative of its surrogate
cost.  In units of \( \sigma_y \), that second derivative is
\( 1 + 1/r^2 \) at a pixel in a flat region, for \( p = 2 \) and
\( q < 2 \) **[derived]**.  The 1 comes from the data term.  The
\( 1/r^2 \) comes from the prior, whose six neighbor weights sum to 1.  So in
flat regions the change per iteration is about \( r^2 \) times the gradient,
and the gradient is close to the distance.

The measurements follow this pattern.  The median ratio of the gradient
statistic to the change was 4.4 at \( r = 0.4 \), 15 at 0.2, 64 at 0.1, and
280 at 0.05.  These ratios are 0.6 to 0.7 times \( 1 + 1/r^2 \).  A threshold
of 0.001 on the change therefore stopped at distances from 0.002 at
\( r = 0.4 \) to 0.107 at \( r = 0.05 \).

The current rule has the same problem, and it adds a dependence on the
offset.  Adding a constant to the image changes \( \lVert x \rVert_1 \), but
it changes neither the VCD path nor \( \lVert \Delta x \rVert_1 \).  On the
test volume, with air at 0 and water at 1000, the current rule at 0.2 percent
stopped after 7 to 21 iterations.  With 10000 added it stopped after 2 or 3.
The offset can also delay the stop.  In a volume of water in true HU,
\( \lVert x \rVert_1 \) is only the size of the residual noise, so the rule
would stop only at the cap of 15 **[inferred]**.

## How many iterations each case needs

The table gives the iterations to reach each distance, and where the current
defaults and the proposed rule stop.  Each "fdk" case has noise whose power
rises with frequency within a slice, as in an FDK image.

| Case | \( r \) | Distance after 15 | Iterations to 0.03 | to 0.01 | Current defaults: iteration, distance | Proposed rule: iteration, distance |
|---|---:|---:|---:|---:|---|---|
| fdk, \( r \) pinned | 0.400 | 0.0003 | 5 | 7 | 7: 0.010 | 9: 0.004 |
| fdk, all automatic | 0.259 | 0.009 | 10 | 15 | 10: 0.030 | 18: 0.005 |
| fdk, \( r \) pinned | 0.200 | 0.024 | 14 | 22 | 13: 0.033 | 27: 0.005 |
| white, automatic \( \sigma_x \) | 0.197 | 0.017 | 11 | 20 | 11: 0.030 | 25: 0.005 |
| fdk, automatic \( \sigma_x \) | 0.188 | 0.030 | 16 | 24 | 13: 0.041 | 30: 0.005 |
| white, \( r \) pinned | 0.100 | 0.081 | 46 | 91 | 13: 0.090 | 108: 0.007 |
| fdk, \( r \) pinned | 0.100 | 0.110 | 46 | 87 | 15: 0.110 | 104: 0.007 |
| fdk, \( r \) pinned | 0.050 | 0.229 | 256 | 444 | 15: 0.229 | 515: 0.007 |

In the last two rows, the current defaults reached the cap of 15 before the
0.2 percent rule fired.  The kind of noise mattered little: at \( r = 0.1 \),
white noise needed 91 iterations to reach 0.01, and fdk noise needed 87.  One
iteration took 0.03 to 0.04 s for the 2.1 million voxels on 4 CPU threads.

The convergence rate should not depend on the size of the volume
**[inferred]**.  In units of \( \sigma_y \), the eigenvalues of the cost's
second derivative lie between 1 and \( 1 + 2/r^2 \) for every image size
**[derived]**.  So the iteration counts above should carry over to large
volumes and to GPUs, where only the time per iteration changes.  The upper
end of this range also explains why the iteration counts grow about as
\( 1/r^2 \).

## The callers of `denoise`

Three kinds of callers use the denoiser, each with its own settings:

| Caller | Iterations | Stop | Start of each call |
|---|---|---|---|
| `denoise` and `denoise_stack`, called directly, as in demo 9 | 15 | 0.2 percent | the input |
| `QGGMRFDenoiserAgent` in `mace.py` | 8 | none | its previous output |
| The stack denoisers of `MACE4DModel` | 15 | 0.05 percent | the input, or the previous output with `denoiser_warm_start` |

**`QGGMRFDenoiserAgent`.**  With automatic parameters, the agent runs at
\( r = 1 \), because `auto_set_sigma_x` and `auto_set_sigma_prox` apply the same
formula to the same estimate.  On the MACE problem of `tests/test_mace.py`, 4
inner iterations gave the same NRMSE as 32 to within 0.00003.  With
`sigma_prox` multiplied by 5, the agent ran at \( r = 0.2 \).  There, 8 inner
iterations reached each NRMSE target at the same MACE iteration as 32.  One
inner iteration needed 46 MACE iterations to reach 0.01, against 28.  So the
default of 8 is enough at both strengths.  An agent at \( r = 0.1 \) or below
would need more **[inferred]**.

**The MACE4D stack denoisers.**  These run near \( r = 0.2 \).  Question 12 of
`plans/mace4d/decisions.md` found that the current rule fires after two or
three iterations of a warm-started call, whether or not the call has reached
its solution.  The gradient statistic does not have this problem.  At the
start of a warm-started call, it measures the distance from the previous
output to the solution for the new input.  So a call continues until it is
within the threshold of that solution, however small its steps are.  A call
whose input barely changed stops after one iteration.

## The automatic noise estimate

`estimate_image_noise_std` takes the standard deviation of a voxel and its
three backward neighbors, and averages it over a support of voxels above a
threshold.  On white noise in a flat volume, this returns
\( 0.798 \sigma \).  The factor comes from the four-value standard deviation
**[derived]**.

| Volume | Noise | Estimate / \( \sigma \) at 20 HU | at 60 HU | at 182 HU |
|---|---|---:|---:|---:|
| flat water | white | 0.798 | 0.798 | 0.798 |
| flat water | fdk | 0.687 | 0.687 | 0.687 |
| phantom, air at 0 | white | 1.226 | 0.920 | 0.836 |
| phantom, air at 0 | fdk | 1.124 | 0.809 | 0.721 |
| phantom, air at -1000 | white | 1.735 | 1.025 | 0.926 |
| phantom, air at -1000 | fdk | 1.635 | 0.869 | 0.770 |

These results show three biases.  The estimate is low on flat images.  Edges
raise it, more so at low noise.  The offset changes it, because the support's
threshold depends on the mean absolute value.

The automatic \( \sigma_x \) is 0.2 times the same statistic, computed on
about 20 rows.  So with `sigma_noise` given, the automatic \( r \) ranged from
0.15 on the flat volumes to 0.83 on the phantom with air at -1000 and 20 HU of
noise.  The automatic strength is therefore weak on images with strong edges
and low noise.

The bias matters for a call of `denoise(image)`.  On the fdk test volume,
`denoise` estimated 131 HU for noise of 182 HU.  Its MAP estimate had an rms
error of 73 HU over the flat region.  With `sigma_noise` = 182 and the
automatic \( \sigma_x \), the error was 39 HU.

A robust alternative is the median absolute first difference, divided by
\( 0.6745 \sqrt{2} \).  It returned 1.000 to 1.031 of the true level on white
noise, on every volume and at every offset.  On fdk noise it returned 0.74 to
0.77.  Any estimate from neighbor differences reads low on noise that is
positively correlated between neighbors **[derived]**.  There are three
options for the automatic estimate:

1. Keep the estimate, and state its behavior in the docstring.
2. Use the robust estimate for `sigma_noise` only.  On flat white noise this
   raises \( \sigma_y \) by about 25%, which strengthens the default
   denoising.
3. Use the robust estimate for `sigma_noise`, and set the automatic
   \( \sigma_x \) to a fixed ratio times it.  This removes the edge and offset
   dependence of the automatic strength, and the ratio needs tuning.

## Limits of this evidence

- The convergence cases used one phantom of 2.1 million voxels and one
  partition seed.
- The MACE test used one 2D problem of 64 by 64 pixels.
- Every run used the CPU.  The iteration counts should hold on a GPU, but
  this was not tested.
- The fdk noise is independent from slice to slice.  Real FDK noise is also
  correlated between slices.
- The test volume's flat-region error is not a measure of image quality
  alone.  At \( r = 0.05 \), the iterates came closer to the clean volume than
  the MAP estimate did, as `experiments/denoise_convergence.md` reports.
