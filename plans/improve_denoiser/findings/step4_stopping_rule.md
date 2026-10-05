# Step 4: the stopping rule and the default number of iterations

This page reports step 4 of
`plans/improve_denoiser/improve_qggmrf_denoiser.md`.  Step 4 asked how the
stopping rule and the default number of iterations of `denoise` should be set
for each place that calls it.  The page has four parts:

1. How far `denoise` stops from the MAP estimate, and a comparison of
   stopping rules.
2. The inner iterations of the MACE denoiser agent.
3. The bias of the automatic noise estimate.  The plan does not list this
   part.  It is here because the estimate sets \( \sigma_y \), and with it
   \( r \), for a call of `denoise(image)`.
4. A test of the stopping rules inside `MACE4DModel`, on GPUs and a real
   scan.  Greg asked for this test on 2026-10-04, after the first three
   parts.

The run records of the first three parts are
`experiments/denoise_convergence.md`, `experiments/mace_inner_iterations.md`,
and `experiments/noise_estimate.md`.  These measurements ran on CPU on
2026-10-03, with mbirtorch at commit b0b1879.  Commit b0b1879 adds PR #15 to
`greg_dev` at 5f62506, which holds the scaling fix of PR #14.  PR #15
compiles the denoiser update once for all noise levels.

The record of the GPU test is `experiments/mace4d_stopping.md`.  The test
ran on 2026-10-04 on 4 H100s of gautschi.  It used two branches, `greg_dev`
at 5f62506 and `prerelease`.  Some runs added a patch with the gradient rule
to `greg_dev`.  Paths that start with `mbirtorch/` or `tests/` are in the
mbirtorch repository.

## Terms

The page uses these terms:

- **\( \sigma_y \)**: the standard deviation of the noise, the parameter
  `sigma_noise` of `denoise`.  When `sigma_noise` is not given, `denoise`
  estimates it.
- **\( r = \sigma_x / \sigma_y \)**: the ratio of the prior's scale
  \( \sigma_x \) to \( \sigma_y \).  A smaller \( r \) gives a stronger
  prior.  For the MACE denoiser agent, the parameter `sigma_prox` takes the
  place of \( \sigma_y \).
- **MAP estimate**: the image that minimizes the cost of `denoise`.
- **VCD**: vectorized coordinate descent, the algorithm of `denoise`.  One
  VCD iteration updates every voxel once, one subset of voxels at a time, with
  16 subsets by default.
- **distance**: the rms difference between an image and the MAP estimate, in
  units of \( \sigma_y \).  The **largest distance** is the largest absolute
  difference at one voxel, in the same units.
- **change**: the rms change of the image in one iteration, in units of
  \( \sigma_y \).
- **exact gradient**: \( \sigma_y \) times the rms gradient of the cost at
  the image after an iteration.
- **gradient statistic**: the same, except that each voxel's gradient is
  taken just before the voxel's update in the sweep.
- **offset**: a constant added to every voxel.  An image in HU has an offset
  of -1000 relative to the same image with air at 0.
- **percent-change rule**: the stopping rule of `denoise` today.  It stops
  when \( 100 \lVert \Delta x \rVert_1 / \lVert x \rVert_1 \) falls below
  `stop_threshold_change_pct`, whose default is 0.2.
- **current defaults**: the percent-change rule at 0.2, together with
  `max_iterations` = 15.
- **MACE4D's rule**: the percent-change rule at 0.05, with a cap of 15
  iterations, as `MACE4DModel` sets it for its stack denoisers.
- **gradient rule**: a rule that stops when the gradient statistic falls
  below a threshold.
- **warm start**: the MACE4D option `denoiser_warm_start`.  With it, each
  call of a stack denoiser starts from the denoiser's previous output
  instead of its input.
- **fdk noise**: noise with the spectrum of an FDK image within each slice,
  and independent from slice to slice.  **fdk_z noise** is fdk noise with a
  correlation of 0.78 between adjacent slices.
- **MACE**: multi-agent consensus equilibrium, as implemented in
  `mbirtorch/mace.py`.

**[derived]** marks a claim that follows from the math.  **[measured]** marks
a claim that rests only on these measurements.  **[inferred]** marks a claim
that the measurements suggest but did not test.

## Summary

On the CPU, the current defaults stopped close to the MAP estimate in only
one setting.  On the 3D test volume, with noise independent between slices
and \( r \) from 0.19 to 0.4, they stopped 0.01 to 0.04 from the MAP
estimate.  In every other case tested, they stopped farther away:

- With fdk_z noise, at the automatic \( r \) of 0.17, they stopped 0.07 away.
- On a 2D slice, at the automatic \( r \) of 0.16, they stopped 0.17 away.
- At \( r = 0.1 \), they stopped 0.09 to 0.38 away.  At \( r = 0.05 \), they
  stopped 0.23 away.
- With 10000 added to the image, and with \( \sigma_y \) and \( \sigma_x \)
  held fixed, the percent-change rule stopped the loop after 1 to 3
  iterations, 0.12 to 0.85 away.

A rule that compares the change with a threshold in units of \( \sigma_y \)
removes the dependence on the offset.  It does not remove the dependence on
\( r \).  The change is smaller than the distance by a factor that grows
about as \( 1 / r^2 \).  So a fixed threshold on the change stops farther
from the MAP estimate as \( r \) falls.

The gradient rule stopped every case at nearly the same distance.  In units
of \( \sigma_y \), the exact gradient is an upper bound on the distance
**[derived]**.  The gradient statistic was at least 1.27
times the distance at every iteration of every case **[measured]**.  The VCD
sweep could compute the gradient statistic with one more sum per subset.  Its
cost was not timed on the CPU.  With a threshold of 0.01, all 13 cases
stopped between 0.004 and 0.007 from the MAP estimate.  The number of
iterations then ranged from 9 at \( r = 0.4 \) to 515 at \( r = 0.05 \).

The GPU test found the same behavior inside MACE4D on a real 4D scan.
Without the warm start, MACE4D's rule stopped every call at its cap of 15
iterations.  The sampled calls then stood 0.08 to 0.17 from their MAP
estimates.  The gradient rule at 0.01 stopped every sampled call 0.0052 to
0.0063 from its MAP estimate.  At the automatic \( r \) of 0.10, the
gradient rule raised the wall time of MACE4D by 21%, and it changed the
final image by 5.8%.  With the gradient rule, the denoiser warm start
changed the final image by 0.6%, against 9.7% with MACE4D's rule.  The
gradient statistic added little or no time per iteration **[inferred]**.

The number of iterations needed depends on \( r \), on the noise, and on
whether the image is 2D or 3D.  At \( r = 0.1 \), the 2D slice needed 131
iterations to reach a distance of 0.01, and the 3D volume needed 87.  The
size of the image mattered little.  Slices of 256 and 500 voxels on a side
were at nearly the same distance after 15 iterations.

The MACE denoiser agent needs no change on the problem tested.  Each call of
the agent runs 8 inner iterations, starting from its previous output.  With
8, MACE reached each accuracy target at the same MACE iteration as with 32,
at \( r = 1 \) and at \( r = 0.2 \).

The automatic noise estimate has three biases.  It returns 0.80 of the true
level on white noise in a flat image.  Edges raise it.  An offset that moves
air from 0 to -1000 also raises it.  When `sigma_noise` is given, the
automatic \( r \) depends on edges even more.  The reason is that the
automatic \( \sigma_x \) comes from a statistic that compares voxels 12 rows
apart on the test volume.  When `sigma_noise` is estimated, \( \sigma_y \)
inherits the three biases.

## Decisions for Greg

1. **Replace the percent-change rule with the gradient rule.**  The change
   would apply to the three code paths that run the VCD loop: `denoise` on
   one device, `denoise` on several devices, and `denoise_stack`.  The
   parameter `stop_threshold_change_pct` would need a new name, such as
   `stop_threshold`, because its unit changes from percent to
   \( \sigma_y \).  A threshold of 0 would still run exactly
   `max_iterations`.  Two callers in the package pass the old parameter.
   MACE4D passes 0.05, and the MACE denoiser agent passes 0.  I propose to
   update both, and to make the old name raise an error that names the new
   parameter.  The stopping rule of the tomography `recon` would not change.
2. **Choose the defaults.**  I propose a threshold of 0.01 and
   `max_iterations` = 200.  With this threshold, the output differed from the
   MAP estimate by less than 1% of the noise level in every case.  That
   distance was less than a tenth of the MAP estimate's own rms error over
   the flat region of the test volume.  At the automatic \( r \) of the test
   images, the rule ran 18 to 64 iterations, against at most 15 today.  At
   \( r = 0.1 \), it ran 104 to 151 iterations.  The cap of 200 was enough
   for \( r \ge 0.1 \) in every tested call that started from its input.
   With the MACE4D warm start, most calls of the second MACE iteration
   reached the cap, as the section on the GPU test reports.  At
   \( r = 0.05 \), the cap would stop the loop 0.04 from the MAP estimate.
   A log line would report each call that reaches `max_iterations`.  One
   frame of the real scan has 49 million voxels.  On one H100, the rule ran
   8 more iterations on it than the current defaults, and these iterations
   took 0.03 s.  The compilation before the first call took 8 to 16 s.
3. **Use the gradient rule in MACE4D.**  I propose that the stack denoisers
   of `MACE4DModel` use the gradient rule at 0.01, with a cap of 200, and
   that the denoiser warm start stay off by default.  The GPU test gives two
   reasons:
   - With the gradient rule, every sampled call stopped within 0.0063 of
     its MAP estimate.  With MACE4D's rule, every call stopped at the cap,
     and the sampled calls were 0.08 to 0.17 away.
   - The final image no longer depended on the warm start.  The warm start
     changed it by 0.63% with the gradient rule, against 9.7% with MACE4D's
     rule.

   The test also found three costs and unknowns:
   - The wall time rose by 21% at the automatic \( r \) of 0.10.  The
     denoiser time rose by a factor of 2.8, and the prox time did not
     change.  So the increase of the wall time depends on the ratio of the
     two, which differs from scan to scan **[inferred]**.  At
     \( r = 0.157 \), the increase was 5.5%, so the cost also depends on
     decision 4.
   - The final image changed by 5.8%.  It became smoother along axis 3, the
     axis of 728 voxels, and between frames.  It became rougher along the
     other two axes.  If the automatic \( \sigma_x \) was tuned while the
     denoisers stopped at 15 iterations, it may need new tuning
     **[inferred]**.  The scan has no reference image, so the test cannot
     say whether the new image is closer to the object.
   - MACE did not converge in 10 iterations.  So the difference between the
     consensus equilibria with the two rules is not known.

   Two cheaper options were not tested in full.  At the automatic \( r \) of
   0.10, a threshold of 0.03 would have stopped the sampled calls after 67
   to 85 iterations instead of 111 to 138.  They would have stopped 0.017 to
   0.019 from their MAP estimates.  From the timing of the runs with the
   gradient rule, the wall time would rise by about 12% instead of 21%
   **[inferred]**.  The effect of this threshold on the final image was not
   measured.  The remedy that question 12 of
   `plans/mace4d/decisions.md` planned, a tolerance that tightens with the
   outer loop, was not tested either.  In MACE4D, \( \sigma_y \) is a
   parameter of the denoiser's strength, not the measured noise of its
   input.  So the reason given in decision 2 for 0.01 does not apply there
   directly **[inferred]**.
4. **The automatic noise estimate.**  I propose no change in this step.  Any
   fix to the estimate would change the default \( r \), because the
   automatic \( \sigma_x \) comes from the same statistic.  The estimate also
   sets the default noise level of the MACE4D denoisers.  So a fix needs its
   own measurement of image quality.  The options are at the end of the
   section on the noise estimate.  The GPU test adds two facts.  On the real
   scan, the automatic estimate was 1.66 times option 2's noise level.  With
   option 2's noise level, the gradient rule raised the wall time of MACE4D
   by 5.5% instead of 21%, because \( r \) was larger.  The final image then
   differed by 6.6% from the image with the automatic noise level.  So
   option 2 gives a different image, not a cheaper way to reach the same
   image.

A rule that needs no new statistic is an alternative to decision 1.  It
estimates the remaining distance from the factor by which the change shrinks
per iteration.  At a threshold of 0.01, it stopped 0.009 to 0.015 from the
MAP estimate.  At 0.03, it stopped 0.015 to 0.079 away.  This rule has no
bound, and on some iterations its estimate was 0.30 times the distance.

## Why the exact gradient is an upper bound on the distance

In units of \( \sigma_y \), the cost is

$$
J(v) = \frac{1}{2} \lVert z - v \rVert^2 + P(v) ,
$$

where \( v = x / \sigma_y \), \( z = y / \sigma_y \), and \( P \) is the
prior.  \( P \) is convex for \( 1 \le q \le p \le 2 \).  A convex function
plus \( \lVert z - v \rVert^2 / 2 \) is 1-strongly convex.  So for any image
\( v \) and the minimizer \( v^* \),

$$
\lVert v - v^* \rVert \le \lVert \nabla J(v) \rVert .
$$

Dividing both sides by the square root of the number of voxels turns this into
a bound on the rms distance.  Let \( f \) be the same cost written in the
unit of the image, so that \( J(v) = f(\sigma_y v) \).  Then

$$
\nabla J(v) = \sigma_y \nabla f(\sigma_y v) .
$$

This is why the exact gradient and the gradient statistic multiply the
gradient of \( f \) by \( \sigma_y \).

\( J \) does not change when the image, \( \sigma_y \), and \( \sigma_x \)
are multiplied by the same constant.  It also does not change when the same
constant is added to the noisy image and to the current image.  So the
gradient statistic does not depend on the image's offset.  It also does not
change when the image, \( \sigma_y \), and \( \sigma_x \) are scaled
together **[derived]**.  Both statements assume that \( \sigma_y \) and
\( \sigma_x \) do not change with the offset.  The automatic parameters do
change with the offset, as the section on the noise estimate shows.

Each subset update of VCD already computes both terms of the gradient at its
voxels.  In `vcd_subset_denoiser`, the gradient of the data term is
`-fm_constant * cur_error_image`, and the gradient of the prior is
`prior_grad`.  The function passes both to `_identity_update_direction`.  The
gradient statistic needs the sum of the squares of their sum over the
subset's voxels.  Each voxel's gradient is taken just before the voxel's
update.  So the gradient statistic approximates the exact gradient rather
than equaling it, and the bound is proven only for the exact gradient.

The measurements confirm the bound.  From iteration 10 on, the median ratio
of the exact gradient to the distance was 1.15 to 1.76 across the cases.  The
smallest ratio was 1.07.  The median ratio of the gradient statistic to the
distance was 1.37 to 2.14.  The gradient statistic was never less than 1.27
times the distance, at any iteration of any case **[measured]**.  So a rule
on the gradient statistic never stopped before the distance was below its
threshold.

## Why a rule on the change fails at small r

VCD divides each voxel's gradient by the second derivative of its surrogate
cost.  Consider a voxel whose differences from its six neighbors are much
smaller than \( T \sigma_x \).  In units of \( \sigma_y \), the second
derivative there is \( 1 + 1/r^2 \), for \( p = 2 \) and \( q < 2 \)
**[derived]**.  The 1 comes from the data term.  The \( 1/r^2 \) comes from
the prior, whose six neighbor weights sum to 1.  So at such voxels the change
per iteration is about \( r^2 \) times the gradient.  The previous section
showed that the median ratio of the gradient to the distance was 1.15 to
2.14.

The measurements follow this pattern.  For fdk noise on the 3D volume, the
median ratio of the gradient statistic to the change was 4.4 at
\( r = 0.4 \), 15 at 0.2, 64 at 0.1, and 280 at 0.05.  These ratios are 0.58
to 0.70 times \( 1 + 1/r^2 \).  A threshold of 0.001 on the change therefore
stopped at distances from 0.002 at \( r = 0.4 \) to 0.107 at
\( r = 0.05 \).

The percent-change rule has the same problem, and it adds a dependence on the
offset.  Adding a constant to the image changes \( \lVert x \rVert_1 \).  It
changes neither the VCD updates nor \( \lVert \Delta x \rVert_1 \), as long as
\( \sigma_y \) and \( \sigma_x \) stay fixed.  On the test volume, with air at
0 and water at 1000, the percent-change rule at 0.2 stopped the loop after 7
to 34 iterations.  With 10000 added, it stopped the loop after 1 to 3
iterations.  The offset can also delay the stop.  In an image of water in
HU, the values are near 0, so \( \lVert x \rVert_1 \) is about the size of
the remaining noise.  The rule would then stop only at the cap of 15
**[inferred]**.

## How many iterations each case needs

The table gives the iterations needed to reach each distance.  It also gives
where the current defaults and the proposed rule stop.  The proposed rule is
a threshold of 0.01 on the gradient statistic.  The Case column uses three
labels:

- "\( r \) given" means that `sigma_noise` = 182 HU and
  \( \sigma_x = 182 r \).
- "automatic \( \sigma_x \)" means that `sigma_noise` = 182 HU and the
  denoiser chose \( \sigma_x \).
- "all automatic" means that the denoiser estimated both, as a call of
  `denoise(image)` does.

| Case | \( r \) | Distance after 15 | Iterations to 0.03 | to 0.01 | Current defaults: iteration, distance | Proposed rule: iteration, distance |
|---|---:|---:|---:|---:|---|---|
| fdk, \( r \) given | 0.400 | 0.0003 | 5 | 7 | 7: 0.010 | 9: 0.004 |
| fdk, all automatic | 0.259 | 0.009 | 10 | 15 | 10: 0.030 | 18: 0.005 |
| fdk, \( r \) given | 0.200 | 0.024 | 14 | 22 | 13: 0.033 | 27: 0.005 |
| white, automatic \( \sigma_x \) | 0.197 | 0.017 | 11 | 20 | 11: 0.030 | 25: 0.005 |
| fdk, automatic \( \sigma_x \) | 0.188 | 0.030 | 16 | 24 | 13: 0.041 | 30: 0.005 |
| white, \( r \) given | 0.100 | 0.081 | 46 | 91 | 13: 0.090 | 108: 0.007 |
| fdk, \( r \) given | 0.100 | 0.110 | 46 | 87 | 15: 0.110 | 104: 0.007 |
| fdk, \( r \) given | 0.050 | 0.229 | 256 | 444 | 15: 0.229 | 515: 0.007 |
| fdk_z, automatic \( \sigma_x \) | 0.167 | 0.070 | 23 | 34 | 15: 0.070 | 42: 0.005 |
| fdk_z, \( r \) given | 0.100 | 0.179 | 52 | 88 | 15: 0.179 | 106: 0.006 |
| 2D slice, fdk, automatic \( \sigma_x \) | 0.162 | 0.172 | 39 | 56 | 15: 0.172 | 64: 0.006 |
| 2D slice, fdk, \( r \) given | 0.100 | 0.383 | 88 | 131 | 15: 0.383 | 151: 0.006 |
| 2D slice, white, \( r \) given | 0.100 | 0.247 | 71 | 120 | 15: 0.247 | 139: 0.007 |

The columns of iterations to each distance, and the column of the proposed
rule, ignore the cap of 15.  Where the current defaults show iteration 15,
they reached the cap before the percent-change rule stopped the loop.  In the
all-automatic case, the denoiser estimated \( \sigma_y \) = 131.2 HU instead
of 182 HU.  Its distances are in units of 131.2 HU.

The kind of noise and the dimension of the image changed the iteration
counts.  At \( r = 0.1 \) on the 3D volume, white noise needed 91 iterations
to reach 0.01, and fdk noise needed 87.  So the correlation within a slice
mattered little.  The fdk_z noise slowed the first iterations.  After 15
iterations, its distance was 0.179, against 0.110 for fdk noise.  The 2D
slice was slower throughout.  These results suggest that VCD removes the error at high
spatial frequencies quickly and the error at low frequencies slowly
**[inferred]**.  Noise correlated between slices has less power at high
frequencies along the slice axis.  A 2D image has a larger share of its
frequencies near zero than a 3D volume **[derived]**.

This dependence explains why the plan reported slower convergence than the
3D volume shows.  On the plan's 2D slice at \( r = 0.1 \), 15 iterations left
the image 69.6 HU, or 0.38 \( \sigma_y \), from the MAP estimate.
`experiments/plan_slice_check.py` reproduced 69.6 HU with the current code.

The number of iterations should not depend on the size of the image
**[inferred]**.  In units of \( \sigma_y \), the cost's second derivative is
the identity plus the prior's second derivative.  The prior's second
derivative is a weighted graph Laplacian.  The weight of each neighbor pair
is \( \sigma_y^2 \rho''(\Delta) / 6 \), where \( \Delta \) is the pair's
difference.  For \( p = 2 \) and \( 1 \le q < 2 \), \( \rho'' \) is largest
at \( \Delta = 0 \), where it equals \( 1 / \sigma_x^2 \).  This was checked
numerically.  So each weight is at most \( 1 / (6 r^2) \).  The eigenvalues
of a graph Laplacian are at most twice the largest sum of weights at a
voxel, and a voxel has at most 6 neighbors.  So the eigenvalues of the cost's
second derivative lie between 1 and \( 1 + 2/r^2 \) for every image size
**[derived]**.

The slices agreed with this.  With the same layout of the array, the slices
of 256 and 500 voxels on a side were 0.383 and 0.373 from the MAP estimate
after 15 iterations.  So the iteration counts above should apply to larger
images with the same noise and the same dimension.  Only the time per
iteration should change.  The upper end \( 1 + 2/r^2 \) also explains why
the iteration counts grow about as \( 1/r^2 \) **[inferred]**.  One
iteration took 0.03 to 0.04 s for the 2.1 million voxels of the 3D volume on
4 CPU threads.

## The callers of `denoise`

Three kinds of callers use the denoiser.  Each kind has its own settings:

| Caller | Iterations | Stop | Start of each call |
|---|---|---|---|
| `denoise` and `denoise_stack`, called directly, as in demo 9 | 15 | percent-change rule at 0.2 | the noisy input |
| `QGGMRFDenoiserAgent` in `mbirtorch/mace.py` | 8 | none | its previous output |
| The stack denoisers of `MACE4DModel` | 15 | percent-change rule at 0.05 | the noisy input, or the previous output with `denoiser_warm_start` |

**`QGGMRFDenoiserAgent`.**  The agent has no automatic parameters.  In
`tests/test_mace.py`, the caller passes the model's automatic `sigma_prox`
and `sigma_x`.  Then \( r = \sigma_x / \sigma_{prox} = 1 \), because
`auto_set_sigma_x` and `auto_set_sigma_prox` apply the same formula to the
same estimate.  On that problem, 8 inner iterations gave the same NRMSE as 32
to within 0.00001 at MACE iterations 1, 10, 20, 40, and 60.  With
`sigma_prox` multiplied by 5 in both agents, the denoiser agent ran at
\( r = 0.2 \).  There, 8 inner iterations reached each NRMSE target at the
same MACE iteration as 32.  Earlier in the run, the NRMSE with 8 was higher.
At MACE iteration 10 it was 0.0585, against 0.0509 with 32.  One inner
iteration needed 46 MACE iterations to reach 0.01, and 8 or more needed 28.
These results indicate that the default of 8 is enough at both values of
\( r \).  The standalone counts above are for calls that start from the
noisy input, and they grow about as \( 1/r^2 \).  So an agent at
\( r = 0.1 \) or below may need more than 8 inner iterations
**[inferred]**.

**The MACE4D stack denoisers.**  With automatic parameters, MACE4D sets the
\( \sigma_y \) of these denoisers to the noise estimate of the initial
image.  It sets \( \sigma_x \) to 0.2 times a similar statistic.  On the
real scan of the GPU test, they ran at \( r = 0.10 \).  Work on question 12
of `plans/mace4d/decisions.md` found that, with the denoiser warm start on,
the percent-change rule at 0.2 stopped each call after two or three
iterations.  The calls stopped before they had finished.  The MACE4D plan
then lowered the threshold to 0.05 as a stopgap.  Question 12 planned a
measurement for a later stage of the MACE4D plan, and the GPU test made that
measurement.  Question 12's planned remedy is an inner tolerance that
tightens with the outer loop.

The gradient statistic does not depend on the warm start in this way.  At the
start of a warm-started call, the exact gradient is an upper bound on the
distance from the previous output to the MAP estimate for the new input
**[derived]**.  So a call with the gradient rule stops only when its
distance to that MAP estimate is below the threshold **[measured]**.  The
size of the call's steps does not matter.  In the GPU test, the 6 sampled
warm-started calls of MACE iterations 5 and 10 stopped 0.0061 to 0.0063 from
their MAP estimates.  The calls that started from their input stopped 0.0055
to 0.0063 away.  At MACE iteration 2, though, most warm-started calls reached
the cap of 200 before the threshold.  A tolerance that tightens with the
outer loop could be a threshold on the gradient statistic **[inferred]**.

## The automatic noise estimate

`estimate_image_noise_std` takes the standard deviation of a voxel and its
three backward neighbors.  It averages this standard deviation over a
support of voxels above a threshold.  On white noise in a flat volume, this
returns \( 0.798 \sigma \).  The factor 0.798 comes from the standard
deviation of four values, which divides by 4 and is biased low
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
raise it, and they raise it more at low noise.  The offset changes it,
because the support's threshold depends on the mean absolute value.

The automatic \( \sigma_x \) is 0.2 times the same statistic, computed on
every 12th row of the 256 rows.  In that subsample, neighbors along the row
axis are 12 voxels apart.  So edges raise the statistic of \( \sigma_x \)
much more than they raise the noise estimate.  On the phantom with 20 HU of
white noise, the statistic of \( \sigma_x \) was 3.33 times the true level,
while the estimate was 1.23 times it.  With `sigma_noise` given, the
automatic \( r \) ranged from 0.15 on the flat volumes to 0.83 on the
phantom with air at -1000 and 20 HU of noise.  These results indicate that
the automatic \( r \) is large, and the denoising weak, on images with strong
edges and low noise.

The bias matters for a call of `denoise(image)`.  On the fdk test volume,
`denoise` estimated 131 HU for noise of 182 HU.  Its MAP estimate had an rms
error of 73 HU over the flat region.  With `sigma_noise` = 182 and the
automatic \( \sigma_x \), the error was 39 HU.  The same estimate sets the
default noise level of the MACE4D denoisers.

A robust alternative is the median absolute first difference, divided by
\( 0.6745 \sqrt{2} \).  It returned 1.000 to 1.031 of the true level on white
noise, on every volume and at both offsets.  On fdk noise it returned 0.74 to
0.77.  Any estimate from neighbor differences is low on noise that is
positively correlated between neighbors **[derived]**.  There are three
options for the automatic estimate:

1. Keep the estimate, and state its behavior in the docstring.
2. Use the robust estimate for `sigma_noise` only.  Across the tested
   volumes, this would multiply \( \sigma_y \) by 0.47 to 1.25, and the
   automatic \( r \) by the inverse factor.
3. Use the robust estimate for `sigma_noise`, and set the automatic
   \( \sigma_x \) to a fixed ratio times it.  This removes the dependence of
   the automatic \( r \) on edges and on the offset.  The ratio needs tuning.

## The GPU test on MACE4D

Greg asked on 2026-10-04 for timing and results on GPUs and real data before
he chooses the stopping rule.  The test ran six MACE4D reconstructions of
the Lilly 4DCT phantom scan on 4 H100s.  The scan has 99 frames of 260 by
260 by 728 voxels, and each run did 10 MACE iterations from the same initial
image.  The record is `experiments/mace4d_stopping.md`, and the tables are
in `experiments/results/mace4d_stopping/`.

This section uses two more terms.  A **MACE iteration** is one iteration of
MACE4D's outer loop.  A **call** is the denoising of one volume.  MACE4D
denoises its volumes in batches, and each volume of a batch stops on its
own.

Every version uses the VCD solver.  The six versions differ in one setting
at a time:
- A: the `prerelease` branch, without the scaling fix of PR #14, with
  MACE4D's rule.
- B: the `greg_dev` branch, with the scaling fix and MACE4D's rule.
- C: B with the gradient rule at 0.01, with a cap of 200 iterations.
- D: C with option 2's noise level.
- Bw and Cw: B and C with the denoiser warm start on.

In the table, the iterations per call are a mean over MACE iterations 2 to
10.  Each run sampled 9 calls, at MACE iterations 1, 5, and 10.  The
distance at the stop is the range over these calls.  The wall time covers
the whole reconstruction.  The last column is the rms difference between
the run's final image and B's, divided by the rms of a final image.  The rms
of every final image was 0.0097 to 0.0099, so the choice of image changes
the ratio by less than 2%.

| Version | \( r \) | Iterations per call | Distance at the stop | Wall time (s) | Wall time relative to B | Difference of the final image from B |
|---|---:|---:|---|---:|---:|---:|
| A | 0.102 | 15 | 0.084 to 0.171 | 1399 | -1.3% | 0.074% |
| B | 0.102 | 15 | 0.084 to 0.172 | 1418 | | |
| C | 0.102 | 106 | 0.0055 to 0.0063 | 1713 | +20.8% | 5.8% |
| D | 0.157 | 44 | 0.0052 to 0.0062 | 1496 | +5.5% | 2.2% |
| Bw | 0.102 | 15 | 0.081 to 0.216 | 1447 | +2.0% | 9.7% |
| Cw | 0.102 | 109 | 0.0055 to 0.0063 | 1790 | +26.2% | 6.2% |

MACE4D's rule never stopped a call of A or B before its cap.  After 15
iterations, the sampled calls were 0.08 to 0.17 from their MAP estimates.
Their largest distance was up to 7.5.

The gradient rule brought every sampled call of C to 0.0055 to 0.0063 from
its MAP estimate.  At the automatic \( r \) of 0.10, it needed a mean of 106
iterations per call, and it raised the wall time by 21%.  The denoiser time
rose by a factor of 2.8, and the prox time did not change.  So the increase
of the wall time depends on the ratio of the two, which differs from scan
to scan **[inferred]**.  At \( r = 0.157 \), with option 2's noise level, the
rule needed 44 iterations, and the wall time rose by 5.5%.  These counts are
consistent with growth as \( 1/r^2 \), as on the CPU.

The gradient statistic added little or no time per iteration
**[inferred]**.  B does not compute the statistic, and C and D do.  A line
through the denoiser times of C and D predicts 67 s per MACE iteration for
B's 15 iterations, and B took 71 s.  On one frame denoised alone, a line
through the sweep times of C and D predicts 114.5 ms for B, and B took
113.5 ms.

The scaling fix changed the wall time of MACE4D by 1.3% and its final image
by 0.074%.  On one frame denoised alone, at \( r = 0.25 \), the fix
mattered.  Without the fix, the largest distance was 10.2.  With the fix, it
was 0.30.  On that frame, the current defaults stopped 0.019 from the MAP
estimate, close to the CPU results at a similar \( r \).

The gradient rule changed the final image by 5.8%.  The last MACE iteration
of B changed the image by 1.15%.  The image became smoother along axis 3,
the axis of 728 voxels, and between frames.  The adjacent-voxel statistic of
option 2 fell by 20% along axis 3 and by 12% between frames.  It rose by 3%
to 7.5% along the other two axes.  The test has no reference image, so it
cannot say which image is closer to the object.

No run had converged after 10 MACE iterations.  In the last MACE iteration
of B, the change of the image shrank by 11%.  If it kept shrinking at that
rate, the remaining changes of B would add up to about 10% **[inferred]**.
So the difference between the consensus equilibria of B and C is not known.

The gradient rule made the final image nearly independent of the warm
start.  With MACE4D's rule, the warm start changed the final image by 9.7%.
With the gradient rule, it changed the final image by 0.63%.

The warm start did not save time with the gradient rule.  At MACE iteration
2, 914 of the 1248 warm-started calls reached the cap of 200.  Their
gradient statistic was then at a median of 0.012.  Over the 10 MACE
iterations, Cw took 4.5% longer than C.

With MACE4D's rule, the warm start makes the percent-change rule at 0.2
stop calls early.  Bw had 6 sampled calls at MACE iterations 5 and 10.  The
percent-change rule at 0.2 would have stopped 4 of them after 1 to 9
iterations.  Question 12 of `plans/mace4d/decisions.md` describes this
behavior.  It also plans a measurement of each setting's distance from a
converged reference, per call and in the final result.  The test made that
measurement, with run C as the reference for the final result.  The final
image of B was 5.8% from that of C, Bw 6.8%, and Cw 0.63%.

Option 2's noise level was 0.00362, and the automatic estimate was 0.00599.
So on this scan, the automatic estimate was 1.66 times option 2's noise
level.  With option 2's noise level, the final image differed from C's by
6.6%.  So option 2 gives a different image, not a cheaper way to reach C's
image.  On this volume, the automatic estimate compares voxels 10 apart,
because the volume is large.  The section on the noise estimate shows that
edges raise such an estimate.  It also shows that noise correlated between
neighbors lowers option 2's estimate.  So the test does not show which
estimate is closer to the true noise level.

## Limits of this evidence

- The convergence cases used one phantom and one partition seed.
- The noise models are simple.  Most cases use fdk noise that is independent
  from slice to slice.  The fdk_z cases add one level of correlation between
  slices.  On one real FDK frame at \( r = 0.25 \), the current defaults
  needed 16 iterations to reach a distance of 0.01.  The 3D test volume with
  fdk noise at \( r = 0.26 \) needed 15.
- The MACE test used one 2D problem of 64 by 64 by 1 voxels.
- The GPU test used one scan, one seed, and 10 MACE iterations, after which
  no run had converged.  It sampled 9 calls per run, all on the middle
  planes.  Its final images have no reference, so the test shows how much
  the versions differ, not which image is closer to the object.
- `denoise` on several devices was not run.  It uses the same VCD update, so
  its convergence per iteration should be similar **[inferred]**.
- Above about 17 million voxels, the noise estimate reads a subsample whose
  neighbors are 2 or more voxels apart.  So its biases on large volumes may
  differ from the table.
- Distance to the MAP estimate is not the same as image quality.  At
  \( r = 0.1 \) and below on the 3D volume, some iterates came closer to the
  clean volume than the MAP estimate did, as
  `experiments/denoise_convergence.md` reports.
