# The clamp on alpha and the scaling of the tomography update

The script is `alpha_clamp.py`, and its results are in
`results/alpha_clamp/`.  The runs used mbirtorch `greg_dev` at commit
78d8cf7.  They ran on one CPU with 4 threads on 2026-10-05.  Every sinogram
is synthetic.

## The question

Step 6 of `improve_qggmrf_denoiser.md` asks two questions.  The first is
whether the tomography models need a scaling adjustment like the one that
PR #14 made in the denoiser.  PR #14 made the denoiser's update give the same
result in any unit of the image.  The second question is whether the clamp
on the step size alpha should be removed, or whether a warning should be
printed when alpha is far above 1.5.

VCD, the vectorized coordinate descent of mbirtorch, is the solver of
`recon` and of the proximal map, `prox_map`.  VCD divides the pixels of the
image into subsets.  Each subset update moves the voxels of one subset along
a direction.  A line search then multiplies the direction by a step
size, alpha.  The code clamps alpha to the interval from
\( 1.19 \times 10^{-7} \) to `max_alpha`.  The default of `max_alpha` is 1.5
(`mbirtorch/_utils.py:97`).  `TranslationModel` sets it to 1.3, with the
comment "A larger line search cap has shown instabilities in this geometry"
(`mbirtorch/translation_model.py:271`).  The denoiser uses 1.5 as a constant
in each of its three updates (`mbirtorch/denoising.py`, lines 229, 282 and
1054).

In this record, "the clamp" means the operation that limits alpha.  "The
default `max_alpha`" means the bound that a model sets for itself: 1.3 in
the translation geometry and 1.5 in the others.

## Summary

**The tomography models need no scaling adjustment.**  Their update gives
the same result in any unit of the image, when the weights do not change
with the scale of the sinogram **[derived]**.  In parallel and cone beam,
multiplying the sinogram by \( 2^{10} \) or \( 2^{-10} \) multiplied the
`recon` result by exactly the same factor **[measured]**.

**With the clamp removed, alpha exceeded the default `max_alpha` in 8 of 42
cases [measured].**  These cases fall into two groups:

- in the translation geometry, 1.7% to 4.1% of the subsets exceeded 1.3, and
  the largest alpha was 2.31;
- in the cone-beam proximal maps at sharpness -1, 1.1% of the subsets
  exceeded 1.5, and the largest alpha was 1.67.

In the other 34 cases, the largest alpha was 1.37.  These cases covered
`recon` and the proximal map in parallel beam, cone beam and the multiaxis
geometry.

**Where the clamp reduced alpha, it changed the cost very little.**  In the
translation geometry, the runs with and without the clamp reached the same
cost within 0.1 iterations of each other **[derived]**.  In the cone-beam
proximal maps, the clamp changed the cost by no more than the rounding of
float32 arithmetic **[measured]**.

**The damping of the slice mean causes the large values of alpha in the
cone-beam proximal maps.**  The cone-beam update damps the mean of the
gradient in each slice of a subset.  With this damping turned off, the
largest alpha fell from 1.67 to 1.03 **[measured]**.

**The clamp is not needed to keep a step from raising the cost.**  The plan
states that the clamp was meant to limit steps from a local model of a
nonlinear problem.  With the positivity constraint off, the alpha of the
line search cannot raise the cost, with or without the clamp **[derived]**.

**Recommendation.**  Keep the clamp and its defaults, and make no scaling
adjustment.  Removing the clamp would gain nothing measurable, and in cone
beam the clamp keeps part of the damping of the slice mean.  Consider a
warning when most subsets of an iteration ask for an alpha above 10.  Before
PR #14, an error in the denoiser asked for an alpha of at least 328 in every
subset **[derived]**.  No subset in these tests asked for more than 3.6
**[measured]**.  The warning is a code change, so it waits for Greg's
decision.

## The subset update

The cost of `recon` is the sum of a data term and a qGGMRF prior.  The data
term is proportional to the weighted squared error of the sinogram, divided
by \( \sigma_y^2 \).  The prior sums a potential function over pairs of
neighboring voxels.  The potential depends on the difference of the two
voxels, and \( \sigma_x \) sets the scale of that difference.  In the
denoiser, the ratio \( r = \sigma_x / \sigma_y \) sets the strength of the
prior.

In the proximal map, a quadratic penalty on the distance of each voxel from
the input image replaces the qGGMRF prior.  Its curvature is
\( 1/\sigma_{prox}^2 \) at each voxel.  This record calls this penalty the
proximal prior.

Each subset update computes a direction and then alpha.  The direction at
voxel \( j \) is

$$
d_j = -\frac{g_j}{h_{f,j} + h_{p,j}} ,
$$

where \( g_j \) is the gradient of the cost, \( h_{f,j} \) is the diagonal
of the Hessian of the data term, and \( h_{p,j} \) is the curvature of the
prior's surrogate.  The code computes \( h_{f,j} \) as `fm_constant *
fm_hessian` (`mbirtorch/tomography_model.py:2486`), and `fm_constant` is
\( 1/\sigma_y^2 \).

The line search sets alpha to minimize a quadratic model of the cost along
the direction \( d \):

$$
\alpha = \frac{-\sum_j g_j d_j}{d^\top H_f d + \sum_j h_{p,j} d_j^2} .
$$

Here \( H_f \) is the Hessian of the data term over the voxels of the
subset.  The code adds \( 1.19 \times 10^{-7} \) to the denominator and then
clamps alpha (`mbirtorch/tomography_model.py:2534`).  This record calls
\( 1.19 \times 10^{-7} \) the small constant.

## Why the tomography update gives the same result in any unit

The cost of `recon` is unchanged when the sinogram, the image, \( \sigma_y \)
and \( \sigma_x \) are all multiplied by the same \( c \).  The data term
divides the squared error by \( \sigma_y^2 \), so its value does not change.
The qGGMRF potential depends on a voxel difference only through its ratio to
\( \sigma_x \), so the prior does not change either **[derived]**.  The
minimizer of the cost therefore scales by \( c \).

Each subset update keeps this property.  Under the scaling, \( g_j \) scales
by \( 1/c \), and both curvatures scale by \( 1/c^2 \).  So \( d_j \) scales
by \( c \).  The numerator and the denominator of alpha do not change, so
alpha does not change **[derived]**.  The small constant therefore has the
same effect at every scale.

The other updates keep this property too.  The cone-beam direction subtracts
part of a weighted mean of \( g \), which scales as \( g \) does.  The
automatic \( \sigma_{prox} \) of the proximal map scales with the image as
\( \sigma_x \) does.

This property assumes that the weights do not change with the scale.  The
weight types 'transmission', 'transmission_root' and 'emission' of
`gen_weights` are functions of the sinogram values
(`mbirtorch/vcd_utils.py:285`).  So these weights change when the sinogram
is multiplied by \( c \), and the result changes with them.

Before PR #14, the denoiser's update did not have this property.  Its
direction used 1 in place of \( h_{f,j} = 1/\sigma_y^2 \).  Take an image in
HU with \( \sigma_y = 182 \) and \( r = 0.1 \).  The correct direction was
then at least 328 times as long as the direction of the code, at every
voxel **[derived]**.  So the line search would have needed an alpha of at
least 328 in every subset, but the clamp held alpha at 1.5.  The plan states
that the factor is at most 1/490.  That bound leaves out the curvature from
the 2 neighbors along the third axis of the image.  The code counts these
neighbors even in a 2D image, where both neighbors are the pixel itself
(`mbirtorch/qggmrf.py:125` and 146) **[derived]**.

## When alpha exceeds 1

**In the denoiser, alpha is at most 1.**  With the direction above, the
numerator equals \( \sum_j (h_{f,j} + h_{p,j}) d_j^2 \).  The forward model
of the denoiser is the identity, so \( H_f \) is diagonal.  Then
\( d^\top H_f d = \sum_j h_{f,j} d_j^2 \), and without the small constant
the ratio is 1 **[derived]**.  With the small constant, alpha equals
\( N / (N + 1.19 \times 10^{-7}) \),
where \( N \) is the numerator.  So alpha falls well below 1 once \( N \)
nears \( 10^{-7} \), and the upper bound of the clamp never acts in the
denoiser.

**In `recon`, alpha differs from 1 by the cross terms of the data term.**
With the direction above,

$$
\alpha = 1 + \frac{\sum_j h_{f,j} d_j^2 - d^\top H_f d}{d^\top H_f d + \sum_j h_{p,j} d_j^2} .
$$

The difference \( \sum_j h_{f,j} d_j^2 - d^\top H_f d \) is minus the sum of
\( H_{f,jk} d_j d_k \) over all pairs \( j \ne k \) in the subset.  Every
entry of \( H_f \) is nonnegative, because the projector and the weights are
nonnegative.  So with this direction, alpha exceeds 1 only when voxels of
the subset that share rays move in opposite directions **[derived]**.  Alpha
falls far below 1 when many voxels of the subset share rays and move
together.  Every model except cone beam uses this direction.

**In cone beam, the damping of the slice mean can raise alpha.**  The
cone-beam direction is

$$
d_j = -w_j \left( g_j - (1 - s_m) \bar g_m \right), \qquad w_j = \frac{1}{h_{f,j} + h_{p,j}} ,
$$

where \( m \) is the slice of voxel \( j \).  Here \( \bar g_m \) is the
mean of \( g \) over the voxels of the subset in slice \( m \), weighted by
\( w_j \).  The damping factor \( s_m \) lies between 0.25 and 0.5
(`mbirtorch/cone_beam.py:42`).  Let \( \phi \) be the fraction of the
weighted sum of squares of \( g \) in slice \( m \) that comes from
\( \bar g_m \):

$$
\phi = \frac{\bar g_m^2 \sum_j w_j}{\sum_j w_j g_j^2} ,
$$

where both sums run over the voxels of the subset in slice \( m \).

Suppose the cross terms of the data term are small compared with the
curvature of the prior.  This holds when the proximal prior is strong.
Suppose also that \( \phi \) and \( s_m \) are the same in every slice of
the subset.  Then

$$
\alpha \approx \frac{1 - (1 - s_m) \phi}{1 - (1 - s_m^2) \phi} .
$$

This ratio is 1 when \( \phi = 0 \) and \( 1/s_m \) when \( \phi = 1 \)
**[derived]**.  So alpha approaches \( 1/s_m \), which is 2 to 4, as
\( \phi \) approaches 1.  In that case the line search undoes most of the
damping.

**The quadratic model can understate the curvature of the prior's
surrogate, but by at most a factor of 2.**  A subset always updates every
slice of each of its pixels, so it updates pairs of adjacent slices
together.  For such a pair of voxels \( j \) and \( k \), the curvature of
the surrogate along \( d \) grows with \( (d_j - d_k)^2 \).  The quadratic
model counts \( d_j^2 + d_k^2 \) instead.  Since \( (d_j - d_k)^2 \) is at
most \( 2(d_j^2 + d_k^2) \), the curvature of the model is at least half of
the curvature of the surrogate **[derived]**.  The data term is quadratic,
so its part of the model is exact.  The proximal prior acts on each voxel
alone, so its part of the model is exact too.

**With the positivity constraint off, alpha cannot raise the cost.**  The
surrogate lies above the prior and equals it at the current image.  Along
\( d \), the cost with the surrogate in place of the prior is a convex
parabola in alpha.  Its curvature is at most twice the curvature of the
model.  So at the alpha of the model, the parabola is no higher than at
alpha = 0.  By convexity, the parabola is no higher at any smaller alpha
either.  So neither the alpha of the model nor a clamped alpha can raise the
cost **[derived]**.  The positivity constraint removes this guarantee,
because it clips the update after alpha is chosen
(`mbirtorch/tomography_model.py:2541`).

## The tests

Greg approved tests of `recon` and of the proximal map in parallel and cone
beam.  The approved cases used noisy data at the automatic \( r \) and at
\( r = 0.1 \).  I made these additions:

- the translation geometry, whose model sets its own `max_alpha`, and the
  multiaxis geometry, so that the tests cover all four tomography models of
  the package;
- noiseless sinograms in parallel and cone beam;
- three values of the sharpness in place of the two values of \( r \), for
  the reason given in Test 2;
- cases that start from a converged image, and cases with one subset;
- a check, in Test 2, of how much neighboring voxels share rays;
- Test 3, which reruns the cases in which alpha exceeded the default
  `max_alpha`, with and without the clamp;
- a check, in Test 3, that a shorter run repeats the first iterations of a
  longer run;
- Test 4, which turns off the damping of the slice mean in cone beam;
- Test 5, which replaces the small constant;
- Test 6, which adds a constant to the input of a cone-beam proximal map.

Each test runs as `python alpha_clamp.py` followed by the name of the test.
The names are `scale` for Test 1, `alpha` for Tests 2 and 3, `overlap` for
the check in Test 2, `prefix` for the check in Test 3, `damping` for Test 4,
`epsilon` for Test 5, and `offset` for Test 6.

## Test 1: `recon` gives the same result in any unit

Test 1 reconstructs a noisy sinogram in parallel and cone beam, with 10
iterations each.  The sinogram and its noise are those of Test 2 below.  For
each scale \( c \) in \( 2^{-10} \), 1 and \( 2^{10} \), the test
reconstructs the sinogram multiplied by \( c \) and divides the result by
\( c \).  The test uses two settings of \( \sigma_y \) and \( \sigma_x \):

- **pinned**: \( \sigma_y \) and \( \sigma_x \) are \( c \) times the
  automatic values at \( c = 1 \);
- **automatic**: `recon` sets \( \sigma_y \) and \( \sigma_x \) itself.

The scales are powers of 2, so multiplying by \( c \) adds no rounding.  The
models compile their kernels, as they do by default.

In both geometries and with both settings, the result divided by \( c \)
equaled the result at \( c = 1 \) exactly **[measured]**.  The automatic
\( \sigma_y \) and \( \sigma_x \) were exactly \( c \) times their values at
\( c = 1 \).  These results agree with the derivation above.

One difference unrelated to the scale appeared in cone beam.  The first
cone-beam run in the process differed from every later cone-beam run, at
every scale, by \( 2.9 \times 10^{-7} \) of the largest voxel value.  This
difference is 2.5 times the float32 machine epsilon.  The first run is also
the run that compiles the kernels.  I did not look for the cause, because
the difference does not depend on the scale.

## Test 2: the largest alpha without the clamp

Test 2 sets `max_alpha` to \( 10^{30} \), which removes the clamp, and
records the alpha of every subset.  If no alpha of a run exceeds the default
`max_alpha`, the clamp would not have changed that run.  The recording
replaces `torch.clamp`, so these runs do not compile their kernels.  In every
run, the number of recorded values equaled the number of subsets.  The mean
of the recorded values in each iteration matched the mean alpha that `recon`
reports, within \( 5.1 \times 10^{-7} \).

The test uses four geometries:

- **parallel**: 64 views over 180 degrees, a detector of 16 rows and 64
  channels, and a recon of 64 by 64 by 16 voxels;
- **cone**: 64 views over 360 degrees, a magnification of 2, and the same
  detector and recon shape;
- **multiaxis**: 64 views over 180 degrees at an elevation of 30 degrees,
  the same detector, and a recon of 64 by 64 by 18 voxels;
- **translation**: a 7 by 7 grid of 49 translations, a detector of 64 rows
  and 64 channels, a magnification of 4, and a recon of 4 by 72 by 72
  voxels.

The first three geometries use the Shepp-Logan phantom of
`generate_demo_data`.  The translation geometry uses the dot phantom of
`gen_translation_phantom`.  Its translations are spaced 3 length units
apart, which is 12 voxel widths.  Its recon has 4 rows along the beam, and
each voxel is 32 times as long in that direction as it is wide.

With the 16 by 64 detector of the other geometries, `generate_demo_data`
gives a translation recon only 1 row thick.  So the translation geometry
follows `tests/test_translation.py` instead.  It differs from that test in
three ways: a 7 by 7 grid of translations in place of 4 by 4, a spacing of 3
along z in place of 2, and a detector of 64 by 64 in place of 40 by 32.

Each case runs `recon`, or the proximal map with a `recon` result as its
input.  The sinograms differ by geometry:

- in parallel and cone beam, the cases use the sinogram without noise and
  with noise;
- in the translation and multiaxis geometries, the cases use the sinogram
  with noise.

The noise is white and Gaussian, with a standard deviation 30 dB below the
rms of the sinogram.  The automatic \( \sigma_y \) assumes this
signal-to-noise ratio.  The input of each proximal map is the `recon` result
at sharpness 1 for the same sinogram.

Each of these cases runs at sharpness -1, 1 and 3.  The approved test plan
asked for \( r = 0.1 \) and the automatic \( r \).  The ratio \( r \) suits
the denoiser, whose image and noise have the same units.  In `recon`, the
image and the sinogram have different units, so the test varies the
sharpness instead.  The automatic \( \sigma_x \) is proportional to 2 raised
to the sharpness.  Sharpness -1, 1 and 3 give \( \sigma_x \) equal to 0.1,
0.4 and 1.6 times an estimate of the standard deviation of the image.
Sharpness 1 is the default.  The automatic \( \sigma_{prox} \) of the
proximal map uses the same formula.

Each case runs 40 iterations from the direct reconstruction, which is the
default start of `recon`.  The default partitions have 4, 16 and 64 subsets
in the first three iterations and 128 subsets in each later iteration.  The
partition makes all subsets the same size by repeating some pixels.  In the
first three geometries, the region of reconstruction has 3096 pixels.  Each
of the 128 subsets holds 25 pixels, so 104 pixels appear in two subsets.  The
translation recon has 288 pixels.  Each of its 128 subsets holds 3 pixels,
so 96 pixels appear in two subsets.

Two more kinds of cases ran in parallel and cone beam, on the sinogram with
noise:

- **converged start**: `recon` and the proximal map at sharpness 1, for 10
  iterations, started from the result of the 40-iteration `recon` at
  sharpness 1.  The first iterations then use a few large subsets on an
  image whose remaining error is small, as in a warm-started call.
- **one subset**: `recon` at sharpness 3 with every pixel in one subset at
  every iteration.  This partition couples the updated voxels the most.

Table 1 gives the largest alpha of each case that started from the direct
reconstruction with the default partitions.  Bold marks a value above the
default `max_alpha`.

Table 1.  The largest alpha with the clamp removed, from the direct
reconstruction with the default partitions.

| Geometry | Operation | Noise | Sharpness -1 | Sharpness 1 | Sharpness 3 |
|---|---|---|---:|---:|---:|
| parallel | `recon` | none | 1.13 | 1.30 | 1.10 |
| parallel | `recon` | 30 dB | 1.14 | 1.31 | 1.07 |
| parallel | `prox_map` | none | 1.01 | 1.06 | 1.10 |
| parallel | `prox_map` | 30 dB | 1.01 | 1.06 | 1.07 |
| cone | `recon` | none | 1.15 | 1.37 | 1.10 |
| cone | `recon` | 30 dB | 1.17 | 1.32 | 1.08 |
| cone | `prox_map` | none | **1.66** | 1.12 | 1.12 |
| cone | `prox_map` | 30 dB | **1.67** | 1.11 | 1.10 |
| multiaxis | `recon` | 30 dB | 1.00 | 0.99 | 0.85 |
| multiaxis | `prox_map` | 30 dB | 0.99 | 1.16 | 1.04 |
| translation | `recon` | 30 dB | **2.09** | **2.15** | **2.30** |
| translation | `prox_map` | 30 dB | **1.89** | **2.24** | **2.31** |

The other cases gave smaller values **[measured]**.  From a converged start,
the largest alpha of `recon` was 1.23 in parallel beam and 1.26 in cone
beam.  The largest alpha of the proximal map from a converged start was 1.04
in both.  With one subset, the largest alpha was 0.029 in parallel beam and
0.067 in cone beam.  In that partition, most voxels share rays with many
others and move together.  So the formula above gives an alpha far below 1
**[inferred]**.

Alpha exceeded the default `max_alpha` in two settings.

In the cone-beam proximal maps at sharpness -1, 1.1% of all subsets exceeded
1.5.  All of them fell in iterations 5 to 11.  In the worst iteration, 16 of
the 128 subsets exceeded 1.5.  The mean alpha of iterations 8 and 9 was
1.34.  By default, `prox_map` runs 3 iterations
(`mbirtorch/tomography_model.py:3186`).  So a call with the default settings
would stop before these iterations.

In the translation geometry, 1.7% to 4.1% of the subsets exceeded 1.3, and
0.3% to 1.7% exceeded 1.5.  In the worst iteration, 11 of the 128 subsets
exceeded 1.3.  The largest mean alpha of any iteration was 0.93.

In the translation geometry, adjacent slices of one pixel share many rays.
The overlap check measured the inner product of the projections of two
neighboring voxels, divided by the product of their norms.  At 12 voxels,
this overlap was 0.22 to 0.31 for adjacent slices in the translation
geometry **[measured]**.  It was 0 in parallel beam and at most 0.12 in cone
beam.  A subset always updates every slice of each of its pixels.  So by the
formula above, updates that move adjacent slices in opposite directions
raise alpha above 1 in the translation geometry **[inferred]**.

Over all cases, the largest mean alpha of any iteration was 0.89 times the
default `max_alpha` **[measured]**.  This value occurred in the cone-beam
proximal maps at sharpness -1.

## Test 3: what the clamp changes

Test 3 reran each of the 8 cases in which alpha exceeded the default
`max_alpha`.  It ran 5, 10, 20 and 40 iterations with three settings of
`max_alpha`: the default, 1.5, and no clamp.  In the cone-beam cases, the
default is 1.5, so these cases have two settings.  A reference cost comes
from 160 iterations with the default `max_alpha`.  The excess cost of a run
is its cost minus the reference cost.

The runs of different lengths share their first iterations.  The prefix
check ran `recon` at sharpness 1 for 5, 20 and 40 iterations, in the
translation geometry and in cone beam.  In each iteration that two of these
runs share, they reported the same mean alpha and the same rms data misfit,
to the last bit **[measured]**.  Also, each run without the clamp at 40
iterations reproduced the cost of the same case in Test 2 exactly.

In the cone-beam proximal maps, the clamp changed the cost by no more than
the rounding of float32 arithmetic **[measured]**.  After 5 iterations, the
excess cost was about 2 in both cases.  The reference costs of the two cases
were 525 and 10143.  The clamp changed the excess cost after 5 iterations by
0.08% or less.  After 10 or more iterations, every run was within 1 part in
a million of the reference cost.  Some of these runs ended below the
reference cost.  So differences of this size are within the rounding of
float32 arithmetic.

These cone-beam cases test the clamp only near convergence.  The clamp
first acted in iteration 5.  By the end of that iteration, the excess cost
was 0.4% of the cost in the noiseless case and 0.02% in the noisy case.

In the translation geometry, the clamp changed the excess cost by less than
1% **[measured]**.  Table 2 shows `recon` at sharpness 1.  Over all six
translation cases and the four iteration counts, the excess cost with the
clamp at 1.3 ranged from 0.68% below to 0.73% above the excess cost without
the clamp.  From 10 to 40 iterations, the excess cost fell by an average
factor of 1.08 to 1.36 per iteration.  At the slowest of these rates, a
difference of 0.73% equals 0.09 iterations **[derived]**.  So the runs with
and without the clamp reached the same cost within 0.1 iterations of each
other.

In the translation geometry, the runs without the clamp showed no sign of
instability.  Their excess cost fell from each sampled iteration count to
the next, as did the excess cost of the runs with the clamp.

Table 2.  Excess cost of `recon` in the translation geometry at sharpness 1.
The reference cost is 28253.

| `max_alpha` | 5 iterations | 10 iterations | 20 iterations | 40 iterations |
|---|---:|---:|---:|---:|
| 1.3, the default | 46982 | 8566 | 1544 | 303.2 |
| 1.5 | 46975 | 8542 | 1537 | 302.8 |
| none | 46975 | 8535 | 1534 | 302.6 |

## Test 4: the damping of the slice mean in cone beam

Test 4 reran the noisy cone-beam proximal map at sharpness -1 without the
clamp, once with the damping of the slice mean and once with this damping
turned off.  With the damping turned off, the largest alpha fell from 1.67
to 1.03, and no subset exceeded 1.5 **[measured]**.  The cost after 40
iterations changed by less than 1 part in 10 million.

The derivation above predicts this effect.  In this geometry, the damping
factors \( s_m \) were 0.252 to 0.265.  So alpha could approach \( 1/s_m \),
which is about 4.  At sharpness -1, the proximal prior is strong.  So the
cross terms of the data term are small compared with the curvature of the
prior, as the derivation assumes.

## Test 5: the small constant in the line search

Late in a run, the small constant in the denominator of alpha lowers alpha.
By iteration 40, the mean alpha of the proximal maps at sharpness -1 and 1
fell to between 0.028 and 0.055 in every geometry except translation
**[measured]**.  Test 5 reran one of these proximal maps, in parallel beam
at sharpness 1, with \( 10^{-30} \) in place of the small constant.

Table 3.  Mean alpha of the parallel-beam proximal map at sharpness 1.

| Small constant | Iteration 10 | Iteration 20 | Iteration 30 | Iteration 40 |
|---|---:|---:|---:|---:|
| \( 1.19 \times 10^{-7} \), the default | 0.994 | 0.443 | 0.092 | 0.055 |
| \( 10^{-30} \) | 0.994 | 0.997 | 0.941 | 0.898 |

The two runs reached the same cost, within 1 part in 10 million.  By
iteration 20, the image changed by less than 0.0003% per iteration in both
runs.  That change is more than 700 times below the default stopping
threshold of 0.2%.  So the small constant acts only after a run would have
met the default stopping rule.  In this test, it did not change the result.
The other proximal maps likely have low values of alpha for the same reason
**[inferred]**.

## Test 6: a proximal map whose input is offset by a constant

Test 6 shows how close the mean alpha of an iteration can come to
`max_alpha` without a scaling error.  It runs the noisy cone-beam proximal
map at sharpness -1 for 5 iterations, starting at the 128-subset partitions.
The start image is the 40-iteration `recon` result at sharpness 1.  The
input is the start image plus a constant of 0.01, 0.03 or 0.1, which is 0.7,
2.1 or 6.9 times \( \sigma_{prox} \).  A constant offset makes the gradient
in each slice of a subset mostly its mean.  By the derivation above, the
line search then undoes most of the damping of the slice mean.

In the first iteration, the mean alpha was 0.960, 0.981 and 0.983 times
`max_alpha` for the three constants **[measured]**.  Without the clamp, 46%,
70% and 74% of the subsets asked for more than 1.5.  The largest alpha was
3.36, 3.56 and 3.59.  In the later iterations, the mean alpha was at most
0.90 times `max_alpha`.

## Recommendation

**Keep the clamp and its defaults.**  Removing the clamp would gain nothing
measurable.  In these tests, the clamp reduced alpha in few subsets.  The
runs with and without the clamp reached the same cost within 0.1 iterations
of each other.

The plan states that the clamp was meant to limit steps from a local model
of a nonlinear problem.  With the positivity constraint off, the clamp is
not needed for that purpose, because alpha cannot raise the cost, as derived
above.  Three reasons still favor keeping the clamp:

- in cone beam, the clamp keeps part of the damping of the slice mean, which
  the line search would otherwise undo, as in Test 6;
- with positivity on, the guarantee derived above does not hold, and these
  tests did not cover that case;
- the translation model's comment reports instabilities with a larger
  `max_alpha`, and these tests covered only one translation setup.

**Make no scaling adjustment.**  The update of every tomography model gives
the same result in any unit, when the weights do not change with the scale
**[derived]**.  Test 1 confirmed this result for `recon` in parallel and
cone beam.  The denoiser's old error was a wrong \( h_{f,j} \) in its
direction, and PR #14 fixed it.  The clamp kept alpha from restoring the
length of the step.  No single alpha could have restored the weight of each
voxel in the direction, which the error also changed.

**Consider a warning when most subsets ask for an alpha far above the
clamp.**  The denoiser's old error asked for an alpha of at least 328 in
every subset.  No subset in these tests asked for more than 3.6.  So a
warning when more than half of the subsets of an iteration ask for an alpha
above 10 would separate the two cases.  Counting subsets keeps one unusual
subset from triggering the warning.  The count costs one comparison and one
addition per subset.  The code would read the count once per iteration,
together with the numbers that it already reads at that point.  One function
could serve `recon` and the denoiser.  The batched denoiser currently
discards alpha (`mbirtorch/denoising.py:1525`), so it would need to keep the
count.

A warning based on the mean alpha after the clamp would be less reliable.
In Test 6, that mean came within 2% of `max_alpha` without any scaling
error.  A user who sets `max_alpha` to 1 or less would also see such means
in normal runs, where alpha is near 1.

**Leave the denoiser's clamp in place.**  In the denoiser, alpha is below 1
by construction, so the upper bound of its clamp never acts.  Removing the
clamp there would change no result.  Step 5 may replace the denoiser's
solver, so this code may change anyway.

Only the warning would change package code, and it waits for Greg's
decision.

## Limits of this evidence

The tests are small and synthetic.  Each recon has at most 74,000 voxels,
and each sinogram has at most 64 views.  The phantoms are Shepp-Logan and a
pattern of dots, and the noise is white.  The weights are constant, and the
positivity constraint is off.  The tests ran on one CPU, so they do not
cover the code paths for several devices.

The tests leave out some settings.  They include no helical cone beam, no
curved detector, and no real scan.  One setup with 4 recon rows represents
the translation geometry.  The translation model's comment does not say
where the instabilities appeared.  So this setup may differ from the one
that showed them.

The recording of alpha ran without compiled kernels.  Compiled runs compute
the same alpha up to rounding **[inferred]**.

A MACE loop calls the proximal map many times from warm starts.  The
converged-start cases test only one such call.

The default partitions can put one pixel twice in a subset when the region
of reconstruction has fewer than about 16,000 pixels **[derived]**.  The
update then moves that pixel twice (`mbirtorch/tomography_model.py:69`).
For such a subset, the factor of 2 derived above can rise to 4, and the
guarantee that alpha cannot raise the cost can fail **[derived]**.  The
partitions of these runs held no pixel twice in one subset **[measured]**.
