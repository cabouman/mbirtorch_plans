# The image-domain score for beam-hardening correction

Date: 2026-09-05.  Status: brainstorm, read-only.  Code citations are to the `geometric_calibration`
branch of mbirtorch at commit `590fea9`.  The numbers below are derived from the formulas stated
beside them, for a uniform disk in exact arithmetic.  None is a measurement.

## The answer

Beam hardening leaves two kinds of artifact, and they need different measures.  The plastic term
leaves cupping, a smooth bias that is consistent with the data, so only a flatness prior can see
it.  Its measure is the low-frequency deviation of the plastic class from its own mean, inside a
mask eroded away from every edge and every metal.  The metal terms leave dark bands and halos,
which are inconsistent with the data, and their measures are signed means of regions tied to the
metal positions against the far plastic.  Both are quadratic in the image once the masks and the
blur are fixed.  If the correction is linear in its coefficients, the direct reconstruction is a
fixed image plus a linear combination of precomputed basis images, and the estimate is one small
linear system.  Five rules make the score safe: the scale is fixed by construction, the masks and
blur are fixed across candidates, every candidate goes through the same code path including the
uncorrected one, the noise floor is measured by an even/odd view split, and the estimate must
agree across slices and blur widths or the verdict is undecided.  The main risk is that real
density structure is read as cupping.  The defense is that hardening parameters are the same in
every slice and region while structure is not.

## What hardening leaves in the image, and which measure sees it

For a uniform disk of attenuation `mu` and radius `R` under the quadratic model
`y = p - b p^2`, the artifact term of the direct reconstruction is `-(8/pi) b mu^2 sqrt(R^2 - r^2)`,
because the projection of `sqrt(R^2 - r^2)` is `(pi/2)(R^2 - t^2)`.  Three things follow.  Cupping
and edge brightening are one term: the dome is deepest at the center and has infinite slope at
the rim.  With `b` sized as the validation harness sizes it, to change the largest sinogram value
by ten percent (`real_scan_validation.py:82`, `:570-572`), the center dips by `(8/pi)(0.05)`,
about 13 percent of `mu`, and the plastic mean shifts by two thirds of that.  The dome's
root-mean-square deviation from the class mean is `sqrt(1/18)` of the center dip, about 3 percent
of `mu`.

The gradient-energy score of the geometry estimator cannot see this.  A blurred edge of height
`mu` and width `sigma` has gradient energy about `2 pi mu^2 R / sigma`, and the dome's interior has
about `0.85 pi (0.13 mu)^2`.  The ratio is about `0.007 sigma / R`, below one part in a thousand at
any blur narrower than the object.  The mask, not the blur, makes cupping visible.

| artifact | origin | sign flips at the right correction | measure |
| --- | --- | --- | --- |
| cupping and edge brightening | plastic curve | yes | low-frequency deviation inside the eroded plastic mask; signed: center minus annulus |
| dark band between metals | cross terms `p m^a` | yes | signed corridor mean minus far-plastic mean |
| halo around a metal | metal-only terms `m^a` | yes | signed annulus mean minus far-plastic mean |
| streaks from metal edges | metal terms, photon starvation, partial volume | partly | directional high-pass energy; diagnostic only |
| shift of the plastic mean | plastic scale | no | invisible to a scale-invariant score; set by `_estimate_plastic_scaling` (`mar.py:724-739`) |

## The candidate scores

"Fixed mask" means a mask computed once from a reference reconstruction and reused for every
candidate.

**Within-class variance after segmentation.**  Reads cupping, bands, halos, streaks, noise, and
real texture together.  Misses a uniform shift of the class and the artifact's sign.  Gamed by a
candidate that shrinks the image, by one that lowers the noise (a weaker correction amplifies
long-path noise less), and by a mask recomputed per candidate, which can drop the cupped center
from the plastic class.  Quadratic with a fixed mask, `x^T (M - M 1 1^T M / N) x`.  Cost `O(N)`.

**Residual after a piecewise-constant fit.**  The same score when the pieces are the classes.  With
connected components as pieces, each plastic part gets its own mean, which absorbs real
part-to-part differences.  Same gaming; quadratic with fixed pieces.

**Low-order polynomial fit inside a class.**  Two scores hide here.  The fitted quadratic
coefficient is a signed cupping measure, linear in the image with a fixed mask, and it crosses
zero at the right correction.  The residual after the fit is blind to cupping, because the fit
removed it, and reads bands, streaks, texture, and noise.  The first is the readout for a
one-parameter plastic search; the second is a band score once cupping is taken out.  Gamed by a
real density gradient whose shape matches the dome.

**Total variation.**  Reads streaks and edges, and cupping only weakly.  Dominated by pixel-scale
noise unless blurred, the lesson of `plans/geometric_calibration/experiments/closed/real_scan_rotation_recon.md`.  Gamed by shrinking and by
smoothing.  Not quadratic.

**Histogram entropy and Otsu separability.**  Cupping widens the plastic peak, so the entropy of the
masked histogram has a minimum.  With absolute bins a shrunk image scores lower, so bins must be
relative to the class mean.  Otsu's between-to-within ratio is a ratio of two quadratics with fixed
masks and not quadratic when the thresholds are recomputed.  Neither is differentiable, and both
read noise.

**Streak-specific measures.**  Energy along lines joining metal centroids, or the gradient
component across the radial direction about each metal.  Quadratic with a fixed direction field.
The trap: photon-starvation streaks have the same shape as hardening streaks and no hardening
correction removes them, so the minimum lands where the correction best hides noise.  Diagnostic
only.

**Low-frequency-only deviation.**  Blur at a fraction of the plastic's extent, then take the
within-class deviation inside the eroded mask.  Reads cupping and bands, ignores noise and fine
texture, still reads texture above the blur width.  Quadratic with fixed mask and blur.  Cost
`O(N)` for a separable blur.  This is the primary score.

**Signed region contrasts.**  Center minus annulus in far plastic; corridor minus far plastic;
per-metal annulus minus far plastic.  Each is linear in the image, so its zero crossing is a root,
not a minimum.  A bracketed root is found by bisection with no local minima, with precision set by
the noise of a region mean divided by the slope.  These are the readouts for a non-linear one- or
two-parameter search.

## What makes a score safe to search on

A homogeneous quadratic score is minimized by the zero image, so the scale must be removed.  Either
fix the linear coefficient of the correction at one, so the search cannot scale the image, or
divide by the square of the far-plastic mean, which makes a ratio of quadratics and turns the
linear system into a generalized eigenproblem of size `K + 1`.

The analog of the matched-processing rule has three parts.  The masks and the blur are computed
once and reused.  Every candidate, including the uncorrected one, goes through the same correction
code and the same reduction, because the current path has a clamp at `mar.py:682` and a floor at
`mar.py:703-718` that a raw reconstruction never passes.  And the noise is matched, which
processing alone cannot do: a stronger correction amplifies long-path noise, so a variance score
prefers weak corrections for a reason unrelated to hardening.  The even/odd split of
`plans/geometric_calibration/experiments/recon_sweep_fine.py` gives the floor.  Score the difference of the even-view and odd-view images
at each candidate, and require the score's range across candidates to exceed it.

The uncorrected sinogram is a candidate, and in the linear form it is the point `a = 0` on one
quadratic surface.  It can win for a bad reason in three ways: noise amplification, as above; the
clamp `max(y - H_m theta_m, 0)` making zeros on metal rays that reconstruct as dark streaks, so a
stronger metal subtraction scores worse; and masks taken from the uncorrected reconstruction.  The
blur and the floor handle the first, an additive correction with no clamp the second, erosion the
third.

## Robustness to real structure

Every flatness score reads texture, holes, and parts.  Three separations exist.  Frequency: the
blur removes texture below its width, and a sweep of the width shows where the estimate stops
moving.  Position: hardening artifacts follow the path-length map and the metal positions, so far
plastic reads the plastic curve alone and the corridors and annuli read the metal terms.  Basis: in
the linear form the minimization projects the image onto the span of the basis images, and texture
moves the estimate only in proportion to its correlation with those images.  A radial density gradient in a round object matches the cupping
basis image exactly, and no single-spectrum measurement separates them.  For any other shape the
cupping basis image follows the chord-length map, which a density gradient does not.

The strongest defense is outside the score.  Hardening parameters are set by the spectrum and the
materials and are the same in every slice and region, while texture is not.  Per-slice and
per-region agreement is therefore a stronger identity here than in the geometry estimator, where
the offset error made per-slice minima disagree for a physical reason.

## Where to score

For the plastic curve, score slices near the central plane with the largest plastic cross-section:
long chords carry the most hardening, and FDK's cone-beam shading is smallest there.  This is the
opposite of the geometry estimator's far-slice choice.  For the metal terms, score the slices that
contain the metal.  Three to five slices, combined by summing their Gram matrices or by the
trimmed mean.

Masks come from one segmentation by `segment_plastic_metal` (`segmentation.py:304-401`).  Erode the
plastic mask by the ramp filter's ringing width plus two blur widths, and by a distance from any
metal voxel, both in ALU and both swept.  Far plastic is the plastic beyond a few metal radii from
every metal; the corridor is the eroded plastic inside the convex hull of two metals; the annulus
is the plastic between the eroded metal edge and a few radii out.  The cupping blur default is
about five percent of the plastic's in-plane extent, swept over one to ten percent; the band
measures use about half the corridor width.

A direct reconstruction is a good proxy for the plastic term and a partial one for the metal
terms.  Cupping is consistent with the data, so MBIR keeps it; a prior does not fight a smooth
bias.  Streaks are not, and the MBIR pass in `recon_plastic_metal` (`tomography_model.py:557`) may
run with MAR weights that discount metal rays (`vcd_utils.py:309-389`), so the metal-only
coefficients matter less in the final image.  FDK's shading and the missing short-scan weighting
(`cone_beam.py:806-807`) are the same for every candidate and bias only through their correlation
with the basis images.  The ramp filter has no apodization (`tomography_utils.py:20-26`), which is
why the blur is needed.

## Multiple materials

No per-material variance term is needed.  The metal interiors are set by the linear add-back at
`mar.py:818-822`, not by the correction, and they are saturated and noisy.  The region structure
isolates one material: metal `k`'s annulus reads its metal-only terms, its corridors read its cross
terms with plastic, and the corridor between metals `j` and `k` reads their joint term.  In the
linear form the coupling between coefficients is the overlap of the masked basis images, and the
condition number of the `K x K` Gram matrix is the identifiability diagnostic.  A large condition
number means the object carries no signal for some coefficient, whose verdict is then undecided.
For one metal at order 3 there are five shape coefficients (`mar.py:764-779`); for two there are
fourteen.

## Consistency checks and the undecided verdict

The reprojection residual is blind to cupping.  For a round object the hardened sinogram is still
in the range of the projector, so `_direct_residual_score` (`geometry_calibration.py:544-584`)
cannot see the artifact the image score reads best.  It does see the inconsistency that bands and
streaks come from, and on real data its minimum was 1.5 percent deep (`plans/geometric_calibration/experiments/closed/real_scan_followup.md`).
It pairs with the corridor and annulus measures as a check, not with cupping.

The check that pairs with cupping is the plastic-only ray trend: bin the corrected residual
`y_c - theta_p p` over plastic-only rays by `p`, and require no trend in the bin means beyond the
noise.  The least-squares fit at `mar.py:534-629` runs over all rays with regularization, so this
trend is not forced to zero by construction, and in an image-score path it is independent.  One
caveat carries over: `p` comes from a segmentation of a reconstruction that carries the artifacts.

The undecided rules: per-slice estimates disagree beyond the half width; the estimate moves with
the blur width or the erosion margin beyond the half width; the score's range is below the
even/odd floor; the Gram matrix is ill-conditioned; the plastic-only trend is not reduced; or the
fitted curve is non-monotone on the data range.  The expected depth is large for the plastic term.
At the harness's hardening the artifact energy is about `(0.03 mu)^2` per voxel against a noise
energy a wide blur makes small, and the signed measures cross zero with a slope set by a 13 percent
center dip.  The limit is systematic, not noise, and the agreement rules are what test it.

## Which scores are quadratic

Quadratic with a fixed mask and blur: within-class variance, the piecewise-constant and
polynomial-fit residuals, the low-frequency deviation, the squared signed contrasts, directional
high-pass energy, and the Otsu numerator and denominator separately.  Linear: the signed contrasts
and the fitted polynomial coefficient.  Not quadratic: total variation, entropy, anything with
per-candidate segmentation, and anything through the ratio-form correction at `mar.py:632-722`,
whose division by the plastic coefficient, clamp, and floor are non-linear in `theta`.

The small linear system needs two linearities.  The correction must be linear in the searched
coefficients, which `BH_correction` (`mar.py:146-203`) and the Chebyshev inverse
(`apply_inverse_beam_hardening_curve` in `utilities.py`) already are.  The reconstruction must be
linear in the sinogram, which the direct reconstruction is and MBIR is not.  With `K` basis
sinograms the cost is `K` reduced-problem direct reconstructions once, then `O(K N_slab)` per
candidate and a `K x K` solve.  A basis sinogram at stride 4, bin 2, and a thirty-row window on the
NSI scan is about 40 MB.  The powers must be taken at full resolution inside the view-batch loop
of `reduce_sinogram` (`geometry_calibration.py:383-451`) before binning.  The metal basis
sinograms need `p` and `m_k` projected from the whole object, so the scout reconstruction must be
whole-extent at coarse binning; only the scoring slab is thin.

## Recommendation

1. Primary: the low-frequency within-class deviation energy on the eroded far-plastic mask, in the
   linear-coefficient form, with the corridor and annulus energies added for the metal terms.
   Assumption: the plastic is homogeneous at the blur scale within the scored regions, so the
   basis-image projection reads hardening and not structure.  Verification: the cylinder phantom of
   `rotation_zero_point_synthetic.py:136-155` without its slab, projected, hardened with
   `y = p - b p^2` at the harness's ten-percent sizing, with noise.  Score `y + a y^2` over a grid
   of `a` that includes zero and `b`.  Record the argmin against `b`, the half width, the even/odd
   floor, and the estimate at blur widths of 1, 2, 5, and 10 percent of the diameter.  Then one
   ablation: add a radial density gradient of 2, 5, and 10 percent and record the bias against the
   gradient size.
2. Check one: per-slice and per-blur agreement under the undecided rules.  Assumption: the
   parameters are global to the scan.  Verification: the NSI no-metal scan on four central slices,
   requiring the four estimates to agree within the half width; then the metal scan with the plastic
   coefficients held at the no-metal values, the single-variable test the scan pair allows.
3. Check two: the plastic-only ray trend.  Assumption: on plastic-only rays the corrected residual
   has no dependence on `p` at the right correction.  Verification: the Shepp-Logan phantom
   (`generate_3d_shepp_logan_low_dynamic_range`, `utilities.py:263`), which has parts of several
   densities; confirm the trend vanishes at the injected `b` and does not vanish at an argmin the
   image score returns when structure biases it.

## Open questions for Greg

- Is the target the coefficients themselves, or the hyperparameters `order`, `alpha`, `beta`, and
  `gamma` with the least-squares fit inside?  The quadratic path needs the first; the signed
  readouts serve the second.
- Should the correction be reparametrized additively, `y + sum_j c_j phi_j(p, m)`, so the linear
  form covers the metal terms, or does the ratio form's physics justify a non-linear search?
- Scatter produces cupping too, and a flatness score absorbs it into the plastic curve.  Is that
  acceptable, or should the plastic-only trend separate them by their different dependence on `p`?
- How far from the central plane can cupping slices sit before FDK shading biases the estimate?
  A sweep on the synthetic cylinder answers it.
- Do the MAR weights in the final MBIR pass make the metal-only coefficients irrelevant to the
  delivered image, so the search can stop at the plastic and cross terms?
