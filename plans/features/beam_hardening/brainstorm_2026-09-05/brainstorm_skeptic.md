# The skeptic's report: estimating beam-hardening parameters from reconstruction quality

Date: 2026-09-05.  Read-only brainstorm.  Line numbers refer to the `geometric_calibration` branch at commit 590fea9.

## Summary

The geometry estimator works because a wrong geometry can only add artifacts, so the score's minimum is the truth.  A beam-hardening correction is different.  The correction family can remove real signal, and every proposed image score rewards removing signal.  Each of the four approaches inherits that.  Three facts from the code sharpen it.  The existing correction is exactly the identity on rays that cross no metal, so cupping is outside the family.  The corrected sinogram always re-inserts the metal from a smooth, noise-free forward projection, so every corrected candidate is partly denoised and wins noise-sensitive scores for the wrong reason.  And the baseline the brainstorm wants to beat has no measured quality in this repository: the MAR golden test projects a phantom with no beam hardening at all.

## The strongest case for the idea

The sinogram fit minimizes the wrong quantity: the residual of `y` against monomials of segmented path lengths (`mar.py:534-629`).  A small residual does not mean a clean image, because on starved metal rays the residual is noise that the fit treats as signal, and the user judges the image, where streaks are visible and measurable.  A polychromatic model, approach (b), has two to four physical parameters, positive spectrum weights, and a monotone concave curve, so it is far more constrained than the fifteen coefficients of the two-metal polynomial, and its parameters transfer across binning unchanged.  The geometry machinery already provides a binned whole-object problem, a coarse grid, a polish, and an undecided verdict for a flat curve (`geometry_calibration.py:200-380`, `1085-1135`).  For an additive model, approach (a) makes each candidate one linear combination of a few basis slices, so a dense sweep is nearly free.  And the existing hyperparameters have never been swept on real data, so even a null result from approach (c) would document an unmeasured sensitivity.

## Attack 1: the scores are gamed, and the uncorrected sinogram is not a fair candidate

Three gaming routes exist in the current family.

The first route is scale.  Within-class variance, total variation, and gradient energy all fall when the image is scaled down.  The family scales the plastic through `theta_0` (`mar.py:671-675`, `718`), and only the least-squares rescale on plastic-only rays (`mar.py:724-739`) holds the scale.  That rescale also makes the correction the identity off the metal: on a ray with no metal, `Sp` is `theta_0`, the clamp is inactive, the corrected value is `y * p_norm / theta_0`, and the rescale returns exactly `theta_0 / p_norm`, so the output is `y`.  Cupping from the plastic alone is therefore outside the family, and its share of any image score is a constant floor.  Any score must be a ratio, for example variance over the squared class contrast, and a ratio closes the scale route only.

The second route is ray removal.  The corrected plastic is `max(y - H_m theta_m, 0)` over a floored denominator (`mar.py:647`, `682`, `703`, `718`).  A large `theta_m` drives the residual below zero on rays through thick metal, the clamp sets those rays to zero, and the re-added metal sinogram (`mar.py:818-824`) fills them with a smooth model projection.  That is metal inpainting.  It removes every streak those rays carried, and it removes the plastic signal on those rays too, which becomes a dark band around the metal.  A streak is high frequency and a band is low frequency, and total variation and gradient energy weigh the streak far more, so they prefer the inpainted candidate.  The score selects removal, not correction.

The third route is the analog of the interpolation lesson, with the direction reversed.  Every corrected candidate has one more denoising step than the measured sinogram: the metal rays are partly replaced by `FP(mask * recon)`, which is smooth and noise free (`mar.py:277-281`), and the clamp deletes negative noise.  A noise-sensitive score therefore prefers more correction on denoising grounds.  The uncorrected sinogram is not in the family at all, because the metal is always re-added, so setting `theta_m` to zero double counts it.  Matched processing has no clean analog when the processing is the correction.

No proposed score is safe on its own.  The least gameable I can construct is the ratio of radial to tangential gradient energy in the plastic class outside a margin around the metal, since streaks are radial about the metal and texture is isotropic, plus a data term on rays with reliable counts.  The data term is what stops ray removal, and it is the sinogram fit returning.  Experiment 1 below measures all three routes.

## Attack 2: the segmentation makes the score self-fulfilling

Otsu's criterion minimizes within-class variance over thresholds (`segmentation.py:222-226`).  Minimizing the same variance over `theta` with a fixed mask, then re-segmenting, is a joint minimization of one objective over mask and correction, and its global optimum is a piecewise-constant image regardless of physics.  On an object with real texture inside a class, a fiber composite or a circuit board's substrate, the score treats the texture as an artifact.  The sinogram fit has no such term, because `y` contains the texture.

The alternation in `recon_plastic_metal` (`tomography_model.py:542-560`) already has wrong fixed points, and a search inherits them.  The first segmentation runs on the FDK reconstruction (`tomography_model.py:532`), whose plastic is cupped, so with `num_metal=2` Otsu can assign the plastic's bright rim to a metal class; the rim is then projected, subtracted, and re-added as metal, and the next reconstruction shows the same rim.  The metal's value is a second fixed point: the re-added metal is `FP(mask * recon)` at whatever value the reconstruction holds (`mar.py:277-281`, `818-824`), so it is inherited from the FDK, never corrected, and a fixed-mask score is indifferent to it.

Evidence.  Segment the FDK reconstruction of the NSI metal scan at `num_metal` of 1 and 2 and count the plastic voxels labeled metal, then run three passes of the alternation and record whether the metal class mean moves.  Minutes.

## Attack 3: the parameters are not identifiable, and the image score chooses worse than the ridge

The columns of `H` are monomials of two normalized sinograms (`mar.py:776-779`, `788-804`).  For a thin metal feature `m` is nearly binary, so `m`, `m^2`, and `m^3` are nearly one column; the bga scan's solder balls are this case.  The ridge (`mar.py:522-529`) picks the minimum-norm combination, and the correction is unchanged along the flat direction because only `H theta` matters on the observed rays.  An image score is equally flat there, so a search wanders along the ridge driven by noise and by the gaming routes of Attack 1.  Within one scan there is no extrapolation, since every corrected ray was also fit.  The reduced problem creates it: a `theta` fit on binned pixels or a subset of rows is applied to `(p, m)` pairs it never saw.  The polychromatic model has the classic form of the problem.  The log-sum-exp curve (`utilities.py:1346-1351`) is a spectrum with attenuation bins, recovering a spectrum from a transmission curve is exponentially ill conditioned, and `fit_beam_hardening_curve` has no regularization (`utilities.py:1311-1330`).

Evidence.  Compute the condition number of `HtH` (`mar.py:496-513`) on the NSI metal scan and on bga at the reduced size, minutes once the columns exist.  Above about 1e6 in float32, the search has flat directions the score cannot resolve.

## Attack 4: dimension, local minima, and gradients through clamps

Golden section is a one-dimensional method (`geometry_calibration.py:1116-1132`), and six to fifteen coefficients admit no coarse grid.  Coordinate descent stalls on the ridges of Attack 3.  Autograd, approach (d), meets zero gradients at every active clamp (`mar.py:682`) and floored pixel (`mar.py:718`), and the Otsu thresholds are not differentiable.  With a fixed mask the gradient exists, but descent from the sinogram fit moves first along the directions the ridge suppressed, where the data term is flat and the image score is not.  Those are the gaming directions.  The score improves at every step while the correction degrades, and nothing in the loop says so.

Evidence.  Fifty descent steps on the image score in experiment 1's synthetic case, plotting score and RMSE per step.  Minutes.

## Attack 5: cost, and the slab is invalid for cone beam

Basis images at production scale are out of reach: one `2000^3` float32 volume is 32 GB.  Approach (a) therefore lives on a few slices, where a basis slice is 16 MB, and the columns feeding them are the problem.  `H`'s columns are built full size (`mar.py:501-513`), and the existing fit holds the measured sinogram, `p`, each `m_k`, and two transient columns, five to seven sinograms of 28.8 GB each at 1800 by 2000 by 2000.  The baseline has not been shown to run at production scale on one GPU, and every variant that computes `H` shares that limit.

The slab does not transfer from geometry to hardening.  `build_reduced_problem` documents that a slab projection carries a term the slab cannot explain, because rays through it cross material outside it (`geometry_calibration.py:220-225`).  For geometry that term sat in the check; for hardening the residual is the target.  In cone beam the columns must therefore come from projecting the whole object, so the reduced problem is a binned whole-object problem, about 450 by 500 by 500 at stride 4 and bin 4, which fits easily.  The view stride adds a confound: 450 views for 500 channels undersamples the metal edges, their aliasing streaks radiate from the metal as hardening streaks do, and a correction that softens the edge through the clamp is rewarded.

Evidence.  Score an uncorrected synthetic sinogram with no hardening at stride 1 and stride 4.  If the score moves more than hardening moves it, the reduced problem measures the reduction.  Minutes.

## Attack 6: binning breaks the transfer

`reduce_sinogram` averages each `bin_factor` block (`geometry_calibration.py:445-447`).  The monomials of a bin mean are not the means of the monomials, and the difference is largest where `m` jumps from zero to its maximum inside one bin, which is the metal edge, the origin of every streak.  The normalization by the sinogram maximum (`mar.py:788-804`) also changes with binning.  A `theta` estimated at bin 4 and applied at bin 1 is therefore biased at exactly the pixels that matter, and the polychromatic model escapes only the normalization part.

Evidence.  Fit `theta` on the synthetic sinogram at bin 1 and at bin 4, apply both at bin 1, and compare the corrected sinograms within three pixels of the metal edge.  Minutes.

## Attack 7: photon starvation, and the fit and the score fail in opposite directions

On rays through thick metal, `y` is noise, clipped, or negative.  The `m^3` column has its energy on exactly those rays, so the least-squares metal coefficients are set by the worst data; the support floor (`mar.py:369`, `419-428`) exists because that once made the QP infeasible and poisoned `theta`.  The image score sees those rays as streaks and rewards zeroing them, Attack 1's ray removal.  MBIR down-weights those rays, so the FBP score minimizes streaks MBIR would suppress anyway.

## Attack 8: scatter adds a degeneracy, not a parameter

A constant transmission offset gives `y = -log(exp(-p) + s)`, concave in `p` like hardening.  Over the range of `p` an object presents, a quadratic fits both, so a scatter parameter is degenerate with the curvature terms in the fit and in any image score.  The two differ only near saturation, on the starved rays of Attack 7.  The discriminating data are the unreliable data.

## Attack 9: the FBP optimum is not the MBIR optimum

The final image is MBIR with weights and a prior; the FBP score has neither.  The prior and the weights remove part of a streak, while a dark band is data consistent and low frequency, so MBIR keeps it.  A `theta` that trades streaks for bands therefore wins the FBP score and loses in the MBIR image.  The gap is unmeasured, and experiment 3 below measures it.

## Attack 10: the hyperparameter search finds the defaults or the edges

Approach (c) sweeps `order`, `alpha`, `beta`, `gamma`, and `num_metal`, about four hundred combinations, each needing a fit and a direct reconstruction.  `beta` and `gamma` are scale free (`mar.py:526-527`, `703`), so they transfer.  Either the surface is flat within the repeatability floor and hours of sweeping confirm the defaults, or the minimum sits at the edge of least regularization, smallest `beta` and largest `order`, which is Attack 1 again.  `num_metal` cannot be compared by a fixed-mask score at all, because each value defines a different mask and so a different score.

## Attack 11: the baseline is unmeasured, and the test data cover only the extremes

The MAR golden test builds `mar_sino` as a monochromatic forward projection (`tests/generate_preprocess_goldens.py:166-169`), and the tests compare against mbirjax's output of the same chain (`tests/test_preprocess_mar.py:110-124`).  They establish parity, not correction.  The only hardening injected in this project's experiments is a quadratic on a 64-row band, which its record calls a proxy (`real_scan_validation.md:181-183`, `299-302`).  So "worse than the existing fit" has no measured reference, and the first experiment of any plan must produce one.  The real scans cover only the extremes: the NSI phantom has one insert of one material, so `num_metal > 1` cannot be tested, and bga's balls are the thin-metal case of Attack 3.

## Attack 12: approach (a) does not represent the existing correction

The existing correction is a ratio with a clamp and a floor (`mar.py:647`), and a direct reconstruction of a ratio is not a linear combination of the columns' reconstructions.  Approach (a) therefore needs an additive model, `y - sum theta_j h_j`, a different correction on metal rays that must be stated as a replacement.  Its gaming route is the simplest of all: with the coefficient of the `R(m)` basis image near the metal's scale, the metal vanishes from the image and every streak with it.

## The three experiments to run first

1. The gaming curve, synthetic, CPU, under an hour.  Build a 2D two-material phantom, harden it with a known polychromatic curve, and add Poisson noise with starvation through the metal.  Score each proposed image measure and the RMSE against the monochromatic truth along two one-parameter paths from the sinogram fit: global scale-down, and increasing metal subtraction up to full inpainting.  Kill: every proposed score reaches its minimum at a point where the RMSE is worse than at the sinogram fit.  Survive: at least one scale-free score has its minimum within the fit's own error of the RMSE minimum.

2. The resolution floor, real, one GPU, about an hour.  On the NSI metal scan at the binned whole-object reduction, compute the surviving score at the sinogram fit and at plus and minus twenty percent on each coefficient in turn, twelve evaluations for `K=1`, and the repeatability floor from the even and odd views as the geometry plan defines it.  Kill: no single-coefficient move exceeds the floor; there is then nothing to search.

3. The MBIR gap, synthetic, one GPU, a few hours.  Take the score-selected `theta` from experiment 1 and the sinogram-fit `theta`, run one MBIR pass with each corrected sinogram through the existing `recon_plastic_metal` path, and measure plastic-class RMSE.  Kill: the score-selected `theta` gives the worse MBIR image, which means the estimator misleads in exactly the case it was built for.

If all three survive, Attack 2's fixed point is the next question.
