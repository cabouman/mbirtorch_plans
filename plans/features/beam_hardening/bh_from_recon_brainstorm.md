# Beam-hardening parameters from reconstruction quality: a viability brainstorm

Date: 2026-09-05.  Status: brainstorm, revised once after a three-reviewer panel and once after
Greg's request for a reorganization, not yet ruled on by Greg.  This page synthesizes six agent
reports written in this session.  All six are in this directory: `brainstorm_score.md`,
`brainstorm_search.md`, `brainstorm_physics.md`, `brainstorm_skeptic.md`,
`brainstorm_literature.md`, and `brainstorm_pipeline.md`.  The three review files are
`review_accuracy.md`, `review_reasoning.md`, and `review_style.md`, and `README.md` lists every
file.  Every number below was read in this session, either from one of those reports or from the
output of a script.  The scripts are `counts_and_binning.py`, `bh_physics_sim.py`, and
`bh_physics_extra.py`, and they are in this directory with their outputs.  Each number names
its source.  Code citations are to the `geometric_calibration` branch of mbirtorch at commit
`590fea9`, with paths from the package directory.

Four terms are used throughout.  The geometry estimator is `estimate_geometry_from_recon`, the
function planned in `plans/features/geometric_calibration/estimate_by_recon_plan.md`.  It
searches geometry parameters by scoring the reconstruction each candidate produces.  The direct
reconstruction is the filtered back projection of `recon_direct`, which is FDK for cone beam.
The existing fit is the sinogram-domain least squares of `correct_sino_plastic_metal`
(`mbirtorch/preprocess/mar.py`), and the correction is the corrected sinogram it produces.  The
reduced problem is the smaller model and sinogram of `build_reduced_problem`
(`mbirtorch/preprocess/geometry_calibration.py`).  It keeps every fourth view, bins the
detector, and reconstructs a slab of a few slices from only the detector rows that feed it.
This page calls those rows the row window.

The page has five parts.  The executive summary and the answer state the conclusion.  The
section on the existing routines describes the code as it is.  The section on modifications
describes changes to those routines, each with the experiments that would support it.  The
section on other techniques describes methods that would replace the existing fit, each with its
experiments.  A short ordered list of the experiments and the open questions close the page.

## Executive summary

Greg asked whether beam-hardening parameters, and the sinogram correction itself, can be
estimated from reconstruction quality with the ideas of the geometry estimator.  The answer is
yes, in a narrower form than the geometry estimator.  Two facts bound the method.  A direct
reconstruction is linear in the sinogram, so a candidate coefficient vector can be scored without
a projector call once a few basis images exist.  A hardening correction can remove real signal,
and a flatness score rewards that removal, so the image cannot choose every coefficient on its
own.  The design that survives keeps the existing sinogram fit as a data term, lets the image
score move only the coefficient directions the reconstruction is sensitive to, and reports
undecided where the object gives no signal.

The work divides into modifications of the existing routines and alternatives to them.  

### Modifications to existing routines

Four modifications are proposed, in order of cost: 
 * run the existing fit on a reduced problem; 
 * choose its ridge strength by an image score, which replaces tuning by eye; 
 * add an image-score warning to the alternation of correction and reconstruction; 
 * replace the division-form correction by a subtraction form whose nonlinear coefficients are fit in the image domain with the sinogram fit as
a data term.  

### Alternatives to existing routines

The main alternative is a physical family of three to five parameters, which fits a
simulated attenuation 2 to 5 times more accurately than the existing cubic and extrapolates to
thicker metal about 50 times more accurately at 200 kV (`bh_physics_extra_output.txt`).  Every
proposal has an experiment with a result that would stop it, and the first experiment is a
synthetic case with polychromatic hardening, because the repository's golden data carry no
hardening at all.

Four decisions are Greg's: which deliverable is wanted, whether the plastic's own hardening
should enter the model, whether to keep the polynomial or move to the physical family, and
whether the NSI export already carries a vendor hardening correction, which a file name
containing "0.5BH" suggests.

## Answer

Estimating beam-hardening parameters from reconstruction quality is viable, in a narrower form
than the geometry estimator.  Two facts determine what the method can and cannot do.

The first fact is favorable.  A direct reconstruction is linear in the sinogram.  Write the
correction as the sinogram minus a fitted nonlinear excess.  Then the reconstruction of any
candidate coefficient vector is a linear combination of a fixed set of images, computed once.  A
search over the coefficients needs no projector call per candidate.  If the score is quadratic
in the image, the estimate is one small convex problem.  The polynomial model of `mar.py` has 6
to 15 columns, of which 4 to 12 are nonlinear and would be searched.

The second fact is unfavorable, and it is where hardening differs from geometry.  A wrong
geometry can only add artifacts, so the best-scoring geometry is the correct one.  A hardening
correction can instead remove real signal.  Any score that rewards a flat image also rewards
that removal.  The published method closest to this construction found that three image-chosen
coefficients were robust while seven were worse than no correction under clinical noise
(`brainstorm_literature.md`, Levi et al. 2021).  That comparison changes the material count
with the coefficient count, so it is evidence rather than a controlled ablation.

The design that follows from both facts is a hybrid.  The existing fit stays.  It holds every
coefficient direction to which the reconstruction is insensitive.  The image score moves only
the directions to which the reconstruction is sensitive, which are the artifact patterns in the
plastic near the metal.  The weight between the two terms is the open design question.
Without a rule for that weight the hybrid is not a decision but a dial, and the modifications
section proposes two rules and an experiment that tests them.

Two cheaper deliverables come first, because neither has been measured:

- a synthetic case with polychromatic hardening, which the repository does not have today;
- a sweep of the existing fit's ridge strength scored on the image, which replaces the tuning a
  user does by eye today.

`brainstorm_physics.md` recommends changing the model itself, and its numbers are these.  The
existing model is a cubic polynomial in the plastic and metal path lengths.  A physical family
with three to five parameters fits a simulated attenuation about 2 to 5 times more accurately
than the cubic.  It extrapolates to thicker metal about 50 times more accurately at 200 kV
(`bh_physics_extra_output.txt`).  A mixture of exponentials with free bins and 8 to 11
parameters fits 15 to 190 times more accurately (`bh_physics_sim_output.txt`).  Which family to
search, and which of three deliverables is wanted, are the first decisions for Greg.

## The existing routines

mbirtorch has three beam-hardening tools today, and the plastic-metal correction is the one this
page builds on.  `BH_correction` (`mar.py:146-203`) applies a polynomial with caller-supplied
coefficients to the sinogram, view batch by view batch through `pipeline.map_view_batches`.
`fit_beam_hardening_curve` and its inverse (`mbirtorch/preprocess/utilities.py`, lines about
1258 to 1600) fit a single-material curve of log-sum-exp form to paired samples, which is a
polychromatic model with equally spaced attenuation bins, and invert it with a Chebyshev fit.
`find_linearization_fit` in `mbirtorch/preprocess/pymbir.py` is a two-bin member of the same
family in closed form.  None of these handles a second material.

The plastic-metal correction is `correct_sino_plastic_metal`, and it runs in four steps.
`segment_plastic_metal` (`mbirtorch/preprocess/segmentation.py:304-401`) segments a
reconstruction into one plastic class and `num_metal` metal classes by multi-threshold Otsu.
The correction forward projects the plastic mask to get a scaled path-length sinogram p, and it
forward projects each metal class's masked reconstruction values to get a sinogram m_k
(`mar.py:271-281`).  It then fits the measured sinogram y to monomial columns in p and m by
ridge least squares with positivity constraints: the columns are p, the cross terms p·m^a with
total metal degree at most `order` − 1, and the metal-only terms m^a with total degree at most
`order` (`mar.py:764-779`), the ridge weights grow with the degree (`mar.py:515-529`), and an
active-set loop adds the most violated pixel constraints and re-solves with OSQP
(`mar.py:534-629`).  The corrected plastic sinogram is then built in three steps: the metal-only
terms are subtracted from y, the result is divided by the coefficient of p with a clamp at zero
and a floor on the denominator (`mar.py:632-722`), and it is rescaled by least squares on
metal-free rays (`mar.py:724-739`).  The metal sinograms are added back at the reconstruction's
own level (`mar.py:818-824`).  This page calls that the division form.

`recon_plastic_metal` (`mbirtorch/tomography_model.py:420-580`) drives the correction.  It makes
one direct reconstruction, then alternates the correction with a full MBIR reconstruction for
`num_BH_iterations` passes, feeding each pass's reconstruction to the next segmentation.  The
hyperparameters `order`, `alpha`, `beta`, `gamma`, and `num_metal` are set by hand.  The MBIR
passes can take the MAR weights of `gen_weights_mar`, which discount rays through metal.

Three facts about this code shape everything below.

The model has no plastic-only nonlinear column.  p enters only linearly (`mar.py:764-779`), and
on a ray with no metal the correction returns y unchanged (`brainstorm_skeptic.md`, attack 1).
The plastic's own cupping is therefore outside the model today.  At the NSI scans' setting of
200 kV behind 0.9 mm of copper, a PMMA-like plastic is linear to one percent over 10 cm.  The
missing column therefore costs nothing at that setting.  At 100 to 150 kV with light
filtration, the plastic's attenuation per centimeter falls 13 to 15 percent over 8 cm.  No
coefficient in the present model can correct that fall (`brainstorm_physics.md`, section 3).

The fit's memory is the production-scale problem.  This is true of the existing fit alone,
before any image score is added.  The columns are full-size sinograms.  At 1800 by 2000 by 2000
each column is 28.8 GB.  The fit holds the measured sinogram, p, and each m_k at once, which is
86.4 GB for one metal.  Its normal equations rebuild 27 columns two at a time
(`brainstorm_pipeline.md`, section 2).

There is no measured baseline.  The golden MAR sinogram is a linear forward projection of a
two-level phantom with no hardening (`tests/generate_preprocess_goldens.py:166-169`).  Those
tests therefore establish parity with mbirjax and say nothing about correction quality
(`brainstorm_skeptic.md`, attack 11).  Any claim that a modified or replaced fit is better or
worse than the existing one needs a synthetic case with a polychromatic truth first.  That case
is experiment 1 below, and every other experiment depends on it.

## Modifications to the existing routines

Four modifications are proposed.  The first three leave the fit and the correction as they are
and change how they are run.  The fourth replaces the correction's form and the way its
nonlinear coefficients are chosen.  Each ends with its supporting experiments.

### Modification 1: run the existing fit on a reduced problem

The proposal is to fit the coefficients on a reduced sinogram and apply them to the full one.
A 48-row window at view stride 4 and detector bin 2 brings a column from 28.8 GB to 86 MB
(`brainstorm_pipeline.md`, section 2).  Four parts of the geometry estimator's machinery
transfer as they are.  The view stride of `reduce_sinogram` changes no pixel value, so it is
exact.  The one-dimensional search of `_search_minimum` serves any scalar hyperparameter, and it
brings a coarse grid, a golden-section refinement, and its notes about edge minima and multiple
minima.  The undecided verdict transfers, with a noise floor from the even-view and odd-view
split.  And the result shape of `CalibrationResult` extends to a vector-valued answer
(`brainstorm_pipeline.md`, section 5).

Three parts do not transfer, and each needs a change.

The slab does not transfer.  In cone beam the rays through a slab's detector rows cross voxels
outside the slab.  The geometry of the rotation experiments has a 5.7 degree half fan.  There
an 8-slice slab's rays reach 0.88 slices past each end of the slab.  At a 15 degree half fan
they reach 2.8 slices (`brainstorm_pipeline.md`, section 1).  Thickening the slab widens the
row window by the factor (sid + r) / (sid − r).  That factor always exceeds one.  No finite slab
is therefore self-contained, and the columns p and m_k for the window rows must be projections
of the full class volumes.  Those projections can still be cheap.  The rays through the window
reach only the bounded distance computed above.  It is therefore enough to project a mask
sub-volume whose height is the slab's height times that factor.  Project it through a model
copy whose detector is cropped to the window, and the window rows are exact.  The factor must
be taken from the window `_slab_row_window` returns, which is widened by a voxel footprint and
rounding (`geometry_calibration.py:133-135`), not from the ideal geometry.  The direct
reconstruction of the window rows into the slab needs no such care, because the ramp filter
runs along channels and a slab voxel receives only the rays through it.

Detector binning does not transfer for the columns.  Binning is linear, so it is exact for the
measured sinogram.  The nonlinear columns are different.  The mean of m squared over a bin
exceeds the square of the mean by the within-bin variance, and at a metal edge that variance is
large.  Two sizes of that bias were computed.  The first size is per pixel at a silhouette edge,
where m runs from zero to its full value across one or two pixels.  There a quadratic column
built from binned m is biased by about 25 percent at bin 2 and 125 percent at bin 4, and a
cubic column by 75 percent at bin 2 (`brainstorm_pipeline.md`, section 1).  The second size is
the bias of the fitted coefficient of a quadratic model on a disk (`counts_and_binning.py`,
rerun in this session).  For a disk of radius 800 pixels it stays under 0.01 percent at bin 4.
For a disk of radius 50 pixels it is 0.23 percent at bin 2 and 1.09 percent at bin 4.  Metal
features are small, so the larger biases apply to them.  Because the fit is linear in the
coefficients, the remedy is to build each column at full resolution inside the view-batch loop
and bin afterward.  That leaves the reduced problem a correctly binned linear least-squares
problem in the coefficients (`brainstorm_pipeline.md`, section 1, and `brainstorm_score.md`).
It is not the same problem as the unbinned fit, because the within-bin residual is discarded.
`brainstorm_pipeline.md` recommends leaving the channel bin factor at 1 instead.  This page
overrides that recommendation for the cost figures, and the bin ablation below is the gate for
the override.

The per-slice agreement rule transfers only for physical parameters.  Hardening parameters are
set by the spectrum and the materials, so they are the same in every slice, while structure is
not (`brainstorm_score.md`).  The polynomial's coefficients are not physical parameters.  They
are a least-squares approximation to the true curve over the set of path-length pairs the
object's rays present.  This page calls that set the presented set.  The presented set differs
between a slice through the middle of the metal and a slice near its end, so the coefficients
may legitimately differ (`review_reasoning.md`, finding 7).  The agreement rule must therefore
compare a slice-invariant quantity: the corrected attenuation at probe path-length pairs that
every scored slice presents.

Applying the coefficients to the full sinogram is the same step the existing correction already
performs, and it can stream.  The correction is a per-pixel function of y, p, and m_k, so it can
run view batch by view batch as `BH_correction` does, with one host output and no second
full-size sinogram.  The batch kernel needs p and m_k for its own views.  No public entry point
projects a subset of views today.  The per-view-batch body already exists
(`mbirtorch/cone_beam.py:161`), and until a public entry exists a model copy carrying the
batch's angles is the way to reach it (`brainstorm_pipeline.md`, section 3;
`brainstorm_search.md`).  The class volumes are the remaining memory question.  One float mask
per class costs 32 GB each at 2K.  A single label volume would cost 8 GB, but only if the
projector could accept a label volume and a class value.  That substitution also changes the
metal columns, which today project the reconstruction's own values inside the mask rather than a
constant (`review_accuracy.md`, problem 2).

The reduced problem also changes what the alternation costs.  Today it is one direct
reconstruction plus `num_BH_iterations` times a fit and a full MBIR.  With the reduced problem it
can be three steps: one direct reconstruction for the masks, an alternation of correction and
direct reconstruction at reduced scale until the coefficients and the masks converge, and one
full MBIR.

Supporting experiments.  On the synthetic case of experiment 1, fit the coefficients at bin 1
and at bin 4 with the columns built before binning, apply both at bin 1, and compare the
corrected sinograms within three pixels of the metal edge and the plastic-region error of the
reconstructions.  Stop the override of the bin-1 recommendation if the bin-4 fit is worse than
the bin-1 fit by more than the existing fit's own error.  Score unhardened synthetic data at view
stride 1 and 4.  The full scan has 1800 views for 2000 channels, the same ratio as 450 for 500,
so aliasing from metal edges is in the acquisition, and this check tests only whether the
reduction adds to it.  On the NSI metal scan, compare the coefficients fit on the reduced
problem with the coefficients fit on the full sinogram, which the existing code produces today,
and record the memory and time of each.  Whether masks from a reduced reconstruction serve the
full correction is measured in experiment 3.

### Modification 2: choose the ridge strength and the floor by an image score

The proposal is a one-dimensional search over the existing fit's ridge strength `beta`, and
optionally its floor `gamma`, scored on a reduced reconstruction, which automates the tuning a
user does by eye today.  `brainstorm_physics.md` ranks this above any coefficient search, and
`brainstorm_search.md` and `brainstorm_pipeline.md` call it the cheapest useful step.  Each
candidate costs one solve of the fit's small normal equations, one active-set pass, one
elementwise correction, and one direct reconstruction of the reduced slab.  A golden-section
search at `_search_minimum`'s defaults spends about 24 evaluations (`brainstorm_search.md`).

The ridge strength is a bias rather than a safeguard, which is why the sweep is worth doing
whatever else is built.  Consider a rod centered in a plastic disk.  Every ray through the rod
crosses nearly the same plastic length.  The cross columns and the metal-only columns are
therefore collinear on the metal rays.  Nothing in the data decides how the fitted excess
divides between them, so the ridge decides it.  The consequence was measured on a probe ray at
half the maximum metal path whose plastic estimate is 10 percent low, which a streak through
the plastic mask produces.  Exact inversion of the true model has no error on that ray.  With the
existing fit's ridge strength β, the corrected plastic errs by +4.4 percent at β of 2e-4, by
+6.7 percent at the default 2e-3, and by +15.4 percent at 2e-2.  The opposite probe, 10 percent
high, errs by −1.2, +0.7, and +8.4 percent (`bh_physics_extra_output.txt`).  The image is
sensitive to that division only through rays whose path lengths differ from the ones the fit
saw, which occur at edges and along streaks.  Those rays are few, so the image constrains only
weakly what the data leave undetermined.

What the sweep cannot fix is also known.  It cannot add columns, so plastic cupping stays out of
reach.  It cannot change the ridge's direction, only its strength.  It cannot repair a
segmentation error, which changes p and m themselves.  And `num_metal` cannot be compared by a
fixed-mask score, because each value defines a different mask and so a different score
(`brainstorm_search.md`, approach c; `brainstorm_skeptic.md`, attack 10).  `beta` is scaled by
the trace of the normal matrix (`mar.py:526-527`), so a value found on binned data transfers
only approximately and the fine level should confirm it.

Supporting experiments.  On the synthetic case, sweep β from 2e-5 to 2e-2 on three slices of the
reduced problem and record the argmin against the β that minimizes the plastic-region error
against the monochromatic truth.  On the NSI metal scan, sweep β over the same five decades,
record the argmin and the depth of the score curve, and compare both against the value a user
chooses by eye.  Stop if the curve's depth is within the even-view and odd-view floor, which
means the surface is flat and the sweep only confirms the default, or if the minimum sits at the
edge of least regularization, which is the signal-removal trap of modification 4.

### Modification 3: an image-score warning inside the alternation

The proposal is one score call after each pass of `recon_plastic_metal`, which warns when the
score gets worse (`brainstorm_pipeline.md`, section 4).  It is the smallest change of the four.
It catches two failures the code already anticipates and never checks against the image: a
non-solved OSQP status that keeps the previous coefficients (`mar.py:620-627`), and a negative
mean of the plastic coefficient (`mar.py:707-708`).  It also catches wrong fixed points of the
alternation.  The first segmentation runs on the direct reconstruction, whose plastic is cupped,
so with `num_metal` of 2 Otsu can assign the plastic's bright rim to a metal class.  The rim is
then projected, subtracted, and re-added as metal, and the next reconstruction shows the same
rim.  The metal's value is a second fixed point, because the added-back metal is the
reconstruction's own projected level, inherited from the direct reconstruction and never
corrected (`brainstorm_skeptic.md`, attack 2).

Supporting experiment, which takes minutes.  Segment the direct reconstruction of the NSI metal
scan at `num_metal` of 1 and 2, count the plastic voxels labeled metal, run three passes of the
alternation, and record whether the metal class mean moves.

### Modification 4: a subtraction-form correction with its nonlinear coefficients fit in the image domain

This is the modification that uses the geometry estimator's idea in full.  It changes the
correction's form, adds an image term to the fit, and adds two diagnostics and an undecided
verdict.  It is described in six parts: the linear structure, the objective, the score, the
safety rules, multiple metals, and the traps with their remedies.

**The linear structure.**  Write the correction as the measured sinogram minus the fitted
nonlinear excess, c(θ) = y − Σ_j θ_j h_j over the nonlinear columns h_j.  Every nonlinear column
is p·m^a or m^a with a of at least one, so c(θ) = y on every ray that misses metal.  The linear
columns are not searched.  The direct reconstruction R then obeys R(c(θ)) = R(y) − Σ_j θ_j
R(h_j).  Each R(h_j) is a basis image, computed once per segmentation on the reduced problem,
and R(y) is one more reconstruction on the same reduced model.  Every candidate θ afterward is a
linear combination of small images.  Three search methods are therefore free of projector
calls: a grid search, a golden-section refinement, and a closed-form solve.  At two metals and
order 3 the stack holds one reconstruction of y plus 15 basis images.  Sixteen images of an
8-slice slab at 500 by 500 occupy 128 MB, at 8 MB per image (`brainstorm_pipeline.md`, section
2).  This is the construction used by empirical beam-hardening correction (Kyriakou et al. 2010)
and by the later automated methods that build on it (Levi et al. 2019, 2021), applied here to
mbirtorch's reduced problem (`brainstorm_literature.md`).

The metal's level needs a stated convention, because the subtraction form differs from the
division form there.  With no add-back, the metal's level on a corrected ray is the one the
fitted linear coefficient gives, and θ = 0 is the measured sinogram itself.  That is the
convention this page adopts for the search, because it makes the uncorrected sinogram a member
of the family.  The division form instead sets the metal to its own reconstructed level
(`mar.py:818-824`).  The two conventions differ by the term Σ_k (ρ_k − θ_k,lin) m_k, with ρ_k the
reconstruction's metal level (`brainstorm_search.md`).  That term is fixed once the linear
coefficients are fixed, so it can be applied after the search as one fixed image.

The subtraction form is a stated replacement for the division form, not an approximation of it.
The two forms agree to first order in the excess and differ at long metal paths.  The
subtraction form drops the clamp at zero and the floor.  The clamp matters for the following
reason.  A large metal coefficient drives thick-metal rays negative, and the clamp sets those
rays to zero.  The added-back metal then fills them with a smooth projection.  That step is
inpainting.  It deletes streaks and plastic signal together and leaves a dark band, and a
gradient or total-variation score prefers that result (`brainstorm_skeptic.md`, attack 1).

**The objective.**  With a score S(x) = xᵀQx that is quadratic in the image for a fixed mask and
blur, S(θ) is quadratic in θ.  The joint objective is

    S(θ) + λ ‖Hθ − y‖² + β ‖θ‖²_W,

where H and y are restricted to the reduced sinogram, λ weights the data term, and β and W are
the existing ridge strength and degree weights (`mar.py:515-529`).  The objective is a convex
quadratic program.  The active-set loop of `_estimate_BH_model_params` (`mar.py:534-629`)
applies with one change.  Its two constraint families exist for the division form, one to keep
the denominator positive and one on the metal-only block.  The subtraction form needs one
family, y − Σ_j θ_j h_j ≥ 0 over all nonlinear columns, so the row assembly changes while the
solver does not.  The loop adds at most `num_constraint_update_iter` most-violated pixels per
family, so it is an approximate guard rather than a guarantee.  The convexity is a property of
this quadratic score.  The published local minima (`brainstorm_literature.md`, Levi et al.)
arose with a thresholded total-variation score and a simplex optimizer, so they are avoided
here by the choice of score, not dissolved.

The data term is what addresses the objection in `brainstorm_skeptic.md` and the failure in
`brainstorm_literature.md`.  Two kinds of direction in θ are invisible to the score: those
whose basis images are nearly collinear on the scored region, and those in the null space of
Q.  The data term and the ridge fix those directions, so noise cannot move them.  The condition
number of the masked Gram matrix of the basis images is the identifiability diagnostic.  A
large value means the object gives no sensitivity for some coefficient, and that coefficient's
verdict is undecided.  A second diagnostic says how much the image is being asked to decide.
The effective degrees of freedom of the image term, the trace of BᵀQB (BᵀQB + λHᵀH + βW)⁻¹ with
B the basis images, is computable once the Gram matrices exist.  That number, not the column
count, is what to compare with the three that worked and the seven that failed in Levi et al.

The weight λ needs a rule that works without a monochromatic truth.  Two candidates exist, and
neither is in the source reports.  The reproducibility rule takes the smallest λ at which the
estimates from the even views and from the odd views agree within the half width of the score
minimum.  The discrepancy rule takes the smallest λ at which the sinogram residual on the
reduced data stays within its noise level.  Experiment 1 tests both against the λ an oracle
would choose on the synthetic case.  Three limiting values of λ are the three positions the
reports take.  At λ → ∞ the estimate is the existing fit, which is the position of
`brainstorm_skeptic.md`.  At λ → 0 it is the pure image fit, which the skeptic, the literature,
and `brainstorm_physics.md` all reject.  The rule chooses between them.

The overall image scale is fixed, for a reason stronger than the gauge argument of the reports.
The correction is the identity on metal-free rays, so the rescale of `_estimate_plastic_scaling`
(`mar.py:724-739`) is exactly one, and it is held at one across candidates.  The plastic's level
cannot move.  The metal's level can, in one case.  When the metal is thin its column m is
nearly binary, m² and m³ are nearly equal to m, and R(m²) is nearly parallel to R(m).  A large
coefficient on a nonlinear column then erases the metal and every streak with it, which is
attack 12 of `brainstorm_skeptic.md`.  Excluding the linear columns does not close that route.
The scale is therefore fixed to the extent that the nonlinear basis images are not collinear
with R(m_k) on the scored region.  The per-scan measure of that collinearity is the angle between
R(m_k) and the span of the nonlinear basis images on the mask, reported beside the condition
number.  The thin-metal case of experiment 1 observes the failure directly.

**The score.**  The image score has two parts, and the signed part is primary.  The signed part
is a set of region contrasts in the plastic near the metal, and it uses three regions.  The
corridor is the plastic region between two metals.  The annulus is the plastic ring around each
metal beyond the edge margin.  The far plastic is the plastic beyond a few metal radii from every
metal.  The contrasts are the mean of the corridor minus the mean of the far plastic, and the
mean of each annulus minus the mean of the far plastic.  Each contrast is linear in the image,
crosses zero at the right correction, and has one root, so a one-parameter search on it is a
bracketed root-finding problem with no local minima (`brainstorm_score.md`).  These contrasts
are primary because they lie where the searched columns act.  Every searched column carries a
factor of m, so the correction changes only rays through metal, and its basis images are streak
and band patterns near the metal.

The second part is the low-frequency deviation of the plastic class from its own mean, inside a
mask eroded away from every edge and every metal.  It measures cupping.  Under the subtraction
form with the existing columns, cupping in the far plastic cannot change.  This part is
therefore a diagnostic of the model rather than a target of the search
(`review_reasoning.md`, finding 2).  It detects the plastic's own hardening and scatter, both
outside the model.  It becomes a target only if a plastic-only column or the physical family is
adopted.  Its expected size on the NSI scans is small, because the plastic there is linear to
one percent (`brainstorm_physics.md`).  The blur width and the erosion margins are specified in
ALU, the package's arbitrary length units, so they do not change with voxel size.  Both are
swept rather than guessed.

The metric of the geometry estimator does not serve here, for a reason worth stating.  For a
round uniform object, the hardened sinogram is still the projection of some image, namely a
cupped one.  The reprojection residual is therefore insensitive to cupping.  Gradient energy is
nearly insensitive to it too.  For a disk, the cupping term is about 0.007 σ/R of the edge term,
where σ is the blur width and R the radius.  It is under one part in a thousand for blurs below
about a seventh of the radius (`brainstorm_score.md`).  Only a flatness measure inside a mask is
sensitive to cupping.  Streaks and bands between separated dense parts are inconsistent with any
image, and the residual is sensitive to those (`brainstorm_physics.md`, section 4).

The slices to score are chosen by the opposite criterion from the geometry estimator's.  The
metal terms are measured on the slices that contain the metal, and the two-metal cross columns
on slices that contain both metals.  The cupping diagnostic is measured on slices near the
central plane with the largest plastic cross-section.  There the chords are longest and FDK's
own shading is smallest (`brainstorm_score.md`).  The slice-selection step must therefore select
slices with dense features, where the geometry estimator's slice selection avoided them.  The
scored set is therefore several row windows, not one slab.

**The safety rules.**  Five rules come from the reports:

- The masks and the blur are computed once from one segmentation and reused for every
  candidate, and the outer plastic rescale is held at one.
- Every candidate, including θ = 0, goes through the same code path and the same reduction.
- The noise floor is measured by scoring the difference between the even-view and odd-view
  reconstructions, and it must bound two opposite biases.  A stronger correction amplifies
  long-path noise, so a variance score prefers weak corrections (`brainstorm_score.md`).  A
  corrected candidate under the division form's convention is also partly denoised on metal
  rays, because the added-back metal is a smooth projection, so a noise-sensitive score prefers
  strong corrections (`brainstorm_skeptic.md`, attack 1).
- A systematic floor is measured beside the noise floor: the score change from moving the
  segmentation threshold by one histogram bin, which is the smallest realistic systematic
  perturbation (`review_reasoning.md`, finding 10).
- The estimate must agree across slices, across blur widths, and across erosion margins.
  Agreement means the corrected attenuation at common probe path-length pairs falls within
  half the width of the score minimum.  Otherwise the verdict is undecided.

A score built on within-class variance can be self-fulfilling.  Within-class variance is Otsu's
own objective (`mbirtorch/preprocess/segmentation.py:222-226`).  Minimizing it over θ with a
mask, then re-segmenting, is one minimization over mask and correction whose optimum is a
piecewise-constant image regardless of physics (`brainstorm_skeptic.md`, attack 2).  Two
things limit the damage.  The fixed mask with erosion breaks the alternation within one
estimate.  The few coefficients can remove only what lies in the span of the basis images, so
texture moves the estimate only in proportion to its correlation with those images.  One case
is not limited at all.  A radial density gradient in a round object matches the cupping basis
image exactly, and no single-spectrum measurement can distinguish the two
(`brainstorm_score.md`).  The NSI phantom scanned without its insert is the reference for that
case.  The plastic is the same object in both scans, so its real structure is measured rather
than assumed, once a registration and an intensity convention between the two scans are
defined.

**Multiple metals.**  Two metals are the hardest case, and a two-metal synthetic case is
required to test it.  The counts of H's columns, from `_generate_metal_exponent_list`'s rule,
are these (`counts_and_binning.py`):

| metals | order 2 | order 3 | order 4 |
| --- | --- | --- | --- |
| 1 | 4 | 6 | 8 |
| 2 | 8 | 15 | 24 |
| 3 | 13 | 29 | 54 |

The searched count excludes the linear columns p and each m_k, so it is 4 at one metal and 12
at two metals at order 3.  The reports quote 5 and 13 or 14 for the same quantity
(`brainstorm_search.md`, `brainstorm_score.md`).  The two-metal count is above the seven that
failed in Levi et al., which is why the data term and the effective-degrees-of-freedom
diagnostic matter.  The polynomial's cross columns of two metals, m_0 m_1 and their higher
forms, are supported only on rays through both metals.  They are decided by few rays and few
slices, so the scoring slab must contain both metals or those columns are dropped
(`brainstorm_search.md`).  The region structure separates the materials, so that each region is
sensitive to one group of terms.  The annulus around metal k is sensitive to its metal-only
terms.  The corridors between metal k and the plastic are sensitive to its cross terms.  The
corridor between metals j and k is sensitive to their joint term (`brainstorm_score.md`).  The
real scans limit what can be tested.  The NSI phantom has one insert of one material.  The bga
scan's solder balls are the thin-metal case, where the monomial columns become nearly collinear
(`brainstorm_skeptic.md`, attacks 3 and 11).  The two-metal synthetic case should include the
centered placement, which is the one that makes the columns collinear, and a thin-metal
placement.

**The traps.**  Eight traps were identified across the six reports.  Each has a mechanism, a
remedy, and an experiment that settles whether the remedy works:

| trap | mechanism | remedy | what settles it |
| --- | --- | --- | --- |
| signal removal | flatness rewards deleting streaks and structure; the clamp inpaints thick-metal rays; a nonlinear column can mimic the metal's own image | subtraction form, linear columns fixed, metal masked, data term with weight λ, the collinearity angle | the one-parameter score curves of experiment 1, including the metal-mimicking path |
| noise-driven preference | a stronger correction amplifies long-path noise; the added-back metal denoises metal rays | blur in ALU, the even/odd floor bounding both directions | the floors of experiment 2 |
| self-fulfilling segmentation | variance is Otsu's objective; the alternation has wrong fixed points | fixed eroded masks, few coefficients, agreement on probe values, the no-metal scan as texture reference | the texture case of experiment 1; the fixed-point count of modification 3 |
| collinear columns | thin metal makes m, m², m³ one column, in the data as well as in the image | the ridge, which is a bias, the Gram condition number, undecided | condition numbers on the NSI metal and bga scans in step 0 |
| photon starvation | starved rays carry streaks no polynomial removes, and the score rewards hiding them | the same MAR weights the final MBIR uses, applied in the score and in every basis image, which keeps linearity | the photon-starvation case of experiment 1 |
| scatter | a constant transmission offset mimics hardening curvature | absorb it for artifact-free plastic; fit or measure it when extrapolation is the goal | the scatter case of the synthetic truth |
| direct reconstruction versus MBIR | the delivered image is MBIR with weights and a prior | direct reconstruction for the search, MBIR along a line of candidates as the check | experiment 3 |
| aliasing from metal edges | 1800 views for 2000 channels undersample metal edges, and aliasing streaks radiate as hardening streaks do | score unhardened data at view stride 1 and 4, which tests only whether the reduction adds to the acquisition's own aliasing | the stride check of modification 1 |

**Supporting experiments.**  Four experiments support this modification, and they are the gates
`brainstorm_skeptic.md` set, revised by the reasoning review.

Experiment 1 is the synthetic truth and the score curves, on CPU in under an hour per case.
Build a thin three-dimensional case of a PMMA disk with iron and aluminum rods.  Harden it with
a tungsten-anode spectrum at 200 kV behind 0.9 mm of copper, which is the NSI scan file's
setting, and build a second case at 100 kV with light filtration.  Use one rod and two rods, in
a centered and an off-center placement, plus a thin-rod placement.  Add Poisson noise at the
photon count in the NSI scan file, and include a photon-starvation case and a scatter case.
Compare the plastic-region rms error against the monochromatic reconstruction for four
estimators: no correction, the existing fit, the subtraction-form closed-form image fit under
each λ rule, and the physical family fitted on the sinogram.  Then record every candidate score
along three one-parameter paths from the existing fit.  Two paths are controls for the division
form: a global scale-down, and an increasing metal subtraction up to full inpainting.  The third
path is the direction in the nonlinear span that best mimics the metal's own image, which is
the leading generalized eigenvector of the masked Gram matrix of the basis images against R(m).
The primary score is committed in advance as the signed contrasts, and the others are
secondary.  The one-metal centered case is the selection set, and the two-metal, off-center,
and thin-rod cases are the confirmation set, which the chosen score must pass unchanged.  Stop
the work if every score reaches its minimum at a point where the rms error is worse than at the
existing fit.  Continue if the primary score's minimum lies within the existing fit's own error
of the rms minimum on both sets.  Run four ablations here, varying one thing each, beside the
two of modification 1:

- the score masks eroded and dilated by one pixel;
- the segmentation threshold moved by one histogram bin, which changes p, m, and every basis
  image;
- a radial density gradient of 2, 5, and 10 percent, and a textured plastic;
- blur widths of 1 to 10 percent of the diameter.

Experiment 2 is identifiability on the NSI metal scan, on one GPU in about an hour.  Use the
reduced problem at the whole axial extent.  Compute the eigenvectors of the masked Gram matrix
of the basis images, ordered by eigenvalue, and along each report the score change per unit
change of the sinogram residual.  That is the identifiability curve.  Compute the even-view and
odd-view noise floor and the segmentation-threshold systematic floor on the same data.  Stop the
work if no direction exceeds both floors.  The same job runs the β sweep of modification 2.

Experiment 3 is the MBIR proxy test, on one GPU in hours.  Run MBIR at four to six coefficient
vectors spaced along the segment between the existing fit's vector and the image-chosen vector,
with one step beyond each end.  Do this on the synthetic case and on the NSI metal scan.  Report
the rank correlation between the direct-reconstruction score and the MBIR plastic-region error,
against the monochromatic truth on the synthetic case and against the registered no-metal
reconstruction on the NSI scan.  Stop the work if the ranks disagree.  The same job runs the
final MBIR with and without the metal-only terms under the MAR weights, which answers whether
those coefficients reach the delivered image at all, and it runs the full correction once with
masks from a reduced reconstruction, which answers the last question of modification 1.

Experiment 4 is the estimator itself.  Build a function in `mar.py` in the style of the planned
`estimate_geometry_from_recon`.  It returns the candidates, their scores, the chosen coefficient
vector, the hyperparameters, the class thresholds, the reduction record, the two diagnostics,
notes, and the undecided verdict.  The function must pass three gates:

- recovery, within a stated tolerance, of the oracle polynomial coefficients on the synthetic
  case, where the oracle is the least-squares fit of the polynomial to the polychromatic truth on
  the presented set;
- an unchanged ranking at detector binning 2;
- on the NSI metal scan, agreement across slices of the corrected attenuation at common probe
  values, and a plastic-region comparison against the registered no-metal reconstruction.

The plastic-only ray residual, binned by path length, is reported in the result as a diagnostic
of the model rather than a gate.  It is constant in the searched coefficients, and it detects
the plastic's own hardening and scatter.

## Other techniques that could replace the existing fit

Five techniques from the reports would replace the polynomial fit rather than modify it.  The
first is the one `brainstorm_physics.md` recommends, and the page's staging puts it after
modification 4 for one reason: the closed-form image fit of modification 4 reuses the existing
fit, constraints, and correction, and it is the simplest test of whether an image score can
decide coefficients at all.  Whatever family is searched, the physical family is the synthetic
truth for every gate from the start, so that the polynomial is never both truth and model.

### Technique 1: a physical family with exact inversion

The physical family is a mixture of exponentials over fixed energy bins whose spectrum comes
from one or two parameters and whose attenuation curves come from tables.  Its unknowns fall in
three groups: one or two spectrum parameters, which are the copper-equivalent filter thickness
and optionally the voltage; one density scale per material; and optionally one scatter
constant.  That is 3 to 5 parameters for one metal, and one more per added metal
(`brainstorm_physics.md`, section 1).  The correction is exact inversion: solve y = f(p, m̂) for
p on each ray by bisection or Newton's method, since f is increasing in p, and output the
monochromatic value at a reference energy.  The inversion is elementwise, so it streams by view
batch like the existing correction.  For starved rays no inversion recovers p, and the right
output there is the model's own prediction from the mask with a weight near zero, which is
sinogram replacement made explicit (`brainstorm_physics.md`, section 6).

The physics report simulated the NSI setting with a Kramers spectrum from a tungsten source,
Elam attenuation tables through the `xraydb` package, and an energy-integrating CsI detector
(`brainstorm_physics.md`; `bh_physics_sim_output.txt`).  The simulation used a grid of PMMA up
to 8 cm and iron up to 1 cm at 200 kV behind 0.9 mm of copper, on which the attenuation reaches
4.35.  The cubic in `mar.py` fits that grid with an rms error of 0.009 and a maximum error of
0.026 in attenuation units.  The cubic's metal curve has an inflection inside the fitted range,
at 0.85 cm of iron at 200 kV.  Beyond that point the cubic is convex.  The true attenuation as a
function of path length is concave everywhere.  Extrapolation was tested by fitting on iron up
to 0.5 cm and evaluating at 1 cm with no plastic.  There the cubic errs by +0.69 at 200 kV, and
by +6.3 and +8.3 at 150 and 100 kV.

The physical family's accuracy was measured against a truth generated with a different detector
and inherent filter.  A one-parameter fit reached a maximum error of 0.0112 and a two-parameter
fit 0.0045, where the cubic on the same truth reached 0.0219.  On the extrapolation test the
one-parameter fit erred by −0.0144 and the two-parameter fit by +0.0140 at 1 cm of iron, against
the cubic's +0.69 (`bh_physics_extra_output.txt`).  The gain in fit is therefore 2 to 5 times,
and the gain in extrapolation about 50 times at 200 kV.  A family with fixed bins and free
weights only is worse than the cubic at 3 parameters (`bh_physics_sim_output.txt`), so the
spectrum parameters are what make the small family work.

The family costs a projector call per candidate when scored on the image, because it is
nonlinear in its parameters.  Each candidate needs one direct reconstruction of the reduced
slab.  A Gauss-Newton step needs one more per parameter for its Jacobian, which is the
basis-image construction applied once per iteration (`brainstorm_search.md`, approach b).  With
3 to 5 parameters a coarse grid and a one-dimensional polish per parameter through
`_search_minimum` are affordable at reduced scale, and the per-slice agreement rule applies to
the parameters themselves.  `brainstorm_physics.md` proposes fitting the family on the
sinogram by nonlinear least squares and using the image score as the check, which needs no
projector call per candidate at all.

The family has three known weaknesses.  Recovering a spectrum from transmission data is
ill-conditioned, which `brainstorm_literature.md` reports for spectrum estimation methods
(question 3).  Fixing the bin energies and fitting only one or two spectrum parameters avoids
that problem.  Tungsten and tin have a K-edge inside the spectrum, which breaks the two-basis
attenuation form used when the materials are unknown; steel, aluminum, and titanium have no
K-edge there, so the form holds for them (`brainstorm_search.md`).  And scatter and
extrapolation are in tension.  A constant transmission offset lowers y most where y is largest,
as hardening does.  A four-bin mixture absorbs a 0.5 to 2 percent offset to within 0.002 to
0.007 rms over 8 cm of plastic (`brainstorm_physics.md`, section 4).  When the goal is
artifact-free plastic on the presented set, absorbing scatter is acceptable, and the fitted
coefficients must not be read as physics.  When the argument for the family is extrapolation,
the fitted parameters must be physics, so scatter must be fitted as its own zero-attenuation bin
or measured on the scanner.  A scatter measurement, for example behind a lead strip, is
therefore a precondition for the family's extrapolation advantage rather than a refinement.

Supporting experiments.  On the NSI metal scan's reduced sinogram, fit the one-parameter family
and the polynomial to y against the mask projections, and compare the residuals on plastic-only
rays and on metal rays.  Then fit both on the thinner half of the metal rays and predict the
thicker half; the polynomial's extrapolation error in simulation was 25 percent of the
attenuation at 200 kV, and the family's was about 50 times smaller.  Stop if the family's
residual on the metal rays is not below the polynomial's.  On the synthetic case, the family is
one of the four estimators of experiment 1, and when it becomes the model the truth must come
from an independent generator, such as the different-detector truth of `bh_physics_extra.py` or
the `spekpy` and `xrayphysics` packages that `brainstorm_physics.md` names.

### Technique 2: a mixture of exponentials with free bins, for unknown materials

When the materials are not named, the family f(p, m) = −log Σ_j w_j exp(−a_j p − b_j m) with
free shared bins has 3J − 1 parameters, 8 at three bins and 11 at four.  It fits the simulated
grid to a maximum error of 0.0018 at three bins and 0.00014 at four, and its extrapolation error
at 1 cm of iron is −0.006 at 200 kV and +0.20 and +0.31 at 150 and 100 kV
(`bh_physics_sim_output.txt`).  It is concave by construction.  Its parameter count is the
polynomial's, so the estimability warning of Levi et al. applies, and `brainstorm_physics.md`
recommends tying a_j and b_j to the photoelectric and Compton basis curves to stop overfitting.
The known failure is collapse to one bin, because the weights are seen only through the
curvature of the curve along each material's axis (`brainstorm_search.md`, approach b).

Supporting experiment.  The same fit-and-predict test as technique 1, with the free family
initialized from the polynomial fit, watching for a collapse to one bin.

### Technique 3: a projection-consistency criterion

Abdurahman and colleagues (2018) estimate the same kind of polynomial coefficients by minimizing
the inconsistency of projection pairs under Grangeat's relation, with no image regions, no
spectrum, and no calibration (`brainstorm_literature.md`).  It is the main alternative to an
image-domain criterion: the same parameter vector with a projection-domain objective.  Its blind
spot is the one the reprojection residual has.  Cupping of a round object is consistent data, so
the criterion is sensitive to streaks and bands and not to cupping.

Supporting experiment.  On the synthetic case, compare the coefficients it chooses with the
image fit's and with the oracle's, on the selection and confirmation sets.

### Technique 4: segmentation-free basis images

Schüller and colleagues (2015) replace the hard segmentation with a nonlinear histogram
deformation of the reconstruction, forward project the original and the deformed volumes,
combine the projections monomially, and combine the resulting basis images by the same image
flatness criterion (`brainstorm_literature.md`).  This keeps the structure of modification 4 and
removes its dependence on `segment_plastic_metal`, which matters where metals and plastic
segment badly, as on the bga scan.

Supporting experiment.  Swap the segmentation for the deformation on the bga scan and on the
thin-rod synthetic case, and compare the score curves and the collinearity angle.

### Technique 5: hardening inside the iterative reconstruction

Polyenergetic statistical reconstruction (Elbakri and Fessler 2002 and 2003; O'Sullivan and
Benac 2007) makes the attenuation at a reference energy the unknown and puts the spectrum in the
forward model, so no sinogram correction exists (`brainstorm_literature.md`).  It is the analog
of the `det_rotation` inside the projectors that the geometry plan defers: the parameter becomes
part of the reconstruction.  It needs a spectrum and attenuation tables, its cost is the
reconstruction's, and it is outside the scope of this brainstorm.  It is recorded here because
technique 1 supplies exactly the spectrum and tables it would need.

## What to run, in order

The steps below are ordered by cost, and each names the section that defines it.

- Step 0, minutes.  Read the NSI scan's metadata and file names to learn whether the export
  already carries a vendor hardening correction; `brainstorm_physics.md` reports a file name
  containing "0.5BH", and every real-scan experiment runs on that scan.  Run the fixed-point
  count of modification 3.  Compute the condition number of HᵀH on the NSI metal scan and on the
  bga scan at reduced size.
- Experiment 1, CPU, under an hour per case: the synthetic truth, the four estimators, the score
  curves, and the ablations of modifications 1 and 4.
- Experiment 2, one GPU, about an hour: identifiability on the NSI metal scan and the β sweep of
  modification 2.
- The fit-and-predict test of technique 1 on the NSI metal scan, which shares experiment 2's
  reduced sinogram.
- Experiment 3, one GPU, hours: the MBIR proxy test, the metal-only-terms check, and the
  reduced-mask check.
- Experiment 4: the estimator and its gates.

## Open questions for Greg

- Which deliverable is wanted: a warning check inside `recon_plastic_metal`, an automation of
  the fit's hyperparameters, or an estimator of the coefficients?  The reports rank the
  hyperparameter automation cheapest and the warning check smallest, with the coefficient
  estimator the most work.
- Should the model gain the plastic's own hardening, as a p² column or through the physical
  family?  Today that hardening is outside the model.  The omission is correct at 200 kV with
  copper filtration.  It is wrong at lower voltages.
- Polynomial or physical family?  The physical family needs the materials named or fitted, the
  voltage from the scan file, and a scatter measurement if its extrapolation is the reason for
  choosing it.
- Does the NSI export already apply a vendor hardening correction?  If the exported sinogram is
  partly corrected, the metal scan's residual hardening is smaller than assumed and the real-scan
  gates must say so.
- Do the MAR weights in the final MBIR make the metal-only coefficients irrelevant to the
  delivered image?  If so, the search shrinks to the cross terms (`brainstorm_score.md`).
- Is a view-subset forward projection an acceptable addition to the projector API?  Without it
  the correction cannot stream at production scale, whichever estimator is built.
- Is the NSI phantom's plastic uniform enough that the no-metal scan can serve as the texture
  reference?  Should the bga scan be the textured-object gate?
- Can scatter be measured on the scanner, for example behind a lead strip?
