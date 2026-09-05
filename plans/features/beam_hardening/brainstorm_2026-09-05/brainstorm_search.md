# Beam-hardening parameters from reconstruction quality: parametrization and search

Date: 2026-09-05.  Status: brainstorm, read-only.  Code citations are to the `geometric_calibration`
branch working tree of mbirtorch at commit `590fea9`.  The three numbers computed for this report
come from `counts_and_binning.py` in this directory.

## Answer

A search over 6 to 15 beam-hardening coefficients is tractable because the direct reconstruction is
linear in the sinogram.  Write the correction as the measured sinogram minus the fitted nonlinear
excess.  The corrected sinogram is then affine in the coefficient vector θ, and the reduced direct
reconstruction of every candidate θ is the same affine combination of a small set of basis images.
The basis costs one forward projection per material class and one filtered back projection per
column, once per segmentation.  Every candidate θ after that costs no projector call.  With a score
that is quadratic in the image, the best θ solves a linear system of size 6 to 15, and the existing
sinogram fit, ridge penalty, and positivity constraints carry over as the regularizer and the
constraint set.  The scale is fixed by construction, because the affine form never subtracts or
rescales a linear column.  The cheapest first step is smaller than that: a one-dimensional search
over the existing fit's `beta` with the image score.  It automates what a user tunes by eye today and
costs one reduced direct reconstruction per candidate.  The polychromatic model is the right
long-term parametrization, with K+J parameters for K metals and J energy bins and no cross terms.
It is nonlinear, so it belongs after the linear version proves that the image score can decide θ at
all.

## Facts the search must respect

The existing model has one linear plastic column, cross columns `p·m^a`, and metal-only columns
`m^a` (`mbirtorch/preprocess/mar.py:764-779`).  The column count at order 3 is 6 for one metal, 15
for two, and 29 for three; at order 4 it is 8, 24, and 54.  Two facts about the model matter for
the search.  The plastic enters only linearly, so the model cannot represent cupping of the plastic
itself.  And the columns are monomials of variables on [0, 1] (`mar.py:788-804`), so the columns of
one metal are nearly collinear, which is what the ridge term resolves (`mar.py:515-529`).

Three steps of the existing correction fix its scale.  The plastic sinogram is divided by the fitted
coefficient of `p` and multiplied back by the plastic's own scale (`mar.py:666-722`).  A global
factor is then refit on metal-free rays (`mar.py:724-739`).  The metal is added back as the recon's
own projection, not as the fitted linear metal term (`mar.py:818-825`).  The net effect is a gauge.
On metal-free rays the corrected sinogram equals the measured one, and inside the metal it equals
what the current recon projects to.  Any replacement must state its own gauge.

At the production size of 1800 views by 2000 rows by 2000 channels, a sinogram is 28.8 GB and a
volume is 32 GB.  The reduced problem of `build_reduced_problem`
(`mbirtorch/preprocess/geometry_calibration.py:200-380`) is small.  At view stride 4 and bin 4, with
32 binned rows feeding an 8-slice slab, it is a 29 MB sinogram and an 8 MB slab image.  At bin 2 with
64 rows it is 115 MB and 32 MB.  A whole-extent reduced sinogram at bin 4 is 0.45 GB.  Everything
below is priced at these two scales.

## The approaches

### (a) Linear basis images

The correction must be affine in θ, and the divisive form is not.  Replace it by subtraction of the
fitted nonlinear excess: corrected = y − Σ_j θ_j h_j over the nonlinear columns only.  The fitted
linear metal is swapped for the recon's metal by adding Σ_k (ρ_k − θ_k,lin) m_k, as today.  Then
recon(θ) = R(y) − Σ_j θ_j R(h_j) + constant, where R is the direct reconstruction.  With a score
S(x) = xᵀQx, S(θ) is a quadratic in θ, and the minimizer solves (BᵀQB)θ = BᵀQb with B the basis
images.  The sinogram fit's normal matrix and vector (`mar.py:492-532`) add as λ(HᵀH, Hᵀy).  The
active-set positivity loop (`mar.py:534-629`) applies unchanged, because it needs only a quadratic
objective and linear constraints.

Parameter count: the nonlinear columns, 5 for one metal and 13 for two at order 3.  The linear
coefficients are not searched.  This is the gauge: the affine form cannot rescale the plastic or the
metal, so the overall scale of the corrected sinogram is the measured sinogram's own.

The basis is built once per segmentation.  It costs K+1 whole-extent forward projections at bin 4,
each a 500-cubed volume into a 0.45 GB sinogram that is then cropped to the window rows.  It then
costs 1+N_col slab direct reconstructions, each a filter over 29 MB and a back projection into an
8 MB slab.  Fifteen slab images occupy 120 MB.  Every candidate θ afterward is a linear combination of
8 MB images, so a grid, a golden-section polish, or the QP are all free.  At the production scale
nothing is reconstructed.  θ is decided on the reduced problem and applied by the view-batched map
described below.

What the image score cannot see: any direction of θ whose basis image lies in the null space of Q.
For a gradient score that is every smooth image, so cupping is invisible to it, as the geometry
design already notes; a within-class variance score sees cupping and streaks both and is quadratic.
What the image score sees wrongly: the linear metal direction.  Subtracting the metal's own basis
image lowers every energy score, so an unconstrained image fit erases the metal.  The linear
coefficients must therefore come from the sinogram fit or from the recon's class means, never from
the image score, and the score should exclude the metal and an edge margin.  What the sinogram fit
cannot see: the collinear directions within one metal's columns, and rays that a streak comes from,
because a streak is a few rays with large residuals that the whole-sinogram sum averages away.  The
image score is that same residual weighted by the ramp filter and the back projection, and that
reweighting is the entire difference between the two fits.

The affine form loses two things.  The division rescales the fit residual and the noise by 1/S_p,
which is why `gamma` exists, and the subtraction leaves them alone, which is better.  The clamp at
zero is lost, and negative corrected values are instead prevented by the active-set constraints.
The two forms agree to first order in the excess, because y/(θ_0(1+ε)) ≈ y(1−ε)/θ_0 and y ≈ θ_0 p
there.  At long metal paths, where the excess is large, they differ at second order.

There are no local minima, because the problem is a convex QP.  With two metals, the columns
`m_0 m_1` are supported only on rays through both metals, so their coefficients are decided by few
rays and few slices.  The scoring slab must hold both metals, or those columns should be dropped.

### (b) A polychromatic model

The model is y = −log Σ_j w_j exp(−Σ_k μ_kj L_k), with L_k the class path lengths from the
segmentation.  The free parameters are J−1 weights and (K+1)J attenuation values, (K+2)J−1 in all:
8 for one metal and 3 bins, 14 at 5 bins, 19 for two metals at 5 bins.  Tying the attenuation to
tabulated photoelectric and Compton basis curves at fixed bin energies reduces the count to one
density scale per material plus the weights, K+J in all: 6 for one metal and 5 bins, 7 for two.
With unknown materials it is 2K+J+1.  The single-material curve already in the package is this model
with equally spaced attenuation bins and log-weights (`mbirtorch/preprocess/utilities.py:1340-1386`).

The gain over monomials is structural.  Cross terms cost nothing, because one spectrum explains the
plastic-only, metal-only, and mixed rays together, so the count grows by one per added metal instead
of by nine.  The curve is monotone and concave by construction, so it extrapolates to path lengths
the reduced data do not contain.  The cost is nonlinearity.  Each evaluation is elementwise on the reduced sinogram, which is cheap,
but the image score needs one reduced direct reconstruction per candidate.  A Gauss-Newton step
instead needs K+J slab reconstructions per iteration for its Jacobian, which is the basis trick of
(a) applied per iteration.  The gauge is the linearization point: the corrected sinogram's
scale is the slope at zero path length, as `fit_inverse_beam_hardening_curve` already chooses
(`utilities.py:1459-1470`).  Identifiability is the known weakness: the weights are seen only through
the curvature of the curve along each material's path-length axis, and the fit can collapse to one
bin.  Initialize from the monomial fit, because the slope at zero is Σ w_j μ_j and the quadratic
coefficient is minus the weighted variance of μ over bins.  The assumption that needs checking is
that a two-basis attenuation form fits the metal present; it fails above a K-edge inside the
spectrum, which tungsten and tin have and steel, aluminum, and titanium do not.

### (c) A search over the existing hyperparameters

The dimensions are `beta` (continuous), `alpha` (continuous), and `order` and `num_metal`
(discrete, a few values each).  For `beta` and `alpha` the normal matrix HᵀH and the vector Hᵀy are
fixed once per segmentation, so each candidate costs a 15-by-15 solve, the active-set scan over the
reduced sinogram, one elementwise correction, and one reduced direct reconstruction.  A
golden-section search at `_search_minimum`'s defaults spends about 24 evaluations
(`geometry_calibration.py:1085-1135`), so a one-dimensional search over log(beta) is about 24 slab
reconstructions, under a minute on a GPU.  The discrete choices multiply that by 6 at most.  This is
the cheapest useful step.  It automates the tuning that happens by eye today, with the same
two-level scheme and the same undecided verdict.

What it fails to fix.  It cannot change which columns exist, so plastic cupping stays out of reach.
It cannot move the ridge's direction, only its strength, because the weight matrix is fixed by
degree.  It cannot repair a segmentation error, which changes p and m themselves.  And it inherits
the divisive form and its clamp, so `gamma` remains a hand-set floor.  `beta` is scaled by
trace(HᵀH)/trace(W) (`mar.py:526-527`), so a value found on binned data transfers only
approximately, and the fine level should confirm it.

### (d) Gradient descent through autograd

The forward pass at the reduced scale is elementwise correction, the FDK filter, the back
projection into the slab, and the score.  The direct path is not under `no_grad`; the two `no_grad`
blocks are in the iterative recon (`mbirtorch/tomography_model.py:3203-3224`), and the adjoint-pair
wrappers supply the projector's backward on a single device (`mbirtorch/autograd.py:58-101`).  The
graph holds a few reduced sinograms and a few slab images, under 1 GB.  At the production scale the
same graph would hold several 28.8 GB arrays, so autograd is a reduced-scale tool only.

The reason not to use it as the optimizer is conditioning.  The Hessian in θ is BᵀQB.  The
basis images of `m`, `m²`, `m³` are nearly collinear, and their Gram matrix is like that of monomials
on [0, 1], whose condition number is 10³ at degree 3 and grows fast with degree.  Plain gradient
descent converges at a rate set by that condition number and would need thousands of iterations,
where the linear solve of (a) takes one.  Autograd earns its place in two cases: a score that is not
quadratic, and a correction that is not affine, such as (b) or a soft segmentation.  Even then the
Jacobian has only 6 to 15 columns, so Gauss-Newton with an autograd-built Jacobian is the right use.
Autograd is also a cheap check of the hand-built basis images of (a).

### (e) Block coordinate descent

Cost is not the reason for blocks here, because every block solve in (a) is already free.  The
reason is identifiability.  The natural blocks are the plastic-only curve, the cross columns of each
metal, and the metal-only columns of each metal.  Each block should be decided from the slices and
rays that see it.  Plastic cupping comes from slices without metal, metal k's blocks from a slab
that holds metal k, and the two-metal cross columns last, from slices holding both.  This is the
analog of the geometry estimator's rule that the offset is read from central slices and the
rotation from far ones.  The chooser's rule inverts: the geometry estimator avoids dense features,
and this one must select them.

### (f) Two levels, and what binning does to a nonlinear fit

A view stride changes no pixel value, so it introduces no bias.  Binning does.  The mean over a bin
of b·p² is b·(p̄² + var(p within the bin)).  Binned data therefore equal the model applied to the
binned path length plus an extra term.  That term is nonzero only where the path length changes
within a bin, which is at edges.  For a disk of radius 800 pixels and y = p − 0.05 p², the least-squares quadratic
coefficient fitted on binned data is biased by under 0.01 percent at bin 4.  For a disk of radius
50 pixels, a metal-sized feature, the bias is 0.2 percent at bin 2 and 1.1 percent at bin 4, roughly
quadratic in the bin factor.  Cubic and cross columns will be biased somewhat more, and the metal
columns are exactly the ones supported at small features.  The consequence is the geometry
estimator's scheme: the coarse level localizes θ to about one percent, and the polish runs at full
channels on the cropped rows with the view stride kept.  If the coarse level proves too biased,
bin rows only, because the in-plane edges that carry the bias lie along channels.

### Applying θ to the full sinogram

The correction is a per-pixel function of y, p, and m_k, so it is a view-batched map like
`BH_correction` (`mar.py:146-203`, through `pipeline.map_view_batches`,
`mbirtorch/preprocess/pipeline.py:99-168`).  The host holds the measured sinogram and the
pre-allocated output, and the device holds one batch.  The kernel needs p and m_k for its
batch, so it must forward project the class volumes for those views only.  The per-view-batch forward
body exists (`mbirtorch/cone_beam.py:161`), but no public entry projects a subset of views.  Today's
correction instead materializes p, every m_k, and two column temporaries as full sinograms
(`mar.py:271-283`, `mar.py:501-513`), which is 3+K full copies.  The remaining production-scale cost
is the class volumes: K+1 float masks at 32 GB each, or one uint8 label volume at 8 GB if the
projector could take a label and a class value.

## Ranked recommendation

1. The `beta` search (c), on the reduced problem, with a within-class variance score on a slab
   through the metal.  Assumption: the score's minimum over `beta` at the reduced scale lies where
   a full-scale reconstruction is judged best.  Smallest experiment: on the NSI metal scan, sweep
   `beta` over five decades at the reduced scale and at full channels on the cropped rows of two
   slices, and compare the two argmins and curve depths against the by-eye value.  One variable,
   `beta`; everything else fixed.

2. The affine correction and closed-form θ (a).  Assumption: the subtract-the-excess form is as
   good as the divisive form on real data.  Smallest experiment: take the θ the existing fit
   produces on the NSI metal scan, apply both forms, run the same MBIR, and compare; the form is the
   only variable.  Then the search test: on synthetic data with a known quadratic-plus-cross model,
   compare θ from the sinogram fit, from the image fit, and from their λ-weighted sum against the
   truth, and record which directions each misses.

3. The polychromatic model (b) with tabulated materials, fitted by Gauss-Newton with per-iteration
   basis images.  Assumption: a two-basis attenuation form fits the metals present.  Smallest
   experiment: fit both models to the same real sinogram and compare the residual on metal rays and
   the corrected reconstructions.

4. Blocks (e) as the ordering inside 2 and 3, and autograd (d) as a Jacobian check and as the tool
   for a non-quadratic score.  Neither is a separate method.

## Open questions for Greg

- Which score for beam hardening?  The geometry score is blind to cupping by design.  A within-class
  variance is quadratic and sees both cupping and streaks.  It is also Otsu's own objective, so the
  segmentation and the score would share one blind spot: a real density gradient in the plastic is
  indistinguishable from cupping.  Is the metal-free NSI scan the way to measure that?
- Should plastic cupping enter the model at all?  Adding `p²` and `p³` columns is one line in the
  exponent list, but then the gauge must fix the slope at zero path length explicitly.
- Is a view-subset forward projection an acceptable addition to the projector API?  Without it the
  correction cannot stream at the production scale.
- For the polychromatic model, are the metals in the real scans below the K-edge range, and do you
  want the materials named by the user or fitted?
- How far should the coarse level be trusted for the metal columns?  The one-percent figure above is
  for a quadratic on a disk; the real gate is a fit on binned versus unbinned synthetic data.
