# Own view, written before reading the agents' reports

## The one structural fact

Direct reconstruction is linear in the sinogram.  If the correction is written as
"subtract the fitted nonlinear excess", c(y; θ) = y − Σ_{j in nonlinear columns} θ_j h_j,
then FBP(c) = FBP(y) − Σ_j θ_j FBP(h_j).  FBP(y) is the uncorrected reconstruction, already
computed at the start of recon_plastic_metal.  The FBP(h_j) are basis images: one per H column,
computed once on a reduced problem (a slab, stride, binning).  Every candidate θ then costs no
projector call.  With a score quadratic in the image on a fixed mask (within-class variance of
the plastic region, dilated away from metal edges), the best θ is a small linear system.

This is the EBHC idea (Kyriakou et al. 2010) placed on mbirtorch's reduced-problem machinery.
The existing correction divides by the coefficient of p, which is not linear in θ; the
subtraction form is the linear alternative.  What the subtraction form loses: when the fit is
exact, the division form returns exactly p (the projected segmentation), while the subtraction
form returns θ_0 p plus the residual.  Both keep the residual, which carries the real structure.

## Two separable contributions of the geometry machinery

A. The reduced problem makes the existing sinogram fit cheap.  Today the fit forms H^T H from
   full-size monomial columns: at 1800 x 2000 x 2000 each column is 28.8 GB, and 15 columns need
   120 column products.  Fitting θ on a reduced sinogram (stride 4, bin 2: 16x smaller) and then
   applying it view batch by view batch is a benefit regardless of any image criterion.  Open
   question: binning bias of a nonlinear map (mean of f(y) vs f(mean y)); for a quadratic term
   the bias is θ_2 Var_within-bin(y), which is small where the sinogram is smooth over a bin and
   large at metal edges.  Quantify on the NSI scan.
B. The image criterion chooses θ by the artifact, not by the data misfit.  The sinogram fit's
   residual is dominated by real structure that the piecewise-constant segmentation misses, so
   the polynomial can absorb structure.  The image criterion with 6 to 15 coefficients can only
   remove what lies in the span of the basis images (cupping and streak patterns), which is a
   built-in guard against flattening real structure.  This guard weakens as columns are added.

## Traps I expect

- Linear columns p and m in the image fit.  FBP(p) is the plastic mask plus FBP's own
  artifacts, which FBP(y) shares.  Fitting the p column would cancel direct-reconstruction
  artifacts that MBIR would not have.  Exclude the linear columns from the image fit: the plastic
  level is the data's own, and the metal level is a choice (the existing code adds m back at
  its projected scale).
- Photon starvation.  Rays through thick metal carry noise streaks that no polynomial removes.
  The image fit would try to cancel them with the m^2 basis image.  Use the same ray mask or
  weights in FBP(y) and in FBP(h_j); linearity survives any fixed linear operator.
- Collinear basis images.  m^2 and m^3 streak images are nearly proportional inside the
  plastic; ridge regularization as the sinogram fit already uses, or a degree cap chosen by a
  coarse sweep.
- Direct versus MBIR.  Score on FBP, check with one MBIR at reduced scale.
- Segmentation coupling.  Same alternation as today, but at reduced scale until θ and masks
  stop moving, then one full MBIR.  Cost today: 1 FDK + 3 x (fit + full MBIR).

## Nonlinear parameters that do not fit the linear inner solve

- A scatter offset s in transmission: y_s = −log(exp(−y) − s).  One FBP of the slab per s
  candidate, then the closed-form inner solve.  Variable projection: outer 1-D search, inner
  linear.
- A physical model (J spectrum bins, tabulated μ_k(E), density scales) has 3 to 6 parameters
  and needs one FBP per candidate; it extrapolates to unobserved path-length combinations by
  physics.  Best use: synthetic truth for gates (so the polynomial is not both truth and model),
  and a possible second-stage refinement.  XCal (Li, Bouman et al. 2025) and the xspec package
  estimate spectra from projections with a dictionary; relevant for the known-materials case.

## Real-data asset

The NSI phantom was scanned with and without the metal insert.  The no-metal reconstruction is
a near ground truth for the plastic's real structure.  The difference between the two
reconstructions, registered, isolates the metal-induced artifacts, so a real-scan gate can
measure artifact energy without a phantom model.

## The first experiment

2D-thin synthetic (a slab of one to three slices, parallel or cone), plastic disk plus one and
two metal disks, polychromatic truth with 3 to 5 bins and tabulated-like attenuation values,
Poisson noise.  Compare: the existing sinogram fit; the image-domain closed-form fit on basis
images; an oracle polynomial fit to the true monochromatic sinogram.  Vary one thing at a
time: noise, mask erosion/dilation by a pixel, real texture and holes in the plastic, metal
size.  Measure: RMS error in the plastic region against the monochromatic reconstruction,
streak energy between the metals, plastic-mean bias.  The result that kills the image idea:
the image fit is worse than the sinogram fit under mask error or texture.
