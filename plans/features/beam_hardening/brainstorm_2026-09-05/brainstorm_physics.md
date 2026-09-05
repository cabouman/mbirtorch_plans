# Beam hardening: the physics, what one scan can identify, and what an estimator should search

Date: 2026-09-05.  Charge: the physics and identifiability of multi-material beam hardening (BH),
for the brainstorm on estimating BH parameters from reconstruction quality.

Provenance.  A number marked (sim) is a calculation I ran in this session with the model of
section 7: a Kramers spectrum without characteristic lines, Elam attenuation tables through the
`xraydb` package, and an energy-integrating 0.6 mm CsI detector.  A number marked (file) was read
from the NSI scan's `.nsipro` configuration.  A number marked (recalled) is from memory and is not
verified here.  The scripts and their output sit beside this report (`bh_physics_*.py`,
`*_output.txt`).

## Answer

The true forward model is a positive mixture of exponentials in the vector of per-material path
lengths, and it is increasing, concave, and zero at zero for any spectrum.  Both NSI scans ran at
200 kV through a 0.9 mm copper filter (file).  At that setting the plastic is linear to one
percent over 10 cm, so the artifact comes from the metal's own curve, whose slope falls 40 percent
between thin iron and 1 cm of iron, plus a cross term of 3 to 10 percent by which the metal
reduces the plastic's attenuation behind it (sim).  The cubic polynomial in `mar.py` fits the
presented combinations to about 0.6 percent at 200 kV, but one step of extrapolation in metal
thickness produces errors of 25 percent at 200 kV and several hundred percent at 100 to 150 kV
(sim).  A log-sum-exp with three or four shared bins fits 10 to 100 times better and extrapolates
100 times better (sim).  From one scan the data pin the map only on the presented set.  When every
ray through the metal crosses the same plastic length, the split between the cross terms and the
metal-only terms is not identifiable, the ridge penalty decides it, and the default penalty
already costs 2 to 7 percent errors on metal rays without noise (sim).  I recommend searching a
physical family of three to five parameters and correcting by exact inversion, with the image
score as the check.  If the polynomial stays, the image score should search the ridge and floor
hyperparameters, not the coefficients.

## 1. The polychromatic model and its parameter count

The measurement on one ray is y = -log [ integral S(E) D(E) exp(-sum_k mu_k(E) L_k) dE / integral
S(E) D(E) dE ].  Only the normalized product S(E) D(E) enters.  With J bins of weight w_j this is
y = -log sum_j w_j exp(-sum_k mu_k(E_j) L_k).

With K+1 known materials the unknowns are J-1 weights and one density scale per material, J+K in
all.  Fixed equally spaced energy bins are a poor basis.  With only the weights free, 6 bins reach
a maximum error of 0.016 where the attenuation reaches 4.35 (sim).  A spectrum generated from
physical knobs is a better basis.  With the voltage from the scan file and the copper-equivalent
filter thickness as the one knob, the fit reaches a maximum error of 0.011 against a truth made
with a different detector and inherent filter, and a second knob for the voltage reaches 0.0045
(sim).  So the physical count for K=1 is 1 to 2 spectrum knobs, 2 density scales, and optionally 1
scatter constant: 3 to 5 parameters.

With unknown materials, the photoelectric-plus-Compton decomposition (Alvarez and Macovski 1976,
recalled) gives mu_k(E) = a_k f_pe(E) + b_k f_c(E), two numbers per material that include the
density.  The count is 2(K+1) + (J-1), or 7 for K=1 with J=4.  A log-sum-exp with free shared
bins, f(p, m) = -log sum_j w_j exp(-a_j p - b_j m), has 3J-1 parameters, 8 at J=3.  The polynomial
has 6 columns for K=1 and 15 for K=2.  The counts are similar.  The difference is shape.  Every
mixture of exponentials is concave, and the polynomial is unconstrained.

## 2. Why cross terms exist, and how large they are

The metal removes the low-energy photons, so the plastic behind it attenuates less per centimeter
than plastic in the open beam.  In the polynomial this is the p times m^a columns. The table gives
the ratio of dy/dp behind the metal to dy/dp without it, and the interaction y(p,m) - y(p,0) -
y(0,m), for PMMA of 2 to 8 cm (sim).

| setting | metal | dy/dp ratio | interaction, percent of y(0,m) | interaction, attenuation units |
| --- | --- | --- | --- | --- |
| 200 kV, 0.9 mm Cu | Fe 1 mm | 0.97 | 3 to 10 | 0.01 to 0.05 |
| 200 kV, 0.9 mm Cu | Fe 10 mm | 0.89 to 0.90 | 2 to 6 | 0.05 to 0.17 |
| 200 kV, 0.9 mm Cu | Al 20 mm | 0.97 | 1 to 4 | 0.01 to 0.05 |
| 150 kV, 1 mm Al | Fe 1 mm | 0.81 to 0.89 | 10 to 28 | 0.13 to 0.34 |
| 150 kV, 1 mm Al | Fe 10 mm | 0.72 to 0.81 | 4 to 11 | 0.17 to 0.51 |

The metal-only nonlinearity is larger than the cross term by about ten to one.  At 200 kV, 1 cm of
iron attenuates by 2.9 where the thin-iron slope predicts 4.9, a deficit of 2.0, and the cross
term at that thickness is at most 0.17 (sim).  Both matter.  An error of 0.17 on a metal ray
equals 0.85 cm of plastic at 0.2 per cm, which is a visible streak.

## 3. Approximation quality

The table is a least-squares fit on a uniform grid of PMMA 0 to 8 cm by iron 0 to 1 cm, no noise,
in attenuation units (sim).  The `mar.py` cubic has the six columns p, pm, pm^2, m, m^2, m^3.  The
full cubic has all nine monomials of degree three or less.  The extrapolation row fits on iron of
0.5 cm or less and reports the error at 1 cm with no plastic.

| setting | y max | mar.py cubic, rms / max | full cubic, max | free LSE J=3, max | free LSE J=4, max |
| --- | --- | --- | --- | --- | --- |
| 200 kV, 0.9 mm Cu | 4.35 | 0.009 / 0.026 | 0.031 | 0.0018 | 0.00014 |
| 150 kV, 1 mm Al | 6.13 | 0.074 / 0.30 | 0.33 | 0.035 | 0.008 |
| 100 kV, 1 mm Al | 8.36 | 0.096 / 0.39 | 0.43 | 0.077 | 0.012 |
| extrapolation error at (p=0, m=1 cm), 200 / 150 / 100 kV | | +0.69 / +6.3 / +8.3 | | -0.006 / +0.20 / +0.31 | |

The polynomial goes wrong in three places.  First, its metal cubic has an inflection inside the
fitted range, at 0.85 cm at 200 kV and 0.75 cm at 150 kV (sim), so it is convex where physics is
concave.  Second, it is linear in the plastic.  At 200 kV with the copper filter this costs
nothing, because the plastic's attenuation per cm falls only 1 percent over 10 cm (sim).  At 100
to 150 kV with light filtration it falls 13 to 15 percent over 8 cm, the plastic-only residual of
the fit is 0.19 to 0.25, and no coefficients can correct cupping, because the model has no p^2
term.  Third, photon starvation. At 200 kV, 1 cm of iron transmits 5.5 percent and 2 cm transmits
0.8 percent; at 100 kV, 1 cm transmits 0.12 percent (sim).  The bias of -log(N/N0) is 1/(2N),
0.005 at 100 photons (sim). The larger low-count effect is a constant floor from scatter or
detector offset, which caps y at
-log(floor), 5.3 at a half-percent floor.

The shared-bin log-sum-exp has the right shape.  It is the negative of a log-sum-exp of affine
functions, so it is jointly concave, increasing, zero at zero, with slope at zero equal to the
mean attenuation and asymptotic slope equal to the smallest bin's attenuation.  The
single-material form in `utilities.py`, with bins at i times theta_0, is the same family with
equally spaced attenuation values; it needs 6 bins at 100 kV for a maximum error of 0.0003 (sim).
The ORNL reader's `find_linearization_fit` in `pymbir.py` is a two-bin member of the same family
in closed form.

## 4. What one scan identifies

The data determine the map only on the set of path-length combinations the object presents. Two
spectra equal on that set give the same data and the same correction.  The correction needs the
map at the estimated combinations (p_hat, m_hat) from the segmentation, and those lie off the
presented set wherever the classes are wrong: along first-pass streaks, and at metal edges where
partial volume makes m_hat too small.

Collinearity is the failure that matters.  For a rod on the axis of a plastic disk, every ray
through the rod crosses nearly the same plastic length, 7.4 to 8.0 cm in my example (sim).  The
columns pm and m are then collinear over the metal rays.  The unregularized fit fails, with
coefficients near 5e4.  The ridge penalty restores it, and at beta of 2e-4 the corrected plastic
on the presented rays is right to 0.5 percent.  But the split between the cross terms and the
metal-only terms is now set by the penalty.  A ray whose plastic estimate is 10 percent low, as
along a first-pass streak, is corrected with an error of +4.4 percent at beta 2e-4, +6.7 percent
at the default 2e-3, and +15 percent at 2e-2 (sim).  Exact inversion of the true model at the same
ray has no error.  For a rod off the axis the unregularized fit is right to 0.1 percent, yet the
default penalty still costs 2 percent rms and 6 percent maximum on the metal rays (sim). So the
ridge strength is a bias rather than a safeguard, and it is worth sweeping.  And an image score
sees the split only through the off-set rays, at edges and along streaks.

Three degeneracies remain.  A constant transmission offset from scatter lowers y most where y is
largest, which is what hardening does; a four-bin curve absorbs a 0.5 to 2 percent offset to
within 0.002 to 0.007 rms over 8 cm of plastic (sim).  The offset is exactly a bin of zero
attenuation, so the family contains it, and the data separate it from hardening only through rays
near saturation.  Second, the mask projections give geometric path lengths in ALU, so one density
scale per material is a real unknown that trades against the spectrum's mean energy; over the
presented set the trade is weak, and both explanations extrapolate physically.  Third, the
plastic's linear coefficient against the reconstruction's scale is a convention, fixed today by
least squares over plastic-only rays.  An image score must be scale-invariant and must not search
it.

One distinction organizes all of this.  Cupping of a single convex uniform body is consistent. It
is the projection of a cupped image, so no test of whether the data are the projection of some
image can see it.  It is identifiable only through the uniformity prior, which the mask projection
and any flatness score both assume, or through physics.  Streaks and bands between separated dense
parts are inconsistency, because the nonlinearity mixes chords that overlap differently at
different angles and the angular moment conditions (Helgason and Ludwig, recalled) fail.  That
part is identifiable from the data alone, and it is what a reprojection residual and a gradient
score read.

## 5. What over- and under-correction look like

Under-correction leaves the effective attenuation falling with path length.  The center of a
uniform body reads low, which is cupping, and rays through two metals read lower still, which is a
dark band between them with bright streaks tangent to the metal.  Over-correction reverses every
sign, giving capping and a bright band.  To first order the residual is linear in the parameter
error, so a signed statistic crosses zero once and its square has one minimum.  Two signed
statistics serve: the center minus the edge of the low-passed plastic class, and the mean of the
plastic between the metals minus the plastic elsewhere.  An unsigned gradient energy has its
minimum at the same place but cannot report the sign.  The bias of both is the object's own
density structure, and the no-metal scan measures it.  At 200 kV through 0.9 mm of copper a
PMMA-like plastic should cup by 1 percent or less (sim), so larger cupping in the no-metal
reconstruction is scatter, a denser material, or a vendor correction already applied.

## 6. What "corrected" should mean

The exact correction solves y = f(p, m_hat) for p, one monotone solve per ray, and outputs
mu_plastic(E_ref) p + mu_metal(E_ref) m_hat.  The `mar.py` correction computes (y - Q(m_hat)) /
P(m_hat), rescales it, and adds back m_hat times the reconstruction's own metal mean.  The two
agree when the polynomial equals f on the presented set, which holds to 0.1 to 0.5 percent at beta
of 2e-4 or below (sim), and they differ off the set by the 4 to 15 percent of section 4.  The
metal's added-back value is a free choice, because a mask times a constant reprojects exactly and
makes no streak; only a quantitative metal density needs E_ref.  For starved rays no inversion
recovers p.  The right output there is the model's own prediction from the mask and a weight near
zero, which is sinogram replacement made explicit; the gamma floor in `mar.py` does part of this
implicitly and does not report where it acted.  The exact correction is elementwise in y and
m_hat, so it streams by view batch with the metal mask projected per batch and needs no second
full-size sinogram.

## 7. The synthetic model for gates

Use tungsten at 200 kV behind 0.9 mm of copper, matching the NSI file, so that the polynomial is
never the truth.  Attenuation curves come from `xraydb` on PyPI, which installed in seconds here
and returns mu(E) by chemical formula and density from the Elam tables; these agree with NIST XCOM
to a few percent in this range (recalled).  Spectra come from a Kramers form, as in my script, or
from `spekpy` (pip from its Bitbucket repository, with characteristic lines) or `xrayphysics`
(conda-forge, LEAP's companion, with a two-material correction to compare against).  The detector
is an energy-integrating 0.6 mm CsI or 0.2 mm GOS layer.  Noise is Poisson at 1e5 photons per
pixel per frame times the file's 6 frames (file), and a scatter constant of 0.5 percent of the
open beam is the degeneracy test.  The ray-level model is the script beside this report, half a
day to clean up.  A 3D sinogram is K+1 mask forward projections per view batch followed by the
elementwise mixture of exponentials, about a day.

## 8. Ranked recommendation

1. Search a physical family and correct by exact inversion.  Parameters: the filter-equivalent
   thickness at the file's voltage, one density scale per material, and a scatter constant, 3 to
   5 numbers for K=1.  Fit them on the reduced problem by nonlinear least squares of y against
   f(p_hat, m_hat), each evaluation one elementwise pass over the reduced sinogram, with the
   image score as the check.  Assumption: each class is one uniform material with a tabulated
   curve up to a density scale, and the spectrum family is close enough; the misspecification
   test puts that error at a quarter of a percent of the largest attenuation (sim).  Smallest
   experiment: on the NSI metal scan's reduced sinogram, fit the one-knob family and the
   polynomial to y against the mask projections, compare residuals on plastic-only and metal
   rays, then fit on the thinner half of the metal rays and predict the thicker half.
2. If the polynomial stays, search beta and gamma by the image score, not theta.  Assumption:
   the streak energy in the plastic class is monotone in the off-set correction error, which
   beta controls (section 4).  Smallest experiment: a beta sweep from 2e-5 to 2e-2 on three
   slices of the reduced problem, on the synthetic case where the error is known and on the
   metal scan.
3. A free shared-bin log-sum-exp with 3 bins, only when the materials are unknown, with a_j and
   b_j tied to the photoelectric-plus-Compton curves to stop overfitting.
4. Do not search the overall scale, the weights of a ten-bin spectrum, or a plastic
   self-hardening term at this filtration.  There is nothing to find in any of them.

## Open questions for Greg

- What are the phantom's plastic and metal, and the metal's diameter and position?  A centered
  metal is the collinear case of section 4.
- Does the NSI export already apply a beam-hardening correction?  Another NSI file name in the
  data directory carries "0.5BH".
- Which scintillator does the detector use, and what is the open-beam count per pixel per frame?
  Both set the synthetic gate.
- Is a quantitative metal density wanted, or only artifact-free plastic?  Only the first needs
  a reference energy.
- Can the scatter constant be measured on the scanner, for example behind a lead strip, so that
  it is not left to the degeneracy of section 4?
