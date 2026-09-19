# Reasoning review of `surveys/beam_hardening/survey.md`

Date: 2026-09-05.  Reviewer charge: judge the reasoning, not the prose.  Read-only except for this
file.  Sources checked: the six brainstorm reports, `experiments/bh_physics_sim_output.txt`,
`experiments/bh_physics_extra_output.txt`, `experiments/counts_and_binning.py`, `own_view_before_reports.md`, and the code at
`mbirtorch/preprocess/mar.py` and `mbirtorch/preprocess/geometry_calibration.py` on
`geometric_calibration` at 590fea9.

Entries are ordered by severity.  Findings 1-6 change conclusions; 7-13 change experiments; 14-18 are
qualifiers.  Section 19 lists what survives review, so the repairs are not mistaken for a rejection.

---

## 1. The headline number for the physical family is contradicted by the file the page cites

**Claim** (Answer): "The physics report adds a stronger recommendation for the model itself: a
three-to-five parameter physical family fits the presented data 10 to 100 times better than the cubic
and extrapolates to thicker metal about 100 times better (`experiments/bh_physics_sim_output.txt`)."

**Problem.**  Those ratios belong to a different and much larger family.  Read from
`experiments/bh_physics_sim_output.txt` at 200 kV / 0.9 mm Cu (lines 22-33):

| model | free params | max abs residual |
| --- | --- | --- |
| `mar.py` cubic | 6 | 0.0260 |
| physical LSE, fixed-energy bins, weights only | 2 | 0.2592 |
| physical LSE, fixed-energy bins, weights only | 3 | 0.0431 |
| physical LSE, fixed-energy bins, weights only | 5 | 0.0163 |
| free shared-bin LSE, J=3 | 8 | 0.00178 |
| free shared-bin LSE, J=4 | 11 | 0.00014 |

The 10-100x figure is the free shared-bin LSE at 8 and 11 parameters (14.6x and 186x).  At the
parameter count the page names, the same file shows the physical family at 3 parameters is **worse
than the cubic** (0.0431 vs 0.0260) and at 5 parameters is 1.6x better, not 10-100x.  The one
directly comparable measurement of the physics report's *actual* proposed family — the one- and
two-knob spectrum fits in `experiments/bh_physics_extra_output.txt` (b) — gives max 0.0112 and 0.0045 against the
cubic's 0.0219 **on the same truth**: a factor of 2 to 5, not 10 to 100.

The extrapolation rows (`experiments/bh_physics_sim_output.txt` lines 32-33, 67-68, 102-103) exist only for the
cubic and the free LSE J=3.  No extrapolation number was ever computed for a 3-5 parameter physical
family in these files.  The nearest measurement is the misspecified one-knob fit, which errs by
-0.0144 at (p=0, m=1 cm) against the cubic's +0.6942: 48x, and only at 200 kV.  At 150 and 100 kV the
free LSE's own ratios are 31x and 27x, so "about 100 times" is the single best case reported as if it
were the rule.

The physics report itself keeps the two families separate — its Answer attributes the ratios to "a
log-sum-exp with three or four shared bins" and separately gives the physical family's 0.011/0.0045.
The page merged them.

**Fix.**  Restate as: the free shared-bin log-sum-exp (8-11 parameters) fits 15-190x better; the 3-5
parameter physical family fits about 2-5x better than the cubic and extrapolates roughly 50x better
at 200 kV.  If the "which family" decision is to be put to Greg on accuracy grounds, it must be put
on those numbers, because a 2x interpolation gain and a big extrapolation gain argue for a different
staging than a 100x gain everywhere does.  Add one line to `experiments/bh_physics_extra.py` computing the
extrapolation gate for the 3- and 5-parameter physical families, since that is the number the
recommendation actually rests on and it does not exist yet.

---

## 2. The primary score measures the one artifact the recommended family cannot change

**Claim** (The score): "The primary score is the low-frequency deviation of the plastic class from its
own mean, inside a mask eroded away from every edge and every metal ... a uniform disk's center dips
by about 13 percent of its attenuation."

**Problem.**  Three of the page's own statements make this incoherent, and the page never puts them
together.

1. "The model has no plastic-only nonlinear column ... So the plastic's own cupping is outside the
   family today."  This is stronger than the page states for the *subtraction* form specifically:
   every nonlinear column is `p·m^a` or `m^a` with `a >= 1`, so **every** searched column vanishes
   identically on metal-free rays.  For any θ, `c(θ) = y` on every ray that misses metal.  The
   estimator's entire image-domain signal comes from rays crossing metal, and the basis images are
   streak- and band-shaped, not dome-shaped.
2. The 13-percent center dip is derived in `brainstorm_score.md` for `y = p − b p²`, a *plastic-only*
   quadratic, at the validation harness's artificial sizing (b chosen to move the largest sinogram
   value by 10 percent).  The page transplants that depth into a section describing a family that
   contains no `p²` column.
3. The page's own physics says the target scan has almost no cupping to find: at 200 kV behind 0.9 mm
   of copper "a PMMA-like plastic is linear to one percent over 10 cm."  That is ~10x smaller than
   the harness sizing, so ~1.3 percent center dip and ~0.3 percent rms — 100x less score energy than
   the quoted figure.

The two masking rules make it worse: the score excludes the metal and an edge margin, and the primary
region is the *far* plastic — precisely where the basis images have least energy.  The measures that
sit where the family acts (corridor and annulus contrasts) are the page's secondary readouts.

**Fix.**  Demote the low-frequency far-plastic deviation to a *model-adequacy* diagnostic (it detects
plastic self-hardening and scatter, both outside the family) and promote the signed corridor and
annulus contrasts to primary, since those are the directions the family spans.  Delete the 13-percent
depth from the design argument or relabel it as the depth of an artifact the family cannot correct.
Recompute the expected depth as the masked energy of the actual basis images `R(h_j)` at the fitted
θ — a number that costs one run of experiment 1 and is currently nowhere in the page or the sources.

---

## 3. The stated correction double-counts the metal, and the "uncorrected candidate" is undefined

**Claim** (basis images): "Write the correction as the measured sinogram minus the fitted nonlinear
excess, `c(θ) = y − Σ_j θ_j h_j` over the nonlinear columns `h_j`, with the linear columns left
alone."  Plus, in the gauge rule: "the metal's level is a choice that the existing add-back makes
(`mar.py:818-824`)."

**Problem.**  `mar.py:818-822` is `out = scale * corrected + Σ_k m_k * metal_scale_k`.  In the existing
divisive form that is not a double count, because `build_y_minus_sm` (`mar.py:677-682`) has already
subtracted the *whole* metal-only block from y, including the linear `m` column.  In the page's
subtraction form the linear columns are "left alone", so `y` still contains the metal's own
attenuation; adding `m_k · scale_k` on top adds it a second time.

`brainstorm_search.md` has this right and the page dropped the term: "The fitted linear metal is
swapped for the recon's metal by adding `Σ_k (ρ_k − θ_k,lin) m_k`, as today."  With that term the
construction is consistent; without it the corrected sinogram is wrong by one full metal projection.

Two consequences follow.  First, the gauge sentence is unsupported as written: the metal's level is
set by whichever of the two conventions is chosen, and the page states neither.  Second, the safety
rule "Every candidate, including the uncorrected one, goes through the same code path" has no
referent.  θ = 0 in the subtraction form is not the uncorrected reconstruction — it is `y` plus the
add-back — which is exactly the skeptic's attack 1 route three ("The uncorrected sinogram is not in
the family at all, because the metal is always re-added, so setting `theta_m` to zero double counts
it").  The page repeats the rule and never answers the objection.

**Fix.**  Write the correction with the difference add-back explicitly, and define the uncorrected
candidate as the point `θ_nonlinear = 0, θ_lin = ρ` (no net metal change), stating that this is what
"uncorrected" means in this family and that it is *not* the raw reconstruction.

---

## 4. The gauge argument fails in exactly the case the page names as a trap

**Claim** (basis images): "The linear columns are never searched: the plastic's level is the data's
own, and the metal's level is a choice ... so the overall scale is fixed by construction."

**Problem.**  Half of this is sound and half is not, and the page asserts both at the same strength.

Sound half: because every nonlinear column carries a factor of `m`, the correction is the identity on
metal-free rays, so `_estimate_plastic_scaling` (`mar.py:724-739`) returns exactly 1 and the plastic's
level genuinely cannot move.  This is a *better* argument than the page gives and worth stating.

Unsound half: "by construction" implies the metal cannot be scaled away.  But the page states two
paragraphs later, and again in the traps table, that "thin metal makes m, m², m³ one column."  When
`m` is near-binary, `m² ≈ m³ ≈ m`, so `R(m²)` is nearly parallel to `R(m)`, and a large coefficient on
a *nonlinear* column erases the metal just as effectively as searching the linear one.  That is
skeptic attack 12 verbatim ("with the coefficient of the `R(m)` basis image near the metal's scale,
the metal vanishes from the image and every streak with it"), and excluding the linear columns from
the search does not close it.  The bga solder balls are named on the page as this case.

**Fix.**  Replace "fixed by construction" with "fixed to the extent the nonlinear columns are not
collinear with the linear ones on the scored region", and make the collinearity measurable: report
the angle between `R(m_k)` and the span of `{R(m_k^a) : a >= 2}` restricted to the score mask, as a
per-scan number alongside the Gram condition number.  Add a thin-metal arm to experiment 1 (see
finding 12) so the failure is observed rather than argued about.

---

## 5. The "convex quadratic program with the existing machinery unchanged" claim is wrong three ways

**Claim** (basis images): "`S(θ) + λ · ‖Hθ − y‖² ... is a convex quadratic program.  The existing ridge
weights and the active-set positivity loop (`mar.py:492-629`) apply unchanged, because they need only
a quadratic objective and linear constraints."

**Problem.**

(a) *The page's own fifth safety rule destroys the quadratic.*  "the score is a ratio to the class
contrast, so that scaling the image cannot lower it."  `brainstorm_score.md` says what that costs:
dividing "by the square of the far-plastic mean ... makes a ratio of quadratics and turns the linear
system into a generalized eigenproblem."  The page keeps both the QP and the ratio and never notes
they are incompatible.  (Given finding 4's sound half, the ratio may be unnecessary here — the scale
is already pinned — which is the cheapest resolution, but it has to be argued, not left implicit.)

(b) *Constraint (1) has no meaning in the subtraction form.*  The existing constraints are, from the
docstring at `mar.py:534-560`, `H_p[i,:] θ_p >= 0` and `y[i] − H_m[i,:] θ_m >= 0`.  The first exists
to keep the *denominator* `Sp` positive.  The subtraction form has no denominator, so that constraint
family is not "unchanged", it is meaningless.

(c) *Constraint (2) covers the wrong column block.*  It bounds only the metal-only block `H_m θ_m`.
The subtraction form also subtracts the cross columns `p·m^a`, whose contribution to
`y − Σ_nonlinear θ_j h_j` is unconstrained in sign.  The claim "negative corrected values are
prevented by the positivity constraints instead" of the clamp is therefore false for the cross block
as the code stands.  Note also that the loop adds at most `num_constraint_update_iter = 10`
most-violated pixel constraints of each type; it is an approximate scheme, not a guarantee of
non-negativity anywhere.

(d) *Unstated linearity requirement.*  `plastic_sino_corrected_scale` (`mar.py:814`) is refit per call
and is a function of the corrected sinogram, hence of θ; a per-candidate refit makes the map
θ → corrected sinogram a ratio, not affine, and breaks the basis-image identity.  It happens to equal
1 here (see finding 4), but the page must say the outer scale is frozen, because that is a
precondition of its central mechanism.

**Fix.**  State the constraint set the subtraction form actually needs — a single family
`y − Σ_{nonlinear} θ_j h_j >= 0` assembled over all nonlinear columns — and say that
`_estimate_BH_model_params`'s row assembly changes even though its solver does not.  Choose between
the ratio score and the QP explicitly.  Add "the outer plastic scale is held fixed across candidates"
to the five safety rules.

---

## 6. The staged recommendation parametrizes the three-way disagreement rather than resolving it

**Claim** (Answer and The family to search): "The design that follows from both facts is a hybrid.
The sinogram fit stays and pins every coefficient direction the image cannot see. ... My
recommendation on the family is staged.  The polynomial with the subtraction form goes first."

**Problem.**  The three source positions map exactly onto three values of one free knob:

- λ → ∞ is the existing sinogram fit, which is the skeptic's position that the image adds nothing
  it can be trusted on.
- λ → 0 is the pure image fit, which the skeptic, the literature (Levi's seven-parameter collapse),
  and the physics report all say fails.
- intermediate λ is the page's recommendation.

The page never gives a rule for choosing λ, and no source supplies one: `brainstorm_search.md` only
proposes comparing "θ from the sinogram fit, from the image fit, and from their λ-weighted sum
against the truth", and the literature reports a hand-set convex weight (0.61 in the Levi SPIE
version).  Experiment 1 selects λ against a synthetic monochromatic truth that does not exist at
deployment.  So the design does not decide between the three positions; it defers the decision to an
unspecified knob and then calibrates that knob on an oracle.

Second, the page misdescribes the physics report's disagreement as being about the model only: "The
physics report adds a stronger recommendation for the model itself."  The physics report's ranked
recommendation 2 is about the *search*, and it is the direct opposite of the page's stage 1: "If the
polynomial stays, search beta and gamma by the image score, not theta."  The page's stage 1 searches
theta.  That override may well be right — the closed-form image fit is genuinely the cleanest test of
the premise — but it is an override and is never engaged.

**Fix.**  Add a deployable λ rule and gate it.  The obvious candidate, which nothing in the page or
the sources tries: choose λ by the even/odd split — the largest λ-driven move that is reproducible
across the two half-view reconstructions — or by discrepancy (the largest image-driven move whose
sinogram residual stays inside the noise level of the reduced data).  Either is testable on
experiment 1's synthetic case against the oracle λ.  Separately, quote the physics report's
recommendation 2 and give the reason for overriding it.

---

## 7. Per-slice and per-region agreement is a physics argument applied to non-physical parameters

**Claim** (Where the geometry recipe transfers): "The undecided verdict ... transfers with more force
than in geometry, because hardening parameters are set by the spectrum and the materials and are the
same in every slice and region while structure is not."

**Problem.**  True of the physical family's parameters.  False of the polynomial's coefficients, which
are the page's stage-1 target.  The polynomial is a misspecified least-squares approximation to `f`
over the *presented set* of `(p, m)` pairs, and the physics report establishes both halves of this:
"the data determine the map only on the set of path-length combinations the object presents", and the
cubic's residual reaches 0.026 on that set.  A slice through the middle of the metal and a slice near
its end present different path-length ranges, so their best-fitting θ legitimately differ.  Per-slice
disagreement is therefore expected under the recommended family and is not evidence of an
unidentifiable estimate.

Given that the page also makes the undecided rules the main defense against the self-fulfilling
segmentation (traps table, "agreement rules"), this weakens a load-bearing safeguard.

**Fix.**  Apply the agreement rule to a slice-invariant quantity rather than to θ: the *corrected
sinogram* or the corrected attenuation at a fixed probe `(p, m)` inside every slice's presented set,
not the coefficients.  Or state that the agreement rule is available in full only for the physical
family, which is one more argument for the physics report's ordering.

---

## 8. Three of the eight traps have remedies that do not remedy

**Photon starvation.**  Remedy given: "the same ray mask in `R(y)` and every `R(h_j)`, which keeps
linearity."  The linearity claim is correct — a fixed diagonal mask `D` commutes with the affine
combination, `R(D(y − Σθh)) = R(Dy) − Σθ R(Dh)` — but linearity was never the trap.  The trap is that
the score rewards hiding starvation streaks.  Masking the rays out (i) does not stop that, it just
changes which artifact the score sees, (ii) introduces its own incompleteness artifact into every
basis image, and (iii) breaks matched processing, because the delivered full-scale correction applies
θ to those rays.  The page answers a computational objection in the slot for a statistical one.

**Collinear columns.**  Remedy given: "ridge and data term".  The physics report's finding is that for
a centered rod the collinearity is *in the data*: "the columns pm and m are then collinear over the
metal rays", so the data term is flat in the same direction the image is flat.  The page states this
correctly in its ridge paragraph ("whatever the data cannot pin the image pins only weakly") and then
contradicts itself in the table by listing the data term as the remedy.  What is left is the ridge
(which the page itself calls "a bias rather than a safeguard") and the undecided verdict.

**Signal removal.**  Settler given: "the gaming curve of experiment 1" — see finding 9.

**Fix.**  For starvation: the remedy is a *weight*, not a mask — down-weight starved rays in the score
with the same weights the final MBIR uses, so the score cannot be won on rays the delivered image
discounts; keep the same weights in every basis image.  For collinearity: strike the data term from
that row and let the row read "ridge (a bias), Gram condition number, undecided" so the honest answer
is visible.

---

## 9. Experiment 1's gaming curve walks paths that do not exist in the proposed family

**Claim** (What to build, 1): "Then follow two one-parameter paths from the existing fit, a global
scale-down and an increasing metal subtraction up to full inpainting, and record every proposed score
along each.  Kill: every score reaches its minimum where the rms error is worse than at the sinogram
fit."

**Problem.**  Both paths were designed by the skeptic for the *existing divisive, clamped* correction.
The page has already ruled both out of its own family: the global scale-down is unreachable because
the linear columns are not searched (finding 4's sound half), and full inpainting is unreachable
because "the subtraction form drops the clamp at zero and the floor."  So the gaming curve tests the
gaming routes of the correction being replaced, not of the one being proposed, and passing it says
nothing about the proposal.

The path that matters for the subtraction form is the near-collinear one: increase the coefficient on
`m²` (and `m³`) along the direction in which `Σ θ_j h_j` best approximates `c · m`, and watch the
metal fade out of the image while the score falls.  That path is absent.

The page also drops the skeptic's *survive* criterion ("at least one scale-free score has its minimum
within the fit's own error of the RMSE minimum"), keeping only the kill.  Not being killed is not the
same as passing, and the page then treats the outcome as producing "the surviving score."

**Fix.**  Keep both old paths as controls for the existing correction, and add a third path along the
best `m`-mimicking direction in the nonlinear span, computed as the leading generalized eigenvector of
the masked Gram matrix of `{R(h_j)}` against `R(m)`.  Restore an explicit survive criterion.

---

## 10. Experiment 2 probes the wrong directions and gates on the wrong floor

**Claim** (What to build, 2): "compute the surviving score at the existing fit and at plus and minus
twenty percent on each coefficient in turn, and the even/odd floor.  Kill: no single-coefficient move
exceeds the floor."

**Problem.**  Two mismatches.

*Directions.*  Coordinate perturbations are the least informative probes of a near-collinear basis.
Moving `θ_{m²}` alone breaks the near-cancellation against `θ_m` and `θ_{m³}` and produces a large
image change; the flat directions are the eigenvectors of the masked Gram matrix with small
eigenvalues, which no coordinate move isolates.  So the test is biased toward passing, and passing it
does not show the score can decide the fit.  The page names the right instrument two sections earlier
("The condition number of the masked Gram matrix ... is the identifiability diagnostic") and then does
not use it in the experiment.  A "twenty percent" relative step is also ill-defined for ridge-shrunk
coefficients near zero and for the sign-alternating coefficients the physics simulation reports
(`theta = [0.206, -0.057, 0.035, 4.615, -2.810, 1.097]`, sim output line 22).

*Floor.*  The even/odd split measures noise repeatability.  `brainstorm_score.md` says what actually
binds: "The limit is systematic, not noise, and the agreement rules are what test it."  A score can
clear the noise floor while being dominated by FDK shading, segmentation error, or real structure.

**Fix.**  Perturb along the eigenvectors of the masked Gram matrix, ordered by eigenvalue, and report
the score change per unit sinogram-residual change in each — that is the identifiability curve, and it
answers the question in one plot.  Add a systematic floor beside the noise floor: the score change
produced by a one-voxel change in the segmentation threshold, which is the smallest realistic
systematic perturbation.  Note also that the traps table names "condition numbers on the NSI metal and
bga scans" as the settler for collinearity, but bga appears in no numbered experiment.

---

## 11. Experiment 3 cannot measure the gap it is named for

**Claim** (What to build, 3): "The MBIR gap. ... Run one MBIR pass with the image-chosen coefficients
and with the sinogram-fit coefficients ... Kill: the image-chosen coefficients give the worse MBIR
image."

**Problem.**  The gap, as the skeptic states it (attack 9), is a claim about *ranking*: "A `theta` that
trades streaks for bands therefore wins the FBP score and loses in the MBIR image."  Two points
cannot measure a ranking disagreement; they can only say which of two θ is better, and if the
image-chosen θ loses, the experiment cannot distinguish "FDK is the wrong proxy" from "λ was badly
chosen" from "the score is gamed."

**Fix.**  Evaluate MBIR at 4-6 θ spaced along the segment between the sinogram-fit θ and the
image-chosen θ (and one step beyond each end), and report the rank correlation between the FDK score
and the MBIR plastic-region error.  That is a direct measurement of the proxy's validity and it costs
a few more MBIR runs on a job already budgeted in hours.

---

## 12. Experiment 4's first gate contradicts a rule the page states, and its last gate is blind

**Claim** (What to build, 4): "Its gates: recovery of injected coefficients on the synthetic case; ...
and ... the plastic-only ray residual showing no trend against path length."

**Problem 1.**  The page's own rule three sections earlier is "The physical family is the synthetic
truth for every gate from the start, so that the polynomial is never both truth and model."  Under
that rule there *are* no injected polynomial coefficients to recover.  The two statements cannot both
hold.

**Problem 2.**  The plastic-only ray residual is constant in the searched parameters.  Every nonlinear
column carries a factor of `m`, so `c(θ) = y` identically on metal-free rays for every θ, and the
residual `y_c − θ_p p` binned over those rays does not depend on the estimate at all.  It is a check
on the segmentation, the plastic's own hardening, and scatter — which is what `brainstorm_score.md`
intends it for — but it cannot gate the estimator.

**Fix.**  Replace gate one with recovery of the *oracle* polynomial coefficients (the least-squares
best fit of the polynomial to the polychromatic truth on the presented set), which is well defined
under the truth rule and is the right target anyway.  Relabel the plastic-only trend as a
model-adequacy check reported in the result record, not as an estimator gate.  Note also that the
truth rule silently breaks at stage 2: when the physical family becomes the model it is also the
truth.  The fix exists in the sources — `experiments/bh_physics_extra.py` (b) already builds a truth with "a
different detector and inherent filter", and the physics report names `spekpy` (characteristic lines)
and `xrayphysics` as independent generators.

---

## 13. Selection on the same data: "every proposed score", then "the surviving score"

**Claim.**  Experiment 1 records "every proposed score" along the gaming paths and applies a
kill criterion.  Experiment 2 then evaluates "the surviving score" on the real scan, and the traps
table's remedies assume one score.

**Problem.**  Six or more candidate scores, several ablation axes, and a disjunctive kill criterion
make it likely that at least one score survives by chance on one synthetic case.  Carrying the winner
forward to experiments 2-4 is selection on the same data with no holdout and no pre-registration.  The
literature warning is adjacent: Levi et al. report that restarting from many initial points did not
recover estimability, i.e. that apparent successes at high parameter count were not reproducible.

**Fix.**  Pre-commit to one primary score before running experiment 1 (finding 2 argues it should be
the signed corridor/annulus contrasts), report the others as secondary, and split experiment 1's cases
into a selection set (one metal, centered) and a confirmation set (two metals, off-centre, thin
metal) that the chosen score must also pass unchanged.

---

## 14. The parameter-count evidence is not the controlled comparison the page implies

**Claim** (Answer): "The published method closest to this construction found that three image-chosen
coefficients were robust while seven were worse than no correction under clinical noise
(`brainstorm_literature.md`, Levi et al. 2021)."

**Problem.**  Per the literature report, ABHC-2's three coefficients are for *two* materials and
ABHC-3's seven are for *three*, with a different expansion.  So the comparison confounds parameter
count with material count and model.  It is still the best evidence available, but it is not the clean
count ablation the sentence suggests.

More importantly, the page never confronts what its own count means against that warning.  The Answer
says "A search over 6 to 15 coefficients", which is the column count of the whole `H` (the table in
"The family to search"), not the searched set.  With the linear columns excluded the searched set is 4
or 5 for one metal and 13 or 14 for two — the two-metal case is *twice* the seven that Levi reports
failing.  The page's answer is that the data term pins most directions, but it never says how many
effective degrees of freedom the image is being asked to decide.

**Fix.**  Add the qualifier about material count.  Replace "6 to 15 coefficients" with the searched
count, and report an effective-degrees-of-freedom number for the joint objective —
`tr(BᵀQB (BᵀQB + λHᵀH + ridge)⁻¹)`, computable for free once the Gram matrices exist — as a headline
diagnostic beside the condition number.  That number is what should be compared with Levi's 3 and 7.

---

## 15. "Leaves the reduced fit exact" overstates, and the omitted binning numbers are the large ones

**Claim** (Where the geometry recipe transfers): "the remedy is to build each column at full
resolution inside the view-batch loop and bin afterward, which leaves the reduced fit exact."

**Problem.**  Binning after building gives `min ‖B(Hθ − y)‖²`, which discards the within-bin residual
variation.  `brainstorm_pipeline.md` states this correctly and more weakly: it "leaves the
least-squares problem in theta a correctly binned linear problem."  The two estimators coincide only
if the model holds exactly; under the misspecification the physics report measures (cubic residual up
to 0.026), binned and unbinned least squares give different θ.  "Exact" is the wrong word for a
misspecified fit.

Second, the quoted bias numbers are the small ones.  The page cites `experiments/counts_and_binning.py`'s
coefficient-level biases for a *smooth disk* (0.23 percent at bin 2, 1.09 percent at bin 4, radius 50
px) directly after the sentence "at a metal edge that variance is large", and concludes "Metal
features are the small ones."  The pipeline report's edge-level numbers — 25 percent at bin 2 and 125
percent at bin 4 for a quadratic column, 75 percent for a cubic column at `g/m ≈ 1` — are omitted.
Those are per-pixel biases at the silhouette edge, a different quantity from the fitted-coefficient
bias, and they are the ones that motivate the remedy.

Third, the pipeline report's actual recommendation is bin_factor 1 on the channel axis ("The cheap
reduction for a beam-hardening search is therefore the view stride, with `bin_factor` left at 1"), and
its API sketch defaults to it.  The page's cost figures assume binning (the 86 MB column is at bin 2,
per `brainstorm_pipeline.md` section 2; the 128 MB basis stack at 500 x 500 is bin 4 of a 2K scan).
The override is defensible given the build-then-bin remedy, but it is not flagged as one.

**Fix.**  Say "leaves the reduced problem a correctly binned linear model" rather than "exact".  Quote
the edge-level biases beside the coefficient-level ones and say which quantity each is.  State that
the page is overriding the pipeline's bin-1 recommendation, and say the ablation "coefficients fitted
at bin 1 and bin 4 and applied at bin 1" is the gate for that override.

---

## 16. The scored problem is described two ways

**Claim.**  "Cost at production scale": "K+1 mask projections onto the window rows, one direct
reconstruction of the slab per basis column."  Experiment 2: "At the reduced whole-extent problem".

**Problem.**  The row-window slab and the whole-extent reduction are different reductions with
different costs (86 MB vs 1.8 GB per column, per `brainstorm_pipeline.md` section 2) and different
scoring properties.  The skeptic and the pipeline reports both land on whole-extent for beam
hardening; the page's cost argument and its basis-image memory figures require the window.  The page
uses each where it is convenient and never reconciles them.

There is also a scoring constraint the page states but does not cost: cupping is read on central
slices, the metal terms on slices holding the metal, and two-metal cross columns need a slab holding
both.  An 8-slice slab may satisfy none of these simultaneously.

**Fix.**  Pick one reduction for the experiments and price it, or state the rule (window for the basis
stack, whole extent for the identifiability diagnostics) and cost both.  Say explicitly which slices
each block of coefficients is read from, since the page adopts `brainstorm_search.md`'s block
ordering (e) without naming it.

---

## 17. Scatter absorption and physical extrapolation are in tension

**Claim.**  Traps table: scatter — "absorb it and do not read coefficients as physics".  The family
section: the physical family's advantage is that "Every mixture of exponentials is concave,
increasing, and zero at zero by construction", so it "extrapolates to path lengths the reduced data do
not contain."

**Problem.**  The extrapolation guarantee is exactly the claim that the fitted parameters *are*
physics.  If a filter-thickness knob or a density scale is absorbing a constant transmission offset —
which the physics report shows it can, "a four-bin curve absorbs a 0.5 to 2 percent offset to within
0.002 to 0.007 rms" — then the fitted curve is not the true attenuation curve off the presented set,
and the extrapolation argument is void in the same regime where it is needed (thick metal, near
saturation, where the physics report says scatter and hardening finally separate).  The page holds
both positions without noticing they cannot both apply to the same fitted parameters.

**Fix.**  Separate the two uses: absorb scatter when the goal is artifact-free plastic on the presented
set; do not absorb it when the argument for the family is extrapolation.  A scatter constant fitted as
its own zero-attenuation bin (which the physics report shows the family already contains) at least
makes the absorption explicit and auditable.  This also promotes the page's last open question —
whether scatter can be measured on the scanner — from a nicety to a precondition for the physical
family's headline advantage.

---

## 18. Smaller reasoning slips

- **"There are no local minima, because the problem is a convex QP"** (implicit in the page's
  convexity claim) sits beside the literature report's finding that Levi et al. "report several local
  minima" and that "more parameters make the cost surface more complex and add local minima."  Both
  are true — the difference is the score class (quadratic vs thresholded TV) and the correction form —
  but the page never says so, leaving a reader to think the published difficulty has been dissolved
  when it has only been avoided by a choice of score that finding 2 puts in doubt.
- **"extrapolates ... about 100 times better"** is the 200 kV case only; the same file gives 31x at
  150 kV and 27x at 100 kV.
- **The two-knob physical fit extrapolates worse than the one-knob**: `experiments/bh_physics_extra_output.txt`
  (b) gives max 0.0208 vs 0.0159 on the held-out thick-metal range.  The page reports only the
  interpolation numbers (0.011, 0.0045), where more knobs look better.
- **"450 views for 500 channels undersample metal edges"** is framed as a reduction artifact, but the
  full scan is 1800 views for 2000 channels — the same ratio.  The aliasing is in the acquisition, not
  created by the stride, which changes what the stride-1/stride-4 comparison can show.
- **`R(y)` "is the uncorrected reconstruction that `recon_plastic_metal` already computes"** — at the
  reduced scale it is not; it has to be computed again on the reduced model.  Trivial, but it is one
  more reduced reconstruction in a cost paragraph that counts them.

---

## 19. Questions a skeptical expert will ask that the page does not answer

Ordered by how much they could change the plan.  For each, whether a source answers it.

1. **How is λ chosen without a truth?**  Not answered by any source.  The page's only λ mention in an
   experiment is "at several λ" on the synthetic case, where the oracle exists.  This is the single
   most important gap: λ is the entire reconciliation (finding 6).
2. **Do the MAR weights in the final MBIR make the metal-only coefficients irrelevant to the
   delivered image?**  `brainstorm_score.md` asks this as an explicit open question for Greg
   ("Do the MAR weights in the final MBIR pass make the metal-only coefficients irrelevant ... so the
   search can stop at the plastic and cross terms?") and notes `vcd_utils.py:309-389` discounts metal
   rays.  The page drops it from its own open-questions list.  If the answer is yes, most of the
   searched coefficients do not matter and the whole design shrinks.
3. **Is the NSI scan a short scan, and does the missing short-scan weighting bias the direct-recon
   score?**  `brainstorm_score.md` notes "the missing short-scan weighting (`cone_beam.py:806-807`)"
   and dismisses it as "the same for every candidate", with the caveat that it biases "through their
   correlation with the basis images".  Nobody asks the question, and an angle-dependent FDK
   systematic correlates with metal-centred basis images by construction.  The page does not mention
   short scan at all.
4. **How are the metal and no-metal scans registered and intensity-matched?**  The no-metal
   comparison is the page's only defensible real reference and appears in experiments 3 and 4.
   `own_view_before_reports.md` says "The difference between the two reconstructions, registered";
   the page drops the word and no report says how.  Without a stated registration and scale
   convention the gate has no defined precision.
5. **Does the estimator have a fixed point inside the alternation?**  Skeptic attack 2's second half
   — Otsu on a cupped FDK assigning the plastic rim to a metal class at `num_metal=2`, and the metal
   class value inherited from the FDK and never corrected because a fixed-mask score is indifferent
   to it — is raised with a minutes-long experiment ("segment the FDK ... at `num_metal` of 1 and 2
   and count the plastic voxels labeled metal, then run three passes ... and record whether the metal
   class mean moves").  The page addresses only the first half (Otsu's objective) and drops the
   experiment entirely.  The page's own plan puts the estimator inside that alternation.
6. **What metal is the synthetic truth?**  The physics report's section 7 recommends tungsten "so that
   the polynomial is never the truth"; the page's own family section says "A K-edge inside the
   spectrum, as tungsten and tin have, breaks the two-basis attenuation form."  Experiment 1 says only
   "metal rods."  If the truth is tungsten and stage 2's model is the two-basis physical family, stage
   2 fails for a reason unrelated to the image score.
7. **How large is a segmentation-threshold error's effect compared with a θ error?**  The page ablates
   "masks eroded and dilated by one pixel", which perturbs the *score* mask.  A threshold error
   changes `p` and `m` themselves and therefore every basis image and every column of `H` —
   `brainstorm_search.md` names this as what a hyperparameter search "cannot repair".  No experiment
   varies the segmentation threshold.
8. **Does the NSI export already carry a vendor correction?**  Both the page and the physics report
   flag the "0.5BH" file name, and the page says "the real-scan gate must say so", but no experiment
   checks it — and experiments 2, 3 and 4 all run on that scan.  This is a minutes-long check that
   gates hours of GPU work.

---

## 20. What survives review

Stated so the repairs above are not read as a rejection.

- **The basis-image identity is correct.**  `R(c(θ)) = R(y) − Σ_j θ_j R(h_j)` holds because `R` is
  linear and the `h_j` are fixed by the segmentation, and it survives any fixed linear operator on
  the sinogram, including a ray mask or weights (the page's linearity claim in the starvation row is
  right, even though it is the wrong answer to that trap).
- **The plastic-scale gauge is sound**, and for a stronger reason than the page gives: every
  nonlinear column vanishes at `m = 0`, so the correction is the identity on metal-free rays and the
  outer plastic rescale is exactly 1.
- **The bounded sub-volume argument is correct and improves on the pipeline report.**  Rays through
  the top window row span `z` in `[h, h·k]` with `k = (sid+r)/(sid−r)`, so a mask sub-volume of
  half-height `h·k` gives the window rows exactly.  `brainstorm_pipeline.md` concluded only that the
  masks must be full-extent and that "the volume traversal is not reduced"; the page's refinement is
  right and is new.  One caveat: `_slab_row_window` widens the window by a voxel footprint plus
  rounding rows (`geometry_calibration.py:133-135`), so `k` should be taken from the returned window,
  not from the ideal geometry.
- **The slab back-projection argument is correct** for FDK: the ramp filter runs along channels, and
  every ray through a slab voxel lands in the window, so cropping rows loses nothing on the
  reconstruction side.
- **The identification of the ridge as a bias rather than a safeguard**, and the conclusion that
  sweeping beta with an image score is worth doing whatever else is built, is well supported by
  `experiments/bh_physics_extra_output.txt` (a) and is the page's most robust product.
