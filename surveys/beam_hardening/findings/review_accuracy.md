# Accuracy review of `surveys/beam_hardening/survey.md`

Reviewer: accuracy seat.  Date 2026-09-05.  Read-only except for this file.

Every number, code citation, report attribution, and literature claim on the page was checked
against its named source.  `experiments/counts_and_binning.py` was rerun in this session with
`/Users/gbuzzard/miniforge3/envs/mbirtorch/bin/python`.  Code was read at
`/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch`, branch `geometric_calibration`,
tip `590fea9` (confirmed).

Thirteen problems, ordered by severity.  Nine are wrong or unsourced numbers or descriptions; four
are inherited looseness worth tightening before the page is quoted onward.

---

## 1. The headline claim about the physical family is attributed to the wrong model (high)

**Page (Answer section):** "The physics report adds a stronger recommendation for the model
itself: a three-to-five parameter physical family fits the presented data 10 to 100 times better
than the cubic and extrapolates to thicker metal about 100 times better
(`experiments/bh_physics_sim_output.txt`)."

**Source checked:** `experiments/bh_physics_sim_output.txt` lines 21-33 (the 200 kV, 0.9 mm Cu block), and
`brainstorm_physics.md` "Answer".

**What the source actually says.**  The 10-to-100x fit and the ~100x extrapolation belong to the
**free shared-bin log-sum-exp**, which has 8 free parameters at J=3 and 11 at J=4 — not to the
three-to-five parameter physical family.  `brainstorm_physics.md` says it plainly: "A log-sum-exp
with three or four shared bins fits 10 to 100 times better and extrapolates 100 times better."
The sim output's small *physical* families are worse than or barely better than the cubic:

| model | free params | rms | vs cubic (rms 0.0088) |
| --- | --- | --- | --- |
| `mar.py` cubic | 6 | 0.0088 | — |
| physical LSE J=3, weights only | 2 | 0.08032 | 9x **worse** |
| physical LSE J=4, weights only | 3 | 0.01901 | 2.2x **worse** |
| physical LSE J=6, weights only | 5 | 0.00643 | 1.4x better |
| free shared-bin LSE J=3 | 8 | 0.00046 | 19x better |
| free shared-bin LSE J=4 | 11 | 0.00004 | 220x better |

The page's own later numbers for the family it means — the one- and two-knob spectrum fits of
`experiments/bh_physics_extra_output.txt` — give 1.4x to 4.9x, not 10 to 100x: 1-knob rms 0.0057 / max 0.0112,
2-knob rms 0.0020 / max 0.0045, against "mar.py cubic on the same truth: rms 0.0077, max |r|
0.0219" (line 11 of that file).  The extrapolation figure is likewise the free LSE's: cubic +0.6942
against free-LSE-J=3 −0.0059 is 118x, while the one- and two-knob extrapolation errors are −0.0144
and +0.0140, roughly 50x.

**Why it matters.**  This sentence is the page's stated reason that "Which family to search is the
first decision for Greg," and it also contradicts the page's own §"The family to search," which
quotes 0.011 and 0.0045 for that family.

**Suggested correction.**  Split the claim: "a free shared-bin log-sum-exp with 3 or 4 bins (8 or
11 parameters) fits the presented data 10 to 100 times better than the cubic and extrapolates about
100 times better; a three-to-five parameter physical family fits 1.4 to 5 times better than the
cubic and extrapolates roughly 50 times better."

---

## 2. `m_k` is not a path-length sinogram (high, code)

**Page:** "`correct_sino_plastic_metal` (`mbirtorch/preprocess/mar.py`) segments a reconstruction
into one plastic class and `num_metal` metal classes, forward projects each class to get the
path-length sinograms p and m_k …"

**Source checked:** `mbirtorch/preprocess/mar.py:255-283` (`_est_plastic_metal_sinos_from_recon`).

**What the code actually does.**  The two classes are projected differently:

- `mar.py:271-272`: `plastic_sino_est = plastic_scale * forward_project(plastic_mask)` — a scaled
  path length, so "path-length sinogram" is fair for `p`.
- `mar.py:277-280`: the metal branch projects `mask * recon`, not `mask`:
  `masked = mask * recon`, then `m = ct_model.forward_project(masked, ...)`.

So `m_k` is the forward projection of the *masked reconstruction values*, not a geometric path
length.  `brainstorm_skeptic.md` attack 2 relies on exactly this distinction ("the re-added metal
is `FP(mask * recon)` at whatever value the reconstruction holds"), and the page itself repeats the
consequence ("the metal's level is a choice that the existing add-back makes").

**Knock-on.**  The page's §"Cost at production scale" closes with "one label volume at 8 GB if the
projector could take a label and a class value."  That substitution is exact only for the plastic
column; the metal columns as written need the reconstruction's own values inside the mask.  (The
same simplification is in `brainstorm_search.md` and `brainstorm_pipeline.md` §3, so the page
inherits it rather than introducing it — but it should be stated as a change to the correction, not
a memory optimization.)

**Suggested correction.**  "forward projects the plastic mask and each metal class's masked
reconstruction to get the sinograms p and m_k (p a scaled path length, m_k the projection of the
metal's own reconstructed values, `mar.py:271-281`)."

---

## 3. "Sixteen basis images … 128 MB" is in neither cited source (medium-high)

**Page:** "Sixteen basis images of an 8-slice slab at 500 by 500 occupy 128 MB
(`brainstorm_pipeline.md`, section 2)."

**Source checked:** `brainstorm_pipeline.md` §2; `brainstorm_search.md` §(a).

**What the sources say.**  Pipeline §2 gives 8 MB per image (`500 * 500 * 8 * 4 = 8.0e6`) and three
totals: "32 MB for one metal at order 3, 80 MB for two metals at order 3, 280 MB for three metals
at order 4."  Neither "sixteen" nor "128 MB" appears.  `brainstorm_search.md` §(a) gives the
nearest statement and a different one: "Fifteen slab images occupy 120 MB."

The per-image figure the page used (8 MB) is right, and 16 is reconstructible as `1 + N_col` for two
metals at order 3 (1 + 15), which is search's counting rule.  But as written the number is
attributed to a source that gives 80 MB for the analogous case, and it disagrees with the other
report by one image.

**Suggested correction.**  Either cite `brainstorm_search.md` §(a) and use its "fifteen … 120 MB",
or state the derivation on the page: "one reconstruction of the measured sinogram plus one per
searched column — 16 images at two metals and order 3 — at 8 MB each
(`brainstorm_pipeline.md`, section 2, for the 8 MB)."

---

## 4. The 86 MB column drops two of its three reduction conditions (medium)

**Page:** "A row window at view stride 4 brings a column to 86 MB."

**Source checked:** `brainstorm_pipeline.md` §2.

**What the source says.**  "With the row window it is `450 * 48 * 1000 * 4 = 8.6e7` bytes, 86 MB."
That is view stride 4 (450 views) **and detector bin 2** (1000 channels, and rows binned 2:1)
**and** a 48-binned-row window.  The immediately preceding sentence in pipeline §2 gives the
stride-4/bin-2 whole-extent figure — 1.8 GB — precisely to show that the stride and the binning
alone are not enough.

At `bin_factor=1`, which is the reduction the page's own §"Where the geometry recipe transfers"
argues for (build columns at full resolution, bin afterward), the same window is about four times
larger.

**Suggested correction.**  "A 48-row window at view stride 4 and detector bin 2 brings a column to
86 MB."

---

## 5. The searched coefficient count contradicts the page's own gauge rule (medium, internal)

**Page (Answer):** "A search over 6 to 15 coefficients then costs no projector call per candidate…"
**Page (§The idea that makes a search tractable):** "The linear columns are never searched: the
plastic's level is the data's own, and the metal's level is a choice that the existing add-back
makes (`mar.py:818-824`)…"

**Source checked:** the page's own column-count table; `brainstorm_search.md` §(a);
`brainstorm_score.md` §Multiple materials; `mar.py:764-779`.

**The conflict.**  6 and 15 are the *total* column counts for one and two metals at order 3 (the
page's table, confirmed by the rerun of `experiments/counts_and_binning.py`).  If the linear columns are held
fixed, the searched dimension is smaller.  The reports do not agree on how much smaller:
`brainstorm_search.md` §(a) says "5 for one metal and 13 for two at order 3" (dropping one linear
column for K=1 and two for K=2 — itself inconsistent); `brainstorm_score.md` says "five shape
coefficients … for two there are fourteen."  Counting from `mar.py:764-779`, the linear columns are
`p` and each `m_k`, giving 4 searched at K=1 and 12 at K=2.

**Suggested correction.**  Say what "6 to 15" counts: "A search over the 4 to 12 nonlinear columns
of a 6- to 15-column model then costs no projector call per candidate," and note in §"three rules"
that the reports quote 5/13 and 5/14 for the same quantity.

---

## 6. The noise trap is stated in one direction only; the skeptic reports the opposite one (medium)

**Page (§The score, and again in the trap table):** "The noise floor is measured by scoring the
difference between the even-view and odd-view reconstructions, because a stronger correction
amplifies long-path noise, so a variance score prefers weak corrections for a reason unrelated to
hardening."  Trap row: "weak-correction preference | a stronger correction amplifies long-path
noise".

**Source checked:** `brainstorm_score.md` §"What makes a score safe to search on" (which the page
transcribes correctly) and `brainstorm_skeptic.md` attack 1, third route.

**What the skeptic says.**  The reverse: "Every corrected candidate has one more denoising step
than the measured sinogram: the metal rays are partly replaced by `FP(mask * recon)`, which is
smooth and noise free (`mar.py:277-281`), and the clamp deletes negative noise.  A noise-sensitive
score therefore prefers more correction on denoising grounds."  The two reports name opposite
biases from the same noise floor.  The page's trap table carries only routes 1 and 2 of attack 1
(scale, ray removal) and not route 3, so the reader is left with one signed prediction where the
sources give two.

**Suggested correction.**  Add the second direction to the trap row and to the safety rule: "the
noise floor must bound both directions — a stronger correction amplifies long-path noise
(`brainstorm_score.md`) while also denoising the metal rays it replaces (`brainstorm_skeptic.md`,
attack 1)."

---

## 7. "The reports rank them cheapest last" reverses the reports' own ranking (medium)

**Page (Open questions):** "Is the deliverable an estimator of the coefficients, an automation of
the fit's hyperparameters, or a warning check inside `recon_plastic_metal`?  The reports rank them
cheapest last, and each is a different amount of work."

**Source checked:** `brainstorm_pipeline.md` §4; `brainstorm_search.md` §Ranked recommendation and
§(c).

**What the sources say.**  Pipeline §4 calls the *middle* option — "Choose the hyperparameters" —
"the cheapest path, and it is the one the sweep-do-not-guess rule points at," and calls the third,
"Serve as a check," "the smallest change."  Search ranks the `beta` search first and calls it "the
cheapest useful step" and "the cheapest first step."  So the reports' cheapest is the page's second
item, not its last.

**Suggested correction.**  "The reports rank the hyperparameter automation cheapest and the warning
check smallest, with the coefficient estimator the most work."

---

## 8. The physical-family parameter count is quoted with one knob but the range needs two (low)

**Page:** "With the voltage from the scan file, the unknowns are one filter-equivalent thickness,
one density scale per material, and optionally one scatter constant: 3 to 5 for one metal…
(`brainstorm_physics.md`, section 1)."

**Source:** `brainstorm_physics.md` §1: "the physical count for K=1 is **1 to 2 spectrum knobs**, 2
density scales, and optionally 1 scatter constant: 3 to 5 parameters."

One knob plus two density scales plus an optional scatter constant is 3 or 4, not 3 to 5.  The 5
requires the second knob — the voltage — which the page itself then uses two sentences later ("a
two-knob fit 0.0045").

**Suggested correction.**  "one or two spectrum knobs (filter-equivalent thickness, and optionally
the voltage), one density scale per material, and optionally one scatter constant: 3 to 5."

---

## 9. `mar.py:492-629` names the active-set loop but spans two functions (low, code)

**Page:** "The existing ridge weights and the active-set positivity loop (`mar.py:492-629`) apply
unchanged…"

**Source checked:** `mar.py`.  `_compute_entry_for_OSQP` is 492-532 (it builds `HtH`, `Hty`, and
the ridge `weight_matrix` at 515-529); `_estimate_BH_model_params`, the active-set loop, is 534-629.
`brainstorm_search.md` cites them separately and correctly ("`mar.py:492-532`" and "`mar.py:534-629`").

The page's single range does cover both features it names in the sentence, so this is imprecision
rather than an error, but a reader following the citation to find the active-set loop lands 42
lines early.

**Suggested correction.**  "The existing ridge weights (`mar.py:515-529`) and the active-set
positivity loop (`mar.py:534-629`)…"

---

## 10. The "one part in a thousand" gradient ratio inherits an unsupported condition (low)

**Page:** "…gradient energy reads the cupping of a disk at under one part in a thousand of its edge
term (`brainstorm_score.md`)."

**Source:** `brainstorm_score.md`: "The ratio is about `0.007 sigma / R`, below one part in a
thousand at any blur narrower than the object."

The page transcribes the report faithfully, so this is not a transcription error.  But the report's
own formula gives below 1e-3 only for `sigma/R < 0.14`; "any blur narrower than the object" would
admit `sigma` up to `2R`, where the ratio is 1.4 percent, fourteen times the quoted bound.

**Suggested correction.**  Quote the formula instead of the bound: "gradient energy reads the
cupping of a disk at about `0.007 sigma/R` of its edge term, under one part in a thousand for blurs
below about a seventh of the radius."

---

## 11. "leaves the reduced fit exact" is stronger than the source (low)

**Page:** "…the remedy is to build each column at full resolution inside the view-batch loop and
bin afterward, which leaves the reduced fit exact (`brainstorm_pipeline.md`, section 1)."

**Source:** pipeline §1: "binning each column after it is built leaves the least-squares problem in
theta a **correctly binned linear problem**; building the columns from binned `p` and `m` does not."

That is exactness *of the binned problem*, not equality with the unbinned fit — the binned problem
still has fewer, averaged rows.  (The "inside the view-batch loop" phrasing comes from
`brainstorm_score.md` §"Which scores are quadratic", not from pipeline §1; worth citing both.)

**Suggested correction.**  "…which leaves the reduced problem a correctly binned linear least
squares in θ (`brainstorm_pipeline.md`, section 1; `brainstorm_score.md`)."

---

## 12. The ridge probe quotes only the worse of the two probe arms (low)

**Page:** "On a probe ray whose plastic estimate is 10 percent low, which a streak through the
plastic mask produces, the corrected plastic errs by +4.4 percent at beta 2e-4, +6.7 percent at the
default 2e-3, and +15.4 percent at 2e-2…"

**Source:** `experiments/bh_physics_extra_output.txt` lines 1-5.  The three numbers are exact, and the
condition "plastic estimate 10 percent low" matches the file's `p=0.9x`.  Two conditions are
dropped: the probe is also at `m=0.5 of max`, and the file reports a second arm at `p=1.1x` with
much smaller errors (−1.2, +0.7, +8.4 percent).  The page's conclusion — that the ridge strength is
a bias worth sweeping — survives either arm, but the quoted magnitudes are the worse half.

**Suggested correction.**  Add the conditions: "…at half the maximum metal path.  The opposite
probe, 10 percent high, errs by −1.2, +0.7, and +8.4 percent."

---

## 13. `estimate_geometry_from_recon` does not exist at the cited commit (low, code)

**Page:** "Only then the estimator.  A function in `mar.py` in the style of
`estimate_geometry_from_recon`…"

**Source checked:** `grep -rn estimate_geometry_from_recon` over the checkout returns nothing at
`590fea9`.  The name is the function under construction, defined in
`mbirtorch_plans/plans/geometric_calibration/estimate_by_recon_plan.md:110`.

Given the page's opening ("Code citations are to the `geometric_calibration` branch of mbirtorch at
commit `590fea9`"), a reader will look for it in the checkout.

**Suggested correction.**  "…in the style of the planned `estimate_geometry_from_recon`
(`plans/geometric_calibration/estimate_by_recon_plan.md`)."

---

## Verified correct

Numbers traced to their source and transcribed correctly:

- **Column-count table** (1/2/3 metals × orders 2/3/4: 4, 6, 8 / 8, 15, 24 / 13, 29, 54) — exact
  match to the rerun of `experiments/counts_and_binning.py`, and the script's rule matches
  `_generate_metal_exponent_list` (`mar.py:206-235`) and the column assembly at `mar.py:764-779`.
- **Binning bias**: "under 0.01 percent for a disk of radius 800 pixels at bin 4, and 0.23 percent
  at bin 2 and 1.09 percent at bin 4 for a disk of radius 50 pixels" — the rerun prints −0.01,
  −0.23, −1.09 percent.  The page is *more* precise than `brainstorm_search.md` (0.2 / 1.1); the
  page's figures are the right ones.  The description of the condition (a quadratic model on a
  disk, columns built from binned p and m) is right.
- **Production sizes**: 28.8 GB per sinogram at 1800 × 2000 × 2000, 86.4 GB for measured + p + m at
  one metal, 27 columns rebuilt two at a time, 32 GB per 2K float mask, 8 GB for a label volume —
  all match `brainstorm_pipeline.md` §2/§3 and the script; the 27 was re-derived from the double
  loop at `mar.py:501-513` (6 + 21).
- **Slab overreach**: 5.7 degree half fan (confirmed at
  `mbirtorch_plans/plans/geometric_calibration/experiments/rotation_zero_point_synthetic.md:63`,
  sid 400 / sdd 800), 0.88 slices per end for an 8-slice slab, 2.8 slices at a 15 degree half fan,
  and the widening factor (sid+r)/(sid−r) — all match `brainstorm_pipeline.md` §1 and recompute.
- **Every physics number except the one in problem 1**: rms 0.009 (sim 0.0088) and max 0.026 with
  attenuation reaching 4.35; three-bin LSE max 0.0018 and four-bin 0.00014; cubic extrapolation
  +0.69 / +6.3 / +8.3 at 200 / 150 / 100 kV and the three-bin form's −0.006; the metal cubic's
  inflection at 0.85 cm (recomputed from the sim's own θ: −2·2.8095 + 6·1.0967·m = 0 → m = 0.854);
  one-knob 0.011 and two-knob 0.0045; PMMA linear to one percent over 10 cm (sim ratio 0.989);
  13 to 15 percent fall over 8 cm at 150 and 100 kV (0.865, 0.846); +4.4 / +6.7 / +15.4 percent
  ridge probe; exact inversion having no error.
- **Score-report derivations**: 13 percent center dip under the harness's ten-percent sizing
  ((8/π)(0.05) = 0.127), the plastic mean shifting by two thirds of it (mean of √(R²−r²) over a
  disk is ⅔ of its center value), the low-frequency primary score, the signed contrasts being
  linear with a root rather than a minimum, the Gram condition number as the identifiability
  diagnostic, the central-plane slice choice inverting the geometry estimator's, and the five
  safety rules (all five present and matching).
- **Code, verified line by line**: `mar.py:764-779` (plastic enters only linearly);
  `mar.py:632-722` the division form, with the clamp at `682` and the gamma floor at `703`/`715-718`;
  `mar.py:818-824` the linear metal add-back; `mar.py:724-739` the rescale on metal-free rays;
  `recon_plastic_metal` alternating correction (`tomography_model.py:546`) and MBIR (`:557`) for
  `num_BH_iterations` passes after one FDK (`:532`); `segmentation.py:222-226` documenting Otsu's
  within-class-variance criterion; `cone_beam.py:161` `_cone_forward_view_batch`;
  `BH_correction` reaching `pipeline.map_view_batches` at `mar.py:202-203`;
  `reduce_sinogram` selecting views without averaging (`geometry_calibration.py:418`) and averaging
  bins (`:446-447`); `_search_minimum` (`:1085-1135`) with its coarse grid, golden-section polish,
  and notes for edge minima and multiple local minima; `CalibrationResult` (`:69-88`);
  `build_reduced_problem`'s docstring on rays crossing material outside the slab (`:220-225`);
  `tests/generate_preprocess_goldens.py:166-169` building a two-level phantom (0.02 / 0.2) and
  forward projecting it linearly.
- **Report attributions**, each confirmed against the cited report and section: skeptic attacks 1,
  2, 3, 5, 11 (identity off the metal, the clamp-as-inpainting route, Otsu self-fulfilment, the
  450-views/500-channels aliasing confound and its stride-1-vs-4 check, the unmeasured baseline,
  the one-insert NSI phantom and thin-metal bga); the three skeptic experiments with their costs
  (CPU under an hour; one GPU about an hour; one GPU a few hours) and all three kill criteria;
  pipeline §1, §2, §3, §5, §7; search §(a), (b), (c), (e), (f) including the first-order agreement
  of the two correction forms, the K-edge caveat (tungsten and tin yes; steel, aluminum, titanium
  no), and the two-metal cross columns needing a slab that holds both metals; physics §1, §3, §4,
  §7 including the "0.5BH" file name in the scan directory and the 200 kV / 0.9 mm Cu setting.
- **Every literature claim**, against `brainstorm_literature.md`: Levi et al. 2021 (three
  coefficients robust, seven "worse than uncorrected FBP … on noisy images"); Kyriakou et al. 2010
  as EBHC and Levi 2019/2021 as its automated descendants; question 2 (ABHC-3 the only
  image-criterion method at three materials, and methods that succeed with several materials
  importing physics to pin most coefficients); question 3 (spectrum estimation from transmission
  data ill-conditioned, Sidky et al. 2005).  Authors, years, and findings all match.
- **The geometry-estimator blur claim** ("there the blur matched the resampling count"), which is
  uncited on the page but is supported by
  `mbirtorch_plans/plans/geometric_calibration/estimate_by_recon.md:76-78`: "every
  comparison is made at a matched resampling count, which the blur enforces."

Not verifiable in this session, and correctly attributed rather than asserted: the NSI scan
settings marked "(file)" in `brainstorm_physics.md` (200 kV, 0.9 mm Cu, 6 frames, the "0.5BH" file
name).  The page attributes each of these to the physics report rather than claiming them
first-hand, which is the right handling.

**Every number on the page traced to a source.**  One (128 MB, problem 3) traces only to a
derivation the page does not show, using a per-image figure from the cited report; one (86 MB,
problem 4) traces to the cited report but under conditions the page omits.
