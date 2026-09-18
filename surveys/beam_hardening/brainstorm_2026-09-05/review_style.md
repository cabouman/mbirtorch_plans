# Style review: `surveys/beam_hardening/survey.md`

Reviewer: style. Checked against `mbirtorch_plans/.claude/writing_style.md` and the
"When communicating" section of `mbirtorch_plans/.claude/claude_prompt.md`.

The page contains no em-dash asides, which is the one warning sign it never trips. The
same asides appear instead as colons, semicolons, and comma-set clauses, and those are
where most entries below fall. Entries are ordered by section. Each gives the quoted
sentence, the rule it breaks, and a rewrite.

---

## Preamble (lines 3-10)

**1.** "This page synthesizes six agent reports written in this session, which sit beside
it: `brainstorm_score.md`, ..."

Rule: metaphor or idiom in a technical statement. "Sit beside it" is the guide's own named
failure, which uses "sitting next door" for a file in the same repository.

Rewrite: "This page synthesizes six agent reports written in this session. All six are in
this directory: `brainstorm_score.md`, `brainstorm_search.md`, `brainstorm_physics.md`,
`brainstorm_skeptic.md`, `brainstorm_literature.md`, and `brainstorm_pipeline.md`."

**2.** "Every number below was read in this session from one of those reports or from the
output of a script beside them (`counts_and_binning.py`, `bh_physics_sim.py`,
`bh_physics_extra.py`), and the source is named beside it."

Rule: parenthetical list inside a claim; 32 words; and "beside" carries two different
meanings in one sentence (in the same directory, and adjacent in the text).

Rewrite: "Every number below was read in this session, either from one of those reports or
from the output of a script. The scripts are `counts_and_binning.py`, `bh_physics_sim.py`,
and `bh_physics_extra.py`, and they are in this directory. Each number names its source."

---

## Answer

**3.** "Estimating beam-hardening parameters from reconstruction quality is viable, but not
as a transplant of the geometry estimator."

Rule: metaphor in a technical statement ("transplant"); undefined referent for a reader who
has not read the source reports ("the geometry estimator").

Rewrite: "Estimating beam-hardening parameters from reconstruction quality is viable. It
cannot reuse the geometry estimator unchanged. That estimator,
`estimate_geometry_from_recon`, searches geometry parameters by scoring the reconstruction
each candidate produces."

**4.** "Two facts decide the shape of the work."

Rule: word used loosely. Facts do not decide, and "the shape of the work" is vague.

Rewrite: "Two facts determine what the method can and cannot do."

**5.** "The first is favorable: a direct reconstruction is linear in the sinogram, so if the
correction is written as the sinogram minus a fitted nonlinear excess, the reconstruction of
every candidate coefficient vector is a linear combination of basis images that are computed
once."

Rule: colon chaining two complete clauses; 44 words; three ideas.

Rewrite: "The first fact is favorable. A direct reconstruction is linear in the sinogram.
Write the correction as the sinogram minus a fitted nonlinear excess. Then the
reconstruction of any candidate coefficient vector is a linear combination of a fixed set of
images, computed once."

**6.** "A search over 6 to 15 coefficients then costs no projector call per candidate, and a
score that is quadratic in the image makes the estimate one small convex problem."

Rule: two ideas in one 30-word sentence; the count "6 to 15" arrives before the reader knows
what sets it.

Rewrite: "A search over the coefficients then needs no projector call per candidate. If the
score is quadratic in the image, the estimate is one small convex problem. The polynomial
model has 6 to 15 coefficients, depending on the order and the number of metals."

**7.** "The second fact is unfavorable, and it is the difference from geometry: a wrong
geometry can only add artifacts, so a quality minimum is the truth, but a hardening
correction can remove real signal, and every flatness score rewards removal."

Rule: colon chaining two complete clauses; 40 words; four ideas.

Rewrite: "The second fact is unfavorable, and it is where hardening differs from geometry. A
wrong geometry can only add artifacts, so the best-scoring geometry is the correct one. A
hardening correction can instead remove real signal. Any score that rewards a flat image
also rewards that removal."

**8.** "The image score moves only the few directions it can see, which are the artifact
patterns in the plastic, and it reports undecided when the object carries no signal for a
direction."

Rule: aside inside a claim; two ideas; 32 words; "see" is a metaphor.

Rewrite: "The image score sets only the coefficient directions to which the reconstruction is
sensitive. Those directions are the artifact patterns in the plastic. When the object gives
no sensitivity for a direction, the estimator reports that direction as undecided."

**9.** "Two cheaper products come first, because neither has been measured: a synthetic case
with real polychromatic hardening, which the repository lacks, and a sweep of the existing
fit's ridge strength scored on the image, which automates what a user tunes by eye today."

Rule: 43 words; a two-item list with a qualifier embedded in each item.

Rewrite: "Two cheaper deliverables come first, because neither has been measured:
- a synthetic case with polychromatic hardening, which the repository does not have today;
- a sweep of the existing fit's ridge strength, scored on the image, which replaces the
  tuning a user does by eye today."

**10.** "The physics report adds a stronger recommendation for the model itself: a
three-to-five parameter physical family fits the presented data 10 to 100 times better than
the cubic and extrapolates to thicker metal about 100 times better
(`bh_physics_sim_output.txt`)."

Rule: colon chaining two complete clauses; 37 words; "the presented data" is undefined
jargon and "the cubic" is an undefined referent at first use.

Rewrite: "`brainstorm_physics.md` recommends changing the model itself. The existing model is
a cubic polynomial in the plastic and metal path lengths. A physical family with three to
five parameters fits the simulated attenuation 10 to 100 times more accurately than the
cubic. It also extrapolates to thicker metal about 100 times more accurately
(`bh_physics_sim_output.txt`)."

**11.** Second paragraph of the Answer as a whole ("The design that follows from both facts
is a hybrid." through "Which family to search is the first decision for Greg.").

Rule: a paragraph carrying more than one idea. It holds three: the hybrid design, the two
cheaper deliverables, and the physics recommendation on the model family.

Rewrite: split into three paragraphs, each opening with its own topic sentence. Keep the
hybrid design first, the cheaper deliverables second, and the model-family recommendation
third, ending on the decision asked of Greg.

---

## What the existing code does, and three facts that shape the design

**12.** Section opening.

Rule: a section that does not open with its big idea. The section opens with two sentences
of mechanism, and its big idea, "Three facts about this code shape everything below,"
arrives fourth.

Rewrite: open with "Three facts about the existing correction shape everything below," then
give the mechanism, then the three facts.

**13.** "`correct_sino_plastic_metal` (`mbirtorch/preprocess/mar.py`) segments a
reconstruction into one plastic class and `num_metal` metal classes, forward projects each
class to get the path-length sinograms p and m_k, and fits the measured sinogram y to
monomial columns in (p, m) by ridge least squares with positivity constraints."

Rule: 44 words; three ideas.

Rewrite: "`correct_sino_plastic_metal` (`mbirtorch/preprocess/mar.py`) segments a
reconstruction into one plastic class and `num_metal` metal classes. It forward projects
each class to get the path-length sinograms p and m_k. It then fits the measured sinogram y
to monomial columns in p and m by ridge least squares with positivity constraints."

**14.** "The corrected plastic sinogram is y minus the metal-only terms, divided by the
coefficient of p, rescaled on metal-free rays, and the metal sinograms are added back
linearly."

Rule: a list of operations given no sentence of its own, plus a second idea after the last
comma.

Rewrite: "The corrected plastic sinogram is built in three steps: the metal-only terms are
subtracted from y, the result is divided by the coefficient of p, and it is rescaled on
metal-free rays. The metal sinograms are then added back linearly."

**15.** "At the NSI scans' setting of 200 kV behind 0.9 mm of copper this costs nothing,
because a PMMA-like plastic is linear to one percent over 10 cm; at 100 to 150 kV with light
filtration the plastic's attenuation per centimeter falls 13 to 15 percent over 8 cm and no
coefficient can correct it (`brainstorm_physics.md`, section 3)."

Rule: 55 words; a semicolon chaining two complete clauses; evidence and interpretation in
the same sentence.

Rewrite: "At the NSI scans' setting of 200 kV behind 0.9 mm of copper, a PMMA-like plastic is
linear to one percent over 10 cm. The missing column therefore costs nothing at that
setting. At 100 to 150 kV with light filtration, the plastic's attenuation per centimeter
falls 13 to 15 percent over 8 cm. No coefficient in the present model can correct that fall
(`brainstorm_physics.md`, section 3)."

**16.** "The fit's memory is the production-scale problem, before any image score enters."

Rule: a qualifier embedded rather than given its own sentence; "enters" used loosely of a
score.

Rewrite: "The fit's memory is the production-scale problem. This is true of the existing fit
alone, before any image score is added."

**17.** "The columns are full-size sinograms, 28.8 GB each at 1800 by 2000 by 2000, and the
fit holds the measured sinogram, p, and each m_k together, 86.4 GB for one metal, while its
normal equations rebuild 27 columns two at a time (`brainstorm_pipeline.md`, section 2)."

Rule: 42 words; three ideas; two appositive asides inside the claim.

Rewrite: "The columns are full-size sinograms. At 1800 by 2000 by 2000 each column is 28.8
GB. The fit holds the measured sinogram, p, and each m_k at once, which is 86.4 GB for one
metal. Its normal equations rebuild 27 columns two at a time (`brainstorm_pipeline.md`,
section 2)."

**18.** "A row window at view stride 4 brings a column to 86 MB. So the reduced problem is
worth having for the existing fit alone."

Rule: undefined terms at first use. "Row window" and "reduced problem" are both used here
and defined two sections later.

Rewrite: "Restricting the fit to a window of detector rows and to every fourth view brings a
column to 86 MB. Call that restriction the reduced problem. It is worth having for the
existing fit alone."

**19.** "The golden MAR sinogram is a linear forward projection of a two-level phantom with
no hardening (`tests/generate_preprocess_goldens.py:166-169`), so the tests establish parity
with mbirjax and say nothing about correction quality (`brainstorm_skeptic.md`, attack 11)."

Rule: a sentence carrying both evidence and interpretation.

Rewrite: "The golden MAR sinogram is a linear forward projection of a two-level phantom with
no hardening (`tests/generate_preprocess_goldens.py:166-169`). These tests therefore
establish parity with mbirjax and say nothing about correction quality
(`brainstorm_skeptic.md`, attack 11)."

---

## Where the geometry recipe transfers, and where it does not

**20.** "The one-dimensional search of `_search_minimum`, with its coarse grid, its
golden-section polish, and its notes about edges and multiple minima, serves any scalar
hyperparameter."

Rule: a list embedded as an aside inside a claim.

Rewrite: "The one-dimensional search of `_search_minimum` serves any scalar hyperparameter.
It brings three parts with it: a coarse grid, a golden-section refinement, and its notes
about edges and multiple minima."

**21.** "The undecided verdict, with per-slice agreement and a noise floor from the even-view
and odd-view split, transfers with more force than in geometry, because hardening parameters
are set by the spectrum and the materials and are the same in every slice and region while
structure is not (`brainstorm_score.md`)."

Rule: 47 words; an aside inside the claim; "with more force" used loosely.

Rewrite: "The undecided verdict transfers, and it applies more widely here than in geometry.
It brings two parts with it: agreement across slices, and a noise floor from the even-view
and odd-view split. Hardening parameters are set by the spectrum and the materials, so they
are the same in every slice and region. Structure is not, so disagreement across slices is
strong evidence against a candidate (`brainstorm_score.md`)."

**22.** "For a round uniform object the hardened sinogram is still the projection of some
image, a cupped one, so the reprojection residual cannot see cupping at all, and gradient
energy reads the cupping of a disk at under one part in a thousand of its edge term
(`brainstorm_score.md`)."

Rule: 47 words; three ideas; "cannot see" and "reads" are metaphors; evidence and
interpretation in one sentence.

Rewrite: "For a round uniform object, the hardened sinogram is still the projection of some
image, namely a cupped one. The reprojection residual is therefore insensitive to cupping.
Gradient energy is nearly insensitive to it too: for a disk, the cupping term is under one
part in a thousand of the edge term (`brainstorm_score.md`)."

**23.** "The slab does not."

Rule: undefined jargon at first use. "Slab" has not been introduced on this page.

Rewrite: "The slab does not transfer. In the geometry estimator, a slab is the small block of
adjacent detector rows and their reconstructed slices on which the search runs."

**24.** "On the synthetic geometry of the rotation experiments, with a 5.7 degree half fan,
an 8-slice slab's rays reach 0.88 slices past each end; at a 15 degree half fan they reach
2.8 slices (`brainstorm_pipeline.md`, section 1)."

Rule: semicolon chaining two complete clauses; a qualifier embedded in the first clause;
"the rotation experiments" is an undefined referent.

Rewrite: "The geometry of the rotation experiments has a 5.7 degree half fan. There an
8-slice slab's rays reach 0.88 slices past each end of the slab. At a 15 degree half fan
they reach 2.8 slices (`brainstorm_pipeline.md`, section 1)."

**25.** "Thickening the slab widens the row window by the factor (sid + r) / (sid − r), which
exceeds one, so no finite slab is self-contained."

Rule: a needed qualifier embedded rather than given its own sentence; evidence and
interpretation in one sentence.

Rewrite: "Thickening the slab widens the row window by the factor (sid + r) / (sid − r).
That factor always exceeds one. No finite slab is therefore self-contained."

**26.** "Those projections can still be cheap: the rays through the window reach only that
bounded distance past the slab, so a mask sub-volume of the slab's height times that factor,
projected through a model copy whose detector is cropped to the window, gives the window
rows exactly."

Rule: colon chaining two complete clauses; 47 words; an aside inside the claim.

Rewrite: "Those projections can still be cheap. The rays through the window reach only the
bounded distance computed above. It is therefore enough to project a mask sub-volume whose
height is the slab's height times that factor. Project it through a model copy whose
detector is cropped to the window, and the window rows are exact."

**27.** "Binning is linear, so it is exact for the measured sinogram, but the mean of m
squared over a bin exceeds the square of the mean by the within-bin variance, and at a metal
edge that variance is large."

Rule: 39 words; three ideas.

Rewrite: "Binning is linear, so it is exact for the measured sinogram. The nonlinear columns
are different: the mean of m squared over a bin exceeds the square of the mean by the
within-bin variance. At a metal edge that variance is large."

**28.** "The size of the bias when columns are instead built from binned p and m was computed
for a quadratic model on a disk: under 0.01 percent for a disk of radius 800 pixels at bin
4, and 0.23 percent at bin 2 and 1.09 percent at bin 4 for a disk of radius 50 pixels
(`counts_and_binning.py`, rerun in this session)."

Rule: 56 words; four numbers that must be matched to two radii and two bin factors while
reading.

Rewrite: "The size of that bias was computed for a quadratic model on a disk
(`counts_and_binning.py`, rerun in this session). For a disk of radius 800 pixels it stays
under 0.01 percent at bin 4. For a disk of radius 50 pixels it is 0.23 percent at bin 2 and
1.09 percent at bin 4." The per-radius numbers could also move to
`counts_and_binning.py`'s companion notes, leaving the actionable statement here.

**29.** "Metal features are the small ones."

Rule: a compressed sentence whose point is not stated literally.

Rewrite: "Metal features are small, so the larger bias is the one that applies to them."

---

## The idea that makes a search tractable: basis images

**30.** Paragraph opening: "Write the correction as the measured sinogram minus the fitted
nonlinear excess, c(θ) = y − Σ_j θ_j h_j over the nonlinear columns h_j, with the linear
columns left alone."

Rule: a paragraph whose first sentence is not its topic sentence. The idea of the paragraph
is that every candidate is free of projector calls, and that arrives last.

Rewrite: open with "Because the reconstruction is linear, every candidate coefficient vector
can be scored without a projector call," then give the construction.

**31.** "Every candidate θ afterward is a linear combination of small images, so a grid, a
polish, or a closed-form solve are all free of projector calls."

Rule: a list embedded in a claim; "a polish" used as a noun, which is loose.

Rewrite: "Every candidate θ afterward is a linear combination of small images. Three search
methods are therefore free of projector calls: a grid search, a golden-section refinement,
and a closed-form solve."

**32.** "This is the construction of empirical beam-hardening correction (Kyriakou et al.
2010) and of its automated descendants (Levi et al. 2019, 2021), placed on mbirtorch's
reduced problem (`brainstorm_literature.md`)."

Rule: metaphor ("descendants"); "placed on" used loosely.

Rewrite: "This is the construction used by empirical beam-hardening correction (Kyriakou et
al. 2010) and by the later automated methods that build on it (Levi et al. 2019, 2021),
applied here to mbirtorch's reduced problem (`brainstorm_literature.md`)."

**33.** "    S(θ) + λ · ‖Hθ − y‖² on the reduced sinogram + the existing ridge term"

Rule: a term used for one object in one place and a different term elsewhere, inside a
formula. English phrases ("on the reduced sinogram", "the existing ridge term") appear as
addends, so the reader cannot tell which symbols the formula actually contains.

Rewrite: write the objective with symbols only, and name the terms in the sentence that
follows: "S(θ) + λ‖Hθ − y‖² + β‖θ‖², where H and y are restricted to the reduced sinogram,
λ weights the data term, and β is the existing ridge strength."

**34.** "The sinogram term is what answers the skeptic and the literature."

Rule: undefined referent and loose usage. "The skeptic" is a report file, not a person, and
"answers" is doing work the sentence never states.

Rewrite: "The sinogram term is what addresses the objection raised in
`brainstorm_skeptic.md` and the failure reported in `brainstorm_literature.md`: it prevents
the image score from moving coefficients the image cannot resolve."

**35.** "Directions of θ whose basis images are nearly collinear on the scored region, or
that lie in the null space of Q, are pinned by the data term and the ridge rather than
wandering with noise."

Rule: 36 words; an aside inside the claim; "wandering with noise" is a metaphor; "pinned"
is used loosely throughout the page.

Rewrite: "Two kinds of direction in θ are invisible to the score: those whose basis images
are nearly collinear on the scored region, and those in the null space of Q. The data term
and the ridge fix those directions, so noise cannot move them."

**36.** "The condition number of the masked Gram matrix of the basis images is the
identifiability diagnostic, and a large value means the object carries no signal for some
coefficient, whose verdict is undecided."

Rule: three ideas in one sentence; the closing relative clause is a separate claim.

Rewrite: "The condition number of the masked Gram matrix of the basis images is the
identifiability diagnostic. A large value means the object gives no sensitivity for some
coefficient. That coefficient's verdict is undecided."

**37.** "Three rules keep the construction honest."

Rule: idiom in a technical statement ("keep honest"), and personification of a construction.

Rewrite: "Three rules prevent the search from lowering the score by removing real signal."

**38.** "The linear columns are never searched: the plastic's level is the data's own, and
the metal's level is a choice that the existing add-back makes (`mar.py:818-824`), so the
overall scale is fixed by construction."

Rule: colon chaining two complete clauses; 38 words; three ideas.

Rewrite: "The linear columns are never searched. The plastic's level comes from the data
itself, and the metal's level is set by the existing add-back step (`mar.py:818-824`). The
overall image scale is therefore fixed, and the search cannot change it."

**39.** "The two agree to first order in the excess and differ at long metal paths; the
subtraction form drops the clamp at zero and the floor, and negative corrected values are
prevented by the positivity constraints instead."

Rule: semicolon chaining two complete clauses; three ideas.

Rewrite: "The two forms agree to first order in the excess and differ at long metal paths.
The subtraction form drops the clamp at zero and the floor. Negative corrected values are
prevented by the positivity constraints instead."

**40.** "The clamp matters, because a large metal coefficient drives thick-metal rays
negative, the clamp zeros them, and the added-back metal fills them with a smooth
projection."

Rule: three chained clauses describing three steps.

Rewrite: "The clamp matters for the following reason. A large metal coefficient drives
thick-metal rays negative, and the clamp sets those rays to zero. The added-back metal then
fills them with a smooth projection."

**41.** "That is inpainting: it deletes streaks and plastic signal together and leaves a dark
band, and a gradient or total-variation score prefers it (`brainstorm_skeptic.md`, attack
1)."

Rule: colon chaining two complete clauses; evidence and interpretation in one sentence.

Rewrite: "That step is inpainting. It deletes streaks and plastic signal together and leaves
a dark band. A gradient or total-variation score prefers that result
(`brainstorm_skeptic.md`, attack 1)."

**42.** Recurring across the page: "flatness score" (Answer), "image score" (Answer, and
sections on the family and on the score), "energy score" and "total-variation score" (basis
images), "primary score" (the score), "variance score" (the score), "surviving score" (what
to build).

Rule: one term per object. Some of these name the proposed score and some name classes of
candidate score, and the page never says which is which.

Rewrite: fix two names and use them everywhere. Use "the image score" for the score this
page proposes, and "candidate scores" for the family from which it is drawn. Where a
specific alternative is meant, name it once ("a total-variation score") and keep that name.

---

## The family to search: polynomial or physical

**43.** Section opening: "The polynomial's column count grows fast with the number of
metals."

Rule: a section that does not open with its big idea. The section's conclusion, the staged
recommendation, is the last paragraph.

Rewrite: open the section with the recommendation, "Start with the polynomial and use the
physical family as the synthetic truth," and let the column counts and the physics numbers
support it.

**44.** "At 200 kV behind 0.9 mm of copper the cubic in `mar.py` fits a grid of PMMA up to 8
cm and iron up to 1 cm with an rms error of 0.009 and a maximum of 0.026 in attenuation
units, where the attenuation reaches 4.35; a log-sum-exp with three shared bins reaches a
maximum of 0.0018 and four bins reach 0.00014."

Rule: 62 words, the longest sentence on the page; semicolon chaining two complete clauses;
four numbers and two models in one sentence.

Rewrite: "The simulation used a grid of PMMA up to 8 cm and iron up to 1 cm at 200 kV behind
0.9 mm of copper, on which the attenuation reaches 4.35. The cubic in `mar.py` fits that
grid with an rms error of 0.009 and a maximum error of 0.026 in attenuation units. A
three-bin exponential mixture reaches a maximum error of 0.0018, and a four-bin mixture
0.00014." The per-model errors could also move to `bh_physics_sim.py`'s companion notes,
leaving the ratio here.

**45.** "Fitted on iron up to 0.5 cm, the cubic errs by +0.69 at 1 cm of iron with no
plastic, and the three-bin form errs by −0.006; at 150 and 100 kV the cubic's extrapolation
errors are +6.3 and +8.3."

Rule: 42 words; semicolon chaining two complete clauses; three results in one sentence.

Rewrite: "Extrapolation was tested by fitting on iron up to 0.5 cm and evaluating at 1 cm
with no plastic. There the cubic errs by +0.69 and the three-bin mixture by −0.006. At 150
and 100 kV the cubic's errors on the same test are +6.3 and +8.3."

**46.** "The cubic's metal curve has an inflection inside the fitted range, at 0.85 cm of
iron at 200 kV, so it turns convex where physics is concave."

Rule: evidence and interpretation in one sentence; "physics is concave" uses a word loosely,
since it is the true attenuation function that is concave, not physics.

Rewrite: "The cubic's metal curve has an inflection inside the fitted range, at 0.85 cm of
iron at 200 kV. Beyond that point the cubic is convex. The true attenuation as a function of
path length is concave everywhere."

**47.** Recurring in this section: "a log-sum-exp with three shared bins", "the three-bin
form", "every mixture of exponentials", "the physical family", "the one-or-two-knob family",
"the two-basis attenuation form".

Rule: one term per object. These name one or two objects with six labels, and
"two-basis attenuation form" is never defined.

Rewrite: define the object once, for example "the physical family is a mixture of
exponentials over a few energy bins," and then use "the physical family" and "the three-bin
mixture" only.

**48.** "A one-knob fit against a truth made with a different detector and inherent filter
reached a maximum error of 0.011, and a two-knob fit 0.0045
(`bh_physics_extra_output.txt`)."

Rule: informal word for a technical object ("knob"); the second clause is elliptical.

Rewrite: "The physical family was also fitted against a truth generated with a different
detector and inherent filter. A one-parameter fit reached a maximum error of 0.011, and a
two-parameter fit reached 0.0045 (`bh_physics_extra_output.txt`)."

**49.** "The cost of the physical family is nonlinearity: each candidate needs one direct
reconstruction of the reduced slab, and a Gauss-Newton step needs one per parameter for its
Jacobian, which is the basis trick applied per iteration (`brainstorm_search.md`, approach
b)."

Rule: colon chaining two complete clauses; 42 words; "the basis trick" is informal, where
the page elsewhere says "basis images".

Rewrite: "The cost of the physical family is that it is nonlinear in its parameters. Each
candidate needs one direct reconstruction of the reduced slab. A Gauss-Newton step needs one
more per parameter for its Jacobian, which is the basis-image construction applied once per
iteration (`brainstorm_search.md`, approach b)."

**50.** "Its known weakness is that recovering a spectrum from transmission data is
ill-conditioned, which the literature reports for spectrum estimation methods
(`brainstorm_literature.md`, question 3); the one-or-two-knob family avoids that by never
freeing the bins."

Rule: semicolon chaining two complete clauses; a qualifier embedded as a relative clause.

Rewrite: "The physical family has one known weakness. Recovering a spectrum from
transmission data is ill-conditioned, which `brainstorm_literature.md` reports for spectrum
estimation methods (question 3). Fixing the bin energies and fitting only one or two scale
parameters avoids that problem."

**51.** "A K-edge inside the spectrum, as tungsten and tin have, breaks the two-basis
attenuation form; steel, aluminum, and titanium do not."

Rule: semicolon chaining two complete clauses; the referent of "do not" is ambiguous, since
it could mean they have no K-edge or that they do not break the form.

Rewrite: "Tungsten and tin have a K-edge inside the spectrum, which breaks the two-basis
attenuation form. Steel, aluminum, and titanium have no K-edge there, so the form holds for
them."

**52.** "For a rod centered in a plastic disk, every ray through the rod crosses nearly the
same plastic length, so the cross columns and the metal-only columns are collinear on the
metal rays and the ridge decides the split."

Rule: 40 words; three ideas; "the split" is an undefined referent at first use.

Rewrite: "Consider a rod centered in a plastic disk. Every ray through the rod crosses nearly
the same plastic length. The cross columns and the metal-only columns are therefore
collinear on the metal rays. Nothing in the data decides how the fitted excess divides
between them, so the ridge decides it."

**53.** "On a probe ray whose plastic estimate is 10 percent low, which a streak through the
plastic mask produces, the corrected plastic errs by +4.4 percent at beta 2e-4, +6.7 percent
at the default 2e-3, and +15.4 percent at 2e-2, where exact inversion of the true model has
no error (`bh_physics_extra_output.txt`)."

Rule: 50 words; an aside inside the claim; a numeric list given no sentence of its own; a
trailing qualifier. The symbol "beta" also appears here for the first time, and the reader
must infer that it is not the λ of the objective on line 125.

Rewrite: "The consequence was measured on a probe ray whose plastic estimate is 10 percent
low, which a streak through the plastic mask produces. Exact inversion of the true model has
no error on that ray. The existing fit's ridge strength is beta, and the corrected plastic
errs by +4.4 percent at beta 2e-4, by +6.7 percent at the default 2e-3, and by +15.4 percent
at 2e-2 (`bh_physics_extra_output.txt`)."

**54.** "The ridge strength is worth sweeping with an image score whatever else is built."

Rule: a qualifier embedded rather than given its own sentence.

Rewrite: "The ridge strength is worth sweeping with an image score. This holds whatever else
is built."

**55.** "And the image can see the split only through rays off the presented set, at edges
and along streaks, so whatever the data cannot pin the image pins only weakly."

Rule: undefined jargon ("the presented set"); "see" and "pin" are metaphors; the closing
clause is a garden-path construction that has to be read twice.

Rewrite: "The image is sensitive to that division only through rays whose path lengths differ
from the ones the fit saw, which occur at edges and along streaks. Those rays are few. The
image therefore constrains only weakly whatever the sinogram data leaves undetermined."

**56.** "The polynomial with the subtraction form goes first, because it reuses the fit, the
constraints, and the correction, and because its closed-form image fit is the cleanest test
of whether an image score can decide coefficients at all."

Rule: 40 words; two reasons and an embedded list in one sentence.

Rewrite: "The polynomial with the subtraction form goes first, for two reasons. It reuses the
existing fit, constraints, and correction. Its closed-form image fit is also the simplest
test of whether an image score can decide coefficients at all."

**57.** "If the first experiments show the polynomial's extrapolation on metal rays in the
reconstructions, the physical family becomes the model, with the same reduced problem, the
same score as the check, and a one-dimensional search per knob through `_search_minimum`."

Rule: 40 words; a three-item list appended as an aside; "extrapolation" used where
"extrapolation error" is meant.

Rewrite: "The first experiments may show the polynomial's extrapolation error on metal rays
in the reconstructions. In that case the physical family becomes the model. It keeps the same
reduced problem and the same score, and it searches one parameter at a time through
`_search_minimum`."

---

## The score

**58.** "The primary score is the low-frequency deviation of the plastic class from its own
mean, inside a mask eroded away from every edge and every metal, plus signed region
contrasts for the metal terms: the corridor between two metals minus the far plastic, and an
annulus around each metal minus the far plastic (`brainstorm_score.md`)."

Rule: 53 words; two definitions in one sentence; a colon list appended to an already
complete claim; "corridor" is an invented term used without definition.

Rewrite: "The image score has two parts. The first is the low-frequency deviation of the
plastic class from its own mean, measured inside a mask eroded away from every edge and
every metal. The second is a set of signed region contrasts for the metal terms: the
straight region between two metals minus the far plastic, and an annulus around each metal
minus the far plastic (`brainstorm_score.md`)."

**59.** "The low-frequency form is the blur of the geometry estimator with a different
purpose: there the blur matched the resampling count, here it removes noise and fine texture
so that cupping and bands remain."

Rule: colon chaining two complete clauses.

Rewrite: "The low-frequency form reuses the blur of the geometry estimator for a different
purpose. In the geometry estimator the blur width matched the resampling count. Here the
blur removes noise and fine texture, so that cupping and bands remain."

**60.** "The blur width and the erosion margins are in ALU and are swept, not guessed."

Rule: undefined abbreviation on a page that otherwise spells its terms out.

Rewrite: "The blur width and the erosion margins are specified in arbitrary length units
(ALU), so they do not change with voxel size. Both are swept rather than guessed."

**61.** "Under the harness's quadratic model sized to change the largest sinogram value by
ten percent, a uniform disk's center dips by about 13 percent of its attenuation and the
plastic mean shifts by two thirds of that (`brainstorm_score.md`, derived for a disk)."

Rule: 42 words; two results in one sentence; "the harness's quadratic model" is an undefined
referent.

Rewrite: "The size of the effect was derived for a disk under a quadratic hardening model,
scaled to change the largest sinogram value by ten percent. Under that model a uniform
disk's center dips by about 13 percent of its attenuation. The plastic mean shifts by two
thirds of that dip (`brainstorm_score.md`)."

**62.** "Which slices to score inverts the geometry estimator's choice."

Rule: a word used loosely. The choice is not inverted; the criterion is reversed.

Rewrite: "The slices to score are chosen by the opposite criterion from the geometry
estimator's."

**63.** "Cupping is read on slices near the central plane with the largest plastic
cross-section, where the chords are longest and FDK's own shading is smallest; the metal
terms are read on the slices that hold the metal (`brainstorm_score.md`)."

Rule: semicolon chaining two complete clauses; a qualifier embedded in the first clause;
"read" used metaphorically.

Rewrite: "Cupping is measured on slices near the central plane with the largest plastic
cross-section. There the chords are longest and FDK's own shading is smallest. The metal
terms are measured on the slices that contain the metal (`brainstorm_score.md`)."

**64.** "The chooser must select dense features where the geometry chooser avoided them."

Rule: invented and undefined jargon ("the chooser", "the geometry chooser").

Rewrite: "The slice-selection step must therefore select slices with dense features, where
the geometry estimator's slice selection avoided them."

**65.** "Five safety rules come from the reports." followed by five sentences.

Rule: a list not introduced by a complete clause and a colon.

Rewrite: "Five safety rules come from the reports:" followed by the five rules as bullets.

**66.** "The noise floor is measured by scoring the difference between the even-view and
odd-view reconstructions, because a stronger correction amplifies long-path noise, so a
variance score prefers weak corrections for a reason unrelated to hardening."

Rule: 36 words; a "because ... so ..." chain carrying three ideas.

Rewrite: "A stronger correction amplifies long-path noise. A variance score therefore prefers
weak corrections, for a reason unrelated to hardening. The noise floor that separates the two
effects is measured by scoring the difference between the even-view and odd-view
reconstructions."

**67.** "The estimate must agree across slices, across blur widths, and across erosion
margins within its half width, or the verdict is undecided."

Rule: an ambiguous referent. "Within its half width" could attach to the erosion margins, to
the estimate, or to the score curve.

Rewrite: "The estimate must agree across slices, across blur widths, and across erosion
margins. Agreement means the estimates fall within half the width of the score minimum.
Otherwise the verdict is undecided."

**68.** Paragraph opening: "Within-class variance is Otsu's own objective
(`mbirtorch/preprocess/segmentation.py:222-226`)."

Rule: a paragraph whose first sentence is not its topic sentence. The paragraph's idea is
that a variance score plus re-segmentation is self-fulfilling, and that three things limit
the damage.

Rewrite: open with "A score built on within-class variance can be self-fulfilling," then give
the Otsu fact as support.

**69.** "And a radial density gradient in a round object matches the cupping basis image
exactly, which no single-spectrum measurement can separate (`brainstorm_score.md`)."

Rule: a relative clause carrying a separate claim, with an ambiguous antecedent and no object
for "separate". It is also listed as one of three things that "limit the damage", but it is a
confound rather than a limit.

Rewrite: "One case is not limited at all. A radial density gradient in a round object matches
the cupping basis image exactly, and no single-spectrum measurement can distinguish the two
(`brainstorm_score.md`)."

**70.** "The NSI phantom scanned without its insert is the reference for that: the plastic is
the same object in both scans, so its real structure is measured, not assumed."

Rule: colon chaining two complete clauses.

Rewrite: "The NSI phantom scanned without its insert is the reference for that case. The
plastic is the same object in both scans, so its real structure is measured rather than
assumed."

---

## Traps, and what settles each

**71.** The section opens directly with a table.

Rule: a section that does not open with its big idea, and a list not introduced by a complete
clause and a colon.

Rewrite: open with one or two sentences, for example: "Eight traps have been identified
across the six reports. Each has a mechanism, a remedy, and an experiment that would settle
whether the remedy works:" and then the table.

**72.** Table cell: "the gaming curve of experiment 1".

Rule: invented jargon. The phrase appears nowhere else, and experiment 1 describes the
procedure without naming it.

Rewrite: "the one-parameter score curves of experiment 1", and use the same phrase in
experiment 1.

**73.** Table cells: "the starvation arm of experiment 1", "the scatter arm of the synthetic
truth".

Rule: undefined jargon. "Arm" is borrowed from clinical trials and is never defined here.

Rewrite: "the photon-starvation case of experiment 1" and "the scatter case of the synthetic
truth", with the same wording in experiment 1.

**74.** Recurring across the page: "direct reconstruction" (lines 16, 96, 113, 179, 279,
293), "FDK" (lines 222, 291, 292), "FBP" (line 257, twice).

Rule: one term per object. The page names one object three ways, and the trap row "FBP
versus MBIR" reads as a fourth method rather than the one already called the direct
reconstruction.

Rewrite: use "direct reconstruction" everywhere, and say once, at first use, that the direct
reconstruction in mbirtorch is FDK.

---

## Multiple metals

**75.** Section opening: "The polynomial's cross columns of two metals, m_0 m_1 and their
higher forms, are supported only on rays through both metals, so they are decided by few rays
and few slices."

Rule: a section that does not open with its big idea; an aside inside the claim; two ideas in
31 words.

Rewrite: open with the section's conclusion, "Two metals are the hardest case, and a
two-metal synthetic case is required to test it," then give the cross-column detail as
support, split into two sentences.

**76.** "The region structure isolates one material: metal k's annulus reads its metal-only
terms, its corridors to the plastic read its cross terms, and the corridor between metals j
and k reads their joint term (`brainstorm_score.md`)."

Rule: colon chaining two complete clauses; a three-item list given no sentence of its own;
"reads" used metaphorically; "corridors" is the invented term from entry 58.

Rewrite: "The region structure separates the materials, so that each region is sensitive to
one group of terms. The annulus around metal k is sensitive to its metal-only terms. The
straight regions between metal k and the plastic are sensitive to its cross terms. The
straight region between metals j and k is sensitive to their joint term
(`brainstorm_score.md`)."

**77.** "The literature's only image-criterion method at three materials, with seven free
coefficients, lost estimability under noise, and methods that succeed with several materials
import physics, a spectrum, cross sections, or a measured material, to pin most of the
coefficients (`brainstorm_literature.md`, question 2)."

Rule: 39 words; an aside inside the claim; a four-item list inside the second clause; two
ideas.

Rewrite: "The literature has one image-criterion method at three materials, and it used seven
free coefficients. That method lost estimability under noise. The methods that do succeed
with several materials add outside information to fix most of the coefficients: a spectrum,
tabulated cross sections, or a measured reference material (`brainstorm_literature.md`,
question 2)."

**78.** "The real scans limit what can be tested: the NSI phantom has one insert of one
material, and the bga scan's solder balls are the thin-metal case where the monomial columns
collapse (`brainstorm_skeptic.md`, attacks 3 and 11)."

Rule: colon chaining two complete clauses; "collapse" used loosely where "become collinear"
is meant.

Rewrite: "The real scans limit what can be tested. The NSI phantom has one insert of one
material. The bga scan's solder balls are the thin-metal case, where the monomial columns
become nearly collinear (`brainstorm_skeptic.md`, attacks 3 and 11)."

**79.** "A two-metal synthetic case is therefore required, and it should include the
collinear placement."

Rule: an undefined referent used with a definite article. Experiment 1 calls the same
arrangement "one placement centered and one off center", so the reader has two names for one
thing.

Rewrite: "A two-metal synthetic case is therefore required. It should include the centered
placement, which is the one that makes the columns collinear."

---

## Cost at production scale

**80.** "The search itself is cheap at any scale, because it runs on the reduced problem: K+1
mask projections onto the window rows, one direct reconstruction of the slab per basis
column, and tens of megabytes of images."

Rule: a claim, its reason, and a three-item list in one 37-word sentence; K is introduced
here for a quantity the page has been calling `num_metal`.

Rewrite: "The search itself is cheap at any scale, because it runs on the reduced problem.
Its cost has three parts: one mask projection onto the window rows per class, one direct
reconstruction of the slab per basis column, and tens of megabytes of images. There are
`num_metal` + 1 classes."

**81.** "The correction is a per-pixel function of y, p, and m_k, so it streams view batch by
view batch as `BH_correction` does through `pipeline.map_view_batches`, with one host output
and no second full-size sinogram."

Rule: 35 words; three ideas; a trailing qualifier.

Rewrite: "The correction is a per-pixel function of y, p, and m_k. It can therefore stream
view batch by view batch, as `BH_correction` already does through
`pipeline.map_view_batches`. That form needs one host output and no second full-size
sinogram."

**82.** "The batch kernel needs p and m_k for its views, and no public entry projects a
subset of views today; the per-view-batch body exists (`mbirtorch/cone_beam.py:161`) and a
model copy with the batch's angles is the available route (`brainstorm_pipeline.md`, section
3; `brainstorm_search.md`)."

Rule: 46 words; a semicolon chaining two clauses that are each already compound; "route" used
loosely.

Rewrite: "The batch kernel needs p and m_k for its own views. No public entry point projects
a subset of views today. The per-view-batch body already exists
(`mbirtorch/cone_beam.py:161`). Until a public entry exists, a model copy carrying the
batch's angles is the way to reach it (`brainstorm_pipeline.md`, section 3;
`brainstorm_search.md`)."

**83.** "The class volumes themselves are the remaining memory question: K+1 float masks at
32 GB each at 2K, or one label volume at 8 GB if the projector could take a label and a class
value."

Rule: colon chaining two complete clauses; a conditional embedded in the second item.

Rewrite: "The class volumes themselves are the remaining memory question. One float mask per
class costs 32 GB each at 2K. A single label volume would cost 8 GB, but only if the
projector could accept a label volume and a class value."

**84.** "The alternation cost changes shape."

Rule: metaphor in a technical statement.

Rewrite: "The reduced problem also changes what the alternation costs."

**85.** "With the reduced problem it can be one FDK for the masks, an alternation of
correction and direct reconstruction at reduced scale until the coefficients and the masks
stop moving, and one full MBIR."

Rule: a three-item list given no sentence of its own; "stop moving" used loosely for
convergence.

Rewrite: "With the reduced problem it can be three steps: one direct reconstruction for the
masks, an alternation of correction and direct reconstruction at reduced scale until the
coefficients and the masks converge, and one full MBIR."

---

## What to build, in order

**86.** "Every step below is an experiment with a result that would stop the work, ordered by
cost."

Rule: a trailing modifier that attaches to the wrong noun, since "ordered by cost" describes
the steps and not the work.

Rewrite: "The steps below are ordered by cost. Each is an experiment with a result that would
stop the work."

**87.** "Build a thin three-dimensional case of a plastic disk with one and with two metal
rods, one placement centered and one off center, hardened with the physical model of
`bh_physics_sim.py` at 200 kV behind 0.9 mm of copper, with Poisson noise at the file's
photon count and a starvation arm."

Rule: 50 words; four stacked modifiers; "the file's photon count" is an undefined referent.

Rewrite: "Build a thin three-dimensional case of a plastic disk with metal rods. Use one rod
and two rods, in a centered and an off-center placement. Harden it with the physical model of
`bh_physics_sim.py` at 200 kV behind 0.9 mm of copper. Add Poisson noise at the photon count
in the NSI scan file, and include a photon-starvation case."

**88.** "Compare the plastic-region rms error against the monochromatic reconstruction for:
no correction, the existing fit, the subtraction-form closed-form image fit at several λ, and
the physical family fitted on the sinogram."

Rule: a list not introduced by a complete clause and a colon. The colon follows the
preposition "for".

Rewrite: "Compare the plastic-region rms error against the monochromatic reconstruction for
four cases: no correction, the existing fit, the subtraction-form closed-form image fit at
several λ, and the physical family fitted on the sinogram."

**89.** "Then follow two one-parameter paths from the existing fit, a global scale-down and
an increasing metal subtraction up to full inpainting, and record every proposed score along
each."

Rule: a two-item list inserted as an aside inside the claim.

Rewrite: "Then follow two one-parameter paths from the existing fit: a global scale-down, and
an increasing metal subtraction up to full inpainting. Record every proposed score along each
path."

**90.** "Kill: every score reaches its minimum where the rms error is worse than at the
sinogram fit."

Rule: undefined notation. "Kill" is used as a label four times and never defined, and the
guide forbids plan notation in favor of plain English.

Rewrite: "Stop the work if every score reaches its minimum at a point where the rms error is
worse than at the sinogram fit." Use the same form for the other three steps.

**91.** "Ablations, one variable each: masks eroded and dilated by one pixel; a radial density
gradient of 2, 5, and 10 percent and a textured plastic; blur widths of 1 to 10 percent of
the diameter; the coefficients fitted at bin 1 and bin 4 and applied at bin 1; unhardened
data at view stride 1 and 4."

Rule: a list introduced by a fragment rather than a complete clause; 56 words in one
semicolon-separated run.

Rewrite: "Run five ablations, varying one thing each:" followed by the five items as bullets.
The style guide also says that run detail belongs in the script or a companion .md file, so
the parameter values could live with the experiment script and leave the list of five
variables here.

**92.** "At the reduced whole-extent problem, compute the surviving score at the existing fit
and at plus and minus twenty percent on each coefficient in turn, and the even/odd floor."

Rule: two undefined terms ("the reduced whole-extent problem", where the page elsewhere says
"the reduced problem"; "the surviving score", which presumably means the score that survived
experiment 1); two ideas in one sentence.

Rewrite: "Use the reduced problem at the full transverse extent. Compute the score that
survived experiment 1 at the existing fit, and at plus and minus twenty percent on each
coefficient in turn. Compute the even-view and odd-view noise floor on the same data."

**93.** "The same job sweeps beta over five decades and records the argmin and the curve depth
against the value chosen by eye, which is the cheapest product on its own."

Rule: an ambiguous relative clause, since "which" could refer to the sweep or to the value
chosen by eye; "product" is used where the Open questions section says "deliverable".

Rewrite: "The same job sweeps beta over five decades. It records the argmin and the depth of
the score curve, and compares both against the value a user chooses by eye. This sweep is the
cheapest deliverable on the list."

**94.** "Run one MBIR pass with the image-chosen coefficients and with the sinogram-fit
coefficients on the synthetic case and on the NSI metal scan, and compare the plastic region
against the monochromatic truth and against the no-metal scan's reconstruction."

Rule: 40 words joined by four instances of "and", so the reader must work out which pairs
group together.

Rewrite: "Run one MBIR pass with the image-chosen coefficients, and a second with the
sinogram-fit coefficients. Do this on the synthetic case and on the NSI metal scan. Compare
the plastic region against the monochromatic truth on the synthetic case, and against the
no-metal scan's reconstruction on the NSI scan."

**95.** "A function in `mar.py` in the style of `estimate_geometry_from_recon`, returning
candidates, scores, the coefficient vector, the hyperparameters, the class thresholds, the
reduction record, notes, and undecided."

Rule: a sentence fragment carrying an eight-item list, not introduced by a complete clause
and a colon.

Rewrite: "Build a function in `mar.py` in the style of `estimate_geometry_from_recon`. It
returns the candidates, their scores, the chosen coefficient vector, the hyperparameters, the
class thresholds, the reduction record, notes, and the undecided verdict."

**96.** "Its gates: recovery of injected coefficients on the synthetic case; an unchanged
ranking at detector binning 2; and on the NSI metal scan, per-slice agreement, a
plastic-region comparison against the no-metal scan, and the plastic-only ray residual
showing no trend against path length."

Rule: a list introduced by a fragment; a nested list inside the third item; 43 words.

Rewrite: "The function must pass three gates:" followed by bullets, with the third bullet's
three NSI checks as sub-bullets.

---

## Open questions for Greg

**97.** "The reports rank them cheapest last, and each is a different amount of work."

Rule: a phrase that costs the reader effort to decode. "Cheapest last" describes the order of
the question's own list, which the reader must reconstruct.

Rewrite: "The reports rank these three by cost. The warning check is the cheapest, the
hyperparameter automation is next, and the coefficient estimator is the most work."

**98.** "Today it is outside the family, which is right at 200 kV with copper filtration and
wrong at lower voltages."

Rule: a qualifier embedded as a relative clause with an ambiguous antecedent, since "it" is
the plastic's hardening and "the family" was used two sentences earlier for the physical
family.

Rewrite: "Today the plastic's own hardening is outside the model. That omission is correct at
200 kV with copper filtration. It is wrong at lower voltages."

**99.** "The physics agent saw a file name in the scan directory carrying "0.5BH"."

Rule: one term per object, and loose usage. The page names its sources as reports everywhere
else, and "carrying" is doing the work of "containing".

Rewrite: "`brainstorm_physics.md` reports a file name in the scan directory that contains the
string "0.5BH"."

**100.** "Is the NSI phantom's plastic uniform enough that the no-metal scan can serve as the
texture reference, and is the bga scan the textured-object gate?"

Rule: two questions in one sentence, which forces one answer to cover both.

Rewrite: "Is the NSI phantom's plastic uniform enough that the no-metal scan can serve as the
texture reference? Should the bga scan be the textured-object gate?"

---

## Readability overall

The page is well organized and the reasoning is easy to follow once each sentence has been
read twice, which is the problem: the guide asks for writing that parses on the first read.
Its structure is genuinely good, with sections in a defensible order and most paragraphs
opening on their topic sentence, and the citation discipline is exemplary. What slows the
reader is sentence-level compression. The page has no em-dash asides, but it has 17
semicolons and many colons that chain complete clauses, and the habit of packing a claim,
its evidence, its qualifier, and its citation into one 40-to-60-word sentence appears in
every section. The best passage on the page is the ridge-strength paragraph, which states
the measurement and then says "Two conclusions follow" in its own sentence; that pattern
should be the model for the rest.

Two other things cost the reader effort. The page names several objects more than one way,
most damagingly the direct reconstruction (also FDK and FBP), the score (flatness, energy,
image, primary, variance, surviving), and the physical family (log-sum-exp, three-bin form,
mixture of exponentials, one-or-two-knob family, two-basis attenuation form). And it uses a
handful of terms that a reader outside this session cannot resolve: the slab, the reduced
problem, the presented set, the corridor, the chooser, ALU, the harness's quadratic model,
the gaming curve, the surviving score, and the skeptic as an actor.

The Answer does not yet stand on its own for a CT expert who has not read the six reports.
The physics in it is clear, and the central asymmetry between geometry and hardening is
stated well enough to survive the length of the sentence carrying it. But the Answer depends
on five things it never defines: what the geometry estimator is and why transplanting it was
ever the plan, what the existing correction does, what "the cubic" is, what "the presented
data" means, and what a basis image is. Two added sentences at the top, one describing the
existing correction and one describing the geometry estimator, would make the section
readable alone. Detail that could move out of the Answer without losing anything actionable:
the "6 to 15 coefficients" range, which the table in a later section supplies, and the "10 to
100 times" and "about 100 times" figures, which belong with the physics numbers they
summarize.

Detail that could move from the page to the source reports or to script companion notes,
without losing actionable information: the four binning-bias percentages (entry 28), the rms
and maximum errors for the cubic and the two exponential mixtures (entry 44), the two
extrapolation numbers at 150 and 100 kV (entry 45), the three probe-ray percentages at three
ridge strengths (entry 53), and the parameter values inside the experiment-1 ablation list
(entry 91). In each case the page can keep the comparison that drives a decision and cite the
script for the numbers behind it.
