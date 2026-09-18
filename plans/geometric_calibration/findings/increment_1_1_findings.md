# Geometric calibration, sub-increment 1.1: the fine sweep on the real scans

Date: 2026-09-05.  Status: measured, texts revised and staged, awaiting Greg's review.  The plan of
record is `estimate_by_recon_plan.md` in this directory.  Greg approved it on 2026-09-05.  The
experiment record is `plans/geometric_calibration/experiments/recon_sweep_fine.md`, and every
number below was read from it or from the sources it cites.  That record's "Units and terms" defines
the raw and normalized trimmed curves, the structure curve, the half width, and the repeatability
floor.  This page uses those terms as that record defines them.  The code is on the
`geometric_calibration` branch of mbirtorch, based on commit `590fea9`.  It is staged and not committed.

## Outcome

The sweep located the rotation the reconstructions prefer.  It also found that the scoring kernel the
plan named cannot locate that rotation.  A slice's lattice angle is an angle that displaces that
slice's own detector rows by a whole number of pixels.  With the bilinear kernel each slice's score
minimum is within 0.007 degrees of its own lattice angle.  It stays within 0.008 degrees of that angle
at every blur from 0 to 4 voxels, and removing the noise term does not move it off the lattice.  The
Fourier kernel shifts the data without smoothing them.  With it the per-slice minima move off the
lattice angles.  A model with one rotation and one channel offset error then fits those
minima.  At a blur of 2 voxels the fitted rotation is 0.1507 degrees on the no-metal scan and 0.1595
degrees on the metal scan, from the structure curves.

Neither figure is one of the two candidates the sweep was run to separate.  The vendor's recorded tilt
is 0.16717 degrees.  A conjugate-view estimate over a tall detector band gives 0.1299 degrees on the
no-metal scan, and that estimate is called the tall-band answer below.  It is recorded in
`closed/real_scan_band_reach.md`, under the experiments directory.  The fitted rotation is 0.0165
degrees below the vendor's value on the no-metal scan and 0.0077 degrees below it on the metal scan.
It is 0.0208 and 0.0296 degrees above the tall-band answer.  That tall-band answer was measured on the
no-metal scan alone, so the second of those two differences compares two scans.

Three numbers describe the spread on the fitted rotation.  On the Fourier job, on the slices that hold
the object, the even-view and odd-view per-slice minima differ by at most 0.0020 degrees at a blur of
2 voxels.  The fitted rotation moves by 0.0033 and 0.0038 degrees on the two scans between a blur of 2
and a blur of 4 voxels.  The largest of the three is the scan-to-scan difference.  The two scans share
one recorded geometry, and their fitted rotations differ by 0.0088 degrees.  That difference is either
a real difference between the two acquisitions or a limit of the score, and this sweep does not decide
which.

One alternative to the gap from the vendor's value stays open.  Any height dependence that is linear
in the kernel row distance shifts the fitted rotation and leaves no residual.  The fit therefore
cannot tell an in-plane rotation from such an effect.  The scan's recorded out-of-plane axis lean is
one candidate, and a LEAP simulation of it moved the in-plane estimate by up to 0.0110 degrees.  That
is comparable to the 0.0165-degree gap (the record's limits, item 4).

## The gates and the verdicts

The plan sets three gates for this sub-increment.  Each is quoted below in the plan's words, followed
by the verdict.  All three were read at a blur of 2 voxels from the raw trimmed curves, in both jobs'
`gates` entries, with the interior-minima count from the curve records.

1. "Each scan's trimmed curve has one minimum whose depth exceeds the repeatability floor."  Passes
   for both kernels.  Each raw trimmed curve has exactly one interior minimum.  With the bilinear
   kernel the depth is 19.40 times the repeatability floor on the no-metal scan and 21.26 times it on
   the metal scan.  With the Fourier kernel it is 13.84 and 16.75 times.
2. "The metal scan's minimum lies inside 0.15 to 0.24 degrees, the bracket its earlier measurements
   set, and a minimum at either end of the searched range fails this gate and widens the range."
   Fails with the bilinear kernel and passes with the Fourier kernel.  The bilinear job's metal
   minimum is 0.1481 degrees, and it is below 0.15 degrees at every blur.  The Fourier job's metal
   minimum is 0.1544 degrees.  It is inside the bracket at blurs of 1 to 4 voxels and outside it
   unblurred.  No minimum in either job lies at an end of the searched range, so the range does not
   widen.  The Fourier pass depends on which form of the trimmed curve is read.  That job's own
   closing comparison marks the normalized trimmed curve as outside the bracket at a blur of 2 voxels,
   at 0.1288 degrees.  The bilinear job fails on both forms of the curve.
3. "The no-metal and metal minima agree within the larger half width, which is the streak test,
   because the two scans share one geometry and differ by the insert."
   Passes for both kernels.  With the bilinear kernel the two scans differ by 0.0031 degrees, and the
   larger half width is 0.0214 degrees.  With the Fourier kernel they differ by 0.0106 degrees, and
   the larger half width is 0.0441 degrees.

One qualification applies to all three verdicts.  The gates are computed on the trimmed curve, and the
sweep found that the trimmed curve does not locate the fitted rotation.  With the Fourier kernel the
raw trimmed minima are 0.1438 and 0.1544 degrees, while the model fits are 0.1507 and 0.1595 degrees.
The gates therefore test the shape of the trimmed curve.  The rotation itself comes from the model
fit.

## The decision

The plan names two outcomes for this sub-increment and a stopping condition.  A no-metal minimum
near 0.167 degrees would confirm the vendor's value.  A minimum near 0.130 degrees would mean that
reconstruction quality prefers the tall-band answer, in which case the estimator proceeds and the
docs stop naming the vendor's value as right.  A curve whose depth fails the repeatability gate
would stop the work for review.  The measured rotation is near neither named value, and the depth
gate passed, so the plan's rule does not stop the work.

The sweep separated a fault in the method from the answer, and the two must be read separately.
The fault is the bilinear kernel that applies a candidate rotation.  Its smoothing depends on the
fractional part of the displacement, and that smoothing set each slice's minimum rather than the
geometry.  Run exactly as the plan specifies, the sweep would have read 0.1450 degrees from a
well-formed curve, 0.1321 normalized, and the result would have been taken as favoring the
tall-band answer.  The answer above comes from the sweep with the kernel replaced.  With the
Fourier kernel the score behaved as the design predicts: the per-slice minima fit one rotation and
one offset term to 0.0010 degrees, the even/odd halves repeat to 0.0020 degrees, and the two scans
agree to 0.0088 degrees.  The bilinear result is also the case the plan's undecided verdict is
designed to refuse, because its per-slice minima disagree by far more than the offset term
explains.  As repaired, the method is not suspect on this evidence.  The unrepaired kernel is what
failed, and the repair is the first requirement carried into sub-increment 1.2.

The decision is to proceed on the plan's second outcome, with the kernel repair attached.  The
four texts that named the vendor's value as right were revised to state the sweep's result, which
is this sub-increment's deliverable, and they are staged.

## What changed in mbirtorch

Four texts were revised and staged on the `geometric_calibration` branch.  All four had said that
direct reconstructions showed the vendor's recorded tilt of 0.167 degrees to be right.  One is the
passage in `docs/source/usr_preprocess.rst`.  The other three are in
`mbirtorch/preprocess/geometry_calibration.py`: the comment above `_CONJUGATE_ROTATION_TOLERANCE`, the
docstring of `estimate_det_rotation`, and the sub-pixel warning in that function.

None of the four now says the vendor's value was shown right.  The docs passage and the docstring
say the sweep put the detector rotation near 0.15 degrees and that the vendor's recorded tilt was
0.167 degrees.  The comment above `_CONJUGATE_ROTATION_TOLERANCE` says only that the estimate's zero
point was off by about a tenth of a degree on one real scan.  The warning carries the advice without
numbers.  The staged diff has the exact wording.  Three of the four keep the advice to prefer a vendor
tilt when the reader supplies one and to check the slices far from the central plane.  The comment
never carried that advice.  The 0.15 degrees in the two texts that give it is the Fourier-kernel
model fit and not a trimmed-curve minimum, and the texts do not say so.

The same staged change shortens the module's docstrings, comment blocks, and warning messages to
what a caller needs, at Greg's request during the review.  Design history and measured figures that
belonged in findings pages were removed, and every Args, Returns, and Raises section was kept.  No
code statement changed except the shortened warning text.  The two test files that cover the module
pass after the change, with 42 tests, and no test was added.

## What is staged

Two files are staged in mbirtorch, on the `geometric_calibration` branch:
`docs/source/usr_preprocess.rst` and `mbirtorch/preprocess/geometry_calibration.py`.  Six files are
staged in this repository: the four experiment scripts `recon_sweep_fine.py`,
`recon_sweep_fine_fourier.py`, `recon_sweep_fine_analysis.py`, and `recon_sweep_fine_tables.py`; the
record `recon_sweep_fine.md`; and this page.  The first five are in
`plans/geometric_calibration/experiments/`.  This repository's `.gitignore` excludes
`.sbatch`, `.jsonl`, `.json`, `.log`, and `.png` files, so the record transcribes what it needs from
those outputs.

## Design consequences, and Greg's decisions

These questions concern how sub-increments 1.2 and 1.3 adopt the repair.  None of them reopens
whether the score works: with the repaired kernel it measured the rotation, and its internal
checks would have caught the fault.

Three decisions shape sub-increment 1.2.  Greg decided all three on 2026-09-05, in the review of
this page: the scoring kernel is the Fourier kernel, because the projector-side rotation is the
larger change and stays parked; the combination across slices is the offset-error model fit, and
the sub-increments state their gates on the fit; and the even/odd split runs by default.  The
three items below record the considerations behind those decisions.

1. The scoring kernel.  It must not smooth by a displacement-dependent amount.  The candidates are
   the Fourier kernel, extended with padding and a taper so that a truncated projection does not
   wrap, or the projector-side rotation the plan parked, which resamples nothing.
2. The combination across slices.  A trimmed curve averages minima that an offset error displaces,
   and here its minimum missed the fitted rotation by 0.005 to 0.031 degrees.  The offset-error
   model fit should replace it.  The fit also returns the offset term, which stays a fitted
   parameter until sub-increment 1.4's synthetic gate identifies it as a channel offset.
3. Standard practice per run.  The even/odd split adds no reconstruction cost and yields the
   repeatability floor and the noise share, so it should run by default.  The slice chooser needs a
   content test: one of the four slices held no object, its noise share stayed above 0.79 at every
   blur, and the trim did not always drop it.

Two numbers pass to sub-increment 1.3.  The half width at the recommended blur of 2 voxels is
about 0.04 degrees.  The undecided threshold sits between a measured failing case, model residuals
of 0.0146 degrees, and a measured passing case, 0.0010 degrees.

Two notes close the section.  The blur default of 2 voxels is confirmed rather than changed.
`apply_calibration` corrects a rotation with the same bilinear kernel, which is a separate
question that no measurement here bears on.

## What is left

Greg reviewed this page and decided the three design questions, so sub-increment 1.2 can start.
That sub-increment builds the score and the slice chooser, which is the helper that picks the
slices the score reads, on the decisions recorded above.

Three smaller items are open.  The sign of the fitted offset error was not reconciled with the sign
convention of the conjugate-view estimator, so only magnitudes have been compared.  The Fourier
kernel's channel-edge wrap was not measured, on a real sinogram or on a synthetic one.  The kernel
check ran on a Mac rather than in the cluster job, so it is not part of that job's record.
