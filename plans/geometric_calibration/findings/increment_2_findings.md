# Geometric calibration, Increment 2: the conjugate-view method

Date: 2026-09-04, revised twice the same day after Greg's reviews.  Status: implemented,
awaiting Greg's review.  The code is on the `geometric_calibration` branch of mbirtorch, on top
of the Increment 1 commit `550b5d3`.  The plan of record is `geometric_calibration_plan.md` in
this directory, and this page reports on its Increment 2.

Units on this page.  An offset is given in channels, which are detector pixels along the channel
axis.  A channel is 1 ALU in the laptop experiments, 1 mm at 512 channels on the cluster, and
0.5 mm at 1024.  A rotation is given in degrees, and its edge displacement is the distance it
moves the edge pixel of the detector, in pixels.

## Outcome

Increment 2 delivers the three functions the plan names.  `estimate_det_channel_offset`
recovers a known offset to 0.02 channels on parallel beam.  The plan's gate for parallel beam is
0.1 channels.  On cone beam at a full fan angle of 20 degrees it recovers the offset to 0.023
channels over the whole search range.  That range was tested on a centered phantom and on an
off-axis rod.  The plan's gate for cone beam is 0.5 channels.  `estimate_det_rotation` recovers
a rotation of 2 or 3 degrees to within 5 percent on both geometries.  `conjugate_difference`
returns the stack of images that both estimates are built from.  All three are in
`mbirtorch/preprocess/geometry_calibration.py`.  Each refuses the scan types the plan lists.

The rotation estimate was nearly deferred, and it was kept on measurement.  Scoring a candidate
rotation resamples the sinogram.  On the small synthetic detectors used on the laptop, that
resampling biased the estimate by 14 to 18 percent of a 0.3 degree rotation.  Greg directed
that the estimate be deferred if that bias proved too large.  The alternative is a rotation
handled inside the projectors.  A further measurement showed that the bias is a property of
rotations that displace the edge pixel by less than about one pixel.  A cluster job at 512 and
1024 channels confirmed that, and the estimate is within 0.5 percent of the angle from 1.3
pixels of edge displacement upward.  On a full-size detector any rotation worth correcting is
well past that.  The estimator was therefore kept.  It uses the cubic kernel, and it warns when
the edge displacement is below one pixel.

## Decisions Greg made on the first draft, and what followed

**The rotation estimate stays, and the cluster job confirmed it.**  Slurm job 15915803 on
gautschi ran `calibration_512_gautschi.py` at 512 and 1024 channels.  The geometry is the one
the LEAP comparison of 2026-09-02 used, at a full fan angle of 14.6 degrees.  The record is
`calibration_512_gautschi.md` in `plans/experiments/features/geometric_calibration/`.  From an
edge displacement of 1.3 pixels upward the rotation estimate is within 0.5 percent of the angle
at both sizes.  Below half a pixel it reads 10 to 24 percent low.  At 0.89 pixels it read 4.5
percent high.  The warning threshold was therefore moved from half a pixel to one pixel.  LEAP's
`estimate_tilt` on the same sinograms reads 10 to 18 percent high below half a pixel, 3.4 percent
high at 0.89 pixels, and within 1 percent above that.  The two estimators err in opposite
directions in the same regime.  These results indicate that the bias is in the resampling of
each estimator and not in the way the tilted test data were made, which the first draft listed as
a doubt.  The claim that the bias depends on the edge displacement and not on the detector size
rests on one matched pair, at 0.45 pixels on both sizes, where the errors were 10 and 12 percent.

**The offset scale error was traced and fixed.**  Greg asked for the trace.  The error on the
off-axis rod ran from +0.035 channels at a true offset of -3.5 to -0.032 at +3.5, in proportion to
the offset.  The trace ruled out three candidates: the trimmed pair set, the comparison region,
and the interpolation between partner views.  Changing none of them changed the error.  The
trace found two causes in the pairing.  The channel offset entered each channel's fan angle with
the wrong sign.  The partner view is found by the column of the mirrored array rather than by
the original channel, and mirroring reverses the sign of the offset term.  The second cause was
that the pairing was computed at the model's offset, which the estimator has no reason to treat
as correct.  With the sign corrected, the first pass alone still errs in proportion to the
offset, by up to 0.035 channels on the rod, and a second pass that re-pairs at the first
estimate removes that trend.  The record `conjugate_offset_recovery.md` shows both passes at
every offset.  At offsets of about one channel the second pass moves the estimate by up to 0.01
channels in either direction.  It doubles the cost of the estimate.

**The view stride stays at 1, and partners come from every view.**  Greg asked that a larger
stride, when memory is limited, be applied to the views of a half rotation, with each kept view
paired with its opposite from the full set.  That scheme was implemented and measured, and one
of its two parts was kept.  Drawing the partners from every view is now the rule at any stride.
A stride therefore thins the reference views without blurring their partners.  Limiting the
references to a half rotation was rejected.  It raised the first-pass error on the off-axis rod
from 0.03 to 0.3 channels.  The cause is the interpolation of the partner views.  A pair compared
from one side carries that error unopposed, and the two sides cancel it when both are
references.  That cancellation holds at a stride only when the partner of a reference view is
itself a reference, which a stride that divides half the view count gives.  The measurement is
not in an archived record, because the half-rotation code was removed; the module's docstring
states it.  At stride 4 the memory saved is about 56 percent, not a factor of four, because the
partner band does not shrink.

**The plan is corrected.**  The plan's "Scoring" subsection now states the conjugate-ray relation
as `beta + pi - 2 gamma` in mbirtorch's conventions.  It carries the derivation from the
projector's coordinate expressions and the numerical check.  A "Corrections after acceptance"
list records the change, and it records that the plan's fallback for a short scan is under
review in Increment 3.  Negating the view angles does not change the relation.
`apply_calibration` negates them for a reversed rotation direction, and the relation holds
because it is a property of how the projector maps an angle value to a source position.

**The trimmed mean drops the worst tenth of view pairs.**  This is a design note.  The score is
a mean over view pairs.  A trimmed mean drops the worst tenth of the pairs before averaging, so
that a few corrupted views do not move the estimate.  The pairs dropped are chosen once, at the
best candidate of a coarse grid scored on every pair, and every candidate is then scored on the
same pairs.  Choosing them anew at each candidate would lower each candidate's score by a
different amount and flatten the minimum.  No corrupted-view test was run.  The plan's
validation section zeroes five percent of the views on a real scan, and that check is not
assigned to an increment.

**Two signatures differ from the plan, as Greg accepted.**  `estimate_det_channel_offset` takes
`det_rotation` and `num_rows` beyond the plan's arguments.  `conjugate_difference` takes
`det_channel_offset`, `det_rotation`, `reduction`, and `num_rows` in place of the plan's `value`.
One behavior differs as well.  The plan says `method='auto'` warns and falls back on a scan
without a full rotation, and the delivered function raises an error and names the reason, until
Increment 3 settles the fallback.

## Increment 3, as Greg decided it

Greg answered the three questions of the first draft.  The conjugate method will be extended to
short scans.  The derivative-filter method is not needed for anything listed, and Greg asked
whether it is useful for something else.  Offset scans are deferred.  Three cautions from the
review of this page go with the first decision.

The short scan pairs fewer rays than the first draft said.  Take a scan over the angles from
zero to 180 degrees plus the full fan angle.  The opposite of the ray at view angle `beta` and
fan angle `gamma` lies at `beta + pi - 2 gamma`, and it is measured only when `beta` is less
than twice the sum of `gamma` and the half fan angle.  The paired rays therefore form a triangle
in the view and channel plane, not two full wedges.  The wedge of views that hold any paired ray
is twice the full fan angle wide, at each end of the scan.  The paired rays are a tenth of all
rays at a full fan angle of 20 degrees, and they sit toward one side of the detector and taper
to a single channel at the far edge of each wedge.  A parallel-beam scan over exactly a half
rotation pairs only its two end views, so it has no conjugate information.

The extension changes the score, not only the pairing.  The score compares every pair over one
rectangular region of channels, shifts the opposites with a circular Fourier shift over the
whole channel axis, and normalizes each pair by its own energy.  A pairing that varies by view
needs a mask per pair, per-pair means over different channel counts, and a trimmed mean that
does not select pairs by the size of their mask.

The extension pairs each ray from one side only.  The start wedge is compared against the end
wedge, and no pair is scored in both directions.  That is the configuration that raised the
first-pass error tenfold on a full rotation.  Two remedies are known: cubic rather than linear
interpolation along the view axis, which attacks the error, and scoring each pair in both
directions from the two bands already in memory.

The derivative-filter question is open rather than answered.  LEAP's statement is that the
derivative reconstruction is featureless noise when the scan covers 360 degrees or more.  That
is a statement about the score's floor, not about where its minimum sits on a short scan, and
Parker weighting before the filter is the usual way to restore the redundancy a short scan
lacks.  The conjugate method's cone-beam gate passed by a factor of 24 on synthetic data without
truncation, stripes, or real scans, so the derivative filter's role as a cone-beam initializer
may return with real data.  A measurement of the derivative score's minimum on a short scan
would settle its worth.

The plan's Increment 3 gates need revisiting.  The search machinery exists, and the cluster
record holds the wall times the plan asks for.  The plan's gate of about fifteen evaluations is
exceeded: the search costs 35 evaluations, and the second pass doubles that on cone beam.  The
plan's run-time estimates rest on fifteen.

## How the estimators work

Every ray of a scan over a full rotation is measured twice, once from each side.  A voxel at
transverse position `x` projects to channel `(x + det_channel_offset) / delta_det_channel`
measured from the detector center, where `delta_det_channel` is the channel pitch.  After a half
rotation the same voxel projects to the mirrored channel.  A view and its opposite view
therefore differ by a shift of twice the offset.  The offset estimator scores each candidate
offset by how well that shift aligns the two.

The opposite view depends on the fan angle in cone beam.  The ray at view angle `beta` and fan
angle `gamma` has its opposite at view angle `beta + pi - 2 gamma`, in the projector's sign
conventions.  Each channel is therefore paired with a different opposite view.  That view rarely
coincides with a measured view, so it is interpolated linearly between the two nearest.
Parallel beam is the case `gamma = 0`.  The fan angle of a channel depends on the channel
offset, so the first pass pairs at the model's offset and the second pass re-pairs at the first
estimate.  The second pass runs for cone beam when the first estimate differs from the pairing
offset by more than the search tolerance of a hundredth of a channel.

The cone-beam comparison uses only a band of rows around the central plane.  The central plane
is the plane through the source that is perpendicular to the rotation axis.  Opposite rays
through a point off that plane reach the detector at different heights.  The band is limited to
the rows where that height difference stays within one row across the object's support.  At the
test geometry that limit gives a band of five rows.  On a scan with a source-to-iso distance of
1000 mm and a support radius of 100 mm the limit gives 11 rows.  That is about one percent of a
1000 row detector.  The `num_rows` argument overrides the limit.

The comparison holds four arrays.  They are the band of every reference view, the band's
mirrored opposites, the band of the partner views, and the spectrum of the opposites.  Together
they are about four times the size of one band.  The sinogram is read in view batches, so no
full-size copy of it is made.

The score is a normalized mean squared difference.  The shift is applied in the Fourier domain,
which is exact for a band-limited signal.  The difference is taken over a fixed interior region
of channels, which excludes the channels the circular shift wraps, and it is divided by the mean
square of the views.  A trimmed mean over view pairs drops the tenth of the pairs that agree
worst.  Agreement is measured against each pair's own energy.

The search has a coarse pass and a golden-section polish.  The coarse pass evaluates 11
candidates in a window of four channels on each side of the model's value.  The polish narrows
the bracket around the best candidate to a hundredth of a channel.  The two take about 24
evaluations, and the coarse grid that chooses the kept pairs takes 11 more.  When the coarse
minimum sits at an edge of the window, the window moves to center on that edge, at the same
width, up to eight times, so the default search reaches offsets of about 36 channels.  The
window stops moving when the channels excluded for the circular shift would leave less than a
quarter of the detector to compare.

The rotation estimate scores each candidate angle in three steps.  It applies the angle to the
row band by cubic resampling about the detector center.  It pairs the rotated views with their
opposites.  It scores the pairs at the model's channel offset.  The search covers five degrees
on each side of zero, which is the cap LEAP places on its detector tilt, and it stops at 0.005
degrees.

## Scans without opposite views

The conjugate-view method needs every ray to be measured from both sides.  A scan over a full
rotation meets that need.  Three kinds of scan fall short of it, and each has a different
answer.

A full rotation with a few views missing is served now.  Each channel's partner view is
interpolated between the two measured views nearest its angle, so a gap of a few views only
widens that interpolation.  The coverage check refuses a scan only when the largest gap between
neighboring angles exceeds both three times the median gap and five degrees.  A scan whose
views advance by the golden angle over a full rotation has gaps of up to 1.6 times the median,
and a test recovers a known offset on it to 0.009 channels.  A scan missing a block of views on
one side has a one-sided set of pairs there, and nothing warns.

A short scan of 180 degrees plus the fan angle is Increment 3's work, as described above.

A scan of less than 180 degrees plus the fan angle has no opposite views at all.  A helical scan
has no opposite view at the same axial position.  For both, the plan's fallback is the residual
method of Increment 4, which applies to any geometry.  That method reconstructs once per
candidate value and scores the sinogram residual.  Until Increment 3 settles the short-scan
case, `method='auto'` raises an error on such a scan and names the reason.

## What was verified

`tests/test_geometry_calibration.py` gained 17 test cases for this increment.  The file's 40
cases pass in about 5 seconds once the compile cache is warm.  Every number in this section was
read from that file's output in this session, except where a record is named.

1. The parallel-beam test used 64 views, 16 rows, and 64 channels.  For a true offset of 1.3
   channels the estimate is off by 0.012 channels.  For a true offset of -2.2 channels it is off
   by 0.010 channels.  With Gaussian noise at 2 percent of the sinogram maximum the errors are
   0.012 and 0.007 channels.
2. The cone-beam test used 128 views at a full fan angle of 20 degrees.  For a true offset of
   1.3 channels the estimate is off by 0.012 channels.  For -2.2 channels it is off by 0.021
   channels.  These two errors are the cone-beam offset error the plan asks to record.
3. A roll of the cone sinogram by two channels moves the estimate by 1.97 channels, against the
   two channels expected.  A roll introduces no interpolation, so this checks the estimator's
   direction and scale on the data alone.
4. `conjugate_difference` at the true offset has a mean absolute value 8.0 times smaller than
   two channels away.
5. The rotation estimate recovers 2.0 degrees as 2.08 on parallel beam and 2.10 on cone beam,
   and -3.0 degrees as -3.02 and -2.96.  A rotation of zero is recovered to within 0.001 degrees.
   A rotation of 0.3 degrees displaces the edge pixel of this detector by 0.17 pixels, and it
   returns with the sub-pixel warning.
6. A true offset of 7.5 channels, outside the default window, is recovered to 0.001 channels
   after the window moves once.
7. The estimators raise an error on these inputs: a scan covering less than a full rotation, a
   helical scan, a multiaxis model, an unknown method name, a rotation beyond five degrees, and a
   rotation on a curved detector.  A full rotation with jittered spacing, a scan of two turns, and
   a golden-angle scan are accepted.
8. The search helper finds the minimum of a parabola to 0.001 in 26 evaluations.  It sets its
   two warning flags correctly: one when the minimum sits at an edge of the bounds, and one when
   the coarse curve has two minima.  The Fourier shift by one channel matches a roll to 2.4e-7.

Three experiment records hold the measurements behind the design, in
`plans/experiments/features/geometric_calibration/`.  `conjugate_offset_recovery.md` sweeps the
true offset from -3.5 to 3.5 channels on two phantoms, with five noise seeds each, shows both
passes, and varies the view stride and the band height.  `rotation_interpolation_bias.md`
compares three resampling kernels and two scores at three small rotations on four geometries,
and then measures the bias against the size of the rotation.  `calibration_512_gautschi.md`
holds the cluster job.

The conjugate-ray sign was checked in three ways.  A reviewing agent derived it from the source
position implied by `cone_beam._cone_pixel_xy_mag` and confirmed it numerically over 200 random
rays.  The offset-recovery experiment repeats the estimate with the sign flipped.  The flipped
sign raises the score minimum by a factor of 5.9 on the Shepp-Logan phantom and 186 on the rod.
It moves the estimate by 0.23 channels on the Shepp-Logan phantom and by 1.9 channels on the
rod.  The score minimum is the sensitive check.  The flipped estimate on the Shepp-Logan phantom
is still within the plan's 0.5 channel gate for cone beam, so that gate alone would not detect a
wrong pairing.

## What the figures show

`estimators_in_action.py` in `plans/experiments/features/geometric_calibration/` draws four
figures.  The scan is synthetic cone beam with 256 views, 32 rows, and 128 channels, at a full
fan angle of 20 degrees.  Its true offset is 2.3 channels and its true rotation is 2 degrees.
The offset search on the rotated data gives 2.43 channels, with every evaluation shown on the
score curve.  The difference between a view and its opposite shows doubled edges at an offset of
zero.  At the estimate the edges are faint, and the uncorrected rotation leaves them.  The
rotation search at that offset gives 1.96 degrees.  With the rotation corrected, the offset
search gives 2.31 channels.  A parameter sweep reconstructs the central slice at four offsets.
Ring artifacts appear at the wrong offsets, and the slice is sharp at the estimate.  LEAP was
not run on this scan; the cluster job compares the two on its own scans.

## The comparison with LEAP

The cluster job ran LEAP's `find_centerCol` and `estimate_tilt` on the same sinograms as the
estimators.  On the rotation the two agree above one pixel of edge displacement, and below it
both are biased, in opposite directions.  On the offset the module's error is within 0.004
channels at 512 channels and 0.001 at 1024.  LEAP's error is 0.000 to 0.003 channels at true
offsets of 0.0 and 7.5, and 0.017 to 0.024 at 1.3 and -2.2.  Those two groups lie on and off a
half-channel grid, which is the pattern a search discretized at half a channel would leave.
That hypothesis was not tested, so the difference is not claimed as an accuracy advantage.  LEAP
is four to six times faster, and it uses one detector row where the module uses a band and every
view.  The one case LEAP found and the module did not, the 7.5 channel offset, led to the moving
window described above.

## Limits of this evidence

Every measurement is on synthetic data, forward projected by the same projector that the
estimator's model describes.  The phantoms are the Shepp-Logan phantom, which is nearly centered,
and an off-axis rod.  The laptop runs used 64 to 256 views, 16 to 64 rows, and 64 to 128 channels
on the CPU with torch.compile off.  The cluster job used 512 and 1024 views and channels on one
H100, at one fan angle of 14.6 degrees, and the laptop gates used one fan angle of 20 degrees.
Five conditions were not tested: real scan data, stripe artifacts, beam hardening, lateral
truncation, and corrupted views.  Peak memory was not measured.  A sinogram that is already
divided across several devices is refused, so the estimators run on the host, and the
estimators' arrays are numpy.

The wall times on the cluster are 0.3 to 0.7 seconds for the offset at 512 channels and 1.9 to
4.0 seconds at 1024, and 2.1 to 2.8 seconds for the rotation at 512 and 9.4 to 11.0 at 1024.
The offset's range within a size is the second pass, which runs at every nonzero offset and
doubles the time.  The 1024 times are about five times the 512 times.

The tilted test data for the kernel comparison were generated at four times the detector
resolution, and whether four times is enough was not checked.  The opposite-direction errors of
the two estimators on the cluster are evidence against a bias in that generation.

## Independent code review

A reviewing agent read the conjugate-view code against the projector source before the
experiments were final.  It verified seven things: the conjugate-ray sign, the offset term of the
fan angle, the direction and size of the shift, the interpolation of the opposite view, the
row-band rule, the search, and the absence of any full-size allocation.

The reviewer found four defects, all now fixed.  Before the fix, the check on angular coverage
refused a scan of two turns and a full rotation with irregular spacing.  Before the fix, a
rotation on a curved detector was refused at one entry point but accepted at another.  Before
the fix, the trimmed mean chose its pairs at each candidate.  Before the fix, the row band was
centered half a row low for an odd row count.

Each draft of this page and its records had a panel of three reviewers, for accuracy,
reasoning, and style.  The first panel asked for the measurement of the bias against the size of
the rotation, which reversed the deferral, and for the offset sweep, which found the scale
error.  The second panel found the stale numbers the fix had left behind, asked for the
first-pass column that shows what the second pass does, showed that a growing search range
coarsens its grid and eats its comparison region, which led to the moving window, and corrected
the short-scan geometry above.

## Files staged for commit

Two files are staged in mbirtorch: `mbirtorch/preprocess/geometry_calibration.py` and
`tests/test_geometry_calibration.py`.  In this repository the staged files are this page, the
corrected plan, the implementation prompt for Increment 3, and these in
`plans/experiments/features/geometric_calibration/`: `conjugate_offset_recovery.py` and its
record, `rotation_interpolation_bias.py` and its record, `estimators_in_action.py`, and
`calibration_512_gautschi.py` and its record.  This repository ignores `.png`, `.sbatch`, and
`.jsonl` files.  The four figures are therefore regenerated by `estimators_in_action.py`, and the
batch file's lines are given in the job's record.

## What is left

Increment 3 is next, and `implementation_prompt.md` in this directory starts it.  Increment 5's
driver and joint grid depend on both estimators of this increment, and both now exist.
