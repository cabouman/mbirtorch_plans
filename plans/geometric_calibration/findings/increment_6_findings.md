# Geometric calibration, Increment 6: the documentation, the Zeiss `det_rotation` argument, the demo, and the real-scan validation

Date: 2026-09-04.  Status: implemented, awaiting Greg's review.  The code is on the
`geometric_calibration` branch of mbirtorch, on top of the Increment 2 commit `4781600`.  It is
staged and not committed.  The plan of record is `geometric_calibration_plan.md` in this
directory.  On 2026-09-04 Greg read `increment_3_evaluation.md` in this directory and chose two of
its options: the real-scan validation and Increment 6.  This page reports on both.  A panel of
three reviewers read it for accuracy, reasoning, and style, and their findings were applied.

## Outcome

The channel offset estimator is now validated on real scans.  The detector rotation estimator
follows a known rotation on real scans but placed its zero point 0.12 degrees wrong on one of them,
where direct reconstructions showed the vendor's recorded tilt to be right.  One real short scan
exists.  Increment 6 is complete.  Five items await Greg's decision and two smaller ones are open,
and all are listed at the end.

Increment 6 passes its three gates.  The documentation builds with zero warnings.  The new demo,
`demo_10_geometry_calibration.py`, runs on a CPU in a few seconds.  It recovers a known channel
offset of 1.7 channels as 1.692 and a known detector rotation of 1.5 degrees as 1.466.  A new test
passes a nonzero `det_rotation` through both Zeiss readers to the sinogram without reading any
scan file.

The real-scan validation passed every gate on all four scans it could measure.  The four are a
Zeiss Versa scan of a ball grid array and an NSI scan of an artifact phantom in three forms: at 200
views, at 1800 views, and at 1800 views with a metal insert.  Three of the four are therefore one
object.  The roll error was within 0.048 channels against a threshold of 0.1.  The three robustness
differences were within 0.069 channels against a threshold of 0.1.  The difference from the
vendor's offset was within 0.074 channels against a threshold of 0.25.  These three tests do not
measure the error against a true offset, because no real scan has one.  They show three things: the
estimator follows a known shift of the data; three changes to the data move it by less than a tenth
of a channel; and it agrees with the scanner's own calibration to a tenth of a channel.

The real scans showed three things the synthetic work could not.  First, the Zeiss scan `z62`
covers 218 degrees, so a real short scan exists.  Second, the detector rotation estimator read
0.047 degrees on the two NSI scans without metal and 0.149 degrees on the scan with metal.  The
vendor's geometry report gives 0.167 degrees for all three.  Third, the direction check gave the
wrong answer on the 200-view NSI scan and the right answer on the other three.  Its own warning
marked the wrong answer.

Two more jobs settled the rotation question.  The first added known rotations of 0.25 to 2 degrees
to the real NSI sinograms and asked whether the estimator followed them.  It did, with a slope of
-1.00 to -1.01 and residuals of 0.001 to 0.006 degrees on all three NSI scans.  The second, which
Greg asked for, reconstructed slices of the two NSI scans directly at four rotations.  The slices far
from the central plane are sharpest at the vendor's 0.167 degrees on both scans.  These results
indicate that the estimator responds to a change in the rotation correctly and places its zero point
wrongly on an object whose structure runs along the detector rows.  Its records are
`real_scan_rotation_check.md` and `real_scan_rotation_recon.md`.

## Units and terms

An offset is given in channels, which are detector pixels along the channel axis.  A detector
rotation is given in degrees, and its edge displacement is the distance the rotation moves the
edge pixel of the detector, in pixels.  A 360-degree scan is one whose views cover a full circle,
and a short scan is one whose views cover 180 degrees plus the fan angle.  A gate is a pass or fail
criterion of the plan, a threshold is the number a gate compares against, and the direction
check's margin is the ratio below which it warns.  An estimator is a
function of the module, and an estimate is the value it returns.

The band-slope method is a fit that the follow-up job defined, not a function of the module.  It
estimates the offset on several row bands and reads the detector rotation from the slope of those
estimates against the bands' heights.

Four quantities recur.  The roll test rolls the sinogram by a whole number of channels and
estimates the offset again.  The roll error is the change in the estimate minus the roll.  A
robustness difference is the change in the offset estimate when the data are changed in one way.
The three changes are these: stripe removal is applied, a quadratic beam-hardening term is added,
or five percent of the views are set to zero.  The conjugate-view score is the mean square of the difference between
each view and its mirrored opposite view, which the conjugate-view estimator minimizes.  The
direction check's margin ratio is the score of the worse rotation direction divided by the score
of the better one, and the check warns when that ratio is below 1.5.

## What was built

Increment 6 made seven changes.  Items 1, 2, 5, 6, and 7 are the plan's Increment 6.  Items 3 and
4 are additions from the evaluation's decision list.

1. `zeiss.get_sino_and_model` and `zeiss_tct.get_sino_and_model` take a `det_rotation` argument in
   radians, and both pass it to `scan_to_sino`.  The default of 0.0 keeps the readers' behavior.
2. `tests/test_reader_det_rotation.py` replaces each reader's scan loader with synthetic arrays.
   It computes the sinogram at a rotation of 0.02 radians and at zero rotation.  It checks that the
   first equals the result of applying `correct_det_rotation` to the second.  Both readers pass.
3. `estimate_det_rotation` raises a warning when the edge displacement is below one pixel.  That
   warning now says the estimate is uncertain there, gives the synthetic figure as synthetic, and
   says that on one real scan the estimate was a third of the vendor's recorded tilt.  It also says
   that applying so small a rotation resamples the whole sinogram bilinearly.  The text "up to 25
   percent of the angle" is gone.  The function's docstring carries the same scope on its bias
   figures.
4. The error the conjugate-view estimator raises on a scan without a full rotation now says that
   no automatic estimate exists for such a scan yet and names `parameter_sweep` as the manual path.
   It also says that on a short scan the sweep's slices carry limited-angle artifacts at every
   candidate, because the direct reconstruction applies no short-scan weighting.  The `method`
   argument docstrings say the same.
5. `docs/source/usr_preprocess.rst` gained a "Geometry calibration" section.  The section covers
   five topics: the preprocessing order; the automatic workflow, which is three calls followed by
   `apply_calibration`; the manual workflow with `parameter_sweep` and the slice viewer; the inputs
   the estimators accept and the inputs they refuse; and the API list.  It quotes one number, that
   the offset agreed with the vendor's value to better than a tenth of a channel on real scans, and
   it says the rotation estimate read well below the vendor's tilt on two scans.
6. `docs/source/demos_and_faqs.rst` lists the new demo.  Its two FAQ answers on blurry
   reconstructions and on artifacts previously told a user to change `det_channel_offset` by hand.
   They now refer the user to four functions: `estimate_det_channel_offset`, `parameter_sweep`,
   `check_rotation_direction`, and `apply_calibration`.
7. `demo/demo_10_geometry_calibration.py` makes a cone-beam scan whose true geometry differs from
   the model's by 1.7 channels of offset and 1.5 degrees of detector rotation.  The demo then runs
   five steps: a direct reconstruction that shows the artifacts; the direction check; the
   three-step estimation sequence; `apply_calibration`; and a second reconstruction with a sweep,
   which shows the artifacts removed.

Two more changes to the module came from the reviews of this page.  `check_rotation_direction`'s
docstring and warning said that a small margin comes from a narrow fan angle.  They now say that
on real scans the margin also depends on the scale of the high-pass filter, and that an answer
that comes with the warning is undecided.  Rendering the module's docstrings through autodoc also
surfaced five broken cross-references.  Two references to private functions became plain literals,
two `ndarray` types became `numpy.ndarray`, and a reference to `calibrate_geometry`, a function
that does not exist, was reworded.  The `CalibrationResult` entry excludes its seven field names
from autodoc, because the class docstring already documents each field.

## What was verified

The three gates of Increment 6 pass.  `make html` in `docs/` builds with zero warnings from a
clean build directory.  The build had zero warnings before the change as well.
`demo_10_geometry_calibration.py` runs to completion under the `Agg` backend and prints the
estimates quoted above.  `tests/test_reader_det_rotation.py` passes for both Zeiss readers.  The 50
tests of `tests/test_geometry_calibration.py`, `tests/test_preprocess_loaders.py`, and the new file
pass together in about 4 seconds.  The full suite was not run, because another session may be
running it.

The real-scan validation is four cluster jobs on gautschi, and each has a record beside its script
in `plans/experiments/features/geometric_calibration/`.  The peaks below are the scripts' own
`getrusage` peaks of host memory.  Slurm job 15925593 ran `real_scan_validation.py` in 21 minutes
with a peak of 50 GB, and its record is `real_scan_validation.md`.  Slurm job 15927130 ran
`real_scan_followup.py` in 35 minutes with a peak of 50 GB, and its record is
`real_scan_followup.md`.  Slurm job 15931130 ran `real_scan_rotation_check.py` in 44 minutes with a
peak of 41 GB, and its record is `real_scan_rotation_check.md`.  Slurm job 15933899 ran
`real_scan_rotation_recon.py` in 10 minutes with a peak of 41 GB, and its record is
`real_scan_rotation_recon.md`.  Every scan loaded at full resolution.

The first job ran seven steps on each scan: it read the scan; ran the three-step sequence of
offset, detector rotation, and offset again; rolled the sinogram by plus and minus 3 channels and
estimated the offset again; ran the three robustness cases; ran the direction check; ran LEAP's
`find_centerCol` and `estimate_tilt` on a band of 128 rows; and reconstructed one slice at six
candidate offsets.  The table gives the gate quantities per scan.

| scan | views | detector channels, count | roll error, channels | largest robustness difference, channels | module minus vendor offset, channels |
| --- | --- | --- | --- | --- | --- |
| NSI phantom, 200 views | 200 | 1496 | 0.003 | 0.026 | -0.074 |
| Zeiss ball grid array | 2401 | 1532 | 0.009 | 0.003 | +0.019 |
| NSI phantom, 1800 views | 1800 | 1496 | 0.003 | 0.018 | -0.053 |
| NSI phantom with metal | 1800 | 1496 | 0.048 | 0.069 | -0.063 |

The three NSI differences share a sign and a size.  All three lie between -0.05 and -0.08
channels.  A pattern of that kind is what a systematic error in either value would produce, and
its cause is not known.  The NSI offsets were estimated at the module's own detector rotation of
0.047 degrees, and the coupling analysis in the evaluation predicts an offset change of about 0.01
channels for a rotation error of 0.12 degrees, so the open rotation question does not account for
the pattern by that estimate.

LEAP's `find_centerCol` agreed with the module's offset on the Zeiss scan to 0.001 channels.  On
the NSI scans LEAP's offset was higher than the module's by 0.98, 0.46, and 0.21 channels, at 200
views, 1800 views, and 1800 views with metal.  The module stayed within 0.08 channels of the
vendor on all three.  The cause of LEAP's difference is not known.  The record proposed the
vendor's detector tilt, which the job held out of the sinogram, but that tilt shifts a row of the
128-row band LEAP saw by at most 0.19 channels, and the three scans carry the same tilt while the
difference varies by a factor of five.

## What the real scans showed

The findings are grouped by parameter rather than by job.

### The detector rotation

The conjugate-view estimator read 0.047 degrees on the two NSI scans without metal and 0.149
degrees on the scan with metal.  The vendor's geometry report gives 0.167 degrees for all three, and
the first job held that tilt out of the sinogram so that it could be estimated.  The two estimates
of 0.047 degrees are the same number to every recorded digit.  Four jobs in all bear on that
disagreement, and this subsection takes them in order.

The follow-up job estimated the rotation a second way, with the band-slope method.  A detector
rotation shifts each row's content along the channel axis in proportion to the row's height above
the central plane.  The offset estimated on a row band at height v therefore differs from the
offset at the central plane by the rotation times v, with the sign of the rotation.  The job
estimated the offset on row bands of 16 rows at up to eleven heights from 600 rows below the
central plane to 600 above.  The Zeiss detector's 968 rows cut that to nine.  It then fitted a
straight line to the estimates against the heights, once over all bands and once over the five
bands within 150 rows of the central plane.  The table gives the fits.  The sign of the fitted
rotation follows the fit's own convention, which the job did not tie to the module's, so the
magnitudes are what the table compares.

| scan | fit | number of bands | rotation from the slope, degrees, fit's sign | residual about the line, channels | module's rotation, degrees | vendor's rotation, degrees |
| --- | --- | --- | --- | --- | --- | --- |
| NSI phantom, 200 views | all bands | 11 | -0.113 | 0.242 | 0.047 | 0.167 |
| NSI phantom, 200 views | within 150 rows | 5 | -0.176 | 0.115 | 0.047 | 0.167 |
| Zeiss ball grid array | all bands | 9 | 0.096 | 0.306 | 0.016 | none |
| Zeiss ball grid array | within 150 rows | 5 | -0.006 | 0.009 | 0.016 | none |
| NSI phantom, 1800 views | all bands | 11 | -0.111 | 0.251 | 0.047 | 0.167 |
| NSI phantom, 1800 views | within 150 rows | 5 | -0.174 | 0.115 | 0.047 | 0.167 |

On the Zeiss scan the two methods agree.  The five near bands lie within 0.009 channels of a line
whose slope gives no rotation, and their estimates span 0.04 channels.  The fits over all bands are
worse on every scan, with residuals of 0.24 to 0.31 channels.  On the Zeiss scan the band 400 rows
below the central plane estimates -0.37 channels while every band within 250 rows of the plane
estimates between 0.52 and 0.58.  These results show the cone-angle bias the evaluation predicted
for bands far from the central plane.  On the two NSI scans without metal the near fit gives a
rotation whose magnitude is within 0.01 degrees of the vendor's, but the five near-band estimates do
not lie on a line.  On both scans they sit about +0.75, +0.40, 0, +0.10, and -0.24 channels from the
central band's estimate, from 150 rows below the central plane to 150 above.  The trend is about
three times as steep below the plane as above it, and a pure detector rotation gives a symmetric
trend.

The third job tested two explanations of the 0.047 degrees.  The first was that the score is nearly
flat in the rotation on this object, so that the search returned a point of its own grid.  The job
varied the search's bounds over 1, 2, and 5 degrees on each side and its coarse grid over 11 and 21
points.  The estimate stayed between 0.0469 and 0.0475 degrees on the two scans without metal and
between 0.147 and 0.150 degrees on the scan with metal.  These results reject the first explanation.
The second was that the estimator under-reads only in the sub-pixel regime.  The job added a known
rotation to the real sinogram with `correct_det_rotation` and estimated again.  It is the rotation's
counterpart of the roll test, and it needs no ground truth.  The table gives the estimates at four
added rotations and the straight-line fit of the estimate against the added rotation.

| scan | estimate at 0.25 degrees added | at 0.5 | at 1.0 | at 2.0 | slope | intercept, degrees | residual, degrees |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NSI phantom, 200 views | -0.204 | -0.458 | -0.958 | -1.958 | -1.001 | 0.044 | 0.001 |
| NSI phantom, 1800 views | -0.204 | -0.458 | -0.958 | -1.958 | -1.002 | 0.044 | 0.001 |
| NSI phantom with metal | -0.050 | -0.319 | -0.814 | -1.817 | -1.006 | 0.193 | 0.006 |

The estimator follows an added rotation with a slope of one.  The sign shows that it reports the
rotation to apply rather than the rotation the data carry.  The intercept is what the estimator
reads with nothing added.  It is 0.044 degrees on the two scans without metal and 0.193 degrees on
the scan with metal.  Taking 0.183 degrees as what the scan with metal carries, which is what its
three larger additions imply, two readings show the estimator's bias near one pixel of edge
displacement on real data.  The direct estimate on that scan was 0.149 degrees at 1.94 pixels,
about 19 percent low.  The addition of 0.25 degrees left about 0.07 degrees to read, and the
estimator read 0.050 degrees at 0.66 pixels, about 25 percent low.  On synthetic data the bias was
within 0.5 percent above 1.34 pixels.  The band-slope method run on the same rotated sinograms gave
slopes of 0.70, 0.71, and 0.80, and its residuals grew to 1.6 channels at 2 degrees.  These results
indicate that the band-slope method does not measure a rotation on these cone-beam scans.

The third job left one question, and the fourth answered it.  The estimator's slope is one, but an
added rotation cannot test its zero point.  Two explanations of the 0.15 degree gap between the
scans without metal and the scan with metal remained: the two acquisitions differ in alignment, or
the alignment is the same and the estimator's zero point depends on the object.  Greg asked for
direct reconstructions of the scans without metal at 0.044 and at 0.167 degrees, because a direct
reconstruction has no prior that could absorb a geometry error.  The fourth job reconstructed five
slices of each NSI scan at four rotations: none, 0.044, 0.167, and 0.19 degrees.  On the slice 752
rows above the central plane of the scan without metal, the phantom's dark horizontal line has a
depth of 0.0080 with no rotation, 0.0097 at 0.044 degrees, 0.0128 at 0.167 degrees, and 0.0121 at
0.19 degrees, and a width at half depth of 8, 6, 4, and 5 rows.  A gradient measure taken after a
two-pixel blur, which removes the pixel-scale noise that the resampling of each candidate smooths,
ranks 0.167 degrees first on every slice away from the central plane, on both scans.  These results
indicate that the vendor's tilt is right for both acquisitions and that the estimator's zero point
is wrong by 0.12 degrees on the scans without metal, which is 1.6 pixels at the edge of the
detector.  On the scan with metal the same measures put 0.167 degrees slightly ahead of 0.19, so
the estimator's zero point there is off by about 0.02 degrees.

The estimator's construction explains the zero-point error.  It rotates a row band of 5 to 11 rows
about the detector center.  Within that band the channel shift a rotation gives is at most the
rotation times half the band height, which is 0.015 channels at 0.167 degrees.  The only
displacement of any size is the vertical one at the edge channels, 2.2 rows at 0.167 degrees, and
that displacement changes the data only where the object's structure varies along the rows.  The
artifact phantom's structure runs along the detector rows, so the score's minimum there is set by
something other than the rotation.  The metal insert adds structure that varies along the rows, and
on that scan the zero-point error fell to 0.02 degrees.  The job's own unblurred sharpness measures
did not show any of this, because the unresampled slice scores highest on them at every height.
Those measures read the interpolation and not the geometry, which is why the fourth record reports
the blurred measure and the dark line instead.

The offset estimate moves with the rotation applied to the comparison, by 0.10 to 0.14 channels
between no rotation and 0.167 degrees.  With the vendor's rotation applied, which the fourth job
showed to be right, the offset estimated on the two scans without metal is 0.15 and 0.18 channels
from the vendor's offset, and on the scan with metal 0.053 channels.  The gate table above quotes
0.05 to 0.07 channels for the NSI scans, because the first job estimated each offset at the
module's own rotation.  The offset gate therefore still passes on the NSI scans at the right
rotation, at 0.18 channels against a threshold of 0.25, but by less than the table shows.  That
movement, up to 0.14 channels, is larger than the evaluation's coupling estimate of about 0.01
channels.  Part of it comes from the rotation being applied about the detector center, which sits
about 17 rows from the comparison band on this scanner.

Greg asked whether LEAP's tilt estimator does better on this rotation.  It does not.  A fifth job,
Slurm job 15936618, gave LEAP the full detector height, because the earlier jobs had given it a band
of 128 rows, and its record is `real_scan_leap_tilt.md`.  With all 1880 rows LEAP returned -1.03
degrees on the scan without metal and -0.010 degrees on the scan with metal, against the 0.167
degrees that direct reconstructions confirmed, and 0.090 degrees on the Zeiss scan, whose rotation
is below 0.02.  LEAP's own cost is monotone over the range from -0.4 to 0.4 degrees on the NSI scans,
and on the Zeiss scan its value at exactly zero rotation, the one angle LEAP does not resample,
stands above its neighbors.  These results indicate that LEAP's cost has no usable minimum on these
scans and that its resampling lowers the cost on its own, which is the bilinear-kernel behavior the
synthetic study found and the module's cubic kernel avoids.  The two estimators therefore fail
differently: LEAP's has no minimum, and the module's has a clear minimum at the wrong zero point.
LEAP's full-height result is also a caution for the first remedy under "Consequences", because a
conjugate comparison over the whole detector height did not find the rotation either.

### The short scan

The Zeiss scan `z62` covers 218 degrees, so the conjugate-view estimator refused it with the
message that names `parameter_sweep`.  The follow-up job measured three things on it.  The sweep
reconstructed one slice at seven offsets from two channels below the vendor's value to two above.
The sharpness measure peaks at the vendor's value, and its values two channels away are 2.2 to 2.5
percent lower.  The slices carry the limited-angle artifact of a direct reconstruction without
short-scan weighting.  LEAP's `find_centerCol` returned an offset 0.125 channels from the
vendor's, and LEAP's `estimate_tilt` returned 0.077 degrees.  The module's direct-residual score
was computed at 46 offsets on a reduced problem of 801 views by 512 rows by 512 channels over the
whole axial extent.  Its minimum sits at the vendor's offset on the coarse grid and 0.12 channels
above it by a parabola fit.  The score two channels away is 1.015 and 1.018 times the minimum,
against 1.5 on the synthetic short scan and 15 on the synthetic 360-degree scan.  Each evaluation
took 5.4 seconds.

These results are scoped to the settings tested.  The direct-residual score used the module's
default high-pass filter widths of 3 rows and 15 channels, fixed in pixels, on a detector binned to
512 channels, and the direct reconstruction applied no short-scan weighting.  The follow-up job
showed on the direction check that those default widths leave the score dominated by pixel-scale
noise on real data and that wider widths restore its margin, and the same score is used here.
The conclusion is therefore narrow.  At its default widths the direct-residual score has a
minimum near the vendor's value on this real short scan and a curve too flat to search.  A sweep
of the filter widths costs about four minutes per width and was not run.  The synthetic probe that
recommended Increment 4's direct form for helical scans rests on a contrast ratio of 12 on
synthetic data, and this is the first real measurement of that score, thirty times shallower than
its synthetic counterpart on a different scan type.  That ratio of 12 should not be treated as a
design input until a filter-width sweep separates the causes.

LEAP and the direct-residual score agree with each other on `z62` to 0.004 channels, and both
differ from the vendor's value by an eighth of a channel.  There is no ground truth, so neither is
known to be right.  Short-scan support also needs the redundancy weighting that the direct
reconstruction lacks, whichever route estimates the offset, because without it the sweep a user
looks at carries the limited-angle artifact at every candidate.

### The direction check

The direction check gave the right answer on three scans and the wrong answer on one.  On the
Zeiss scan and the two 1800-view NSI scans it kept the readers' angles, with margin ratios of
3.63, 9.07, and 17.32.  On the 200-view NSI scan it asked for the angles to be negated, with a
ratio of 1.05, and it warned.  The follow-up job ran the check on that scan at three bin factors.
At the default of 2 it gave the negated direction at 1.05.  At 4 it gave the readers' direction at
1.37, and at 8 it gave the readers' direction at 3.72.  It then scored both directions at a bin
factor of 4 with four filter widths, from the module's 3 rows and 15 channels up to eight times
those.  The ratio rose from 1.37 to 3.64, while the fraction of the reduced sinogram's energy the
filter kept rose from 0.9 to 42 percent.  These results indicate that the default widths leave the
score computed on the finest scale of the data, where noise dominates, on a scan of 200 views.

The Zeiss scan behaves the other way.  Its ratio fell from 3.63 at a bin factor of 2 to 2.78 at 4,
and from 2.78 to 1.51 as the filter widths grew.  Its discriminating structure is at the fine scale
the wider filters remove.  No single width serves both scans.  The check's own margin told the two
cases apart, because the only wrong answer came with the warning.

## Consequences

Four things follow.  The first three are decisions for Greg, and the fourth records what the docs
now say.

The detector rotation estimator has a zero-point error that depends on the object, and it needs a
change before it can be trusted without a vendor value.  On the phantom whose structure runs along
the detector rows the error was 0.12 degrees, and on the same phantom with a metal insert it was
0.02 degrees.  The estimator follows a change in the rotation with a slope of one, so the error is
in where its score's minimum sits and not in its response.  Two remedies are worth costing.  The
first widens the row band and corrects the cone-angle mismatch of the far rows before the
comparison, so that the horizontal displacement a rotation gives the object's edges enters the score.
The second uses a slice sweep like the fourth job's as the estimator, choosing the rotation whose
far slices are sharpest after a blur.  Neither was tried.  The band-slope method is not a remedy,
because it followed the same rotations with a slope of 0.7 and its residuals grew with the rotation.
Until a remedy is built, the docs and the module tell a user to prefer the vendor's tilt where one
exists and to check the far slices.

A real short scan exists, which is the condition the evaluation set for Increment 3.  The
conjugate-view estimator on short scans is the route the evaluation named for that case.  Two
things go with it.  The direct reconstruction needs short-scan redundancy weighting before either
the sweep or the direct-residual score is useful on such a scan, and that weighting may also
restore the direct-residual score's depth.  The evaluation's estimate of three days covered the
estimator alone.

The direction check should not be trusted on a single setting.  A rule that raises the bin factor
until the ratio clears the margin is not sound.  The Zeiss scan's ratio fell as the bin factor rose,
and raising the bin factor changed the 200-view scan's answer.  Such a rule can therefore stop at a
large margin and report the wrong direction.  A sound rule runs the check at several filter scales.
It reports a direction only when every setting that clears the margin gives the same direction, and
it returns undecided otherwise.  The check took 0.35 seconds at a bin factor of 8 and 11 seconds at
2.  These costs indicate that several settings cost less than one default run.

The docs section says what the real scans support.  It states three things: the offset agreement to
a tenth of a channel; that the rotation estimate followed added rotations with a slope of one but
read 0.044 degrees where direct reconstructions showed the vendor's 0.167 to be right; and that a
direction answer that comes with the warning is undecided.  The module's warning and docstrings say
the same, and both tell a user to prefer a vendor tilt and to check the far slices.

## Decisions made without asking

Greg could not be asked during the work, so these decisions were made and are listed for review.

- The robustness cases changed a band of 64 rows around the central plane rather than the whole
  sinogram.  The estimators read only that band, so the change reached them.  The beam-hardening
  case used a quadratic term sized to change the band's maximum by ten percent, as a proxy for a
  fitted correction.
- The NSI sinograms were built in the job by repeating the reader's steps with the vendor's tilt
  held out, because the reader applies that tilt and a tilt already applied cannot be estimated.
- The jobs requested two GPUs, because that request allocates 252 GB of host memory, and used one
  GPU.  The first job's memory rule read the node's memory rather than the job's allocation, so
  every scan loaded at full resolution.  The largest scan needed 101 GB by that rule, and the
  allocation covered it.
- The `det_rotation` argument sits before `verbose` in the four reader signatures.  The public
  signatures are keyword-only, so no caller can be affected.
- The Zeiss translation reader got the argument too, although the calibration module refuses its
  geometry, because the plan names both Zeiss readers.
- The demo uses `compile_mode='off'`, because compiling the projectors would take longer than the
  rest of the demo at its size.  It prints the direction check's small-margin warning, because the
  two geometry errors are still in the data when the check runs.
- The fourth job reconstructed five slices per scan at four fixed rotations rather than sweeping
  the rotation finely, and its two blur-based measures were computed on the Mac from the saved
  slices after the job's own measures proved to read the interpolation.
- The module's warning and docstring texts were revised twice: once for Increment 6 as the
  evaluation asked, and once more after the reviews of this page, when the real-scan results
  contradicted a synthetic bound the first revision had quoted.  A third revision followed the
  fourth job, and it tells a user to prefer a vendor tilt and to check the far slices.

## Files staged for commit

The following files are staged on the `geometric_calibration` branch of mbirtorch:

- `mbirtorch/preprocess/zeiss.py` and `mbirtorch/preprocess/zeiss_tct.py`, the `det_rotation`
  argument;
- `mbirtorch/preprocess/geometry_calibration.py`, the warning, error, and docstring texts;
- `docs/source/usr_preprocess.rst` and `docs/source/demos_and_faqs.rst`;
- `tests/test_reader_det_rotation.py`, new;
- `demo/demo_10_geometry_calibration.py`, new.

The following files are staged in this repository: `increment_3_evaluation.md` and this page in
this directory, and in `plans/experiments/features/geometric_calibration/` the scripts
`residual_score_probe.py`, `real_scan_validation.py`, `real_scan_followup.py`,
`real_scan_rotation_check.py`, `real_scan_rotation_recon.py`, and `real_scan_leap_tilt.py` with
their records, and `real_scan_rotation_recon_metrics.py`, which computed the fourth record's blurred
measures.  The batch files are transcribed in the records,
because this repository's `.gitignore` excludes them.  The plan's working copy carries an
uncommitted edit from another session, the Increment 2 line that reads "COMPLETE 2026-08-04", and
it is not staged here.

## What is left

Greg's review of Increment 6 and of the five records is the next step.  Five items await his
decision, and each is stated with its cost.

1. Decide what to do about the rotation estimator's zero point.  The two remedies under
   "Consequences" are each about two days with a synthetic and a real-scan check.  Leaving the
   estimator as it is, with the caution now in the docs, costs nothing.
2. Build Increment 3, the conjugate-view estimator on short scans, now that a real short scan
   exists.  The evaluation's estimate was three days for the estimator.  The short-scan redundancy
   weighting for the direct reconstruction is separate work of about a day, and both routes to a
   short-scan offset need it.
3. Revise the direction check to run at several filter scales and report a direction only when
   the settings agree.  That is about a day with its test.
4. Build the reduced Increment 5, the driver without the joint grid, which the evaluation's rule
   allows now that the real-scan gates passed.  That is about a day.  The driver should apply the
   vendor's tilt rather than the estimate when the reader supplies one, given item 1.
5. Correct two entries in the plan's working copy.  The Increment 2 line should read 2026-09-04.
   The "Corrections after acceptance" list has no entry yet for the order of work Greg chose on
   2026-09-04.

Two smaller items are also open.  The depot holds at least three Zeiss scans the jobs did not read,
and their angular ranges would say how common short scans are among the scans in hand.  LEAP's
offset differs from the module's by up to a channel on the NSI scans, and no test of the cause has
been run.
