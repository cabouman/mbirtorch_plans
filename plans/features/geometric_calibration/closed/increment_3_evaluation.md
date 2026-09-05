# Geometric calibration: the accuracy of the current estimators against Increments 3 to 5

Date: 2026-09-04, with a status section added 2026-09-05.  Status: Greg chose two of this page's
options on 2026-09-04, the real-scan validation and Increment 6, and both are complete.  The work
they led to overturned some of this page's claims, and the section "Status on 2026-09-05" below
says which.  The rest of the page stands as the evaluation that was made before that work.  Greg
asked for the evaluation on 2026-09-04: before Increment 3 begins, weigh the accuracy of the
existing estimators for the channel offset and the detector rotation against the benefits and the
complexity of Increments 3 to 5.  The evaluation was reviewed by a panel of three, for accuracy
against its sources, for reasoning, and for style, and the panel's findings were applied.

## Status on 2026-09-05

Greg chose items 1 and 2 of "Decision for Greg" on 2026-09-04, and both are done.  The real-scan
validation grew into five cluster jobs, each with a record in
`plans/experiments/features/geometric_calibration/`: `real_scan_validation.md`,
`real_scan_followup.md`, `real_scan_rotation_check.md`, `real_scan_rotation_recon.md`, and
`real_scan_leap_tilt.md`.  Increment 6 is implemented and passes its three gates, and its findings
page is `increment_6_findings.md` in this directory.  Everything is staged in both repositories and
not committed.  Item 3, the reduced Increment 5, was not built.  Item 4's condition has been met,
because a real short scan exists.  Item 5, stopping after Increment 6, was not taken.

Three of this page's conclusions held on real scans.  The offset estimator passed every gate on the
four scans the conjugate-view method accepted: a roll error within 0.048 channels, robustness
differences within 0.069 channels, and agreement with the vendors' offsets within 0.074 channels
(`real_scan_validation.md`).  LEAP's `find_centerCol` agreed with it to 0.001 channels on the Zeiss
scan and read 0.21 to 0.98 channels higher on the three NSI scans, for a reason not yet known.  The
plan's first release needed only Increment 6, and Increment 6 delivered it.  The residual-score
probe's finding that a thin slab hides the minimum also held: the direct-residual score keeps the
whole axial extent in every later use.

Four of this page's conclusions did not hold, and the first matters most.  This page said the
detector rotation estimator was accurate where the rotation matters and that its residual at the
detector edge was small in absolute terms.  On the NSI scan of an artifact phantom without metal the
estimator read 0.047 degrees where the vendor's geometry report gave 0.167.  A job that added known
rotations of 0.25 to 2 degrees to the real sinogram showed the estimator following them with a slope
of -1.00 and residuals of a few thousandths of a degree, so its response is right
(`real_scan_rotation_check.md`).  Direct reconstructions of the slices far from the central plane
at four rotations then showed the vendor's 0.167 degrees to be right: the phantom's dark line is
deepest and narrowest there, and blurred at 0.044 (`real_scan_rotation_recon.md`).  The estimator's
zero point is therefore wrong by 0.12 degrees on that scan, which is 1.6 pixels at the detector
edge, and by about 0.02 degrees on the same phantom with a metal insert.  The cause is the
estimator's construction: it compares a row band of 5 to 11 rows, within which the only signal a
rotation leaves is the vertical displacement at the edge channels, and this phantom's structure runs
along the rows.  The module's warning and docstring and the docs now tell a user to prefer a vendor
tilt when the reader supplies one and to check the far slices.  LEAP's `estimate_tilt` does no
better: its cost is monotone over the range where the rotation lies on the NSI scans, with the full
detector height or a band, and it returned its search bound or a drift on every real scan
(`real_scan_leap_tilt.md`).

The second conclusion that did not hold is the alternative recorded under "Alternatives worth
recording", the rotation read from the offset's variation across row bands.  On the real NSI scans
that method followed the added rotations with slopes of 0.70 to 0.80 and its residuals grew with
the rotation, so it does not measure a rotation on cone-beam data and should not be built
(`real_scan_rotation_check.md`).  Its near-center fits had given 0.174 to 0.176 degrees, near the
vendor's value, but with an asymmetric trend that the cone angle explains.

The third is this page's account of the roll-recovery test.  The test cancels any bias that moves
with the data, so it detects the direction and scale of the estimate and a gross failure, and not a
bias from stripes or beam hardening.  The validation job therefore ran the plan's robustness pairs
and the vendor and LEAP comparisons as well, and the gate table above rests on all three.

The fourth is the cost of Increment 4's direct form.  This page said the direct-residual score could
serve Increment 4 in about a day, on a synthetic helical probe with a contrast ratio of 12.  On the
real short scan `z62` that score has its minimum within 0.12 channels of the vendor's offset, but the
score two channels away is only 1.015 to 1.018 times the minimum, at the module's default filter
widths, so a search cannot use it there (`real_scan_followup.md`).  That is the only real measurement
of the score, and it is thirty times shallower than its synthetic counterpart, so the synthetic
helical ratio should not be treated as a design input until the filter widths are swept.

Three further facts came from the real scans.  The Zeiss scan `z62` covers 218 degrees, so a real
short scan exists and the condition this page set for Increment 3 has been met.  The direction check
gave the wrong answer with a margin ratio of 1.05 on the 200-view NSI scan at its default binning,
where its score is dominated by pixel-scale noise, and the right answer at a ratio of 3.72 with a
bin factor of 8; on the Zeiss scan the ratio fell as the binning or the filter widths grew, so a
sound rule must require agreement across settings rather than a threshold on one
(`real_scan_followup.md`).  The offset estimate on the NSI scans moves by 0.10 to 0.14 channels
between no rotation and the vendor's 0.167 degrees, which is larger than the coupling estimate under
"The coupling between the two estimates" and puts the offset 0.15 to 0.18 channels from the vendor's
value at the right rotation, still inside the 0.25 channel threshold (`real_scan_rotation_check.md`).

The open decisions are the five listed at the end of `increment_6_findings.md`: what to do about
the rotation estimator's zero point, with a far-slice sharpness sweep now the favored remedy;
Increment 3 for the short scan that exists, together with short-scan weighting for the direct
reconstruction; a multi-setting rule for the direction check; the reduced Increment 5; and the two
corrections to the plan's working copy noted below.  The depot's remaining Zeiss scans have not been
read, and the cause of LEAP's offset difference on the NSI scans has not been tested.


## Sources and units

Every number on this page was read in this session from the record cited beside it.  Two records
are in this directory: `increment_1_findings.md` and `increment_2_findings.md`.  Five more are in
`plans/experiments/features/geometric_calibration/`: `conjugate_offset_recovery.md`,
`rotation_interpolation_bias.md`, `calibration_512_gautschi.md`, `direction_score_contrast.md`, and
`residual_score_probe.md`.  The last of those was run for this evaluation.  Code citations refer
to the `geometric_calibration` branch of mbirtorch at commit `4781600`.

This page uses two units.  An offset is given in channels, which are detector pixels along the
channel axis.  A rotation is given in degrees.  The edge displacement of a rotation is the distance
the rotation moves the edge pixel of the detector, in pixels.  A gate is the accuracy threshold the
plan sets for accepting an increment.

## The answer

The two estimators of Increment 2 are already more accurate than every gate the plan sets for
Increments 3 to 5.  That comparison holds on the scans the estimators accept, which are full
rotations of a parallel-beam or cone-beam scan.  `estimate_det_channel_offset` recovers a known
offset to within 0.023 channels on synthetic data at every detector size tested.  On the cluster,
at detector widths of 512 and 1024 channels, it recovers offsets of up to 2.2 channels to within
0.004 channels.  The plan's gate for the residual method of Increment 4 is 0.25 channels.
`estimate_det_rotation` leaves a residual of at most 0.054 pixels at the edge of the detector in
every cluster case.  No method measured so far has shown better accuracy than these two on a full
rotation.  The residual score probed for this evaluation matched the conjugate-view estimator's
error on the same sinogram, at 0.009 channels with the opposite sign.

Increments 3 to 5 therefore add coverage and convenience rather than accuracy.  Increment 3 adds
coverage of short scans of 180 degrees plus the fan angle.  Increment 4 adds coverage of helical
scans, multiaxis scans with elevation, and scans of less than a half rotation, and it would cover
short scans as well.  Increment 5 adds the one-call driver.  The driver carries the one accuracy
gain among the three: on a scan with a detector rotation, its third step, the offset estimate
repeated after the rotation is corrected, moved the offset estimate by 0.12 channels in the one
synthetic case measured.  A user can take that step by hand today.

No scan that needs Increment 3 or 4 has been named.  The NSI reader usually produces full
rotations, and it has no helical path.  Both readers pass the scan's own angular range through, so
a short scan is possible from either.  None of the scans on the cluster's depot has been
identified as a short scan.  A user with a short or helical scan is not without a path today,
because `parameter_sweep` accepts those scans and reconstructs one slice per candidate for the
user to judge by eye.  What such a user lacks is an automatic estimate.

A probe run for this evaluation shows that Increment 4 can be simpler than the plan describes.
The module's existing direct-residual score, applied over the whole axial extent, recovered a
known offset on a synthetic helical scan to 0.010 channels.  That run used no iterative
reconstruction and froze no reconstruction settings.  On a thin slab the same score had a shallow
minimum.  Increment 1 predicted that result.  On a short scan reconstructed without Parker
weighting the minimum was also shallow.

The largest gap in the evidence is that every measurement is synthetic.  A cluster job on real
scans from depot would close part of that gap.  The plan's roll-recovery test detects the
direction and the scale of an estimate and any gross failure on real data.  It does not detect a
bias from stripes or beam hardening, because those move with the data when the sinogram is
rolled.  The plan's robustness checks and a comparison with the vendor's values and with LEAP
are what test those biases, and the same job can run them.

The plan's first release needs none of Increments 3 to 5.  The plan defines that release as three
items: `parameter_sweep`, `estimate_det_channel_offset`, and the two rewritten FAQ answers.  The
two functions exist.  The FAQ answers are Increment 6.

The recommendation is to do three pieces of work in order: the real-scan validation job, then
Increment 6, then a reduced Increment 5.  Increments 3 and 4 should be deferred until a scan that
needs one of them is named.  The section "Decision for Greg" gives each option with its cost, its
gate, and what follows from each outcome.

## What the estimators deliver today

### The channel offset

The offset estimate is accurate to a few hundredths of a channel on every synthetic case measured.
The table gives the largest error in each record.

| record | geometry | detector width, channels | full fan angle, degrees | true offsets, channels | noise | largest error, channels |
| --- | --- | --- | --- | --- | --- | --- |
| `increment_2_findings.md`, the tests | parallel | 64 | 0 | 1.3 and -2.2 | none, and 2 percent | 0.012 |
| `increment_2_findings.md`, the tests | cone | 64 | 20 | 1.3 and -2.2 | none | 0.021 |
| `conjugate_offset_recovery.md` | cone, two phantoms | 64 | 20 | -3.5 to 3.5 in seven steps | none, and 2 percent | 0.023 |
| `increment_2_findings.md`, the tests | cone, golden-angle views | 64 | 20 | 1.3 | none | 0.009 |
| `calibration_512_gautschi.md` | cone | 512 | 14.6 | 0.0, 1.3, and -2.2 | none, and 2 percent | 0.004 |
| `calibration_512_gautschi.md` | cone | 1024 | 14.6 | 0.0, 1.3, and -2.2 | none, and 2 percent | 0.001 |

The cluster job also ran a true offset of 7.5 channels, and that case failed at the time of the
job.  The search window was then fixed at four channels on each side of the model's value, so the
search stopped at the edge of the window and returned 4 channels (`calibration_512_gautschi.md`).
The window now moves to center on the edge where the coarse minimum sits.  With the moving window,
the test suite recovers a true offset of 7.5 channels to 0.001 channels on a 64-channel detector
(`increment_2_findings.md`).  The moving window has not been run at 512 or 1024 channels.

The offset estimate is insensitive to noise and to the sampling choices.  Noise at 2 percent of
the sinogram maximum changes it by less than 0.005 channels (`conjugate_offset_recovery.md`).  The
view stride and the height of the row band change it by at most 0.008 channels (the same record).

LEAP's `find_centerCol` is less accurate on the same sinograms.  On the cluster it erred by 0.017
to 0.024 channels at offsets of 1.3 and -2.2, and by 0.003 or less at 0.0 and 7.5
(`calibration_512_gautschi.md`).

The plan's gates are far larger than these errors.  Increment 2's gates were 0.1 channels for
parallel beam and 0.5 for cone beam.  Increment 3's search gate is 0.1 channels.  Increment 4's
gate is 0.25 channels on cone beam.  The cluster errors are 60 to 250 times smaller than the
Increment 4 gate.  A gate is a threshold and not a prediction, so meeting a looser gate does not
by itself mean a method is less accurate.  The point is narrower.  Nothing in Increments 3 to 5
asks for more accuracy than the estimators already show.

Computing the offset estimate takes seconds.  On one H100 node it took 0.3 to 0.7 seconds at 512
channels and 1.9 to 4.0 seconds at 1024, on the host in numpy (`calibration_512_gautschi.md`).
The search evaluates the score 35 times, and the second pass on cone beam doubles that
(`increment_2_findings.md`).

### The detector rotation

The rotation estimator is accurate at the rotations that displace the detector edge by more than
a pixel.  Below one pixel of edge displacement its relative error is larger, and the absolute
residual stays small.  The record `calibration_512_gautschi.md` reports the error as a percentage
of the angle.  That percentage is 10 to 24 below half a pixel of edge displacement, 4.5 at 0.89
pixels, and within 0.5 from 1.34 pixels upward.

The quantity that affects a reconstruction is the residual displacement after the correction.
For the edge channel that residual is the error in the angle, in radians, times half the channel
count, and it is a vertical displacement.  The channel displacement of the edge row is the same
error times half the row count.  The two are equal on a square detector.  On the cluster's
detector, which has one eighth as many rows as channels, the channel displacement is one eighth of
the vertical one.  The table lists the vertical residual, computed from the record's estimates.

| detector width, channels | true rotation, degrees | estimated rotation, degrees | edge displacement, pixels | residual after correction, pixels |
| --- | --- | --- | --- | --- |
| 512 | 0.05 | 0.038 | 0.22 | 0.054 |
| 512 | 0.1 | 0.090 | 0.45 | 0.045 |
| 512 | 0.3 | 0.299 | 1.34 | 0.004 |
| 512 | 1.0 | 1.005 | 4.47 | 0.022 |
| 1024 | 0.05 | 0.044 | 0.45 | 0.054 |
| 1024 | 0.1 | 0.105 | 0.89 | 0.045 |
| 1024 | 0.3 | 0.301 | 2.68 | 0.009 |
| 1024 | 1.0 | 1.003 | 8.94 | 0.027 |

The residual is at most 0.054 pixels in every case.  That figure belongs to the five angles and
two detector widths tested, and it does not transfer as a constant.  The residual is the relative
error times the edge displacement, so it grows with the angle and with the detector width.  Two
bounds follow from the measured relative errors.  Below one pixel of edge displacement the
residual is at most 0.24 times the displacement, so it stays under a quarter of a pixel at any
detector width.  Above 1.34 pixels it is at most 0.5 percent of the displacement, which is 0.09
pixels for a rotation of 1 degree on 2048 channels and 0.9 pixels at the 5 degree cap on 4096
channels.  These bounds indicate that the residual is small wherever a rotation is likely, and
that it reaches a pixel only at the cap on the widest detectors.

Part of the tabulated residual is the search's stopping rule rather than resampling bias.  The
rotation search stops when its bracket is narrower than 0.005 degrees.  That width is 0.022
pixels of edge displacement at 512 channels and 0.045 pixels at 1024.  Five of the eight residuals
in the table are at or below that width.

Correcting a small rotation has a cost the estimate does not show.  The estimator resamples with
a cubic kernel, and `apply_calibration` applies the rotation with the bilinear kernel of
`_rotation_kernel`.  `rotation_interpolation_bias.md` records that the bilinear kernel smooths the
data by an amount that grows with the angle.  For a rotation below one pixel of edge displacement,
the user therefore weighs the blur of the correction against the misregistration it removes.  That
trade was not measured.

The module's warning in the sub-pixel regime says the bias is "up to 25 percent of the angle".
That statement is true, but it gives the user no way to judge whether the residual matters.  A
replacement is proposed under "Alternatives worth recording".

Computing the rotation estimate takes 2.1 to 2.8 seconds at 512 channels and 9.4 to 11.0 seconds
at 1024 (`calibration_512_gautschi.md`).  The search evaluates the score about 26 times, which
follows from its bounds of 5 degrees on each side and its stopping width.  Each evaluation
resamples the row band of every view with the cubic kernel.

### The coupling between the two estimates

The offset estimate and the rotation estimate are coupled.  One synthetic case had a true offset
of 2.3 channels and a true rotation of 2 degrees.  Before the rotation was corrected, the offset
estimate was 2.43 channels.  The rotation estimate at that offset was 1.96 degrees.  After the
rotation was corrected, the offset estimate was 2.31 channels (`increment_2_findings.md`, "What the
figures show").  The coupling at 2 degrees was therefore 0.13 channels, measured as the uncorrected
estimate minus the true offset.

The coupling shrinks in proportion to the angle, or faster.  It comes from two effects.  The first
is the finite height of the row band, across which a rotation shifts the channels by different
amounts.  That effect is linear in the angle, with a coefficient no larger than the band's
half-height, which is 2 to 5 rows for cone beam.  The second is the vertical displacement a
rotation gives the edge channels.  On the 128-channel detector of that case it is 2.2 rows at 2
degrees and 0.055 rows at 0.05 degrees.  The cluster record holds a measurement of the scaling.
LEAP's `find_centerCol` on the rotated sinograms returned offsets that grew from 0.03 channels at
0.05 degrees to 0.42 channels at 1 degree, and the record says the module's estimate shows the
same coupling (`calibration_512_gautschi.md`).  That growth is close to linear.  At 0.05 degrees
the coupling is therefore expected to be about 0.003 channels, which is below the search
tolerance of 0.01 channels.  The module's own coupling at a small rotation was not measured.

### What the estimators accept and refuse

The conjugate-view method needs every ray measured from both sides.  It accepts a parallel-beam or
cone-beam scan over a full rotation, with irregular spacing, a few dropped views, golden-angle
ordering, or more than one turn.  The rotation estimate also needs a flat detector.

The method raises an error on three kinds of scan: a scan over less than a full rotation, a
helical scan, and a multiaxis scan.  It does not serve an offset scan, whose detector is displaced
by hundreds of channels, because the search range is a few channels and the comparison excludes
only the channels the shift wraps (`geometry_calibration.py:1157-1159`).  A multiaxis model at
zero elevation is parallel beam, and it could be accepted with a small change.

Every entry point also refuses a sinogram that is already divided across several devices, through
`_sharding.reject_shards`, and the estimators run on the host in numpy
(`increment_2_findings.md`).  A multi-device reconstruction at production size holds its sinogram
in that divided form.  Calibration on such a run therefore needs the host copy of the sinogram,
and no increment addresses that.

## What the evidence does not cover

Every measurement is on synthetic data, forward projected by the same projector the estimator's
model describes.  The phantoms are the Shepp-Logan phantom and an off-axis rod.  The fan angles of
the estimator measurements are 14.6 and 20 degrees.  Five conditions were not tested: real scan
data, stripe artifacts, beam hardening, lateral truncation, and corrupted views.  The
conjugate-view method compares data with data, so a shared projector error biases it less than it
biases a method that fits a model.  Stripes and truncation act on the data themselves, and only
real data show them.

Peak memory and cost at production size were not measured.  The comparison holds four arrays of
the size of one row band and the spectrum of one, which is twice a band in bytes.  At 2048
channels, a band of 16 rows, and 4000 views, one band is about half a gigabyte, so the comparison
holds a few gigabytes of host memory.  The rotation search reads and resamples both bands at each
of its evaluations.  The cluster record shows the 1024-channel times at about five times the
512-channel times, and nothing above 1024 has been run.

A cluster job on real scans would test four things, and the roll-recovery test is only one of
them.  The plan's roll-recovery test rolls a real sinogram by a known integer number of channels
with `np.roll`, which introduces no interpolation, and it requires the estimate to move by the roll
to within 0.1 channels.  That test detects the direction and the scale of the estimate and any
gross failure on real data.  It does not detect a bias from stripes or from beam hardening,
because a rolled sinogram keeps every stripe and every hardening streak in the same place relative
to the object.  The plan's robustness checks detect those biases: an estimate with and without
beam-hardening correction, with and without stripe removal, and with five percent of the views
zeroed, each pair agreeing to 0.1 channels.  A comparison with the vendor's values, which the NSI
reader supplies, and with LEAP's `find_centerCol` on the same sinogram tests the absolute accuracy.
The job should also report each scan's angular range and detector shape, which settles whether any
scan on depot is a short scan.  Alignment of the views with `align_sino_views` must be off during
the job, because alignment removes part of the offset error the calibration is meant to find.

The scans for the job are on the cluster's depot, listed on 2026-09-04 in this session.  The Zeiss
scans are `/depot/bouman/data/ORNL/versa/ParAM-Round-1_Z62.txrm` at 6.9 GB,
`/depot/bouman/data/ORNL/versa/SiC-SiC_CompositeFFOV_tomo-A.txrm` at 13.7 GB, two Purdue BGA scans
in `/depot/bouman/data/Zeiss/purdue_BGA/` at 7.2 GB each, and
`/depot/bouman/data/Zeiss/purdue/Scan_tomo-A.txrm`.  The NSI scans are in
`/depot/bouman/data/Lilly/`: the client sample `NSI_sample_1`, and the tarballs
`demo_nsi_vert_metal_all_views.tgz` and `demo_nsi_vert_no_metal_all_views.tgz`.  The first tarball
is the public demo dataset, which an mbirjax experiment script downloads from
`https://www.datadepot.rcac.purdue.edu/bouman/data/demo_nsi_vert_metal_all_views.tgz`.  The second
was not checked.

Results on the client sample are published only with Greg's approval.  The Z62 or BGA scan and the
public NSI tarball are the candidates.

## What Increments 3 to 5 would add, and what each costs

### Increment 3: the conjugate-view method on short scans

The benefit is an automatic estimate on scans over 180 degrees plus the fan angle.  The plan's
risk section says that many industrial scans have that coverage.  Whether any available scan does
is not known.  The NSI reader computes each angle as the view index times the scan's angle step,
modulo 360 degrees, and it reads the angle step from the scan header
(`mbirtorch/preprocess/nsi.py:381` and `:279`).  Its comment on the header's total-angle field says
that value is usually 360 (`:252-253`).  The Zeiss readers take the angles from the scan file
(`mbirtorch/preprocess/zeiss.py:277`).  A short scan can therefore reach the estimators from either
reader, and neither reader treats it as the usual case.  The angular ranges of the depot scans
were not read.

The complexity is in the score, not in the pairing.  Increment 2's findings page gives the
geometry of the pairing.  The paired rays form a triangle in the plane of view angle and channel.
The views that contain paired rays lie in two wedges, each twice the fan angle wide.  At a 20
degree fan the paired rays are about a tenth of all rays.  The score today does three things: it
compares every pair over one rectangular region of channels, it shifts the opposite views with a
circular Fourier shift over the whole channel axis, and it normalizes each pair by its own energy.
A pairing that varies by view needs a mask per pair in all three places, and a trimmed mean that
does not select pairs by the size of their mask.  The coverage check must accept the scan when
enough rays have partners.

The accuracy is uncertain, and the plan's gate of 0.1 channels is plausible.  The pairs are
one-sided.  On a full rotation, one-sided pairing raised the first-pass error on the off-axis rod
from 0.03 to 0.3 channels (`increment_2_findings.md`).  Two remedies are known: scoring each pair
from both wedges, and interpolating the partner views with a cubic kernel along the view axis.
Neither remedy has been measured.  A tenth of the rays is still tens of thousands of measurements
on a real detector, so the noise floor is not the concern.  The systematic error is the concern.

Short-scan support needs changes beyond the estimator.  `recon_direct` applies no short-scan
redundancy weighting (`mbirtorch/cone_beam.py:806-808`).  A `parameter_sweep` on a short scan
therefore shows the direct reconstruction's own short-scan artifacts at every candidate, and a
user picking the sharpest slice would see them.  The plan defers the direct-reconstruction
weighting an offset scan needs.  A short scan needs a weighting of the same kind.

Deferring this increment leaves `method='auto'` raising an error on a short scan.  The plan's
corrections list records that behavior as provisional until Increment 3.  If the increment is
deferred, the error message should name `parameter_sweep` as the manual path.  The plan's first
release named `estimate_det_channel_offset` "by the conjugate-view or derivative-filter method",
and with the derivative filter dropped and Increment 3 deferred, the released function raises on
the scan class the plan's risk section calls common.

The plan's effort estimate is three days.  This evaluation agrees.

### Increment 4: the residual method

The benefit is an automatic estimate on every geometry: helical scans, multiaxis scans with
elevation, scans of less than a half rotation, and short scans.  No reader in the package produces
a helical scan.  A user with helical data from another source, or with a multiaxis scan, can
already calibrate by eye with `parameter_sweep`, and would need this increment to calibrate
automatically.

The plan's form of the method is the most complex of the three increments.  The plan names three
costs: four reconstruction settings that must be frozen, a reduced iterative reconstruction per
candidate, and about fifteen candidates per search.  The settings are frozen so that scores are
comparable across candidates.  The delivered search costs 35 evaluations rather than fifteen.  The
plan also names the method's weakness.  A reconstruction fits part of a geometry error, so the
residual understates that error.  Increment 1 measured a second weakness.  A thin slab leaves a
term in the residual that the slab cannot explain, because rays through the slab cross material
outside it.  For the direction check, a slab of 4 or 8 slices gave a ratio of 1.02 to 1.79
between the two directions, and the whole volume gave 2.17 to 7.28
(`direction_score_contrast.md`).  The check therefore keeps the whole axial extent.  Whether an
offset search has the same weakness was the question the probe answers.

The probe answers it, and the answer changes the form Increment 4 should take.
`residual_score_probe.py` scored the module's existing direct-residual score as a function of
`det_channel_offset` (`residual_score_probe.md`).  That score is the one `check_rotation_direction`
uses.  It reconstructs the reduced sinogram directly, forward projects the result, and high-pass
filters both.  The data were synthetic cone-beam data at a 20 degree fan with a true offset of 1.3
channels.  Four cases were run: a full rotation with a slab of 8 of 32 slices, the same scan with
the whole axial extent, a two-turn helical scan with axial shifts of 4 ALU each way over the whole
axial extent, and a short scan of 200 degrees over the whole axial extent.

The minimum sat at the true offset in every case, and its depth depended on the case.  The table
gives the fitted estimate and the ratio of the score two channels away to the score at the
minimum.  Every score in the table was computed over the central half of the rows.

| case | fitted estimate, channels | error, channels | score ratio, 2 channels above the minimum | score ratio, 2 channels below the minimum |
| --- | --- | --- | --- | --- |
| full rotation, slab of 8 slices | 1.274 | -0.026 | 1.54 | 1.52 |
| full rotation, whole axial extent | 1.292 | -0.009 | 15.07 | 15.85 |
| helical, whole axial extent | 1.290 | -0.010 | 12.08 | 12.76 |
| short scan, whole axial extent | 1.285 | -0.015 | 1.47 | 1.61 |

The thin slab fails in the way Increment 1 predicted.  The slab's score curve is the whole-extent
curve plus an extra term.  Across the coarse grid that extra term varies by 8.5 percent of the
term's mean.  Over the same grid the whole-extent curve varies by a factor of 20.5
(`residual_score_probe.md`).  These results indicate that a residual method must keep the whole
axial extent.  Each candidate therefore costs a reduced reconstruction over the whole axial extent
rather than over a slab.

The helical case is the one that matters.  The conjugate-view method refuses a helical scan, and
the direct residual on one gives an error of 0.010 channels and a ratio of 12.  On the short scan
the minimum is shallow, with a ratio of about 1.5.  Two effects are consistent with that shallow
minimum.  First, `recon_direct` applies no Parker weighting, so its reconstruction is wrong at
every candidate.  Second, a short scan measures each ray once, so a wrong offset produces less
inconsistency between views.  The probe does not separate the two effects.

Noise did not move the minimum.  Noise at 2 percent of the sinogram maximum moved the mean fitted
estimate over three seeds by at most 0.0011 channels, with a standard deviation of at most 0.0007
channels (`residual_score_probe.md`).

On the same full-rotation sinogram the conjugate-view estimator returned 1.309 channels, an error
of 0.009 channels.  That estimate took under a tenth of a second.  One residual evaluation took
0.29 seconds on the CPU, with no view stride and no binning, and one grid pass of 46 candidates
took about 13 seconds (`residual_score_probe.md`).  These results indicate that the residual score
is not an alternative to the conjugate-view method where opposite views exist, and that it is a
working estimator for the helical case.

The probe changes the design of Increment 4.  The plan's residual method reconstructs iteratively
with four frozen reconstruction settings, so that scores are comparable across candidates and so
that the reconstruction does not fit the geometry error.  The direct residual has neither problem,
because a filtered back projection fits nothing.  It trades the four reconstruction settings for
three score settings: the two widths of the high-pass filter, the fraction of rows scored, and the
name of the reconstruction filter.  Those three must be fixed and recorded in
`CalibrationResult.reduction`, for the reason the plan freezes its four.  The probe measured the
row fraction changing the ratio by a factor of 1.0 to 1.7, and it did not vary the filter widths.
On this evidence Increment 4 could be the direct residual over the whole axial extent, joined to
the search that already exists.  The module already contains that score.  The rough effort
estimate is one day in place of the plan's three.

Five limits apply to that conclusion.  First, the probe is an inverse crime: the direct
reconstruction was re-projected through the projector that made the data, so it reproduced the
data far better than it will reproduce a real scan.  The short-scan row shows what happens when the
direct reconstruction is wrong at every candidate, because its ratio fell from 15 to 1.5.  Beam
hardening, scatter, the cone-beam approximation at a real cone angle, and truncation all make a
direct reconstruction wrong in that way.  Second, the high-pass filter's widths are 3 rows and 15
channels, fixed in pixels (`sino_high_pass_filtering`, `preprocess/utilities.py:1131`).  On the
probe's 64-channel detector that filter keeps only the finest structure, and on a 2048-channel
detector the same widths pass a band near the pixel scale, where the noise is.  The ratios of 12
to 16 therefore do not transfer, and a cluster gate should sweep the two widths.  Third, the
fitted estimate was below the true offset in all eight probe rows, by 0.009 to 0.043 channels.  A
bias of one sign across four scan types points at the score or at the five-point parabola fit, and
it is the size of the accuracy being claimed, so it needs an explanation before a gate is set on
it.  Fourth, the cost without a slab is higher than the plan's figures.  The plan's per-evaluation
estimates assumed a slab of one sixteenth of the slices, and without the slab each direct
reconstruction and forward projection handles sixteen times the voxels.  By the plan's own figures
that makes an evaluation a few seconds rather than a few tenths at N = 1024, and 35 evaluations a
few minutes.  Fifth, the evidence is one phantom at one fan angle on a small detector.

The plan's gate is 0.25 channels on cone beam.  That gate is 10 to 250 times larger than the
errors measured for the conjugate-view estimator.  These numbers indicate that Increment 4 adds
coverage and not accuracy.  Its value stays zero until the project has a scan of one of the
geometries it serves.

### Increment 5: the driver

The benefit is the one-call workflow the plan shows.  `calibrate_geometry(ct_model, sino)` returns
a dictionary of results, and `apply_calibration` applies those results to the model and the
sinogram.  The driver automates a three-step sequence: the offset estimate, then the rotation
estimate, then the offset estimate again.  That sequence has been run by hand once on synthetic
data (`estimators_in_action.py`, reported in `increment_2_findings.md`).  It recovered both
parameters, with the offset within 0.01 channels and the rotation within 0.04 degrees.  The
offset error equals the offset search's tolerance.  The rotation error is eight times the rotation
search's stopping width, and it is a residual of 0.045 pixels at the edge of that 128-channel
detector.

The complexity is low.  The driver calls the two estimators in sequence and passes each estimate
to the next call.  Both estimators already accept those values through their `det_rotation` and
`det_channel_offset` arguments.  It is perhaps fifty to a hundred lines with its tests.  The
third step should run unconditionally, because it costs one offset estimate, and the driver should
record how far it moved the estimate.

Two points of the plan's Increment 5 need a decision.  The plan's open question is whether
`calibrate_geometry` should change the model or stay read-only.  This evaluation recommends
read-only, so that `apply_calibration` stays the one function that changes state, as the module's
docstring promises.  The plan's gate for the increment is the recovery of a joint perturbation of
both parameters, and no record has tested that at full size.  The cluster record perturbed the
offset with the rotation at zero and the rotation with the offset at zero.  A reduced Increment 5
should therefore include one synthetic joint case at 512 channels.

The joint five-by-five grid the plan attaches is unlikely to improve the result enough to justify
its cost.  The plan's own gate for the grid is that the grid improves on the coordinate-descent
result on at least one synthetic case.  If it does not, the plan drops the grid.  The one hand-run
case ended within the offset search's tolerance, and the coupling at a small rotation is expected
below that tolerance.  The grid would cost 25 evaluations of the rotation score, each a cubic
resampling of both bands.  The recommendation is to build the driver without the grid.

## Alternatives worth recording

The detector rotation could be read from the offset estimate's variation across row bands, with no
resampling at all.  A detector rotated by an angle shifts the content of a row at height v along
the channels by the angle times v.  Mirroring the opposite view reverses the sign of that shift.  A
view and its mirrored opposite at height v therefore differ by twice the offset plus or minus twice
the angle times v.  The offset estimator run on a row band at height v therefore returns the offset
plus or minus the angle times v.  A line fit through the estimates at several heights gives the
angle from its slope and the offset from its intercept.  The sign depends on the conventions of the
rotation kernel and the mirror, and it would be settled by the same numerical check that settled
the conjugate-ray sign in Increment 2.  The shift is in units of channels, so a detector whose
pixels are not square needs the ratio of the row pitch to the channel pitch in the fit.

Consider a detector of 1000 square pixels per side and a rotation of 0.05 degrees.  The estimates
at the top and bottom row bands would then differ by about 0.9 channels.  The offset estimator
resolves a difference of that size to a few thousandths of a channel.

The alternative is sound for parallel beam and for multiaxis at zero elevation, and it is doubtful
for cone beam.  In cone beam the module limits its row band to the rows where opposite rays through
the support land within one row of each other, which is 5 to 11 rows around the central plane
(`increment_2_findings.md`).  A band 500 rows from the central plane compares rays whose heights
differ by about 100 rows on a scan whose support radius is a tenth of the source distance.  That
height mismatch is odd in the height, which is the same parity as the slope the fit reads, so on an
object whose structure changes with height it would bias the slope.  The alternative is recorded
for two reasons.  It would remove resampling from the rotation estimate, so neither the sub-pixel
regime nor the choice of kernel would affect it.  It would also estimate the offset and the
rotation jointly from one fit, which would replace the coordinate descent and the joint grid of
Increment 5.

A multiaxis model at zero elevation is parallel beam.  `_geometry_kind` classifies it as
multiaxis, and the conjugate-view method refuses it.  Accepting it needs a guard on the elevations
and a change in `_view_angles`, which today flattens the model's two-column angle array and would
have to take the azimuth column.

The rotation warning should say two things in place of the percentage.  It should give the
residual in pixels at the edge of the detector, which the cluster record puts at 0.054 pixels or
less in the sub-pixel regime tested.  It should also say that applying the correction resamples
the whole sinogram bilinearly, and that for a rotation this small the blur of that resampling may
cost more than the misregistration it removes.

## Decision for Greg

Five options are open.  Items 1 and 2 are not independent, because the FAQ answers and the
warning text of Increment 6 should claim only what the real-scan job supports.

1. **The real-scan validation job.**  One cluster job on gautschi would do four things.  It reads
   one Zeiss scan and one public NSI scan from the depot, with view alignment off.  It runs the
   roll-recovery test on each.  It runs the plan's three robustness pairs and compares the
   estimates with the NSI vendor values and with LEAP's `find_centerCol`, for which a python
   environment with both LEAP and mbirtorch installed already exists on scratch.  It reports each
   scan's angular range and detector shape.  The gates are the plan's: a roll difference within
   0.1 channels, each robustness pair within 0.1 channels, and vendor agreement within 0.25
   channels.  If every gate passes, items 2 and 3 proceed and the accuracy claim above stands on
   real data.  If a robustness pair fails, the estimator needs the preprocessing step that pair
   names before it ships, and Increment 6 documents that.  If the roll test fails, the estimator
   has a defect on real data, and nothing else proceeds until it is traced.  The rough effort
   estimate is one day plus queue time.  Most of that day goes to the two scan readers and to
   writing the job.
2. **Increment 6.**  Increment 6 covers three items: the documentation, the Zeiss `det_rotation`
   argument, and the demo.  Four things found here belong in it: the preprocessing order with
   alignment last, the refusal of a divided sinogram, the revised rotation warning, and an error
   message on a short scan that names `parameter_sweep`.  This item completes the plan's first
   release.  The rough effort estimate is three days, as the plan says.
3. **A reduced Increment 5.**  This item builds the driver without the joint grid, read-only, with
   an unconditional third step, and with one synthetic joint case at 512 channels as its gate.
   The rough effort estimate is one day.
4. **Increments 3 and 4, deferred.**  Either is built when a scan that needs it is named, and the
   scan inventory of item 1 makes that trigger checkable.  Increment 3 is the better tool for a
   short scan, because it compares data with data.  Increment 4 is the only automatic tool for a
   helical or multiaxis scan.  The probe shows that Increment 4 can be the module's existing
   direct-residual score over the whole axial extent.  That score recovered a known offset on a
   synthetic helical scan to 0.010 channels, and built that way the increment is roughly a day of
   work.  Until either is built, a user with such a scan gets an error from the estimators and a
   manual path through `parameter_sweep`.  If Greg accepts the deferral, the plan's "Corrections
   after acceptance" list needs an entry, or the next session reads the plan and starts
   Increment 3.
5. **Stop after item 2.**  The first release is complete after Increment 6, and items 1, 3, and 4
   are optional relative to it.  Stopping there ships the estimators with their accuracy shown on
   synthetic data only.

Greg decides the order of items 1, 2, and 3.  This evaluation puts item 1 first because it tests
the claim the other items depend on.

## Notes on the plan's working copy

The plan's Increment 2 line now reads "COMPLETE 2026-08-04", and the date should be 2026-09-04.
That edit is in the working copy of this repository and is not committed.  The executive summary
still names the derivative-filter method as Increment 3, and the "Corrections after acceptance"
list records Greg's decision of 2026-09-04 to replace it.  The plan's list of changes in v3 says
the estimators handle an offset scan by restricting the comparison to the overlapping channels,
and the delivered module's docstring says an offset scan is not served.  None of these lines was
changed by this evaluation.

## Files

This page is new.  `residual_score_probe.py` and `residual_score_probe.md` are new in
`plans/experiments/features/geometric_calibration/`.  No file in mbirtorch was changed.
