# The short scan, the rotation from row bands, and the direction check: what `real_scan_followup.py` measured

Date: 2026-09-04.  Slurm job 15927130 on gautschi, one NVIDIA H100 80GB HBM3, torch 2.13.0+cu130,
LEAP 1.26, mbirtorch 0.0.2 at commit `4781600` on the `geometric_calibration` branch.  The job
asked for two GPUs so that it would hold 252 GB of host memory, and it pinned mbirtorch to one
device with `MBIRTORCH_NUM_DEVICES`.  `sacct` reported `COMPLETED`, exit code `0:0`, elapsed
`00:34:45`, and a batch-step `MaxRSS` of 30323.50M.  The script's own peak was 49.7 GB, measured
with `getrusage`.  `sacct` samples periodically, so the smaller of the two numbers is expected.
Every number below was read from the job's output in the same session.  That output is the log,
`/scratch/gautschi/buzzard/leap_cmp/real_scan_followup_15927130.log`, and the 89 JSON lines of
`results_real_scan_followup/real_scan_followup.jsonl` in the same directory.  The job also wrote
`z62_sweep.png` and `z62_sweep.npz` there.  This repository ignores such files, so this page is the
durable copy.

Units.  An offset is given in channels, and a channel is the detector pitch of the scan in
question.  Three pitches appear here: 0.127 mm on the NSI scans, 0.0135 mm on `z62`, and 0.1496 mm
on `bga`.  A rotation is given in degrees, and its edge displacement is the distance it moves the
edge channel of the detector, in pixels.  A band height is a distance along the detector's row
axis, in ALU, measured from the row the central plane reaches.  The estimators under test are
`estimate_det_channel_offset`, `estimate_det_rotation`, and `check_rotation_direction` in
`mbirtorch/preprocess/geometry_calibration.py`, called the module below.  A vendor value is the
value the scanner's own calibration recorded, as the reader for that scanner reports it.

## The answers

Part B measured the detector rotation a second way, and its result favors the vendor tilt without
settling the question.  The band fit within 150 rows of the central plane gives an angle of
magnitude 0.176 degrees on `nsi_small` and 0.174 degrees on `nsi_no_metal`.  The vendor tilt on
both scans is 0.167 degrees.  The two differences are 0.008 and 0.007 degrees, and the standard
error of each fitted angle is 0.036 degrees.  The module's `estimate_det_rotation` returned 0.047
degrees on both scans, which is 3.5 standard errors from the fitted angle.  Two things keep the fit
from settling the rotation.  The five near-band estimates do not follow a straight line.  On both
scans they sit about +0.75, +0.40, 0, +0.10, and -0.24 channels from the central band's estimate,
from 150 rows below the central plane to 150 above, so the trend is about three times as steep below
the plane as above it, and a pure rotation would give a symmetric trend.  The conjugate score computed
at the vendor's rotation on resampled data is also higher than the score at zero rotation on both
scans, and smoothing can only lower that score, so the view-pair comparison itself does not prefer
the vendor's rotation.  These results indicate that the module's resampling-based estimate
under-reads the rotation on these two scans if the vendor's value is right, and that the band fit's
asymmetry leaves open how much of its slope is the cone angle rather than the rotation.  This record
cannot say which value is right.

On `bga` the two methods agree with each other.  The band fit within 150 rows gives 0.006 degrees
and the module gives 0.016 degrees.  Both are below 0.02 degrees.  These results indicate that this
Zeiss scan carries no detector rotation worth correcting.  The Zeiss reader records no vendor tilt,
so there is nothing here to compare the two values against.

The sign of the fitted angle is opposite to the sign of the vendor tilt and of the module's
estimate.  The band fit gives -0.176 degrees on `nsi_small`, while the vendor records +0.167 degrees
and the module returns +0.047 degrees.  This job did not settle the sign convention between the band
slope and the module's rotation.  Only the magnitudes are compared above.

Part A measured the short scan `z62`, and two of its three methods agree on the offset to 0.004
channels.  LEAP's `find_centerCol` gives -0.802 channels and the residual score's fitted minimum
gives -0.806 channels.  The vendor value is -0.928 channels, which is 0.125 channels from LEAP and
0.122 channels from the residual score.  The sharpness measure over seven reconstructed slices named
the vendor value the sharpest of the seven.  These results indicate that the two search methods
agree with each other and disagree with the vendor value by about an eighth of a channel.  The
sharpness measure does not resolve that disagreement.

The residual score has a minimum on this real short scan, and that minimum is far shallower than any
synthetic case.  The score 2 channels above the minimum is 1.015 times the minimum, and the score 2
channels below it is 1.018 times the minimum.  The synthetic short scan in `residual_score_probe.md`
gave 1.47 and 1.61, and the synthetic full rotation there gave 15.07 and 15.85.  These results
indicate that the minimum sits in a plausible place and that it is the shallowest one this feature
has measured.  A minimum 1.5 percent deep cannot be located reliably on data whose noise floor is
not known in advance.  One evaluation of the score cost 5.4 seconds here, and the 46 evaluations
cost 248 seconds together.  LEAP's whole offset search on the same scan cost 1.06 seconds.

Part C found that the small margin the direction check gave on `nsi_small` comes from the pixel
scale of the reduced problem.  At a bin factor of 2 the check returned the negated direction at a
ratio of 1.05, which is the first job's answer.  At a bin factor of 4 it returned the direction as
given at a ratio of 1.37.  At a bin factor of 8 it returned the direction as given at a ratio of
3.72.  Wider filters at a fixed bin factor of 4 raise the ratio the same way, from 1.37 at the
module's default widths to 3.64 at eight times those widths.  At the default widths the filter keeps
0.9 percent of the reduced sinogram's energy on `nsi_small`.  These results indicate that the
default score on `nsi_small` is computed almost entirely on the finest scale of the data, where
noise dominates.  They also indicate that more binning or wider filters restore the margin and
change the answer to the direction as given, which is the answer every other real scan has returned.

More binning and wider filters do not raise the margin on `bga`.  The check returned the direction
as given at every setting on that scan, so the margin there was never in doubt.  Its ratio fell from
3.63 at a bin factor of 2 to 2.78 at a bin factor of 4.  It fell again from 2.78 at the default
filter widths to 1.51 at eight times those widths.

## The three questions

The first job is `real_scan_validation.py` in this directory, and its record is
`real_scan_validation.md`.  It ran the conjugate-view estimators on five real scans and left three
questions.  This job was written to settle them.

The three questions are these.  The Zeiss scan `z62` covers 218 degrees, so the conjugate-view
method refused it and nothing was measured on it.  On the NSI scans the module's rotation estimate
was 0.047 degrees and the vendor tilt was 0.167 degrees.  The module's own estimate cannot say which
of the two is right, because it applies each candidate angle by resampling the data.  On `nsi_small`
the direction check returned the negated direction at a score ratio of 1.05, and both scores were
near 1.4.  Two scores that close and that large point at something both directions share.

## The scans

The script's dataset list holds four of the first job's five scans.  `nsi_metal` is not among them.
Every scan loaded at full resolution.  The NSI sinograms were built inside the script rather than
through `nsi.get_sino_and_model`.  That reader applies the vendor's detector tilt to the sinogram,
and a tilt already applied cannot be estimated.  The script repeats the reader's steps with the tilt
held out and keeps it as the vendor value.

| dataset | scanner | views | rows | channels | pitch, mm | coverage, deg | vendor offset, channels | vendor tilt, deg | parts run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `nsi_small` | NSI | 200 | 1880 | 1496 | 0.1270 | 358.20 | -14.125 | 0.1672 | B, C |
| `z62` | Zeiss Versa | 801 | 2048 | 2048 | 0.0135 | 217.99 | -0.928 | | A |
| `bga` | Zeiss Versa | 2401 | 968 | 1532 | 0.1496 | 359.83 | 0.528 | | B, C |
| `nsi_no_metal` | NSI | 1800 | 1880 | 1496 | 0.1270 | 359.80 | -14.125 | 0.1672 | B |

The script chooses the parts by coverage.  A scan the conjugate-view method refuses gets Part A, and
a scan it accepts gets Part B.  `z62` was refused with the same message the first job recorded: `The
conjugate-view method needs views over a full rotation.  The angles cover 218.0 degrees, with a gap
of 142.0 degrees between neighboring views.`  Part C runs on `nsi_small` and `bga` only, which are
the two datasets the script's `DIRECTION_DATASETS` names.

The four datasets took these times, in the order of the table above: 77.0, 326.9, 1072.1, and 588.4
seconds.  Their peak GPU allocations, in the same order, were 4.38, 3.19, 9.59, and 0.18 GB.

## Part A: the short scan `z62`

### The offset sweep

The sweep reconstructs one slice at each of seven candidate offsets and scores each slice by a
sharpness measure.  The seven candidates are the vendor value and six steps from it.  The sharpness
measure is the mean squared finite difference along both axes of the slice divided by the mean
square of the slice.  `recon_direct` applies no Parker weighting, so its reconstruction of a short
scan is itself approximate at every candidate.

| step from vendor, channels | offset, channels | sharpness |
| --- | --- | --- |
| -2.0 | -2.928 | 0.0140121 |
| -1.0 | -1.928 | 0.0141892 |
| -0.5 | -1.428 | 0.0143029 |
| +0.0 | -0.928 | 0.0143640 |
| +0.5 | -0.428 | 0.0143317 |
| +1.0 | +0.072 | 0.0142666 |
| +2.0 | +1.072 | 0.0140436 |

The measure named the vendor value the sharpest, and the seven values span 2.5 percent.  The measure
falls off on both sides of the vendor value without a second peak.  These results indicate that the
sharpness measure prefers the vendor value over any candidate half a channel or more away.  The
sweep's finest step is half a channel, and the two search estimates below sit 0.12 channels from the
vendor value, so this sweep cannot separate them from it.

The figure `z62_sweep.png` is the durable record of what the sweep looks like, because this
repository ignores PNG files.  Its seven panels sit in a row of four and a row of three, one panel
per candidate, on a shared gray scale.  Each panel shows a bright, almost featureless disk on a dark
background, with pale and dark wedges above and below the disk.  Those wedges are the limited-angle
artifact of a scan that covers 218 degrees.  The seven panels look alike to the eye.

### LEAP on the same scan

LEAP was given the 128 central detector rows, rows 960 to 1088, in the sinogram order the LEAP
comparison established.  That order reverses the channel axis, so the offset below is negated back
into this package's convention.

| quantity | value |
| --- | --- |
| `centerCol` | 1024.3025 |
| offset, channels | -0.8025 |
| offset minus the vendor value, channels | +0.1252 |
| `find_centerCol` metric | 0.00040987 |
| tilt, degrees | 0.0772 |
| offset seconds | 1.06 |
| tilt seconds | 0.05 |

LEAP's `find_centerCol` may assume a full rotation, and nothing here checks whether it does.  The
run raised the LEAP warning `rebin::rebin_parallel - Warning: non-equi-spaced angles can only be
rebinned with bilinear interpolation` twice.  LEAP's tilt of 0.0772 degrees is well inside its five
degree search bound.  The two NSI tilts the first job recorded sat at that bound.  Nothing else here
measures a tilt on `z62`, so LEAP's value stands alone.

### The residual score against the offset

The residual score is `_direct_residual_score` in the module.  The reduced model reconstructs the
reduced sinogram directly and forward projects the result.  Both the reduced sinogram and the
projection are then high-pass filtered.  The score is the mean squared difference of the two
filtered arrays divided by the mean square of the filtered sinogram.  The mean is taken over the
central half
of the detector rows.  The probe kept every view and binned the detector by 4.  That reduction gave
a sinogram of shape (801, 512, 512) and a reconstruction of shape (512, 512, 512).  Setup took 7.2
seconds, and the 46 evaluations took 5.39 seconds each on average.

The coarse grid holds 25 candidates, in steps of 0.25 channels from the vendor value.

| step from vendor, channels | score |
| --- | --- |
| -3.00 | 0.155661 |
| -2.75 | 0.154914 |
| -2.50 | 0.154202 |
| -2.25 | 0.153533 |
| -2.00 | 0.152912 |
| -1.75 | 0.152348 |
| -1.50 | 0.151841 |
| -1.25 | 0.151396 |
| -1.00 | 0.151017 |
| -0.75 | 0.150708 |
| -0.50 | 0.150471 |
| -0.25 | 0.150308 |
| +0.00 | 0.150226 |
| +0.25 | 0.150227 |
| +0.50 | 0.150312 |
| +0.75 | 0.150477 |
| +1.00 | 0.150722 |
| +1.25 | 0.151044 |
| +1.50 | 0.151436 |
| +1.75 | 0.151894 |
| +2.00 | 0.152414 |
| +2.25 | 0.152990 |
| +2.50 | 0.153615 |
| +2.75 | 0.154283 |
| +3.00 | 0.154988 |

The coarse curve falls to a single minimum and rises on both sides of it.  The coarse argmin is at
+0.00 channels from the vendor value, with a score of 0.150226.  The fine grid, in steps of 0.1
channels from -1.0 to +1.0, has its argmin at +0.10 channels, with a score of 0.150216.  The
parabola through the five fine points centered on that argmin opens upward and has its minimum at
+0.1216 channels.  That fitted step puts the offset at -0.806 channels, which is 0.004 channels from
LEAP's -0.802 channels.

| quantity | value |
| --- | --- |
| coarse argmin, channels from vendor | +0.00 |
| fine argmin, channels from vendor | +0.10 |
| parabola window start, channels from vendor | -0.10 |
| parabola opens upward | yes |
| fitted minimum, channels from vendor | +0.1216 |
| fitted offset, channels | -0.806 |
| contrast ratio 2 channels above the minimum | 1.015 |
| contrast ratio 2 channels below the minimum | 1.018 |
| seconds per evaluation | 5.39 |

The two contrast ratios are 1.015 and 1.018.  The score therefore rises by 1.5 to 1.8 percent at 2
channels from the minimum.  The synthetic short scan in `residual_score_probe.md` rose by 47 to 61
percent at the same distance and at the same row fraction.  The synthetic full rotation there rose
by a factor of 15.  These results indicate that the residual score has a minimum on this real
short scan, in a plausible place and with the right shape.  They also indicate that the minimum is
about thirty times shallower here than on the synthetic short scan.  The whole coarse grid spans 3.6
percent of the minimum score, so a search on this scan works within a few percent of a floor whose
size it cannot know.  On the evidence here the minimum is usable as a check against another method
and is not usable on its own.

## Part B: the rotation from row bands

A detector rotated by an angle shifts each row's content along the channels in proportion to the
row's height above the central plane.  The offset estimated on a band of rows at height v therefore
equals the offset plus or minus the angle times v, in matching units.  The slope of the offset
estimate against the band height is therefore the rotation.  This method applies no resampling to
the data, which is what makes it independent of the module's own rotation estimate.
Each band is a window of 16 detector rows whose center sits a fixed number of rows above or below
the row the central plane reaches.  Every band keeps every view and bins nothing.

The module warns for every band that leaves the central plane out, with this message: `The
reduction's row window does not contain the row the central plane reaches, so the cone-beam
conjugate comparison is biased by the cone angle.`  The counts of that warning per scan are these:

- twice per band on ten of the eleven `nsi_small` bands;
- twice per band on ten of the eleven `nsi_no_metal` bands;
- once or twice per band on eight of the nine `bga` bands.

It raised no warning on the central band of any scan.  No other warning appeared in any band.  Every
band ran 24 evaluations and kept 90 percent of the view pairs.

### The bands

| dataset | center offset, rows | band height, ALU | estimate, channels | vendor difference, channels | score | seconds |
| --- | --- | --- | --- | --- | --- | --- |
| `nsi_small` | -600 | -76.250 | -13.278 | +0.848 | 0.686679 | 2.32 |
| `nsi_small` | -400 | -50.850 | -13.524 | +0.601 | 0.004841 | 2.26 |
| `nsi_small` | -250 | -31.800 | -13.740 | +0.385 | 0.011109 | 2.29 |
| `nsi_small` | -150 | -19.100 | -13.399 | +0.726 | 0.014869 | 2.28 |
| `nsi_small` | -75 | -9.575 | -13.737 | +0.388 | 0.001215 | 2.29 |
| `nsi_small` | 0 | -0.050 | -14.149 | -0.024 | 0.000482 | 2.30 |
| `nsi_small` | +75 | +9.475 | -14.047 | +0.079 | 0.000700 | 2.31 |
| `nsi_small` | +150 | +19.000 | -14.393 | -0.268 | 0.002717 | 2.26 |
| `nsi_small` | +250 | +31.700 | -14.566 | -0.441 | 0.000860 | 2.27 |
| `nsi_small` | +400 | +50.750 | -14.913 | -0.788 | 0.002914 | 2.27 |
| `nsi_small` | +600 | +76.150 | -15.741 | -1.615 | 0.003186 | 2.30 |
| `bga` | -400 | -59.840 | -0.374 | -0.902 | 0.055714 | 81.91 |
| `bga` | -250 | -37.400 | +0.522 | -0.006 | 0.118141 | 41.00 |
| `bga` | -150 | -22.440 | +0.582 | +0.055 | 0.076896 | 82.26 |
| `bga` | -75 | -11.220 | +0.549 | +0.022 | 0.082060 | 82.35 |
| `bga` | 0 | 0.000 | +0.548 | +0.020 | 0.001695 | 82.05 |
| `bga` | +75 | +11.220 | +0.541 | +0.014 | 0.088783 | 82.11 |
| `bga` | +150 | +22.440 | +0.546 | +0.019 | 0.109059 | 81.99 |
| `bga` | +250 | +37.400 | +0.530 | +0.002 | 0.108062 | 41.10 |
| `bga` | +400 | +59.840 | +1.731 | +1.204 | 0.052652 | 82.09 |
| `nsi_no_metal` | -600 | -76.250 | -13.338 | +0.788 | 0.688032 | 29.95 |
| `nsi_no_metal` | -400 | -50.850 | -13.522 | +0.603 | 0.004913 | 29.93 |
| `nsi_no_metal` | -250 | -31.800 | -13.734 | +0.391 | 0.011170 | 30.05 |
| `nsi_no_metal` | -150 | -19.100 | -13.386 | +0.739 | 0.014958 | 30.02 |
| `nsi_no_metal` | -75 | -9.575 | -13.740 | +0.385 | 0.001215 | 30.04 |
| `nsi_no_metal` | 0 | -0.050 | -14.136 | -0.011 | 0.000459 | 30.02 |
| `nsi_no_metal` | +75 | +9.475 | -14.036 | +0.090 | 0.000706 | 29.99 |
| `nsi_no_metal` | +150 | +19.000 | -14.377 | -0.252 | 0.002742 | 29.91 |
| `nsi_no_metal` | +250 | +31.700 | -14.558 | -0.433 | 0.000892 | 29.95 |
| `nsi_no_metal` | +400 | +50.750 | -14.912 | -0.786 | 0.002956 | 29.90 |
| `nsi_no_metal` | +600 | +76.150 | -15.728 | -1.602 | 0.003263 | 29.98 |

`bga` has 968 detector rows, so its bands at 600 rows from the center would leave the detector and
were skipped.  Two `bga` bands, at -250 and +250 rows, recorded one central-plane warning each and
took about half the seconds of the other bands.  Both ran 24 evaluations like the rest.  This record
does not explain that difference.

### The fits

Each fit is the least-squares straight line of the offset estimate against the band height.  Both
quantities are in ALU, so the slope has no units and its arc tangent is the rotation.  The second
fit uses only the bands within 150 rows of the central plane, because the cone angle biases the
bands far from it.  The angle and its negative are both recorded, because the sign convention is not
settled.

| dataset | fit | bands | slope | angle, deg | negated angle, deg | intercept, channels | rms residual, channels | edge displacement, pixels |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `nsi_small` | all bands | 11 | -0.00197526 | -0.1132 | +0.1132 | -14.1359 | 0.2419 | 1.477 |
| `nsi_small` | within 150 rows | 5 | -0.00306455 | -0.1756 | +0.1756 | -13.9462 | 0.1149 | 2.292 |
| `bga` | all bands | 9 | +0.00167210 | +0.0958 | -0.0958 | +0.5753 | 0.3063 | 1.281 |
| `bga` | within 150 rows | 5 | -0.00010660 | -0.0061 | +0.0061 | +0.5535 | 0.0093 | 0.082 |
| `nsi_no_metal` | all bands | 11 | -0.00193796 | -0.1110 | +0.1110 | -14.1341 | 0.2505 | 1.450 |
| `nsi_no_metal` | within 150 rows | 5 | -0.00303745 | -0.1740 | +0.1740 | -13.9361 | 0.1153 | 2.272 |

The all-bands fits show the cone-angle bias the script's docstring anticipated.  On the NSI scans
their root mean square residual is 0.242 and 0.250 channels, against 0.115 channels for the fits
within 150 rows.  On `bga` the all-bands residual is 0.306 channels against 0.009 channels within
150 rows, a factor of 33.  The two fits also disagree about the angle on every scan.  These results
indicate that the bands far from the central plane do not lie on the line the bands near it define.
The fit within 150 rows is therefore the one to read.

Two bands on `bga` depart from the line by more than the others.  The band at -400 rows estimates
-0.374 channels and the band at +400 rows estimates +1.731 channels.  Every `bga` band within 250
rows of the central plane estimates between 0.52 and 0.58 channels.  The all-bands fit is drawn
through those two departing bands and the fit within 150 rows is not.  That difference is why the
two fits disagree about the sign of the angle on this scan.

One band on each NSI scan carries a score far above the rest.  On both scans the band at -600 rows
scores about 0.69, which is more than 45 times the largest score of any other band on the same scan.
Every other band on those two scans scores between 0.0005 and 0.015.  These results indicate that
the band's own score marks the outer band where the conjugate comparison fails worst.

### The module's own rotation estimate

The module's `estimate_det_rotation` ran on the same scan, at the channel offset the all-bands fit
gives as its intercept.

| dataset | rotation, deg | rotation, radians | vendor tilt, deg | difference, deg | edge displacement, pixels | seconds |
| --- | --- | --- | --- | --- | --- | --- |
| `nsi_small` | 0.047144 | 0.00082282 | 0.1672 | -0.1200 | 0.615 | 3.91 |
| `bga` | 0.015528 | 0.00027102 | | | 0.208 | 43.33 |
| `nsi_no_metal` | 0.047144 | 0.00082282 | 0.1672 | -0.1200 | 0.615 | 33.48 |

The two NSI scans returned the same rotation to every recorded digit, 0.0008228244347475705
radians, as they did in the first job.  All three estimates raised the sub-pixel warning, which
`estimate_det_rotation` gives when the estimated angle moves the edge channel by less than one
pixel.  On `nsi_small` it read: `estimate_det_rotation: the estimate displaces the edge channels by
0.62 pixels, where the resampling of each candidate biases it by up to 25 percent of the angle.`
The warning names a bias of up to 25 percent of the angle.  A bias that size cannot account for the
factor of 3.5 between this estimate and the vendor tilt.

### The conjugate scores

Each conjugate score is the mean square of `conjugate_difference` over every element it returns, at
the same channel offset the module's rotation estimate used.  These scores are comparable with each
other and not with the estimator's own score in the tables above, which is trimmed to 90 percent of
the view pairs.

| dataset | rotation label | rotation, deg | mean square | resampled | difference shape |
| --- | --- | --- | --- | --- | --- |
| `nsi_small` | zero | +0.000000 | 6.4860e-05 | no | (200, 7, 1496) |
| `nsi_small` | module | +0.047144 | 6.0044e-05 | yes | (200, 7, 1496) |
| `nsi_small` | vendor | +0.167165 | 6.5619e-05 | yes | (200, 7, 1496) |
| `nsi_small` | vendor negated | -0.167165 | 7.2514e-05 | yes | (200, 7, 1496) |
| `bga` | zero | +0.000000 | 4.1292e-04 | no | (2401, 3, 1532) |
| `bga` | module | +0.015528 | 4.1338e-04 | yes | (2401, 3, 1532) |
| `nsi_no_metal` | zero | +0.000000 | 6.1343e-05 | no | (1800, 7, 1496) |
| `nsi_no_metal` | module | +0.047144 | 5.6494e-05 | yes | (1800, 7, 1496) |
| `nsi_no_metal` | vendor | +0.167165 | 6.2081e-05 | yes | (1800, 7, 1496) |
| `nsi_no_metal` | vendor negated | -0.167165 | 6.9331e-05 | yes | (1800, 7, 1496) |

One caveat belongs with this table, and the script's docstring states it.  `conjugate_difference`
applies a nonzero rotation by bilinear resampling and does not resample at all at a rotation of
zero.  Resampling smooths the data, and smoothing lowers a mean square on its own.  The zero entry
is therefore the only one whose data are not smoothed, and every other entry has an advantage the
table cannot separate from the rotation itself.

Two comparisons survive that caveat, because smoothing can only lower a score.  On both NSI scans
the vendor tilt scores above the zero entry, at 6.562e-05 against 6.486e-05 and at 6.208e-05 against
6.134e-05.  On `bga` the module's own answer scores above the zero entry, at 4.1338e-04 against
4.1292e-04.  These results indicate that the conjugate comparison prefers no rotation at all to the
vendor tilt on the NSI scans.  They also indicate that it prefers no rotation to the module's own
estimate on `bga`.

The two preferences do not agree with the band fit in the same way.  On `bga` the band fit within
150 rows gives 0.006 degrees, which is a rotation of nearly zero, so the conjugate preference agrees
with it.  On the NSI scans the band fit gives a magnitude of about 0.175 degrees, so the conjugate
preference for zero disagrees with it.  This record therefore does not treat the conjugate scores as
a third opinion on the rotation.

One ordering in the table is not confounded, because both entries are resampled by the same amount.
On both NSI scans the vendor tilt scores below its own negation, at 6.562e-05 against 7.251e-05 and
at 6.208e-05 against 6.933e-05.  These results indicate that the conjugate comparison prefers the
vendor's sign to the opposite sign.  That sign is the one the module and the NSI reader both use.

## Part C: the direction check

The check scores a direct reconstruction with the angles as given and with every angle negated, and
returns the direction that scores lower.  A value of 1.0 keeps the angles as the reader built them
and a value of -1.0 asks for them to be negated.  The ratio is the worse score divided by the better
one, and the check warns when it falls below 1.5.  The view stride is the largest divisor of the
view count that is at most 4, which is 4 on `nsi_small` and 1 on `bga`.

### The check against the bin factor

The bin factor sets the pixel scale of the reduced problem, and the high-pass filter's widths are
fixed in pixels.  A coarser bin therefore filters over a larger part of the detector.

| dataset | bin factor | value | score as given | score negated | ratio | view stride | reduced sinogram shape | margin warning | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `nsi_small` | 2 | -1.0 | 1.474534 | 1.405799 | 1.049 | 4 | (50, 940, 748) | yes | 11.17 |
| `nsi_small` | 4 | 1.0 | 0.179774 | 0.247079 | 1.374 | 4 | (50, 470, 374) | yes | 1.01 |
| `nsi_small` | 8 | 1.0 | 0.019469 | 0.072393 | 3.718 | 4 | (50, 235, 187) | no | 0.35 |
| `bga` | 2 | 1.0 | 0.126555 | 0.458893 | 3.626 | 1 | (2401, 484, 766) | no | 72.07 |
| `bga` | 4 | 1.0 | 0.169988 | 0.471748 | 2.775 | 1 | (2401, 242, 383) | no | 18.82 |

`bga` has 1532 channels, which 8 does not divide, so its bin factor of 8 was skipped with the reason
`the bin factor does not divide both detector counts`.  The `nsi_small` row at a bin factor of 2
repeats the first job's measurement at the same view stride of 4.  It gives the same answer and the
same ratio.  Every direction entry also caught the model's parameter check warning `Cone angle is
more than 45 degrees.  This will likely produce recon artifacts.`  The first record explains that
warning, which says nothing about the geometry the job used.  The `nsi_small` entry at a bin factor
of 2 also caught fourteen copies of the `torch.jit.script_method` deprecation warning, which is
unrelated to this feature.

### The check against the filter width

Both direction scores were computed again at four filter widths, on one reduced problem per scan at
a bin factor of 4.  The two reduced models are built the way `check_rotation_direction` builds them,
one from the model as given and one from a copy whose view angles are negated.  Each model
reconstructs and forward projects once, because neither step depends on the filter widths.  The
energy fraction is the mean square of the filtered reduced sinogram divided by the mean square of
the reduced sinogram.  It is therefore the share of the sinogram's energy the filter keeps.

| dataset | sigma row | sigma col | score given | score negated | ratio | better | energy fraction | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `nsi_small` | 3 | 15 | 0.179774 | 0.247079 | 1.374 | given | 0.0086 | 0.47 |
| `nsi_small` | 6 | 30 | 0.044182 | 0.092842 | 2.101 | given | 0.0360 | 0.96 |
| `nsi_small` | 12 | 60 | 0.011662 | 0.034703 | 2.976 | given | 0.1427 | 2.21 |
| `nsi_small` | 24 | 120 | 0.004148 | 0.015119 | 3.645 | given | 0.4197 | 4.68 |
| `bga` | 3 | 15 | 0.169988 | 0.471748 | 2.775 | given | 0.0668 | 12.20 |
| `bga` | 6 | 30 | 0.222439 | 0.451505 | 2.030 | given | 0.1068 | 24.79 |
| `bga` | 12 | 60 | 0.227701 | 0.374833 | 1.646 | given | 0.1863 | 57.58 |
| `bga` | 24 | 120 | 0.194962 | 0.294991 | 1.513 | given | 0.3178 | 119.35 |

The first row of each scan is the module's own default width, 3 detector rows and 15 channels.  Its
score agrees with the bin-factor table's score for the same scan to ten digits.  That agreement
confirms that the filter widths are the only thing this table changes.  The setup the four rows of
each scan share took 0.50 seconds on `nsi_small` and 7.88 seconds on `bga`.

The energy fraction says how much of the reduced sinogram the score reads.  At the module's default
widths the filter keeps 0.86 percent of the reduced sinogram's energy on `nsi_small` and 6.7 percent
on `bga`.  These results indicate that the default score is computed on the finest scale of the
data, and that the finest scale carries under a hundredth of the energy on `nsi_small`.  Widening
the filter on `nsi_small` raises both the energy fraction and the score ratio at every step.  These
two results together indicate that most of what the default score reads on `nsi_small` is noise.

The two scans behave in opposite ways as the filter widens.  On `nsi_small` the ratio rises from
1.374 to 3.645, and on `bga` it falls from 2.775 to 1.513.  These results indicate that a wider
filter helps when the default score is dominated by noise and hurts when it is not.

## Limits of this evidence

The evidence is narrow in several ways.  One short scan was measured, and it is a Zeiss Versa scan
of 218 degrees.  Nothing here says how the residual score behaves on a short scan from another
scanner or at another coverage.  Two scanners are represented across the four scans.  `nsi_small`
and `nsi_no_metal` are the same object on the same scanner at 200 and 1800 views.  The two NSI
results in Part B are therefore not independent of each other.

The sign convention was not settled.  The band fit gives a negative angle on both NSI scans, and the
vendor tilt and the module's estimate are both positive.  This record compares magnitudes only.  A
later job has to establish which way each convention runs before the band fit can be used to correct
data.

The conjugate-score comparison is confounded by resampling.  Every nonzero rotation in that table
was applied by bilinear resampling, and the zero rotation was not.  Smoothing lowers a mean square
on its own.  The two comparisons this record draws from that table are the ones the confound cannot
reverse, because smoothing can only lower a score.  No comparison between two nonzero rotations of
different size is safe there.

The band method was not tested on synthetic data with a known rotation at this detector size.  The
only synthetic check is the one the script's comments record, on a detector of 96 rows and 96
channels.  There the recovered slope ran from 0.49 to 1.61 times the applied angle, over four
applied rotations and four applied channel shears.  The bands here span up to 1200 rows, which is
more than ten times that detector's extent.  The same per-band scatter would determine the angle far
better over that longer span.  That argument is not a measurement.

The fitted angle carries an uncertainty larger than the agreement this record claims for it.  The
standard error of the fitted angle is 0.036 degrees for each NSI fit within 150 rows.  That number
is computed in this record from the band heights and estimates the job recorded.  The vendor tilt of
0.167 degrees is within one standard error of the fitted magnitudes of 0.176 and 0.174 degrees.  The
module's 0.047 degrees is 3.5 standard errors from those magnitudes.  These results indicate that
the band fit distinguishes the vendor tilt from the module's estimate on these two scans.  They also
indicate that it does not pin the angle down more closely than about four hundredths of a degree.

Two smaller limits apply to Part A.  `recon_direct` applies no Parker weighting, so every
reconstruction of `z62` in this job is approximate at every candidate offset.  That applies to the
sweep's slices and to the reconstruction inside each residual score alike.  LEAP's `find_centerCol`
may assume a full rotation, and nothing here checks whether it does.  Its agreement with the
residual score is therefore not proof that either value is right.

One limit applies to Part C.  Both scans were reduced at a bin factor of 4 for the filter-width
table, and the module's default is 2.  The filter widths and the bin factor both change the physical
scale the filter works at.  This job varied them one at a time rather than over a grid, so it does
not separate the two.

## The batch file

The job was submitted from `/scratch/gautschi/buzzard/leap_cmp` with `sbatch
real_scan_followup.sbatch`.  That file sits beside this page on disk, and this repository ignores
batch files, so its lines are transcribed here.  It requests `-A bouman -p ai -q normal -N 1
--gpus-per-node=2 --cpus-per-task=28 -t 03:00:00`, names the job `real_scan_followup`, and writes
its log to `/scratch/gautschi/buzzard/leap_cmp/real_scan_followup_%j.log`.  A comment explains why
it asks for two GPUs.  The `ai` partition refuses `--mem` and gives 126 GB of host memory per GPU
requested, and a full-size real scan needs more than that.  The same comment records that the
partition requires 14 CPUs per GPU.  The file then sources `~/load_conda_cuda.sh`, sets `set -e`,
and changes to `/scratch/gautschi/buzzard/leap_cmp`.  It exports six environment variables:

- `TORCHINDUCTOR_CACHE_DIR` to `torch_cache_followup` under that directory;
- `MPLBACKEND` to `Agg`;
- `MBIRTORCH_NUM_DEVICES` to 1;
- `PYTHONPATH` to that directory;
- `REAL_SCAN_RESULTS` to `results_real_scan_followup` under it and `REAL_SCAN_DATA` to `data` under
  it.

A comment above those last two says why they are exported before the interpreter starts.  The
`record` function this job imports from the first job fixes its output directory at import time from
`REAL_SCAN_RESULTS`.  `REAL_SCAN_DATA` is the directory the first job extracted the NSI tarballs
into, and `extract_tarball` skips a directory that is already there.  The file creates both
directories with `mkdir -p`.  It then runs a one-line check that stops the job visibly if the
interpreter is not the one it expects:
`venv/bin/python -c "import torch, mbirtorch; assert torch.cuda.is_available(); print(torch.__version__, mbirtorch.__file__)"`.
Finally it runs `venv/bin/python -u real_scan_followup.py`.
