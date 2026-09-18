# The search's own controls and a known rotation added to the data: what `real_scan_rotation_check.py` measured

Date: 2026-09-04.  Slurm job 15931130 on gautschi, one NVIDIA H100 80GB HBM3, torch 2.13.0+cu130,
mbirtorch 0.0.2 at commit `4781600` on the `geometric_calibration` branch.  The job asked for two
GPUs so that it would hold 252 GB of host memory, and it pinned mbirtorch to one device with
`MBIRTORCH_NUM_DEVICES`.  `sacct` reported `COMPLETED`, exit code `0:0`, elapsed `00:44:06`, and a
batch-step `MaxRSS` of 42948588 KB, which is 40.96 GB.  The script's own peak was 41.44 GB, measured
with `getrusage`.  `sacct` samples periodically, so the smaller of the two numbers is expected.
Every number below was read from the job's output in the same session.  That output is the log,
`/scratch/gautschi/buzzard/leap_cmp/real_scan_rotation_check_15931130.log`, and the 166 JSON lines
of `results_real_scan_rotation_check/real_scan_rotation_check.jsonl` in the same directory.  A few
statements below describe how the code is built rather than what the job returned.  Those come from
the source rather than from the job's output, and each says so where it appears.

Units.  A rotation is given in degrees, and the module returns it in radians.  The edge displacement
of a rotation is the distance it moves the edge channel of the detector, in pixels.  An offset is
given in channels, and a channel is the detector pitch, which is 0.127 mm on all three scans.  A
band height is a distance along the detector's row axis, in ALU, measured from the row the central
plane reaches.

Definitions.  The module is `mbirtorch/preprocess/geometry_calibration.py`, and the two estimators
under test in it are `estimate_det_rotation` and `estimate_det_channel_offset`.  The search is
`_search_minimum` in the same file, which both estimators call.  The band-slope method is the fit
the follow-up job defines.  That method estimates the channel offset on each of several bands of
detector rows.  It then fits a straight line of that estimate against the band height.  The slope of
that line is the rotation.  An added rotation is an angle applied to the whole sinogram with
`correct_det_rotation` before the estimates run on it.  The script's own name for an added rotation
is `delta_degrees`.  A vendor value is the value the scanner's geometry report recorded, as the NSI
reader reports it.

## The answers

The search's own controls did not choose the rotation estimate.  Six settings of the search's bounds
and coarse count ran on each scan.  The six values span 0.0004 degrees on `nsi_small`, 0.0005 degrees
on `nsi_no_metal`, and 0.0023 degrees on `nsi_metal`.  They average 0.047 degrees on each scan
without metal and 0.149 degrees on the scan with metal.  The bounds ranged over a factor of five and
the coarse count over a factor of two.  The ratio of the largest score on the searched range to the
smallest ranged from 6.1 to 218.  These results indicate that the estimate is a reading of the data
rather than a point the search's controls placed.  The first of the two hypotheses the script's
docstring names is therefore rejected.

The module's rotation estimate followed a rotation added to the data on all three scans.  Four
angles from 0.25 to 2.0 degrees were added with `correct_det_rotation`.  A straight line of the
estimate against the added angle has a slope of -1.0014 on `nsi_small`, -1.0016 on `nsi_no_metal`,
and -1.0061 on `nsi_metal`.  The root mean square residual of those three lines is 0.0013, 0.0011,
and 0.0061 degrees.  Adding an angle to the data therefore lowers the estimate by that same angle.
These results indicate that the sign convention is that the estimate is the rotation to apply.  The
intercept of each line is what the estimator reads with nothing added.  The three intercepts are
0.044, 0.044, and 0.193 degrees.

On `nsi_metal` the three larger additions imply more rotation than the direct estimate reads.  Those
three additions are 0.5, 1.0, and 2.0 degrees, and each of them moves the edge channel by more than
four pixels.  They imply a carried rotation of 0.181 to 0.186 degrees.  The direct estimate at no
addition is 0.149 degrees, which moves the edge channel by 1.94 pixels.  The addition of 0.25
degrees left about 0.07 degrees for the estimator to read, and the estimator read 0.050 degrees at
0.66 pixels.  These two shortfalls are the estimator's bias near one pixel of edge displacement,
measured here on real data.

The band-slope method followed the same additions with a slope well below one.  Its slope against
the added angle is 0.70 on `nsi_small`, 0.71 on `nsi_no_metal`, and 0.80 on `nsi_metal`.  The root
mean square residual of its own band line also grew with the added angle on all three scans.  That
residual ran from about 0.2 channels at the smallest addition to about 1.6 channels at the largest.
A method that measured the rotation would return the added angle and would keep its band line
straight.  These results indicate that the band-slope method does not measure a rotation on these
cone-beam scans.

The three scans were read with one geometry report.  That report gives a detector tilt of 0.167
degrees, and the reader applies it to all three scans.  The estimator reads two different values on
those scans.  It reads 0.044 degrees on the two scans without metal and 0.193 degrees on the scan
with metal.  Two explanations remain open.  The first is that the two acquisitions differ in
their detector alignment and the report describes the scan with metal.  The second is that the
alignment is the same on both and the estimator's zero point depends on the object.  An added
rotation cannot separate the two, because it tests the slope of the estimate and not its zero point.
This record cannot say which explanation is right.

The offset estimated at three fixed rotations divides the scans the same way.  On `nsi_small` and
`nsi_no_metal` the offset at no rotation is the closest of the three to the vendor's offset, at
0.037 and 0.022 channels from it.  On `nsi_metal` the offset at the vendor's rotation of 0.167
degrees is the closest of the three, at 0.053 channels from it.  These results indicate that the
offset estimate prefers no rotation on the two scans without metal and the vendor's rotation on the
scan with metal.  That is the same division the rotation estimate makes.  This record does not treat
it as a second measurement of the rotation.  The three offsets on one scan span 0.14 channels at
most, and nothing here says which of the three is right.

## The question this job answers

The two earlier jobs are `real_scan_validation.py` and `real_scan_followup.py` in this directory,
and their records are `real_scan_validation.md` and `real_scan_followup.md`.  Both left one number
unexplained.  The module's `estimate_det_rotation` returned 0.0008228244347475705 radians, which is
0.047 degrees, on `nsi_small` and on `nsi_no_metal`.  Those two returns were identical to every
recorded digit although the two scans have different view counts.  The vendor's geometry report gives
0.167 degrees on the same scanner.  The follow-up job's band fit within 150 rows gave a magnitude of
0.176 degrees on `nsi_small` and 0.174 degrees on `nsi_no_metal`.

The script's docstring names two hypotheses about the 0.047 degrees.  The first is that the search
returned a point of its own lattice rather than a reading of the data.  The second is that the
estimator under-reads only when the angle is small, because its estimate there moves the edge
channel by 0.62 pixels.  Part 1 measures the first hypothesis and Part 2 measures the second.
Neither part needs a ground truth.

## The scans

The three scans are the same phantom on the same NSI scanner.  `nsi_small` and `nsi_no_metal` are
that phantom without a metal insert, at 200 and 1800 views.  `nsi_metal` is the phantom with the
insert, at 1800 views.  Every scan loaded at full resolution.  The sinograms were built inside the
script rather than through `nsi.get_sino_and_model`, because that reader applies the vendor's
detector tilt and a tilt already applied cannot be estimated.  The script repeats the reader's steps
with the tilt held out and keeps it as the vendor value.

| dataset | views | rows | channels | pitch, mm | coverage, deg | vendor offset, channels | vendor tilt, deg | load seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `nsi_small` | 200 | 1880 | 1496 | 0.1270 | 358.20 | -14.125 | 0.1672 | 4.0 |
| `nsi_no_metal` | 1800 | 1880 | 1496 | 0.1270 | 359.80 | -14.125 | 0.1672 | 29.9 |
| `nsi_metal` | 1800 | 1880 | 1496 | 0.1270 | 359.80 | -14.125 | 0.1672 | 30.8 |

The vendor tilt of 0.1672 degrees moves the edge channel by 2.182 pixels on all three scans.  The
log shows why the three vendor values are the same.  All three loads print the same corrected
coordinate of the (0,0) detector pixel, `[95.707 123.072 416.49152]`, and the same rotation axis,
normal, and horizontal unit vectors.  The metal scan's directory also holds a geometry report file
whose name carries the no-metal phantom's name, `Geometry Report
[JB-033_ArtifactPhantom_Vertical_NoMetal].rtf`.  These facts indicate that one geometry report
supplies the vendor values for all three scans.

The metal insert shows in the sinogram range.  The largest sinogram value is 0.722 on `nsi_small`,
0.725 on `nsi_no_metal`, and 3.181 on `nsi_metal`.  No scan held a nonfinite value.

The three datasets took 110.7, 1264.8, and 1253.5 seconds, in the order of the table above.  Their
peak GPU allocation was 1.41 GB on each.

## Part 1: the search's own controls

The search runs in two stages, and the description in this paragraph comes from the module's source.
A coarse pass evaluates `num_coarse` equally spaced candidates over the bounds.  A golden-section
search then narrows the bracket around the coarse minimum until the bracket is narrower than a fixed
width, which is 0.005 degrees for the rotation.  The search returns the evaluated candidate with the
smallest score.  Part 1 runs that search six times per scan, at three settings of the bounds and two
of the coarse count.  Every run uses the same sinogram and the same channel offset, so the search's
controls are the only thing that changes.

A bounds entry of `default` is the module's own range of five degrees on each side.  An entry of 1.0
or 2.0 is that many degrees on each side.  The score ratio is the largest score on the run's curve
divided by the smallest, which says how much the score changes over the range searched.

| dataset | bounds, deg | coarse count | value, deg | value, radians | evaluations | score ratio | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `nsi_small` | default | 11 | 0.0471 | 0.0008228244347475705 | 26 | 109.9 | 4.00 |
| `nsi_small` | default | 21 | 0.0472 | 0.0008243347577869141 | 35 | 109.9 | 5.02 |
| `nsi_small` | 1.0 | 11 | 0.0472 | 0.000824032693179046 | 23 | 6.1 | 1.94 |
| `nsi_small` | 1.0 | 21 | 0.0475 | 0.0008294530226370501 | 31 | 6.1 | 2.62 |
| `nsi_small` | 2.0 | 11 | 0.0474 | 0.0008265918256041148 | 24 | 19.4 | 2.19 |
| `nsi_small` | 2.0 | 21 | 0.0472 | 0.0008240326931790443 | 33 | 19.4 | 3.11 |
| `nsi_no_metal` | default | 11 | 0.0471 | 0.0008228244347475705 | 26 | 115.6 | 33.19 |
| `nsi_no_metal` | default | 21 | 0.0472 | 0.0008243347577869141 | 35 | 115.6 | 48.43 |
| `nsi_no_metal` | 1.0 | 11 | 0.0472 | 0.000824032693179046 | 23 | 6.4 | 21.49 |
| `nsi_no_metal` | 1.0 | 21 | 0.0469 | 0.0008186123637210379 | 31 | 6.4 | 29.10 |
| `nsi_no_metal` | 2.0 | 11 | 0.0474 | 0.0008265918256041148 | 24 | 20.5 | 23.68 |
| `nsi_no_metal` | 2.0 | 21 | 0.0472 | 0.0008240326931790443 | 33 | 20.5 | 33.34 |
| `nsi_metal` | default | 11 | 0.1486 | 0.002592696201606864 | 26 | 217.8 | 33.94 |
| `nsi_metal` | default | 21 | 0.1486 | 0.0025942065246462066 | 35 | 217.8 | 49.44 |
| `nsi_metal` | 1.0 | 11 | 0.1495 | 0.0026098633838044843 | 23 | 20.8 | 21.61 |
| `nsi_metal` | 1.0 | 21 | 0.1472 | 0.002569361945173373 | 31 | 20.8 | 28.86 |
| `nsi_metal` | 2.0 | 11 | 0.1480 | 0.0025830638011223274 | 24 | 60.2 | 23.81 |
| `nsi_metal` | 2.0 | 21 | 0.1495 | 0.002609863383804486 | 33 | 60.2 | 33.95 |

The six values on one scan span very little.  On `nsi_small` they run from 0.0471 to 0.0475 degrees,
a spread of 0.0004 degrees.  On `nsi_no_metal` they run from 0.0469 to 0.0474 degrees, a spread of
0.0005 degrees.  On `nsi_metal` they run from 0.1472 to 0.1495 degrees, a spread of 0.0023 degrees.
Each spread is smaller than the 0.005 degree width at which the search stops.  The score ratio
changes by a factor of 18 on `nsi_small` across the six settings, from 6.1 to 109.9.  These results
indicate that the estimate
does not move when the search's controls move.  On the two scans without metal the estimate differs
from the vendor tilt by 0.120 degrees, and the six settings move the estimate by less than 0.001
degrees.  These two numbers indicate that the search's controls do not account for that difference.

The two scans without metal returned the same value at five of the six settings.  Those five returns
agree to every digit of the recorded radians, although the two scans have different view counts and
different scores.  The score at the returned point is 4.789e-04 on `nsi_small` and 4.559e-04 on
`nsi_no_metal` at the default bounds and 11 coarse points.  The search's construction explains the
agreement, and this paragraph's account of it comes from the module's source.  The coarse grid
depends only on the bounds and the coarse count.  Each golden-section point is then placed at a
fixed fraction of the current bracket.  Each step drops one end of the bracket according to a single
comparison of two scores.  The set of points the search can visit is therefore fixed before any data
are read.  Which of those points it returns depends only on the sequence of comparisons.  Two scans
of one object give score curves of the same shape, so they make the same comparisons and stop at the
same point.

The sixth setting shows what happens when one comparison goes the other way.  At bounds of 1.0
degree and 21 coarse points, `nsi_small` returned 0.0475 degrees and `nsi_no_metal` returned 0.0469
degrees.  The two values differ by 0.0006 degrees, which is smaller than the 0.005 degree width at
which the search stops.  Each returned value also lies between two candidates the other run
evaluated.  These results indicate that the two scans' comparison sequences diverged at one step.
They also indicate that the two answers ended within one stopping width of each other.

The sub-pixel warning fired on all six runs on `nsi_small` and on all six on `nsi_no_metal`.  It
fired on none of the six on `nsi_metal`.  On the two scans without metal the warning read:
`estimate_det_rotation: the estimate displaces the edge channels by 0.62 pixels, where the
resampling of each candidate biases it by up to 25 percent of the angle.`  One `nsi_no_metal` run
reported 0.61 pixels instead of 0.62.  The six `nsi_metal` estimates displace the edge channel by
1.92 to 1.95 pixels, which is above the one pixel at which the module warns.  Every run in this
table kept 90 percent of the view pairs.  That is 180 of 200 pairs on `nsi_small` and 1620 of 1800
on the other two scans.

## Part 2: a known rotation added to the data

Part 2 has three measurements.  The first estimates the channel offset at three fixed rotations, on
the data as they were loaded.  The second and third add four known rotations to the data and run the
module's estimate and the band-slope method on each rotated sinogram.  Both parts of the job hold
the channel offset fixed at one estimate per scan, so the rotation is the only thing that varies
within a dataset.  That estimate is the entry at a rotation of zero in the table below.

### The offset at three fixed rotations

The three rotations are zero, the module's own rotation estimate on the two scans without metal, and
the vendor's detector tilt.  The comparison applies the rotation it is given to the band before it
pairs the views.  Every run below made 24 evaluations.

| dataset | rotation, deg | offset, channels | vendor difference, channels | score |
| --- | --- | --- | --- | --- |
| `nsi_small` | 0.000 | -14.1618 | -0.0365 | 4.8460e-04 |
| `nsi_small` | 0.047 | -14.1990 | -0.0737 | 4.4844e-04 |
| `nsi_small` | 0.167 | -14.3061 | -0.1808 | 4.8534e-04 |
| `nsi_no_metal` | 0.000 | -14.1469 | -0.0216 | 4.6175e-04 |
| `nsi_no_metal` | 0.047 | -14.1779 | -0.0526 | 4.2531e-04 |
| `nsi_no_metal` | 0.167 | -14.2751 | -0.1498 | 4.6191e-04 |
| `nsi_metal` | 0.000 | -14.2751 | -0.1498 | 2.4665e-04 |
| `nsi_metal` | 0.047 | -14.2460 | -0.1207 | 2.0044e-04 |
| `nsi_metal` | 0.167 | -14.1779 | -0.0526 | 1.4489e-04 |

The vendor's offset agrees best with a different rotation on the scan with metal than on the two
without it.  On `nsi_small` and `nsi_no_metal` the smallest vendor difference is at no rotation, at
0.037 and 0.022 channels.  On `nsi_metal` the smallest vendor difference is at the vendor's rotation
of 0.167 degrees, at 0.053 channels.  These results indicate that the offset estimate makes the same
division between the scans that the rotation estimate makes.

The entries at 0.047 degrees reproduce the first job's offset estimates on the two scans without
metal.  Those entries are -14.199 and -14.178 channels, and `real_scan_validation.md` records the
same two numbers.  The first job estimated the offset a second time at each scan's own rotation
estimate, which was 0.047 degrees on those two scans.  Its `nsi_metal` value of -14.188 channels was
estimated at 0.149 degrees, which lies between the 0.047 and 0.167 entries here.  These agreements
indicate that this job reproduces the first job's offset measurements.

The scores in this table cannot be compared across rotations.  A nonzero rotation is applied by
resampling.  Resampling smooths the data, and smoothing lowers a mean square on its own.  The entry
at zero rotation is the only one whose data are not smoothed.

### The module's estimate at each added rotation

Each angle was applied to the whole sinogram with `correct_det_rotation`, which returns a second
array of the sinogram's size.  The estimate then ran on the rotated sinogram at the same channel
offset the rest of the dataset used.  The last column is the estimate plus the added angle.  That
sum is what the estimator would read on the data before the addition, under the sign convention that
the estimate is the rotation to apply.

| dataset | added angle, deg | estimate, deg | edge displacement, pixels | sub-pixel warning | estimate plus added angle, deg |
| --- | --- | --- | --- | --- | --- |
| `nsi_small` | 0.25 | -0.204452 | 2.669 | no | 0.045548 |
| `nsi_small` | 0.50 | -0.458074 | 5.980 | no | 0.041926 |
| `nsi_small` | 1.00 | -0.958161 | 12.509 | no | 0.041839 |
| `nsi_small` | 2.00 | -1.958161 | 25.564 | no | 0.041839 |
| `nsi_no_metal` | 0.25 | -0.204452 | 2.669 | no | 0.045548 |
| `nsi_no_metal` | 0.50 | -0.457514 | 5.973 | no | 0.042486 |
| `nsi_no_metal` | 1.00 | -0.958161 | 12.509 | no | 0.041839 |
| `nsi_no_metal` | 2.00 | -1.958161 | 25.564 | no | 0.041839 |
| `nsi_metal` | 0.25 | -0.050423 | 0.658 | yes | 0.199577 |
| `nsi_metal` | 0.50 | -0.318840 | 4.162 | no | 0.181160 |
| `nsi_metal` | 1.00 | -0.814182 | 10.629 | no | 0.185818 |
| `nsi_metal` | 2.00 | -1.817461 | 23.727 | no | 0.182539 |

Each estimate is negative and close to the negative of the added angle.  The straight-line fit of
the estimate against the added angle gives these numbers:

| dataset | slope | intercept, deg | rms residual, deg |
| --- | --- | --- | --- |
| `nsi_small` | -1.0014 | +0.0441 | 0.0013 |
| `nsi_no_metal` | -1.0016 | +0.0444 | 0.0011 |
| `nsi_metal` | -1.0061 | +0.1930 | 0.0061 |

The three slopes are -1.00 to -1.01 and the three residuals are 0.001 to 0.006 degrees.  These
results indicate that the estimator follows an added rotation one for one.  They also settle the
sign convention.  Adding an angle with `correct_det_rotation` lowers the estimate by that angle, so
the estimate is the rotation to apply rather than the rotation the detector carries.  The intercept
is then the rotation each scan carried before the addition, and those intercepts are 0.044, 0.044,
and 0.193 degrees.

The `nsi_metal` line is the only one whose four points do not all sit above one pixel of edge
displacement.  Its three larger additions give a carried rotation of 0.181, 0.186, and 0.183 degrees.
Its addition of 0.25 degrees gives 0.200 degrees.  That run is the only one in this table that raised
the sub-pixel warning.  The warning read: `estimate_det_rotation: the estimate displaces the edge
channels by 0.66 pixels, where the resampling of each candidate biases it by up to 25 percent of the
angle.`  The direct estimate
at no addition, from Part 1, is 0.149 degrees at 1.94 pixels.  The three larger additions therefore
imply a rotation 0.03 degrees above what the direct estimate reads.  The addition of 0.25 degrees
left about 0.07 degrees for the estimator to read, taking 0.183 degrees as the carried rotation.  The
estimator read 0.050 degrees there.  The two shortfalls are 19 percent of the angle at 1.94 pixels
and 25 percent of the angle at 0.66 pixels.  These results indicate that the estimator under-reads a
rotation whose edge displacement is near one pixel, on this real scan.

The two scans without metal give no such comparison, because all four of their additions sit well
above one pixel of edge displacement.  Their smallest addition moves the edge channel by 2.67 pixels.
Their intercepts of 0.044 degrees agree with the Part 1 estimates of 0.047 degrees to 0.003 degrees.
That agreement is not a test of the sub-pixel bias.  The Part 1 estimates on those two scans sit at
0.62 pixels, and every addition in this table sits above 2.6 pixels.

Each run in this table made 26 evaluations and kept 90 percent of the view pairs.  The score ratio
over the searched range rose with the added angle on every scan.  It ran from 117.7 to 209.5 on
`nsi_small`, from 123.9 to 221.9 on `nsi_no_metal`, and from 280.3 to 402.9 on `nsi_metal`.  The
estimate itself took 3.2 to 3.9 seconds on `nsi_small`, 33.0 to 38.2 seconds on `nsi_no_metal`, and
32.2 to 38.3 seconds on `nsi_metal`.

`correct_det_rotation` took these seconds per added rotation, in the order 0.25, 0.5, 1.0, 2.0
degrees:

- `nsi_small`: 1.31, 1.32, 1.30, 1.31;
- `nsi_no_metal`: 11.72, 12.34, 9.78, 9.42;
- `nsi_metal`: 8.75, 8.70, 8.63, 8.72.

### The band-slope method at each added rotation

The band-slope method ran on the same rotated sinograms.  Each fit used the five bands within 150
rows of the central plane, at row offsets of -150, -75, 0, +75, and +150.  Each band is a window of
16 detector rows, keeps every view, and bins nothing.  Every band made 24 evaluations and kept 90
percent of the view pairs.  The module's central-plane warning fired twice for each of the four bands
away from the central plane, on every scan and at every added angle.  It did not fire for the central
band.  That warning read: `The reduction's row window does not contain the row the central
plane reaches, so the cone-beam conjugate comparison is biased by the cone angle.`

| dataset | added angle, deg | fitted angle magnitude, deg | band line rms residual, channels |
| --- | --- | --- | --- |
| `nsi_small` | 0.25 | 0.010932 | 0.2108 |
| `nsi_small` | 0.50 | 0.201414 | 0.3445 |
| `nsi_small` | 1.00 | 0.571494 | 0.6684 |
| `nsi_small` | 2.00 | 1.241557 | 1.5617 |
| `nsi_no_metal` | 0.25 | 0.012395 | 0.1990 |
| `nsi_no_metal` | 0.50 | 0.203725 | 0.3257 |
| `nsi_no_metal` | 1.00 | 0.577211 | 0.6401 |
| `nsi_no_metal` | 2.00 | 1.255029 | 1.5223 |
| `nsi_metal` | 0.25 | 0.228470 | 0.2366 |
| `nsi_metal` | 0.50 | 0.406183 | 0.1581 |
| `nsi_metal` | 1.00 | 0.790052 | 0.3628 |
| `nsi_metal` | 2.00 | 1.619414 | 1.6210 |

Every fitted angle in this table is positive, so its magnitude is the fitted angle itself.  The
straight-line fit of that magnitude against the added angle gives these numbers:

| dataset | slope | intercept, deg | rms residual, deg |
| --- | --- | --- | --- |
| `nsi_small` | 0.7008 | -0.1506 | 0.0135 |
| `nsi_no_metal` | 0.7078 | -0.1515 | 0.0132 |
| `nsi_metal` | 0.7987 | +0.0122 | 0.0144 |

The three slopes are 0.70 to 0.80, against 1.00 for a method that measured the rotation.  These
results indicate that the band-slope method reads about three quarters of an added rotation on these
scans.  The residual of its own band line also grew with the added angle.  On `nsi_small` it grew
from 0.211 to 1.562 channels and on `nsi_no_metal` from 0.199 to 1.522 channels, over the four
additions in order.  On `nsi_metal` it grew from 0.237 to 1.621 channels, with one dip to 0.158
channels at an added 0.5 degrees.  A rotation moves each row's content along the channels in
proportion to the row's height.  Adding a pure rotation to the data should therefore change the
slope of the band line without changing how far the bands sit from it.  These results indicate that
the five band estimates depart further from a straight line as the added rotation grows.

Taken together, the slope and the growing residual indicate that the band-slope method does not
measure a rotation on these cone-beam scans.  The follow-up job's magnitude of 0.176 degrees on
`nsi_small` and 0.174 degrees on `nsi_no_metal` therefore cannot be read as a rotation.  This record
does not say what the band slope does measure.  The module's own warning names one candidate.  The
cone angle biases the offset estimate on every band away from the central plane.

Each band took 2.1 to 3.1 seconds on `nsi_small`, 29.5 to 43.7 seconds on `nsi_no_metal`, and 29.6
to 44.1 seconds on `nsi_metal`.  Each fit used five bands, so the band-slope method cost about five
times those seconds per added angle.

## Limits of this evidence

The evidence is narrow in several ways.  One scanner is represented.  One phantom is represented, in
two forms, without and with a metal insert.  `nsi_small` and `nsi_no_metal` are the same object and
the same form at 200 and 1800 views, so their two results are not independent of each other.  The
question of whether the estimator's zero point depends on the object rests on a comparison of two
forms of one phantom.

The zero point of the estimator was not tested.  An added rotation moves the data by a known amount,
so it tests the slope of the estimate against the rotation.  It leaves the estimate's value at zero
added rotation untested, because no measurement here knows the rotation the data already carried.
Every statement in this record about the 0.044 and 0.193 degree intercepts is therefore a statement
about what the estimator reads, and not about the detector.

The added rotation and the estimator's own candidates share one resampling kernel.
`correct_det_rotation` rotates each view with bilinear interpolation, which comes from the source of
`mbirtorch/preprocess/utilities.py`.  At this commit the estimator also applies each candidate
rotation by bilinear resampling, which `real_scan_followup.md` records for the same commit.  A bias
that both share would not show up in the slope measured here.  The
sub-pixel warning names a bias of up to 25 percent of the angle from that resampling.  The one
sub-pixel point in this job under-read by 25 percent, on `nsi_metal` at an added 0.25 degrees.

The band-slope method was not tested on synthetic data with a known rotation at this detector size.
The only synthetic check is the one the follow-up job's comments record, on a detector of 96 rows
and 96 channels.  A slope of 0.70 to 0.80 measured here on real data cannot be separated into a
property of the method and a property of these scans without such a control.

## The batch file

The job was submitted from `/scratch/gautschi/buzzard/leap_cmp` with `sbatch
real_scan_rotation_check.sbatch`.  That file sits beside this page on disk, and this repository
ignores batch files, so its lines are transcribed here.  It requests `-A bouman -p ai -q normal -N 1
--gpus-per-node=2 --cpus-per-task=28 -t 03:00:00`, names the job `real_scan_rotation_check`, and
writes its log to `/scratch/gautschi/buzzard/leap_cmp/real_scan_rotation_check_%j.log`.  A comment
explains why it asks for two GPUs.  The `ai` partition refuses `--mem` and gives 126 GB of host
memory per GPU requested.  The same comment records that this job holds two sinograms at once,
because each added rotation makes a second array of the sinogram's size.  It records that the
largest sinogram here is 20 GB.  It also records that the partition requires 14 CPUs per GPU.  The
file then sources `~/load_conda_cuda.sh`, sets `set -e`, and changes to
`/scratch/gautschi/buzzard/leap_cmp`.
It exports six environment variables:

- `TORCHINDUCTOR_CACHE_DIR` to `torch_cache_rotation_check` under that directory;
- `MPLBACKEND` to `Agg`;
- `MBIRTORCH_NUM_DEVICES` to 1;
- `PYTHONPATH` to that directory;
- `REAL_SCAN_RESULTS` to `results_real_scan_rotation_check` under it and `REAL_SCAN_DATA` to `data`
  under it.

A comment above those last two says why they are exported before the interpreter starts.  The
`record` function this job imports from the first job fixes its output directory at import time from
`REAL_SCAN_RESULTS`.  `REAL_SCAN_DATA` is the directory the first job extracted the NSI tarballs
into, and `extract_tarball` skips a directory that is already there.  The file creates both
directories with `mkdir -p`.  It then runs a one-line check that stops the job visibly if the
interpreter is not the one it expects:
`venv/bin/python -c "import torch, mbirtorch; assert torch.cuda.is_available(); print(torch.__version__, mbirtorch.__file__)"`.
Finally it runs `venv/bin/python -u real_scan_rotation_check.py`.
