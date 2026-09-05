# The band-height sweep on the real NSI scans, the view-identity check, and the geometry vectors: what `real_scan_band_height.py` measured

Date: 2026-09-05.  Slurm job 15949700 on gautschi, one NVIDIA H100 80GB HBM3 used of two
requested, torch 2.13.0+cu130, mbirtorch 0.0.2 at commit `4781600` on the `geometric_calibration`
branch.  `sacct` reported `COMPLETED`, exit code `0:0`, elapsed `00:49:02`.  The script's own peak
was 44.0 GB of host memory, measured with `getrusage`.  Every number below was read in this
session from the job's output, which is the log
`/scratch/gautschi/buzzard/leap_cmp/real_scan_band_height_15949700.log` and the JSON lines of
`results_real_scan_band_height/real_scan_band_height.jsonl` in the same directory.

Units.  A rotation is given in degrees, and its edge displacement is the distance it moves the
edge channel of the detector, in pixels.  An offset is given in channels, at the detector pitch of
0.127 mm.  A band is the window of detector rows the rotation estimator compares, and its height
is in detector rows.  The cross-row statistic of a band is the mean squared difference between
neighboring rows divided by the mean square of the band, computed at a view stride of 4.  The two
scans are the artifact phantom without and with its metal insert, 1800 views each, loaded with the
vendor's tilt held out exactly as the earlier jobs loaded them (`real_scan_validation.md`).

## The answers

The no-metal scan's rotation estimate moves with the band height, and far too slowly to reach the
vendor's value.  Across bands of 7 to 201 rows the estimate runs 0.0471, 0.0471, 0.0495, 0.0557,
0.0581, 0.0705 degrees, against the vendor's 0.167.  The movement is monotone, and at 100 rows on
each side of the central plane it has recovered a fifth of the missing 0.12 degrees.  The
synthetic sweep of `rotation_zero_point_synthetic.md`, whose phantom also holds its cross-row
structure outside the default band, recovered its full angle by 32 rows on each side.  The real
scan does not behave like that synthetic case, and its cross-row structure at 470 to 752 rows was
not reached by any band of this sweep.

The metal scan's estimate is near its truth at every band height.  It runs 0.1486, 0.1509,
0.1571, 0.1920, 0.1911, 0.1562 degrees across the same six bands, and the direct reconstructions
of `real_scan_rotation_recon.md` put that scan's rotation between 0.167 and 0.19.  The two scans
share one acquisition geometry and one recorded misalignment, so the difference between their
sweeps is the object.  The metal scan's cross-row statistic is 3.4e-04 to 6.8e-04 against the
no-metal scan's 1.5e-04 to 1.7e-04, which is the same difference stated as a number.

The 200-view scan is a subset of the 1800-view scan, exactly.  Small view 0 matches large view 0,
small view 50 matches large view 450, and small view 150 matches large view 1350, each with a
largest absolute difference of exactly 0.0 over the full 1880 by 1496 frame.  The identical
rotation estimates the earlier jobs recorded on the two scans were therefore guaranteed by the
data.

The vendor's geometry has a real component that no in-plane rotation can correct.  From the
`.nsipro` vectors of the no-metal scan: the in-plane angle between the rotation axis and the
detector columns is 0.16717 degrees, which matches the loaded model's vendor value to seven
digits and so validates the parse; the out-of-plane angle, the lean of the axis along the
detector normal, is 0.07885 degrees; the total angle is 0.18452 degrees.  Whether that lean
explains the no-metal anchor was tested separately and it does not: `leap_axis_tilt.md` measured
the lean's effect on the estimate as 0.011 degrees at most, and this job's metal sweep shows a
scan that carries the same lean and estimates well.

## Question 1: the estimate against the band height

The channel offset was estimated once per scan at the default settings and held fixed across the
sweep, so the band height is the only thing that varies within a scan.  The no-metal offset was
-14.147 channels, 0.022 from the vendor's value; the metal offset was -14.275 channels, 0.150
from it.  Both searches made 24 evaluations and kept 1620 of 1800 view pairs, and neither warned.

| scan | band, rows | row window | estimate, deg | edge displacement, px | min score | max/min | cross-row | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no metal | 7 | 954-961 | 0.0471 | 0.62 | 4.56e-04 | 115.6 | 1.60e-04 | 36.8 |
| no metal | 17 | 949-966 | 0.0471 | 0.62 | 4.56e-04 | 116.0 | 1.67e-04 | 57.2 |
| no metal | 33 | 941-974 | 0.0495 | 0.65 | 4.73e-04 | 112.3 | 1.66e-04 | 94.7 |
| no metal | 65 | 925-990 | 0.0557 | 0.73 | 5.47e-04 | 92.2 | 1.59e-04 | 171.4 |
| no metal | 101 | 907-1008 | 0.0581 | 0.76 | 7.46e-04 | 62.4 | 1.57e-04 | 261.9 |
| no metal | 201 | 857-1058 | 0.0705 | 0.92 | 1.04e-03 | 38.4 | 1.54e-04 | 518.7 |
| metal | 7 | 954-961 | 0.1486 | 1.94 | 1.65e-04 | 217.8 | 3.62e-04 | 36.7 |
| metal | 17 | 949-966 | 0.1509 | 1.97 | 1.69e-04 | 211.9 | 3.44e-04 | 57.6 |
| metal | 33 | 941-974 | 0.1571 | 2.05 | 1.83e-04 | 197.9 | 3.48e-04 | 95.5 |
| metal | 65 | 925-990 | 0.1920 | 2.51 | 2.93e-04 | 139.3 | 4.34e-04 | 171.6 |
| metal | 101 | 907-1008 | 0.1911 | 2.50 | 7.65e-04 | 73.1 | 5.78e-04 | 260.4 |
| metal | 201 | 857-1058 | 0.1562 | 2.04 | 6.74e-03 | 21.8 | 6.80e-04 | 511.9 |

Three readings of this table matter.  First, every no-metal estimate sits below one pixel of edge
displacement, so the module's sub-pixel warning fired on all six, and the resampling bias of that
regime is in all six; every metal estimate sits above 1.9 pixels and none warned.  Second, the
score's floor grows with the band on both scans, from a max/min ratio of 116 down to 38 on the
no-metal scan and 218 down to 22 on the metal scan.  The growing floor is the cone mismatch the
default band's rule guards against, and at these heights it still leaves a well-formed minimum.
Third, the metal scan's estimates wander by about 0.04 degrees across the heights, which says the
taller bands are not more precise here, only differently biased.

The cost grew in proportion to the band height, from 37 seconds at 7 rows to 519 at 201.  The
estimate at 201 rows reads 100 rows on each side of the central plane, still 370 rows short of
the nearest structure.  A follow-up job, `real_scan_band_reach.py`, runs bands of 501 and 1001
rows on the no-metal scan to test whether reaching the structure recovers the estimate.

## Question 2: are the two no-metal files one acquisition?

| small view | matched large view | angle difference, deg | largest absolute difference | identical |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0.0 | 0.0 | yes |
| 50 | 450 | 0.0 | 0.0 | yes |
| 150 | 1350 | 0.00001 | 0.0 | yes |

The 200-view file keeps every ninth view of the 1800-view acquisition, with the frames unchanged
to the float.  Every conclusion drawn from the pair therefore rests on one acquisition, not two.

## Question 3: the geometry vectors

The three vectors were read from the no-metal scan's `.nsipro` with the reader's own parser, and
the detector column vector was built as the reader builds it.  The parse is validated by the
in-plane angle, which reproduces the loaded model's vendor value.

| quantity | value |
| --- | --- |
| in-plane angle, axis to detector columns within the detector plane | 0.16717 deg |
| the loaded model's vendor tilt | 0.16717 deg |
| out-of-plane angle, the axis's lean along the detector normal | 0.07885 deg |
| total angle, axis to detector columns | 0.18452 deg |

The out-of-plane lean displaces a point at the phantom's roughly 30 mm radius vertically by about
0.65 detector pixels between a view and its opposite, varying with the view angle.  Its measured
effect on the estimator is in `leap_axis_tilt.md`.

## Limits of this evidence

Five limits apply.  The sweep's tallest band reaches 100 rows from the central plane, and this
object's cross-row structure begins about 470 rows out, so this record says how the estimate
moves with the band and not what happens when the band reaches the structure.  The channel offset
was held at the value estimated at zero rotation, so the offset coupling the earlier jobs
measured is frozen into every row of the table.  One acquisition and one object in two forms
supply all the real data.  The band statistic was computed at a view stride of 4.  The two
questions beyond the sweep were measured on the no-metal scan only.

## The batch file

The job was submitted from `/scratch/gautschi/buzzard/leap_cmp` with `sbatch
real_scan_band_height.sbatch`.  That file is the rotation-check job's batch file with the job
name, log, compile-cache directory, and results directory renamed to `real_scan_band_height`.  It
requests `-A bouman -p ai -q normal -N 1 --gpus-per-node=2 --cpus-per-task=28 -t 03:00:00`, and
the two GPUs are for the 252 GB of host memory, because the view comparison holds the 200-view
scan beside the 1800-view scan.  It exports the same six environment variables as the earlier
jobs, runs the one-line interpreter check, and runs `venv/bin/python -u real_scan_band_height.py`.
