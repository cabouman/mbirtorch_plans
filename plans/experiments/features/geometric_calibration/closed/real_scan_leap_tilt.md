# LEAP's tilt estimate with the full detector height: what `real_scan_leap_tilt.py` measured

Date: 2026-09-04.  Slurm job 15936618 on gautschi, one NVIDIA H100 80GB HBM3, torch 2.13.0+cu130,
LEAP 1.26, mbirtorch 0.0.2 at commit `4781600`.  The job asked for two GPUs for 252 GB of host
memory and pinned mbirtorch to one device.  `sacct` reported `COMPLETED`, exit code `0:0`, elapsed
`00:09:46`, and a batch-step `MaxRSS` of 31165180 KB; the script's own peaks were 40.6, 41.1, and
49.1 GB on the three scans.  Every number below was read in the same session from the job's log,
`/scratch/gautschi/buzzard/leap_cmp/real_scan_leap_tilt_15936618.log`, and its JSON lines in
`results_real_scan_leap_tilt/`.

Units.  A detector rotation is given in degrees.  LEAP's cost is the mean square of its
`conjugate_difference`, which is dimensionless, and it is given here times a million.

## The answer

LEAP's `estimate_tilt` does not do better than the module's estimator on these scans, with or
without the full detector height.  The question arose because the earlier jobs gave LEAP a band of
128 rows, and a detector rotation shifts each row's content along the channels in proportion to the
row's height, so the band removed most of that signal.  With all 1880 rows LEAP returned -1.03
degrees on the NSI scan without metal, whose rotation direct reconstructions put at the vendor's
0.167 degrees (`real_scan_rotation_recon.md`).  It returned -0.010 degrees on the scan with metal,
whose rotation is near 0.167 degrees as well.  It returned 0.090 degrees on the Zeiss scan, whose
rotation the module and a row-band fit both put below 0.02 degrees.  On the 128-row band it
returned the bound of its search, 5.000 degrees, on the first scan, and 0.006 and -0.403 degrees on
the other two, as the earlier jobs found.

LEAP's own cost has no minimum near the true rotation on the NSI scans.  The job evaluated LEAP's
`conjugate_difference` at seventeen angles from -0.4 to 0.4 degrees.  On the scan without metal the
cost falls steadily from 0.4 to -0.4 degrees with the full detector, by 8 percent over the range,
and it falls steadily the other way on the 128-row band, by 24 percent.  On the scan with metal the
cost varies by 0.2 percent over the range with the full detector.  On the Zeiss scan the cost has an
interior minimum near 0.05 degrees with the full detector, and the value at exactly zero rotation
stands 1.7 percent above its neighbors.  These results indicate three things.  LEAP's search returns
the bound or a drift because its cost is monotone over the range where the rotation lies.  The cost
at zero rotation, which is the one angle LEAP does not resample, sits above the resampled angles on
the Zeiss scan, so the resampling lowers LEAP's cost on its own.  The synthetic study of this
feature found that a bilinear kernel is unusable for this search for that reason, and the module
resamples with a cubic kernel to avoid it (`rotation_interpolation_bias.md`).

The two estimators therefore fail differently.  LEAP's cost has no usable minimum on these scans.
The module's cost has a clear minimum, with the score at the ends of its search range 6 to 218 times
the minimum (`real_scan_rotation_check.md`), but on the phantom without metal that minimum sits at
the wrong zero point.  LEAP's estimate took 1.4 seconds on a 20 GB sinogram, which indicates that it
does not read every row it is given.

## What was measured

Each scan was loaded as the earlier jobs loaded it, with the vendor's tilt held out of the NSI
sinograms.  LEAP's model took the scan's own view angles as 180 degrees minus mbirtorch's, the
channel axis reversed, the center row set to the row of the central plane, and the center column set
from the vendor's offset, as LEAP's documentation asks.  The full detector and the 128-row band were
run one after the other on the same loaded scan.

| scan | rows given | LEAP `estimate_tilt`, degrees | at its search bound | cost argmin on the grid, degrees | cost, largest over smallest | seconds |
| --- | --- | --- | --- | --- | --- | --- |
| NSI phantom, 1800 views, no metal | 1880 | -1.029 | no | -0.40 | 1.084 | 1.38 |
| NSI phantom, 1800 views, no metal | 128 | 5.000 | yes | +0.40 | 1.243 | 0.09 |
| NSI phantom with metal | 1880 | -0.010 | no | -0.15 | 1.002 | 1.35 |
| NSI phantom with metal | 128 | 0.006 | no | +0.40 | 1.016 | 0.09 |
| Zeiss ball grid array | 968 | 0.090 | no | +0.05 | 1.118 | 0.44 |
| Zeiss ball grid array | 128 | -0.403 | no | -0.40 | 1.196 | 0.04 |

A cost argmin at the edge of the grid, -0.40 or +0.40, means the cost was monotone over the range.
The reference values are 0.167 degrees for the two NSI scans and under 0.02 degrees for the Zeiss
scan.

## Limits of this evidence

Three scans from two scanners were run, and the reference rotation of the NSI scans rests on direct
reconstructions at four fixed angles.  LEAP's `estimate_tilt` and `conjugate_difference` were used
as its documentation describes, with the geometry conventions the LEAP comparison established, and
no other LEAP calibration route, such as its `inconsistencyReconstruction`, was tried.  How LEAP
resamples inside `conjugate_difference` was inferred from the shape of its cost at zero rotation and
not read from its source.

## The batch file

The job was submitted with `sbatch real_scan_leap_tilt.sbatch`, which is the direct-reconstruction
job's batch file with the job name, log, cache directory, and results directory renamed to
`real_scan_leap_tilt` and a walltime of two hours.
