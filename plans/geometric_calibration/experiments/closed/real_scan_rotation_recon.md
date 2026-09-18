# Direct reconstructions of the NSI scans at four detector rotations: what `real_scan_rotation_recon.py` measured

Date: 2026-09-04.  Slurm job 15933899 on gautschi, one NVIDIA H100 80GB HBM3, torch
2.13.0+cu130, mbirtorch 0.0.2 at commit `4781600` on the `geometric_calibration` branch.  The job
asked for two GPUs so that it would hold 252 GB of host memory, and it pinned mbirtorch to one
device.  `sacct` reported `COMPLETED`, exit code `0:0`, elapsed `00:09:35`, and a batch-step
`MaxRSS` of 40394904 KB.  The script's own peaks were 40.6 and 41.4 GB on the two scans.  Every
number below was read in the same session from the job's output, which is the log
`/scratch/gautschi/buzzard/leap_cmp/real_scan_rotation_recon_15933899.log` and the JSON lines of
`results_real_scan_rotation_recon/real_scan_rotation_recon.jsonl` in the same directory, or was
computed from the slice arrays the job saved there by the companion script
`real_scan_rotation_recon_metrics.py` beside this page, which ran on a Mac.  The job also wrote one
figure per slice.  This repository ignores such files, so this page is the durable copy.

Units.  A detector rotation is given in degrees.  An offset is given in channels, which are the
detector pitch of 0.127 mm on these scans.  A slice's height is given in rows from the central
plane, which is the plane through the source that is perpendicular to the rotation axis.  The
estimator is `estimate_det_rotation` in `mbirtorch/preprocess/geometry_calibration.py`.

## The answer

The vendor's recorded tilt of 0.167 degrees is the right detector rotation for the NSI scan without
metal, and the estimator's 0.044 degrees is not.  Greg asked for direct reconstructions at the two
rotations, because a direct reconstruction has no prior that could absorb a geometry error.  The
job reconstructed five slices of the scan at four rotations: none, 0.044 degrees, 0.167 degrees,
and 0.19 degrees.  On the slice 752 rows above the central plane the phantom's dark horizontal line
has a depth of 0.0080 with no rotation applied, 0.0097 at 0.044 degrees, 0.0128 at 0.167 degrees,
and 0.0121 at 0.19 degrees.  Its width at half depth is 8, 6, 4, and 5 rows in the same order.  A
sharpness measure taken after a two-pixel blur, which removes the pixel-scale differences that
resampling causes, ranks 0.167 degrees first on every slice away from the central plane, on both
scans.  These results indicate that the estimator's zero point is wrong by about 0.12 degrees on
this scan, which is 1.6 pixels at the edge of the detector.  The estimator follows an added rotation
with a slope of one on the same scan (`real_scan_rotation_check.md`), so its error is in where it
puts zero and not in how it responds to a change.

On the scan with metal the same measures put 0.167 degrees slightly ahead of 0.19 degrees, the
value the added rotations implied for that scan.  The estimator's zero point there is off by about
0.02 degrees.  These results indicate that the vendor's report describes both acquisitions, and
that the estimator's zero point depends on the object.  On the phantom without metal, whose
structure runs along the detector rows, the error is 0.12 degrees.  On the same phantom with a
metal insert, which adds structure across the rows, the error is 0.02 degrees.

The job's own sharpness numbers do not show this, and the reason is worth recording.  The slice
with no rotation applied is the only one that is not resampled, and it scores highest on both of
the job's measures on every slice, while the 0.044 degree slice scores lowest.  Bilinear resampling
smooths pixel-scale noise by an amount set by the fractional part of the displacement, and at these
heights that fraction is largest for 0.044 degrees.  A gradient measure on an unblurred slice
therefore reads the interpolation and not the geometry.  The blurred measure and the dark line's
depth and width read the geometry.

## What was measured

The job used `parameter_sweep` with the parameter `det_rotation`.  For one slice of the volume that
function crops the detector to the rows the slice needs, rotates that band of every view by the
candidate angle with the bilinear kernel that `apply_calibration` uses, and reconstructs the slice
directly.  The two NSI scans were loaded as the earlier jobs loaded them, with the vendor's tilt held
out of the sinogram.  Five slices were reconstructed per scan, at a tenth, a quarter, a half, three
quarters, and nine tenths of the way through the volume's 1880 slices.  The middle slice lies on the
central plane, where a detector rotation moves nothing, and the other four lie 470 and 752 rows
below and above it.  The table gives the slices.

| slice | height, mm | rows from the central plane | seconds for four rotations, without metal | with metal |
| --- | --- | --- | --- | --- |
| 188 | -27.4 | -752 | 19.5 | 12.5 |
| 470 | -17.1 | -470 | 8.7 | 8.8 |
| 940 | 0.0 | 1 | 3.4 | 3.4 |
| 1409 | 17.1 | 470 | 8.7 | 8.8 |
| 1691 | 27.4 | 752 | 12.3 | 12.4 |

The slice at 752 rows below the central plane lies below the phantom, and its two scans give the
same numbers to three digits, so it carries no information about the rotation.  The four other
slices are read below.  Each scan took under five minutes in all, of which 31 seconds was the
load.

## The job's sharpness measures

The job recorded two measures per slice and rotation.  The first is the mean squared finite
difference along both axes divided by the slice's mean square.  The second is the variance of the
four-neighbor Laplacian divided by the mean square.

| scan | slice | rows from the central plane | gradient measure at 0 | at 0.044 | at 0.167 | at 0.19 | Laplacian measure at 0 | at 0.044 | at 0.167 | at 0.19 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| without metal | 470 | -470 | 0.0409 | 0.0117 | 0.0133 | 0.0118 | 0.161 | 0.0313 | 0.0362 | 0.0306 |
| without metal | 940 | 1 | 0.0228 | 0.0194 | 0.0172 | 0.0169 | 0.0857 | 0.0711 | 0.0615 | 0.0601 |
| without metal | 1409 | 470 | 0.0373 | 0.0103 | 0.0104 | 0.0129 | 0.147 | 0.0269 | 0.0268 | 0.0394 |
| without metal | 1691 | 752 | 0.0602 | 0.0168 | 0.0258 | 0.0231 | 0.236 | 0.0450 | 0.0786 | 0.0722 |
| with metal | 470 | -470 | 0.0396 | 0.0112 | 0.0130 | 0.0116 | 0.156 | 0.0304 | 0.0353 | 0.0301 |
| with metal | 940 | 1 | 0.0085 | 0.0080 | 0.0074 | 0.0073 | 0.0173 | 0.0154 | 0.0132 | 0.0128 |
| with metal | 1409 | 470 | 0.0363 | 0.0100 | 0.0102 | 0.0126 | 0.143 | 0.0260 | 0.0260 | 0.0382 |
| with metal | 1691 | 752 | 0.0782 | 0.0213 | 0.0332 | 0.0297 | 0.309 | 0.0588 | 0.104 | 0.0953 |

The unresampled slice scores three to four times higher than any resampled one on every slice
away from the central plane.  These results indicate that both measures are dominated by
pixel-scale noise, which resampling smooths.  On the central plane, where a rotation moves nothing,
the measures still fall from no rotation to 0.19 degrees, which shows the same smoothing with no
geometry in it.

## The measures after a blur

The companion script blurs each slice with a Gaussian of two pixels before it takes the gradient
measure.  The blur removes the pixel-scale noise that the resampling smooths, so what remains is
the sharpness of the phantom's edges.

| scan | slice | rows from the central plane | blurred measure at 0 | at 0.044 | at 0.167 | at 0.19 | sharpest |
| --- | --- | --- | --- | --- | --- | --- | --- |
| without metal | 470 | -470 | 0.00095 | 0.00096 | 0.00105 | 0.00104 | 0.167 |
| without metal | 1409 | 470 | 0.00080 | 0.00079 | 0.00081 | 0.00080 | 0.167 |
| without metal | 1691 | 752 | 0.00106 | 0.00112 | 0.00129 | 0.00123 | 0.167 |
| with metal | 1409 | 470 | 0.00077 | 0.00078 | 0.00081 | 0.00080 | 0.167 |
| with metal | 1691 | 752 | 0.00111 | 0.00112 | 0.00129 | 0.00125 | 0.167 |

The rotation of 0.167 degrees gives the sharpest slice in every row of the table.  The unresampled
slice is no longer the sharpest in any row.  These results indicate that the ordering follows the
geometry and not the interpolation.  The spread is largest at 752 rows, where the two candidates
of 0.044 and 0.167 degrees displace the center of rotation by 1.3 channels relative to each other,
and it is smallest at 470 rows above the plane, where the phantom's structure at that height is
weak.

## The dark line

The phantom carries a thin dark horizontal line, and a wrong rotation smears it.  The companion
script finds the line as the row inside the phantom's disk with the lowest mean over 240 central
columns, takes the mean vertical profile through it, and measures the dip's depth below the
surrounding level and its width at half that depth.

| scan | slice | rows from the central plane | depth at 0 | at 0.044 | at 0.167 | at 0.19 | width in rows at 0 | at 0.044 | at 0.167 | at 0.19 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| without metal | 470 | -470 | 0.0021 | 0.0021 | 0.0021 | 0.0021 | 8 | 8 | 9 | 9 |
| without metal | 1409 | 470 | 0.0005 | 0.0006 | 0.0007 | 0.0007 | 7 | 4 | 4 | 4 |
| without metal | 1691 | 752 | 0.0080 | 0.0097 | 0.0128 | 0.0121 | 8 | 6 | 4 | 5 |
| with metal | 1409 | 470 | 0.0021 | 0.0021 | 0.0020 | 0.0020 | 3 | 3 | 3 | 3 |
| with metal | 1691 | 752 | 0.0014 | 0.0014 | 0.0013 | 0.0013 | 3 | 3 | 2 | 2 |

The slice 752 rows above the central plane on the scan without metal is the decisive row.  There
the line's depth rises by 60 percent from no rotation to 0.167 degrees and its width falls from 8
rows to 4.  The line at 0.044 degrees is 24 percent shallower and half again as wide as at 0.167.
On the slice 470 rows below the plane the measure does not resolve the line at any rotation, and on
the scan with metal the line is faint and equally narrow at 0.167 and 0.19 degrees.  These results
indicate that the line measure decides the question where the line is resolved, and agrees with the
blurred measure there.

## What the figures show

The job's figure for each slice shows the four reconstructions on one gray scale and the
difference between the 0.044 and 0.167 degree slices.  On the slices away from the central plane
the difference image holds the phantom's edges and its dark line, and it holds noise elsewhere.  A
zoomed figure made by the companion script from the saved arrays, on the slice 752 rows above the
plane of the scan without metal, shows the dark line's right end and the rim's notches.  The line
is broad and soft with no rotation applied, broad at 0.044 degrees, thin and sharp at 0.167 degrees,
and slightly softer again at 0.19 degrees.  The notches on the rim follow the same order.

## Limits of this evidence

Six limits apply.  One scanner and one phantom in two forms were reconstructed.  The candidates
were four fixed rotations, so the measures say which of the four is best and not where the optimum
lies between them; the two nearest, 0.167 and 0.19 degrees, differ by 0.023 degrees and the
measures separate them by a few percent.  Every nonzero candidate was applied by bilinear
resampling, and the blurred measure removes most but not all of what that resampling changes.  The
line measure resolves the line on two slices of five.  The reconstructions are direct, so the
cone-beam approximation is in every one of them alike.  The blurred and line measures were computed
on a Mac from the saved arrays rather than in the job, and the companion script beside this page is
what computed them.

## The batch file

The job was submitted from `/scratch/gautschi/buzzard/leap_cmp` with
`sbatch real_scan_rotation_recon.sbatch`.  That file is the follow-up job's batch file with four
changes: the job name `real_scan_rotation_recon`, the log
`/scratch/gautschi/buzzard/leap_cmp/real_scan_rotation_recon_%j.log`, the compile cache directory
`torch_cache_rotation_recon`, and the results directory `results_real_scan_rotation_recon`.  It
requests `-A bouman -p ai -q normal -N 1 --gpus-per-node=2 --cpus-per-task=28 -t 03:00:00`,
sources `~/load_conda_cuda.sh`, sets `set -e`, exports `TORCHINDUCTOR_CACHE_DIR`, `MPLBACKEND=Agg`,
`MBIRTORCH_NUM_DEVICES=1`, `PYTHONPATH`, `REAL_SCAN_RESULTS`, and `REAL_SCAN_DATA`, runs the
one-line interpreter check, and runs `venv/bin/python -u real_scan_rotation_recon.py`.
