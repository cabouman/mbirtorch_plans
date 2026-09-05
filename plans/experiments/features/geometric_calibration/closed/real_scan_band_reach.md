# Bands that reach the structure on the no-metal NSI scan: what `real_scan_band_reach.py` measured

Date: 2026-09-05.  Slurm job 15951659 on gautschi, one NVIDIA H100 80GB HBM3 used of two
requested, torch 2.13.0+cu130, mbirtorch 0.0.2 at commit `4781600` on the `geometric_calibration`
branch.  `sacct` reported `COMPLETED`, exit code `0:0`, elapsed `01:08:25`.  The script's own peak
was 138.7 GB of host memory, which is why the job asks for two GPUs, whose allocation carries
252 GB.  Every number below was read in this session from the job's output, which is the log
`/scratch/gautschi/buzzard/leap_cmp/real_scan_band_reach_15951659.log` and the JSON lines of
`results_real_scan_band_reach/real_scan_band_reach.jsonl` in the same directory.  The script is a
small wrapper that reruns `real_scan_band_height.py` on the no-metal scan at two tall bands, with
that job's loading, fixed offset, and recorded fields.

Units are the band-height record's: rotations in degrees, offsets in channels at the 0.127 mm
pitch, band heights in detector rows, edge displacement in pixels.

## The answer

The estimate recovers most of its error before the band reaches the object's structure, and then
it stops.  At the default band of 7 rows the estimate is 0.0471 degrees, as before.  At 501 rows,
which is 250 rows on each side of the central plane and short of the structure at 470 to 752
rows, it is 0.1299 degrees.  At 1001 rows, whose edge at 500 rows reaches the nearest structure,
it is 0.1299 degrees again, identical to every recorded digit, which is the search returning the
same point of its deterministic lattice.  The vendor's value is 0.167 degrees.

Two readings follow.  First, the signal that moves the estimate lives between 100 and 250 rows
from the central plane, because the band-height sweep read 0.0705 at 100 rows
(`real_scan_band_height.md`) and this job reads 0.1299 at 250, while adding rows 250 to 500
changes nothing.  The structure named from the reconstructions, at 470 rows and beyond, is not
what the recovery used.  Second, the tall-band answer sits 0.037 degrees below the vendor's
value, and no measurement so far says which of the two is right for reconstruction quality.  The
far-slice job tested only the candidates 0, 0.044, 0.167, and 0.19 degrees, and its record states
that it ranks those four and does not locate the optimum between them
(`real_scan_rotation_recon.md`).  A fine sweep of reconstructed far slices over roughly 0.10 to
0.20 degrees would settle it, and it is the same computation the proposed reconstruction
estimator makes (`estimate_by_recon.md`).

## The measurements

The channel offset was estimated once at the default settings, -14.147 channels, 0.022 from the
vendor's value, and held fixed, as in the band-height job.

| band, rows | row window | estimate, deg | edge displacement, px | min score | max/min | cross-row | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | 954-961 | 0.0471 | 0.62 | 4.56e-04 | 115.6 | 1.60e-04 | 36.4 |
| 501 | 707-1208 | 0.1299 | 1.70 | 4.32e-03 | 10.9 | 1.77e-04 | 1249.0 |
| 1001 | 457-1458 | 0.1299 | 1.70 | 7.13e-03 | 9.7 | 1.96e-04 | 2578.7 |

Three details of the table matter.  The tall-band estimates sit at 1.70 pixels of edge
displacement, above the one-pixel regime where the resampling bias lives, and neither raised the
sub-pixel warning; the default band's estimate sits at 0.62 pixels and did.  The score's floor
grows with the band, from a max/min ratio of 116 down to about 10, and the minimum stays
well-formed.  The cross-row statistic barely moves, 1.6e-04 to 2.0e-04, so even the 1001-row band
is nearly uniform across rows on average, and whatever the recovery reads is a small part of the
band.

## What this says about the tall band as a remedy

A tall band repairs most of the zero-point error on this scan, at a real cost.  One estimate at
501 rows took 21 minutes and one at 1001 took 43, both at every view, and the 1001-row band drove
the process to 139 GB of host memory.  A view stride of 4 would cut the time by about four at a
cost the stride study bounded at 0.008 channels for the offset (`conjugate_offset_recovery.md`),
and 501 rows suffices, but the method still lands 0.037 degrees from the vendor's value with no
measurement yet saying which is right.  The reconstruction estimator reaches the same question
from the other side in minutes, so the fine far-slice sweep above is the next measurement either
way.

## Limits of this evidence

Four limits apply.  One scan was measured, and the two tall bands violate the cone rule the
default band obeys, by design; the growing score floor is that violation, and at a larger cone
angle it could do more than raise the floor.  The offset was held at the value estimated at zero
rotation.  The identical estimates at 501 and 1001 rows agree to the search's stopping width of
0.005 degrees, not to their printed digits.  The 0.1299 itself carries the search lattice's
granularity for the same reason.

## The batch file

The job was submitted from `/scratch/gautschi/buzzard/leap_cmp` with `sbatch
real_scan_band_reach.sbatch`.  That file is the band-height job's batch file with the job name,
log, compile-cache directory, and results directory renamed to `real_scan_band_reach`, a walltime
of four hours, and the final line `venv/bin/python -u real_scan_band_reach.py`.  The two GPUs are
for the 252 GB of host memory the 1001-row band needs.
