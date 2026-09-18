# A rotation axis leaning out of the detector plane, simulated with LEAP: what `leap_axis_tilt.py` measured

Date: 2026-09-05.  Slurm job 15951439 on gautschi, one NVIDIA H100 80GB HBM3, torch
2.13.0+cu130, LEAP 1.26, mbirtorch 0.0.2 at commit `4781600` plus the staged working-tree edits
on the `geometric_calibration` branch.  `sacct` reported `COMPLETED`, exit code `0:0`, elapsed
`00:00:49`; the script's own peak was 10.7 GB of host memory.  Every number below was read in
this session from the job's output, which is the log
`/scratch/gautschi/buzzard/leap_cmp/leap_axis_tilt_15951439.log` and the JSON lines of
`results_leap_axis_tilt/leap_axis_tilt.jsonl` in the same directory.

Units.  A rotation is given in degrees.  The in-plane angle turns the rotation axis within the
detector plane, which an in-plane rotation of the sinogram can correct.  The out-of-plane angle
leans the axis along the detector normal, which no in-plane rotation can correct.  A band is the
window of detector rows the rotation estimator compares.

## The question

The no-metal NSI scan's config vectors split its misalignment into an in-plane rotation of 0.167
degrees and an out-of-plane lean of 0.079 degrees (`real_scan_band_height.md`).  The rotation
estimator anchors near 0.047 degrees on that scan.  This job asks whether the lean is the anchor:
it generates scans with a leaning axis inside LEAP's modular-beam projector, which shares no code
with mbirtorch, and runs mbirtorch's estimators on them.  The geometry, the two phantoms, and the
in-plane angle of 1.5 degrees repeat `rotation_zero_point_synthetic.md`, so the in-plane cases
also replicate that experiment through an independent projector.  The out-of-plane angle of 0.65
degrees is the real lean scaled to this smaller detector: it displaces a point at the cylinder's
28-pixel radius vertically by about 0.6 pixels between a view and its opposite, which is what
0.079 degrees does at the real phantom's roughly 236-pixel radius.

## The answer

The lean does not move the in-plane estimate.  With no in-plane angle, a lean of 0.65 degrees
changes the estimate by -0.0007 degrees at the default band and by 0.0000 at 127 rows, on both
phantoms.  With the in-plane angle of 1.5 degrees present, the lean changes the estimate by
-0.0110 degrees at the default band on the far phantom, -0.0086 on the near phantom, and -0.0038
on both at 127 rows.  Every change is two orders of magnitude below the 0.12-degree error on the
real scan.  These results reject the lean as the anchor of the real scan's estimate, at this
displacement scale and on these noiseless phantoms.  The metal scan agrees from the real side:
it carries the same lean and estimates its rotation well (`real_scan_band_height.md`).

The in-plane behavior replicates across projectors.  On the far phantom, whose slab sits outside
the default band, the default band reads -1.1036 degrees for a true magnitude of 1.5 and the
taller bands recover it, -1.5279 at 65 rows and -1.4365 at 127.  On the near phantom the default
band already reads -1.4528.  The mbirtorch-generated version of the same experiment gave +1.137,
+1.526, +1.469, and +1.423 for the same cases (`rotation_zero_point_synthetic.md`).  The
magnitudes agree case by case, and the sign difference is the two constructions' conventions: one
tilts the data, the other tilts the axis.  A failure that reproduces through a projector that
shares no code with mbirtorch is a property of the estimator and the object, not of any
generation path.

## What was measured

The geometry matches the earlier synthetic run: 128 views over a full rotation, a flat detector
of 128 rows by 160 channels at 1 mm pitch, source to iso 400 mm, source to detector 800 mm.  LEAP
chose the volume, 160 by 160 by 128 voxels of 0.5 mm.  The two phantoms are the cylinder of
radius 28 voxels with one darker slab of 5 slices, centered 78 percent of the way to the top
('far', slices 111 to 116) or on the central plane ('near', slices 62 to 67).

Each case orbits the nominal cone-beam gantry about a tilted axis, built per view for LEAP's
modular-beam geometry.  At the view of angle zero the gantry is exactly nominal and the axis is
what leans, which is how a real scanner is misaligned.  The recorded axis vectors confirm each
case: the in-plane cases tilt the axis by sin(1.5 degrees) toward the detector's column plane,
the out-of-plane cases by sin(0.65 degrees) along the normal, and the leaning cases' sinograms
differ from the flat ones (mean 7.5884 against 7.5917), so the null result is a measurement and
not a no-op.  mbirtorch's estimators then ran on each sinogram: the channel offset at defaults,
then the rotation at four band heights with that offset passed in.

| phantom | in-plane, deg | out-of-plane, deg | offset, ch | rotation at default band | at 33 rows | at 65 | at 127 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| far | 0.0 | 0.0 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| far | 1.5 | 0.0 | -0.0124 | -1.1036 | -1.3514 | -1.5279 | -1.4365 |
| far | 0.0 | 0.65 | +0.0000 | -0.0007 | +0.0000 | +0.0000 | +0.0000 |
| far | 1.5 | 0.65 | -0.0124 | -1.1146 | -1.3514 | -1.5279 | -1.4403 |
| near | 0.0 | 0.0 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| near | 1.5 | 0.0 | -0.0235 | -1.4528 | -1.4312 | -1.5125 | -1.4389 |
| near | 0.0 | 0.65 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| near | 1.5 | 0.65 | -0.0216 | -1.4614 | -1.4374 | -1.5154 | -1.4427 |

Every LEAP projection took under a hundredth of a second on the H100, and the whole job took 35
seconds of script time.

## The geometry checks, and how the failed one reads

Three checks ran before the cases, and one of them failed in a way that needs its verdict stated.

The check that matters passed.  The no-tilt modular sinogram, converted to mbirtorch's
convention, was compared against mbirtorch's own projection of the same phantom on the same
grid.  The agreement is 1.5 percent normalized error with the rows as they are, against 9.0
percent with the rows flipped, so the modular geometry corresponds to the estimating model with
the detector rows running the right way.  Two different projectors explain the 1.5 percent.

The check against LEAP's own cone-beam geometry failed, and the failure decodes as a frame
choice, not a physics error.  The no-tilt modular projection differs from LEAP's cone-beam
projection of the same phantom by 19 percent at the worst pixel, 1.5 percent normalized.  LEAP's
own cone-to-modular converter names the cause: its source positions differ from the derived ones
by 565.7 mm, which is the source distance times the square root of two, its column vectors by
1.414, and its row vectors by exactly 0.0.  The derived nominal pose therefore differs from
LEAP's cone convention by a rigid quarter turn about the rotation axis, which is the ambiguity
the script's comments document: LEAP's docstrings do not fix where the view angle zero sits.  A
rigid turn about the axis changes nothing for these phantoms, which are rotationally symmetric,
and the row vectors, the one sign a turn cannot hide, agree exactly.  The script's printed
warning, that nothing below the failed check means anything, is written for a geometry error and
overstates this case; the passing mbirtorch check and the case-by-case replication of the
earlier synthetic run are the evidence the cases stand on.

One caveat stays with that reading.  The mbirtorch comparison used the rotationally symmetric
cylinder, so it pins the row direction and the scale and not the channel direction or the view
order.  For these phantoms neither of those can change any measured number, because the objects
are the same at every angle about the axis, but a future use of this harness on an asymmetric
object should first rerun the lopsided-probe comparison against mbirtorch rather than against
LEAP's cone geometry.

## Limits of this evidence

Six limits apply.  The phantoms are noiseless, rotationally symmetric, and sharp-edged, so a
lean's effect that needs noise, asymmetry, or soft edges to matter would not show here.  One
lean size was tested, chosen to match the real scan's displacement at the object's radius; the
detector is 12 times narrower than the real one, so effects that scale with absolute size are
not reproduced.  The offset was estimated per case rather than held at truth, and the in-plane
cases show it off by up to 0.024 channels.  The estimating model's cone rule set the default
band; its height is in the tables of the JSON file rather than printed here.  The quarter-turn
frame difference against LEAP's cone geometry is understood but not corrected, so the harness
should not be reused on an asymmetric object without the check described above.  The sign
convention between an axis tilt and the estimator's answer was recorded, not derived.

## The batch file

The job was submitted from `/scratch/gautschi/buzzard/leap_cmp` with `sbatch
leap_axis_tilt.sbatch`.  That file is the band-height job's batch file with one GPU instead of
two (`--gpus-per-node=1 --cpus-per-task=14`), a walltime of one hour, the job name, log, and
compile-cache directory renamed to `leap_axis_tilt`, the results directory exported as
`LEAP_AXIS_TILT_RESULTS=$PWD/results_leap_axis_tilt`, no data directory, and the final line
`venv/bin/python -u leap_axis_tilt.py`.  The venv holds both LEAP and mbirtorch, which the
one-line interpreter check verifies before the script starts.
