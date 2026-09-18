# What geometry metadata the two Zeiss .txrm files hold: what `zeiss_metadata_probe.py` found

Date: 2026-09-05.  Slurm job 15989128 on gautschi ran the probe on the two files the earlier jobs
used: the bga scan `/depot/bouman/data/Zeiss/purdue_BGA/17U1-250TC-Normal_Tomo_No_HART.txrm` and
the z62 scan `/depot/bouman/data/ORNL/versa/ParAM-Round-1_Z62.txrm`.  `sacct` reported `COMPLETED`,
exit code `0:0`, elapsed `00:00:22`.  Every number below was read in this session from the job's
log, `/scratch/gautschi/buzzard/leap_cmp/zeiss_metadata_probe_15989128.log`.  This repository
ignores `.log` files, so this page is the durable copy of what it quotes.  The question came from
Greg during the review of the fine-sweep findings: do the Zeiss files record a detector tilt, and
do they hold per-view correction data or a per-view coordinate frame?

## The answers

The files record no detector tilt.  The format has an explicit field for one, named
`CameraFineRotation`, and it is 0.0 in both files, in both the places the scan carries it
(`ImageInfo/CameraFineRotation` and `DetAssemblyInfo/CameraFineRotation`).  On the bga file the
instrument configuration's per-camera `CameraFineRotationDeg` entries are 0.0 as well.  The only
stream with "tilt" in its name is a metrology-sensor configuration entry.  For these scanners the
reconstruction estimator is therefore the only source of a tilt value.

The files do hold per-view correction data, and the reader already applies the totals.  Each file
stores one x shift and one y shift per view under `Alignment/X-Shifts` and `Alignment/Y-Shifts`,
which `read_metadata` in `mbirtorch/preprocess/zeiss.py` reads and `correct_sino_shifts` applies
as per-view offsets.  Behind the totals the file stores the decomposition by cause, which the
reader does not parse: encoder shifts, static runout with a 361-point calibration table over
-180.05 to 180.05 degrees, temperature drift, and source drift, with a flag per cause saying
whether it was folded into the totals.  On both files the encoder term dominates the totals.

The files also hold the per-view coordinate frame Greg asked about.  `PositionInfo/MotorPositions`
records the position of every stage axis at every view, in raw and ideal variants, with the axis
names and units beside it.  The bga file has 13 axes per view and the z62 file 14, and each file's
`TotalAxis` field matches its count.

## The shift magnitudes

The table gives the per-view arrays of the main scan of each file, in the pixel-scale units the
reader applies.  Each row is a float32 array with one value per view: 2401 views for bga and 801
for z62.

| stream | bga min | bga max | z62 min | z62 max |
| --- | --- | --- | --- | --- |
| `Alignment/X-Shifts` (the applied total) | -9.98 | 10.10 | -21.4 | 20.8 |
| `Alignment/Y-Shifts` (the applied total) | -1.07 | 1.04 | -7.24 | 7.56 |
| `Alignment/EncoderXShifts` | -9.85 | 9.98 | -19.7 | 19.9 |
| `Alignment/StaticRunoutX` | -0.159 | 0.137 | -0.160 | 0.171 |
| `Alignment/TemperatureXShifts` | -0.073 | 0.072 | -0.019 | 0.017 |
| `Alignment/SourceDriftXShifts` | -0.179 | 0.000 | -0.721 | 0.000 |

Two readings of the table matter.  The totals reach ten to twenty pixels, so a reconstruction of
these scans without the per-view correction would be badly blurred, and the reader's application
of the totals is doing real work.  The non-encoder terms are tenths of a pixel or less, so parsing
the decomposition would change little; its value is diagnostic.

The applied flags of the bga main scan read: encoder, spot, and static-runout shifts applied;
metrology, reference, source-spot, stage, and user shifts not applied.  The flags are stored as
the float32 bit patterns of the integers 0 and 1.

## What the reader parses, and what it does not

`read_metadata` parses the per-view angles, the per-view source and detector distances, the
per-view x, y, and z stage positions, the two alignment shift totals, the reconstruction center
shift (returned as the negated `det_channel_offset`), the axis name and unit lists, the pixel
size, and the per-view currents, voltages, and exposure times.  Of the streams the probe's
geometry keywords matched, 601 of 611 on bga and 648 of 658 on z62 are not parsed.  Most of the
unparsed ones are configuration echoes.  The substantive unparsed data are the shift
decomposition, the applied flags, and the motor-position tables described above.

The bga axis list reads: Sample X, Sample Y, Sample Z, Sample Theta, Source Z, Detector Z, CCD_Z,
CCD_X, Flat Panel X, Flat Panel Z, MkIV Filter Wheel, DCT, Source X.  The z62 list replaces the
flat-panel and filter entries with autoloader and filter-wheel-stage axes and is otherwise the
same.

## Limits of this evidence

Three limits apply.  Two files from two instruments were probed, and stream sets vary with the
instrument and software version.  The probe decodes each stream heuristically, and some float
arrays print as text in the log; every value quoted here came from a stream that decoded cleanly
as float32.  The units of the shift arrays are not named in the file; the values are pixel-scale
and the reader applies them as pixel offsets, and no independent check of that reading was made
here.

## The batch file

The job was submitted from `/scratch/gautschi/buzzard/leap_cmp` with `sbatch
zeiss_metadata_probe.sbatch`.  That file requests `-A bouman -p ai -q normal -N 1
--gpus-per-node=1 --cpus-per-task=14 -t 00:15:00`, writes the log
`zeiss_metadata_probe_%j.log`, sources `~/load_conda_cuda.sh`, and runs
`venv/bin/python -u zeiss_metadata_probe.py`.  The work is file reading on the CPU; the GPU is
requested because the partition allocates CPUs and host memory per GPU.
