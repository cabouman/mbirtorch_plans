# The estimators on five real scans, against the vendor values and LEAP: what `real_scan_validation.py` measured

Date: 2026-09-04.  Slurm job 15925593 on gautschi, one NVIDIA H100 80GB HBM3, torch 2.13.0+cu130,
LEAP 1.26, mbirtorch 0.0.2 at commit `4781600` on the `geometric_calibration` branch.  The job
asked for two GPUs so that it would hold 252 GB of host memory, and it pinned mbirtorch to one
device with `MBIRTORCH_NUM_DEVICES`.  `sacct` reported `COMPLETED`, exit code `0:0`, elapsed
`00:21:25`, and a batch-step `MaxRSS` of 46489160 KB.  The script's own peak was 50.2 GB, measured
with `getrusage`.  `sacct` samples periodically, so the smaller of the two numbers is expected.
Every number below was read from the job's output in the same session.  That output is the log,
`/scratch/gautschi/buzzard/leap_cmp/real_scan_validation_15925593.log`, and the 64 JSON lines of
`results_real_scan/real_scan_validation.jsonl` in the same directory.  The job also wrote four
`*_sweep.png` figures and four `*_sweep.npz` arrays there.  This repository ignores such files, so
this page is the durable copy.

Units.  An offset is given in channels, and a channel is the detector pitch of the scan in
question.  Three pitches appear here: 0.127 mm on the NSI scans, 0.0135 mm on `z62`, and 0.1496 mm
on `bga`.  A rotation is given in degrees, and its edge displacement is the distance it moves the
edge channel of the detector, in pixels.  The estimators under test are `estimate_det_channel_offset`,
`estimate_det_rotation`, and `check_rotation_direction` in
`mbirtorch/preprocess/geometry_calibration.py`, called the module below.  A vendor value is the
value the scanner's own calibration recorded, as the reader for that scanner reports it.  The LEAP
comparison is the study recorded in `plans/features/leap_comparison/`.

## The answers

The estimators passed all three gates on every scan they could measure.  Four of the five scans
were measured.  The largest roll error was 0.048 channels, against a gate of 0.1 channels.  The
largest robustness difference was 0.069 channels, against a gate of 0.1 channels.  The largest
difference from a vendor value was 0.074 channels, against a gate of 0.25 channels.  These results
indicate that the offset estimator holds up on real data from two scanners, at least to a tenth of
a channel.

The offset estimates agree with the vendor values to better than 0.08 channels.  On `bga` the
estimate is 0.546 channels and the vendor value 0.528 channels.  On the three NSI scans the
estimates are -14.199, -14.178, and -14.188 channels, against a vendor value of -14.125 channels.
LEAP's `find_centerCol` agrees with the module on `bga` to 0.001 channels.  On the NSI scans LEAP
reads higher than the module by 0.98 channels on `nsi_small`, 0.46 channels on `nsi_no_metal`, and
0.21 channels on `nsi_metal`.  The NSI data carry a detector tilt that the vendor recorded and this
job held out of the sinogram.  That tilt displaces the edge channel by 2.18 pixels.  The module
estimates the rotation and applies it before it pairs opposite views.  LEAP's `find_centerCol`
searches on one detector row, as `calibration_512_gautschi.md` records, so a tilt stays in what it
reads.  That difference is a hypothesis for the gap on the NSI scans, and two facts count against it.
Over the 128-row band LEAP was given, that tilt shifts a row's content by at most 0.19 channels,
which is smaller than any of the three gaps.  The three scans also carry the same tilt, and the
gap differs by a factor of five among them.  The cause of the gap is therefore not known.  This
job did not test it.

The rotation estimates on the NSI scans do not match the vendor tilt.  The vendor tilt is 0.167
degrees on all three.  The module read 0.047 degrees on `nsi_small` and on `nsi_no_metal`, and
0.149 degrees on `nsi_metal`.  The differences are -0.120, -0.120, and -0.019 degrees.  The two
scans without metal returned the same rotation to every recorded digit, 0.0008228244347475705
radians, though one has 200 views and the other 1800.  This record cannot say which of the two
values is right, because the vendor tilt is itself an estimate and nothing else here measures the
tilt.  The follow-up job measures the rotation a second way.  LEAP's `estimate_tilt` returned
5.000000 degrees on `nsi_small` and on `nsi_no_metal`, which is the bound of its five degree
search.  A value at the bound is a failed search rather than an estimate.  On `nsi_metal` LEAP
returned 0.006 degrees and on `bga` -0.402 degrees, against the module's 0.149 and 0.017 degrees.

The rotation-direction check agreed with the reader's geometry on three scans and warned on the
fourth.  On `bga`, `nsi_no_metal`, and `nsi_metal` it returned 1.0, which keeps the angles as the
reader built them, with ratios of 3.63, 9.07, and 17.32.  On `nsi_small` it returned
-1.0, which asks for the angles to be negated, with a ratio of 1.05.  `nsi_small` and
`nsi_no_metal` are the same object on the same scanner, at 200 views and at 1800 views, so those
two answers disagree.  The module warned on `nsi_small` and on no other scan, because its ratio
fell below the margin of 1.5 that the check expects.  These results indicate that the check's own
warning marked the one answer that disagrees with the rest.

The fan angle did not order the margins the way the check's documentation leads one to expect.
`bga` has a full fan angle of 44.7 degrees and the NSI scans have 18.4 degrees.  The check's
docstring says the separation between the two directions grows with the fan angle.  `bga`'s ratio
of 3.63 is smaller than the NSI ratios of 9.07 and 17.32.  The reduced problems differ as well.
`bga` ran at a view stride of 1 and the NSI scans at the default stride of 4.  The docstring also
says a ratio measured on one scan does not transfer to another, so these four ratios cannot be
compared directly.

One scan could not be measured.  `z62` covers 218.0 degrees, with a gap of 142.0 degrees between
neighboring views, so the module refused it as a short scan.  Nothing else was measured on it.
This is the first real scan of that kind the feature has met.  A follow-up job,
`real_scan_followup.py` beside this page, takes it up.  That job had not run when this page was
written.

## The gates

The three gates are the ones the script's docstring names.  A roll of the sinogram by a known
number of channels must move the offset estimate by that number, within 0.1 channels.  Each
robustness case must leave the offset estimate within 0.1 channels of the estimate on the
unmodified data.  The offset estimate must agree with the vendor value within 0.25 channels.  The
third one is a sanity check rather than a hard gate, because the vendor value is itself an
estimate.

| dataset | largest roll error, channels | largest robustness difference, channels | vendor difference, channels | roll gate | robustness gate | vendor gate |
| --- | --- | --- | --- | --- | --- | --- |
| `nsi_small` | 0.003 | 0.026 | -0.074 | pass | pass | pass |
| `bga` | 0.009 | 0.003 | +0.019 | pass | pass | pass |
| `nsi_no_metal` | 0.003 | 0.018 | -0.053 | pass | pass | pass |
| `nsi_metal` | 0.048 | 0.069 | -0.063 | pass | pass | pass |
| `z62` | not measured | not measured | not measured | not measured | not measured | not measured |

## The scans

The five scans come from two scanners.  Three are NSI scans of one object, an artifact phantom
scanned vertically.  `nsi_small` keeps 200 views of it, `nsi_no_metal` keeps 1800 views of the same
object, and `nsi_metal` keeps 1800 views of it with a metal insert added.  The other two are Zeiss
Versa scans.  `z62` is the scan the depot names `ParAM-Round-1_Z62`, and `bga` is a ball grid array package.
Every scan loaded at full resolution.

The full fan angle below is `2 * atan(channels * pitch / 2 / source_detector_dist)`.  The vendor
tilt and its edge displacement exist only for the NSI scans, because the NSI reader records a
detector tilt and the Zeiss reader does not.

| dataset | scanner | views | rows | channels | pitch, mm | source to axis, mm | source to detector, mm | full fan, deg | coverage, deg | vendor offset, channels | vendor tilt, deg | edge displacement, pixels |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `nsi_small` | NSI | 200 | 1880 | 1496 | 0.1270 | 168.03 | 585.45 | 18.43 | 358.20 | -14.125 | 0.1672 | 2.18 |
| `z62` | Zeiss Versa | 801 | 2048 | 2048 | 0.0135 | 50.01 | 55.69 | 27.88 | 217.99 | -0.928 | | |
| `bga` | Zeiss Versa | 2401 | 968 | 1532 | 0.1496 | 25.55 | 278.45 | 44.74 | 359.83 | 0.528 | | |
| `nsi_no_metal` | NSI | 1800 | 1880 | 1496 | 0.1270 | 168.03 | 585.45 | 18.43 | 359.80 | -14.125 | 0.1672 | 2.18 |
| `nsi_metal` | NSI | 1800 | 1880 | 1496 | 0.1270 | 168.03 | 585.45 | 18.43 | 359.80 | -14.125 | 0.1672 | 2.18 |

The NSI sinograms were built inside the script rather than through `nsi.get_sino_and_model`.  That
reader applies the vendor's detector tilt to the sinogram, and a tilt already applied cannot be
estimated.  The script repeats the reader's steps with the tilt held out and keeps it as the vendor
value.

## The estimates

Each scan was measured in three steps, in this order: the channel offset at the model's geometry,
then the rotation, then the channel offset again at the estimated rotation.  The third value is the
one the gates use.  The offset search ran 24 evaluations on every scan and the rotation search 26.
Each estimator kept 90 percent of the view pairs on every scan, which is the trimmed mean's share.

| dataset | step | value | vendor difference | pairs kept | seconds | notes and warnings |
| --- | --- | --- | --- | --- | --- | --- |
| `nsi_small` | offset | -14.162 channels | | 180 of 200 | 1.00 | none |
| `nsi_small` | rotation | 0.0471 deg | -0.120 deg | 180 of 200 | 4.18 | the sub-pixel warning, at 0.62 pixels |
| `nsi_small` | offset at rotation | -14.199 channels | -0.074 channels | 180 of 200 | 0.99 | none |
| `bga` | offset | 0.548 channels | | 2161 of 2401 | 15.78 | none |
| `bga` | rotation | 0.0170 deg | | 2161 of 2401 | 45.95 | the sub-pixel warning, at 0.23 pixels |
| `bga` | offset at rotation | 0.546 channels | +0.019 channels | 2161 of 2401 | 15.81 | none |
| `nsi_no_metal` | offset | -14.147 channels | | 1620 of 1800 | 11.10 | none |
| `nsi_no_metal` | rotation | 0.0471 deg | -0.120 deg | 1620 of 1800 | 34.43 | the sub-pixel warning, at 0.62 pixels |
| `nsi_no_metal` | offset at rotation | -14.178 channels | -0.053 channels | 1620 of 1800 | 11.63 | none |
| `nsi_metal` | offset | -14.275 channels | | 1620 of 1800 | 11.66 | none |
| `nsi_metal` | rotation | 0.1486 deg | -0.019 deg | 1620 of 1800 | 34.51 | none, at 1.94 pixels |
| `nsi_metal` | offset at rotation | -14.188 channels | -0.063 channels | 1620 of 1800 | 11.62 | none |

The sub-pixel warning is the one `estimate_det_rotation` raises when the estimated angle moves the
edge channel by less than one pixel.  On `nsi_small` it read in full: `estimate_det_rotation: the
estimate displaces the edge channels by 0.62 pixels, where the resampling of each candidate biases
it by up to 25 percent of the angle.`  It is called the sub-pixel warning below.  It fired on three
of the four measured scans, and not on `nsi_metal`, whose estimate moves the edge channel by 1.94
pixels.  The offset steps raised no warning on any scan.  The searches recorded no notes beyond the
sub-pixel warning text.

## The roll test

The roll test needs no ground truth.  The sinogram is rolled along the channel axis by a whole
number of channels, and the offset is estimated again.  The estimate should move by the number of
channels rolled.  The roll is circular and moves no value between samples.  The roll error below is
the difference between the estimates minus the roll.

| dataset | roll, channels | estimate, channels | roll error, channels | seconds |
| --- | --- | --- | --- | --- |
| `nsi_small` | +3 | -11.198 | +0.0006 | 0.95 |
| `nsi_small` | -3 | -17.201 | -0.0025 | 0.96 |
| `bga` | +3 | 3.547 | +0.0006 | 15.82 |
| `bga` | -3 | -2.462 | -0.0087 | 15.82 |
| `nsi_no_metal` | +3 | -11.179 | -0.0013 | 11.35 |
| `nsi_no_metal` | -3 | -17.180 | -0.0025 | 11.40 |
| `nsi_metal` | +3 | -11.187 | +0.0006 | 11.63 |
| `nsi_metal` | -3 | -17.236 | -0.0477 | 11.61 |

Seven of the eight roll errors are within 0.01 channels.  The eighth, on `nsi_metal` at a roll of
-3 channels, is 0.048 channels.  These results indicate that the estimator tracks a known shift of
the data to better than a twentieth of a channel on real scans.

## The robustness cases

Each case changes a band of 64 detector rows centered on the row the central plane reaches, then
estimates the offset again, then restores the band.  The estimators compare a band of rows around
that same row, so a change inside the band is a change to what they read.  The stripe case runs
`remove_all_stripe` on the band.  The beam-hardening case adds a quadratic term sized to change the
band's largest value by ten percent, as a proxy for a linearization correction.  The zeroed-view
case sets 5 percent of the views to zero inside the band.  The difference below is against the
third estimate of the same scan.

| dataset | case | difference, channels | correction seconds | estimate seconds |
| --- | --- | --- | --- | --- |
| `nsi_small` | stripes | -0.0019 | 2.63 | 0.94 |
| `nsi_small` | beam hardening | -0.0229 | 0.07 | 0.96 |
| `nsi_small` | zeroed views, 10 of 200 | -0.0260 | | 0.94 |
| `bga` | stripes | 0.0000 | 32.66 | 14.56 |
| `bga` | beam hardening | 0.0000 | 0.46 | 14.60 |
| `bga` | zeroed views, 120 of 2401 | -0.0031 | | 14.58 |
| `nsi_no_metal` | stripes | 0.0000 | 12.96 | 11.29 |
| `nsi_no_metal` | beam hardening | -0.0180 | 0.33 | 11.25 |
| `nsi_no_metal` | zeroed views, 90 of 1800 | +0.0050 | | 11.32 |
| `nsi_metal` | stripes | +0.0019 | 13.20 | 11.74 |
| `nsi_metal` | beam hardening | -0.0031 | 0.32 | 11.53 |
| `nsi_metal` | zeroed views, 90 of 1800 | -0.0693 | | 11.72 |

The largest difference is 0.069 channels, on `nsi_metal` with views zeroed.  The zeroed-view case
is the largest difference on three of the four scans.  These results indicate that missing views
move the estimate more than either correction does, and that all three stay well inside a tenth of
a channel.

## The rotation-direction check

The check scores a direct reconstruction with the angles as given and with every angle negated, and
returns the direction that scores lower.  A value of 1.0 keeps the angles as the reader built them
and a value of -1.0 asks for them to be negated.  The ratio is the worse score divided by the
better one, and the check warns when it falls below 1.5.  `bga` needed a retry, because 2401 views
have no divisor between 2 and 4 and the default view stride of 4 was refused.  The retry used a
view stride of 1 and a bin factor of 2.

| dataset | answer | score, as given | score, negated | ratio | view stride, bin factor | margin warning | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `nsi_small` | -1.0 | 1.4745 | 1.4058 | 1.05 | 4, 2 | yes | 12.2 |
| `bga` | 1.0 | 0.1266 | 0.4589 | 3.63 | 1, 2 | no | 70.7 |
| `nsi_no_metal` | 1.0 | 0.0279 | 0.2531 | 9.07 | 4, 2 | no | 26.1 |
| `nsi_metal` | 1.0 | 0.0098 | 0.1700 | 17.32 | 4, 2 | no | 26.2 |

Every direction entry also caught the warning `Cone angle is more than 45 degrees.  This will
likely produce recon artifacts.`  That warning is raised by the model's parameter check at
construction.  At that point the row pitch is still the default of 1 ALU, because the reader has
not yet applied the real pitch.  The warning therefore says nothing about the geometry the job
used.  The `nsi_small` entry also caught
fourteen copies of the `torch.jit.script_method` deprecation warning, which is unrelated to this
feature.

## LEAP on the same data

LEAP was given the 128 central detector rows of each scan, in the sinogram order the LEAP
comparison established.  That order reverses the channel axis, so LEAP's offset carries the
opposite sign and the table below negates it.  The band is rows 893 to 1021 on the NSI scans and
rows 420 to 548 on `bga`.  The difference and the sum below are both against the module's rotation
estimate on the same scan.  Both are given because the two packages' tilt conventions have not been
checked against each other on real data.

| dataset | LEAP offset, channels | minus the module's third estimate, channels | LEAP tilt, deg | minus the module's rotation, deg | plus the module's rotation, deg | offset seconds | tilt seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `nsi_small` | -13.216 | +0.983 | 5.000 | +4.953 | +5.047 | 0.24 | 0.11 |
| `bga` | 0.545 | -0.001 | -0.402 | -0.419 | -0.385 | 2.37 | 0.05 |
| `nsi_no_metal` | -13.722 | +0.456 | 5.000 | +4.953 | +5.047 | 1.00 | 0.10 |
| `nsi_metal` | -13.977 | +0.211 | 0.006 | -0.143 | +0.154 | 0.99 | 0.11 |

LEAP's tilt of 5.000000 degrees on two scans is the bound of its five degree search.  These results
indicate a failed search rather than an estimate, so those two rows carry no tilt information.

## The offset sweep

The sweep reconstructs one slice at each of six candidate offsets.  The six are the vendor value,
the module's third estimate, and the estimate moved by half a channel and by a whole channel each
way.  Each slice
is scored by a sharpness measure, the mean squared finite difference along both axes divided by the
mean square of the slice.  The panel the measure called sharpest is named below.

| dataset | sharpest | vendor | estimate | estimate -0.5 ch | estimate +0.5 ch | estimate -1.0 ch | estimate +1.0 ch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `nsi_small` | estimate +0.5 ch | 0.185033 | 0.185048 | 0.184974 | 0.185408 | 0.185058 | 0.184818 |
| `bga` | vendor | 0.070934 | 0.070783 | 0.070312 | 0.064530 | 0.055713 | 0.054681 |
| `nsi_no_metal` | vendor | 0.023014 | 0.022993 | 0.022666 | 0.022975 | 0.022425 | 0.022700 |
| `nsi_metal` | vendor | 0.0085304 | 0.0085072 | 0.0078691 | 0.0081491 | 0.0070601 | 0.0072402 |

The measure cannot separate the vendor value from the estimate on any scan.  The two candidates sit
0.019 to 0.074 channels apart, and their sharpness values differ by 0.01 to 0.27 percent.  On
`nsi_small` the six values span only 0.32 percent, and the measure named a candidate half a channel
from the estimate as the sharpest.  These results indicate that this sharpness measure decides
nothing at the scale the estimators work at.

The four figures are the durable record of what the sweeps look like, because this repository
ignores PNG files.  In `nsi_small_sweep.png` the six panels show the artifact phantom as a bright
disk on a noisy background.  The disk carries a dark dome, a row of dark blocks, two bright wedges,
and a rim of small blocks.  In
`nsi_no_metal_sweep.png` the six panels show the same phantom at 1800 views, cleaner and marked by
faint concentric rings.  In `nsi_metal_sweep.png` the six panels show that phantom with its metal
insert, a small saturated blob with a horizontal streak through it.  On those three figures the six
panels look alike to the eye.  In `bga_sweep.png` the six panels show the ball grid array package
inside a bright circular holder.  There the two candidates a whole channel from the estimate split
the holder's rim into a doubled arc.

## The short scan

`z62` failed at the first estimate, with this message: `The conjugate-view method needs views over
a full rotation.  The angles cover 218.0 degrees, with a gap of 142.0 degrees between neighboring
views.`  The scan runs from -109.0 to +109.0 degrees.  The conjugate-view method pairs each view
with its opposite, and a scan of 218 degrees has no opposite for most of its views.  The refusal
came from `_require_conjugate_geometry` before any work was done, and the job moved on to the next
scan.  This is the first real short scan the feature has met.  `real_scan_followup.py`, beside this
page, takes it up.

## Limits of this evidence

The evidence is narrow in several ways.  Two scanners are represented, and only four scans were
measured.  Three of those four are the same NSI object, at two view counts and with and without a
metal insert, so they are not four independent scans.  No parameter here has ground truth.  The
vendor values that the third gate compares against are themselves estimates from the scanner's own
calibration, so a disagreement does not say which side is wrong.

The robustness cases are weaker than a real correction in two ways.  The beam-hardening case is a
quadratic term sized to change one band's largest value by ten percent.  It is a proxy for a
linearization correction rather than the correction any of these scans needs.  Every case changed a
band of 64 rows rather than the whole sinogram.  That band is what the estimators read, so the
change did reach them, but a correction fitted to a whole scan could move the estimate differently.

The resolution rule read the wrong memory.  It compares five sinogram-sized arrays against 70
percent of what `/proc/meminfo` reports as `MemAvailable`.  On this node that field reported about
960 GB, while the job held 252 GB.  Every scan therefore loaded at full resolution.  The largest
scan needs 101 GB by that rule, which is still under 70 percent of 252 GB.  These results indicate
that the job's own allocation would have led to the same choice on all five scans.

The sweep's sharpness measure cannot rank the candidates that matter.  It differs by 0.01 to 0.27
percent between the vendor value and the estimate on all four scans, and those two candidates are
under 0.08 channels apart.  On `nsi_small` it is flat within 0.32 percent across all six
candidates.  A person looking at the panels can see the whole-channel errors on `bga` and cannot
see them on the NSI scans, which matches what the numbers say.

## The batch file

The job was submitted from `/scratch/gautschi/buzzard/leap_cmp` with
`sbatch real_scan_validation.sbatch`.  That file is not in this repository, which ignores batch
files, so its lines are transcribed here.  It requests `-A bouman -p ai -q normal -N 1
--gpus-per-node=2 --cpus-per-task=28 -t 03:00:00`, names the job `real_scan_validation`, and writes
its log to `/scratch/gautschi/buzzard/leap_cmp/real_scan_validation_%j.log`.  A comment explains why
it asks for two GPUs.  The `ai` partition refuses `--mem` and gives 126 GB of host memory per GPU
requested, and a full-size real scan needs more than that.  The same comment records that the
partition requires 14 CPUs per GPU.  The file then sources `~/load_conda_cuda.sh`, sets `set -e`,
and changes to `/scratch/gautschi/buzzard/leap_cmp`.  It exports six environment variables:

- `TORCHINDUCTOR_CACHE_DIR` to `torch_cache_real_scan` under that directory;
- `MPLBACKEND` to `Agg`;
- `MBIRTORCH_NUM_DEVICES` to 1;
- `PYTHONPATH` to that directory;
- `REAL_SCAN_RESULTS` to `results_real_scan` under it and `REAL_SCAN_DATA` to `data` under it.

It creates those last two directories with `mkdir -p`.  It then runs a one-line check that stops
the job visibly if the interpreter is not the one it expects:
`venv/bin/python -c "import torch, mbirtorch; assert torch.cuda.is_available(); print(torch.__version__, mbirtorch.__file__)"`.
Finally it runs `venv/bin/python -u real_scan_validation.py`.
