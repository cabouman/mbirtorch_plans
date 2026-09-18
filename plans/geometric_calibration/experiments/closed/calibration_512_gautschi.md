# The estimators at 512 and 1024 channels, against LEAP: what `calibration_512_gautschi.py` measured

Date: 2026-09-04.  Slurm job 15915803 on gautschi, one NVIDIA H100 80GB, torch 2.13.0+cu130,
LEAP 1.26, mbirtorch 0.0.2 on the `geometric_calibration` branch.  The Increment 2 module was
copied over that checkout.  The job ran for 4 minutes 8 seconds.  Every number below was read
from the job's log, `/scratch/gautschi/buzzard/leap_cmp/calibration_512_15915803.log`, in the
same session.  The job also wrote `results_calibration/calibration_512.jsonl` in that directory.
This repository ignores such files, so this page is the durable copy.

Units.  An offset is given in channels, which are 1 mm at 512 channels and 0.5 mm at 1024.  A
rotation is given in degrees, and its edge displacement is the distance it moves the edge pixel
of the detector, in pixels.  The estimator under test is `estimate_det_channel_offset` and
`estimate_det_rotation` in `mbirtorch/preprocess/geometry_calibration.py`, called the module
below.  The LEAP comparison is the study recorded in `plans/features/leap_comparison/`.

## The answers

The rotation estimate is accurate at full size.  From an edge displacement of 1.3 pixels upward
its error is within 0.5 percent of the angle, at both 512 and 1024 channels.  Below half a pixel
it reads 10 to 24 percent low.  At 0.89 pixels it reads 4.5 percent high.  These results
indicate that the resampling bias is a function of the edge displacement, and the warning
threshold of the module was moved from half a pixel to one pixel on them.  LEAP's `estimate_tilt`
on the same sinograms reads 10 to 18 percent high below half a pixel, 3.4 percent high at 0.89
pixels, and within 1 percent above that.  These results indicate that both estimators lose
accuracy below about one pixel of edge displacement, in opposite directions.

The offset estimate is accurate at full size.  Its error is within 0.005 channels at 512
channels and within 0.001 channels at 1024, on clean and on noisy data.  LEAP's `find_centerCol`
errs by 0.017 to 0.024 channels at true offsets of 1.3 and -2.2, and by 0.003 channels or less
at 0.0 and 7.5.  One case failed.  A true offset of 7.5 channels lies outside the default search
window, which was four channels on each side of the model's value.  The search stopped at the
edge of that window, issued a warning, and returned 4 channels.  LEAP found the offset.  The
module's window now moves to center on the edge where the coarse minimum sits.

Both estimates finish in seconds.  On the H100 node the offset estimate took 0.3 to 0.7 seconds
at 512 channels and 1.9 to 4.0 seconds at 1024.  The rotation estimate took 2.1 to 2.8 seconds at
512 and 9.4 to 11.0 seconds at 1024.  Both run on the host in numpy, and the GPU was used only
to make the data.  LEAP's `find_centerCol` took 0.1 seconds at 512 and 0.9 at 1024, and its
`estimate_tilt` 0.01 and 0.05 seconds.

## Geometry and data

The geometry is the LEAP comparison's: circular cone beam, a flat detector, source to axis 1000
mm, source to detector 2000 mm, N views over a full rotation, N channels of 2 * 256 / N mm, and
N / 8 rows.  The full fan angle is 14.6 degrees.

The phantom is the Shepp-Logan phantom at the model's recon shape.  Noisy data add Gaussian
noise with standard deviation 2 percent of the sinogram maximum.

The tilted sinograms were generated at four times the detector resolution, rotated at that
resolution with the bilinear kernel, and binned by four.  This is the method of the laptop
experiment in `rotation_interpolation_bias.md`.  The generation took 5 seconds at 512 channels
and 70 seconds at 1024.

LEAP's `centerCol` is the column of the ray through the rotation axis.  In the module's terms
that is the detector center plus the offset in channels.  The sinogram order was converted with
the map the LEAP comparison established.  That map reverses the channel axis, so LEAP's offset
carries the opposite sign, and the tables below negate it.

## Channel offset

Errors are in channels.

| N | true offset | data | module error | LEAP error | module seconds | LEAP seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 0.0 | clean | +0.002 | 0.000 | 0.40 | 0.11 |
| 512 | 0.0 | noisy | 0.000 | +0.003 | 0.35 | 0.12 |
| 512 | 1.3 | clean | +0.004 | +0.021 | 0.67 | 0.11 |
| 512 | 1.3 | noisy | +0.004 | +0.022 | 0.68 | 0.12 |
| 512 | -2.2 | clean | -0.003 | -0.024 | 0.68 | 0.11 |
| 512 | -2.2 | noisy | -0.001 | -0.020 | 0.68 | 0.12 |
| 512 | 7.5 | clean | -3.500, at the edge of the window | -0.002 | 0.67 | 0.11 |
| 512 | 7.5 | noisy | -3.500, at the edge of the window | -0.003 | 0.68 | 0.11 |
| 1024 | 0.0 | clean | 0.000 | 0.000 | 1.87 | 0.86 |
| 1024 | 0.0 | noisy | -0.001 | +0.001 | 1.86 | 0.85 |
| 1024 | 1.3 | clean | +0.001 | +0.017 | 3.72 | 0.86 |
| 1024 | 1.3 | noisy | +0.001 | +0.018 | 3.75 | 0.85 |
| 1024 | -2.2 | clean | 0.000 | -0.019 | 3.76 | 0.88 |
| 1024 | -2.2 | noisy | 0.000 | -0.018 | 3.75 | 0.86 |
| 1024 | 7.5 | clean | -3.500, at the edge of the window | -0.001 | 3.71 | 0.90 |
| 1024 | 7.5 | noisy | -3.500, at the edge of the window | 0.000 | 3.99 | 0.89 |

The second pass changed the estimate in only one case, the clean case at N = 512 and a true
offset of -2.2 channels.  It moved the estimate from -2.2012 to -2.2031 channels.  Every other
case returned the first estimate.  These results indicate that the fan-angle pairing at a zero
offset is already accurate at a full fan angle of 14.6 degrees and offsets of two channels.  The
laptop experiment at a 20 degree fan and offsets of 3.5 channels showed a first-pass error of
0.035 channels, so the second pass is expected to matter more at wider fans and larger offsets.

The offset times within one size differ by the second pass.  At 512 channels the cases at a
true offset of zero took 0.35 to 0.40 seconds and every other case 0.67 to 0.68 seconds.  The
second pass runs whenever the first estimate differs from the model's offset by more than the
search tolerance, and it doubles the time.

LEAP's errors fall into two groups.  They are 0.003 channels or less at true offsets of 0.0 and
7.5, and 0.017 to 0.024 channels at 1.3 and -2.2.  The first pair of offsets lies on a
half-channel grid and the second does not, which is the pattern a search discretized at half a
channel would leave.  That hypothesis was not tested.

## Detector rotation

The offset is zero in these cases.  Errors are percentages of the true rotation.

| N | true, degrees | edge displacement, pixels | module, degrees | error | LEAP, degrees | error | module seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 0.0 | 0.00 | -0.001 | | 0.000 | | 2.8 |
| 512 | 0.05 | 0.22 | 0.038 | -24 percent | 0.059 | +18 percent | 2.1 |
| 512 | 0.1 | 0.45 | 0.090 | -10 percent | 0.113 | +13 percent | 2.1 |
| 512 | 0.3 | 1.34 | 0.299 | -0.4 percent | 0.298 | -0.8 percent | 2.1 |
| 512 | 1.0 | 4.47 | 1.005 | +0.5 percent | 1.002 | +0.2 percent | 2.2 |
| 1024 | 0.0 | 0.00 | 0.000 | | 0.000 | | 9.4 |
| 1024 | 0.05 | 0.45 | 0.044 | -12 percent | 0.055 | +10 percent | 9.4 |
| 1024 | 0.1 | 0.89 | 0.105 | +4.5 percent | 0.103 | +3.4 percent | 9.5 |
| 1024 | 0.3 | 2.68 | 0.301 | +0.4 percent | 0.300 | +0.1 percent | 9.8 |
| 1024 | 1.0 | 8.94 | 1.003 | +0.3 percent | 1.002 | +0.2 percent | 11.0 |

Two cases share an edge displacement of 0.45 pixels: 0.1 degrees on 512 channels and 0.05
degrees on 1024.  They err by -10 and -12 percent.  These results indicate that the bias is a
function of the edge displacement and not of the detector size.  The laptop measurement at 64
channels predicted that, and this one matched pair is the evidence for it here.  The case at
0.89 pixels errs by 4.5 percent, and every case from 1.34 pixels upward by under 0.5 percent.
These results indicate that a warning threshold of one pixel is placed about right, and the
module's threshold was moved there from half a pixel.

The rotation and the offset are coupled in both estimates.  On the rotated sinograms LEAP's
`find_centerCol` returned offsets that grow with the rotation, from 0.03 channels at 0.05
degrees to 0.42 channels at 1 degree on 512 channels.  LEAP's search uses one detector row, and a
rotation moves that row's content along the channels.  The module's estimate shows the same
coupling, which is why the offset is estimated first, then the rotation, then the offset again.

## The batch file

The job was submitted from `/scratch/gautschi/buzzard/leap_cmp` with
`sbatch calibration_512_gautschi.sbatch`.  That file is not in this repository, which ignores
batch files.  It does four things.  It requests
`-A bouman -p ai -q normal -N 1 --gpus-per-node=1 -n 14 -t 02:00:00`.  It sources
`~/load_conda_cuda.sh`.  It sets five environment variables:

- `TORCHINDUCTOR_CACHE_DIR` to `torch_cache_calib` under the same directory;
- `MPLBACKEND` to `Agg`;
- `MBIRTORCH_NUM_DEVICES` to 1;
- `PYTHONPATH` to the same directory;
- `CALIBRATION_RESULTS` to `results_calibration` under it.

It then runs `venv/bin/python -u calibration_512_gautschi.py`.
