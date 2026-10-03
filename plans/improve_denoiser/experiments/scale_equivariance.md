# Scale-equivariance test of `denoise`

Script: `scale_equivariance.py`.  Measured on mbirtorch `greg_dev` at 44da2e9,
whose `denoising.py` is identical to 63207f7.  Run on CPU (4 threads), 2026-10-03.

## Setup

The phantom is a 256 x 256 x 4 volume in HU: a water disk at 1000 HU with
three inserts, plus white Gaussian noise with standard deviation 182 HU.  For
each scale c, the script denoises c y with `sigma_noise` = 182 c.  In the
fixed-prior case it also sets `sigma_x` = 18.2 c, so r = 0.1.  In the
automatic case `sigma_x` comes from the automatic regularization.  Every run
uses the same partition, 20 iterations, and `stop_threshold_change_pct=0`.

The script runs two updates.  The released update is the code as it stands.
The patched update sets the second derivative of the data term to
`fm_constant` instead of 1.

## Results

Each distance is an rms distance in HU, after dividing the output by c.

| Update | Prior | c | mean alpha | distance moved from y | distance to the c = 1 result |
|---|---|---:|---:|---:|---:|
| released | fixed | 1e-3 | 0.974 | 175.911 | 175.058 |
| released | fixed | 1 | 1.500 | 0.956 | 0 |
| released | fixed | 1e3 | 1.500 | 0.000 | 0.956 |
| released | automatic | 1e-3 | 0.926 | 160.084 | 159.698 |
| released | automatic | 1 | 1.500 | 0.423 | 0 |
| released | automatic | 1e3 | 1.500 | 0.000 | 0.423 |
| patched | fixed | all three | 1.000 | 175.914 | 0.000 |
| patched | automatic | all three | 1.000 | 160.089 | 0.000 |

The released update fails the test.  At c = 1 it moves the image by less than
1 HU in 20 iterations, and at c = 1e3 it does not move the image at all.  The
clamp holds alpha at 1.5 in both cases.

The patched update passes the test at every c, with both priors.  Alpha is
1.000.  The automatic `sigma_x` divided by c is 34.812 at every c, so the
automatic regularization is itself scale equivariant (r = 0.19 here).
