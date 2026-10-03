# The automatic noise estimate of `denoise`

Script: `noise_estimate.py`, with the volumes of `phantom3d.py`.  Measured on
mbirtorch at b0b1879 (`greg_dev` at 5f62506 plus PR #15), on CPU,
2026-10-03.  The results below are the script's output.

## Setup

The script compares `estimate_image_noise_std` with the true noise level on
three volumes of 256 by 256 by 32 voxels:

- **flat**: water at 1000 everywhere, so the volume has no edges.
- **phantom**: the volume of `phantom3d.py`, with air at 0, water at 1000, and
  six inserts.
- **phantom in HU**: the phantom minus 1000, so air is -1000 and water is 0.

Each volume gets white noise or fdk noise, at 20, 60, or 182 HU per voxel.
The fdk noise has a power that rises with frequency within each slice.  Its
correlation between neighbors is 0.57 within a slice and 0 between slices.

The script also reports the automatic \( r = \sigma_x / \sigma_y \) in two
ways: with `sigma_noise` set to the true level, and with `sigma_noise`
estimated, as in a call of `denoise(image)`.  The last column is a robust
candidate estimate: the median absolute first difference, pooled over the
three axes and all voxels, divided by \( 0.6745 \sqrt{2} \).

## How the current estimate works

The estimate takes the standard deviation of four values: a voxel and its
three backward neighbors.  It averages this standard deviation over a
support: the voxels above 5% of the mean absolute value plus a noise floor.
A first pass sets the floor to 0, and a second pass sets it to the first
pass's estimate.

For white noise in a flat region, the mean of this standard deviation is
\( 0.798 \sigma \).  The factor is \( \sqrt{3/4} \) times
\( c_4(4) = 0.9213 \), because the standard deviation divides by 4 and not
by 3 **[derived; a simulation gave 0.7976]**.

The automatic \( \sigma_x \) of the denoiser is 0.2 times the same statistic,
computed on a subsample of about 20 rows (`auto_set_sigma_x`).  So the
automatic \( \sigma_x \) follows the statistic's response to edges.

## Results

| Volume | Noise | \( \sigma \) (HU) | Estimate / \( \sigma \) | \( r \), \( \sigma \) given | \( r \), \( \sigma \) estimated | Candidate / \( \sigma \) |
|---|---|---:|---:|---:|---:|---:|
| flat | white | 20, 60, 182 | 0.798 | 0.159 | 0.200 | 1.000 |
| flat | fdk | 20, 60, 182 | 0.687 | 0.151 | 0.219 | 0.743 |
| phantom | white | 20 | 1.226 | 0.666 | 0.543 | 1.031 |
| phantom | white | 60 | 0.920 | 0.307 | 0.334 | 1.020 |
| phantom | white | 182 | 0.836 | 0.197 | 0.235 | 1.007 |
| phantom | fdk | 20 | 1.124 | 0.658 | 0.585 | 0.771 |
| phantom | fdk | 60 | 0.809 | 0.298 | 0.367 | 0.763 |
| phantom | fdk | 182 | 0.721 | 0.188 | 0.259 | 0.753 |
| phantom in HU | white | 20 | 1.735 | 0.830 | 0.504 | 1.031 |
| phantom in HU | white | 60 | 1.025 | 0.344 | 0.336 | 1.020 |
| phantom in HU | white | 182 | 0.926 | 0.222 | 0.238 | 1.007 |
| phantom in HU | fdk | 20 | 1.635 | 0.820 | 0.530 | 0.771 |
| phantom in HU | fdk | 60 | 0.869 | 0.338 | 0.382 | 0.763 |
| phantom in HU | fdk | 182 | 0.770 | 0.215 | 0.270 | 0.753 |

The current estimate has three biases:

1. **It is low on flat images.**  It returned 0.798 of the true level for white
   noise and 0.687 for fdk noise.
2. **Edges raise it, more so at low noise.**  On the phantom with white noise,
   it returned 1.226 of the true level at 20 HU and 0.836 at 182 HU.
3. **The offset changes it.**  The same phantom in HU gave 1.735 at 20 HU,
   against 1.226 with air at 0.  The offset moves the support's threshold,
   which depends on the mean absolute value.

The automatic \( r \) inherits these biases.  With \( \sigma \) given, it
ranged from 0.15 on the flat volumes to 0.83 on the phantom in HU at 20 HU.
So the automatic strength is weak on images with strong edges and low noise.

The candidate estimate was within 3.1% of the true level for white noise on
every volume.  Edges changed it by at most 3.1%, and the offset did not change
it.  For fdk noise
it returned 0.74 to 0.77 of the true level.  Any estimate from neighbor
differences reads low on noise that is positively correlated between
neighbors **[derived]**.  So no estimate of this kind recovers the per-voxel
level of fdk noise without a model of the correlation.
