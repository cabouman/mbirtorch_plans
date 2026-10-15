# The median in place of the mean in the noise estimate

Script: `noise_median.py`, with the volumes of `noise_estimate.py` and
`phantom3d.py`.  Measured on mbirtorch `prerelease` at 694a87a, on CPU,
2026-10-15.  The denoiser code is the same at 296f738, the head of
`prerelease` when this record was written.  The table below is the script's
output.

## Question

Charlie answered the proposal of 2026-10-12 with a suggestion for the noise
estimate.  Edges raise the estimate, and the median in place of the mean
could fix this.  This record tests the suggestion.  It also compares the
suggestion with the estimate of `greg_dev`, which uses pairs of voxels 5
apart.

The noise estimate of `prerelease` works as `noise_estimate.md` describes.
For each voxel of a support, it takes the standard deviation of four values:
the voxel and its three backward neighbors.  It returns the mean of these
standard deviations over the support.  The support is the set of voxels
above 5% of the mean absolute value, plus a noise floor.

## Setup

The volumes, the kinds of noise, and the noise levels are those of
`noise_estimate.md`.  The volumes have 256 by 256 by 32 voxels:

- **flat**: water at 1000 everywhere, so the volume has no edges.
- **phantom**: air at 0, water at 1000, and six inserts.
- **phantom in HU**: the phantom minus 1000, so air is -1000 and water is 0.

The script compares four estimates with the true noise level:

- **mean**: the estimate of `prerelease`.
- **median**: the same code, with the median in place of the mean in
  `_get_estimate_of_recon_std`.  Both passes of the estimate use that
  function.
- **median scaled**: the median divided by 0.7691.
- **pairs 5 apart**: the estimate of `greg_dev` at 235c82d.  It is the
  median absolute difference of voxels 5 apart along each axis, divided by
  \( 0.6745 \sqrt{2} \).  The script holds a copy of the function.

The factor 0.7691 makes the median correct for white noise in a flat region.
The standard deviation of four independent normal values, dividing by 4, is
\( \sigma \sqrt{X/4} \).  Here \( X \) has a chi-square distribution with 3
degrees of freedom.  The median of \( X \) is 2.366, so the median of the
standard deviation is \( 0.7691 \sigma \) **[derived]**.

The automatic \( \sigma_x \) of the denoiser is 0.2 times the statistic of
`_get_estimate_of_recon_std`, computed on every 12th row.  So a median in that
function changes \( \sigma_x \) as well as the noise estimate.  The last two
columns give the automatic \( r = \sigma_x / \sigma_y \) with the true noise
level given, for the mean and for the median.

## Results

The four estimate columns are divided by the true noise level.

| Volume | Noise | \( \sigma \) (HU) | mean | median | median scaled | pairs 5 apart | \( r \), mean | \( r \), median |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| flat | white | 20, 60, 182 | 0.798 | 0.769 | 1.000 | 1.000 | 0.159 | 0.154 |
| flat | fdk | 20, 60, 182 | 0.687 | 0.649 | 0.844 | 1.004 | 0.151 | 0.143 |
| phantom | white | 20 | 1.226 | 0.820 | 1.066 | 1.075 | 0.666 | 0.182 |
| phantom | white | 60 | 0.920 | 0.821 | 1.067 | 1.060 | 0.307 | 0.180 |
| phantom | white | 182 | 0.836 | 0.804 | 1.046 | 1.039 | 0.197 | 0.173 |
| phantom | fdk | 20 | 1.124 | 0.697 | 0.907 | 1.079 | 0.658 | 0.171 |
| phantom | fdk | 60 | 0.809 | 0.693 | 0.901 | 1.064 | 0.298 | 0.169 |
| phantom | fdk | 182 | 0.721 | 0.676 | 0.879 | 1.043 | 0.188 | 0.162 |
| phantom in HU | white | 20 | 1.735 | 1.112 | 1.446 | 1.075 | 0.830 | 0.469 |
| phantom in HU | white | 60 | 1.025 | 0.971 | 1.262 | 1.060 | 0.344 | 0.238 |
| phantom in HU | white | 182 | 0.926 | 0.902 | 1.172 | 1.039 | 0.222 | 0.203 |
| phantom in HU | fdk | 20 | 1.635 | 0.954 | 1.240 | 1.079 | 0.820 | 0.468 |
| phantom in HU | fdk | 60 | 0.869 | 0.788 | 1.024 | 1.064 | 0.338 | 0.229 |
| phantom in HU | fdk | 182 | 0.770 | 0.727 | 0.945 | 1.043 | 0.215 | 0.192 |

The mean column and the first \( r \) column match `noise_estimate.md` to
the last digit.  That record measured `greg_dev` before its new estimate.  So
the estimate of `prerelease` today gives the same numbers as the estimate that
`greg_dev` had before its change.

## What the results show

The median removes most of the effect of edges.  On the phantom with air at
0 and white noise, the scaled median read 1.05 to 1.07 of the true level.
The mean read 0.84 to 1.23 on the same volumes.

The median needs a scale factor.  Without the factor, it read 0.769 of the
true level on white noise in the flat volume.  The mean has the same kind of
bias, with a factor of 0.798.

The median keeps two other biases of the mean.  On fdk noise, the scaled
median read 0.84 to 0.91 of the true level.  Adjacent voxels of fdk noise are
correlated, so they differ less than independent voxels would.  In HU, the
scaled median read 0.95 to 1.45 of the true level.  The offset moves the
threshold of the support, so the support holds different voxels.

The pairs 5 apart read 1.00 to 1.08 of the true level in every case.  Edges
cause the excess above 1.00.

These results indicate that Charlie's median and the estimate of `greg_dev`
share their main idea.  Both take a median, so that edges have little
effect.  The estimate of `greg_dev` adds three things to the median:

1. a scale factor that makes it correct for white noise;
2. pairs 5 apart, so that correlation between neighbors lowers it less;
3. no support threshold, so that an offset does not change it.

A median in `_get_estimate_of_recon_std` would also change the automatic
\( \sigma_x \), because that function computes the statistic of
\( \sigma_x \).  On the phantom with 20 HU of white noise, the automatic
\( r \) fell from 0.67 to 0.18.  So the denoising would become much stronger
on images with strong edges and low noise.  The estimate of `greg_dev`
changes only the noise level, so the automatic \( \sigma_x \) does not
change.

## Limits

- The volumes come from one phantom, with two simple noise models.
- The fdk noise is independent from slice to slice.  Noise that is
  correlated over more than 5 voxels would lower the estimate from pairs 5
  apart as well **[inferred]**.
- The volumes have about 2 million voxels.  Above about 17 million voxels,
  the estimate of `prerelease` reads a subsample whose neighbors are 2 or
  more voxels apart, as `noise_estimate.md` notes.
