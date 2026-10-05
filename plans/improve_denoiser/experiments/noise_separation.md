# The separation of the voxel pairs in option 2's noise estimate

Script: `noise_separation.py`, with the volumes of `phantom3d.py`.  The
script uses numpy and scipy only, not mbirtorch.  It ran on CPU on
2026-10-05.  The results below are the script's output.

## Why this test was run

Option 2 of `findings/step4_stopping_rule.md` estimates the noise level as
the median absolute difference of adjacent voxels, divided by
\( 0.6745 \sqrt{2} \).  On 2026-10-05, Greg proposed to compare voxels 10
apart instead.  This record measures how the separation of the two voxels
changes the estimate.

## Setup

For each separation \( d \) of 1, 2, 3, 5, and 10 voxels, the script takes
every pair of voxels that lie \( d \) apart along one axis.  The pairs of the
three axes are pooled.  The estimate is the median absolute difference of
the pairs, divided by \( 0.6745 \sqrt{2} \).  For white Gaussian noise in a
flat image, this estimate equals \( \sigma \) in the limit of many pairs.  A
pair is left out if either voxel is exactly 0, as in `robust_sigma.py`.  No
voxel of these volumes is exactly 0, so no pair was left out.

The volumes are those of `noise_estimate.md`, each 256 by 256 by 32 voxels:

- **flat**: water at 1000 everywhere, so the volume has no edges.
- **phantom**: air at 0 and a body of water at 1000, 180 by 230 voxels in
  each slice.  The body holds six cylindrical inserts, each about 27 voxels
  across and 20 to 26 slices long.
- **phantom in HU**: the phantom minus 1000, so air is -1000.

Each volume gets one of three kinds of noise, at 20, 60, or 182 HU per voxel:

- **white**: noise that is independent between voxels.
- **fdk**: noise with the spectrum of an FDK image within each slice.  Its
  correlation between neighbors is 0.57 within a slice and 0 between slices.
- **fdk_z**: fdk noise with a correlation of 0.78 between adjacent slices.

The **edge bias** is the amount by which the estimate on the phantom exceeds
the estimate on the flat volume with the same noise.

## Results

Each entry is the estimate divided by the true \( \sigma \).

| Volume | Noise | \( \sigma \) (HU) | d = 1 | d = 2 | d = 3 | d = 5 | d = 10 |
|---|---|---:|---:|---:|---:|---:|---:|
| flat | white | 20, 60, 182 | 1.000 | 0.999 | 1.000 | 1.000 | 0.999 |
| flat | fdk | 20, 60, 182 | 0.743 | 1.002 | 1.032 | 1.004 | 1.001 |
| flat | fdk_z | 20, 60, 182 | 0.584 | 0.927 | 1.014 | 1.002 | 1.001 |
| phantom | white | 20 | 1.031 | 1.041 | 1.053 | 1.075 | 1.134 |
| phantom | white | 60 | 1.020 | 1.030 | 1.040 | 1.060 | 1.113 |
| phantom | white | 182 | 1.007 | 1.015 | 1.023 | 1.039 | 1.081 |
| phantom | fdk | 20 | 0.771 | 1.045 | 1.086 | 1.079 | 1.136 |
| phantom | fdk | 60 | 0.763 | 1.033 | 1.073 | 1.064 | 1.115 |
| phantom | fdk | 182 | 0.753 | 1.018 | 1.056 | 1.043 | 1.083 |
| phantom | fdk_z | 20 | 0.603 | 0.966 | 1.066 | 1.077 | 1.136 |
| phantom | fdk_z | 60 | 0.597 | 0.955 | 1.054 | 1.063 | 1.115 |
| phantom | fdk_z | 182 | 0.590 | 0.942 | 1.037 | 1.041 | 1.082 |

On the flat volume, the three noise levels gave the same ratios to three
decimals.  The phantom in HU gave the same ratios as the phantom, so the
table leaves it out.  The difference of two voxels does not depend on an
offset.

## Observations

1. **Correlated noise makes the estimate from adjacent pairs low.**  On the
   flat volume, the estimate from pairs 1 voxel apart was 0.74 of the true
   level with fdk noise and 0.58 with fdk_z noise.  At \( d = 5 \) and
   \( d = 10 \), the estimate on the flat volume was 0.999 to 1.004 of the
   true level with all three kinds of noise.
2. **Edges make the estimate from distant pairs high, more so at low
   noise.**  On the phantom with white noise, the estimate was 1.04 to 1.08
   of the true level at \( d = 5 \), and 1.08 to 1.13 at \( d = 10 \).  A
   pair whose two voxels lie on opposite sides of an edge differs by the step
   of the edge as well as by the noise.  The larger the separation, the more
   pairs have their two voxels on opposite sides of an edge **[derived]**.
3. **At \( d = 2 \) and \( d = 3 \), correlated noise biased the estimate in
   some cases.**  On the flat volume, the estimate was 0.93 of the true
   level with fdk_z noise at \( d = 2 \), and 1.03 with fdk noise at
   \( d = 3 \).

On the flat volume, the correlation of the test noise did not bias the
estimate at \( d = 5 \).  If a real scan's noise is correlated over more
voxels than the test noise, pairs 5 apart could give a low estimate
**[inferred]**.  A real scan may also have more or fewer edges per voxel than
the phantom, so its edge bias may be larger or smaller **[inferred]**.
