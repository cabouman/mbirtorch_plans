# The two rotation implementations share one convention: what `rotation_kernel_conventions.py` measured

Date: 2026-09-05.  The script is `rotation_kernel_conventions.py` in this directory.  It ran on the
Mac in the miniforge `mbirtorch` environment, torch 2.13.0 on the CPU, against the
`geometric_calibration` branch working tree of mbirtorch, which is commit `4781600` plus the staged
Increment 6 edits.  Every number below was read from that run's output in this session.

## The question

Every synthetic detector-rotation example in this feature was made by rotating a sinogram with
`_rotation_kernel`, the bilinear kernel in `mbirtorch/preprocess/utilities.py` that
`correct_det_rotation`, `apply_calibration`, and `reduce_sinogram` use.  The rotation estimator
scores each candidate angle with a different implementation, OpenCV's cubic `warpAffine` inside
`_rotated_band` (`mbirtorch/preprocess/geometry_calibration.py`).  A recovery test built from such
data checks the estimator against the correction's convention.  This script measures how exactly
the two implementations agree, and how large a sign error or a center error would look, so that
the recovery tests' power against such errors is a measurement rather than a claim.

## The answer

The two implementations compute the same map.  Rotating one smooth image with `_rotation_kernel`
and with OpenCV's bilinear `warpAffine` about the same center gives interior values that agree to
a relative maximum difference of 6.2e-07 or less at every angle tested.  A sign error between the
two would show at 1.5e-02 to 9.8e-02, and a center error of half a channel at 1.3e-04 to 8.9e-04.
These results indicate that the synthetic rotation recoveries do test the sign and the scale of
the convention shared by the correction and the estimator, and would have caught a disagreement
in either.

The estimator's scoring path agrees with the correction path to the size of its kernel
difference.  `_rotated_band`, which resamples with the cubic kernel, differs from
`_rotation_kernel` by 2.3e-03 to 3.7e-03 on the same smooth image, and by 1.7e-02 to 9.6e-02 when
its angle is negated.  The cubic-versus-bilinear difference is therefore an order of magnitude
below the sign control at every angle.  It is also larger than the half-channel center control,
so this comparison alone cannot rule out a sub-channel center disagreement between the two
production paths.  That the two paths rotate about the same center rests on their source.
`_rotation_kernel` defaults to the array center, `((num_rows - 1) / 2, (num_cols - 1) / 2)`
(`utilities.py:234-236`).  `_rotated_band` names the same physical point expressed in its band's
indices, `((num_channels - 1) / 2, (center_row - band_lo) / bin_factor)` with `center_row` the
full detector's `(num_rows - 1) / 2` (`geometry_calibration.py:1296-1310`).

## What was measured

The image is a sum of six Gaussian blobs on a constant background, 96 rows by 128 channels, seed
0.  A smooth image keeps the cubic and bilinear kernels close, so a convention difference would
stand out.  Each comparison reports the maximum absolute difference over the interior, eight
pixels in from every edge, divided by the maximum absolute interior value.  The reference in every
column is `_rotation_kernel` at the stated angle.

| angle, degrees | OpenCV bilinear, same center and angle | OpenCV bilinear, angle negated | OpenCV bilinear, center moved 0.5 channels | `_rotated_band`, same angle | `_rotated_band`, angle negated |
| --- | --- | --- | --- | --- | --- |
| 0.30 | 3.9e-07 | 1.5e-02 | 1.3e-04 | 2.3e-03 | 1.7e-02 |
| -0.50 | 6.2e-07 | 2.5e-02 | 2.2e-04 | 3.4e-03 | 2.7e-02 |
| 2.00 | 5.9e-07 | 9.8e-02 | 8.9e-04 | 3.7e-03 | 9.6e-02 |

## Limits of this evidence

This measurement compares implementations with each other, not with a physical detector.  An
error in the shared convention itself, one that both implementations carry, is invisible here and
to every synthetic recovery test.  The evidence that the shared convention matches a real tilted
detector is `real_scan_rotation_recon.md`: applying the vendor's recorded 0.167 degrees with
`_rotation_kernel` to a real scan sharpened the slices far from the central plane.  One image,
one size, and three angles were tested, all on the CPU.  The interior border of eight pixels
excludes the rim, where the two implementations zero out-of-bounds samples by slightly different
rules.
