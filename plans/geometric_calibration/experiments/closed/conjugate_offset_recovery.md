# Conjugate-view offset recovery: what `conjugate_offset_recovery.py` measured

Date: 2026-09-04.  The script is `conjugate_offset_recovery.py` in this directory.  It ran in the
`mbirtorch` conda environment on a Mac laptop, with torch 2.13.0 on the CPU and torch.compile
off.  The mbirtorch checkout was the `geometric_calibration` branch during Increment 2, after the
fan-angle pairing was corrected and the estimator gained its second pass.  Every number below
was read from that run's output in the same session.  The estimator is
`estimate_det_channel_offset` in `mbirtorch/preprocess/geometry_calibration.py`, called the
module below.  Distances are in arbitrary length units, written ALU, and the detector pitch is
1 ALU, so offsets in ALU are offsets in channels.

## The answer

The module recovers a known channel offset to within 0.023 channels over the range of -3.5 to
3.5 channels.  The data are synthetic cone beam at a full fan angle of 20 degrees.  Two phantoms
were used, a nearly centered one and an off-axis rod.  Noise at 2 percent of the sinogram
maximum changes the estimate by less than 0.005 channels.  The view stride and the band height
change the error by at most 0.008 channels.  A deliberately wrong conjugate-ray sign raises the
score minimum by a factor of 5.9 to 186.  It moves the estimate by 0.2 to 1.9 channels.

## What was measured

The geometry has these values: 128 views, 16 rows, 64 channels, a source-to-detector distance of
181.5 ALU, and a source-to-iso distance of 90.7 ALU.  The two distances give a full fan angle of
20 degrees.  One phantom is the Shepp-Logan phantom, which is nearly centered.  The other is a
rod of radius 3 voxels, placed 8 voxels from the axis.  That placement puts its signal on one
side of the detector.

For each true offset a generating model with that offset forward projected the phantom.  The
estimating model had an offset of zero, so the search window of four channels on each side of
zero held every true offset.  The first pass pairs the views at the model's offset of zero, and
the second pass re-pairs at the first estimate.  The noisy cases add Gaussian noise with
standard deviation 2 percent of the sinogram maximum, over five seeds, and report the second
pass.

| phantom | true offset | error, first pass | error, second pass | error, noisy mean | noisy standard deviation |
| --- | --- | --- | --- | --- | --- |
| Shepp-Logan | -3.5 | +0.014 | +0.004 | +0.003 | 0.003 |
| Shepp-Logan | -2.2 | +0.020 | +0.021 | +0.023 | 0.002 |
| Shepp-Logan | -1.0 | -0.007 | 0.000 | +0.004 | 0.003 |
| Shepp-Logan | 0.0 | +0.001 | +0.001 | 0.000 | 0.001 |
| Shepp-Logan | +1.3 | +0.004 | +0.012 | +0.014 | 0.004 |
| Shepp-Logan | +2.6 | +0.002 | -0.011 | -0.011 | 0.001 |
| Shepp-Logan | +3.5 | -0.014 | -0.001 | -0.003 | 0.002 |
| off-axis rod | -3.5 | +0.035 | +0.004 | +0.001 | 0.000 |
| off-axis rod | -2.2 | +0.023 | -0.003 | -0.004 | 0.001 |
| off-axis rod | -1.0 | +0.006 | -0.002 | -0.003 | 0.003 |
| off-axis rod | 0.0 | -0.001 | -0.001 | -0.003 | 0.001 |
| off-axis rod | +1.3 | -0.007 | +0.009 | +0.008 | 0.001 |
| off-axis rod | +2.6 | -0.019 | -0.006 | -0.005 | 0.002 |
| off-axis rod | +3.5 | -0.032 | +0.002 | +0.002 | 0.001 |

The first pass errs in proportion to the offset on the rod.  Its error runs from +0.035
channels at a true offset of -3.5 to -0.032 at +3.5, which is a slope of about one percent.  The
second pass removes that trend.  Its errors on the rod are within 0.009 channels with no trend.
On the Shepp-Logan phantom the first pass errs by up to 0.020 channels and the second by up to
0.021, and at offsets of about one channel the second pass moves the estimate by up to 0.01
channels in either direction.  These results indicate that the second pass is needed at offsets
of a few channels on an off-axis object, and that at small offsets it changes the estimate by
about the search tolerance.  The largest error that remains, 0.023 channels as a noisy mean on
the Shepp-Logan phantom at -2.2, reproduces to 0.002 across the seeds, so it is systematic and
not a search residual.

## The scale error found in the earlier run

An earlier run of this script came before the pairing was corrected.  It showed an error on the
rod that grew in proportion to the offset, at about one percent of it.  The trace found two
causes.  First, the channel offset entered each channel's fan angle with the wrong sign, so the
partner view was chosen wrongly.  Second, the pairing was computed at the model's offset rather
than at the true offset.  The sign was corrected, and the second pass was added.  The first-pass
column above shows the second cause on its own, with the sign already corrected.

## View stride and band height

The true offset here is 1.3 channels.  A stride thins the reference views.  The partner views
are drawn from every view at any stride.

| phantom | stride 1 | stride 2 | stride 4 | 5 rows | 9 rows | 16 rows |
| --- | --- | --- | --- | --- | --- | --- |
| Shepp-Logan | +0.012 | +0.014 | +0.014 | +0.012 | +0.006 | +0.004 |
| off-axis rod | +0.009 | +0.009 | +0.004 | +0.009 | +0.009 | +0.009 |

Neither the stride nor the band height changes the error by more than 0.008 channels.  The
default band at this geometry is 5 rows, which is the cone-beam limit.  The rod is identical in
every row.  This measurement therefore does not test the limit.

## The conjugate-ray sign

The module pairs the ray at view angle `beta` and fan angle `gamma` with the view at
`beta + pi - 2 gamma`.  The script repeats the estimate with `beta + pi + 2 gamma`, in both
passes.

| phantom | score minimum, flipped over module | estimate, flipped | estimate, module |
| --- | --- | --- | --- |
| Shepp-Logan | 5.90 | +1.086 | +1.312 |
| off-axis rod | 185.63 | -0.603 | +1.309 |

The flipped sign raises the score minimum by a factor of 5.9 on the Shepp-Logan phantom and by a
factor of 186 on the rod.  It moves the estimate by 0.23 channels on the Shepp-Logan phantom and
by 1.9 channels on the rod.  These results indicate that the score minimum, not the offset error,
is the sensitive check of the fan-angle pairing.  On the Shepp-Logan phantom the flipped estimate
is still within the plan's 0.5 channel gate for cone beam.  These results indicate that the gate
alone would not detect a wrong pairing.
