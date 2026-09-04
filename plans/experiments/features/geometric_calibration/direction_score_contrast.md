# Direction-score contrast: what `direction_score_contrast.py` measured

Date: 2026-09-04.  The script is `direction_score_contrast.py` in this directory.  It ran on the
Mac in the `mbirtorch` conda environment, with torch 2.13.0 on the CPU and torch.compile off.  It
ran against the `geometric_calibration` branch of mbirtorch at the state of Increment 1.  Every
number below was read from that run's output in the same session.

## The answers

Two answers came out of this measurement.  The direction check keeps the whole axial extent of
the reconstruction, because a thin slab gives almost no contrast between the two rotation
directions.  The check scores the residual over the central half of the detector rows, because
that raised the contrast in every case measured and narrower fractions added little.

## Definitions

The check scores the geometry as given and the geometry with every view angle negated, and it
picks the lower score.  The score is the mean squared difference between the high-pass filtered
sinogram and the high-pass filtered forward projection of a direct reconstruction.  That
difference is divided by the mean square of the filtered sinogram.  The high-pass filter is
`sino_high_pass_filtering` at its default widths of 3 detector rows and 15 channels.  Those widths
are in pixels, so on the small detectors here the filter removes more of the signal than it would
on a full-size detector, and the scores below are not portable to another detector size.

The contrast is the ratio of the wrong-direction score to the correct-direction score.  A ratio
near 1 means the check cannot distinguish the two directions.

Two design questions needed a measurement.  The reduced problem normally reconstructs a thin
slab.  Does a thin slab leave enough contrast between the two directions?  And does restricting
the residual to the central detector rows raise the contrast?

## Effect of slab thickness on contrast

Part A used one cone-beam geometry.  It has 128 views, 32 detector rows, and 64 channels.  The
source-to-detector distance is 256 ALU and the source-to-iso distance is 128 ALU.  Those distances
give a half fan angle of 7.1 degrees.  The recon is 64 by 64 by 32, so 32 slices is the whole
volume at bin 1, and 16 slices is the whole volume at bin 2.  The ratio is given for three scored
row fractions.

| view stride | bin | slab slices | rows kept | all rows | central half | central third |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 1 | 4 | 10 to 21 | 1.03 | 1.06 | 1.08 |
| 2 | 1 | 8 | 8 to 23 | 1.06 | 1.26 | 1.36 |
| 2 | 1 | 16 | 3 to 28 | 1.14 | 3.46 | 6.28 |
| 2 | 1 | 32 | 0 to 32 | 2.86 | 6.04 | 7.28 |
| 2 | 2 | 4 | 4 to 26 | 1.02 | 1.04 | 1.06 |
| 2 | 2 | 8 | 0 to 30 | 1.05 | 1.23 | 1.74 |
| 2 | 2 | 16 | 0 to 32 | 2.17 | 3.20 | 3.49 |
| 4 | 2 | 8 | 0 to 30 | 1.06 | 1.23 | 1.79 |
| 4 | 2 | 12 | 0 to 32 | 1.35 | 3.45 | 3.93 |
| 4 | 2 | 16 | 0 to 32 | 2.27 | 3.56 | 4.00 |
| 1 | 1 | 8 | 8 to 23 | 1.06 | 1.26 | 1.36 |

A thin slab gives almost no contrast.  With 4 or 8 slices the correct-direction score is itself
large.  Over all rows it ranges from 0.41 to 1.31 in the normalized units.  The kept rows measure
material outside the slab, and the slab cannot explain that material.  That unexplained part is
nearly the same for both directions, so the ratio sits near 1.

The contrast appears when the slab covers most of the material the kept rows intersect.  At the
shipped settings of stride 4 and bin 2, scored over the central half, a slab of 8 of 16 slices
gives 1.23, a slab of 12 gives 3.45, and the whole volume gives 3.56.  At bin 1, 16 of 32 slices
gives 3.46 and the whole volume gives 6.04.  These results indicate that a slab of half the volume
recovers most of the contrast, and that the whole volume recovers a little more.

The whole volume was chosen for the check.  The difference in cost between half the volume and
the whole volume is small for a check that runs once, and the whole volume needs no rule for how
thick a slab must be.  A slab thinner than half the volume is not usable for this score.

The view stride does not affect the contrast.  The ratio changes by less than 0.02 between
strides of 1 and 2 at 8 slices.

## Effect of the scored row fraction, with the whole axial extent kept

Part B used three geometries.  In each one the source-to-iso distance is half the
source-to-detector distance.

- small: 128 views, 32 rows, 64 channels, half fan angle 7.1 degrees, recon 64 by 64 by 32;
- medium: 256 views, 64 rows, 128 channels, half fan angle 7.1 degrees, recon 128 by 128 by 64;
- medium narrow fan: the medium geometry with both distances doubled, which gives a half fan
  angle of 3.6 degrees.

In the cases labeled noisy, Gaussian noise was added to the sinogram.  Its standard deviation is
2 percent of the sinogram maximum.  Each cell is the ratio at the given scored row fraction.

| geometry | stride | bin | data | 1.00 | 0.67 | 0.50 | 0.33 | 0.20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small | 4 | 2 | clean | 3.55 | 4.72 | 5.38 | 5.68 | 5.65 |
| small | 4 | 2 | noisy | 3.32 | 4.37 | 4.97 | 5.27 | 5.27 |
| small | 2 | 2 | clean | 3.27 | 4.20 | 4.60 | 4.71 | 4.68 |
| small | 2 | 2 | noisy | 3.05 | 3.88 | 4.26 | 4.38 | 4.38 |
| small | 4 | 1 | clean | 5.30 | 8.02 | 9.02 | 8.42 | 7.67 |
| small | 4 | 1 | noisy | 4.19 | 6.02 | 6.74 | 6.50 | 6.22 |
| medium | 4 | 2 | clean | 6.22 | 9.84 | 11.49 | 12.90 | 13.65 |
| medium | 4 | 2 | noisy | 5.60 | 8.59 | 9.93 | 11.09 | 11.73 |
| medium | 2 | 2 | clean | 5.97 | 9.10 | 10.39 | 11.27 | 11.61 |
| medium | 2 | 2 | noisy | 5.30 | 7.86 | 8.89 | 9.62 | 9.92 |
| medium | 4 | 1 | clean | 11.77 | 22.50 | 26.63 | 28.56 | 28.09 |
| medium | 4 | 1 | noisy | 7.33 | 11.55 | 12.98 | 13.74 | 13.94 |
| medium narrow fan | 4 | 2 | clean | 4.36 | 5.70 | 6.32 | 6.78 | 6.94 |
| medium narrow fan | 4 | 2 | noisy | 3.80 | 4.93 | 5.47 | 5.86 | 6.02 |
| medium narrow fan | 2 | 2 | clean | 4.01 | 5.04 | 5.41 | 5.62 | 5.67 |
| medium narrow fan | 2 | 2 | noisy | 3.45 | 4.33 | 4.66 | 4.86 | 4.94 |
| medium narrow fan | 4 | 1 | clean | 7.25 | 12.05 | 13.52 | 12.84 | 12.01 |
| medium narrow fan | 4 | 1 | noisy | 4.36 | 6.28 | 6.99 | 7.05 | 6.92 |

Restricting the residual to the central half of the detector rows raised the ratio by a factor
of 1.35 to 2.3.  This holds in all eighteen cases in the table.  Fractions below one half changed
the ratio by less than 20 percent, up or down.  The correct-direction score is what falls as the
scored fraction narrows.  In the clean cases it ranges from 0.0085 to 0.0171 with every row
scored, and from 0.0069 to 0.0102 with the central half scored.  The wrong-direction score rises
slightly.  These results indicate that the outer rows add error from the direct reconstruction's
own cone-angle approximation.  That error is unrelated to the rotation direction under test.

The table shows three other effects.  Binning by two lowers the contrast.  Against the unbinned
cases at the same stride the ratio falls by a factor of 1.5 to 2.3 on clean data and by 1.15 to
1.4 with noise.  Two causes are consistent with this, and the measurement does not separate them.
Binning smooths the inconsistency the wrong direction produces, and binning also doubles the
high-pass filter's width in ALU, because the filter's width is fixed in pixels.  Halving the fan
angle lowers the contrast.  At bin 2 and the central half the ratio falls by a factor of 1.8 to
1.9.  Noise at 2 percent of the maximum lowers the contrast by up to about half at bin 1, and by
at most about 15 percent at bin 2.

The lowest ratio in the column the check uses, the central half, is 4.26.  At the check's
defaults of stride 4 and bin 2 the lowest is 4.97.  The test suite measures 2.20 at a smaller
geometry of 32 views, 16 rows, and 32 channels, with stride 2 and bin 1.  These results indicate
that the contrast falls as the problem gets smaller, and that the smallest case tested is the
weakest.

## How `check_rotation_direction` uses these results

`check_rotation_direction` builds its reduced problem with `num_slab_slices=None`.  That value
keeps the whole axial extent.  `_direct_residual_score` then takes the residual over the central
half of the detector rows.  The defaults are every fourth view and a detector binned by two.
These defaults keep the cost low.  At these defaults the contrast is 4.97 or more in every case
measured here, and the check warns when the ratio it observes is below 1.5.
