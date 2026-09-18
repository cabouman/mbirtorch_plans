# The rotation estimate against the band height on synthetic data: what `rotation_zero_point_synthetic.py` measured

Date: 2026-09-05.  The script is `rotation_zero_point_synthetic.py` in this directory.  It ran on
the Mac in the miniforge `mbirtorch` environment, torch 2.13.0 on the CPU with six threads and
torch.compile off, against the `geometric_calibration` branch working tree of mbirtorch.  Every
number below was read from the run's output in this session.  The whole run took about nine
minutes, of which the two fine projections took 256 and 261 seconds.

Units.  A rotation is given in degrees.  A band is the window of detector rows the estimator
compares, and its height is in detector rows.  The cross-row statistic of a band is the mean
squared difference between neighboring rows divided by the mean square of the band.

## The question

On a real NSI scan the rotation estimator read 0.047 degrees where the vendor's 0.167 degrees is
right, and the suspected cause is that the default band of a few rows holds almost nothing a
rotation changes for an object whose cross-row structure sits far from the central plane.  This
script builds that situation with a known angle: a cylinder that is the same in every slice, one
darker slab of slices either far from the central plane or on it, and a tilt of 1.5 degrees
injected at four times the detector resolution.  The tilt displaces the edge channel by 2.09
pixels, which is the regime of the real scan and above the sub-pixel region where the resampling
kernel's own bias dominates.

## The answer

The failure reproduces, and it belongs to the default band.  On the phantom whose slab sits about
50 rows from the central plane, the default 11-row band estimates 1.137 degrees for a true 1.500,
an error of 24 percent.  On the same phantom with the slab moved to the central plane, the same
band estimates 1.423 degrees, an error of 5 percent.  The two runs differ only in the slab's
position, so the default band's estimate depends on the object in the way the real scans showed.

A taller band recovers the estimate, and it recovers before it reaches the slab.  On the far-slab
phantom the estimate rises with the band height: 1.118 at 9 rows, 1.195 at 17, 1.348 at 33, 1.526
at 65, and 1.469 at 127.  The 65-row band ends about 32 rows from the central plane, which is
short of the slab at 50 rows, and its cross-row statistic is 3.2e-06, as small as the default
band's.  The estimate there is nonetheless within 1.8 percent of the truth.  These results
indicate that the taller band regains the rotation through the channel shear a candidate angle
applies across the band, which grows with the band's height and needs only structure along the
channels, such as the cylinder's own boundary.  Reaching the object's cross-row structure is not
required.

The size of the shear term explains the recovery.  A residual angle error shifts a band row's
channels in proportion to the row's distance from the band center.  At the default band's five
rows that shift is a few hundredths of a channel for the 0.36 degree error observed, and at 32
rows it is about 0.4 channels, which the cylinder's edge makes visible.

Two controls behaved as they should.  The same far-slab phantom projected with no tilt returned
exactly 0.0000 degrees at every band height, so a taller band invents no rotation.  The near-slab
phantom's estimates at 65 and 127 rows are 1.506 and 1.468 degrees, within 2.2 percent, so the
taller bands are accurate when the default band already is.

The cross-row statistic does not mark the unreliable runs by itself.  The far-slab run at 65 rows
has the same tiny statistic as the default band and a good estimate, and the near-slab default
run has a statistic four orders larger and a good estimate too.  What does mark the failure is
disagreement across band heights: the far-slab estimates at the default and at 65 rows differ by
0.39 degrees, while the near-slab pair differs by 0.08.  A reliability check should therefore
compare estimates at two or three band heights rather than read the band's gradient energy.

## What was measured

The geometry is a circular cone beam with a flat detector: 128 views over a full rotation, 128
rows, 160 channels, both pitches 1 ALU, source to iso 400 ALU, and source to detector 800 ALU.
The half fan angle is 5.7 degrees and the half cone angle 4.6 degrees.  The phantom is a cylinder
of radius 35 percent of the recon's half width, the same in every slice, with one slab of 8 fine
slices multiplied by 0.65.  The far position centers the slab 78 percent of the way from the
central slice to the top, and the near position centers it on the central slice.  The fine model
has four times the detector counts and a quarter of the pitches, the tilt is applied by
`reduce_sinogram` at that resolution with the bilinear kernel, and the same call bins by four.
The estimating model's channel offset is zero, which is the true value, so nothing estimates it.

## Results

The table is the run's own, transcribed.  The band column is the height asked for, and the window
is the rows the module used; `default` lets the module pick, and its cone rule gave 11 rows here.
The max/min column is the largest score of the search divided by the smallest.

| case | band | rows | window | estimate, deg | error, deg | min score | max/min | cross-row | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| far slab, tilted | default | 11 | 58-69 | +1.1370 | -0.3630 | 6.58e-06 | 378 | 3.2e-06 | 0.5 |
| far slab, tilted | 9 | 9 | 60-69 | +1.1184 | -0.3816 | 4.73e-06 | 376 | 3.2e-06 | 0.3 |
| far slab, tilted | 17 | 17 | 56-73 | +1.1950 | -0.3050 | 1.31e-05 | 389 | 3.2e-06 | 0.4 |
| far slab, tilted | 33 | 33 | 48-81 | +1.3475 | -0.1525 | 2.07e-05 | 833 | 3.2e-06 | 0.5 |
| far slab, tilted | 65 | 65 | 32-97 | +1.5264 | +0.0264 | 2.67e-05 | 2074 | 3.2e-06 | 0.8 |
| far slab, tilted | 127 | 127 | 0-127 | +1.4690 | -0.0310 | 1.09e-04 | 1430 | 1.8e-03 | 1.4 |
| near slab, tilted | default | 11 | 58-69 | +1.4226 | -0.0774 | 5.83e-05 | 594 | 1.8e-02 | 0.3 |
| near slab, tilted | 9 | 9 | 60-69 | +1.4250 | -0.0750 | 6.63e-05 | 634 | 2.3e-02 | 0.3 |
| near slab, tilted | 17 | 17 | 56-73 | +1.4111 | -0.0889 | 4.96e-05 | 509 | 1.1e-02 | 0.4 |
| near slab, tilted | 33 | 33 | 48-81 | +1.4040 | -0.0960 | 3.69e-05 | 751 | 5.2e-03 | 0.5 |
| near slab, tilted | 65 | 65 | 32-97 | +1.5063 | +0.0063 | 3.71e-05 | 1655 | 2.6e-03 | 0.9 |
| near slab, tilted | 127 | 127 | 0-127 | +1.4676 | -0.0324 | 1.12e-04 | 1431 | 2.8e-03 | 1.4 |
| far slab, no tilt | default | 11 | 58-69 | +0.0000 | +0.0000 | 3.96e-07 | 3932 | 2.1e-11 | 0.3 |
| far slab, no tilt | 9 | 9 | 60-69 | +0.0000 | +0.0000 | 3.96e-07 | 2839 | 1.3e-11 | 0.3 |
| far slab, no tilt | 17 | 17 | 56-73 | +0.0000 | +0.0000 | 3.96e-07 | 7931 | 5.2e-11 | 0.3 |
| far slab, no tilt | 33 | 33 | 48-81 | +0.0000 | +0.0000 | 3.96e-07 | 27572 | 2.1e-10 | 0.5 |
| far slab, no tilt | 65 | 65 | 32-97 | +0.0000 | +0.0000 | 3.96e-07 | 89749 | 8.3e-10 | 0.8 |
| far slab, no tilt | 127 | 127 | 0-127 | +0.0000 | +0.0000 | 4.10e-07 | 255934 | 1.1e-03 | 1.5 |

Every no-tilt run raised the module's sub-pixel warning, because an estimate of zero displaces the
edge channel by zero pixels.  No other run warned.  Every search minimum in the table is deep, at
a max/min ratio of 376 or more, including the runs whose estimate is 24 percent wrong.  These
ratios repeat the real-scan finding that a deep minimum does not mean a right one.

## The default band's cone rule looks overcautious here

The default band is limited to the rows where opposite rays through the support land within one
row of each other, which gave 11 rows at this geometry.  The 65-row and 127-row bands violate
that rule and estimate better on both phantoms.  The mismatch a taller band admits does not
depend on the candidate angle, so it raises the score's floor without moving its minimum much,
in the same way the thin-slab term raised the direct-residual score without moving it.  This
holds at the 4.6 degree half cone angle measured and was not tested at a larger one.

## Limits of this evidence

Seven limits apply.  The data are synthetic, made by the projector that also defines the pairing,
and the tilt was injected with the package's own bilinear kernel at four times the detector
resolution.  One geometry, one fan angle, and one cone angle were run.  The cylinder is as tall
as the volume, so its end faces land on the outermost detector rows; the 127-row band includes
them, which is visible in its cross-row statistic, and they may explain why both phantoms read
about 2 percent low there.  The edge displacement is 2.09 pixels, while the real no-metal scan's
estimate sits at 0.62 pixels, where the resampling bias adds a further error this run does not
carry.  The channel offset is exactly right here, so the offset coupling the real scans show is
absent.  The detector is small at 128 by 160.  Everything ran on the CPU.
