# Why the rotation estimate uses cubic resampling and warns at small edge displacements

Date: 2026-09-04.  The script is `rotation_interpolation_bias.py` in this directory.  It ran in
the `mbirtorch` conda environment on a Mac laptop, with torch 2.13.0 on the CPU and torch.compile
off.  The mbirtorch checkout was the `geometric_calibration` branch during Increment 2, after the
row band of the conjugate comparison was centered on the central plane.  Every number below was
read from that run's output in the same session.

## Conclusion

A conjugate-view estimate of the detector rotation that resamples the sinogram at each candidate
angle is biased when the rotation displaces the edge pixel of the detector by less than about
half a pixel.  With the cubic kernel the bias is 14 to 18 percent of the angle at an edge
displacement of 0.17 pixels and 0 to 6 percent from 0.56 pixels upward.  The bilinear kernel is
unusable on cone beam at small displacements.  `estimate_det_rotation` therefore resamples with
the cubic kernel, and it warns when its estimate displaces the edge pixel by less than one pixel.
The threshold was half a pixel on this measurement, and the cluster measurement in
`calibration_512_gautschi.md`, where the error was 4.5 percent at 0.89 pixels, moved it to one.

## Definitions

A detector rotated by an angle records every view rotated by that same angle.  Mirroring a view in
the channel direction reverses the sign of that rotation.  A view and its opposite view therefore
differ by twice the angle.

The estimator under test keeps every view.  It rotates a row band from each view by the candidate
angle, pairs each view with its opposite view by the conjugate-ray rule, and scores how closely
the two agree.  The search reuses the two stages of `estimate_det_channel_offset`.  A coarse pass
evaluates candidates on a grid, and a golden-section stage then refines the minimum.  Every view
pair is scored, with no trimming.

The edge displacement of a rotation is the distance the edge pixel of the detector moves, which
is the angle in radians times half the channel count.  A rotation of 0.3 degrees displaces the
edge pixel of a 64 channel detector by 0.17 pixels and of a 2000 channel detector by 5 pixels.

## How the tilted test data were made

The tilted test data were generated so that no interpolation kernel was favored.  A rotation
applied by resampling at the detector's own resolution smooths the data in the way the bilinear
estimator does.  The bilinear estimator would then recover that rotation exactly, and the
agreement would prove nothing.  The sinogram was instead generated at four times the detector
resolution.  It was rotated at that resolution with the bilinear kernel, then binned by four.
Whether four times is enough was not checked.  At four times, a rotation of 0.3 degrees still
displaces the fine grid's edge pixel by only 0.67 pixels, so the generation smooths the data by
about a quarter of what the estimator's own resampling does at the detector's resolution.

## Part A: three kernels and two scores at three small rotations

Every geometry used the Shepp-Logan phantom.  Four configurations were run: parallel beam with 64
views, 32 rows, and 64 channels; and cone beam with 128 views at a full fan angle of 20 degrees,
in three detector sizes of 32 rows by 64 channels, 64 rows by 64 channels, and 64 rows by 128
channels.  The true rotations were 0.3, -0.5, and 0.0 degrees, and the search interval was one
degree on each side of zero.

Five variants were compared.  The interpolation kernel of the rotation was bilinear, cubic, or
Lanczos, through `cv2.warpAffine`.  The score was the module's normalized mean squared
difference, labeled `ssd`, or one minus the normalized correlation, labeled `ncc`.  The `ssd`
score with the Lanczos kernel was not run, so five variants remain.  Each variant ran under two
row-band settings.  The first is the default band of `geometry_calibration`, which is 16 rows for
parallel beam and 5 rows for the cone geometries here.  The second is a fixed band of 16 rows,
so the parallel-beam rows are the same under both settings.

Each cell below is the estimate in degrees.  An asterisk marks a search whose coarse minimum fell
at an edge of the search interval, or whose coarse scores had two local minima.  The detector
sizes are rows by channels.

| geometry | rows | true | ssd-linear | ncc-linear | ssd-cubic | ncc-cubic | ncc-lanczos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| parallel 32 by 64 | 16 | +0.3 | +0.321 | +0.321 | +0.246 | +0.247 | +0.267 |
| parallel 32 by 64 | 16 | -0.5 | -0.519 | -0.517 | -0.443 | -0.444 | -0.496 |
| parallel 32 by 64 | 16 | 0.0 | 0.000 | 0.000 | 0.000 | 0.000 | -0.011 |
| cone 32 by 64 | 5 | +0.3 | +0.550* | +0.495 | +0.180 | +0.171 | +0.045 |
| cone 32 by 64 | 16 | +0.3 | +0.462 | +0.399 | +0.258 | +0.261 | +0.294 |
| cone 32 by 64 | 5 | -0.5 | -0.696 | -0.661 | -0.315 | -0.330 | -0.120 |
| cone 32 by 64 | 16 | -0.5 | -0.624 | -0.578 | -0.465 | -0.472 | -0.479 |
| cone 32 by 64 | 5 | 0.0 | +0.333* | -0.265* | +0.002 | -0.016 | -0.011 |
| cone 32 by 64 | 16 | 0.0 | +0.227* | +0.125* | 0.000 | 0.000 | -0.011 |
| cone 64 by 64 | 5 | +0.3 | +1.000* | +1.000* | +0.225 | +0.184 | -0.011 |
| cone 64 by 64 | 16 | +0.3 | +1.000* | +1.000* | +0.153 | +0.144 | -0.011 |
| cone 64 by 64 | 5 | -0.5 | -1.000* | -1.000* | -0.374 | -0.442 | +0.032 |
| cone 64 by 64 | 16 | -0.5 | -1.000* | -1.000* | -0.346 | -0.354 | -0.011 |
| cone 64 by 64 | 5 | 0.0 | +1.000* | -1.000* | +0.020 | -0.020 | -0.011 |
| cone 64 by 64 | 16 | 0.0 | -1.000* | -1.000* | 0.000 | +0.001 | -0.011 |
| cone 64 by 128 | 5 | +0.3 | +0.710* | +0.528* | +0.146 | +0.172 | +0.016 |
| cone 64 by 128 | 16 | +0.3 | +0.482 | +0.403 | +0.196 | +0.228 | +0.153 |
| cone 64 by 128 | 5 | -0.5 | -0.839* | -0.675 | -0.321* | -0.616 | -0.022 |
| cone 64 by 128 | 16 | -0.5 | -0.620 | -0.558 | -0.605 | -0.569 | -0.335 |
| cone 64 by 128 | 5 | 0.0 | +0.519* | -0.330* | +0.001 | -0.003 | +0.015 |
| cone 64 by 128 | 16 | 0.0 | -0.287* | -0.176* | 0.000 | 0.000 | -0.011 |

The bilinear kernel is unusable on cone beam at these rotations.  On the 64 row by 64 channel
detector, every bilinear search placed its minimum at a bound of the search interval.  On the
other two cone detectors it reported a rotation of 0.1 to 0.5 degrees where the true rotation was
zero.  These results indicate that bilinear resampling lowers the score as the candidate angle
grows.  The cause is that the smoothing it applies also grows with the angle.  At these
rotations that effect is larger than the alignment signal itself.

The cubic kernel underestimates the magnitude of the rotation.  On parallel beam it gives 0.246
degrees for a true 0.3 and -0.443 for a true -0.5.  On the cone geometries it gives 0.146 to
0.261 degrees for a true 0.3, and -0.315 to -0.616 degrees for a true -0.5, with one search at 5
rows on the largest detector that stopped at a second minimum.  The cubic kernel does find zero
when the rotation is zero, to within 0.02 degrees.

The Lanczos kernel is inconsistent.  For a true rotation of 0.3 degrees it gives from -0.011 to
0.294 degrees.

Replacing the mean squared difference with the normalized correlation changes the cubic
estimates by at most 0.07 degrees.  It does not rescue the bilinear kernel.

The parallel-beam bilinear estimate is accurate to about 0.02 degrees, giving 0.321 for a true 0.3
and -0.519 for a true -0.5.  Parallel beam has no cone-angle term in the pairing rule, and its row
band is the full 16 rows.  In this one case the smoothing effect is small compared with the
alignment signal.

## Part B: the bias against the size of the rotation

The bias depends on the edge displacement.  Part B holds the detector at 64 channels and raises
the true rotation from 0.3 to 3 degrees, so the edge displacement runs from 0.17 to 1.68 pixels.
That range covers what a 2000 channel detector reaches at rotations of 0.01 to 0.1 degrees.  The
search interval was five degrees on each side of zero, the row band was 16 rows, and the score
was `ssd`.  Each cell is the estimate in degrees and its error as a percentage of the true
rotation.

| geometry | true | edge displacement | ssd-linear | ssd-cubic |
| --- | --- | --- | --- | --- |
| parallel 32 by 64 | 0.3 | 0.17 pixels | +0.321 (+7 percent) | +0.247 (-18 percent) |
| parallel 32 by 64 | 1.0 | 0.56 pixels | +0.959 (-4 percent) | +0.972 (-3 percent) |
| parallel 32 by 64 | 2.0 | 1.12 pixels | +1.814 (-9 percent) | +2.039 (+2 percent) |
| parallel 32 by 64 | 3.0 | 1.68 pixels | +2.918 (-3 percent) | +2.949 (-2 percent) |
| cone 32 by 64 | 0.3 | 0.17 pixels | +0.461 (+54 percent) | +0.259 (-14 percent) |
| cone 32 by 64 | 1.0 | 0.56 pixels | +1.017 (+2 percent) | +1.019 (+2 percent) |
| cone 32 by 64 | 2.0 | 1.12 pixels | +1.848 (-8 percent) | +2.119 (+6 percent) |
| cone 32 by 64 | 3.0 | 1.68 pixels | +3.039 (+1 percent) | +3.012 (+0 percent) |

The cubic kernel's bias falls from 14 to 18 percent at 0.17 pixels to 0 to 6 percent from 0.56
pixels upward.  The bilinear kernel's bias falls from 54 percent to within 9 percent over the same
range on cone beam, and it stays within 9 percent on parallel beam throughout.  These results
indicate that the bias is a property of sub-pixel edge displacements, not of the method.  A
rotation large enough to matter on a full-size detector displaces the edge pixel by several
pixels.

The rotations in Part B are larger than the rotations a real scanner shows, and the detector is
small, so the measurement reaches the same edge displacements by a different route.  Whether the
bias at a given edge displacement is the same on a large detector at a small rotation was not
measured.  A run at 512 or more channels on the cluster would settle it.

## What the module does with this

`estimate_det_rotation` resamples each candidate with the cubic kernel through `cv2.warpAffine`,
and it scores the mean squared difference.  It warns when its estimate displaces the edge pixel
by less than one pixel.  A rotation applied inside the projectors would need no resampling and
would have no such regime.
