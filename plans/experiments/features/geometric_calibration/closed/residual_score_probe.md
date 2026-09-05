# The residual score as a function of the channel offset: what `residual_score_probe.py` measured

Date: 2026-09-04.  The script is `residual_score_probe.py` in this directory.  It ran in the
`mbirtorch` conda environment on a Mac laptop, with torch 2.13.0, numpy 2.4.6, six threads on the
CPU, and torch.compile off.  The mbirtorch checkout was the `geometric_calibration` branch after the
conjugate-view estimators were added.  Every number below was read from that run's output in the
same session.  The run's stdout is saved at
`/private/tmp/claude-501/-Users-gbuzzard-Documents-PyCharm-Projects-Research-mbirtorch/8c5fc8a2-fffe-42ce-9136-72b210546780/scratchpad/residual_score_probe.log`.
The whole run took 165.4 seconds.  Distances are in arbitrary length units, written ALU, and the
detector channel pitch is 1 ALU, so offsets in ALU are offsets in channels.

The score is `_direct_residual_score` in `mbirtorch/preprocess/geometry_calibration.py`, called the
module below.  Today only `check_rotation_direction` uses it.  The question here is whether the same
score can also estimate `det_channel_offset`.

## The answer

The score has a minimum at the true channel offset in all four scan cases, and noise does not move
that minimum.  The cases differ in how deep the minimum is.  With the whole axial extent kept, a
full circular scan and a helical scan give a score at 2 channels away that is 7.4 to 15.9 times the
score at the minimum.  With a slab of 8 slices out of 32, the same ratio is 1.5 at the central half
of the rows and 1.1 over every row.  On a short scan the ratio is 1.4 to 1.6.

The score is therefore usable on a full circular scan and on a helical scan when the whole axial
extent is kept.  It is not usable on a thin slab.  On a short scan the minimum is in the right
place, but the ratio is too small for the minimum to be located reliably on other data.

Slab thickness matters, and it matters in the way the hypothesis predicted.  The 8-slice curve is
the whole-extent curve plus a nearly constant term.  Over the coarse grid that term runs from 0.3572
to 0.3889, a spread of 8.5 percent of its own mean.  The whole-extent curve varies by a factor of
20.5 over the same grid.  These results indicate that the slab adds a residual that does not depend
on the offset and is much larger than the part of the score that does.  That residual comes from the
material outside the slab, which the slab cannot explain.

The score is accurate but slightly biased low.  The fitted estimate is below the true offset of 1.3
in every case, by 0.009 to 0.043 channels.  The conjugate-view estimator on the same data returns
+1.3090, an error of +0.009 channels.

## Definitions

The residual score for one model and one reduced sinogram is computed as follows.  The model
reconstructs the reduced sinogram directly, which is a filtered back projection.  It then forward
projects that reconstruction.  Both the reduced sinogram and the projection are high-pass filtered
by `sino_high_pass_filtering` at its default widths of 3 detector rows and 15 channels.  The score is
the mean squared difference of the two filtered arrays divided by the mean square of the filtered
sinogram.  The mean is taken over the central `row_fraction` of the detector rows.  The filter widths
are in pixels, so on the small detector here the filter removes more of the signal than it would on
a full-size detector, and the score values are not portable to another detector size.

The candidate is the value of `det_channel_offset` set on the reduced model before the score is
computed.  The coarse grid runs from -2.0 to +4.0 channels in steps of 0.25.  The fine grid runs from
0.8 to 1.8 channels in steps of 0.05.

The fitted estimate is the minimum of a parabola.  The parabola is fitted by least squares to the
five fine-grid points centered on the fine-grid argmin.  The error is the fitted estimate minus the
true offset of 1.3 channels.

The contrast ratio at +2 channels is the coarse-grid score at the candidate nearest 3.3, which is
3.25, divided by the smallest coarse-grid score.  The ratio at -2 channels uses the candidate
nearest -0.7, which is -0.75.  A ratio near 1 means the minimum is shallow.

The noisy cases add Gaussian noise to the full sinogram before it is reduced.  Its standard deviation
is 2 percent of the sinogram maximum.  Three seeds were used, 0, 1, and 2.  The noisy cases score the
fine grid only, at a row fraction of 0.5.

The seconds per evaluation is the mean over the 46 candidates of one row fraction.  Each evaluation
holds one `set_params` call, one direct reconstruction, and one forward projection.

## What was measured

The geometry is a circular cone beam with a flat detector: 128 views, 32 detector rows, and 64
channels.  The source-to-detector distance is 181.4810 ALU and the source-to-iso distance is 90.7405
ALU.  Those two distances give a full fan angle of 20 degrees.  The phantom is the 3D Shepp-Logan
phantom at the model's own recon shape.  A generating model with `det_channel_offset` set to 1.3
forward projected it.  The estimating model has `det_channel_offset` set to 0.0.  The reduced problem
keeps every view and bins nothing, so the only reduction is the slab.

The four cases differ in the view angles and in the slab.  Their shapes are these.

| case | view angles | recon shape | reduced sinogram shape | reduced recon shape | rows kept |
| --- | --- | --- | --- | --- | --- |
| full_slab8 | 0 to 2 pi | (64, 64, 32) | (128, 16, 64) | (64, 64, 8) | 7 to 23 of 32 |
| full_whole | 0 to 2 pi | (64, 64, 32) | (128, 32, 64) | (64, 64, 32) | 0 to 32 of 32 |
| helical_whole | 0 to 4 pi | (64, 64, 48) | (128, 32, 64) | (64, 64, 48) | 0 to 32 of 32 |
| short_whole | 0 to 200 degrees | (64, 64, 32) | (128, 32, 64) | (64, 64, 32) | 0 to 32 of 32 |

The helical case adds per-view axial shifts running from -4 to +4 ALU.  The short scan covers 180
degrees plus the full fan angle of 20 degrees.  `recon_direct` applies no Parker weighting, so its
reconstruction of a short scan is itself approximate.  The short-scan case therefore tests only
whether the score's minimum still sits at the true offset.  The generating and estimating models have
the same recon shape in every case, so the phantom is the same for both.

## Results

The noisy columns were measured at a row fraction of 0.5 only.

| case | row fraction | coarse argmin | fitted estimate | error | ratio at +2 | ratio at -2 | noisy mean | noisy s.d. | s per evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_slab8 | 0.50 | +1.25 | +1.2739 | -0.026 | 1.54 | 1.52 | +1.2739 | 0.0007 | 0.145 |
| full_slab8 | 1.00 | +1.25 | +1.2570 | -0.043 | 1.13 | 1.11 | not run | not run | 0.146 |
| full_whole | 0.50 | +1.25 | +1.2915 | -0.009 | 15.07 | 15.85 | +1.2911 | 0.0001 | 0.289 |
| full_whole | 1.00 | +1.25 | +1.2885 | -0.012 | 9.11 | 9.55 | not run | not run | 0.286 |
| helical_whole | 0.50 | +1.25 | +1.2904 | -0.010 | 12.08 | 12.76 | +1.2902 | 0.0004 | 0.339 |
| helical_whole | 1.00 | +1.25 | +1.2900 | -0.010 | 7.37 | 7.73 | not run | not run | 0.341 |
| short_whole | 0.50 | +1.25 | +1.2854 | -0.015 | 1.47 | 1.61 | +1.2843 | 0.0002 | 0.286 |
| short_whole | 1.00 | +1.25 | +1.2669 | -0.033 | 1.42 | 1.49 | not run | not run | 0.288 |

Every case meets two of the three tests for a usable minimum.  Every fitted estimate is within 0.043
channels of the true offset, which is inside the 0.1 channel gate.  Noise at 2 percent of the
sinogram maximum moves the mean fitted estimate by at most 0.0011 channels away from the clean
value, and its standard deviation across three seeds is at most 0.0007 channels.

The three cases that keep the whole axial extent are not equally deep.  The full circular scan gives
the largest ratios, 15.07 and 15.85 at a row fraction of 0.5.  The helical scan gives 12.08 and
12.76.  The short scan gives 1.47 and 1.61.  Scoring the central half of the rows raises the ratio
over scoring every row in all four cases, by a factor of 1.0 to 1.7.

The conjugate-view estimator was run on the same clean sinogram of the two full-rotation cases.  Both
cases share one geometry and one sinogram, so both give the same answer.

| case | conjugate estimate | error | seconds |
| --- | --- | --- | --- |
| full_slab8 | +1.3090 | +0.009 | under 0.1 |
| full_whole | +1.3090 | +0.009 | under 0.1 |

The conjugate estimator is more accurate than the residual score and much cheaper.  Its error is
+0.009 channels against the residual score's -0.009 at best and -0.043 at worst.  It ran in under a
tenth of a second, against 0.29 seconds for one residual evaluation and about 13 seconds for the 46
candidates of one grid pass.

## Why the thin slab fails

The 8-slice slab raises the whole score curve without changing its shape much.  Subtracting the
whole-extent curve from the 8-slice curve, point by point on the coarse grid at a row fraction of
0.5, leaves a difference that runs from 0.3572 to 0.3889.  That is a spread of 8.5 percent of the
difference's own mean of 0.3714.  Over the same grid the whole-extent score itself runs from 0.012286
to 0.252337, a factor of 20.5.

These results confirm the hypothesis.  Rays through the slab cross material outside the slab, and the
slab cannot represent that material.  The mismatch that follows is nearly the same for every
candidate offset.  It therefore adds a nearly constant term to the score, and that term lowers the
contrast ratio.  The absolute depth of the minimum remains.  The 8-slice curve still falls by 0.257
from its value at -2.0 channels to its minimum.  The ratio is what the added term reduces.  A search
on real data would have to work with the ratio, because the noise floor there is not known in
advance.

This finding matches the earlier one in `direction_score_contrast.md`.  There a slab of 4 or 8 slices
left almost no contrast between the two rotation directions, for the same reason.

## Why the short scan is weak

The short-scan score curve is flat compared with the full-rotation curve.  Over the coarse grid at a
row fraction of 0.5 it runs from 0.033261 to 0.063696, a factor of 1.9.  The full circular scan over
the same grid runs from 0.012286 to 0.252337, a factor of 20.5.  The short scan's minimum score is
2.7 times the full scan's minimum score.

Two effects are consistent with this, and the measurement does not separate them.  A short scan has
no Parker weighting in `recon_direct`, so its reconstruction is wrong even at the true offset, and
that error adds a floor to the score.  A short scan also measures each ray once rather than twice,
so a wrong offset produces less inconsistency between views.  The minimum still sits at the true
offset, with a fitted error of -0.015 channels at a row fraction of 0.5.

## Limits of this evidence

Six limits apply.  The data are synthetic and were generated by the same projector that the score
uses.  That is an inverse crime, meaning the model that fits the data is exactly the model that made
them, so the agreement is better than it would be on measured data.  One phantom was used, the 3D
Shepp-Logan phantom, which is nearly centered.  One fan angle was used, 20 degrees full.  The
detector is small at 32 rows by 64 channels, and the high-pass filter's widths are fixed in pixels,
so the score values do not transfer to a full-size detector.  Everything ran on the CPU, so the
seconds per evaluation are not a guide to cost on a GPU.  The short-scan case has no Parker weighting
in `recon_direct`, so its reconstruction is approximate at every candidate.

Two more notes on the run.  No case raised an exception.  The script collected warnings raised inside
the scored regions rather than printing them, and it counted zero.
