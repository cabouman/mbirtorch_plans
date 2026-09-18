# Fine sweep of the detector rotation on the two NSI scans

Date: 2026-09-05.  Two resampling kernels were compared.  The scripts are `recon_sweep_fine.py` and
`recon_sweep_fine_fourier.py`.

Three Slurm jobs on gautschi produced everything below, and all three completed.  Job 15970242 ran
`recon_sweep_fine.py`, which applies each candidate rotation with the bilinear kernel the corrections
use.  For that job `sacct` reported four values: state `COMPLETED`, exit code `0:0`, elapsed time
`00:19:47`, and a batch-step `MaxRSS` of 41003092 KB.  Job 15971893 ran `recon_sweep_fine_fourier.py`,
which is the same sweep with the Fourier kernel in place of the bilinear kernel.  For that job `sacct`
reported three values: state `COMPLETED`, exit code `0:0`, and elapsed time `00:20:05`.  Job 15980973
ran `recon_sweep_fine_tables.py` on the two jobs' records and `recon_sweep_fine_analysis.py` on their
saved stacks.  For that job `sacct` reported three values: state `COMPLETED`, exit code `0:0`, and
elapsed time `00:03:28`.

The two sweep jobs each used one NVIDIA H100 80GB HBM3 GPU.  Each of their batch files requests two of
those GPUs, for the host memory the partition allocates per GPU.  Those two jobs recorded torch
2.13.0+cu130 and mbirtorch 0.0.2, at commit `590fea9` on the `geometric_calibration` branch
(`tables.txt`, ENVIRONMENT).  The analysis job imports neither of those two packages, and it requests
one GPU for the same reason of host memory.

Every number below was read in this session from those jobs' output.  The bilinear job's output is the
log `/scratch/gautschi/buzzard/leap_cmp/recon_sweep_fine_15970242.log` and three files in
`/scratch/gautschi/buzzard/leap_cmp/results_recon_sweep_fine`.  Those three files are these:
`recon_sweep_fine.jsonl`, `tables.txt`, and `analysis.txt`.  The sweep job appended the JSON lines.
The third job wrote the tables and the analysis.  The Fourier job's output is the log
`recon_sweep_fine_fourier_15971893.log` and the same three files in
`results_recon_sweep_fine_fourier`.  The kernel check is `check_kernel()` in
`recon_sweep_fine_fourier.py`.  It was run on a Mac, and its output is `kernel_check.json`.  Each
sweep job wrote three figures per scan, and the analysis job wrote three more, so each results
directory holds six figures per scan.  This repository ignores five kinds of file: `.jsonl`, `.json`,
`.log`, `.png`, and `.sbatch`.  This page is therefore the durable copy of what it quotes.

## Units and terms

This section defines the units and the terms used below.  It also distinguishes the two detector rows
that appear below.

A detector rotation is given in degrees in prose and in radians in the formulas below.  An offset is
given in channels.  One channel is one detector pitch, which is 0.127 mm on these scans.  A slice's
height is given in rows from the central plane.  The central plane passes through the source and is
perpendicular to the rotation axis.  The blur width is the standard deviation of a Gaussian in
reconstruction voxels.  On these scans `delta_voxel` is also the reconstruction pixel pitch, and
`voxel_row_aspect` is 1.0.  A blur of 2 voxels is therefore a blur of 2 reconstruction pixels.

Two detector rows appear below, and they are different rows.  Both kernels turn about the detector's
center row.  That row is `(num_rows - 1) / 2`, which is 939.5 on these 1880-row scans
(`recon_sweep_fine_analysis.py`, `scan_record`).  The central-plane row is 956.90 here (`tables.txt`).
The kernel row distance of a slice is its detector row on the rotation axis minus the detector's
center row.  The plan of record measures a slice's height from the central-plane row instead.  That
plan is `plans/geometric_calibration/estimate_by_recon_plan.md`.  The two measures differ by
a constant of 17.4 rows.  Every fit below uses the kernel row distance, and the 17.4-row difference is
not modeled here.

The lattice of a slice is the set of candidate rotations that displace that slice's rows by a whole
number of pixels.  A row at kernel row distance `i` moves by `a * i` pixels along the channels under a
rotation of `a` radians.  The lattice is therefore the set of angles `a = k / |i|` for integer `k`
(`recon_sweep_fine_analysis.py`, `lattice_points`).  At a lattice angle bilinear resampling copies
pixels, so it smooths nothing.  At a half-pixel displacement it averages neighbors, so it smooths
most.

The even/odd split uses the two half reconstructions each job saved.  Each slice was reconstructed
from the even-index views and from the odd-index views separately.  The mean of the two halves is
the all-view slice.  The half difference is half of the even-view slice minus the odd-view slice,
and it holds noise only.  That noise has the same variance as the noise in the mean.  The noise
curve is the negative gradient energy of the half difference divided by the mean's mean square, so
it is the noise's share of the mean's score.  The structure curve is built from two
subtractions.  The half difference's gradient energy is subtracted from the mean's gradient energy,
and the half difference's mean square from the mean's mean square.  The structure curve is the score
formed from those two differences (`recon_sweep_fine_analysis.py`, `split_scores`).  The noise share
of a slice is the value of its noise curve divided by its all-view score, at the same candidate and
blur.  The identity check is the job's direct test of the claim that the mean of the two halves is
the all-view slice.

The offset-error model is `a_s = a_true + d / i_s`.  Its symbols are these: `a_s` is the rotation at
which slice `s` has its score minimum, `a_true` is the rotation the model fits, `i_s` is that slice's
kernel row distance, both angles are in radians, and `d` is in channels.  The model says that a
residual channel offset displaces each slice's minimum by an amount that falls with the slice's kernel
row distance.  The sign of `d` below is the sign the fit returns (`recon_sweep_fine_analysis.py`,
`fit_offset_model`).  The plan's section "How the score reads the geometry" states the mechanism, and
it puts each slice's minimum at the true rotation minus `d / i`.  The fitted `d` is therefore the
negation of the plan's `d`.

Four terms come from the plan's "Units and terms".  The trimmed curve is the per-candidate mean over
the scored slices after the highest and the lowest slice scores are dropped.  A curve's depth is its
largest score minus its smallest over the searched range.  The half width of a minimum is the distance
at which a quadratic fit near it rises two percent above the fitted minimum.  The repeatability floor
is the largest absolute difference between the even-view curve and the odd-view curve.  The largest is
taken over the candidates.

The trimmed curve takes two forms, and they give different answers below.  The raw trimmed curve trims
the slice scores as the job computed them.  The normalized trimmed curve trims them after each slice's
scores are divided by the magnitude of that slice's own mean over the candidates
(`recon_sweep_fine.py`, `normalize_rows`).  Each form is named where it is used.

## The answers

The rotation that reconstruction quality prefers is near 0.150 degrees on the scan without metal and
near 0.160 degrees on the scan with the metal insert.  Those two values are fitted parameters of the
offset-error model, and not places where any single reconstruction was sharpest.  They come from
fitting that model to the Fourier job's structure-curve minima at a blur of 2 voxels
(`results_recon_sweep_fine_fourier/analysis.txt`, both scans).  The fitted rotations are 0.1507 and
0.1595 degrees.  The same model fitted to the job's own per-slice minima gives 0.1501 and 0.1594
degrees.  The three structure-curve minima behind the no-metal fit are 0.1748, 0.1272, and 0.1374
degrees, on slices 470, 1409, and 1691.  The three behind the metal fit are 0.1808, 0.1393, and 0.1475
degrees.

Neither fitted value is one of the two candidates the sweep was run to separate.  The vendor's
recorded tilt is 0.16717 degrees (`tables.txt`).  A second earlier estimate comes from the
conjugate-view method applied to a tall detector band, and it gives 0.1299 degrees on the no-metal
scan (`closed/real_scan_band_reach.md`).  That estimate is called the tall-band answer below.

The bilinear kernel does not measure that rotation.  With it the four slices' score minima disagree,
and each one is within 0.007 degrees of the lattice angle of its own row (`analysis.txt`, PART 1).
Removing the noise term does not move those minima off the lattice.  Neither does a blur of up to 4
voxels.  At every blur each minimum stays within 0.008 degrees of its own lattice angle.  The
offset-error model fitted to the structure curves leaves residuals of up to 0.0146 degrees on the
no-metal scan at a blur of 2 voxels.  At that blur the model's own `d / i_s` term spans about 0.005
degrees over the three slices, while the minima themselves span 0.0275 degrees.  The fitted term
therefore explains almost none of the spread.

The one variable that was changed was the kernel.  A Fourier shift multiplies every frequency by a
phase of unit magnitude, so the amount of smoothing it applies does not depend on the fractional part
of the displacement.  That kernel changed three things: the noise curves became flat, the per-slice
minima moved off the lattice, and the offset-error model fitted those minima.  On the no-metal scan at
a blur of 2 voxels the fit's residuals are at most 0.0010 degrees.  The three minima it fits span
0.0476 degrees.  The model therefore explains a spread of nearly five hundredths of a degree to within
a thousandth.  The bilinear fit left 0.0146 degrees unexplained.  These results indicate that the
bilinear kernel's displacement-dependent smoothing placed the per-slice minima of the first job.  The
geometry did not place them.

The bilinear job's trimmed curve has a narrow and repeatable minimum all the same.  That minimum is
not at the rotation the geometry sets.  A trimmed curve can therefore have a narrow and repeatable
minimum while the quantity that minimum locates is not the geometry.

Two numbers measure two parts of the spread on the 0.150 and 0.160 degrees.  The first is the
repeatability, which is the difference between the even-view answer and the odd-view answer.  At a
blur of 2 voxels the Fourier job's per-slice minima differ between the two parities by at most 0.0020
degrees, on the slices that hold the object.  The fitted rotation inherits at most that figure, and
its own parity repeatability was not computed.  The second is the dependence on the blur.  The fitted
rotation moves by 0.0033 degrees on the no-metal scan and 0.0038 degrees on the metal scan between a
blur of 2 and a blur of 4 voxels.

The largest measured term is neither of those two.  The two scans share one recorded geometry, and the
Fourier fit gives 0.1507 degrees on one and 0.1595 degrees on the other.  Those differ by 0.0088
degrees, and the per-slice minima of the two scans differ by up to 0.0124 degrees at the same blur.
That difference is either a real difference between the two acquisitions or a limit of the score.
This record does not decide which.

Three further terms are not covered by any of those numbers.  They are the two-shear approximation of
the Fourier kernel, the cone-beam approximation, and the choice of kernel itself.  The bilinear kernel
gives 0.1336 degrees where the Fourier kernel gives 0.1507 degrees.  That spread is not carried as an
uncertainty, because the evidence below shows the bilinear kernel to be biased.  Only one unbiased
kernel was tried, so no second method confirms the 0.150 degrees.

## What was measured

Each job swept the detector rotation on the two NSI scans and scored four reconstructed slices per
scan.  The scans are the NSI artifact phantom without and with a metal insert, 1800 views each.  They
were read with the vendor's tilt held out of the sinogram, as the earlier jobs read them
(`closed/real_scan_validation.md`).  Both jobs report four scan values: a vendor tilt of 0.16717
degrees, a vendor offset of -14.125 channels, a central-plane row of 956.90, and an angular coverage
of 359.80 degrees (`tables.txt`).  The reconstruction shape is 1496 by 1496 by 1880 voxels, at a
`delta_voxel` of 0.03645 and a `voxel_row_aspect` of 1.0.

`parameter_sweep` does the reconstruction, as it did in the four-candidate job
(`closed/real_scan_rotation_recon.md`).  For one slice it crops the detector to the rows that slice
needs, rotates that band of every view by the candidate angle, and reconstructs the slice directly.
The score is the gradient energy of the blurred slice divided by that slice's mean square.  The score
is negated, so a lower score is better.  Every candidate is resampled, because the candidate grids
start at 0.10 degrees and no candidate is zero.

The two jobs ran the same grids.  The no-metal scan was swept from 0.100 to 0.200 degrees, and the
metal scan from 0.100 to 0.240 degrees, both in steps of 0.005 degrees.  Two candidates were added to
each grid by name: 0.130 degrees and the vendor's 0.16717 degrees.  That second added candidate is
called the vendor candidate below.  The grids hold 22 candidates on the no-metal scan and 30 on the
metal scan.  Each reconstructed slice was scored at five blur widths: 0, 1, 2, 3, and 4 voxels.  Each
candidate was reconstructed twice, once from the 900 even-index views and once from the 900 odd-index
views.

The four slices are the slices of the earlier job that lie away from the central plane.  The table
gives them, with the mean squares the bilinear job recorded at the vendor candidate (`tables.txt`,
SLICES).

| slice | rows from the central plane | kernel row distance | mean square, no metal | mean square, with metal | holds the object |
| --- | --- | --- | --- | --- | --- |
| 188 | -751.5 | -734.1 | 3.208e-07 | 3.138e-07 | no |
| 470 | -469.5 | -452.1 | 4.835e-05 | 4.954e-05 | yes |
| 1409 | +469.5 | +486.9 | 5.738e-05 | 5.834e-05 | yes |
| 1691 | +751.5 | +768.9 | 3.474e-05 | 2.543e-05 | yes |

The identity check passed on every slice of both jobs.  The mean of the even-view and odd-view slices
differs from the all-view slice by at most a relative 6.7e-07 at any pixel (`tables.txt`, SLICES, both
jobs).  In root mean square the difference is at most a relative 2.6e-07.  These results indicate that
the two halves together deliver the all-view reconstruction.  The two halves also use the same 1800
views that one all-view reconstruction uses, so the even/odd split adds no views.  No all-view sweep
was timed here, so the equal cost is arithmetic on view counts and not a measurement.

The analysis job reproduced the two jobs' own scores from the saved stacks to a relative 3.0e-09 or
better on every slice (`analysis.txt`, PART 2, both jobs).  That check is what allows the even/odd
split and the lattice comparison to be read as statements about the jobs' scores.

The cost of one job is about twenty minutes.  The table gives the per-scan figures from the
`resources` entries of each job.

| job | scan | load, s | seconds for the scan | host peak, GB | GPU peak, GB |
| --- | --- | --- | --- | --- | --- |
| 15970242, bilinear | no metal | 31.2 | 525.8 | 40.6 | 7.15 |
| 15970242, bilinear | with metal | 30.9 | 640.2 | 41.2 | 7.15 |
| 15971893, Fourier | no metal | 31.6 | 533.7 | 40.6 | 7.15 |
| 15971893, Fourier | with metal | 30.7 | 645.4 | 41.3 | 7.15 |

Sweeping one slice of the no-metal scan took 24.7 to 43.2 seconds in the two jobs, per half of the
even/odd split.  On the metal scan it took 33.7 to 54.3 seconds per half (`tables.txt`, SLICES).  The
one all-view reconstruction of the identity check added 2.2 to 3.5 seconds per slice.

## The empty slice

The slice at 751.5 rows below the central plane holds no object.  On the no-metal scan its mean square
at the vendor candidate is 3.208e-07, against 3.474e-05 on the slice at 751.5 rows above the plane
(`tables.txt`).  The empty slice's mean square is therefore smaller by a factor of 108.  On the metal
scan the same two figures are 3.138e-07 and 2.543e-05, a factor of 81.  The four-candidate job
reported the same slice as lying below the phantom (`closed/real_scan_rotation_recon.md`).

The analysis leaves that slice out of the fits and the combined structure curves.  The rule is this: a
slice is treated as holding no object when its mean square is below five percent of the largest
slice's mean square (`recon_sweep_fine_analysis.py`, `CONTENT_FRACTION`).  Both jobs' trimmed curves
still include it, because the jobs compute the trimmed curve over all four slices.  The trim does not
always drop it either.  Take the Fourier job's normalized trimmed curve on the no-metal scan at a blur
of 2 voxels.  The two slices it drops at the first candidate are 1409 and 470, so the surviving pair
is the empty slice and slice 1691 (`tables.txt`, TRIMMED CURVES, normalized).  Where a number below
comes from a fit it covers the three slices that hold the object.

## What the bilinear kernel gave

### The per-slice minima are at the lattice angles

The four slices' score minima disagree with each other and agree with their own lattice angles.  The
table gives the job's per-slice minima at a blur of 2 voxels against the nearest lattice angle
(`analysis.txt`, PART 1, both scans).

| scan | slice | kernel row distance | minimum, deg | nearest lattice angle, deg | difference, deg |
| --- | --- | --- | --- | --- | --- |
| no metal | 188 | -734.1 | 0.1558 | 0.1561 | -0.0003 |
| no metal | 470 | -452.1 | 0.1321 | 0.1267 | +0.0054 |
| no metal | 1409 | +486.9 | 0.1182 | 0.1177 | +0.0006 |
| no metal | 1691 | +768.9 | 0.1477 | 0.1490 | -0.0014 |
| with metal | 188 | -734.1 | 0.1543 | 0.1561 | -0.0018 |
| with metal | 470 | -452.1 | 0.1329 | 0.1267 | +0.0062 |
| with metal | 1409 | +486.9 | 0.1194 | 0.1177 | +0.0017 |
| with metal | 1691 | +768.9 | 0.1497 | 0.1490 | +0.0007 |

The minima span 0.1182 to 0.1558 degrees, and the lattice angles of the same rows span 0.1177 to
0.1561 degrees.  The agreement is not a coincidence of a crowded lattice.  A slice's lattice spacing
is `1 / |i|` radians, which is 0.1267, 0.1177, and 0.0745 degrees on the three slices that hold the
object.  On the no-metal scan each of those slices therefore has exactly one lattice angle inside the
searched range (`analysis.txt`, PART 1).  Those three angles differ from each other by up to 0.031
degrees, and every minimum is within 0.007 degrees of its own.

The lattice is anchored on the detector's center row, 939.5, which is a property of the resampler.
The central-plane row is 956.90, so 939.5 has no meaning in the scanner's geometry.  An agreement with
a lattice anchored on such a row is itself evidence against a physical explanation.  The two scans
also give the same three minima on the slices that hold the object, to within 0.0020 degrees.  These
results indicate that what places a slice's minimum is a property of the slice's row and not of the
metal insert.

### The noise curves have minima at the same lattice angles

The half difference holds no object and no geometry, so any minimum in its score is the kernel's.  On
the three slices that hold the object, each noise curve's minimum is within 0.0058 degrees of that
slice's lattice angle.  That holds at every blur and on both scans.  The size of the modulation falls
with the blur.  The table gives the depth of the noise curve as a fraction of the magnitude of its
mean, on the same three slices (`analysis.txt`, PART 2).

| scan | slice | noise curve depth at blur 0 | at blur 2 |
| --- | --- | --- | --- |
| no metal | 470 | 0.971 | 0.089 |
| no metal | 1409 | 1.029 | 0.091 |
| no metal | 1691 | 0.770 | 0.057 |
| with metal | 470 | 0.950 | 0.084 |
| with metal | 1409 | 0.958 | 0.088 |
| with metal | 1691 | 0.736 | 0.057 |

Unblurred, the depth of the noise curve is 0.74 to 1.03 times the magnitude of its own mean.  At a
blur of 2 voxels it is 0.057 to 0.091 times that magnitude.  These results indicate two things.  The
first is that a blur reduces the modulation of the noise term and does not remove it.  The second is
that the kernel alone, acting on noise, produces a score minimum at a lattice angle.  No object
content is needed to produce one.

### Removing the noise does not move the minima

The structure curves put their minima in the same places the job's own curves did.  The table gives
the structure curve minima at a blur of 2 voxels against the nearest lattice angle (`analysis.txt`,
PART 2).

| scan | slice | structure minimum, deg | nearest lattice angle, deg | difference, deg |
| --- | --- | --- | --- | --- |
| no metal | 470 | 0.1337 | 0.1267 | +0.0069 |
| no metal | 1409 | 0.1191 | 0.1177 | +0.0014 |
| no metal | 1691 | 0.1466 | 0.1490 | -0.0024 |
| with metal | 470 | 0.1365 | 0.1267 | +0.0097 |
| with metal | 1409 | 0.1205 | 0.1177 | +0.0029 |
| with metal | 1691 | 0.1494 | 0.1490 | +0.0004 |

Subtracting the noise term moves each minimum by at most 0.0036 degrees from the job's own value.
The construction predicts that the subtraction leaves the minima where they were.  The half difference
holds noise only, so subtracting its gradient energy removes the noise term's share of the modulation.
The subtraction therefore cannot touch the structure term's own modulation.  These results indicate
that the resampling lowers the gradient energy of the object's structure as well as that of the
noise.

A blur does not separate the two either.  The structure term's modulation and the geometry signal both
act on the same high-frequency content of the edges, so a blur reduces both together and does not move
the minimum.  The evidence is that the structure minima stay near the lattice at every blur.  At blurs
of 1 to 4 voxels each structure minimum is within 0.0098 degrees of its own lattice angle
(`analysis.txt`, PART 2, both scans).  The one larger case is slice 470 of the metal scan unblurred,
at 0.0350 degrees.

### The offset-error model does not explain the disagreement

Fitting the offset-error model to the three slices that hold the object leaves residuals of up to
0.0153 degrees over the blurs the table below covers.  The table gives the fit on the structure curves
(`analysis.txt`, PART 2).  The residuals are listed for slices 470, 1409, and 1691 in that order.

| scan | blur | fitted rotation, deg | offset error, channels | residuals, deg |
| --- | --- | --- | --- | --- |
| no metal | 1 | 0.1331 | -0.016 | -0.0024, -0.0112, +0.0136 |
| no metal | 2 | 0.1336 | -0.021 | -0.0026, -0.0120, +0.0146 |
| with metal | 1 | 0.1357 | -0.026 | -0.0025, -0.0117, +0.0142 |
| with metal | 2 | 0.1360 | -0.025 | -0.0027, -0.0126, +0.0153 |

Three slices and two parameters leave one degree of freedom, so the residual is one number and not
three.  The comparison that matters is the residual against the term the model fits, and not against
the spread of the minima.  At a blur of 2 voxels on the no-metal scan the fitted `d / i_s` term spans
about 0.005 degrees over the three slices, and the residuals reach 0.0146 degrees.  The residuals are
also about seven times the 0.0020-degree repeatability of the per-slice minima, so the misfit is well
above the resolution of the data.

The same one constraint can also be stated as a prediction.  Least squares spreads a discrepancy
over three points, and the prediction form does not.  Two slices determine the two parameters
exactly, so solving on slices 470 and 1409 and predicting slice 1691 tests what the residual
tests.  At a blur of 2 voxels on the no-metal scan the pair gives a rotation of 0.1261 degrees and
an offset error of -0.060 channels.  That solution predicts 0.1217 degrees for slice 1691 where
0.1466 degrees was measured, a difference of 0.0250 degrees.  On the metal scan the same prediction
gives 0.1234 degrees where 0.1494 degrees was measured, a difference of 0.0261 degrees.  These
results indicate that the spread of the bilinear job's per-slice minima is not a readout of an
offset error.

### The trimmed curves

The trimmed curves are well formed and their minima are consistent between the scans.  At a blur of 2
voxels the raw trimmed curve has one interior minimum on each scan, at 0.1450 degrees without metal
and 0.1481 degrees with metal (`tables.txt`, CLOSING COMPARISON).  Those two differ by 0.0031 degrees.
The normalized trimmed curves at the same blur give 0.1321 and 0.1349 degrees, a difference of 0.0028
degrees.  The depth of each raw trimmed curve is 19 to 21 times the repeatability floor.

The trimmed curve therefore has a narrow and repeatable minimum.  The per-slice evidence above says
that the kernel places that minimum, and not the geometry.

## The kernel check

The kernel check compares the two kernels on synthetic images, with no reconstruction involved
(`recon_sweep_fine_fourier.py`, `check_kernel`).  It rotates a Gaussian blob in a 96-row block whose
rotation center lies 700 rows away.  It then compares each kernel's output with a reference blob, on
the rows away from the block's edges.  The blob's peak value is 1, so the errors below are fractions
of the peak.  The table below is read from `kernel_check.json`.  That file was produced by running
`check_kernel` on a Mac.

| rotation, deg | bilinear error | Fourier error | largest difference between the kernels |
| --- | --- | --- | --- |
| 0.05 | 9.75e-04 | 5.45e-05 | 9.87e-04 |
| 0.13 | 1.18e-03 | 1.37e-04 | 1.21e-03 |
| 0.24 | 1.20e-03 | 2.38e-04 | 1.24e-03 |

The reference is not an exact rotation.  `check_kernel` builds it from the first-order shear map,
which is the map the Fourier kernel itself implements.  The bilinear kernel implements a true
cosine-and-sine rotation instead.  The table therefore measures each kernel's interpolation error
against that shared first-order map.  These results indicate that the Fourier kernel interpolates
more accurately at these angles, by a factor of 5 at 0.24 degrees and 18 at 0.05 degrees.  What
differs between the kernels at these angles is how much they smooth.

The second-order difference is invisible to that test by construction, so it is bounded analytically
instead.  The two shears compose to the rotation matrix plus a second-order term.  That term displaces
a point by about `a * a / 2` times the point's distance from the center of the rotation.  At the
farthest scored row, 768.9 rows away, the displacement is under 0.01 pixels at 0.24 degrees and about
0.003 pixels at 0.15 degrees.  No measurement here bounds it.

A round trip was also measured.  A Fourier rotation of 0.002 radians followed by its reverse returns
the block to within 1.74e-04 of its original values.  Any invertible operator passes such a test, so
it shows that the kernel loses little and not that the kernel is correct.

The third check measures the smoothing directly.  A block of white noise of unit variance was rotated
at nine angles from 0.05 to 0.25 degrees, and the surviving noise power was recorded.  The Fourier
kernel left 0.9926 to 0.9961 of the power across the nine angles.  The bilinear kernel left 0.3238 to
0.5832 of the power.  The power the bilinear kernel leaves therefore varies by a factor of 1.80 across
the same angles.  These results indicate that the bilinear kernel removes between 42 and 68 percent of
the noise power, by an amount that changes from candidate to candidate.  The Fourier kernel removes
under one percent at every candidate.

The Fourier kernel treats the data as periodic, so it needs edge handling.  It applies the rotation
as two shears.  Each shear shifts whole rows, or whole columns, by a non-integer amount.  Both
shifts are done in the Fourier domain (`recon_sweep_fine_fourier.py`,
`fourier_rotation_kernel`).  Before the shift along the rows, the block is mirrored at its row edges
by 16 rows.  A further 16 rows are added to the band the kernel receives.  Those extra rows are
cropped after the shift, so the wrap and its ringing are not in the rows that are scored.  Along the
channels no taper is applied.  The two ends of every row are air, whose attenuation is near zero, so
the wrap joins two near-zero values.  The absence of a channel taper was not measured on the real
sinogram.

## What the Fourier kernel gave

### The noise curves are flat

The noise curves of the Fourier job have no minimum at the lattice angles.  On the three slices that
hold the object the depth of the noise curve is at most 0.005 of the magnitude of its mean
(`analysis.txt`, PART 2).  That holds at every blur and on both scans.  The bilinear job's figures
for the same slices are 0.736 to 1.029 unblurred and 0.057 to 0.091 at a blur of 2 voxels.  These
results indicate that the change of kernel removed the modulation the first job measured.

### The per-slice minima move off the lattice

With the Fourier kernel the per-slice minima are no longer at the lattice angles.  The table gives the
Fourier job's per-slice minima at a blur of 2 voxels against the nearest lattice angle
(`analysis.txt`, PART 1).

| scan | slice | kernel row distance | minimum, deg | nearest lattice angle, deg | difference, deg |
| --- | --- | --- | --- | --- | --- |
| no metal | 470 | -452.1 | 0.1738 | 0.1267 | +0.0471 |
| no metal | 1409 | +486.9 | 0.1270 | 0.1177 | +0.0093 |
| no metal | 1691 | +768.9 | 0.1373 | 0.1490 | -0.0117 |
| with metal | 470 | -452.1 | 0.1805 | 0.1267 | +0.0537 |
| with metal | 1409 | +486.9 | 0.1394 | 0.1177 | +0.0218 |
| with metal | 1691 | +768.9 | 0.1474 | 0.1490 | -0.0016 |

The minima are 0.002 to 0.054 degrees away from the lattice.  The bilinear job's were within 0.007
degrees of it.  The slice below the plane now has the highest minimum of the three.  In the bilinear
job the highest was the slice farthest above the plane.  These results indicate that the ordering of
the per-slice minima changed with the kernel.

### The offset-error model fits

The offset-error model fits the three per-slice minima.  The table gives the fit on the structure
curves (`analysis.txt`, PART 2).  The residuals are listed for slices 470, 1409, and 1691 in that
order.

| scan | blur | fitted rotation, deg | offset error, channels | residuals, deg |
| --- | --- | --- | --- | --- |
| no metal | 1 | 0.1509 | -0.201 | -0.0000, -0.0000, +0.0000 |
| no metal | 2 | 0.1507 | -0.192 | -0.0002, -0.0008, +0.0010 |
| with metal | 1 | 0.1599 | -0.207 | -0.0001, -0.0006, +0.0007 |
| with metal | 2 | 0.1595 | -0.168 | -0.0001, -0.0004, +0.0005 |

The fit has one degree of freedom here too, so the residual is one number and not three.  What makes
it convincing is the size of what the model explains.  At a blur of 2 voxels on the no-metal scan the
three minima span 0.0476 degrees and the residuals reach 0.0010 degrees.  The bilinear fit at the same
blur explained about 0.005 degrees and left 0.0146 degrees unexplained.  The Fourier residual is also
half the 0.0020-degree repeatability of the per-slice minima, so the fit cannot be told from perfect
at the resolution of these data.

The same fit was also run on the job's own per-slice minima, which still contain the noise term.  At a
blur of 2 voxels it gives 0.1501 degrees and -0.188 channels on the no-metal scan, and 0.1594 degrees
and -0.167 channels on the metal scan.  The noise term therefore moves the answer by less than 0.001
degrees here.

Stating the one constraint as a prediction gives the same conclusion without that spreading.  At a
blur of 2 voxels on the no-metal scan the pair of slices 470 and 1409 gives a rotation of 0.1502
degrees and an offset error of -0.195 channels.  That solution predicts 0.1357 degrees for slice
1691 where 0.1374 degrees was measured, a difference of 0.0018 degrees.  On the metal scan the same
prediction gives 0.1466 degrees where 0.1475 degrees was measured, a difference of 0.0009
degrees.  These results indicate that three slices at three heights are consistent with one rotation
and one offset error.

The second fitted parameter behaves as the model's offset error, and its magnitude is a fraction of a
channel on both scans.  One comparison is with the conjugate-view offset, estimated at the vendor's
rotation of 0.167 degrees.  That offset differs from the vendor's offset by -0.1498 channels on the
no-metal scan and by -0.0526 channels on the metal scan (`closed/real_scan_rotation_check.md`).  The
fitted offset error here is -0.192 channels on the no-metal scan and -0.168 channels on the metal
scan.  Only the magnitudes are being compared.  The two quantities are close on the no-metal scan and
differ by a factor of three on the metal scan.  Neither the sign convention nor the reference point has
been reconciled between them.

### The trimmed curves no longer locate the fitted rotation

The trimmed curves of the Fourier job do not report the rotation the model fit gives.  At a blur of 2
voxels the raw trimmed curve has one interior minimum on each scan, at 0.1438 degrees without metal
and 0.1544 degrees with metal (`tables.txt`, CLOSING COMPARISON).  The normalized trimmed curves at
the same blur give 0.1261 and 0.1288 degrees.  The model fit at that blur gives 0.1507 and 0.1595
degrees.

The reason is visible in the per-slice table above.  The three minima on the no-metal scan span 0.127
to 0.174 degrees.  The trimmed curve drops the highest and the lowest score at each candidate, and it
does not combine the slices through the model.  These results indicate that the trimmed curve is not
the right combination once the per-slice minima are spread by an offset error.

## The blur

The blur sets how much of the score is noise, and the noise share falls steeply with it.  The table
covers the no-metal scan of the Fourier job (`analysis.txt`).  For each blur it gives four quantities:
the noise share of the score, the fitted rotation, the largest residual of the model fit, and the half
width of the structure curve of slice 1409.  The noise share is given for each of the three slices, at
the vendor candidate.

| blur, voxels | noise share, slice 470 | slice 1409 | slice 1691 | fitted rotation, deg | largest residual, deg | half width, slice 1409, deg |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.952 | 0.957 | 0.941 | 0.1487 | 0.0034 | 0.0138 |
| 1 | 0.551 | 0.601 | 0.558 | 0.1509 | 0.0000 | 0.0292 |
| 2 | 0.125 | 0.146 | 0.145 | 0.1507 | 0.0010 | 0.0485 |
| 3 | 0.034 | 0.039 | 0.041 | 0.1496 | 0.0019 | 0.0665 |
| 4 | 0.012 | 0.014 | 0.015 | 0.1474 | 0.0031 | 0.0868 |

The metal scan behaves the same way.  Its noise shares at a blur of 2 voxels are 0.137, 0.142, and
0.187 on the three slices.  Its fitted rotations run 0.1502, 0.1599, 0.1595, 0.1581, and 0.1557
degrees over the five blurs.  Its largest residuals over the same five blurs are 0.0072, 0.0007,
0.0005, 0.0024, and 0.0030 degrees.

Three conclusions follow.  The model fits best at blurs of 1 and 2 voxels on both scans.  The half
width grows with the blur, so a larger blur locates the minimum less precisely.  The fitted rotation
moves by 0.0033 degrees on the no-metal scan and 0.0038 degrees on the metal scan between a blur of 2
and a blur of 4 voxels.  That change is a dependence on a setting.  It is not a limit on
repeatability.

A blur of 2 voxels is the recommendation.  At that blur the model residuals are at most 0.0010
degrees, the noise share is about a seventh, and the half width of slice 1409's structure curve is
0.0485 degrees.  A blur of 1 voxel fits the model equally well.  More than half of the score at that
blur is noise.  The even/odd split is what makes that share visible.  The three grounds for the
recommendation are the noise share, the residual, and the half width, so the recommendation does not
depend on any gate outcome below.

## The gates

The three gates of sub-increment 1.1 are stated in the plan.  The jobs compute them on the trimmed
curves and record them under the kind `gates`.  The table gives the raw trimmed curves at a blur of 2
voxels for both kernels (`tables.txt`, TRIMMED CURVES and CLOSING COMPARISON, both jobs).

| kernel | scan | trimmed minimum, deg | half width, deg | depth divided by the repeatability floor | interior minima | inside 0.15 to 0.24 | difference between the scans, deg | agree within the larger half width |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bilinear | no metal | 0.1450 | 0.0214 | 19.40 | 1 | not asked | 0.0031 | yes |
| bilinear | with metal | 0.1481 | 0.0193 | 21.26 | 1 | no | 0.0031 | yes |
| Fourier | no metal | 0.1438 | 0.0389 | 13.84 | 1 | not asked | 0.0106 | yes |
| Fourier | with metal | 0.1544 | 0.0441 | 16.75 | 1 | yes | 0.0106 | yes |

The depth gate and the agreement gate pass for both kernels at this blur.  The bracket gate fails for
the bilinear kernel and passes for the Fourier kernel.  The bilinear job's metal minimum is below 0.15
degrees at every blur.  It comes closest at a blur of 3 voxels, at 0.1482 degrees.  The Fourier job's
metal minimum is inside the bracket at blurs of 1, 2, 3, and 4 voxels.  Unblurred it is outside the
bracket, at 0.1459 degrees.

The bracket verdict above is read from the raw trimmed curve, and it depends on that choice.  The
Fourier job's own closing comparison marks the normalized trimmed curve as outside the bracket at a
blur of 2 voxels, at 0.1288 degrees (`tables.txt`, CLOSING COMPARISON, normalized).  It marks that
curve as outside the bracket at blurs of 0 and 3 voxels as well.  The bilinear job fails the bracket
gate on both forms of the curve, so only the Fourier pass turns on which form is read.

The unblurred curves are the weakest on the depth gate.  Both scans show two interior minima on the
bilinear job's raw trimmed curve unblurred.  The Fourier job's unblurred depths are 3.56 and 2.30
times the repeatability floor, against 13.84 and 16.75 at a blur of 2 voxels.

The findings page `plans/geometric_calibration/findings/increment_1_1_findings.md` states these gates
in the plan's words, gives the verdicts, and records the decision the sub-increment ends with.

## Limits of this evidence

Twelve limits apply.

1. One scanner and one phantom in two forms were measured.
2. Four slices were reconstructed per scan and one of the four holds no object, so every fit rests on
   three slices.
3. The reconstructions are direct, so the same cone-beam approximation is in all of them.  For the
   bilinear result the cone-beam approximation is not a candidate explanation.  The detector's center
   row has no role in the cone-beam geometry, and a cone-beam effect cannot vanish when only the
   kernel changes.  For the Fourier fit the cone-beam approximation stays open, and no measurement
   here separates a cone-beam effect from a geometry effect.
4. Any height dependence that is linear in the kernel row distance shifts the fitted rotation and
   leaves no residual.  This fit therefore cannot tell an in-plane rotation from such an effect.  The
   recorded out-of-plane lean of the rotation axis, 0.07885 degrees, is one such candidate
   (`closed/real_scan_band_height.md`).  A LEAP simulation of that lean moved the in-plane estimate by
   up to 0.0110 degrees (`leap_axis_tilt.md`).  That is comparable to the 0.0165-degree difference
   between the fitted rotation and the vendor's tilt.  This record does not decide the question.
5. The Fourier kernel applies the rotation as two shears.  That composition reproduces the rotation
   matrix only to first order in the angle.  The second-order error is bounded analytically in "The
   kernel check" above, and no measurement here bounds it.
6. The Fourier shift along a row treats the row as periodic, so it assumes the two ends of the row
   match.  That assumption held here.  The object is inside the field of view, and the per-view
   background correction leaves the row ends as zero-mean air.  On a scan whose object extends beyond
   the field of view the two ends differ.  The wrap then rings with an amplitude proportional to the
   mismatch times the sine of pi times the fractional shift.  That fractional shift depends on the
   candidate, so the ringing would depend on the candidate as well.  No such scan was measured here.
7. The Fourier kernel mirrors 16 rows at each row edge inside a band whose height varies with the
   slice and the candidate.  A row-dependent bias from that padding would be absorbed by the fitted
   `d / i_s` term, and nothing here separates the two.
8. The kernel check was run on a Mac, using the script in this directory.  Job 15971893's own `kernel`
   entry records only the kernel's name and its two row margins, so the check is not part of that
   job's record.
9. The sign of the fitted offset error was not reconciled with the sign convention of the
   conjugate-view estimator, so only magnitudes were compared.
10. No known rotation was imposed on these scans.  The claim above is therefore about what
    reconstruction quality prefers, and not about what the geometry is.
11. The fitted rotation depends on the blur by a few thousandths of a degree, and that dependence has
    no model here.
12. The candidate grid is 0.005 degrees, and the minima are located by a quadratic fit through five
    candidates rather than by reconstructing at the fitted value.

## The batch files

Three batch files were submitted from `/scratch/gautschi/buzzard/leap_cmp`.  This repository ignores
`.sbatch` files, so they are transcribed here.

`recon_sweep_fine.sbatch` requests `-A bouman -p ai -q normal -N 1 --gpus-per-node=2
--cpus-per-task=28 -t 02:00:00`.  It names the job `recon_sweep_fine` and writes the log
`/scratch/gautschi/buzzard/leap_cmp/recon_sweep_fine_%j.log`.  Its comment records why it asks for two
GPUs.  The `ai` partition refuses `--mem` and gives 126 GB of host memory per GPU.  A full-size NSI
scan is a 20 GB sinogram, and loading one needs several arrays of that size at once.  It sources
`~/load_conda_cuda.sh` and sets `set -e`.  It then exports six variables:
`TORCHINDUCTOR_CACHE_DIR=$PWD/torch_cache_sweep_fine`, `MPLBACKEND=Agg`, `MBIRTORCH_NUM_DEVICES=1`,
`PYTHONPATH=$PWD`, `REAL_SCAN_RESULTS=$PWD/results_recon_sweep_fine`, and
`REAL_SCAN_DATA=$PWD/data`.  It then runs the one-line interpreter check and
`venv/bin/python -u recon_sweep_fine.py`.

`recon_sweep_fine_fourier.sbatch` is the same file with three changes: the job name and log become
`recon_sweep_fine_fourier`, `REAL_SCAN_RESULTS` becomes `$PWD/results_recon_sweep_fine_fourier`, and
the last line runs `recon_sweep_fine_fourier.py`.  The compile cache directory is shared with the
first sweep, because the reduced models have the same shapes.

`recon_sweep_fine_analysis.sbatch` requests `-A bouman -p ai -q normal -N 1 --gpus-per-node=1
--cpus-per-task=14 -t 01:00:00`.  It names the job `recon_sweep_fine_analysis` and writes the log
`/scratch/gautschi/buzzard/leap_cmp/recon_sweep_fine_analysis_%j.log`.  It exports `MPLBACKEND=Agg`
and `PYTHONPATH=$PWD`.  It also checks three library versions, which the log records as numpy 2.4.6,
scipy 1.17.1, and matplotlib 3.11.1.  For each of the two results directories it exports
`RECON_SWEEP_FINE_RESULTS` and runs `recon_sweep_fine_tables.py` into `tables.txt` and
`recon_sweep_fine_analysis.py` into `analysis.txt`.  The work is numpy and scipy on the CPU, and the
GPU is requested only because the `ai` partition allocates host memory and CPUs per GPU.
