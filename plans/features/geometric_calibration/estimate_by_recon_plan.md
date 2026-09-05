# `estimate_geometry_from_recon`: implementation plan

Date: 2026-09-05.  Status: DRAFT v2, awaiting Greg's review.  Version 1 had a panel review of
three, for accuracy, reasoning, and style, and this version applies their findings.  Greg chose
the function name `estimate_geometry_from_recon` on 2026-09-05, and the two `estimate_by_recon`
file names keep the earlier short form.  The design this plan implements is
`estimate_by_recon.md` in this directory.  Measured numbers here were read in this session from
the record cited beside each.  Records cited by bare name live in
`plans/experiments/features/geometric_calibration/`, the closed campaign's under `closed/`
there; the pages of this directory's earlier campaign are under `closed/` here.  Day figures and
thresholds marked provisional are estimates.  Code citations refer to the
`geometric_calibration` branch working tree of mbirtorch, which is commit `4781600` plus the
staged Increment 6 edits.

## Summary

`estimate_geometry_from_recon` estimates `det_rotation`, and optionally `det_channel_offset`
with it, by reconstructing a few slices at candidate values and scoring their quality.  It is a standalone,
optional estimator beside the conjugate-view ones.  This plan is Increment 1 of `geometric_calibration_plan_v2.md`, so its five
sub-increments are numbered 1.1 to 1.5, each reviewed before the next starts:

- Sub-increment 1.1 runs the measurement the estimator's premise rests on, a fine sweep of
  reconstructed far slices on the real scans, and states what each outcome means for the rest of
  the plan.
- Sub-increment 1.2 adds the score and the slice chooser to the module, with synthetic tests, and
  measures the score's resolution.
- Sub-increment 1.3 adds the public function in its rotation-only mode, with the two-level search, the
  uncertainty fields, the undecided verdict, and the guard that keeps an undecided result out of
  `apply_calibration`.
- Sub-increment 1.4 adds the joint mode for the offset and the rotation.
- Sub-increment 1.5 adds the streak and hardening measurements and the documentation.

The estimator accepts every scan `parameter_sweep` accepts, including helical and multiaxis
scans, which the conjugate-view method refuses.  Two refusals carry over from `parameter_sweep`:
`det_rotation` on a curved detector, and, until `recon_direct` has short-scan redundancy
weighting, a short scan, with the message pattern the module already uses
(`mbirtorch/preprocess/geometry_calibration.py:828-835`).  The weighting is separate work,
estimated at a day.

## Units and terms

Lengths are in ALU, the package's arbitrary length units.  An offset is given in channels, which
are detector pixels along the channel axis.  A rotation is given in degrees in prose and stored
in radians, as the module stores it.  The edge displacement of a rotation is the distance it
moves the edge pixel of the detector, in pixels.  The central plane is the plane through the
source that is perpendicular to the rotation axis, and a slice's height is its distance from
that plane in detector rows.  A short scan covers 180 degrees plus the fan angle.  A dense
feature is a region whose attenuation sits far above the object's typical values, such as metal,
and streaks are the line artifacts dense features and beam hardening draw in a reconstruction.

The two real scans are the NSI artifact phantom without and with a metal insert, 1800 views
each, loaded with the vendor's recorded values held out (`real_scan_validation.md`).  The band
estimator is `estimate_det_rotation`, which compares a band of detector rows with the mirrored
opposite views.  The band estimator's zero point is the angle it returns on the data as given,
the argmin of its score over candidate angles.  The tall-band answer, 0.130 degrees, is that
estimator's value when its band is widened to hundreds of rows (`real_scan_band_reach.md`).  The
far-slice job is the earlier experiment that reconstructed slices far from the central plane at
four fixed rotations (`real_scan_rotation_recon.md`).

Six terms name parts of this estimator.  The chooser is `_choose_scoring_slices`, the helper of
sub-increment 1.2 that picks the scoring slices.  The scout is the one reduced whole-extent direct
reconstruction the chooser reads, shared with the coarse level.  The coarse level is the search
over the caller's bounds at reduced resolution, and the fine level is the search at full
resolution around the coarse minimum; a coarse step is the coarse grid's spacing, the bounds
width divided by one less than `num_coarse`.  The rotation kernel is `_rotation_kernel`, the
bilinear resampler the corrections use.  The search lattice is the fixed set of points a
deterministic search can visit, set by the bounds, the coarse count, and the golden-section rule
before any data are read (`real_scan_rotation_check.md`).

Four terms describe a score curve.  The trimmed curve is the per-candidate mean over the scored
slices after the highest and the lowest slice scores are dropped; at four slices it is the mean
of the middle two.  A curve's depth is its largest score minus its smallest over the searched
range.  The half width of a minimum is the distance at which a quadratic fit near it rises two
percent above the fitted minimum.  The repeatability floor is the largest absolute difference,
over the candidates, between the curve scored on the even-index views and the curve scored on
the odd-index views.

## How the score reads the geometry

One mechanism drives the design, and three choices below follow from it.  At a slice `i` rows
from the central plane, a residual offset error `d` and a rotation error `a` displace the slice's
center of rotation together, by about `d + a * i` channels.  Here `d` is in channels, `a` in
radians, and `i` in rows, where `i` is the detector row the slice projects to at the rotation
axis, so the voxel pitch and the voxel aspect ratios do not enter.  The product is in channels
when `delta_det_row` equals `delta_det_channel`, and it is scaled by
`delta_det_row / delta_det_channel` when they differ.  Each scored slice therefore has its
score minimum at the true rotation minus `d / i`, not at the true rotation.  Three consequences:

- The chooser pairs scoring slices symmetrically about the central plane where the object
  allows, because the `d / i` term cancels to first order between a slice at `+i` and one at
  `-i`.  The far-slice job's own slices, at 470 and 752 rows on both sides, are such pairs
  (`real_scan_rotation_recon.md`).
- The spread of the per-slice minima is not noise.  It is a readout of the supplied offset's
  error, and the result records it as such.
- The joint mode is two nearly separate estimates, the offset from central slices and the
  rotation from far ones, alternated for two rounds.  The rounds are a cost bound, not a
  convergence argument.

A rotation-free object is the mirror case.  The score needs structure that changes along the
axis at the scoring slices; the conjugate band estimator instead needs in-plane structure across
its band (`rotation_zero_point_synthetic.md`).  A cylinder that is the same in every slice
therefore gives this score nothing even where the band estimator succeeds, and the undecided
gate of sub-increment 1.3 is built on exactly that object.

## The API

One public function joins `mbirtorch/preprocess/geometry_calibration.py`:

```python
def estimate_geometry_from_recon(ct_model, sino, parameters='det_rotation', *,
                                 bounds=None, num_coarse=11, num_slices=4,
                                 slice_indices=None, blur=None, det_channel_offset=None,
                                 filter_name='ramp', view_stride=4,
                                 bin_factor=2) -> dict[str, CalibrationResult]
```

The return value is a dictionary mapping each parameter name to a `CalibrationResult` whose
`method` is `'recon'`, and the docstring shows `results['det_rotation'].value`.  `parameters`
accepts one name or a sequence of names.  `bounds` and `num_coarse` accept one value or a
per-parameter dictionary, because the two parameters carry different units.  `det_channel_offset`
supplies a known offset in rotation-only mode, defaulting to the model's value, as
`estimate_det_rotation` does.  `slice_indices` overrides the chooser.  `view_stride` and
`bin_factor` set the coarse level, and the stride actually used is the largest divisor of the
view count not exceeding the request, because `build_reduced_problem` requires a divisor
(`geometry_calibration.py:277`; the `bga` scan's 2401 views have no divisor between 2 and 4,
`real_scan_validation.md`).

The stored score is the negated blurred gradient energy.  Gradient energy is largest at the best
candidate, `_search_minimum` minimizes, and `CalibrationResult.score` documents lower as better,
so the negation is stated once here and the words below say minimum throughout.  Every
candidate, including zero, is routed through the rotation kernel, so no candidate is scored on
unresampled data; `reduce_sinogram` today skips the kernel at exactly zero
(`geometry_calibration.py:426`), and the estimator must not take that path.

`blur` is a width in ALU.  Its default is twice the full model's `delta_voxel`, raised at the
coarse level to twice that level's voxel pitch, because a blur below the voxel pitch of the
level being scored is no blur at all and the coarse level would then read the interpolation.  The rotation and the offset
get separate blurs: the rotation's as above, and the offset's smaller or zero, because offset
candidates are applied through `set_params` with no resampling to defeat, and the doubled-edge
signature the offset score reads is at the pixel scale.

An undecided answer sets `value` to NaN, states the reason in `reduction`, and warns.
`apply_calibration` gains a refusal of any non-finite value, in sub-increment 1.3, because today a NaN
rotation would be written over the sinogram in place by `_rotate_views_in_place`
(`geometry_calibration.py:707`) and a NaN offset would be set on the model.  The undecided rules
ship complete in sub-increment 1.3: the per-slice minima disagree beyond what the `d / i` term above
can explain, the two search levels disagree by more than one coarse step, or the chooser finds
no usable slice.  Nothing here changes state, and `apply_calibration` stays the one function that
does.  Like every entry point of the module, the function refuses a sinogram already divided
across devices and runs on the host.

The slice scoring stays in torch, so a later differentiable version changes the search and not
the score.  The reused reduction and search helpers are numpy today, and this plan does not
change them.

## The increments

**Sub-increment 1.1.  The fine sweep on the real scans.  Rough estimate 1 day.**  This sub-increment adds
one cluster experiment, `plans/experiments/features/geometric_calibration/recon_sweep_fine.py`,
with its record.  The script reconstructs the four far slices of the far-slice job, at 470 and
752 rows on both sides of the central plane, for the no-metal and metal NSI scans.  The
candidate rotations run from 0.10 to 0.20 degrees on the no-metal scan and 0.10 to 0.24 on the
metal scan, in 0.005-degree steps, with the two named candidates, 0.130 and 0.1672 degrees,
added to the grid explicitly, because a 0.005-degree grid from 0.10 does not contain 0.1672
(`real_scan_band_reach.md`, `closed/real_scan_followup.md`).  Each slice is scored at four blur widths spanning one to four
voxels, which costs only rescoring, so the blur default is chosen from data.  A quadratic fit
near each trimmed curve's minimum gives its location and its half width, and the even-odd view
split gives the repeatability floor, all three as defined under "Units and terms".  It has
three gates.  Each scan's trimmed curve has one minimum whose depth exceeds the
repeatability floor.  The metal scan's minimum lies inside 0.15 to 0.24 degrees, the bracket its
earlier measurements set, and a minimum at either end of the searched range fails this gate
and widens the range (`real_scan_rotation_recon.md`, `real_scan_rotation_check.md`).  The
no-metal and metal minima agree within the larger half width, which is the streak test, because
the two scans share one geometry and differ by the insert (`real_scan_band_height.md`).

The sub-increment ends with a decision, and the plan states the outcomes now.  A no-metal minimum
near 0.167 degrees confirms the vendor's value and the estimator proceeds as the remedy.  A
minimum near 0.130 says reconstruction quality itself prefers the tall-band answer; the
estimator still proceeds, because it is then the tool that measured the truth, and the docs
stop naming the vendor's value as right.  A curve whose depth fails the repeatability gate stops
the plan for review, because the metric's premise failed on the decisive scan.  One
deliverable follows the decision: the four staged texts that today assert the vendor's 0.167
degrees as right, in `docs/source/usr_preprocess.rst` and the module's docstrings and warning,
are revised to match the measured outcome before any commit.

**Sub-increment 1.2.  The score and the slice chooser.  Rough estimate 1 day.**  This sub-increment adds
two private helpers to `mbirtorch/preprocess/geometry_calibration.py`.  `_recon_slice_score`
reconstructs one slice directly, blurs it at the ALU width, and returns the negated normalized
gradient energy.  `_choose_scoring_slices` picks slices from one reduced whole-extent
reconstruction, which is also the coarse level's model, so the two share one build.  It picks by
content that changes along the axis for the rotation and by in-plane content for the offset, it
prefers symmetric pairs about the central plane, it avoids slices holding dense features, and it
takes the central-plane row from the module's existing arithmetic
(`geometry_calibration.py:980`).  The synthetic script gains a third phantom, a cylinder with no
slab, which does not exist there today.  Tests run on the far-slab, near-slab, and no-slab
phantoms of `plans/experiments/features/geometric_calibration/rotation_zero_point_synthetic.py`.
It has three gates.  The score ranks the injected 1.5 degrees first on the far-slab phantom,
where the band estimator under-reads by 24 percent (`rotation_zero_point_synthetic.md`).  The
ranking is unchanged at a detector binning of 2 with the recon grid held fixed, so only the blur
is under test.  The chooser, run at the coarse binning the estimator will use, returns slices
inside the slab on the slabbed phantoms and reports no usable slice on the no-slab one.  The
sub-increment also measures the score's half width on the far-slab phantom, which sets the synthetic
threshold of sub-increment 1.3.

**Sub-increment 1.3.  The rotation-only mode.  Rough estimate 2 days.**  This sub-increment adds the
public function in its rotation-only mode, the two-level search, the three uncertainty fields,
the undecided verdict, and the non-finite refusal in `apply_calibration`, with tests in
`tests/test_geometry_calibration.py`.  The coarse level searches the caller's bounds at the
reduced resolution; the fine level searches one coarse step around the coarse minimum at full
resolution on the cropped rows, with the fine tolerance no finer than the half width sub-increment 1.1
measured, so the search does not report points of its own lattice
(`real_scan_rotation_check.md`).  It has six gates.  On the far-slab phantom the estimate is
within the threshold sub-increment 1.2 measured, provisionally 0.05 degrees, of the injected 1.5
degrees.  On the no-slab cylinder the function returns undecided, and `apply_calibration`
refuses the undecided result with an error.  On the no-metal scan, a cluster job with two GPUs
recovers the sub-increment 1.1 minimum within its half width.  On the same job the coarse level's
answer is within one coarse step of the fine answer; if this gate fails, the coarse level
reduces views only and the job reruns.  On the same scan the six-setting search control, three
bounds by two coarse counts, moves the estimate by less than the half width, which is the
lattice test the conjugate estimator needed (`real_scan_rotation_check.md`).  A rotation added
to the real sinogram in place with `_rotate_views_in_place` moves the estimate by minus the
added angle to within the half width, which ties the sign to what `apply_calibration` applies
without allocating a second sinogram.

**Sub-increment 1.4.  The joint mode.  Rough estimate 1 day.**  This sub-increment adds the joint mode,
which alternates the offset estimate on central slices and the rotation estimate on far slices
for two rounds.  Offset candidates are applied through `set_params`, and the offset score uses
its own blur.  It has three gates.  A synthetic joint perturbation of 1.3 channels and 1.5
degrees, the sizes the existing synthetics use (`conjugate_offset_recovery.md`,
`rotation_zero_point_synthetic.md`), is recovered to 0.5 channels and the sub-increment 1.2 rotation
threshold.  On the `bga` scan the joint offset is within 0.5 channels of the conjugate
estimator's 0.546 (`real_scan_validation.md`), a threshold that record's sweep supports, since
its sharpness separated candidates half a channel apart and not closer.  The per-slice spread on
a synthetic case with a deliberately wrong supplied offset reads back that offset's error
through the `d / i` conversion to within a factor of two.  A deliverable beside the gates: the
docstring states each estimator's resolution, a few hundredths of a channel for the conjugate
offset and about half a channel for this mode, so the joint mode is documented as the fallback.

**Sub-increment 1.5.  Hardening and documentation.  Rough estimate 1 day.**  This sub-increment runs the
hardening measurement and writes the documentation.  The cluster job extends the validation
harness's quadratic hardening from its 64-row band to the whole sinogram, sized the same way,
to change the largest value by ten percent (`real_scan_validation.md`), and reruns the
sub-increment 1.1 scoring on the hardened no-metal scan.  It has three gates.  The hardened minimum
agrees with the unhardened one within the half width.  The documentation builds with no
warnings.  No existing module test is modified.  The deliverables beside the gates: an
`estimate_geometry_from_recon` entry in the calibration section of
`docs/source/usr_preprocess.rst` with the use cases relative to the existing estimators, and one
sentence stating how the future driver calls the rotation mode when a reader supplies no vendor
tilt, which is what decision 4 of `closed/status_2026-09-05.md` waits on.

## Expected run time

Every figure here is an estimate anchored on cited measurements.  One slice at four candidates
took 3.4 to 19.5 seconds at full resolution on the H100 (`real_scan_rotation_recon.md`).  A fine
level of about 24 evaluations over four slices, the count `_search_minimum` spends at its
defaults (`real_scan_validation.md`), is therefore about two to eight minutes per parameter.
One coarse evaluation is 16 to 64 times cheaper.  The scout adds one reduced whole-extent
direct reconstruction, shared with the chooser.  The real-scan jobs request two GPUs for the
252 GB of host memory, as every job on these scans has.  The tall-band alternative measured 21
minutes for one estimate at 501 rows and 43 minutes with a 139 GB peak at 1001
(`real_scan_band_reach.md`); this estimator's whole search is cheaper than either single
estimate.  A CPU run is expected to be minutes at the coarse level and slow at the fine level,
and no measurement backs a tighter statement.

The five sub-increments total six days against the design's two to three.  The difference is
sub-increment 1.1, which the design did not budget, and the controls the panel review added.

## Validation beyond the gates

Two cross-checks run as experiments rather than tests.  The LEAP harness supplies in-plane
tilted sinograms from a projector that shares no code with mbirtorch (`leap_axis_tilt.py`), and
running sub-increment 1.3's estimator on them checks the pipeline against independent data; its
record's caution applies, because its phantoms are rotationally symmetric and axially sparse, so
only the slab slices can be scored (`leap_axis_tilt.md`).  The out-of-plane lean of the rotation
axis is already bounded for this estimator the way it was for the band estimator: the lean did
not move an in-plane estimate on synthetic data, and the metal scan carries the same lean and
measures well (`leap_axis_tilt.md`, `real_scan_band_height.md`).

## Out of scope

Three things stay outside this plan.  Short-scan redundancy weighting for `recon_direct` is its
own small feature, and both modes refuse short scans until it exists.  A `det_rotation` inside
the projectors, the change the calibration plan defers, would remove the resampling from every
candidate and is the path to the fully differentiable and the iterative versions.  The
reprojection-residual check the design listed is dropped from the undecided rules: on a slab it
carries the unexplained-material term the module already documents
(`geometry_calibration.py:281`), and on real data its minimum was 1.5 percent deep
(`real_scan_followup.md`); the whole-extent scout could host such a check later if a use
appears.

## Risks

- The score's landscape on real data has an unmeasured roughness.  The qualifier matters
  because the polish is local.  The coarse grid brackets the minimum first, and the blur
  smooths the landscape.
- Streaks from the insert could move a minimum.  The sub-increment 1.1 gate that compares the two
  scans' minima measures exactly that.
- An object without axial structure gives the score nothing.  The no-slab gate makes the answer
  undecided rather than wrong, and the chooser's no-usable-slice path reports the reason.
- An object with structure on one side only defeats the symmetric pairing.  The result records
  the one-sidedness, and the per-slice spread then carries the offset-error term the mechanism
  section states.
- Very small rotations sit below one pixel of edge displacement, where resampling bias lives.
  A new warning states it for this estimator; only the resampling clause of the band
  estimator's warning applies here.
- The search lattice can masquerade as precision on a flat curve.  The six-setting control of
  sub-increment 1.3 is the gate for it.
