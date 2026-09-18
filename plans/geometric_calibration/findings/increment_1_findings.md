# Geometric calibration, Increment 1: the reduced problem, the parameter sweep, and the direction check

Date: 2026-09-04.  Status: implemented, awaiting Greg's review.  The code is on the
`geometric_calibration` branch of mbirtorch, branched from `prerelease` at commit `39a9eff`.  It
is staged and not committed.  The plan of record is `geometric_calibration_plan.md` in this
directory, and this page reports on its Increment 1.

## Outcome

Increment 1 is complete, and every gate the plan set for it passes.  The new module
`mbirtorch/preprocess/geometry_calibration.py` holds five public names: `CalibrationResult`,
`build_reduced_problem`, `parameter_sweep`, `apply_calibration`, and `check_rotation_direction`.
`mbirtorch/preprocess/__init__.py` exports that module.  A user can now sweep one of three
parameters: `det_channel_offset`, `det_row_offset`, or `det_rotation`.  The sweep returns one
reconstructed slice per candidate, and the slice viewer pages through them.  This is the manual
workflow the plan describes.

The automatic estimators are not yet written.  They are Increments 2 through 4, and the driver
that combines them is Increment 5.

Five decisions were made during the work without asking first, because this session could not
ask.  Each is listed in the next section with its alternative and the cost of each branch.  The
decision that most affects later increments is the choice of score used by
`check_rotation_direction`, which the plan did not specify.

## Decisions for Greg

**The score used by `check_rotation_direction`.**  The plan lists the function in Increment 1
and names no method.  The three scoring methods are Increments 2 through 4, so the check needed a
score that exists now, and one was written.  It is the mean squared difference between the
high-pass filtered sinogram and the high-pass filtered forward projection of a direct
reconstruction, divided by the mean square of the filtered sinogram, over the central half of the
detector rows.  Two measurements determined its form, and both are recorded in
`plans/experiments/features/geometric_calibration/direction_score_contrast.md`.  With a thin
slab the two directions scored almost the same.  The kept rows measure material outside the slab,
and the slab cannot explain that material, so that error term dominated both scores.  The check
therefore keeps the whole axial extent.  Scoring the central half of the rows then raised the
ratio between the two directions by a factor of 1.35 to 2.3.

Three choices are open.  The first keeps this score, and its cost is that it fixes a filter, a
row fraction, and a normalization that the residual method of Increment 4 will inherit or
contradict.  The second defers the check to Increment 2 and scores it with the conjugate-view
method, which compares data with data and so has no unexplained-material term.  Its cost is that
the check ships one increment later.  The third keeps the score and moves it behind the Increment
4 residual gate.  The score is a private function with one call site, so any of the three is a
small change.

**A margin below which the check warns.**  The check picks the lower of two scores.  It now warns
when the worse score is less than 1.5 times the better one.  That threshold is provisional.  The
smallest ratio measured was 2.20, on the test geometry, and the ratio falls with the fan angle, so
a narrow-fan industrial scan may fall below it.  The alternative is to refuse rather than warn,
and its cost is that a scan near the threshold gets no answer at all.

**Cone beam only for the direction check.**  For parallel beam, negating the view angles mirrors
the reconstruction and changes nothing else.  The two directions therefore fit the data equally
well, and the check refuses parallel beam with that reason.  For multiaxis parallel beam with
nonzero elevation the two directions do differ, because the row map depends on the sign of the
in-plane depth.  The check refuses multiaxis parallel beam and reports it as unsupported.  The
refusal is a limit of the current code, not a property of the geometry, and a later increment
could add it.  The alternative is to support it now, at the cost of a test geometry and a
measurement that this increment did not make.

**A row crop, a data reducer, and three new arguments.**  The plan sets a cone or multiaxis slab
through `recon_shape` and `recon_slice_offset`.  The builder also crops the detector rows to the
rows any ray through the slab can reach, and compensates `det_row_offset` for the crop by the rule
`apply_detector_crop` uses.  The crop makes the reduced sinogram small, and it removes most of the
measurements that come only from material outside the slab.  Its cost is a new mechanism, which
was the source of one of the two defects the code review found.  The alternative is to keep every
row and accept a larger reduced sinogram and a larger unexplained term.  The builder returns the
reduction record the plan describes, and the plan names no function that consumes it.
`reduce_sinogram` is that function.  It is public, so a caller can reduce a weights array or a
second sinogram the same way.  The alternative is to return the reduced sinogram from the builder,
at the cost of passing the sinogram into a function that otherwise only reads the model.  The
builder also takes `slice_index`, which the sweep needs, and `row_margin`, which a sweep of
`det_row_offset` needs, and it accepts `num_slab_slices=None` for the whole axial extent, which
the direction check needs.

**`det_row_offset` in the sweep.**  The plan defers an estimator for the row offset, because a
wrong row offset shifts the volume and the shifted volume explains the data almost as well.  A
row-offset sweep on a cone-beam scan therefore shows the object at a different height in each
candidate, and its sharpness changes only in proportion to the cone angle.  The sweep accepts the
parameter and its docstring says what the stack shows.  The alternative is to refuse the
parameter, at the cost of a viewing tool that LEAP also provides.

**The default slab has an even slice count.**  The plan's default of 8 slices puts no slice at the
slab's center, so the requested slice sits half a slice between two reduced slices.  The reduction
record carries `slice_in_slab`, the nearest one.  The sweep is unaffected, because it uses one
slice.  A default of 7 or 9 would make the center exact, at the cost of departing from the plan.
The plan's 8 was kept.

The implementation prompt asked that two decisions be confirmed with Greg at the start of the
session.  The first is the split of the parameter table into 17 user-facing and 12 research
parameters.  The second is the order of the two lines of work, calibration and the parameter
system.  This session could not ask, so the work follows the plan's order, with calibration first,
and both questions are put here.  The plan's own open question, whether `calibrate_geometry`
should change the model or stay read-only, is also still open.

## Limits of this evidence

Every measurement on this page comes from synthetic data.  The phantoms are Shepp-Logan volumes
forward projected by the same projector that then reconstructs and scores them, which is the
inverse crime the plan names.  The plan says the residual method is the one the inverse crime
flatters most, and the direction score is a residual score.

The sizes are small and the device is the CPU.  The test models have 32 views, 16 rows, and 32
channels, with torch.compile off on all but one 16 by 8 by 16 model.  The contrast experiment
reaches 256 views, 64 rows, and 128 channels.  Nothing ran on a GPU, on more than one device, or
on real data.  No wall time was measured for any user-facing function, and the plan's largest
stated uncertainty, first-time compilation, was not exercised.

Three effects the plan lists as risks were not exercised: lateral truncation, stripes, and beam
hardening.  The high-pass filter inside the direction score has fixed widths in pixels, so its
scores depend on the detector size and do not transfer between scans.

Two code paths were tested but not measured.  A multiaxis model at 10 degrees of elevation and a
curved-detector cone model each reproduce the full direct reconstruction through the sweep, which
checks their row windows.  Neither was used for a direction check or a contrast measurement.

## What was verified

`tests/test_geometry_calibration.py` checks the four gates of Increment 1.  It holds 23 test
cases.  It runs on the CPU in about 4 seconds once the compile cache is warm.  Every number in this
section was read from that file's output in this session.

1. `parameter_sweep` returns shape `(num_rows, num_cols, num_candidates)` in float32, and the
   viewer's data model `VolumeStack` pages through the last axis with its default `slice_axis`.
2. The reduced model's field of view in ALU equals the full model's at bin factors 1, 2, and 4,
   for parallel and cone beam.  The measured difference is zero at every factor.
3. `build_reduced_problem` raises in two cases: when the bin factor does not divide the detector
   counts, and when the view stride does not divide the view count.  The first case is tested at
   33 channels with a bin factor of 2.
4. `apply_calibration` is the only function that changes state.  The model's parameters and the
   sinogram are unchanged after three operations: the reduction, the sweep, and the direction
   check.  `apply_calibration` sets the offset, rotates the sinogram in place, and negates the
   view angles.

Two further checks test the slab construction.  The sweep candidate equal to the model's current
value reproduces the full model's direct reconstruction of the requested slice.  The relative
maximum error is 0.0 for parallel beam and between 1.4e-7 and 2.1e-7 for cone beam.  The check ran
for three parameters on cone beam and two on parallel beam, which refuses `det_row_offset`.  It
also ran for a slice away from the middle of the volume, for a multiaxis model, and for a curved
detector.  This result checks the row crop.  A missing row would change the reconstruction, so
agreement with the full model shows that the crop contains every detector row a ray through the
slab can reach.

The second check bins the detector by two and reconstructs an off-axis rod.  The centroid of the
reconstructed rod is within 0.06 reduced voxels of the full model's centroid.  This holds for
parallel and cone beam.  These results indicate that the detector offsets in ALU transfer to the
binned model unchanged.

The direction check was measured on synthetic cone-beam data.  The test geometry has 32 views, 16
rows, and 32 channels, at a half fan angle of 7.1 degrees.  On that geometry the wrong-direction
score is 2.20 times the correct-direction score.  The check returns the correct answer for a
sinogram simulated in either direction.  On the larger geometries of the contrast experiment the
lowest ratio in the scored column is 4.26, and 4.97 at the check's defaults.  These results
indicate that the contrast falls as the problem gets smaller.

Two existing test files call the one changed function in `preprocess/utilities.py`:
`tests/test_preprocess_entry_forms.py` and `tests/test_sharded_pipeline.py`.  Those two files and
`tests/test_preprocess_loaders.py` pass, 44 tests in all.  `tests/test_preprocess_utilities.py`
compares output against stored reference files, and the suite's default options deselect it.
The full suite was not run, because another session may be running it.

## How the reduced problem is built

The reduced problem is three things: the reduced model, the reduced sinogram, and the reduction
record that ties them together.  `build_reduced_problem` makes the model and the record, and
`reduce_sinogram` makes the sinogram from the record.

`build_reduced_problem` reduces the model in three steps, and each step keeps the geometry in ALU
unchanged.  It keeps every `view_stride`-th view.  It bins the detector by `bin_factor` in rows
and channels.  It multiplies the detector pitches by the same factor and recomputes the recon
geometry from the new pitches, so the result is the full model's field of view with coarser
voxels.  It then selects a slab of `num_slab_slices` slices around a chosen slice of the full
model.

The slab selection differs by geometry.  In parallel beam the slab is a band of detector rows,
because row r is slice r.  In cone beam and multiaxis parallel beam the slab is set through
`recon_shape` and `recon_slice_offset`.  The builder then crops the detector rows to the rows any
ray through the slab can reach.  It computes that set of rows from the projector's slice-to-row
map.  The crop does not apply to a helical scan.  A helical scan keeps every row and every slice,
and the plan requires that it is not thinned axially.

`reduce_sinogram` reads a block of kept views and kept rows at a time.  The full sinogram is
therefore never copied.  The result is a small new array.  When a detector rotation is requested,
`reduce_sinogram` rotates each block before the crop.  The rotation is about the full detector's
center.  The result therefore equals a crop of the rotated full sinogram, and a test checks that
equality for a crop at the edge of the detector under a rotation of 0.1 radians.

The sweep uses the builder at its cheapest setting for viewing.  It keeps every view at the full
channel resolution and reconstructs one slice.  The row crop is small for a slice near the center
of the volume and grows with the slice's distance from it, because the cone widens.

## Costs not yet measured

The direction check runs two direct reconstructions and two forward projections, one pair per
direction, each at one sixteenth of the sinogram's elements.  By the plan's benchmark numbers at
N = 1024, a direct reconstruction takes 5.0 s and a forward projection 8.6 s, so the check would
take about 1.7 s of projector work.  This spends the row reduction that the plan's cost model
counts on for the estimators, which the check does not use.  No wall time was measured.

The sweep's cost per candidate is one filter pass over the cropped rows and one back projection
into one slice, plus the cost of setting the parameter.  Setting a detector offset on the reduced
model rebuilds the model's projector functions, and the first changed value costs one
recompilation.  The runtime-offsets findings measured that later values cost none, because
`torch._dynamo.config.specialize_float` is false in torch 2.13.0.  That claim is inherited from
those findings and was not measured in this increment, whose tests run with compilation off.  The
filter pass could be moved out of the candidate loop for the two offsets, because the filter does
not depend on them.

## Consolidation after Greg's review

Greg's review of the first draft asked that the module not duplicate geometry arithmetic that the
model classes own.  Three changes followed.  The slice-to-z map and its inverse are now the model
methods `recon_slice_z` and `nearest_recon_slice`, and the three host-side sites in
`cone_beam.py` that wrote the same expression call the first of them.  The per-pixel magnification
range over the support is now `ConeBeamModel.pixel_magnification_bounds`, which both the axial
padding in `auto_set_recon_geometry` and the row crop use.  The batch staging helper the module
had written for itself is gone, and `pipeline._stage_batch` accepts a tensor instead.  The
affected cone, split-sinogram, and preprocessing tests pass.

## Smaller deviations from the plan

- `_rotation_kernel` in `preprocess/utilities.py` gained an optional `center` argument.  It
  defaults to the array's center.  The argument lets a crop of rows be rotated about the full
  detector's center.  `correct_det_rotation` is unchanged, and its tests pass.
- The reduced model inherits the full model's `compile_mode`.  The inheritance is explicit in
  `build_reduced_problem`, because `copy_ct_model` does not copy that setting.
- A caller-supplied `use_ror_mask` array has the full model's shape, so the reduced model uses the
  default mask instead.  The docstring says so.
- `parameter_sweep` accepts exactly three parameters.  A sweep over the source distances is not
  supported yet.  Changing a source distance changes the set of detector rows the slab reaches,
  and it changes the integer interpolation radii in the projector.
- A weights array reduced by `reduce_sinogram` holds the mean weight of each bin rather than the
  sum.  The factor is the same for every candidate.  It therefore does not change which candidate
  scores lowest, and the docstring of `reduce_sinogram` records this behavior.

## Independent code review

An Opus reviewer read the module against the projector code before the tests were final.  The
reviewer found one defect and one wrong index.  Both are fixed, and a test now covers each.  The
read margin for a rotated crop of rows ignored the crop's distance from the detector center.  A
crop far from the center could therefore read too few rows.  Where the rotation sampled beyond
those rows, the result was zero.  The sweep read the wrong slice from a helical reduction, whose
slice grid is recomputed from the reduced geometry.  The reviewer also confirmed five other items:
the row crop, the crop compensation, the binning arithmetic, the field of view, and the rebuild of
the projectors after the view angles are negated.  Each check cited the source file it used.

This page and its companion had a panel of three reviewers, for accuracy, reasoning, and style.
Their findings were applied in one pass, and the "Limits of this evidence" and "Costs not yet
measured" sections came from that review.

## Files staged for commit

In mbirtorch, on the `geometric_calibration` branch:

- `mbirtorch/preprocess/geometry_calibration.py`, new;
- `mbirtorch/preprocess/__init__.py`, one added import;
- `mbirtorch/preprocess/utilities.py`, the `center` argument of `_rotation_kernel`;
- `mbirtorch/preprocess/pipeline.py`, `_stage_batch` accepts a tensor;
- `mbirtorch/tomography_model.py`, the new methods `recon_slice_z` and `nearest_recon_slice`;
- `mbirtorch/cone_beam.py`, three sites use `recon_slice_z`, and the new
  `pixel_magnification_bounds` serves the axial padding and the row crop;
- `tests/test_geometry_calibration.py`, new.

In this repository: this page, and `direction_score_contrast.py` with its companion `.md` in
`plans/experiments/features/geometric_calibration/`.

## What is left

Increment 2, the conjugate-view method, is next in the plan's order.  Before it starts, Greg's
answers are needed on the decisions above, on the parameter-table split, and on the order of the
two lines of work.
