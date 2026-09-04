# Geometric calibration utilities

Date: 2026-09-03.  Status: DRAFT v3, revised 2026-09-03 after Greg's review of v2, awaiting
approval.

Every `mbirtorch` file:line citation below is against the working tree at commit `39a9eff`.  That
tree was clean when this plan was written.  Citations that begin with `LEAP` are against LEAP
v1.26 at commit `0c8846f42b2e59340d5559fc1271d590a292f9a0`, which is the version the LEAP
comparison pins (`../leap_comparison/leap_comparison.md:49`).  Supporting scripts go in
`plans/experiments/features/geometric_calibration/`, which follows the layout rule in
`../../README.md`.

## Executive summary

mbirtorch should estimate its own scan geometry from the sinogram, and today it cannot.  The
package's own FAQ tells a user with a blurry reconstruction to change `det_channel_offset` by hand
(`docs/source/demos_and_faqs.rst:100-103`).  A second FAQ entry gives the same instruction to a
user with center rings (`:147-149`).  The LEAP comparison names this the widest capability gap
between the two packages (`../leap_comparison/leap_comparison.md:177`).

The work adds one new module, `mbirtorch/preprocess/geometry_calibration.py`, which runs after a
reader's `get_sino_and_model` and before reconstruction.  Each estimator takes a sinogram and a
model, then scores a set of candidate values.  It returns the best value, the candidate values,
and the score at each one.  No estimator changes state.

Users gain three capabilities:

- A scan whose vendor metadata is wrong or missing can be calibrated from its own data.
- A user who prefers to judge by eye gets a stack of reconstructed slices, one per candidate
  value, which the existing slice viewer displays.
- A user can check the rotation direction, which the FAQ names as a common failure
  (`docs/source/demos_and_faqs.rst:142-145`).

The first release is deliberately narrow.  A user with a blurry reconstruction or center rings
runs one function, gets a value for `det_channel_offset`, and gets a slice stack that shows why.
That release is `parameter_sweep`, `estimate_det_channel_offset` by the conjugate-view or
derivative-filter method, and the two rewritten FAQ answers.  Everything else here is a later
release.

The work has eight increments, each reviewed before the next starts.  Increments 1 through 6 are
the calibration strand, and increments 7 and 8 are a separate parameter-system strand that the
calibration work does not depend on.

1. Increment 1 creates the calibration module, the reduced-problem builder, and `parameter_sweep`.
2. Increment 2 adds the conjugate-view method, which compares each view with its opposite view.
3. Increment 3 adds the derivative-filter method and the coarse-to-fine search.
4. Increment 4 adds the residual method, which is the geometry-generic fallback.
5. Increment 5 adds the coordinate-descent driver `calibrate_geometry`.
6. Increment 6 updates the documentation, gives the Zeiss readers a `det_rotation` argument, and
   adds a demo.
7. Increment 7 makes the two detector offsets call-time tensor inputs and sets `recompile_flag` to
   false for both.
8. Increment 8 applies the parameter-table review, adds a `ParamDict` that restores editor
   support for `set_params`, and adds the name-coordination test.

## Motivation

LEAP solves for scan geometry from the projection data, and mbirtorch does not.  LEAP provides
three estimators: `find_centerCol` for the center of rotation, `find_tau` for the lateral source
offset, and `estimate_tilt` for detector rotation about the optical axis.  Four more utilities
build on those three, including a joint two-parameter search and a parameter sweep that
reconstructs one slice per candidate. The LEAP comparison lists nine LEAP calibration and
measurement utilities against two mbirtorch ones
(`../leap_comparison/leap_comparison.md:162-177`), and the LEAP inventory records what each one
computes (`../leap_comparison/leap_comparison_sources/leap_inventory.md:316-327`).

LEAP also supplies the score function those sweeps need.  `inconsistencyReconstruction` replaces
the ramp filter with a derivative, and LEAP's docstring states that its output is pure noise when
the geometry is calibrated and the scan covers 360 degrees or more
(`../leap_comparison/leap_comparison_sources/leap_inventory.md:191`).  Among LEAP's calibration
utilities, this is the only one that changes a filter rather than running a search.  The direct
reconstruction filter in mbirtorch accepts one name today (`mbirtorch/tomography_utils.py:19-21`),
so adding a second name is a small change.

## What exists today

mbirtorch already has the routines a calibration search needs.  It has no such search.  The
relevant routines are these:

- `estimate_sino_view_offset` forward projects a caller-supplied reconstruction, high-pass filters
  both arrays with `sino_high_pass_filtering` (`mbirtorch/preprocess/utilities.py:1123-1166`), and
  estimates a per-view shift with OpenCV's `findTransformECC` (`:1072-1120`).
- `align_sino_views` applies those per-view shifts with bilinear interpolation (`:1205-1228`,
  kernel at `:1169-1203`).
- `correct_det_rotation` rotates every view in the detector plane by a given angle (`:265-292`,
  kernel at `:210-262`).
- `copy_ct_model` builds a model of the same class with different view parameters or a different
  detector size (`mbirtorch/utilities.py:1001-1101`).
- `slice_viewer` displays one or more three-dimensional arrays with synchronized navigation
  (`mbirtorch/viewer.py:2353-2355`).

A calibration would change these geometry parameters.  The two detector offsets are
`det_channel_offset` and `det_row_offset` (`mbirtorch/_utils.py:76-77`, documented at
`docs/source/usr_parameters.rst:113-128`), both measured in arbitrary length units (ALU) and both
defaulting to zero.  The cone geometry adds `source_detector_dist`, `source_iso_dist`, and
`recon_slice_offset` (`mbirtorch/cone_beam.py:334-338`), and derives magnification from the two
distances (`:499-505`).

Detector tilt is the physical rotation of the detector about the optical axis, and `det_rotation`
is the quantity that describes it. `det_rotation` is not a model parameter; it exists only as a
preprocessing correction applied to the sinogram before the model is built. The NSI reader
computes it from the scanner's own basis vectors (`mbirtorch/preprocess/nsi.py:616-631`) and
applies it inside the fused scan-to-sinogram pass (`:130-136`).  Both Zeiss readers pass a
hard-coded zero instead (`mbirtorch/preprocess/zeiss.py:120` and
`mbirtorch/preprocess/zeiss_tct.py:108`).  `zeiss.py` also has no vendor value for the row offset
and sets it to zero (`:283`), while its channel offset comes from the vendor's `center_shift`
(`:282`).

Each detector offset appears in exactly one arithmetic expression per compiled projector body,
applied to data.  The seven expressions are at `mbirtorch/parallel_beam.py:44`,
`mbirtorch/cone_beam.py:101` and `:156`, `mbirtorch/multiaxis_parallel.py:49` and `:80`, and
`mbirtorch/translation_model.py:78` and `:96`.  The integer tap radii that set the unrolled loop
lengths come from the detector pitches, the voxel pitches, and the source distances, never from an
offset (`mbirtorch/parallel_beam.py:172-180` and `mbirtorch/cone_beam.py:542-566`).

## Design

The design has six parts: where the code lives, how a user runs a calibration, how a candidate is
scored, how a search is run, what the search runs on, and how long a calibration is expected to
take.

### Placement and API

The new module `mbirtorch/preprocess/geometry_calibration.py` is exported the way the other
preprocessing modules are exported (`mbirtorch/preprocess/__init__.py:1-8`).  Every function takes
the sinogram and the model, and works through the model's own `forward_project`, `back_project`,
and `recon_direct`, so one implementation serves every geometry that supplies those three methods.

```python
CalibrationResult = NamedTuple(
    parameter: str, value: float, score: float,   # the parameter name, the estimate, its score
    candidates: np.ndarray, scores: np.ndarray,   # (num_candidates,) values scored and their scores
    method: str,                                  # 'conjugate', 'derivative', or 'residual'
    reduction: dict)                              # the reduced problem and the frozen recon settings

def estimate_det_channel_offset(ct_model, sino, *, method='auto', bounds=None,
                                num_coarse=11, reduction=None) -> CalibrationResult
def estimate_det_rotation(ct_model, sino, *, method='auto', bounds=None,
                          num_coarse=11, reduction=None) -> CalibrationResult
def calibrate_geometry(ct_model, sino, parameters=('det_channel_offset', 'det_rotation'),
                       rounds=2, **kwargs) -> dict[str, CalibrationResult]
def check_rotation_direction(ct_model, sino, **kwargs) -> CalibrationResult
def conjugate_difference(ct_model, sino, *, value=None) -> np.ndarray
def apply_calibration(ct_model, sino, results) -> tuple[TomographyModel, np.ndarray]
def parameter_sweep(ct_model, sino, parameter, values, *, slice_index=None,
                    filter_name='ramp') -> np.ndarray
def build_reduced_problem(ct_model, *, view_stride=4, bin_factor=2,
                          num_slab_slices=8) -> tuple[TomographyModel, dict]
```

`apply_calibration` is the only function that changes state.  `det_channel_offset` is a model
parameter, and `apply_calibration` sets it on the model.  `det_rotation` is not a model parameter,
so `apply_calibration` applies it by rotating the sinogram.  It returns the modified model and the
rotated sinogram.

`parameter_sweep` reconstructs one slice per candidate value and returns the slice stack, of shape
`(num_rows, num_cols, num_candidates)`. The candidate index is last because `slice_viewer`'s
`slice_axis` defaults to the last axis (`mbirtorch/viewer.py:2377-2378`), so a user who opens the
stack with the default pages through candidates.  `conjugate_difference` returns the difference
image behind a conjugate-view score, so a user can look at what the number summarizes.

### Workflow

Two workflows are supported, an automatic one and a manual one.  In the automatic workflow the
user calls three functions between the reader and the reconstruction:

```python
from mbirtorch.preprocess import geometry_calibration, nsi

sino, ct_model = nsi.get_sino_and_model(dataset_dir)
results = geometry_calibration.calibrate_geometry(ct_model, sino)
ct_model, sino = geometry_calibration.apply_calibration(ct_model, sino, results)
recon, recon_dict = ct_model.recon(sino)
```

`calibrate_geometry` is the grouping convenience function of that workflow.  By default it
estimates two parameters, `det_channel_offset` and then `det_rotation`.  It returns a dictionary
that maps each parameter name to its `CalibrationResult`, so the caller can read the chosen value,
the candidates, and the scores before applying anything.  The individual estimators stay public,
so a script that wants one parameter calls `estimate_det_channel_offset` directly.

In the manual workflow the user reconstructs one slice per candidate, looks at the stack, and sets
the value by hand:

```python
import numpy as np
import mbirtorch
from mbirtorch.preprocess import geometry_calibration

values = np.linspace(-4.0, 4.0, 17)
slices = geometry_calibration.parameter_sweep(ct_model, sino, 'det_channel_offset', values)
mbirtorch.slice_viewer(slices, title='det_channel_offset sweep')
ct_model.set_params(det_channel_offset=1.0)   # the value chosen from the viewer
recon, recon_dict = ct_model.recon(sino)
```

The manual workflow replaces the hand adjustment the FAQ describes today.  Two FAQ answers
tell a user to change `det_channel_offset` by hand (`docs/source/demos_and_faqs.rst:100-103` and
`:147-149`).  `parameter_sweep` runs that search in one call and shows the result in one viewer
window.

### Scoring

Three scoring methods are proposed, in the order the increments build them, which is also the
order of increasing generality.  **The conjugate-view method** compares each view with its
opposite view, and for parallel beam it is exact.  A voxel at rotated coordinate `x` projects to
channel `n = (x + d) / delta + c` (`mbirtorch/parallel_beam.py:44`).  Here `d` is
`det_channel_offset` and `c` is the center channel.  Rotating by pi negates `x` (`:38-40`), and
mirroring the channel axis about `c` then maps the same voxel to `c + (x - d) / delta`.  That
differs from `n` by exactly `2 * d / delta` channels, so half of the measured shift between a view
and its mirrored opposite is `det_channel_offset` in channel units.  The same comparison also
estimates the detector tilt, which is the rotation angle that best aligns a view with its mirrored
opposite.

For cone beam the mirror relation is wrong in two ways.  Rotating by pi negates the rotated
coordinate `y` as well as `x`, and the flat-detector channel coordinate `u = pixel_mag * x`
depends on `y` through `pixel_mag` (`mbirtorch/cone_beam.py:75` and `:93`).  A band of rows around
the center row removes that cone-angle error.  The larger error is the fan angle.  The central
plane of a circular cone-beam scan is a fan beam, and there the conjugate of the ray at view angle
`beta` and fan angle `gamma` lies at `beta + pi + 2 * gamma`.  Mirroring the view at `beta + pi`
therefore pairs each channel with the wrong ray, by an amount that grows with the distance from
the center channel, which is why LEAP uses fan-specific cost functions
(`../leap_comparison/leap_comparison_sources/leap_inventory.md:318`).  The method is therefore
restricted to a central band in rows and in channels, it needs the source distances from the
reader, and its docstring states that the estimate degrades with fan angle.  If its 20-degree gate
fails, the derivative-filter method becomes the cone-beam initializer.

**The derivative-filter method** is a direct reconstruction with the ramp filter replaced by a
derivative.  In mbirtorch this is one added name in `generate_direct_recon_filter`, whose
supported list currently holds only `"ramp"` (`mbirtorch/tomography_utils.py:19-21`).  The score
is an image-domain energy or variance measure on the resulting slice.  It is built before the
residual method for one reason: it performs no fit, so nothing in it can absorb a geometry error.
Its cost is one filter pass and one back projection per candidate.

**The residual method** reconstructs a reduced problem per candidate and scores the weighted
sinogram residual of its forward projection. It reuses the two steps `estimate_sino_view_offset`
already performs, a forward projection of a reconstruction and `sino_high_pass_filtering` applied
to both arrays (`mbirtorch/preprocess/utilities.py:1101-1105`).  The score is the weighted sum of
squared differences between the two filtered arrays.  It works for every geometry and every
angular coverage, so it is the documented fallback for a scan without the symmetry the other two
methods need.

The residual method needs four settings fixed, or its numbers are not comparable across
candidates.  `recon` stops on a relative-change rule, and `auto_regularize_flag` defaults to true
(`mbirtorch/_utils.py:97`), so both the iteration count and the regularization would otherwise
vary per candidate.  The sweep runs a fixed `max_iterations` with the relative-change stop
disabled, sets `auto_regularize_flag` to false, freezes the regularization parameters at values
derived once from the unperturbed model, and uses one fixed weights array. Those four settings are
recorded in `CalibrationResult.reduction`.

A reconstruction absorbs part of a geometry error, which is the residual method's weakness.  Run
few iterations and the score reports the prior; run many and the reconstruction fits the wrong
geometry with a data-consistent but artifact-laden image.  A thin slab lowers the contrast
further, because rays through it also pass through material the reduced reconstruction does not
represent, which adds a large term that is nearly the same for every candidate.  The Increment 4
gate is therefore a contrast requirement as well as a single-minimum requirement.

The high-pass filter makes the score insensitive to cupping and to scatter, because both are
smooth across the detector.  It does not remove beam-hardening streaks between dense features,
which are edge-correlated.  A trimmed mean over views limits the effect of corrupted views, and
the measured sinogram is filtered once outside the candidate loop.

### Candidate evaluation

When the candidate is a model parameter, the evaluation sets it on the model.  Three reasons make
this the default.  The strongest is that the measured data must not change across candidates:
resampling smooths the sinogram by an amount that grows with the shift, which lowers a residual or
an image-energy score monotonically in the shift.  The second is geometry generality, because the
offset enters the same expression in the flat and curved cone branches
(`mbirtorch/cone_beam.py:88-101`).  The third is that after Increment 7, setting either detector
offset causes no Dynamo retrace.

Resampling the sinogram is the alternative, and for a flat detector it is exact.  For
`det_rotation` it is the only option, because the model has no tilt parameter.  Two costs come
with it.  `correct_det_rotation` and `_translate_views_bilinear` both interpolate bilinearly
(`mbirtorch/preprocess/utilities.py:210-262` and `:1169-1203`), so a search resamples the sinogram
once and applying the estimate resamples it again.  `correct_det_rotation` also zeroes any output
pixel whose sample fell outside the view (`:259-262`), so a rotation score is computed over a mask
that excludes the corners any candidate zeroes.  For an NSI scan the better fix is to re-run the
reader with the corrected tilt, because the reader applies the tilt inside `scan_to_sino`
(`mbirtorch/preprocess/nsi.py:130-136`).  Both Zeiss readers need a `det_rotation` argument before
their users can do the same, which Increment 6 adds.

LEAP puts the tilt inside the projector instead, which is the route mbirtorch does not take.  LEAP
rotates the detector coordinates inside its cone-beam separable-footprint kernels, and it does so
per pixel.  The forward kernel rotates the row and column coordinates of the detector pixel it is
filling (`LEAP src/projectors_SF.cu:1981-1998`).  The back projector applies the same rotation to
the detector coordinate that a voxel projects to (`LEAP src/projectors_SF.cu:1868-1889`).  LEAP
accepts a tilt only for flat-panel cone beam (`LEAP src/parameters.cpp:690-693`), and it caps the
angle at five degrees (`LEAP src/parameters.cpp:1764-1766`).  mbirtorch instead resamples the
sinogram with `correct_det_rotation`.  Both routes approximate the same rotation of the detector
plane.  The resampling route costs one interpolation of the data and leaves the projectors
unchanged.

### Search and the reduced problem

Each parameter is searched with one coarse pass followed by a bracketed polish.  The coarse pass
evaluates `num_coarse` values across the starting bounds, eleven values over plus or minus four
channels by default.  It establishes that the score curve has a single minimum, and it gives the
user the curve to look at.  A golden-section or parabolic polish then refines the bracket around
the coarse minimum. Together they reach a resolution near one hundredth of a channel in roughly
fifteen evaluations, against 63 for three fixed rounds of 21 candidates.

The evaluation count matters only for the residual method, which reconstructs per candidate.  One
parameter costs about fifteen reduced reconstructions, and two parameters over two
coordinate-descent rounds cost about sixty.  Each round searches one parameter and holds the
others fixed.  A joint grid of five by five over `det_channel_offset` and `det_rotation` follows
those rounds, because the two are correlated and coordinate descent stalls in a diagonal valley.
The reduced model is built once and reused, so the shape-dependent compiles are paid once.

The reduced problem is what makes a per-candidate reconstruction fast enough to run, and it has
three parts.  The first keeps every fourth view, built with
`copy_ct_model(ct_model, new_angles=...)` (`mbirtorch/utilities.py:1001-1002`).  The stride must
divide the view count, so that a 360-degree scan keeps its conjugate pairs.  A helical cone scan
must pass `new_helical_z_shifts` as well, or `copy_ct_model` raises an exception (`:1071-1074`).

The second part bins the detector by a factor of two to four.  `copy_ct_model` changes the
detector size through `new_num_det_rows` and `new_num_det_cols` but does not rescale the pitches
(`:1088-1091`), and it drops `recon_shape` so that the automatic pass recomputes it
(`:1095-1097`).  The builder therefore sets `delta_det_channel` and `delta_det_row` for the bin
factor and then calls `auto_set_recon_geometry`, so that `delta_voxel` and `recon_shape` follow
the new pitches.  Without that call the reduced model reconstructs a fraction of the field of
view, and lateral truncation is exactly the condition that biases a `det_channel_offset` estimate
(`mbirtorch/tomography_model.py:2128-2132`).  The offsets are in ALU rather than in pixels
(`docs/source/usr_parameters.rst:113-128`), so a binned estimate transfers to the full-resolution
model unchanged.  That holds only when the bin factor divides the channel and row counts exactly,
so the builder refuses a factor that does not.  A dropped leftover channel moves the binned
detector center by half a bin, which at four times binning is two channels of bias.

The third part keeps a thin central slab of slices.  For cone and multi-axis parallel geometries
the volume is centered on `recon_slice_offset` (`mbirtorch/cone_beam.py:152-157` and
`mbirtorch/multiaxis_parallel.py:78-80`), so a small `recon_shape[2]` and a chosen
`recon_slice_offset` select any slab.  For parallel beam `verify_valid_params` requires
`recon_shape[2]` to equal `sinogram_shape[1]` (`mbirtorch/parallel_beam.py:228-232`), and detector
row `r` is recon slice `r`, so a parallel-beam slab is a sinogram row crop.  A helical scan is not
thinned axially at all, because every ray through a slab comes from a different axial position.

The builder pins the reduced model to one device with
`configure_devices(devices=[ct_model.torch_device])`.  Without that pin the copy inherits
automatic device selection (`mbirtorch/utilities.py:1098-1100`) and runs its own widening search
with a memory preflight (`mbirtorch/tomography_model.py:1140-1147`).  The pin also makes the
scores reproducible, and the device list is recorded in `CalibrationResult.reduction`.

### Expected run time

Every figure in this subsection is an estimate, derived from two measured records rather than from
a calibration run.  The benchmark record supplies two single-GPU times at N = 1024 on one H100: a
forward projection of 8.636 s and a direct reconstruction of 4.980 s
(`../../experiments/features/leap_comparison/results/leap_benchmark_results.md:101` and `:103`).
The quality record supplies the reconstruction those two times are measured against.  With its
default stop rule at N = 1024, mbirtorch stopped after 12 of 100 iterations and took 214.006 s of
iteration time (`../leap_comparison/quality_results.md:468`).

The reduced problem removes about a factor of 64 from the sinogram work, at the four-times end of
the binning range the builder supports.  It keeps every fourth view, bins the detector by four in
each direction, and reconstructs a slab of about one sixteenth of the slices.  The view stride and
the channel binning alone remove a factor of 16, and the row reduction removes at least another
factor of 4.  Dividing the two N = 1024 times by 64 puts a reduced direct reconstruction at about
0.08 s and a reduced forward projection at about 0.13 s.

The estimates for the four calibration steps follow from those two reduced times:

- One derivative-filter evaluation is one filter pass and one back projection, so it costs a few
  tenths of a second.
- The conjugate-view method reconstructs nothing.  It compares view pairs over the reduced
  sinogram, which costs seconds.
- The derivative-filter search takes about fifteen evaluations, so it costs under ten seconds.
- The residual method reconstructs once per candidate.  One reduced iteration costs about 0.28 s,
  which is 214.006 s divided by 12 iterations and then by 64.  A fixed budget of three to five
  iterations over fifteen candidates is then ten to twenty seconds.

The full driver is estimated at under a minute.  It runs two coordinate-descent rounds over two
parameters and then a five by five joint grid, which is about 85 evaluations.  At a few tenths of
a second per evaluation those 85 evaluations take under a minute, against the 214.006 s
reconstruction quoted above.  A run that falls back to the residual method for every evaluation
would take one to two minutes instead.

First-time compilation is the largest uncertainty in these estimates.  A cold process compiles the
reduced shapes before it can score anything, and that compilation can cost more than the
calibration itself.  The quality study timed one direct reconstruction at N = 256 in three
conditions: 10.694 s as the first GPU work in its process, 3.087 s in a second process, and
0.028 s warm (`../leap_comparison/quality_results.md:225-227`).  The mitigation is that
`build_reduced_problem` uses fixed reduced shapes, so the inductor cache holds them from the
second calibration onward.

### Ordering, scope, and deferred work

Calibration runs after the corrections that change where signal lands on the detector.  The
required order is this: defective-pixel interpolation, background offset correction, stripe
removal, calibration, then `align_sino_views`.  Stripe removal comes first because a gain stripe
sits at a fixed channel, `sino_high_pass_filtering` keeps narrow features by construction, and the
mirror in the conjugate-view method maps a stripe onto a different channel.  A calibration run on
data whose stripes have not been removed is expected to be biased.  `align_sino_views` comes last
because it shifts each view independently (`mbirtorch/preprocess/utilities.py:1228`).  A
systematic `det_channel_offset` error appears as a systematic per-view shift, so running
`align_sino_views` first would remove part of the error the calibration is meant to estimate.
Within the calibration the parameter order is `det_channel_offset`, then `det_rotation`.

Three cases are out of scope, and each is refused with a message rather than estimated badly.  A
helical scan cannot use the conjugate-view method, because it has no opposite view at the same
axial position.  `det_rotation` is refused on a curved-detector model, because
`correct_det_rotation` rotates the flat detector plane and the channel coordinate on a curved
detector is an arc parameter (`mbirtorch/cone_beam.py:96-99`).  Setting `det_channel_offset` stays
correct there, because the offset enters the same expression (`:101`).  A ball-phantom fit is out
of scope, because it needs a dedicated calibration scan that these users do not have.

A laterally displaced flat detector is not a separate case.  On a flat detector that displacement
is exactly a change of `det_channel_offset`, which is the parameter the first release estimates.
The large-displacement version of it is the offset scan, in which the displacement is large enough
that a view and its opposite view overlap only in part.  The estimators handle that case by
restricting the comparison to the channels where the two views do overlap.  An offset scan also
needs a direct-reconstruction weighting, and that weighting is a separate feature.  It is item 2
of the LEAP comparison (`../leap_comparison/leap_comparison.md:544-555`), and it is out of scope
here.

`estimate_det_row_offset` is deferred to a second release, because the residual score is nearly
flat in `det_row_offset`.  A reconstruction made under a wrong row offset is a vertically shifted
volume, and a shifted volume explains the data almost as well as the correct one does.  The
curvature that is left in the score is proportional to the cone angle, so it goes to zero as the
geometry approaches parallel beam.  The NSI reader sets `recon_slice_offset` to minus
`det_row_offset` divided by the magnification (`mbirtorch/preprocess/nsi.py:423`).  That identity
is evidence that the two parameters are interchangeable, and it is not the cause of the flatness.
A row-offset estimator needs a constraint from outside the sinogram.

`estimate_magnification` is deferred for two reasons.  First, `delta_voxel` is derived from the
magnification in `auto_set_recon_geometry` (`mbirtorch/cone_beam.py:596`), so a magnification
error and a voxel-pitch error produce similar reconstructions.  Second, changing `source_iso_dist`
changes the magnification (`:499-505`), the magnification sets the integer tap radii
(`:551-559`), and a change to a tap radius changes the traced graph, so a magnification search
cannot avoid a retrace.

LEAP's lateral source offset `tau` is `det_channel_offset` to first order.  The LEAP comparison
records `tau` as having no mbirtorch counterpart (`../leap_comparison/leap_comparison.md:167`),
and the paragraphs below replace that entry with the relation between the two parameters.
LEAP's manual calls `tau` the horizontal translation of the rotation stage
(`LEAP documentation/LEAP.tex:175`).  Its fan-beam transform puts the source at
`R * theta - tau * theta_perp` and sets the detector coordinate to
`u = (x . theta_perp + tau) / (R - x . theta)` (`LEAP documentation/LEAP.tex:431-435`).  Here `R`
is the source-to-object distance, `theta` is the unit vector from the rotation axis toward the
source, and `theta_perp` is `theta` turned by ninety degrees.

The two packages describe the same scan from different reference lines.  In LEAP the detector is
perpendicular to the line from the source to the detector center, and `centerCol` is the column at
the foot of that perpendicular.  `tau` then displaces the rotation axis sideways from that line.
In mbirtorch the single parameter `det_channel_offset` is the detector center's offset from the
source-to-axis line, and mbirtorch takes that line to be perpendicular to the detector.  The two
descriptions differ by a rotation of the detector about the vertical axis, which is a yaw of angle
`arctan(tau / sod)`.

To first order a LEAP `tau` is an mbirtorch `det_channel_offset` of `tau * sdd / sod`, plus a
constant shift of every view angle.  Here `sod` is the source-to-object distance, which mbirtorch
calls `source_iso_dist`, and `sdd` is the source-to-detector distance, which mbirtorch calls
`source_detector_dist`.  What is left after that first-order match is the yaw.  The yaw displaces
a ray that meets the detector at distance `u` from the center by about `u^2 * yaw / sdd`.  One
worked example fixes the four quantities: `tau` is 10 mm, `sod` is 1000 mm, `sdd` is 2000 mm, and
`u` is 100 mm.  The yaw is then 0.01 radians and the displacement is about 0.05 mm, which is half
of a 0.1 mm pixel.  At a `tau` of 1 mm the same calculation gives 0.005 mm.

`estimate_det_channel_offset` is therefore the mbirtorch form of LEAP's `find_tau`, to first
order, and the residual yaw is second order.  The NSI reader already makes that first-order
conversion.  It folds a lateral source offset into `det_channel_offset` by multiplying that offset
by the magnification (`mbirtorch/preprocess/nsi.py:701-702`).

One resampling can remove any detector pose error, and it generalizes the tilt correction above.
Two flat detector planes that share one source position are related by a homography, because a
pixel of one plane and the matching pixel of the other lie on the same ray through the source.
Any detector pose error is therefore removable by one projective resampling per view.  That covers
in-plane tilt, yaw, and pitch, and it covers the two detector offsets as well.  The resampling
carries the measured data onto the ideal detector, which is the one perpendicular to the
source-to-axis line, so no projector needs to change.  The resampling generalizes
`correct_det_rotation`, which is the in-plane case of the same map.  A source position error
cannot be removed this way.  Moving the source moves the center of projection, and no map of the
detector plane can undo that.

## Parameter system

### The runtime-offsets change

This strand makes the two detector offsets call-time inputs, reviews what `recompile_flag` means
for every parameter in the table, and replaces a header comment that is not true.  The prototype
for the first part is recorded in `../runtime_offsets/runtime_offsets_findings.md`, with the patch
in the same directory as `runtime_offsets.diff`.  The change has three edits.  `recompile_flag` is
set to false for both offsets (`mbirtorch/_utils.py:76-77`).  The parallel and cone geometries
pass their offsets to the compiled bodies as zero-dimensional float32 tensors on the calling
device, built by a new `TomographyModel._runtime_scalar`.  Each of the two driver loops passes the
device that holds the view parameters it is about to use (`mbirtorch/projectors.py:561` and
`:618`), so every device in a sharded run builds its own scalar.

The measured benefit is modest.  `torch._dynamo.config.specialize_float` is false in this torch,
so only the first changed value retraces and every later value is free
(`../runtime_offsets/runtime_offsets_findings.md:126-146`).  The prototype measured that one-time
cost at 2140.5 ms for parallel beam and 7266.6 ms for cone beam on a cold inductor cache (`:263`
and `:274`).  A calibration sweep therefore saves about 2.1 s or 7.3 s once per process, and
nothing after that.  The motivation is correct flag semantics, an accurate header comment, and
simplicity, as much as speed.

The memo key inside `_runtime_scalar` must include the parameter name.  Both cone offsets default
to zero, so a key of value and device alone returns the same tensor object for both arguments.
Dynamo then installs an aliasing guard, and that guard fails as soon as one offset changes
(`:240-249`).

The prototype covered parallel beam and cone beam only.  In the prototype the multi-axis parallel
and translation geometries accept a new `device` argument and ignore it, so they still pass Python
floats.  In the current tree neither `_view_batch_args` takes a `device` argument at all
(`mbirtorch/multiaxis_parallel.py:304` and `mbirtorch/translation_model.py:287`).  Increment 7
gives both geometries the tensor form.

`recon_slice_offset` is not part of Increment 7.  The prototype did not attempt it (`:416`),
nothing here searches over it, and it is a geometry-supplied parameter rather than a table entry.
Every geometry-supplied parameter is created with `recompile_flag` set to true
(`mbirtorch/tomography_model.py:214-216`), so a geometry would first need a way to set that field
itself.

### Review of the parameter table

The table holds 29 parameters (`mbirtorch/_utils.py:70-108`), and nine of them have
`recompile_flag` set to true today.  One rule governs whether a parameter needs it:
`recompile_flag` may be set to false when every value derived from the parameter is either
recomputed per call or served from a cache that compares its key.  All three caches in the package
compare their keys (`mbirtorch/tomography_model.py:866-884` and `:885-901`,
`mbirtorch/cone_beam.py:446-450`), which is why the two detector offsets are safe and why
`use_ror_mask` never needed the flag.

Each parameter falls into one of four classes.  A shape or layout changer keeps the flag, because
a stale `Placement` is silently wrong on a sharded run.  A parameter in the "can drop" class
reaches only data, and clearing the flag also removes a Dynamo retrace.  A parameter in the "keeps
the flag" class sets an integer tap radius; clearing its flag would be safe under the rule above,
but a radius change produces a different traced graph and must retrace anyway.  For every
remaining parameter the flag is irrelevant, because the parameter reaches no compiled body.

| Parameter | `recompile_flag` today | Class | Reason |
| --- | --- | --- | --- |
| `geometry_type` | False | irrelevant | names the model class |
| `file_format` | False | irrelevant | records the persistence format version at construction |
| `sinogram_shape` | True | shape or layout | sets `sino_placement.axis_len` (`tomography_model.py:911-912`) |
| `delta_det_channel` | True | keeps the flag | sets the integer tap radii (`parallel_beam.py:176-180`, `cone_beam.py:542-566`) |
| `delta_det_row` | True | keeps the flag | sets the cone integer tap radii (`cone_beam.py:542-566`) |
| `det_row_offset` | True | can drop | appears in one data expression per geometry (`cone_beam.py:156`) |
| `det_channel_offset` | True | can drop | appears in one data expression per geometry (`parallel_beam.py:44`) |
| `sigma_y` | False | irrelevant | weights the forward model, read per recon |
| `alu_unit` | False | irrelevant | labels the length unit, reaches no body |
| `alu_value` | False | irrelevant | scales the length unit, reaches no body |
| `recon_shape` | True | shape or layout | sets `recon_placement.axis_len` (`tomography_model.py:913-914`) |
| `delta_voxel` | True | keeps the flag | sets the integer tap radii (`cone_beam.py:542-566`) |
| `voxel_row_aspect` | True | keeps the flag | sets the tap radii through the row pitch (`parallel_beam.py:178-180`) |
| `voxel_slice_aspect` | True | keeps the flag | sets the cone tap radii through the slice pitch (`cone_beam.py:548-565`) |
| `sigma_x` | False | irrelevant | scales the prior, read per recon |
| `sigma_prox` | False | irrelevant | scales the proximal map, read per recon |
| `p` | False | irrelevant | sets a prior exponent, read per recon |
| `q` | False | irrelevant | sets a prior exponent, read per recon |
| `T` | False | irrelevant | sets the prior threshold, read per recon |
| `qggmrf_nbr_wts` | False | irrelevant | weights the prior neighbours, read per recon |
| `auto_regularize_flag` | False | irrelevant | selects the automatic regularization pass |
| `positivity_flag` | False | irrelevant | clamps the update, read per recon |
| `snr_db` | False | irrelevant | sets `sigma_y` in the automatic pass |
| `sharpness` | False | irrelevant | sets `sigma_x` and `sigma_prox` in the automatic pass |
| `granularity` | False | irrelevant | sizes the partitions, read per recon |
| `partition_sequence` | False | irrelevant | orders the partitions, read per recon |
| `verbose` | False | irrelevant | sets the logging level |
| `max_alpha` | False | irrelevant | caps the step size, read per recon |
| `use_ror_mask` | False | irrelevant | keys its two derived caches, so no rebuild is needed (`tomography_model.py:866-884`, `:885-901`) |

The four classes hold these counts: two shape or layout changers, two that can drop the flag, five
that keep it with nothing to gain, and 20 for which the flag is irrelevant.  Increment 7 changes
exactly the two table parameters in the "can drop" class.

Three geometry-supplied parameters need the same review.  The view parameters, held under the name
in `view_params_name`, must keep the flag, because the projectors read them at build time and
pre-place one copy per device (`mbirtorch/projectors.py:438-444`). `source_detector_dist` and
`source_iso_dist` keep it for the tap-radius reason (`mbirtorch/cone_beam.py:551-559`).
`recon_slice_offset` can drop it, and its damping-profile cache compares its key (`:446-450`), so
no separate invalidation is needed.

### Coordinating the parameter names

mbirtorch has no `ParamNames` type, so a question about keeping `ParamNames` and the parameter
table in agreement has no subject in this package.  The module docstring of `parameter_handler.py`
records the absence, listing the YAML save and load and "the ParamNames Literal typing machinery"
as not implemented (`mbirtorch/parameter_handler.py:5-7`).  A repository-wide search finds the
name only in that sentence.  mbirjax has the type and a test that keeps it current
(`mbirjax/parameter_handler.py:18` and `mbirjax tests/test_utilities.py:21-27`), and neither is
ported here.

What `ParamNames` bought mbirjax was editor support, and the port lost it.  `set_params` takes
`**kwargs` (`mbirtorch/parameter_handler.py:260`), so an editor has no list of names to offer as
completions and no types to check the values against.  The mbirjax `ParamNames` Literal gave an
IDE both.  A user of mbirtorch today gets neither, and a misspelled name is caught only at run
time, by the `ValueError` that lists the valid names (`:306-311`).

The replacement is a typed dictionary of the user-facing parameters, unpacked into the keyword
arguments:

```python
class ParamDict(TypedDict, total=False):
    sharpness: float                        # the 17 user-facing parameters, and only those
    snr_db: float
    sigma_y: float
    sigma_prox: float
    positivity_flag: bool
    verbose: int
    recon_shape: tuple[int, int, int]
    delta_voxel: float
    voxel_row_aspect: float
    voxel_slice_aspect: float
    delta_det_channel: float
    delta_det_row: float
    det_channel_offset: float
    det_row_offset: float
    use_ror_mask: bool | np.ndarray
    alu_unit: str
    alu_value: float

def set_params(self, no_warning=False, no_compile=False, **kwargs: Unpack[ParamDict]):
```

That form is PEP 692, and `total=False` makes every key optional.  `typing.Unpack` is available
from Python 3.11 onward, and mbirtorch already requires Python 3.11 or later
(`pyproject.toml:23`).  Type checkers read the annotation, and PyCharm reads it as well, so a user
gets name completion and a type check on each value.  Nothing changes at run time, because
`Unpack` is an annotation and `set_params` still receives a plain `**kwargs`.

The rest of the table stays usable and stops being advertised.  A research parameter is still
accepted by `set_params`, because the unknown-name check reads the parameter table rather than
`ParamDict` (`mbirtorch/parameter_handler.py:304-311`).  It is not a key of `ParamDict`, so an
editor does not offer it, and it is not documented in `docs/source/usr_parameters.rst`.

The proposed split follows the user documentation as it stands today, with two moves, and it is
for Greg to confirm.  `docs/source/usr_parameters.rst` documents 19 of the 29 table parameters,
and 17 of those 19 become the user-facing set listed above.  The two moves are `max_alpha` and
`qggmrf_nbr_wts`, which are documented today and are proposed as research parameters.  `max_alpha`
limits the step size of a VCD update, and `qggmrf_nbr_wts` sets the prior's weights along the row,
column, and slice directions.  Neither describes the scan, and neither is how a user asks for a
sharper or a softer image.  The other 12 parameters are the research set: `geometry_type`,
`file_format`, `sinogram_shape`, `sigma_x`, `p`, `q`, `T`, `qggmrf_nbr_wts`,
`auto_regularize_flag`, `granularity`, `partition_sequence`, and `max_alpha`.  Three of those 12
are internal bookkeeping rather than research knobs, because `geometry_type`, `file_format`, and
`sinogram_shape` are set by the model itself.

The proposed test asserts three statements.  Every key of `ParamDict` is a key of
`recon_defaults_dict`.  Every parameter name hard-coded inside `parameter_handler.py` is a key of
`recon_defaults_dict`.  The set of parameter headings in `docs/source/usr_parameters.rst` equals
the set of `ParamDict` keys.

Each of the three assertions catches a different drift.  The first catches a `ParamDict` key that
the table no longer holds.  The second catches a rename that misses a hard-coded list.  Two such
lists exist, `["sigma_y", "sigma_x", "sigma_prox"]` at `mbirtorch/parameter_handler.py:319` and
`["sharpness", "snr_db"]` at `:321`, and a rename in the table would leave either one naming a
parameter that no longer exists, with nothing failing.  The third is an equality rather than a
containment, so it catches a user-facing parameter that gained a `ParamDict` key with no
documentation section, and one that gained a section with no key.  `all_param_keys`
(`mbirtorch/_utils.py:118`) is read nowhere in the package, so this test is its one job.

The third assertion does not hold today, and Increment 8 has to make it hold.  The documentation
still gives `max_alpha` and `qggmrf_nbr_wts` sections in its user-facing parameter list.  Those
two sections move to a research-parameter section of the same file, which the test does not read.

### The rename and the header comment

The flag itself should be renamed.  It is called `recompile_flag` (`mbirtorch/_utils.py:62-63`)
and it compiles nothing.  Setting a flagged parameter rebuilds bindings and compiles nothing.  It
sets one boolean (`mbirtorch/parameter_handler.py:317-318`) whose only effect is a call to
`refresh_device_bindings` (`:343-344`).  That method rebuilds the two `Placement` objects, drops
the device caches, and reconstructs the `Projectors` (`mbirtorch/tomography_model.py:903-918`).
The compiled projector bodies are reused unchanged, because a module-level cache keyed by function
and device index holds them (`mbirtorch/projectors.py:41` and `:158-160`).  The rename is
therefore small.  The proposed name is `rebuild_bindings`.  Its docstring should say three things:
that setting the parameter rebuilds the device placements and the projector bindings, that it
compiles nothing, and that the rule above decides when the field must be true.  Nothing outside
`parameter_handler.py` reads the field, so the rename changes two lines in `_utils.py`, three
occurrences in all, and four lines in `parameter_handler.py`.

The `_utils.py` header says the names, values, and recompile flags "are fixed by an external
reference and must not be changed here" (`mbirtorch/_utils.py:3-5`).  A second copy of that
statement appears immediately before the first dictionary (`:69`).  The statement is already
false. mbirtorch's table has 29 entries and mbirjax's has 30.  The extra mbirjax entry is a
deprecated `use_gpu`, with the flag set to true, that mbirtorch does not have
(`mbirjax/_utils.py:128`).  The comment also names no reference, so a reader cannot check it.

The replacement should say what a reader needs to know.  The table alone defines three things: the
parameter names, their defaults, and whether setting a parameter rebuilds the device placements
and projector bindings.  The table was derived from mbirjax and has since diverged, so a change to
the mbirtorch table does not need to match mbirjax.  A change to a name or a default is a public
API change, and `docs/source/usr_parameters.rst` must change with it.

## Increments

Every day estimate below is rough.

**Increment 1.  Calibration module skeleton.  Rough estimate 4 days.** This increment creates
`mbirtorch/preprocess/geometry_calibration.py` with `CalibrationResult`,
`build_reduced_problem`, `parameter_sweep`, `apply_calibration`, and `check_rotation_direction`,
and exports it from `mbirtorch/preprocess/__init__.py`.  It has four gates.  `parameter_sweep`
returns shape `(num_rows, num_cols, num_candidates)`, and `slice_viewer` pages through candidates
with its default `slice_axis`.  The reduced model's field of view in ALU equals the full model's
to within one voxel, at every binning factor tested.  `build_reduced_problem` raises when the bin
factor does not divide the detector counts, tested at an odd channel count.  `apply_calibration`
is the only function that changes state.

**Increment 2.  Conjugate-view method.  Rough estimate 4 days.** This increment adds the
opposite-view comparison for `det_channel_offset` and `det_rotation`, plus `conjugate_difference`
as a diagnostic image.  It has three gates.  On parallel-beam synthetic data the estimate recovers
a known offset to better than 0.1 channel.  On cone-beam synthetic data at a full fan angle of 20
degrees it recovers the offset to better than 0.5 channel, with the bias recorded.  The function
raises when the angular coverage is below 360 degrees, on a helical model, and on a
curved-detector model for `det_rotation`.

**Increment 3.  Derivative-filter method and the search.  Rough estimate 3 days.** This increment
adds a derivative filter name to `generate_direct_recon_filter`, the image-domain score that reads
it, the coarse pass, and the bracketed polish.  It has four gates.  On a calibrated 360-degree
synthetic scan the derivative reconstruction's score sits within a stated factor of the score on
pure noise, and the increment's script records that factor.  The existing ramp results are
unchanged.  The search recovers a known offset to better than 0.1 channel in about fifteen
evaluations.  On the N = 512 benchmark geometry the increment's script records the calibration
wall time and the reconstruction wall time, so the estimates under "Expected run time" can be
checked against a measurement.

**Increment 4.  Residual method.  Rough estimate 3 days.** This increment adds the weighted
high-pass residual score with its four frozen recon settings, recorded in
`CalibrationResult.reduction`.  It has two gates.  On cone-beam synthetic data the estimate
recovers a known perturbation to better than 0.25 channel.  In every test case the score curve has
a single minimum whose depth exceeds the score's run-to-run spread by a stated factor.

**Increment 5.  Coordinate-descent driver.  Rough estimate 3 days.** This increment adds
`calibrate_geometry`, which runs two coordinate-descent rounds and then a five by five joint grid
over `det_channel_offset` and `det_rotation`.  It has two gates.  A joint perturbation of both
parameters is recovered to the per-parameter gates above.  The joint grid improves on the
coordinate-descent result on at least one synthetic case, or the grid is dropped.

**Increment 6.  Documentation, Zeiss readers, and demo.  Rough estimate 3 days.** This increment
adds a calibration section to `docs/source/usr_preprocess.rst` after the existing preprocessing
entries (`:67-85`), and rewrites the two FAQ answers that tell the user to change
`det_channel_offset` by hand (`docs/source/demos_and_faqs.rst:100-103` and `:147-149`).  It gives
`zeiss.get_sino_and_model` and `zeiss_tct.get_sino_and_model` a `det_rotation` argument that
defaults to 0.0 and flows to `scan_to_sino`, a two-line change per reader. It adds a demo script
to `demo/` at the repository root, which holds nine demo scripts today.  It has three gates.  The
documentation builds, the demo runs to completion, and a Zeiss reader test passes a non-zero
`det_rotation` through to the sinogram.

**Increment 7.  Runtime detector offsets.  Rough estimate 3 days.** This increment makes both
detector offsets call-time tensor inputs in all four geometries and sets `recompile_flag` to false
for both, in `mbirtorch/_utils.py`, `tomography_model.py`, `projectors.py`, and the four geometry
modules.  The comment at `mbirtorch/parallel_beam.py:25-28` says the scalar parameters specialize
as constants and are fixed per model; this change makes that comment wrong, so Increment 7 edits
it.  One new test per geometry asserts zero Dynamo retraces over a sequence of offset values, and
asserts value parity with a model constructed at the same offset.  A second changes one cone
offset, leaves the other, and asserts no new Dynamo frame.  Two gates need other hardware: a
two-GPU run, and a CUDA run of `tests/test_triton_parallel.py` and `tests/test_triton_cone.py`.
Neither was run for the prototype, which ran on virtual CPU devices with no Triton installed
(`../runtime_offsets/runtime_offsets_findings.md:453-462`).  `tests/test_device_policy.py:982-994`
needs its parameter changed, because it uses `det_channel_offset` as its example of a flagged
parameter that changes no array shape.

**Increment 8.  Parameter table review, `ParamDict`, and name coordination.  Rough estimate 3
days.** This increment applies the classification table above, renames `recompile_flag` to
`rebuild_bindings`, replaces both header comments, adds `ParamDict` and the `Unpack` annotation on
`set_params`, splits `docs/source/usr_parameters.rst` into a user-facing list and a
research-parameter section, and adds the three-assertion name test.  The files touched are
`mbirtorch/_utils.py`, `mbirtorch/parameter_handler.py`, `docs/source/usr_parameters.rst`, and
`tests/`.  It has three gates.  The new test passes.  The full suite passes with no test modified.
Setting a research parameter through `set_params` still works, and it still warns where it warned
before.

## Validation plan

Synthetic validation perturbs a known geometry and measures recovery.  The package has three
phantom generators: `generate_3d_shepp_logan_low_dynamic_range` (`mbirtorch/utilities.py:263`),
`gen_cube_phantom` (`:1613`), and `generate_demo_data` (`:1715`).  The gate is recovery of the
injected perturbation to a fraction of a detector pixel, and each increment above names its own
fraction.

Generating synthetic data with the same projector that scores it is an inverse crime.  The test
can then succeed because the model and the data share an error. mbirtorch has no analytic
ray-traced phantom that would avoid this (`../leap_comparison/leap_comparison.md:17`).  The
residual method is flattered most, because at the true geometry the model fits simulated data
exactly.  One source avoids most of the inverse crime at no new code: reconstruct a real scan at
full resolution, then forward project that volume onto a different voxel grid to make the test
sinogram.

The primary real-data gate is a roll-recovery test, and it needs no trusted ground truth.  A known
integer channel roll is applied to a real sinogram with `np.roll`, which introduces no
interpolation.  The estimator then runs on the original sinogram and on the rolled one. The gate
is that the difference between the two estimates equals the roll times the channel pitch, to
within 0.1 channel.  This tests the estimator on real noise, real stripes, and real beam
hardening.

A vendor comparison is the secondary check.  A calibrated NSI scan supplies vendor values for the
channel offset, the row offset, and the tilt (`mbirtorch/preprocess/nsi.py:421-423`), so the
estimate can be compared against the channel offset and the tilt.  Agreement to within 0.25
channel is a sanity check rather than a hard gate, because the vendor value is itself an estimate.
`zeiss.py` supplies a vendor channel offset only (`mbirtorch/preprocess/zeiss.py:282`), because
its row offset and tilt are hard-coded to zero (`:283` and `:120`), so for a Zeiss scan the
comparison is against the estimate's own repeatability and against reconstructed image quality.

The repository ships no real scan data, and its preprocessing tests run against stored synthetic
goldens (`tests/test_preprocess_loaders.py:15-16`).  The real scans named in this plans repository
are SiC, BGA, z62, and Lilly, from the flash-remediation validation
(`../../archive/flash_remediation/flash_remediation_plan.md:1-9`).  Those are the candidate scans,
and the lead developer has to confirm which are available and calibrated.

Robustness validation covers three effects that can bias a score.  Two estimates are made on the
same scan, one with a beam-hardening correction applied and one without, and the check is that
they agree to within 0.1 channel.  Two more are made with and without stripe removal, against the
same threshold, because a stripe is the effect most likely to bias the conjugate-view method.  A
third check sets five percent of the views to zero and confirms that the trimmed mean keeps the
estimate inside the same threshold.

## Risks and open questions

The conjugate-view method needs opposite views, and many industrial scans cover only 180 degrees
plus the fan angle.  The mitigation is that `method='auto'` warns and falls back to the
derivative-filter method.

A strong artifact can shift the location of the residual score's minimum, and truncated
projections degrade the conjugate-view method, which is the limitation LEAP records for
`find_centerCol` (`../leap_comparison/leap_comparison_sources/leap_inventory.md:318`).  The
mitigation is that every estimator returns its candidates and scores, so a flat or double minimum
is visible, and that the lateral-truncation warning already exists
(`mbirtorch/tomography_model.py:2128-2132`) and can gate the method.

`det_rotation` is not a model parameter, so applying an estimated tilt resamples the sinogram, and
the cone-beam conjugate-view method carries a fan-angle bias that grows with the distance from the
center channel.  Three mitigations apply.  `apply_calibration` returns the rotated sinogram
explicitly.  Increment 6 gives both Zeiss readers a `det_rotation` argument, so the tilt can be
applied at read time. The 20-degree gate in Increment 2 decides whether the derivative-filter
method becomes the cone-beam initializer.

Setting a flagged parameter today also drops `prox_data` and the damping cache, and other code may
depend on that side effect.  After Increment 7, changing `det_row_offset` also stops refreshing
anything, while `auto_set_recon_geometry` still derives `recon_shape` and `recon_slice_offset`
from that offset (`mbirtorch/cone_beam.py:617-655`).  Three mitigations apply: the findings page's
conclusion that neither cache is stale after an offset change
(`../runtime_offsets/runtime_offsets_findings.md:465-470`), the parity tests in Increment 7, and a
note in `apply_calibration` saying that re-running `auto_set_recon_geometry` is the caller's job
after a row-offset change.

Renaming `recompile_flag` renames a field of a public dataclass, which a downstream caller could
read, and the `_utils.py` header states that the flag values must not be changed.  The mitigation
is that nothing outside `parameter_handler.py` reads the field, that the `Param` repr can carry
both names for one release, and that the panel review decides the header question.

Two questions are open for the panel.  Should `calibrate_geometry` default to changing the model,
or stay read-only as proposed here?  Ten of the 29 table parameters have no section in
`docs/source/usr_parameters.rst` today, and the split proposed under "Coordinating the parameter
names" leaves them undocumented as research parameters.  Is that the right split?

### Should the geometry parameters change?

The recommendation is no change to the geometry parameters now, and four reasons support it.  The
row offset needs a constraint from outside the sinogram rather than a new parameter, for the
reason given under deferred work.  A magnification search uses the parameters that already exist,
and it costs one retrace per candidate, which is acceptable while the candidates are few.  LEAP's
`tau` reduces to `det_channel_offset` to first order, and what is left of it is a detector yaw
that a per-view projective resampling can remove.  A detector pose parameter inside the projectors
would be justified only by a pose error large enough to matter at the resolution the user asks
for, and no dataset in hand shows one.

The question should be revisited on evidence.  The evidence would be a calibrated NSI or Zeiss
scan that still shows geometry artifacts after `det_channel_offset` and `det_rotation` have both
been calibrated.

## References

The plan documents cited above are given relative to this file.
`../leap_comparison/leap_comparison.md` holds sections 1 and 2 and the calibration table at 162 to
177.  `../leap_comparison/leap_comparison_sources/leap_inventory.md` holds section 7.5 at 316 to
327.  The prototype record is `../runtime_offsets/runtime_offsets_findings.md`, with
`runtime_offsets.diff` in the same directory.  The flash-remediation plan is
`../../archive/flash_remediation/flash_remediation_plan.md`, and the layout rule is in
`../../README.md`.

The run-time estimates rest on two measurement records.  They are
`../leap_comparison/quality_results.md` and
`../../experiments/features/leap_comparison/results/leap_benchmark_results.md`.

Source files are cited in place above.  They lie in `mbirtorch/`, `mbirtorch/preprocess/`,
`docs/source/`, `tests/`, and `demo/`, plus `pyproject.toml` at the repository root.  The
comparison package is `mbirjax`, in the same research directory, where `mbirjax/_utils.py`,
`mbirjax/parameter_handler.py`, and `tests/test_utilities.py` are cited.  LEAP files are cited
with a `LEAP` prefix, and they lie in `src/` and `documentation/` of the LEAP tree at the commit
named at the top of this plan.

## Changes in v3

1. The new module is `mbirtorch/preprocess/geometry_calibration.py`, renamed from
   `calibration.py`, in the executive summary, the API subsection, and Increment 1.
2. "Coordinating the parameter names" now proposes a `ParamDict` typed dictionary and an `Unpack`
   annotation on `set_params`, a split of the table into 17 user-facing and 12 research
   parameters for Greg to confirm, and a three-assertion name test in place of the earlier
   two-containment one.  Increment 8 carries that work.
3. "Candidate evaluation" now records how LEAP applies a detector tilt inside its cone-beam
   kernels, and how that route compares with resampling the sinogram.
4. A "Workflow" subsection gives the automatic call sequence and the manual one.
5. The offset-scan case is now stated correctly.  A laterally displaced flat detector is a change
   of `det_channel_offset`, the estimators handle an offset scan, and only the
   direct-reconstruction weighting an offset scan needs is out of scope.
6. `tau` is now given as `det_channel_offset` to first order, with the residual detector yaw and a
   worked example, in place of the claim that it has no mbirtorch counterpart.
7. A paragraph records that one per-view projective resampling can remove any detector pose error,
   and that a source position error is not removable that way.
8. The flatness in `det_row_offset` is now attributed to the vertically shifted reconstruction
   rather than to the NSI identity.
9. A "Should the geometry parameters change?" subsection recommends no change now, with its four
   reasons and the evidence that would reopen the question.
10. An "Expected run time" subsection derives estimates from the benchmark record and the quality
    record, and Increment 3 gains a gate that records the calibration wall time against the
    reconstruction wall time.
11. The status line reads DRAFT v3, and this list was added.
