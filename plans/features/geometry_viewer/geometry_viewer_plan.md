# Geometry viewer, plan of record

Date: 2026-09-09.  Status: APPROVED direction (Greg, 2026-09-09).  Increment 1 is
done (2026-09-09): `geometry_conventions.md` records the confirmed conventions, and
`gv1_conventions_probe.py` passed its gate for all six geometries with a largest
error of 0.11 detector pixel.  Increment 2 is done (2026-09-09): `geometry_scene.py`
and its 67 tests pass, and `gv2_scene_findings.md` records the surface and the drawing
choices.  Increment 3 is done (2026-09-09): `geometry_viewer.py` draws the five
panels, 36 viewer tests pass, and `gv3_static_figure_findings.md` reviews eight figures
against the conventions record with no disagreement.  Increment 4 is done
(2026-09-09): the view slider, the source-path toggle, the 3D zoom, and the comparison
overlay are built, 152 tests pass, and a slider step on an 1800-view model takes 41 ms
against the 100 ms gate (`gv4_interaction_findings.md`, `gv4_timing.md`).  Increment 5
is next, after Greg has used the interactive viewer.
Decisions recorded on 2026-09-09: the first frontend is matplotlib; the primary
uses are checking a real scan's geometry before reconstruction and supporting the
geometric calibration work; the prototype is built in this repository for a later
port into mbirtorch; data overlays (a sinogram on the detector, a reconstruction in
the volume) come after the geometry itself works.

Citations.  mbirtorch file paths are given from the package directory, so
`mbirtorch/cone_beam.py` means `mbirtorch/mbirtorch/cone_beam.py` in the sibling
repository.  Experiment scripts live in `plans/experiments/geometry_viewer/` and are
cited by bare name.  The comparison studies cited are
`plans/features/tigre comparison/TIGRE_comparison.md` and
`plans/features/leap_comparison/leap_comparison.md`.

## Purpose

The viewer draws the scan geometry that a `TomographyModel` describes.  A user builds
a model from a scanner's files, opens the viewer, and sees the source, the detector,
the reconstruction volume, and the rotation axis in one picture, together with the
numbers that the picture implies.

Two uses drive the design.  The first use is checking a real scan before
reconstruction.  A detector offset with the wrong sign, a detector too small for the
object, or a helical travel that leaves slices uncovered is visible in a drawing
within seconds, while the same error takes a full reconstruction to see in the
output.  The TIGRE comparison ranked this capability as the largest gap in
mbirtorch's plotting, and both TIGRE and LEAP have one.  The second use is
supporting geometric calibration.  The calibration work estimates offsets and a
detector rotation, and the viewer should show an estimated geometry against a
vendor geometry so that the size and direction of a correction are visible.

## The geometry the viewer must draw

All four models share one object frame.  Voxel (i, j, k) of the reconstruction has
object coordinates y, x, z, in that order: y runs along the row index with pitch
`delta_voxel * voxel_row_aspect`, x runs along the column index with pitch
`delta_voxel`, and z runs along the slice index with pitch
`delta_voxel * voxel_slice_aspect` plus `recon_slice_offset`.  The volume is centered
on the origin in x and y.  This frame is the one the projector code uses in
`_cone_pixel_xy_mag` (`mbirtorch/cone_beam.py`), `_parallel_hfan_math`
(`mbirtorch/parallel_beam.py`), and their multiaxis and translation counterparts.

Each model applies a per-view action to the object and holds the source and detector
fixed.  The parallel and cone models rotate the object about z by the view angle.  The
multiaxis model rotates the object by the azimuth and tilts the rays by the
elevation.  The translation model shifts the object by a per-view vector.  A helical
cone scan adds a per-view z shift.  Reading the cone code, the source sits on the
positive y axis at `source_iso_dist`, and the detector plane sits on the negative y
side at `source_detector_dist` from the source.  The offsets `det_channel_offset`
and `det_row_offset` shift the detector grid relative to the central ray.  These
readings come from the projector code and not from a document, because the
documentation figures do not define axes.  Increment 1 confirms them numerically.

The viewer draws the object fixed and the source and detector moving.  In the code
the object moves and the source stays.  A user of a scanner thinks of the object as
fixed and the gantry as moving, and TIGRE and LEAP draw it that way.  The two
pictures are the same geometry.  Converting between them is a rotation of the
source and detector by the negative of the view angle, and a shift by the negative
of the translation or z shift.

## What the viewer shows

The figure has four drawing panels and one text panel.  The panels share one scene
so that they always agree.

- **3D view.**  The source, the detector outline, the volume box, the rotation axis,
  the rotation direction as an arc with an arrowhead, and the rays from the source
  to the four detector corners, for one view.  The region of reconstruction
  cylinder is drawn inside the box when `use_ror_mask` is on.  The panel can be
  rotated with the mouse.
- **Index markers.**  Detector pixel (row 0, channel 0) and voxel (0, 0, 0) are
  marked in every panel where they appear.  A mirrored channel order or a wrong
  offset sign is visible from these two markers alone.  CIL's sketch does this,
  and `reference_sketches.md` in this directory records what the three reference
  packages draw.
- **Top view (the xy plane).**  The fan angle, the lateral field of view at the
  rotation axis, and the volume's footprint.  A lateral offset of the detector is
  visible here as an asymmetric fan.
- **Side view (the yz plane).**  The cone angle, the detector height, the axial
  field of view, the slice offset, and the helical travel as a vertical range.
- **Detector face.**  The detector grid with the projected outline of the volume
  box for the current view.  This panel shows whether the volume projects outside
  the detector, and where the central ray lands relative to the grid center.
- **Text panel.**  The parameters and the derived quantities: magnification, fan
  and cone angles in degrees, field of view at the rotation axis in ALU, voxel
  pitch, the volume's extent, the psf radius, and a truncation statement.

Interaction is limited to what the two uses need.  A slider steps through views,
and the panels redraw for that view.  A toggle overlays every source position as
the trajectory.  A second geometry can be overlaid in a second color for the
calibration use, drawn from a second model or from a dictionary of parameter
overrides, and the text panel then lists the differences.

Two overlays are deferred.  A sinogram view can be painted onto the detector face
for the current view, and a reconstruction or phantom can be drawn as a silhouette
in the volume box.  Both are useful, and both add scope, so they wait for the
geometry itself to work.

## Design rule: draw what the projector computes

The viewer must never be a second implementation of the geometry.  The scene
computes the detector coordinates of a point with the same formulas the projector
uses, and a test checks the scene against the projector.  The test forward-projects
single voxels at the volume corners with `sparse_forward_project`, finds the
centroid of each footprint in detector rows and channels, and compares it with the
scene's prediction.  The comparison uses a tolerance of half a detector pixel and
not exact equality, following the float rule in `.claude/lessons.md` section 2.  A
viewer that fails this test would mislead the user in exactly the situation it is
meant for.

## Architecture

The viewer follows the slice viewer's structure (`mbirtorch/viewer.py`).  A pure
numpy scene class takes the model's parameters and a view index and returns
drawable primitives: the source point, the detector outline as a polyline (a
rectangle for a flat panel and an arc for a curved one), the volume box, the axis,
the corner rays, and the volume's projected outline on the detector.  The scene
imports numpy only, so every geometric statement is testable without a display.  A
matplotlib class draws the scene and owns the slider and toggle.  It imports
pyplot on first construction and never at import time.  The scene reads parameters
through `get_params`, so it depends on the parameter names and not on the model
classes.

The prototype lives in `plans/experiments/geometry_viewer/` in this repository, and
the port moves two files into mbirtorch.  The slice viewer was built the same way,
as a package-independent file with a thin wrapper.  The port adds the public entry
point, a demo, a docs page, and a row in `plans/API_specification.md`.

## Increments

Each increment ends with a review stop.  A gate is a test or a measurement that must
pass before the next increment starts.  Increments 1 to 3 run in order.  Increment 4
follows 3.  Increment 5 may start once 3 is done.  Increment 6 is a separate
session in the mbirtorch repository.

**Increment 1.  Conventions record.**  The deliverable is `geometry_conventions.md`
in this directory, one section per geometry, stating where the source, the detector,
the rotation axis, and the offsets sit in the object frame, and which way the view
angle turns.  The probe script `gv1_conventions_probe.py` confirms each statement.
For each geometry it builds a small model with offsets, unequal detector pitches, and
non-unit aspect ratios, forward-projects single voxels at the volume corners and
center, and compares the footprint centroids with a prediction written from the
geometric statement rather than copied from the code.  Gate: every centroid agrees
with its prediction within half a detector pixel, in every view, for parallel, flat
cone, curved cone, helical cone, multiaxis, and translation.  The script's companion
`gv1_conventions_probe.md` records the numbers.

**Increment 2.  The scene.**  `geometry_scene.py` with the class described above and
`test_geometry_scene.py` with headless tests.  Gate: the corner-projection test from
the design rule passes for the six geometries of Increment 1, and the scene handles
the edge cases: a single detector row, an infinite `source_detector_dist` in the cone
model, a curved detector, and a translation model, which has no rotation axis and
draws the translation path instead.

**Increment 3.  The static figure.**  `geometry_viewer.py` draws the five panels for
one view and saves the figure to a file.  Gate: figures for the six geometries are
saved under the Agg backend, and a review of the six images against the
conventions record finds no disagreement.  The review is by eye and is recorded in
a findings page with the images.

**Increment 4.  Interaction and comparison.**  The view slider, the trajectory
toggle, and the second-geometry overlay.  Three items from the Increment 3 review
join this increment: a zoom control for the 3D panel, because a 12 ALU volume in a
200 ALU cube is twenty pixels wide; moving the parallel-beam corner rays and the
rotation-direction arc from the drawing layer into the scene, so that the drawing
layer computes no geometry at all; and a larger default drawing distance for the
parallel-type geometries, because at 1.5 half-extents the drawn source touches the
volume box.  Gate: the slider redraws a 1800-view
model in under 100 ms per step on a laptop, the trajectory of 1800 views draws
as one polyline and not as 1800 markers, and the overlay of a model against a copy
with `det_channel_offset` changed by ten channels shows the shift on the detector
face and lists it in the text panel.

**Increment 5.  Data overlays.**  A sinogram view on the detector face and a
reconstruction silhouette in the volume box.  Gate: the sinogram view is drawn with
the same row and channel orientation the detector-face grid uses, checked by
painting a synthetic sinogram with one bright pixel at a known row and channel.

**Increment 6.  The port.**  Move the scene and viewer into mbirtorch, add the entry
point, a demo, a docs page, and the API specification row.  This increment runs in
the mbirtorch repository and is planned separately.

## Open items from the Increment 4 review

- **The fit statement is wrong for helical scans.**  `volume_fits_detector` asks
  whether every volume corner projects inside the detector in every view.  A
  helical volume never does, because each view sees only part of the axial travel,
  so the text panel says "no" for every helical scan.  The statement should count
  the views in which the volume leaves the detector, and for a helical scan it
  should compare the volume's z extent with the detector's swept coverage instead.
- **The comparison section runs out of room** in the text panel when two
  geometries differ in many parameters.  A panel or window of its own is the fix.
- **The slider has run only under the Agg backend.**  The partial-redraw path
  follows the slice viewer's, but it has not been watched on a screen.  Greg's use
  of the viewer is the first display test, before any port.

## Risks and open questions

- **matplotlib's 3D panel has no depth ordering** and slows down with many artists.
  The three 2D panels carry the quantitative content, and the 3D panel is for
  orientation.  If the 3D panel proves too weak, a second frontend can be added
  behind the same scene class.
- **The detector rotation is not a model parameter.**  The NSI reader applies
  `det_rotation` to the sinogram inside preprocessing and the model never sees it.
  For the calibration use the viewer could accept an optional rotation and draw the
  detector turned about its normal.  Whether to add this is open.
- **Object truncation cannot be drawn from the model alone.**  The automatic
  reconstruction shape is set from the detector, so the volume always fits the
  detector by construction.  Truncation of the object is visible only with data,
  which is the sinogram overlay of Increment 5.
- **Long helical travel** makes the side view tall and the detector small.  The side
  view may need its own scale, or a range indicator in place of a to-scale drawing.
- **The experiment directory.**  The initial prompt places scripts in
  `plans/experiments/geometry_viewer/`.  The other feature studies use
  `plans/experiments/features/<name>/`.  This plan follows the initial prompt.

## Terms

- **Object frame**: the x, y, z coordinates in which voxel positions are computed,
  with z along the rotation axis.
- **View action**: what a model does to the object for one view: a rotation, a
  rotation and tilt, a translation, or a rotation and z shift.
- **Scene**: the set of drawable primitives for one view, computed by the pure
  numpy class.
- **ALU**: arbitrary length unit, the unit of every distance in a model.
