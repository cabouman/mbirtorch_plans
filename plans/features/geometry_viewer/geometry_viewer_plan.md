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
is next, after Greg has used the interactive viewer.  Greg's first use (2026-09-09
and 2026-09-10) produced four changes: a display fix for backends outside the blit
path, labels on the source and detector, an angle-0 reference, and the display
convention that negative z is the top of every drawing, recorded in
`geometry_conventions.md`.  `gv_show_example.py` opens the viewer on any of the six
geometries.
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
  box for the current view, seen from the source with row 0 at the top and the
  channel index increasing to the right, as `imshow` shows one sinogram view.
  This panel shows whether the volume projects outside the detector, and where
  the central ray lands relative to the grid center.
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

## The Python interface (Greg, 2026-09-10)

The primary interface is a Python call used the way the slice viewer is used.  The
slice viewer's call is `mbirtorch.slice_viewer(*volumes, title='', ..., block=True)`:
it builds the viewer, shows it, and returns the viewer object; with `block=False` it
returns at once and a module-level registry keeps the window alive until the next
blocking call returns.  The geometry viewer's call mirrors it:
`geometry_viewer(model, view_index=0, show_trajectory=False, compare=None,
show_reference=True, zoom='scan', title='', block=True)`, returning the
`GeometryFigure`, with the same nonblocking registry and the same docstring shape.
The port (Increment 6) exposes it as `mbirtorch.geometry_viewer` beside
`mbirtorch.slice_viewer`.  The web page below is an additional route to the same
drawing, not a replacement for this call.  Built 2026-09-10: `geometry_viewer` in
`geometry_viewer.py`, with `show_geometry` kept as the earlier name.

## Web packaging (Greg, 2026-09-10)

Greg asked for the viewer on huggingface.co, as an entry in Charlie Bouman's Thingy
Repository, whose builder's guide sets the rules.  The viewer needs neither torch nor
mbirtorch, because the scene is built from a parameter dictionary and the figure is
matplotlib, so it can run on a Hugging Face Space.  Two packagings are built from one
Gradio app in `plans/experiments/geometry_viewer/web/`.

The primary packaging is a Gradio-Lite static page, `web/lite/index.html`, which
runs the Python in the visitor's browser through Pyodide.  It is the route the guide
gives for Python on a free account.  Gradio-Lite was last published on 2025-09-10 as
version 5.45.0 and its source is gone from the Gradio repository, so the page pins that
release, the app uses no Gradio 6 feature, and the page cannot be tested in this
session, whose network policy blocks the CDN that serves the runtime.  The secondary
packaging is a standard Gradio Space, `web/space/`, which runs the same app on a
server, needs a paid Hugging Face plan, and is tested here end to end with a headless
browser.

The guide's acceptance list adds three files or fields: a `license` field in the
README metadata (bsd-3-clause, mbirtorch's license), a one-line `short_description`,
and a square `icon.png` of about 400 by 400 pixels at the Space root, which the build
script renders from the scene.  The web app replaces the matplotlib widgets with
Gradio controls and shows the figure as an image, so the 3D panel cannot be rotated
there.  One new module, `geometry_defaults.py`, copies mbirtorch's automatic
reconstruction geometry so that a scene can be built from scan parameters alone; a
test pins the copy to the real models.  The record is `gv5_web_findings.md`.
Built 2026-09-10: 224 tests pass, the Gradio Space runs in a headless browser under
Gradio 6.26 and under Gradio 5.45, and a render takes 0.7 to 1.2 s on the server.

## Open items from the Increment 4 review

- **The fit statement is wrong for helical scans.**  `volume_fits_detector` asks
  whether every volume corner projects inside the detector in every view.  A
  helical volume never does, because each view sees only part of the axial travel,
  so the text panel says "no" for every helical scan.  The statement should count
  the views in which the volume leaves the detector, and for a helical scan it
  should compare the volume's z extent with the detector's swept coverage instead.
- **The fit statement tests the wrong shape.**  `volume_fits_detector` projects
  the corners of the reconstruction box.  The automatic box is the square around the
  field-of-view circle, so its corners always project outside the detector and every
  automatically sized cone scan reads "volume fits det: no".  When the
  region-of-reconstruction mask is on, the check should project the cylinder's
  outline instead, and the detector-face panel should draw it.
- **The web page fixes the 3D camera.**  The figure is an image there, so it cannot
  be rotated.  Two numbers for the camera angles, which `GeometryFigure` already
  accepts, would give the page what the mouse gives the desktop.
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
