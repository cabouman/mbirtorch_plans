# Geometry viewer, plan of record

Status: COMPLETE
Updated: 2026-09-19
Code: mbirtorch prerelease d90fd69 holds Increments 1 to 7 (d9882d2, cde72a0, e662d27, 41fca86, all 2026-09-12)
Next step: Correct the stale status lines and the pre-move module paths in this plan and in the gv8 and gv9 records, then move the folder to archive/, moving experiments/web/ and the two gv5 build scripts to tools/ instead of deleting them.

Date: 2026-09-09.  Status: APPROVED direction (Greg, 2026-09-09).  Increment 1 is
done (2026-09-09): `reference/geometry_conventions.md` records the confirmed conventions, and
`gv1_conventions_probe.py` passed its gate for all six geometries with a largest
error of 0.11 detector pixel.  Increment 2 is done (2026-09-09): `geometry_scene.py`
and its 67 tests pass, and `gv2_scene_findings.md` records the surface and the drawing
choices.  Increment 3 is done (2026-09-09): `geometry_viewer.py` draws the five
panels, 36 viewer tests pass, and `gv3_static_figure_findings.md` reviews eight figures
against the conventions record with no disagreement.  Increment 4 is done
(2026-09-09): the view slider, the source-path toggle, the 3D zoom, and the comparison
overlay are built, 152 tests pass, and a slider step on an 1800-view model takes 41 ms
against the 100 ms gate (`gv4_interaction_findings.md`, `gv4_timing.md`).  Increment 5
was done on 2026-09-11, after the open items of the Increment 4 review.  Increment 6,
the port, was done on 2026-09-11 and is staged in the mbirtorch repository for review;
`gv8_port_findings.md` is its record.  Increment 7, the two overlays at scale, was
done on 2026-09-12 and is staged there as well; `gv9_overlay_findings.md` is its
record.  Greg's first use (2026-09-09
and 2026-09-10) produced four changes: a display fix for backends outside the blit
path, labels on the source and detector, an angle-0 reference, and the display
convention that negative z is the top of every drawing, recorded in
`reference/geometry_conventions.md`.  `gv_show_example.py` opens the viewer on any of the six
geometries.
Decisions recorded on 2026-09-09: the first frontend is matplotlib; the primary
uses are checking a real scan's geometry before reconstruction and supporting the
geometric calibration work; the prototype is built in this repository for a later
port into mbirtorch; data overlays (a sinogram on the detector, a reconstruction in
the volume) come after the geometry itself works.

Citations.  mbirtorch file paths are given from the package directory, so
`mbirtorch/cone_beam.py` means `mbirtorch/mbirtorch/cone_beam.py` in the sibling
repository.  Experiment scripts live in `plans/geometry_viewer/experiments/` and are
cited by bare name.  The comparison studies cited are
`surveys/tigre_comparison/survey.md` and
`surveys/leap_comparison/leap_comparison.md` (version 1, deleted 2026-09-18; `git show cd4aac3:plans/features/leap_comparison/leap_comparison.md`).

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
  and `plans/geometry_viewer/findings/reference_sketches.md` in this directory records what the three reference
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

The prototype lives in `plans/geometry_viewer/experiments/` in this repository, and
the port moves two files into mbirtorch.  The slice viewer was built the same way,
as a package-independent file with a thin wrapper.  The port adds the public entry
point, a demo, a docs page, and a row in `reference/api_specification.md`.

## Increments

Each increment ends with a review stop.  A gate is a test or a measurement that must
pass before the next increment starts.  Increments 1 to 3 run in order.  Increment 4
follows 3.  Increment 5 may start once 3 is done.  Increment 6 is a separate
session in the mbirtorch repository.

**Increment 1.  Conventions record.**  The deliverable is `reference/geometry_conventions.md`
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
Done 2026-09-11: `sinogram=` and `recon=` on `GeometryFigure` and `geometry_viewer`,
with `set_sinogram` and `set_recon`; the gate passes on the rendered pixels, a
forward-projected phantom lands inside the drawn rims, and a slider step with a
sinogram takes 25 ms on the 1800-view model.  Later the same day (Greg's request):
three toggles show and hide the sinogram, the phantom, and the comparison, and the
phantom is also drawn as an outline in the top and side views and, in the 3D panel,
as a wire outline that follows the support slice by slice, a parallelepiped for the
sheared cube phantom, which is projected onto the detector face as well.  The
record is `gv7_data_overlays_findings.md`.

**Increment 6.  The port.**  Move the scene and viewer into mbirtorch, add the entry
point, a demo, a docs page, and the API specification row.  This increment runs in
the mbirtorch repository and is planned separately.
Decision (Greg, 2026-09-11): the port's first step is a single source of truth for
the point projection.  `GeometryScene.detector_coordinates` holds a copy of the
projector's per-geometry formulas, pinned by the corner-projection test, and every
drawing on the detector face goes through it.  mbirtorch has no public method that
maps object points to detector indices; its per-geometry mapping lives in private
per-view helpers that take pixel indices.  The port adds `project_points` to
`TomographyModel`, implemented in each model class from the same helpers the
projector uses, and the ported scene calls it and drops its copy.  The web page
cannot import torch and keeps the numpy copy, pinned by the same test, as
`geometry_defaults.py` is kept honest today.  Until then the prototype draws from
the tested copy and writes no new projection formula.
Done 2026-09-11: `TomographyModel.project_points` is implemented in each model class
from the projector's own helpers, with the projector's arithmetic bitwise unchanged;
the scene and the figure live in `mbirtorch/geometry_scene.py` and
`mbirtorch/geometry_figure.py`, and `mbirtorch.geometry_viewer` is exported beside
`mbirtorch.slice_viewer`; the tests, a demo, a docs page, and the API rows are in
place, and the two docstrings are corrected.  Decision (Greg, 2026-09-11): the web
page keeps a separate numpy scene, so the prototype's files in the experiment
directory stay as the page's sources and the page was not touched.  The record is
`gv8_port_findings.md`.

**Increment 7.  The overlays at scale.**  Decisions (Greg, 2026-09-11 and
2026-09-12): a sinogram larger than about 128 pixels across is subsampled for
display, on the device that holds it, and `vmin` and `vmax` fix its gray scale; the
phantom's outline in the 3D panel follows the phantom's shape rather than its
bounding rectangles, because the old outline reached outside the region of
reconstruction for demo 3's round phantom; the phantom is read a chunk at a time
with no full-size temporaries; and two candidate outlines, sections in single
planes and slab silhouettes, were built and compared before one was chosen.
Done 2026-09-12: sections across the support's thinnest direction, at most nine,
with the legend naming the count and a point budget that coarsens long outlines;
the slab kind stays behind a constant.  The record is `gv9_overlay_findings.md`.

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
Gradio app in `plans/geometry_viewer/experiments/web/`.

The primary packaging is a Gradio-Lite static page, `web/lite/index.html`, which
runs the Python in the visitor's browser through Pyodide.  It is the route the guide
gives for Python on a free account.  Gradio-Lite was last published on 2025-09-10 as
version 5.45.0 and its source is gone from the Gradio repository, so the page pins that
release and the app uses no Gradio 6 feature.  The cloud session that built the page
could not open it, because its network policy blocked the CDN that serves the runtime.
The secondary packaging is a standard Gradio Space, `web/space/`, which runs the same
app on a server, needs a paid Hugging Face plan, and was tested end to end with a
headless browser.

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
Deployed 2026-09-10: the static Space failed before the app started.  The runtime is
frozen at its last release but resolves gradio's dependencies at load time, and three
of them had moved on; the viewer also used a matplotlib 3.10 argument that Pyodide's
matplotlib 3.8.4 rejects.  A shim in `index.html` now holds the runtime's dependencies
at their releases of 2025-09-10 and gives gradio's queue a thread-free runner, the
viewer guards the argument, and under Pyodide the app reports an invalid value in its
status line instead of raising.  The page runs in a headless Chromium: the first figure
appears 13 s after the page loads and a render takes about 0.4 s in the browser.  The
record is the last section of `gv5_web_findings.md`.  The Space was re-uploaded from
`web/lite/` the same day and runs, with the same times, in the headless browser; Greg's
own browser is the acceptance check that remains.  A guide for other Thingy builders,
with the shim to paste, is `reference/gradio_lite_space_guide.md`.

## Open items from the Increment 4 review

- **The fit statement is wrong for helical scans.**  `volume_fits_detector` asks
  whether every volume corner projects inside the detector in every view.  A
  helical volume never does, because each view sees only part of the axial travel,
  so the text panel says "no" for every helical scan.  The statement should count
  the views in which the volume leaves the detector, and for a helical scan it
  should compare the volume's z extent with the detector's swept coverage instead.
  Done 2026-09-10: `fit_report` applies the helical rule, the text panel counts the
  views the volume leaves the detector in and prints the swept z range, and
  `gv6_review_items_findings.md` records the numbers.
- **The fit statement tests the wrong shape.**  `volume_fits_detector` projects
  the corners of the reconstruction box.  The automatic box is the square around the
  field-of-view circle, so its corners always project outside the detector and every
  automatically sized cone scan reads "volume fits det: no".  When the
  region-of-reconstruction mask is on, the check should project the cylinder's
  outline instead, and the detector-face panel should draw it.
  Done 2026-09-10: the statement tests the cylinder's two rims when the mask is on,
  and the detector face draws them.  The cylinder is the mask's own ellipse, through
  the centers of the outermost voxels, which a forward projection confirmed on
  2026-09-11.  The statement is two rows, "lateral fit" and "axial fit" (Greg,
  2026-09-11).  The automatically sized cone scan fits laterally and misses axially
  by 6.8 rows, and the findings page says why that is the honest answer.
- **The web page fixes the 3D camera.**  The figure is an image there, so it cannot
  be rotated.  Two numbers for the camera angles, which `GeometryFigure` already
  accepts, would give the page what the mouse gives the desktop.
  Done 2026-09-10: two number boxes under the toggles set the camera; a change costs
  one render, as every control change does.
- **The comparison section runs out of room** in the text panel when two
  geometries differ in many parameters.  A panel or window of its own is the fix.
  Done 2026-09-10: a comparison opens a second figure, `compare_figure`, that tables
  every difference with the parameters first; the text panel keeps the parameter
  differences and a count of the derived quantities that differ; `save` writes the
  window beside the main file; the web page shows the table as a second plot.
- **The text panel is cramped at its font floor.**  Found 2026-09-10: the
  multiaxis example with a comparison reaches the 5 point floor, because its drawing
  note is six lines, and the block placement then runs the static block's last line
  into the comparison header.  A shorter note for the parallel-type geometries, or a
  placement that keeps the blank line at the floor, would fix it.
  Done 2026-09-11: the scene's drawing notes are shorter, the two convention notes
  share a line, and the comparison block drops its sentence about the derived
  quantities before it drops a parameter entry.  The multiaxis and parallel examples
  with a comparison now sit above the floor, at 5.1 points, with every block whole.
- **The slider has run only under the Agg backend.**  The partial-redraw path
  follows the slice viewer's, but it has not been watched on a screen.  Greg's use
  of the viewer is the first display test, before any port.
  Closed 2026-09-10: Greg's use of the viewer on the 9th and 10th was that test.
- **The multiaxis side view's source marker covered part of the
  `recon_slice_offset` label.**  Done 2026-09-10: the label is a moving artist placed
  on the side of its segment away from the source, and the label overlap test now
  checks the source marker against the labels of the side view at three views.
- **The top view's labels collide in some views.**  Found 2026-09-10 by running the
  extended overlap test on the top view at three views: the channel-offset label
  runs into the angle-0 caption in three geometries, the source-travel label into
  the source's label in two, the detector-iso label touches the pixel-0 label in
  two, and in the multiaxis geometry the channel-offset label reaches the source's
  marker.  The top view needs a placement pass of its own, after which the overlap
  test can cover it at several views as it now covers the side view.
  Done 2026-09-11: three placement rules and shorter offset labels ("chan offset",
  "row offset") clear the top view at views 0, 4, and 7 of every geometry, and the
  overlap test measures the top view at view 0.  Seven transient collisions remain
  at other views, listed in `gv6_review_items_findings.md`; Greg's decision is that
  a label clear in most views need not be clear in every view.
- **The detector-face legend covered the panel's content.**  Found by Greg,
  2026-09-11.  Done the same day: the legend sits in a band under the panel, two
  rows of three entries, measured against every panel and the widgets in every
  geometry, with and without a comparison and a sinogram.

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
  `plans/geometry_viewer/experiments/`.  The other feature studies use
  `plans/experiments/features/<name>/`.  This plan follows the initial prompt.  (The layout of
  the time; since 2026-09-18 every plan uses `plans/<name>/experiments/`.)

## Terms

- **Object frame**: the x, y, z coordinates in which voxel positions are computed,
  with z along the rotation axis.
- **View action**: what a model does to the object for one view: a rotation, a
  rotation and tilt, a translation, or a rotation and z shift.
- **Scene**: the set of drawable primitives for one view, computed by the pure
  numpy class.
- **ALU**: arbitrary length unit, the unit of every distance in a model.
