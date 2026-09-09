# gv4: interaction and comparison, as built

Date: 2026-09-09.  Files changed: `geometry_scene.py`, `geometry_viewer.py`,
`test_geometry_scene.py`, `test_geometry_viewer.py`.  Files added:
`test_geometry_interaction.py`, `gv4_timing.py`, `gv4_timing.md`,
`gv4_render_figures.py`, and three images under `figures/`.  Status: built; 152
tests pass and a slider step on an 1800-view model costs 42 ms against a gate
of 100 ms.

## What was built

Increment 4 adds three controls, a second geometry, and two primitives that
moved from the drawing layer into the scene.

The controls sit in a row under the panels.  A `Slider` labeled "View" steps
through the views one whole view at a time and calls `set_view`; a scan with one
view builds no slider and hides its axes.  A `CheckButtons` labeled "source
path" calls `set_show_trajectory`.  A second `CheckButtons` labeled "3D zoom to
volume" calls `set_zoom`.  The widget conventions are the slice viewer's, in
`mbirtorch/viewer.py`: an integer `valstep`, the slider's own `drawon` turned
off so that a step goes through this class's redraw, and `CheckButtons` for the
two toggles.

The 3D zoom has two states.  In `'scan'`, which is the default and what
Increment 3 drew, the source, the detector, and the volume share one cubic box.
In `'volume'` the cube is three volume extents wide and centered on the volume,
and the axes limits clip whatever leaves it.  A cone geometry's volume is
about twenty pixels wide in the scan view, which is what the Increment 3 review
asked to fix.

The comparison overlay draws a second geometry in one color with dashed lines.
`GeometryFigure(..., compare=...)` and `set_compare(...)` accept a
`GeometryScene`, a `TomographyModel`, or a dictionary of parameter overrides
such as `dict(det_channel_offset=12.5)`, which is applied to a copy of the
first scene's parameters.  Each drawing panel gains the comparison's detector
outline, source, central ray, and pixel-0 marker, and the detector face gains
its projected volume outline, the point where its central ray lands, and its
grid center.  The volume box is shared unless the two volumes differ, in which
case the comparison's box is drawn as well.  The comparison follows the slider
and the source-path toggle.  `set_compare(None)` removes every comparison
artist.

The text panel gained a comparison section.  It lists each differing parameter
and derived quantity as `name: primary -> comparison`.  A value that reads the
same to three significant figures is printed to seven, so that a listed
difference is a visible difference.

Three changes are in the scene, so that the drawing layer computes no geometry.

- `ViewScene.corner_rays` of a parallel-type geometry is now four parallel
  segments.  Each runs along `ray_direction` and ends on its detector corner,
  and each starts in the plane through the drawn source, so a corner ray is as
  long as the drawn central ray.  Before this change the four rays converged on
  the drawn source, which is the picture of a cone beam; the Increment 3
  viewer worked around it in the drawing layer.
- `ViewScene.rotation_direction_arc` is a new primitive: a short arc on the
  circle the source travels, starting at the drawn source, with its last two
  points giving the direction an arrowhead points.  `source_travel_sense` says
  the same thing in words, either `'clockwise seen from +z'` or
  `'counterclockwise seen from +z'`.  Both are None for the translation
  geometry, for a single view, for two neighboring views at one angle, and for
  a source on the rotation axis.
- `DEFAULT_DRAWING_DISTANCE_FACTOR` rose from 1.5 to 2.5.  At 1.5 the drawn
  source of a parallel-type geometry sat close enough to the volume box to
  touch it.  No test depended on the old value: the one test that names a
  factor passes 1.5 and 4.0 explicitly and checks only that a larger factor
  draws the source farther out.

Two more scene changes serve the comparison and the speed.
`GeometryScene.differences(other)` returns the differing parameters and derived
quantities with both values, `with_parameters(overrides)` returns a copy with
some parameters replaced, and `drawing_options()` carries the drawing choices
into that copy.  `trajectory()` now computes the whole scan at once instead of
building one `ViewScene` per view.

## How a redraw stays fast

The drawing layer was rebuilt around artists that live as long as the figure.
Every artist is created once in `_create_artists`, and a view change replaces
its data with `set_data`, `set_data_3d`, or `set_verts`.  Nothing is cleared and
nothing is created per view, with one exception: the 3D arrowhead is
matplotlib's own `quiver`, which builds an arrow from a start and a direction,
so it is made again for each view.

The panel limits are computed from a sample of at most sixteen views and then
held.  Holding them has two effects.  The ticks, the grids, the legends, and
the text block do not change with the view, so they can be painted once into a
background that a view change reuses.  The panels also stop moving under the
drawing while the slider is dragged.  A view whose content would fall outside
the limits makes them grow and forces one full repaint, which is the case a
sample can miss.

A view change then restores the cached background, draws the artists that
moved, and blits, which is the partial-redraw idea `mbirtorch/viewer.py` uses
for its slider rows.  The artists that move are marked animated, which is what
keeps a full repaint from drawing them twice; `save` unmarks them for the
duration of the write so that a saved image holds everything.  The fast path
runs only on the backends where it is verified, Agg and TkAgg, and only when
the canvas reports blit support.  Everywhere else, and whenever the background
has to change, the whole figure repaints.

Three drawings are still made from a description the scene gives rather than
from positions it returns.  The region of reconstruction is drawn as an ellipse
from its center, semi-axes, and heights.  A ray is sampled along its length,
because the 3D panel clips a line point by point and a two-point segment whose
ends are both outside the volume zoom would disappear.  A projected volume edge
that crosses the detector's boundary is sampled so that the red part starts
where the edge leaves the grid.  None of the three is a geometric statement
about the scan.

## Commands and results

Testing:

```
cd plans/experiments/geometry_viewer
MPLBACKEND=Agg PYTHONPATH=/home/user/cabouman/mbirtorch \
    /home/user/gv_env/bin/python -m pytest -q test_geometry_scene.py \
    test_geometry_viewer.py test_geometry_interaction.py
```

The result is 152 passed in 95 seconds, made up of 93 scene tests, 36 viewer
tests, and 23 interaction tests.  Increment 3 ended at 103 tests, so 49 are
new.  The scene gained 26 tests: the parallel corner rays, the drawing
distance, the arc and its sense in each geometry and with falling angles and
with one view, the vectorized trajectory against the per-view scenes for all
six geometries, and the comparison surface.  The 23 interaction tests are the
slider range and its effect on the drawing, the hidden slider of a single-view
scan, the source path as one polyline per panel, both toggles, the zoom cube,
the ten-channel comparison on the detector face and in the text panel, the
comparison following the slider, its removal, that construction never calls
`show`, that `show_geometry` passes `block` through, that a view reached by
stepping is pixel for pixel the view drawn from new, and that each of the six
geometries takes every control.

Two existing tests were adjusted rather than left to fail.  One counted the
figure's axes, which is now five panels plus three widget axes.  The other read
the detector-face line data, which now carries a row of NaN between the twelve
edges because they are drawn as one polyline.  One existing test was
strengthened: the overshoot test now checks that the overshoot line carries
points, because that line exists in every view and is empty when the volume
fits.

Timing:

```
cd plans/experiments/geometry_viewer
MPLBACKEND=Agg PYTHONPATH=/home/user/cabouman/mbirtorch \
    /home/user/gv_env/bin/python gv4_timing.py
```

Rendering the three figures:

```
cd plans/experiments/geometry_viewer
MPLBACKEND=Agg PYTHONPATH=/home/user/cabouman/mbirtorch \
    /home/user/gv_env/bin/python gv4_render_figures.py
```

## The gate: 100 ms per slider step

The measurement is an 1800-view helical cone scan built from parameters alone,
with a detector of 40 by 64, a reconstruction of 24 by 24 by 16, and 30 ALU of
helical travel.  One `set_view` call is one slider step and the time includes
the redraw.  The machine is a Linux container with 4 CPUs, which is slower per
core than the laptop the gate names.  `gv4_timing.md` holds the full table.

| what | before | after | gate |
| --- | --- | --- | --- |
| slider step, mean | 402 ms | 42 ms | 100 ms |
| slider step, worst of 20 | 494 ms | 44 ms | 100 ms |
| slider step with the path on, mean | not measured | 39 ms | 100 ms |
| `scene.trajectory()` over 1800 views | 505 ms | 0.3 ms | none |
| trajectory toggle | 882 ms | 270 ms | none |
| figure build | 462 ms | 848 ms | none |

The gate passes with about a factor of two in hand.  The "before" column is the
Increment 3 code on the same model and machine, timing `set_view` followed by a
canvas draw, because that `set_view` rebuilt the artists and left the drawing
to a later draw.  Rebuilding the artists alone cost 87 ms on average and 165 ms
at worst, so even the part that did no drawing was near the gate.

Two numbers moved the wrong way, and both are one-time costs.  A figure now
takes 0.8 s to build, because it draws the whole figure once to make the
background, measures where the text blocks end, and draws again.  A trajectory
toggle costs 270 ms, because the path changes the panel limits and so the
background; a toggle is a click and not a drag.

The same script also times a figure built with `blit=False`, which repaints the
whole figure on every step.  That path costs 221 ms per step and fails the
gate, which is where the five-fold difference comes from.  Most of the cost is
text: a full repaint renders about 1500 text artists, counting every tick
label, every legend entry, and the twenty-odd lines of the text panel, and the
text panel alone costs more to render than every line and marker of the four
drawing panels together.

## The three figures, reviewed by eye

### The comparison: ten channels of detector offset

![comparison with ten channels of offset](figures/gv4_compare_offset.png)

The flat cone geometry at view 2, against a copy whose `det_channel_offset` is
ten channels larger, which is -1.70 ALU against 9.30 ALU at a channel pitch of
1.10 ALU.

- The comparison's projected volume outline on the detector face is the
  primary's, moved along the channel axis: yes.  The render script prints the
  shift as 10.00 channels, and the test compares the two lines' data and finds
  exactly ten.
- The shift is visible in the picture and not only in the numbers: yes.  The
  purple outline runs from about channel 8 to channel 34 and the dashed olive
  one from about 18 to 44.
- The text panel lists the changed parameter with both values: yes.  It reads
  `det_channel_offset: -1.7 -> 9.3`, and under it `detector_center_u: 1.7 ->
  -9.3`, which is the same change stated as the position of the grid center,
  and `fan_angle_deg: 15.03808 -> 15.00761`, which is the fan turning slightly
  less symmetric.
- The comparison is visible in the other three panels: yes.  The dashed olive
  detector outline sits beside the blue one in the 3D view, the top view, and
  the side view, and the two central rays part company where the offset moves
  the grid.
- Where the comparison's central ray lands on the detector face is marked in
  the comparison color: yes, an olive circle at channel 32 against the black
  circle at channel 22.

### The 3D zoom to the volume

![the 3D panel zoomed to the volume](figures/gv4_zoom_volume.png)

The same geometry with the 3D panel in the `'volume'` state.

- The cube is about three volume extents wide: yes, 36.0 ALU for a volume box
  12.0 ALU across, against 247.4 ALU in the scan state.
- The volume box, the region of reconstruction, the rotation axis, and the
  voxel marker are legible: yes.  All four are a few hundred pixels across
  where they were about twenty in the scan state.
- The rays and the axis are clipped by the limits rather than squeezed: yes.
  The central ray crosses the box and stops at the panel's edge, and the
  detector and the source are outside the cube and are not drawn.
- The other three panels are unchanged by the zoom: yes, the zoom is the 3D
  panel's own control.
- One thing reads poorly.  The four corner rays are light gray, which was
  chosen so that they would not compete with the volume in the scan state, and
  at this scale against the panel's gray grid they are hard to see.  The
  central ray carries the beam direction instead.

### The 1800-view helical scan with the source path

![1800 views of a helical scan with the source path](figures/gv4_helical_1800_trajectory.png)

The model of `gv4_timing.py`, drawn at view 450 of 1800.

- The source path of 1800 views is one smooth curve and not 1800 markers: yes.
  It is one dash-dot line per panel, and the test checks that each panel holds
  exactly one such line, that it carries no marker, and that it holds 1800
  points.
- The path is a helix: yes.  The top view shows one circle, because the two
  turns lie on top of each other seen from +z, and the side view shows the path
  climbing 30 ALU, which the panel names as the source z range.
- The 3D view shows the two turns as two rings 30 ALU apart in z: yes.
- The volume does not fit the detector, and the panel says so: yes, "no (worst
  overshoot 19.2 px)", and the projected volume outline is drawn red where it
  leaves the grid.  This is the nature of a helical scan and not an error in the
  drawing: the volume spans 30 ALU of travel plus its own 8 ALU, and the
  detector sees 20 ALU at the rotation axis in any one view.  A helical
  reconstruction covers the volume with all of the views together.
- The volume box is invisible in the 3D panel at this scale: 12 ALU in a 500
  ALU cube.  The zoom control is the answer, and it is why the review asked for
  it.

## The Increment 3 figures, re-rendered

`gv3_render_figures.py` was run again, so that the eight committed figures show
the new scene.  Every checklist statement of `gv3_static_figure_findings.md`
still holds, with two numbers changed by the larger drawing distance.  The gv3
record is left as it was written.

- The parallel figure's drawn source is now at (4.17, 19.56, 0), one drawing
  distance of 20 ALU from the origin, where the record says (2.50, 11.74, 0)
  and 12 ALU.  The statement it supports still holds: the source sits to the +x
  side of the +y axis at view angle 0.21.  The farthest volume corner is 11.79
  ALU from the origin, so the drawn source now stands clear of the box.
- The multiaxis figure's drawn source is now at (3.24, 15.18, -4.80) and its
  detector origin at (-3.24, -15.18, 4.80), where the record says (1.94, 9.11,
  -2.88) and (-1.94, -9.11, 2.88).  The statement it supports still holds: a
  positive elevation puts the source below the xy plane and the detector center
  above it.
- The parallel and multiaxis rays are still parallel and not convergent, and
  they are now parallel because the scene says so rather than because the
  drawing layer worked around the scene.
- Every other statement is about a detector index, a projected outline, an
  angle, or a marker's corner, and none of those depends on the drawing
  distance.  The detector-face numbers the record quotes are unchanged.
- The panels are framed a little wider than they were, because the limits now
  hold over a sample of views instead of fitting the one view drawn.  Nothing
  is cut off, and the drawing is a few percent smaller in the 2D panels.

## Open items

- **The comparison section can run out of room.**  The derived quantities and
  the drawing note nearly fill the text panel, so the panel's font drops from
  7.5 to 6.0 points while a comparison is drawn, and the comparison section is
  capped at six entries and then at the lines that measurably fit.  A
  comparison between two different geometry kinds differs in about thirty
  entries, and the panel then shows one and counts the rest.
  `GeometryScene.differences(other)` returns them all, so nothing is lost to a
  caller.  A better answer would give the comparison its own panel or its own
  window.
- **The interactive path is untested on a display.**  This container has no
  display, so the partial redraw was exercised only under Agg, where a blit
  costs the same work but paints nothing visible.  The slider's own row is
  repainted the way `mbirtorch/viewer.py` repaints its slider rows, and that
  code path has not been watched on a screen.  Increment 6 should check the
  slider by hand on TkAgg before the port lands.
- **`derived_quantities` still walks the views.**  It costs 59 ms on 1800
  views, because `volume_fits_detector` projects the volume's eight corners one
  view at a time.  It is called once per figure and once per comparison.
  Vectorizing it would mean writing the projection a second time, which is what
  the plan's design rule warns against, so it was left alone.
- **The 3D arrowhead is rebuilt per view.**  `quiver` builds an arrow from a
  start and a direction, and there is no way to move one in place, so the
  artist is removed and made again on every view change.  It costs about a
  millisecond and it is the only artist that is not reused.
- **A comparison with fewer views holds its last view.**  The comparison
  follows the slider by index, and a comparison of a different length stops at
  its own last view rather than raising.  The text panel names the view it is
  holding.  Whether that is the right rule for comparing two scans of different
  lengths is open.
- **The comparison's pixel-0 marker is not drawn on the detector face.**  Pixel
  (0, 0) is index (0, 0) for both geometries, so the marker would sit on the
  primary's.  It is drawn in the three panels where the two differ.
- **The figures are not in the repository.**  This repository's `.gitignore`
  excludes `*.png`, and no PNG under `figures/` is tracked, including the eight
  the Increment 3 record shows.  The eight Increment 3 images and the three new
  ones total 2.5 MB.  Adding them takes
  `git add -f plans/experiments/geometry_viewer/figures`, and without that the
  images this page and the Increment 3 page review are on disk only.
