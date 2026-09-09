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

## Display test follow-up, 2026-09-09

Greg ran `gv_show_example.py` on a machine with a display and reported three
things.  The 3D view showed no source and no detector, and moving the slider
changed nothing in it.  The detector panel was not labeled.  The drawing did
not show the direction the geometry projects at angle 0, which is the position
a user reads a view angle from.

This section records what each report led to.  The first was a drawing-layer
error, and the lead diagnosed and fixed it in the committed code.  The other
two are new drawing: labels on the source and the detector, and a fixed
angle-0 reference.  Files changed: `geometry_scene.py`, `geometry_viewer.py`,
`test_geometry_scene.py`, `test_geometry_viewer.py`, and
`test_geometry_interaction.py`.  Status: 171 tests pass, and a slider step on
the 1800-view helical model costs 50 to 53 ms against a gate of 100 ms.

### Why nothing that moves was drawn

The cause was that every moving artist was marked animated for the life of the
figure.  A full draw skips an animated artist, and only the partial redraw
paints one.  The partial redraw runs on the backends where it is verified, Agg
and TkAgg, and Greg's session ran on the macosx backend.  On that backend
nothing that carries the current view was ever painted.  What his window showed
is exactly the static artists: the volume box, the region of reconstruction,
the rotation axis, the voxel marker, the detector grid, and the text panel.  A
saved file looked right, because `save` unmarks the artists for the duration of
the write.

The rule that fixed it is `GeometryFigure._animate_moving`.  A moving artist is
animated only when blitting is enabled, the backend is in `BLIT_BACKENDS`, and
the canvas reports blit support.  Everywhere else the moving artists stay
ordinary artists, and a full draw paints them.  Two tests at the end of
`test_geometry_interaction.py` hold the rule.  One drives a figure built with
`blit=False` through a plain draw and checks that the draw paints what a draw
with every artist unmarked paints.  The other checks that the artists are
animated under Agg with blitting on and are not animated with it off.

Every artist added below obeys that rule without a second mechanism.  A moving
artist is registered in the `_moving` list, and `_set_animated` walks that
list, so a new label is animated exactly when the source and the detector
are.  A
view-independent artist is an ordinary static artist and is not in the list at
all.  `test_the_labels_are_animated_with_the_other_moving_artists` and
`test_the_reference_artists_are_not_animated` check the two cases.

### The labels

The source and the detector now carry a text label in the 3D view, the top
view, and the side view, and the detector's pixel-0 marker carries one in the
top view.  Each label is set at `ANNOTATION_FONT_SIZE` and takes the color of
the element it names.  A label moves with its element, so it is a moving
artist.  The two 3D labels are moved with `set_position_3d`, which matplotlib
3.11 provides.  The five 2D labels are annotations, and a view change moves the
data point each one is attached to.  The vertical part of a 2D label's offset
from that point is fixed when the label is created.  The horizontal part is
chosen per view by the rules below.

Where each label sits was chosen to keep the labels of one panel apart.  The
rules are these:

* The source's label sits beside the source, offset a little farther than the
  others.  The central ray ends at the source, and a smaller offset put the
  label on that line.  In the side view it sits below the source, because the
  `recon_slice_offset` label is above it there.
* The pixel-0 label sits higher above the detector than the detector's own
  label, so that the two are on separate lines when the detector is short
  against the width of the panel.
* The detector's label sits at the end of the detector farthest from the
  pixel-0 marker.  That keeps it off the middle of the panel, where the rays
  and the offset segment are, and away from the pixel-0 label.
* In the top view the detector's label reads outward from its end of the
  detector.  The channel-offset label sits at the middle of the detector and
  reaches toward that end, and a label reading inward ran into it.
* The source's label and the pixel-0 label read toward the middle of their
  panel.  Both name a point that can sit at the edge of a panel, and a label
  reading outward there ran off the panel and onto the tick labels.
* A comparison's detector is named "comparison" once, in the top view, below
  the detector line.  A label in every panel would say the same thing four
  times.
* A 3D label is hidden when the point it names is outside the 3D panel's cube,
  which `_clip_3d_artists` does.  The by-eye section below says why.

### The angle-0 reference

`GeometryScene.reference_view` is a new method.  It returns the `ViewScene` of
the geometry with the view action set to the identity, which is where the
source and the detector sit at angle 0.  The identity differs by geometry kind:
angle 0 and z shift 0 for the parallel and cone geometries, azimuth 0 for the
multiaxis geometry, and a zero translation vector for the translation geometry.

The multiaxis reference keeps the elevation of view 0.  A multiaxis geometry
has no position free of elevation, because its rays are tilted out of the xy
plane in every view, so the azimuth alone goes to zero.  The docstring says so.

The method builds the reference by replacing view 0's own view parameters with
the identity and taking that copy's view 0.  Every entry of the result
therefore means what the same entry of `view` means.  A scan whose view 0 is
already the identity gets its own view 0 back, which
`test_reference_view_is_view_zero_when_view_zero_is_the_identity` checks entry
by entry to within 1e-9.  For the probe's flat cone scan, whose first angle is
-0.3 radians, the reference source sits at (0, source_iso_dist, 0) while the
view-0 source does not.

The viewer draws the reference in the 3D view and the top view.  Four things
are drawn: the reference source as a hollow star in the source color, the
reference detector outline dotted in the detector color, the reference central
ray dotted from that source to that detector origin with an arrowhead at the
detector end, and a label on the arrow.  All four are drawn at an alpha of
0.45, so that the reference cannot be read as part of the view drawn.  The top
view's label is "projection direction, angle 0" on two lines, and the 3D view's
is the short form "angle 0".  The translation geometry has no view angle, so
its two labels say "no translation" instead.

The side view and the detector face get no reference.  The side view is the yz
plane, in which a rotation about z moves nothing, so a reference drawn there
would lie on top of the view drawn.  The detector face is in index coordinates,
which the view action does not change at all.

The reference artists do not depend on the view, so they are static artists and
they belong to the background.  Turning them off therefore repaints the whole
figure, which is what the source-path toggle does for the same reason.  The
panel limits measure the reference whether or not it is drawn, because the
limits are computed once and a reference outside them would be cut off when the
toggle turned it on.

The reference is on by default.  `GeometryFigure` takes `show_reference=True`,
`set_show_reference` turns it on and off, and a third `CheckButtons` labeled
"angle-0 reference" sits beside the other two toggles.  The view slider is
shorter than it was to make room for that toggle.

One case is worth knowing.  A scan whose view 0 is at angle 0 draws the
reference on top of its own view 0, and the two separate as soon as the slider
moves.  `gv_show_example.py` builds such a scan and opens at view 0.

### What a slider step costs now

A slider step on the 1800-view helical model of `gv4_timing.py` costs 50 to
53 ms on average, against the gate of 100 ms.  Three runs of the script gave
averages of 50, 50, and 53 ms.  The worst single step in the three runs was
66 ms, in a run with the source path drawn.  The step cost 42 ms on average
before this change.  These results indicate that the seven new labels cost 8
to 11 ms per step.  The gate still passes with about a factor of two in hand.
The reference costs a step nothing, because it is a static artist and a step
does not redraw it.

### The by-eye check of the figures

The three figures the follow-up names were read on screen: `gv3_cone_flat.png`,
`gv4_compare_offset.png`, and `gv3_multiaxis.png`.  The other eight figures
were read as well, because the label rules had to hold for every one of them.
Overlapping labels were the problem in every case but one, where a label was
drawn outside its panel.  Each change is listed here with the figure that
showed the problem:

* The source's label sat on the central ray in the side view of
  `gv3_multiaxis.png`, because that ray ends at the source.  Its vertical
  offset is larger than the other labels' now.
* The source's label then ran into the `recon_slice_offset` label in the side
  view of `gv3_cone_curved.png`, where both sat just above the z = 0 line.  In
  the side view it sits below the source instead of above it.
* The detector's label sat at the center of the detector, where the offset
  label and the pixel-0 label already were, in every figure.  It moved to the
  end of the detector farthest from the pixel-0 marker, in the 3D view and in
  both projected views.
* The top view's detector label then ran into the channel-offset label in
  `gv3_parallel.png`, where the detector is tilted and a vertical offset does
  not separate the two.  It reads outward from its end of the detector now.
* The pixel-0 label ran onto the tick labels of the top view in
  `gv3_multiaxis.png` when it read outward, so it reads inward.
* The pixel-0 label also ran into the channel-offset label in
  `gv3_translation.png`, where both sat below the detector.  It sits above the
  detector now, and the channel-offset label still sits below.
* The pixel-0 label then ran into the detector's own label in
  `gv4_helical_1800_trajectory.png`.  That scan's detector is 53 ALU wide in a
  panel 500 ALU across, so its two ends are close together and no horizontal
  rule separates two labels of 50 ALU.  The pixel-0 label sits higher above
  the detector than the detector's label does now, which puts the two on
  separate lines whatever the panel's scale.
* The top view's reference label ran off the right edge of the panel as one
  line in `gv3_cone_flat.png`.  It is two lines now, and it sits a fifth of the
  way back along the reference ray instead of on the arrowhead.
* The 3D view's reference label sat on the detector's own label in
  `gv3_cone_flat.png`, so it moved to the middle of the reference detector's
  low-row edge.
* The comparison's label ran into the channel-offset label in
  `gv4_compare_offset.png`, so it hangs farther below the detector than the
  primary's labels do.
* The 3D labels were drawn outside their own panel in `gv4_zoom_volume.png`,
  which is the volume zoom.  Matplotlib clips a 3D line to the axes limits
  when the line asks for it, and it clips neither a 3D text artist nor an
  arrowhead built by `quiver`.  The source sits far outside the volume cube,
  so its label landed on the figure beside the panel, and the reference's
  arrowhead drew a stray segment there.  `_clip_3d_artists` hides each of
  those artists when the point it is attached to is outside the cube.  It
  covers the source's label, the detector's label, the reference's label and
  its arrowhead, and the source-travel label and arrowhead that were already
  on the page.

After those changes no label in any of the eleven figures overlaps another
label, and none is drawn outside its own panel.  Two labels that were
already on the page sit close together in the multiaxis side view,
`recon_slice_offset -0.9` and `det_row_offset -1.35`.  They are on separate
lines and both are readable, and neither is new, so they were left alone.

### Open items from this follow-up

* **The example script's docstring names two toggles.**
  `gv_show_example.py` describes the view slider and two toggles, and there are
  three now.  That file was outside this change.
* **The reference has not been seen on a display.**  This container has no
  display, so the reference and the labels were checked under Agg and in saved
  files only.  What Greg's report showed is that Agg and a windowed backend can
  differ, and the rule that caused that difference now has two tests, but the
  window itself is still unwatched here.
* **The reference is drawn for a geometry that has no view angle.**  The
  translation geometry's reference is the gantry at zero translation, and its
  labels say "no translation".  Whether a fixed reference earns its space in a
  translation scan, whose views differ by a few ALU, is worth Greg's opinion.
* **A label can still be crowded at a small view angle.**  In the 3D view the
  reference and the view drawn coincide when the view angle is near zero, so
  their labels sit close together.  At view 2 of the probe scans, which is 12
  degrees, they are legible and adjacent.  A viewer that hid the reference's
  labels when the two nearly coincide would read better, and that is not built.
