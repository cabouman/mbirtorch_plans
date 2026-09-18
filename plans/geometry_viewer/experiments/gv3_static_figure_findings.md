# gv3: the static geometry figure, as built

Date: 2026-09-09.  Files: `geometry_viewer.py` (the drawing layer),
`gv3_render_figures.py` (the render script), `test_geometry_viewer.py` (its
tests), and eight images under `figures/`.  Status: built; 36 viewer tests pass
and the 67 scene tests still pass.

## What was built

`geometry_viewer.py` draws a `GeometryScene` in five panels.  It computes no
geometry.  Every position it draws comes from `GeometryScene.view(v)`,
`trajectory()`, `volume_corners()`, `volume_z_range()`, `ror_cylinder()`, or
`uv_to_indices()`, and every number it prints comes from
`derived_quantities()`.  Two drawings are made from a description the scene
gives rather than from a position it returns, and both are named in the
Limitations section below.

The public surface is one class and one function.

- `GeometryFigure(scene_or_model, view_index=0, show_trajectory=False,
  figsize=(15, 9), title=None, elevation_deg, azimuth_deg)` builds the figure.
  `GeometryFigure.from_model(model)` builds one from a model instead.
- `set_view(index)` redraws the four drawing panels for another view.
  `set_show_trajectory(flag)` turns the source's path over all views on and
  off.  `save(path, dpi=110)` writes an image.  `figure` is the matplotlib
  `Figure`, and `panel_axes` is the five axes.
- `show_geometry(scene_or_model, ...)` builds a figure and opens a window.  It
  is the only place that calls `show`, and under a backend with no window it
  prints a line and returns.

The layout is a 2 by 3 grid.  The 3D view takes the whole left column, because
it needs the room.  The top view and the side view share the upper right, and
the detector face and the text panel share the lower right.

The panels are these.

- **3D view.**  This panel draws eight things: the source as a star, the
  detector outline with its face lightly filled for a flat panel, the four
  corner rays and the central ray, the volume box as its twelve edges, the
  region of reconstruction as an elliptic cylinder, the rotation axis as a
  dashed line with an arc and an arrowhead for the source's travel, the
  translation path for the translation geometry, and the two index markers.
  The source's path over all views is added when the trajectory is on.  The
  three axes share one scale in a cubic box, so an angle in the drawing is the
  angle in the geometry.
- **Top view**, the xy plane seen from +z.  The same scene projected, plus the
  region of reconstruction as an ellipse, the rotation axis as a point at the
  origin, and the detector's channel offset as a short thick segment with its
  name and value.
- **Side view**, the yz plane seen from +x.  The same scene projected, plus the
  z = 0 line as a dotted line, the volume's z center as a segment labeled
  `recon_slice_offset`, and the detector's row offset as a labeled segment.
- **Detector face**, channel index horizontal and row index vertical, with row
  0 at the bottom so that the row index increases upward as v does along +z.
  The grid runs from index -0.5 to index num - 0.5.  The twelve edges of the
  volume box are drawn from `volume_outline_on_detector`, in red where they
  leave the grid.  Three markers sit on it: where the central ray lands, the
  grid's center, and pixel (0, 0).
- **Text panel.**  Eighteen aligned `name : value unit` lines to three
  significant figures, then the view drawn, then the scene's drawing note.

The colors are one per element, in the module's `COLORS` dictionary, and each
element keeps its color in every panel: source orange, detector blue, rays
light gray, central ray near black, volume purple, region of reconstruction
cyan, rotation axis gray, detector pixel 0 magenta, voxel 0 green, trajectory
brown, and overshoot red.

## Commands and results

Rendering the eight figures:

```
cd plans/geometry_viewer/experiments
MPLBACKEND=Agg PYTHONPATH=/home/user/cabouman/mbirtorch \
    /home/user/gv_env/bin/python gv3_render_figures.py
```

The run takes 12 seconds and writes eight PNG files of about 200 kB each into
`figures/`.  The script takes no arguments.

Testing:

```
cd plans/geometry_viewer/experiments
MPLBACKEND=Agg PYTHONPATH=/home/user/cabouman/mbirtorch \
    /home/user/gv_env/bin/python -m pytest -q test_geometry_viewer.py
```

The result is 36 passed in 15 seconds.  The tests check seven things:

- importing `geometry_viewer` leaves `matplotlib.pyplot` unimported, which is
  checked in a separate interpreter because the test process has pyplot loaded;
- the first figure built does import pyplot;
- each of the six geometries builds a figure with five axes, redraws for
  another view, accepts the trajectory toggle, and saves a PNG above 10 kB;
- a view index outside the scan raises `IndexError`;
- the detector-face panel's twelve edges are entries of the scene's
  `volume_outline_on_detector`, compared both against the array the figure kept
  and against the line data read back out of the axes;
- a volume enlarged until it misses the detector is drawn partly in the
  overshoot color;
- the text panel carries the scene's numbers, with no negative zero.

Running the scene's own tests alongside these gives 103 passed.

## The figures, reviewed against the conventions record

Each figure below is followed by a checklist.  Each line states what the figure
should show under `reference/geometry_conventions.md` and whether it does.  The numbers
quoted are the scene's, printed alongside the render.

### Parallel beam, view 2

![parallel geometry](figures/gv3_parallel.png)

- At view angle 0.21 the source direction is (sin 0.21, cos 0.21, 0), so the
  drawn source sits to the +x side of the +y axis in the top view: yes.  It is
  drawn at (2.50, 11.74, 0), one drawing distance of 12 ALU from the origin.
- The rays are parallel and not convergent: yes.  This needed a change; see the
  Limitations section.
- The pixel 0 marker is at the low-channel, low-row corner of the detector,
  which is the -x, -z corner at angle 0 after the inverse rotation: yes.  It is
  drawn at (-24.80, -6.98, -7.50), and the low-channel end of the panel points
  toward -x turned by the view angle.
- A positive `det_channel_offset` of 1.35 places the detector grid center at
  u = -1.35: yes, and the text panel prints that value.
- The same positive offset moves the image toward a higher channel index: yes.
  On the detector face the central ray lands at channel 20.7 and the grid
  center sits at channel 19.5.
- The row pitch drawn is `delta_voxel` and not `delta_det_row`: yes.  The text
  panel prints a row pitch of 1 where `delta_det_row` is 0.9, and the drawing
  note says why.
- The rows are not shifted, because `det_row_offset` takes no part in a
  parallel projection: yes.  The grid center and the central ray land on the
  same row, 7.5.
- The volume's 16 slices fill the detector's 16 rows exactly: yes.  The
  projected volume outline runs from row -0.5 to row 15.5.
- Voxel 0 is at the -x, -y, -z corner of the volume box: yes, at
  (-5.50, -5.63, -7.50).

### Cone beam with a flat detector, view 2

![flat cone geometry](figures/gv3_cone_flat.png)

- The source is at (source_iso_dist sin 0.21, source_iso_dist cos 0.21, 0):
  yes, at (20.85, 97.80, 0) with `source_iso_dist` 100.
- The point where the central ray meets the detector lies opposite the source
  at `source_detector_dist - source_iso_dist` from the origin: yes, at
  (-20.85, -97.80, 0), which is 100 ALU from the origin.
- A negative `det_channel_offset` of -1.70 places the grid center at
  u = +1.70 and moves the image toward a lower channel index: yes.  The central
  ray lands at channel 22.0 and the grid center sits at channel 23.5.
- A positive `det_row_offset` of 0.65 places the grid center at v = -0.65 and
  moves the image toward a higher row index: yes.  The central ray lands at row
  12.0 and the grid center sits at row 11.5.
- `recon_slice_offset` of 1.4 moves the volume toward +z, which the side view
  shows against the z = 0 line: yes.  The volume's z range is -1.8 to 4.6.
- The magnification of a point at the origin is
  `source_detector_dist / source_iso_dist`: yes, 2, and the projected volume
  box on the detector face is twice the volume's size in ALU.
- The volume projects inside the detector: yes, the text panel says so and the
  detector face holds no red.
- The source travels clockwise seen from +z as the view index grows: yes, the
  arc in the top view turns from +y toward +x.

### Cone beam with a flat detector, view 0

![flat cone geometry at view 0](figures/gv3_cone_flat_view0.png)

- View 0 has a negative angle of -0.300, so the source sits on the -x side of
  the +y axis: yes, at (-29.55, 95.53, 0).
- The rotation arrow still points clockwise seen from +z, because the view
  angles grow with the view index: yes.
- The detector offsets and the projected volume box are unchanged from view 2
  in index coordinates, apart from the rotation of the volume: yes.  The
  central ray lands at channel 22.0 and row 12.0 in both figures.

### Cone beam with a curved detector, view 2

![curved cone geometry](figures/gv3_cone_curved.png)

- The detector is an arc in the top view and in the 3D view, curved in the
  channel direction only: yes.
- The arc lies on a cylinder of radius `source_detector_dist` about the source:
  yes.  Pixel 0 is drawn at (-70.56, -39.59, -29.40), which is 100.0 ALU from
  the source, and `source_detector_dist` is 100.
- The rows are spaced on the plane tangent to the cylinder at the central ray
  and not on the cylinder: yes, by construction in the scene.  The figure
  cannot show this difference, because the row positions of the two rules
  differ by less than a line width here.
- A positive `det_channel_offset` of 2.30 places the grid center at u = -2.30
  and moves the image toward a higher channel index: yes.  The central ray
  lands at channel 32.7 and the grid center sits at channel 31.5.
- A negative `det_row_offset` of -1.15 places the grid center at v = +1.15 and
  moves the image toward a lower row index: yes.  The central ray lands at row
  22.6 and the grid center sits at row 23.5.
- The fan angle is the detector arc divided by `source_detector_dist`: yes,
  73.3 degrees for an arc of 128 ALU and a radius of 100 ALU.
- The large fan angle and the short source distance make the projected volume
  box strongly perspective on the detector face: yes.  The magnification runs
  from 5.0 at the volume's near face to 3.1 at its far face, and the two drawn
  faces differ by that ratio of 1.6.

### Cone beam, helical scan, view 2

![helical cone geometry](figures/gv3_cone_helical.png)

- A positive helical z shift moves the object toward -z, so in this
  fixed-object drawing the source and the detector move toward +z: yes.  At
  view 2 the shift is 1.2 and the source is drawn at z = 1.2.
- The total helical travel is the range of the shifts: yes, 4.2 ALU, which the
  text panel prints.
- The axial field of view includes the helical travel: yes, 22.4 ALU against a
  detector height of 36.4 ALU at a magnification of 2, which is 18.2 plus 4.2.

### Cone beam, helical scan, view 2, with the trajectory on

![helical cone geometry with the trajectory](figures/gv3_cone_helical_trajectory.png)

- The trajectory rises toward +z with the view index: yes.  The source's z runs
  0, 0.6, 1.2, 1.8, 2.4, 3.0, 3.6, 4.2 over the eight views, and the side view
  shows the path sloping up toward +y.
- The trajectory is one polyline and not one marker per view: yes, it is a
  single dash-dot line in each panel.
- The top view shows the source's path turning clockwise seen from +z: yes.
- The side view names the source's z range: yes, 4.2 ALU in the lower right.

### Multiaxis parallel, view 2

![multiaxis geometry](figures/gv3_multiaxis.png)

- The rays are parallel and tilted out of the xy plane by the elevation of
  0.300 radians: yes, visible in the side view as four parallel lines climbing
  toward the detector.
- The source is placed on the +y side, which puts it below the xy plane for a
  positive elevation and the detector center above it: yes.  The source is
  drawn at (1.94, 9.11, -2.88) and the detector origin at (-1.94, -9.11, 2.88).
  This is the convention the scene documents and not a measurement, and the
  figure's drawing note says so.
- The detector plane is perpendicular to the direction of travel, so it leans
  by the elevation: yes, visible in the side view.
- A positive `det_channel_offset` of 1.85 places the grid center at u = -1.85
  and moves the image toward a higher channel index: yes.  The central ray
  lands at channel 25.2 and the grid center sits at channel 23.5.
- A negative `det_row_offset` of -1.35 places the grid center at v = +1.35 and
  moves the image toward a lower row index: yes.  The central ray lands at row
  12.1 and the grid center sits at row 13.5.
- A negative `recon_slice_offset` of -0.9 moves the volume toward -z: yes.  The
  volume's z range is -5.5 to 3.7 and its center is at -0.9.
- The azimuth turns the same way as a parallel or cone view angle: yes, the
  arc in the top view turns clockwise seen from +z.

### Translation, view 2

![translation geometry](figures/gv3_translation.png)

- The source and the detector sit as in the flat cone geometry: yes, the source
  at y = 100 and the detector plane at y = -100 before the translation.
- Each view moves the object by minus the translation vector, so in this
  fixed-object drawing the source and the detector move by plus the vector:
  yes.  At view 2 the vector is (0.7, 0.5, 0.3) and the source is drawn at
  (0.70, 100.50, 0.30).
- There is no rotation axis, and the path of the translation vectors takes its
  place: yes.  The legend names a translation path and no rotation axis, and
  the top view labels the path.
- The volume is centered on z = 0, because this model has no
  `recon_slice_offset`: yes, the text panel prints a volume z center of 0 and
  the side view draws no offset segment.
- A negative `det_channel_offset` of -2.15 places the grid center at u = +2.15
  and moves the image toward a lower channel index: yes.  The central ray lands
  at channel 21.6 and the grid center sits at channel 23.5.
- A positive `det_row_offset` of 1.45 places the grid center at v = -1.45 and
  moves the image toward a higher row index: yes.  The central ray lands at row
  12.7 and the grid center sits at row 11.5.

No figure disagrees with `reference/geometry_conventions.md`.

## What the review changed

Four things were wrong or illegible in the first renders, and each was fixed.

The parallel and multiaxis geometries drew four rays converging on a point.
The scene's `corner_rays` run from `source_draw` to the four detector corners,
and `source_draw` is a finite stand-in for a source that is infinitely far
away.  The picture was therefore a cone beam.  The viewer now draws those four
rays parallel for a geometry with no source, using the ray direction and the
drawing distance the `ViewScene` itself carries.

The 3D panel's box was shaped like the data.  The cone geometries then drew as
a long thin tunnel, because their source-detector distance is 200 ALU and their
volume is 12 ALU across.  The camera looked along that tunnel, so nothing in it
could be told apart.  The box is now a cube with one common scale on the three
axes.  That costs empty space along the short axes and keeps every angle true.

The 3D camera looked almost straight down the y axis.  At that azimuth the y
axis and the z axis both run up the screen, and a detector standing upright
looked like one lying flat.  The camera now sits 22 degrees above the xy plane
at an azimuth of -70 degrees, which is 20 degrees around from the -y axis.  The
source at +y is still at the back of the picture at view angle zero.

The side view's labels sat on top of one another.  A cone geometry's side view
is about five times wider than it is tall, so there is little vertical room for
text.  The three labels now go at three places along the panel: the detector's
at the near end, the volume's in the middle, and the trajectory's in the far
corner.  The "z = 0" text was dropped, because the dotted line and the z axis
ticks say the same thing.

## Limitations, and items for Increment 4

Two drawings are made from a description rather than from a scene position.
Both are documented in the code and both are candidates for moving into the
scene.

- The rotation-direction arc is drawn on the circle the source travels on, at
  the source's own radius and height, sweeping 0.5 radians.  Its direction
  comes from the sign of the step between the current view's angle and the next
  one, so a model whose angles fall gets an arc the other way.  The scene has
  no primitive for this arc.
- The parallel corner rays are drawn from the `ViewScene`'s ray direction and
  from the distance between its drawn source and its detector origin.  A
  primitive for the rays of a source-free geometry would belong in the scene.

Three things are drawn to scale and are therefore small.

- The volume box in the 3D panel is 12 ALU across in a 200 ALU cube for the
  cone geometries, so it is about twenty pixels wide.  The plan anticipates
  this: the three 2D panels carry the quantitative content and the 3D panel is
  for orientation.  A zoom control or a second 3D panel scaled to the volume
  would fix it.
- The translation path is a few ALU across in the same 200 ALU frame, so it is
  a small squiggle beside the volume.  The top view labels it.
- The detector offset segments are one or two ALU long in the same frame, so
  they are a few pixels long.  Their labels carry the value, and the text panel
  prints the grid center in (u, v).

Four smaller items are open.

- The side view of a cone geometry is a wide short strip, because equal aspect
  is kept and the panel's width is the source-detector distance.  The plan
  lists a separate scale for this panel as an option.
- `set_view` clears and redraws the panel axes, which is fast enough for these
  eight-view models but was not measured against the plan's gate of 100 ms per
  step for 1800 views.  Increment 4 owns that gate and the slider.
- `trajectory()` builds one `ViewScene` per view, so the trajectory toggle
  costs 1800 scene builds on a 1800-view model.  The figure caches the result,
  so the cost is paid once.  Increment 4 should measure it.
- The figures are PNG files, and this repository's `.gitignore` excludes
  `*.png`, so they are not tracked (Greg, 2026-09-09).  The image links on this
  page work in a checkout that has run `gv3_render_figures.py`, which
  regenerates all eight in about twelve seconds.
