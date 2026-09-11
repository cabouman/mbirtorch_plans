# gv6: the open items of the Increment 4 review

Date: 2026-09-10.  Files: `geometry_scene.py`, `geometry_viewer.py`, `web/app.py`,
`test_geometry_scene.py`, `test_geometry_viewer.py`, `test_geometry_interaction.py`,
`test_web_app.py`.

Five open items of the plan were worked in one day, after the Space was fixed.  The
fit statement now tests the region of reconstruction and has a rule for helical
scans.  The comparison has a window of its own.  The web page has two camera
fields.  The side view's `recon_slice_offset` label moves out of the source's way.  A
sixth item, the slider's first test on a screen, was closed by Greg's own use of the
viewer.  Each section below gives what changed, the evidence, and what was found on
the way.  Opus wrote the code of the first, second, and fifth items and of the
comparison window from a specification, and the specification's author reviewed the
diffs and the figures.

## The fit statement tests the region of reconstruction

`volume_fits_detector` projected the eight corners of the reconstruction box.  The
automatic box is the square around the field-of-view circle, so its corners project
past the detector by construction, and every automatically sized cone scan read
"volume fits det : no".  The statement now tests the shape the reconstruction
solves for.  When `use_ror_mask` is True that shape is the elliptic cylinder that
`ror_cylinder` describes, and when there is no mask it is the box.

The cylinder is tested through its two rims.  Each rim is sampled at 90 points, and
the two rims bound the whole cylinder's projection.  Two facts give that bound.  A
point's channel index does not depend on its z in any of the four geometries.  Its
row index is an affine function of z at fixed x and y, so along every line of the
cylinder parallel to the axis the row index is extreme at one end, and both ends
lie on a rim.  The projection is the scene's `project_points`, the one function that
agrees with the projector, so the statement adds no geometry of its own.  The
detector face draws the two projected rims in the color of the region of
reconstruction, split at the detector's edge the way the box edges are split.  The
panel therefore shows the shape the statement is about.

The numbers for the default cone scan, a 96 by 128 detector at a source-detector
distance of 512 ALU and a source-isocenter distance of 256 ALU, with the automatic
reconstruction geometry:

```
shape       worst channel overshoot    worst row overshoot
box              27.9 px                   10.3 px
cylinder          0.00 px                   6.80 px
```

The cylinder fits in channels and misses in rows.  The rows miss because the beam
narrows toward the source.  The top rim on the source side is closer to the source
than the isocenter is, so it is magnified more and projects above the detector's
first row.  That miss is a real property of a cone-beam scan whose volume is sized
from the field of view at the isocenter.  The honest statement for this scan is
therefore "lateral fit : yes (cylinder)" and "axial fit : no (6.8 px over)".  Whether a second statement about the field of view at the isocenter would
serve users better is a question for Greg.

The cylinder's semi-axes were corrected on the way.  The scene had drawn the
ellipse at n/2 voxel pitches, half the volume's physical width, and with that
ellipse the default cone scan missed by 0.50 channel.  mbirtorch's mask keeps the
voxels whose centers lie inside the ellipse at (n - 1) / 2 pitches, half a voxel
smaller.  The Increment 5 work forward-projected a volume of ones inside the mask
and compared the shadow's channel edges with both ellipses over the views of three
geometries.  The shadow matches the mask's ellipse to 0.1 channel on average, and
the larger ellipse misses by about one channel.  The scene now uses the mask's
ellipse, which is the projector's own, and the half channel is gone.  The record is
`gv7_data_overlays_findings.md`.

The text panel gives the statement in two rows, "lateral fit" and "axial fit",
because one answer would hide which direction misses (Greg, 2026-09-11).  The
lateral row is about the channels and names the shape tested.  The axial row is
about the rows or, for a helical scan, about the swept coverage explained in the
next section, whose range it prints.  A third row, "leaves det in views", counts
the views in which the shape reaches past any detector edge, which separates a
scan that misses everywhere from one that misses in a few views.

## The helical rule

A helical volume is taller than one view's detector by design, so it leaves the
detector in every view, and demanding that every view contain it is the wrong
question.  `fit_report` applies a second rule when the geometry is a cone scan with
a nonzero helical travel.  The scan fits when the shape stays inside the detector's
channel range in every view and the detector's axial coverage, swept over the scan,
contains the volume's z extent.

The axial coverage of one view comes from `project_points` as well.  The two points
(0, 0, z_min) and (0, 0, z_max) on the line x = y = 0 are projected.  The row index
is an affine function of z along that line, so the two rows give the z at the
detector's first and last row edge.  The union of the per-view intervals is merged
interval by interval.  A scan that leaves a gap between two groups of views
therefore reads "not covered", even though the gap lies inside the swept range.  A test with such a
gap checks that.

For the probe's helical configuration the swept range is -9.8 to 12.6 ALU, which is
one view's 18.2 ALU plus the 4.2 ALU travel.  Its volume leaves the detector in no
view, and the statement reads "yes (helical rule)".  An automatically sized helical
scan of 180 views with a 50 ALU travel sweeps -49 to 49 ALU and covers its volume.
It leaves the detector in 180 of 180 views, because each view sees only part of the
travel, and it reads "yes (helical rule)".  The count of views is all or nothing for a helical scan
that misses in rows, because every view sees the same fraction of the travel.

## The side view's `recon_slice_offset` label

The label was a fixed annotation above the end of its segment, and the multiaxis
example drew the source's marker over two of its characters.  The label is now a
moving artist.  Each view places it at the segment's end on the side away from the
source, reading outward from the volume box.  The gap is one line rather than the
six points other labels use, for two reasons.  The source marker reaches seven
points past its point.  The row-offset label reads back toward the volume at nearly
the same height.

The label overlap test grew with it.  It now checks the source marker's box against
every label, and it runs the side view at three views per geometry, so the moving
label is checked where the source has moved.  Run on the top view at three views,
the same check finds seven collisions that predate this work, and the plan lists
them as a new item.

## The camera fields of the web page

The web page shows the figure as an image, so the 3D panel cannot be turned with
the mouse.  Two number boxes under the toggles now set the camera's elevation and
azimuth in degrees, with the viewer's defaults of -25 and -40.  A blank box keeps
the default.  A change costs one render, as every control change on the page does,
so the page is no slower for having them.

## Two things found on the way

The extra rows of the fit statement exposed an ordering fault in the text panel.
`_place_text_blocks` cut the comparison section to the lines the panel had room for
in the same pass that shrank the font.  A cut is never taken back, so the section
lost entries the smaller font would have had room for.  The cut now happens
only in a pass that left the font alone, and the comparison test that failed on
the extra rows lists three entries where it listed one.

`derived_quantities` on an 1800-view helical scan takes 30 ms against 23 ms before,
because each view projects 182 points instead of eight.  A slider step on the same
model takes 23.0 ms against 20.2 ms, under the 100 ms gate.

## The comparison window

The text panel listed the differences between the two geometries in a block that
was capped at six entries and cut further to the lines the panel had room for.
At the smallest font the multiaxis example showed no entry at all.  A comparison
now opens a second figure, `GeometryFigure.compare_figure`, which tables every
difference `GeometryScene.differences` returns.  Each row holds the name, the
primary geometry's value, and the comparison's value.  The parameters come first,
then a rule, then the derived quantities.  The window's height follows the row
count up to ten inches, after which the rows left out are counted in a last row.
The hardest case tried, a helical cone scan against a translation scan, has 30
rows and fits.  The text panel keeps what a calibration user must see beside the
drawing: the header, the parameters that differ, and one sentence counting the
derived quantities that differ and pointing at the window.  `save` writes the
window beside the main file with `_comparison` added to the name, and removing
the comparison closes the window.

On the web page the window is a second plot under the status line, shown only
while a comparison is drawn.  Gradio 5.45 accepts an update that carries a
figure and a visibility flag for a `Plot` output, which was checked by running
the page's own function through Gradio's output processing.

The multiaxis example still reaches the text panel's font floor of 5 points
with a comparison, because its drawing note is six lines long.  At the floor the
static block's last line touches the comparison header.  That is a
placement fault that predates the window; the plan lists it.

## The top view's labels and the detector-face legend

The extended overlap test had found seven collisions in the top view at other
views, and Greg found that the detector-face legend covered the panel's content.
A placement pass on 2026-09-11 addressed both.  The pass was guided by Greg's rule:
remove the systematic collisions with placement rules, but a label that is clear in
most views and touches another in a few is acceptable, because the labels move
with the view.

Three rules and one wording change did the work.  The pixel (0, 0) label sits on
the side of its marker away from the detector iso, so that it reads outward from
the end of the detector while the iso's labels stack inward.  The "source travel"
label sits on the side of the arc's end away from the source.  The two offset
labels read "chan offset" and "row offset" instead of the full parameter names,
which stay in the text panel; the shorter row label also removed two side-view
collisions in the multiaxis example.  Counting label-label, marker-label, and
out-of-panel faults over the three panels: at the three views the test measures,
11 before and 0 after; over all eight views of the probe configurations, 27 before
and 7 after.  The test now measures the top view at view 0, the view a figure
opens with, and the side view at three views.  The seven that remain are all at
views the test does not measure: the channel-offset label over the pixel (0, 0)
label at view 5 of the flat and helical cone scans, where the detector is drawn
under the angle-0 caption; the pixel (0, 0) label 2.3 pixels past the panel's top
edge at view 3 of the curved cone scan; the detector's label over the angle-0
caption at view 6 of the parallel scan; and the source's label over the detector's
at view 5 of the two cone scans in the side view.

The detector-face legend sits in a band under the panel, two rows of three
entries.  The panel gives up the bottom of its grid cell for the band, and because
the panel holds an equal aspect, only the curved cone scan's panel shrinks, from
3.53 by 2.79 inches to 3.06 by 2.42.  The legend's place was measured against the
five panels, the slider, the toggles, and the figure's edge in every geometry, with
and without a comparison and a sinogram, at the desktop size and the web size, and
it overlaps nothing.  A new test repeats that measurement.

The text panel's font floor was reached less often after the same day's fit rows
added a line.  The scene's drawing notes were shortened, the two convention notes
share one line, and the comparison block drops its sentence about the derived
quantities before it drops a parameter entry, since the window holds everything.
At the web figure's size the cone scan with a comparison keeps its parameter line
and the sentence, at the 5 point floor, and the multiaxis and parallel examples
with a comparison sit above the floor at 5.1 points with every block whole.

## Tests

The suite grew from 229 to 255 tests, and all pass with gradio 5.45.0.  The web
tests also pass under matplotlib 3.8.4, the release inside Pyodide.  The new tests
cover these things:

- the fit statement: the cylinder-against-box answer, the automatic cone scan's
  overshoot, two swept-coverage identities, the helical rule, a helical scan with
  a gap, the rim outline, and the fit points without a mask;
- the text panel and the detector face: the three fit rows, the swept-z row, and
  the region's line on the detector face;
- the labels: the source marker against every label of the side view at three
  views;
- the comparison window: its contents and order, its closing, the companion
  file, and the text panel's shortened block;
- the web page: the camera fields and the third output of `render`.
