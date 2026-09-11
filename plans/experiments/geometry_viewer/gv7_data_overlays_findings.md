# gv7: the data overlays (Increment 5)

Date: 2026-09-11.  Files: `geometry_viewer.py`, `geometry_scene.py`,
`test_geometry_viewer.py`, `test_geometry_interaction.py`, `gv_show_example.py`.

The viewer can now draw data beside the geometry.  A sinogram is painted on the
detector face, one view at a time, and a reconstruction or a phantom is drawn as a
silhouette in the volume box of the top view and the side view.  Both are optional
arguments of `GeometryFigure` and of `geometry_viewer`, and both can be added or
removed later with `set_sinogram` and `set_recon`.  The plan's gate, that a
synthetic sinogram with one bright pixel lands at its row and channel, passes on
the rendered pixels.  A forward projection through mbirtorch's own projector then
tied the painted sinogram to the drawn region of reconstruction, and it corrected
the region's radius on the way.  Opus wrote the code from a specification, and
the specification's author reviewed the diff and the figures and made the
measurement of the last section.

## The sinogram on the detector face

The detector face paints `sinogram[view]` with `imshow` in gray, at an extent that
puts array element (r, c) at channel c and row r, with `origin='upper'`.  The
panel's row axis is inverted, so row 0 is drawn at the top, which is how `imshow`
shows one view of a sinogram and how the panel already drew its grid.  The gray
scale is fixed over the whole array, so stepping through the views compares them.
The image is a moving artist: a view change replaces its data and nothing else, and
it takes part in the partial redraw with the other moving artists.  The image sits
above the detector's translucent rectangle and below every line and marker, so the
projected outlines of the volume box and of the region of reconstruction stay
readable over it.  The panel's title says "with sinogram" while one is drawn, and
the text panel's footer names the overlays drawn.

The gate test builds a sinogram of zeros with one bright pixel, moves the slider to
its view, renders the figure, finds the brightest pixel inside the detector panel,
and converts its display position back to data coordinates.  The result rounds to
the pixel's channel and row.  This check goes through the screen, including the
inverted row axis, and not through the array alone.

## The silhouette in the volume box

The silhouette is the support of the array, the voxels whose absolute value exceeds
a threshold, and the default threshold is a tenth of the largest absolute value.
The top view draws the support projected along z and the side view draws it
projected along x, each as a translucent fill in the volume's color through a
masked array, so the outside of the support is transparent.  The extent of each
image comes from `scene.voxel_centers` at the corner voxels, half a pitch beyond
their centers, so the drawing adds no geometry of its own.  The silhouette is
static, because the object is the thing the drawing holds fixed.  The 3D panel
draws neither overlay.

The gate tests place a 2 by 2 by 2 block of voxels at the low corner of the volume
and at the high corner, in the multiaxis configuration, whose voxel pitches differ
and whose `recon_slice_offset` is not zero.  Each test renders the figure with and
without the silhouette and takes the pixels that differ, because the fill's color
composited over white is the color of the volume box's antialiased outline and
color matching cannot tell the two apart.  The patch's center, converted to data
coordinates, lies within one voxel pitch of the block's center in both panels, and
at the low corner it lies at the marker of voxel (0, 0, 0) that the panels already
draw.  A single voxel was too small for this check: it hides under the box's
outline, and in the cone configuration it is narrower than a pixel.

## The projector ties the sinogram to the region of reconstruction

A phantom of ones inside mbirtorch's mask was forward projected with the real
model of the probe's flat cone configuration, and the painted sinogram's lit extent
was compared with the projected rims of the region in every view.  The comparison
found the drawn region half a voxel too large.  `GeometryScene.ror_cylinder` had
used semi-axes of half the volume's physical width, which is n/2 voxel pitches,
while mbirtorch's mask, `vcd_utils.get_2d_ror_mask`, keeps the voxels whose centers
lie inside the ellipse at (n - 1)/2 pitches.

The two candidates were then compared directly.  A volume of ones inside the mask
was forward projected in the flat cone, the curved cone, and the parallel
configurations, and the channel edges of the shadow, at a tenth of the largest
value, were compared with the projected rims of each ellipse in every view:

```
configuration    ellipse         mean edge disagreement    worst
cone flat        n/2 pitches      -1.10, +1.01 channels     1.47
                 (n-1)/2          -0.12, +0.03              0.62
cone curved      n/2              -1.12, +1.04              1.79
                 (n-1)/2          -0.07, -0.01              0.81
parallel         n/2              -0.30, +0.75              0.89
                 (n-1)/2          +0.19, +0.26              0.33
```

The shadow matches the mask's ellipse.  The scene now uses it, and the test that
pins the cylinder's semi-axes asserts (n - 1)/2.  This changed the fit statement of
the default cone scan: its cylinder had missed the detector by 0.50 channel, which
was exactly this half voxel, and now fits in channels and misses only in rows.
`vcd_utils.get_support_radius` stays half a voxel larger on purpose, because it
bounds the outer edge of every voxel the projectors update; the fit statement asks
a different question, where the mass is.

The projector test keeps a tolerance of two channels and one row, for a reason
that is now stated correctly in its docstring.  The rims run through the voxel
centers, the voxels' material reaches half a voxel further, which is 0.91 channel
in that scan, and the projector's footprint spreads the mass by up to half a
channel more.  Any pixel with some mass counts as lit in that test, so the lit edge
lies outside the rims by up to about 1.4 channels.

## Timing

A slider step on the 1800-view helical scan of `gv4_timing.py`, with a sinogram
painted, takes 25.0 ms on average and 25.7 ms at worst over ten steps on this Mac,
against 22.9 ms without an overlay and the 100 ms gate.  The sinogram costs about
2 ms per step.  The container `gv4_timing.md` was measured in reported 41.8 ms
without an overlay, so the gate holds there with the same margin.

## What the example shows

`gv_show_example.py` has two new constants, `SHOW_SINOGRAM` and `SHOW_RECON`, both
False.  With either on, it builds mbirtorch's cube phantom, forward projects it
with the model, and passes the arrays to the viewer.  In the review figure of the
flat cone configuration the phantom's shadow sits inside both drawn outlines with
row 0 at the top, and it slants toward higher channels as the row index grows,
which is the cube phantom's own shift with the slice index seen the right way up.
The silhouette sits inside the volume box in both panels.  In a cone geometry those
panels span the whole scan, about 220 ALU, so a 12 ALU volume is a twentieth of the
panel and the silhouette is a few pixels wide; the parallel-type geometries draw
the panels a few volume widths across, and the silhouette is plain there.

## Tests

The suite grew from 255 to 266 tests, and all pass with gradio 5.45.0.  The new
tests cover the two gates on the rendered pixels, the projector's shadow against
the rims, the shape checks of both arrays, removal of each overlay, the animated
rule, a comparison added and removed with overlays drawn, and the timing gate.
