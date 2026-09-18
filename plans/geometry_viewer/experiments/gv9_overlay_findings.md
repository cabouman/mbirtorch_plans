# gv9: the two overlays at scale (Increment 7)

Date: 2026-09-12.  Files in the `mbirtorch` repository: `mbirtorch/geometry_figure.py`,
`tests/test_geometry_figure.py`, `tests/test_geometry_interaction.py`, and
`docs/source/usr_geometry_viewer.rst`.  Status: built, reviewed, and staged for Greg's
review; nothing is committed.  Opus wrote the code from a specification, and the
specification's author reviewed the diff and the figures and made the measurements of
the last two sections.

The two data overlays now cost little at production size, and the phantom's outline in
the 3D panel follows the phantom's shape.  Greg raised both questions on 2026-09-11
after running the viewer on demo 3, whose reconstruction drew an outline that reached
outside the region of reconstruction.  This page records what was wrong with that
outline, what replaced it, how the two candidate replacements compared, and what the
sinogram overlay does with a large detector.

## What was wrong with the old outline

The old outline was built from bounding rectangles and not from the support.  For each
slice it took the bounding rectangle of the support, and it drew rails through the
corners of those rectangles across the slices.  For a round object the rectangle is a
square around the circle, so its corners sit outside the circle by a factor of root two.
Measured on demo 3's phantom with every voxel outside the region of reconstruction set
to zero:

| Quantity | Value |
|---|---|
| Support voxels outside the region's ellipse | 0 |
| Region of reconstruction radius | 63.5 ALU |
| Widest slice's rectangle, half width | 63.0 ALU |
| That rectangle's corner, distance from the axis | 89.1 ALU |

The drawing reached past the region while the support did not.  Seen from above the
four rails ran diagonally from the small rectangles of the end slices to the corners of
the widest one, which drew an X, and seen from the side the rails stepped by whole
voxels, up to 8 voxels between neighboring slices near the ends of this phantom.  For
the cube phantom every rectangle is the same size, so the old outline happened to be
exact there.

## Sections across the thinnest direction

The 3D panel now draws sections of the support.  A section is the outline of the
support in one plane, drawn along the cell edges of the voxels in it, which is the
same outline the top and side silhouettes use.  A box slice is four segments and a
round slice is a staircase of one-voxel steps.  The sections lie across the axis along
which the support is thinnest, spread evenly from the first plane with support to the
last, and at most nine are drawn.  When the support is no more than nine planes thick,
every plane is drawn.  The 3D panel's legend names the count, for example
"phantom (9 of 88 sections across y)", because the panel shows a few planes of the
shape and a reader has to know which.

The thinnest direction is chosen for two reasons.  A thin object is seen face on when
it is cut across its thin direction.  A translation scan's volume is a few rows deep,
so its sections lie across y and show the board and its components face on, which the
top and side views cannot, since both see the board edge on.  When the thin extent is
nine planes or fewer, every voxel of the support appears in some section.  Ties go to
z, then y, then x, so a volume filled everywhere is cut across its slices.

The detector face projects the same sections.  A section lies on the object, so its
projection lies inside the object's shadow at every view, and the envelope of the
projected sections approaches the shadow as sections are added.  The cube phantom's
test measures this in every view of its scan: the outline never reaches outside the lit
region, and it falls short of the lit edge by at most 1.583 pixels.  The shortfall is
the half pitch of the cube's thin direction.  The sections sit at the row centers, and
the cube's material extends half a row pitch beyond them, which shows in the channel
direction as the view turns.  The test's tolerance is therefore one pixel outside and
two pixels inside.

## The point budget

Many small objects make long outlines.  Measured with the cell-edge outline:

| Slice | Outline points |
|---|---|
| disk, 128 wide | 803 |
| disk, 2048 wide | 12947 |
| 1048 small blobs on a 2048 face | 12563 |

The sections share a budget of 8000 points.  A section whose outline would exceed its
share is coarsened: the plane's mask is reduced by blocks of 2, 3, or more voxels,
taking a block as in the support when any of its voxels is, until the outline fits.  A
three-voxel component becomes one block and stays visible, and components closer than
the block size merge.  The text panel's footer says by how much the outline was
coarsened.  A test puts 600 blobs of 3 by 3 voxels on a 300 by 300 face and checks that
the outline stays under the budget and passes within one coarse block of every blob;
that case coarsens by a factor of 7.

## Sections against slab silhouettes

Greg proposed an alternative on 2026-09-11: outline everything in a slab of planes, so
that every voxel of the support appears in some outline.  Both were built behind one
constant, `PHANTOM_SECTION_KIND`, and rendered on four cases at a view near 45
degrees: a Shepp-Logan head in a cone scan, the sheared cube phantom, the translation
dots, and 80 beads scattered through a cone scan's volume.  The detector-face agreement
was measured over every view of each scan.  A point of the projected outline counts as
off the lit region when it falls on a pixel below a tenth of the sinogram's largest
value, after the lit region is widened by one pixel.

| Case | Kind | Sections | Points | Fraction off the lit region | Worst distance |
|---|---|---|---|---|---|
| head | section | 9 across y | 3872 | 0.002 | 1.00 px |
| head | slab | 9 across y | 4392 | 0.131 | 4.00 px |
| cube | section | 9 across y | 2304 | 0.068 | 2.24 px |
| cube | slab | 9 across y | 2304 | 0.026 | 1.41 px |
| beads | section | 9 across y | 176 | 0.102 | 1.00 px |
| beads | slab | 9 across y | 768 | 0.407 | 15.81 px |

Sections are the default.  A slab silhouette is drawn at the slab's center plane, so
where the object tapers within the slab the outline lies outside the object, and its
projection lies outside the shadow.  On the head that reaches 4 pixels, and on the
beads, where a bead sits anywhere in a slab a thirteenth of the volume deep, 16 pixels.
The cube's two rows disagree by less than one pixel either way, because its section
is the same in every row.  The slab kind stays in the code behind the constant, for a
user who wants completeness over fidelity in a sparse volume, and its own test holds.
The cost of sections is that a bead between two section planes is absent from the 3D
outline; the beads case draws 9 of its 118 planes, and the legend says so, while the
top and side silhouettes show every bead.

The translation dots are not in the table because their measurement says nothing
about the two kinds.  The dots volume is wider than the detector's field of view, so
44 percent of the outline's points fall off the detector in each view.  Of the points
on the detector, 22 percent fall on unlit pixels at the tenth-of-maximum threshold
and 8.5 percent at a five-hundredth, with a worst distance of 8.2 pixels.  Every one of
the 12901 projected dot centers lands on a lit pixel.  These results indicate that the
projection is right and that the residual is the two-voxel coarse block drawn around
each dot: a one-voxel dot in this geometry is a needle 43 ALU deep whose shadow is a
streak about one pixel wide, and the block's outline reaches two pixels from it.

## Reading the array a chunk at a time

The phantom overlay no longer holds anything the size of the array.  The old code
built the absolute value of the whole array, its boolean support, and kept the support
for the life of the figure; at a 2K cube that is 32 GB of temporaries and an 8 GB
support.  The array is now read in chunks of slices, 16 million elements at a time, and
each chunk is thresholded where it lives.  A tensor on a GPU is thresholded on the GPU,
and only the reductions cross to the host: the three projections of the support, which
are what the silhouettes and the axis choice need, and the planes the sections are cut
from.  When no threshold is given, one more chunked pass finds the largest absolute
value first.  A divided array, the multi-device form, is refused with a message that
names `gather()`.  A test runs the pass with chunks of 50 elements and checks that the
projections equal the direct computation, and a test on this Mac's MPS device checks
that a device tensor draws the same sections as the host array.

## The sinogram overlay

A large sinogram is subsampled for display.  When the larger of the detector's two
dimensions exceeds 128 pixels, every s-th row and every s-th channel is kept, with s the
smallest stride that brings that dimension to 128 or under, the same in both directions
so that the kept pixels stay square.  The slice is taken on the object as given, so a
tensor on a GPU transfers only the kept pixels, and a numpy array is a strided view
with no copy at all.  The image's extent centers each kept sample on the detector pixel
it came from.  A feature narrower than the stride, such as one bright pixel, can fall
between the kept samples; the constant's comment says so.  `vmin` and `vmax` now pass
through `geometry_viewer`, `GeometryFigure`, and `set_sinogram`, as they do for the
slice viewer, and the default gray scale is the range of the subsample.  The test
paints a 16 by 16 block into a detector of 300 by 2000 pixels, which gives a stride of
16 and an image of 19 by 125 samples, and checks the block's sample and its position.

## Cost

A slider step on the 1800-view helical scan, with a sinogram painted and a ball phantom
of radius 8 voxels drawn, takes 29.1 ms on average and 30.6 ms at worst, against the
100 ms gate.  Building each of the four comparison figures took between 0.56 and 0.66 s.

## Tests

The figure and interaction files hold 148 tests, and the full mbirtorch suite gives
1063 passed and 133 skipped in 202 s with four workers:

```
cd mbirtorch
~/miniforge3/envs/mbirtorch/bin/python -m pytest -n 4 tests ci
```

The new tests cover the subsampled sinogram, its gray scale, a sinogram tensor on a
device, the divided form for both overlays, the sections of a block and of the cube
phantom, the axis choice for a board, a wafer, a rod, and a full volume, the nine-section
limit and its legend, the blob field and its coarsening, the chunked pass, a phantom
tensor on a device, and the slab kind's coverage of every plane.

## Open items

- The dots case shows that the coarsening rounds a thin object's outline outward by
  up to one block.  A user who needs the outline of a field of one-voxel objects at
  full resolution can raise `PHANTOM_OUTLINE_POINT_BUDGET`.
- The strided subsample drops a feature narrower than the stride.  A block average
  would keep it as a faint value, at the cost of a reshape on the device and cropping
  to a multiple of the stride.
- The package viewer has still only run under Agg here.
