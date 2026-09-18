# Geometry conventions of the mbirtorch models

Date: 2026-09-09.  Status: confirmed numerically for six geometries.

This page states where the source, the detector, the volume, the rotation axis,
and the detector offsets sit in the object frame for each mbirtorch geometry
model.  The geometry viewer draws from these statements, so each statement is
written to be drawable and each is confirmed by measurement.  The measurement is
`plans/geometry_viewer/experiments/gv1_conventions_probe.py`, which
forward-projects single voxels and compares the footprint centroid on the
detector with a prediction written from the statements below.  The numbers of
that run are in `gv1_conventions_probe.md` in the same directory.  Every
statement below agrees with the projector to within 0.12 detector pixel.

Distances are in ALU, the arbitrary length unit that every model parameter uses.
Angles are in radians.

## Object frame

The object frame is a right-handed (x, y, z) frame whose z axis is the rotation
axis.  Voxel (i, j, k) of the reconstruction has object coordinates

    x = delta_voxel * (j - (num_cols - 1) / 2)
    y = delta_voxel_row * (i - (num_rows - 1) / 2)
    z = delta_voxel_slice * (k - (num_slices - 1) / 2) + recon_slice_offset

with `delta_voxel_row = voxel_row_aspect * delta_voxel` and
`delta_voxel_slice = voxel_slice_aspect * delta_voxel`.  The column index j runs
along x, the row index i runs along y, and the slice index k runs along z.  The
volume is centered on the origin in x and in y.  In z it is centered on
`recon_slice_offset`.

Two models have no `recon_slice_offset` parameter.  The parallel and translation
models reject the name, and their volumes are always centered on z = 0.  The
cone and multiaxis models accept it.

The detector is a grid of `num_det_rows` rows by `num_det_channels` channels,
and a sinogram has shape (num_views, num_det_rows, num_det_channels).  Two
physical coordinates describe a point on the detector: u along the channel
direction and v along the row direction, both measured from the point where the
central ray meets the detector.  A point at (u, v) has fractional detector
indices

    channel = (u + det_channel_offset) / delta_det_channel + (num_det_channels - 1) / 2
    row     = (v + det_row_offset) / delta_det_row + (num_det_rows - 1) / 2

The channel index therefore increases with u and the row index increases with v.
The center of the detector grid sits at u = -det_channel_offset and
v = -det_row_offset.  A positive `det_channel_offset` moves the detector grid
toward negative u, which moves a fixed object's image toward higher channel
index.  A positive `det_row_offset` moves the grid toward negative v, which
moves the image toward higher row index.  The static method
`ConeBeamModel.detector_mn_to_uv` states the same rule in the other direction.

Each model applies a per-view action to the object and holds the source and the
detector fixed.  The parallel and cone models rotate the object about z by the
view angle.  For a positive angle the rotation carries the +x axis toward the +y
axis, which is counterclockwise when seen from a point on the +z axis looking
toward the origin.  A drawing that holds the object fixed and moves the gantry
must rotate the source and the detector by the negative of the view angle, so in
such a drawing the source turns clockwise as seen from +z.  The multiaxis model
rotates the object by the azimuth in the same sense and tilts the rays by the
elevation.  The translation model shifts the object.

The residual disagreement between prediction and projector is about 0.1
detector pixel and comes from the projector's trapezoid footprint weights,
which do not place a sampled footprint's centroid exactly at its center.  It is
not a geometric disagreement.  A footprint much narrower than a detector pixel
lands on a single cell, and its centroid then rounds to an integer.

Confirmed by gv1_conventions_probe.py on 2026-09-09, for the index-to-coordinate
rule, the detector index rule, and the rotation sense.  The probe confirms them
in each of the four models separately, so the shared rule above is a summary of
four measurements rather than an assumption.

## Parallel beam

The object frame is the shared frame above, with two restrictions.
`voxel_slice_aspect` must equal one, and the slice count must equal the detector
row count.  There is no `recon_slice_offset`.

The source is infinitely far away on the positive y side, and the rays are
parallel to the direction (0, -1, 0).  The detector is the xz plane.  Its
position along y does not matter, because the projection is parallel.  Its u
axis is +x and its v axis is +z.

The channel coordinate of an object point is its x coordinate, so the channel
index increases with +x.  The detector center sits at u = -det_channel_offset,
and a positive `det_channel_offset` moves the image toward higher channel index.

The row index is the reconstruction slice index.  Detector row m receives recon
slice m, with no spreading and no offset.  The row index increases with +z.  The
row pitch that the projector uses is `delta_voxel`, and the parameters
`delta_det_row` and `det_row_offset` take no part in the projection at all.
Changing those two parameters leaves a sinogram unchanged.  A drawing of a
parallel beam detector should therefore use `delta_voxel` as the row pitch and
should not shift the rows.

The view angle rotates the object counterclockwise about z as seen from +z.  In
a fixed-object drawing the rays travel along (-sin(angle), -cos(angle), 0), so
the source direction is (sin(angle), cos(angle), 0) at whatever distance the
drawing chooses.  That direction turns clockwise as seen from +z.

Confirmed by gv1_conventions_probe.py on 2026-09-09.

## Cone beam, including the curved detector and the helical scan

The object frame is the shared frame above.  `recon_slice_offset` moves the
volume toward +z.

The source is a point at (0, source_iso_dist, 0), on the positive y axis.  A
flat detector is the plane y = source_iso_dist - source_detector_dist, on the
negative y side, at `source_detector_dist` from the source.  Its normal is the y
axis, its u axis is +x, and its v axis is +z.  The detector coordinates of a
voxel are the coordinates of the point where the ray from the source through the
voxel center meets that plane.  Written out, a point at (x, y, z) projects to
u = M x and v = M z with M = source_detector_dist / (source_iso_dist - y), the
magnification of that point.

The channel index increases with +x and the row index increases with +z.  The
detector center sits at u = -det_channel_offset and v = -det_row_offset.  A
positive `det_channel_offset` moves the image toward higher channel index, and a
positive `det_row_offset` moves the image toward higher row index.

A curved detector is a cylinder of radius `source_detector_dist` whose axis
passes through the source parallel to z.  Its channel coordinate is arc length
along the cylinder, measured from the central ray and positive toward +x, so
u = source_detector_dist * atan2(x, source_iso_dist - y).  Its row coordinate is
the same v as for a flat detector, v = M z.  The detector is therefore curved in
the channel direction only, and its rows are spaced on the plane tangent to the
cylinder at the central ray.  The row coordinate is not the height at which the
ray crosses the cylinder.  The two rules differ by the factor
hypot(x, source_iso_dist - y) / (source_iso_dist - y), so they separate only at
a large fan angle.  The probe uses a large fan angle for that reason.  There the
tangent plane rule agrees within 0.05 row, and the cylinder rule is wrong by up
to 0.75 row.

A helical scan subtracts the view's z shift from the object z.  The object z
that the projector uses is z - helical_z_shifts[view].  A positive shift
therefore moves the object toward negative z, and in a fixed-object drawing the
source and the detector move toward positive z as the shift grows.  The total
helical travel is the range of the shifts.

The view angle rotates the object counterclockwise about z as seen from +z.  In
a fixed-object drawing the source lies at
(source_iso_dist * sin(angle), source_iso_dist * cos(angle), 0), which turns
clockwise as seen from +z.  The point where the central ray meets the detector
lies opposite the source, at distance
source_detector_dist - source_iso_dist from the origin.

The model accepts an infinite `source_detector_dist`, and the magnification is
then one.  A flat detector then gives u = x and v = z, which is a parallel beam
that still uses `delta_det_row` and `det_row_offset` for its rows, unlike
`ParallelBeamModel`.  The probe did not cover that case.

Confirmed by gv1_conventions_probe.py on 2026-09-09.

## Multiaxis parallel

The object frame is the shared frame above.  `recon_slice_offset` moves the
volume toward +z, and `voxel_slice_aspect` may differ from one.

The rays are parallel and their direction is tilted out of the xy plane by the
elevation.  The assumed direction of travel is
(0, -cos(elevation), sin(elevation)).  The detector plane is perpendicular to
that direction.  Its u axis is +x, and its v axis is
(0, sin(elevation), cos(elevation)), which is the +z axis turned by the
elevation.  A point at (x, y, z) therefore projects to u = x and
v = z cos(elevation) + y sin(elevation).  The channel coordinate does not depend
on the elevation, and the row coordinate picks up the in-plane depth y as well
as z.

The channel index increases with +x.  The row index increases with +z for any
elevation smaller than 90 degrees in magnitude.  At a positive elevation the row
index also increases with +y, so a voxel farther from the detector projects
higher.  The detector center sits at u = -det_channel_offset and
v = -det_row_offset, and both offsets act as in the shared rule.

The elevation sign has a measured part and a convention part.  What the
measurement establishes is the row coordinate
v = z cos(elevation) + y sin(elevation), and it separates that from the opposite
sign by more than five rows.  What the measurement cannot establish is which end
of a ray holds the source, because a parallel projection is the same in both
directions.  Decision (Greg, 2026-09-09): the source is on the positive y side,
as in the other three models.  The direction of travel is then
(0, -cos(elevation), sin(elevation)).  The elevation is the angle at which the
source looks up at the object, in the sense of an altitude angle: for a positive
elevation the object is above the source's horizon, the rays climb from the
source through the object to the detector, and the detector center sits above
the xy plane.  The `MultiAxisParallelModel` docstring should state the elevation
this way.  This choice changes no detector index.

The azimuth rotates the object counterclockwise about z as seen from +z, exactly
as the parallel and cone view angles do.

Confirmed by gv1_conventions_probe.py on 2026-09-09, except for the position of
the source along the ray, which the measurement cannot settle.

## Translation

The object frame is the shared frame above, with one restriction.  There is no
`recon_slice_offset`, so the volume is centered on z = 0.

The source and the detector sit exactly as in the flat cone geometry.  The
source is a point at (0, source_iso_dist, 0), and the detector is the plane
y = source_iso_dist - source_detector_dist with its u axis along +x and its v
axis along +z.  The channel index increases with +x and the row index increases
with +z.  The detector center sits at u = -det_channel_offset and
v = -det_row_offset, and both offsets act as in the shared rule.  An infinite
`source_detector_dist` is rejected by the model.

The object does not rotate.  Each view moves the object by minus the view's
translation vector, so the object point that the projector uses is the voxel
center minus (t_x, t_y, t_z).  The three components mean the following.  A
positive `t_x` moves the object toward -x, which moves its image toward lower
channel index.  A positive `t_y` moves the object toward -y, which is away from
the source, so the magnification of the object falls and its image contracts
toward the point where the central ray meets the detector.  A positive `t_z`
moves the object toward -z, which moves its image toward lower row index.

There is no rotation axis to draw.  The path of the translation vectors takes
its place, and in a fixed-object drawing the source and the detector move by
plus the translation vector.

Confirmed by gv1_conventions_probe.py on 2026-09-09.

## Display convention: negative z is the top of a drawing

Decision (Greg, 2026-09-10): every drawing of the geometry puts negative z at the
top.  The reason is array indexing.  The slice index k runs along +z, and indices
increase from top to bottom in an array and in an `imshow` of a slice, so a
drawing with -z up shows the volume the way its array is indexed.

This is a presentation rule and not a geometry rule.  The object frame above stays
right-handed with +z along increasing slice index, every statement in this record
stands, and the scene computes nothing differently.  The viewer inverts the axes
of its panels, and one constant, `Z_UP_SIGN` in `geometry_viewer.py`, holds the
choice.

The group's reference picture fixes the rest of the orientation.  The slide
"Parallel Beam Geometry - top view" (from Balke et al., Separable Models for
cone-beam MBIR Reconstruction, 2018) draws the beam running from a source on the
left to a detector on the right, with z pointing down the rotation axis, y
pointing from the isocenter toward the source, and x pointing toward the viewer.
Its detector has channels increasing along +x and rows increasing along +z, with
`sino[0, 0, 0]` at the top far corner, and it names the point where the
source-to-detector line meets the detector the detector iso.  Every panel of the
viewer follows that picture: the beam runs left to right and -z is up.

- The top view is the xy plane seen from -z, with y increasing to the left and x
  increasing downward on the screen.  It is a view from -z like an `imshow` of a
  reconstruction slice, turned so that the source is on the left.  In it the
  object rotates clockwise with increasing view angle and the source travels
  counterclockwise, which agrees with the `vcls` docstring quoted below.
- The side view is the yz plane seen from +x, with y increasing to the left and
  -z up, so the source is again on the left.
- The detector face is the view from the source toward the detector with -z up:
  the channel index increases to the right and row 0 is at the top, which is how
  `imshow` shows one sinogram view.  The point where the central ray meets the
  detector is labeled the detector iso, as on the slide.
- The 3D view has its z axis inverted and is seen from the +x side and above, so
  that it resembles the slide.

For an NSI scan this convention also shows the object upright, because the NSI
reader's row direction is parallel to a rotation axis that points physically
down (see the next section).

## Real scans: which physical row is row 0

The conventions above are the model's frame.  A real scan's row order is set by
its reader, and only the NSI reader records the physical orientation.

The NSI reader applies the vendor's `flipV`, `flipH`, and `rotate` correction
flags from the `.nsipro` file, so the frames leave the reader in the orientation
the vendor software displays.  The reader then defines the row direction as
r_v = r_n x r_h, with r_n from source to detector and r_h along the rows from left
to right, and reads the physical position of the pixel in the first row and
column from the geometry report.  In the reader's test fixture r_n = (0, 1, 0),
r_h = (1, 0, 0), and the rotation axis r_a = (0, 0, -1), so r_v = (0, 0, -1),
parallel to a rotation axis that the reader's docstring describes as pointing
down.  For NSI data the row index therefore increases physically downward, row 0
is the top row of the detector, and the model's +z axis points physically
downward.  A display with row 0 at the top, which is matplotlib's default, shows
an NSI radiograph right side up.

The Zeiss cone-beam reader applies no flip and records nothing about
orientation.  The Zeiss translation reader flips the frames vertically, with a
comment that the flip was found necessary for the object to come out upright.
The pymbir reader takes the HDF5 sinogram as stored.

The geometry viewer draws the model's frame, with the row index increasing
upward along +z on its detector-face panel.  For an NSI scan that panel is
therefore vertically flipped relative to the slice viewer's display of the same
view.  Both are correct.  One shows the physical detector as the vendor displays
it, and the other shows the model's frame.

## Disagreements with existing docstrings

Two docstrings state a convention in words.  One of them disagrees with the
measurement, and the other is stated in a mirrored picture.

The `TranslationModel` docstring's z sign is wrong in the object frame.  It says
of `translation_vectors`: "Positive x shifts the object left, z shifts up, and y
shifts away from the source."  The probe found that a positive `t_x` moves the
object toward -x, a positive `t_y` moves it toward -y, and a positive `t_z`
moves it toward -z.  The statement about y agrees, because the source is on the
+y side.  The statement about x agrees if "left" means -x.  The statement about
z disagrees, because a positive `t_z` moves the object toward -z and not toward
+z.  One reading makes the sentence consistent.  In an image of the sinogram
drawn with row 0 at the top, a positive `t_z` moves the image toward lower row
index, which is upward on the screen.  Under that reading the sentence describes
the image and not the object, while the same sentence describes the object for x.
A viewer should follow the measured signs and not this sentence.

The `vcls.show_image_with_projection_rays` docstring states the rotation sense
in a mirrored picture.  It says: "Looking down at the object with the detector at
the top of the FoV, 0 degrees points from bottom to top of the object.  As the
rotation angle increases, the object rotates clockwise, which means that if the
object is kept in a fixed view, then the projection angle rotates
counterclockwise."  The probe found that the object rotation carries +x toward
+y, which is counterclockwise seen from +z in the right-handed object frame.
The two statements are compatible.  The docstring's picture is an image of a
reconstruction slice, in which the column index x runs to the right and the row
index y runs downward, so -y is at the top and the detector at y < 0 appears at
the top.  That picture is a mirror of the right-handed view from +z, and the same
rotation appears clockwise in it.  A drawing that puts +y upward must call the
object's rotation counterclockwise.

The geometry viewer plan's reading of the cone code is confirmed.  The plan says
that the source sits on the positive y axis at `source_iso_dist` and that the
detector plane sits on the negative y side at `source_detector_dist` from the
source.  Both hold.
