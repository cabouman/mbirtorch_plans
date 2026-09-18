# gv1_conventions_probe: run record

Date: 2026-09-09.  Script: `gv1_conventions_probe.py`.

## What the run does

The script tests a written statement of each model's geometry against the
projector.  For six scan geometries it builds a small model, forward-projects
single voxels one at a time with `sparse_forward_project`, and measures the
intensity-weighted centroid of each footprint in detector rows and channels.
It then predicts the same centroid from plain 3D geometry in numpy: the voxel
center in object coordinates, the view action, the source position, the
detector surface, the ray from the source through the voxel center, and the
row and channel index of the intersection.  The prediction uses no formula
copied from the projector, so agreement is evidence about the geometry and not
a restatement of the code.

The gate is agreement within half a detector pixel in both row and channel, for
every unclipped voxel-view pair, in all six geometries.

## Run command

```
cd plans/geometry_viewer/experiments
PYTHONPATH=/home/user/cabouman/mbirtorch /home/user/gv_env/bin/python gv1_conventions_probe.py
```

The run takes about two seconds on the CPU and exits with status 0 when every
gate passes.  Models are built with `compile_mode='off'`.  Setting the gate to
0.05 pixel instead of 0.5 makes the same run exit with status 1, which confirms
that the gate reaches the exit status.

## Parameters

All six geometries use eight views, twelve probe voxels, a `delta_voxel` of
1.0, and a reconstruction shape of ten rows by twelve columns by eight slices.
The parallel geometry uses sixteen slices instead, because that model requires
the slice count to equal the detector row count.  The probe voxels are the
eight corners of the volume, the center voxel, and three off-center voxels.
Each off-center voxel has, in all three directions, an index that is neither at
an end nor at the center.

Every geometry that rotates the object uses the same eight view angles in
radians: -0.30, 0.0, 0.21, 0.55, 1.10, 1.90, 2.40, and 3.00.  The angles are
not equally spaced and one of them is negative.

The detector and voxel parameters per geometry are:

| geometry | detector (rows x channels) | delta_det_row, delta_det_channel | det_row_offset, det_channel_offset | voxel_row_aspect, voxel_slice_aspect | recon_slice_offset | source distances (SDD, SID) |
|---|---|---|---|---|---|---|
| parallel | 16 x 40 | 0.9, 1.1 | -0.85, +1.35 | 1.25, 1.0 | not a parameter | none |
| cone flat | 24 x 48 | 1.3, 1.1 | +0.65, -1.70 | 1.2, 0.8 | 1.4 | 200, 100 |
| cone curved | 48 x 64 | 1.3, 2.0 | -1.15, +2.30 | 1.2, 1.0 | 1.4 | 100, 26 |
| cone helical | 28 x 48 | 1.3, 1.1 | +1.40, -0.95 | 1.2, 0.8 | 1.4 | 200, 100 |
| multiaxis | 28 x 48 | 0.95, 1.1 | -1.35, +1.85 | 1.3, 1.15 | -0.9 | none |
| translation | 24 x 48 | 1.25, 1.1 | +1.45, -2.15 | 1.15, 0.9 | not a parameter | 200, 100 |

Every offset is non-integer, and both signs appear across the six geometries.
The two detector pitches differ in every geometry.  The parallel model requires
`voxel_slice_aspect` to equal one, so only the other geometries carry a slice
aspect ratio away from one.

The remaining view-dependent parameters are:

- Helical z shifts: 0.0, 0.6, 1.2, 1.8, 2.4, 3.0, 3.6, 4.2.  They are all
  non-negative and therefore not symmetric about zero.
- Multiaxis elevations in radians: 0.349, 0.349, 0.300, -0.200, 0.349, 0.349,
  0.100, 0.349.  Most views sit at about 20 degrees, one view is negative, and
  one is near zero.
- Translation vectors as (t_x, t_y, t_z): (-4.0, -2.5, -1.2), (-2.0, -1.5,
  -0.6), (0.7, 0.5, 0.3), (2.0, 1.5, 0.9), (4.0, 2.5, 1.5), (-3.0, -0.5,
  -0.9), (1.0, 1.0, 0.6), and (3.0, 2.0, 1.2).  Every component is nonzero in
  every view.

The curved cone geometry uses a short source distance and a wide detector on
purpose.  Its large fan angle is what separates the two candidate rules for the
row coordinate of a curved detector, which is discussed below.

## Results

Every geometry passed the half-pixel gate, and no voxel-view pair was skipped.

| geometry | pairs | max row error | max channel error | skipped | result |
|---|---|---|---|---|---|
| parallel | 96 | 0.0000 | 0.1115 | 0 | pass |
| cone flat | 96 | 0.0769 | 0.0814 | 0 | pass |
| cone curved | 96 | 0.0460 | 0.0835 | 0 | pass |
| cone helical | 96 | 0.0692 | 0.0848 | 0 | pass |
| multiaxis | 96 | 0.0607 | 0.1047 | 0 | pass |
| translation | 96 | 0.0818 | 0.0505 | 0 | pass |

The errors are in detector pixels.  The largest error anywhere is 0.11 pixel,
which is about a fifth of the gate.  These results indicate that the stated
geometry is the geometry the projector implements, in all six cases.

The residual of about 0.1 pixel is a property of the measurement and not of the
geometry.  The projector spreads a voxel over detector cells with a trapezoid
weight, and a sampled trapezoid does not have its first moment exactly at its
center unless the trapezoid is a triangle.  The centroid quantization survey
below measures that effect.

## Sign audit

Several signs cannot be read unambiguously from the projector source, so the
script determines them by measurement.  It evaluates the prediction again with
one convention deliberately flipped and reports the largest error.  A flipped
convention that produces an error above the gate is a convention the
measurement pins.

| geometry | flipped convention | max row error | max channel error |
|---|---|---|---|
| parallel | channel offset sign | 0.0000 | 2.5661 |
| parallel | rotation sense | 0.0000 | 9.7003 |
| cone flat | channel offset sign | 0.0769 | 3.1724 |
| cone flat | row offset sign | 1.0769 | 0.0814 |
| cone flat | rotation sense | 0.6701 | 18.6362 |
| cone curved | channel offset sign | 0.0460 | 2.3835 |
| cone curved | row offset sign | 1.8122 | 0.0835 |
| cone curved | rotation sense | 7.0049 | 20.1494 |
| cone curved | curved row on cylinder | 0.7497 | 0.0835 |
| cone helical | channel offset sign | 0.0692 | 1.8073 |
| cone helical | row offset sign | 2.2181 | 0.0848 |
| cone helical | rotation sense | 0.7551 | 18.6382 |
| cone helical | z shift sign | 13.7841 | 0.0848 |
| multiaxis | channel offset sign | 0.0607 | 3.4610 |
| multiaxis | row offset sign | 2.9028 | 0.1047 |
| multiaxis | rotation sense | 3.7654 | 10.0933 |
| multiaxis | elevation sign | 5.4376 | 0.1047 |
| translation | channel offset sign | 0.0818 | 3.9596 |
| translation | row offset sign | 2.4011 | 0.0505 |
| translation | translation sign | 5.3046 | 15.9103 |

Every flip in the table exceeds the half-pixel gate in at least one direction.
The script fails if any flip stays inside the gate, because a convention that
cannot be distinguished is a convention this run does not establish.

Two flips deserve a comment.  The flipped rotation sense for the flat and
helical cone geometries moves the row centroid by only 0.67 and 0.76 pixel,
because a rotation about z changes a row only through the magnification.  The
same flip moves the channel centroid by 18.6 pixels, which is what settles the
rotation sense.

## Direct parameter checks

Two claims cannot be seen in a footprint centroid, so the script checks them
directly.  Both held.

- The parallel model ignores `delta_det_row` and `det_row_offset`.  Changing
  the two parameters to 2.7 and 5.5 left the sinogram of a single voxel
  unchanged, with a maximum difference of exactly zero.
- The translation model has no `recon_slice_offset` parameter.  Reading it
  raises `NameError`.

## Centroid quantization survey

The survey measures how the centroid method degrades when a voxel projects to
less than one detector pixel.  It repeats the parallel geometry at three
channel pitches and reports the largest channel error.

| delta_det_channel | footprint width in channels | max channel error |
|---|---|---|
| 1.1 | 1.14 | 0.1115 |
| 4.0 | 0.31 | 0.3789 |
| 8.0 | 0.16 | 0.4395 |

A narrow footprint lands entirely on the nearest channel, so its centroid
rounds to an integer and the error approaches half a pixel.  The projector
weight is `clip((W + 1) / 2 - |n_p - n|, 0, min(1, W))` for channel `n`, where
`W` is the projected voxel width in channels and `n_p` is the projected center.
At `W` near one this weight is a triangle, and a sampled triangle has its first
moment exactly at its center.  At `W` well below one the weight is flat over
the single cell that contains `n_p`.  These results indicate that Increment 2's
scene test should keep the voxel pitch near the detector pitch.  A test that
does not will spend most of its half-pixel budget on quantization.

## Surprises

Six findings were not evident from reading the projector code alone.

The parallel model does not use its detector row geometry at all.  A detector
row index equals a reconstruction slice index, and `delta_det_row` and
`det_row_offset` take no part in the projection.  The row pitch that matters is
the voxel slice pitch, and the model requires `voxel_slice_aspect` to equal one
and the slice count to equal the detector row count.  A viewer that draws a
parallel detector with `delta_det_row` as its row pitch would draw a detector
that the projector does not use.

The parallel and translation models have no `recon_slice_offset` parameter.
Only the cone and multiaxis models accept one.

The curved cone detector is curved in the channel direction only.  Its channel
coordinate is arc length on a cylinder of radius `source_detector_dist`.  Its
row coordinate is the height at which the ray crosses the plane tangent to that
cylinder at the central ray, and not the height at which the ray crosses the
cylinder itself.  The cylinder rule disagrees with the measurement by up to
0.75 row in this configuration, which is above the gate, while the tangent
plane rule agrees within 0.046 row.  The two rules differ by the factor
`hypot(x, SID - y) / (SID - y)`, so they separate only at a large fan angle.
This is why the curved configuration uses a short source distance.

The helical z shift moves the object toward negative z.  The object z used by
the projector is the voxel z minus the view's shift.  In a drawing that holds
the object fixed, the source and detector therefore move toward positive z as
the shift grows.

A positive translation `t_z` also moves the object toward negative z, which
disagrees with the model's docstring.  The disagreement is recorded in
`../../../reference/geometry_conventions.md`.

The multiaxis geometry does not determine which end of a ray holds the source.
The projection is parallel, so the measurement fixes the ray direction only up
to sign.  Placing the source on the positive y side, as the other three models
do, then puts the source below the xy plane at positive elevation.  That last
statement is a convention and not a measured fact.

## Limitations

The run tests centroids and not footprint shapes.  A voxel's projected width
and weight enter the measurement only through the symmetry of the weight
profile, so this run says nothing about whether the projected voxel width is
correct.

The run tests one small model per geometry.  Confidence rests on the twelve
probe voxels covering every sign of every index, and on the sign audit showing
that each convention is separately pinned, not on a parameter sweep.
