# gv2: the geometry scene, as built

Date: 2026-09-09.  Files: `geometry_scene.py` (the scene), `test_geometry_scene.py`
(its tests).  Status: built; 67 tests pass.

## What was built

`geometry_scene.py` is the model layer of the geometry viewer.  It imports numpy
only.  It turns a model's parameters into the points and polylines a drawing needs,
and it computes detector indices with the conventions of
`reference/geometry_conventions.md`.  The matplotlib viewer will draw from it and own no
geometry of its own.

The public surface has one class and a few constants.

- `GeometryScene(params, kind, ...)` holds the parameters of one model.  `kind` is
  one of `GEOMETRY_KINDS`: parallel, cone, multiaxis, translation.
- `GeometryScene.from_model(model)` reads the parameters through `get_params` and
  the kind from the class name.  `required_parameter_names(kind)` lists what it
  reads.  Nothing else of the model is touched.
- `project_points(points_xyz, view_index)` maps object-frame points to fractional
  detector indices, returned as (row, channel).  This is the function the gate
  test checks against the projector.
- `voxel_centers(ijk)`, `volume_corners()`, `volume_z_range()`, `ror_cylinder()`
  describe the volume.
- `view(view_index)` returns a `ViewScene` with the drawable primitives of one
  view in the fixed-object picture: the source (or a drawn stand-in), the ray
  direction, the detector center, origin, axes, outline, corners, and pixel 0,
  the volume corners and voxel 0, the rotation axis or the translation path, the
  four corner rays, the projected outline of the volume on the detector, and the
  region-of-reconstruction cylinder.
- `trajectory()` returns the source and detector-center paths over all views.
- `derived_quantities()` returns the numbers a text panel reports, with stable
  key names, and `drawing_note()` names every position that is a drawing choice.

## Tests

The command is

```
cd plans/geometry_viewer/experiments
PYTHONPATH=/home/user/cabouman/mbirtorch /home/user/gv_env/bin/python -m pytest -q test_geometry_scene.py
```

and the result is 67 passed in about two seconds on the CPU.  The tests reuse the
six configurations, the probe voxels, and the centroid measurement of
`gv1_conventions_probe.py`.

The gate test compares `project_points` with the measured footprint centroid of
each probe voxel in every view.  The gate is half a detector pixel.

| geometry | pairs | max row error | max channel error |
|---|---|---|---|
| parallel | 96 | 0.0000 | 0.1115 |
| cone flat | 96 | 0.0769 | 0.0814 |
| cone curved | 96 | 0.0460 | 0.0835 |
| cone helical | 96 | 0.0692 | 0.0848 |
| multiaxis | 96 | 0.0607 | 0.1047 |
| translation | 96 | 0.0818 | 0.0505 |

These errors equal the probe's own errors to four decimals.  That equality is
expected, because the scene and the probe implement the same statements.

A second test checks the fixed-object picture.  It re-derives each voxel's
detector indices from the moved primitives of `view(v)` alone, by intersecting
the ray from the drawn source with the drawn detector plane, and compares with
`project_points`.  The two agree to 1e-14 in every geometry.  The first version
of this test failed in all six geometries.  The fault was in the test's helper,
which passed (v, u) to a function that takes (u, v).  The scene was right.

The remaining tests cover the edge cases the plan lists and the design rules: one
detector row for parallel and cone; an infinite source-detector distance in the
cone model, which the model accepts and the scene treats as a parallel projection;
a curved detector, whose outline lies on the cylinder and whose rows use the
tangent plane; the translation model, which has a path and no rotation axis; the
helical trajectory rising with the shift; the multiaxis source side as a documented
choice; the region of reconstruction matching mbirtorch's inscribed ellipse; the
drawing distance leaving the projection unchanged; and the fit check detecting a
volume that does not fit.

## Drawing choices

Four positions in a drawing are choices and not geometry.  Each is a named
constant or constructor argument, and `drawing_note()` reports which apply.

- A parallel projection has no source position, and its detector may sit anywhere
  along the rays.  Both are drawn at `DEFAULT_DRAWING_DISTANCE_FACTOR` (1.5) times
  the volume's largest half-extent from the origin, on opposite sides.
- A cone model with an infinite source-detector distance is drawn the same way.
- The multiaxis source is on the +y side (`MULTIAXIS_SOURCE_ON_PLUS_Y`), so a
  positive elevation puts it below the xy plane.  The measurement cannot settle
  this, and the constant can be flipped without changing any detector index.
- A curved detector's outline is sampled at `CURVED_ARC_SAMPLES` (33) points per
  edge.  The rotation axis is drawn 1.25 times the volume's z half-extent past
  the volume center in each direction.

## Notes on the conventions record

The record was sufficient to implement the scene without further measurement.
One point deserves a sentence there: the parallel model's row pitch is
`delta_voxel` and its row offset is zero, and the scene therefore keeps a
separate `row_pitch` and `row_offset` from the raw parameters.  The record says
this; the scene's `_read_detector` docstring repeats it because a reader of the
code will not have the record open.
