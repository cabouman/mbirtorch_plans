# gv8: the port into mbirtorch (Increment 6)

Date: 2026-09-11.  Files in the `mbirtorch` repository: `mbirtorch/geometry_rules.py`,
`mbirtorch/geometry_scene.py`, `mbirtorch/geometry_figure.py`, the four model files
and `mbirtorch/tomography_model.py`, `mbirtorch/view_utils.py`, `mbirtorch/__init__.py`,
`tests/geometry_probe.py`, `tests/test_project_points.py`, `tests/test_geometry_scene.py`,
`tests/test_geometry_figure.py`, `tests/test_geometry_interaction.py`,
`tests/test_viewer_wrapper.py`, `demo/demo_11_geometry_viewer.py`,
`docs/source/usr_geometry_viewer.rst`, `docs/source/figs/geometry_viewer_cone.png`, and
five other docs pages.  File in this repository: `reference/api_specification.md`.
Status: built, reviewed, and staged in both repositories for Greg's review.  Nothing is
committed.  Opus wrote the code of each of the three parts from a specification, and
the specification's author reviewed the diffs and the figures.

The geometry viewer is now part of mbirtorch.  `mbirtorch.geometry_viewer(model, ...)`
opens it beside `mbirtorch.slice_viewer`, with the same nonblocking registry.  The
viewer draws every point on the detector through a new model method,
`TomographyModel.project_points`.  The web page was not touched.  Greg's decision on
the design question of the handoff was to leave the page a separate numpy scene, so the
prototype's files in this directory stay as the page's sources, and the page did not
need to be rebuilt.

## One statement of the projection

`project_points(points_xyz, view_index)` maps object-frame points to fractional
detector indices.  It is a method of `TomographyModel`, and each model class implements
it from the same functions its projection bodies use.  To make that possible, each
projector helper was split into two steps.  The pixel-index-to-position step is
`pixel_xy` in the new module `geometry_rules.py`.  The position-to-detector step stays
in the geometry's own module.  The two index rules, `channel_index` and `row_index`,
and the rotation about z are written once in `geometry_rules.py`, and the projector
bodies and `project_points` both call them.

The projector's arithmetic did not change.  The split moves operations into functions
and reorders nothing.  A fingerprint script recorded forward and back projections of the
six probe geometries before the change and compared them after it.  All twelve arrays
are bitwise identical.  The golden parity tests against mbirjax pass, and so do the
projector-facing test files.

The parallel geometry's rows come from the slice map.  The parallel projector has no
row formula, because detector row r receives recon slice r by array layout.
`project_points` therefore uses the fractional inverse of `recon_slice_z`, which the
model already holds as the one host-side statement of that map.

The method computes in float64 on the CPU.  A geometric query is small, and a device
transfer would cost more than the arithmetic.  The argument `view_index` is one index
or a sequence of indices.  The result is `(row, channel)`, each `(N,)` for one view and
`(V, N)` for several.  The sequence form is what lets the scene's fit statement project
every view in one call.

The tests in `tests/test_project_points.py` check three things against each other.  The
projector, through the footprint centroids of single voxels, agrees with
`project_points` within the plan's half pixel for the six probe geometries.  An
independent numpy prediction, copied from `gv1_conventions_probe.py` into
`tests/geometry_probe.py`, agrees with `project_points` to 1e-8 pixel.  That comparison
first disagreed by 2e-6 pixel, because the models store their view parameters as
float32 and the prediction had used the float64 constants of the configuration.  With
the stored values read back from the model, the largest disagreement is 7.1e-15 pixel.
The prototype's own `GeometryScene.project_points` agreed to 1e-9 pixel in a check
made once during the port.  That check is not a package test, because a package test
must not depend on this repository.  Two more tests pin the
rules the conventions record names: parallel rows are slice indices and ignore the row
parameters, and a curved detector's rows use the tangent plane and not the cylinder.

## The scene and the figure in the package

`mbirtorch/geometry_scene.py` is the prototype's scene built from a model instead of a
parameter dictionary.  It reads the parameters by name through `get_params`, as before,
and it projects points through `model.project_points`.  Its own copy of the projection
formulas is gone: `detector_coordinates`, `point_magnification`, and `uv_to_indices`
were deleted.  The drawing geometry stays in numpy, because the source, the detector,
the rays, and the axis are positions the projector never computes.  The test that
re-derives the projection from the moved primitives still pins that drawing geometry to
the projector.

A comparison or a reference view now builds a model copy.  `with_parameters` constructs
a new model of the same class from the model's own parameters, applies every optional
parameter with `set_params`, and then applies the overrides.  It does not use
`build_model` or `copy_ct_model`, because both rerun the automatic reconstruction
geometry, which replaces a `delta_voxel` or a `recon_slice_offset` that was set by hand.
A copy costs about 12 ms.

The fit statement projects every view in one call.  The prototype walked the views one
`project_points` call at a time, which was cheap in numpy and would cost a torch call
per view now.  The overshoots and the axial coverages are computed with array
operations, and every entry of the report keeps its meaning.  The ported scene agrees
with the prototype scene to the last bit on every derived quantity and every projected
index, over the six probe geometries, the two automatic cone scans, and the gapped
helical case.

The figure module is `mbirtorch/geometry_figure.py`, named for its class.  The function
`geometry_viewer` is exported lazily through the package's `__getattr__`.  A module
named `geometry_viewer.py` would have been bound to the package attribute of that name
the moment anything imported it, and the function would have been shadowed.  The slice
viewer avoids the same clash by holding `slice_viewer` in `viewer.py`.  Three small
changes follow the slice viewer as well: the earlier name `show_geometry` was dropped,
the `AXLIM_CLIP` guard for matplotlib releases before 3.10 was kept, and a blocking
show collects garbage under TkAgg afterward.  The docstrings and comments keep their
technical content and carry no history (Greg, 2026-09-11): the attributions, the
dates, the plan notation, and the references to this directory's records were
removed, and the facts they pointed at were kept.

The package surface follows the slice viewer's.  `view_utils.py` re-exports
`GeometryScene`, `GeometryFigure`, and `geometry_viewer`, and `__init__.py` resolves
the three names lazily, so a headless `import mbirtorch` still imports no matplotlib.
The existing test of that rule now also watches the two new modules.

The three test files moved with the two modules.  They hold 112, 85, and 53 tests
against the prototype's 113, 85, and 54.  The two tests dropped were the one that left
a name out of a parameter dictionary, which a model cannot do, and the one for
`show_geometry`.  Every scene the prototype built from a dictionary is built from a
model or through `with_parameters` now.  The timing test measures a slider step on the
1800-view helical scan with a sinogram painted: 26.4 ms on average and 27.0 ms at
worst, against the 100 ms gate.

The six example figures were rendered from the package and read by eye.  Each shows the
cube phantom, its sinogram, the source path, the angle-0 reference, and a comparison
ten channels over, and every panel matches what the prototype's records describe.  The
default cone example carries a channel offset of 3 ALU and a row offset of -2 ALU, and
its text panel reads 2.99 channels over laterally and 8.8 rows over axially.  The
earlier findings gave the zero-offset answers for the same scan: a lateral fit and an
axial miss of 6.8 rows.  The two offsets move those answers by 3 channels and 2 rows,
which is what the figure reports.

## The demo, the docs page, and the API rows

`demo/demo_11_geometry_viewer.py` builds a cone-beam scan with the two offsets, projects
the cube phantom, prints where the volume's center lands, and opens the viewer with the
sinogram, the phantom, and the ten-channel comparison.  Under the Agg backend it prints
the landing line and the viewer's no-window line and exits cleanly.

`docs/source/usr_geometry_viewer.rst` is the docs page.  It says what the viewer is for,
what the five panels show, what the controls and the two overlays do, and where a point
lands.  It carries the demo's figure and the documentation of `geometry_viewer`,
`GeometryFigure`, and its setters.  `project_points` is documented on the tomography
model's page.  The new page is linked from the API list, the utilities page, the API
overview, and the demos page, which also gained a FAQ entry.  The docs build with
warnings as errors, and ten docstring cross-references were rewritten so that it does.

The API specification gained two rows: `geometry_viewer` under Viewing and
`project_points` under Model management, whose rule now names the one method there
that takes an array.  The slice viewer's row was corrected to say that it returns the
viewer object.

## Two docstring corrections and one more

The two docstrings named in `reference/geometry_conventions.md` were corrected.
`TranslationModel` now says that each view moves the object by minus its vector, with
the effect of each component on the image.  `MultiAxisParallelModel` now defines the
elevation as the angle at which the source, on the +y side, looks at the object, with
the row coordinate `v = z cos(elevation) + y sin(elevation)`.  A third docstring in the
translation module said the object shifts by `(t_x, t_y)`.  It says `(-t_x, -t_y)` now.

## Found on the way

`copy_ct_model` loses a hand-set `delta_voxel`.  It calls `build_model`, which reruns
the automatic reconstruction geometry and re-pins only `recon_shape`.  For the scanner
reader flow that rerun is deliberate and tested.  For the split-sinogram reconstruction,
whose half models come from `copy_ct_model`, a parent whose voxel pitch was set by hand
gets halves at the automatic pitch.  This was not changed, and it is recorded as a
separate task for Greg to decide.

## Tests and commands

Every suite involved passes.  The full mbirtorch suite, run as CI runs it with four
workers, gives 1034 passed and 133 skipped in 267 s:

```
cd mbirtorch
~/miniforge3/envs/mbirtorch/bin/python -m pytest -n 4 tests ci
```

The prototype's four non-web test files pass against the refactored package, 278
tests, because they drive the same projector:

```
cd plans/geometry_viewer/experiments
MPLBACKEND=Agg ~/miniforge3/envs/mbirtorch/bin/python -m pytest -q \
    test_geometry_scene.py test_geometry_viewer.py test_geometry_interaction.py \
    test_geometry_defaults.py
```

The golden parity tests are opt-in and pass:

```
cd mbirtorch
~/miniforge3/envs/mbirtorch/bin/python -m pytest -q -m goldens tests/test_vs_goldens.py
```

## Open items

- The package viewer has not been watched on a display.  Every check here ran under
  Agg.  Greg's first use of `mbirtorch.geometry_viewer` on a screen is the display test
  of the port, as his use of the prototype was for Increment 4.
- The two small items of the handoff's Part 2, the phantom outline cut at the
  detector's edge and the seven remaining label collisions, were skipped at Greg's
  direction.
- The web page's modules are the prototype's, unchanged.  If the page is ever rebuilt
  from the package's modules instead, `test_web_app.py` and the build script need the
  new sources.
