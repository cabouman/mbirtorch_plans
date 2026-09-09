"""Headless tests for geometry_scene.py.

The gate test is ``test_projection_matches_projector``.  For each of the six
scan geometries of ``gv1_conventions_probe.py`` it forward-projects single
voxels with the real projector, measures where each footprint lands on the
detector, and compares that with ``GeometryScene.project_points``.  A viewer
that failed this test would draw a geometry the projector does not use.

The rest of the tests check the fixed-object picture, the drawing primitives,
the parameter interface, and the four edge cases the plan lists: a single
detector row, an infinite source-detector distance, a curved detector, and the
translation geometry.

Run:
    cd plans/experiments/geometry_viewer
    PYTHONPATH=<mbirtorch clone> python -m pytest -q test_geometry_scene.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gv1_conventions_probe as probe  # noqa: E402
from geometry_scene import (CURVED_ARC_SAMPLES, GeometryScene,  # noqa: E402
                            required_parameter_names)

# The gate: the largest allowed difference between a measured footprint
# centroid and the scene's prediction, in detector pixels.  Half a pixel is the
# tolerance the plan's design rule sets.
GATE_PIXELS = 0.5

# Position and direction comparisons are geometric identities, so they get a
# tight tolerance rather than the gate's half pixel.
GEOMETRY_TOLERANCE = 1e-8

CONFIGS_BY_NAME = {cfg['name']: cfg for cfg in probe.CONFIGS}

# Filled in by the gate test and printed by this file's __main__ block, so the
# run record can quote a number per geometry.
MAX_ERRORS = {}


def build_scene(cfg):
    """Build the model of one probe configuration and a scene from it."""
    model = probe.build_model(cfg)
    return model, GeometryScene.from_model(model)


# ── the gate ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_projection_matches_projector(name):
    """Every projected voxel center lands where the projector puts it.

    The comparison uses footprint centroids, which are only meaningful when the
    footprint is at least about as wide as a detector pixel.  Every probe
    configuration keeps the voxel pitch near the detector pitch for that
    reason, and the probe's own record explains the failure mode.
    """
    cfg = CONFIGS_BY_NAME[name]
    model, scene = build_scene(cfg)

    max_row_error = 0.0
    max_channel_error = 0.0
    num_pairs = 0
    for voxel in probe.probe_voxels(cfg['recon_shape']):
        measured = probe.measure_footprint_centroids(model, cfg, voxel)
        center = scene.voxel_centers([voxel])
        for view, (reason, row, channel) in enumerate(measured):
            if reason is not None:
                continue
            row_pred, channel_pred = scene.project_points(center, view)
            row_error = abs(row - float(row_pred[0]))
            channel_error = abs(channel - float(channel_pred[0]))
            assert row_error <= GATE_PIXELS, (
                f'{name} voxel {voxel} view {view}: row {row} against '
                f'predicted {float(row_pred[0])}')
            assert channel_error <= GATE_PIXELS, (
                f'{name} voxel {voxel} view {view}: channel {channel} against '
                f'predicted {float(channel_pred[0])}')
            max_row_error = max(max_row_error, row_error)
            max_channel_error = max(max_channel_error, channel_error)
            num_pairs += 1

    assert num_pairs > 0
    MAX_ERRORS[name] = (num_pairs, max_row_error, max_channel_error)


@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_voxel_centers_match_the_probe(name):
    """The scene's voxel coordinates agree with the probe's own rule."""
    cfg = CONFIGS_BY_NAME[name]
    _, scene = build_scene(cfg)
    scalars = probe.geometry_scalars(cfg)
    voxels = probe.probe_voxels(cfg['recon_shape'])
    expected = np.array([probe.voxel_center_xyz(i, j, k, scalars)
                         for i, j, k in voxels])
    assert np.allclose(scene.voxel_centers(voxels), expected,
                       atol=GEOMETRY_TOLERANCE)


# ── the fixed-object picture ─────────────────────────────────────────────────

@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_volume_outline_matches_project_points(name):
    """view(v).volume_outline_on_detector is project_points of the corners."""
    cfg = CONFIGS_BY_NAME[name]
    _, scene = build_scene(cfg)
    for view in range(scene.num_views):
        view_scene = scene.view(view)
        row, channel = scene.project_points(view_scene.volume_corners, view)
        expected = np.stack([row, channel], axis=1)
        assert view_scene.volume_outline_on_detector.shape == (8, 2)
        assert np.allclose(view_scene.volume_outline_on_detector, expected,
                           atol=GEOMETRY_TOLERANCE)


def detector_indices_from_moved_detector(scene, view_scene, point):
    """Detector indices of one object-frame point, from the moved primitives.

    This repeats the projection using only the drawing primitives of
    ``view(v)``: the source position (or the ray direction), the detector
    origin, and the detector's two axes, all of which carry the inverse of the
    view's action.  Agreement with ``project_points`` therefore shows that the
    inverse action was applied consistently to the source and to the detector.

    A flat detector is a plane, so the ray from the source through the point is
    intersected with it.  A curved detector's channel coordinate is arc length,
    so the angle from the central ray is used instead.  ``uv_to_indices`` takes
    (u, v) in that order and returns (row, channel).
    """
    point = np.asarray(point, dtype=np.float64).reshape(3)
    origin = view_scene.detector_origin
    u_axis = view_scene.detector_u_axis
    v_axis = view_scene.detector_v_axis

    if view_scene.source is None:
        # Parallel rays: slide the point along the ray onto the detector plane.
        direction = view_scene.ray_direction
        step = np.dot(origin - point, direction)
        hit = point + step * direction
        u = np.dot(hit - origin, u_axis)
        v = np.dot(hit - origin, v_axis)
        return scene.uv_to_indices(u, v)

    to_point = point - view_scene.source
    forward = np.dot(to_point, view_scene.ray_direction)
    if scene.use_curved_detector:
        radius = scene.source_detector_dist
        u = radius * np.arctan2(np.dot(to_point, u_axis), forward)
        v = radius * np.dot(to_point, v_axis) / forward
    else:
        # The plane through detector_origin with the detector's normal.
        normal = view_scene.detector_normal
        denominator = np.dot(to_point, normal)
        step = np.dot(origin - view_scene.source, normal) / denominator
        hit = view_scene.source + step * to_point
        u = np.dot(hit - origin, u_axis)
        v = np.dot(hit - origin, v_axis)
    return scene.uv_to_indices(u, v)


@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_moved_source_and_detector_reproduce_the_projection(name):
    """The drawn source and detector project a point where the scene says.

    This is the check that the fixed-object picture is right.  The scene's
    ``project_points`` moves the object and holds the source fixed, which is
    what the projector does.  ``view(v)`` holds the object fixed and moves the
    source and the detector.  Re-deriving the detector indices from the moved
    primitives must give the same answer.
    """
    cfg = CONFIGS_BY_NAME[name]
    _, scene = build_scene(cfg)
    voxels = probe.probe_voxels(cfg['recon_shape'])
    centers = scene.voxel_centers(voxels)
    for view in range(scene.num_views):
        view_scene = scene.view(view)
        rows, channels = scene.project_points(centers, view)
        for index, point in enumerate(centers):
            row, channel = detector_indices_from_moved_detector(
                scene, view_scene, point)
            assert abs(float(row) - rows[index]) < 1e-6
            assert abs(float(channel) - channels[index]) < 1e-6


@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_detector_center_sits_at_the_grid_center(name):
    """The detector center is the (u, v) point that maps to the grid center."""
    cfg = CONFIGS_BY_NAME[name]
    _, scene = build_scene(cfg)
    row, channel = scene.uv_to_indices(-scene.det_channel_offset,
                                       -scene.row_offset)
    assert float(row) == pytest.approx((scene.num_det_rows - 1) / 2.0)
    assert float(channel) == pytest.approx((scene.num_det_channels - 1) / 2.0)
    quantities = scene.derived_quantities()
    assert quantities['detector_center_u'] == pytest.approx(
        -scene.det_channel_offset)
    assert quantities['detector_center_v'] == pytest.approx(-scene.row_offset)


@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_view_primitives_have_the_promised_shapes(name):
    """Every primitive of every view has the shape a drawing expects."""
    cfg = CONFIGS_BY_NAME[name]
    _, scene = build_scene(cfg)
    view_scene = scene.view(0)
    assert view_scene.source_draw.shape == (3,)
    assert view_scene.ray_direction.shape == (3,)
    assert np.linalg.norm(view_scene.ray_direction) == pytest.approx(1.0)
    assert view_scene.detector_center.shape == (3,)
    assert view_scene.detector_u_axis.shape == (3,)
    assert view_scene.detector_v_axis.shape == (3,)
    assert view_scene.detector_corners.shape == (4, 3)
    assert view_scene.detector_pixel0.shape == (3,)
    assert view_scene.volume_corners.shape == (8, 3)
    assert view_scene.voxel0_center.shape == (3,)
    assert view_scene.corner_rays.shape == (4, 2, 3)
    assert view_scene.detector_outline.ndim == 2
    assert view_scene.detector_outline.shape[1] == 3
    # The outline is closed.
    assert np.allclose(view_scene.detector_outline[0],
                       view_scene.detector_outline[-1],
                       atol=GEOMETRY_TOLERANCE)
    # Each corner ray starts at the drawn source and ends on a corner.
    for index in range(4):
        assert np.allclose(view_scene.corner_rays[index, 0],
                           view_scene.source_draw, atol=GEOMETRY_TOLERANCE)
        assert np.allclose(view_scene.corner_rays[index, 1],
                           view_scene.detector_corners[index],
                           atol=GEOMETRY_TOLERANCE)

    sources, centers = scene.trajectory()
    assert sources.shape == (scene.num_views, 3)
    assert centers.shape == (scene.num_views, 3)
    assert np.allclose(sources[0], view_scene.source_draw,
                       atol=GEOMETRY_TOLERANCE)
    assert np.allclose(centers[0], view_scene.detector_center,
                       atol=GEOMETRY_TOLERANCE)


@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_derived_quantities_are_plain_values(name):
    """Every derived quantity is a plain float, int, bool, or string."""
    cfg = CONFIGS_BY_NAME[name]
    _, scene = build_scene(cfg)
    quantities = scene.derived_quantities()
    for key, value in quantities.items():
        assert isinstance(value, (float, int, bool, str)), (key, type(value))
    assert quantities['geometry_kind'] == scene.kind
    if scene.is_parallel_type:
        assert quantities['magnification'] == pytest.approx(1.0)
        assert quantities['fan_angle_deg'] == 0.0
        assert quantities['cone_angle_deg'] == 0.0
    else:
        assert quantities['fan_angle_deg'] > 0.0
        assert quantities['cone_angle_deg'] > 0.0
        assert quantities['magnification'] == pytest.approx(
            scene.source_detector_dist / scene.source_iso_dist)
    # The probe's configurations keep the volume inside the detector's field of
    # view, so the corner test must agree.
    fits, overshoot = scene.volume_fits_detector()
    assert fits is quantities['volume_fits_detector']
    assert overshoot == pytest.approx(quantities['worst_overshoot_pixels'])


# ── the parameter interface ──────────────────────────────────────────────────

@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_from_model_reads_only_named_parameters(name):
    """A scene from a model equals a scene from that model's parameter values.

    The dictionary is built by asking the model for one parameter name at a
    time, which is the only access ``from_model`` is allowed.  Agreement shows
    that the scene depends on the parameter names and on nothing else about the
    model.
    """
    cfg = CONFIGS_BY_NAME[name]
    model, from_model = build_scene(cfg)
    kind = GeometryScene.kind_of_model(model)
    params = {parameter: model.get_params(parameter)
              for parameter in required_parameter_names(kind)}
    from_dict = GeometryScene(params, kind)

    assert from_dict.kind == from_model.kind
    assert from_dict.num_views == from_model.num_views
    assert from_dict.derived_quantities() == from_model.derived_quantities()
    voxels = probe.probe_voxels(cfg['recon_shape'])
    centers = from_model.voxel_centers(voxels)
    for view in range(from_model.num_views):
        expected = from_model.project_points(centers, view)
        actual = from_dict.project_points(centers, view)
        assert np.allclose(actual[0], expected[0], atol=GEOMETRY_TOLERANCE)
        assert np.allclose(actual[1], expected[1], atol=GEOMETRY_TOLERANCE)


def test_kind_of_model_covers_the_four_classes():
    """Each model class is recognized, including multiaxis's class string."""
    for cfg in probe.CONFIGS:
        model = probe.build_model(cfg)
        assert GeometryScene.kind_of_model(model) == cfg['kind']


def test_missing_parameter_is_reported():
    """A dictionary that lacks a needed name fails at construction."""
    cfg = CONFIGS_BY_NAME['cone flat']
    model = probe.build_model(cfg)
    params = {parameter: model.get_params(parameter)
              for parameter in required_parameter_names('cone')}
    del params['source_iso_dist']
    with pytest.raises(ValueError, match='source_iso_dist'):
        GeometryScene(params, 'cone')


def test_unknown_kind_is_reported():
    with pytest.raises(ValueError, match='Unknown geometry kind'):
        GeometryScene({}, 'spiral')


# ── the parallel row rule ────────────────────────────────────────────────────

def test_parallel_rows_ignore_the_detector_row_parameters():
    """A parallel scene's rows do not move when the row parameters change.

    The parallel projector sends recon slice m to detector row m, so
    ``delta_det_row`` and ``det_row_offset`` take no part.  A scene that used
    them would draw the rows in the wrong places.
    """
    cfg = CONFIGS_BY_NAME['parallel']
    model, scene = build_scene(cfg)
    voxels = probe.probe_voxels(cfg['recon_shape'])
    centers = scene.voxel_centers(voxels)
    rows_before, _ = scene.project_points(centers, 3)

    params = dict(scene.params)
    params['delta_det_row'] = 3.7
    params['det_row_offset'] = 6.25
    changed = GeometryScene(params, 'parallel')
    rows_after, _ = changed.project_points(centers, 3)
    assert np.allclose(rows_after, rows_before, atol=GEOMETRY_TOLERANCE)
    assert changed.row_pitch == pytest.approx(scene.delta_voxel)
    assert changed.row_offset == 0.0

    # The same rule holds for the drawn detector height.
    assert changed.detector_size()[1] == pytest.approx(
        scene.num_det_rows * scene.delta_voxel)


# ── edge case: a single detector row ─────────────────────────────────────────

def single_row_parallel_config():
    """A parallel configuration with one detector row and one recon slice.

    The parallel model requires the slice count to equal the detector row
    count, so a one-row detector means a one-slice volume.
    """
    return dict(
        name='parallel one row',
        kind='parallel',
        sinogram_shape=(4, 1, 40),
        recon_shape=(10, 12, 1),
        angles=probe.VIEW_ANGLES[:4],
        params=dict(delta_det_channel=1.1, delta_det_row=0.9,
                    det_channel_offset=1.35, det_row_offset=-0.85,
                    delta_voxel=1.0, voxel_row_aspect=1.25,
                    voxel_slice_aspect=1.0),
    )


def test_single_detector_row_projects_correctly():
    """One detector row: every voxel lands on row 0, and the projector agrees."""
    cfg = single_row_parallel_config()
    model, scene = build_scene(cfg)
    assert scene.num_det_rows == 1
    # The probe's voxel list assumes at least three slices.  With one slice,
    # every voxel sits in slice 0, so clamp the slice index and drop repeats.
    voxels = sorted({(i, j, 0) for i, j, _ in
                     probe.probe_voxels(cfg['recon_shape'])})
    num_pairs = 0
    for voxel in voxels:
        measured = probe.measure_footprint_centroids(model, cfg, voxel)
        center = scene.voxel_centers([voxel])
        for view, (reason, row, channel) in enumerate(measured):
            if reason is not None:
                continue
            row_pred, channel_pred = scene.project_points(center, view)
            assert abs(row - float(row_pred[0])) <= GATE_PIXELS
            assert abs(channel - float(channel_pred[0])) <= GATE_PIXELS
            num_pairs += 1
    assert num_pairs > 0


def test_single_detector_row_primitives_are_finite():
    """A one-row detector still has a drawable outline and finite numbers."""
    cfg = single_row_parallel_config()
    _, scene = build_scene(cfg)
    view_scene = scene.view(0)
    assert np.all(np.isfinite(view_scene.detector_outline))
    assert np.all(np.isfinite(view_scene.corner_rays))
    quantities = scene.derived_quantities()
    assert quantities['detector_height'] == pytest.approx(scene.delta_voxel)
    assert np.isfinite(quantities['axial_fov_alu'])


def test_single_detector_row_cone():
    """A one-row cone detector gives a small cone angle and finite drawing."""
    cfg = dict(CONFIGS_BY_NAME['cone flat'])
    cfg = dict(cfg, name='cone one row', sinogram_shape=(8, 1, 48),
               recon_shape=(10, 12, 1))
    _, scene = build_scene(cfg)
    assert scene.num_det_rows == 1
    quantities = scene.derived_quantities()
    assert 0.0 < quantities['cone_angle_deg'] < 2.0
    view_scene = scene.view(2)
    assert np.all(np.isfinite(view_scene.detector_outline))
    assert view_scene.detector_outline.shape == (5, 3)


# ── edge case: an infinite source-detector distance ──────────────────────────

def infinite_distance_cone_config():
    """A cone configuration whose source-detector distance is infinite."""
    cfg = dict(CONFIGS_BY_NAME['cone flat'])
    return dict(cfg, name='cone infinite sdd',
                source_detector_dist=np.inf, source_iso_dist=100.0)


def test_infinite_source_detector_dist_is_a_parallel_projection():
    """An infinite source-detector distance gives magnification one.

    The cone model accepts the infinite distance, and its projector then uses
    u = x and v = z, which is a parallel projection that still uses
    ``delta_det_row`` and ``det_row_offset`` for its rows.  The scene must say
    the same, and it must place the source and the detector by the drawing
    rule, because neither has a finite position.
    """
    cfg = infinite_distance_cone_config()
    model, scene = build_scene(cfg)
    assert scene.magnification == pytest.approx(1.0)
    assert scene.is_parallel_type is True
    assert scene.row_pitch == pytest.approx(scene.delta_det_row)
    assert scene.row_offset == pytest.approx(scene.det_row_offset)

    view_scene = scene.view(0)
    assert view_scene.source is None
    assert np.all(np.isfinite(view_scene.source_draw))
    assert np.all(np.isfinite(view_scene.detector_outline))
    assert 'infinite' in scene.derived_quantities()['drawing_note']

    # The projector must agree with the scene here as it does elsewhere.
    num_pairs = 0
    for voxel in probe.probe_voxels(cfg['recon_shape']):
        measured = probe.measure_footprint_centroids(model, cfg, voxel)
        center = scene.voxel_centers([voxel])
        for view, (reason, row, channel) in enumerate(measured):
            if reason is not None:
                continue
            row_pred, channel_pred = scene.project_points(center, view)
            assert abs(row - float(row_pred[0])) <= GATE_PIXELS
            assert abs(channel - float(channel_pred[0])) <= GATE_PIXELS
            num_pairs += 1
    assert num_pairs > 0


def test_infinite_source_detector_dist_rejected_where_it_makes_no_sense():
    """The two geometries that cannot use an infinite distance say so."""
    cfg = dict(CONFIGS_BY_NAME['translation'])
    model = probe.build_model(cfg)
    params = {name: model.get_params(name)
              for name in required_parameter_names('translation')}
    params['source_detector_dist'] = np.inf
    with pytest.raises(ValueError, match='finite source_detector_dist'):
        GeometryScene(params, 'translation')

    cfg = CONFIGS_BY_NAME['cone curved']
    model = probe.build_model(cfg)
    params = {name: model.get_params(name)
              for name in required_parameter_names('cone')}
    params['source_detector_dist'] = np.inf
    with pytest.raises(ValueError, match='curved detector'):
        GeometryScene(params, 'cone')


# ── edge case: a curved detector ─────────────────────────────────────────────

def test_curved_detector_outline_lies_on_the_cylinder():
    """A curved outline is an arc at the cylinder's radius from the source.

    The cylinder's axis passes through the source parallel to z, so every point
    of the detector surface is at ``source_detector_dist`` from the source when
    measured in the xy plane.
    """
    cfg = CONFIGS_BY_NAME['cone curved']
    _, scene = build_scene(cfg)
    assert scene.use_curved_detector is True
    for view in (0, 3, 5):
        view_scene = scene.view(view)
        assert view_scene.detector_outline.shape == (2 * CURVED_ARC_SAMPLES + 1, 3)
        offsets = view_scene.detector_outline - view_scene.source
        horizontal = np.hypot(offsets[:, 0], offsets[:, 1])
        assert np.allclose(horizontal, scene.source_detector_dist, atol=1e-6)
        corner_offsets = view_scene.detector_corners - view_scene.source
        assert np.allclose(np.hypot(corner_offsets[:, 0], corner_offsets[:, 1]),
                           scene.source_detector_dist, atol=1e-6)


def test_curved_detector_rows_use_the_tangent_plane():
    """The curved row rule is the tangent plane and not the cylinder crossing.

    The two rules differ by hypot(x, source_iso_dist - y) / (source_iso_dist -
    y), so they separate only at a large fan angle.  The curved probe
    configuration has a large fan angle for that reason, and the projector was
    measured to follow the tangent plane.  This test states the difference so
    that a later change to the cylinder rule would be caught.
    """
    cfg = CONFIGS_BY_NAME['cone curved']
    _, scene = build_scene(cfg)
    voxels = probe.probe_voxels(cfg['recon_shape'])
    centers = scene.voxel_centers(voxels)
    view = 0
    rows, _ = scene.project_points(centers, view)

    projector_frame = scene.to_projector_frame(centers, view)
    x = projector_frame[:, 0]
    depth = scene.source_iso_dist - projector_frame[:, 1]
    z = projector_frame[:, 2]
    tangent_v = scene.source_detector_dist * z / depth
    cylinder_v = scene.source_detector_dist * z / np.hypot(x, depth)
    tangent_rows, _ = scene.uv_to_indices(np.zeros_like(tangent_v), tangent_v)
    cylinder_rows, _ = scene.uv_to_indices(np.zeros_like(cylinder_v),
                                           cylinder_v)
    assert np.allclose(rows, tangent_rows, atol=GEOMETRY_TOLERANCE)
    assert np.max(np.abs(tangent_rows - cylinder_rows)) > 0.5


# ── edge case: the translation geometry ──────────────────────────────────────

def test_translation_has_a_path_and_no_rotation_axis():
    """Translation draws the path of the rotation center, not an axis."""
    cfg = CONFIGS_BY_NAME['translation']
    _, scene = build_scene(cfg)
    view_scene = scene.view(0)
    assert view_scene.rotation_axis is None
    assert view_scene.translation_path.shape == (scene.num_views, 3)
    assert np.allclose(view_scene.translation_path,
                       probe.TRANSLATION_VECTORS, atol=1e-6)
    # In the fixed-object picture the source moves by plus the translation
    # vector, so the source path is the path shifted to the source.
    sources, _ = scene.trajectory()
    expected = (np.array([0.0, scene.source_iso_dist, 0.0])
                + probe.TRANSLATION_VECTORS)
    assert np.allclose(sources, expected, atol=1e-6)
    # No mask is drawn: the translation model turns the mask off.
    assert scene.ror_cylinder() is None


def test_rotating_geometries_have_a_rotation_axis():
    """The three rotating geometries draw an axis through the volume's z range."""
    for name in ('parallel', 'cone flat', 'multiaxis'):
        cfg = CONFIGS_BY_NAME[name]
        _, scene = build_scene(cfg)
        view_scene = scene.view(0)
        assert view_scene.translation_path is None
        axis = view_scene.rotation_axis
        assert axis.shape == (2, 3)
        assert np.allclose(axis[:, :2], 0.0, atol=GEOMETRY_TOLERANCE)
        z_min, z_max = scene.volume_z_range()
        assert axis[0, 2] <= z_min + GEOMETRY_TOLERANCE
        assert axis[1, 2] >= z_max - GEOMETRY_TOLERANCE


# ── edge case: a helical cone scan ───────────────────────────────────────────

def test_helical_trajectory_rises_with_the_z_shift():
    """The drawn source rises toward +z as the helical shift grows.

    The projector subtracts the shift from the object's z, so in the
    fixed-object picture the source and the detector move toward +z by the
    shift.
    """
    cfg = CONFIGS_BY_NAME['cone helical']
    _, scene = build_scene(cfg)
    sources, centers = scene.trajectory()
    assert np.allclose(sources[:, 2], probe.HELICAL_Z_SHIFTS, atol=1e-6)
    assert np.all(np.diff(sources[:, 2]) > 0)
    # The detector center rises by the same shift.
    assert np.allclose(centers[:, 2] - centers[0, 2],
                       probe.HELICAL_Z_SHIFTS - probe.HELICAL_Z_SHIFTS[0],
                       atol=1e-6)
    # The source turns clockwise seen from +z, so its in-plane angle from the
    # +y axis follows minus the view angle.
    angles = np.arctan2(sources[:, 0], sources[:, 1])
    assert np.allclose(angles, np.arctan2(np.sin(probe.VIEW_ANGLES),
                                          np.cos(probe.VIEW_ANGLES)),
                       atol=1e-6)
    assert scene.derived_quantities()['helical_travel_alu'] == pytest.approx(
        float(np.ptp(probe.HELICAL_Z_SHIFTS)))


# ── the multiaxis source-side convention ────────────────────────────────────

def test_multiaxis_source_side_is_a_documented_choice():
    """A positive elevation puts the source below the xy plane by default.

    The convention flag reverses the source and the detector and leaves every
    detector index unchanged, because a parallel projection is the same in both
    directions along a ray.
    """
    cfg = CONFIGS_BY_NAME['multiaxis']
    model = probe.build_model(cfg)
    default_scene = GeometryScene.from_model(model)
    flipped_scene = GeometryScene.from_model(
        model, multiaxis_source_on_plus_y=False)

    view = 0
    assert probe.MULTIAXIS_ELEVATIONS[view] > 0
    default_view = default_scene.view(view)
    assert default_view.source_draw[2] < 0.0
    assert default_view.detector_center[2] > 0.0
    assert default_view.source is None

    flipped_view = flipped_scene.view(view)
    assert flipped_view.source_draw[2] > 0.0

    voxels = probe.probe_voxels(cfg['recon_shape'])
    centers = default_scene.voxel_centers(voxels)
    for index in range(default_scene.num_views):
        expected = default_scene.project_points(centers, index)
        actual = flipped_scene.project_points(centers, index)
        assert np.allclose(actual[0], expected[0], atol=GEOMETRY_TOLERANCE)
        assert np.allclose(actual[1], expected[1], atol=GEOMETRY_TOLERANCE)
    assert 'convention' in default_scene.derived_quantities()['drawing_note']


# ── the region of reconstruction and the drawing distance ───────────────────

def test_ror_cylinder_matches_the_inscribed_ellipse():
    """The drawn region is the ellipse inscribed in the volume's x and y width.

    mbirtorch's default mask is that ellipse, applied to every slice, and
    ``vcd_utils.get_support_radius`` returns its larger semi-axis.
    """
    cfg = CONFIGS_BY_NAME['cone flat']
    _, scene = build_scene(cfg)
    cylinder = scene.ror_cylinder()
    half_x = 0.5 * scene.num_cols * scene.delta_voxel
    half_y = 0.5 * scene.num_rows * scene.delta_voxel_row
    assert cylinder['semi_axis_x'] == pytest.approx(half_x)
    assert cylinder['semi_axis_y'] == pytest.approx(half_y)
    assert cylinder['radius'] == pytest.approx(max(half_x, half_y))
    z_min, z_max = scene.volume_z_range()
    assert cylinder['z_min'] == pytest.approx(z_min)
    assert cylinder['z_max'] == pytest.approx(z_max)

    params = dict(scene.params)
    params['use_ror_mask'] = False
    assert GeometryScene(params, 'cone').ror_cylinder() is None


def test_drawing_distance_does_not_change_the_projection():
    """The drawing distance moves the picture and not the detector indices.

    A parallel projection's detector coordinates do not depend on where along
    the rays the detector plane sits, so a different drawing distance must give
    the same indices and a different drawn position.
    """
    cfg = CONFIGS_BY_NAME['parallel']
    model = probe.build_model(cfg)
    near = GeometryScene.from_model(model, drawing_distance_factor=1.5)
    far = GeometryScene.from_model(model, drawing_distance_factor=4.0)
    assert far.drawing_distance > near.drawing_distance

    voxels = probe.probe_voxels(cfg['recon_shape'])
    centers = near.voxel_centers(voxels)
    for view in range(near.num_views):
        expected = near.project_points(centers, view)
        actual = far.project_points(centers, view)
        assert np.allclose(actual[0], expected[0], atol=GEOMETRY_TOLERANCE)
        assert np.allclose(actual[1], expected[1], atol=GEOMETRY_TOLERANCE)
    assert (np.linalg.norm(far.view(0).detector_center)
            > np.linalg.norm(near.view(0).detector_center))


def test_volume_fits_detector_detects_a_volume_that_does_not():
    """A volume grown past the detector's field of view is reported.

    The statement is geometric and takes no account of the projector's point
    spread, which is what the plan calls the psf-free statement.
    """
    cfg = CONFIGS_BY_NAME['cone flat']
    _, scene = build_scene(cfg)
    fits, overshoot = scene.volume_fits_detector()
    assert fits is True
    assert overshoot == 0.0

    params = dict(scene.params)
    params['recon_shape'] = (60, 60, 8)
    grown = GeometryScene(params, 'cone')
    fits, overshoot = grown.volume_fits_detector()
    assert fits is False
    assert overshoot > 1.0


def test_view_index_is_checked():
    cfg = CONFIGS_BY_NAME['parallel']
    _, scene = build_scene(cfg)
    with pytest.raises(IndexError):
        scene.view(scene.num_views)
    with pytest.raises(IndexError):
        scene.project_points([[0.0, 0.0, 0.0]], -1)


if __name__ == '__main__':
    # Print the per-geometry gate errors for the run record.
    for cfg in probe.CONFIGS:
        test_projection_matches_projector(cfg['name'])
    print(f'{"geometry":16s} {"pairs":>6s} {"max row err":>12s} '
          f'{"max chan err":>13s}')
    for name, (pairs, row_error, channel_error) in MAX_ERRORS.items():
        print(f'{name:16s} {pairs:6d} {row_error:12.4f} {channel_error:13.4f}')
