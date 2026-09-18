"""Pin geometry_defaults.py against the real mbirtorch models.

Why this test exists.  ``geometry_defaults.py`` is a copy of mbirtorch's
parameter defaults and of each model's ``auto_set_recon_geometry``, written so
that the web packaging of the viewer can build a scene where mbirtorch and
torch are not installed.  A copy can drift from its original.  This test is
what keeps the copy honest: it builds the real model, calls
``default_parameters`` with the same arguments, and asserts that the two agree
on every parameter a scene reads.

What is compared.  The names are
``geometry_scene.required_parameter_names(kind)``, which is exactly the set
``GeometryScene.from_model`` reads.  Shapes, integers, booleans, and strings
must be equal.  Floats must agree to a relative 1e-6, and a per-view array must
agree entry by entry to 1e-6.  The tolerance covers the float32 storage the
models use for their view arrays and nothing more, so a changed rule fails
here.

The cases.  The first case is the six scan geometries of
``gv1_conventions_probe.py``, built from that script's own constructor
arguments and detector parameters.  The second is a randomized set for each of
the four geometry kinds: ten or more parameter sets per kind, from a seeded
generator, so that a rule that happens to agree at one size does not pass.

Two copied helpers are checked on their own, because no model reports them.
``translation_vectors`` is compared with ``gen_translation_vectors`` and
``cube_phantom`` with ``gen_cube_phantom``.

Run:
    cd plans/geometry_viewer/experiments
    PYTHONPATH=<mbirtorch clone> python -m pytest -q test_geometry_defaults.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mbirtorch  # noqa: E402

import geometry_defaults  # noqa: E402
import gv1_conventions_probe as probe  # noqa: E402
from geometry_defaults import (DETECTOR_PARAMETER_NAMES,  # noqa: E402
                               default_parameters)
from geometry_scene import (GEOMETRY_KINDS,  # noqa: E402
                            required_parameter_names)

#: The relative tolerance for a float and for one entry of a per-view array.
#: The models store their view arrays as float32, whose relative resolution is
#: about 6e-8, so this tolerance passes that storage and little else.
FLOAT_TOLERANCE = 1e-6

#: How many randomized parameter sets each geometry kind is checked with.
RANDOM_CASES = 12

#: The seed of each kind's generator, so that a failure is reproducible.
SEEDS = dict(parallel=1, cone=2, multiaxis=3, translation=4)


# ── comparing one model with one dictionary ──────────────────────────────────

def assert_parameters_match(kind, model, params, case=''):
    """Assert that every scene parameter of ``model`` equals ``params``.

    Args:
        kind (str): the geometry kind.
        model: the mbirtorch model.
        params (dict): what ``default_parameters`` returned.
        case (str, optional): a description added to a failure message.
    """
    names = required_parameter_names(kind)
    assert set(params) == set(names), (
        f'{case}: default_parameters returned the names {sorted(params)}, but '
        f'a {kind} scene needs {sorted(names)}.')

    for name in names:
        mine = params[name]
        theirs = model.get_params(name)
        message = f'{case}: {name} is {mine!r} here and {theirs!r} in the model'

        if isinstance(theirs, (bool, str)) or isinstance(mine, (bool, str)):
            assert type(mine) is type(theirs), message
            assert mine == theirs, message
            continue
        if isinstance(theirs, tuple):
            assert tuple(mine) == tuple(theirs), message
            assert all(isinstance(entry, int) for entry in mine), (
                f'{case}: {name} must hold plain integers; got {mine!r}.')
            continue

        mine_array = np.asarray(mine, dtype=np.float64)
        theirs_array = np.asarray(theirs, dtype=np.float64)
        assert mine_array.shape == theirs_array.shape, message
        assert np.allclose(mine_array, theirs_array, rtol=FLOAT_TOLERANCE,
                           atol=0.0), message


def build_model(kind, sinogram_shape, geometry, detector):
    """Build the mbirtorch model of one case.

    The model is constructed from the geometry's own arguments, so that its
    constructor runs ``auto_set_recon_geometry`` exactly as a user's would.
    The detector overrides are set afterwards, which is the order
    ``default_parameters`` reproduces and the order a user follows.

    Args:
        kind (str): the geometry kind.
        sinogram_shape (tuple of int): the sinogram shape.
        geometry (dict): the geometry's own arguments, as
            ``default_parameters`` takes them.
        detector (dict): the detector overrides, or an empty dictionary.

    Returns:
        A ``TomographyModel`` subclass instance.
    """
    if kind == 'parallel':
        model = mbirtorch.ParallelBeamModel(sinogram_shape,
                                            geometry['angles'],
                                            compile_mode='off')
    elif kind == 'cone':
        model = mbirtorch.ConeBeamModel(
            sinogram_shape, geometry['angles'],
            source_detector_dist=geometry['source_detector_dist'],
            source_iso_dist=geometry['source_iso_dist'],
            helical_z_shifts=geometry.get('helical_z_shifts'),
            use_curved_detector=geometry.get('use_curved_detector', False),
            compile_mode='off')
    elif kind == 'multiaxis':
        model = mbirtorch.MultiAxisParallelModel(sinogram_shape,
                                                 geometry['angles'],
                                                 compile_mode='off')
    else:
        model = mbirtorch.TranslationModel(
            sinogram_shape, geometry['translation_vectors'],
            source_detector_dist=geometry['source_detector_dist'],
            source_iso_dist=geometry['source_iso_dist'],
            compile_mode='off')
    if detector:
        # no_warning suppresses the notice that a reconstruction parameter was
        # changed by hand.  This call does not re-run auto_set_recon_geometry,
        # which is the mbirtorch behavior default_parameters matches.
        model.set_params(no_warning=True, **detector)
    return model


def check_case(kind, sinogram_shape, geometry, detector, case):
    """Build both sides of one case and compare them."""
    model = build_model(kind, sinogram_shape, geometry, detector)
    params = default_parameters(kind, sinogram_shape, **geometry, **detector)
    assert_parameters_match(kind, model, params, case)


# ── the six probe geometries ─────────────────────────────────────────────────

def probe_case(name):
    """The geometry arguments and detector overrides of one probe geometry.

    ``gv1_conventions_probe.py`` sets a reconstruction shape and voxel
    parameters by hand, so that its footprints stay on the detector.  Those
    are not automatic geometry and are left out here.  What is taken is the
    geometry itself: the sinogram shape, the view arrays, the two distances,
    the curved-detector flag, and the four detector parameters.
    """
    cfg = dict((entry['name'], entry) for entry in probe.CONFIGS)[name]
    kind = cfg['kind']
    geometry = {}
    if kind == 'cone':
        geometry['angles'] = cfg['angles']
        geometry['helical_z_shifts'] = cfg['z_shifts']
        geometry['use_curved_detector'] = cfg['use_curved_detector']
    elif kind == 'multiaxis':
        geometry['angles'] = np.stack([cfg['angles'], cfg['elevations']],
                                      axis=1)
    elif kind == 'parallel':
        geometry['angles'] = cfg['angles']
    else:
        geometry['translation_vectors'] = cfg['translation_vectors']
    for name_of_distance in ('source_detector_dist', 'source_iso_dist'):
        if name_of_distance in cfg:
            geometry[name_of_distance] = cfg[name_of_distance]
    detector = {name_of_parameter: cfg['params'][name_of_parameter]
                for name_of_parameter in DETECTOR_PARAMETER_NAMES}
    return kind, cfg['sinogram_shape'], geometry, detector


@pytest.mark.parametrize('name', [cfg['name'] for cfg in probe.CONFIGS])
def test_probe_geometry(name):
    """Each probe geometry's automatic parameters match its model's."""
    kind, sinogram_shape, geometry, detector = probe_case(name)
    check_case(kind, sinogram_shape, geometry, detector, f'probe {name}')


# ── randomized parameter sets ────────────────────────────────────────────────

def random_case(kind, rng, elevation_limit_deg=44.0):
    """One randomized case, as the arguments both sides take.

    The ranges are wide enough that the counts, the pitches, and the distances
    all change, and narrow enough that every case is a geometry a model
    accepts: the translation geometry's volume stays well inside the source
    distance.

    Args:
        kind (str): the geometry kind.
        rng: a numpy generator.
        elevation_limit_deg (float, optional): the largest multiaxis elevation
            in degrees.  The default stays under the 45 degrees above which
            the model warns.  A caller that wants the elevation clamp of the
            multiaxis rule exercised passes a larger limit.
    """
    num_views = int(rng.integers(2, 64))
    num_det_rows = int(rng.integers(4, 80))
    num_det_channels = int(rng.integers(4, 96))
    detector = dict(
        delta_det_channel=float(rng.uniform(0.3, 3.0)),
        delta_det_row=float(rng.uniform(0.3, 3.0)),
        det_channel_offset=float(rng.uniform(-6.0, 6.0)),
        det_row_offset=float(rng.uniform(-6.0, 6.0)),
    )
    source_detector_dist = float(rng.uniform(150.0, 2000.0))
    source_iso_dist = source_detector_dist * float(rng.uniform(0.2, 0.9))
    start_deg = float(rng.uniform(-90.0, 90.0))
    end_deg = start_deg + float(rng.uniform(30.0, 360.0))

    if kind == 'parallel':
        geometry = dict(angles=geometry_defaults.view_angles(
            num_views, start_deg, end_deg))
    elif kind == 'cone':
        geometry = dict(
            angles=geometry_defaults.view_angles(num_views, start_deg,
                                                 end_deg),
            source_detector_dist=source_detector_dist,
            source_iso_dist=source_iso_dist,
            use_curved_detector=bool(rng.integers(0, 2)))
        if rng.integers(0, 2):
            geometry['helical_z_shifts'] = geometry_defaults.helical_z_shifts(
                num_views, float(rng.uniform(0.0, 120.0)))
    elif kind == 'multiaxis':
        geometry = dict(angles=geometry_defaults.azimuth_elevation_pairs(
            num_views, start_deg, end_deg,
            float(rng.uniform(-elevation_limit_deg, elevation_limit_deg))))
    else:
        num_x = int(rng.integers(1, 8))
        num_z = int(rng.integers(1, 8))
        num_views = num_x * num_z
        geometry = dict(
            translation_vectors=geometry_defaults.translation_vectors(
                num_x, num_z, float(rng.uniform(1.0, 30.0)),
                float(rng.uniform(1.0, 30.0))),
            source_detector_dist=source_detector_dist,
            source_iso_dist=source_iso_dist)
    shape = (num_views, num_det_rows, num_det_channels)
    return shape, geometry, detector


@pytest.mark.parametrize('kind', GEOMETRY_KINDS)
def test_randomized_geometries(kind):
    """Randomized parameter sets of each kind match their models.

    The generator's seed is fixed per kind, so a failure names a case that can
    be built again.
    """
    rng = np.random.default_rng(SEEDS[kind])
    for case in range(RANDOM_CASES):
        sinogram_shape, geometry, detector = random_case(kind, rng)
        check_case(kind, sinogram_shape, geometry, detector,
                   f'{kind} random case {case}')


@pytest.mark.filterwarnings('ignore:One or more elevation angles')
@pytest.mark.parametrize('kind', GEOMETRY_KINDS)
def test_formulas_at_other_detector_values(kind, monkeypatch):
    """The copied rules match the model's rules at any detector, not only the
    default one.

    Why this test is needed beside the one above.  A model constructor runs
    ``auto_set_recon_geometry`` while the detector still holds its defaults of
    one ALU pitch and zero offset, so a case built through a constructor
    exercises the copied formulas at those values alone.  Two pieces of the
    formulas are then invisible: the parallel rule's ratio of the row pitch to
    the voxel pitch, which is one at the default detector, and the multiaxis
    rule's floor on the cosine of the elevation, which binds only above 84
    degrees.

    How the other values are reached.  ``DEFAULT_DETECTOR`` is what
    ``default_parameters`` computes the automatic geometry from, so this test
    replaces it with the case's own detector.  On the model's side the
    detector is set and ``auto_set_recon_geometry`` is called again, which is
    the call a user makes after changing a detector parameter.  The elevation
    range is widened so that the cosine floor binds in some cases.
    """
    rng = np.random.default_rng(100 + SEEDS[kind])
    for case in range(RANDOM_CASES):
        sinogram_shape, geometry, detector = random_case(
            kind, rng, elevation_limit_deg=89.0)
        monkeypatch.setattr(geometry_defaults, 'DEFAULT_DETECTOR',
                            dict(detector))
        model = build_model(kind, sinogram_shape, geometry, detector)
        if kind == 'translation':
            # The first automatic run replaced the row aspect ratio with its
            # own value, and the rule keeps a ratio that is not one.  Putting
            # the ratio back to one lets the second run choose it again, which
            # is the state a construction with this detector would be in.
            model.set_params(no_warning=True, voxel_row_aspect=1.0)
        model.auto_set_recon_geometry(no_compile=True, no_warning=True)
        params = default_parameters(kind, sinogram_shape, **geometry,
                                    **detector)
        assert_parameters_match(kind, model, params,
                                f'{kind} detector case {case}')


# ── the two rules the copy states in words ───────────────────────────────────

def test_detector_override_does_not_change_the_recon_shape():
    """A detector override leaves the automatic reconstruction geometry alone.

    This is the mbirtorch behavior ``default_parameters`` matches: setting a
    detector parameter through ``set_params`` stores the value and does not
    re-run ``auto_set_recon_geometry``.  A user who wants the automatic
    geometry of the new detector calls that method by hand.
    """
    shape = (36, 40, 64)
    angles = geometry_defaults.view_angles(36)
    plain = default_parameters('cone', shape, angles=angles,
                               source_detector_dist=800.0,
                               source_iso_dist=400.0)
    overridden = default_parameters('cone', shape, angles=angles,
                                    source_detector_dist=800.0,
                                    source_iso_dist=400.0,
                                    delta_det_channel=2.5,
                                    det_channel_offset=7.0)
    assert overridden['recon_shape'] == plain['recon_shape']
    assert overridden['delta_voxel'] == plain['delta_voxel']
    assert overridden['delta_det_channel'] == 2.5
    assert overridden['det_channel_offset'] == 7.0

    model = mbirtorch.ConeBeamModel(shape, angles,
                                    source_detector_dist=800.0,
                                    source_iso_dist=400.0, compile_mode='off')
    model.set_params(no_warning=True, delta_det_channel=2.5,
                     det_channel_offset=7.0)
    assert tuple(model.get_params('recon_shape')) == plain['recon_shape']


def test_translation_vectors_match_mbirtorch():
    """The translation grid is the one ``gen_translation_vectors`` builds."""
    mine = geometry_defaults.translation_vectors(5, 3, 20.0, 12.0)
    theirs = mbirtorch.gen_translation_vectors(5, 3, 20.0, 12.0)
    assert mine.shape == theirs.shape
    assert np.allclose(mine, theirs, rtol=FLOAT_TOLERANCE, atol=0.0)


#: The reconstruction shapes the cube phantom is checked at.  The first four
#: are the shapes the web page's six geometries choose at their default
#: controls, which are the shapes the page builds a phantom for; three of the
#: six geometries share (128, 128, 96).  The rest are awkward on purpose: a
#: shape not divisible by four, many more slices than rows, one slice, a volume
#: too narrow in one direction to hold a block, and the smallest volume the
#: rule gives a block at all.
CUBE_PHANTOM_SHAPES = (
    (128, 128, 96), (128, 128, 196), (128, 128, 110), (6, 160, 48),
    (13, 7, 5), (9, 9, 40), (32, 32, 1), (3, 20, 6), (4, 4, 4),
)


@pytest.mark.parametrize('shape', CUBE_PHANTOM_SHAPES)
def test_cube_phantom_matches_mbirtorch(shape):
    """The cube phantom is the one ``gen_cube_phantom`` builds, exactly.

    Exact equality is the right test here, not a tolerance.  Every index in the
    rule is integer arithmetic, and the one value the phantom carries is a
    single division that both sides compute the same way, so the two arrays
    agree bit for bit or the rule has been copied wrongly.
    """
    mine = geometry_defaults.cube_phantom(shape)
    theirs = np.asarray(mbirtorch.gen_cube_phantom(shape).cpu())
    assert mine.dtype == np.float32
    assert mine.shape == theirs.shape
    assert np.array_equal(mine, theirs)


# ── the errors a bad call raises ─────────────────────────────────────────────

def test_unknown_kind_and_missing_arguments_raise():
    """A wrong kind, a missing argument, and a mismatched view count raise."""
    with pytest.raises(ValueError, match='Unknown geometry kind'):
        default_parameters('fan', (8, 8, 8), angles=np.zeros(8))
    with pytest.raises(ValueError, match='needs source_detector_dist'):
        default_parameters('cone', (8, 8, 8), angles=np.zeros(8))
    with pytest.raises(ValueError, match='does not use these arguments'):
        default_parameters('parallel', (8, 8, 8), angles=np.zeros(8),
                           source_iso_dist=1.0)
    with pytest.raises(ValueError, match='holds 4 views'):
        default_parameters('parallel', (8, 8, 8), angles=np.zeros(4))
