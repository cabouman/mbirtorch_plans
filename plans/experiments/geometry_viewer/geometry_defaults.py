"""The automatic geometry of an mbirtorch model, computed without mbirtorch.

Why this module exists.  ``GeometryScene`` needs a complete parameter
dictionary: the sinogram shape, the reconstruction shape, the voxel pitch, the
detector pitches and offsets, and the per-view arrays.  A user of the viewer
supplies the scan, and an mbirtorch model supplies the rest, because a model
constructor calls ``auto_set_recon_geometry`` and that method chooses
``recon_shape``, ``delta_voxel``, and (for two geometries) one more parameter.
The web packaging of the viewer runs where mbirtorch and torch are not
installed, so this module computes those same values in plain numpy.

This file is a deliberate copy of mbirtorch's rules.  Every number below is
taken from one of five places in the mbirtorch source: the defaults in
``_utils.py``, each model class's ``__init__``, each model class's
``auto_set_recon_geometry``, ``utilities.calc_tct_recon_params``, and
``utilities.gen_translation_vectors``.  A copy can drift from its original when
the original changes.  ``test_geometry_defaults.py`` is what keeps this copy
honest: it builds the real mbirtorch model and asserts that every parameter a
scene reads matches ``model.get_params(name)``, for the six probe
configurations and for randomized parameter sets in each geometry.  Run that
test against a new mbirtorch clone before trusting this file with it.

What ``default_parameters`` reproduces.  A model constructor stores the
geometry's own parameters, records the sinogram shape, and then runs
``auto_set_recon_geometry`` while the detector parameters still hold their
defaults of one ALU pitch and zero offset.  So the automatic reconstruction
geometry is computed from the default detector, and a detector override is
applied afterwards.  That is what ``set_params`` does in mbirtorch: setting
``delta_det_channel``, ``delta_det_row``, ``det_channel_offset``, or
``det_row_offset`` stores the value and does NOT re-run
``auto_set_recon_geometry``.  A user who wants the automatic geometry of the
overridden detector calls ``auto_set_recon_geometry`` by hand.  This module
matches the behavior a user gets without that call, because that is the
behavior of ``gv_show_example.py`` and of the web app.

The per-view arrays are float32.  Every model constructor casts its view array
with ``np.asarray(..., dtype=np.float32)``, and the automatic geometry then
reads that array.  Two of the rules take a floor or a ceiling of a value
derived from the array, so the cast can change an integer in the
reconstruction shape.  The arrays here carry the same cast, so the two agree.

Run nothing here.  The module has no side effects and no command-line
interface.  ``web/app.py`` and the tests are its callers.
"""

import numpy as np

__all__ = ['default_parameters', 'view_angles', 'helical_z_shifts',
           'azimuth_elevation_pairs', 'translation_vectors',
           'GEOMETRY_KINDS', 'DETECTOR_PARAMETER_NAMES',
           'DEFAULT_DETECTOR', 'AXIAL_PAD_FRACTION']

#: The geometry kinds this module builds parameters for.  These are the kinds
#: ``geometry_scene.GEOMETRY_KINDS`` names, one per mbirtorch model class.
GEOMETRY_KINDS = ('parallel', 'cone', 'multiaxis', 'translation')

#: The detector parameters a caller may override, with the default value each
#: one holds while the automatic reconstruction geometry is computed.  The
#: values are the ones ``_utils._forward_model_defaults_dict`` sets.
DEFAULT_DETECTOR = dict(delta_det_channel=1.0, delta_det_row=1.0,
                        det_channel_offset=0.0, det_row_offset=0.0)

#: The names of those four parameters, in a fixed order.
DETECTOR_PARAMETER_NAMES = ('delta_det_channel', 'delta_det_row',
                            'det_channel_offset', 'det_row_offset')

#: The voxel aspect ratios, from ``_utils._recon_model_defaults_dict``.  The
#: translation geometry replaces the row aspect with a computed one; see
#: :func:`_translation_recon_geometry`.
DEFAULT_VOXEL_ROW_ASPECT = 1.0
DEFAULT_VOXEL_SLICE_ASPECT = 1.0

#: The reconstruction-of-region mask flag, from
#: ``_utils._reconstruction_defaults_dict``.  ``TranslationModel.__init__``
#: overrides it, because a translation scan's object usually spans the whole
#: field of view.
DEFAULT_USE_ROR_MASK = True
TRANSLATION_USE_ROR_MASK = False

#: The cone model's per-end axial padding fraction.  ``ConeBeamModel.__init__``
#: sets it to zero and no parameter a scene reads can change it, so the two
#: padding terms of ``ConeBeamModel.auto_set_recon_geometry`` are exactly zero
#: slices and this module does not compute the ray excess they scale.  A model
#: whose fraction is nonzero has a taller reconstruction than this module
#: reports; the test asserts the equality that the zero fraction gives.
AXIAL_PAD_FRACTION = 0.0

#: The slice offset both the cone and the multiaxis model start from.  The cone
#: model's automatic geometry then replaces it with the center of the helical z
#: range.
DEFAULT_RECON_SLICE_OFFSET = 0.0


# ── the per-view arrays ──────────────────────────────────────────────────────
# These build the view arrays the way gv_show_example.py builds them, so that
# the web app and the example show the same scan for the same numbers.

def view_angles(num_views, start_deg=0.0, end_deg=360.0):
    """Evenly spaced view angles in radians, over a range given in degrees.

    The angles run over the half-open range, so a full turn does not measure
    angle zero twice.  ``gv_show_example.py`` builds one full turn the same
    way, with ``np.linspace(0, 2 * np.pi, num_views, endpoint=False)``.

    Args:
        num_views (int): the number of views, at least one.
        start_deg, end_deg (float, optional): the ends of the angular range in
            degrees.  Defaults to one full turn.

    Returns:
        ndarray: the angles in radians, (num_views,), float32.
    """
    num_views = _positive_count(num_views, 'the number of views')
    start, end = float(start_deg), float(end_deg)
    if start == end:
        raise ValueError('The angular range is empty; give an end angle that '
                         'differs from the start angle.')
    angles = np.linspace(np.radians(start), np.radians(end), num_views,
                         endpoint=False)
    return angles.astype(np.float32)


def helical_z_shifts(num_views, travel_alu):
    """Per-view z shifts of a helical scan, centered on zero.

    The shifts run from minus half the travel to plus half the travel, which is
    the rule ``gv_show_example.py`` uses.  Both ends are measured, so the range
    of the shifts is the travel.

    Args:
        num_views (int): the number of views, at least one.
        travel_alu (float): the total axial travel in ALU.  Zero gives a
            circular scan.

    Returns:
        ndarray: the shifts in ALU, (num_views,), float32.
    """
    num_views = _positive_count(num_views, 'the number of views')
    travel = float(travel_alu)
    shifts = np.linspace(-0.5 * travel, 0.5 * travel, num_views)
    return shifts.astype(np.float32)


def azimuth_elevation_pairs(num_views, start_deg=0.0, end_deg=360.0,
                            elevation_deg=0.0):
    """The (azimuth, elevation) pairs of a multiaxis scan, in radians.

    The azimuths are :func:`view_angles` and the elevation is the same in every
    view, which is the scan ``gv_show_example.py`` builds.

    Args:
        num_views (int): the number of views, at least one.
        start_deg, end_deg (float, optional): the azimuth range in degrees.
        elevation_deg (float, optional): the elevation in degrees, held for
            every view.

    Returns:
        ndarray: the pairs, (num_views, 2), float32.
    """
    azimuths = view_angles(num_views, start_deg, end_deg)
    elevations = np.full(azimuths.shape, np.radians(float(elevation_deg)))
    return np.stack([azimuths, elevations], axis=1).astype(np.float32)


def translation_vectors(num_x_translations, num_z_translations, x_spacing,
                        z_spacing):
    """The object positions of a translation scan, as a grid.

    This is the rule of ``mbirtorch.gen_translation_vectors``, copied: the
    positions run along x inside a loop over z, both centered on zero, and the
    y component is zero everywhere.  The scan has one view per position, so the
    view count is the product of the two counts.

    Args:
        num_x_translations, num_z_translations (int): the grid counts, each at
            least one.
        x_spacing, z_spacing (float): the spacings in ALU.

    Returns:
        ndarray: the positions, (num_x_translations * num_z_translations, 3),
        as (dx, dy, dz), float32.
    """
    num_x = _positive_count(num_x_translations, 'the number of x translations')
    num_z = _positive_count(num_z_translations, 'the number of z translations')
    x_spacing, z_spacing = float(x_spacing), float(z_spacing)

    num_views = num_x * num_z
    vectors = np.zeros((num_views, 3))
    x_center = (num_x - 1) / 2
    z_center = (num_z - 1) / 2
    index = 0
    for row in range(num_z):
        for column in range(num_x):
            vectors[index] = [(column - x_center) * x_spacing, 0,
                              (row - z_center) * z_spacing]
            index += 1
    return vectors.astype(np.float32)


# ── the complete parameter dictionary ────────────────────────────────────────

def default_parameters(kind, sinogram_shape, **geometry):
    """The parameters a scene of this geometry kind needs, with mbirtorch's
    automatic reconstruction geometry.

    The returned dictionary holds exactly the names
    ``geometry_scene.required_parameter_names(kind)`` lists, with the values an
    mbirtorch model of the same geometry would report after construction and
    after a ``set_params`` call carrying the detector overrides.

    Args:
        kind (str): one of :data:`GEOMETRY_KINDS`.
        sinogram_shape (tuple of int): (num_views, num_det_rows,
            num_det_channels).
        **geometry: the geometry's own arguments and the optional detector
            overrides.  A parallel geometry takes ``angles``, (num_views,) in
            radians.  A cone geometry takes ``angles``,
            ``source_detector_dist``, ``source_iso_dist``, and optionally
            ``helical_z_shifts`` and ``use_curved_detector``.  A multiaxis
            geometry takes ``angles``, (num_views, 2) as azimuth and elevation
            in radians.  A translation geometry takes ``translation_vectors``,
            (num_views, 3) in ALU, ``source_detector_dist``, and
            ``source_iso_dist``.  Every kind takes the four detector overrides
            named in :data:`DETECTOR_PARAMETER_NAMES`.

    Returns:
        dict: the parameter values, keyed by mbirtorch parameter name.

    Raises:
        ValueError: for an unknown geometry kind, an unknown or missing
            geometry argument, or a shape that does not match a view array.
    """
    if kind not in GEOMETRY_KINDS:
        raise ValueError(f'Unknown geometry kind {kind!r}; expected one of '
                         f'{GEOMETRY_KINDS}.')
    shape = tuple(int(n) for n in sinogram_shape)
    if len(shape) != 3:
        raise ValueError('sinogram_shape must hold three entries: the number '
                         'of views, of detector rows, and of detector '
                         f'channels; got {sinogram_shape!r}.')
    for size, name in zip(shape, ('views', 'detector rows',
                                  'detector channels')):
        if size < 1:
            raise ValueError(f'The number of {name} must be at least one; got '
                             f'{size}.')

    overrides = {name: geometry.pop(name) for name in DETECTOR_PARAMETER_NAMES
                 if name in geometry}
    detector = dict(DEFAULT_DETECTOR)

    builders = dict(parallel=_parallel_parameters, cone=_cone_parameters,
                    multiaxis=_multiaxis_parameters,
                    translation=_translation_parameters)
    params = builders[kind](shape, detector, geometry)

    params['sinogram_shape'] = shape
    params.update(detector)
    for name, value in overrides.items():
        params[name] = float(value)
    return params


def _parallel_parameters(shape, detector, geometry):
    """The parameters of a parallel geometry.

    The rule is ``ParallelBeamModel.auto_set_recon_geometry``.  The
    magnification is one, the voxel pitch is the channel pitch, the two lateral
    counts are ceilings and the slice count is a rounding, and the slice count
    therefore equals the detector's row count while the two pitches are equal.
    """
    angles = _view_array(geometry.pop('angles', None), 'angles', 1, shape[0])
    _no_extra_arguments(geometry, 'parallel')

    _, num_det_rows, num_det_channels = shape
    delta_det_channel = detector['delta_det_channel']
    delta_det_row = detector['delta_det_row']
    magnification = 1.0
    delta_voxel = delta_det_channel / magnification
    delta_voxel_row = DEFAULT_VOXEL_ROW_ASPECT * delta_voxel

    num_recon_rows = int(np.ceil(num_det_channels * delta_det_channel
                                 / (delta_voxel_row * magnification)))
    num_recon_cols = int(np.ceil(num_det_channels * delta_det_channel
                                 / (delta_voxel * magnification)))
    num_recon_slices = int(np.round(num_det_rows
                                    * ((delta_det_row / delta_voxel)
                                       / magnification)))
    return dict(
        recon_shape=(num_recon_rows, num_recon_cols, num_recon_slices),
        delta_voxel=delta_voxel,
        voxel_row_aspect=DEFAULT_VOXEL_ROW_ASPECT,
        voxel_slice_aspect=DEFAULT_VOXEL_SLICE_ASPECT,
        use_ror_mask=DEFAULT_USE_ROR_MASK,
        angles=angles,
    )


def _cone_parameters(shape, detector, geometry):
    """The parameters of a cone geometry, circular or helical.

    The rule is ``ConeBeamModel.auto_set_recon_geometry``.  The lateral counts
    are the detector's channel coverage at iso, the slice count covers the
    detector's height at iso swept over the helical travel, and the slice
    offset is the center of the helical z range.  The per-end axial padding is
    zero slices; see :data:`AXIAL_PAD_FRACTION`.
    """
    angles = _view_array(geometry.pop('angles', None), 'angles', 1, shape[0])
    shifts = geometry.pop('helical_z_shifts', None)
    if shifts is None:
        shifts = np.zeros_like(angles)
    else:
        shifts = _view_array(shifts, 'helical_z_shifts', 1, shape[0])
    use_curved_detector = bool(geometry.pop('use_curved_detector', False))
    source_detector_dist, source_iso_dist = _two_distances(geometry, 'cone')
    _no_extra_arguments(geometry, 'cone')

    view_params_array = np.stack([angles, shifts], axis=1)
    if np.isinf(source_detector_dist):
        magnification = 1
    else:
        magnification = source_detector_dist / source_iso_dist

    _, num_det_rows, num_det_channels = shape
    delta_det_channel = detector['delta_det_channel']
    delta_det_row = detector['delta_det_row']
    delta_voxel = delta_det_channel / magnification
    delta_voxel_row = DEFAULT_VOXEL_ROW_ASPECT * delta_voxel
    delta_voxel_slice = DEFAULT_VOXEL_SLICE_ASPECT * delta_voxel

    num_recon_rows = int(np.round(num_det_channels
                                  * ((delta_det_channel / delta_voxel_row)
                                     / magnification)))
    num_recon_cols = int(np.round(num_det_channels
                                  * ((delta_det_channel / delta_voxel)
                                     / magnification)))
    z_shifts = view_params_array[:, 1]
    z_min, z_max = float(np.min(z_shifts)), float(np.max(z_shifts))
    z_travel = z_max - z_min
    height_at_iso = num_det_rows * (delta_det_row / magnification)
    num_recon_slices = max(1, int(np.ceil((height_at_iso + z_travel)
                                          / delta_voxel_slice)))
    recon_slice_offset = 0.5 * (z_min + z_max)

    return dict(
        recon_shape=(num_recon_rows, num_recon_cols, num_recon_slices),
        delta_voxel=delta_voxel,
        voxel_row_aspect=DEFAULT_VOXEL_ROW_ASPECT,
        voxel_slice_aspect=DEFAULT_VOXEL_SLICE_ASPECT,
        use_ror_mask=DEFAULT_USE_ROR_MASK,
        view_params_array=view_params_array,
        source_detector_dist=source_detector_dist,
        source_iso_dist=source_iso_dist,
        use_curved_detector=use_curved_detector,
        recon_slice_offset=recon_slice_offset,
    )


def _multiaxis_parameters(shape, detector, geometry):
    """The parameters of a multiaxis parallel geometry.

    The rule is ``MultiAxisParallelModel.auto_set_recon_geometry``.  The volume
    is the largest box the detector covers: its lateral half width is half the
    detector's channel coverage, and its z half height is half the row coverage
    divided by the smallest ``|cos(elevation)|`` over the views, floored at
    0.1 so that a view looking straight down does not ask for infinite z.
    """
    angles = _view_array(geometry.pop('angles', None), 'angles', 2, shape[0])
    _no_extra_arguments(geometry, 'multiaxis')

    _, num_det_rows, num_det_channels = shape
    delta_det_channel = detector['delta_det_channel']
    delta_det_row = detector['delta_det_row']
    max_u = (num_det_channels * delta_det_channel) / 2.0
    max_v = (num_det_rows * delta_det_row) / 2.0

    elevations = angles[:, 1]
    max_radius_xy = max_u
    min_cos_elevation = np.min(np.abs(np.cos(elevations)))
    min_cos_elevation = max(min_cos_elevation, 0.1)
    max_radius_z = max_v / min_cos_elevation

    delta_voxel = delta_det_channel
    delta_voxel_row = DEFAULT_VOXEL_ROW_ASPECT * delta_voxel
    delta_voxel_slice = DEFAULT_VOXEL_SLICE_ASPECT * delta_voxel
    num_recon_cols = int(np.floor(2 * max_radius_xy / delta_voxel))
    num_recon_rows = int(np.floor(2 * max_radius_xy / delta_voxel_row))
    num_recon_slices = int(np.floor(2 * max_radius_z / delta_voxel_slice))

    return dict(
        recon_shape=(num_recon_rows, num_recon_cols, num_recon_slices),
        delta_voxel=delta_voxel,
        voxel_row_aspect=DEFAULT_VOXEL_ROW_ASPECT,
        voxel_slice_aspect=DEFAULT_VOXEL_SLICE_ASPECT,
        use_ror_mask=DEFAULT_USE_ROR_MASK,
        angles=angles,
        recon_slice_offset=DEFAULT_RECON_SLICE_OFFSET,
    )


def _translation_parameters(shape, detector, geometry):
    """The parameters of a translation geometry.

    The rule is ``TranslationModel.auto_set_recon_geometry``, which calls
    ``utilities.calc_tct_recon_params``.  That function chooses the voxel
    pitch, the row aspect ratio, and all three counts; see
    :func:`_translation_recon_geometry`.  The translation model also turns the
    cylindrical mask off, because its object usually spans the whole field of
    view.
    """
    vectors = _view_array(geometry.pop('translation_vectors', None),
                          'translation_vectors', 3, shape[0])
    source_detector_dist, source_iso_dist = _two_distances(geometry,
                                                           'translation')
    _no_extra_arguments(geometry, 'translation')
    if np.isinf(source_detector_dist):
        raise ValueError('The translation geometry needs a finite '
                         'source-detector distance; every view would '
                         'otherwise carry the same information.')

    recon_shape, delta_voxel, voxel_row_aspect = _translation_recon_geometry(
        source_detector_dist, source_iso_dist, detector['delta_det_row'],
        detector['delta_det_channel'], shape, vectors,
        DEFAULT_VOXEL_ROW_ASPECT, DEFAULT_VOXEL_SLICE_ASPECT)

    return dict(
        recon_shape=recon_shape,
        delta_voxel=delta_voxel,
        voxel_row_aspect=voxel_row_aspect,
        voxel_slice_aspect=DEFAULT_VOXEL_SLICE_ASPECT,
        use_ror_mask=TRANSLATION_USE_ROR_MASK,
        translation_vectors=vectors,
        source_detector_dist=source_detector_dist,
        source_iso_dist=source_iso_dist,
    )


def _translation_recon_geometry(source_detector_dist, source_iso_dist,
                                delta_det_row, delta_det_channel,
                                sinogram_shape, vectors, voxel_row_aspect,
                                voxel_slice_aspect):
    """``mbirtorch.utilities.calc_tct_recon_params``, copied.

    The voxel pitch is the coarser of the two detector pitches at iso.  The row
    pitch follows a heuristic on the average view slope, and the row aspect
    ratio is that pitch over the voxel pitch.  The column and slice counts
    cover the range of the translations.  The row count is chosen so that the
    number of unknowns is twice the number of measurements, and is capped so
    that the volume reaches no further than halfway to the source.

    Returns:
        (tuple, float, float): the reconstruction shape, the voxel pitch, and
        the row aspect ratio.
    """
    num_views, num_det_rows, num_det_channels = sinogram_shape
    magnification = source_detector_dist / source_iso_dist

    detect_box = np.array([delta_det_channel * num_det_channels,
                           delta_det_row * num_det_rows])
    average_view_slope = (detect_box / 4) / source_detector_dist

    pitch_at_iso_vector = (np.array([delta_det_row, delta_det_channel])
                           / magnification)
    pitch_at_iso = np.max(pitch_at_iso_vector)
    delta_voxel = float(pitch_at_iso)
    delta_voxel_slice = voxel_slice_aspect * delta_voxel

    nominal_row_pitch = 4.0 * pitch_at_iso_vector / average_view_slope
    nominal_row_pitch = np.max(nominal_row_pitch)
    delta_recon_row = float(np.maximum(nominal_row_pitch, pitch_at_iso))
    if voxel_row_aspect == 1.0:
        voxel_row_aspect = delta_recon_row / delta_voxel
    else:
        delta_recon_row = voxel_row_aspect * delta_voxel

    cube = np.amax(vectors, axis=0) - np.amin(vectors, axis=0)
    recon_box = np.ceil(np.array([cube[0], cube[2]])
                        / np.array([delta_voxel, delta_voxel_slice]))

    num_pixels_per_view = ((recon_box[0] + num_det_rows)
                           * (recon_box[1] + num_det_channels)) / num_views
    num_measurements_per_view = num_det_channels * num_det_rows
    num_recon_rows = 2 * np.ceil(num_measurements_per_view
                                 / num_pixels_per_view)
    # The volume must reach no further than halfway to the source.  mbirtorch
    # prints a message when this cap falls below one row; here the cap is
    # returned as it is, and the scene then reports a volume of zero rows.
    max_recon_rows = np.floor((source_iso_dist - cube[1]) / delta_recon_row)
    num_recon_rows = np.minimum(num_recon_rows, max_recon_rows)

    num_recon_cols, num_recon_slices = recon_box
    recon_shape = (int(num_recon_rows), int(num_recon_cols),
                   int(num_recon_slices))
    return recon_shape, delta_voxel, voxel_row_aspect


# ── argument checking ────────────────────────────────────────────────────────

def _positive_count(value, name):
    """``value`` as an integer of at least one, or a ValueError naming it."""
    count = int(value)
    if count < 1:
        raise ValueError(f'{name.capitalize()} must be at least one; got '
                         f'{count}.')
    return count


def _view_array(value, name, columns, num_views):
    """One per-view array, cast to float32 and checked against the view count.

    Args:
        value (array_like or None): the array a caller passed.
        name (str): the argument's name, for the error message.
        columns (int): 1 for a plain per-view array, or the number of columns.
        num_views (int): the view count ``sinogram_shape`` holds.

    Returns:
        ndarray: the array, float32, of shape (num_views,) or
        (num_views, columns).
    """
    if value is None:
        raise ValueError(f'A {name} array is needed for this geometry.')
    array = np.asarray(value, dtype=np.float32)
    if columns == 1:
        array = array.reshape(-1)
        wanted = f'({num_views},)'
    else:
        wanted = f'({num_views}, {columns})'
        if array.ndim != 2 or array.shape[1] != columns:
            raise ValueError(f'{name} must have shape {wanted}; got '
                             f'{tuple(array.shape)}.')
    if array.shape[0] != num_views:
        raise ValueError(f'{name} holds {array.shape[0]} views but '
                         f'sinogram_shape holds {num_views}.')
    return array


def _two_distances(geometry, kind):
    """The source-detector and source-iso distances, checked."""
    for name in ('source_detector_dist', 'source_iso_dist'):
        if name not in geometry:
            raise ValueError(f'A {kind} geometry needs {name}.')
    source_detector_dist = float(geometry.pop('source_detector_dist'))
    source_iso_dist = float(geometry.pop('source_iso_dist'))
    if source_iso_dist == 0.0:
        raise ValueError('The source-iso distance must not be zero.')
    return source_detector_dist, source_iso_dist


def _no_extra_arguments(geometry, kind):
    """Raise when a caller passed an argument this geometry does not use."""
    if geometry:
        raise ValueError(f'A {kind} geometry does not use these arguments: '
                         f'{sorted(geometry)}.')
