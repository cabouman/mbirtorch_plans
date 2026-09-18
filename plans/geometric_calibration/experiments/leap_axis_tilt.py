"""Does a rotation axis that leans out of the detector plane move mbirtorch's in-plane rotation
estimate, and does a taller comparison band remove that movement?

``estimate_det_rotation`` in ``mbirtorch/preprocess/geometry_calibration.py`` compares a band of
detector rows around the central plane with the mirrored opposite views.  On a real NSI scan it read
0.047 degrees where 0.167 degrees is the right answer.  That scan's configuration vectors say the
physical misalignment has two parts: an in-plane rotation of the detector relative to the rotation
axis, 0.167 degrees, which rotating the sinogram corrects, and a lean of the rotation axis along the
detector normal, 0.079 degrees, which no in-plane rotation of the sinogram can correct.  The
hypothesis this job tests is that the second part biases the estimate of the first at the band the
module picks by default.

The data are made by LEAP, a third-party projector that shares no code with mbirtorch.  LEAP's
modular-beam geometry takes a source position, a detector center, and two detector direction vectors
for every view, so a scan whose rotation axis points anywhere can be generated inside the projector,
with no resampling of a sinogram anywhere.  Each case here orbits the nominal cone-beam gantry about
a tilted axis: the axis is the z axis turned within the detector plane by the in-plane angle and then
leaned along the detector normal by the out-of-plane angle, both measured at the view where the
gantry is in its nominal pose.  At that view the detector is exactly nominal and the axis is what
leans, which is how the real scanner is misaligned.  mbirtorch's estimators then run on the resulting
sinograms.

Two phantoms are used, both from the earlier laptop run of this question.  Each is a cylinder that is
the same in every slice with one darker slab of slices in it.  In the 'far' phantom the slab sits 78
percent of the way from the central slice to the top of the volume, outside the default band; in the
'near' phantom it sits on the central plane, inside it.  Each phantom is run at four angle pairs: no
tilt, in-plane only, out-of-plane only, and both.  The in-plane angle of 1.5 degrees displaces the
edge channel by 2.1 pixels, which is what the earlier run used.  The out-of-plane angle of 0.65
degrees displaces a point at the cylinder's radius vertically by about 0.6 pixels between a view and
its opposite, which is the real scan's regime scaled to this much smaller detector: the real scan
leans 0.079 degrees at an object radius of about 236 pixels, and the cylinder here has a radius of
about 28 pixels on the detector, so the angle scales up by about eight.

Before any case runs, the same phantom is projected twice, once with LEAP's own cone-beam geometry
and once with the no-tilt modular geometry derived here, and the largest difference between them is
recorded.  The two geometry types use different kernels, so a small difference is expected, but a
large one means the modular geometry is not the cone-beam geometry and nothing after it means
anything.

What to expect, as things to compare the numbers against rather than as assertions:

  * The pair (0, 0) is the control.  It says whether the geometry and the conversion into mbirtorch's
    sinogram convention are right: the offset should read about zero channels and the rotation about
    zero degrees at every band.
  * The pair (1.5, 0) should reproduce the earlier laptop run: the default band should under-read the
    1.5 degrees on the far phantom, and the taller bands should recover it.  The sign is recorded,
    not predicted, because it depends on conventions on both sides.
  * The pairs (0, 0.65) and (1.5, 0.65) are the new measurement.  The first says whether a lean alone
    moves the estimate off zero; the second says whether it changes an estimate that has a real
    in-plane angle to find.

Everything is printed as aligned tables and appended to a JSON-lines file, one record per
measurement, so a job that runs out of time still leaves what it finished.

Run on one GPU through ``leap_axis_tilt.sbatch``.
"""
import json
import math
import os
import resource
import sys
import time
import traceback
import warnings

os.environ.setdefault('MBIRTORCH_NUM_DEVICES', '1')

import numpy as np
import torch

import mbirtorch
from mbirtorch.preprocess import geometry_calibration as gc

# ── run parameters ────────────────────────────────────────────────────────────────────────────────
NUM_THREADS = 14                    # the CPUs this job asks for; the estimators run on the host

# The geometry, matching the earlier laptop run of this question.  Both detector pitches are 1 mm, so
# a length in mm is a length in detector pixels, and the two source distances give a magnification of
# two.
NUM_VIEWS = 128
NUM_DET_ROWS = 128
NUM_DET_CHANNELS = 160
DELTA_DET_CHANNEL = 1.0
DELTA_DET_ROW = 1.0
SOURCE_ISO_DIST = 400.0
SOURCE_DETECTOR_DIST = 800.0

# The phantom, built on LEAP's own volume grid.  The cylinder's radius is a fraction of the volume's
# half width, the slab is a run of slices multiplied by a value below one, and the slab's center is a
# fraction of the way from the volume's central slice to its top, so zero puts it on the central
# plane.
CYLINDER_RADIUS_FRACTION = 0.35
SLAB_SLICE_FRACTION = 0.04          # slab thickness as a fraction of the volume's slices
SLAB_VALUE = 0.65
PHANTOMS = (
    dict(label='far', slab_fraction=0.78),
    dict(label='near', slab_fraction=0.0),
)

# The angle pairs every phantom is run at, in degrees.  The first of each pair turns the axis within
# the detector plane, which an in-plane rotation of the sinogram can correct; the second leans it
# along the detector normal, which no in-plane rotation can correct.
CASES = (
    (0.0, 0.0),
    (1.5, 0.0),
    (0.0, 0.65),
    (1.5, 0.65),
)

# The band heights each case is estimated at, in detector rows.  None omits the argument, which
# leaves the module its own default; on a cone model the module cuts that default down from 16 rows
# using the cone geometry, and the tables print the height it actually used.  The heights are odd so
# that each band is centered on a row, and the largest covers all but one row of the detector.
BAND_ROWS = (None, 33, 65, 127)

# The check that the modular geometry really is the cone-beam geometry.  The run says so loudly in
# the table when the largest difference, relative to the largest value of the cone-beam projection,
# is above this.
GEOMETRY_CHECK_TOLERANCE = 1e-4

# A second, deliberately lopsided volume for that check.  The phantoms above are the same at every
# angle about the z axis and so cannot show a channel direction that runs the wrong way; this one
# adds a block off the axis and below the central plane, which can.  It is only ever used in the
# check, never in a measured case.
PROBE_VALUE = 0.6
PROBE_OFFSET_FACTOR = 1.6           # block center, in cylinder radii from the axis
PROBE_SIZE_FRACTION = 0.25          # block half width, as a fraction of the cylinder radius
PROBE_HEIGHT_FRACTION = 0.45        # block center, as a fraction of the way to the bottom slice

# The circular mask LEAP applies to the volume is switched off, by asking for a diameter far larger
# than the volume, so that the cone-beam and the modular projections clip the volume identically.
# The phantoms are zero out there in any case.
NO_MASK_DIAMETER = 1.0e6

# Whether to also check the conversion into mbirtorch's sinogram convention by projecting the same
# phantom with mbirtorch itself.  The cylinder with its slab is lopsided along z, so this says
# whether the detector rows of the two libraries run the same way.
CHECK_AGAINST_MBIRTORCH = True

RESULTS = os.environ.get(
    'LEAP_AXIS_TILT_RESULTS',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_leap_axis_tilt'))
JSONL = os.path.join(RESULTS, 'leap_axis_tilt.jsonl')


# ── recording ─────────────────────────────────────────────────────────────────────────────────────

def _plain(value):
    """Convert a value json.dumps cannot serialize, so a numpy scalar or array can be recorded."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def record(kind, seconds, **fields):
    """Append one measurement to the JSON-lines file and print it.

    Every entry carries the kind of measurement and the wall time it took, so the file can be read
    without knowing which fields a kind adds.
    """
    entry = dict(kind=kind, seconds=float(seconds))
    entry.update(fields)
    os.makedirs(RESULTS, exist_ok=True)
    with open(JSONL, 'a') as handle:
        handle.write(json.dumps(entry, default=_plain) + '\n')
    print(json.dumps(entry, default=_plain), flush=True)


# ── the nominal geometry, read off LEAP's own definitions ─────────────────────────────────────────
# set_conebeam's docstring (leapctype.py line 486) writes the forward projection as
#
#   Pf(u, phi, v) = integral f( R theta(phi) - tau theta_perp(phi) + Delta phi z_hat
#                               + l / sqrt(1 + u^2 + v^2)
#                                 [ -theta(phi) + u theta_perp(phi) + v z_hat ] ) dl
#
# with R = sod, u = s / sdd and v = t / sdd, where s and t are the detector coordinates in mm.  With
# tau and the helical pitch both zero, the source of view phi sits at sod * theta(phi), and the point
# of the detector at (s, t) sits at (sod - sdd) * theta(phi) + s theta_perp(phi) + t z_hat.  So the
# module center is at (sod - sdd) theta(phi), the direction of increasing column index is
# theta_perp(phi), and the direction of increasing row index is z_hat.  That is exactly what
# set_modularbeam (line 638) asks for: a source position y, a module center c, a column vector u_hat
# and a row vector v_hat, with the ray to (s, t) running from y to c + s u_hat + t v_hat.
#
# The docstring does not write out theta(phi), but leapctype's own drawing code does.  In drawSystem
# (line 6356 and following) the pose it draws has the source on the +y axis and the detector at -y,
# and it puts detector column 0 at +x and the last column at -x, with the row index increasing with
# z.  With theta on +y, theta_perp = z_hat x theta is on -x, so column 0 is at +s and the last column
# at -s: the column direction is +theta_perp, not -theta_perp, and the row direction is +z_hat.  That
# fixes both signs.  What it does not fix is where phi = 0 sits, because drawSystem's static pose and
# its source trajectory (line 6502) disagree with each other by half a turn.  A different zero is a
# rigid rotation of the whole system about z, and every phantom measured here is the same at every
# angle about z, so it changes none of the measurements; the check below reports the view shift that
# best lines the two geometries up, so the disagreement is measured rather than assumed.
NOMINAL_SOURCE = (SOURCE_ISO_DIST, 0.0, 0.0)
NOMINAL_CENTER = (SOURCE_ISO_DIST - SOURCE_DETECTOR_DIST, 0.0, 0.0)
NOMINAL_COL = (0.0, 1.0, 0.0)       # theta_perp at the nominal pose: increasing column index
NOMINAL_ROW = (0.0, 0.0, 1.0)       # z_hat: increasing row index


def rodrigues(vectors, axis, radians):
    """Rotate each row of ``vectors`` about the unit vector ``axis`` through the origin.

    The rotation is right-handed about ``axis``, so rotating (1, 0, 0) about (0, 0, 1) by an angle
    gives (cos, sin, 0), which is the theta(phi) of set_conebeam's docstring.
    """
    axis = np.asarray(axis, dtype=np.float64)
    vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
    cosine, sine = math.cos(radians), math.sin(radians)
    along = vectors @ axis
    return (vectors * cosine + np.cross(axis, vectors) * sine
            + np.outer(along * (1.0 - cosine), axis))


def tilt_axis(in_plane_degrees, out_of_plane_degrees):
    """The direction the gantry orbits, for one pair of misalignment angles.

    The axis is the nominal z axis turned within the detector plane, about the detector normal, by
    the in-plane angle, and then leaned out of that plane, about the detector's channel direction, by
    the out-of-plane angle, both taken at the nominal pose.  At these angles the order of the two
    does not matter to the fourth decimal place.  A turn within the detector plane is what an
    in-plane rotation of the sinogram can undo; a lean along the normal is what it cannot.
    """
    normal = np.cross(NOMINAL_COL, NOMINAL_ROW)     # u_hat x v_hat, pointing back at the source
    turned = rodrigues(NOMINAL_ROW, normal, math.radians(in_plane_degrees))[0]
    leaned = rodrigues(turned, NOMINAL_COL, math.radians(out_of_plane_degrees))[0]
    return leaned / np.linalg.norm(leaned)


def modular_arrays(phis_degrees, axis):
    """The four modular-beam arrays for a gantry that orbits ``axis``.

    Every view is the nominal pose rotated about the tilted axis by that view's angle, so the view
    whose angle is zero is exactly nominal and the axis is the only thing out of place.  The source
    position and the module center are rotated as points and the two detector vectors as directions,
    which is the same rotation because the axis passes through the center of rotation.
    """
    nominal = np.array([NOMINAL_SOURCE, NOMINAL_CENTER, NOMINAL_ROW, NOMINAL_COL], dtype=np.float64)
    blocks = [rodrigues(nominal, axis, math.radians(float(phi))) for phi in phis_degrees]
    stacked = np.stack(blocks, axis=0)
    return tuple(np.ascontiguousarray(stacked[:, index], dtype=np.float32) for index in range(4))


# ── the phantoms, on LEAP's volume grid ───────────────────────────────────────────────────────────

def cylinder_with_slab(grid, slab_fraction):
    """A cylinder that is the same in every slice, with one darker slab of slices in it.

    The volume is in LEAP's own order, which is (z, y, x).  ``slab_fraction`` is the fraction of the
    way from the central slice to the top of the volume at which the slab is centered, so zero puts
    it on the central plane.  Returns the volume, the slices the slab occupies, and the cylinder's
    radius in voxels.
    """
    num_slices, num_y, num_x = grid['shape']
    rows, columns = np.indices((num_y, num_x))
    radius = CYLINDER_RADIUS_FRACTION * min(num_y, num_x) / 2.0
    inside = (((rows - (num_y - 1) / 2.0) ** 2 + (columns - (num_x - 1) / 2.0) ** 2)
              <= radius ** 2)
    volume = np.zeros((num_slices, num_y, num_x), dtype=np.float32)
    volume[:, inside] = 1.0
    thickness = max(1, int(round(SLAB_SLICE_FRACTION * num_slices)))
    half_extent = (num_slices - 1) / 2.0
    center = half_extent + slab_fraction * half_extent
    low = int(round(center - (thickness - 1) / 2.0))
    low = max(0, min(low, num_slices - thickness))
    volume[low:low + thickness] *= SLAB_VALUE
    return np.ascontiguousarray(volume), (low, low + thickness), radius


def with_probe_block(volume, radius):
    """The same volume with one block added off the axis and below the central plane.

    The cylinder is the same at every angle about z, so a projection of it cannot show a detector
    channel direction that runs backwards.  This block breaks that symmetry, and it also sits away
    from the central plane, so a row direction that runs backwards shows up too.  It is used only in
    the geometry check.
    """
    num_slices, num_y, num_x = volume.shape
    half = max(1, int(round(PROBE_SIZE_FRACTION * radius)))
    center_y = int(round((num_y - 1) / 2.0 + PROBE_OFFSET_FACTOR * radius))
    center_x = int(round((num_x - 1) / 2.0))
    center_z = int(round((num_slices - 1) / 2.0 * (1.0 - PROBE_HEIGHT_FRACTION)))
    out = volume.copy()
    out[max(0, center_z - half):center_z + half + 1,
        max(0, center_y - half):center_y + half + 1,
        max(0, center_x - half):center_x + half + 1] += PROBE_VALUE
    return np.ascontiguousarray(out), (center_z, center_y, center_x, half)


# ── LEAP ──────────────────────────────────────────────────────────────────────────────────────────

def read_volume_grid(leapct):
    """The volume LEAP chose for this geometry, read back from it after set_default_volume."""
    order = int(leapct.get_volumeDimensionOrder())
    num_x, num_y, num_z = (int(leapct.get_numX()), int(leapct.get_numY()), int(leapct.get_numZ()))
    return {
        'dimension_order': order,
        'order_text': 'zyx' if order == 1 else 'xyz',
        'shape': (num_z, num_y, num_x),
        'num_x': num_x, 'num_y': num_y, 'num_z': num_z,
        'voxel_width': float(leapct.get_voxelWidth()),
        'voxel_height': float(leapct.get_voxelHeight()),
        'offset_x': float(leapct.get_offsetX()),
        'offset_y': float(leapct.get_offsetY()),
        'offset_z': float(leapct.get_offsetZ()),
    }


def apply_volume(leapct, grid):
    """Put the recorded volume back on the model, after any change of geometry."""
    leapct.set_volume(grid['num_x'], grid['num_y'], grid['num_z'],
                      voxelWidth=grid['voxel_width'], voxelHeight=grid['voxel_height'],
                      offsetX=grid['offset_x'], offsetY=grid['offset_y'], offsetZ=grid['offset_z'])
    leapct.set_diameterFOV(NO_MASK_DIAMETER)


def set_cone_geometry(leapct, phis, grid=None):
    """LEAP's own cone-beam geometry at the constants at the top of this file.

    The argument order and the centered detector, ``centerRow`` and ``centerCol`` at half of one less
    than the count, are the ones bench_leap_vs_mbirtorch.py uses (its run_leap, lines 425 to 431).
    """
    leapct.set_conebeam(NUM_VIEWS, NUM_DET_ROWS, NUM_DET_CHANNELS, DELTA_DET_ROW, DELTA_DET_CHANNEL,
                        0.5 * (NUM_DET_ROWS - 1), 0.5 * (NUM_DET_CHANNELS - 1), phis,
                        SOURCE_ISO_DIST, SOURCE_DETECTOR_DIST)
    leapct.set_flatDetector()
    if grid is not None:
        apply_volume(leapct, grid)


def set_modular_geometry(leapct, arrays, grid):
    """The modular-beam geometry built above, on the same volume."""
    sources, centers, rows, columns = arrays
    leapct.set_modularbeam(NUM_VIEWS, NUM_DET_ROWS, NUM_DET_CHANNELS, DELTA_DET_ROW,
                           DELTA_DET_CHANNEL, sources, centers, rows, columns)
    apply_volume(leapct, grid)


def project(leapct, volume, device):
    """One forward projection on the GPU, and the seconds it took.

    The volume goes to the GPU and the projections come back as a host array, the way
    bench_leap_vs_mbirtorch.py's run_leap does it (lines 441 to 452).
    """
    volume_gpu = torch.as_tensor(volume, device=device).contiguous()
    projections = torch.zeros((NUM_VIEWS, NUM_DET_ROWS, NUM_DET_CHANNELS), dtype=torch.float32,
                              device=device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    leapct.project(projections, volume_gpu)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    out = np.ascontiguousarray(projections.cpu().numpy(), dtype=np.float32)
    del volume_gpu, projections
    torch.cuda.empty_cache()
    return out, seconds


# ── the two libraries' sinogram conventions ───────────────────────────────────────────────────────
# The LEAP comparison measured the correspondence between the two libraries: LEAP's view angle is
# 180 degrees minus mbirtorch's, the detector channel axis runs the other way, and the detector rows
# do not change.  quality_leap_vs_mbirtorch.py states it as an index map on a sinogram taken at
# phis = 0, 360/N, ... (its swap_sinogram_convention, lines 71 to 86: LEAP's view j matches
# mbirtorch's view i when j = (N/2 - i) mod N, with the channels reversed).  real_scan_leap_tilt.py
# states the same thing the other way round, by handing LEAP the angles 180 - degrees(angles)
# directly (its leap_model, lines 60 to 78).  The second form is used here, because then LEAP's view
# i is mbirtorch's view i and the only thing left to do to a sinogram is reverse its channels.  The
# two are the same map: with mbirtorch's angles 360 i / N in degrees, 180 - 360 i / N is the angle
# LEAP's view (N/2 - i) mod N carries in the first form.

def leap_phis(angles):
    """LEAP's view angles in degrees, for mbirtorch's view angles in radians."""
    return np.ascontiguousarray(180.0 - np.degrees(angles), dtype=np.float32)


def leap_sinogram_to_mbirtorch(sinogram):
    """A LEAP sinogram in mbirtorch's convention: the channels reversed, the rows and views kept."""
    return np.ascontiguousarray(sinogram[:, :, ::-1], dtype=np.float32)


# ── mbirtorch ─────────────────────────────────────────────────────────────────────────────────────

def cone_model():
    """The estimating model: the same geometry as LEAP's, on the host with compilation off.

    Only the sinogram side of this model is used by the estimators, so its recon grid is left to
    mbirtorch.  The device is pinned before anything reads it, because a model left to choose for
    itself takes the GPU, and the GPU is LEAP's here.  This is the model of
    rotation_zero_point_synthetic.py's cone_model at an oversampling of one.
    """
    angles = np.linspace(0, 2 * np.pi, NUM_VIEWS, endpoint=False)
    shape = (NUM_VIEWS, NUM_DET_ROWS, NUM_DET_CHANNELS)
    model = mbirtorch.ConeBeamModel(shape, angles, source_detector_dist=SOURCE_DETECTOR_DIST,
                                    source_iso_dist=SOURCE_ISO_DIST, compile_mode='off')
    model.configure_devices(devices=['cpu'])
    # Changing the detector pitch changes the recon grid the model would use, so the recon geometry
    # is set again after it.
    model.set_params(delta_det_channel=DELTA_DET_CHANNEL, delta_det_row=DELTA_DET_ROW)
    model.auto_set_recon_geometry()
    model.set_params(no_warning=True, verbose=0, det_channel_offset=0.0)
    return model, angles


def normalized_rmse(test, reference):
    """||test - reference|| / ||reference||, both flattened, in float64."""
    test = np.asarray(test, dtype=np.float64).ravel()
    reference = np.asarray(reference, dtype=np.float64).ravel()
    denominator = np.linalg.norm(reference)
    if denominator == 0.0:
        return float('nan')
    return float(np.linalg.norm(test - reference) / denominator)


def relative_max_difference(test, reference):
    """The largest difference between two arrays, over the largest value of the reference."""
    scale = float(np.max(np.abs(np.asarray(reference, dtype=np.float64))))
    if scale == 0.0:
        return float('nan')
    return float(np.max(np.abs(np.asarray(test, dtype=np.float64) - reference))) / scale


def best_view_shift(test, reference):
    """The cyclic shift of the views that lines two sinograms up best, and its difference.

    A different choice of where the view angle zero sits is a rigid rotation of the whole system
    about z, which shows up as a shift of the view index.  The search is over that shift only, on
    per-view column profiles, which is the cheap form bench_leap_vs_mbirtorch.py uses in its
    best_view_alignment (lines 527 to 553).
    """
    profiles_test = test.sum(axis=1).astype(np.float64)
    profiles_reference = reference.sum(axis=1).astype(np.float64)
    costs = [float(np.linalg.norm(np.roll(profiles_test, shift, axis=0) - profiles_reference))
             for shift in range(test.shape[0])]
    shift = int(np.argmin(costs))
    return shift, relative_max_difference(np.roll(test, shift, axis=0), reference)


# ── the measurements ──────────────────────────────────────────────────────────────────────────────

def timed_estimate(function, *args, **kwargs):
    """Run one estimator, and return its result, its wall time, and the warnings it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        start = time.perf_counter()
        result = function(*args, **kwargs)
        seconds = time.perf_counter() - start
    return result, seconds, [str(item.message) for item in caught]


def run_case(model, sino, phantom_label, in_plane, out_of_plane, leap_seconds):
    """The channel offset and then the rotation at every band height, for one sinogram."""
    offset, offset_seconds, offset_warnings = timed_estimate(gc.estimate_det_channel_offset,
                                                             model, sino)
    offset_value = float(offset.value)
    offset_channels = offset_value / DELTA_DET_CHANNEL
    offset_window = tuple(int(value) for value in offset.reduction['row_window'])
    record('offset', offset_seconds, phantom=phantom_label, in_plane_degrees=in_plane,
           out_of_plane_degrees=out_of_plane, offset_alu=offset_value,
           offset_channels=offset_channels, row_window=list(offset_window),
           band_rows=offset_window[1] - offset_window[0], leap_seconds=leap_seconds,
           warnings=offset_warnings)
    print(f'  offset {offset_channels:+.4f} channels over rows {offset_window[0]}-'
          f'{offset_window[1]}, {offset_seconds:.1f} s, {len(offset_warnings)} warning(s)',
          flush=True)

    rotations = []
    for num_rows in BAND_ROWS:
        # A height of None leaves the argument out, so the module picks the band itself from the
        # cone geometry.  The offset just estimated is passed in, because the rotation comparison
        # shifts the opposite views by twice the offset.
        keywords = {} if num_rows is None else dict(num_rows=num_rows)
        result, seconds, messages = timed_estimate(gc.estimate_det_rotation, model, sino,
                                                   det_channel_offset=offset_value, **keywords)
        degrees = math.degrees(float(result.value))
        scores = [float(value) for value in result.scores]
        window = tuple(int(value) for value in result.reduction['row_window'])
        row = dict(phantom=phantom_label, in_plane_degrees=in_plane,
                   out_of_plane_degrees=out_of_plane, num_rows=num_rows,
                   row_window=list(window), band_rows=window[1] - window[0],
                   estimate_degrees=degrees, min_score=min(scores),
                   score_ratio=max(scores) / max(min(scores), 1e-30), seconds=seconds,
                   warnings=messages)
        # ``row`` already carries its own seconds, so it supplies record's second argument.
        record('rotation', **row)
        rotations.append(row)
        asked = 'default' if num_rows is None else str(num_rows)
        print(f'  rotation, band {asked} ({row["band_rows"]} rows, {window}): {degrees:+.4f} '
              f'degrees, {seconds:.1f} s, {len(messages)} warning(s)', flush=True)
    return dict(phantom=phantom_label, in_plane_degrees=in_plane, out_of_plane_degrees=out_of_plane,
                offset_channels=offset_channels, offset_seconds=offset_seconds,
                offset_warnings=offset_warnings, leap_seconds=leap_seconds, rotations=rotations)


# ── the report ────────────────────────────────────────────────────────────────────────────────────

def band_label(num_rows):
    return 'default' if num_rows is None else str(num_rows)


def print_summary(results):
    """One line per case: the two angles put in, the offset, and the rotation at every band."""
    header = (f'{"phantom":<8}{"in-plane":>10}{"out-plane":>11}{"offset":>10}'
              + ''.join(f'{band_label(rows):>10}' for rows in BAND_ROWS)
              + f'{"LEAP s":>9}')
    print()
    print('rotation estimates in degrees, one column per band height in detector rows')
    print(header)
    print('-' * len(header))
    for case in results:
        line = (f'{case["phantom"]:<8}{case["in_plane_degrees"]:>10.3f}'
                f'{case["out_of_plane_degrees"]:>11.3f}{case["offset_channels"]:>+10.4f}')
        for row in case['rotations']:
            line += f'{row["estimate_degrees"]:>+10.4f}'
        print(line + f'{case["leap_seconds"]:>9.2f}')


def print_detail(results):
    """One line per band height, with what the search saw and what it warned about."""
    header = (f'{"phantom":<8}{"in-plane":>10}{"out-plane":>11}{"band":>9}{"rows":>6}  '
              f'{"window":<12}{"estimate":>10}{"min score":>12}{"max/min":>9}{"seconds":>9}'
              f'{"warnings":>9}')
    print()
    print(header)
    print('-' * len(header))
    for case in results:
        for row in case['rotations']:
            window = f'{row["row_window"][0]}-{row["row_window"][1]}'
            print(f'{row["phantom"]:<8}{row["in_plane_degrees"]:>10.3f}'
                  f'{row["out_of_plane_degrees"]:>11.3f}{band_label(row["num_rows"]):>9}'
                  f'{row["band_rows"]:>6}  {window:<12}{row["estimate_degrees"]:>+10.4f}'
                  f'{row["min_score"]:>12.4e}{row["score_ratio"]:>9.3f}{row["seconds"]:>9.1f}'
                  f'{len(row["warnings"]):>9}')


def estimate_at(results, phantom, in_plane, out_of_plane, num_rows):
    """The rotation estimate of one case at one band height, or None when it is not there."""
    for case in results:
        if (case['phantom'] == phantom and case['in_plane_degrees'] == in_plane
                and case['out_of_plane_degrees'] == out_of_plane):
            for row in case['rotations']:
                if row['num_rows'] == num_rows:
                    return row['estimate_degrees']
    return None


def print_verdicts(results):
    """What the lean did to the estimate, phantom by phantom, in the two comparisons that matter."""
    print()
    print('what the out-of-plane lean did to the in-plane estimate')
    lean = max(pair[1] for pair in CASES)
    turn = max(pair[0] for pair in CASES)
    for phantom in PHANTOMS:
        label = phantom['label']
        for num_rows in (None, BAND_ROWS[-1]):
            flat_zero = estimate_at(results, label, 0.0, 0.0, num_rows)
            leaned_zero = estimate_at(results, label, 0.0, lean, num_rows)
            flat_turn = estimate_at(results, label, turn, 0.0, num_rows)
            leaned_turn = estimate_at(results, label, turn, lean, num_rows)
            if None in (flat_zero, leaned_zero, flat_turn, leaned_turn):
                print(f'{label} phantom, {band_label(num_rows)} band: a case is missing')
                continue
            print(f'{label} phantom, {band_label(num_rows)} band:')
            print(f'  with no in-plane angle, the lean of {lean} degrees moves the estimate from '
                  f'{flat_zero:+.4f} to {leaned_zero:+.4f} degrees, a change of '
                  f'{leaned_zero - flat_zero:+.4f}')
            print(f'  with an in-plane angle of {turn} degrees, it moves the estimate from '
                  f'{flat_turn:+.4f} to {leaned_turn:+.4f} degrees, a change of '
                  f'{leaned_turn - flat_turn:+.4f}, against a true angle of {turn}')


def print_warnings(results):
    """The distinct warnings the runs raised, with how many runs raised each one."""
    counts = {}
    for case in results:
        for message in case['offset_warnings']:
            counts[message[:100]] = counts.get(message[:100], 0) + 1
        for row in case['rotations']:
            for message in row['warnings']:
                counts[message[:100]] = counts.get(message[:100], 0) + 1
    print()
    if not counts:
        print('no warnings')
        return
    print('warnings, by the first 100 characters of each message:')
    for prefix, count in counts.items():
        print(f'  {count} run(s): {prefix}')


# ── the checks that come before the measurements ──────────────────────────────────────────────────

def compare_with_leaps_own_conversion(leapct, phis, grid):
    """Print LEAP's own cone-to-modular vectors beside the ones derived at the top of this file.

    leapctype has a convert_conebeam_to_modularbeam, so LEAP will state its own answer to the
    question the comment above NOMINAL_SOURCE works out by hand.  This only reports; nothing here
    uses LEAP's numbers, because a difference between the two is exactly what is worth seeing.  The
    cone-beam geometry is put back afterwards.
    """
    start = time.perf_counter()
    try:
        leapct.convert_conebeam_to_modularbeam()
        sources = np.asarray(leapct.get_sourcePositions(), dtype=np.float64)
        centers = np.asarray(leapct.get_moduleCenters(), dtype=np.float64)
        rows = np.asarray(leapct.get_rowVectors(), dtype=np.float64)
        columns = np.asarray(leapct.get_colVectors(), dtype=np.float64)
    except Exception as error:                          # the version here may not have all of these
        record('leap_own_conversion', time.perf_counter() - start, available=False,
               error=repr(error))
        set_cone_geometry(leapct, phis, grid)
        return
    names = ('source', 'center', 'row', 'column')
    mine = [np.asarray(array, dtype=np.float64) for array in
            modular_arrays(phis, np.array([0.0, 0.0, 1.0]))]
    theirs = [np.asarray(array, dtype=np.float64)
              for array in (sources, centers, rows, columns)]
    differences = {name: float(np.max(np.abs(ours - leaps)))
                   for name, ours, leaps in zip(names, mine, theirs)}
    record('leap_own_conversion', time.perf_counter() - start, available=True,
           largest_difference=differences,
           leap_first_view=dict(zip(names, [array[0].tolist() for array in theirs])),
           derived_first_view=dict(zip(names, [array[0].tolist() for array in mine])))
    print()
    print('the derived modular vectors against the ones LEAP converts to, first view')
    for name, ours, leaps in zip(names, mine, theirs):
        print(f'  {name:<7} derived {np.round(ours[0], 4)}   LEAP {np.round(leaps[0], 4)}   '
              f'largest difference over all views {differences[name]:.4e}')
    set_cone_geometry(leapct, phis, grid)


def check_modular_matches_cone(leapct, phis, grid, volume, label, device, search_shift):
    """Project one volume with both geometries and say how far apart they are.

    The no-tilt modular geometry is meant to be LEAP's cone-beam geometry written a different way,
    so the two projections should agree.  The two geometry types use different kernels, so a small
    difference is expected.  When ``search_shift`` is set, the best cyclic shift of the views is
    reported as well, because a different choice of where the view angle zero sits shows up as
    exactly such a shift and changes nothing else.
    """
    start = time.perf_counter()
    set_cone_geometry(leapct, phis, grid)
    cone, cone_seconds = project(leapct, volume, device)
    set_modular_geometry(leapct, modular_arrays(phis, np.array([0.0, 0.0, 1.0])), grid)
    modular, modular_seconds = project(leapct, volume, device)
    at_rest = relative_max_difference(modular, cone)
    fields = dict(volume=label, relative_max_difference=at_rest,
                  nrmse=normalized_rmse(modular, cone), cone_max=float(np.max(cone)),
                  modular_max=float(np.max(modular)), cone_seconds=cone_seconds,
                  modular_seconds=modular_seconds, tolerance=GEOMETRY_CHECK_TOLERANCE)
    if search_shift:
        shift, shifted = best_view_shift(modular, cone)
        fields.update(best_view_shift=shift, relative_max_difference_at_best_shift=shifted)
    passed = min(at_rest, fields.get('relative_max_difference_at_best_shift', at_rest))
    fields['passed'] = bool(passed <= GEOMETRY_CHECK_TOLERANCE)
    record('geometry_check', time.perf_counter() - start, **fields)
    print(f'geometry check on the {label} volume: largest difference {at_rest:.3e} of the '
          f'cone-beam maximum', flush=True)
    if search_shift:
        print(f'  at the best view shift of {fields["best_view_shift"]} views: '
              f'{fields["relative_max_difference_at_best_shift"]:.3e}', flush=True)
    if not fields['passed']:
        print(f'  WARNING: above the tolerance of {GEOMETRY_CHECK_TOLERANCE:.0e}; the modular '
              f'geometry is not the cone-beam geometry and nothing below it means anything',
              flush=True)
    del cone, modular
    return fields


def check_conversion_against_mbirtorch(grid, volume, sino_mbirtorch):
    """Project the same volume with mbirtorch and compare, to fix the direction of the rows.

    The cylinder with its slab is lopsided along z, so this says whether the detector rows of the two
    libraries run the same way, which the channel reversal alone does not settle.  mbirtorch's recon
    grid is set to LEAP's, in mbirtorch's own order (y, x, z), which is the transpose
    bench_leap_vs_mbirtorch.py uses in leap_volume_to_mbirtorch (lines 103 to 105).
    """
    start = time.perf_counter()
    if abs(grid['voxel_width'] - grid['voxel_height']) > 1e-9:
        record('mbirtorch_conversion_check', time.perf_counter() - start, ran=False,
               reason='LEAP chose a voxel height that differs from its voxel width, and this model '
                      'has one voxel pitch', voxel_width=grid['voxel_width'],
               voxel_height=grid['voxel_height'])
        return
    model, _angles = cone_model()
    model.set_params(delta_voxel=grid['voxel_width'],
                     recon_shape=(grid['num_y'], grid['num_x'], grid['num_z']))
    in_mbirtorch_order = np.ascontiguousarray(np.transpose(volume, (1, 2, 0)))
    projected = np.asarray(model.forward_project(in_mbirtorch_order), dtype=np.float32)
    options = {'rows_as_is': sino_mbirtorch, 'rows_flipped': sino_mbirtorch[:, ::-1, :]}
    errors = {name: normalized_rmse(candidate, projected) for name, candidate in options.items()}
    chosen = min(errors, key=errors.get)
    record('mbirtorch_conversion_check', time.perf_counter() - start, ran=True,
           orientation=chosen, nrmse=errors, expected='rows_as_is',
           agrees_with_expected=bool(chosen == 'rows_as_is'),
           mbirtorch_mean=float(projected.mean()), leap_mean=float(sino_mbirtorch.mean()))
    print(f'conversion check against mbirtorch: {errors} (the LEAP comparison expects rows_as_is)',
          flush=True)
    del model, in_mbirtorch_order, projected


# ── the run ───────────────────────────────────────────────────────────────────────────────────────

def main():
    run_start = time.perf_counter()
    os.makedirs(RESULTS, exist_ok=True)
    torch.set_num_threads(NUM_THREADS)
    assert torch.cuda.is_available(), 'this job needs a GPU for LEAP'
    device = torch.device('cuda:0')

    from leapctype import tomographicModels
    leapct = tomographicModels()
    leapct.set_gpu(0)

    model, angles = cone_model()
    phis = leap_phis(angles)

    record('environment', 0.0, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
           mbirtorch=mbirtorch.__version__, mbirtorch_file=mbirtorch.__file__,
           leap=str(leapct.version()), threads=NUM_THREADS, results=RESULTS, jsonl=JSONL,
           num_views=NUM_VIEWS, num_det_rows=NUM_DET_ROWS, num_det_channels=NUM_DET_CHANNELS,
           source_iso_dist=SOURCE_ISO_DIST, source_detector_dist=SOURCE_DETECTOR_DIST,
           band_rows=[band_label(rows) for rows in BAND_ROWS],
           cases=[list(pair) for pair in CASES], argv=sys.argv)
    print(f'mbirtorch {mbirtorch.__version__} at {mbirtorch.__file__}, torch {torch.__version__}, '
          f'LEAP {leapct.version()}, {torch.cuda.get_device_name(0)}', flush=True)

    # LEAP picks the volume for this geometry and the phantoms are built on it.
    set_cone_geometry(leapct, phis)
    leapct.set_default_volume()
    grid = read_volume_grid(leapct)
    apply_volume(leapct, grid)
    leapct.print_parameters()
    record('volume', 0.0, **grid, mask_diameter=NO_MASK_DIAMETER)
    print(f'LEAP volume {grid["shape"]} in {grid["order_text"]} order, voxel '
          f'{grid["voxel_width"]} by {grid["voxel_height"]} mm', flush=True)

    volumes = {}
    cylinder_radius = None
    for phantom in PHANTOMS:
        volume, slab, radius = cylinder_with_slab(grid, phantom['slab_fraction'])
        volumes[phantom['label']] = volume
        cylinder_radius = radius
        record('phantom', 0.0, phantom=phantom['label'], slab_fraction=phantom['slab_fraction'],
               slab_slices=list(slab), radius_voxels=radius,
               radius_mm=radius * grid['voxel_width'],
               radius_detector_pixels=(radius * grid['voxel_width']
                                       * SOURCE_DETECTOR_DIST / SOURCE_ISO_DIST
                                       / DELTA_DET_CHANNEL))
        print(f'{phantom["label"]} phantom: cylinder radius {radius:.1f} voxels '
              f'({radius * grid["voxel_width"]:.1f} mm), slab slices {slab[0]} to {slab[1]}',
              flush=True)

    # The checks that say whether the geometry below is the geometry it is meant to be.
    compare_with_leaps_own_conversion(leapct, phis, grid)
    check_modular_matches_cone(leapct, phis, grid, volumes['far'], 'far phantom', device,
                               search_shift=False)
    probe, block = with_probe_block(volumes['far'], cylinder_radius)
    record('probe_volume', 0.0, block_center_zyx=list(block[:3]), block_half_width=block[3],
           value=PROBE_VALUE)
    check_modular_matches_cone(leapct, phis, grid, probe, 'lopsided probe', device,
                               search_shift=True)
    del probe

    # The cases.  A case that fails is recorded and skipped, so the tables still hold the ones that
    # finished; a case that is missing shows up as such in the comparisons at the end.
    results = []
    for phantom in PHANTOMS:
        for in_plane, out_of_plane in CASES:
            print(f'\n{phantom["label"]} phantom, in-plane {in_plane} degrees, out-of-plane '
                  f'{out_of_plane} degrees', flush=True)
            case_start = time.perf_counter()
            try:
                axis = tilt_axis(in_plane, out_of_plane)
                arrays = modular_arrays(phis, axis)
                set_modular_geometry(leapct, arrays, grid)
                sino_leap, leap_seconds = project(leapct, volumes[phantom['label']], device)
                record('projection', leap_seconds, phantom=phantom['label'],
                       in_plane_degrees=in_plane, out_of_plane_degrees=out_of_plane,
                       axis=axis.tolist(), first_view_source=arrays[0][0].tolist(),
                       first_view_center=arrays[1][0].tolist(),
                       first_view_row=arrays[2][0].tolist(),
                       first_view_col=arrays[3][0].tolist(), sinogram_max=float(np.max(sino_leap)),
                       sinogram_mean=float(np.mean(sino_leap)))
                sino = leap_sinogram_to_mbirtorch(sino_leap)
                del sino_leap
                if (CHECK_AGAINST_MBIRTORCH and phantom['label'] == 'far' and in_plane == 0.0
                        and out_of_plane == 0.0):
                    check_conversion_against_mbirtorch(grid, volumes['far'], sino)
                results.append(run_case(model, sino, phantom['label'], in_plane, out_of_plane,
                                        leap_seconds))
                del sino
            except Exception:
                record('case_error', time.perf_counter() - case_start, phantom=phantom['label'],
                       in_plane_degrees=in_plane, out_of_plane_degrees=out_of_plane,
                       traceback=traceback.format_exc())

    print_summary(results)
    print_detail(results)
    print_verdicts(results)
    print_warnings(results)

    # Linux reports ru_maxrss in kilobytes, and it is the largest the whole process has been.
    record('resources', time.perf_counter() - run_start,
           max_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2,
           cases=len(results))
    print('\nLEAP_AXIS_TILT DONE', flush=True)


if __name__ == '__main__':
    main()
