"""A fine sweep of the detector rotation on the two NSI scans, scored on four slices.

An earlier job reconstructed four slices far from the central plane of each scan at four fixed
detector rotations.  Of those four candidates the vendor's 0.167 degrees gave the sharpest slices.
That job could not say where the best rotation lies between its candidates, because it only had
four of them.  A later measurement, which compares conjugate views over a tall band of detector
rows, returned 0.130 degrees.  This job asks where the reconstructions themselves put the best
rotation, on a grid fine enough to answer.

The job reruns the far-slice reconstruction on a grid of candidate rotations spaced 0.005 degrees
apart.  ``parameter_sweep`` does the work: for one slice of the volume it crops the detector to the
rows that slice needs, rotates that band of every view by the candidate angle, and reconstructs the
slice directly.  Each reconstructed slice is then scored.  The score is the gradient energy of the
slice after a Gaussian blur, divided by the blurred slice's mean square, and negated so that lower
is better.  The blur is what makes the score read the geometry: without it the score is dominated
by the bilinear resampling that applies the candidate rotation, which smooths pixel-scale noise by
an amount that depends on the fractional part of the displacement.  The unblurred score is computed
anyway, as a diagnostic.

The four slices are combined at each candidate by dropping the highest and the lowest slice score
and averaging the rest.  A trimmed mean is used because the four slices differ in how much
structure they hold, and one slice that carries no structure would otherwise set the shape of the
combined curve.  The minimum of the combined curve is then located by fitting a quadratic through a
few candidates around the lowest one.  The fit also gives a half width, which says how sharply the
curve rises away from its minimum, and therefore how precisely the minimum is placed.

The job measures how repeatable the curve is by splitting the views.  It scores the even-numbered
views and the odd-numbered views separately and compares the two curves.  The largest difference
between them over the candidates is a floor: a feature of the combined curve smaller than that
floor is not a feature of the geometry.  The split costs nothing extra.  The direct reconstruction
filter is scaled by pi divided by the model's view count, so a reconstruction from half the views
is on the same scale as one from all of them, and by linearity the all-view slice is the mean of
the even-view and odd-view slices.  The two half sweeps together therefore cost what one all-view
sweep would cost, and their mean is the all-view result.  The job checks that claim directly: at
one candidate it also reconstructs the slice from all views and compares.

Results are appended to a JSON-lines file as each slice is done, so a job cut short leaves what it
finished.  Every reconstructed stack is saved as an ``.npz`` file, and three figures per scan are
drawn.  Run parameters are at the top of the file.

Cost.  The earlier job took two to five seconds per candidate on the far slices of these scans,
and this job has 22 candidates on the scan without metal and 30 on the scan with metal.  Four
slices per scan then come to roughly five to ten minutes of reconstruction, plus a few minutes of
scoring per scan, half a minute to load each scan, and the compilation of each reduced model.
"""
import math
import os
import resource
import sys
import time
import traceback
from gc import collect as collect_garbage    # 'gc' below is the calibration module

os.environ.setdefault('MBIRTORCH_NUM_DEVICES', '1')

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

import mbirtorch
from mbirtorch.preprocess import geometry_calibration as gc
from mbirtorch.utilities import copy_ct_model

# The loader modules of the earlier jobs sit in the 'closed' subdirectory of this directory, and on
# the cluster every module sits flat in one directory that is on PYTHONPATH.  Adding the
# subdirectory here covers the local layout and does nothing on the cluster.
_CLOSED = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'closed')
if os.path.isdir(_CLOSED) and _CLOSED not in sys.path:
    sys.path.insert(0, _CLOSED)

# Importing the first job fixes where 'record' writes, and importing the follow-up job points that
# name at the follow-up's file, so the line below that points it at this job's file comes after
# both imports.
import real_scan_validation as first_job
import real_scan_followup as second_job
from real_scan_validation import git_commit, record
from real_scan_followup import load_scan

torch.set_num_threads(14)       # the CPUs the batch file asks for per GPU

# ── run parameters ────────────────────────────────────────────────────────────────────────────────
# The datasets, in the order they run.  Each path is an NSI tarball that the first job's
# 'extract_tarball' unpacks into DATA, and a directory already there is left alone.
DATASETS = (
    dict(name='nsi_no_metal', reader='nsi',
         path='/depot/bouman/data/Lilly/demo_nsi_vert_no_metal_all_views.tgz'),
    dict(name='nsi_metal', reader='nsi',
         path='/depot/bouman/data/Lilly/demo_nsi_vert_metal_all_views.tgz'),
)
# The candidate rotations searched on each scan, in degrees, as (lowest, highest).  Both ranges
# bracket the two values under test, 0.130 and the vendor's 0.167 degrees.  The range on the scan
# with metal reaches higher, because an estimate on that scan reached 0.19 degrees.
CANDIDATE_RANGES_DEGREES = {'nsi_no_metal': (0.10, 0.20), 'nsi_metal': (0.10, 0.24)}
# The spacing of the candidate grid, in degrees.  At 750 detector rows from the central plane a
# step of 0.005 degrees moves the center of rotation by about 0.065 channels.
CANDIDATE_STEP_DEGREES = 0.005
# Candidates that get their own name in the records and their own line in the figures.  The scan's
# vendor rotation is added to this list under the name IDENTITY_CANDIDATE.
NAMED_CANDIDATES_DEGREES = (0.130,)
# The slices reconstructed, as fractions of the way through the volume's slices.  These are the
# four slices of the earlier job that lie away from the central plane, where a detector rotation
# moves something.  The earlier job found the lowest of the four to lie below the object on these
# scans, so its curve may be flat; the trimmed mean and the normalized curves both guard against
# one flat slice setting the answer.
SLICE_FRACTIONS = (0.1, 0.25, 0.75, 0.9)
# The Gaussian blur widths the score is computed at, as multiples of the reconstruction voxel
# pitch.  The 0.0 entry is unblurred.  It is a diagnostic only, because the earlier job showed an
# unblurred gradient score reads the resampling that applies the rotation rather than the geometry.
BLUR_VOXELS = (0.0, 1.0, 2.0, 3.0, 4.0)
# Candidates in the window the quadratic is fitted through, centered on the lowest candidate.
PARABOLA_POINTS = 5
# The half width is the distance from the fitted minimum at which the fitted parabola has risen by
# this fraction of the magnitude of the fitted minimum value.
HALF_WIDTH_RISE = 0.02
# The candidate at which the all-view reconstruction is also computed, so that the mean of the
# even-view and odd-view reconstructions can be compared with it.  This name is also the name the
# scan's vendor rotation is recorded under.
IDENTITY_CANDIDATE = 'vendor'
# The two scans the closing comparison reads, and the range the scan with metal is asked to place
# its minimum inside.
GATE_SCAN_NAMES = ('nsi_no_metal', 'nsi_metal')
METAL_BRACKET_DEGREES = (0.15, 0.24)

RESULTS = first_job.RESULTS
DATA = first_job.DATA
JSONL = os.path.join(RESULTS, 'recon_sweep_fine.jsonl')
# 'record' reads the first job's module-level JSONL name each time it is called, so pointing that
# name at this job's file sends every entry of this job there.
first_job.JSONL = JSONL

PARITIES = ('even', 'odd', 'mean')
CURVE_KINDS = ('raw', 'normalized')


# ── scoring and curve analysis ────────────────────────────────────────────────────────────────────
# Everything in this section is plain numpy and scipy.  It takes arrays and returns numbers, so a
# later script can import it and rerun the analysis on the saved stacks without a GPU.

def blur_key(blur):
    """The dictionary key one blur width is recorded under."""
    return str(float(blur))


def score_slice(image, sigma_pixels):
    """The negative gradient energy of a blurred slice, divided by the blurred slice's mean square.

    The gradient energy is the mean squared finite difference along each axis, summed over the two
    axes.  A reconstruction made with the right geometry has sharper edges than one made with the
    wrong geometry, so the gradient energy is largest at the best candidate.  The sign is flipped
    so that the best candidate has the lowest score, which is the convention the calibration module
    uses.  Dividing by the mean square removes the overall brightness of the slice.

    Args:
        image: the slice, as a two-dimensional array.
        sigma_pixels (float): the standard deviation of the Gaussian blur, in reconstruction
            pixels.  Zero means no blur.

    Returns:
        float: the score, or NaN when the blurred slice has zero mean square.
    """
    image = np.asarray(image, dtype=np.float64)
    smooth = gaussian_filter(image, float(sigma_pixels)) if sigma_pixels > 0 else image
    energy = float(np.mean(smooth ** 2))
    if not energy > 0.0:
        return float('nan')
    gradient = float(np.mean(np.diff(smooth, axis=0) ** 2) + np.mean(np.diff(smooth, axis=1) ** 2))
    return float(-gradient / energy)


def score_stack(stack, sigmas_pixels):
    """Score every candidate slice of one stack at each blur width.

    Args:
        stack: the sweep output, shape ``(recon_rows, recon_cols, num_candidates)``.
        sigmas_pixels (dict): blur key to blur standard deviation in reconstruction pixels.

    Returns:
        dict: blur key to the list of scores, one per candidate.
    """
    scores = {key: [] for key in sigmas_pixels}
    for k in range(stack.shape[2]):
        # The conversion to float64 is done once per candidate and reused by every blur width.
        image = np.asarray(stack[:, :, k], dtype=np.float64)
        for key, sigma in sigmas_pixels.items():
            scores[key].append(score_slice(image, sigma))
    return scores


def trimmed_curve(matrix):
    """Combine the slices at each candidate by dropping the highest and lowest slice score.

    Args:
        matrix: scores, shape ``(num_slices, num_candidates)``.

    Returns:
        tuple: the combined curve over candidates, and a list that gives, for each candidate, the
        two slice indices that were dropped there.  A candidate with fewer than three finite slice
        scores gets NaN and an empty list.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    num_candidates = matrix.shape[1]
    curve = np.full(num_candidates, np.nan)
    dropped = []
    for c in range(num_candidates):
        column = matrix[:, c]
        finite = np.nonzero(np.isfinite(column))[0]
        if finite.size < 3:
            dropped.append([])
            continue
        order = finite[np.argsort(column[finite])]
        dropped.append([int(order[0]), int(order[-1])])
        curve[c] = float(np.mean(column[order[1:-1]]))
    return curve, dropped


def normalize_rows(matrix):
    """Divide each slice's scores by the magnitude of that slice's mean over the candidates.

    The slices differ in brightness and in how much structure they hold, so their raw scores differ
    in size as well as in shape.  Dividing each slice by its own mean makes the four slices weigh
    alike in the combined curve.  Every score is negative, so the mean of a row is negative and its
    magnitude is a positive scale factor, which leaves the position of that row's minimum where it
    was.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    with np.errstate(invalid='ignore'):
        scale = np.abs(np.nanmean(matrix, axis=1, keepdims=True))
        scale = np.where(scale > 0.0, scale, np.nan)
        return matrix / scale


def parabola_fit(candidates, curve, center_index, num_points=None, rise_fraction=None):
    """Fit a quadratic through a window of the curve and locate its minimum.

    The window holds ``num_points`` candidates centered on ``center_index``, moved inward when that
    would run off either end of the grid.  The half width is the distance from the fitted minimum
    at which the parabola has risen by ``rise_fraction`` of the magnitude of the fitted minimum
    value.  It is NaN when the parabola opens downward, because then there is no minimum to
    measure a width around.

    Args:
        candidates: the candidate rotations in degrees, shape ``(num_candidates,)``.
        curve: the score at each candidate, same shape.
        center_index (int): the candidate the window is centered on.
        num_points (int, optional): candidates in the window.  Defaults to ``PARABOLA_POINTS``.
        rise_fraction (float, optional): the rise the half width is measured at.  Defaults to
            ``HALF_WIDTH_RISE``.

    Returns:
        dict: the window, the fitted coefficients, the location and value of the fitted minimum,
        whether the parabola opens upward, and the half width in degrees.
    """
    num_points = PARABOLA_POINTS if num_points is None else int(num_points)
    rise_fraction = HALF_WIDTH_RISE if rise_fraction is None else float(rise_fraction)
    candidates = np.asarray(candidates, dtype=np.float64)
    curve = np.asarray(curve, dtype=np.float64)
    nan = float('nan')
    empty = dict(window=[], coefficients=[], location_degrees=nan, value=nan, opens_upward=False,
                 half_width_degrees=nan)
    width = int(min(num_points, curve.size))
    if width < 3:
        return empty
    lo = int(min(max(int(center_index) - width // 2, 0), curve.size - width))
    x, y = candidates[lo:lo + width], curve[lo:lo + width]
    good = np.isfinite(y)
    if int(np.count_nonzero(good)) < 3:
        return empty
    a, b, c = (float(v) for v in np.polyfit(x[good], y[good], 2))
    result = dict(window=[lo, lo + width], coefficients=[a, b, c], location_degrees=nan, value=nan,
                  opens_upward=bool(a > 0.0), half_width_degrees=nan)
    if a == 0.0:
        return result
    location = -b / (2.0 * a)
    value = a * location ** 2 + b * location + c
    result['location_degrees'] = float(location)
    result['value'] = float(value)
    if a > 0.0:
        result['half_width_degrees'] = float(math.sqrt(rise_fraction * abs(value) / a))
    return result


def curve_summary(candidates, curve, num_points=None, rise_fraction=None):
    """Describe one score curve: where it is lowest, how deep it is, and how sharp its minimum is.

    ``interior_minima`` counts the candidates that are strictly lower than both of their
    neighbors.  One such candidate means the curve has a single dip, and more than one means the
    curve is not a simple dip over this range.  ``argmin_at_end`` says whether the lowest candidate
    is the first or the last of the range, which means the range did not bracket the minimum.
    """
    candidates = np.asarray(candidates, dtype=np.float64)
    curve = np.asarray(curve, dtype=np.float64)
    nan = float('nan')
    if not np.any(np.isfinite(curve)):
        return dict(values=curve.tolist(), argmin_index=-1, argmin_degrees=nan, argmin_at_end=False,
                    depth=nan, interior_minima=0, parabola=parabola_fit(candidates, curve, 0,
                                                                        num_points, rise_fraction))
    argmin = int(np.nanargmin(curve))
    interior = int(np.count_nonzero((curve[1:-1] < curve[:-2]) & (curve[1:-1] < curve[2:])))
    return dict(values=curve.tolist(), argmin_index=argmin,
                argmin_degrees=float(candidates[argmin]),
                argmin_at_end=bool(argmin in (0, curve.size - 1)),
                depth=float(np.nanmax(curve) - np.nanmin(curve)), interior_minima=interior,
                parabola=parabola_fit(candidates, curve, argmin, num_points, rise_fraction))


def slice_locations(candidates, matrix):
    """Where each slice's own curve is lowest, by the lowest candidate and by the quadratic fit.

    The spread of these numbers over the slices says how much the four slices disagree.  Both
    lists are the same for the raw and the normalized scores, because normalizing divides a slice's
    curve by a positive number.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    summaries = [curve_summary(candidates, matrix[i]) for i in range(matrix.shape[0])]
    return dict(argmin_degrees=[s['argmin_degrees'] for s in summaries],
                parabola_degrees=[s['parabola']['location_degrees'] for s in summaries])


def analyze_scan(candidates_degrees, matrices, named_indices):
    """Build and describe the combined score curves of one scan.

    The primary curve is the trimmed mean of the raw slice scores.  The normalized curve is the
    same trimmed mean taken after each slice is divided by the magnitude of its own mean, and it is
    a companion readout rather than the definition.  Both are built for the even views, the odd
    views, and their mean.  The repeatability floor of a blur width is the largest difference
    between the even-view and odd-view curves over the candidates, and the mean curve is only
    believable where it is deeper than that floor.

    Args:
        candidates_degrees: the candidate rotations in degrees.
        matrices (dict): parity to blur key to a ``(num_slices, num_candidates)`` score array.
        named_indices (dict): candidate name to its index in ``candidates_degrees``.

    Returns:
        dict: ``curves[kind][blur key][parity]`` is a curve summary and ``floors[kind][blur key]``
        is the repeatability floor.
    """
    candidates = np.asarray(candidates_degrees, dtype=np.float64)
    curves = {kind: {} for kind in CURVE_KINDS}
    floors = {kind: {} for kind in CURVE_KINDS}
    for key in matrices['mean']:
        built = {kind: {} for kind in CURVE_KINDS}
        for parity in PARITIES:
            matrix = np.asarray(matrices[parity][key], dtype=np.float64)
            locations = slice_locations(candidates, matrix)
            for kind in CURVE_KINDS:
                rows = matrix if kind == 'raw' else normalize_rows(matrix)
                curve, dropped = trimmed_curve(rows)
                summary = curve_summary(candidates, curve)
                summary['dropped_slices'] = dropped
                summary['slice_locations'] = locations
                summary['score_at_named'] = {name: float(curve[index])
                                             for name, index in named_indices.items()}
                built[kind][parity] = summary
        for kind in CURVE_KINDS:
            even = np.asarray(built[kind]['even']['values'], dtype=np.float64)
            odd = np.asarray(built[kind]['odd']['values'], dtype=np.float64)
            with np.errstate(invalid='ignore'):
                floor = float(np.nanmax(np.abs(even - odd)))
            mean_summary = built[kind]['mean']
            mean_summary['floor'] = floor
            mean_summary['depth_over_floor'] = (float(mean_summary['depth'] / floor)
                                                if floor > 0.0 else float('inf'))
            mean_summary['depth_exceeds_floor'] = bool(mean_summary['depth'] > floor)
            floors[kind][key] = floor
            curves[kind][key] = built[kind]
    return dict(candidates_degrees=candidates.tolist(), named_indices=dict(named_indices),
                curves=curves, floors=floors)


# ── figures ───────────────────────────────────────────────────────────────────────────────────────

def save_trimmed_curves_figure(name, analysis, named_degrees):
    """Draw the even-view, odd-view, and mean raw trimmed curves, with one panel per blur width."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    candidates = np.asarray(analysis['candidates_degrees'], dtype=np.float64)
    keys = list(analysis['curves']['raw'])
    figure, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 4.2), squeeze=False)
    for axis, key in zip(axes[0], keys):
        panel = analysis['curves']['raw'][key]
        for parity, style in (('even', ':'), ('odd', '--'), ('mean', '-')):
            axis.plot(candidates, panel[parity]['values'], style, label=parity)
        for label, degrees in named_degrees.items():
            axis.axvline(degrees, color='0.6', linestyle='--', linewidth=0.8)
            axis.annotate(label, (degrees, 0.02), xycoords=('data', 'axes fraction'), fontsize=7,
                          rotation=90, ha='right', va='bottom')
        # The marker is drawn only for a fit that has a minimum inside the range searched.  A fit
        # whose parabola opens downward has no minimum, and one that lands outside the range would
        # stretch the axis away from the candidates.
        fitted = panel['mean']['parabola']
        location, value = fitted['location_degrees'], fitted['value']
        if (fitted['opens_upward'] and np.isfinite(location) and np.isfinite(value)
                and candidates[0] <= location <= candidates[-1]):
            axis.plot([location], [value], 'o', color='k', markersize=5, label='fitted minimum')
        axis.set_xlim(candidates[0], candidates[-1])
        axis.set_title(f'blur {key} voxels\nfloor {panel["mean"]["floor"]:.3g}, '
                       f'depth {panel["mean"]["depth"]:.3g}', fontsize=9)
        axis.set_xlabel('det_rotation, degrees')
        axis.legend(fontsize=7)
    axes[0][0].set_ylabel('trimmed score, lower is better')
    figure.suptitle(f'{name}: trimmed score over four slices, by view parity')
    figure.tight_layout()
    figure.savefig(os.path.join(RESULTS, f'{name}_trimmed_curves.png'), dpi=110)
    plt.close(figure)


def save_slice_curves_figure(name, analysis, matrices, rows_from_central_plane):
    """Draw each slice's own normalized curve, with one panel per blur width."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    candidates = np.asarray(analysis['candidates_degrees'], dtype=np.float64)
    keys = list(analysis['curves']['raw'])
    figure, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 4.2), squeeze=False)
    for axis, key in zip(axes[0], keys):
        rows = normalize_rows(np.asarray(matrices['mean'][key], dtype=np.float64))
        for i in range(rows.shape[0]):
            axis.plot(candidates, rows[i], label=f'{rows_from_central_plane[i]:+.0f} rows')
        axis.set_title(f'blur {key} voxels', fontsize=9)
        axis.set_xlabel('det_rotation, degrees')
        axis.legend(fontsize=7)
    axes[0][0].set_ylabel('normalized score, lower is better')
    figure.suptitle(f'{name}: each slice normalized by its own mean over the candidates')
    figure.tight_layout()
    figure.savefig(os.path.join(RESULTS, f'{name}_slice_curves.png'), dpi=110)
    plt.close(figure)


def save_named_slice_figure(name, slice_index, z_alu, images, named_degrees):
    """Draw one slice at each named candidate, and the first named slice minus the last.

    Every slice panel uses one gray scale, taken from the first image, so that a difference in
    brightness between the panels is visible.  The difference panel gets a symmetric scale of its
    own.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = list(images)
    low, high = np.nanpercentile(images[labels[0]], [1.0, 99.5])
    figure, axes = plt.subplots(1, len(labels) + 1, figsize=(5.0 * (len(labels) + 1), 5.4))
    for axis, label in zip(axes[:-1], labels):
        axis.imshow(images[label], cmap='gray', vmin=low, vmax=high)
        axis.set_title(f'{label}: {named_degrees[label]:.4f} deg', fontsize=9)
        axis.set_xticks([]); axis.set_yticks([])
    difference = images[labels[0]].astype(np.float64) - images[labels[-1]].astype(np.float64)
    scale = float(np.nanpercentile(np.abs(difference), 99.5))
    axes[-1].imshow(difference, cmap='gray', vmin=-scale, vmax=scale)
    axes[-1].set_title(f'{labels[0]} minus {labels[-1]}', fontsize=9)
    axes[-1].set_xticks([]); axes[-1].set_yticks([])
    figure.suptitle(f'{name}: slice {slice_index}, z = {z_alu:+.1f} mm, mean of the even-view and '
                    'odd-view reconstructions')
    figure.tight_layout()
    figure.savefig(os.path.join(RESULTS, f'{name}_slice_{slice_index:04d}_named.png'), dpi=110)
    plt.close(figure)


# ── the candidate grid ────────────────────────────────────────────────────────────────────────────

def named_candidate_degrees(scan):
    """The candidates that get their own name, as a dict from name to degrees."""
    named = {f'{float(value):.3f}': float(value) for value in NAMED_CANDIDATES_DEGREES}
    named[IDENTITY_CANDIDATE] = math.degrees(float(scan.vendor_det_rotation))
    return named


def build_candidates(name, named_degrees):
    """The candidate grid for one scan, with the named candidates merged into it.

    The grid runs from the low end of the scan's range to the high end in steps of
    ``CANDIDATE_STEP_DEGREES``.  Each named candidate is added to it, the result is sorted, and two
    candidates within a billionth of a degree of each other are treated as one.

    Returns:
        tuple: the candidates in degrees, and a dict from candidate name to its index.
    """
    low, high = CANDIDATE_RANGES_DEGREES[name]
    grid = np.arange(low, high + CANDIDATE_STEP_DEGREES / 2.0, CANDIDATE_STEP_DEGREES)
    values = np.sort(np.concatenate([grid, np.asarray(list(named_degrees.values()),
                                                      dtype=np.float64)]))
    kept = [float(values[0])]
    for value in values[1:]:
        if float(value) - kept[-1] > 1e-9:
            kept.append(float(value))
    candidates = np.asarray(kept, dtype=np.float64)
    indices = {}
    for label, degrees in named_degrees.items():
        index = int(np.argmin(np.abs(candidates - degrees)))
        if abs(candidates[index] - degrees) > 1e-9:
            raise RuntimeError(f'the named candidate {label} at {degrees} degrees is not on the grid')
        indices[label] = index
    return candidates, indices


# ── the half-view models ──────────────────────────────────────────────────────────────────────────

def build_half_models(scan, slice_indices):
    """Build the even-view and odd-view models and check they reconstruct the same geometry.

    Each half model keeps every other view of the full model.  The reconstruction geometry must be
    the one the full model uses, because ``parameter_sweep`` selects the slice to reconstruct by
    the full model's slice index and by the height in ALU that index maps to.  A half model whose
    ``recon_shape``, ``recon_slice_offset``, or ``delta_voxel`` differed from the full model's
    would reconstruct a different slice under the same index.  The three parameters are copied over
    when they differ, and the height of every requested slice is then checked again.

    Args:
        scan: the loaded scan.
        slice_indices (sequence of int): the full model's slices this job reconstructs.

    Returns:
        tuple: a dict from parity name to (model, sinogram view), and a dict of the checks.
    """
    full = scan.ct_model
    angles = gc._view_angles(full)
    reference = dict(recon_shape=tuple(int(s) for s in full.get_params('recon_shape')),
                     recon_slice_offset=float(full.get_params('recon_slice_offset')),
                     delta_voxel=float(full.get_params('delta_voxel')))
    reference_z = [float(full.recon_slice_z(int(k))) for k in slice_indices]
    halves, checks = {}, dict(full=reference, full_slice_z=reference_z, num_views=int(angles.size))
    for parity, start in (('even', 0), ('odd', 1)):
        half_angles = angles[start::2]
        half = copy_ct_model(full, new_angles=half_angles,
                             new_helical_z_shifts=np.zeros(half_angles.size))
        half.compile_mode = full.compile_mode
        found = dict(recon_shape=tuple(int(s) for s in half.get_params('recon_shape')),
                     recon_slice_offset=float(half.get_params('recon_slice_offset')),
                     delta_voxel=float(half.get_params('delta_voxel')))
        reset = found != reference
        if reset:
            half.set_params(**reference)
        half_z = [float(half.recon_slice_z(int(k))) for k in slice_indices]
        largest = max(abs(a - b) for a, b in zip(half_z, reference_z))
        if largest > 1e-9:
            raise RuntimeError(f'the {parity}-view model puts the requested slices up to {largest} '
                               'ALU away from where the full model puts them')
        checks[parity] = dict(num_views=int(half_angles.size), geometry_before_reset=found,
                              half_geometry_reset=bool(reset), slice_z=half_z,
                              largest_slice_z_difference=float(largest))
        # The sinogram of one parity is a strided view of the full sinogram, so no copy is made.
        halves[parity] = (half, scan.sino[start::2])
    return halves, checks


# ── the job ───────────────────────────────────────────────────────────────────────────────────────

def run_scan(scan):
    """Sweep the detector rotation on one loaded scan and analyze the resulting score curves.

    Returns the analysis dict that :func:`analyze_scan` produced, or None when the scan cannot be
    swept or when too few slices finished to combine.
    """
    name = scan.name
    if scan.vendor_det_rotation is None:
        record(name, 'skip', 0.0, reason='the scan has no vendor detector rotation to compare with')
        return None
    named_degrees = named_candidate_degrees(scan)
    candidates, named_indices = build_candidates(name, named_degrees)
    values_radians = [math.radians(float(d)) for d in candidates]
    identity_index = named_indices[IDENTITY_CANDIDATE]

    full = scan.ct_model
    recon_shape = [int(s) for s in full.get_params('recon_shape')]
    num_slices = recon_shape[2]
    delta_voxel = float(full.get_params('delta_voxel'))
    voxel_row_aspect = float(full.get_params('voxel_row_aspect'))
    # A blur width is a multiple of the voxel pitch, so in ALU it is that multiple of delta_voxel.
    # The reconstruction pixel pitch is delta_voxel as well, so the same multiple is the blur's
    # standard deviation in reconstruction pixels.  The pixel is voxel_row_aspect times delta_voxel
    # in the other in-plane direction, and that aspect is recorded beside the scores.
    sigmas_pixels = {blur_key(blur): float(blur) for blur in BLUR_VOXELS}
    magnification = float(full.get_params('source_detector_dist')
                          / full.get_params('source_iso_dist'))
    slice_indices = [int(round(fraction * (num_slices - 1))) for fraction in SLICE_FRACTIONS]

    halves, checks = build_half_models(scan, slice_indices)
    record(name, 'half_models', 0.0, candidates_degrees=candidates.tolist(),
           named_indices=named_indices, named_degrees=named_degrees,
           slice_indices=slice_indices, delta_voxel=delta_voxel,
           voxel_row_aspect=voxel_row_aspect, recon_shape=recon_shape, **checks)

    matrices = {parity: {key: [] for key in sigmas_pixels} for parity in PARITIES}
    done_slices, rows_from_central_plane, named_images = [], [], []
    for slice_index in slice_indices:
        try:
            z_alu = float(full.recon_slice_z(slice_index))
            # The detector row the slice's center reaches, for a point on the rotation axis.
            row = scan.central_row + z_alu * magnification / scan.delta_det_row
            stacks, seconds = {}, {}
            for parity in ('even', 'odd'):
                half_model, half_sino = halves[parity]
                start = time.perf_counter()
                stacks[parity] = gc.parameter_sweep(half_model, half_sino, 'det_rotation',
                                                    values_radians, slice_index=slice_index)
                seconds[parity] = time.perf_counter() - start
                np.savez(os.path.join(RESULTS, f'{name}_slice_{slice_index:04d}_{parity}.npz'),
                         stack=stacks[parity], candidates_degrees=candidates,
                         slice_index=slice_index, z_alu=z_alu)
            stacks['mean'] = 0.5 * (stacks['even'] + stacks['odd'])

            # The all-view reconstruction at one candidate, to check that the mean of the two half
            # reconstructions is the all-view reconstruction.
            start = time.perf_counter()
            identity_stack = gc.parameter_sweep(full, scan.sino, 'det_rotation',
                                                [values_radians[identity_index]],
                                                slice_index=slice_index)
            seconds['identity'] = time.perf_counter() - start
            reference = np.asarray(identity_stack[:, :, 0], dtype=np.float64)
            approximation = np.asarray(stacks['mean'][:, :, identity_index], dtype=np.float64)
            reference_max = float(np.max(np.abs(reference)))
            reference_rms = float(np.sqrt(np.mean(reference ** 2)))
            difference = approximation - reference
            identity_rel_max = float(np.max(np.abs(difference)) / max(reference_max, 1e-30))
            identity_rms_rel = float(np.sqrt(np.mean(difference ** 2)) / max(reference_rms, 1e-30))

            scores = {parity: score_stack(stacks[parity], sigmas_pixels) for parity in PARITIES}
            for parity in PARITIES:
                for key in sigmas_pixels:
                    matrices[parity][key].append(scores[parity][key])
            done_slices.append(slice_index)
            rows_from_central_plane.append(row - scan.central_row)
            named_images.append(dict(
                slice_index=slice_index, z_alu=z_alu, rows_from_central_plane=row - scan.central_row,
                images={label: np.array(stacks['mean'][:, :, index], copy=True)
                        for label, index in named_indices.items()}))
            record(name, 'slice', seconds['even'] + seconds['odd'], slice_index=slice_index,
                   num_slices=num_slices, z_alu=z_alu, detector_row_on_axis=row,
                   rows_from_central_plane=row - scan.central_row,
                   slice_shape=list(stacks['mean'].shape[:2]),
                   candidates_degrees=candidates.tolist(), named_indices=named_indices,
                   seconds_even=seconds['even'], seconds_odd=seconds['odd'],
                   seconds_identity=seconds['identity'], identity_rel_max=identity_rel_max,
                   identity_rms_rel=identity_rms_rel, identity_reference_max=reference_max,
                   mean_square_at_identity=float(np.mean(approximation ** 2)),
                   blur_voxels=list(BLUR_VOXELS), scores=scores)
            del stacks, identity_stack, reference, approximation, difference
            collect_garbage()
        except Exception:
            record(name, 'slice', 0.0, slice_index=slice_index, traceback=traceback.format_exc())

    if len(done_slices) < 3:
        record(name, 'skip', 0.0, done_slices=done_slices,
               reason='fewer than three slices finished, so the trimmed mean has nothing to trim')
        return None

    matrices = {parity: {key: np.asarray(rows, dtype=np.float64)
                         for key, rows in per_blur.items()}
                for parity, per_blur in matrices.items()}
    analysis = analyze_scan(candidates, matrices, named_indices)
    analysis.update(name=name, slice_indices=done_slices,
                    rows_from_central_plane=rows_from_central_plane,
                    named_degrees=named_degrees)
    record_curves(name, analysis)
    save_trimmed_curves_figure(name, analysis, named_degrees)
    save_slice_curves_figure(name, analysis, matrices, rows_from_central_plane)
    # The named slices are drawn for the slice highest above the central plane, which is where a
    # detector rotation moves the most and where the earlier job found the object's structure.
    highest = max(named_images, key=lambda entry: entry['rows_from_central_plane'])
    save_named_slice_figure(name, highest['slice_index'], highest['z_alu'], highest['images'],
                            named_degrees)
    return analysis


def record_curves(name, analysis):
    """Write one record per view parity and blur width, holding both trimmed curves."""
    for key in analysis['curves']['raw']:
        for parity in PARITIES:
            record(name, 'curve', 0.0, blur=key, parity=parity,
                   candidates_degrees=analysis['candidates_degrees'],
                   named_indices=analysis['named_indices'],
                   raw=analysis['curves']['raw'][key][parity],
                   normalized=analysis['curves']['normalized'][key][parity])


def run_dataset(spec):
    """Load one scan, sweep it, and record what the run cost."""
    name = spec['name']
    dataset_start = time.perf_counter()
    scan, analysis = None, None
    try:
        scan = load_scan(spec)
        if scan is None:
            return None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        analysis = run_scan(scan)
    except Exception:
        record(name, 'error', time.perf_counter() - dataset_start, traceback=traceback.format_exc())
    finally:
        record(name, 'resources', time.perf_counter() - dataset_start,
               max_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2,
               gpu_peak_gb=(torch.cuda.max_memory_allocated(0) / 1024.0 ** 3
                            if torch.cuda.is_available() else None))
        del scan
        collect_garbage()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return analysis


# ── the closing comparison ────────────────────────────────────────────────────────────────────────

def gate_fields(analyses, kind, key):
    """Compare the two scans' minima at one blur width and one kind of trimmed curve.

    The two scans see the same detector, so their fitted minima should agree.  They agree when the
    distance between them is no larger than the wider of the two half widths.  The scan with metal
    is also asked to place its minimum inside a stated range, with the lowest candidate away from
    both ends of the range, which says the range bracketed the minimum.
    """
    nan = float('nan')
    fields = {}
    minima, widths = [], []
    for role, name in zip(('first', 'second'), GATE_SCAN_NAMES):
        analysis = analyses.get(name)
        summary = None
        if analysis is not None:
            summary = analysis['curves'][kind].get(key, {}).get('mean')
        if summary is None:
            fields[role] = dict(name=name, present=False, minimum_degrees=nan,
                                half_width_degrees=nan, argmin_at_end=None,
                                depth_exceeds_floor=None)
            minima.append(nan)
            widths.append(nan)
            continue
        fitted = summary['parabola']
        fields[role] = dict(name=name, present=True,
                            minimum_degrees=fitted['location_degrees'],
                            half_width_degrees=fitted['half_width_degrees'],
                            argmin_degrees=summary['argmin_degrees'],
                            argmin_at_end=summary['argmin_at_end'],
                            depth=summary['depth'], floor=summary['floor'],
                            depth_over_floor=summary['depth_over_floor'],
                            depth_exceeds_floor=summary['depth_exceeds_floor'])
        minima.append(fitted['location_degrees'])
        widths.append(fitted['half_width_degrees'])
    difference = abs(minima[0] - minima[1])
    largest_width = np.nanmax(widths) if np.any(np.isfinite(widths)) else nan
    fields['difference_degrees'] = float(difference)
    fields['agree_within_half_width'] = bool(np.isfinite(difference) and np.isfinite(largest_width)
                                             and difference <= largest_width)
    low, high = METAL_BRACKET_DEGREES
    metal = fields['second']
    fields['metal_min_inside_bracket'] = bool(metal['present']
                                              and np.isfinite(metal['minimum_degrees'])
                                              and low <= metal['minimum_degrees'] <= high
                                              and not metal['argmin_at_end'])
    return fields


def write_gates(analyses):
    """Write one record per blur width that compares the two scans."""
    for blur in BLUR_VOXELS:
        key = blur_key(blur)
        record('job', 'gates', 0.0, blur=key,
               raw=gate_fields(analyses, 'raw', key),
               normalized=gate_fields(analyses, 'normalized', key))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(mbirtorch.__file__)))
    record('job', 'environment', 0.0, torch=torch.__version__,
           gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none',
           mbirtorch=mbirtorch.__version__, mbirtorch_file=mbirtorch.__file__,
           mbirtorch_commit=git_commit(package_root), results=RESULTS, data=DATA, jsonl=JSONL,
           candidate_ranges_degrees=CANDIDATE_RANGES_DEGREES,
           candidate_step_degrees=CANDIDATE_STEP_DEGREES,
           named_candidates_degrees=list(NAMED_CANDIDATES_DEGREES),
           slice_fractions=list(SLICE_FRACTIONS), blur_voxels=list(BLUR_VOXELS),
           parabola_points=PARABOLA_POINTS, half_width_rise=HALF_WIDTH_RISE,
           identity_candidate=IDENTITY_CANDIDATE,
           # The load resolution is a run parameter of the follow-up job, whose 'load_scan' reads it.
           downsample_factor=list(second_job.DOWNSAMPLE_FACTOR), argv=sys.argv)
    analyses = {}
    for spec in DATASETS:
        analyses[spec['name']] = run_dataset(spec)
    write_gates(analyses)
    print('RECON_SWEEP_FINE DONE', flush=True)


if __name__ == '__main__':
    main()
