"""Three follow-up measurements on real scans, after the first job left three questions open.

The first job is ``real_scan_validation.py`` in this directory.  It ran the conjugate-view
estimators on five real scans.  It left three questions.

The Zeiss scan ``z62`` runs from -109 to +109 degrees, so it is a short scan.  The conjugate-view
method refused it, and nothing else was measured on it.  On the NSI scan ``nsi_small`` the module's
rotation estimate was 0.047 degrees and the vendor's tilt from the geometry report was 0.167
degrees.  One of those two is wrong.  The module's own estimate cannot say which, because it
applies each candidate angle by resampling the data.  On the same scan ``check_rotation_direction``
returned the negated direction with a score ratio of 1.05, and both scores were near 1.4.  Those
numbers indicate that the high-pass residual is dominated by something both directions share.
Noise at the pixel scale is the likely cause, because the filter widths are fixed in pixels.

Part A measures the short scan.  It sweeps ``det_channel_offset`` and saves one reconstructed slice
per candidate.  It asks LEAP for its own estimate.  It then probes the direct-residual score as a
function of the offset.  That probe is the first measurement of the score on a real short scan.  On
synthetic data the score's minimum sat at the true offset on a short scan.  The score two channels
away from that minimum was only 1.4 to 1.6 times the minimum.  Part A settles whether the minimum
is deeper or shallower on real data.

Part B measures the detector rotation a second way, with no resampling.  A detector rotated by an
angle shifts each row's content along the channels in proportion to the row's height above the
central plane.  The offset estimated on a band of rows at height v therefore equals the offset plus
or minus the angle times v, in matching units.  The slope of the estimate against the band height
is the rotation.  Part B fits that slope over several band heights.  It reports the root mean
square residual of each fit, because the cone angle biases the bands far from the central plane.
It also records the conjugate score at several rotations, so the record shows which rotation the
conjugate comparison itself prefers.

Part C tests whether the direction check is dominated by pixel-scale noise.  It runs the check at
three bin factors, which changes the pixel scale of the reduced problem.  It then scores both
directions at four filter widths, from the module's defaults of 3 rows and 15 channels up to eight
times those.  Each width also gets the fraction of the reduced sinogram's energy that the filter
keeps.  That fraction says how much of what is scored is noise.

Every dataset runs in its own try/except and every part in its own, with tracebacks recorded.
Results are appended to a JSON-lines file as each measurement is taken, so a job cut short leaves
what it finished.  The measured tables are transcribed to ``real_scan_followup.md`` in this
directory.

Run parameters are at the top of the file.  The batch file that submits it,
``real_scan_followup.sbatch``, is in this directory as well.  That batch file puts the directory on
PYTHONPATH, so the helpers of the first job are imported here rather than copied.
"""
import math
import os
import resource
import sys
import time
import traceback
import warnings
from gc import collect as collect_garbage    # 'gc' below is the calibration module, as in the
                                             # other scripts of this feature

os.environ.setdefault('MBIRTORCH_NUM_DEVICES', '1')

import numpy as np
import torch

import mbirtorch
from mbirtorch.preprocess import geometry_calibration as gc
from mbirtorch.preprocess import zeiss as zeiss_reader
from mbirtorch.preprocess.utilities import sino_high_pass_filtering
from mbirtorch.utilities import copy_ct_model

# The first job's helpers.  Its directory is on PYTHONPATH, and importing it also fixes where
# 'record' writes, which the next block changes to this job's file.
import real_scan_validation as first_job
from real_scan_validation import (central_band, clean_nonfinite, extract_tarball,
                                  find_nsi_dataset_dir, git_commit, leap_estimates, model_param,
                                  nsi_sino_and_model, record, save_sweep_figure, search_fields,
                                  sharpness, timed_estimate)

torch.set_num_threads(14)       # the CPUs the batch file asks for per GPU

# ── run parameters ────────────────────────────────────────────────────────────────────────────────
# The datasets, in the order they run.  An 'nsi' path is a tarball that is extracted into DATA, and
# a 'zeiss' path is the .txrm file itself.  Every scan is loaded at full resolution.
DATASETS = (
    dict(name='nsi_small', reader='nsi', path='/depot/bouman/data/Lilly/demo_data_nsi.tgz'),
    dict(name='z62', reader='zeiss', path='/depot/bouman/data/ORNL/versa/ParAM-Round-1_Z62.txrm'),
    dict(name='bga', reader='zeiss',
         path='/depot/bouman/data/Zeiss/purdue_BGA/17U1-250TC-Normal_Tomo_No_HART.txrm'),
    dict(name='nsi_no_metal', reader='nsi',
         path='/depot/bouman/data/Lilly/demo_nsi_vert_no_metal_all_views.tgz'),
)
DOWNSAMPLE_FACTOR = (1, 1)          # every detector pixel is kept
VERBOSE = 1                         # the readers' own geometry printout goes to the log

# Part A, the short scan.  The sweep candidates are these many channels from the vendor offset.
SWEEP_CHANNEL_STEPS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
PROBE_VIEW_STRIDE = 1               # the residual probe keeps every view
PROBE_BIN_FACTOR = 4                # detector binning of the probe, reduced if it does not divide
PROBE_COARSE_STEPS = np.arange(-3.0, 3.01, 0.25)    # channels from the vendor offset
PROBE_FINE_STEPS = np.arange(-1.0, 1.01, 0.1)       # channels from the vendor offset
PROBE_ROW_FRACTION = 0.5            # the central fraction of rows the score is taken over
PROBE_CONTRAST_CHANNELS = 2.0       # the contrast ratio is measured this far from the minimum
PARABOLA_POINTS = 5                 # fine points the parabola is fitted through

# Part B, the rotation from the offset's variation across row bands.
# A synthetic check of this fit ran on a Mac before the job was written.  It used the code below on
# a cone-beam scan of 96 views, 96 channels, and a 20 degree full fan angle.  With no rotation
# applied the fitted slope was zero to five decimal places.  Every applied rotation gave a nonzero
# slope of the right order of magnitude.  The recovered slope was still a poor measure of the
# applied angle there.  Over four applied rotations and four applied channel shears it ran from 0.49
# to 1.61 times the applied angle, and at one detector height the rotation's slope came out with the
# opposite sign to the shear's.  The band estimates scattered by 0.02 to 0.07 channels about the
# fitted line.  On that detector, whose bands spanned 32 to 96 rows, a scatter that size leaves the
# slope poorly determined.  The bands below span up to 1200 rows, which is more than ten times as
# long, so the same per-band scatter would determine the angle far better.  A detector shorter than
# 1216 rows drops the outer bands and shortens that span.  The root mean square residual recorded
# with each fit says whether the line came out straight.
BAND_CENTER_OFFSETS = (-600, -400, -250, -150, -75, 0, 75, 150, 250, 400, 600)   # rows
BAND_ROWS = 16                      # rows in each band
BAND_VIEW_STRIDE = 1                # every view is compared
NEAR_CENTER_ROWS = 150              # the second fit uses only the bands within this many rows

# Part C, the direction check against binning and filter width.
DIRECTION_DATASETS = ('nsi_small', 'bga')
DIRECTION_BIN_FACTORS = (2, 4, 8)
DIRECTION_VIEW_STRIDE_LIMIT = 4     # the stride is the largest divisor of the view count at most this
FILTER_WIDTHS = ((3.0, 15.0), (6.0, 30.0), (12.0, 60.0), (24.0, 120.0))     # (sigma_row, sigma_col)

RESULTS = first_job.RESULTS
DATA = first_job.DATA
JSONL = os.path.join(RESULTS, 'real_scan_followup.jsonl')
# 'record' reads the first job's module-level JSONL name each time it is called, so pointing that
# name at this job's file sends every entry of this job there.
first_job.JSONL = JSONL

# The warning the module raises for every band that does not contain the central plane.  It is
# expected in Part B and is recorded once rather than once per band.
CENTRAL_PLANE_WARNING = 'does not contain the row the central plane reaches'


# ── small numerical helpers ───────────────────────────────────────────────────────────────────────

def common_divisor_at_most(first, second, limit):
    """The largest divisor of both ``first`` and ``second`` that is at most ``limit``."""
    for candidate in range(int(limit), 0, -1):
        if first % candidate == 0 and second % candidate == 0:
            return candidate
    return 1


def divisor_at_most(value, limit):
    """The largest divisor of ``value`` that is at most ``limit``."""
    for candidate in range(int(limit), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def parabola_minimum(candidates, scores):
    """The minimum of the parabola through the points centered on the argmin of ``scores``.

    Returns:
        tuple: ``(estimate, window_start, opens_upward)``.  ``window_start`` is the first candidate
        of the fitted window, and ``opens_upward`` is False when the fit has no minimum.
    """
    k = int(np.argmin(scores))
    half = PARABOLA_POINTS // 2
    lo = int(min(max(k - half, 0), candidates.size - PARABOLA_POINTS))
    x = np.asarray(candidates[lo:lo + PARABOLA_POINTS], dtype=np.float64)
    y = np.asarray(scores[lo:lo + PARABOLA_POINTS], dtype=np.float64)
    a, b, _ = np.polyfit(x, y, 2)
    if a <= 0.0:
        return float('nan'), float(x[0]), False
    return float(-b / (2.0 * a)), float(x[0]), True


def contrast(candidates, scores, target):
    """The candidate nearest ``target``, its score, and that score divided by the smallest score."""
    k = int(np.argmin(np.abs(np.asarray(candidates) - target)))
    smallest = float(np.min(scores))
    return float(candidates[k]), float(scores[k]), float(scores[k] / max(smallest, 1e-30))


def line_fit(heights, estimates):
    """The least-squares straight line of the offset estimate against the band height.

    Both arguments are in ALU, so the slope has no units and its arc tangent is the detector
    rotation.  Returns the slope, the intercept in ALU, and the root mean square residual in ALU.
    """
    heights = np.asarray(heights, dtype=np.float64)
    estimates = np.asarray(estimates, dtype=np.float64)
    slope, intercept = np.polyfit(heights, estimates, 1)
    residual = estimates - (slope * heights + intercept)
    return float(slope), float(intercept), float(np.sqrt(np.mean(residual ** 2)))


# ── the loaded scan ───────────────────────────────────────────────────────────────────────────────

class Scan:
    """One loaded scan: its sinogram, its model, and the geometry numbers every part reads.

    The central row is the detector row that the central plane of the scan reaches, computed the
    way ``_ConjugatePairs._default_reduction`` computes it.
    """

    def __init__(self, name, sino, ct_model, vendor_det_rotation):
        self.name = name
        self.sino = sino
        self.ct_model = ct_model
        self.vendor_det_rotation = vendor_det_rotation
        shape = ct_model.get_params('sinogram_shape')
        self.num_views, self.num_rows, self.num_channels = (int(s) for s in shape)
        delta_det_channel, delta_det_row, det_row_offset = ct_model.get_params(
            ['delta_det_channel', 'delta_det_row', 'det_row_offset'])
        self.delta_det_channel = float(delta_det_channel)
        self.delta_det_row = float(delta_det_row)
        self.det_row_offset = float(det_row_offset)
        self.vendor_offset = float(ct_model.get_params('det_channel_offset'))
        self.central_row = (self.num_rows - 1) / 2.0 + self.det_row_offset / self.delta_det_row

    def channels(self, value_alu):
        """A length in ALU along the channel axis, expressed in channels."""
        return float(value_alu) / self.delta_det_channel


def load_scan(spec):
    """Load one dataset at full resolution and record its ``scan`` entry.

    Returns the Scan, or None when the data are not there.  The NSI datasets are loaded with the
    vendor's detector tilt held out, so that the tilt can be compared with an estimate.
    """
    name, reader, path = spec['name'], spec['reader'], spec['path']
    if reader == 'nsi':
        if not os.path.exists(path):
            record(name, 'skip', 0.0, path=path, reason='the tarball is not there')
            return None
        directory, extract_seconds, extracted = extract_tarball(path)
        record(name, 'extract', extract_seconds, path=path, directory=directory, extracted=extracted)
        dataset_dir = find_nsi_dataset_dir(directory)
        if dataset_dir is None:
            record(name, 'skip', 0.0, directory=directory,
                   reason='no .nsipro file and Radiographs directory in it or one level down')
            return None
    else:
        if not os.path.exists(path):
            record(name, 'skip', 0.0, path=path, reason='the .txrm file is not there')
            return None
        dataset_dir = path

    start = time.perf_counter()
    if reader == 'nsi':
        sino, ct_model, vendor_det_rotation = nsi_sino_and_model(dataset_dir, DOWNSAMPLE_FACTOR)
    else:
        sino, ct_model = zeiss_reader.get_sino_and_model(dataset_dir,
                                                         downsample_factor=DOWNSAMPLE_FACTOR,
                                                         verbose=VERBOSE)
        vendor_det_rotation = None      # the Zeiss reader has no vendor tilt
    load_seconds = time.perf_counter() - start
    if sino.dtype != np.float32:
        # A copy here is a second array of the sinogram's size, so it is made only if the reader
        # returned another dtype.
        sino = sino.astype(np.float32)

    scan = Scan(name, sino, ct_model, vendor_det_rotation)
    nonfinite, sino_min, sino_max = clean_nonfinite(sino)
    angles = gc._view_angles(ct_model)
    entry = dict(shape=list(sino.shape), model_class=type(ct_model).__name__,
                 downsample_factor=list(DOWNSAMPLE_FACTOR),
                 alu_unit=model_param(ct_model, 'alu_unit'),
                 delta_det_channel=scan.delta_det_channel, delta_det_row=scan.delta_det_row,
                 source_iso_dist=model_param(ct_model, 'source_iso_dist'),
                 source_detector_dist=model_param(ct_model, 'source_detector_dist'),
                 vendor_offset_alu=scan.vendor_offset,
                 vendor_offset_channels=scan.channels(scan.vendor_offset),
                 det_row_offset=scan.det_row_offset, central_row=scan.central_row,
                 angle_min_degrees=float(np.degrees(angles.min())),
                 angle_max_degrees=float(np.degrees(angles.max())),
                 angular_coverage_degrees=math.degrees(gc._angular_coverage(angles)),
                 sino_min=sino_min, sino_max=sino_max, nonfinite_count=nonfinite,
                 nonfinite_fraction=nonfinite / float(sino.size))
    if vendor_det_rotation is not None:
        entry['vendor_det_rotation_degrees'] = math.degrees(vendor_det_rotation)
        entry['vendor_edge_displacement_pixels'] = abs(vendor_det_rotation) * scan.num_channels / 2.0
    record(name, 'scan', load_seconds, **entry)
    return scan


def is_short_scan(scan):
    """Whether the conjugate-view method refuses this scan, and the reason it gives.

    The method needs views over a full rotation, and ``_require_conjugate_geometry`` is the check it
    runs.  Any ValueError from that check is treated here as a short scan.
    """
    try:
        gc._require_conjugate_geometry(scan.ct_model, 'det_channel_offset')
        return False, None
    except ValueError as error:
        return True, str(error)


# ── Part A: the real short scan ───────────────────────────────────────────────────────────────────

def part_a_sweep(scan):
    """Reconstruct one slice per candidate channel offset and save the stack and a figure.

    The conjugate-view estimators refuse a scan whose views do not cover a full rotation.  The
    workflow the module offers instead is a person choosing the value by eye, and this stack is
    what that person looks at.  One caveat belongs with the slices.  ``recon_direct`` applies no
    Parker weighting, so its reconstruction of a short scan is itself approximate.
    """
    values = [scan.vendor_offset + step * scan.delta_det_channel for step in SWEEP_CHANNEL_STEPS]
    labels = [f'vendor {step:+.1f} ch' for step in SWEEP_CHANNEL_STEPS]
    start = time.perf_counter()
    stack = gc.parameter_sweep(scan.ct_model, scan.sino, 'det_channel_offset', values)
    seconds = time.perf_counter() - start
    measures = [sharpness(stack[:, :, k]) for k in range(stack.shape[2])]
    np.savez(os.path.join(RESULTS, f'{scan.name}_sweep.npz'), stack=stack,
             values=np.asarray(values), sharpness=np.asarray(measures), labels=np.asarray(labels))
    save_sweep_figure(scan.name, stack, values, measures, scan.delta_det_channel, labels)
    finite = [k for k, value in enumerate(measures) if np.isfinite(value)]
    sharpest = labels[max(finite, key=lambda k: measures[k])] if finite else None
    record(scan.name, 'sweep', seconds, steps_channels=list(SWEEP_CHANNEL_STEPS),
           values_alu=[float(value) for value in values],
           values_channels=[scan.channels(value) for value in values], labels=labels,
           sharpness=measures, sharpest=sharpest, slice_shape=list(stack.shape[:2]))
    del stack


def part_a_leap(scan):
    """LEAP's own channel offset and tilt on the same scan.

    LEAP's ``find_centerCol`` may assume a full rotation, and nothing here checks whether it does.
    Whatever it returns is recorded.  The difference from the vendor offset is the number to read.
    """
    start = time.perf_counter()
    leap = leap_estimates(scan.ct_model, scan.sino, scan.delta_det_channel, scan.delta_det_row)
    seconds = time.perf_counter() - start
    leap['vendor_difference_channels'] = scan.channels(leap['leap_offset_alu'] - scan.vendor_offset)
    if scan.vendor_det_rotation is not None:
        leap['vendor_tilt_difference_degrees'] = (leap['leap_tilt_degrees']
                                                  - math.degrees(scan.vendor_det_rotation))
    record(scan.name, 'leap', seconds, **leap)


def probe_grid(reduced, sino_reduced, filtered, values):
    """The direct-residual score at every candidate offset, and the seconds each took."""
    scores = np.empty(values.size)
    timings = []
    for k, value in enumerate(values):
        start = time.perf_counter()
        reduced.set_params(det_channel_offset=float(value))
        scores[k] = gc._direct_residual_score(reduced, sino_reduced, filtered,
                                              row_fraction=PROBE_ROW_FRACTION)
        timings.append(time.perf_counter() - start)
    return scores, timings


def part_a_residual_probe(scan):
    """The direct-residual score as a function of the channel offset, on a real short scan.

    ``residual_score_probe.py`` in this directory ran the same probe on synthetic data.  There the
    score had its minimum at the true offset in every case.  On the synthetic short scan the score
    two channels away from that minimum was only 1.4 to 1.6 times the minimum.  This run says
    whether the minimum is deeper or shallower on real data.  The whole axial extent is kept,
    because on synthetic data a thin slab added a large term that did not depend on the offset and
    hid the minimum.
    """
    bin_factor = common_divisor_at_most(scan.num_rows, scan.num_channels, PROBE_BIN_FACTOR)
    view_stride = PROBE_VIEW_STRIDE if scan.num_views % PROBE_VIEW_STRIDE == 0 else 1
    start = time.perf_counter()
    reduced, reduction = gc.build_reduced_problem(scan.ct_model, view_stride=view_stride,
                                                  bin_factor=bin_factor, num_slab_slices=None)
    sino_reduced = gc.reduce_sinogram(scan.sino, reduction)
    filtered = sino_high_pass_filtering(sino_reduced)
    setup_seconds = time.perf_counter() - start

    coarse_values = scan.vendor_offset + PROBE_COARSE_STEPS * scan.delta_det_channel
    fine_values = scan.vendor_offset + PROBE_FINE_STEPS * scan.delta_det_channel
    coarse_scores, coarse_timings = probe_grid(reduced, sino_reduced, filtered, coarse_values)
    fine_scores, fine_timings = probe_grid(reduced, sino_reduced, filtered, fine_values)
    timings = coarse_timings + fine_timings

    coarse_argmin_step = float(PROBE_COARSE_STEPS[int(np.argmin(coarse_scores))])
    fine_argmin_step = float(PROBE_FINE_STEPS[int(np.argmin(fine_scores))])
    fitted_step, window_start, opens_upward = parabola_minimum(PROBE_FINE_STEPS, fine_scores)
    contrasts = {}
    for side, target in (('above', coarse_argmin_step + PROBE_CONTRAST_CHANNELS),
                         ('below', coarse_argmin_step - PROBE_CONTRAST_CHANNELS)):
        where, value, ratio = contrast(PROBE_COARSE_STEPS, coarse_scores, target)
        contrasts[f'contrast_{side}_step'] = where
        contrasts[f'contrast_{side}_score'] = value
        contrasts[f'contrast_{side}_ratio'] = ratio

    record(scan.name, 'residual_probe', float(np.sum(timings)),
           bin_factor=bin_factor, view_stride=view_stride,
           reduced_sinogram_shape=list(reduction['sinogram_shape']),
           reduced_recon_shape=list(reduction['recon_shape']),
           row_fraction=PROBE_ROW_FRACTION, setup_seconds=setup_seconds,
           coarse_steps_channels=[float(v) for v in PROBE_COARSE_STEPS],
           coarse_scores=[float(v) for v in coarse_scores],
           fine_steps_channels=[float(v) for v in PROBE_FINE_STEPS],
           fine_scores=[float(v) for v in fine_scores],
           coarse_argmin_step_channels=coarse_argmin_step,
           coarse_min_score=float(np.min(coarse_scores)),
           fine_argmin_step_channels=fine_argmin_step,
           fine_min_score=float(np.min(fine_scores)),
           parabola_window_start_channels=window_start, parabola_opens_upward=opens_upward,
           fitted_step_channels=fitted_step,
           fitted_offset_alu=scan.vendor_offset + fitted_step * scan.delta_det_channel,
           seconds_per_evaluation=float(np.mean(timings)), num_evaluations=len(timings),
           **contrasts)
    del sino_reduced, filtered, reduced


# ── Part B: the rotation from the offset's variation across row bands ─────────────────────────────

def band_reduction(scan, row_lo):
    """A reduction record for the band of rows ``[row_lo, row_lo + BAND_ROWS)``.

    The record holds the keys ``_ConjugatePairs._default_reduction`` puts in one, with the row
    window moved to the requested band.  Every band keeps every view and bins nothing, so the only
    thing that changes from band to band is the rows compared.
    """
    stride = BAND_VIEW_STRIDE
    return {'geometry': gc._geometry_kind(scan.ct_model),
            'view_stride': stride,
            'bin_factor': 1,
            'row_window': (int(row_lo), int(row_lo) + BAND_ROWS),
            'axial_thinning': True,
            'full_sinogram_shape': (scan.num_views, scan.num_rows, scan.num_channels),
            'sinogram_shape': (scan.num_views // stride, BAND_ROWS, scan.num_channels),
            'devices': [str(scan.ct_model.torch_device)]}


def part_b_bands(scan):
    """Estimate the channel offset on several bands of rows above and below the central plane.

    Each band is a window of ``BAND_ROWS`` rows whose center sits a fixed number of rows above or
    below the central plane.  The module warns for every band that leaves the central plane out,
    and that warning is recorded once.  Returns the list of per-band records the fits are made from.
    """
    base_lo, _ = central_band(scan.ct_model, BAND_ROWS)      # the band at offset zero
    bands = []
    warning_recorded = False
    for offset_rows in BAND_CENTER_OFFSETS:
        row_lo = base_lo + int(offset_rows)
        if row_lo < 0 or row_lo + BAND_ROWS > scan.num_rows:
            record(scan.name, 'band', 0.0, offset_rows=int(offset_rows), row_lo=row_lo,
                   reason='the band would leave the detector')
            continue
        reduction = band_reduction(scan, row_lo)
        result, seconds, messages = timed_estimate(gc.estimate_det_channel_offset, scan.ct_model,
                                                   scan.sino, reduction=reduction)
        expected = [text for text in messages if CENTRAL_PLANE_WARNING in text]
        other = [text for text in messages if CENTRAL_PLANE_WARNING not in text]
        if expected and not warning_recorded:
            record(scan.name, 'band_warning', 0.0, message=expected[0],
                   note='the module raises this for every band away from the central plane, and it '
                        'is recorded once')
            warning_recorded = True
        # The band's height is measured from its own center after rounding, which is within half a
        # row of the center the run parameter asked for.
        center_row = row_lo + (BAND_ROWS - 1) / 2.0
        value = float(result.value)
        entry = dict(offset_rows=int(offset_rows), row_window=[row_lo, row_lo + BAND_ROWS],
                     center_row=center_row,
                     height_alu=(center_row - scan.central_row) * scan.delta_det_row,
                     value_alu=value, value_channels=scan.channels(value),
                     vendor_difference_channels=scan.channels(value - scan.vendor_offset))
        bands.append(entry)
        record(scan.name, 'band', seconds, num_expected_warnings=len(expected), warnings=other,
               **entry, **search_fields(result))
    return bands


def record_band_fit(scan, label, bands):
    """Fit the offset estimates against the band heights and record the line.

    The slope is the tangent of the detector rotation.  The sign convention of that rotation is not
    settled here, so the angle and its negative are both recorded.  The root mean square residual
    says how straight the line is.  A bias from the cone angle would show up in that residual.
    Returns the intercept in ALU, or None when there are too few bands to fit.
    """
    if len(bands) < 2:
        record(scan.name, 'band_fit', 0.0, fit=label, num_bands=len(bands),
               reason='a straight line needs at least two bands')
        return None
    heights = [band['height_alu'] for band in bands]
    estimates = [band['value_alu'] for band in bands]
    slope, intercept, rms_alu = line_fit(heights, estimates)
    angle_degrees = math.degrees(math.atan(slope))
    record(scan.name, 'band_fit', 0.0, fit=label, num_bands=len(bands),
           offsets_rows=[band['offset_rows'] for band in bands],
           heights_alu=heights, estimates_alu=estimates,
           slope=slope, angle_degrees=angle_degrees, angle_degrees_negated=-angle_degrees,
           intercept_alu=intercept, intercept_channels=scan.channels(intercept),
           rms_residual_channels=scan.channels(rms_alu),
           edge_displacement_pixels=abs(math.atan(slope)) * scan.num_channels / 2.0)
    return intercept


def part_b_comparisons(scan, offset_alu):
    """The module's rotation estimate and the conjugate score at several rotations.

    The scores show which rotation the conjugate comparison itself prefers.  One caveat belongs
    with them.  ``conjugate_difference`` applies a nonzero rotation by bilinear resampling, and it
    does not resample at all at a rotation of zero.  The zero entry is therefore the only one whose
    data are not smoothed, and smoothing lowers the score on its own.
    """
    result, seconds, messages = timed_estimate(gc.estimate_det_rotation, scan.ct_model, scan.sino,
                                               det_channel_offset=offset_alu)
    module_rotation = float(result.value)
    entry = dict(parameter='det_rotation', det_channel_offset_alu=float(offset_alu),
                 value_radians=module_rotation, value_degrees=math.degrees(module_rotation),
                 edge_displacement_pixels=abs(module_rotation) * scan.num_channels / 2.0,
                 warnings=messages, **search_fields(result))
    if scan.vendor_det_rotation is not None:
        entry['vendor_det_rotation_degrees'] = math.degrees(scan.vendor_det_rotation)
        entry['vendor_difference_degrees'] = math.degrees(module_rotation - scan.vendor_det_rotation)
    record(scan.name, 'module_rotation', seconds, **entry)

    rotations = [('zero', 0.0), ('module', module_rotation)]
    if scan.vendor_det_rotation is not None:
        rotations.append(('vendor', float(scan.vendor_det_rotation)))
        rotations.append(('vendor_negated', -float(scan.vendor_det_rotation)))
    for label, rotation in rotations:
        start = time.perf_counter()
        difference = gc.conjugate_difference(scan.ct_model, scan.sino,
                                             det_channel_offset=offset_alu, det_rotation=rotation)
        mean_square = float(np.mean(difference.astype(np.float64) ** 2))
        seconds = time.perf_counter() - start
        record(scan.name, 'conjugate_score', seconds, rotation_label=label,
               rotation_radians=float(rotation), rotation_degrees=math.degrees(rotation),
               det_channel_offset_alu=float(offset_alu), mean_square=mean_square,
               difference_shape=list(difference.shape),
               resampled=bool(rotation != 0.0))
        del difference


def part_b(scan):
    """The whole of Part B: the bands, the two fits, and the comparisons at the fitted intercept."""
    bands = part_b_bands(scan)
    intercept = record_band_fit(scan, 'all_bands', bands)
    near = [band for band in bands if abs(band['offset_rows']) <= NEAR_CENTER_ROWS]
    record_band_fit(scan, f'within_{NEAR_CENTER_ROWS}_rows', near)
    if intercept is not None:
        part_b_comparisons(scan, intercept)


# ── Part C: the direction check against binning and filter width ──────────────────────────────────

def part_c_binning(scan):
    """The rotation-direction check at three bin factors.

    The bin factor sets the pixel scale of the reduced problem, and the high-pass filter's widths
    are fixed in pixels.  Coarser bins therefore filter over a larger part of the detector.  If the
    small score margin the first job saw comes from pixel-scale noise, the margin should grow with
    the bin factor.
    """
    stride = divisor_at_most(scan.num_views, DIRECTION_VIEW_STRIDE_LIMIT)
    for bin_factor in DIRECTION_BIN_FACTORS:
        if scan.num_rows % bin_factor or scan.num_channels % bin_factor:
            record(scan.name, 'direction', 0.0, bin_factor=bin_factor, view_stride=stride,
                   reason='the bin factor does not divide both detector counts')
            continue
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                start = time.perf_counter()
                result = gc.check_rotation_direction(scan.ct_model, scan.sino, view_stride=stride,
                                                     bin_factor=bin_factor)
                seconds = time.perf_counter() - start
            scores = [float(value) for value in result.scores]
            record(scan.name, 'direction', seconds, bin_factor=bin_factor, view_stride=stride,
                   value=float(result.value), scores=scores,
                   ratio=max(scores) / max(min(scores), 1e-30),
                   reduced_sinogram_shape=list(result.reduction['sinogram_shape']),
                   warnings=[str(item.message) for item in caught])
        except Exception:
            record(scan.name, 'direction', 0.0, bin_factor=bin_factor, view_stride=stride,
                   traceback=traceback.format_exc())


def residual_score_at_widths(sino_reduced, filtered_sino, projection, sigma_row, sigma_col):
    """The module's direct-residual score with the filter widths chosen by the caller.

    ``_direct_residual_score`` calls ``sino_high_pass_filtering`` with its defaults, so its lines
    are repeated here with the widths this function is given.  Both the sinogram and the projection
    are filtered at the same widths, which is what makes the two comparable.  The projection is
    passed in because it does not depend on the widths.
    """
    filtered_projection = sino_high_pass_filtering(projection, sigma_row=sigma_row,
                                                   sigma_col=sigma_col)
    num_rows = sino_reduced.shape[1]
    keep = max(1, int(round(num_rows * PROBE_ROW_FRACTION)))
    lo = (num_rows - keep) // 2
    filtered_sino = filtered_sino[:, lo:lo + keep]
    filtered_projection = filtered_projection[:, lo:lo + keep]
    energy = np.mean(filtered_sino ** 2, dtype=np.float64)
    residual = np.mean((filtered_sino - filtered_projection) ** 2, dtype=np.float64)
    return float(residual / energy)


def part_c_filters(scan):
    """Both direction scores at four filter widths, on one reduced problem.

    The two reduced models are built the way ``check_rotation_direction`` builds them: one from the
    model as given and one from a copy whose view angles are negated.  Each model reconstructs the
    reduced sinogram once and forward projects the result once, because neither step depends on the
    filter widths.  The widths then change only the filtering.
    """
    valid = [b for b in DIRECTION_BIN_FACTORS
             if b <= 4 and scan.num_rows % b == 0 and scan.num_channels % b == 0]
    bin_factor = max(valid) if valid else 1
    stride = divisor_at_most(scan.num_views, DIRECTION_VIEW_STRIDE_LIMIT)
    reduction_kwargs = dict(view_stride=stride, bin_factor=bin_factor, num_slab_slices=None)

    start = time.perf_counter()
    reduced, reduction = gc.build_reduced_problem(scan.ct_model, **reduction_kwargs)
    required, _, _ = scan.ct_model.get_all_params()
    reversed_full = copy_ct_model(scan.ct_model, new_angles=-np.asarray(required['angles']),
                                  new_helical_z_shifts=np.asarray(required['helical_z_shifts']))
    reversed_full.compile_mode = scan.ct_model.compile_mode
    reversed_reduced, _ = gc.build_reduced_problem(reversed_full, **reduction_kwargs)
    sino_reduced = gc.reduce_sinogram(scan.sino, reduction)
    sino_energy = float(np.mean(sino_reduced.astype(np.float64) ** 2))
    projections = {}
    for label, model in (('given', reduced), ('negated', reversed_reduced)):
        recon = model.recon_direct(sino_reduced)
        projections[label] = np.asarray(model.forward_project(recon))
        del recon
    setup_seconds = time.perf_counter() - start
    record(scan.name, 'direction_filter_setup', setup_seconds, bin_factor=bin_factor,
           view_stride=stride, reduced_sinogram_shape=list(reduction['sinogram_shape']),
           reduced_recon_shape=list(reduction['recon_shape']), sino_mean_square=sino_energy)

    for sigma_row, sigma_col in FILTER_WIDTHS:
        start = time.perf_counter()
        filtered = sino_high_pass_filtering(sino_reduced, sigma_row=sigma_row, sigma_col=sigma_col)
        kept_fraction = float(np.mean(filtered.astype(np.float64) ** 2) / max(sino_energy, 1e-30))
        scores = [residual_score_at_widths(sino_reduced, filtered, projections[label], sigma_row,
                                           sigma_col) for label in ('given', 'negated')]
        seconds = time.perf_counter() - start
        record(scan.name, 'direction_filter', seconds, bin_factor=bin_factor, view_stride=stride,
               sigma_row=sigma_row, sigma_col=sigma_col, score_given=scores[0],
               score_negated=scores[1], ratio=max(scores) / max(min(scores), 1e-30),
               better='given' if scores[0] <= scores[1] else 'negated',
               filtered_energy_fraction=kept_fraction, row_fraction=PROBE_ROW_FRACTION)
        del filtered
    del sino_reduced, projections, reduced, reversed_reduced, reversed_full


# ── one dataset ───────────────────────────────────────────────────────────────────────────────────

def run_part(scan, kind, function):
    """Run one part, and record a traceback rather than losing the rest of the dataset to it."""
    try:
        function(scan)
    except Exception:
        record(scan.name, kind, 0.0, traceback=traceback.format_exc())


def run_dataset(spec):
    """Load one scan and run the parts that apply to it."""
    name = spec['name']
    dataset_start = time.perf_counter()
    scan = None
    try:
        scan = load_scan(spec)
        if scan is None:
            return
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        short, reason = is_short_scan(scan)
        record(name, 'coverage', 0.0, short_scan=short, reason=reason,
               angular_coverage_degrees=math.degrees(
                   gc._angular_coverage(gc._view_angles(scan.ct_model))))

        if short:
            run_part(scan, 'sweep', part_a_sweep)
            run_part(scan, 'leap', part_a_leap)
            run_part(scan, 'residual_probe', part_a_residual_probe)
        else:
            run_part(scan, 'band', part_b)
        if name in DIRECTION_DATASETS:
            run_part(scan, 'direction', part_c_binning)
            run_part(scan, 'direction_filter', part_c_filters)
    except Exception:
        record(name, 'error', time.perf_counter() - dataset_start, traceback=traceback.format_exc())
    finally:
        # Linux reports ru_maxrss in kilobytes, and it is the largest the whole process has been,
        # so the number carries over from earlier datasets.
        record(name, 'resources', time.perf_counter() - dataset_start,
               max_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2,
               gpu_peak_gb=(torch.cuda.max_memory_allocated(0) / 1024.0 ** 3
                            if torch.cuda.is_available() else None))
        del scan
        collect_garbage()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    os.makedirs(RESULTS, exist_ok=True)
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(mbirtorch.__file__)))
    record('job', 'environment', 0.0, torch=torch.__version__,
           gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none',
           mbirtorch=mbirtorch.__version__, mbirtorch_file=mbirtorch.__file__,
           mbirtorch_commit=git_commit(package_root), results=RESULTS, data=DATA, jsonl=JSONL,
           argv=sys.argv)
    for spec in DATASETS:
        run_dataset(spec)
    print('REAL_SCAN_FOLLOWUP DONE', flush=True)


if __name__ == '__main__':
    main()
