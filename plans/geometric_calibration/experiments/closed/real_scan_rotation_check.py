"""Two measurements that settle what the module's rotation estimate reads on the NSI scans.

The two earlier jobs are ``real_scan_validation.py`` and ``real_scan_followup.py``, both in this
directory.  On the NSI scans without metal they left one number unexplained.  The module's
``estimate_det_rotation`` returned 0.0008228244347475705 radians, which is 0.047 degrees, on both
``nsi_small`` and ``nsi_no_metal``.  Those two returns were identical to every recorded digit,
although the two scans have different view counts and their searches reported different channel
offsets and different scores.  The vendor's geometry report gives a detector tilt of 0.167 degrees
on the same scanner.  A fit of the channel offset against the row-band height gave a magnitude of
0.174 to 0.176 degrees.  On ``nsi_metal``, which is the same object with a metal insert, the module
returned 0.149 degrees.

Two hypotheses about that 0.047 degrees are open.  The first is that the search returned a point of
its own lattice rather than a reading of the data, because the score is nearly flat in the rotation
on this object, whose structure runs along the detector rows.  The second is that the module
under-reads only when the angle is small, because its estimate there moves the edge channel by 0.62
pixels.  This job measures both hypotheses, and neither measurement needs a ground truth.

Part 1 runs the rotation search over six settings of the search's own controls.  The bounds take
three values and the coarse count takes two.  Each run records every candidate the search evaluated
and the score at it.  Each run also records the ratio of the largest score on that curve to the
smallest, which says how much the score changes over the range searched.  A value that does not move
when the bounds and the coarse count move is a fixed point of the search.  A value that moves with
them was chosen by the search rather than by the data.

Part 2 puts a known rotation into the real data and asks whether each estimate follows it.  Four
angles from 0.25 to 2.0 degrees are applied with ``correct_det_rotation``, which resamples every
view.  Each rotated sinogram then gets the module's estimate and the band-slope fit of the follow-up
job.  A straight line of each estimate against the applied angle gives a slope and an intercept.  A
slope near one in magnitude means the estimate follows a change in the rotation, and the intercept
is then what the data carried before the injection.  A slope well below one in magnitude means the
estimate does not follow the rotation on this object.

Part 2 also estimates the channel offset at three fixed rotations, on the data as they were loaded.
The earlier jobs found the offset estimate 0.05 to 0.07 channels from the vendor value on these
scans.  These three runs say whether that difference depends on the rotation the comparison applies.

Every dataset runs in its own try/except and every part in its own, with tracebacks recorded.  Each
injected angle also runs in its own try/except, so one failed angle does not cost the rest.  Results
are appended to a JSON-lines file as each measurement is taken, so a job cut short leaves what it
finished.

Run parameters are at the top of the file.  The batch file that submits it,
``real_scan_rotation_check.sbatch``, is in this directory as well.  That batch file puts the
directory on PYTHONPATH, so the helpers of the two earlier jobs are imported here rather than
copied.
"""
import math
import os
import resource
import sys
import time
import traceback
from gc import collect as collect_garbage    # 'gc' below is the calibration module, as in the
                                             # other scripts of this feature

os.environ.setdefault('MBIRTORCH_NUM_DEVICES', '1')

import torch

import mbirtorch
from mbirtorch.preprocess import geometry_calibration as gc
from mbirtorch.preprocess.utilities import correct_det_rotation

# The two earlier jobs' helpers.  Their directory is on PYTHONPATH.  Importing the first job fixes
# where 'record' writes, and importing the follow-up job points that name at the follow-up's file,
# so the line below that points it at this job's file comes after both imports.
#
# Three of the imported helpers call others that are not imported here.  'load_scan' calls
# 'extract_tarball', 'find_nsi_dataset_dir', and 'nsi_sino_and_model' of the first job.
# 'part_b_bands' calls 'central_band' of the first job and 'band_reduction' of the follow-up job.
import real_scan_validation as first_job
import real_scan_followup as second_job
from real_scan_validation import git_commit, record, search_fields, timed_estimate
from real_scan_followup import Scan, line_fit, load_scan, part_b_bands, record_band_fit

torch.set_num_threads(14)       # the CPUs the batch file asks for per GPU

# ── run parameters ────────────────────────────────────────────────────────────────────────────────
# The datasets, in the order they run.  Each path is an NSI tarball that the first job's
# 'extract_tarball' unpacks into DATA, and a directory already there is left alone.  'nsi_small' and
# 'nsi_no_metal' are the same object on the same scanner at 200 and 1800 views.  'nsi_metal' is that
# object with a metal insert.  Every scan is loaded with the vendor's detector tilt held out, which
# is what 'load_scan' does for an NSI path.
DATASETS = (
    dict(name='nsi_small', reader='nsi', path='/depot/bouman/data/Lilly/demo_data_nsi.tgz'),
    dict(name='nsi_no_metal', reader='nsi',
         path='/depot/bouman/data/Lilly/demo_nsi_vert_no_metal_all_views.tgz'),
    dict(name='nsi_metal', reader='nsi',
         path='/depot/bouman/data/Lilly/demo_nsi_vert_metal_all_views.tgz'),
)

# Part 1.  Every combination of a bounds entry and a coarse count runs one rotation search.  A
# number here is the half-width of a range centered on zero, in degrees.  None is the module's own
# default, which is five degrees on each side.
SEARCH_BOUNDS_DEGREES = (None, 1.0, 2.0)
SEARCH_COARSE_COUNTS = (11, 21)

# Part 2.  Each angle is applied to the whole sinogram before the estimates run on it.
INJECTED_DELTAS_DEGREES = (0.25, 0.5, 1.0, 2.0)
# The rotations the channel offset is estimated at, on the data as they were loaded.  The second is
# the module's own rotation estimate on these scans and the third is the vendor's detector tilt.
OFFSET_ROTATIONS_DEGREES = (0.0, 0.047, 0.167)
# The band fit of each injected angle uses only the bands within this many rows of the central
# plane.  It is the follow-up job's second fit, whose root mean square residual on the NSI scans was
# about half that of the fit over every band.
NEAR_CENTER_ROWS = 150

# The load resolution, the band centers, and the band height come from the follow-up job's own run
# parameters, because its 'load_scan' and 'part_b_bands' read them.  They are recorded in the
# environment entry so this job's file says what they were.

RESULTS = first_job.RESULTS
DATA = first_job.DATA
JSONL = os.path.join(RESULTS, 'real_scan_rotation_check.jsonl')
# 'record' reads the first job's module-level JSONL name each time it is called, so pointing that
# name at this job's file sends every entry of this job there.
first_job.JSONL = JSONL
# The band fit uses only the bands within NEAR_CENTER_ROWS of the central plane, so the far bands
# that 'part_b_bands' would also estimate are dropped from its run parameters here.  On an 1800-view
# scan that saves about five minutes per injected angle.
second_job.BAND_CENTER_OFFSETS = tuple(offset for offset in second_job.BAND_CENTER_OFFSETS
                                       if abs(offset) <= NEAR_CENTER_ROWS)


# ── small helpers ─────────────────────────────────────────────────────────────────────────────────

def bounds_radians(half_width_degrees):
    """The search range in radians for one entry of ``SEARCH_BOUNDS_DEGREES``.

    Returns None for a None entry, which leaves the module its own default range.
    """
    if half_width_degrees is None:
        return None
    return -math.radians(half_width_degrees), math.radians(half_width_degrees)


def curve_fields(result):
    """The whole score curve of one search, with the candidates converted to degrees.

    ``candidates`` and ``scores`` hold every evaluation the search made, sorted by candidate.  The
    ratio of the largest score to the smallest says how much the score changes over the range
    searched.  A ratio near one means the score is nearly flat there.
    """
    candidates = [math.degrees(float(value)) for value in result.candidates]
    scores = [float(value) for value in result.scores]
    return dict(candidates_degrees=candidates, scores=scores,
                score_ratio=max(scores) / max(min(scores), 1e-30))


# ── Part 1: the search's own lattice ──────────────────────────────────────────────────────────────

def part_1_search(scan, offset_alu):
    """Run the rotation search over six settings of its bounds and its coarse count.

    Every run uses the same sinogram and the same channel offset, so the search's controls are the
    only thing that changes.  A value that is the same in all six runs did not come from the bounds
    or from the coarse grid.  A value that moves with them was set by the search rather than by the
    data.  The value is recorded as its full repr as well, because the earlier jobs found two scans
    agreeing to every printed digit and a rounded number cannot show that.
    """
    for half_width in SEARCH_BOUNDS_DEGREES:
        for num_coarse in SEARCH_COARSE_COUNTS:
            bounds = bounds_radians(half_width)
            result, seconds, messages = timed_estimate(
                gc.estimate_det_rotation, scan.ct_model, scan.sino,
                det_channel_offset=offset_alu, bounds=bounds, num_coarse=num_coarse)
            value = float(result.value)
            record(scan.name, 'search', seconds,
                   bounds_degrees=None if half_width is None else [-half_width, half_width],
                   num_coarse=num_coarse, det_channel_offset_alu=float(offset_alu),
                   value_radians=value, value_radians_repr=repr(value),
                   value_degrees=math.degrees(value),
                   edge_displacement_pixels=abs(value) * scan.num_channels / 2.0,
                   warnings=messages, **search_fields(result), **curve_fields(result))


# ── Part 2: a known rotation injected into the real data ──────────────────────────────────────────

def part_2_offsets_at_rotation(scan):
    """Estimate the channel offset at three fixed rotations, on the data as they were loaded.

    The comparison applies the rotation it is given to the band before it pairs the views.  The
    earlier jobs found the offset estimate 0.05 to 0.07 channels from the vendor value on these
    scans, with the rotation left at zero.  These three runs say whether that difference depends on
    the rotation applied.
    """
    for rotation_degrees in OFFSET_ROTATIONS_DEGREES:
        rotation = math.radians(rotation_degrees)
        result, seconds, messages = timed_estimate(gc.estimate_det_channel_offset, scan.ct_model,
                                                   scan.sino, det_rotation=rotation)
        value = float(result.value)
        record(scan.name, 'offset_at_rotation', seconds, rotation_degrees=rotation_degrees,
               rotation_radians=rotation, value_alu=value, value_channels=scan.channels(value),
               vendor_offset_channels=scan.channels(scan.vendor_offset),
               vendor_difference_channels=scan.channels(value - scan.vendor_offset),
               warnings=messages, **search_fields(result))


def injected_module_estimate(scan_rotated, offset_alu, delta_degrees):
    """The module's rotation estimate on the sinogram that carries the injected angle.

    The sign convention between ``correct_det_rotation`` and the estimator is not settled here.  The
    estimate less the injected angle and the estimate plus it are therefore both recorded.  One of
    those two is what the estimator would read on the data before the injection, under one
    convention each.  Returns the estimate in degrees.
    """
    result, seconds, messages = timed_estimate(gc.estimate_det_rotation, scan_rotated.ct_model,
                                               scan_rotated.sino, det_channel_offset=offset_alu)
    value = float(result.value)
    value_degrees = math.degrees(value)
    record(scan_rotated.name, 'injected_module', seconds, delta_degrees=delta_degrees,
           det_channel_offset_alu=float(offset_alu), value_radians=value,
           value_degrees=value_degrees, value_minus_delta_degrees=value_degrees - delta_degrees,
           value_plus_delta_degrees=value_degrees + delta_degrees,
           edge_displacement_pixels=abs(value) * scan_rotated.num_channels / 2.0,
           warnings=messages, **search_fields(result), **curve_fields(result))
    return value_degrees


def injected_band_fit(scan_rotated, delta_degrees):
    """The band-slope fit on the sinogram that carries the injected angle.

    ``part_b_bands`` and ``record_band_fit`` record their own entries, and those entries have no
    field for the injected angle.  The summary entry recorded here carries that angle along with
    the numbers of the fit.  In the file, the band entries of one injected angle lie between that
    angle's ``injected_module`` entry and its ``injected_band_summary`` entry.  Returns the
    magnitude of the fitted angle in degrees, or None when there were too few bands to fit.
    """
    label = f'within_{NEAR_CENTER_ROWS}_rows_delta_{delta_degrees}'
    bands = part_b_bands(scan_rotated)
    near = [band for band in bands if abs(band['offset_rows']) <= NEAR_CENTER_ROWS]
    intercept = record_band_fit(scan_rotated, label, near)
    if intercept is None:
        record(scan_rotated.name, 'injected_band_summary', 0.0, delta_degrees=delta_degrees,
               fit=label, num_bands=len(near), reason='a straight line needs at least two bands')
        return None
    # 'record_band_fit' returns only the intercept, so the fit is made a second time here to get the
    # slope and the residual with it.  Both fits are of the same points, so they agree.
    slope, _, rms_alu = line_fit([band['height_alu'] for band in near],
                                 [band['value_alu'] for band in near])
    angle_degrees = math.degrees(math.atan(slope))
    record(scan_rotated.name, 'injected_band_summary', 0.0, delta_degrees=delta_degrees, fit=label,
           num_bands=len(near), slope=slope, angle_degrees=angle_degrees,
           angle_magnitude_degrees=abs(angle_degrees), intercept_alu=intercept,
           intercept_channels=scan_rotated.channels(intercept),
           rms_residual_channels=scan_rotated.channels(rms_alu))
    return abs(angle_degrees)


def record_injection_fit(scan, which, points):
    """Fit a straight line of one estimate against the injected angle and record it.

    The slope says whether the estimate follows the injected angle.  The intercept is the value the
    line gives at an injected angle of zero, which is the rotation the data carried before the
    injection.  ``line_fit`` is a least-squares straight line and its arithmetic does not depend on
    the units, so degrees are passed to it here where the follow-up job passes ALU.
    """
    if len(points) < 2:
        record(scan.name, 'injection_fit', 0.0, which=which, num_points=len(points),
               reason='a straight line needs at least two points')
        return
    deltas = [float(delta) for delta, _ in points]
    estimates = [float(estimate) for _, estimate in points]
    slope, intercept, rms = line_fit(deltas, estimates)
    record(scan.name, 'injection_fit', 0.0, which=which, num_points=len(points),
           deltas_degrees=deltas, estimates_degrees=estimates, slope=slope,
           intercept_degrees=intercept, rms_residual_degrees=rms)


def part_2_injections(scan, offset_alu):
    """Apply four known rotations to the real data and fit each estimate against the applied angle.

    Each angle is applied with ``correct_det_rotation``, which returns a second array of the
    sinogram's size.  That array is dropped before the next angle is applied, so the job holds two
    sinograms at once and not five.
    """
    module_points, band_points = [], []
    for delta_degrees in INJECTED_DELTAS_DEGREES:
        rotated, scan_rotated = None, None
        try:
            start = time.perf_counter()
            rotated = correct_det_rotation(scan.sino, math.radians(delta_degrees))
            record(scan.name, 'inject', time.perf_counter() - start, delta_degrees=delta_degrees,
                   shape=list(rotated.shape), dtype=str(rotated.dtype))
            scan_rotated = Scan(scan.name, rotated, scan.ct_model, scan.vendor_det_rotation)
            value_degrees = injected_module_estimate(scan_rotated, offset_alu, delta_degrees)
            module_points.append((delta_degrees, value_degrees))
            angle = injected_band_fit(scan_rotated, delta_degrees)
            if angle is not None:
                band_points.append((delta_degrees, angle))
        except Exception:
            record(scan.name, 'inject', 0.0, delta_degrees=delta_degrees,
                   traceback=traceback.format_exc())
        finally:
            del scan_rotated, rotated
            collect_garbage()
    record_injection_fit(scan, 'module', module_points)
    record_injection_fit(scan, 'band', band_points)


# ── one dataset ───────────────────────────────────────────────────────────────────────────────────

def run_part(scan, kind, function):
    """Run one part, and record a traceback rather than losing the rest of the dataset to it."""
    try:
        function(scan)
    except Exception:
        record(scan.name, kind, 0.0, traceback=traceback.format_exc())


def run_dataset(spec):
    """Load one scan, estimate its channel offset once, and run the two parts.

    Both parts hold the channel offset fixed at that one estimate, so the rotation is the only thing
    that varies within a dataset.
    """
    name = spec['name']
    dataset_start = time.perf_counter()
    scan = None
    try:
        scan = load_scan(spec)
        if scan is None:
            return
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        result, seconds, messages = timed_estimate(gc.estimate_det_channel_offset, scan.ct_model,
                                                   scan.sino)
        offset_alu = float(result.value)
        record(name, 'offset', seconds, value_alu=offset_alu,
               value_channels=scan.channels(offset_alu),
               vendor_offset_channels=scan.channels(scan.vendor_offset),
               vendor_difference_channels=scan.channels(offset_alu - scan.vendor_offset),
               warnings=messages, **search_fields(result))

        run_part(scan, 'search', lambda loaded: part_1_search(loaded, offset_alu))
        run_part(scan, 'offset_at_rotation', part_2_offsets_at_rotation)
        run_part(scan, 'inject', lambda loaded: part_2_injections(loaded, offset_alu))
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
           datasets=[spec['name'] for spec in DATASETS],
           search_bounds_degrees=list(SEARCH_BOUNDS_DEGREES),
           search_coarse_counts=list(SEARCH_COARSE_COUNTS),
           injected_deltas_degrees=list(INJECTED_DELTAS_DEGREES),
           offset_rotations_degrees=list(OFFSET_ROTATIONS_DEGREES),
           near_center_rows=NEAR_CENTER_ROWS,
           downsample_factor=list(second_job.DOWNSAMPLE_FACTOR),
           band_center_offsets=list(second_job.BAND_CENTER_OFFSETS),
           band_rows=second_job.BAND_ROWS, band_view_stride=second_job.BAND_VIEW_STRIDE,
           argv=sys.argv)
    for spec in DATASETS:
        run_dataset(spec)
    print('REAL_SCAN_ROTATION_CHECK DONE', flush=True)


if __name__ == '__main__':
    main()
