"""Three measurements on the NSI scans, about the rotation estimate's zero point.

The earlier jobs are ``real_scan_validation.py``, ``real_scan_followup.py``, and
``real_scan_rotation_check.py``, all in this directory.  On the NSI scans the module's
``estimate_det_rotation`` returned 0.047 degrees where the vendor's geometry report gives a
detector tilt of 0.167 degrees, and direct reconstructions showed the vendor's value to be the
right one.  The injection test of the third job found the estimate follows an added rotation with a
slope of one, so what is wrong is the zero point and not the scale.

The estimate compares a band of detector rows around the central plane, and a detector rotation
moves content along the rows.  The object in these scans has its cross-row structure 470 to 752
rows from the central plane, so the default band, which is a few rows tall, may hold almost nothing
the comparison can read.  Question 1 tests that.  The rotation is estimated at six band heights,
from the module's own default up to 201 rows.  If the band is the problem, the estimate moves from
0.047 toward 0.167 degrees as the band grows.  The tallest band here reaches 100 rows on each side
of the central plane, which is short of where this object's structure sits, so the sweep says
whether the estimate moves with the band and how fast, not whether a band that reaches the structure
reads the vendor's tilt.  Each run also reports how much the band it used changes along the rows, as
the mean squared difference between neighboring rows divided by the mean square of the band.  That
number is a candidate for a check the module could make before it trusts an estimate, and it is
recorded here at the same settings the estimates ran at.

Question 2 asks whether the 200-view scan and the 1800-view scan are the same acquisition.  The
third job found the two returning the same rotation to every recorded digit although their view
counts and their reported channel offsets differ.  Three views of the 200-view scan are matched to
the nearest angle in the 1800-view scan and the two frames are subtracted.  A difference of exactly
zero means the same frames are in both files.

Question 3 asks how much of the misalignment between the rotation axis and the detector is not an
in-plane rotation at all.  The reader's tilt comes from ``nsi.calc_det_rotation``, which projects
the rotation axis onto the detector plane and measures the angle there.  Any component of the axis
along the detector normal is dropped by that projection, and no in-plane rotation of the sinogram
can correct it.  The four unit vectors of that geometry are recorded here, three of them read from
the configuration file and the fourth built from two of those, along with their pairwise dot
products, the vendor tilt, the out-of-plane angle, and the total angle between the rotation axis and
the detector columns.

The cost of one estimate grows in proportion to the band height.  The default band on these scans
is a few rows and the earlier job's estimate at it took about 34 seconds on the 1800-view scan, so
the 201-row estimate is roughly 29 times that and the whole sweep is about half an hour per scan.

Every dataset runs in its own try/except and every part in its own, with tracebacks recorded.
Results are appended to a JSON-lines file as each measurement is taken, so a job cut short leaves
what it finished.

Run parameters are at the top of the file.  The batch file that submits it,
``real_scan_band_height.sbatch``, is in this directory as well.  That batch file puts the directory
on PYTHONPATH, so the helpers of the earlier jobs are imported here rather than copied.
"""
import glob
import math
import os
import resource
import sys
import time
import traceback
from gc import collect as collect_garbage    # 'gc' below is the calibration module, as in the
                                             # other scripts of this feature

os.environ.setdefault('MBIRTORCH_NUM_DEVICES', '1')

import numpy as np
import torch

import mbirtorch
import mbirtorch.preprocess as mtp
from mbirtorch.preprocess import geometry_calibration as gc
from mbirtorch.preprocess import nsi as nsi_reader

# The earlier jobs' helpers.  Their directory is on PYTHONPATH.  Importing the first job fixes where
# 'record' writes, and importing the follow-up job points that name at the follow-up's file, so the
# line below that points it at this job's file comes after both imports.
#
# 'load_scan' of the follow-up job calls 'extract_tarball', 'find_nsi_dataset_dir', and
# 'nsi_sino_and_model' of the first job, and it loads an NSI scan with the vendor's detector tilt
# held out, which is what every measurement here needs.
import real_scan_validation as first_job
import real_scan_followup as second_job
from real_scan_validation import (extract_tarball, find_nsi_dataset_dir, git_commit, record,
                                  search_fields, timed_estimate)
from real_scan_followup import load_scan

torch.set_num_threads(14)       # the CPUs the batch file asks for per GPU

# ── run parameters ────────────────────────────────────────────────────────────────────────────────
# The datasets, in the order they run.  Each path is an NSI tarball that the first job's
# 'extract_tarball' unpacks into DATA, and a directory already there is left alone.  'nsi_no_metal'
# is the object without a metal insert at 1800 views and 'nsi_metal' is the same object with one.
DATASETS = (
    dict(name='nsi_no_metal', reader='nsi',
         path='/depot/bouman/data/Lilly/demo_nsi_vert_no_metal_all_views.tgz'),
    dict(name='nsi_metal', reader='nsi',
         path='/depot/bouman/data/Lilly/demo_nsi_vert_metal_all_views.tgz'),
)

# Question 1.  The band heights the rotation is estimated at, in detector rows.  None omits the
# argument, which leaves the module its own default.  The largest entry is 100 rows on each side of
# the central plane, which is still short of the 470 to 752 rows where this object's cross-row
# structure sits; the cost grows with the height, and a band that reached the structure would take
# hours.
BAND_ROWS = (None, 17, 33, 65, 101, 201)
# The band statistic keeps every fourth view.  The statistic is an average over the band, so a
# quarter of the views gives the same number to well within what it is read for, at a quarter of the
# cost and a quarter of the memory.
BAND_STATISTIC_VIEW_STRIDE = 4

# Question 2.  The 200-view scan, loaded while the 1800-view scan is in memory, and the views of it
# that are matched by angle against the 1800-view scan.
SMALL_DATASET = dict(name='nsi_small', reader='nsi',
                     path='/depot/bouman/data/Lilly/demo_data_nsi.tgz')
VIEW_COMPARISON_DATASET = 'nsi_no_metal'
SMALL_VIEW_INDICES = (0, 50, 150)

# Question 3.  The scan whose configuration file the geometry vectors are read from.
VECTOR_DATASET = 'nsi_no_metal'

# The load resolution comes from the follow-up job's own run parameters, because its 'load_scan'
# reads them.  It is recorded in the environment entry so this job's file says what it was.

RESULTS = first_job.RESULTS
DATA = first_job.DATA
JSONL = os.path.join(RESULTS, 'real_scan_band_height.jsonl')
# 'record' reads the first job's module-level JSONL name each time it is called, so pointing that
# name at this job's file sends every entry of this job there.
first_job.JSONL = JSONL


# ── small helpers ─────────────────────────────────────────────────────────────────────────────────

def curve_fields(result):
    """The whole score curve of one search, with the candidates converted to degrees.

    ``candidates`` and ``scores`` hold every evaluation the search made, sorted by candidate.  The
    ratio of the largest score to the smallest says how much the score changes over the range
    searched, and a ratio near one means the score is nearly flat there.
    """
    candidates = [math.degrees(float(value)) for value in result.candidates]
    scores = [float(value) for value in result.scores]
    return dict(candidates_degrees=candidates, scores=scores, score_min=min(scores),
                score_ratio=max(scores) / max(min(scores), 1e-30))


def band_statistic(scan, row_window):
    """How much the band the estimator used changes along the detector rows.

    The band is cut out with the same row window the estimator's own record describes, at full
    detector resolution and at the view stride above.  The number returned is the mean squared
    difference between neighboring rows divided by the mean square of the band, so it does not
    depend on how bright the object is.  A band with no structure across the rows gives a number
    near zero, which is where the rotation comparison has nothing to read.
    """
    low, high = (int(value) for value in row_window)
    reduction = {'view_stride': BAND_STATISTIC_VIEW_STRIDE, 'bin_factor': 1,
                 'row_window': (low, high),
                 'full_sinogram_shape': tuple(int(size) for size in scan.sino.shape),
                 'devices': [str(scan.ct_model.torch_device)]}
    start = time.perf_counter()
    band = gc.reduce_sinogram(scan.sino, reduction)
    energy = float(np.mean(band ** 2, dtype=np.float64))
    if band.shape[1] < 2 or energy == 0.0:
        rows, ratio = None, None
    else:
        rows = float(np.mean(np.diff(band, axis=1) ** 2, dtype=np.float64))
        ratio = rows / energy
    fields = dict(band_shape=list(band.shape), band_view_stride=BAND_STATISTIC_VIEW_STRIDE,
                  band_mean_square=energy, row_difference_mean_square=rows,
                  cross_row_ratio=ratio, band_seconds=time.perf_counter() - start)
    del band
    return fields


def nsi_dataset_dir(spec):
    """The directory ``load_scan`` loaded this dataset from, found the same way it finds it.

    ``extract_tarball`` returns at once when the directory is already there, so this costs nothing
    after the scan has been loaded.
    """
    directory, _, _ = extract_tarball(spec['path'])
    return find_nsi_dataset_dir(directory)


def nsi_geometry_vectors(dataset_dir):
    """The four unit vectors of the NSI geometry, as the reader builds them.

    Three of them are strings in the .nsipro configuration file and are read with the reader's own
    parser: the rotation axis, the detector normal, and the detector rows.  The fourth, the detector
    columns, is the cross product of the normal and the rows, which is how
    ``calc_source_detector_params`` builds the vector it hands to ``calc_det_rotation``.  The sign
    convention on the rotation axis is the reader's: it points down.  Returns None when the
    configuration file is not there.
    """
    config_paths = glob.glob(os.path.join(dataset_dir, '*.nsipro'))
    if not config_paths:
        return None
    tag_section_list = [['axis', 'Result'],         # unit vector along the rotation axis
                        ['normal', 'Result'],       # unit vector from the source to the detector
                        ['horizontal', 'Result']]   # unit vector along the detector rows
    fields = nsi_reader._read_str_from_config(config_paths[0], tag_section_list)
    r_a, r_n, r_h = (np.array([np.single(item) for item in field.split(' ')]) for field in fields)
    if r_a[1] > 0:                  # the reader makes the rotation axis point down
        r_a = -r_a
    r_n = mtp.unit_vector(r_n)      # the reader normalizes the normal before it builds the columns
    r_v = np.cross(r_n, r_h)
    return r_a, r_n, r_h, r_v


# ── Question 1: the estimate against the band height ──────────────────────────────────────────────

def part_1_band_sweep(scan, offset_alu):
    """Estimate the rotation at every band height, with the channel offset held fixed.

    The channel offset is the one estimate made on this scan at the default settings, so the band
    height is the only thing that changes across these runs.  The row window the module actually
    used is recorded beside the height asked for, because the module clamps a height to the detector
    and centers the band on the row the central plane reaches.
    """
    for num_rows in BAND_ROWS:
        # A height of None leaves the argument out, so the module picks the band itself from the
        # cone geometry.
        keywords = {} if num_rows is None else dict(num_rows=num_rows)
        result, seconds, messages = timed_estimate(
            gc.estimate_det_rotation, scan.ct_model, scan.sino, det_channel_offset=offset_alu,
            **keywords)
        value = float(result.value)
        window = tuple(int(edge) for edge in result.reduction['row_window'])
        record(scan.name, 'band_sweep', seconds, num_rows_asked=num_rows, row_window=list(window),
               band_rows=window[1] - window[0], det_channel_offset_alu=float(offset_alu),
               value_radians=value, value_radians_repr=repr(value),
               value_degrees=math.degrees(value),
               edge_displacement_pixels=abs(value) * scan.num_channels / 2.0,
               vendor_det_rotation_degrees=(None if scan.vendor_det_rotation is None
                                            else math.degrees(scan.vendor_det_rotation)),
               warnings=messages, **search_fields(result), **curve_fields(result),
               **band_statistic(scan, window))


# ── Question 2: are the two no-metal scans the same acquisition? ──────────────────────────────────

def part_2_same_acquisition(scan):
    """Match three views of the 200-view scan to the nearest angle of this scan and subtract them.

    The 200-view scan is loaded here, while the larger scan is already in memory, and freed before
    the next dataset.  The angles are compared on the circle, so a pair near zero and a pair near
    360 degrees are close.  A largest absolute difference of exactly zero means the two files hold
    the same frames.
    """
    small = None
    try:
        small = load_scan(SMALL_DATASET)
        if small is None:
            record(scan.name, 'views_compare', 0.0, reason='the 200-view scan is not there')
            return
        if tuple(small.sino.shape[1:]) != tuple(scan.sino.shape[1:]):
            record(scan.name, 'views_compare', 0.0, small_shape=list(small.sino.shape),
                   large_shape=list(scan.sino.shape),
                   reason='the two scans have different detector shapes')
            return
        angles_small = np.mod(gc._view_angles(small.ct_model), 2 * np.pi)
        angles_large = np.mod(gc._view_angles(scan.ct_model), 2 * np.pi)
        for index in SMALL_VIEW_INDICES:
            if index >= angles_small.size:
                record(scan.name, 'views_compare', 0.0, small_index=index,
                       small_num_views=int(angles_small.size),
                       reason='the 200-view scan has no such view')
                continue
            gaps = np.abs(angles_large - angles_small[index])
            gaps = np.minimum(gaps, 2 * np.pi - gaps)
            nearest = int(np.argmin(gaps))
            start = time.perf_counter()
            small_view, large_view = small.sino[index], scan.sino[nearest]
            largest = float(np.max(np.abs(small_view - large_view)))
            record(scan.name, 'views_compare', time.perf_counter() - start, small_index=index,
                   large_index=nearest, small_angle_degrees=float(np.degrees(angles_small[index])),
                   large_angle_degrees=float(np.degrees(angles_large[nearest])),
                   angle_difference_degrees=float(np.degrees(gaps[nearest])),
                   view_shape=list(small_view.shape), largest_absolute_difference=largest,
                   identical=(largest == 0.0), small_view_max=float(np.max(small_view)),
                   large_view_max=float(np.max(large_view)),
                   small_num_views=int(angles_small.size), large_num_views=int(angles_large.size))
    finally:
        del small
        collect_garbage()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── Question 3: what the vendor geometry says beyond an in-plane rotation ─────────────────────────

def part_3_geometry_vectors(scan, spec):
    """Record the vendor's geometry vectors, their pairwise angles, and the vendor tilt.

    ``calc_det_rotation`` projects the rotation axis onto the detector plane and measures the angle
    between that projection and the detector columns.  The component of the axis along the detector
    normal is dropped by the projection, and that component is the part of the misalignment that no
    in-plane rotation of the sinogram can correct.  It is recorded here as the angle whose sine is
    the dot product of the axis with the normal.  The total angle between the axis and the detector
    columns is recorded beside it, so the two say how much of the misalignment the tilt covers.
    """
    dataset_dir = nsi_dataset_dir(spec)
    if dataset_dir is None:
        record(scan.name, 'vectors', 0.0, reason='the NSI dataset directory was not found')
        return
    vectors = nsi_geometry_vectors(dataset_dir)
    if vectors is None:
        record(scan.name, 'vectors', 0.0, directory=dataset_dir,
               reason='no .nsipro file in the dataset directory')
        return
    r_a, r_n, r_h, r_v = vectors
    det_rotation = float(nsi_reader.calc_det_rotation(r_a, r_n, r_h, r_v))
    out_of_plane = math.degrees(math.asin(float(np.clip(np.dot(r_a, r_n), -1.0, 1.0))))
    axis_to_columns = math.degrees(math.acos(float(np.clip(np.dot(r_a, r_v), -1.0, 1.0))))
    record(scan.name, 'vectors', 0.0, directory=dataset_dir,
           r_a=[float(value) for value in r_a], r_n=[float(value) for value in r_n],
           r_h=[float(value) for value in r_h], r_v=[float(value) for value in r_v],
           norms=[float(np.linalg.norm(vector)) for vector in (r_a, r_n, r_h, r_v)],
           dot_a_n=float(np.dot(r_a, r_n)), dot_a_h=float(np.dot(r_a, r_h)),
           dot_a_v=float(np.dot(r_a, r_v)), dot_h_v=float(np.dot(r_h, r_v)),
           dot_n_h=float(np.dot(r_n, r_h)), dot_n_v=float(np.dot(r_n, r_v)),
           det_rotation_radians=det_rotation, det_rotation_degrees=math.degrees(det_rotation),
           # The tilt the loaded model carried comes from the same function through the reader, so
           # the two should agree; the difference says whether this parse matches the reader's.
           model_det_rotation_degrees=(None if scan.vendor_det_rotation is None
                                       else math.degrees(scan.vendor_det_rotation)),
           out_of_plane_degrees=out_of_plane, axis_to_columns_degrees=axis_to_columns)


# ── one dataset ───────────────────────────────────────────────────────────────────────────────────

def run_part(scan, kind, function):
    """Run one part, and record a traceback rather than losing the rest of the dataset to it."""
    try:
        function(scan)
    except Exception:
        record(scan.name, kind, 0.0, traceback=traceback.format_exc())


def run_dataset(spec):
    """Load one scan, estimate its channel offset once, and run the parts that apply to it.

    The band sweep holds the channel offset fixed at that one estimate, so the band height is the
    only thing that varies within a dataset.  The scan is freed at the end of the dataset, before
    the next one is loaded.
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

        run_part(scan, 'band_sweep', lambda loaded: part_1_band_sweep(loaded, offset_alu))
        if name == VIEW_COMPARISON_DATASET:
            run_part(scan, 'views_compare', part_2_same_acquisition)
        if name == VECTOR_DATASET:
            run_part(scan, 'vectors', lambda loaded: part_3_geometry_vectors(loaded, spec))
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
    job_start = time.perf_counter()
    record('job', 'environment', 0.0, torch=torch.__version__,
           gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none',
           mbirtorch=mbirtorch.__version__, mbirtorch_file=mbirtorch.__file__,
           mbirtorch_commit=git_commit(package_root), results=RESULTS, data=DATA, jsonl=JSONL,
           datasets=[spec['name'] for spec in DATASETS], band_rows=list(BAND_ROWS),
           band_statistic_view_stride=BAND_STATISTIC_VIEW_STRIDE,
           view_comparison_dataset=VIEW_COMPARISON_DATASET,
           small_dataset=SMALL_DATASET['name'], small_view_indices=list(SMALL_VIEW_INDICES),
           vector_dataset=VECTOR_DATASET,
           downsample_factor=list(second_job.DOWNSAMPLE_FACTOR), argv=sys.argv)
    for spec in DATASETS:
        run_dataset(spec)
    # The host figure is the largest the whole process has been.  The device figure is the peak
    # since the last dataset reset it, which is the last dataset's peak.
    record('job', 'resources', time.perf_counter() - job_start,
           max_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2,
           gpu_peak_gb=(torch.cuda.max_memory_allocated(0) / 1024.0 ** 3
                        if torch.cuda.is_available() else None))
    print('REAL_SCAN_BAND_HEIGHT DONE', flush=True)


if __name__ == '__main__':
    main()
