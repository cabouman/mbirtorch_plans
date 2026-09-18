"""The conjugate-view estimators on five real scans, with no trusted ground truth.

Every earlier measurement of this feature used data the projector itself made, so the estimator and
the data shared a model.  This job runs the estimators on scans from an NSI scanner and a Zeiss
Versa scanner.  Those data carry real noise, real stripes, and real beam hardening.  The job
answers four questions.  Does a known integer roll of the sinogram along the channel axis move the
offset estimate by the same amount?  Does the estimate move when stripe removal, a beam-hardening
correction, or a set of dead views changes the data?  Does it agree with the value the vendor's own
calibration recorded?  And what do LEAP's ``find_centerCol`` and ``estimate_tilt`` return on the
same data?

The job has three gates.  The roll test is the primary one, because it needs no ground truth.  Its
gate is that the difference between the estimate on the rolled sinogram and the estimate on the
original is within 0.1 channels of the roll.  Each robustness pair must agree within 0.1 channels.
The vendor comparison must agree within 0.25 channels.  That third one is a sanity check rather
than a hard gate, because the vendor value is itself an estimate.

The beam-hardening case is a proxy for a linearization correction.  It adds a quadratic term to a
band of the sinogram, sized to change the band's largest value by ten percent.  Nothing here
claims that term is the correction a real scan needs.

The NSI sinogram is built here rather than through ``nsi.get_sino_and_model``, because that reader
applies the vendor's detector tilt to the sinogram.  A tilt already applied cannot be estimated, so
the steps of ``nsi._compute_sino_and_params`` are repeated below with the tilt held out and kept as
the vendor value to compare against.

Results are appended to a JSON-lines file as each measurement is taken, so a job cut short still
leaves what it finished, and the datasets are ordered cheapest first for the same reason.  The
measured tables are transcribed to ``real_scan_validation.md`` beside this file.

Run parameters are at the top of the file.  The batch file that submits it,
``real_scan_validation.sbatch``, is beside it.
"""
import glob
import json
import math
import os
import resource
import subprocess
import sys
import tarfile
import time
import traceback
import warnings
from gc import collect as collect_garbage    # 'gc' below is the calibration module, as in the
                                             # other scripts of this feature

os.environ.setdefault('MBIRTORCH_NUM_DEVICES', '1')

import numpy as np
import torch

import mbirtorch
import mbirtorch.preprocess as mtp
from mbirtorch.preprocess import geometry_calibration as gc
from mbirtorch.preprocess import nsi as nsi_reader
from mbirtorch.preprocess import zeiss as zeiss_reader

torch.set_num_threads(14)       # the CPUs the batch file asks for per GPU

# ── run parameters ────────────────────────────────────────────────────────────────────────────────
# The datasets, cheapest first.  An 'nsi' path is a tarball that is extracted into DATA; a 'zeiss'
# path is the .txrm file itself.
DATASETS = (
    dict(name='nsi_small', reader='nsi', path='/depot/bouman/data/Lilly/demo_data_nsi.tgz'),
    dict(name='z62', reader='zeiss', path='/depot/bouman/data/ORNL/versa/ParAM-Round-1_Z62.txrm'),
    dict(name='bga', reader='zeiss',
         path='/depot/bouman/data/Zeiss/purdue_BGA/17U1-250TC-Normal_Tomo_No_HART.txrm'),
    dict(name='nsi_no_metal', reader='nsi',
         path='/depot/bouman/data/Lilly/demo_nsi_vert_no_metal_all_views.tgz'),
    dict(name='nsi_metal', reader='nsi',
         path='/depot/bouman/data/Lilly/demo_nsi_vert_metal_all_views.tgz'),
)
SUBSAMPLE_VIEW_FACTOR = 1           # every view is kept
VERBOSE = 1                         # the readers' own geometry printout goes to the log
VIEW_BATCH = 64                     # views touched per step when the sinogram is changed in place
ROLL_CHANNELS = (3, -3)             # the known channel rolls of the primary test
ROBUSTNESS_ROWS = 64                # rows of the band that the robustness cases modify
LEAP_ROWS = 128                     # rows of the band handed to LEAP
ZEROED_VIEW_FRACTION = 0.05         # fraction of views set to zero in the third robustness case
ZEROED_VIEW_SEED = 0
BH_PEAK_CHANGE = 0.1                # the quadratic term changes the band's maximum by this fraction
SWEEP_CHANNEL_STEPS = (-0.5, 0.5, -1.0, 1.0)    # sweep candidates around the estimate, in channels
MEMORY_ARRAYS = 5                   # sinogram-sized arrays the load is assumed to need at once
MEMORY_FRACTION = 0.7               # fraction of MemAvailable those arrays may use
RESULTS = os.environ.get('REAL_SCAN_RESULTS',
                         os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_real_scan'))
DATA = os.environ.get('REAL_SCAN_DATA',
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
JSONL = os.path.join(RESULTS, 'real_scan_validation.jsonl')


# ── recording ─────────────────────────────────────────────────────────────────────────────────────

def _plain(value):
    """Convert a value json.dumps cannot serialize, so a numpy scalar or array can be recorded."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def record(dataset, kind, seconds, **fields):
    """Append one measurement to the JSON-lines file and print it.

    Every entry carries the dataset name, the kind of measurement, and the wall time it took, so
    the file can be read without knowing which fields a kind adds.
    """
    entry = dict(dataset=dataset, kind=kind, seconds=float(seconds))
    entry.update(fields)
    os.makedirs(RESULTS, exist_ok=True)
    with open(JSONL, 'a') as handle:
        handle.write(json.dumps(entry, default=_plain) + '\n')
    print(json.dumps(entry, default=_plain), flush=True)


def model_param(ct_model, name, default=None):
    """The model's value for a parameter, or ``default`` when this geometry does not have it."""
    try:
        return ct_model.get_params(name)
    except NameError:
        return default


# ── choosing the load resolution ──────────────────────────────────────────────────────────────────

def available_memory_bytes():
    """The host memory Linux reports as available, in bytes, or None off Linux."""
    try:
        with open('/proc/meminfo') as handle:
            for line in handle:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def nsi_scan_shape(dataset_dir):
    """The (views, rows, channels) an NSI scan loads at, read from its text config file.

    The fields and their order are the ones ``nsi.load_scans_and_params`` reads, so the shape here
    matches what that function would return.  A landscape scan swaps the two detector counts, and
    the configured crop removes the same number of pixels from every border.  Returns None when the
    config file is missing or a field cannot be read, and the caller then keeps full resolution.
    """
    config_paths = glob.glob(os.path.join(dataset_dir, '*.nsipro'))
    if not config_paths:
        return None
    tag_section_list = [['source', 'Result'], ['reference', 'Result'], ['pitch', 'Object Radiograph'],
                        ['width pixels', 'Detector'], ['height pixels', 'Detector'],
                        ['number', 'Object Radiograph'], ['Rotation range', 'CT Project Configuration'],
                        ['rotate', 'Correction'], ['flipH', 'Correction'], ['flipV', 'Correction'],
                        ['angleStep', 'Object Radiograph'], ['clockwise', 'Processed'],
                        ['axis', 'Result'], ['normal', 'Result'], ['horizontal', 'Result'],
                        ['crop', 'Radiograph']]
    try:
        fields = nsi_reader._read_str_from_config(config_paths[0], tag_section_list)
        num_channels, num_rows = int(fields[3]), int(fields[4])
        num_views = len(range(0, int(fields[5]), SUBSAMPLE_VIEW_FACTOR))
        if int(fields[7]) in (90, 270):
            num_channels, num_rows = num_rows, num_channels
        max_crop = max(int(value) for value in fields[15].split())
    except Exception:
        # Any failure here only costs the resolution rule; the load itself decides the shape.
        return None
    return num_views, num_rows - 2 * max_crop, num_channels - 2 * max_crop


def zeiss_scan_shape(path):
    """The (views, rows, channels) a Zeiss scan loads at, read from three fields of the .txrm file.

    These are the same OLE fields ``zeiss.read_metadata`` reads.  Only the three are read here,
    because ``read_metadata`` also loads the reference images.  Returns None when a field cannot be
    read, and the caller then keeps full resolution.
    """
    import olefile
    try:
        with olefile.OleFileIO(path) as ole:
            num_channels = zeiss_reader._read_ole_value(ole, 'ImageInfo/ImageWidth', '<I')
            num_rows = zeiss_reader._read_ole_value(ole, 'ImageInfo/ImageHeight', '<I')
            num_views = zeiss_reader._read_ole_value(ole, 'ImageInfo/NoOfImages', '<I')
        return len(range(0, int(num_views), SUBSAMPLE_VIEW_FACTOR)), int(num_rows), int(num_channels)
    except Exception:
        # Any failure here only costs the resolution rule; the load itself decides the shape.
        return None


def choose_downsample(shape):
    """Pick the detector downsampling from the scan's size and the free host memory.

    The load holds several arrays of the scan's size at once, so the rule compares that many
    float32 arrays against most of what the machine reports as available.  Returns the factor and
    the two numbers the decision used.  A scan whose size could not be read keeps full resolution.
    """
    memory = available_memory_bytes()
    if shape is None:
        return (1, 1), None, memory
    array_bytes = 4 * int(shape[0]) * int(shape[1]) * int(shape[2])
    if memory is not None and MEMORY_ARRAYS * array_bytes > MEMORY_FRACTION * memory:
        factor = (2, 2)
    else:
        factor = (1, 1)
    return factor, array_bytes, memory


# ── loading ───────────────────────────────────────────────────────────────────────────────────────

def extract_tarball(path):
    """Extract an NSI tarball into DATA unless its directory is already there.

    Returns the directory the tarball's members sit in, the seconds the extraction took, and
    whether anything was extracted.  A tarball with one top-level directory keeps that name; one
    with several members at the top gets a directory named after the tarball.
    """
    with tarfile.open(path) as handle:
        names = handle.getnames()
        top = sorted({name.split('/')[0] for name in names if name.split('/')[0]})
        if len(top) == 1:
            target, destination = os.path.join(DATA, top[0]), DATA
        else:
            target = os.path.join(DATA, os.path.basename(path).split('.')[0])
            destination = target
        if os.path.exists(target):
            return target, 0.0, False
        os.makedirs(destination, exist_ok=True)
        start = time.perf_counter()
        if hasattr(tarfile, 'data_filter'):
            handle.extractall(destination, filter='data')
        else:
            handle.extractall(destination)
        return target, time.perf_counter() - start, True


def find_nsi_dataset_dir(directory):
    """The directory the NSI reader wants: the one holding a .nsipro file and a Radiographs
    directory.  Some tarballs wrap that directory in one more level, so one level down is searched
    as well.  Returns None when neither holds both."""

    def complete(candidate):
        return (bool(glob.glob(os.path.join(candidate, '*.nsipro')))
                and bool(glob.glob(os.path.join(candidate, 'Radiographs*'))))

    if complete(directory):
        return directory
    for entry in sorted(os.listdir(directory)):
        child = os.path.join(directory, entry)
        if os.path.isdir(child) and complete(child):
            return child
    return None


def nsi_sino_and_model(dataset_dir, downsample_factor):
    """Build the NSI sinogram and model with the vendor's detector tilt held out.

    These are the steps of ``nsi._compute_sino_and_params``, with one change: the tilt that the
    reader would apply to the sinogram is returned instead, so that the estimate can be compared
    with it.  The configured crop is taken from the config file, as the reader does.
    """
    obj_scan, blank_scan, dark_scan, nsi_params, defective_pixel_array = \
        nsi_reader.load_scans_and_params(dataset_dir, subsample_view_factor=SUBSAMPLE_VIEW_FACTOR,
                                         verbose=VERBOSE, offset_correction=True)
    crop = int(nsi_params['max_crop'])
    cone_beam_params, optional_params = nsi_reader.convert_nsi_to_mbirtorch_params(
        nsi_params, downsample_factor=downsample_factor, crop_pixels_sides=crop,
        crop_pixels_top=crop, crop_pixels_bottom=crop)
    obj_scan, blank_scan, dark_scan, defective_pixel_array = mtp.crop_view_data(
        obj_scan, blank_scan, dark_scan, crop_pixels_sides=crop, crop_pixels_top=crop,
        crop_pixels_bottom=crop, defective_pixel_array=defective_pixel_array)
    vendor_det_rotation = float(optional_params.pop('det_rotation'))
    sino = mtp.scan_to_sino(obj_scan, blank_scan, dark_scan, defective_pixel_array,
                            downsample_factor=downsample_factor, det_rotation=0.0)
    del obj_scan, blank_scan, dark_scan
    sino = mtp.correct_background_offset(sino, option='per_view')
    cone_beam_params['geometry_type'] = str(mbirtorch.ConeBeamModel)
    sino, ct_model = mtp.finalize_model(sino, cone_beam_params, optional_params)
    return sino, ct_model, vendor_det_rotation


def clean_nonfinite(sino):
    """Replace every nonfinite value by zero, one batch of views at a time.

    Returns the count replaced and the smallest and largest value that remain.  The work is
    batched because a mask over the whole sinogram would be another array of its size.
    """
    count, low, high = 0, math.inf, -math.inf
    for start in range(0, sino.shape[0], VIEW_BATCH):
        block = sino[start:start + VIEW_BATCH]
        bad = ~np.isfinite(block)
        num_bad = int(bad.sum())
        if num_bad:
            block[bad] = 0.0
            count += num_bad
        low = min(low, float(block.min()))
        high = max(high, float(block.max()))
    return count, low, high


# ── the measurements ──────────────────────────────────────────────────────────────────────────────

def timed_estimate(function, *args, **kwargs):
    """Run one estimator, and return its result, its wall time, and the warnings it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        start = time.perf_counter()
        result = function(*args, **kwargs)
        seconds = time.perf_counter() - start
    return result, seconds, [str(item.message) for item in caught]


def search_fields(result):
    """The parts of a CalibrationResult that describe how the search went."""
    reduction = result.reduction
    return dict(score=float(result.score), search_notes=reduction.get('search_notes'),
                pairs_kept=reduction.get('pairs_kept'), num_pairs=reduction.get('num_pairs'),
                first_pass=reduction.get('first_pass'), bounds=reduction.get('bounds'),
                num_evaluations=int(result.candidates.size))


def central_band(ct_model, num_band_rows):
    """The detector rows ``[lo, hi)`` of a band centered on the row the central plane reaches.

    The estimators compare a band of rows around that same row, so a change made inside this band
    is a change to what they read.
    """
    _, num_rows, _ = (int(s) for s in ct_model.get_params('sinogram_shape'))
    delta_det_row, det_row_offset = ct_model.get_params(['delta_det_row', 'det_row_offset'])
    central_row = (num_rows - 1) / 2.0 + det_row_offset / delta_det_row
    height = max(1, min(int(num_band_rows), num_rows))
    lo = int(round(central_row - (height - 1) / 2.0))
    lo = max(0, min(lo, num_rows - height))
    return lo, lo + height


def roll_in_place(sino, shift):
    """Roll the sinogram along the channel axis, one batch of views at a time.

    Rolling in batches keeps the temporary to one batch.  The roll is circular and moves no value
    between samples, so it changes the geometry the data imply without any interpolation.
    """
    for start in range(0, sino.shape[0], VIEW_BATCH):
        stop = min(start + VIEW_BATCH, sino.shape[0])
        sino[start:stop] = np.roll(sino[start:stop], shift, axis=2)


def sharpness(image):
    """The mean squared finite difference along both axes of a slice, divided by its mean square.

    A reconstruction made with the right channel offset has sharper edges than one made with the
    wrong offset, so this number should peak at the best candidate.
    """
    image = np.asarray(image, dtype=np.float64)
    energy = float(np.mean(image ** 2))
    if not energy > 0.0:
        return float('nan')
    return float((np.mean(np.diff(image, axis=0) ** 2) + np.mean(np.diff(image, axis=1) ** 2)) / energy)


def leap_estimates(ct_model, sino, delta_det_channel, delta_det_row):
    """LEAP's channel offset and tilt on a band of central rows, in this package's terms.

    LEAP's ``centerCol`` is the column of the ray through the rotation axis, which is the detector
    center plus the offset in channels.  Two conventions differ between the two packages and the
    LEAP comparison settled both: LEAP's view angle is 180 degrees minus this package's, and the
    channel axis runs the other way.  The channel reversal mirrors the offset, so its sign is
    negated below.  Only a band of rows is sent, so the GPU holds a fraction of the sinogram.
    """
    from leapctype import tomographicModels
    lo, hi = central_band(ct_model, LEAP_ROWS)
    data = np.ascontiguousarray(sino[:, lo:hi][:, :, ::-1], dtype=np.float32)
    num_views, num_band_rows, num_channels = data.shape
    angles = gc._view_angles(ct_model)
    phis = np.ascontiguousarray(180.0 - np.degrees(angles), dtype=np.float32)
    leapct = tomographicModels()
    leapct.set_gpu(0)
    leapct.set_conebeam(num_views, num_band_rows, num_channels, float(delta_det_row),
                        float(delta_det_channel), (num_band_rows - 1) / 2.0,
                        (num_channels - 1) / 2.0, phis,
                        float(ct_model.get_params('source_iso_dist')),
                        float(ct_model.get_params('source_detector_dist')))
    leapct.set_flatDetector()
    start = time.perf_counter()
    metric = leapct.find_centerCol(data)
    offset_seconds = time.perf_counter() - start
    center_col = float(leapct.get_centerCol())
    offset_alu = -(center_col - (num_channels - 1) / 2.0) * float(delta_det_channel)
    start = time.perf_counter()
    tilt_degrees = float(leapct.estimate_tilt(data))
    tilt_seconds = time.perf_counter() - start
    return dict(band_rows=[lo, hi], leap_centerCol=center_col, leap_metric=float(metric),
                leap_offset_alu=float(offset_alu),
                leap_offset_channels=float(offset_alu / delta_det_channel),
                leap_tilt_degrees=tilt_degrees, leap_offset_seconds=offset_seconds,
                leap_tilt_seconds=tilt_seconds)


def save_sweep_figure(name, stack, values, measures, delta_det_channel, labels):
    """Write the sweep's slices to a PNG, one panel per candidate, on a shared gray scale."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    low, high = np.nanpercentile(stack, [1.0, 99.0])
    columns = math.ceil(stack.shape[2] / 2)
    figure, axes = plt.subplots(2, columns, figsize=(4.5 * columns, 9.0), squeeze=False)
    for k, axis in enumerate(axes.ravel()):
        if k >= stack.shape[2]:
            axis.axis('off')
            continue
        axis.imshow(stack[:, :, k], cmap='gray', vmin=low, vmax=high)
        axis.set_title(f'{labels[k]}\n{values[k] / delta_det_channel:+.3f} channels, '
                       f'sharpness {measures[k]:.4g}', fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f'{name}: det_channel_offset sweep')
    figure.tight_layout()
    figure.savefig(os.path.join(RESULTS, f'{name}_sweep.png'), dpi=120)
    plt.close(figure)


def divisor_at_most(value, limit):
    """The largest divisor of ``value`` that is at most ``limit``."""
    for candidate in range(int(limit), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


# ── one dataset ───────────────────────────────────────────────────────────────────────────────────

def run_dataset(spec):
    """Load one scan and take every measurement on it."""
    name, reader, path = spec['name'], spec['reader'], spec['path']
    dataset_start = time.perf_counter()
    sino = ct_model = None
    try:
        # Find the scan on disk, and learn its size before anything large is read.
        if reader == 'nsi':
            if not os.path.exists(path):
                record(name, 'skip', 0.0, path=path, reason='the tarball is not there')
                return
            directory, extract_seconds, extracted = extract_tarball(path)
            record(name, 'extract', extract_seconds, path=path, directory=directory,
                   extracted=extracted)
            dataset_dir = find_nsi_dataset_dir(directory)
            if dataset_dir is None:
                record(name, 'skip', 0.0, directory=directory,
                       reason='no .nsipro file and Radiographs directory in it or one level down')
                return
            shape = nsi_scan_shape(dataset_dir)
        else:
            if not os.path.exists(path):
                record(name, 'skip', 0.0, path=path, reason='the .txrm file is not there')
                return
            dataset_dir = path
            shape = zeiss_scan_shape(path)

        downsample_factor, array_bytes, memory = choose_downsample(shape)
        record(name, 'resolution', 0.0, full_shape=list(shape) if shape else None,
               downsample_factor=list(downsample_factor), array_bytes=array_bytes,
               mem_available_bytes=memory)

        start = time.perf_counter()
        if reader == 'nsi':
            sino, ct_model, vendor_det_rotation = nsi_sino_and_model(dataset_dir, downsample_factor)
        else:
            sino, ct_model = zeiss_reader.get_sino_and_model(dataset_dir,
                                                             downsample_factor=downsample_factor,
                                                             subsample_view_factor=SUBSAMPLE_VIEW_FACTOR,
                                                             verbose=VERBOSE)
            vendor_det_rotation = None      # the Zeiss reader has no vendor tilt
        load_seconds = time.perf_counter() - start
        if sino.dtype != np.float32:
            # A copy here is a second array of the sinogram's size, so it is made only if the
            # reader returned another dtype.
            sino = sino.astype(np.float32)

        delta_det_channel, delta_det_row, det_row_offset = ct_model.get_params(
            ['delta_det_channel', 'delta_det_row', 'det_row_offset'])
        delta_det_channel, delta_det_row = float(delta_det_channel), float(delta_det_row)
        vendor_offset = float(ct_model.get_params('det_channel_offset'))
        angles = gc._view_angles(ct_model)
        num_channels = int(ct_model.get_params('sinogram_shape')[2])
        nonfinite, sino_min, sino_max = clean_nonfinite(sino)

        entry = dict(shape=list(sino.shape), model_class=type(ct_model).__name__,
                     downsample_factor=list(downsample_factor),
                     alu_unit=model_param(ct_model, 'alu_unit'),
                     delta_det_channel=delta_det_channel, delta_det_row=delta_det_row,
                     source_iso_dist=model_param(ct_model, 'source_iso_dist'),
                     source_detector_dist=model_param(ct_model, 'source_detector_dist'),
                     vendor_offset_alu=vendor_offset,
                     vendor_offset_channels=vendor_offset / delta_det_channel,
                     det_row_offset=float(det_row_offset),
                     angle_min_degrees=float(np.degrees(angles.min())),
                     angle_max_degrees=float(np.degrees(angles.max())),
                     angular_coverage_degrees=math.degrees(gc._angular_coverage(angles)),
                     sino_min=sino_min, sino_max=sino_max, nonfinite_count=nonfinite,
                     nonfinite_fraction=nonfinite / float(sino.size))
        if vendor_det_rotation is not None:
            entry['vendor_det_rotation_degrees'] = math.degrees(vendor_det_rotation)
            entry['vendor_edge_displacement_pixels'] = abs(vendor_det_rotation) * num_channels / 2.0
        record(name, 'scan', load_seconds, **entry)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # The three estimates.  The offset and the rotation are coupled, so the offset is estimated
        # first, then the rotation at that offset, then the offset again at that rotation.
        result, seconds, messages = timed_estimate(gc.estimate_det_channel_offset, ct_model, sino)
        offset1 = float(result.value)
        record(name, 'estimate', seconds, step='offset', parameter='det_channel_offset',
               value_alu=offset1, value_channels=offset1 / delta_det_channel, warnings=messages,
               **search_fields(result))

        result, seconds, messages = timed_estimate(gc.estimate_det_rotation, ct_model, sino,
                                                   det_channel_offset=offset1)
        rotation = float(result.value)
        entry = dict(step='rotation', parameter='det_rotation', value_radians=rotation,
                     value_degrees=math.degrees(rotation),
                     edge_displacement_pixels=abs(rotation) * num_channels / 2.0, warnings=messages,
                     **search_fields(result))
        if vendor_det_rotation is not None:
            entry['vendor_difference_degrees'] = math.degrees(rotation - vendor_det_rotation)
        record(name, 'estimate', seconds, **entry)

        result, seconds, messages = timed_estimate(gc.estimate_det_channel_offset, ct_model, sino,
                                                   det_rotation=rotation)
        offset3 = float(result.value)
        record(name, 'estimate', seconds, step='offset_at_rotation', parameter='det_channel_offset',
               value_alu=offset3, value_channels=offset3 / delta_det_channel,
               vendor_difference_channels=(offset3 - vendor_offset) / delta_det_channel,
               warnings=messages, **search_fields(result))

        # The roll test.  A roll of k channels should move the estimate by k channels, and it needs
        # no ground truth.  The sinogram is rolled back afterward, so the next test starts from the
        # data as loaded.
        for shift in ROLL_CHANNELS:
            start = time.perf_counter()
            roll_in_place(sino, shift)
            roll_seconds = time.perf_counter() - start
            result, seconds, messages = timed_estimate(gc.estimate_det_channel_offset, ct_model,
                                                       sino, det_rotation=rotation)
            roll_in_place(sino, -shift)
            difference = (float(result.value) - offset3) / delta_det_channel
            record(name, 'roll', seconds, roll_channels=shift, value_alu=float(result.value),
                   value_channels=float(result.value) / delta_det_channel,
                   difference_channels=difference, difference_minus_roll=difference - shift,
                   difference_plus_roll=difference + shift, roll_seconds=roll_seconds,
                   warnings=messages, **search_fields(result))

        # The robustness cases.  Each changes a band of rows around the central plane, which is the
        # part of the sinogram the estimator reads, and each is undone from the saved band.
        lo, hi = central_band(ct_model, ROBUSTNESS_ROWS)
        band = sino[:, lo:hi].copy()
        try:
            start = time.perf_counter()
            sino[:, lo:hi] = mtp.remove_all_stripe(band)
            stripe_seconds = time.perf_counter() - start
            result, seconds, messages = timed_estimate(gc.estimate_det_channel_offset, ct_model,
                                                       sino, det_rotation=rotation)
            record(name, 'robustness', seconds, case='stripes', band_rows=[lo, hi],
                   value_alu=float(result.value),
                   difference_channels=(float(result.value) - offset3) / delta_det_channel,
                   correction_seconds=stripe_seconds, warnings=messages, **search_fields(result))
            sino[:, lo:hi] = band

            # A proxy for a linearization correction.  The quadratic term is sized to change the
            # band's largest value by ten percent.
            band_max = float(band.max())
            if np.isfinite(band_max) and band_max > 0.0:
                second_coefficient = BH_PEAK_CHANGE / band_max
                start = time.perf_counter()
                sino[:, lo:hi] = mtp.BH_correction(band, alpha=[1.0, second_coefficient])
                bh_seconds = time.perf_counter() - start
                result, seconds, messages = timed_estimate(gc.estimate_det_channel_offset, ct_model,
                                                           sino, det_rotation=rotation)
                record(name, 'robustness', seconds, case='beam_hardening', band_rows=[lo, hi],
                       band_max=band_max, second_coefficient=second_coefficient,
                       value_alu=float(result.value),
                       difference_channels=(float(result.value) - offset3) / delta_det_channel,
                       correction_seconds=bh_seconds, warnings=messages, **search_fields(result))
                sino[:, lo:hi] = band
            else:
                record(name, 'robustness', 0.0, case='beam_hardening', band_max=band_max,
                       reason='the band has no positive maximum to size the quadratic term against')

            rng = np.random.default_rng(ZEROED_VIEW_SEED)
            num_zeroed = max(1, int(round(ZEROED_VIEW_FRACTION * sino.shape[0])))
            zeroed = rng.choice(sino.shape[0], size=num_zeroed, replace=False)
            sino[zeroed, lo:hi] = 0.0
            result, seconds, messages = timed_estimate(gc.estimate_det_channel_offset, ct_model,
                                                       sino, det_rotation=rotation)
            record(name, 'robustness', seconds, case='zeroed_views', band_rows=[lo, hi],
                   num_zeroed_views=num_zeroed, value_alu=float(result.value),
                   difference_channels=(float(result.value) - offset3) / delta_det_channel,
                   warnings=messages, **search_fields(result))
        finally:
            sino[:, lo:hi] = band
            del band

        # The rotation-direction check.  Its reduced problem needs a view stride that divides the
        # view count and a bin factor that divides both detector counts, which a real scan may not
        # allow at the defaults, so the largest divisors below the defaults are the fallback.
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                start = time.perf_counter()
                try:
                    result = gc.check_rotation_direction(ct_model, sino)
                    retried = None
                except ValueError:
                    stride = divisor_at_most(sino.shape[0], 4)
                    binning = min(divisor_at_most(sino.shape[1], 2), divisor_at_most(sino.shape[2], 2))
                    retried = [stride, binning]
                    result = gc.check_rotation_direction(ct_model, sino, view_stride=stride,
                                                        bin_factor=binning)
                seconds = time.perf_counter() - start
            scores = [float(value) for value in result.scores]
            record(name, 'direction', seconds, value=float(result.value), scores=scores,
                   ratio=max(scores) / max(min(scores), 1e-30), retried_with=retried,
                   warnings=[str(item.message) for item in caught])
        except Exception:
            record(name, 'direction', 0.0, error=traceback.format_exc())

        # LEAP on the same data, for a second opinion from an independent implementation.
        try:
            start = time.perf_counter()
            leap = leap_estimates(ct_model, sino, delta_det_channel, delta_det_row)
            seconds = time.perf_counter() - start
            leap['offset_difference_channels'] = (leap['leap_offset_alu'] - offset3) / delta_det_channel
            leap['tilt_difference_degrees'] = leap['leap_tilt_degrees'] - math.degrees(rotation)
            leap['tilt_sum_degrees'] = leap['leap_tilt_degrees'] + math.degrees(rotation)
            record(name, 'leap', seconds, **leap)
        except Exception as error:
            record(name, 'leap', 0.0, error=repr(error))

        # The sweep: one reconstructed slice per candidate offset, for a person to look at.
        step = delta_det_channel
        values = [vendor_offset, offset3] + [offset3 + fraction * step for fraction in SWEEP_CHANNEL_STEPS]
        labels = ['vendor', 'estimate'] + [f'estimate {fraction:+.1f} ch' for fraction in SWEEP_CHANNEL_STEPS]
        start = time.perf_counter()
        stack = gc.parameter_sweep(ct_model, sino, 'det_channel_offset', values)
        sweep_seconds = time.perf_counter() - start
        measures = [sharpness(stack[:, :, k]) for k in range(stack.shape[2])]
        np.savez(os.path.join(RESULTS, f'{name}_sweep.npz'), stack=stack,
                 values=np.asarray(values), sharpness=np.asarray(measures),
                 labels=np.asarray(labels))
        save_sweep_figure(name, stack, values, measures, step, labels)
        finite = [k for k, value in enumerate(measures) if np.isfinite(value)]
        sharpest = labels[max(finite, key=lambda k: measures[k])] if finite else None
        record(name, 'sweep', sweep_seconds, values_alu=[float(value) for value in values],
               values_channels=[float(value) / step for value in values], labels=labels,
               sharpness=measures, sharpest=sharpest, slice_shape=list(stack.shape[:2]))
        del stack
    except Exception:
        record(name, 'error', time.perf_counter() - dataset_start, traceback=traceback.format_exc())
    finally:
        # Linux reports ru_maxrss in kilobytes, and it is the largest the whole process has been,
        # so the number carries over from earlier datasets.
        record(name, 'resources', time.perf_counter() - dataset_start,
               max_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2,
               gpu_peak_gb=(torch.cuda.max_memory_allocated(0) / 1024.0 ** 3
                            if torch.cuda.is_available() else None))
        del sino, ct_model
        collect_garbage()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def git_commit(directory):
    """The short commit of the git tree at ``directory``, or the reason it could not be read."""
    try:
        finished = subprocess.run(['git', '-C', directory, 'rev-parse', '--short', 'HEAD'],
                                  capture_output=True, text=True, timeout=60)
        return finished.stdout.strip() or finished.stderr.strip()
    except Exception as error:
        return repr(error)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(mbirtorch.__file__)))
    record('job', 'environment', 0.0, torch=torch.__version__,
           gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none',
           mbirtorch=mbirtorch.__version__, mbirtorch_file=mbirtorch.__file__,
           mbirtorch_commit=git_commit(package_root), results=RESULTS, data=DATA, argv=sys.argv)
    for spec in DATASETS:
        run_dataset(spec)
    print('REAL_SCAN_VALIDATION DONE', flush=True)


if __name__ == '__main__':
    main()
