"""Does the rotation estimate lose its zero point when the object's structure sits far from the
central plane, and does a taller comparison band bring the estimate back?

``estimate_det_rotation`` in ``mbirtorch/preprocess/geometry_calibration.py`` compares a band of
detector rows around the central plane with the mirrored opposite views, and a detector rotation
moves content along the rows.  An object that is the same in every slice therefore puts almost
nothing into that comparison, and the angle the search returns is then set by whatever else the
score contains.  On a real NSI scan whose structure sits 470 to 752 rows from the central plane the
estimate read 0.047 degrees where the vendor's tilt of 0.167 degrees is the right answer.

This script builds that situation on synthetic data, where the true angle is known.  Two phantoms
are used.  Both are a cylinder that is the same in every slice, and both carry one darker slab of
slices.  In the first the slab sits far above the central plane, near the top of the volume; in the
second it sits on the central plane.  Within the volume the slab is the only structure that changes
along the detector rows, so the two phantoms differ only in whether the default band reaches it.
The cylinder is as tall as the volume, so its two end faces land on the outermost detector rows and
are themselves structure across the rows; the band statistic reported with each run says what the
band the estimator used actually held.

A known tilt is put into the data the way the earlier tilt measurements of this feature did it: the
projection is made at four times the detector resolution, rotated there, and binned back down, so
the tilt is not limited by the detector's own sampling.  The same phantom is also projected with no
tilt, as a control that says what the estimate reads when there is nothing to find.  Each of the
three cases is then estimated at six band heights, from the module's own default up to the whole
detector.

Each run also reports a statistic of the band the estimator used: the mean squared difference
between neighboring rows divided by the mean square of the band.  That number says how much of the
band changes along the rows, which is the quantity the hypothesis says is missing, so it is a
candidate for a check the module could make before it trusts an estimate.

Everything runs on the CPU with torch.compile off.  The fine projection is the expensive step and
it runs once per phantom.  Nothing is written to disk; the printed table is the record.

Run from the mbirtorch repository root with the mbirtorch conda environment:

    PYTHONPATH=. /Users/gbuzzard/miniforge3/envs/mbirtorch/bin/python \
        <this directory>/rotation_zero_point_synthetic.py
"""
import math
import os
import time
import warnings

os.environ['MBIRTORCH_NUM_DEVICES'] = '1'

import numpy as np
import torch

import mbirtorch
from mbirtorch.preprocess import geometry_calibration as gc

# ── run parameters ────────────────────────────────────────────────────────────────────────────────
NUM_THREADS = 6

# The geometry.  Both detector pitches are 1 ALU, so a length in ALU is a length in detector pixels,
# and the two source distances give a magnification of two.
NUM_VIEWS = 128
NUM_DET_ROWS = 128
NUM_DET_CHANNELS = 160
DELTA_DET_CHANNEL = 1.0
DELTA_DET_ROW = 1.0
SOURCE_ISO_DIST = 400.0
SOURCE_DETECTOR_DIST = 800.0

# The tilt is put into the data at this many times the detector resolution and binned back down, so
# the tilted sinogram is not limited by the detector's own sampling.
OVERSAMPLING = 4
# The true angle.  It displaces the edge channel by radians(1.5) * 160 / 2 = 2.1 pixels, which is
# the regime of the real scan and above the fraction of a pixel where the resampling kernel's own
# bias dominates.
TRUE_ROTATION_DEGREES = 1.5

# The phantom.  The cylinder's radius is a fraction of the recon's half width in x and y, the slab
# is a run of slices multiplied by a value below one, and the slab's center is a fraction of the way
# from the volume's central slice to its top.  A fraction of zero puts the slab on the central
# plane.
CYLINDER_RADIUS_FRACTION = 0.35
SLAB_SLICES = 8                     # slices of the fine recon grid
SLAB_VALUE = 0.65
FAR_SLAB_FRACTION = 0.78
NEAR_SLAB_FRACTION = 0.0

# The three cases.  Each names the phantom it uses and the angle put into its data.  The cases that
# share a phantom share its projection, so the fine projection runs once per distinct slab position.
CASES = (
    dict(label='far slab, tilted', slab_fraction=FAR_SLAB_FRACTION,
         true_degrees=TRUE_ROTATION_DEGREES),
    dict(label='near slab, tilted', slab_fraction=NEAR_SLAB_FRACTION,
         true_degrees=TRUE_ROTATION_DEGREES),
    dict(label='far slab, no tilt', slab_fraction=FAR_SLAB_FRACTION, true_degrees=0.0),
)

# The band heights each case is estimated at, in detector rows.  None omits the argument, which
# leaves the module its own default; on a cone model the module cuts that default down from 16 rows
# using the cone geometry, and the table prints the height it actually used.  The heights here are
# odd so that each band is centered on a row, and the largest covers all but one row of the
# detector.
BAND_ROWS = (None, 9, 17, 33, 65, 127)

# The band statistic keeps every view.  The sinograms here are small enough that it costs nothing.
BAND_VIEW_STRIDE = 1

# The verdicts call an estimate a reading of the true angle when it is within this fraction of it,
# and within this many degrees of zero for the case with no tilt.
VERDICT_FRACTION = 0.1
VERDICT_ZERO_DEGREES = 0.1


# ── the models and the phantoms ───────────────────────────────────────────────────────────────────

def cone_model(oversampling=1):
    """The test model, on the CPU with compilation off.

    An oversampling factor multiplies the detector counts and divides both pitches, so the fine
    model covers the same field of view as the detector-resolution model and its recon grid is the
    same volume at a finer voxel.  The channel offset is zero in both, and it is the true value, so
    nothing here estimates it.
    """
    angles = np.linspace(0, 2 * np.pi, NUM_VIEWS, endpoint=False)
    shape = (NUM_VIEWS, NUM_DET_ROWS * oversampling, NUM_DET_CHANNELS * oversampling)
    model = mbirtorch.ConeBeamModel(shape, angles, source_detector_dist=SOURCE_DETECTOR_DIST,
                                    source_iso_dist=SOURCE_ISO_DIST, compile_mode='off')
    # The device is pinned before anything reads it, because a model left to choose for itself
    # takes the GPU this machine has and this run is meant to be on the CPU.
    model.configure_devices(devices=['cpu'])
    # Changing the detector pitch changes the recon grid the model would use, so the recon geometry
    # is set again after it.
    model.set_params(delta_det_channel=DELTA_DET_CHANNEL / oversampling,
                     delta_det_row=DELTA_DET_ROW / oversampling)
    model.auto_set_recon_geometry()
    model.set_params(no_warning=True, verbose=0, det_channel_offset=0.0)
    return model


def cylinder_with_slab(recon_shape, slab_fraction):
    """A cylinder that is the same in every slice, with one darker slab of slices in it.

    ``slab_fraction`` is the fraction of the way from the central slice to the top of the volume at
    which the slab is centered, so zero puts it on the central plane.  Returns the phantom and the
    slices the slab occupies.
    """
    num_rows, num_cols, num_slices = (int(size) for size in recon_shape)
    rows, cols = np.indices((num_rows, num_cols))
    radius = CYLINDER_RADIUS_FRACTION * min(num_rows, num_cols) / 2.0
    inside = (((rows - (num_rows - 1) / 2.0) ** 2 + (cols - (num_cols - 1) / 2.0) ** 2)
              <= radius ** 2)
    phantom = np.zeros((num_rows, num_cols, num_slices), dtype=np.float32)
    phantom[inside] = 1.0
    half_extent = (num_slices - 1) / 2.0
    center = half_extent + slab_fraction * half_extent
    low = int(round(center - (SLAB_SLICES - 1) / 2.0))
    low = max(0, min(low, num_slices - SLAB_SLICES))
    phantom[:, :, low:low + SLAB_SLICES] *= SLAB_VALUE
    return phantom, (low, low + SLAB_SLICES)


def slab_fractions():
    """The distinct slab positions of ``CASES``, in the order they first appear."""
    positions = []
    for case in CASES:
        if case['slab_fraction'] not in positions:
            positions.append(case['slab_fraction'])
    return positions


# ── the measurements ──────────────────────────────────────────────────────────────────────────────

def timed_estimate(function, *args, **kwargs):
    """Run one estimator, and return its result, its wall time, and the warnings it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        start = time.perf_counter()
        result = function(*args, **kwargs)
        seconds = time.perf_counter() - start
    return result, seconds, [str(item.message) for item in caught]


def cross_row_statistic(model, sino, row_window):
    """How much the band the estimator used changes along the detector rows.

    The band is cut out with the same reduction the estimator's own record describes, at full
    detector resolution.  The number returned is the mean squared difference between neighboring
    rows divided by the mean square of the band, so it does not depend on how bright the object is.
    A band with no structure across the rows gives a number near zero, and the hypothesis under test
    is that the estimate is unreliable exactly there.  Returns None for a band of one row or a band
    that is entirely zero.
    """
    low, high = (int(value) for value in row_window)
    reduction = {'view_stride': BAND_VIEW_STRIDE, 'bin_factor': 1, 'row_window': (low, high),
                 'full_sinogram_shape': tuple(int(size) for size in sino.shape),
                 'devices': [str(model.torch_device)]}
    band = gc.reduce_sinogram(sino, reduction)
    energy = float(np.mean(band ** 2, dtype=np.float64))
    if band.shape[1] < 2 or energy == 0.0:
        return None
    return float(np.mean(np.diff(band, axis=1) ** 2, dtype=np.float64)) / energy


def run_case(model, sino, case):
    """Estimate the rotation of one sinogram at every band height, and return one row per height."""
    rows = []
    for num_rows in BAND_ROWS:
        # A height of None leaves the argument out, so the module picks the band itself from the
        # cone geometry.
        keywords = {} if num_rows is None else dict(num_rows=num_rows)
        result, seconds, messages = timed_estimate(gc.estimate_det_rotation, model, sino,
                                                   **keywords)
        degrees = math.degrees(float(result.value))
        scores = [float(value) for value in result.scores]
        window = tuple(int(value) for value in result.reduction['row_window'])
        row = dict(label=case['label'], num_rows=num_rows, row_window=window,
                   band_rows=window[1] - window[0], estimate_degrees=degrees,
                   error_degrees=degrees - case['true_degrees'], min_score=min(scores),
                   score_ratio=max(scores) / max(min(scores), 1e-30),
                   cross_row=cross_row_statistic(model, sino, window), seconds=seconds,
                   warnings=messages)
        rows.append(row)
        asked = 'default' if num_rows is None else str(num_rows)
        print(f'  {case["label"]}, band {asked} ({row["band_rows"]} rows, {window}): '
              f'{degrees:+.4f} degrees, error {row["error_degrees"]:+.4f}, {seconds:.1f} s',
              flush=True)
    return rows


# ── the report ────────────────────────────────────────────────────────────────────────────────────

def print_table(rows_by_case):
    """One aligned line per run, in the order the cases are listed at the top of the file."""
    header = (f'{"case":<19}{"band":>9}{"rows":>6}  {"window":<12}{"estimate":>10}{"error":>10}'
              f'{"min score":>12}{"max/min":>9}{"cross-row":>11}{"seconds":>9}{"warnings":>9}')
    print()
    print(header)
    print('-' * len(header))
    for case in CASES:
        for row in rows_by_case.get(case['label'], []):
            asked = 'default' if row['num_rows'] is None else str(row['num_rows'])
            window = f'{row["row_window"][0]}-{row["row_window"][1]}'
            cross = 'none' if row['cross_row'] is None else f'{row["cross_row"]:.3e}'
            print(f'{row["label"]:<19}{asked:>9}{row["band_rows"]:>6}  {window:<12}'
                  f'{row["estimate_degrees"]:>+10.4f}{row["error_degrees"]:>+10.4f}'
                  f'{row["min_score"]:>12.4e}{row["score_ratio"]:>9.3f}{cross:>11}'
                  f'{row["seconds"]:>9.1f}{len(row["warnings"]):>9}')


def verdict(case, rows):
    """One line saying what the band sweep did to this case's estimate."""
    truth = case['true_degrees']
    tolerance = VERDICT_ZERO_DEGREES if truth == 0.0 else VERDICT_FRACTION * abs(truth)
    first, last = rows[0], rows[-1]
    best = min(rows, key=lambda row: abs(row['error_degrees']))
    if abs(first['error_degrees']) <= tolerance:
        judgment = 'the default band already reads the true angle'
    elif abs(last['error_degrees']) <= tolerance:
        judgment = 'the default band misses the true angle and the tallest band reads it'
    elif abs(last['error_degrees']) < abs(first['error_degrees']):
        judgment = ('the estimate moves toward the true angle as the band grows, without reaching '
                    'it')
    else:
        judgment = 'a taller band does not move the estimate toward the true angle'
    return (f'{case["label"]}: true {truth:+.3f} degrees.  Default band ({first["band_rows"]} '
            f'rows) {first["estimate_degrees"]:+.4f}, tallest band ({last["band_rows"]} rows) '
            f'{last["estimate_degrees"]:+.4f}, closest {best["estimate_degrees"]:+.4f} at '
            f'{best["band_rows"]} rows.  {judgment}.')


def print_warnings(rows_by_case):
    """The distinct warnings the runs raised, with how many runs raised each one."""
    counts = {}
    for rows in rows_by_case.values():
        for row in rows:
            for message in row['warnings']:
                prefix = message[:100]
                counts[prefix] = counts.get(prefix, 0) + 1
    if not counts:
        print('no warnings')
        return
    print('warnings, by the first 100 characters of each message:')
    for prefix, count in counts.items():
        print(f'  {count} run(s): {prefix}')


# ── the run ───────────────────────────────────────────────────────────────────────────────────────

def main():
    torch.set_num_threads(NUM_THREADS)
    estimating = cone_model()
    fine = cone_model(oversampling=OVERSAMPLING)
    fine_shape = tuple(int(size) for size in fine.get_params('sinogram_shape'))
    recon_shape = tuple(int(size) for size in fine.get_params('recon_shape'))
    print(f'mbirtorch {mbirtorch.__version__} at {mbirtorch.__file__}, torch {torch.__version__}, '
          f'{NUM_THREADS} threads')
    print(f'detector {NUM_VIEWS} views, {NUM_DET_ROWS} rows, {NUM_DET_CHANNELS} channels; '
          f'fine sinogram {fine_shape}, fine recon {recon_shape}')
    print(f'true rotation {TRUE_ROTATION_DEGREES} degrees, which displaces the edge channel by '
          f'{math.radians(TRUE_ROTATION_DEGREES) * NUM_DET_CHANNELS / 2:.2f} pixels')

    # One reduction record for every case: keep every view, bin the fine detector back down to the
    # detector resolution, and keep every row.  The rotation is applied by reduce_sinogram before
    # the binning, at the fine resolution.
    reduction = {'view_stride': 1, 'bin_factor': OVERSAMPLING, 'row_window': (0, fine_shape[1]),
                 'full_sinogram_shape': fine_shape, 'devices': [str(fine.torch_device)]}

    rows_by_case = {}
    for fraction in slab_fractions():
        phantom, slab = cylinder_with_slab(recon_shape, fraction)
        # The recon's axial extent is the detector's height at iso, so a slab at this fraction of
        # the volume's half height lands at about the same fraction of the detector's half height.
        print(f'\nslab at {fraction:.2f} of the way to the top: fine recon slices {slab[0]} to '
              f'{slab[1]}, which is about {fraction * (NUM_DET_ROWS - 1) / 2:.0f} detector rows '
              f'from the central plane')
        start = time.perf_counter()
        sino_fine = np.asarray(fine.forward_project(phantom), dtype=np.float32)
        print(f'  fine projection {time.perf_counter() - start:.1f} s, shape {sino_fine.shape}')
        del phantom
        for case in CASES:
            if case['slab_fraction'] != fraction:
                continue
            sino = gc.reduce_sinogram(sino_fine, reduction,
                                      det_rotation=-math.radians(case['true_degrees']))
            rows_by_case[case['label']] = run_case(estimating, sino, case)
            del sino
        del sino_fine

    print_table(rows_by_case)
    print()
    for case in CASES:
        print(verdict(case, rows_by_case[case['label']]))
    print()
    print_warnings(rows_by_case)


if __name__ == '__main__':
    main()
