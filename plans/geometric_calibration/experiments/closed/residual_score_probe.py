"""Does the direct-reconstruction residual have a usable minimum in det_channel_offset?

The module ``mbirtorch/preprocess/geometry_calibration.py`` has a private score,
``_direct_residual_score``.  It reconstructs a reduced sinogram directly, forward projects the
result, high-pass filters the sinogram and the projection, and returns the normalized mean squared
difference over the central rows.  Today only ``check_rotation_direction`` uses it.  This script
scores it as a function of ``det_channel_offset`` on synthetic cone-beam data whose true offset is
known.  It asks three things.  Does the score have a minimum at the true offset?  How much does the
score rise two channels away from that minimum?  Does noise move the minimum?

The script runs four scan cases.  Two are full-rotation scans that differ only in the slab: one
reconstructs a slab of 8 slices, and one reconstructs the whole axial extent.  The third is a
helical scan and the fourth is a short scan, both over the whole axial extent.  Comparing the two
full-rotation cases tests the hypothesis that a thin slab gives a shallow minimum.  A ray through a
thin slab also crosses material outside it, and the slab cannot explain that material.

Note on the short scan.  ``recon_direct`` applies no Parker weighting, so its reconstruction of a
short scan is itself approximate.  That case tests only whether the score's minimum still sits at
the true offset.

Run from the mbirtorch repository root with the mbirtorch conda environment:

    PYTHONPATH=. /Users/gbuzzard/miniforge3/envs/mbirtorch/bin/python \
        <this directory>/residual_score_probe.py

Everything runs on the CPU with torch.compile off.  The measured tables and the conclusions are in
``residual_score_probe.md`` beside this file.
"""
import os
import sys
import time
import warnings

os.environ['MBIRTORCH_NUM_DEVICES'] = '1'

import numpy as np
import torch

import mbirtorch
from mbirtorch.preprocess import geometry_calibration as gc
from mbirtorch.preprocess.utilities import sino_high_pass_filtering

# ── run parameters ────────────────────────────────────────────────────────────────────────────────
NUM_THREADS = 6

# The geometry.  The channel pitch is 1 ALU, so an offset in ALU is an offset in channels.  The two
# source distances put the 64 channels across a full fan angle of 20 degrees.
NUM_VIEWS = 128
NUM_DET_ROWS = 32
NUM_DET_CHANNELS = 64
FULL_FAN_DEGREES = 20.0
SOURCE_DETECTOR_DIST = NUM_DET_CHANNELS / 2 / np.tan(np.radians(FULL_FAN_DEGREES / 2))
SOURCE_ISO_DIST = SOURCE_DETECTOR_DIST / 2

# The generating model carries the true offset and the estimating model starts at zero.
TRUE_OFFSET = 1.3
START_OFFSET = 0.0

# The four scan cases, as (name, num_slab_slices).  A value of None keeps the whole axial extent.
CASES = (('full_slab8', 8), ('full_whole', None), ('helical_whole', None), ('short_whole', None))
HELICAL_Z_HALF_RANGE = 4.0          # helical_z_shifts run from -4 to +4 ALU
HELICAL_TURNS = 2                   # helical angles run from 0 to 4 pi
SHORT_SCAN_EXTRA_DEGREES = FULL_FAN_DEGREES   # a short scan is 180 degrees plus the full fan angle

# The candidates.  The coarse grid shows the shape of the curve and the fine grid locates the
# minimum.  Both are in ALU, which is channels here.
COARSE_CANDIDATES = np.arange(-2.0, 4.01, 0.25)
FINE_CANDIDATES = np.arange(0.8, 1.81, 0.05)
ROW_FRACTIONS = (0.5, 1.0)

# The contrast is the score this far from the true offset divided by the coarse minimum score.
CONTRAST_DELTA = 2.0

# The parabola is fitted to this many fine points centered on the fine argmin.
PARABOLA_POINTS = 5

# The noisy cases add Gaussian noise at this fraction of the sinogram maximum.
NOISE_FRACTION_OF_MAX = 0.02
NOISE_SEEDS = (0, 1, 2)
NOISE_ROW_FRACTION = 0.5

LOG_PATH = ('/private/tmp/claude-501/-Users-gbuzzard-Documents-PyCharm-Projects-Research-mbirtorch/'
            '8c5fc8a2-fffe-42ce-9136-72b210546780/scratchpad/residual_score_probe.log')

# Warning messages raised inside the score calls, collected rather than printed.
WARNINGS_SEEN = []


class _Tee:
    """Write every character to the real stdout and to the log file."""

    def __init__(self, stream, handle):
        self.stream, self.handle = stream, handle

    def write(self, text):
        self.stream.write(text)
        self.handle.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.handle.flush()


# ── the models and the data ───────────────────────────────────────────────────────────────────────

def case_angles(case):
    """The view angles and the per-view axial shifts of one case.  The shifts are None for a
    circular scan."""
    if case == 'helical_whole':
        angles = np.linspace(0, HELICAL_TURNS * 2 * np.pi, NUM_VIEWS, endpoint=False)
        z_shifts = np.linspace(-HELICAL_Z_HALF_RANGE, HELICAL_Z_HALF_RANGE, NUM_VIEWS)
        return angles, z_shifts
    if case == 'short_whole':
        return np.linspace(0, np.pi + np.radians(SHORT_SCAN_EXTRA_DEGREES), NUM_VIEWS,
                           endpoint=False), None
    return np.linspace(0, 2 * np.pi, NUM_VIEWS, endpoint=False), None


def build_model(case, det_channel_offset):
    """A cone-beam model for one case at one channel offset, on the CPU with compilation off."""
    angles, z_shifts = case_angles(case)
    kwargs = dict(source_detector_dist=SOURCE_DETECTOR_DIST, source_iso_dist=SOURCE_ISO_DIST,
                  compile_mode='off')
    if z_shifts is not None:
        kwargs['helical_z_shifts'] = z_shifts
    model = mbirtorch.ConeBeamModel((NUM_VIEWS, NUM_DET_ROWS, NUM_DET_CHANNELS), angles, **kwargs)
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0, det_channel_offset=float(det_channel_offset))
    return model


# ── scoring ───────────────────────────────────────────────────────────────────────────────────────

def score_at(reduced, sino_reduced, filtered, offset, row_fraction):
    """One evaluation of the module's residual score, and the seconds it took.

    The timed region holds the ``set_params`` that installs the candidate and the score call, which
    together are the cost of one candidate.  Warnings raised inside the region are collected in
    ``WARNINGS_SEEN`` instead of being printed, so the tables stay readable.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        start = time.perf_counter()
        reduced.set_params(det_channel_offset=float(offset))
        value = gc._direct_residual_score(reduced, sino_reduced, filtered, row_fraction=row_fraction)
        elapsed = time.perf_counter() - start
    for entry in caught:
        WARNINGS_SEEN.append(f'{entry.category.__name__}: {str(entry.message)}')
    return value, elapsed


def score_grid(reduced, sino_reduced, filtered, candidates, row_fraction, timings):
    """The score at every candidate of one grid.  Each evaluation's duration is appended to
    ``timings``."""
    scores = np.empty(candidates.size)
    for k, offset in enumerate(candidates):
        scores[k], elapsed = score_at(reduced, sino_reduced, filtered, offset, row_fraction)
        timings.append(elapsed)
    return scores


def parabola_minimum(candidates, scores):
    """The minimum of the parabola through the points centered on the argmin.

    Returns:
        tuple: ``(estimate, window_start, opens_upward)``.  ``window_start`` is the first candidate
        of the fitted window, and ``opens_upward`` is False when the fit has no minimum.
    """
    k = int(np.argmin(scores))
    half = PARABOLA_POINTS // 2
    lo = int(min(max(k - half, 0), candidates.size - PARABOLA_POINTS))
    x = candidates[lo:lo + PARABOLA_POINTS]
    y = scores[lo:lo + PARABOLA_POINTS]
    a, b, _ = np.polyfit(x, y, 2)
    if a <= 0.0:
        return float('nan'), float(x[0]), False
    return float(-b / (2.0 * a)), float(x[0]), True


def print_grid(label, candidates, scores):
    """Print one grid as a two-column table, so every number can be checked against the log."""
    print(f'    {label}')
    for offset, value in zip(candidates, scores):
        print(f'      {offset:+6.2f}  {value:.6f}')


def contrast(candidates, scores, target):
    """The score at the candidate nearest ``target``, that candidate, and the ratio of that score to
    the smallest score on the grid."""
    k = int(np.argmin(np.abs(candidates - target)))
    smallest = float(np.min(scores))
    return float(candidates[k]), float(scores[k]), float(scores[k] / max(smallest, 1e-30))


# ── one case ──────────────────────────────────────────────────────────────────────────────────────

def run_case(case, num_slab_slices):
    """Run the whole procedure for one case and print its tables."""
    print(f'\n=== case {case} (num_slab_slices={num_slab_slices}) ===')
    generating = build_model(case, TRUE_OFFSET)
    estimating = build_model(case, START_OFFSET)
    generating_shape = tuple(int(s) for s in generating.get_params('recon_shape'))
    estimating_shape = tuple(int(s) for s in estimating.get_params('recon_shape'))
    print(f'  generating model recon shape {generating_shape}, '
          f'estimating model recon shape {estimating_shape}')

    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(generating_shape)
    sino = np.asarray(generating.forward_project(phantom), dtype=np.float32)
    print(f'  full sinogram shape {tuple(sino.shape)}, maximum {float(sino.max()):.4f}')

    reduced, reduction = gc.build_reduced_problem(estimating, view_stride=1, bin_factor=1,
                                                 num_slab_slices=num_slab_slices)
    print(f'  reduced sinogram shape {reduction["sinogram_shape"]}, '
          f'reduced recon shape {reduction["recon_shape"]}, '
          f'row window {reduction["row_window"]} of {NUM_DET_ROWS} rows, '
          f'axial thinning {reduction["axial_thinning"]}')
    sino_reduced = gc.reduce_sinogram(sino, reduction)
    filtered = sino_high_pass_filtering(sino_reduced)

    for row_fraction in ROW_FRACTIONS:
        timings = []
        print(f'  row fraction {row_fraction:.2f}')
        coarse = score_grid(reduced, sino_reduced, filtered, COARSE_CANDIDATES, row_fraction, timings)
        print_grid('coarse grid (candidate, score)', COARSE_CANDIDATES, coarse)
        coarse_argmin = float(COARSE_CANDIDATES[int(np.argmin(coarse))])
        print(f'    coarse argmin {coarse_argmin:+.2f}, minimum score {float(np.min(coarse)):.6f}')
        for sign, target in ((+1, TRUE_OFFSET + CONTRAST_DELTA), (-1, TRUE_OFFSET - CONTRAST_DELTA)):
            where, value, ratio = contrast(COARSE_CANDIDATES, coarse, target)
            side = '+2' if sign > 0 else '-2'
            print(f'    contrast at {side} channels: candidate {where:+.2f}, score {value:.6f}, '
                  f'ratio {ratio:.2f}')

        fine = score_grid(reduced, sino_reduced, filtered, FINE_CANDIDATES, row_fraction, timings)
        print_grid('fine grid (candidate, score)', FINE_CANDIDATES, fine)
        fine_argmin = float(FINE_CANDIDATES[int(np.argmin(fine))])
        fitted, window_start, opens_upward = parabola_minimum(FINE_CANDIDATES, fine)
        print(f'    fine argmin {fine_argmin:+.3f}, parabola window starts {window_start:+.2f}, '
              f'opens upward {opens_upward}')
        print(f'    fitted estimate {fitted:+.4f}, error {fitted - TRUE_OFFSET:+.4f} channels')
        print(f'    seconds per evaluation: mean {float(np.mean(timings)):.3f} over {len(timings)} '
              f'evaluations, total {float(np.sum(timings)):.1f} s')

    # Noise, on the fine grid only, at one row fraction.
    print(f'  noise at {NOISE_FRACTION_OF_MAX:.2f} of the sinogram maximum, '
          f'row fraction {NOISE_ROW_FRACTION:.2f}')
    noisy_estimates = []
    for seed in NOISE_SEEDS:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, NOISE_FRACTION_OF_MAX * float(sino.max()), sino.shape)
        noisy = (sino + noise).astype(np.float32)
        noisy_reduced = gc.reduce_sinogram(noisy, reduction)
        noisy_filtered = sino_high_pass_filtering(noisy_reduced)
        timings = []
        scores = score_grid(reduced, noisy_reduced, noisy_filtered, FINE_CANDIDATES,
                            NOISE_ROW_FRACTION, timings)
        fitted, window_start, opens_upward = parabola_minimum(FINE_CANDIDATES, scores)
        noisy_estimates.append(fitted)
        print(f'    seed {seed}: fine argmin {float(FINE_CANDIDATES[int(np.argmin(scores))]):+.3f}, '
              f'window starts {window_start:+.2f}, opens upward {opens_upward}, '
              f'fitted {fitted:+.4f}, error {fitted - TRUE_OFFSET:+.4f}')
    noisy_estimates = np.array(noisy_estimates)
    print(f'    noisy fitted estimate: mean {float(noisy_estimates.mean()):+.4f}, '
          f'standard deviation {float(noisy_estimates.std(ddof=0)):.4f}, '
          f'mean error {float(noisy_estimates.mean()) - TRUE_OFFSET:+.4f}')

    # The conjugate estimator, for comparison, on the full-rotation cases only.
    if case in ('full_slab8', 'full_whole'):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            start = time.perf_counter()
            result = gc.estimate_det_channel_offset(estimating, sino)
            elapsed = time.perf_counter() - start
        for entry in caught:
            WARNINGS_SEEN.append(f'{entry.category.__name__}: {str(entry.message)}')
        print(f'  conjugate estimator: value {result.value:+.4f}, '
              f'error {result.value - TRUE_OFFSET:+.4f} channels, {elapsed:.1f} s')


def main():
    torch.set_num_threads(NUM_THREADS)
    print(f'residual_score_probe: {NUM_VIEWS} views, {NUM_DET_ROWS} rows, {NUM_DET_CHANNELS} '
          f'channels, full fan angle {FULL_FAN_DEGREES:.1f} degrees')
    print(f'source-to-detector distance {SOURCE_DETECTOR_DIST:.4f} ALU, source-to-iso distance '
          f'{SOURCE_ISO_DIST:.4f} ALU, true offset {TRUE_OFFSET} channels')
    print(f'torch {torch.__version__}, numpy {np.__version__}, threads {torch.get_num_threads()}')
    start = time.perf_counter()
    for case, num_slab_slices in CASES:
        try:
            run_case(case, num_slab_slices)
        except Exception as error:  # a case that raises is a result
            print(f'  case {case} raised {type(error).__name__}: {error}')
    print(f'\nwarnings raised inside the scored regions: {len(WARNINGS_SEEN)}')
    for message in sorted(set(WARNINGS_SEEN)):
        print(f'  {message}')
    print(f'\ntotal wall time {time.perf_counter() - start:.1f} s')


if __name__ == '__main__':
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, 'w') as log_handle:
        original = sys.stdout
        sys.stdout = _Tee(original, log_handle)
        try:
            main()
        finally:
            sys.stdout = original
