"""Can a conjugate-view comparison recover a detector rotation when each candidate angle resamples
the sinogram?

The estimator under test scores a candidate angle by rotating a band of every kept view by that
angle, pairing each view with its mirrored opposite, and taking the mean squared difference or one
minus the normalized correlation.  The rotation is a resampling, and a resampling smooths the data
by an amount that depends on the angle.  Part A measures the bias that smoothing leaves in the
estimate, for three interpolation kernels and two scores, on four synthetic geometries.  Part B
measures how the bias depends on the size of the rotation at one detector size, which is what
decided the kernel and the warning that ``estimate_det_rotation`` uses.

The tilted test data are made fairly.  A resampled tilt at the detector's own resolution would
match the bilinear estimator by construction, so the sinogram is generated at four times the
detector resolution, rotated there, and binned by four.  The measured tables and the conclusion
are in ``rotation_interpolation_bias.md`` beside this file.

Run from the mbirtorch repository root with the mbirtorch conda environment.  Everything runs on
the CPU with torch.compile off.  The four-times sinograms take a few minutes to generate.
"""
import math
import os
import time

os.environ['MBIRTORCH_NUM_DEVICES'] = '1'

import cv2
import numpy as np
import torch

import mbirtorch
from mbirtorch.preprocess import geometry_calibration as gc

# ── run parameters ────────────────────────────────────────────────────────────────────────────────
NUM_THREADS = 6
OVERSAMPLING = 4
# (geometry, num_views, num_det_rows, num_det_channels, full fan angle in degrees or None)
CASES = [('parallel', 64, 32, 64, None), ('cone', 128, 32, 64, 20), ('cone', 128, 64, 64, 20),
         ('cone', 128, 64, 128, 20)]
TRUE_TILTS_DEGREES = (0.3, -0.5, 0.0)
# (label, cv2 interpolation flag, score): 'ssd' is the module's mean squared difference, 'ncc' is
# one minus the normalized correlation.
VARIANTS = [('ssd-linear', cv2.INTER_LINEAR, 'ssd'), ('ncc-linear', cv2.INTER_LINEAR, 'ncc'),
            ('ssd-cubic', cv2.INTER_CUBIC, 'ssd'), ('ncc-cubic', cv2.INTER_CUBIC, 'ncc'),
            ('ncc-lanczos', cv2.INTER_LANCZOS4, 'ncc')]
BAND_ROWS = (None, 16)          # None is the module's default band for the geometry
SEARCH_BOUNDS = (-math.radians(1.0), math.radians(1.0))
NUM_COARSE = 11
TOLERANCE = math.radians(0.005)


def make_model(kind, num_views, num_det_rows, num_det_channels, sdd=None, oversampling=1):
    """A parallel or cone model at the given detector size, with the pitches divided by the
    oversampling so the field of view in ALU is the same at every oversampling."""
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
    shape = (num_views, num_det_rows * oversampling, num_det_channels * oversampling)
    if kind == 'parallel':
        model = mbirtorch.ParallelBeamModel(shape, angles, compile_mode='off')
    else:
        model = mbirtorch.ConeBeamModel(shape, angles, source_detector_dist=sdd,
                                        source_iso_dist=sdd / 2, compile_mode='off')
    model.set_params(delta_det_channel=1.0 / oversampling, delta_det_row=1.0 / oversampling)
    model.auto_set_recon_geometry()
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0)
    return model


def fair_tilted_sinogram(kind, num_views, num_det_rows, num_det_channels, sdd, tilt):
    """The sinogram, at the detector's resolution, of a scan whose detector is rotated by ``tilt``
    radians: generated at the oversampled resolution, rotated there, and binned."""
    fine = make_model(kind, num_views, num_det_rows, num_det_channels, sdd, OVERSAMPLING)
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(fine.get_params('recon_shape'))
    sino_fine = fine.forward_project(phantom).astype(np.float32)
    reduction = {'view_stride': 1, 'bin_factor': OVERSAMPLING,
                 'row_window': (0, num_det_rows * OVERSAMPLING),
                 'full_sinogram_shape': sino_fine.shape, 'devices': ['cpu']}
    # A detector rotated by tilt records the data rotated by -tilt, which apply_calibration
    # undoes by rotating by +tilt.
    return gc.reduce_sinogram(sino_fine, reduction, det_rotation=-tilt)


def rotated_band(sino, reduction, theta, interpolation):
    """The band of rows named by ``reduction``, rotated by ``theta`` about the full detector's
    center with cv2, read from a wider band so the rotation samples nothing outside it."""
    num_views, num_rows, num_channels = reduction['full_sinogram_shape']
    lo, hi = reduction['row_window']
    stride = reduction['view_stride']
    center_row = (num_rows - 1) / 2.0
    margin = gc._rotation_row_margin(theta, max(abs(lo - center_row), abs(hi - 1 - center_row)),
                                     num_channels)
    band_lo, band_hi = max(0, lo - margin), min(num_rows, hi + margin)
    band = np.ascontiguousarray(sino[::stride, band_lo:band_hi, :], dtype=np.float32)
    out = np.empty((band.shape[0], hi - lo, num_channels), np.float32)
    matrix = cv2.getRotationMatrix2D(((num_channels - 1) / 2.0, center_row - band_lo),
                                     math.degrees(theta), 1.0)
    for i in range(band.shape[0]):
        rotated = cv2.warpAffine(band[i], matrix, (num_channels, band_hi - band_lo),
                                 flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out[i] = rotated[lo - band_lo:hi - band_lo]
    return out


def ncc_score(views, opposites, shift, margin):
    """One minus the normalized correlation between the views and the shifted opposites over the
    interior channels, as a trimmed mean over pairs."""
    num_channels = views.shape[2]
    shifted = gc._fourier_shift_channels(opposites, shift)
    region = slice(margin, num_channels - margin)
    a = views[:, :, region].astype(np.float64)
    b = shifted[:, :, region].astype(np.float64)
    a = a - a.mean(axis=(1, 2), keepdims=True)
    b = b - b.mean(axis=(1, 2), keepdims=True)
    numerator = (a * b).sum(axis=(1, 2))
    denominator = np.sqrt((a * a).sum(axis=(1, 2)) * (b * b).sum(axis=(1, 2))) + 1e-30
    per_pair = 1.0 - numerator / denominator
    keep = np.argsort(per_pair)[:max(1, int(round(per_pair.size * 0.9)))]
    return float(per_pair[keep].mean())


def estimate_tilt(model, sino, interpolation, score_name, num_rows, bounds=None):
    """The tilt in degrees that the conjugate-view search returns for one variant.  Every view
    pair is scored; no pair is trimmed."""
    problem = gc._ConjugatePairs(model, None, num_rows)
    offset = problem.model_offset
    margin = problem.channel_margin(offset)
    columns = np.arange(problem.num_channels)
    every_pair = np.arange(problem.num_views)

    def score_at(theta):
        band = rotated_band(sino, problem.reduction, theta, interpolation)
        mirrored = band[:, :, ::-1]
        opposites = np.empty_like(band)
        for i in range(problem.num_views):
            weight = problem.partner_weight[i][:, None]
            opposites[i] = ((1.0 - weight) * mirrored[problem.partner_low[i], :, columns]
                            + weight * mirrored[problem.partner_high[i], :, columns]).T
        if score_name == 'ncc':
            return ncc_score(band, opposites, 2.0 * offset / problem.delta, margin)
        prepared = problem.prepare(band, opposites, margin)
        return problem.score(prepared, opposites, offset, every_pair)

    best, _, _, notes = gc._search_minimum(score_at, bounds or SEARCH_BOUNDS, NUM_COARSE, TOLERANCE)
    return math.degrees(best), notes, problem.reduction['row_window']


# Part B: the bias against the true tilt at one detector size, so that the edge displacement in
# pixels, which is what the resampling smoothing competes with, spans the range a large detector
# reaches at a small tilt.  The search bounds are five degrees on each side of zero.
PART_B_CASES = [('parallel', 64, 32, 64, None), ('cone', 128, 32, 64, 20)]
PART_B_TILTS_DEGREES = (0.3, 1.0, 2.0, 3.0)
PART_B_VARIANTS = [('ssd-linear', cv2.INTER_LINEAR), ('ssd-cubic', cv2.INTER_CUBIC)]
PART_B_BOUNDS = (-math.radians(5.0), math.radians(5.0))


def part_b():
    print('Part B: bias against the true tilt at one detector size, 16 rows, ssd score')
    for kind, num_views, num_det_rows, num_det_channels, fan in PART_B_CASES:
        sdd = None if fan is None else num_det_channels / 2 / math.tan(math.radians(fan / 2))
        model = make_model(kind, num_views, num_det_rows, num_det_channels, sdd)
        fine = make_model(kind, num_views, num_det_rows, num_det_channels, sdd, OVERSAMPLING)
        phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(fine.get_params('recon_shape'))
        sino_fine = fine.forward_project(phantom).astype(np.float32)
        reduction = {'view_stride': 1, 'bin_factor': OVERSAMPLING,
                     'row_window': (0, num_det_rows * OVERSAMPLING),
                     'full_sinogram_shape': sino_fine.shape, 'devices': ['cpu']}
        print(f'  {kind}: {num_views} views, {num_det_rows} rows, {num_det_channels} channels')
        for tilt in PART_B_TILTS_DEGREES:
            sino = gc.reduce_sinogram(sino_fine, reduction, det_rotation=-math.radians(tilt))
            displacement = math.radians(tilt) * num_det_channels / 2
            cells = []
            for label, interpolation in PART_B_VARIANTS:
                estimate, notes, _ = estimate_tilt(model, sino, interpolation, 'ssd', 16, PART_B_BOUNDS)
                flag = '*' if notes else ''
                cells.append(f'{label} {estimate:+.3f} ({(estimate - tilt) / tilt * 100:+.0f} percent){flag}')
            print(f'    true tilt {tilt:.1f} degrees, edge displacement {displacement:.2f} pixels: '
                  + ' | '.join(cells))


def main():
    torch.set_num_threads(NUM_THREADS)
    print('Part A: three kernels and two scores at three true tilts')
    for kind, num_views, num_det_rows, num_det_channels, fan in CASES:
        sdd = None if fan is None else num_det_channels / 2 / math.tan(math.radians(fan / 2))
        model = make_model(kind, num_views, num_det_rows, num_det_channels, sdd)
        start = time.time()
        data = {tilt: fair_tilted_sinogram(kind, num_views, num_det_rows, num_det_channels, sdd,
                                           math.radians(tilt))
                for tilt in TRUE_TILTS_DEGREES}
        print(f'{kind}: {num_views} views, {num_det_rows} rows, {num_det_channels} channels, full '
              f'fan angle {fan} degrees (data generated in {time.time() - start:.0f} s)')
        offset = gc.estimate_det_channel_offset(model, data[0.0]).value
        print(f'  channel offset estimated on the untilted data: {offset:+.4f}')
        for tilt, sino in data.items():
            cells = []
            for label, interpolation, score_name in VARIANTS:
                for num_rows in BAND_ROWS:
                    estimate, notes, rows = estimate_tilt(model, sino, interpolation, score_name,
                                                          num_rows)
                    flag = '*' if notes else ''
                    cells.append(f'{label}/{rows[1] - rows[0]} rows {estimate:+.3f}{flag}')
            print(f'  true tilt {tilt:+.1f} degrees: ' + ' | '.join(cells))
    print('An asterisk marks a search whose coarse minimum sat at an edge of the bounds or whose '
          'coarse curve had more than one local minimum.')
    part_b()


if __name__ == '__main__':
    main()
