"""The conjugate-view estimators at a full-size detector, against LEAP on the same data.

This script runs on one GPU of the gautschi cluster, in the environment of the LEAP comparison,
which holds LEAP 1.26 and mbirtorch in one interpreter.  It answers three questions that the
laptop measurements could not.  Does the rotation estimate's bias stay small at a detector of
512 or more channels, where a rotation of a tenth of a degree displaces the edge pixel by
several pixels?  What does each estimator cost in wall time at that size?  And what do LEAP's
`find_centerCol` and `estimate_tilt` return on the same sinograms?

The geometry is the one the LEAP comparison used: circular cone beam, a flat detector, a source
to rotation axis distance of 1000 mm and a source to detector distance of 2000 mm, N views over a
full rotation, N detector columns, and a detector pixel of 2 * 256 / N mm, so the detector always
covers a 256 mm field of view.  The detector here has fewer rows than columns, because the
estimators use a band of rows around the central plane, and fewer rows keep the sinograms small.

The tilted sinograms are made at four times the detector resolution, rotated there, and binned by
four, as in the laptop experiment.  The results are appended to a JSON-lines file as each
measurement is taken, so a job cut short still leaves what it finished, and the measured tables
are transcribed to ``calibration_512_gautschi.md`` beside this file.

Run parameters are at the top of the file.  The batch file that submits it,
``calibration_512_gautschi.sbatch``, lives beside the LEAP comparison's environment on the
cluster, because this repository ignores batch files; its lines are transcribed in the record.
"""
import json
import math
import os
import sys
import time

os.environ.setdefault('MBIRTORCH_NUM_DEVICES', '1')

import numpy as np
import torch

import mbirtorch
from mbirtorch.preprocess import geometry_calibration as gc

# ── run parameters ────────────────────────────────────────────────────────────────────────────────
SIZES = (512, 1024)                 # detector columns and views; the rows are ROWS_FRACTION of this
ROWS_FRACTION = 1 / 8
OVERSAMPLING = 4                    # for the tilted data
SOURCE_ISO_MM = 1000.0
SOURCE_DETECTOR_MM = 2000.0
FIELD_OF_VIEW_MM = 256.0
TRUE_OFFSETS_PIXELS = (0.0, 1.3, -2.2, 7.5)        # channel offsets, in detector pixels
TRUE_ROTATIONS_DEGREES = (0.0, 0.05, 0.1, 0.3, 1.0)
NOISE_FRACTION_OF_MAX = 0.02
RESULTS = os.environ.get('CALIBRATION_RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'))
JSONL = os.path.join(RESULTS, 'calibration_512.jsonl')


def record(entry):
    """Append one measurement to the JSON-lines file and print it."""
    os.makedirs(RESULTS, exist_ok=True)
    with open(JSONL, 'a') as handle:
        handle.write(json.dumps(entry) + '\n')
    print(json.dumps(entry), flush=True)


def geometry(n):
    """The LEAP comparison's geometry at size n, with a shorter detector."""
    pixel = 2.0 * 256.0 / n
    return dict(num_views=n, num_det_rows=max(16, int(n * ROWS_FRACTION)), num_det_channels=n,
                pixel_mm=pixel)


def cone_model(n, det_channel_offset=0.0, oversampling=1):
    g = geometry(n)
    angles = np.linspace(0, 2 * np.pi, g['num_views'], endpoint=False)
    shape = (g['num_views'], g['num_det_rows'] * oversampling, g['num_det_channels'] * oversampling)
    model = mbirtorch.ConeBeamModel(shape, angles, source_detector_dist=SOURCE_DETECTOR_MM,
                                    source_iso_dist=SOURCE_ISO_MM)
    model.set_params(delta_det_channel=g['pixel_mm'] / oversampling,
                     delta_det_row=g['pixel_mm'] / oversampling)
    model.auto_set_recon_geometry()
    model.configure_devices(num_devices=1)
    model.set_params(no_warning=True, verbose=0, det_channel_offset=det_channel_offset)
    return model


def phantom_for(model):
    """The Shepp-Logan phantom at the model's recon shape."""
    return mbirtorch.generate_3d_shepp_logan_low_dynamic_range(model.get_params('recon_shape'))


def leap_model(n, det_channel_offset_mm=0.0):
    """A LEAP model of the same geometry.  LEAP's centerCol is the column of the ray through the
    rotation axis, which in mbirtorch's terms is the detector center plus the offset in pixels."""
    from leapctype import tomographicModels
    g = geometry(n)
    leapct = tomographicModels()
    leapct.set_gpu(0)
    phis = np.ascontiguousarray(np.linspace(0.0, 360.0, g['num_views'], endpoint=False), dtype=np.float32)
    center_col = 0.5 * (g['num_det_channels'] - 1) + det_channel_offset_mm / g['pixel_mm']
    leapct.set_conebeam(g['num_views'], g['num_det_rows'], g['num_det_channels'], g['pixel_mm'],
                        g['pixel_mm'], 0.5 * (g['num_det_rows'] - 1), center_col, phis,
                        SOURCE_ISO_MM, SOURCE_DETECTOR_MM)
    leapct.set_flatDetector()
    return leapct, center_col


def to_leap(sinogram):
    """mbirtorch's (view, row, channel) order to LEAP's, as the LEAP comparison established: LEAP's
    view j is mbirtorch's view (N/2 - j) mod N, with the channel axis reversed and the rows
    unchanged.  The map is its own inverse."""
    num_views = sinogram.shape[0]
    out = sinogram[:, :, ::-1][::-1]
    out = np.roll(out, num_views // 2 + 1, axis=0)
    return np.ascontiguousarray(out, dtype=np.float32)


def leap_estimates(n, sinogram, true_offset_mm):
    """LEAP's centerCol search and tilt estimate on the sinogram, returned in mbirtorch's terms:
    the channel offset in mm and the tilt in degrees.  LEAP's own error metric is recorded too."""
    g = geometry(n)
    leapct, center_col_at_zero = leap_model(n)
    data = to_leap(sinogram)
    start = time.perf_counter()
    metric = leapct.find_centerCol(data)
    offset_seconds = time.perf_counter() - start
    center_col = leapct.get_centerCol()
    # The channel reversal in to_leap mirrors the offset's sign.
    offset_mm = -(center_col - 0.5 * (g['num_det_channels'] - 1)) * g['pixel_mm']
    start = time.perf_counter()
    tilt_degrees = float(leapct.estimate_tilt(data))
    tilt_seconds = time.perf_counter() - start
    return dict(leap_offset_mm=float(offset_mm), leap_centerCol=float(center_col),
                leap_metric=float(metric), leap_offset_seconds=offset_seconds,
                leap_tilt_degrees=tilt_degrees, leap_tilt_seconds=tilt_seconds)


def offsets(n, phantom):
    """The channel offset at size n: mbirtorch's estimate and LEAP's, on clean and noisy data."""
    g = geometry(n)
    estimating = cone_model(n)
    for true_pixels in TRUE_OFFSETS_PIXELS:
        true_mm = true_pixels * g['pixel_mm']
        generating = cone_model(n, det_channel_offset=true_mm)
        sino = np.asarray(generating.forward_project(phantom), dtype=np.float32)
        noisy = sino + np.random.default_rng(0).normal(0, NOISE_FRACTION_OF_MAX * sino.max(), sino.shape).astype(np.float32)
        for label, data in (('clean', sino), ('noisy', noisy)):
            start = time.perf_counter()
            result = gc.estimate_det_channel_offset(estimating, data)
            seconds = time.perf_counter() - start
            entry = dict(kind='offset', N=n, pixel_mm=g['pixel_mm'], data=label,
                         true_offset_mm=true_mm, true_offset_pixels=true_pixels,
                         estimate_mm=result.value, error_pixels=(result.value - true_mm) / g['pixel_mm'],
                         first_pass_mm=result.reduction['first_pass'], seconds=seconds,
                         num_evaluations=int(result.candidates.size), notes=result.reduction['search_notes'])
            try:
                entry.update(leap_estimates(n, data, true_mm))
                entry['leap_error_pixels'] = (entry['leap_offset_mm'] - true_mm) / g['pixel_mm']
            except Exception as error:      # LEAP is optional: record the failure and go on
                entry['leap_error'] = repr(error)
            record(entry)


def rotations(n, phantom):
    """The rotation estimate at size n on fairly tilted data, and LEAP's tilt on the same data."""
    g = geometry(n)
    estimating = cone_model(n)
    fine = cone_model(n, oversampling=OVERSAMPLING)
    fine_phantom = phantom_for(fine)
    start = time.perf_counter()
    sino_fine = np.asarray(fine.forward_project(fine_phantom), dtype=np.float32)
    record(dict(kind='data', N=n, oversampling=OVERSAMPLING, seconds=time.perf_counter() - start,
                shape=list(sino_fine.shape)))
    del fine_phantom
    reduction = {'view_stride': 1, 'bin_factor': OVERSAMPLING,
                 'row_window': (0, sino_fine.shape[1]), 'full_sinogram_shape': sino_fine.shape,
                 'devices': [str(estimating.torch_device)]}
    for true_degrees in TRUE_ROTATIONS_DEGREES:
        sino = gc.reduce_sinogram(sino_fine, reduction, det_rotation=-math.radians(true_degrees))
        start = time.perf_counter()
        result = gc.estimate_det_rotation(estimating, sino)
        seconds = time.perf_counter() - start
        estimate_degrees = math.degrees(result.value)
        entry = dict(kind='rotation', N=n, true_degrees=true_degrees,
                     edge_displacement_pixels=math.radians(true_degrees) * n / 2,
                     estimate_degrees=estimate_degrees,
                     error_percent=(100.0 * (estimate_degrees - true_degrees) / true_degrees
                                    if true_degrees else None),
                     seconds=seconds, num_evaluations=int(result.candidates.size),
                     notes=result.reduction['search_notes'])
        try:
            leap = leap_estimates(n, sino, 0.0)
            entry.update(leap)
        except Exception as error:
            entry['leap_error'] = repr(error)
        record(entry)


def main():
    record(dict(kind='environment', torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none',
                mbirtorch=mbirtorch.__version__, argv=sys.argv))
    for n in SIZES:
        model = cone_model(n)
        phantom = phantom_for(model)
        start = time.perf_counter()
        model.forward_project(phantom)          # pays the compile before anything is timed
        record(dict(kind='warmup', N=n, seconds=time.perf_counter() - start))
        offsets(n, phantom)
        rotations(n, phantom)
    print('CALIBRATION_512 DONE', flush=True)


if __name__ == '__main__':
    main()
