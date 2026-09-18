"""How well does the conjugate-view estimator recover a known channel offset, and what does its
accuracy depend on?  The first pass, which pairs the views at the model's offset, is reported
beside the final estimate, so the effect of the second pass is visible across the range.

The estimator is ``estimate_det_channel_offset`` in ``mbirtorch/preprocess/geometry_calibration.py``.
This script measures its error on synthetic cone-beam data at a full fan angle of 20 degrees, over
a range of true offsets, with and without noise over several seeds, for two phantoms, for three
view strides, for three band heights, and with the conjugate-ray sign deliberately flipped.  The
measured tables and the conclusions are in ``conjugate_offset_recovery.md`` beside this file.

Run from the mbirtorch repository root with the mbirtorch conda environment.  Everything runs on
the CPU with torch.compile off, and the whole script takes a few minutes.
"""
import math
import os

os.environ['MBIRTORCH_NUM_DEVICES'] = '1'

import numpy as np
import torch

import mbirtorch
from mbirtorch.preprocess import geometry_calibration as gc

# ── run parameters ────────────────────────────────────────────────────────────────────────────────
NUM_THREADS = 6
NUM_VIEWS, NUM_DET_ROWS, NUM_DET_CHANNELS = 128, 16, 64
FULL_FAN_DEGREES = 20.0
TRUE_OFFSETS = (-3.5, -2.2, -1.0, 0.0, 1.3, 2.6, 3.5)       # channels; the pitch is 1
NOISE_FRACTION_OF_MAX = 0.02
NOISE_SEEDS = range(5)
VIEW_STRIDES = (1, 2, 4)
BAND_ROWS = (5, 9, 16)
SIGN_CHECK_OFFSET = 1.3


def cone_model(det_channel_offset):
    """The test model: cone beam at the fan angle above, source-to-iso half the source-to-detector
    distance, on the CPU with compilation off."""
    angles = np.linspace(0, 2 * np.pi, NUM_VIEWS, endpoint=False)
    sdd = NUM_DET_CHANNELS / 2 / math.tan(math.radians(FULL_FAN_DEGREES / 2))
    model = mbirtorch.ConeBeamModel((NUM_VIEWS, NUM_DET_ROWS, NUM_DET_CHANNELS), angles,
                                    source_detector_dist=sdd, source_iso_dist=sdd / 2,
                                    compile_mode='off')
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0, det_channel_offset=det_channel_offset)
    return model


def phantoms(recon_shape):
    """The Shepp-Logan phantom and a rod of radius 3 voxels away from the axis."""
    shepp = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
    rows, cols, _ = np.indices(recon_shape)
    rod = (((rows - 20) ** 2 + (cols - 12) ** 2) <= 9).astype(np.float32)
    return [('shepp-logan', shepp), ('off-axis rod', rod)]


def with_stride(model, stride):
    """The estimator's default reduction record with the view stride replaced."""
    record = gc._ConjugatePairs._default_reduction(model, None)
    record['view_stride'] = stride
    record['sinogram_shape'] = (NUM_VIEWS // stride,) + tuple(record['sinogram_shape'][1:])
    return record


def flipped_sign_ratio(model, sino):
    """The score minimum with the conjugate-ray sign flipped, divided by the score minimum with
    the module's sign, and the flipped estimate.  The flip replaces beta + pi - 2 gamma with
    beta + pi + 2 gamma in the partner angles, in both passes of the estimate."""
    right = gc.estimate_det_channel_offset(model, sino)
    original_init = gc._ConjugatePairs.__init__

    def flipped_init(self, ct_model, reduction=None, num_rows=None, pairing_offset=None):
        original_init(self, ct_model, reduction, num_rows, pairing_offset)
        angles = gc._view_angles(ct_model)
        center = (self.num_channels - 1) / 2.0
        u = (np.arange(self.num_channels) - center) * self.delta + self.pairing_offset
        gamma = np.arctan(u / ct_model.get_params('source_detector_dist'))
        target = angles[self.reference_indices][:, None] + np.pi + 2.0 * gamma[None, :]
        low, high, self.partner_weight = self._partners(angles, target)
        self.partner_indices = np.unique(np.concatenate([low.ravel(), high.ravel()]))
        self.partner_low = np.searchsorted(self.partner_indices, low)
        self.partner_high = np.searchsorted(self.partner_indices, high)

    gc._ConjugatePairs.__init__ = flipped_init
    try:
        wrong = gc.estimate_det_channel_offset(model, sino)
    finally:
        gc._ConjugatePairs.__init__ = original_init
    return wrong.score / right.score, wrong.value, right.value


def main():
    torch.set_num_threads(NUM_THREADS)
    estimating = cone_model(0.0)
    for name, phantom in phantoms(estimating.get_params('recon_shape')):
        print(f'phantom {name}: {NUM_VIEWS} views, {NUM_DET_ROWS} rows, {NUM_DET_CHANNELS} channels, '
              f'full fan angle {FULL_FAN_DEGREES:.0f} degrees')
        for true in TRUE_OFFSETS:
            sino = cone_model(true).forward_project(phantom).astype(np.float32)
            result = gc.estimate_det_channel_offset(estimating, sino)
            errors = []
            for seed in NOISE_SEEDS:
                noise = np.random.default_rng(seed).normal(0, NOISE_FRACTION_OF_MAX * sino.max(), sino.shape)
                noisy = (sino + noise).astype(np.float32)
                errors.append(gc.estimate_det_channel_offset(estimating, noisy).value - true)
            notes = f'  notes {result.reduction["search_notes"]}' if result.reduction['search_notes'] else ''
            print(f'  true {true:+.1f}: first pass error {result.reduction["first_pass"] - true:+.4f} | '
                  f'clean error {result.value - true:+.4f} | noisy mean error '
                  f'{np.mean(errors):+.4f}, standard deviation {np.std(errors):.4f} over '
                  f'{len(errors)} seeds{notes}')
        sino = cone_model(SIGN_CHECK_OFFSET).forward_project(phantom).astype(np.float32)
        cells = [f'stride {stride}: {gc.estimate_det_channel_offset(estimating, sino, reduction=with_stride(estimating, stride)).value - SIGN_CHECK_OFFSET:+.4f}'
                 for stride in VIEW_STRIDES]
        print(f'  true {SIGN_CHECK_OFFSET:+.1f}, error by view stride: ' + ' | '.join(cells))
        cells = [f'{rows} rows: {gc.estimate_det_channel_offset(estimating, sino, num_rows=rows).value - SIGN_CHECK_OFFSET:+.4f}'
                 for rows in BAND_ROWS]
        print(f'  true {SIGN_CHECK_OFFSET:+.1f}, error by band height: ' + ' | '.join(cells))
        ratio, wrong, right = flipped_sign_ratio(estimating, sino)
        print(f'  true {SIGN_CHECK_OFFSET:+.1f}, conjugate-ray sign flipped: score minimum ratio '
              f'{ratio:.2f}, estimate {wrong:+.4f} against {right:+.4f} with the module\'s sign')
        result = gc.estimate_det_channel_offset(estimating, sino)
        print(f'  true {SIGN_CHECK_OFFSET:+.1f}, first pass {result.reduction["first_pass"]:+.4f}, '
              f'second pass {result.value:+.4f}')


if __name__ == '__main__':
    main()
