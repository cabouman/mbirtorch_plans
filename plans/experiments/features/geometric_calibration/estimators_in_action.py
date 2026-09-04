"""Pictures of the conjugate-view estimators at work on synthetic cone-beam data.

The script makes a scan with a known channel offset and a known detector rotation, then draws
what a user would see: the score curve of the offset search, the difference between a view and
its opposite at the wrong offset and at the estimate, the score curve of the rotation search,
and a parameter sweep of reconstructed slices.  The figures are written as PNG files beside the
results directory.  Run from the mbirtorch repository root with the mbirtorch conda environment;
the geometry is small so the run takes under a minute on a CPU.
"""
import math
import os

os.environ['MBIRTORCH_NUM_DEVICES'] = '1'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import mbirtorch
from mbirtorch.preprocess import geometry_calibration as gc
from mbirtorch.preprocess.utilities import correct_det_rotation

# ── run parameters ────────────────────────────────────────────────────────────────────────────────
NUM_VIEWS, NUM_DET_ROWS, NUM_DET_CHANNELS = 256, 32, 128
FULL_FAN_DEGREES = 20.0
TRUE_OFFSET = 2.3               # channels; the pitch is 1
TRUE_ROTATION_DEGREES = 2.0
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')


def cone_model(det_channel_offset):
    angles = np.linspace(0, 2 * np.pi, NUM_VIEWS, endpoint=False)
    sdd = NUM_DET_CHANNELS / 2 / math.tan(math.radians(FULL_FAN_DEGREES / 2))
    model = mbirtorch.ConeBeamModel((NUM_VIEWS, NUM_DET_ROWS, NUM_DET_CHANNELS), angles,
                                    source_detector_dist=sdd, source_iso_dist=sdd / 2, compile_mode='off')
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0, det_channel_offset=det_channel_offset)
    return model


def main():
    torch.set_num_threads(6)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generating = cone_model(TRUE_OFFSET)
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(generating.get_params('recon_shape'))
    sino = np.asarray(generating.forward_project(phantom), dtype=np.float32)
    # The measured data also carry a detector rotation.
    sino = correct_det_rotation(sino, -math.radians(TRUE_ROTATION_DEGREES))
    estimating = cone_model(0.0)

    # 1. The offset search and its score curve.
    offset = gc.estimate_det_channel_offset(estimating, sino, det_rotation=0.0)
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(offset.candidates, offset.scores, 'o-', markersize=3)
    axis.axvline(TRUE_OFFSET, color='gray', linestyle='--', label=f'true offset {TRUE_OFFSET}')
    axis.axvline(offset.value, color='red', linestyle=':', label=f'estimate {offset.value:.3f}')
    axis.set_xlabel('candidate det_channel_offset (channels)')
    axis.set_ylabel('conjugate-view score')
    axis.set_title('Offset search: every evaluation, before the rotation is corrected')
    axis.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'offset_score_curve.png'), dpi=120)

    # 2. The difference between each view and its opposite, at a wrong offset and at the estimate.
    wrong = gc.conjugate_difference(estimating, sino, det_channel_offset=0.0)
    right = gc.conjugate_difference(estimating, sino, det_channel_offset=offset.value)
    scale = np.abs(wrong).max()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for axis, image, title in ((axes[0], wrong, 'offset 0 (wrong)'), (axes[1], right, f'offset {offset.value:.2f} (estimate)')):
        # One view pair's difference over the row band, channels along the horizontal axis.
        axis.imshow(image[0], cmap='gray', vmin=-scale, vmax=scale, aspect='auto')
        axis.set_title(f'view minus opposite view, {title}')
        axis.set_xlabel('channel')
        axis.set_ylabel('row in band')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'conjugate_difference.png'), dpi=120)

    # 3. The rotation search at the estimated offset.
    rotation = gc.estimate_det_rotation(estimating, sino, det_channel_offset=offset.value)
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(np.degrees(rotation.candidates), rotation.scores, 'o-', markersize=3)
    axis.axvline(TRUE_ROTATION_DEGREES, color='gray', linestyle='--', label=f'true rotation {TRUE_ROTATION_DEGREES} degrees')
    axis.axvline(math.degrees(rotation.value), color='red', linestyle=':', label=f'estimate {math.degrees(rotation.value):.3f} degrees')
    axis.set_xlabel('candidate det_rotation (degrees)')
    axis.set_ylabel('conjugate-view score')
    axis.set_title('Rotation search at the estimated offset')
    axis.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'rotation_score_curve.png'), dpi=120)

    # 4. A second offset pass with the rotation corrected, and the sweep a user would look at.
    corrected = correct_det_rotation(sino, rotation.value)
    offset2 = gc.estimate_det_channel_offset(estimating, corrected)
    values = np.array([-2.0, 0.0, offset2.value, 4.0])
    stack = gc.parameter_sweep(estimating, corrected, 'det_channel_offset', values)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for axis, value, k in zip(axes, values, range(4)):
        axis.imshow(stack[:, :, k], cmap='gray', vmin=0, vmax=np.percentile(stack, 99.5))
        axis.set_title(f'det_channel_offset = {value:.2f}' + (' (estimate)' if k == 2 else ''))
        axis.axis('off')
    fig.suptitle('parameter_sweep: the central slice reconstructed at four candidate offsets')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'parameter_sweep.png'), dpi=120)

    print(f'true offset {TRUE_OFFSET}: first estimate {offset.value:.4f} (rotation uncorrected), '
          f'after the rotation correction {offset2.value:.4f}')
    print(f'true rotation {TRUE_ROTATION_DEGREES} degrees: estimate {math.degrees(rotation.value):.4f}')
    print('figures in', OUTPUT_DIR)


if __name__ == '__main__':
    main()
