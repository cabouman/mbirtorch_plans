"""How well does the direct-reconstruction residual separate the two rotation directions?

This script produced the numbers behind the design of ``check_rotation_direction`` in
``mbirtorch/preprocess/geometry_calibration.py`` (Increment 1 of the geometric calibration plan).
Part A varies the slab thickness of the reduced problem at one small cone-beam geometry.  Part B
keeps the whole axial extent and varies the fraction of central detector rows the residual is
taken over, at three geometries, with and without noise.  The measured tables and the conclusions
are in ``direction_score_contrast.md`` beside this file.

The score for one model and one reduced sinogram is the mean squared difference between the
high-pass filtered sinogram and the high-pass filtered forward projection of the model's direct
reconstruction, divided by the mean squared filtered sinogram.  The ratio reported is the score
with every view angle negated divided by the score with the angles as given, so a ratio well
above 1 means the check separates the two directions.

Run from the mbirtorch repository root with the mbirtorch conda environment.  Everything runs on
the CPU with torch.compile off, which is what the reduced problem inherits from the models here.
"""
import os
import time

os.environ['MBIRTORCH_NUM_DEVICES'] = '1'

import numpy as np
import torch

import mbirtorch
from mbirtorch.preprocess import geometry_calibration as gc
from mbirtorch.preprocess.utilities import sino_high_pass_filtering

# ── run parameters ────────────────────────────────────────────────────────────────────────────────
NUM_THREADS = 6
# Part A: one geometry, (view_stride, bin_factor, num_slab_slices) per row.  32 slices is the whole
# volume at bin 1, and 16 slices is the whole volume at bin 2.
PART_A_GEOMETRY = dict(num_views=128, num_det_rows=32, num_det_channels=64, sdd_over_channels=4)
PART_A_CASES = [(2, 1, 4), (2, 1, 8), (2, 1, 16), (2, 1, 32), (2, 2, 4), (2, 2, 8), (2, 2, 16),
                (4, 2, 8), (4, 2, 12), (4, 2, 16), (1, 1, 8)]
# Part B: three geometries, whole axial extent, (view_stride, bin_factor) per row, and the fraction
# of central rows scored.  The noisy case adds Gaussian noise at 2 percent of the sinogram maximum.
PART_B_GEOMETRIES = [
    ('small', dict(num_views=128, num_det_rows=32, num_det_channels=64, sdd_over_channels=4)),
    ('medium', dict(num_views=256, num_det_rows=64, num_det_channels=128, sdd_over_channels=4)),
    ('medium narrow fan', dict(num_views=256, num_det_rows=64, num_det_channels=128,
                               sdd_over_channels=8)),
]
PART_B_CASES = [(4, 2), (2, 2), (4, 1)]
PART_B_FRACTIONS = (1.0, 2 / 3, 1 / 2, 1 / 3, 1 / 5)
NOISE_FRACTION_OF_MAX = 0.02
SEED = 0


def make_cone(num_views, num_det_rows, num_det_channels, sdd_over_channels):
    """A cone-beam model at the given size, on the CPU with compilation off, and its Shepp-Logan
    sinogram.  The source-to-iso distance is half the source-to-detector distance."""
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
    sdd = sdd_over_channels * num_det_channels
    model = mbirtorch.ConeBeamModel((num_views, num_det_rows, num_det_channels), angles,
                                    source_detector_dist=sdd, source_iso_dist=sdd / 2,
                                    compile_mode='off')
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0)
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(model.get_params('recon_shape'))
    sino = model.forward_project(phantom)
    return model, sino


def negated_copy(model):
    """The same model with every view angle negated."""
    required, _, _ = model.get_all_params()
    copy = mbirtorch.copy_ct_model(model, new_angles=-np.asarray(required['angles']),
                                   new_helical_z_shifts=np.asarray(required['helical_z_shifts']))
    copy.compile_mode = 'off'
    return copy


def filtered_pair(model, sino_reduced):
    """The high-pass filtered reduced sinogram and the high-pass filtered forward projection of the
    model's direct reconstruction of it."""
    recon = model.recon_direct(sino_reduced)
    projection = model.forward_project(recon)
    return sino_high_pass_filtering(sino_reduced), sino_high_pass_filtering(projection)


def residual(filtered_sino, filtered_projection, row_fraction=1.0):
    """The normalized residual over the central ``row_fraction`` of the rows."""
    num_rows = filtered_sino.shape[1]
    keep = max(1, int(round(num_rows * row_fraction)))
    lo = (num_rows - keep) // 2
    fs, fp = filtered_sino[:, lo:lo + keep], filtered_projection[:, lo:lo + keep]
    return float(np.mean((fs - fp) ** 2) / np.mean(fs ** 2))


def part_a():
    print('Part A: slab thickness at the small geometry')
    model, sino = make_cone(**PART_A_GEOMETRY)
    model_neg = negated_copy(model)
    print('  recon shape', model.get_params('recon_shape'))
    for stride, bin_factor, num_slab in PART_A_CASES:
        start = time.time()
        kwargs = dict(view_stride=stride, bin_factor=bin_factor, num_slab_slices=num_slab)
        reduced, reduction = gc.build_reduced_problem(model, **kwargs)
        reduced_neg, _ = gc.build_reduced_problem(model_neg, **kwargs)
        sino_reduced = gc.reduce_sinogram(sino, reduction)
        pairs = [filtered_pair(m, sino_reduced) for m in (reduced, reduced_neg)]
        cells = []
        for fraction, label in ((1.0, 'all rows'), (1 / 2, 'central half'), (1 / 3, 'central third')):
            scores = [residual(*p, fraction) for p in pairs]
            cells.append(f'{label} {scores[0]:.4f}/{scores[1]:.4f} ratio {scores[1] / scores[0]:.2f}')
        print(f'  stride {stride} bin {bin_factor} slab {num_slab:2d} rows {reduction["row_window"]}: '
              + ' | '.join(cells) + f' ({time.time() - start:.1f} s)')


def part_b():
    print('Part B: row fraction with the whole axial extent')
    rng = np.random.default_rng(SEED)
    for label, geometry in PART_B_GEOMETRIES:
        model, sino = make_cone(**geometry)
        model_neg = negated_copy(model)
        noisy = sino + rng.normal(0, NOISE_FRACTION_OF_MAX * sino.max(), sino.shape).astype(np.float32)
        sdd = geometry['sdd_over_channels'] * geometry['num_det_channels']
        half_fan = np.degrees(np.arctan(geometry['num_det_channels'] / 2 / sdd))
        half_cone = np.degrees(np.arctan(geometry['num_det_rows'] / 2 / sdd))
        print(f'  {label}: recon {model.get_params("recon_shape")}, half fan angle {half_fan:.1f} '
              f'degrees, half cone angle {half_cone:.1f} degrees')
        for stride, bin_factor in PART_B_CASES:
            kwargs = dict(view_stride=stride, bin_factor=bin_factor, num_slab_slices=None)
            reduced, reduction = gc.build_reduced_problem(model, **kwargs)
            reduced_neg, _ = gc.build_reduced_problem(model_neg, **kwargs)
            for data, tag in ((sino, 'clean'), (noisy, 'noisy')):
                sino_reduced = gc.reduce_sinogram(data, reduction)
                pairs = [filtered_pair(m, sino_reduced) for m in (reduced, reduced_neg)]
                cells = []
                for fraction in PART_B_FRACTIONS:
                    scores = [residual(*p, fraction) for p in pairs]
                    cells.append(f'{fraction:.2f}: {scores[0]:.4f}/{scores[1]:.4f} = '
                                 f'{scores[1] / scores[0]:.2f}')
                print(f'    stride {stride} bin {bin_factor} {tag}: ' + '  '.join(cells))


if __name__ == '__main__':
    torch.set_num_threads(NUM_THREADS)
    part_a()
    part_b()
