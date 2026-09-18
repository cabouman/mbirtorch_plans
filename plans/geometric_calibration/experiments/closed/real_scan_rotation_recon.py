"""Direct reconstructions of the NSI scans at four detector rotations, for a comparison by eye.

The three earlier jobs left the zero point of the detector rotation estimate open on the NSI scans
without metal.  The estimator reads 0.044 degrees there, the vendor's geometry report gives 0.167
degrees, and on the scan with metal the estimator reads 0.19 degrees.  Greg asked for direct
reconstructions of the scans without metal at 0.044 and at 0.167 degrees, because a direct
reconstruction has no prior that could absorb a geometry error, and because the two candidates are
told apart by the slices far from the central plane.  At 600 rows from that plane the two rotations
move the center of rotation by 1.3 channels relative to each other, which a direct reconstruction
shows as doubled edges and rings.

The job uses ``parameter_sweep`` with the parameter ``det_rotation``.  For one slice of the volume it
crops the detector to the rows that slice needs, rotates that band of every view with the same
bilinear kernel that ``apply_calibration`` uses, and reconstructs the slice directly.  It does that
at four rotations: none, 0.044 degrees, 0.167 degrees, and 0.19 degrees.  Five slices are
reconstructed per scan, from near the bottom of the volume to near the top, and the middle slice is
the control, because a detector rotation moves nothing on the central plane.  Two sharpness measures
are recorded per slice and rotation, and every slice is saved as an array and drawn in a figure with
one panel per rotation and one panel for the difference between the 0.044 and 0.167 degree slices.

The scan with metal runs as a second case.  There the estimator and the vendor nearly agree, so its
slices should change little between 0.167 and 0.19 degrees and more between those and 0.044.

Results are appended to a JSON-lines file as each slice is done.  The measured tables are transcribed
to ``real_scan_rotation_recon.md`` beside this file.  Run parameters are at the top of the file, and
the batch file ``real_scan_rotation_recon.sbatch`` beside it submits the job.  That batch file puts
this directory on PYTHONPATH, so the helpers of the earlier jobs are imported rather than copied.
"""
import math
import os
import resource
import sys
import time
import traceback
from gc import collect as collect_garbage    # 'gc' below is the calibration module

os.environ.setdefault('MBIRTORCH_NUM_DEVICES', '1')

import numpy as np
import torch

import mbirtorch
from mbirtorch.preprocess import geometry_calibration as gc

import real_scan_validation as first_job
import real_scan_followup as second_job
from real_scan_validation import git_commit, record, sharpness
from real_scan_followup import load_scan

torch.set_num_threads(14)

# ── run parameters ────────────────────────────────────────────────────────────────────────────────
DATASETS = (
    dict(name='nsi_no_metal', reader='nsi',
         path='/depot/bouman/data/Lilly/demo_nsi_vert_no_metal_all_views.tgz'),
    dict(name='nsi_metal', reader='nsi',
         path='/depot/bouman/data/Lilly/demo_nsi_vert_metal_all_views.tgz'),
)
# The rotations applied before each reconstruction, in degrees: none, the estimator's value on the
# scans without metal, the vendor's tilt, and the estimator's value on the scan with metal.
ROTATIONS_DEGREES = (0.0, 0.044, 0.167, 0.19)
# The slices reconstructed, as fractions of the way through the volume's slices.
SLICE_FRACTIONS = (0.1, 0.25, 0.5, 0.75, 0.9)
# The two rotations whose slices are differenced in the figure.
DIFFERENCE_PAIR_DEGREES = (0.044, 0.167)

RESULTS = first_job.RESULTS
DATA = first_job.DATA
JSONL = os.path.join(RESULTS, 'real_scan_rotation_recon.jsonl')
first_job.JSONL = JSONL


def laplacian_variance(image):
    """The variance of the four-neighbor Laplacian of a slice, divided by the slice's mean square.

    A second sharpness measure, sensitive to fine edges.  Doubled edges and rings from a wrong
    geometry raise the Laplacian's variance as well as lower it, so the two measures are read
    together with the figures rather than alone.
    """
    image = np.asarray(image, dtype=np.float64)
    energy = float(np.mean(image ** 2))
    if not energy > 0.0:
        return float('nan')
    lap = (-4.0 * image[1:-1, 1:-1] + image[:-2, 1:-1] + image[2:, 1:-1]
           + image[1:-1, :-2] + image[1:-1, 2:])
    return float(np.var(lap) / energy)


def save_figure(name, slice_index, z_alu, stack, measures):
    """Draw one panel per rotation and one for the difference between the two candidates."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    k_a = ROTATIONS_DEGREES.index(DIFFERENCE_PAIR_DEGREES[0])
    k_b = ROTATIONS_DEGREES.index(DIFFERENCE_PAIR_DEGREES[1])
    low, high = np.nanpercentile(stack, [1.0, 99.5])
    num_panels = len(ROTATIONS_DEGREES) + 1
    figure, axes = plt.subplots(1, num_panels, figsize=(5.0 * num_panels, 5.4))
    for k, axis in enumerate(axes[:-1]):
        axis.imshow(stack[:, :, k], cmap='gray', vmin=low, vmax=high)
        axis.set_title(f'{ROTATIONS_DEGREES[k]:.3f} deg\nsharpness {measures["sharpness"][k]:.4g}, '
                       f'laplacian {measures["laplacian"][k]:.4g}', fontsize=9)
        axis.set_xticks([]); axis.set_yticks([])
    difference = stack[:, :, k_a] - stack[:, :, k_b]
    scale = float(np.nanpercentile(np.abs(difference), 99.5))
    axes[-1].imshow(difference, cmap='gray', vmin=-scale, vmax=scale)
    axes[-1].set_title(f'{DIFFERENCE_PAIR_DEGREES[0]} minus {DIFFERENCE_PAIR_DEGREES[1]} deg', fontsize=9)
    axes[-1].set_xticks([]); axes[-1].set_yticks([])
    figure.suptitle(f'{name}: slice {slice_index}, z = {z_alu:+.1f} mm, direct reconstruction at four '
                    'detector rotations')
    figure.tight_layout()
    figure.savefig(os.path.join(RESULTS, f'{name}_slice_{slice_index:04d}_rotations.png'), dpi=110)
    plt.close(figure)


def run_dataset(spec):
    """Load one scan and reconstruct the chosen slices at the four rotations."""
    name = spec['name']
    dataset_start = time.perf_counter()
    scan = None
    try:
        scan = load_scan(spec)
        if scan is None:
            return
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        num_slices = int(scan.ct_model.get_params('recon_shape')[2])
        values = [math.radians(r) for r in ROTATIONS_DEGREES]
        for fraction in SLICE_FRACTIONS:
            slice_index = int(round(fraction * (num_slices - 1)))
            try:
                z_alu = float(scan.ct_model.recon_slice_z(slice_index))
                # The detector row the slice's center reaches, for a point on the rotation axis.
                magnification = float(scan.ct_model.get_params('source_detector_dist')
                                      / scan.ct_model.get_params('source_iso_dist'))
                row = scan.central_row + z_alu * magnification / scan.delta_det_row
                start = time.perf_counter()
                stack = gc.parameter_sweep(scan.ct_model, scan.sino, 'det_rotation', values,
                                           slice_index=slice_index)
                seconds = time.perf_counter() - start
                measures = dict(sharpness=[sharpness(stack[:, :, k]) for k in range(stack.shape[2])],
                                laplacian=[laplacian_variance(stack[:, :, k]) for k in range(stack.shape[2])])
                np.savez(os.path.join(RESULTS, f'{name}_slice_{slice_index:04d}_rotations.npz'),
                         stack=stack, rotations_degrees=np.asarray(ROTATIONS_DEGREES),
                         slice_index=slice_index, z_alu=z_alu)
                save_figure(name, slice_index, z_alu, stack, measures)
                k_a = ROTATIONS_DEGREES.index(DIFFERENCE_PAIR_DEGREES[0])
                k_b = ROTATIONS_DEGREES.index(DIFFERENCE_PAIR_DEGREES[1])
                difference = stack[:, :, k_a] - stack[:, :, k_b]
                record(name, 'slice', seconds, slice_index=slice_index, num_slices=num_slices,
                       z_alu=z_alu, detector_row_on_axis=row,
                       rows_from_central_plane=row - scan.central_row,
                       rotations_degrees=list(ROTATIONS_DEGREES), slice_shape=list(stack.shape[:2]),
                       sharpness=measures['sharpness'], laplacian_variance=measures['laplacian'],
                       sharpest=float(ROTATIONS_DEGREES[int(np.nanargmax(measures['sharpness']))]),
                       laplacian_largest=float(ROTATIONS_DEGREES[int(np.nanargmax(measures['laplacian']))]),
                       difference_rms=float(np.sqrt(np.mean(difference.astype(np.float64) ** 2))),
                       slice_rms=float(np.sqrt(np.mean(stack[:, :, k_a].astype(np.float64) ** 2))))
                del stack
            except Exception:
                record(name, 'slice', 0.0, slice_index=slice_index, traceback=traceback.format_exc())
    except Exception:
        record(name, 'error', time.perf_counter() - dataset_start, traceback=traceback.format_exc())
    finally:
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
           rotations_degrees=list(ROTATIONS_DEGREES), slice_fractions=list(SLICE_FRACTIONS),
           argv=sys.argv)
    for spec in DATASETS:
        run_dataset(spec)
    print('REAL_SCAN_ROTATION_RECON DONE', flush=True)


if __name__ == '__main__':
    main()
