"""The fine sweep of ``recon_sweep_fine.py`` with a Fourier-shift rotation in place of the bilinear one.

The fine sweep found that each slice's score curve has its minimum where the candidate rotation
displaces that slice's detector rows by a whole number of pixels.  The bilinear resampling that
applies a candidate rotation smooths the data by an amount that depends on the fractional part of the
displacement: nothing at a whole-pixel displacement, most at a half pixel.  Smoothing lowers the
gradient energy the score reads, so the score prefers whole-pixel displacements for reasons that have
nothing to do with the geometry, and a blur of the reconstruction does not remove the preference.

This job repeats the sweep with one change.  The candidate rotation is applied by two shears, each a
shift of whole rows or whole columns by a non-integer amount, and each shift is done in the Fourier
domain.  A Fourier shift multiplies each frequency by a phase of unit magnitude, so it moves the data
without smoothing it at any frequency, and the amount of smoothing no longer depends on the
displacement.  For the small angles here the two shears reproduce the bilinear kernel's rotation to
within a few thousandths of a pixel; the difference is the second-order term of the rotation matrix.

Everything else is the job it wraps: the scans, the slices, the candidates, the even-view and odd-view
split, the scores, the analysis, the figures, and the records.  If the per-slice minima move off the
whole-pixel angles and agree with each other, the disagreement the first job saw was the resampling.
If they stay where they were, it was not.

A Fourier shift treats the data as periodic, so the block of rows handed to the kernel is widened by
extra rows on each side and mirrored at its row edges before the shift along the rows, and the extra
rows are cropped afterward.  Along the channels the two ends of every row are air, so the periodic
wrap joins two near-zero values and needs no padding.  Run parameters are at the top; the batch file
beside this script sets the results directory.
"""
import math
import os
import sys

import numpy as np
import torch

# The job this script wraps sits beside it, and the loader modules sit in its 'closed' subdirectory
# locally and flat on the cluster.
_HERE = os.path.dirname(os.path.abspath(__file__))
for path in (_HERE, os.path.join(_HERE, 'closed')):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

import recon_sweep_fine as job
from mbirtorch.preprocess import geometry_calibration as gc

# ── run parameters ────────────────────────────────────────────────────────────────────────────────
# Rows added on each side of the block the kernel receives, beyond the rows the rotation itself
# needs.  The shift along the rows wraps the block's first and last rows into each other, and the
# mirrored padding keeps that wrap, and its ringing, away from the rows that are kept.
EXTRA_ROW_MARGIN = 16
# Rows of mirrored padding applied inside the kernel before the shift along the rows.
REFLECT_ROWS = 16

_bilinear_row_margin = gc._rotation_row_margin


def fourier_shift(block, shifts, dim):
    """Shift ``block`` along ``dim`` by a non-integer number of samples that varies along the other
    in-plane axis, in the Fourier domain.

    Args:
        block (tensor): ``(num_views, num_rows, num_channels)``.
        shifts (tensor): the shift of each line along ``dim``, one value per index of the other
            in-plane axis.  A positive shift moves content toward higher indices.
        dim (int): 1 to shift each column along the rows, 2 to shift each row along the channels.
    """
    length = block.shape[dim]
    spectrum = torch.fft.rfft(block, dim=dim)
    frequencies = torch.fft.rfftfreq(length, device=block.device, dtype=torch.float64)
    # The phase is exp(-2 pi i f s) for a shift s; the two axes broadcast it across the block.
    if dim == 2:
        angle = -2.0 * math.pi * frequencies[None, :] * shifts.to(torch.float64)[:, None]   # (rows, freq)
        phase = torch.polar(torch.ones_like(angle), angle)[None, :, :]
    else:
        angle = -2.0 * math.pi * frequencies[:, None] * shifts.to(torch.float64)[None, :]   # (freq, cols)
        phase = torch.polar(torch.ones_like(angle), angle)[None, :, :]
    shifted = torch.fft.irfft(spectrum * phase.to(spectrum.dtype), n=length, dim=dim)
    return shifted.to(block.dtype)


def fourier_rotation_kernel(sino_batch, det_rotation, center=None):
    """Rotate each view's (row, channel) plane by ``det_rotation`` radians with two Fourier shears.

    The output pixel at row ``i`` and channel ``j`` takes the input at row ``i + a (j - c_col)`` and
    channel ``j - a (i - c_row)``, which is what the bilinear kernel samples up to the second-order
    terms of the rotation.  The first shear moves each row along the channels by ``a (i - c_row)``,
    and the second moves each column along the rows by ``-a (j - c_col)``.  The column shear is done
    on a block mirrored at its row edges, and the mirrored rows are cropped afterward.
    """
    num_views, num_rows, num_cols = sino_batch.shape
    device = sino_batch.device
    a = float(det_rotation)
    if center is None:
        center_row, center_col = (num_rows - 1) / 2.0, (num_cols - 1) / 2.0
    else:
        center_row, center_col = (float(c) for c in center)
    if a == 0.0:
        return sino_batch.clone()
    rows = torch.arange(num_rows, device=device, dtype=torch.float64)
    cols = torch.arange(num_cols, device=device, dtype=torch.float64)
    sheared = fourier_shift(sino_batch, a * (rows - center_row), dim=2)
    pad = min(REFLECT_ROWS, num_rows - 1)
    padded = torch.cat([sheared[:, 1:pad + 1].flip(1), sheared, sheared[:, -pad - 1:-1].flip(1)], dim=1)
    rotated = fourier_shift(padded, -a * (cols - center_col), dim=1)
    return rotated[:, pad:pad + num_rows]


def wide_row_margin(det_rotation, max_row_distance, num_channels):
    """The rows the bilinear kernel would need, plus the extra rows the Fourier shift needs."""
    return _bilinear_row_margin(det_rotation, max_row_distance, num_channels) + EXTRA_ROW_MARGIN


def install():
    """Replace the module's rotation kernel and row margin with the Fourier versions."""
    gc._rotation_kernel = fourier_rotation_kernel
    gc._rotation_row_margin = wide_row_margin


def check_kernel(device='cpu'):
    """Compare the two kernels on synthetic images and report what the Fourier kernel changes.

    Three checks run on a block whose rows continue outside it, as a band cut from a taller
    detector does, with the rotation center 700 rows away.  A smooth blob is rotated by both kernels
    and by the exact formula, so the interpolation error of each kernel is measured.  A rotation
    followed by its reverse must return the block.  A block of white noise is rotated over a range
    of angles, and the noise power that survives is recorded per angle: the bilinear kernel's
    smoothing changes it with the fractional part of the displacement, and the Fourier kernel's
    must not.  Returns a dict of the results, which ``main`` records.
    """
    from mbirtorch.preprocess.utilities import _rotation_kernel as bilinear
    generator = torch.Generator(device='cpu').manual_seed(0)
    num_rows, num_cols, center = 96, 512, (700.0, 255.5)
    rows = torch.arange(num_rows, dtype=torch.float32, device=device)[:, None]
    cols = torch.arange(num_cols, dtype=torch.float32, device=device)[None, :]
    blob = torch.exp(-((rows - 40.0) ** 2 + (cols - 300.0) ** 2) / (2 * 12.0 ** 2))[None]
    keep = slice(20, num_rows - 20)          # rows away from the block's edges
    results = dict(angles_degrees=[], bilinear_error=[], fourier_error=[], kernel_difference=[])
    for degrees in (0.05, 0.13, 0.24):
        a = math.radians(degrees)
        source_row = rows + a * (cols - center[1])
        source_col = cols - a * (rows - center[0])
        exact = torch.exp(-((source_row - 40.0) ** 2 + (source_col - 300.0) ** 2) / (2 * 12.0 ** 2))
        b = bilinear(blob, a, center=center)[0]
        f = fourier_rotation_kernel(blob, a, center=center)[0]
        results['angles_degrees'].append(degrees)
        results['bilinear_error'].append(float((b - exact)[keep].abs().max()))
        results['fourier_error'].append(float((f - exact)[keep].abs().max()))
        results['kernel_difference'].append(float((b - f)[keep].abs().max()))
    back = fourier_rotation_kernel(fourier_rotation_kernel(blob, 0.002, center=center), -0.002, center=center)
    results['rotate_and_back'] = float((back - blob)[:, keep].abs().max())
    noise = torch.randn(4, 64, num_cols, generator=generator).to(device)
    angles = [0.05 + 0.025 * k for k in range(9)]
    results['noise_angles_degrees'] = angles
    results['noise_power_fourier'] = [float((fourier_rotation_kernel(noise, math.radians(d), center=center)[:, 20:-20] ** 2).mean())
                                      for d in angles]
    results['noise_power_bilinear'] = [float((bilinear(noise, math.radians(d), center=center)[:, 20:-20] ** 2).mean())
                                       for d in angles]
    return results


def main():
    install()
    job.record('job', 'kernel', 0.0, kernel='fourier_shear', extra_row_margin=EXTRA_ROW_MARGIN,
               reflect_rows=REFLECT_ROWS, **check_kernel())
    job.main()


if __name__ == '__main__':
    main()
