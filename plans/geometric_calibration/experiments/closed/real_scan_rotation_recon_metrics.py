"""Two sharpness measures for the slices that ``real_scan_rotation_recon.py`` saved.

The job's own sharpness numbers are dominated by the resampling that applies each candidate
rotation, because bilinear interpolation smooths pixel-scale noise by an amount that depends on the
fractional part of the displacement.  This script reads the saved slice arrays and computes two
measures that are less sensitive to that smoothing.  The first is the gradient energy of the slice
after a Gaussian blur of two pixels, divided by the blurred slice's mean square.  The second is the
depth and the width at half depth of the phantom's dark horizontal line, taken from the mean
vertical profile over 240 central columns.  The tables it prints were transcribed to
``real_scan_rotation_recon.md``.

Run parameters are at the top.  The script ran on a Mac in the mbirtorch conda environment on the
``.npz`` files copied from the cluster.
"""
import os

import numpy as np
from scipy.ndimage import gaussian_filter

# ── run parameters ────────────────────────────────────────────────────────────────────────────────
SLICE_DIR = os.environ.get('ROTATION_RECON_RESULTS', 'results_real_scan_rotation_recon')
SLICES = (('nsi_no_metal', 470), ('nsi_no_metal', 1409), ('nsi_no_metal', 1691),
          ('nsi_metal', 1409), ('nsi_metal', 1691))
BLUR_SIGMA = 2.0            # pixels
LINE_HALF_WINDOW = 12       # rows on each side of the dark line in the vertical profile
LINE_COLUMNS = 120          # columns on each side of the disk's center in the profile
INTERIOR_MARGIN = 40        # rows inside the disk's edges excluded when the line is found


def blurred_sharpness(image):
    """Gradient energy after a Gaussian blur, divided by the blurred image's mean square."""
    smooth = gaussian_filter(np.asarray(image, dtype=np.float64), BLUR_SIGMA)
    energy = float(np.mean(smooth ** 2))
    return float((np.mean(np.diff(smooth, axis=0) ** 2) + np.mean(np.diff(smooth, axis=1) ** 2)) / energy)


def dark_line(stack):
    """The row of the phantom's dark horizontal line, found on the slice at the third rotation."""
    reference = stack[:, :, 2]
    mask = gaussian_filter(reference, 3) > 0.5 * float(gaussian_filter(reference, 3).max())
    rows, cols = np.nonzero(mask)
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    center_col = (c0 + c1) // 2
    interior = reference[r0 + INTERIOR_MARGIN:r1 - INTERIOR_MARGIN,
                         center_col - LINE_COLUMNS:center_col + LINE_COLUMNS].mean(axis=1)
    return r0 + INTERIOR_MARGIN + int(np.argmin(interior)), center_col


def line_depth_and_width(image, line_row, center_col):
    """The dark line's dip depth below the surrounding level and its width at half depth, in rows."""
    profile = image[line_row - LINE_HALF_WINDOW:line_row + LINE_HALF_WINDOW + 1,
                    center_col - LINE_COLUMNS:center_col + LINE_COLUMNS].mean(axis=1)
    base = float(np.median(np.concatenate([profile[:5], profile[-5:]])))
    depth = base - float(profile.min())
    width = int(np.count_nonzero(profile < base - 0.5 * depth))
    return depth, width


def main():
    for name, slice_index in SLICES:
        data = np.load(os.path.join(SLICE_DIR, f'{name}_slice_{slice_index:04d}_rotations.npz'))
        stack, rotations = data['stack'], list(data['rotations_degrees'])
        sharp = [blurred_sharpness(stack[:, :, k]) for k in range(stack.shape[2])]
        line_row, center_col = dark_line(stack)
        lines = [line_depth_and_width(stack[:, :, k], line_row, center_col) for k in range(stack.shape[2])]
        print(name, slice_index, 'line row', line_row)
        for k, rotation in enumerate(rotations):
            print(f'  {rotation:.3f} deg: blurred sharpness {sharp[k]:.5f}, line depth {lines[k][0]:.4f}, '
                  f'line width {lines[k][1]} rows')


if __name__ == '__main__':
    main()
