"""Do the package's two rotation implementations share one geometric convention?

Question.  Every synthetic detector-rotation example in this feature was made by rotating a
sinogram with ``_rotation_kernel``, the bilinear kernel that ``correct_det_rotation``,
``apply_calibration``, and ``reduce_sinogram`` use.  The rotation estimator scores each candidate
angle with a different implementation, OpenCV's cubic ``warpAffine`` inside ``_rotated_band``.  A
recovery test built that way checks the estimator against the correction's convention.  It shows
that the two agree.  It cannot show that either one is right, which is why the real-scan
reconstruction job exists.  This script measures the agreement between the two implementations
directly, and it measures the size of the disagreement that a sign error or a center error would
produce, so that "the comparison would have caught such an error" is a measurement rather than a
claim.

Method.  One smooth image, a sum of Gaussian blobs, stands in for one view.  The image is smooth
so that the cubic and bilinear kernels differ little on it, and a convention difference would
stand out.  Four comparisons run at each test angle, each reported as the maximum absolute
difference over the interior divided by the maximum absolute interior value:

1. ``_rotation_kernel`` against OpenCV bilinear about the same center at the same angle.  A value
   near zero means the two implementations compute the same map.
2. The same pair with the OpenCV angle negated.  This is the size a sign error would show.
3. The same pair with the OpenCV center moved by half a channel.  This is the size a half-pixel
   center error would show.
4. ``_rotated_band`` (the estimator's scoring path, cubic) against ``_rotation_kernel`` (the
   correction path, bilinear) at the same angle, and against the negated angle as a control.

The interior excludes a border of BORDER pixels, because the two implementations zero
out-of-bounds samples by slightly different rules at the rim.
"""

import math

import cv2
import numpy as np
import torch

from mbirtorch.preprocess import geometry_calibration as gc
from mbirtorch.preprocess.utilities import _rotation_kernel

# Run parameters.
NUM_ROWS = 96
NUM_CHANNELS = 128
ANGLES_DEGREES = (0.3, -0.5, 2.0)
BORDER = 8            # pixels excluded on every side of the comparison
NUM_BLOBS = 6
BLOB_SIGMA_RANGE = (6.0, 14.0)
SEED = 0


def smooth_image():
    """A sum of Gaussian blobs on a small constant background, float32, (NUM_ROWS, NUM_CHANNELS)."""
    rng = np.random.default_rng(SEED)
    rows = np.arange(NUM_ROWS, dtype=np.float64)[:, None]
    cols = np.arange(NUM_CHANNELS, dtype=np.float64)[None, :]
    img = np.full((NUM_ROWS, NUM_CHANNELS), 0.05)
    for _ in range(NUM_BLOBS):
        r0 = rng.uniform(0.25 * NUM_ROWS, 0.75 * NUM_ROWS)
        c0 = rng.uniform(0.25 * NUM_CHANNELS, 0.75 * NUM_CHANNELS)
        sigma = rng.uniform(*BLOB_SIGMA_RANGE)
        img += rng.uniform(0.3, 1.0) * np.exp(-((rows - r0) ** 2 + (cols - c0) ** 2) / (2 * sigma ** 2))
    return img.astype(np.float32)


def rotate_kernel(img, radians):
    """The package's bilinear rotation about the array center, as a (NUM_ROWS, NUM_CHANNELS) array."""
    batch = torch.from_numpy(img)[None]
    return _rotation_kernel(batch, radians).numpy()[0]


def rotate_cv2(img, radians, center_shift_channels=0.0):
    """OpenCV bilinear rotation about the array center, optionally with the center moved."""
    center = ((NUM_CHANNELS - 1) / 2.0 + center_shift_channels, (NUM_ROWS - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, math.degrees(radians), 1.0)
    return cv2.warpAffine(img, matrix, (NUM_CHANNELS, NUM_ROWS), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def rotated_band(img, radians):
    """The estimator's scoring path: ``_rotated_band`` over the whole detector at bin factor 1."""
    reduction = {'full_sinogram_shape': (1, NUM_ROWS, NUM_CHANNELS),
                 'sinogram_shape': (1, NUM_ROWS, NUM_CHANNELS),
                 'view_stride': 1, 'bin_factor': 1, 'row_window': (0, NUM_ROWS),
                 'devices': ['cpu']}
    return gc._rotated_band(img[None], reduction, radians)[0]


def rel_max(a, b):
    """max |a - b| over the interior, divided by max |a| over the interior."""
    interior = np.s_[BORDER:-BORDER, BORDER:-BORDER]
    return float(np.max(np.abs(a[interior] - b[interior])) / np.max(np.abs(a[interior])))


def main():
    img = smooth_image()
    print(f'image {NUM_ROWS} x {NUM_CHANNELS}, border {BORDER}, blobs {NUM_BLOBS}, seed {SEED}')
    header = (f'{"angle, deg":>10}  {"1 same map":>12}  {"2 sign error":>12}  '
              f'{"3 center+0.5":>12}  {"4 band vs kernel":>16}  {"4 control, -angle":>17}')
    print(header)
    for degrees in ANGLES_DEGREES:
        radians = math.radians(degrees)
        reference = rotate_kernel(img, radians)
        same = rel_max(reference, rotate_cv2(img, radians))
        sign = rel_max(reference, rotate_cv2(img, -radians))
        center = rel_max(reference, rotate_cv2(img, radians, center_shift_channels=0.5))
        band = rel_max(reference, rotated_band(img, radians))
        band_control = rel_max(reference, rotated_band(img, -radians))
        print(f'{degrees:>10.2f}  {same:>12.2e}  {sign:>12.2e}  {center:>12.2e}  '
              f'{band:>16.2e}  {band_control:>17.2e}')


if __name__ == '__main__':
    main()
