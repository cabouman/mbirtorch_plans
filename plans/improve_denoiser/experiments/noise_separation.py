"""Option 2's noise estimate at several separations of the voxel pairs.

Option 2 of findings/step4_stopping_rule.md estimates the noise level as the
median absolute difference of voxel pairs, divided by 0.6745 sqrt(2).  For
white Gaussian noise this ratio is sigma.  Noise that is correlated between
neighbors makes the differences of adjacent voxels smaller, so the estimate
is low.  Pairs that are farther apart are less correlated, but the image's
own variation over the separation adds to their differences.

For each volume, kind of noise, and noise level of noise_estimate.py, the
script reports the estimate divided by the true sigma for pairs 1, 2, 3, 5,
and 10 voxels apart along each axis, pooled over the three axes.  A pair
with an exact zero is left out, as in robust_sigma.py.

Run with numpy and scipy:
    python noise_separation.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phantom3d import make_phantom  # noqa: E402

SHAPE = (256, 256, 32)
VOLUMES = ('flat', 'phantom', 'phantom_hu')
NOISES = ('white', 'fdk', 'fdk_z')
SIGMAS = (20.0, 60.0, 182.0)
SEPARATIONS = (1, 2, 3, 5, 10)
MAD_TO_SIGMA = 1.0 / (0.6745 * np.sqrt(2.0))


def make_volume(volume, noise, sigma):
    """Return the noisy volume, as noise_estimate.py makes it."""
    clean, noisy, _ = make_phantom(SHAPE, noise=noise, sigma=sigma)
    if volume == 'flat':
        return noisy - clean + 1000.0
    if volume == 'phantom_hu':
        return noisy - 1000.0
    return noisy


def robust_estimate(image, separation):
    """The median absolute difference of the voxel pairs that lie
    `separation` voxels apart along each axis, divided by 0.6745 sqrt(2)."""
    pieces = []
    for axis in range(3):
        if image.shape[axis] <= separation:
            continue
        first = np.take(image, np.arange(image.shape[axis] - separation), axis=axis)
        second = np.take(image, np.arange(separation, image.shape[axis]), axis=axis)
        keep = (first != 0) & (second != 0)
        pieces.append(np.abs(second - first)[keep])
    return float(np.median(np.concatenate(pieces))) * MAD_TO_SIGMA


def main():
    header = ''.join(f'{f"d = {d}":>9}' for d in SEPARATIONS)
    print(f'{"volume":<12}{"noise":<7}{"sigma":>7}{header}')
    for volume in VOLUMES:
        for noise in NOISES:
            for sigma in SIGMAS:
                noisy = make_volume(volume, noise, sigma)
                ratios = ''.join(f'{robust_estimate(noisy, d) / sigma:>9.3f}' for d in SEPARATIONS)
                print(f'{volume:<12}{noise:<7}{sigma:>7g}{ratios}', flush=True)


if __name__ == '__main__':
    main()
