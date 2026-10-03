"""Bias of QGGMRFDenoiser's automatic noise estimate (step 4 of the plan).

For each volume, kind of noise, and noise level, the script reports:
  estimate / sigma      estimate_image_noise_std divided by the true sigma
  r given sigma         the automatic sigma_x / sigma_y when sigma_noise = sigma
  r estimated           the automatic sigma_x / sigma_y when sigma_noise is
                        estimated, as in a call of denoise(image)
  MAD / sigma           a robust candidate: the median absolute first
                        difference, pooled over the three axes and all voxels,
                        divided by 0.6745 sqrt(2)
For white noise, the candidate is unbiased away from edges [derived].

The current estimate takes the standard deviation, with ddof = 0, of four
values: a voxel and its three backward neighbors.  It averages this over a
support of voxels above 5% of the mean absolute value plus the estimate of a
first pass.  For white noise in a flat region, the mean of this standard
deviation is 0.798 sigma [derived: sqrt(3/4) times c4(4) = 0.9213].

The automatic sigma_x is 0.2 times the same statistic, computed on every
num_rows // 20-th row (subsample_views), which is every 12th row here.  In
that subsample, neighbors along the row axis are 12 voxels apart.

The volumes are made by phantom3d.make_phantom:
  flat          water at 1000 everywhere, so the volume has no edges
  phantom       air at 0, water at 1000, and six inserts
  phantom_hu    the phantom minus 1000, so air is -1000 and water is 0, as in HU

Run on any machine with mbirtorch installed:
    python noise_estimate.py
"""

import sys
from pathlib import Path

import numpy as np

import mbirtorch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phantom3d import make_phantom  # noqa: E402

# ---- Run parameters ----------------------------------------------------------
SHAPE = (256, 256, 32)
VOLUMES = ('flat', 'phantom', 'phantom_hu')
NOISES = ('white', 'fdk')
SIGMAS = (20.0, 60.0, 182.0)


def make_volume(volume, noise, sigma):
    """Return (clean, noisy) for one volume, kind of noise, and noise level."""
    clean, noisy, _ = make_phantom(SHAPE, noise=noise, sigma=sigma)
    if volume == 'flat':
        return np.full(SHAPE, 1000.0), noisy - clean + 1000.0
    if volume == 'phantom_hu':
        return clean - 1000.0, noisy - 1000.0
    return clean, noisy


def automatic(noisy, sigma_noise):
    """Return (sigma_y, sigma_x) as denoise settles them, with sigma_noise None
    meaning the automatic estimate."""
    denoiser = mbirtorch.QGGMRFDenoiser(SHAPE)
    denoiser.configure_devices(devices=['cpu'])
    denoiser.set_params(no_warning=True, verbose=0)
    np.random.seed(0)
    denoiser.initialize_denoiser(image=noisy.astype(np.float32), sigma_noise=sigma_noise)
    return float(denoiser.get_params('sigma_y')), float(denoiser.get_params('sigma_x'))


def mad_estimate(noisy):
    """The median absolute first difference over all voxels and axes, divided
    by 0.6745 sqrt(2), which is sigma for white Gaussian noise."""
    differences = np.concatenate([np.abs(np.diff(noisy, axis=a)).ravel() for a in range(3)])
    return float(np.median(differences)) / (0.6745 * np.sqrt(2.0))


def main():
    print(f'{"volume":<12}{"noise":<7}{"sigma":>7}{"estimate/sigma":>16}{"r given sigma":>15}'
          f'{"r estimated":>13}{"MAD/sigma":>11}')
    for volume in VOLUMES:
        for noise in NOISES:
            for sigma in SIGMAS:
                _, noisy = make_volume(volume, noise, sigma)
                denoiser = mbirtorch.QGGMRFDenoiser(SHAPE)
                estimate = float(denoiser.estimate_image_noise_std(noisy.astype(np.float32)))
                sigma_y, sigma_x = automatic(noisy, sigma)
                sigma_y_auto, sigma_x_auto = automatic(noisy, None)
                print(f'{volume:<12}{noise:<7}{sigma:>7g}{estimate / sigma:>16.3f}'
                      f'{sigma_x / sigma_y:>15.3f}{sigma_x_auto / sigma_y_auto:>13.3f}'
                      f'{mad_estimate(noisy) / sigma:>11.3f}', flush=True)


if __name__ == '__main__':
    main()
