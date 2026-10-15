"""Charlie's suggestion for the noise estimate: the median in place of the mean.

The noise estimate of QGGMRFDenoiser on the mbirtorch branch prerelease takes,
for each voxel of a support, the standard deviation of four values: the voxel
and its three backward neighbors.  It returns the mean of these standard
deviations over the support.  Charlie suggested the median in place of the
mean, so that edges raise the estimate less.

For each volume, kind of noise, and noise level, the script reports four
estimates divided by the true noise level:
  mean            the estimate of prerelease, estimate_image_noise_std
  median          the same, with the median in place of the mean in
                  _get_estimate_of_recon_std, which both passes use
  median scaled   median / 0.7691.  For white noise in a flat region, the
                  median of the standard deviation of four values is 0.7691
                  times the noise level [derived, see MEDIAN_FACTOR].
  pairs 5 apart   the estimate of greg_dev at 235c82d: the median absolute
                  difference of voxels 5 apart along each axis, divided by
                  0.6745 sqrt(2).  Pairs with a voxel that is exactly 0 are
                  left out.  The function below is a copy of greg_dev's.

The automatic sigma_x of the denoiser is 0.2 times the statistic of
_get_estimate_of_recon_std, computed on a subsample of the rows.  So a median
inside that function changes sigma_x too.  The script reports the automatic
r = sigma_x / sigma_y with the true noise level given, for the mean and for
the median.

The volumes are those of noise_estimate.py:
  flat          water at 1000 everywhere, so the volume has no edges
  phantom       air at 0, water at 1000, and six inserts
  phantom_hu    the phantom minus 1000, so air is -1000 and water is 0, as in HU
Each gets white noise or fdk noise, as phantom3d.py describes.

Run from any folder except one that holds a folder named mbirtorch, with the
prerelease branch of mbirtorch installed:
    python noise_median.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy import stats

import mbirtorch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from noise_estimate import NOISES, SHAPE, SIGMAS, VOLUMES, make_volume  # noqa: E402

# The standard deviation (dividing by 4) of four independent standard normal
# values is sqrt(X / 4), where X has a chi-square distribution with 3 degrees
# of freedom.  Its median is sqrt(median(X) / 4).
MEDIAN_FACTOR = float(np.sqrt(stats.chi2.median(3) / 4.0))

# The constants of greg_dev's estimate.
PAIR_SEPARATION = 5
NORMAL_ABS_MEDIAN = 0.6744897501960817
POINT_BUDGET = 5_000_000


class MedianDenoiser(mbirtorch.QGGMRFDenoiser):
    """The prerelease denoiser with the median in place of the mean in its noise statistic."""

    def _get_estimate_of_recon_std(self, noisy_image, support_indicator):
        """Return the median, over the support, of the standard deviation of each voxel and
        its three backward neighbors.  The rest is the code of prerelease."""
        def views():
            yield noisy_image
            for axis in range(3):
                yield np.roll(noisy_image, 1, axis=axis)

        count = 4
        mean = np.array(noisy_image, dtype=noisy_image.dtype, copy=True)
        for index, value in enumerate(views()):
            if index:
                mean += value
        mean /= count
        spread = np.zeros_like(mean)
        deviation = np.empty_like(mean)
        for value in views():
            np.subtract(value, mean, out=deviation)
            deviation *= deviation
            spread += deviation
        spread /= count
        np.sqrt(spread, out=spread)
        return np.median(spread[support_indicator.astype(bool)])


def pair_stride(shape, separation, point_budget):
    """Copy of greg_dev's _pair_stride: the smallest stride at which the pairs
    number at most point_budget."""
    def num_pairs(stride):
        total = 0
        for axis in range(3):
            if shape[axis] > separation:
                lines = 1
                for other in range(3):
                    if other != axis:
                        lines *= -(-shape[other] // stride)
                total += (shape[axis] - separation) * lines
        return total

    stride = 1
    while num_pairs(stride) > point_budget:
        stride += 1
    return stride


def pair_estimate(image):
    """Copy of greg_dev's estimate_image_noise_std for a numpy volume."""
    shape = image.shape
    separation = min(PAIR_SEPARATION, max(shape) - 1)
    stride = pair_stride(shape, separation, POINT_BUDGET)
    differences = []
    for axis in range(3):
        if shape[axis] <= separation:
            continue
        steps = [stride, stride, stride]
        steps[axis] = 1
        lines = image[::steps[0], ::steps[1], ::steps[2]].astype(np.float64)
        first = [slice(None)] * 3
        second = [slice(None)] * 3
        first[axis] = slice(0, shape[axis] - separation)
        second[axis] = slice(separation, shape[axis])
        first, second = lines[tuple(first)], lines[tuple(second)]
        kept = (first != 0) & (second != 0)
        differences.append(np.abs(second - first)[kept])
    differences = np.concatenate(differences)
    return float(np.median(differences)) / (np.sqrt(2.0) * NORMAL_ABS_MEDIAN)


def automatic_r(denoiser_class, noisy, sigma):
    """Return sigma_x / sigma_y as initialize_denoiser sets them with sigma_noise given."""
    denoiser = denoiser_class(SHAPE)
    denoiser.configure_devices(devices=['cpu'])
    denoiser.set_params(no_warning=True, verbose=0)
    np.random.seed(0)
    denoiser.initialize_denoiser(image=noisy, sigma_noise=sigma)
    return float(denoiser.get_params('sigma_x')) / float(denoiser.get_params('sigma_y'))


def main():
    print(f'median factor for white noise: {MEDIAN_FACTOR:.4f}')
    print('| Volume | Noise | sigma (HU) | mean | median | median scaled | pairs 5 apart '
          '| r, mean | r, median |')
    for volume in VOLUMES:
        for noise in NOISES:
            for sigma in SIGMAS:
                _, noisy = make_volume(volume, noise, sigma)
                noisy = noisy.astype(np.float32)
                mean = float(mbirtorch.QGGMRFDenoiser(SHAPE).estimate_image_noise_std(noisy))
                median = float(MedianDenoiser(SHAPE).estimate_image_noise_std(noisy))
                pairs = pair_estimate(noisy)
                r_mean = automatic_r(mbirtorch.QGGMRFDenoiser, noisy, sigma)
                r_median = automatic_r(MedianDenoiser, noisy, sigma)
                print(f'| {volume} | {noise} | {sigma:g} | {mean / sigma:.3f} | '
                      f'{median / sigma:.3f} | {median / MEDIAN_FACTOR / sigma:.3f} | '
                      f'{pairs / sigma:.3f} | {r_mean:.3f} | {r_median:.3f} |', flush=True)


if __name__ == '__main__':
    main()
