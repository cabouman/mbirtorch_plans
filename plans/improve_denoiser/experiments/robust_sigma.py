"""The noise level of option 2 for the MACE4D denoisers, from the initial image.

MACE4DModel sets the noise level of its three denoisers with
QGGMRFDenoiser.estimate_image_noise_std, applied to the initial image reshaped
to (frames * rows, columns, slices).  That estimate reads a strided subsample
of at most about five million points.  On the full Lilly volume, 99 frames of
260 x 260 x 728 voxels, the stride is 10, so the estimate compares voxels 10
apart.

Option 2 of findings/step4_stopping_rule.md replaces the estimate by the median
absolute difference of adjacent voxels, divided by 0.6745 sqrt(2).  For white
Gaussian noise this ratio is sigma.  The script reports four values:
  current           the library's estimate, computed as MACE4DModel computes it
  current_adjacent  the library's statistic on the central blocks below, where
                    the neighbors are adjacent voxels
  robust            option 2: the median absolute difference of adjacent voxels
                    in the central blocks, divided by 0.6745 sqrt(2)
  robust_strided    the same median on the library's strided subsample
The central blocks are the middle half of each spatial axis of the frames that
subsample_views picks, which are the frames MACE4DModel reads for sigma_x.  A
pair with an exact zero is left out, so that a masked exterior does not enter
the median.

The values go to --out as JSON and are printed.

Run with mbirtorch installed:
    python robust_sigma.py --init <init_dir>/init_recon.npy --out robust_sigma.json
"""

import argparse
import json

import numpy as np

import mbirtorch as mt

MAD_TO_SIGMA = 1.0 / (0.6745 * np.sqrt(2.0))


def median_absolute_difference(blocks):
    """The median absolute difference of adjacent voxels along the three axes
    of each block, over the pairs with no exact zero."""
    pieces = []
    for block in blocks:
        block = np.asarray(block, dtype=np.float32)
        for axis in range(3):
            first = np.take(block, np.arange(block.shape[axis] - 1), axis=axis)
            second = np.take(block, np.arange(1, block.shape[axis]), axis=axis)
            keep = (first != 0) & (second != 0)
            pieces.append(np.abs(second - first)[keep])
    return float(np.median(np.concatenate(pieces)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--init', required=True, help='init_recon.npy, of shape (frames, x, y, z).')
    parser.add_argument('--out', required=True, help='JSON file for the results.')
    args = parser.parse_args()

    init = np.load(args.init, mmap_mode='r')
    num_frames, rows, cols, slices = init.shape

    # The library's estimate, as MACE4DModel._estimate_global_sigma computes it.
    image_3d = np.asarray(init, dtype=np.float32).reshape(-1, cols, slices)
    denoiser = mt.QGGMRFDenoiser(image_3d.shape)
    denoiser.set_params(no_warning=True, verbose=0)
    current = float(denoiser.estimate_image_noise_std(image_3d))
    num_elements = image_3d.size
    stride = int(round((num_elements / min(5_000_000, num_elements)) ** (1 / 3)))
    strided = image_3d[::stride, ::stride, ::stride]
    robust_strided = median_absolute_difference([strided]) * MAD_TO_SIGMA

    frame_denoiser = mt.QGGMRFDenoiser((rows, cols, slices))
    frame_denoiser.set_params(no_warning=True, verbose=0)
    chosen = [int(t) for t in frame_denoiser.subsample_views(np.arange(num_frames))]
    box = tuple(slice(n // 4, n // 4 + max(n // 2, 2)) for n in (rows, cols, slices))
    blocks = [np.asarray(init[t][box], dtype=np.float32) for t in chosen]
    robust = median_absolute_difference(blocks) * MAD_TO_SIGMA
    block_shape = tuple(int(s.stop - s.start) for s in box)
    per_block = []
    for block in blocks:
        block_denoiser = mt.QGGMRFDenoiser(block.shape)
        block_denoiser.set_params(no_warning=True, verbose=0)
        per_block.append(float(block_denoiser.estimate_image_noise_std(block)))

    result = dict(init=args.init, shape=list(init.shape), current=current,
                  current_stride=stride, current_adjacent=float(np.mean(per_block)),
                  robust=robust, robust_strided=robust_strided, frames=chosen,
                  block=[[s.start, s.stop] for s in box], block_shape=list(block_shape))
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
