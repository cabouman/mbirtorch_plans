"""The residual noise of the final images of the MACE4D runs.

For each 4D image, the script computes the median absolute difference of
adjacent voxels, divided by 0.6745 sqrt(2).  robust_sigma.py computes option
2's noise level in the same way, on the central blocks of the frames it lists
in robust_sigma.json.  With --frames_from that file, the script uses the same
frames, so the value of the initial image equals option 2's noise level.
Without it, the script uses every fourth frame.  The script reports the value
over the three spatial axes together, along each spatial axis, and along
time, between each chosen frame and the next one.  A pair with an exact zero
is left out.  A smoother image gives smaller values.

The values go to --out as CSV and are printed.

Run with numpy:
    python image_noise.py --images init=<init_recon.npy> A=<run_dir>/recon.npy ... \
        --frames_from <init_dir>/robust_sigma.json --out image_noise.csv
"""

import argparse
import csv
import json
import os

import numpy as np

MAD_TO_SIGMA = 1.0 / (0.6745 * np.sqrt(2.0))
FRAME_STEP = 4                      # the frame step without --frames_from


def absolute_differences(first, second):
    keep = (first != 0) & (second != 0)
    return np.abs(second - first)[keep]


def noise_levels(path, frames):
    """The robust noise level over the spatial axes, along each axis, and along time."""
    image = np.load(path, mmap_mode='r')
    num_frames, rows, cols, slices = image.shape
    box = tuple(slice(n // 4, n // 4 + max(n // 2, 2)) for n in (rows, cols, slices))
    frames = frames if frames is not None else list(range(0, num_frames, FRAME_STEP))
    axes = [[], [], []]
    time = []
    for t in frames:
        block = np.asarray(image[t][box], dtype=np.float32)
        for axis in range(3):
            first = np.take(block, np.arange(block.shape[axis] - 1), axis=axis)
            second = np.take(block, np.arange(1, block.shape[axis]), axis=axis)
            axes[axis].append(absolute_differences(first, second))
        if t + 1 < num_frames:
            time.append(absolute_differences(block, np.asarray(image[t + 1][box], dtype=np.float32)))
    axes = [np.concatenate(pieces) for pieces in axes]
    row = dict(spatial=float(np.median(np.concatenate(axes)) * MAD_TO_SIGMA))
    for axis, values in enumerate(axes):
        row[f'axis_{axis + 1}'] = float(np.median(values) * MAD_TO_SIGMA)
    row['time'] = float(np.median(np.concatenate(time)) * MAD_TO_SIGMA) if time else ''
    row['frames'] = len(frames)
    row['block_shape'] = 'x'.join(str(s.stop - s.start) for s in box)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--images', nargs='+', required=True, help='label=path pairs of 4D images.')
    parser.add_argument('--frames_from', default=None, help='robust_sigma.json, whose frames are used.')
    parser.add_argument('--out', required=True, help='CSV file for the results.')
    args = parser.parse_args()
    frames = [int(t) for t in json.load(open(args.frames_from))['frames']] if args.frames_from else None
    rows = []
    for item in args.images:
        label, path = item.split('=', 1)
        if not os.path.exists(path):
            print(f'{label}: no file {path}', flush=True)
            continue
        row = dict(label=label, **noise_levels(path, frames))
        rows.append(row)
        print({k: (round(v, 7) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
