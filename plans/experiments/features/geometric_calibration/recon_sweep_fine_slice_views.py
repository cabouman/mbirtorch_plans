"""Zoomed comparisons of one reconstructed slice at four candidate rotations, for viewing.

For each requested slice the script loads the fine-sweep job's even-view and odd-view stacks,
forms the all-view slice as their mean, and draws one figure: the full slice with the crop marked
and the slice's score curve, then a zoomed crop at each of four candidates, then each crop's
difference from the best of the four after the scoring blur.  The crop sits on the object's
boundary where the candidates differ most, so it shows the object's edge rather than the holder
ring at the field's rim.  The scores shown are recomputed from the stacks and checked against the
job's own record.

The script needs no GPU and no mbirtorch.  It reads the stacks and the JSON-lines record from the
results directory and writes its figures there.  Run parameters are at the top.
"""
import json
import os

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, uniform_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── run parameters ────────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.environ.get('RECON_SWEEP_FINE_RESULTS', 'results_recon_sweep_fine_fourier')
STACKS_DIR = os.environ.get('RECON_SWEEP_FINE_STACKS', RESULTS_DIR)
SCAN = 'nsi_no_metal'
SLICES = (1691, 1409)
# The candidates drawn: named degrees, None for the slice's own best grid candidate, or 'vendor'.
SHOW_DEGREES = (0.130, None, 0.150, 'vendor')
VENDOR_DEGREES = 0.16716526473827872        # the vendor candidate the jobs inserted
BLUR = 2.0                                  # the scoring blur, in reconstruction pixels
CROP_ROWS, CROP_COLS = 260, 380


def score(image, sigma=BLUR):
    """The job's score: negative blurred gradient energy over the blurred mean square."""
    smooth = gaussian_filter(np.asarray(image, dtype=np.float64), sigma)
    energy = float(np.mean(smooth ** 2))
    return float(-(np.mean(np.diff(smooth, axis=0) ** 2)
                   + np.mean(np.diff(smooth, axis=1) ** 2)) / energy)


def slice_record(slice_index):
    with open(os.path.join(RESULTS_DIR, 'recon_sweep_fine.jsonl')) as handle:
        for line in handle:
            entry = json.loads(line)
            if (entry.get('dataset') == SCAN and entry.get('kind') == 'slice'
                    and entry.get('slice_index') == slice_index):
                return entry
    raise KeyError(slice_index)


def best_crop(reference, other):
    """The crop that shows the object's boundary where the two candidates differ most.

    The object is the bright phantom, found by thresholding the smoothed slice at half its
    maximum.  The crop is centered where the blurred difference between the two candidate
    reconstructions has the most energy on a band around that object's boundary.
    """
    smooth = gaussian_filter(reference, 3.0)
    mask = smooth > 0.5 * float(smooth.max())
    band = binary_dilation(mask, iterations=12) & ~binary_erosion(mask, iterations=12)
    difference = gaussian_filter(reference, BLUR) - gaussian_filter(other, BLUR)
    energy = uniform_filter((difference * band) ** 2, size=(CROP_ROWS, CROP_COLS))
    r, c = np.unravel_index(int(np.argmax(energy)), energy.shape)
    r0 = int(np.clip(r - CROP_ROWS // 2, 0, reference.shape[0] - CROP_ROWS))
    c0 = int(np.clip(c - CROP_COLS // 2, 0, reference.shape[1] - CROP_COLS))
    return r0, c0


def main():
    for slice_index in SLICES:
        even = np.load(os.path.join(STACKS_DIR, f'{SCAN}_slice_{slice_index:04d}_even.npz'))
        odd = np.load(os.path.join(STACKS_DIR, f'{SCAN}_slice_{slice_index:04d}_odd.npz'))
        mean = 0.5 * (even['stack'].astype(np.float64) + odd['stack'].astype(np.float64))
        candidates = np.asarray(even['candidates_degrees'], dtype=np.float64)
        entry = slice_record(slice_index)
        recorded = np.asarray(entry['scores']['mean'][str(BLUR)], dtype=np.float64)
        vendor = candidates[int(np.argmin(np.abs(candidates - VENDOR_DEGREES)))]

        # The scores recomputed here must match the job's own record.
        check = [score(mean[:, :, k]) for k in (0, len(candidates) // 2, len(candidates) - 1)]
        worst = max(abs(a - recorded[k]) / abs(recorded[k])
                    for a, k in zip(check, (0, len(candidates) // 2, len(candidates) - 1)))
        print(f'slice {slice_index}: recomputed scores match the record to a relative {worst:.1e}')

        own_best = float(candidates[int(np.argmin(recorded))])
        shown = [own_best if d is None else (vendor if d == 'vendor' else float(d))
                 for d in SHOW_DEGREES]
        shown = sorted(dict.fromkeys(round(v, 6) for v in shown))
        indices = [int(np.argmin(np.abs(candidates - v))) for v in shown]
        sharp = [-recorded[k] * 1e3 for k in indices]
        best_shown = int(np.argmax(sharp))

        r0, c0 = best_crop(mean[:, :, indices[0]], mean[:, :, indices[-1]])
        crops = [mean[r0:r0 + CROP_ROWS, c0:c0 + CROP_COLS, k] for k in indices]
        low, high = np.percentile(crops[best_shown], [1.0, 99.5])
        # The differences are shown after the scoring blur, which averages the noise down and
        # leaves the displaced edges the score reads.
        blurred = [gaussian_filter(c, BLUR) for c in crops]
        diffs = [b - blurred[best_shown] for b in blurred]
        diff_scale = max(float(np.percentile(np.abs(d), 99.5)) for j, d in enumerate(diffs)
                         if j != best_shown)

        n = len(indices)
        figure = plt.figure(figsize=(4.1 * n, 12.2))
        grid = figure.add_gridspec(3, n, height_ratios=[1.25, 1.0, 1.0])
        axis = figure.add_subplot(grid[0, : max(1, n // 2)])
        full_low, full_high = np.percentile(mean[:, :, best_shown], [0.5, 99.9])
        axis.imshow(mean[:, :, best_shown], cmap='gray', vmin=full_low, vmax=full_high)
        axis.add_patch(plt.Rectangle((c0, r0), CROP_COLS, CROP_ROWS, fill=False, color='red',
                                     lw=1.5))
        axis.set_title(f'slice {slice_index}, all views, at {shown[best_shown]:.4f} deg',
                       fontsize=10)
        axis.set_xticks([]); axis.set_yticks([])
        axis = figure.add_subplot(grid[0, max(1, n // 2):])
        axis.plot(candidates, -recorded * 1e3, '.-', color='0.3')
        for v in shown:
            axis.axvline(v, color='tab:red', linestyle=':', linewidth=1.0)
            axis.annotate(f'{v:.3f}', (v, axis.get_ylim()[0]), fontsize=8, rotation=90,
                          va='bottom', ha='right', color='tab:red')
        axis.set_xlabel('det_rotation, degrees')
        axis.set_ylabel('blurred sharpness x1e3 (higher is sharper)')
        axis.set_title(f'score of this slice, blur {BLUR:.0f} voxels', fontsize=10)
        for j, (v, crop, s) in enumerate(zip(shown, crops, sharp)):
            axis = figure.add_subplot(grid[1, j])
            axis.imshow(crop, cmap='gray', vmin=low, vmax=high)
            delta = 100.0 * (s - sharp[best_shown]) / sharp[best_shown]
            name = ' (vendor)' if abs(v - vendor) < 1e-6 else ''
            axis.set_title(f'{v:.4f} deg{name}\nsharpness {s:.3f}, {delta:+.1f}% vs best',
                           fontsize=9)
            axis.set_xticks([]); axis.set_yticks([])
        for j, d in enumerate(diffs):
            axis = figure.add_subplot(grid[2, j])
            axis.imshow(d, cmap='gray', vmin=-diff_scale, vmax=diff_scale)
            axis.set_title(f'{shown[j]:.4f} minus {shown[best_shown]:.4f} deg, blurred',
                           fontsize=9)
            axis.set_xticks([]); axis.set_yticks([])
        figure.suptitle(f'{SCAN}, slice {slice_index} '
                        f'({entry["rows_from_central_plane"]:+.0f} rows from the central plane): '
                        'the reconstruction at four candidate rotations', fontsize=12)
        figure.tight_layout()
        out = os.path.join(RESULTS_DIR, f'{SCAN}_slice_{slice_index:04d}_candidates.png')
        figure.savefig(out, dpi=110)
        plt.close(figure)
        print('wrote', out)


if __name__ == '__main__':
    main()
