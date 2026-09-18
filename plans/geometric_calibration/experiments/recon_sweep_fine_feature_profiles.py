"""Profiles of two phantom features per candidate rotation: the horizontal groove and the teeth.

The fine-sweep jobs saved every even-view and odd-view slice stack, and this script reads two
features out of them, one curve per candidate rotation.  The first feature is the deep horizontal
groove across the disk on the upper far slice.  A vertical line crosses it, and the profile along
that line shows the groove as a dip; the same profile averaged over 240 columns gives the dip's
depth and its width at half depth with the noise averaged down, and a per-block pairing gives the
spread of the candidate differences.  The second feature is the ring of boundary teeth on the
other upper slice.  The intensity around the circle through mid-tooth height alternates between
tooth material and gap air, and the tooth mean minus the gap mean is the modulation a blurred
boundary reduces.

The script needs no GPU and no mbirtorch.  It reads the stacks and the JSON-lines record from the
results directory, writes its figures there, and prints its numbers.  Run parameters are at the
top.
"""
import json
import os

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, map_coordinates, median_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── run parameters ────────────────────────────────────────────────────────────────────────────────
# The directory with the job's JSON-lines record; the stacks and the figures default to it too.
RESULTS_DIR = os.environ.get('RECON_SWEEP_FINE_RESULTS', 'results_recon_sweep_fine_fourier')
STACKS_DIR = os.environ.get('RECON_SWEEP_FINE_STACKS', RESULTS_DIR)
SCAN = 'nsi_no_metal'
# The candidates drawn: named degrees, None for the slice's own best grid candidate, or 'vendor'.
SHOW_DEGREES = (0.130, None, 0.150, 'vendor')
VENDOR_DEGREES = 0.16716526473827872        # the vendor candidate the jobs inserted
BLUR = 2.0                                  # the scoring blur, in reconstruction pixels

GROOVE_SLICE = 1691
GROOVE_HALF_COLS = 2        # columns averaged on each side of the single line
GROOVE_COLUMNS = 120        # columns on each side of the disk's center in the averaged profile
GROOVE_HALF_WINDOW = 20     # rows kept on each side of the groove
GROOVE_MARGIN = 40          # rows inside the disk's edges excluded when the groove is found
GROOVE_BLOCK = 10           # columns per block in the paired comparison

MODULATION_SLICE = 1409
THETA_STEPS = 2880          # angular samples of the boundary
TOOTH_MIN_PROTRUSION = 3.0  # pixels a tooth must protrude above the smoothed boundary
EXCLUDE_COS = 0.98          # rays this close to horizontal cross the groove and are dropped
RADIAL_AVERAGE = (-2, -1, 0, 1, 2)   # circles averaged around the mid-tooth radius
ZOOM_START, ZOOM_WIDTH = 200.0, 80.0  # the angular stretch drawn, in degrees


# ── shared helpers ────────────────────────────────────────────────────────────────────────────────

def slice_record(slice_index):
    """The job's record of one slice, from the JSON-lines file."""
    with open(os.path.join(RESULTS_DIR, 'recon_sweep_fine.jsonl')) as handle:
        for line in handle:
            entry = json.loads(line)
            if (entry.get('dataset') == SCAN and entry.get('kind') == 'slice'
                    and entry.get('slice_index') == slice_index):
                return entry
    raise KeyError(slice_index)


def load_mean(slice_index):
    """The all-view slice stack as the mean of the even-view and odd-view stacks."""
    even = np.load(os.path.join(STACKS_DIR, f'{SCAN}_slice_{slice_index:04d}_even.npz'))
    odd = np.load(os.path.join(STACKS_DIR, f'{SCAN}_slice_{slice_index:04d}_odd.npz'))
    mean = 0.5 * (even['stack'].astype(np.float64) + odd['stack'].astype(np.float64))
    return mean, np.asarray(even['candidates_degrees'], dtype=np.float64)


def chosen_candidates(candidates, recorded):
    """The candidate values drawn, resolved from SHOW_DEGREES and sorted."""
    vendor = float(candidates[int(np.argmin(np.abs(candidates - VENDOR_DEGREES)))])
    own_best = float(candidates[int(np.argmin(recorded))])
    values = [own_best if d is None else (vendor if d == 'vendor' else float(d))
              for d in SHOW_DEGREES]
    return sorted(dict.fromkeys(round(v, 6) for v in values)), vendor


def label(value, vendor):
    return f'{value:.4f}' + (' (vendor)' if abs(value - vendor) < 1e-6 else '')


def boundary_radius(smooth, cy, cx, theta):
    """The outermost half-maximum crossing along each ray from the center."""
    half = 0.5 * float(smooth.max())
    radii = np.arange(20.0, min(smooth.shape) / 2.0 - 2.0, 0.5)
    rows = cy + radii[None, :] * np.sin(theta)[:, None]
    cols = cx + radii[None, :] * np.cos(theta)[:, None]
    values = map_coordinates(smooth, [rows.ravel(), cols.ravel()], order=1).reshape(rows.shape)
    above = values > half
    last = above.shape[1] - 1 - np.argmax(above[:, ::-1], axis=1)
    return np.where(above.any(axis=1), radii[last], np.nan)


# ── the groove ────────────────────────────────────────────────────────────────────────────────────

def find_groove(reference):
    """The groove's row and the disk's center column, found as the earlier far-slice measures
    found the dark line: the minimum of the mean vertical profile over the disk's central
    columns, away from the disk's edges."""
    smooth = gaussian_filter(reference, 3.0)
    mask = smooth > 0.5 * float(smooth.max())
    rows, cols = np.nonzero(mask)
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    center_col = int((c0 + c1) // 2)
    interior = reference[r0 + GROOVE_MARGIN:r1 - GROOVE_MARGIN,
                         center_col - GROOVE_COLUMNS:center_col + GROOVE_COLUMNS].mean(axis=1)
    return r0 + GROOVE_MARGIN + int(np.argmin(interior)), center_col


def depth_and_width(span, profile):
    """The dip's depth below its shoulders and its width at half depth, in rows.

    The base is the median of the profile's first and last five samples.  The width comes from
    the two half-depth crossings nearest the minimum, each interpolated linearly between samples.
    """
    base = float(np.median(np.concatenate([profile[:5], profile[-5:]])))
    m = int(np.argmin(profile))
    depth = base - float(profile[m])
    half = base - 0.5 * depth
    left = span[0]
    for j in range(m, 0, -1):
        if profile[j - 1] >= half > profile[j]:
            left = span[j - 1] + (profile[j - 1] - half) / (profile[j - 1] - profile[j])
            break
    right = span[-1]
    for j in range(m, profile.size - 1):
        if profile[j] < half <= profile[j + 1]:
            right = span[j] + (half - profile[j]) / (profile[j + 1] - profile[j])
            break
    return depth, float(right - left)


def groove_figure():
    mean, candidates = load_mean(GROOVE_SLICE)
    entry = slice_record(GROOVE_SLICE)
    recorded = np.asarray(entry['scores']['mean'][str(BLUR)], dtype=np.float64)
    shown, vendor = chosen_candidates(candidates, recorded)
    indices = {v: int(np.argmin(np.abs(candidates - v))) for v in shown}
    best = shown[int(np.argmax([-recorded[indices[v]] for v in shown]))]

    reference = mean[:, :, indices[best]]
    groove_row, center_col = find_groove(reference)
    lo, hi = groove_row - GROOVE_HALF_WINDOW, groove_row + GROOVE_HALF_WINDOW + 1
    span = np.arange(lo, hi)
    print(f'slice {GROOVE_SLICE}: groove at row {groove_row}, disk center column {center_col}')

    figure, axes = plt.subplots(1, 4, figsize=(19.0, 4.9),
                                gridspec_kw=dict(width_ratios=[1.15, 1.0, 1.0, 1.0]))
    axis = axes[0]
    view = reference[groove_row - 60:groove_row + 60,
                     center_col - GROOVE_COLUMNS - 40:center_col + GROOVE_COLUMNS + 40]
    low, high = np.percentile(view, [0.5, 99.5])
    axis.imshow(view, cmap='gray', vmin=low, vmax=high)
    x = GROOVE_COLUMNS + 40
    axis.plot([x, x], [60 - GROOVE_HALF_WINDOW, 60 + GROOVE_HALF_WINDOW], color='tab:red', lw=1.2)
    axis.set_title(f'slice {GROOVE_SLICE}: the groove and the line', fontsize=10)
    axis.set_xticks([]); axis.set_yticks([])

    blocks = None
    for axis, mode, title in ((axes[1], 'line', 'one line, as reconstructed'),
                              (axes[2], 'average', f'{2 * GROOVE_COLUMNS} columns averaged'),
                              (axes[3], 'blurred', f'averaged, after the {BLUR:.0f}-voxel blur')):
        for v in shown:
            image = mean[:, :, indices[v]]
            if mode == 'blurred':
                image = gaussian_filter(image, BLUR)
            if mode == 'line':
                profile = image[lo:hi, center_col - GROOVE_HALF_COLS:
                                center_col + GROOVE_HALF_COLS + 1].mean(axis=1)
                axis.plot(span, profile, label=label(v, vendor))
                continue
            band = image[lo:hi, center_col - GROOVE_COLUMNS:center_col + GROOVE_COLUMNS]
            profile = band.mean(axis=1)
            depth, width = depth_and_width(span, profile)
            axis.plot(span, profile,
                      label=f'{label(v, vendor)}: depth {depth:.5f}, width {width:.2f}')
            print(f'slice {GROOVE_SLICE} {mode} at {v:.4f} deg: groove depth {depth:.6f}, '
                  f'width at half depth {width:.3f} rows')
            if mode == 'blurred':
                per_block = band.reshape(band.shape[0], -1, GROOVE_BLOCK).mean(axis=2)
                block_depths = [depth_and_width(span, per_block[:, b])[0]
                                for b in range(per_block.shape[1])]
                blocks = blocks or {}
                blocks[v] = np.asarray(block_depths)
        axis.set_title(f'slice {GROOVE_SLICE}: {title}', fontsize=10)
        axis.set_xlabel('row')
        axis.set_ylabel('reconstructed value')
        axis.legend(fontsize=7)
        axis.grid(alpha=0.3)
    print('per-block groove depth relative to the best candidate '
          f'({GROOVE_BLOCK} columns per block; negative means shallower):')
    for v in shown:
        if v == best:
            continue
        difference = blocks[v] - blocks[best]
        print(f'  {v:.4f} minus {best:.4f}: mean {difference.mean():+.6f}, '
              f'sem {difference.std(ddof=1) / np.sqrt(difference.size):.6f}, '
              f'over {difference.size} blocks')
    figure.suptitle(f'{SCAN}, slice {GROOVE_SLICE}: the horizontal groove per candidate rotation',
                    fontsize=12)
    figure.tight_layout()
    out = os.path.join(RESULTS_DIR, f'{SCAN}_slice_{GROOVE_SLICE}_groove_profiles.png')
    figure.savefig(out, dpi=110)
    plt.close(figure)
    print('wrote', out)


# ── the teeth ─────────────────────────────────────────────────────────────────────────────────────

def modulation_figure():
    mean, candidates = load_mean(MODULATION_SLICE)
    entry = slice_record(MODULATION_SLICE)
    recorded = np.asarray(entry['scores']['mean'][str(BLUR)], dtype=np.float64)
    shown, vendor = chosen_candidates(candidates, recorded)
    indices = {v: int(np.argmin(np.abs(candidates - v))) for v in shown}
    best = shown[int(np.argmax([-recorded[indices[v]] for v in shown]))]

    reference = mean[:, :, indices[best]]
    smooth = gaussian_filter(reference, 3.0)
    mask = smooth > 0.5 * float(smooth.max())
    cy, cx = (float(w.mean()) for w in np.nonzero(mask))
    theta = np.linspace(0.0, 2.0 * np.pi, THETA_STEPS, endpoint=False)
    rb = boundary_radius(smooth, cy, cx, theta)
    rb = np.nan_to_num(rb, nan=np.nanmedian(rb))
    baseline = median_filter(rb, size=121)
    tooth = (rb - baseline) > TOOTH_MIN_PROTRUSION
    usable = np.abs(np.cos(theta)) < EXCLUDE_COS
    r_base = float(np.median(baseline))
    r_tip = float(np.median(rb[tooth])) if tooth.any() else r_base + 8.0
    r_mid = 0.5 * (r_base + r_tip)
    tooth_arc = binary_erosion(tooth, iterations=3) & usable
    gap_arc = binary_erosion(~tooth, iterations=3) & usable
    print(f'slice {MODULATION_SLICE}: sampling radius {r_mid:.1f} px, '
          f'{int(tooth_arc.sum())} tooth and {int(gap_arc.sum())} gap samples')

    curves = {}
    for v in shown:
        image = mean[:, :, indices[v]]
        curves[v] = np.mean([map_coordinates(image, [cy + (r_mid + dr) * np.sin(theta),
                                                     cx + (r_mid + dr) * np.cos(theta)], order=1)
                             for dr in RADIAL_AVERAGE], axis=0)

    flips = np.diff(gap_arc.astype(int))
    starts = list(np.nonzero(flips == 1)[0] + 1)
    ends = list(np.nonzero(flips == -1)[0] + 1)
    if gap_arc[0]:
        starts = [0] + starts
    if gap_arc[-1]:
        ends = ends + [gap_arc.size]
    gaps = [(s, e) for s, e in zip(starts, ends) if e - s >= 4]

    depths, gap_means = {}, {}
    for v in shown:
        depths[v] = float(curves[v][tooth_arc].mean() - curves[v][gap_arc].mean())
        gap_means[v] = np.array([curves[v][s:e].mean() for s, e in gaps])
        print(f'slice {MODULATION_SLICE} at {v:.4f} deg: modulation depth {depths[v]:.6f}, '
              f'tooth level {curves[v][tooth_arc].mean():.6f}, gap level {curves[v][gap_arc].mean():.6f}')
    print('per-gap fill relative to the best candidate (positive means shallower):')
    for v in shown:
        if v == best:
            continue
        difference = gap_means[v] - gap_means[best]
        print(f'  {v:.4f} minus {best:.4f}: mean {difference.mean():+.6f}, '
              f'sem {difference.std(ddof=1) / np.sqrt(difference.size):.6f}, over {difference.size} gaps')

    figure, axes = plt.subplots(1, 3, figsize=(17.0, 5.2),
                                gridspec_kw=dict(width_ratios=[1.0, 1.8, 0.8]))
    axis = axes[0]
    low, high = np.percentile(reference, [0.5, 99.9])
    axis.imshow(reference, cmap='gray', vmin=low, vmax=high)
    axis.plot(cx + r_mid * np.cos(theta), cy + r_mid * np.sin(theta), color='tab:red', lw=0.8)
    axis.set_title(f'slice {MODULATION_SLICE}: the sampling circle', fontsize=10)
    axis.set_xticks([]); axis.set_yticks([])
    axis = axes[1]
    zoom = (np.degrees(theta) >= ZOOM_START) & (np.degrees(theta) <= ZOOM_START + ZOOM_WIDTH)
    for v in shown:
        axis.plot(np.degrees(theta[zoom]), curves[v][zoom], lw=1.0, label=label(v, vendor))
    axis.set_xlabel('angle around the disk, degrees')
    axis.set_ylabel('reconstructed value on the circle')
    axis.set_title('a stretch of the circle: teeth high, gaps low', fontsize=10)
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    axis = axes[2]
    positions = np.arange(len(shown))
    axis.bar(positions, [depths[v] for v in shown], color='0.6')
    for j, v in enumerate(shown):
        axis.text(j, depths[v], f'{depths[v]:.5f}', ha='center', va='bottom', fontsize=8)
    axis.set_xticks(positions)
    axis.set_xticklabels([label(v, vendor).replace(' ', '\n') for v in shown], fontsize=8)
    axis.set_ylabel('tooth mean minus gap mean')
    axis.set_title('notch modulation per candidate', fontsize=10)
    figure.suptitle(f'{SCAN}, slice {MODULATION_SLICE}: the tooth-and-gap modulation per candidate '
                    'rotation', fontsize=12)
    figure.tight_layout()
    out = os.path.join(RESULTS_DIR, f'{SCAN}_slice_{MODULATION_SLICE}_notch_modulation.png')
    figure.savefig(out, dpi=110)
    plt.close(figure)
    print('wrote', out)


def main():
    groove_figure()
    modulation_figure()


if __name__ == '__main__':
    main()
