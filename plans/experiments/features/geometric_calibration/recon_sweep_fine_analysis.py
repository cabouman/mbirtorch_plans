"""Two analyses of the stacks that ``recon_sweep_fine.py`` saved: the resampling lattice, and a
score with the noise removed.

The fine sweep found that each slice's score curve has its minimum where the candidate rotation
displaces that slice's detector rows by a whole number of pixels.  The candidate rotation is applied
by bilinear resampling about the detector's center row.  A row at distance ``i`` from that center
moves by ``a * i`` pixels along the channels for a rotation of ``a`` radians.  When that displacement
is a whole number the resampling copies pixels and smooths nothing, and when it is a half pixel the
resampling averages neighbors and smooths most.  Smoothing lowers the gradient energy of the
reconstruction, so the score, which is the negative gradient energy, is lowest at the whole-number
displacements ``a = k / i`` for integer ``k``.  These angles form a lattice that depends only on the
slice's row, and this script compares each slice's minimum with that lattice.

The noise-removed score uses the two half reconstructions the job saved.  The even-view and odd-view
reconstructions hold the same object and independent noise.  Their mean is the all-view
reconstruction, and half their difference holds noise only, with the same variance as the noise in
the mean, and passed through the same resampling.  The gradient energy of the half difference is
therefore the noise's share of the mean's gradient energy, including the part the resampling
modulates.  Subtracting it leaves the gradient energy of the object's structure.  The script scores
that structure term on its own, normalized by the mean square of the structure, which is the mean's
mean square minus the half difference's.  It then repeats the per-slice analysis, fits the minima
over the slices with content to the model in which a residual channel offset ``d`` moves each
slice's minimum by ``d / i``, and combines the slices as the job did.

The script runs on the saved ``.npz`` stacks and the JSON-lines record of the job, without a GPU.  It
prints its tables and saves its figures beside the stacks.  Run parameters are at the top.
"""
import json
import math
import os

import numpy as np
from scipy.ndimage import gaussian_filter

# ── run parameters ────────────────────────────────────────────────────────────────────────────────
# The directory holding the job's JSON-lines record and its .npz stacks.
RESULTS_DIR = os.environ.get('RECON_SWEEP_FINE_RESULTS', 'results_recon_sweep_fine')
JSONL_NAME = 'recon_sweep_fine.jsonl'
SCANS = ('nsi_no_metal', 'nsi_metal')
# The blur widths analyzed, as multiples of the reconstruction voxel pitch, which is also the blur's
# standard deviation in reconstruction pixels on these scans.
BLUR_VOXELS = (0.0, 1.0, 2.0, 3.0, 4.0)
# A slice whose mean square is below this fraction of the largest slice's mean square holds no
# object and is left out of the fits and the combined curves.
CONTENT_FRACTION = 0.05
# The quadratic fit and the half width use the job's own settings.
PARABOLA_POINTS = 5
HALF_WIDTH_RISE = 0.02
SAVE_FIGURES = True


# ── small helpers, the same definitions the job uses ─────────────────────────────────────────────

def load_entries(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def gradient_energy(image):
    """The mean squared finite difference along each axis, summed over the two axes."""
    return float(np.mean(np.diff(image, axis=0) ** 2) + np.mean(np.diff(image, axis=1) ** 2))


def parabola_fit(candidates, curve, center_index):
    """The minimum of the quadratic through PARABOLA_POINTS candidates centered on ``center_index``,
    and its half width at a rise of HALF_WIDTH_RISE of the fitted minimum's magnitude."""
    width = int(min(PARABOLA_POINTS, curve.size))
    lo = int(min(max(int(center_index) - width // 2, 0), curve.size - width))
    x, y = candidates[lo:lo + width], curve[lo:lo + width]
    good = np.isfinite(y)
    nan = float('nan')
    if int(np.count_nonzero(good)) < 3:
        return dict(location_degrees=nan, half_width_degrees=nan, opens_upward=False)
    a, b, c = np.polyfit(x[good], y[good], 2)
    if a == 0.0:
        return dict(location_degrees=nan, half_width_degrees=nan, opens_upward=False)
    location = -b / (2.0 * a)
    value = a * location ** 2 + b * location + c
    half_width = math.sqrt(HALF_WIDTH_RISE * abs(value) / a) if a > 0.0 else nan
    return dict(location_degrees=float(location), half_width_degrees=float(half_width),
                opens_upward=bool(a > 0.0))


def curve_summary(candidates, curve):
    """The lowest candidate, the quadratic fit around it, the depth, and the count of interior
    minima of one curve."""
    curve = np.asarray(curve, dtype=np.float64)
    argmin = int(np.nanargmin(curve))
    interior = int(np.count_nonzero((curve[1:-1] < curve[:-2]) & (curve[1:-1] < curve[2:])))
    fit = parabola_fit(candidates, curve, argmin)
    return dict(argmin_degrees=float(candidates[argmin]), argmin_at_end=argmin in (0, curve.size - 1),
                depth=float(np.nanmax(curve) - np.nanmin(curve)), interior_minima=interior, **fit)


def trimmed_curve(matrix):
    """The mean over slices at each candidate after the highest and lowest slice scores are dropped.
    With three slices this is the middle slice's score."""
    matrix = np.asarray(matrix, dtype=np.float64)
    ordered = np.sort(matrix, axis=0)
    return ordered[1:-1].mean(axis=0) if matrix.shape[0] >= 3 else matrix.mean(axis=0)


def normalize_rows(matrix):
    """Each slice's curve divided by the magnitude of its own mean over the candidates."""
    scale = np.abs(np.mean(matrix, axis=1, keepdims=True))
    return matrix / scale


def lattice_points(i_kernel, low, high):
    """The whole-pixel displacement angles ``k / |i|`` in degrees inside ``[low, high]``."""
    points = []
    k = 1
    while True:
        angle = math.degrees(k / abs(i_kernel))
        if angle > high:
            return points
        if angle >= low:
            points.append((k, angle))
        k += 1


def nearest_lattice(angle, i_kernel):
    """The nearest whole-pixel displacement angle to ``angle`` for a row at distance ``i_kernel``."""
    k = max(1, int(round(math.radians(angle) * abs(i_kernel))))
    return k, math.degrees(k / abs(i_kernel))


# ── reading the job's record ──────────────────────────────────────────────────────────────────────

def scan_record(entries, name):
    """What the analysis needs from the job's record of one scan."""
    scan = next(e for e in entries if e['dataset'] == name and e['kind'] == 'scan')
    slices = [e for e in entries if e['dataset'] == name and e['kind'] == 'slice' and 'scores' in e]
    curves = [e for e in entries if e['dataset'] == name and e['kind'] == 'curve']
    num_rows = int(scan['shape'][1])
    center_row = (num_rows - 1) / 2.0        # the row the resampling turns about
    candidates = np.asarray(slices[0]['candidates_degrees'], dtype=np.float64)
    named = slices[0]['named_indices']
    for entry in slices:
        entry['i_kernel'] = entry['detector_row_on_axis'] - center_row
    # The per-slice fitted minima the job recorded, from the mean-parity raw curves, per blur.
    job_minima = {}
    for entry in curves:
        if entry['parity'] == 'mean':
            job_minima[entry['blur']] = entry['raw']['slice_locations']['parabola_degrees']
    return dict(name=name, num_rows=num_rows, center_row=center_row, candidates=candidates,
                named=named, slices=slices, job_minima=job_minima,
                vendor_degrees=scan.get('vendor_det_rotation_degrees'))


# ── the noise split ───────────────────────────────────────────────────────────────────────────────

def split_scores(even, odd, sigmas):
    """Score the mean, the noise, and the structure of one slice at every candidate and blur.

    Returns a dict with, per blur key, three curves over the candidates: ``mean`` is the job's
    score of the all-view slice; ``noise`` is the negative gradient energy of the half difference
    divided by the mean's mean square, so it is the noise's share of ``mean``; ``structure`` is the
    negative of the difference of the two gradient energies divided by the difference of the two
    mean squares, which is the score of the object alone.
    """
    keys = [str(float(b)) for b in sigmas]
    out = {key: dict(mean=[], noise=[], structure=[]) for key in keys}
    for k in range(even.shape[2]):
        m = 0.5 * (even[:, :, k].astype(np.float64) + odd[:, :, k].astype(np.float64))
        h = 0.5 * (even[:, :, k].astype(np.float64) - odd[:, :, k].astype(np.float64))
        for key, sigma in zip(keys, sigmas):
            mb = gaussian_filter(m, sigma) if sigma > 0 else m
            hb = gaussian_filter(h, sigma) if sigma > 0 else h
            ge_m, ge_h = gradient_energy(mb), gradient_energy(hb)
            ms_m, ms_h = float(np.mean(mb ** 2)), float(np.mean(hb ** 2))
            out[key]['mean'].append(-ge_m / ms_m)
            out[key]['noise'].append(-ge_h / ms_m)
            structure_ms = ms_m - ms_h
            out[key]['structure'].append(-(ge_m - ge_h) / structure_ms if structure_ms > 0 else float('nan'))
    return {key: {part: np.asarray(v) for part, v in parts.items()} for key, parts in out.items()}


def fit_offset_model(minima_degrees, i_kernels):
    """Fit ``a_s = a_true + d / i_s`` over slices, with ``a`` in radians and ``d`` in channels.

    Returns the fitted rotation in degrees, the offset error in channels, and the residual of each
    slice in degrees.  With two slices the fit is exact and the residuals are zero.
    """
    a = np.radians(np.asarray(minima_degrees, dtype=np.float64))
    design = np.stack([np.ones_like(a), 1.0 / np.asarray(i_kernels, dtype=np.float64)], axis=1)
    solution, *_ = np.linalg.lstsq(design, a, rcond=None)
    residual = a - design @ solution
    return math.degrees(solution[0]), float(solution[1]), np.degrees(residual)


# ── one scan ──────────────────────────────────────────────────────────────────────────────────────

def analyze(record):
    name, candidates = record['name'], record['candidates']
    low, high = float(candidates[0]), float(candidates[-1])
    keys = [str(float(b)) for b in BLUR_VOXELS]
    print(f'\n================ {name} ================')
    print(f"detector rows {record['num_rows']}, resampling center row {record['center_row']}, "
          f"vendor {record['vendor_degrees']:.5f} deg, candidates {low:.3f} to {high:.3f} deg")

    # Part 1: the lattice against the job's own per-slice minima.
    largest_ms = max(e['mean_square_at_identity'] for e in record['slices'])
    print('\nPART 1.  Whole-pixel displacement angles (the lattice) against each slice\'s minimum')
    print('slice | rows from center | mean square | content | lattice angles in range (k: deg) | '
          'job minimum at each blur (deg) | nearest lattice point at blur 2 (deg, difference)')
    for entry in record['slices']:
        i_k = entry['i_kernel']
        entry['has_content'] = entry['mean_square_at_identity'] >= CONTENT_FRACTION * largest_ms
        lattice = ', '.join(f'{k}: {ang:.4f}' for k, ang in lattice_points(i_k, low, high))
        minima = [record['job_minima'][key][record['slices'].index(entry)] for key in keys]
        k2, near = nearest_lattice(minima[keys.index('2.0')], i_k)
        print(f"{entry['slice_index']} | {i_k:+.1f} | {entry['mean_square_at_identity']:.3e} | "
              f"{'yes' if entry['has_content'] else 'no'} | {lattice} | "
              f"{', '.join(f'{m:.4f}' for m in minima)} | k={k2}: {near:.4f}, "
              f"{minima[keys.index('2.0')] - near:+.4f}")

    # Part 2: the noise split from the saved stacks.
    print('\nPART 2.  The noise split')
    split = {}
    for entry in record['slices']:
        index = entry['slice_index']
        even = np.load(os.path.join(RESULTS_DIR, f'{name}_slice_{index:04d}_even.npz'))['stack']
        odd = np.load(os.path.join(RESULTS_DIR, f'{name}_slice_{index:04d}_odd.npz'))['stack']
        split[index] = split_scores(even, odd, BLUR_VOXELS)
        # The job's mean-parity scores should be reproduced exactly up to float64 rounding.
        worst = max(float(np.max(np.abs(split[index][key]['mean'] - np.asarray(entry['scores']['mean'][key]))
                                 / np.max(np.abs(entry['scores']['mean'][key]))))
                    for key in keys)
        print(f'slice {index}: mean-parity scores reproduced to a relative {worst:.1e}')
        del even, odd

    print('\nnoise share of the score at the vendor candidate, per blur (noise / mean):')
    for entry in record['slices']:
        index, j = entry['slice_index'], record['named']['vendor']
        shares = ', '.join(f"{key}: {split[index][key]['noise'][j] / split[index][key]['mean'][j]:.3f}"
                           for key in keys)
        print(f'  slice {index} ({entry["i_kernel"]:+.0f} rows): {shares}')

    print('\nnoise curve (half difference only): lowest candidate and nearest lattice point, per blur')
    for entry in record['slices']:
        index, i_k = entry['slice_index'], entry['i_kernel']
        parts = []
        for key in keys:
            s = curve_summary(candidates, split[index][key]['noise'])
            k, near = nearest_lattice(s['location_degrees'], i_k) if np.isfinite(s['location_degrees']) else (0, float('nan'))
            parts.append(f"{key}: min {s['location_degrees']:.4f} (lattice {near:.4f}, depth/|mean| "
                         f"{s['depth'] / abs(np.mean(split[index][key]['noise'])):.3f})")
        print(f'  slice {index} ({i_k:+.0f} rows): ' + '; '.join(parts))

    print('\nstructure curve (noise removed): per-slice fitted minimum, half width, and nearest lattice point')
    print('blur | slice | rows from center | minimum deg | half width deg | depth/|mean| | nearest lattice deg | difference')
    structure_minima = {key: {} for key in keys}
    for key in keys:
        for entry in record['slices']:
            index, i_k = entry['slice_index'], entry['i_kernel']
            s = curve_summary(candidates, split[index][key]['structure'])
            structure_minima[key][index] = s
            k, near = nearest_lattice(s['location_degrees'], i_k) if np.isfinite(s['location_degrees']) else (0, float('nan'))
            print(f"{key} | {index} | {i_k:+.0f} | {s['location_degrees']:.4f} | {s['half_width_degrees']:.4f} | "
                  f"{s['depth'] / abs(np.mean(split[index][key]['structure'])):.3f} | {near:.4f} | "
                  f"{s['location_degrees'] - near:+.4f}")

    content = [e for e in record['slices'] if e['has_content']]
    print(f'\nslices with content: {[e["slice_index"] for e in content]}')
    print('\noffset-error model over the slices with content, on the structure curves: '
          'a_s = a_true + d / i_s')
    print('blur | fitted rotation deg | offset error channels | residual per slice deg')
    for key in keys:
        minima = [structure_minima[key][e['slice_index']]['location_degrees'] for e in content]
        rotation, offset, residual = fit_offset_model(minima, [e['i_kernel'] for e in content])
        print(f"{key} | {rotation:.4f} | {offset:+.3f} | {', '.join(f'{r:+.4f}' for r in residual)}")
    print('the same model on the job\'s own per-slice minima (the score with the noise still in it):')
    for key in keys:
        minima = [record['job_minima'][key][record['slices'].index(e)] for e in content]
        rotation, offset, residual = fit_offset_model(minima, [e['i_kernel'] for e in content])
        print(f"{key} | {rotation:.4f} | {offset:+.3f} | {', '.join(f'{r:+.4f}' for r in residual)}")

    # The same model solved exactly on the two content slices on opposite sides of the detector
    # center that are nearest to mirror images of each other, then checked on the remaining slices.
    below = [e for e in content if e['i_kernel'] < 0]
    above = [e for e in content if e['i_kernel'] > 0]
    if below and above:
        pair = min(((a, b) for a in below for b in above), key=lambda ab: abs(ab[0]['i_kernel'] + ab[1]['i_kernel']))
        others = [e for e in content if e not in pair]
        print(f"\nthe model solved on the pair of slices {pair[0]['slice_index']} ({pair[0]['i_kernel']:+.0f} rows) "
              f"and {pair[1]['slice_index']} ({pair[1]['i_kernel']:+.0f} rows), checked on the other slices")
        print('blur | score | rotation deg | offset error channels | other slice: predicted deg, measured deg, difference')
        for key in keys:
            for label, source in (('structure', lambda e: structure_minima[key][e['slice_index']]['location_degrees']),
                                  ('job', lambda e: record['job_minima'][key][record['slices'].index(e)])):
                rotation, offset, _ = fit_offset_model([source(pair[0]), source(pair[1])],
                                                       [pair[0]['i_kernel'], pair[1]['i_kernel']])
                checks = []
                for e in others:
                    predicted = rotation + math.degrees(offset / e['i_kernel'])
                    measured = source(e)
                    checks.append(f"{e['slice_index']}: {predicted:.4f}, {measured:.4f}, {measured - predicted:+.4f}")
                print(f"{key} | {label} | {rotation:.4f} | {offset:+.3f} | {'; '.join(checks)}")

    print('\ncombined structure curves over the slices with content')
    print('blur | kind | argmin deg | fit deg | half width deg | depth/|mean| | interior minima | '
          'value at 0.130 minus min, as a fraction of depth | same at vendor')
    combined = {}
    for key in keys:
        matrix = np.stack([split[e['slice_index']][key]['structure'] for e in content])
        for kind, rows in (('raw', matrix), ('normalized', normalize_rows(matrix))):
            curve = trimmed_curve(rows) if rows.shape[0] >= 3 else rows.mean(axis=0)
            label = 'trimmed' if rows.shape[0] >= 3 else 'mean'
            s = curve_summary(candidates, curve)
            combined[(key, kind)] = curve
            depth = s['depth']
            at_named = {n: (curve[j] - np.nanmin(curve)) / depth for n, j in record['named'].items()}
            print(f"{key} | {kind} {label} | {s['argmin_degrees']:.4f} | {s['location_degrees']:.4f} | "
                  f"{s['half_width_degrees']:.4f} | {depth / abs(np.mean(curve)):.3f} | {s['interior_minima']} | "
                  f"{at_named.get('0.130', float('nan')):.2f} | {at_named.get('vendor', float('nan')):.2f}")

    if SAVE_FIGURES:
        save_figures(record, split, content, combined, keys)
    return dict(split=split, structure_minima=structure_minima, content=content, combined=combined)


def save_figures(record, split, content, combined, keys):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    name, candidates = record['name'], record['candidates']
    low, high = float(candidates[0]), float(candidates[-1])
    # Noise-only curves with the lattice, one panel per blur.
    figure, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 4.2), squeeze=False)
    for axis, key in zip(axes[0], keys):
        for entry in record['slices']:
            curve = split[entry['slice_index']][key]['noise']
            line, = axis.plot(candidates, curve / abs(np.mean(curve)), label=f"{entry['i_kernel']:+.0f} rows")
            for _, angle in lattice_points(entry['i_kernel'], low, high):
                axis.axvline(angle, color=line.get_color(), linestyle=':', linewidth=0.8)
        axis.set_title(f'blur {key} voxels', fontsize=9)
        axis.set_xlabel('det_rotation, degrees')
        axis.legend(fontsize=7)
    axes[0][0].set_ylabel('noise score, normalized')
    figure.suptitle(f'{name}: score of the half difference (noise only); dotted lines are whole-pixel '
                    'displacements of each slice')
    figure.tight_layout()
    figure.savefig(os.path.join(RESULTS_DIR, f'{name}_noise_modulation.png'), dpi=110)
    plt.close(figure)
    # Structure curves per slice, normalized, with the lattice.
    figure, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 4.2), squeeze=False)
    for axis, key in zip(axes[0], keys):
        for entry in record['slices']:
            curve = split[entry['slice_index']][key]['structure']
            style = '-' if entry in content else ':'
            axis.plot(candidates, curve / abs(np.mean(curve)), style, label=f"{entry['i_kernel']:+.0f} rows")
        axis.axvline(0.130, color='0.6', linestyle='--', linewidth=0.8)
        axis.axvline(record['vendor_degrees'], color='0.6', linestyle='--', linewidth=0.8)
        axis.set_title(f'blur {key} voxels', fontsize=9)
        axis.set_xlabel('det_rotation, degrees')
        axis.legend(fontsize=7)
    axes[0][0].set_ylabel('structure score, normalized')
    figure.suptitle(f'{name}: score with the noise removed, each slice normalized by its own mean; '
                    'dotted curves are slices without content')
    figure.tight_layout()
    figure.savefig(os.path.join(RESULTS_DIR, f'{name}_structure_curves.png'), dpi=110)
    plt.close(figure)
    # Combined structure curves.
    figure, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 4.2), squeeze=False)
    for axis, key in zip(axes[0], keys):
        for kind, style in (('raw', '-'), ('normalized', '--')):
            curve = combined[(key, kind)]
            axis.plot(candidates, curve / abs(np.mean(curve)), style, label=kind)
        axis.axvline(0.130, color='0.6', linestyle='--', linewidth=0.8)
        axis.axvline(record['vendor_degrees'], color='0.6', linestyle='--', linewidth=0.8)
        axis.set_title(f'blur {key} voxels', fontsize=9)
        axis.set_xlabel('det_rotation, degrees')
        axis.legend(fontsize=7)
    axes[0][0].set_ylabel('combined structure score, normalized')
    figure.suptitle(f'{name}: combined structure score over the slices with content')
    figure.tight_layout()
    figure.savefig(os.path.join(RESULTS_DIR, f'{name}_structure_combined.png'), dpi=110)
    plt.close(figure)


def main():
    entries = load_entries(os.path.join(RESULTS_DIR, JSONL_NAME))
    for name in SCANS:
        if not any(e['dataset'] == name and e['kind'] == 'slice' and 'scores' in e for e in entries):
            print(f'\n{name}: no slice records')
            continue
        analyze(scan_record(entries, name))


if __name__ == '__main__':
    main()
