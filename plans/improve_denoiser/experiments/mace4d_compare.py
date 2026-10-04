"""Compare the MACE4D runs of the GPU test of the denoiser's stopping rule.

Reads the run folders of mace4d_stopping.py and writes:
  timing_summary.csv        one row per run: the noise level, sigma_x, r, the
                            wall time, the time of all iterations, the times
                            per iteration (iteration 1, which includes
                            compilation, and the mean of the later
                            iterations), the denoiser iterations, the last
                            consensus change, and the peak device memory
  timing_per_iteration.csv  the timing logs of all runs in one table
  denoiser_iterations.csv   per run, iteration, and orientation: the median,
                            mean, and largest iteration count of the volumes,
                            and the fraction that reached the cap
  recon_differences.csv     per pair of runs: the relative rms difference of
                            the final 4D images, the smallest, median, and
                            largest relative rms difference of a frame, and
                            the largest absolute difference over the largest
                            absolute value
  slices.png                the middle frame's middle slice along each spatial
                            axis, for every run, and each run's difference
                            from the baseline run, on a color scale per run

Run with numpy and matplotlib:
    python mace4d_compare.py --runs A=<dir> B=<dir> ... --baseline B --out <dir>
"""

import argparse
import csv
import itertools
import json
import os

import numpy as np


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    for row in rows[1:]:
        fields += [k for k in row if k not in fields]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def timing(label, run):
    """The timing rows of one run and its summary row."""
    summary = json.load(open(os.path.join(run, 'summary.json')))
    rows = read_csv(os.path.join(run, 'logs', 'timing_log.csv'))
    keys = ['prox_total_sec', 'denoise_total_sec', 'makespan_sec', 'iteration_total_sec']
    values = {k: np.array([float(r[k]) for r in rows]) for k in keys}
    mean_iterations = np.array([float(r['denoise_mean_iterations']) for r in rows])
    memory = summary.get('peak_device_memory', {})
    row = dict(label=label, sigma=summary['denoiser_sigma'], sigma_x=summary['denoiser_sigma_x'],
               r=summary['r'], sigma_source=summary.get('sigma_source'),
               warm_start=summary['args'].get('denoiser_warm_start'), rule=summary['args'].get('rule'),
               iterations=len(rows), wall_seconds=summary['wall_seconds'],
               torch_threads=summary.get('torch_threads'))
    row['iterations_total_seconds'] = float(np.sum(values['iteration_total_sec']))
    for k in keys:
        row[f'{k}_first'] = float(values[k][0])
        row[f'{k}_later_mean'] = float(np.mean(values[k][1:])) if len(rows) > 1 else ''
    row['denoise_mean_iterations_first'] = float(mean_iterations[0])
    row['denoise_mean_iterations_later_mean'] = float(np.mean(mean_iterations[1:])) if len(rows) > 1 else ''
    row['consensus_change_pct_last'] = float(rows[-1]['consensus_change_pct'])
    row['peak_allocated_gb_max'] = max((m['max_allocated_gb'] for m in memory.values()), default='')
    row['peak_reserved_gb_max'] = max((m['max_reserved_gb'] for m in memory.values()), default='')
    per_iteration = [dict(label=label, **r) for r in rows]
    return row, per_iteration


def iteration_counts(label, run):
    """Per iteration and orientation: the statistics of the volumes' iteration counts."""
    rows = read_csv(os.path.join(run, 'volumes.csv'))
    calls = read_csv(os.path.join(run, 'calls.csv'))
    cap = {(c['iteration'], c['orientation']): int(c['max_iterations']) for c in calls}
    groups = {}
    for r in rows:
        groups.setdefault((r['iteration'], r['orientation']), []).append(int(r['iterations']))
    out = []
    for (iteration, orientation), counts in sorted(groups.items(), key=lambda t: (int(t[0][0]), t[0][1])):
        counts = np.array(counts)
        limit = cap[(iteration, orientation)]
        out.append(dict(label=label, iteration=int(iteration), orientation=orientation,
                        volumes=len(counts), median=float(np.median(counts)), mean=float(counts.mean()),
                        max=int(counts.max()), cap=limit, fraction_at_cap=float(np.mean(counts >= limit))))
    return out


def differences(labels, runs):
    """The differences between the final images of every pair of runs.

    The second run of a pair is the reference of the relative values.  Each
    image is read once, one frame at a time.
    """
    images = {label: np.load(os.path.join(runs[label], 'recon.npy'), mmap_mode='r') for label in labels}
    pairs = list(itertools.combinations(labels, 2))
    num, den, worst, peak = ({pair: 0.0 for pair in pairs} for _ in range(4))
    per_frame = {pair: [] for pair in pairs}
    for t in range(images[labels[0]].shape[0]):
        frames = {label: np.asarray(images[label][t], dtype=np.float64) for label in labels}
        for a, b in pairs:
            xt, yt = frames[a], frames[b]
            d2 = float(np.sum((xt - yt) ** 2))
            y2 = float(np.sum(yt ** 2))
            num[a, b] += d2
            den[a, b] += y2
            per_frame[a, b].append(np.sqrt(d2 / y2) if y2 > 0 else np.nan)
            worst[a, b] = max(worst[a, b], float(np.max(np.abs(xt - yt))))
            peak[a, b] = max(peak[a, b], float(np.max(np.abs(yt))))
    rows = []
    for a, b in pairs:
        frame_values = np.array(per_frame[a, b])
        rows.append(dict(first=a, second=b, relative_rms=float(np.sqrt(num[a, b] / den[a, b])),
                         frame_relative_rms_min=float(np.nanmin(frame_values)),
                         frame_relative_rms_median=float(np.nanmedian(frame_values)),
                         frame_relative_rms_max=float(np.nanmax(frame_values)),
                         max_abs_over_max=worst[a, b] / peak[a, b]))
    return rows


def slices(labels, runs, baseline, path):
    """One figure: the middle slices of the middle frame, and the differences from the baseline."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    images = {label: np.load(os.path.join(runs[label], 'recon.npy'), mmap_mode='r') for label in labels}
    t = images[baseline].shape[0] // 2

    def cuts(volume):
        volume = np.asarray(volume[t], dtype=np.float32)
        return [volume[volume.shape[0] // 2], volume[:, volume.shape[1] // 2], volume[:, :, volume.shape[2] // 2]]

    base = cuts(images[baseline])
    vmax = float(np.percentile(np.concatenate([c.ravel() for c in base]), 99.5))
    fig, axes = plt.subplots(2 * len(labels), 3, figsize=(12, 4.2 * len(labels)), squeeze=False)
    planes = {label: cuts(images[label]) for label in labels}
    for i, label in enumerate(labels):
        # Each run's differences have their own color scale, which the titles state.
        spread = max(float(np.percentile(np.abs(c - b), 99.5)) for c, b in zip(planes[label], base)) or 1.0
        for j, (cut, ref) in enumerate(zip(planes[label], base)):
            axes[2 * i, j].imshow(cut.T, cmap='gray', vmin=0, vmax=vmax, origin='lower')
            axes[2 * i, j].set_title(f'{label}, frame {t}, axis {j + 1} middle')
            axes[2 * i + 1, j].imshow((cut - ref).T, cmap='RdBu', vmin=-spread, vmax=spread, origin='lower')
            axes[2 * i + 1, j].set_title(f'{label} minus {baseline} (+/-{spread:.2g})')
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=80)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--runs', nargs='+', required=True, help='label=run_dir pairs.')
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    runs = dict(item.split('=', 1) for item in args.runs)
    labels = [label for label in runs if os.path.exists(os.path.join(runs[label], 'summary.json'))]
    os.makedirs(args.out, exist_ok=True)

    summary_rows, per_iteration, counts = [], [], []
    for label in labels:
        row, rows = timing(label, runs[label])
        summary_rows.append(row)
        per_iteration += rows
        counts += iteration_counts(label, runs[label])
    write_csv(os.path.join(args.out, 'timing_summary.csv'), summary_rows)
    write_csv(os.path.join(args.out, 'timing_per_iteration.csv'), per_iteration)
    write_csv(os.path.join(args.out, 'denoiser_iterations.csv'), counts)
    write_csv(os.path.join(args.out, 'recon_differences.csv'), differences(labels, runs))
    if args.baseline in labels:
        slices(labels, runs, args.baseline, os.path.join(args.out, 'slices.png'))
    for row in summary_rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)


if __name__ == '__main__':
    main()
