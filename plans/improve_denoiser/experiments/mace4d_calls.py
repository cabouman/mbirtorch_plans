"""The distance of each sampled MACE4D denoiser call from that call's MAP estimate.

mace4d_stopping.py keeps a few denoiser calls of each run.  A sample holds one
hyperplane volume of the call: its input, its warm start if there was one, its
output, the call's parameters, and the call's pixel partition.  For each sample
this script does four things.

1. It computes the call's MAP estimate.  The denoiser starts from the call's
   output and runs the gradient rule at REFERENCE_THRESHOLD, with a cap of
   REFERENCE_MAX_ITERATIONS iterations.
2. It computes the exact gradient of the cost at the MAP estimate, after the
   last update.  In units of sigma_y, its rms is an upper bound on the
   remaining distance from the true minimizer.
3. It reports the distance of the call's output from the MAP estimate.
4. It runs the denoiser again from the call's start, the input or the warm
   start, one iteration at a time up to TRAJECTORY_ITERATIONS, with the call's
   own partition.  After each iteration it records the distance, the percent
   change, and the gradient statistic.  From this trajectory it reports where
   each rule in RULES stops, how far from the MAP estimate it stops, and how
   many iterations reach each distance in TARGETS.

Every distance is an rms over the volume, in units of sigma_y.  The trajectory
uses the patched library's solver.  For a sample of a run of that solver
(versions B, C, and D), the trajectory after the call's iteration count
reproduces the call's output, and replay_difference checks this.  For a sample
of version A, replay_difference is the difference between the old and the new
solver after the same number of iterations.

The MAP estimate and the trajectory depend only on the start, the input, and
the parameters, so they are cached by a hash of these under --cache.  Samples
of different runs with the same input share them.

Files written to --out:
  calls_summary.csv   one row per sample
  trajectories.csv    one row per sample and iteration

Needs the patched library (gradient_rule.patch).  Run on one GPU:
    python mace4d_calls.py --runs <run_dir> [<run_dir> ...] --out <dir> --cache <dir>
To use several GPUs, run one process per GPU with --shard k/n and its own
--out, then combine the outputs with --merge:
    CUDA_VISIBLE_DEVICES=k python mace4d_calls.py --runs ... --shard k/n --out <dir>/shard_k --cache <dir>
    python mace4d_calls.py --merge <dir>/shard_* --out <dir>
A shard holds whole groups of samples with the same input: the samples of the
first MACE iteration, or of the one-frame check, share a group per
orientation, and every other sample is a group of its own.
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

import mbirtorch as mt
from mbirtorch import qggmrf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mace4d_stopping import check_library  # noqa: E402

# ---- Run parameters ----------------------------------------------------------
REFERENCE_THRESHOLD = 1e-3          # the gradient statistic at which the MAP run stops
REFERENCE_MAX_ITERATIONS = 20000
TRAJECTORY_ITERATIONS = 400
TARGETS = (0.1, 0.03, 0.01)
# Each rule is (name, statistic, threshold, cap).
RULES = (('percent_0.05_cap_15', 'percent', 0.05, 15),
         ('percent_0.2_cap_15', 'percent', 0.2, 15),
         ('gradient_0.03_cap_200', 'gradient', 0.03, 200),
         ('gradient_0.01_cap_200', 'gradient', 0.01, 200))


def make_denoiser(shape, meta, partition, device):
    """A stack denoiser set up as MACE4DModel sets up its denoisers."""
    denoiser = mt.QGGMRFDenoiser(shape)
    denoiser.configure_devices(devices=[device])
    denoiser.set_params(no_warning=True, verbose=0, sigma_y=meta['sigma_y'], sigma_noise=meta['sigma_y'],
                        sigma_x=meta['sigma_x'], p=meta['p'], q=meta['q'], T=meta['T'],
                        qggmrf_nbr_wts=meta['qggmrf_nbr_wts'],
                        granularity=[int(g) for g in meta['granularity']],
                        partition_sequence=[int(v) for v in meta['partition_sequence']],
                        use_ror_mask=bool(meta['use_ror_mask']), auto_regularize_flag=False)
    denoiser.initialize_denoiser(partition=partition)
    return denoiser


def exact_gradient(x, z, meta, device):
    """sigma_y times the rms gradient of the cost at x, over every voxel."""
    shape = x.shape
    flat_x = torch.as_tensor(x, device=device).reshape(-1, shape[2]).contiguous()
    flat_z = torch.as_tensor(z, device=device).reshape(-1, shape[2])
    b = qggmrf.get_b_from_nbr_wts(meta['qggmrf_nbr_wts'])
    params = (b, meta['sigma_x'], meta['p'], meta['q'], meta['T'])
    fm_constant = 1.0 / meta['sigma_y'] ** 2
    total = 0.0
    num_pixels = flat_x.shape[0]
    step = max(1, num_pixels // 8)
    with torch.no_grad():
        for start in range(0, num_pixels, step):
            indices = torch.arange(start, min(start + step, num_pixels), device=device)
            prior_grad, _ = qggmrf.qggmrf_gradient_and_hessian_at_indices(flat_x, shape, indices, params)
            gradient = prior_grad - fm_constant * (flat_z[indices] - flat_x[indices])
            total += float(torch.sum(gradient.double() ** 2))
    return meta['sigma_y'] * float(np.sqrt(total / flat_x.numel()))


def key_of(*arrays, meta):
    digest = hashlib.sha1()
    for a in arrays:
        digest.update(np.ascontiguousarray(a).tobytes())
    for name in ('sigma_y', 'sigma_x', 'p', 'q', 'T', 'qggmrf_nbr_wts'):
        digest.update(repr(meta[name]).encode())
    return digest.hexdigest()[:20]


def reference(z, x, meta, partition, device, cache):
    """The MAP estimate of a call, its iteration count, its last gradient
    statistic, and its exact gradient, from the cache when present."""
    path = os.path.join(cache, f'ref_{key_of(z, meta=meta)}.npz')
    if os.path.exists(path):
        saved = np.load(path)
        return saved['map'], int(saved['iterations']), float(saved['statistic']), float(saved['exact'])
    denoiser = make_denoiser(z.shape, meta, partition, device)
    tick = time.time()
    out, info = denoiser.denoise_stack(z[None], init_stack=x[None], max_iterations=REFERENCE_MAX_ITERATIONS,
                                       stop_threshold_gradient=REFERENCE_THRESHOLD, batch_size=1,
                                       do_initialization=False)
    out = np.asarray(out[0], dtype=np.float32)
    iterations = int(info['num_iterations'][0])
    statistic = float(info['gradient_statistic'][0][-1])
    exact = exact_gradient(out, z, meta, device)
    print(f'    MAP run: {iterations} iterations in {time.time() - tick:.0f} s, statistic {statistic:.2e}, '
          f'exact gradient {exact:.2e}', flush=True)
    np.savez(path, map=out, iterations=iterations, statistic=statistic, exact=exact)
    return out, iterations, statistic, exact


def trajectory(z, start, ref, meta, partition, device, cache):
    """The distance, the percent change, and the gradient statistic after
    each iteration from ``start``, from the cache when present."""
    path = os.path.join(cache, f'traj_{key_of(z, start, partition, meta=meta)}.npy')
    if os.path.exists(path):
        return np.load(path)
    denoiser = make_denoiser(z.shape, meta, partition, device)
    sigma = meta['sigma_y']
    z_t = torch.as_tensor(z[None], device=device)
    current = torch.as_tensor(start[None], device=device).clone()
    ref_t = torch.as_tensor(ref, device=device).double()
    rows = []
    for _ in range(TRAJECTORY_ITERATIONS):
        current, info = denoiser.denoise_stack(z_t, init_stack=current, max_iterations=1,
                                               stop_threshold_change_pct=0.0, batch_size=1,
                                               do_initialization=False)
        distance = float(torch.sqrt(torch.mean((current[0].double() - ref_t) ** 2))) / sigma
        rows.append((distance, info['nmae_pct'][0][-1], info['gradient_statistic'][0][-1]))
    rows = np.array(rows)
    np.save(path, rows)
    return rows


def replay(z, start, partition, meta, device, iterations):
    """The image after ``iterations`` iterations from ``start``."""
    denoiser = make_denoiser(z.shape, meta, partition, device)
    out, _ = denoiser.denoise_stack(z[None], init_stack=start[None], max_iterations=iterations,
                                    stop_threshold_change_pct=0.0, batch_size=1, do_initialization=False)
    return np.asarray(out[0], dtype=np.float32)


def stop_of(rows, statistic, threshold, cap):
    """The iteration at which a rule stops on a trajectory, and the distance there."""
    column = 1 if statistic == 'percent' else 2
    below = np.flatnonzero(rows[:cap, column] < threshold)
    k = int(below[0]) + 1 if below.size else cap
    if k > rows.shape[0]:
        return '', ''
    return k, float(rows[k - 1, 0])


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def merge(folders, out):
    """Combine the output files of the shards."""
    os.makedirs(out, exist_ok=True)
    for name in ('calls_summary.csv', 'trajectories.csv'):
        rows = []
        for folder in folders:
            if os.path.exists(os.path.join(folder, name)):
                rows += read_csv(os.path.join(folder, name))
        write_csv(os.path.join(out, name), rows)
        print(f'{name}: {len(rows)} rows', flush=True)


def group_of(meta):
    """Samples of one group have the same input, so they share the cache."""
    if int(meta['iteration']) <= 1:
        return f"{meta['iteration']}_{meta['orientation']}"
    return f"{meta['label']}_{meta['iteration']}_{meta['orientation']}"


def sample_paths(runs, shard):
    """The sample files of the runs, or the part of them that belongs to shard k of n."""
    paths = [p for run in runs for p in sorted(glob.glob(os.path.join(run, 'samples', '*.npz')))]
    if shard is None:
        return paths
    k, n = (int(v) for v in shard.split('/'))
    groups = {p: group_of(json.loads(str(np.load(p)['meta']))) for p in paths}
    names = sorted(set(groups.values()))
    mine = {name for i, name in enumerate(names) if i % n == k}
    return [p for p in paths if groups[p] in mine]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--runs', nargs='+', help='Run folders of mace4d_stopping.py.')
    parser.add_argument('--out', required=True)
    parser.add_argument('--cache')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shard', default=None, help='k/n: process part k of n, counting from 0.')
    parser.add_argument('--merge', nargs='+', default=None, help='Shard output folders to combine.')
    args = parser.parse_args()
    if args.merge:
        merge(args.merge, args.out)
        return
    if not args.runs or not args.cache:
        parser.error('give --runs and --cache, or --merge')
    check_library()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)
    print('library under test:', mt.__file__, flush=True)

    summary_rows, trajectory_rows = [], []
    paths = sample_paths(args.runs, args.shard)
    print(f'{len(paths)} samples', flush=True)
    for path in paths:
        tick = time.time()
        saved = np.load(path)
        meta = json.loads(str(saved['meta']))
        z = saved['input']
        x = saved['output']
        start = saved['init'] if 'init' in saved.files else z
        partition = torch.as_tensor(saved['partition'])
        sigma = meta['sigma_y']
        print(f"{meta['label']} iteration {meta['iteration']} {meta['orientation']} plane "
              f"{meta['plane']}: shape {z.shape}, r {meta['sigma_x'] / sigma:.4f}", flush=True)
        ref, ref_iterations, ref_statistic, ref_exact = reference(z, x, meta, partition, args.device,
                                                                  args.cache)
        rows = trajectory(z, start, ref, meta, partition, args.device, args.cache)
        replayed = replay(z, start, partition, meta, args.device, meta['iterations'])
        row = dict(label=meta['label'], iteration=meta['iteration'], orientation=meta['orientation'],
                   plane=meta['plane'], sigma_y=sigma, sigma_x=meta['sigma_x'],
                   r=meta['sigma_x'] / sigma, warm_start=int(meta['warm_start']),
                   call_iterations=meta['iterations'], call_max_iterations=meta['max_iterations'],
                   call_percent_last=meta['percent_change'][-1],
                   call_gradient_last=(meta['gradient_statistic'][-1]
                                       if meta.get('gradient_statistic') else ''),
                   distance_start=float(np.sqrt(np.mean((start.astype(np.float64) - ref) ** 2))) / sigma,
                   distance_call=float(np.sqrt(np.mean((x.astype(np.float64) - ref) ** 2))) / sigma,
                   replay_difference=float(np.max(np.abs(replayed - x))) / sigma,
                   reference_iterations=ref_iterations, reference_statistic=ref_statistic,
                   reference_exact_gradient=ref_exact, seconds=round(time.time() - tick, 1))
        for target in TARGETS:
            reached = np.flatnonzero(rows[:, 0] < target)
            row[f'iterations_to_{target:g}'] = int(reached[0]) + 1 if reached.size else ''
        for name, statistic, threshold, cap in RULES:
            k, distance = stop_of(rows, statistic, threshold, cap)
            row[f'{name}_stop'] = k
            row[f'{name}_distance'] = distance
        summary_rows.append(row)
        print('   ', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                     if k.startswith(('distance', 'replay', 'iterations_to', 'percent', 'gradient', 'seconds'))},
              flush=True)
        for step, (distance, percent, gradient) in enumerate(rows, start=1):
            trajectory_rows.append(dict(label=meta['label'], iteration=meta['iteration'],
                                        orientation=meta['orientation'], step=step,
                                        distance=distance, percent_change=percent,
                                        gradient_statistic=gradient))

    write_csv(os.path.join(args.out, 'calls_summary.csv'), summary_rows)
    write_csv(os.path.join(args.out, 'trajectories.csv'), trajectory_rows)
    print(f'{len(summary_rows)} samples written to {args.out}', flush=True)


if __name__ == '__main__':
    main()
