"""Warm-process iteration curves, and reconstruction images at N = 512.

Part A re-runs the iteration curves of quality_leap_vs_mbirtorch.py with
discarded warm-up reconstructions first, so that no timed run pays a
torch.compile cost.  In the earlier study the runs ascended in k inside one
process and the k = 1 and k = 2 runs carried the compilation, which made the
measured curve run backwards at its left end.  It also times the direct
reconstruction three times, which the earlier study never did.

Part B saves slices of the phantom and of five reconstructions at N = 512 and
composes them into figures, so the error numbers in the curves can be looked at.

Everything is written to a new JSON-lines file under new operation names, so the
earlier records are untouched.

Modes:
  curves   the warm iteration curve for one library at one size
  slices   the reconstructions of Part B for one library, saved as .npy
  compose  build the figures from the saved slices (no GPU needed)
  plot     redraw the error-against-wall-time figure from the warm records
"""

import argparse
import json
import os
import time

import numpy as np

import bench_leap_vs_mbirtorch as bench
import quality_leap_vs_mbirtorch as quality

# The settings the N = 256 sweep chose, checked against the sweep's own records
# at start-up rather than trusted as literals.
BEST_SETTING = {
    'mbirtorch': {'sharpness': -1.0, 'snr_db': None},
    'leap': {'tv_weight': 10.0, 'tv_delta': 0.001},
}

K_BY_SIZE = {256: quality.K_VALUES, 512: quality.K_VALUES, 1024: [5, 10]}

# The warm-up reconstructions, discarded.  k = 3 is what was asked for; the
# second run at k = 5 is here because mbirtorch's default partition sequence,
# [2, 4, 6] + [7, 8, 9, 10] * 25 into granularity
# [1, 2, 4, 8, 16, 32, 64, 128, 128, 128, 128], uses 4 subsets at iteration 1,
# 16 at iteration 2, 64 at iteration 3 and 128 from iteration 4 on.  A k = 3
# warm-up therefore never reaches the fourth and last of those shapes.
WARMUP_K = [3, 5]

# Part B.  k = 14 is the LEAP iteration count whose warm time is closest to
# mbirtorch's warm time to target at N = 512; k = 10 is mbirtorch's own
# target-reaching point; k = 100 is where LEAP first reaches the target.
SLICE_RUNS = {'mbirtorch': [10], 'leap': [14, 100]}
GRAY_WINDOW = (0.0, 0.045)


def sphere_centres_mm():
    """The two inner spheres, as (x, y, z) in mm and radius in mm."""
    return [((30.0, -20.0, 25.0), 25.0, 'A, value 0.5'),
            ((-25.0, 35.0, -30.0), 18.0, 'B, value 2.0')]


def mm_to_index(value_mm, n, delta):
    return int(np.rint(value_mm / delta + (n - 1) / 2.0))


def slice_plan(n):
    """Which planes to save, in mbirtorch's (y, x, z) index order.

    NOTE, and it is a property of the phantom rather than a choice made here:
    no axis-aligned plane contains both inner sphere centres.  Sphere A is
    visible in planes with y between -45 and 5 mm, sphere B in planes with y
    between 17 and 53 mm, and those ranges do not overlap; the same is true of
    the z ranges.  So one coronal plane is saved through each inner sphere
    rather than one through both.
    """
    delta = bench.geometry(n)['delta_voxel_mm']
    plan = [('axial_centre', 'z', n // 2,
             'axial, the centre slice z = N/2; only the large sphere is cut here')]
    for (cx, cy, cz), radius, label in sphere_centres_mm():
        plan.append((f'coronal_sphere_{label[0]}', 'y', mm_to_index(cy, n, delta),
                     f'coronal (x, z) through inner sphere {label}'))
    return plan


def take_slice(volume, axis, index):
    """One plane of a volume held in mbirtorch's (y, x, z) order."""
    if axis == 'z':
        return np.ascontiguousarray(volume[:, :, index])
    if axis == 'y':
        return np.ascontiguousarray(volume[index, :, :])
    return np.ascontiguousarray(volume[:, index, :])


def check_setting(library, args):
    """Confirm the hardcoded best setting is what the sweep actually chose."""
    if not args.sweep_jsonl or not os.path.exists(args.sweep_jsonl):
        print(f'sweep jsonl not available, using the recorded best setting for '
              f'{library} as given', flush=True)
        return
    setting, nrmse, k = quality.best_setting(args.sweep_jsonl, library, 256)
    if setting != BEST_SETTING[library]:
        raise SystemExit(f'the sweep chose {setting} for {library}, not '
                         f'{BEST_SETTING[library]}')
    print(f'{library} best setting confirmed against the sweep: {setting} '
          f'(nrmse {nrmse:.5f} at k={k})', flush=True)


def best_of(run, sync, repeats=3):
    times = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        run()
        sync()
        times.append(time.perf_counter() - start)
    return min(times), times


# ── mbirtorch ────────────────────────────────────────────────────────────────
def mbirtorch_session(n, args):
    import torch
    import mbirtorch

    g = bench.geometry(n)
    device = torch.device('cuda:0')
    setting = BEST_SETTING['mbirtorch']

    def sync():
        torch.cuda.synchronize()

    recorder = bench.Recorder(args.jsonl, {
        'library': 'mbirtorch', 'library_version': mbirtorch.__version__,
        'torch_version': torch.__version__,
        'gpu_name': torch.cuda.get_device_name(0), 'N': n, 'setting': setting,
    })

    phantom = np.load(quality.paths(args.results_dir, n)['phantom'])
    sinogram = np.load(quality.paths(args.results_dir, n)['sinogram'])
    weights = np.load(quality.paths(args.results_dir, n)['weights'])
    mask = quality.cylinder_mask(n)[:, :, None]

    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float32)
    model = mbirtorch.ConeBeamModel((n, n, n), angles,
                                    source_detector_dist=g['source_detector_dist_mm'],
                                    source_iso_dist=g['source_iso_dist_mm'])
    model.set_params(delta_det_channel=g['delta_det_mm'],
                     delta_det_row=g['delta_det_mm'],
                     delta_voxel=g['delta_voxel_mm'], recon_shape=(n, n, n))
    model.configure_devices(num_devices=1)
    model.set_params(sharpness=setting['sharpness'])
    if setting.get('snr_db') is not None:
        model.set_params(snr_db=setting['snr_db'])

    sinogram_gpu = torch.as_tensor(sinogram, device=device)
    weights_gpu = torch.as_tensor(weights, device=device)

    def direct():
        return model.recon_fdk(sinogram_gpu, output_sharded=True)

    def iterate(initial, k, stop=0.0):
        np.random.seed(0)
        return model.recon(sinogram_gpu, weights=weights_gpu,
                           init_recon=initial, max_iterations=k,
                           stop_threshold_change_pct=stop, print_logs=False,
                           logfile_path=None, output_sharded=True)

    # Warm-up, discarded, but timed so the record says what it cost.
    sync()
    start = time.perf_counter()
    warm_direct = direct()
    sync()
    recorder.write(operation='warmup_discarded', what='direct_recon',
                   time_s=time.perf_counter() - start)
    for k in WARMUP_K:
        sync()
        start = time.perf_counter()
        iterate(warm_direct.clone(), k)
        sync()
        recorder.write(operation='warmup_discarded', what=f'recon_k{k}',
                       time_s=time.perf_counter() - start)
        torch.cuda.empty_cache()

    # The direct reconstruction, now measured warm, three times.
    holder = {}

    def timed_direct():
        holder['value'] = direct()

    direct_time, all_times = best_of(timed_direct, sync)
    direct_recon = holder['value']
    direct_nrmse = quality.masked_nrmse(direct_recon.cpu().numpy(), phantom, mask)
    recorder.write(operation='direct_recon_warm', time_s=direct_time,
                   all_times_s=all_times, nrmse=direct_nrmse)

    if args.mode == 'slices':
        save_slices(args, n, 'mbirtorch', 'direct', None, direct_recon.cpu().numpy(),
                    direct_nrmse, direct_time, direct_time, recorder)
        for k in SLICE_RUNS['mbirtorch']:
            sync()
            start = time.perf_counter()
            recon, _info = iterate(direct_recon.clone(), k)
            sync()
            elapsed = time.perf_counter() - start
            volume = recon.cpu().numpy()
            value = quality.masked_nrmse(volume, phantom, mask)
            save_slices(args, n, 'mbirtorch', 'iterative', k, volume, value,
                        elapsed, elapsed + direct_time, recorder)
            del recon
            torch.cuda.empty_cache()
        return

    for k in K_BY_SIZE[n]:
        sync()
        start = time.perf_counter()
        recon, _info = iterate(direct_recon.clone(), k)
        sync()
        elapsed = time.perf_counter() - start
        value = quality.masked_nrmse(recon.cpu().numpy(), phantom, mask)
        recorder.write(operation='iteration_point_warm', k=k, nrmse=value,
                       time_iterations_s=elapsed,
                       time_total_s=elapsed + direct_time,
                       direct_time_s=direct_time)
        del recon
        torch.cuda.empty_cache()


# ── LEAP ─────────────────────────────────────────────────────────────────────
def leap_session(n, args):
    import torch
    from leapctype import tomographicModels
    from leap_filter_sequence import filterSequence, TV

    g = bench.geometry(n)
    device = torch.device('cuda:0')
    setting = BEST_SETTING['leap']

    def sync():
        torch.cuda.synchronize()

    leapct = tomographicModels()
    leapct.set_gpu(0)
    recorder = bench.Recorder(args.jsonl, {
        'library': 'leap', 'library_version': str(leapct.version()),
        'torch_version': torch.__version__,
        'gpu_name': torch.cuda.get_device_name(0), 'N': n, 'setting': setting,
    })

    phis = np.ascontiguousarray(
        np.linspace(0.0, 360.0, n, endpoint=False), dtype=np.float32)
    leapct.set_conebeam(n, n, n, g['delta_det_mm'], g['delta_det_mm'],
                        0.5 * (n - 1), 0.5 * (n - 1), phis,
                        g['source_iso_dist_mm'], g['source_detector_dist_mm'])
    leapct.set_flatDetector()
    leapct.set_volume(n, n, n, voxelWidth=g['delta_voxel_mm'],
                      voxelHeight=g['delta_voxel_mm'])

    phantom_mbirtorch = np.load(quality.paths(args.results_dir, n)['phantom'])
    phantom = bench.phantom_for_leap(phantom_mbirtorch)
    del phantom_mbirtorch
    mask = quality.cylinder_mask(n)[None, :, :]

    sinogram_gpu = torch.as_tensor(quality.swap_sinogram_convention(
        np.load(quality.paths(args.results_dir, n)['sinogram'])),
        device=device).contiguous()
    weights_gpu = torch.as_tensor(quality.swap_sinogram_convention(
        np.load(quality.paths(args.results_dir, n)['weights'])),
        device=device).contiguous()

    result = torch.zeros(phantom.shape, dtype=torch.float32, device=device)

    def direct():
        leapct.FBP(sinogram_gpu, result)

    def iterate(start_volume, k):
        result.copy_(start_volume)
        filters = filterSequence(1.0)
        filters.append(TV(leapct, delta=setting['tv_delta'], p=1.2,
                          weight=setting['tv_weight']))
        leapct.RWLS(sinogram_gpu, result, k, filters=filters, W=weights_gpu,
                    preconditioner='SQS', nonnegativityConstraint=True)

    # The same warm-up protocol as mbirtorch, so the two are treated alike.
    # LEAP has nothing to compile, so these should cost what the timed runs do.
    sync()
    start = time.perf_counter()
    direct()
    sync()
    recorder.write(operation='warmup_discarded', what='direct_recon',
                   time_s=time.perf_counter() - start)
    warm_start = result.clone()
    for k in WARMUP_K:
        sync()
        start = time.perf_counter()
        iterate(warm_start, k)
        sync()
        recorder.write(operation='warmup_discarded', what=f'recon_k{k}',
                       time_s=time.perf_counter() - start)
    del warm_start

    direct_time, all_times = best_of(direct, sync)
    direct_recon = result.clone()
    direct_nrmse = quality.masked_nrmse(direct_recon.cpu().numpy(), phantom, mask)
    recorder.write(operation='direct_recon_warm', time_s=direct_time,
                   all_times_s=all_times, nrmse=direct_nrmse)

    if args.mode == 'slices':
        save_slices(args, n, 'leap', 'direct', None,
                    bench.leap_volume_to_mbirtorch(direct_recon.cpu().numpy()),
                    direct_nrmse, direct_time, direct_time, recorder)
        for k in SLICE_RUNS['leap']:
            sync()
            start = time.perf_counter()
            iterate(direct_recon, k)
            sync()
            elapsed = time.perf_counter() - start
            volume = result.cpu().numpy()
            value = quality.masked_nrmse(volume, phantom, mask)
            save_slices(args, n, 'leap', 'iterative', k,
                        bench.leap_volume_to_mbirtorch(volume), value,
                        elapsed, elapsed + direct_time, recorder)
        return

    for k in K_BY_SIZE[n]:
        sync()
        start = time.perf_counter()
        iterate(direct_recon, k)
        sync()
        elapsed = time.perf_counter() - start
        value = quality.masked_nrmse(result.cpu().numpy(), phantom, mask)
        recorder.write(operation='iteration_point_warm', k=k, nrmse=value,
                       time_iterations_s=elapsed,
                       time_total_s=elapsed + direct_time,
                       direct_time_s=direct_time)


# ── Part B: slices ───────────────────────────────────────────────────────────
def save_slices(args, n, library, kind, k, volume_mbirtorch_order, nrmse,
                iteration_time, total_time, recorder):
    """Write the planes of one volume as .npy and record what they are."""
    directory = os.path.join(args.results_dir, f'slices_{n}')
    os.makedirs(directory, exist_ok=True)
    tag = f'{library}_{kind}' + ('' if k is None else f'_k{k}')
    files = {}
    for name, axis, index, description in slice_plan(n):
        plane = take_slice(volume_mbirtorch_order, axis, index)
        path = os.path.join(directory, f'{tag}__{name}.npy')
        np.save(path, plane)
        files[name] = {'file': os.path.basename(path), 'axis': axis,
                       'index': index, 'description': description}
    recorder.write(operation='slice_set', which=library, kind=kind, k=k,
                   nrmse=nrmse, time_iterations_s=iteration_time,
                   time_total_s=total_time, tag=tag, files=files)
    print(f'saved slices for {tag}', flush=True)


def save_phantom_slices(n, args):
    directory = os.path.join(args.results_dir, f'slices_{n}')
    os.makedirs(directory, exist_ok=True)
    recorder = bench.Recorder(args.jsonl, {'library': 'phantom', 'N': n})
    phantom = np.load(quality.paths(args.results_dir, n)['phantom'])
    files = {}
    for name, axis, index, description in slice_plan(n):
        plane = take_slice(phantom, axis, index)
        path = os.path.join(directory, f'phantom__{name}.npy')
        np.save(path, plane)
        files[name] = {'file': os.path.basename(path), 'axis': axis,
                       'index': index, 'description': description}
    recorder.write(operation='slice_set', which='phantom', kind='truth', k=None,
                   nrmse=0.0, time_total_s=0.0, tag='phantom', files=files)


def compose(n, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    directory = os.path.join(args.results_dir, f'slices_{n}')
    records = [json.loads(l) for l in open(args.jsonl) if l.strip()]
    sets = {r['tag']: r for r in records if r.get('operation') == 'slice_set'}
    delta = bench.geometry(n)['delta_voxel_mm']
    plan = slice_plan(n)

    def load(tag, name):
        return np.load(os.path.join(directory, sets[tag]['files'][name]['file']))

    def title_for(tag):
        r = sets[tag]
        if r['which'] == 'phantom':
            return 'phantom (truth)'
        name = 'LEAP' if r['which'] == 'leap' else 'mbirtorch'
        if r['kind'] == 'direct':
            label = 'FBP' if r['which'] == 'leap' else 'recon_fdk'
            return (f'{name} {label}\nNRMSE {r["nrmse"]:.5f}, '
                    f'{r["time_total_s"]:.2f} s')
        return (f'{name} k = {r["k"]}\nNRMSE {r["nrmse"]:.5f}, '
                f'{r["time_total_s"]:.2f} s')

    # The inset region for each row: none on the axial row, the sphere itself
    # on each coronal row.  A coronal plane is (x, z), so the row index of the
    # plotted array is x and the column index is z.
    insets = {plan[0][0]: None}
    for ((cx, cy, cz), radius, label), (name, _axis, _index, _d) in zip(
            sphere_centres_mm(), plan[1:]):
        half = int(np.rint(1.9 * radius / delta))
        insets[name] = (mm_to_index(cx, n, delta), mm_to_index(cz, n, delta), half)

    def figure_for(tags, out_name, difference=False, vmax=None):
        rows = len(plan)
        figure, axes = plt.subplots(rows, len(tags),
                                    figsize=(3.5 * len(tags), 3.7 * rows),
                                    squeeze=False)
        for column, tag in enumerate(tags):
            for row, (name, _axis, _index, description) in enumerate(plan):
                panel = axes[row][column]
                image = load(tag, name)
                if difference:
                    image = np.abs(image - load('phantom', name))
                    lo, hi = 0.0, vmax
                    cmap = 'inferno'
                else:
                    lo, hi = GRAY_WINDOW
                    cmap = 'gray'
                panel.imshow(image, cmap=cmap, vmin=lo, vmax=hi,
                             interpolation='nearest')
                panel.set_xticks([])
                panel.set_yticks([])
                if row == 0:
                    panel.set_title(title_for(tag), fontsize=9)
                if column == 0:
                    panel.set_ylabel(description.split(';')[0].split(' through ')[0]
                                     + ('' if row == 0 else
                                        f'\nsphere {plan[row][0][-1]}'),
                                     fontsize=8)
                box = insets[name]
                if box is not None:
                    centre_row, centre_col, half = box
                    r0 = max(0, centre_row - half); r1 = min(n, centre_row + half)
                    c0 = max(0, centre_col - half); c1 = min(n, centre_col + half)
                    inset = panel.inset_axes([0.60, 0.02, 0.38, 0.38])
                    inset.imshow(image[r0:r1, c0:c1], cmap=cmap, vmin=lo,
                                 vmax=hi, interpolation='nearest')
                    inset.set_xticks([]); inset.set_yticks([])
                    for spine in inset.spines.values():
                        spine.set_edgecolor('tab:cyan'); spine.set_linewidth(1.4)
                    panel.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0,
                                              fill=False, edgecolor='tab:cyan',
                                              linewidth=1.0))
        window = (f'absolute difference from the phantom, common scale 0 to '
                  f'{vmax:.4f} per mm' if difference else
                  f'gray window {GRAY_WINDOW[0]} to {GRAY_WINDOW[1]} per mm')
        figure.suptitle(f'N = {n}, noisy cone beam, one H100 — {window}',
                        fontsize=10)
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        out = os.path.join(args.results_dir, out_name)
        figure.savefig(out, dpi=150)
        plt.close(figure)
        print(f'wrote {out}', flush=True)

    figure_for(['phantom', 'leap_iterative_k14', 'mbirtorch_iterative_k10'],
               f'quality_slices_matched_time_{n}.png')
    figure_for(['phantom', 'leap_iterative_k100', 'mbirtorch_iterative_k10'],
               f'quality_slices_matched_quality_{n}.png')
    figure_for(['phantom', 'leap_direct', 'mbirtorch_direct'],
               f'quality_slices_direct_{n}.png')

    # One common difference scale for every difference panel: the largest
    # 99.9th percentile over the panels being drawn, so no panel saturates.
    tags = ['leap_direct', 'mbirtorch_direct', 'leap_iterative_k14',
            'mbirtorch_iterative_k10', 'leap_iterative_k100']
    peak = 0.0
    for tag in tags:
        for name, _axis, _index, _d in plan:
            peak = max(peak, float(np.percentile(
                np.abs(load(tag, name) - load('phantom', name)), 99.9)))
    vmax = float(np.ceil(peak * 1000.0) / 1000.0)
    figure_for(tags, f'quality_slices_difference_{n}.png', difference=True,
               vmax=vmax)


# ── Part A figure ────────────────────────────────────────────────────────────
def plot_warm(n, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    records = [json.loads(l) for l in open(args.jsonl) if l.strip()]
    figure, axes = plt.subplots(figsize=(7.6, 4.8))
    colors = {'leap': 'tab:orange', 'mbirtorch': 'tab:blue'}
    bests = []
    for library in ('leap', 'mbirtorch'):
        points = [r for r in records if r.get('library') == library
                  and r.get('N') == n
                  and r.get('operation') == 'iteration_point_warm']
        if not points:
            continue
        points.sort(key=lambda r: r['k'])
        bests.append(min(r['nrmse'] for r in points))
        axes.plot([r['time_total_s'] for r in points],
                  [r['nrmse'] for r in points], 'o-', color=colors[library],
                  label=f'{library}, warm process')
        for r in points:
            if r['k'] in (1, 5, 10, 20, 100):
                axes.annotate(str(r['k']), (r['time_total_s'], r['nrmse']),
                              textcoords='offset points', xytext=(5, 4),
                              fontsize=7, color=colors[library])
        direct = [r for r in records if r.get('library') == library
                  and r.get('N') == n and r.get('operation') == 'direct_recon_warm']
        if direct:
            axes.plot([direct[0]['time_s']], [direct[0]['nrmse']], '*',
                      markersize=13, color=colors[library],
                      label=f'{library}, direct reconstruction')
    if len(bests) == 2:
        target = 1.02 * max(bests)
        axes.axhline(target, color='grey', linestyle=':', linewidth=1.2)
        axes.text(0.015, target, f'common target {target:.5f}', fontsize=8,
                  color='grey', va='bottom',
                  transform=axes.get_yaxis_transform())
    axes.set_xscale('log')
    axes.set_xlabel('wall time, seconds (direct reconstruction plus k iterations)')
    axes.set_ylabel('NRMSE against the phantom, inside the cylinder')
    axes.set_title(f'Noisy cone-beam reconstruction at N = {n}, one H100\n'
                   'warm process: no run pays a compilation cost')
    axes.grid(True, which='both', alpha=0.3)
    axes.legend(fontsize=8)
    figure.tight_layout()
    out = os.path.join(args.results_dir, f'quality_nrmse_vs_time_{n}.png')
    figure.savefig(out, dpi=160)
    print(f'wrote {out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True,
                        choices=['curves', 'slices', 'compose', 'plot', 'phantom-slices'])
    parser.add_argument('--lib', choices=['mbirtorch', 'leap'])
    parser.add_argument('--N', type=int, required=True)
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--jsonl', required=True)
    parser.add_argument('--sweep-jsonl', default=None,
                        help='the earlier study, used only to confirm the best setting')
    args = parser.parse_args()

    if args.mode in ('curves', 'slices'):
        check_setting(args.lib, args)
        if args.lib == 'mbirtorch':
            mbirtorch_session(args.N, args)
        else:
            leap_session(args.N, args)
    elif args.mode == 'phantom-slices':
        save_phantom_slices(args.N, args)
    elif args.mode == 'compose':
        compose(args.N, args)
    else:
        plot_warm(args.N, args)


if __name__ == '__main__':
    main()
