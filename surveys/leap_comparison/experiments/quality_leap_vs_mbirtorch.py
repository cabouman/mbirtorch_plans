"""Noisy reconstruction quality against iteration count and wall time.

For one noisy cone-beam sinogram, this measures each library's error against
the phantom after exactly k iterations, for a list of k, at that library's best
regularization setting, and reports how many iterations and how much wall time
each needs to first reach a common target error.

The geometry, the phantom shape, the volume index conventions and the
sinogram alignment between the two libraries are all reused from
bench_leap_vs_mbirtorch.py, which established and measured them.

Modes:
  data       build the phantom, project it, add Poisson noise, save the noisy
             sinogram and the weights for one size
  mbirtorch  reconstruct with mbirtorch and record error against iteration
  leap       reconstruct with LEAP and record error against iteration
  plot       draw the error-against-wall-time figure from the JSON lines

All runs of one library at one size happen in a single process, so mbirtorch
pays its compilation once rather than once per k.
"""

import argparse
import json
import os
import time

import numpy as np

import bench_leap_vs_mbirtorch as bench

# The phantom of the timing study scaled to attenuation per mm: its three
# sphere values 1, 0.5 and 2 become 0.02, 0.01 and 0.04 per mm.  The longest
# line through it is then a few nepers, which is a realistic transmission scan.
ATTENUATION_SCALE = 0.02
PHOTON_COUNT = 10000.0          # I0, incident photons per detector pixel
NOISE_SEED = 1234

# The iteration counts each reconstruction is run to, independently.
K_VALUES = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]
K_VALUES_LARGE = [5, 10, 20, 40]

# The parameter sweeps, run at N = 256 only.
MBIRTORCH_SETTINGS = [
    {'sharpness': -1.0, 'snr_db': None},
    {'sharpness': 0.0, 'snr_db': None},
    {'sharpness': 1.0, 'snr_db': None},
    {'sharpness': 2.0, 'snr_db': None},
    {'sharpness': 0.0, 'snr_db': 25.0},
    {'sharpness': 0.0, 'snr_db': 35.0},
]
LEAP_SETTINGS = [{'tv_weight': w, 'tv_delta': d}
                 for d in (0.001, 0.01)
                 for w in (1e-2, 1e-1, 1e0, 1e1, 1e2)]


def paths(results_dir, n):
    return {
        'phantom': os.path.join(results_dir, f'qphantom_{n}.npy'),
        'sinogram': os.path.join(results_dir, f'qsino_{n}.npy'),
        'weights': os.path.join(results_dir, f'qweights_{n}.npy'),
        'clean': os.path.join(results_dir, f'qclean_{n}.npy'),
    }


def attenuation_phantom(n):
    """The timing study's phantom, in attenuation per mm, mbirtorch order."""
    return (ATTENUATION_SCALE * bench.make_phantom(n)).astype(np.float32)


def swap_sinogram_convention(sinogram):
    """Convert a sinogram between mbirtorch's and LEAP's conventions.

    The timing study measured the alignment between the two libraries at
    N = 256: LEAP's view j matches mbirtorch's view i when j = (N/2 - i) mod N,
    that is phi_LEAP = 180 degrees - phi_mbirtorch, together with a reversal of
    the detector column index and no change to the detector rows.  That map is
    its own inverse, so this one function converts in both directions.  Its
    correctness at each size is checked, not assumed: the LEAP process compares
    its own forward projection of the phantom against the mapped mbirtorch one
    and records the error.
    """
    num_views = sinogram.shape[0]
    out = sinogram[:, :, ::-1][::-1]
    out = np.roll(out, num_views // 2 + 1, axis=0)
    return np.ascontiguousarray(out)


def cylinder_mask(n):
    """True inside the inscribed cylinder of radius N/2 - 2 voxels."""
    axis = np.arange(n) - (n - 1) / 2.0
    radius_squared = axis[:, None] ** 2 + axis[None, :] ** 2
    return radius_squared <= (n / 2.0 - 2.0) ** 2


def masked_nrmse(volume, phantom, mask_expanded, chunk=32):
    """||volume - phantom|| / ||phantom|| inside the mask, chunked over axis 0.

    Chunking keeps the float64 accumulation off the whole volume at once, which
    matters at N = 1024 where a float64 copy would be 8 GB.
    """
    error = 0.0
    norm = 0.0
    for start in range(0, volume.shape[0], chunk):
        stop = min(start + chunk, volume.shape[0])
        mask = mask_expanded[start:stop] if mask_expanded.shape[0] > 1 else mask_expanded
        a = volume[start:stop].astype(np.float64)
        b = phantom[start:stop].astype(np.float64)
        difference = (a - b) * mask
        error += float(np.sum(difference * difference))
        truth = b * mask
        norm += float(np.sum(truth * truth))
    return float(np.sqrt(error / norm)) if norm > 0 else float('nan')


# ── data ─────────────────────────────────────────────────────────────────────
def run_data(n, args):
    """Project the phantom with mbirtorch and add transmission noise.

    The noiseless sinogram comes from mbirtorch's projector rather than LEAP's.
    That is an inverse crime, but a symmetric one: the timing study measured the
    two forward projectors to agree to 0.05 percent at N = 256, so neither
    library is reconstructing data its own projector made and the other's did
    not.  Whichever projector had been used, the same argument applies.
    """
    import torch
    import mbirtorch

    g = bench.geometry(n)
    device = torch.device('cuda:0')
    recorder = bench.Recorder(args.jsonl, {'library': 'data', 'N': n})

    phantom = attenuation_phantom(n)
    np.save(paths(args.results_dir, n)['phantom'], phantom)

    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float32)
    model = mbirtorch.ConeBeamModel((n, n, n), angles,
                                    source_detector_dist=g['source_detector_dist_mm'],
                                    source_iso_dist=g['source_iso_dist_mm'])
    model.set_params(delta_det_channel=g['delta_det_mm'],
                     delta_det_row=g['delta_det_mm'],
                     delta_voxel=g['delta_voxel_mm'], recon_shape=(n, n, n))
    model.configure_devices(num_devices=1)

    clean = model.forward_project(torch.as_tensor(phantom, device=device),
                                  output_sharded=True)
    torch.cuda.synchronize()
    maximum_line_integral = float(clean.max().item())

    # counts ~ Poisson(I0 exp(-p)); the attenuation estimate is -log(counts/I0)
    # with counts floored at 1 so the log is finite, and the weight is the
    # transmission counts/I0, which is the reciprocal of the variance of that
    # estimate up to the constant I0.
    generator = torch.Generator(device=device).manual_seed(NOISE_SEED)
    rate = PHOTON_COUNT * torch.exp(-clean)
    counts = torch.poisson(rate, generator=generator)
    del rate
    counts = torch.clamp(counts, min=1.0)
    noisy = -torch.log(counts / PHOTON_COUNT)
    weights = counts / PHOTON_COUNT
    torch.cuda.synchronize()

    np.save(paths(args.results_dir, n)['clean'], clean.cpu().numpy())
    np.save(paths(args.results_dir, n)['sinogram'], noisy.cpu().numpy())
    np.save(paths(args.results_dir, n)['weights'], weights.cpu().numpy())
    recorder.write(operation='data', photon_count=PHOTON_COUNT, seed=NOISE_SEED,
                   max_line_integral_nepers=maximum_line_integral,
                   clean_sino_mean=float(clean.mean().item()),
                   noisy_sino_mean=float(noisy.mean().item()),
                   min_counts=float(counts.min().item()),
                   attenuation_values=[0.02, 0.01, 0.04])
    print(f'wrote noisy data for N={n}', flush=True)


def best_setting(jsonl_path, library, sweep_n):
    """The setting with the lowest error at any k in the sweep at sweep_n."""
    best = None
    with open(jsonl_path) as handle:
        for line in handle:
            record = json.loads(line)
            if (record.get('library') != library or record.get('N') != sweep_n
                    or record.get('operation') != 'iteration_point'
                    or record.get('phase') != 'sweep'):
                continue
            if best is None or record['nrmse'] < best['nrmse']:
                best = record
    if best is None:
        raise SystemExit(f'no sweep records for {library} at N={sweep_n} in {jsonl_path}')
    return best['setting'], best['nrmse'], best['k']


# ── mbirtorch ────────────────────────────────────────────────────────────────
def run_mbirtorch(n, args):
    import torch
    import mbirtorch

    g = bench.geometry(n)
    device = torch.device('cuda:0')
    uuids = bench.gpu_uuids()
    k_values = args.k_values

    recorder = bench.Recorder(args.jsonl, {
        'library': 'mbirtorch', 'library_version': mbirtorch.__version__,
        'torch_version': torch.__version__,
        'gpu_name': torch.cuda.get_device_name(0), 'N': n,
    })

    phantom = np.load(paths(args.results_dir, n)['phantom'])
    sinogram = np.load(paths(args.results_dir, n)['sinogram'])
    weights = np.load(paths(args.results_dir, n)['weights'])
    mask = cylinder_mask(n)[:, :, None]          # mbirtorch order is (y, x, z)

    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float32)
    model = mbirtorch.ConeBeamModel((n, n, n), angles,
                                    source_detector_dist=g['source_detector_dist_mm'],
                                    source_iso_dist=g['source_iso_dist_mm'])
    model.set_params(delta_det_channel=g['delta_det_mm'],
                     delta_det_row=g['delta_det_mm'],
                     delta_voxel=g['delta_voxel_mm'], recon_shape=(n, n, n))
    model.configure_devices(num_devices=1)
    recorder.write(operation='default_regularization',
                   snr_db=float(model.get_params('snr_db')),
                   sharpness=float(model.get_params('sharpness')))

    sinogram_gpu = torch.as_tensor(sinogram, device=device)
    weights_gpu = torch.as_tensor(weights, device=device)

    # The direct reconstruction, which is also both libraries' starting point.
    torch.cuda.synchronize()
    start = time.perf_counter()
    direct = model.recon_fdk(sinogram_gpu, output_sharded=True)
    torch.cuda.synchronize()
    direct_time = time.perf_counter() - start
    direct_nrmse = masked_nrmse(direct.cpu().numpy(), phantom, mask)
    recorder.write(operation='direct_recon', time_s=direct_time,
                   nrmse=direct_nrmse)

    settings = args.settings_json or (
        MBIRTORCH_SETTINGS if args.phase == 'sweep' else [args.setting])
    for setting in settings:
        model.set_params(sharpness=setting['sharpness'])
        if setting.get('snr_db') is not None:
            model.set_params(snr_db=setting['snr_db'])
        for k in k_values:
            np.random.seed(0)
            initial = direct.clone()
            torch.cuda.synchronize()
            start = time.perf_counter()
            recon, _info = model.recon(sinogram_gpu, weights=weights_gpu,
                                       init_recon=initial, max_iterations=k,
                                       stop_threshold_change_pct=0.0,
                                       print_logs=False, logfile_path=None,
                                       output_sharded=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            value = masked_nrmse(recon.cpu().numpy(), phantom, mask)
            recorder.write(operation='iteration_point', phase=args.phase,
                           setting=setting, k=k, nrmse=value,
                           time_iterations_s=elapsed,
                           time_total_s=elapsed + direct_time,
                           direct_time_s=direct_time)
            del recon, initial
            torch.cuda.empty_cache()

    if args.phase != 'sweep':
        # Steady-state cost per iteration: three k = 10 runs in this same
        # process, so only the first can carry any compilation.
        for repeat in range(args.steady_repeats):
            np.random.seed(0)
            initial = direct.clone()
            torch.cuda.synchronize()
            start = time.perf_counter()
            model.recon(sinogram_gpu, weights=weights_gpu, init_recon=initial,
                        max_iterations=10, stop_threshold_change_pct=0.0,
                        print_logs=False, logfile_path=None, output_sharded=True)
            torch.cuda.synchronize()
            recorder.write(operation='steady_state', k=10, repeat=repeat,
                           setting=args.setting,
                           time_iterations_s=time.perf_counter() - start)
            torch.cuda.empty_cache()

        # What the library does when its own stopping rule is left in place.
        np.random.seed(0)
        initial = direct.clone()
        torch.cuda.synchronize()
        start = time.perf_counter()
        recon, info = model.recon(sinogram_gpu, weights=weights_gpu,
                                  init_recon=initial, max_iterations=100,
                                  print_logs=False, logfile_path=None,
                                  output_sharded=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        value = masked_nrmse(recon.cpu().numpy(), phantom, mask)
        params = info.get('recon_params', {})
        lengths = {key: len(val) for key, val in params.items()
                   if isinstance(val, (list, tuple))}
        completed = max(lengths.values()) if lengths else None
        recorder.write(operation='default_stop_rule', setting=args.setting,
                       max_iterations=100, iterations_completed=completed,
                       trace_lengths=lengths, nrmse=value,
                       time_iterations_s=elapsed,
                       time_total_s=elapsed + direct_time)


# ── LEAP ─────────────────────────────────────────────────────────────────────
def run_leap(n, args):
    import torch
    from leapctype import tomographicModels
    from leap_filter_sequence import filterSequence, TV

    g = bench.geometry(n)
    device = torch.device('cuda:0')
    k_values = args.k_values

    leapct = tomographicModels()
    leapct.set_gpu(0)
    recorder = bench.Recorder(args.jsonl, {
        'library': 'leap', 'library_version': str(leapct.version()),
        'torch_version': torch.__version__,
        'gpu_name': torch.cuda.get_device_name(0), 'N': n,
    })

    phis = np.ascontiguousarray(
        np.linspace(0.0, 360.0, n, endpoint=False), dtype=np.float32)
    leapct.set_conebeam(n, n, n, g['delta_det_mm'], g['delta_det_mm'],
                        0.5 * (n - 1), 0.5 * (n - 1), phis,
                        g['source_iso_dist_mm'], g['source_detector_dist_mm'])
    leapct.set_flatDetector()
    leapct.set_volume(n, n, n, voxelWidth=g['delta_voxel_mm'],
                      voxelHeight=g['delta_voxel_mm'])

    phantom_mbirtorch = np.load(paths(args.results_dir, n)['phantom'])
    phantom = bench.phantom_for_leap(phantom_mbirtorch)     # LEAP order (z, y, x)
    del phantom_mbirtorch
    mask = cylinder_mask(n)[None, :, :]

    sinogram = swap_sinogram_convention(np.load(paths(args.results_dir, n)['sinogram']))
    weights = swap_sinogram_convention(np.load(paths(args.results_dir, n)['weights']))
    sinogram_gpu = torch.as_tensor(sinogram, device=device).contiguous()
    weights_gpu = torch.as_tensor(weights, device=device).contiguous()
    del sinogram, weights

    # Check the alignment at THIS size rather than trusting the N = 256 result:
    # project the phantom with LEAP and compare against the mapped mbirtorch
    # sinogram of the same phantom, which the data step wrote before noise.
    check_volume = torch.as_tensor(phantom, device=device).contiguous()
    check_projection = torch.zeros_like(sinogram_gpu)
    leapct.project(check_projection, check_volume)
    torch.cuda.synchronize()
    reference = swap_sinogram_convention(
        np.load(paths(args.results_dir, n)['clean']))
    everywhere = np.ones((1, 1, 1), dtype=bool)
    recorder.write(operation='alignment_check',
                   note="LEAP's own forward projection of the phantom against "
                        "mbirtorch's noiseless one, mapped into LEAP's "
                        'convention; this checks the view and detector mapping '
                        'at this size instead of assuming the N = 256 result',
                   nrmse=masked_nrmse(check_projection.cpu().numpy(), reference,
                                      everywhere))
    del check_volume, check_projection, reference
    torch.cuda.empty_cache()

    result = torch.zeros(phantom.shape, dtype=torch.float32, device=device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    leapct.FBP(sinogram_gpu, result)
    torch.cuda.synchronize()
    direct_time = time.perf_counter() - start
    direct = result.clone()
    direct_nrmse = masked_nrmse(direct.cpu().numpy(), phantom, mask)
    recorder.write(operation='direct_recon', time_s=direct_time, nrmse=direct_nrmse)

    settings = args.settings_json or (
        LEAP_SETTINGS if args.phase == 'sweep' else [args.setting])
    for setting in settings:
        for k in k_values:
            result.copy_(direct)
            filters = filterSequence(1.0)
            filters.append(TV(leapct, delta=setting['tv_delta'], p=1.2,
                              weight=setting['tv_weight']))
            torch.cuda.synchronize()
            start = time.perf_counter()
            leapct.RWLS(sinogram_gpu, result, k, filters=filters, W=weights_gpu,
                        preconditioner='SQS', nonnegativityConstraint=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            value = masked_nrmse(result.cpu().numpy(), phantom, mask)
            recorder.write(operation='iteration_point', phase=args.phase,
                           setting=setting, k=k, nrmse=value,
                           time_iterations_s=elapsed,
                           time_total_s=elapsed + direct_time,
                           direct_time_s=direct_time)

    if args.phase != 'sweep':
        for repeat in range(args.steady_repeats):
            result.copy_(direct)
            filters = filterSequence(1.0)
            filters.append(TV(leapct, delta=args.setting['tv_delta'], p=1.2,
                              weight=args.setting['tv_weight']))
            torch.cuda.synchronize()
            start = time.perf_counter()
            leapct.RWLS(sinogram_gpu, result, 10, filters=filters, W=weights_gpu,
                        preconditioner='SQS', nonnegativityConstraint=True)
            torch.cuda.synchronize()
            recorder.write(operation='steady_state', k=10, repeat=repeat,
                           setting=args.setting,
                           time_iterations_s=time.perf_counter() - start)


# ── figure ───────────────────────────────────────────────────────────────────
def run_plot(n, args):
    """Error against wall time, both libraries, log time axis.

    Two curves per library.  The solid one is the wall time actually measured,
    direct reconstruction plus k iterations.  The dashed one removes the
    one-time compilation that mbirtorch pays inside whichever run happens to
    come first in a process: it is k times the steady-state cost of an
    iteration plus the direct reconstruction.  Both are drawn because the solid
    curve is what a user waits for the first time and the dashed one is what
    the same work costs once the process is warm.

    Points are joined in order of increasing k, not increasing time.  Ordering
    by time would draw a curve that doubles back on itself wherever a small-k
    run paid the compilation that a larger-k run did not.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    records = [json.loads(line) for line in open(args.jsonl) if line.strip()]
    figure, axes = plt.subplots(figsize=(7.6, 4.8))
    colors = {'leap': 'tab:orange', 'mbirtorch': 'tab:blue'}
    for library in ('leap', 'mbirtorch'):
        points = [r for r in records
                  if r.get('library') == library and r.get('N') == n
                  and r.get('operation') == 'iteration_point'
                  and r.get('phase') == 'main']
        if not points:
            continue
        points.sort(key=lambda r: r['k'])
        direct = [r for r in records if r.get('library') == library
                  and r.get('N') == n and r.get('operation') == 'direct_recon']
        steady = [r for r in records if r.get('library') == library
                  and r.get('N') == n and r.get('operation') == 'steady_state']
        axes.plot([r['time_total_s'] for r in points],
                  [r['nrmse'] for r in points], 'o-', color=colors[library],
                  label=f'{library}, measured')
        if direct and steady:
            per_iteration = max(steady, key=lambda r: r['repeat'])['time_iterations_s'] / 10.0
            axes.plot([r['k'] * per_iteration + direct[0]['time_s'] for r in points],
                      [r['nrmse'] for r in points], 's--', color=colors[library],
                      alpha=0.55, markersize=4,
                      label=f'{library}, compilation removed')
        for r in points:
            if r['k'] in (1, 5, 20, 100, 40):
                axes.annotate(str(r['k']), (r['time_total_s'], r['nrmse']),
                              textcoords='offset points', xytext=(5, 4),
                              fontsize=7, color=colors[library])
        if direct:
            axes.plot([direct[0]['time_s']], [direct[0]['nrmse']], '*',
                      markersize=13, color=colors[library],
                      label=f'{library}, direct reconstruction')

    # The common target: 1.02 times the larger of the two libraries' best.
    bests = []
    for library in ('leap', 'mbirtorch'):
        values = [r['nrmse'] for r in records if r.get('library') == library
                  and r.get('N') == n and r.get('operation') == 'iteration_point'
                  and r.get('phase') == 'main']
        if values:
            bests.append(min(values))
    if len(bests) == 2:
        target = 1.02 * max(bests)
        axes.axhline(target, color='grey', linestyle=':', linewidth=1.2)
        axes.text(0.015, target, f'common target {target:.5f}', fontsize=8,
                  color='grey', va='bottom', transform=axes.get_yaxis_transform())
    axes.set_xscale('log')
    axes.set_xlabel('wall time, seconds (direct reconstruction plus k iterations)')
    axes.set_ylabel('NRMSE against the phantom, inside the cylinder')
    axes.set_title(f'Noisy cone-beam reconstruction at N = {n}, one H100\n'
                   'points are iteration counts k')
    axes.grid(True, which='both', alpha=0.3)
    axes.legend(fontsize=8)
    figure.tight_layout()
    out = os.path.join(args.results_dir, f'quality_nrmse_vs_time_{n}.png')
    figure.savefig(out, dpi=160)
    print(f'wrote {out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True,
                        choices=['data', 'mbirtorch', 'leap', 'plot'])
    parser.add_argument('--N', type=int, required=True)
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--jsonl', required=True)
    parser.add_argument('--phase', choices=['sweep', 'main'], default='sweep')
    parser.add_argument('--sweep-n', type=int, default=256,
                        help='the size whose sweep chooses the setting for --phase main')
    parser.add_argument('--k-set', choices=['full', 'large'], default='full')
    parser.add_argument('--steady-repeats', type=int, default=3)
    parser.add_argument('--settings-json', default=None,
                        help='a JSON list of settings that replaces the built-in '
                             'sweep grid, for probing outside it')
    args = parser.parse_args()
    args.k_values = K_VALUES if args.k_set == 'full' else K_VALUES_LARGE
    args.settings_json = json.loads(args.settings_json) if args.settings_json else None
    args.setting = None
    if args.phase == 'main' and args.mode in ('mbirtorch', 'leap'):
        setting, nrmse, k = best_setting(args.jsonl, args.mode, args.sweep_n)
        args.setting = setting
        print(f'best {args.mode} setting from the N={args.sweep_n} sweep: '
              f'{setting} (nrmse {nrmse:.5f} at k={k})', flush=True)

    if args.mode == 'data':
        run_data(args.N, args)
    elif args.mode == 'mbirtorch':
        run_mbirtorch(args.N, args)
    elif args.mode == 'leap':
        run_leap(args.N, args)
    else:
        run_plot(args.N, args)


if __name__ == '__main__':
    main()
