"""Reconstruct the ORNL Inconel scan with mbirtorch the way the collaborators'
comparison script does, and record time and memory.

The steps follow their script: build the cone-beam model with their
parameters, run the direct (FDK) reconstruction, then run recon from that
start with the statistical weights.  The device layout is left to the library
unless --num-devices is given, as their script leaves it.  Their call keeps
the default stop rule, so the run may end before the iteration limit; the
number of iterations actually run is recorded.

Peak GPU memory is sampled from NVML during each phase (the device total and
this process's share), and torch's own peak allocated and reserved counters
are printed after the reconstruction.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import mbirtorch
import ornl_data
from gpu_sampler import GpuSampler, host_peak_gib

OUT_DIR = '/scratch/gautschi/buzzard/leap_ornl/out/'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--view-step', type=int, default=1)
    ap.add_argument('--det-step', type=int, default=1)
    ap.add_argument('--iterations', type=int, default=100)
    ap.add_argument('--stop-pct', type=float, default=0.2,
                    help='stop_threshold_change_pct; 0.2 is the library default their script keeps')
    ap.add_argument('--num-devices', type=int, default=0, help='0 leaves the choice to the library')
    ap.add_argument('--tag', default='full')
    ap.add_argument('--save-volume', action='store_true')
    ap.add_argument('--fdk-repeats', type=int, default=2,
                    help='timed FDK runs; the collaborators averaged 25, the first here includes any warm-up')
    args = ap.parse_args()

    assert torch.cuda.is_available(), 'NOT ON GPU'
    print('library under test:', mbirtorch.__file__, flush=True)
    print('torch', torch.__version__, 'visible GPUs', torch.cuda.device_count(), flush=True)

    proj, weights, params = ornl_data.load(args.view_step, args.det_step)
    ornl_data.describe(params)
    cone = params['proj_params']['cone_params']
    vol = params['vol_params']
    mis = params['miscalib']
    angles = params['proj_params']['angles']
    recon_shape = (int(vol['n_vox_x']), int(vol['n_vox_y']), int(vol['n_vox_z']))
    sod = cone['src_orig']
    sdd = cone['src_orig'] + cone['orig_det']

    model = mbirtorch.ConeBeamModel(proj.shape, angles, source_detector_dist=sdd, source_iso_dist=sod)
    model.set_params(recon_shape=recon_shape, snr_db=30, sharpness=1.0, positivity_flag=False,
                     delta_det_channel=cone['pix_y'], delta_det_row=cone['pix_x'],
                     delta_voxel=vol['vox_xy'], det_channel_offset=-mis['delta_u'],
                     det_row_offset=-mis['delta_v'], use_ror_mask=False, verbose=1)
    if args.num_devices > 0:
        model.configure_devices(num_devices=args.num_devices)
    model.print_params()

    sampler = GpuSampler()
    results = dict(tag=args.tag, view_step=args.view_step, det_step=args.det_step,
                   sinogram_shape=list(proj.shape), recon_shape=list(recon_shape),
                   mbirtorch_commit=os.popen('git -C /scratch/gautschi/buzzard/leap_ornl/mbirtorch_src '
                                             'rev-parse --short HEAD').read().strip())

    results['fdk_seconds_each'] = []
    for _ in range(max(1, args.fdk_repeats)):
        t0 = time.time()
        fdk = model.recon_direct(proj)
        torch.cuda.synchronize()
        results['fdk_seconds_each'].append(time.time() - t0)
        print(f'mbirtorch FDK took {results["fdk_seconds_each"][-1]:.1f} s', flush=True)
    results['fdk_seconds'] = results['fdk_seconds_each'][-1]
    results['fdk_memory'] = sampler.report('FDK')
    sampler.reset()
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)

    t0 = time.time()
    recon, recon_dict = model.recon(proj, weights=weights, max_iterations=args.iterations,
                                    stop_threshold_change_pct=args.stop_pct, init_recon=fdk)
    torch.cuda.synchronize()
    results['recon_seconds'] = time.time() - t0
    print(f'mbirtorch recon took {results["recon_seconds"]:.1f} s', flush=True)
    results['recon_memory'] = sampler.report('recon')
    sampler.stop()
    gib = 1024 ** 3
    results['torch_peak_allocated_gib'] = [torch.cuda.max_memory_allocated(i) / gib
                                           for i in range(torch.cuda.device_count())]
    results['torch_peak_reserved_gib'] = [torch.cuda.max_memory_reserved(i) / gib
                                          for i in range(torch.cuda.device_count())]
    print('torch peak allocated (GiB):', ['%.2f' % x for x in results['torch_peak_allocated_gib']])
    print('torch peak reserved (GiB):', ['%.2f' % x for x in results['torch_peak_reserved_gib']])
    results['host_peak_rss_gib'] = host_peak_gib()
    print(f'host peak RSS {results["host_peak_rss_gib"]:.1f} GiB', flush=True)

    recon_params = dict(recon_dict['recon_params'])
    for key, value in recon_params.items():
        if hasattr(value, '__len__') and not isinstance(value, str):
            print(f'  recon_params.{key}: length {len(value)}')
            if key in ('fm_rmse', 'prior_loss', 'stop_pct', 'alpha_values'):
                results[key] = [float(v) for v in value]
        else:
            print(f'  recon_params.{key}: {value}')
    if 'fm_rmse' in results:
        results['iterations_run'] = len(results['fm_rmse'])
    results['notes'] = str(recon_dict.get('notes'))
    with open(os.path.join(OUT_DIR, f'mbirtorch_{args.tag}_recon_log.txt'), 'w') as f:
        f.write(str(recon_dict.get('recon_log')))

    results['recon_stats'] = dict(min=float(recon.min()), max=float(recon.max()),
                                  mean=float(recon.mean()))
    np.savez(os.path.join(OUT_DIR, f'mbirtorch_{args.tag}_central_slices.npz'),
             row=recon[recon.shape[0] // 2], column=recon[:, recon.shape[1] // 2],
             slice=recon[:, :, recon.shape[2] // 2])
    if args.save_volume:
        np.save(os.path.join(OUT_DIR, f'mbirtorch_{args.tag}_recon.npy'), recon.astype(np.float32))
    with open(os.path.join(OUT_DIR, f'mbirtorch_{args.tag}_results.json'), 'w') as f:
        json.dump(results, f, indent=1)
    print('MBIRTORCH RUN DONE', flush=True)


if __name__ == '__main__':
    main()
