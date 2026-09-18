"""Reconstruct the ORNL Inconel scan with LEAP the way the collaborators'
comparison script does, and record time and memory.

The steps follow their script: set LEAP's cone-beam geometry and volume from
their parameters, run FBP (LEAP's name for FDK), then run their own MBIR loop
(utils.MBIR_LEAP_optim.MBIR: voxel-wise preconditioned Nesterov OGM2 on a
weighted least-squares cost with LEAP's anisotropic TV regularizer) for the
given number of iterations from the FBP start.  All arrays stay in host
memory; LEAP streams chunks of them through the GPUs inside each projection
call.  The XrayPhysics import of their script is dropped because it served
only the beam-hardening correction, which is already applied to the tiff.

LEAP's debug log level is switched on for the FBP and the first two
iterations so the chunking decisions are recorded, then switched back so the
remaining iterations log only the loop's own progress line.
"""
import argparse
import json
import os
import time

import numpy as np
from leapctype import tomographicModels

import ornl_data
from gpu_sampler import GpuSampler, host_peak_gib
from utils.MBIR_LEAP_optim import MBIR

OUT_DIR = '/scratch/gautschi/buzzard/leap_ornl/out/'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--view-step', type=int, default=1)
    ap.add_argument('--det-step', type=int, default=1)
    ap.add_argument('--iterations', type=int, default=100)
    ap.add_argument('--beta', type=float, default=0.7)
    ap.add_argument('--tag', default='full')
    ap.add_argument('--save-volume', action='store_true')
    ap.add_argument('--max-slices', type=int, default=0,
                    help='set_maxSlicesForChunking; 0 keeps LEAP\'s default of 128')
    ap.add_argument('--fdk-repeats', type=int, default=2,
                    help='timed FBP runs; the collaborators averaged 25')
    ap.add_argument('--gpus', type=int, default=0,
                    help='use only the first this many GPUs; 0 keeps LEAP\'s default of all')
    args = ap.parse_args()

    leapct = tomographicModels()
    leapct.about()
    # number_of_gpus counts the devices on the machine; get_gpus lists the ones in use.
    print('LEAP GPUs on the machine:', leapct.number_of_gpus(), flush=True)
    if args.gpus > 0:
        leapct.set_gpus(list(range(args.gpus)))
    print('LEAP GPUs in use:', list(leapct.get_gpus()), flush=True)
    if args.max_slices > 0:
        leapct.set_maxSlicesForChunking(args.max_slices)

    proj, weights, params = ornl_data.load(args.view_step, args.det_step)
    ornl_data.describe(params)
    cone = params['proj_params']['cone_params']
    vol = params['vol_params']
    mis = params['miscalib']
    dims = params['proj_params']['dims']
    numRows, numCols = int(dims[0]), int(dims[2])
    scan_angles = (params['proj_params']['angles'] * 180 / np.pi).astype('float32')
    numAngles = len(scan_angles)
    pixelWidth, pixelHeight = cone['pix_x'], cone['pix_y']
    d_so = cone['src_orig']
    d_sd = d_so + cone['orig_det']
    centerRow = numRows / 2 - mis['delta_v'] / pixelHeight - 0.5
    centerCol = numCols / 2 - mis['delta_u'] / pixelWidth - 0.5
    magnif = d_sd / d_so
    voxelWidth, voxelHeight = pixelWidth / magnif, pixelHeight / magnif

    leapct.set_conebeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol,
                        scan_angles, d_so, d_sd)
    leapct.set_volume(numX=int(vol['n_vox_x']), numY=int(vol['n_vox_y']), numZ=int(vol['n_vox_z']),
                      voxelWidth=voxelWidth, voxelHeight=voxelHeight)
    leapct.set_diameterFOV(1.0e16)
    leapct.print_parameters()

    TV_params = dict(delta=1e-4, beta=args.beta, p=1.2)
    rec_params = params['rec_params']
    rec_params['num_iter'] = args.iterations
    rec_params['debug'] = False
    rec_params['verbose'] = True

    sampler = GpuSampler()
    results = dict(tag=args.tag, view_step=args.view_step, det_step=args.det_step,
                   sinogram_shape=list(proj.shape),
                   recon_shape=[int(vol['n_vox_z']), int(vol['n_vox_y']), int(vol['n_vox_x'])],
                   beta=args.beta, leap_gpus_in_use=[int(g) for g in leapct.get_gpus()],
                   max_slices=args.max_slices)

    leapct.set_log_debug()
    results['fdk_seconds_each'] = []
    for _ in range(max(1, args.fdk_repeats)):
        t0 = time.time()
        f = leapct.FBP(proj)
        results['fdk_seconds_each'].append(time.time() - t0)
        print(f'LEAP FDK took {results["fdk_seconds_each"][-1]:.1f} s', flush=True)
    results['fdk_seconds'] = results['fdk_seconds_each'][-1]
    results['fdk_memory'] = sampler.report('FDK')
    sampler.reset()

    rec_params['x_init'] = f
    # Time LEAP's three GPU entry points from the loop so the remainder of an
    # iteration, which is host arithmetic in their loop, can be attributed.
    # Debug logging stays on for the Lipschitz projection and two iterations
    # (three project calls), then LEAP goes back to its status level.
    timers = {'project': 0.0, 'backproject': 0.0, 'TVgradient': 0.0}
    calls = {'project': 0, 'backproject': 0, 'TVgradient': 0}

    def timed(name, fn):
        def wrapper(*a, **k):
            t = time.time()
            out = fn(*a, **k)
            timers[name] += time.time() - t
            calls[name] += 1
            if name == 'project' and calls[name] == 3:
                leapct.set_log_status()
            return out
        return wrapper

    leapct.project = timed('project', leapct.project)
    leapct.backproject = timed('backproject', leapct.backproject)
    leapct.TVgradient = timed('TVgradient', leapct.TVgradient)
    t0 = time.time()
    x, cost = MBIR(proj, weights, leapct, rec_params, vol, TV_params)
    results['recon_seconds'] = time.time() - t0
    print(f'LEAP MBIR took {results["recon_seconds"]:.1f} s for {args.iterations} iterations',
          flush=True)
    results['recon_memory'] = sampler.report('MBIR')
    results['gpu_call_seconds'] = dict(timers)
    results['gpu_calls'] = dict(calls)
    print('time inside LEAP calls (s):', {k: round(v, 1) for k, v in timers.items()},
          'calls:', calls, flush=True)
    sampler.stop()
    results['host_peak_rss_gib'] = host_peak_gib()
    print(f'host peak RSS {results["host_peak_rss_gib"]:.1f} GiB', flush=True)
    results['iterations_run'] = args.iterations
    results['recon_stats'] = dict(min=float(x.min()), max=float(x.max()), mean=float(x.mean()))
    np.savez(os.path.join(OUT_DIR, f'leap_{args.tag}_central_slices.npz'),
             z=x[x.shape[0] // 2], y=x[:, x.shape[1] // 2], x=x[:, :, x.shape[2] // 2])
    if args.save_volume:
        np.save(os.path.join(OUT_DIR, f'leap_{args.tag}_recon.npy'), x.astype(np.float32))
    with open(os.path.join(OUT_DIR, f'leap_{args.tag}_results.json'), 'w') as f_out:
        json.dump(results, f_out, indent=1)
    print('LEAP RUN DONE', flush=True)


if __name__ == '__main__':
    main()
