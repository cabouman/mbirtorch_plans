"""Denoise one frame of the real MACE4D initial image, as a call of denoise would.

This is the one-GPU part of the GPU test of the stopping rule.  The frame is a
direct reconstruction (FDK) of 48 views of the Lilly 4DCT phantom scan, taken
from the initial image that the MACE4D runs share.  The script calls denoise
with the automatic regularization, on one device, and times two parts: the
initialization, which estimates the noise level and sigma_x and draws the
pixel partition, and the sweep.  A first sweep of one iteration compiles the
update and is not timed.

With --rule library, denoise keeps its defaults: at most 15 iterations, and a
stop at a change of 0.2 percent.  With --rule gradient, it stops on the
gradient statistic at --gradient_threshold, with a cap of
--max_iterations, which needs the patched library (gradient_rule.patch).
--sigma_noise robust sets the noise level to option 2's estimate from the
frame: the median absolute difference of adjacent voxels divided by
0.6745 sqrt(2), as in robust_sigma.py.

The output is written as a sample in the format of mace4d_stopping.py, so that
mace4d_calls.py computes its distance from the MAP estimate:
  <out_dir>/samples/frame_<label>.npz   the frame, the output, the partition,
                                        and the parameters
  <out_dir>/frame_<label>.json          the times and the settings

Run on one GPU with mbirtorch installed:
    python frame_check.py --init <init_dir>/init_recon.npy --out_dir <dir> --label B
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

import mbirtorch as mt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mace4d_stopping import check_library, json_value, library_state  # noqa: E402
from robust_sigma import MAD_TO_SIGMA, median_absolute_difference  # noqa: E402

SEED = 0


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--init', required=True, help='init_recon.npy, of shape (frames, x, y, z).')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--frame', type=int, default=None, help='Defaults to the middle frame.')
    parser.add_argument('--rule', choices=['library', 'gradient'], default='library')
    parser.add_argument('--gradient_threshold', type=float, default=0.01)
    parser.add_argument('--max_iterations', type=int, default=None,
                        help='Defaults to 15 for the library rule and 200 for the gradient rule.')
    parser.add_argument('--sigma_noise', default=None, help="A value, 'robust', or omitted for automatic.")
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    check_library()
    state = library_state()
    print('library under test:', state, flush=True)
    init = np.load(args.init, mmap_mode='r')
    frame_index = init.shape[0] // 2 if args.frame is None else args.frame
    frame = np.array(init[frame_index], dtype=np.float32)
    if args.sigma_noise is None:
        sigma_noise = None
    elif args.sigma_noise == 'robust':
        sigma_noise = median_absolute_difference([frame]) * MAD_TO_SIGMA
    else:
        sigma_noise = float(args.sigma_noise)

    denoiser = mt.QGGMRFDenoiser(frame.shape)
    denoiser.configure_devices(devices=[args.device])
    denoiser.set_params(no_warning=True, verbose=0)

    np.random.seed(SEED)              # the partition is drawn from the global state
    synchronize(args.device)
    tick = time.time()
    denoiser.initialize_denoiser(image=frame, sigma_noise=sigma_noise)
    synchronize(args.device)
    init_seconds = time.time() - tick
    partition = denoiser.denoise_data['partition']

    tick = time.time()
    denoiser.denoise(frame, max_iterations=1, stop_threshold_change_pct=0.0, logfile_path=None,
                     print_logs=False, do_initialization=False)
    synchronize(args.device)
    compile_seconds = time.time() - tick

    if args.rule == 'library':
        max_iterations = 15 if args.max_iterations is None else args.max_iterations
        rule = dict(stop_threshold_change_pct=0.2)
    else:
        max_iterations = 200 if args.max_iterations is None else args.max_iterations
        rule = dict(stop_threshold_gradient=args.gradient_threshold)
    synchronize(args.device)
    tick = time.time()
    output, info = denoiser.denoise(frame, max_iterations=max_iterations, logfile_path=None,
                                    print_logs=False, do_initialization=False, **rule)
    synchronize(args.device)
    sweep_seconds = time.time() - tick
    output = np.asarray(output, dtype=np.float32)

    names = ['sigma_y', 'sigma_x', 'p', 'q', 'T', 'qggmrf_nbr_wts', 'granularity', 'partition_sequence',
             'use_ror_mask']
    params = {n: json_value(v) for n, v in zip(names, denoiser.get_params(names))}
    iterations = int(info['recon_params']['num_iterations'])
    # denoise returns the percent change of each iteration under this name.
    percent = [float(v) for v in info['recon_params']['stop_threshold_change_pct']]
    meta = dict(label=f'frame_{args.label}', iteration=0, orientation='frame', plane=frame_index,
                rule=args.rule, gradient_threshold=args.gradient_threshold if args.rule == 'gradient' else None,
                max_iterations=max_iterations, iterations=iterations, percent_change=percent,
                gradient_statistic=info.get('gradient_statistic'), warm_start=False, **params)
    os.makedirs(os.path.join(args.out_dir, 'samples'), exist_ok=True)
    np.savez(os.path.join(args.out_dir, 'samples', f'frame_{args.label}.npz'), meta=json.dumps(meta),
             input=frame, output=output, partition=partition.cpu().numpy().astype(np.int64))
    record = dict(label=args.label, library=state, init=args.init, frame=frame_index, shape=list(frame.shape),
                  device=args.device, sigma_noise_setting=args.sigma_noise, sigma_noise=sigma_noise,
                  sigma_y=params['sigma_y'], sigma_x=params['sigma_x'],
                  r=params['sigma_x'] / params['sigma_y'], rule=args.rule, max_iterations=max_iterations,
                  iterations=iterations, init_seconds=init_seconds, compile_seconds=compile_seconds,
                  sweep_seconds=sweep_seconds, seconds_per_iteration=sweep_seconds / max(iterations, 1),
                  percent_change=percent, gradient_statistic=info.get('gradient_statistic'))
    with open(os.path.join(args.out_dir, f'frame_{args.label}.json'), 'w') as f:
        json.dump(record, f, indent=1)
    print(json.dumps({k: v for k, v in record.items() if k not in ('percent_change', 'gradient_statistic')},
                     indent=1), flush=True)


if __name__ == '__main__':
    main()
