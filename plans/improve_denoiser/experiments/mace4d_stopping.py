"""One MACE4D reconstruction of the GPU test of the denoiser's stopping rule.

The test compares the stopping rule of the qGGMRF stack denoisers inside
MACE4DModel on the Lilly 4DCT phantom scan.  Each run of this script is one
version of the test, and mace4d_stopping.md lists the versions.  The library
under test is whatever mbirtorch the interpreter imports, so a version is
chosen by the environment the script runs in.

The reconstruction is set up as in the Lilly driver of mbirtorch_applications
(nsi_4d/Lilly_recon_4d.py), with that driver's defaults: six frames per
rotation, a frame overlap factor of 2.0, transmission_root weights, sharpness
1.0 on the frame models and 0.0 on the denoisers, and the temporal filter on.
Two settings differ from the driver.  The MACE stop threshold is 0, so every
run does --max_mace_iterations iterations.  The seed is fixed, so every run
draws the same partitions.

The script wraps three library functions to record the stack denoisers, and
it changes nothing they compute:
  MACE.step                     records the number of the MACE iteration.
  HyperplaneAgent._run_slab     records the orientation and the planes of each
                                denoiser task.
  QGGMRFDenoiser.denoise_stack  records each call's iteration counts and its
                                last stopping statistics.  With --rule gradient
                                it also passes stop_threshold_gradient and
                                max_iterations, which only the patched library
                                (gradient_rule.patch) accepts.
At the iterations in --sample_iterations, the call that holds the middle plane
of each orientation is sampled: the input, the warm start if there is one, and
the output of that plane's volume are kept with the call's parameters and its
pixel partition.  mace4d_calls.py later computes each sampled call's MAP
estimate.  The samples are copied to the host inside the task, which takes a
few milliseconds, and they are written to disk after the reconstruction.

Files written to --run_dir:
  logs/run_info.txt, logs/timing_log.csv, logs/task_log.csv
                  the logs of MACE4DModel.recon
  calls.csv       one row per denoiser call
  volumes.csv     one row per volume of each call
  samples/*.npz   the sampled calls
  summary.json    the settings, the library, the wall time, and the peak
                  device memory
  recon.npy       the final 4D image

Run on a gautschi GPU node through mace4d_stopping.sh, or on any machine:
    python mace4d_stopping.py --data_path <NSI dataset> --run_dir <dir> \\
        --init_dir <dir> --label <name> [--rule gradient] [--sigma_noise <value>]
With --max_mace_iterations 0, the run computes the initial image, caches it in
--init_dir, and writes no recon.npy.  The option --synthetic replaces the
dataset by a small simulated scan, for a check of the script on a CPU.
"""

import argparse
import csv
import json
import os
import subprocess
import threading
import time

import numpy as np
import torch

import mbirtorch as mt
from mbirtorch import mace as _mace
from mbirtorch import mace4d as _mace4d

# ---- Fixed settings of the reconstruction --------------------------------------
FRAMES_PER_ROTATION = 6
FRAME_OVERLAP_FACTOR = 2.0
WEIGHT_TYPE = 'transmission_root'
FRAME_SHARPNESS = 1.0                 # the frame models, which make the initial image
DENOISER_SHARPNESS = 0.0
NBR_WEIGHT_TIME = 1.0
MACE_PRIOR_WEIGHT = 0.5
RHO_MANN = 0.5
PROX_NUM_ITERATIONS = 3
PROX_STOP_THRESHOLD = 0.02
LIBRARY_MAX_ITERATIONS = 15           # MACE4D's own cap, recorded for the per-call counts

CALL_FIELDS = ['iteration', 'orientation', 'start', 'stop', 'device', 'volumes', 'seconds',
               'max_iterations', 'iterations_min', 'iterations_median', 'iterations_mean',
               'iterations_max', 'at_cap', 'percent_change_last_median', 'percent_change_last_max',
               'gradient_last_median', 'gradient_last_max', 'sampled', 'sample_copy_seconds']
VOLUME_FIELDS = ['iteration', 'orientation', 'plane', 'iterations', 'percent_change_last',
                 'gradient_last']


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data_path', type=str, default=None, help='NSI dataset directory.')
    parser.add_argument('--run_dir', type=str, required=True, help='Output folder of this run.')
    parser.add_argument('--init_dir', type=str, required=True,
                        help='Folder of the cached initial image, shared by all runs.')
    parser.add_argument('--label', type=str, required=True, help='Name of this version.')
    parser.add_argument('--max_mace_iterations', type=int, default=10)
    parser.add_argument('--rule', choices=['library', 'gradient'], default='library',
                        help="'library' keeps MACE4D's stopping rule; 'gradient' stops each volume "
                             'on the gradient statistic.')
    parser.add_argument('--gradient_threshold', type=float, default=0.01)
    parser.add_argument('--denoiser_max_iterations', type=int, default=200,
                        help='The cap of the gradient rule.')
    parser.add_argument('--sigma_noise', type=float, default=None,
                        help='Noise level of the three denoisers.  Omit for the automatic estimate.')
    parser.add_argument('--denoiser_warm_start', action='store_true')
    parser.add_argument('--sample_iterations', type=str, default='1,5,10')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_frames', type=int, default=None)
    parser.add_argument('--downsampling', type=int, default=1)
    parser.add_argument('--num_devices', type=int, default=None)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--synthetic', action='store_true',
                        help='Use a small simulated scan instead of --data_path.')
    parser.add_argument('--allow_cpu', action='store_true',
                        help='Run without a GPU, for a check of the script.')
    args = parser.parse_args()
    if args.data_path is None and not args.synthetic:
        parser.error('give --data_path or --synthetic')
    return args


class Recorder:
    """The records of the stack denoisers, filled by the wrappers below."""

    def __init__(self, args, run_dir):
        self.args = args
        self.run_dir = run_dir
        self.lock = threading.Lock()
        self.local = threading.local()
        self.iteration = 0
        self.planes = {}              # orientation -> number of hyperplanes
        self.call_rows = []
        self.volume_rows = []
        self.samples = []
        self.sample_iterations = {int(s) for s in args.sample_iterations.split(',') if s}
        self.sampled_keys = set()

    def flush(self):
        """Append the rows recorded so far to the two CSV files."""
        with self.lock:
            calls, self.call_rows = self.call_rows, []
            volumes, self.volume_rows = self.volume_rows, []
        for name, fields, rows in (('calls.csv', CALL_FIELDS, calls),
                                   ('volumes.csv', VOLUME_FIELDS, volumes)):
            path = os.path.join(self.run_dir, name)
            new = not os.path.exists(path)
            with open(path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                if new:
                    writer.writeheader()
                writer.writerows(rows)

    def write_samples(self):
        folder = os.path.join(self.run_dir, 'samples')
        os.makedirs(folder, exist_ok=True)
        for sample in self.samples:
            meta = sample.pop('meta')
            name = f"it{meta['iteration']:02d}_{meta['orientation']}.npz"
            np.savez(os.path.join(folder, name), meta=json.dumps(meta), **sample)


def json_value(value):
    """A parameter value in a form json can write: a bool, a float, or a list."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if np.ndim(value) == 0:
        return float(value)
    return [json_value(v) for v in value]


def to_host(tensor):
    """A host copy of one volume."""
    if torch.is_tensor(tensor):
        return tensor.detach().to('cpu', copy=True).numpy()
    return np.array(tensor, copy=True)


def install_wrappers(rec):
    """Wrap MACE.step, HyperplaneAgent._run_slab, and QGGMRFDenoiser.denoise_stack."""
    axis_names = {axis: name.replace('-', '') for name, axis in _mace4d._ORIENTATIONS}

    class TrackingMACE(_mace.MACE):
        def step(self, *a, **k):
            with rec.lock:
                rec.iteration = self.iteration + 1
            result = super().step(*a, **k)
            rec.flush()
            return result

    _mace4d.MACE = TrackingMACE

    run_slab = _mace.HyperplaneAgent._run_slab

    def tagged_run_slab(self, w, region, device):
        axis = self.axis
        name = axis_names[axis]
        with rec.lock:
            rec.planes[name] = int(w.shape[axis])
            iteration = rec.iteration
        rec.local.call = dict(orientation=name, start=int(region[axis].start),
                              stop=int(region[axis].stop), iteration=iteration,
                              device=str(device), planes=int(w.shape[axis]))
        try:
            return run_slab(self, w, region, device)
        finally:
            rec.local.call = None

    _mace.HyperplaneAgent._run_slab = tagged_run_slab

    denoise_stack = mt.QGGMRFDenoiser.denoise_stack

    def recorded_denoise_stack(self, stack, *a, **k):
        call = getattr(rec.local, 'call', None)
        if call is None:
            return denoise_stack(self, stack, *a, **k)
        if rec.args.rule == 'gradient':
            k['stop_threshold_gradient'] = rec.args.gradient_threshold
            k['max_iterations'] = rec.args.denoiser_max_iterations
        max_iterations = int(k.get('max_iterations', LIBRARY_MAX_ITERATIONS))
        real = call['stop'] - call['start']
        middle = call['planes'] // 2
        key = (call['iteration'], call['orientation'])
        sample = (call['iteration'] in rec.sample_iterations and call['start'] <= middle < call['stop']
                  and key not in rec.sampled_keys)
        copy_seconds = 0.0
        if sample:
            rec.sampled_keys.add(key)
            tick = time.time()
            j = middle - call['start']
            z = to_host(stack[j])
            init = k.get('init_stack')
            w0 = None if init is None else to_host(init[j])
            copy_seconds += time.time() - tick
        start = time.time()
        out, info = denoise_stack(self, stack, *a, **k)
        seconds = time.time() - start
        counts = np.asarray(info['num_iterations'][:real])
        last_pct = np.array([h[-1] if h else np.nan for h in info['nmae_pct'][:real]])
        gradients = info.get('gradient_statistic')
        last_grad = (np.array([g[-1] if g else np.nan for g in gradients[:real]])
                     if gradients is not None else np.full(real, np.nan))
        if sample:
            tick = time.time()
            x = to_host(out[j])
            copy_seconds += time.time() - tick
            names = ['sigma_y', 'sigma_x', 'p', 'q', 'T', 'qggmrf_nbr_wts', 'granularity',
                     'partition_sequence', 'use_ror_mask']
            params = {n: json_value(v) for n, v in zip(names, self.get_params(names))}
            meta = dict(label=rec.args.label, iteration=call['iteration'],
                        orientation=call['orientation'], plane=middle, slab=[call['start'], call['stop']],
                        device=call['device'], rule=rec.args.rule,
                        gradient_threshold=rec.args.gradient_threshold if rec.args.rule == 'gradient' else None,
                        max_iterations=max_iterations, iterations=int(counts[j]),
                        percent_change=[float(v) for v in info['nmae_pct'][j]],
                        gradient_statistic=(None if gradients is None
                                            else [float(v) for v in gradients[j]]),
                        warm_start=w0 is not None, **params)
            partition = self.denoise_data['partition']
            entry = dict(meta=meta, input=z, output=x, partition=to_host(partition).astype(np.int64))
            if w0 is not None:
                entry['init'] = w0
            with rec.lock:
                rec.samples.append(entry)
        row = dict(iteration=call['iteration'], orientation=call['orientation'], start=call['start'],
                   stop=call['stop'], device=call['device'], volumes=real, seconds=round(seconds, 4),
                   max_iterations=max_iterations, iterations_min=int(counts.min()),
                   iterations_median=float(np.median(counts)), iterations_mean=float(counts.mean()),
                   iterations_max=int(counts.max()), at_cap=int(np.sum(counts >= max_iterations)),
                   percent_change_last_median=float(np.nanmedian(last_pct)),
                   percent_change_last_max=float(np.nanmax(last_pct)),
                   gradient_last_median=(float(np.nanmedian(last_grad))
                                         if np.isfinite(last_grad).any() else ''),
                   gradient_last_max=(float(np.nanmax(last_grad))
                                      if np.isfinite(last_grad).any() else ''),
                   sampled=int(sample), sample_copy_seconds=round(copy_seconds, 4))
        volume_rows = [dict(iteration=call['iteration'], orientation=call['orientation'],
                            plane=call['start'] + v, iterations=int(counts[v]),
                            percent_change_last=float(last_pct[v]),
                            gradient_last=(float(last_grad[v]) if np.isfinite(last_grad[v]) else ''))
                       for v in range(real)]
        with rec.lock:
            rec.call_rows.append(row)
            rec.volume_rows.extend(volume_rows)
        return out, info

    mt.QGGMRFDenoiser.denoise_stack = recorded_denoise_stack


def check_library():
    """Stop unless mbirtorch is imported from the folder that EXPECT_LIBRARY names, when it is set."""
    expected = os.environ.get('EXPECT_LIBRARY')
    if expected and not os.path.realpath(mt.__file__).startswith(os.path.realpath(expected) + os.sep):
        raise SystemExit(f'mbirtorch is imported from {mt.__file__}, not from {expected}')


def library_state():
    """The file, version, commit, and changed files of the mbirtorch under test."""
    folder = os.path.dirname(os.path.dirname(os.path.abspath(mt.__file__)))

    def git(*command):
        try:
            return subprocess.run(['git', '-C', folder, *command], capture_output=True, text=True,
                                  timeout=60).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ''

    return dict(file=mt.__file__, version=mt.__version__, commit=git('rev-parse', 'HEAD'),
                changed_files=git('diff', '--name-only').splitlines(), torch=torch.__version__)


def synthetic_problem(verbose):
    """A small simulated cone-beam scan of a static Shepp-Logan phantom, over two rotations."""
    num_views, rows, cols = 96, 16, 24
    angles = np.linspace(0.0, 4 * np.pi, num_views, endpoint=False)
    ct_model = mt.ConeBeamModel((num_views, rows, cols), angles, source_detector_dist=200.0,
                                source_iso_dist=100.0)
    ct_model.set_params(no_warning=True, verbose=verbose)
    recon_shape = ct_model.get_params('recon_shape')
    phantom = mt.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
    sino = np.asarray(ct_model.forward_project(phantom), dtype=np.float32)
    rng = np.random.default_rng(1)
    sino = sino + 0.02 * float(np.max(sino)) * rng.standard_normal(sino.shape).astype(np.float32)
    return sino, ct_model


def main():
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)
    for name in ('calls.csv', 'volumes.csv'):
        if os.path.exists(os.path.join(run_dir, name)):
            os.remove(os.path.join(run_dir, name))

    # Preflight: a silent CPU fallback and the wrong library are the two worst failures.
    if not args.allow_cpu:
        assert torch.cuda.is_available(), 'NOT ON GPU'
    check_library()
    state = library_state()
    print('library under test:', state, flush=True)
    print('torch threads:', torch.get_num_threads(), ' OMP_NUM_THREADS:', os.environ.get('OMP_NUM_THREADS'),
          ' MKL_NUM_THREADS:', os.environ.get('MKL_NUM_THREADS'),
          ' CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'), flush=True)
    if args.rule == 'gradient':
        import inspect
        assert 'stop_threshold_gradient' in inspect.signature(mt.QGGMRFDenoiser.denoise_stack).parameters, \
            '--rule gradient needs the library with gradient_rule.patch applied'

    rec = Recorder(args, run_dir)
    install_wrappers(rec)

    if args.synthetic:
        sino, ct_model = synthetic_problem(args.verbose)
        weights = None
        dataset = 'synthetic'
    else:
        import mbirtorch.preprocess as mtp
        sino, ct_model = mtp.nsi.get_sino_and_model(
            args.data_path, downsample_factor=[args.downsampling, args.downsampling], auto_crop=True)
        weights = mt.gen_weights(sino, weight_type=WEIGHT_TYPE)
        dataset = os.path.abspath(args.data_path)
    ct_model.set_params(sharpness=FRAME_SHARPNESS, positivity_flag=True, verbose=args.verbose)

    mace_model = mt.MACE4DModel(ct_model, frames_per_rotation=FRAMES_PER_ROTATION,
                                frame_overlap_factor=FRAME_OVERLAP_FACTOR, num_frames=args.num_frames)
    mace_model.set_params(mace_prior_weight=MACE_PRIOR_WEIGHT, rho_mann=RHO_MANN,
                          prox_num_iterations=PROX_NUM_ITERATIONS,
                          prox_stop_threshold=PROX_STOP_THRESHOLD, verbose=args.verbose,
                          sharpness=DENOISER_SHARPNESS, sigma_noise=args.sigma_noise,
                          nbr_weight_time=NBR_WEIGHT_TIME,
                          denoiser_warm_start=args.denoiser_warm_start)
    if args.num_devices is not None:
        mace_model.set_device_pool(args.num_devices)
    devices = [str(d) for d in mace_model.devices]
    print(f'Time frames: {mace_model.num_frames}; devices: {devices}', flush=True)

    start = time.time()
    recon_4d, recon_dict = mace_model.recon(
        sino, weights=weights, max_iterations=args.max_mace_iterations, stop_threshold_change_pct=0.0,
        init_dir=os.path.abspath(args.init_dir), log_dir=os.path.join(run_dir, 'logs'), seed=args.seed)
    wall = time.time() - start
    rec.flush()
    if args.max_mace_iterations > 0:      # a run of 0 iterations only computes the initial image
        np.save(os.path.join(run_dir, 'recon.npy'), recon_4d)
    rec.write_samples()

    memory = {}
    for d in mace_model.devices:
        d = torch.device(d)
        if d.type == 'cuda':
            memory[str(d)] = dict(max_allocated_gb=torch.cuda.max_memory_allocated(d) / 2 ** 30,
                                  max_reserved_gb=torch.cuda.max_memory_reserved(d) / 2 ** 30)
    params = recon_dict['recon_params']
    sigma = float(params['denoiser sigma (global)'])
    sigma_x = float(params['denoiser sigma_x'])
    summary = dict(label=args.label, args=vars(args), library=state, dataset=dataset, devices=devices,
                   torch_threads=torch.get_num_threads(), wall_seconds=wall,
                   denoiser_sigma=sigma, denoiser_sigma_x=sigma_x, r=sigma_x / sigma,
                   sigma_source=params.get('denoiser sigma source'),
                   iterations_completed=params.get('iterations completed'),
                   recon_shape=list(recon_4d.shape), peak_device_memory=memory,
                   planes=rec.planes, samples=len(rec.samples))
    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
