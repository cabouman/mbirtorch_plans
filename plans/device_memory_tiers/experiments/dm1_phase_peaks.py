"""Attribute the peak of allocated device memory to the phases of a VCD
reconstruction of the ORNL scan, without touching the library.

A background thread samples torch's allocated bytes on every device every
few milliseconds and keeps the maximum per phase label.  Thin wrappers on
the model's own methods set the label on entry and restore it on exit, so
the setup phases and the parts of a subset step are told apart: the direct
reconstruction, the error sinogram formation, the Hessian diagonal, and
within a step the delta forward projection, the back projection, and the
rest.  The device count is pinned to four through the environment so the
attribution and the allocator runs describe the same layout.  Sampling can
miss a transient shorter than the period, so the overall peak from torch's
own counter is printed beside the sampled peaks.
"""
import argparse
import os
import threading
import time

import numpy as np
import torch

import mbirtorch
import ornl_data

GIB = 1024 ** 3


class PhaseSampler:
    def __init__(self, period=0.005):
        self.n = torch.cuda.device_count()
        self.period = period
        self.stack = ['outside any phase']
        self.peaks = {}
        self.lock = threading.Lock()
        self._stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self._stop:
            with self.lock:
                label = self.stack[-1]
            for i in range(self.n):
                used = torch.cuda.memory_allocated(i)
                key = (label, i)
                if used > self.peaks.get(key, 0):
                    self.peaks[key] = used
            time.sleep(self.period)

    def push(self, label):
        with self.lock:
            self.stack.append(label)

    def pop(self):
        with self.lock:
            self.stack.pop()

    def stop(self):
        self._stop = True
        self.thread.join()

    def report(self):
        labels = []
        for label, _ in self.peaks:
            if label not in labels:
                labels.append(label)
        print('=== sampled peak of allocated memory per phase (GiB per card)', flush=True)
        for label in labels:
            per_card = [self.peaks.get((label, i), 0) / GIB for i in range(self.n)]
            print(f'  {label:<40s} ' + ' '.join(f'{x:6.2f}' for x in per_card), flush=True)


def phased(sampler, label, fn):
    def wrapper(*a, **k):
        sampler.push(label)
        try:
            return fn(*a, **k)
        finally:
            for i in range(sampler.n):
                torch.cuda.synchronize(i)
            sampler.pop()
    return wrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iterations', type=int, default=4)
    args = ap.parse_args()
    assert torch.cuda.is_available()
    print('library under test:', mbirtorch.__file__, 'pinned devices:', os.environ.get('MBIRTORCH_NUM_DEVICES'), flush=True)
    proj, weights, params = ornl_data.load()
    cone = params['proj_params']['cone_params']
    vol = params['vol_params']
    mis = params['miscalib']
    angles = params['proj_params']['angles']
    sod = cone['src_orig']
    sdd = sod + cone['orig_det']
    model = mbirtorch.ConeBeamModel(proj.shape, angles, source_detector_dist=sdd, source_iso_dist=sod)
    model.set_params(recon_shape=(int(vol['n_vox_x']), int(vol['n_vox_y']), int(vol['n_vox_z'])),
                     snr_db=30, sharpness=1.0, positivity_flag=False,
                     delta_det_channel=cone['pix_y'], delta_det_row=cone['pix_x'],
                     delta_voxel=vol['vox_xy'], det_channel_offset=-mis['delta_u'],
                     det_row_offset=-mis['delta_v'], use_ror_mask=False, verbose=1)

    # The allocator's counters exist only once a device has been used, so
    # touch every card before the sampler starts and before the reset.
    for i in range(torch.cuda.device_count()):
        torch.zeros(1, device=f'cuda:{i}')
        torch.cuda.synchronize(i)
    sampler = PhaseSampler()
    model.recon_direct = phased(sampler, 'direct reconstruction', model.recon_direct)
    model._initial_error_state = phased(sampler, 'error sinogram formation', model._initial_error_state)
    model.compute_hessian_diagonal = phased(sampler, 'hessian diagonal', model.compute_hessian_diagonal)
    original_create = model.create_vcd_subset_updater

    def create_wrapped(*a, **k):
        updater = original_create(*a, **k)
        return phased(sampler, 'subset step, other parts', updater)

    model.create_vcd_subset_updater = create_wrapped
    model.sparse_forward_project = phased(sampler, 'subset step, delta forward projection', model.sparse_forward_project)
    model.sparse_back_project = phased(sampler, 'subset step, back projection', model.sparse_back_project)

    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    t0 = time.time()
    fdk = model.recon_direct(proj)
    print(f'direct reconstruction took {time.time() - t0:.1f} s', flush=True)
    t0 = time.time()
    recon, recon_dict = model.recon(proj, weights=weights, max_iterations=args.iterations,
                                    stop_threshold_change_pct=0.0, init_recon=fdk)
    print(f'{args.iterations} iterations took {time.time() - t0:.1f} s', flush=True)
    sampler.stop()
    sampler.report()
    print('torch peak allocated per card (GiB):',
          ' '.join(f'{torch.cuda.max_memory_allocated(i) / GIB:6.2f}' for i in range(torch.cuda.device_count())), flush=True)
    print('torch peak reserved per card (GiB): ',
          ' '.join(f'{torch.cuda.max_memory_reserved(i) / GIB:6.2f}' for i in range(torch.cuda.device_count())), flush=True)
    print('PHASE PEAKS DONE', flush=True)


if __name__ == '__main__':
    main()
