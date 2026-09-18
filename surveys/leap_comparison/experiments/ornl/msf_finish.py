"""Finish the reporting of a multi-slice fusion run.

The fusion job saves the final volume, metrics, and images only if it
reaches the end of its iterations inside the walltime.  This script makes
those outputs from whatever the run left behind: the final fusion volume
when it exists, otherwise the newest consensus checkpoint.  It computes the
weighted sinogram residual, saves the central slices, and writes the
side-by-side image and a results file.
"""
import json
import os
import sys
import time

import numpy as np
import torch

import mbirtorch
import ornl_data

OUT_DIR = '/scratch/gautschi/buzzard/leap_ornl/out/msf/'
TAG = sys.argv[1] if len(sys.argv) > 1 else 'sigma002'


def log(msg):
    print(f'[finish {time.strftime("%H:%M:%S")}] {msg}', flush=True)


final = os.path.join(OUT_DIR, f'ornl_{TAG}_fusion.npy')
checkpoint = os.path.join(OUT_DIR, f'ornl_{TAG}_checkpoint.npy')
if os.path.exists(final):
    src, from_checkpoint = final, False
elif os.path.exists(checkpoint):
    src, from_checkpoint = checkpoint, True
else:
    raise FileNotFoundError(f'no fusion volume or checkpoint for tag {TAG}')
log(f'volume: {src}')
fusion = np.load(src)
traces = np.load(os.path.join(OUT_DIR, f'ornl_{TAG}_traces.npz'))
iterations_recorded = int(len(traces['consensus_spread']))
log(f'{iterations_recorded} iterations recorded, final spread '
    f'{traces["consensus_spread"][-1]:.2e}')

proj, weights, params = ornl_data.load()
cone = params['proj_params']['cone_params']
vol = params['vol_params']
mis = params['miscalib']
angles = params['proj_params']['angles']
recon_shape = (int(vol['n_vox_x']), int(vol['n_vox_y']), int(vol['n_vox_z']))
model = mbirtorch.ConeBeamModel(proj.shape, angles,
                                source_detector_dist=cone['src_orig'] + cone['orig_det'],
                                source_iso_dist=cone['src_orig'])
model.set_params(recon_shape=recon_shape, snr_db=30, sharpness=1.0,
                 positivity_flag=False,
                 delta_det_channel=cone['pix_y'], delta_det_row=cone['pix_x'],
                 delta_voxel=vol['vox_xy'], det_channel_offset=-mis['delta_u'],
                 det_row_offset=-mis['delta_v'], use_ror_mask=False, verbose=0)


def rms_w(volume):
    ax = model.forward_project(np.ascontiguousarray(volume))
    num = 0.0
    den = 0.0
    for v0 in range(0, proj.shape[0], 64):
        r = proj[v0:v0 + 64] - ax[v0:v0 + 64]
        w = weights[v0:v0 + 64]
        num += float(np.sum(w * r * r))
        den += float(np.sum(w))
    del ax
    return float(np.sqrt(num / den))


results = {'tag': TAG, 'volume': os.path.basename(src),
           'from_checkpoint': from_checkpoint,
           'iterations_recorded': iterations_recorded,
           'final_spread': float(traces['consensus_spread'][-1])}
results['rms_w_fusion'] = rms_w(fusion)
log(f'rms_w fusion: {results["rms_w_fusion"]:.5f}')

standard = np.load('/scratch/gautschi/buzzard/leap_ornl/out/mbirtorch_full_recon.npy',
                   mmap_mode='r')
postproc = np.load(os.path.join(OUT_DIR, f'ornl_{TAG}_postproc.npy'), mmap_mode='r')
mid = [n // 2 for n in recon_shape]
np.savez(os.path.join(OUT_DIR, f'ornl_{TAG}_slices.npz'),
         standard=np.asarray(standard[:, :, mid[2]]),
         postproc=np.asarray(postproc[:, :, mid[2]]),
         fusion=fusion[:, :, mid[2]],
         standard_row=np.asarray(standard[mid[0]]),
         fusion_row=fusion[mid[0]])
with open(os.path.join(OUT_DIR, f'ornl_{TAG}_finish_results.json'), 'w') as f:
    json.dump(results, f, indent=1)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
s = np.asarray(standard[:, :, mid[2]])
lo, hi = np.percentile(s, [1, 99.5])
label = (f'multi-slice fusion, {iterations_recorded} iterations'
         + (' (checkpoint)' if from_checkpoint else ''))
fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
for ax_, (img, title) in zip(axes, [
        (s, 'standard qGGMRF recon'),
        (np.asarray(postproc[:, :, mid[2]]), 'DRUNet postprocessing'),
        (fusion[:, :, mid[2]], label)]):
    ax_.imshow(img, cmap='gray', vmin=lo, vmax=hi)
    ax_.set_title(title)
    ax_.axis('off')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, f'ornl_{TAG}_slices.png'), dpi=110)
n = s.shape[0]
sl = (slice(n // 4, n // 2), slice(n // 4, n // 2))
fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
for ax_, (img, title) in zip(axes, [
        (s[sl], 'standard, zoom'),
        (np.asarray(postproc[:, :, mid[2]])[sl], 'postprocessing, zoom'),
        (fusion[:, :, mid[2]][sl], 'fusion, zoom')]):
    ax_.imshow(img, cmap='gray', vmin=lo, vmax=hi)
    ax_.set_title(title)
    ax_.axis('off')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, f'ornl_{TAG}_slices_zoom.png'), dpi=110)
log('FINISH DONE')
