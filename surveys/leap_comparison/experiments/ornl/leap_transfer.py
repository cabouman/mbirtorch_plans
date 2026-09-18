"""Separate LEAP's transfer time from its kernel time on one H100.

LEAP accepts numpy arrays, which it copies to the card and back on every
call, and torch tensors already on the card, which it uses in place.  Timing
the same call both ways on the full-size ORNL scan gives the transfer share
directly.  The raw copy rates of the sinogram and the volume are timed with
torch for reference.  One card is used so that no chunking happens and the
whole arrays move once per call.  FBP runs last and in place, because it
filters its input.
"""
import time

import numpy as np
import torch
from leapctype import tomographicModels

import ornl_data

proj, weights, params = ornl_data.load()
del weights
cone = params['proj_params']['cone_params']
vol = params['vol_params']
mis = params['miscalib']
angles = params['proj_params']['angles']
sod = cone['src_orig']
sdd = sod + cone['orig_det']
dims = params['proj_params']['dims']
numRows, numCols = int(dims[0]), int(dims[2])
scan_angles = (angles * 180 / np.pi).astype('float32')
pixelWidth, pixelHeight = cone['pix_x'], cone['pix_y']
centerRow = numRows / 2 - mis['delta_v'] / pixelHeight - 0.5
centerCol = numCols / 2 - mis['delta_u'] / pixelWidth - 0.5
magnif = sdd / sod

leapct = tomographicModels()
leapct.set_gpu(0)
leapct.set_conebeam(len(scan_angles), numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol,
                    scan_angles, sod, sdd)
leapct.set_volume(numX=int(vol['n_vox_x']), numY=int(vol['n_vox_y']), numZ=int(vol['n_vox_z']),
                  voxelWidth=pixelWidth / magnif, voxelHeight=pixelHeight / magnif)
leapct.set_diameterFOV(1.0e16)
print('LEAP GPUs in use:', list(leapct.get_gpus()), flush=True)
gib = 1024 ** 3


def report(label, seconds, nbytes=None):
    rate = f', {nbytes / seconds / 1e9:.1f} GB/s' if nbytes else ''
    print(f'{label}: {seconds:.2f} s{rate}', flush=True)


# Raw copy rates.
torch.cuda.synchronize()
t = time.time()
g_dev = torch.from_numpy(proj).cuda()
torch.cuda.synchronize()
report('sinogram host to device, pageable', time.time() - t, proj.nbytes)
t = time.time()
back = g_dev.cpu().numpy()
report('sinogram device to host, pageable', time.time() - t, proj.nbytes)
del back
f_shape = (int(vol['n_vox_z']), int(vol['n_vox_y']), int(vol['n_vox_x']))
f_dev = torch.zeros(f_shape, dtype=torch.float32, device='cuda')

# Kernel-only calls: data already on the card.
for rep in range(2):
    torch.cuda.synchronize(); t = time.time()
    leapct.project(g_dev, f_dev)
    torch.cuda.synchronize(); report(f'project, data on device, run {rep}', time.time() - t)
f_dev.fill_(0.0)
for rep in range(2):
    torch.cuda.synchronize(); t = time.time()
    leapct.backproject(g_dev, f_dev)
    torch.cuda.synchronize(); report(f'backproject, data on device, run {rep}', time.time() - t)

# The same calls with host arrays: LEAP copies in and out itself.
f_host = leapct.allocate_volume()
for rep in range(2):
    t = time.time()
    leapct.backproject(proj, f_host)
    report(f'backproject, data on host, run {rep}', time.time() - t)
g_host = leapct.allocate_projections()
for rep in range(2):
    t = time.time()
    leapct.project(g_host, f_host)
    report(f'project, data on host, run {rep}', time.time() - t)
del g_host

# FBP both ways, in place so the input is filtered rather than copied.
q = proj.copy()
t = time.time()
leapct.FBP(q, f_host, inplace=True)
report('FBP, data on host, in place', time.time() - t)
del q
torch.cuda.synchronize(); t = time.time()
leapct.FBP(g_dev, f_dev, inplace=True)
torch.cuda.synchronize(); report('FBP, data on device, in place', time.time() - t)
print('peak device memory allocated by torch (GiB):', torch.cuda.max_memory_allocated() / gib)
print('TRANSFER DONE', flush=True)
