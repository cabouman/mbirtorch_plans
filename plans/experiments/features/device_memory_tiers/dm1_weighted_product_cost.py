"""Time what one subset step spends on the weighted error sinogram, on one
H100, for the ORNL scan as one of four view shards.

The loop forms the product of the weights and the error sinogram as a
whole per-device array at every subset step, then back projects it at the
subset's pixels and frees it.  This script times the product and the back
projection separately for a shard of 533 views, which is what each of four
devices holds, at the finest partition the loop uses, 128 subsets.  The
product's time times the subset count is what a fused back projection would
remove from an iteration, and the product's size is the transient it would
remove from the peak.
"""
import time

import numpy as np
import torch

import mbirtorch
import ornl_data

proj, weights, params = ornl_data.load()
cone = params['proj_params']['cone_params']
vol = params['vol_params']
mis = params['miscalib']
angles = params['proj_params']['angles']
sod = cone['src_orig']
sdd = sod + cone['orig_det']
shard_views = len(angles) // 4
proj = np.ascontiguousarray(proj[:shard_views])
weights = np.ascontiguousarray(weights[:shard_views])
angles = angles[:shard_views]

model = mbirtorch.ConeBeamModel(proj.shape, angles, source_detector_dist=sdd, source_iso_dist=sod)
model.set_params(recon_shape=(int(vol['n_vox_x']), int(vol['n_vox_y']), int(vol['n_vox_z'])),
                 delta_det_channel=cone['pix_y'], delta_det_row=cone['pix_x'],
                 delta_voxel=vol['vox_xy'], det_channel_offset=-mis['delta_u'],
                 det_row_offset=-mis['delta_v'], use_ror_mask=False, verbose=0)
model.configure_devices(num_devices=1)
gib = 1024 ** 3
error = torch.as_tensor(proj, device='cuda')
weight = torch.as_tensor(weights, device='cuda')
del proj, weights
print(f'shard: {tuple(error.shape)}, {error.numel() * 4 / gib:.2f} GiB per array', flush=True)

# The finest partition: 128 subsets of the full pixel grid.
recon_shape = model.get_params('recon_shape')
num_pixels = int(recon_shape[0]) * int(recon_shape[1])
rng = np.random.default_rng(0)
perm = rng.permutation(num_pixels)
subset = torch.as_tensor(np.sort(perm[: num_pixels // 128]), dtype=torch.int64, device='cuda')
print(f'subset of {subset.numel()} pixels', flush=True)


def timed(label, fn, repeats=3):
    fn()
    torch.cuda.synchronize()
    best = None
    for _ in range(repeats):
        t = time.time()
        out = fn()
        torch.cuda.synchronize()
        best = min(best, time.time() - t) if best is not None else time.time() - t
    print(f'{label}: {best:.3f} s', flush=True)
    return out


product = timed('weights * error, whole shard', lambda: weight * error)
timed('sparse back projection of the product, one subset', lambda: model.sparse_back_project(product, subset))
timed('sparse back projection of the error alone, one subset', lambda: model.sparse_back_project(error, subset))
print('peak allocated (GiB):', torch.cuda.max_memory_allocated() / gib, flush=True)
print('PRODUCT COST DONE', flush=True)
