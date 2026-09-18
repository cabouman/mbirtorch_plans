"""Render early fusion results from the newest checkpoint: the central axial
slice and a zoom for the standard recon, the DRUNet postprocessing, and the
fusion checkpoint, plus the consensus traces so far.  Reads the volumes
memory-mapped and touches no GPU."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '/scratch/gautschi/buzzard/leap_ornl/out/msf/'
standard = np.load('/scratch/gautschi/buzzard/leap_ornl/out/mbirtorch_full_recon.npy', mmap_mode='r')
postproc = np.load(OUT + 'ornl_sigma002_postproc.npy', mmap_mode='r')
fusion = np.load(OUT + 'ornl_sigma002_checkpoint.npy', mmap_mode='r')
traces = np.load(OUT + 'ornl_sigma002_traces.npz')
k = len(traces['consensus_spread'])
print('iterations recorded:', k, 'spread:', traces['consensus_spread'])

mid = standard.shape[2] // 2
s = np.asarray(standard[:, :, mid]); p = np.asarray(postproc[:, :, mid]); f = np.asarray(fusion[:, :, mid])
lo, hi = np.percentile(s, [1, 99.5])
panels = [(s, 'standard qGGMRF recon'), (p, 'DRUNet postprocessing'),
          (f, f'fusion checkpoint after {5 * (k // 5)} iterations, sigma 0.02')]
fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
for ax, (img, title) in zip(axes, panels):
    ax.imshow(img, cmap='gray', vmin=lo, vmax=hi); ax.set_title(title); ax.axis('off')
fig.tight_layout(); fig.savefig(OUT + 'early_slices.png', dpi=110)

n = s.shape[0]; sl = (slice(n // 4, n // 2), slice(n // 4, n // 2))
fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
for ax, (img, title) in zip(axes, panels):
    ax.imshow(img[sl], cmap='gray', vmin=lo, vmax=hi); ax.set_title(title + ', zoom'); ax.axis('off')
fig.tight_layout(); fig.savefig(OUT + 'early_slices_zoom.png', dpi=110)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.semilogy(traces['consensus_spread'], 'o-', label='consensus spread')
ax.semilogy(traces['consensus_change'][1:], 's-', label='consensus change')
ax.set_xlabel('iteration'); ax.legend(); ax.grid(True, which='both', alpha=0.3)
ax.set_title('fusion convergence traces so far')
fig.tight_layout(); fig.savefig(OUT + 'early_traces.png', dpi=110)
print('EARLY RENDER DONE')
