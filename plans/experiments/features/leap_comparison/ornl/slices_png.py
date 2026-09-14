"""Render the central axial slices of the two full-size reconstructions side
by side, and their difference after matching orientation.

mbirtorch stores a volume as (row, column, slice) and the driver saved the
axial slice recon[:, :, nz // 2] under the key 'slice'.  LEAP stores a volume
as (z, y, x) and the driver saved x[nz // 2] under the key 'z'.  The two
conventions differ by a rotation or a flip in the axial plane, so the script
tries the eight axial orientations of the LEAP slice and keeps the one with
the smallest normalized RMSE against the mbirtorch slice.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out_dir = '/scratch/gautschi/buzzard/leap_ornl/out/'
tag = sys.argv[1] if len(sys.argv) > 1 else 'full'
m = np.load(out_dir + f'mbirtorch_{tag}_central_slices.npz')['slice'].astype(np.float32)
l = np.load(out_dir + f'leap_{tag}_central_slices.npz')['z'].astype(np.float32)
print('mbirtorch axial slice', m.shape, 'LEAP axial slice', l.shape)

best = None
for k in range(4):
    for flip in (False, True):
        cand = np.rot90(l, k)
        if flip:
            cand = cand[:, ::-1]
        if cand.shape != m.shape:
            continue
        nrmse = np.sqrt(np.mean((cand - m) ** 2)) / np.sqrt(np.mean(m ** 2))
        print(f'rot90 k={k} flip={flip}: NRMSE {nrmse:.4f}')
        if best is None or nrmse < best[0]:
            best = (nrmse, k, flip, cand)
nrmse, k, flip, l_aligned = best
print(f'best orientation: rot90 k={k} flip={flip}, NRMSE {nrmse:.4f}')
print('mbirtorch slice mean', float(m.mean()), 'LEAP slice mean', float(l_aligned.mean()))

lo, hi = np.percentile(m, [1, 99.5])
fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
axes[0].imshow(m, cmap='gray', vmin=lo, vmax=hi)
axes[0].set_title('mbirtorch, 15 iterations, central axial slice')
axes[1].imshow(l_aligned, cmap='gray', vmin=lo, vmax=hi)
axes[1].set_title('LEAP loop, 15 iterations, same slice')
d = l_aligned - m
axes[2].imshow(d, cmap='gray', vmin=-0.2 * (hi - lo), vmax=0.2 * (hi - lo))
axes[2].set_title(f'LEAP minus mbirtorch, NRMSE {nrmse:.3f}')
for ax in axes:
    ax.axis('off')
fig.tight_layout()
fig.savefig(out_dir + f'central_slices_{tag}.png', dpi=110)
print('wrote', out_dir + f'central_slices_{tag}.png')

# A zoomed quarter of the slice, where the pores are.
n = m.shape[0]
sl = (slice(n // 4, n // 2), slice(n // 4, n // 2))
fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
axes[0].imshow(m[sl], cmap='gray', vmin=lo, vmax=hi)
axes[0].set_title('mbirtorch, zoom')
axes[1].imshow(l_aligned[sl], cmap='gray', vmin=lo, vmax=hi)
axes[1].set_title('LEAP loop, zoom')
for ax in axes:
    ax.axis('off')
fig.tight_layout()
fig.savefig(out_dir + f'central_slices_{tag}_zoom.png', dpi=110)
print('wrote', out_dir + f'central_slices_{tag}_zoom.png')
