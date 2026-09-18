"""Intensity profiles across pores for the fusion run's central slice.

Reads the central-slice file the finisher wrote and finds two line segments,
one horizontal and one vertical, inside the zoom window of the slice image,
each chosen to cross as many pores as possible.  Plots the standard
reconstruction, the DRUNet postprocessing, and the fusion along both
segments, with the segments drawn on the zoomed slice.

Usage: python msf_profiles.py <slices.npz> <out.png>
"""
import sys

import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PITCH_MM = 0.0172847      # voxel pitch of the ORNL reconstruction
LENGTH = 300              # segment length in pixels, about 5.2 mm

z = np.load(sys.argv[1])
standard, postproc, fusion = z['standard'], z['postproc'], z['fusion']
n = standard.shape[0]
win = (slice(n // 4, n // 2), slice(n // 4, n // 2))   # the zoom window

# Pores: dark connected components well inside the part, found on the fusion
# slice because it has the least grain.
part = fusion > 0.15
interior = ndimage.binary_erosion(part, iterations=25)
level = np.median(fusion[interior])
pores, count = ndimage.label(interior & (fusion < 0.8 * level))
sizes = ndimage.sum(np.ones_like(fusion), pores, index=np.arange(1, count + 1))
keep = np.zeros(count + 1, bool)
keep[1:] = sizes >= 6
pores = np.where(keep[pores], pores, 0)
print(f'part level {level:.4f}, pores kept {int(keep.sum())}')


def best_segment(vertical):
    """Segment inside the window and the interior crossing the most pores."""
    best = (0, None)
    r0, c0 = win[0].start, win[1].start
    r1, c1 = win[0].stop, win[1].stop
    for a in range(r0 if not vertical else c0, (r1 if not vertical else c1), 2):
        for b in range((c0 if not vertical else r0), (c1 if not vertical else r1) - LENGTH, 3):
            if vertical:
                rows, cols = slice(b, b + LENGTH), a
            else:
                rows, cols = a, slice(b, b + LENGTH)
            if not interior[rows, cols].all():
                continue
            crossed = len(set(pores[rows, cols].tolist()) - {0})
            if crossed > best[0]:
                best = (crossed, (rows, cols))
    return best


(nh, seg_h), (nv, seg_v) = best_segment(False), best_segment(True)
print(f'horizontal segment row {seg_h[0]} cols {seg_h[1].start}-{seg_h[1].stop} crosses {nh} pores')
print(f'vertical segment col {seg_v[1]} rows {seg_v[0].start}-{seg_v[0].stop} crosses {nv} pores')

fig = plt.figure(figsize=(11, 14))
gs = fig.add_gridspec(3, 1, height_ratios=[1.6, 1, 1])
ax = fig.add_subplot(gs[0])
lo, hi = np.percentile(standard, [1, 99.5])
ax.imshow(standard[win], cmap='gray', vmin=lo, vmax=hi,
          extent=(win[1].start, win[1].stop, win[0].stop, win[0].start))
ax.plot([seg_h[1].start, seg_h[1].stop], [seg_h[0], seg_h[0]], color='tab:red', lw=1.5, label='segment A')
ax.plot([seg_v[1], seg_v[1]], [seg_v[0].start, seg_v[0].stop], color='tab:blue', lw=1.5, label='segment B')
ax.set_title('standard recon, zoom window of the central slice, with the two segments')
ax.legend(loc='lower left')
ax.set_xlabel('column (pixel)')
ax.set_ylabel('row (pixel)')

x = np.arange(LENGTH) * PITCH_MM
for axp, seg, name, color in [(fig.add_subplot(gs[1]), seg_h, 'A (horizontal)', 'tab:red'),
                               (fig.add_subplot(gs[2]), seg_v, 'B (vertical)', 'tab:blue')]:
    axp.plot(x, standard[seg], color='0.6', lw=1.0, label='standard recon')
    axp.plot(x, postproc[seg], color='tab:green', lw=1.3, label='DRUNet postprocessing')
    axp.plot(x, fusion[seg], color='k', lw=1.3, label='multi-slice fusion')
    axp.set_title(f'intensity along segment {name}', color=color)
    axp.set_xlabel('position along the segment (mm)')
    axp.set_ylabel('reconstruction value')
    axp.set_xlim(0, x[-1])
    axp.grid(alpha=0.3)
    axp.legend(loc='lower left', ncol=3, fontsize=9)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=120)
print('wrote', sys.argv[2])
