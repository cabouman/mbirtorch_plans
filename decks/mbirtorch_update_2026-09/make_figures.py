"""Make the charts for mbirtorch_update_deck.tex and copy its images.

Run from this directory with a Python that has matplotlib and numpy:

    python make_figures.py

Every number below is copied from a record in this repository, named beside
it.  The charts are written as PDF into images/.  The copied PNG figures are
listed in README.md with their sources; the repository ignores PNG files, so
the copies live only in the working tree.
"""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
IMAGES = HERE / 'images'
IMAGES.mkdir(exist_ok=True)
PLANS = HERE.parents[1]
MBIRTORCH = PLANS.parent / 'mbirtorch'

ACCENT = '#3D6B99'
GRAY = '#8A8A8A'
BRICK = '#A23B3B'
GREEN = '#2E7D32'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})


def memory_layers():
    """GPU memory of the ORNL scan on four H100 GPUs, before and after the
    2026-09-14 changes.  Source: plans/device_memory_tiers/experiments/
    dm1_record.md, section "Verification after the changes"."""
    labels = ['before', 'after', 'after, with the\nallocator setting']
    allocated = [29.03, 24.89, 24.89]
    reserved = [38.45, 31.56, 28.56]
    combined = [156.32, 124.65, 116.68]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6),
                                   gridspec_kw={'width_ratios': [1.35, 1]})
    x = np.arange(3)
    w = 0.36
    ax1.bar(x - w / 2, reserved, w, color=GRAY, label='reserved by the allocator')
    ax1.bar(x + w / 2, allocated, w, color=ACCENT, label='arrays in use')
    for i in range(3):
        ax1.text(x[i] - w / 2, reserved[i] + 0.6, f'{reserved[i]:.1f}', ha='center', fontsize=9)
        ax1.text(x[i] + w / 2, allocated[i] + 0.6, f'{allocated[i]:.1f}', ha='center', fontsize=9)
    ax1.set_xticks(x, labels)
    ax1.set_ylabel('GiB on the busiest GPU')
    ax1.set_ylim(0, 44)
    ax1.set_title('Peak on one GPU', fontsize=11)
    ax1.legend(frameon=False, fontsize=9, loc='upper right')
    ax2.bar(x, combined, 0.5, color=[GRAY, ACCENT, ACCENT])
    for i in range(3):
        ax2.text(x[i], combined[i] + 2, f'{combined[i]:.0f}', ha='center', fontsize=9)
    ax2.set_xticks(x, labels)
    ax2.set_ylabel('GiB, four GPUs combined')
    ax2.set_ylim(0, 175)
    ax2.set_title('Peak reported by nvidia-smi', fontsize=11)
    fig.tight_layout()
    fig.savefig(IMAGES / 'memory_layers.pdf')
    plt.close(fig)


def ornl_gpu_memory():
    """Peak memory of the ORNL scan on four H100 GPUs: mbirtorch against the
    collaborators' OGM2 loop on LEAP.  Source: surveys/leap_comparison/
    surveys/leap_comparison/findings/ornl_reproduction.md, Section 3.1."""
    mb = [39.7, 39.3, 39.3, 38.0]
    leap = [18.9, 6.2, 6.2, 6.2]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6),
                                   gridspec_kw={'width_ratios': [1.5, 1]})
    x = np.arange(4)
    w = 0.38
    ax1.bar(x - w / 2, mb, w, color=ACCENT, label='mbirtorch, arrays resident')
    ax1.bar(x + w / 2, leap, w, color=BRICK, label='ORNL OGM2 loop on LEAP, host arrays streamed')
    for i in range(4):
        ax1.text(x[i] - w / 2, mb[i] + 0.6, f'{mb[i]:.1f}', ha='center', fontsize=9)
        ax1.text(x[i] + w / 2, leap[i] + 0.6, f'{leap[i]:.1f}', ha='center', fontsize=9)
    ax1.set_xticks(x, [f'GPU {i}' for i in range(4)])
    ax1.set_ylabel('peak GiB per GPU (nvidia-smi)')
    ax1.set_ylim(0, 60)
    ax1.set_title('GPU memory', fontsize=11)
    ax1.legend(frameon=False, fontsize=8.5, loc='upper center', ncol=1)
    host = [72.6, 175.4]
    ax2.bar([0, 1], host, 0.5, color=[ACCENT, BRICK])
    for i in range(2):
        ax2.text(i, host[i] + 2.5, f'{host[i]:.1f}', ha='center', fontsize=9)
    ax2.set_xticks([0, 1], ['mbirtorch', 'ORNL loop\non LEAP'])
    ax2.set_ylabel('peak GiB of host memory')
    ax2.set_ylim(0, 200)
    ax2.set_title('Host memory', fontsize=11)
    fig.tight_layout()
    fig.savefig(IMAGES / 'ornl_gpu_memory.pdf')
    plt.close(fig)


def gather_time():
    """Time to gather a sharded 8.9 GiB volume to the host, before and after
    the rebuild of Shards.gather.  Source: surveys/leap_comparison/
    surveys/leap_comparison/findings/host_gather.md, the summary table."""
    labels = ['1 GPU', '2 GPUs', '4 GPUs']
    before = [2.3, 4.8, 5.7]
    after = [0.25, 0.26, 0.24]
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    x = np.arange(3)
    w = 0.36
    ax.bar(x - w / 2, before, w, color=GRAY, label='before')
    ax.bar(x + w / 2, after, w, color=ACCENT, label='after')
    for i in range(3):
        ax.text(x[i] - w / 2, before[i] + 0.08, f'{before[i]:.1f} s', ha='center', fontsize=9)
        ax.text(x[i] + w / 2, after[i] + 0.08, f'{after[i]:.2f} s', ha='center', fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel('seconds')
    ax.set_ylim(0, 6.6)
    ax.set_title('Gather of an 8.9 GiB volume to the host', fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(IMAGES / 'gather_time.pdf')
    plt.close(fig)


def memory_levels():
    """The four levels of GPU memory on one GPU of the ORNL scan, four GPUs,
    15 iterations.  Source: surveys/leap_comparison/findings/ornl_reproduction.md,
    Section 3.2, the four-device row."""
    labels = ['reported by nvidia-smi', 'reserved by the allocator',
              'peak of arrays in use', 'arrays held for the whole run']
    values = [39.7, 38.4, 29.0, 15.3]
    colors = [GRAY, GRAY, ACCENT, ACCENT]
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    y = np.arange(4)
    ax.barh(y, values, 0.55, color=colors)
    for i in range(4):
        ax.text(values[i] + 0.6, y[i], f'{values[i]:.1f} GiB', va='center', fontsize=10)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 48)
    ax.set_xlabel('GiB on one GPU')
    ax.set_title('One GPU of four, the ORNL scan', fontsize=11)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    fig.tight_layout()
    fig.savefig(IMAGES / 'memory_levels.pdf')
    plt.close(fig)


def copy_images():
    """Copy the figures made elsewhere.  Each source is a record in this
    repository or the mbirtorch documentation."""
    copies = {
        MBIRTORCH / 'docs/source/figs/geometry_viewer_cone.png': 'geometry_viewer_cone.png',
        PLANS / 'surveys/leap_comparison/experiments/results/quality_nrmse_vs_time_512.png':
            'quality_nrmse_vs_time_512.png',
    }
    for src, name in copies.items():
        if src.exists():
            shutil.copy(src, IMAGES / name)
        else:
            print('missing', src)


def fusion_zoom(npz_path):
    """Three zoomed panels of one axial slice of the ORNL fusion run: the
    qGGMRF reconstruction, DRUNet postprocessing, and the fusion consensus.
    Source: the slice arrays the finisher wrote on gautschi at
    /scratch/gautschi/buzzard/leap_ornl/out/msf/ornl_sigma002_slices.npz
    (keys standard, postproc, fusion; 1360 by 1360 float32).  The window is
    240 voxels square at the top edge of the part, chosen for its inclusions;
    one gray scale, the 1st to 99.5th percentile of the qGGMRF window, is
    shared by the three panels."""
    d = np.load(npz_path)
    rows, cols = slice(310, 550), slice(560, 800)
    panels = [('qGGMRF reconstruction', d['standard'][rows, cols]),
              ('DRUNet postprocessing', d['postproc'][rows, cols]),
              ('fusion consensus', d['fusion'][rows, cols])]
    lo, hi = np.percentile(panels[0][1], [1, 99.5])
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.25))
    for ax, (title, a) in zip(axes, panels):
        ax.imshow(a, cmap='gray', vmin=lo, vmax=hi, interpolation='nearest')
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.01, wspace=0.03)
    fig.savefig(IMAGES / 'ornl_fusion_zoom.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    import sys
    memory_layers()
    memory_levels()
    ornl_gpu_memory()
    gather_time()
    copy_images()
    print('wrote', sorted(p.name for p in IMAGES.iterdir()))
    if len(sys.argv) > 1:
        fusion_zoom(sys.argv[1])  # path to ornl_sigma002_slices.npz
    else:
        print('fusion_zoom skipped: pass the path to ornl_sigma002_slices.npz')
