"""Make the charts for devices_talk_deck.tex.

Run from any directory with a Python that has matplotlib and numpy:

    python make_figures.py

The charts read the demo results in demos/results/ for one machine, HOST.
They are written as PDF into images/.  The script also builds the MACE
device figure from plans/mace4d/figures/fig6_devices.tex with pdflatex and
copies it into images/.  The deck reads two more charts from the update
deck's images/ folder; README.md lists every image and its source.
"""
import csv
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
IMAGES = HERE / 'images'
IMAGES.mkdir(exist_ok=True)
RESULTS = HERE / 'demos' / 'results'
MACE_FIGURES = HERE.parents[1] / 'plans' / 'mace4d' / 'figures'
HOST = 'apple_m4_max'

ACCENT = '#3D6B99'
GRAY = '#8A8A8A'
BRICK = '#A23B3B'
GREEN = '#2E7D32'
AMBER = '#B26B1F'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})

# NVIDIA H100 SXM5 80GB datasheet: float32 without tensor cores, and HBM3 bandwidth.
H100_TFLOPS = 67.0
H100_TB_PER_S = 3.35


def read_rows(demo):
    """Return the rows of demos/results/<demo>_<HOST>.csv as dicts."""
    with open(RESULTS / f'{demo}_{HOST}.csv', newline='') as f:
        return list(csv.DictReader(f))


def machine_value(device, quantity):
    """Return one measured value from the machine demo."""
    for row in read_rows('machine'):
        if row['device'] == device and row['quantity'] == quantity:
            return float(row['value'])
    raise KeyError((device, quantity))


def thread_scaling():
    """Speedup of add and sin on the M4 Max CPU against torch's thread count.
    Source: demos/demo_threads.py."""
    rows = read_rows('threads')
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for name, color, label in [('sin', GREEN, 'sin: many operations per byte'),
                               ('add', ACCENT, 'add: one operation per 12 bytes')]:
        threads = [int(r['threads']) for r in rows if r['operation'] == name]
        speedup = [float(r['speedup']) for r in rows if r['operation'] == name]
        ax.plot(threads, speedup, 'o-', color=color, label=label, lw=2, ms=5)
    top = max(int(r['threads']) for r in rows)
    ax.plot([1, 12], [1, 12], '--', color=GRAY, lw=1, label='speedup equal to threads')
    ax.axvspan(12.5, top + 0.5, color=GRAY, alpha=0.12, lw=0)
    ax.text(14.5, 12.2, 'the 4 efficiency\ncores join', ha='center', va='top', fontsize=9, color='#5A5A5A')
    add_rate = np.median([float(r['GB_per_s']) for r in rows
                          if r['operation'] == 'add' and int(r['threads']) >= 4])
    ax.annotate(f'memory bandwidth\nreached, about {add_rate:.0f} GB/s', xy=(4, 2.6), xytext=(5.5, 0.7),
                fontsize=9, color=ACCENT, arrowprops=dict(arrowstyle='->', color=ACCENT))
    ax.set_xlabel('torch threads')
    ax.set_ylabel('speedup over one thread')
    ax.set_xticks([1, 2, 4, 6, 8, 10, 12, 14, 16])
    ax.set_xlim(0.5, top + 0.5)
    ax.set_ylim(0, 12.5)
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    fig.tight_layout()
    fig.savefig(IMAGES / 'thread_scaling.pdf')
    plt.close(fig)


def operation_size():
    """Time of one c = a + b against the number of values, on the M4 Max CPU
    and GPU, with a wait for the device after each operation.
    Source: demos/demo_sizes.py."""
    rows = read_rows('sizes')
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for device, color, label in [('mps', ACCENT, 'GPU (MPS)'), ('cpu', GRAY, 'CPU, 12 threads')]:
        n = np.array([int(r['n']) for r in rows if r['device'] == device])
        us = np.array([float(r['microseconds']) for r in rows if r['device'] == device])
        ax.loglog(n, us, 'o-', color=color, label=label, lw=2, ms=4)
    gpu_rate = machine_value('mps', 'bandwidth, c = a + b')
    n_line = np.array([2**20, 2**28])
    ax.loglog(n_line, 12 * n_line / (gpu_rate * 1e9) * 1e6, '--', color=ACCENT, lw=1)
    ax.text(2**23.2, 12 * 2**22 / (gpu_rate * 1e9) * 1e6, f'{gpu_rate:.0f} GB/s', color=ACCENT,
            fontsize=9, ha='left', va='top')
    ax.text(2**2, 230, 'fixed cost: issuing the\noperation and waiting for it', fontsize=9,
            color=ACCENT, va='bottom')
    ax.text(2**14.6, 3.2, 'torch starts to split\nthe work across threads', fontsize=9,
            color='#5A5A5A', va='bottom')
    ax.set_xlabel('values in each array')
    ax.set_ylabel('microseconds per operation')
    ax.set_ylim(0.3, 3e4)
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    fig.tight_layout()
    fig.savefig(IMAGES / 'operation_size.pdf')
    plt.close(fig)


def roofline():
    """Attainable arithmetic rate against arithmetic intensity: the measured M4
    Max GPU and the H100 specification.  Sources: demos/demo_machine.py and the
    NVIDIA H100 datasheet."""
    gpu_tflops = machine_value('mps', 'matrix product')
    gpu_tb = machine_value('mps', 'bandwidth, c = a + b') / 1e3
    intensity = np.logspace(-2, 4, 400)
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for peak, bw, color, label, above in [
            (H100_TFLOPS, H100_TB_PER_S, GRAY, 'H100, float32 (specification)', True),
            (gpu_tflops, gpu_tb, ACCENT, 'M4 Max GPU (measured)', False)]:
        ax.loglog(intensity, np.minimum(peak, bw * intensity), color=color, lw=2, label=label)
        balance = peak / bw
        ax.plot([balance], [peak], 'o', color=color, ms=5)
        if above:
            ax.text(balance * 1.5, peak * 1.2, f'balance point:\n{balance:.0f} operations per byte',
                    fontsize=8.5, color=color, ha='left', va='bottom')
        else:
            ax.text(balance * 1.15, peak * 0.62, f'balance point:\n{balance:.0f} operations per byte',
                    fontsize=8.5, color=color, ha='left', va='top')
    add_ai = 1 / 12
    ax.plot([add_ai], [gpu_tb * add_ai], 's', color=BRICK, ms=6)
    ax.text(add_ai * 1.4, gpu_tb * add_ai * 0.8, 'add, c = a + b\n1/12 operation per byte',
            fontsize=8.5, color=BRICK, ha='left', va='top')
    mm_ai = 8192 / 6
    ax.plot([mm_ai], [gpu_tflops], 's', color=BRICK, ms=6)
    ax.text(mm_ai * 1.1, gpu_tflops * 0.12, 'matrix product,\nn = 8192', fontsize=8.5, color=BRICK,
            ha='center', va='top')
    ax.annotate('', xy=(mm_ai, gpu_tflops * 0.8), xytext=(mm_ai * 1.05, gpu_tflops * 0.14),
                arrowprops=dict(arrowstyle='->', color=BRICK, lw=0.8))
    ax.set_xlabel('arithmetic intensity (operations per byte moved)')
    ax.set_ylabel('TFLOP/s')
    ax.set_ylim(3e-3, 1e3)
    ax.legend(frameon=False, fontsize=8.5, loc='upper left')
    fig.tight_layout()
    fig.savefig(IMAGES / 'roofline.pdf')
    plt.close(fig)


def mace_devices():
    """Build plans/mace4d/figures/fig6_devices.tex with pdflatex and copy the PDF."""
    with tempfile.TemporaryDirectory() as build:
        subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                        f'-output-directory={build}', 'fig6_devices.tex'],
                       cwd=MACE_FIGURES, check=True, capture_output=True)
        shutil.copy(Path(build) / 'fig6_devices.pdf', IMAGES / 'mace_devices.pdf')


if __name__ == '__main__':
    thread_scaling()
    operation_size()
    roofline()
    mace_devices()
    print(f'Wrote the charts into {IMAGES}')
