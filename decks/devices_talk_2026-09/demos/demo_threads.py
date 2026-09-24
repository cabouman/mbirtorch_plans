"""CPU speed against the number of threads, for two kinds of work.

torch splits each CPU operation across its pool of threads.  The script times
two operations on arrays of 2**26 float32 values (256 MiB each) as the pool
grows from 1 thread to every core:

  add, c = a + b   one arithmetic operation per 12 bytes moved, so memory
                   bandwidth limits it once a few cores are busy;
  sin, y = sin(x)  tens of arithmetic operations per 8 bytes moved, so the
                   number of cores limits it.

The script also times a 2048 by 2048 matrix product.  On a Mac, torch sends
it to Apple's Accelerate library, which does not use torch's threads, so its
speed should not follow the thread count.  Its rate is recorded in the GB_per_s
column as TFLOP/s.

Run from this directory:  python demo_threads.py
"""
import os

import torch

from devtime import host_label, time_call, write_csv

N = 2**26
CPU = torch.device('cpu')


def thread_counts():
    """Return 1, 2, 4, every second count up to 16, a few larger counts, and
    the number of cores this process may use (a Slurm job's allocation)."""
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    counts = [1, 2, 4] + list(range(6, 17, 2)) + [24, 32, 48, 64, 96]
    return sorted({c for c in counts if c <= cores} | {cores})


def sweep(operations, rows):
    """Time each operation at every thread count and append the rows.  Each
    entry of operations is a call and the amount of work behind the rate
    column: bytes for add and sin, thousands of GFLOP for the matrix product."""
    base = {}
    for threads in thread_counts():
        torch.set_num_threads(threads)
        line = f'{threads:7d}'
        for name, (fn, work) in operations.items():
            seconds = time_call(fn, CPU, repeats=10, warmup=2)
            base.setdefault(name, seconds)
            speedup = base[name] / seconds
            rate = work / seconds / 1e9
            rows.append((threads, name, f'{seconds * 1e3:.3f}', f'{speedup:.2f}', f'{rate:.2f}'))
            line += f'   {name} {seconds * 1e3:8.2f} ms {speedup:6.2f}x {rate:7.2f}'
        print(line)


def main():
    a = torch.rand(N)
    b = torch.rand(N)
    c = torch.empty(N)
    m = torch.rand(2048, 2048)
    rows = []
    print(f'{host_label()}: speedup over one thread, and GB/s (add, sin) or TFLOP/s (matmul)')
    # The matrix product runs in its own sweep, after the others, because
    # Accelerate's own threads would otherwise compete with torch's.
    sweep({'add': (lambda: torch.add(a, b, out=c), 3 * 4 * N),
           'sin': (lambda: torch.sin(a, out=c), 2 * 4 * N)}, rows)
    sweep({'matmul': (lambda: m @ m, 2 * 2048**3 / 1e3)}, rows)
    path = write_csv('threads', ['threads', 'operation', 'ms', 'speedup', 'GB_per_s'], rows)
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
