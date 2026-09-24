"""A GPU call returns before the GPU finishes.

The script multiplies two 8192 by 8192 float32 matrices on the GPU (CUDA or
MPS) and reads the clock at three points: when the call returns, after
synchronize() returns, and after .item() reads one value of the product.
Each reported time is the median over five products.

Run from this directory:  python demo_async.py
"""
import statistics
import time

import torch

from devtime import accelerator, device_label, synchronize, write_csv

N = 8192
REPEATS = 5


def main():
    device = accelerator()
    if device.type == 'cpu':
        print('No GPU was found.  On the CPU a call returns only after its work is done.')
        return
    a = torch.randn(N, N, device=device)
    b = torch.randn(N, N, device=device)
    (a @ b)[0, 0].item()  # The first product includes one-time setup, so it is not timed.

    returned, synchronized, item = [], [], []
    for _ in range(REPEATS):
        synchronize(device)
        start = time.perf_counter()
        c = a @ b
        returned.append(time.perf_counter() - start)
        synchronize(device)
        synchronized.append(time.perf_counter() - start)
        del c

        start = time.perf_counter()
        c = a @ b
        c[0, 0].item()
        item.append(time.perf_counter() - start)
        del c

    rows = [('the call returned', statistics.median(returned)),
            ('synchronize() returned', statistics.median(synchronized)),
            ('.item() returned', statistics.median(item))]
    rate = 2 * N**3 / statistics.median(synchronized) / 1e12
    print(f'{N} by {N} matrix product on {device_label(device)}')
    for label, seconds in rows:
        print(f'  {label:<24s} {seconds * 1e3:9.2f} ms')
    print(f'  rate from the synchronized time: {rate:.1f} TFLOP/s')
    path = write_csv('async', ['device', 'reading', 'ms'],
                     [(device_label(device), label, f'{seconds * 1e3:.3f}') for label, seconds in rows])
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
