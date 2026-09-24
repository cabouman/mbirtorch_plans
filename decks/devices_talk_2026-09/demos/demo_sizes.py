"""Time of one operation against the size of its arrays.

The script times c = a + b for arrays of 1 to 2**28 float32 values on each
device.  Small arrays take a fixed time, the cost of issuing one operation.
Large arrays take a time in proportion to their size, set by the memory
bandwidth.  The size where the two meet is the smallest array that uses the
device well.

Run from this directory:  python demo_sizes.py
"""
import torch

from devtime import devices, device_label, host_label, time_call, write_csv

SIZES = [2**k for k in range(0, 29, 2)]


def main():
    rows = []
    print(f'{host_label()}: time of c = a + b')
    for device in devices():
        label = device_label(device)
        for n in SIZES:
            a = torch.rand(n, device=device)
            b = torch.rand(n, device=device)
            c = torch.empty(n, device=device)
            repeats = 50 if n < 2**20 else 10
            seconds = time_call(lambda: torch.add(a, b, out=c), device, repeats=repeats, warmup=5)
            rows.append((label, n, f'{seconds * 1e6:.3f}', f'{12 * n / seconds / 1e9:.3f}'))
            print(f'  {label:<8s} n = 2**{n.bit_length() - 1:<3d} {seconds * 1e6:12.1f} microseconds'
                  f'  {12 * n / seconds / 1e9:8.1f} GB/s')
            del a, b, c
    path = write_csv('sizes', ['device', 'n', 'microseconds', 'GB_per_s'], rows)
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
