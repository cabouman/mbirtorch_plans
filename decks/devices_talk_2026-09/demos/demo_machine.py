"""Measured rates of this machine, for each device: the matrix product rate,
the memory bandwidth, the cost of one small operation, and the copy rates
between host and GPU.

  matrix product   c = a @ b on float32 matrices: 2 n**3 operations
  bandwidth        c = a + b on 2**27 float32 values: 12 bytes per value
  small operation  y = x + 1 on one value: the fixed cost of an operation
  copies           a 1 GiB float32 tensor, host to GPU and back

Run from this directory:  python demo_machine.py
"""
import time

import torch

from devtime import devices, device_label, host_label, synchronize, time_call, write_csv

BANDWIDTH_N = 2**27
COPY_N = 2**28
SMALL_REPEATS = 2000


def matmul_rate(device):
    """Return the float32 matrix product rate in TFLOP/s."""
    n = 4096 if device.type == 'cpu' else 8192
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    return 2 * n**3 / time_call(lambda: a @ b, device, repeats=5) / 1e12


def bandwidth(device):
    """Return the rate of c = a + b in GB/s, counting two reads and one write."""
    a = torch.rand(BANDWIDTH_N, device=device)
    b = torch.rand(BANDWIDTH_N, device=device)
    c = torch.empty(BANDWIDTH_N, device=device)
    return 3 * 4 * BANDWIDTH_N / time_call(lambda: torch.add(a, b, out=c), device) / 1e9


def small_operation_cost(device):
    """Return the time of one y = x + 1 on a one-value tensor, in microseconds."""
    x = torch.ones(1, device=device)
    for _ in range(100):
        x + 1
    synchronize(device)
    start = time.perf_counter()
    for _ in range(SMALL_REPEATS):
        x + 1
    synchronize(device)
    return (time.perf_counter() - start) / SMALL_REPEATS * 1e6


def copy_rates(device):
    """Return the host-to-GPU and GPU-to-host copy rates in GB/s.  On a CUDA
    GPU the copies also run from pinned host memory."""
    rates = {}
    host = torch.rand(COPY_N)
    nbytes = 4 * COPY_N
    rates['host to GPU'] = nbytes / time_call(lambda: host.to(device), device, repeats=5) / 1e9
    on_gpu = host.to(device)
    rates['GPU to host'] = nbytes / time_call(lambda: on_gpu.to('cpu'), device, repeats=5) / 1e9
    if device.type == 'cuda':
        pinned = host.pin_memory()
        rates['host to GPU, pinned'] = nbytes / time_call(
            lambda: pinned.to(device, non_blocking=True), device, repeats=5) / 1e9
        back = torch.empty(COPY_N, pin_memory=True)
        rates['GPU to host, pinned'] = nbytes / time_call(
            lambda: back.copy_(on_gpu, non_blocking=True), device, repeats=5) / 1e9
    return rates


def main():
    rows = []
    print(host_label())
    for device in devices():
        label = device_label(device)
        measured = [('matrix product', matmul_rate(device), 'TFLOP/s'),
                    ('bandwidth, c = a + b', bandwidth(device), 'GB/s'),
                    ('one small operation', small_operation_cost(device), 'microseconds')]
        if device.type != 'cpu':
            measured += [(name, rate, 'GB/s') for name, rate in copy_rates(device).items()]
        for name, value, unit in measured:
            print(f'  {label:<8s} {name:<22s} {value:9.2f} {unit}')
            rows.append((label, name, f'{value:.3f}', unit))
    path = write_csv('machine', ['device', 'quantity', 'value', 'unit'], rows)
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
