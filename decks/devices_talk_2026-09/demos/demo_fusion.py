"""Eager torch against torch.compile on a chain of elementwise operations.

The chain is the qGGMRF surrogate weight b_tilde(delta) of
mbirtorch/qggmrf.py (b_tilde_by_definition), applied to 2**26 differences
between neighboring voxels.  In eager mode each of its operations is one
kernel that reads its input from memory and writes its output back.
torch.compile fuses the chain into one kernel that reads each difference once
and writes each weight once.

The script reports, for each device: the eager time, the time of the first
compiled call (which includes compiling), the compiled time after that, and
the largest difference between the eager and compiled results.

Run from this directory:  python demo_fusion.py
"""
import time

import torch

from devtime import devices, device_label, host_label, synchronize, time_call, write_csv

N = 2**26
F32_EPS = torch.finfo(torch.float32).eps


def b_tilde(delta, sigma_x=1.0, p=2.0, q=1.2, T=1.0):
    """Return rho'(delta) / (2 delta) for the qGGMRF potential, as in mbirtorch/qggmrf.py."""
    abs_delta = torch.clamp(torch.abs(delta), min=T * sigma_x * F32_EPS)
    delta_scale = abs_delta / (T * sigma_x)
    ds_q_minus_p = delta_scale ** (q - p)
    ds_p_minus_2 = abs_delta ** (p - 2)
    numerator = ds_p_minus_2 / (2 * sigma_x ** p)
    numerator = numerator * ds_q_minus_p * ((q / p) + ds_q_minus_p)
    return numerator / (1 + ds_q_minus_p) ** 2


def main():
    rows = []
    print(f'{host_label()}: b_tilde on {N} values')
    for device in devices():
        label = device_label(device)
        delta = torch.randn(N, device=device)
        eager = time_call(lambda: b_tilde(delta), device)

        compiled_fn = torch.compile(b_tilde)
        synchronize(device)
        start = time.perf_counter()
        result = compiled_fn(delta)
        synchronize(device)
        first_call = time.perf_counter() - start
        compiled = time_call(lambda: compiled_fn(delta), device)

        reference = b_tilde(delta)
        difference = float(((result - reference).abs() / reference.abs()).max())
        rate = 8 * N / compiled / 1e9
        print(f'  {label:<8s} eager {eager * 1e3:8.2f} ms   compiled {compiled * 1e3:8.2f} ms'
              f'   speedup {eager / compiled:5.2f}   first compiled call {first_call:6.2f} s'
              f'   largest relative difference {difference:.1e}')
        rows.append((label, f'{eager * 1e3:.3f}', f'{compiled * 1e3:.3f}', f'{eager / compiled:.2f}',
                     f'{first_call:.2f}', f'{rate:.1f}', f'{difference:.2e}'))
        del delta, result, reference
    path = write_csv('fusion', ['device', 'eager_ms', 'compiled_ms', 'speedup', 'first_call_s',
                                'compiled_GB_per_s', 'max_relative_difference'], rows)
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
