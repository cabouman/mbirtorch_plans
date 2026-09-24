"""Helpers shared by the demos: choose a device, wait for it, time a call,
and record the results.

A call that runs on a GPU returns before the GPU finishes.  Every timing here
therefore waits for the device before it reads the clock.
"""
import csv
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

RESULTS = Path(__file__).resolve().parent / 'results'


def accelerator():
    """Return the GPU device, CUDA first and then MPS, or the CPU if there is no GPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def devices():
    """Return the CPU, followed by the GPU device if there is one."""
    gpu = accelerator()
    return [torch.device('cpu')] if gpu.type == 'cpu' else [torch.device('cpu'), gpu]


def synchronize(device):
    """Wait until every operation queued on the device has finished."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def time_call(fn, device, repeats=10, warmup=2):
    """Return the median time of fn() in seconds.

    The warmup calls are not timed, so compilation and the first memory
    allocation are excluded.  The device is synchronized before and after
    each timed call, and fn's result is dropped before the next call.
    """
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        fn()
        synchronize(device)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def host_label():
    """Return the CPU model, such as 'Apple M4 Max', or the machine type if the model is not found."""
    try:
        if sys.platform == 'darwin':
            return subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                  capture_output=True, text=True, check=True).stdout.strip()
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return platform.machine()


def device_label(device):
    """Return the device type, with the GPU model for a CUDA device."""
    if device.type == 'cuda':
        return f'cuda ({torch.cuda.get_device_name(device)})'
    return device.type


def write_csv(demo, header, rows):
    """Write the rows to results/<demo>_<host>.csv and return the path."""
    RESULTS.mkdir(exist_ok=True)
    host = re.sub(r'[^a-z0-9]+', '_', host_label().lower()).strip('_')
    path = RESULTS / f'{demo}_{host}.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return path
