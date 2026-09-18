"""Sample GPU memory from NVML in a background thread while work runs.

Two peaks are kept per visible GPU: the device's total used memory, which is
what the collaborators' scripts report, and the memory attributed to this
process alone, which excludes other processes on a shared card.  The host
peak is the resident set size from getrusage, as in their scripts.
"""
import os
import resource
import threading
import time

import pynvml


class GpuSampler:
    def __init__(self, interval=0.05):
        pynvml.nvmlInit()
        self.count = pynvml.nvmlDeviceGetCount()
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.count)]
        self.names = [pynvml.nvmlDeviceGetName(h) for h in self.handles]
        self.interval = interval
        self.pid = os.getpid()
        self.reset()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def reset(self):
        self.peak_total = [0] * self.count
        self.peak_process = [0] * self.count
        self.peak_sum_total = 0
        self.peak_sum_process = 0

    def _run(self):
        while not self._stop:
            sum_total = 0
            sum_process = 0
            for i, h in enumerate(self.handles):
                used = pynvml.nvmlDeviceGetMemoryInfo(h).used
                mine = 0
                try:
                    for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
                        if p.pid == self.pid and p.usedGpuMemory is not None:
                            mine += p.usedGpuMemory
                except pynvml.NVMLError:
                    pass
                self.peak_total[i] = max(self.peak_total[i], used)
                self.peak_process[i] = max(self.peak_process[i], mine)
                sum_total += used
                sum_process += mine
            self.peak_sum_total = max(self.peak_sum_total, sum_total)
            self.peak_sum_process = max(self.peak_sum_process, sum_process)
            time.sleep(self.interval)

    def stop(self):
        self._stop = True
        self._thread.join()
        pynvml.nvmlShutdown()

    def report(self, label):
        gib = 1024 ** 3
        lines = [f'=== GPU memory peaks, {label} ({self.count} visible GPUs)']
        for i in range(self.count):
            lines.append(f'  GPU {i} ({self.names[i]}): total used {self.peak_total[i] / gib:.2f} GiB, '
                         f'this process {self.peak_process[i] / gib:.2f} GiB')
        lines.append(f'  combined peak: total used {self.peak_sum_total / gib:.2f} GiB, '
                     f'this process {self.peak_sum_process / gib:.2f} GiB')
        text = '\n'.join(lines)
        print(text, flush=True)
        return dict(label=label, peak_total_gib=[x / gib for x in self.peak_total],
                    peak_process_gib=[x / gib for x in self.peak_process],
                    peak_sum_total_gib=self.peak_sum_total / gib,
                    peak_sum_process_gib=self.peak_sum_process / gib)


def host_peak_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
