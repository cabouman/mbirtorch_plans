"""GPU smoke test: PyTorch version, CUDA build, each visible GPU, and an fp32 matmul rate per GPU.

An A100 should show about 19 TFLOP/s (its fp32 peak is 19.5).  Run by the sbatch files beside it.
"""
import time
import torch

print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| available", torch.cuda.is_available())
n = torch.cuda.device_count()
print("devices", n)
N = 8192
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    x = torch.randn(N, N, device=i)
    x @ x  # warm-up
    torch.cuda.synchronize(i)
    t = time.perf_counter()
    for _ in range(10):
        y = x @ x
    torch.cuda.synchronize(i)
    dt = time.perf_counter() - t
    print(f"  cuda:{i} {p.name}, {p.total_memory / 2**30:.1f} GiB, fp32 matmul {10 * 2 * N**3 / dt / 1e12:.1f} TFLOP/s")
