# Increment 1 of the device memory plan: the measurements

Companion to the scripts in this directory and to
`plans/features/device_memory_tiers/device_memory_tiers_plan.md`, Section 6.
This file holds the run detail and the numbers.  The plan holds their
interpretation.

## What runs

Job `dm1_allocator_and_product.sbatch` on four H100 cards of gautschi, with
the device count pinned to four through `MBIRTORCH_NUM_DEVICES`, in the
ORNL harness directory `/scratch/gautschi/buzzard/leap_ornl/`.  The data and
the model parameters are the ORNL Inconel scan's, as in
`plans/experiments/features/leap_comparison/ornl/ornl_harness.md`.

| step | script | what it records |
|---|---|---|
| phase peaks | `dm1_phase_peaks.py --iterations 4` | the sampled peak of allocated memory per card for each phase, and torch's own peak counters |
| allocator default | `run_mbirtorch.py --iterations 15 --stop-pct 0` | peak allocated, peak reserved, NVML peak per card, and the time |
| allocator expandable, twice | the same with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | the same |
| allocator default, 30 iterations | `run_mbirtorch.py --iterations 30 --stop-pct 0` | the time, for the steady-state time per iteration |
| weighted product cost | `dm1_weighted_product_cost.py` on one card | the time of the whole-shard product and of a subset back projection |

The phase sampler wraps the model's own methods from the driver and reads
torch's allocated bytes on every card every five milliseconds.  The
library is not changed.  The steady-state time per iteration is the
30-iteration time less the 15-iteration time, divided by 15.

## Ledger pricing used by the plan

`price_phases.py` prints every ledger phase for the ORNL geometry with the
torch bodies' batch terms removed.  Run on 2026-09-14 from the mbirtorch
repository root at commit 41fca86 with the miniforge environment.

| phase, busiest card | four devices | two devices |
|---|---|---|
| error sinogram formation | 29.06 GiB | 57.89 GiB |
| subset delta forward projection, granularity 4 | 27.23 GiB | 53.53 GiB |
| initial forward projection | 26.66 GiB | 52.41 GiB |
| per-iteration statistics, squared error | 25.97 GiB | 51.71 GiB |

The error sinogram formation phase holds the sinogram, the weights, the
forward projection, the scaled projection, and the error sinogram, 5.32
GiB each on four devices and 10.64 GiB each on two, plus the initial
reconstruction.

## Results

Job 16406732, submitted 2026-09-14.  To be filled from the log
`results/dm1_allocator_and_product_16406732.log` and the results files.
