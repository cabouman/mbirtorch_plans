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

Job 16406732 ran on h016 on 2026-09-14, four H100 cards, device count
pinned to four.  The phase-attribution step of that job failed at its
first line, because torch's allocator counters exist only once a device has
been used, and the script now touches every card first.  The attribution
was resubmitted alone as job 16409250 with the corrected script.  The
results files are `results/mbirtorch_alloc_*_results.json`.

### The allocator setting

| run | 15 iterations | peak allocated, busiest card | peak reserved, busiest card | NVML peak, busiest card | NVML peak, four cards | host peak |
|---|---|---|---|---|---|---|
| default | 567.6 s | 29.03 GiB | 38.45 GiB | 39.66 GiB | 156.32 GiB | 64.6 GiB |
| expandable segments, run 1 | 565.3 s | 29.03 GiB | 33.87 GiB | 35.08 GiB | 137.98 GiB | 64.6 GiB |
| expandable segments, run 2 | 566.6 s | 29.03 GiB | 33.87 GiB | 35.08 GiB | 137.98 GiB | 64.7 GiB |

The setting changed nothing about the arrays in use and removed 4.6 GiB of
reserve per card, from 9.4 GiB to 4.8 GiB, at no time cost.  The two runs
with the setting agree to the byte on every memory figure and to 0.2
percent in time.  The direct reconstruction's NVML peak over four cards
fell as well, from 83.4 GiB to 78.8 GiB.

### The steady-state time per iteration

The 30-iteration run at the default allocator took 1104.5 s and the
15-iteration run 567.6 s, so one iteration costs 35.8 s in the steady
state and the one-time costs of a run, the compilation, the error sinogram,
and the Hessian diagonal, are about 31 s.  The pinned 15-iteration run was
also 51 s faster than the unpinned full reproduction's 619 s, which chose
four devices by its own search.

### The weighted product's cost

On one card, for one four-device shard of 533 views (5.32 GiB per array)
and a subset of 14450 pixels, which is the finest partition's subset size:

| operation | time |
|---|---|
| `weights * error`, the whole shard | 0.006 s |
| sparse back projection of the product at one subset | 0.071 s |
| sparse back projection of the error alone at one subset | 0.071 s |

The product runs at the card's memory bandwidth.  At the finest partition
of 128 subsets it costs about 0.8 s of a 35.8 s iteration, and the back
projection reads the product and the error alike, so fusing the product
into the back projection could save at most that 2 percent of an
iteration.  The peak of arrays in use during this step was 21.3 GiB.

### The peak attributed to phases

Job 16409250 ran on h010 on 2026-09-14 with the corrected script, on the
library as it was before the co-live shard change, four cards pinned,
four iterations, default allocator.  The log is
`results/dm1_phase_peaks_16409250.log`.

| phase | sampled peak of allocated memory, GiB per card |
|---|---|
| direct reconstruction | 15.15, 15.12, 15.12, 15.12 |
| initial error state: the dot products and the error formation | 29.03, 28.83, 28.83, 28.83 |
| subset step, back projection | 24.89, 24.12, 24.12, 23.38 |
| subset step, delta forward projection | 24.16, 23.97, 23.97, 23.97 |
| subset step, other parts | 21.74, 21.55, 21.55, 21.55 |
| outside the labeled phases | 20.62, 20.43, 25.75, 20.43 |

torch's own peak counter gave 29.03, 28.83, 28.83, 28.83 GiB, the same as
the sampled peak of the initial error state, so the sampler missed nothing
at the peak.  The initial error state is the run's peak by 4.1 GiB over the
next phase.  Two cautions on reading the table.  The labels nest, so the
Hessian diagonal's back projection, which runs through the same method as
a subset step's, is counted under the back projection label.  And one
stretch outside the labeled phases reached 25.75 GiB on one card only, most
likely the placement of an array or the Hessian's assembly, which the
wrappers did not cover.  A finer set of labels is needed before that
stretch can be named.

What this decides.  The subset back projection, which holds the weighted
error product, is 4.1 GiB below the peak, so fusing that product into the
kernel would not lower the peak.  The initial error state holds five
sinogram-sized shards in both of its parts, the dot products and the error
formation, so forming the error in place lowers the peak of an unweighted
run and leaves the peak of a weighted run at the dot products until those
change as well.

## Verification after the changes (job 16416780)

The job ran on h001 on 2026-09-14 with the committed changes (mbirtorch
23c4a43), the same four-card pinned setup, and the same three steps: the
phase attribution, the 15-iteration run at the allocator default, and the
same run with expandable segments.  The log is
`results/dm2_verify_16416780.log` and the results files are
`results/mbirtorch_site5_*_results.json`.

| quantity | before the changes | after, default allocator | after, with the setting |
|---|---|---|---|
| 15 iterations | 567.6 s | 567.9 s | 568.2 s |
| peak allocated, largest card | 29.03 GiB | 24.89 GiB | 24.89 GiB |
| peak reserved, largest card | 38.45 GiB | 31.56 GiB | 28.56 GiB |
| NVML peak, four cards combined | 156.32 GiB | 124.65 GiB | 116.68 GiB |

The changes removed 4.1 GiB from the peak of arrays in use at no time
cost, and with the allocator setting the reported figure per card fell
from 39.7 GiB to about 30.  The attribution confirms where the peak went.

| phase | before, GiB per card | after, GiB per card |
|---|---|---|
| initial error state | 29.03 | 20.62 |
| subset step, back projection | 24.89 | 24.89 |
| subset step, delta forward projection | 24.16 | 24.16 |
| direct reconstruction | 15.15 | 15.15 |

The peak now sits at the subset back projection, the phase the ledger
ranked next once the setup temporaries were gone.  Every measured phase
sits at or below its modeled value with the torch bodies' batch terms
removed, the back projection by 1.1 GiB, so the ledger remains a safe
preflight.  The reconstruction results moved only within the golden
tests' relative gates, as the scale decision accepted.
