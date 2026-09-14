# Reproducing the ORNL comparison of LEAP and mbirtorch on the Inconel scan

Date: 2026-09-14.  Status: MEASURED.  Every number below is read from the
results files in `plans/experiments/features/leap_comparison/ornl/results/`
or from the two ORNL PDFs, which are Greg's files and are kept outside
this repository.

## Summary

Collaborators at ORNL (Obaid Rahman) reconstructed one Metrotom scan of an
Inconel additive-manufacturing sample with LEAP and with mbirtorch on four
A100 40 GB cards.  They reported that LEAP needed one sixth of the GPU memory
at about the same time per iteration.  This page reproduces their
measurement on four gautschi H100 cards with their code and data, and
explains the memory difference from the two codes.

The reproduction agrees with their table.  Peak GPU memory was 156 GiB
across the four cards for mbirtorch and 22 GiB for LEAP, against their
144 GB and 24 GB.  Peak host memory was 73 GiB for mbirtorch and 175 GiB for
LEAP, against their 86 GB and 175 GB.  Fifteen iterations took 619 s for
mbirtorch and 830 s for the LEAP loop on the H100s, against 1496 s and
1666 s on their A100s.

The memory difference is a difference in where the arrays live.  The ORNL
LEAP loop keeps every array in host memory, and each LEAP projection call
streams jobs of at most 128 detector rows or 128 volume slices through the
cards.  A job and its slab of the volume take about 5 GiB, so a card holds
about 6 GiB during a projection.  The one exception is the regularizer,
whose gradient puts two whole volumes on the first card, which is the 19 GiB
peak on that card in both their run and ours.  mbirtorch keeps the error
sinogram, the weights, the reconstruction, and the Hessian diagonal resident
on the devices for the whole run.  On four devices that persistent set is
15 GiB per card, and the peak of arrays in use is 29 GiB per card.
`nvidia-smi` reports 39 GiB per card.  The difference is memory that torch's
caching allocator has reserved but is not using.

The per-iteration time is close because the two loops do different amounts
of work per iteration.  A LEAP iteration is one forward projection, one back
projection, one TV gradient, and host arithmetic on the 21 GiB arrays.  On
four H100s those take 17 s, 11 s, 3 s, and about 21 s, for 52 s per
iteration.  An mbirtorch iteration updates every voxel through its subsets
and averaged 41 s over the run, which includes the compilation, the error
sinogram, and the Hessian diagonal.  What an iteration achieves is a
separate question that this page does not measure.  The central slices of
the two 15-iteration reconstructions agree to 5.7 percent NRMSE.

## 1. What the collaborators ran

The data are one full-turn cone-beam scan: 2132 views, a 1456 x 1840 detector
at 0.127 mm, source to axis 110.080 mm, source to detector 808.814 mm.  Their
parameter reader pads the scanner's 1186 x 1261 x 1199 volume to
1360 x 1360 x 1296 voxels of 0.0172847 mm.  A sinogram-shaped float32 array is
21.3 GiB and a volume is 8.9 GiB.  The projections had beam-hardening and
scatter corrections applied with LEAP beforehand and were read from a float16
tiff.  The statistical weights are the dark-subtracted counts, shifted per
view by the scanner's recorded object shifts.

Both arms start from an FDK reconstruction.  The mbirtorch arm calls `recon`
with the weights, `snr_db=30`, `sharpness=1.0`, no positivity, no
region-of-reconstruction mask, and the library's default stop rule, and
leaves the device choice to the library.  The LEAP arm is their own loop, in
`utils/MBIR_LEAP_optim.py`: a voxel-wise preconditioned Nesterov OGM2 descent
on the weighted least-squares cost with LEAP's anisotropic total variation
regularizer (delta 1e-4, weight 0.7, p 1.2).  The loop holds the
projections, the weights, the error, and the weighted error in host numpy
arrays, and calls LEAP's `project`, `backproject`, and `TVgradient` once each
per iteration.

Their slides report 15 iterations of each loop on the server `mdf-a100` with
four A100 40 GB cards and 1 TB of host memory.  The code in the repository is
set to 100 iterations, so the slide table is the measurement to reproduce.

| quantity | mbirtorch 0.0.2 | LEAP 1.26 |
|---|---|---|
| FDK, average of 25 runs | 27.31 s | 26.78 s |
| MBIR, 15 iterations | 1496.34 s | 1666.16 s |
| peak GPU memory, four cards combined | 143.97 GB | 23.96 GB |
| peak host memory | 85.78 GB | 175.27 GB |

Their screenshot gives the per-card peaks.  mbirtorch held 35.6 GB on one
card and 35.4 GB on each of the other three.  LEAP held 18.8 GB on one card
and 6.2 GB on each of the other three.  The screenshot also shows the LEAP
loop's own timer at 98 to 106 s per iteration.

## 2. The reproduction

The harness runs the collaborators' loader and their LEAP loop unchanged, as
copies of their `utils` package, with drivers of our own around them.  The
drivers and a companion file with the run detail live in
`plans/experiments/features/leap_comparison/ornl/`.

- `ornl_data.py` calls their parameter reader once, caches the weights and
  parameters, and reads the corrected tiff.
- `run_mbirtorch.py` builds the cone-beam model with their parameters, runs
  the direct reconstruction, then `recon` from that start with the weights.
  Their script left the device count to the library, and so does this one.
- `run_leap.py` sets LEAP's geometry as their script does, runs `FBP`, then
  their `MBIR` loop from that start.  It times LEAP's three entry points from
  inside the loop.
- `gpu_sampler.py` samples every visible card from NVML during each phase,
  both the card's total and this process's share.
- `fdk_pieces.py` times the parts of the FDK for both codes (Section 3.4).
- `leap_transfer.py` times LEAP's calls with host arrays and with device
  tensors on one card (Section 3.5).

What differs from the collaborators' runs:

- The cards are H100 80 GB instead of A100 40 GB, so times are not
  comparable card for card.  The memory figures are.
- mbirtorch is the `greg_dev` tip 41fca86 of 2026-09-12, and LEAP is 1.26
  built from source on 2026-09-02.
- The FDK is timed twice instead of averaged over 25 runs.  The first run
  includes any warm-up cost.
- The XrayPhysics import of their LEAP script is dropped.  It served only
  the correction step, which was already applied to the tiff.

## 3. Results

### 3.1 The reproduction on four H100s

Both arms ran 15 iterations.  mbirtorch's stop rule did not fire: the
relative change was 0.84 percent after the last iteration, above the 0.2
percent threshold.  The library chose four devices on its own, as it did for
the collaborators.

| quantity | mbirtorch | LEAP loop |
|---|---|---|
| FDK, first run | 25.9 s | 12.2 s |
| FDK, second run | 18.0 s | 12.3 s |
| MBIR, 15 iterations | 619.0 s | 829.8 s |
| time per iteration | 41.3 s, averaged over the run | 52.0 s, steady state |
| peak GPU memory per card, NVML | 39.7, 39.3, 39.3, 38.0 GiB | 18.9, 6.2, 6.2, 6.2 GiB |
| peak GPU memory, four cards combined | 156.4 GiB | 22.2 GiB |
| peak host memory | 72.6 GiB | 175.4 GiB |

The mbirtorch average includes the one-time costs of the run: compiling the
solver, forming the error sinogram, and computing the Hessian diagonal.  The
LEAP steady state is the interval between the loop's own timer lines from
the second iteration on.  The first LEAP iteration took 58.1 s.

The NVML peaks match the collaborators' per-card figures.  The 6.2 GiB on
three cards is their 6.16 GB, and the 18.9 GiB on the first card is their
18.79 GB.  mbirtorch's per-card peak is 39 GiB here against their 35 GB, on
cards with twice the memory.

The central axial slices of the two reconstructions agree to 5.7 percent
NRMSE once the LEAP slice is rotated by a half turn and flipped to
mbirtorch's orientation.  The figure `results/central_slices_full.png` shows
the two slices and their difference.  The difference is streaks along the
sample's edges and the corners outside the field of view, where the two
regularizers differ.

### 3.2 Where mbirtorch's GPU memory goes

torch's own counters split the NVML figure into arrays in use and the
allocator's reserve.

| devices | iterations | persistent set per card, ledger | peak allocated per card | peak reserved per card | NVML peak per card |
|---|---|---|---|---|---|
| 4 | 15 | 15.3 GiB | 29.0 GiB | 38.4 GiB | 39.7 GiB |
| 2 | 5 | 30.4 GiB | 57.9 GiB | 69.8 GiB | 71.1 GiB |

The persistent set is the memory ledger's price for the arrays the loop
holds for its whole run, from `price_ornl_geometry.py`.  The difference
between it and the peak allocated is the widest subset step's transient:
the weighted error sinogram for the device's views, the prior gradient and
Hessian, and the projection kernels' batch buffers.  The difference between
allocated and reserved, about 9 GiB per card on four devices and 12 GiB on
two, is memory the caching allocator holds for reuse.  The NVML figure adds
the CUDA context.

For this scan the sinogram-shaped residents are the larger share.  On four
devices the error sinogram and the weights are 10.6 GiB of the 15.3 GiB
persistent set, and the reconstruction and the Hessian diagonal are 4.5 GiB.

### 3.3 Ablations at full size, five iterations each, two H100s

| arm | FDK | time per iteration, steady | forward per call | back per call | TV gradient per call | NVML peak per card, loop | NVML peak per card, FDK |
|---|---|---|---|---|---|---|---|
| LEAP, one card | 29.7 s | 101 s | | | | 31.3 GiB | 52.6 GiB |
| LEAP, two cards | 21.2 s | 71.3 s | 30.0 s | 19.6 s | 3.6 s | 18.9, 6.2 GiB | 10.1, 10.5 GiB |
| LEAP, two cards, chunk cap 32 | 32.2 s | 78.7 s | 31.5 s | 28.5 s | 3.7 s | 18.9, 4.5 GiB | 7.7, 7.8 GiB |
| LEAP, four cards (Section 3.1) | 12.2 s | 52.0 s | 17.0 s | 10.8 s | 3.4 s | 18.9, 6.2, 6.2, 6.2 GiB | 10.5, 9.8, 10.5, 10.1 GiB |
| mbirtorch, two devices | 30.8 s | 73.9 s, averaged | | | | 71.1, 68.5 GiB | 38.5, 33.4 GiB |

The per-call times are the timers around LEAP's entry points divided by the
number of calls.  The one-card arm ran before those timers were added.

Three results stand out.  On one card LEAP does not chunk at all, and the
whole projection set and volume go on the card.  The card held 31 GiB during
the loop and 53 GiB during the FDK.  The FDK's filtering needs a second copy
of the projections.  The loop's time per iteration on one card is 101 s, the same
as the collaborators saw on four A100s.  A lower chunk cap saves 1.7 GiB on
the chunk cards and 2.5 GiB during the FDK, at a tenth more time per
iteration.

### 3.4 The FDK split into its parts, four H100s

Greg asked why LEAP's FBP beats mbirtorch's warm FDK when LEAP streams its
arrays through the cards.  The script `fdk_pieces.py` (job 16401543) times
the parts of both.  mbirtorch is pinned to four devices, the count its
policy chose, and each part is timed twice.  The warm figures are below.

| part | mbirtorch | LEAP |
|---|---|---|
| sinogram, host to devices | 2.2 s, one device at a time | inside each job, that job's rows of every view |
| filter | 0.27 s, on the devices | 1.5 s, measured on the whole host array |
| back projection | 9.2 s, kernel plus band reduce | 10.5 s, chunked, copies included |
| volume, devices to host | 7.1 to 8.0 s | inside each job, one 0.9 GiB slab |
| whole | 19.7 s | 12.2 s |

The back projection kernel is not the slow part.  mbirtorch's kernel and
band reduce take 9.2 s against LEAP's 10.5 s for its chunked back projection,
and the filter is a tenth of LEAP's.  The difference is the volume's trip
back to the host.  mbirtorch's gather copies each device's shard into its
slot of one host array, and the shards are cut along the volume's last axis,
so every copy lands in a strided destination and the four copies run one
after another at about 1.2 GB/s.  LEAP's slabs are cut along the first axis
of its (z, y, x) volume, so each slab is one contiguous block, and the four
threads copy their slabs concurrently.  The host-to-device side costs
mbirtorch 2.2 s for 21 GiB, which is the pageable copy rate.

This cost is paid only when the caller asks for the volume on the host.
Inside `recon` with no initial reconstruction the FDK result stays on the
devices.  The collaborators' script asked for the FDK on the host and then
passed it back as the initial reconstruction, which moves the volume twice.

The two kernels differ in how they weight the detector samples, not in
whether they are transposes.  Both are matched pairs: LEAP's default
projector is `SF`, which `get_projector` confirmed, and its unmatched
voxel-driven option was not used.  LEAP's `coneBeamBackprojectorKernel_SF`
runs one thread per (x, y) column and eight slices.  For each view it
collapses the horizontal footprint to two column samples with precomputed
weights, and for each slice it reads the projection through a 3-D texture at
a fractional row coordinate, so the row interpolation is done by the texture
hardware.  That is two texture fetches per voxel and view, or four where the
footprint spans three rows.  mbirtorch's `_cone_back_kernel` runs a tile of
pixels by slices per program.  For each view it forms the trapezoid weight of
every tap in a 3 by 3 stencil in the kernel, the radius being 1 for this
geometry, and gathers the nine samples from global memory.  It is the exact
transpose of the forward kernel's rectangle footprint model.  LEAP's texture
scheme is what its manual calls "very, very slightly less accurate" than the
rectangle model.

### 3.5 LEAP's transfer share on one card

Greg asked whether LEAP is compute bound at this size.  LEAP accepts torch
tensors already on a card, and then a call moves nothing between host and
device, so the same call timed with host arrays and with device tensors
separates the copies from the kernel.  The script `leap_transfer.py` ran
both on one H100 in the interactive session.

| call, one H100 | data on device | data on host | transfer share |
|---|---|---|---|
| forward projection | 47.7 s | 50.3 to 54.2 s | 5 to 12 percent |
| back projection | 24.0 s | 26.9 to 28.5 s | 11 to 16 percent |
| FBP, in place | 25.3 s | 34.1 s | 26 percent |

The raw copies ran at 10.9 GB/s to the card and 4.2 GB/s back, for the
21 GiB sinogram in pageable memory.  On one card LEAP is compute bound: a
projection's copies are about a tenth of the call.  With chunking on four
cards the same calls took 17.0 s and 10.5 s, which is 2.8 and 2.3 times
faster for four times the cards, so about a third of a four-card call is
chunk overhead: the per-job copies of overlapping rows through pageable
memory, the texture copies, and the host-side row extraction.  Two
comparisons follow.  LEAP's back-projection kernel is about 1.5 times faster
per card than mbirtorch's, 24.0 s on one card against 9.2 s on four with the
data resident, which agrees with the ratio of 1.19 measured at N=1024.  And
mbirtorch's multi-device path carries no chunk overhead, which is why it
wins on four cards despite the slower kernel.

## 4. Why LEAP uses so little GPU memory

LEAP owns no arrays.  Its Python interface takes numpy arrays in host memory,
and each call copies what it needs to the GPUs and back.  With more than one
GPU, `project` splits the detector rows into jobs of at most
`maxSlicesForChunking` rows, 128 by default, and `backproject` splits the
volume into jobs of at most 128 slices.  One OpenMP thread per GPU takes jobs
from the list.  For each job the card receives the projection rows of the
job and the volume slices those rows can see, computes, and returns the
result.  The relevant code is `project_multiGPU` and
`backproject_FBP_multiGPU` in `src/tomographic_models.cpp` of LEAP 1.26.

For this geometry the jobs are small.  The script `leap_chunks.py` prices
them from LEAP's own slab formulas, and the job ranges LEAP printed in its
debug log match the script's ranges.

| call | job | projection part | volume part | widest job |
|---|---|---|---|---|
| forward projection | 128 detector rows | 1.87 GiB | 139 to 278 slices, 0.96 to 1.92 GiB | 3.79 GiB |
| back projection | 128 volume slices | 155 to 291 rows, 2.27 to 4.25 GiB | 0.88 GiB | 5.13 GiB |

LEAP adds a reserve of 0.25 GB per job, and a CUDA context takes about
0.5 GiB per card.  These figures give the 6.2 GiB measured on the chunk
cards.

The larger figure on the first card comes from the regularizer.
`TVgradient` on a host array copies the whole volume to one card and
allocates a whole output volume there when the two fit.  It chunks only when
they do not.  The code is `tomographicModels::TVgradient` in the same file.  Two volumes
of 8.9 GiB are the 18.9 GiB on the first card.

Two conditions switch the chunking off.  With one GPU, LEAP chunks only when
the whole job does not fit, and otherwise places the whole projection set and
the whole volume on the card, as the one-card ablation shows.  With any
number of GPUs, a job that LEAP's memory estimate says fits is not split
further.

The price of this design is paid in host memory and host time.  Their loop
holds five sinogram-shaped arrays and seven volumes in host memory, which is
the 175 GiB host peak in both their table and ours.  Every iteration moves
the whole volume to the cards and the whole sinogram back for the forward
projection, and the reverse for the back projection.  The loop's own
arithmetic on 21 GiB arrays runs in single-threaded numpy.  That part of the
iteration does not shrink with more cards.  It took about 18 s on two cards
and 21 s on four, while the two projections fell from 50 s to 28 s.

## 5. What this means for mbirtorch

The device memory mbirtorch reports has three layers, and only the first is
the arrays the algorithm needs.  On four devices the persistent set is 15
GiB per card, the peak of arrays in use is 29 GiB, and NVML reports 39 GiB.
A comparison by `nvidia-smi` therefore overstates mbirtorch's need by about
a third.  Whether the caching allocator's reserve can be reduced without a
time cost, for example through `PYTORCH_CUDA_ALLOC_CONF`, is a measurement
this page does not make.

The host-resident reconstruction plan in
`plans/features/device_memory_tiers/host_resident_layout_plan.md` keeps the
reconstruction-shaped arrays in host memory and moves rows per subset.  For
this scan those arrays are 4.5 GiB of the 15.3 GiB persistent set per card
on four devices, because the sinogram is 2.4 times the volume.  The larger
lever here is on the sinogram side, which that plan lists among its
alternatives.

LEAP's loop reaches its low device footprint by keeping 175 GiB in host
memory and by moving the whole problem through the cards on every call.
mbirtorch runs the same scan in 73 GiB of host memory.  On a cluster node
with 126 GB of host memory per GPU, the LEAP loop needs two GPUs' worth of
host memory to run at all.

## 6. Files

- Slides and the one-page note from ORNL: `MBIRtorch_v_LEAP_ORNL_09Sep2026.pdf`
  and `LEAP_vs_MBIR_Torch_Practical_Comparison_20260910.pdf`, kept outside
  this repository.
  The one-page note is a separate micro-benchmark on volumes of at most 96
  voxels per side with mbirtorch's compiler and Triton disabled.  It is not
  the measurement reproduced here.
- Harness, run detail, and records: `plans/experiments/features/leap_comparison/ornl/`
  with `ornl_harness.md` and `results/` holding the results files, the job
  logs, and the slice figures.  The repository's ignore rules exclude
  `.json`, `.log`, `.png`, and `.sbatch` files, so those stay in the working
  tree and on the cluster and are not committed, as for the earlier LEAP
  comparison.  The tables above carry every number they hold.
- Cluster copies: `/scratch/gautschi/buzzard/leap_ornl/` on gautschi,
  including the two full reconstructions as float32 `.npy`.
