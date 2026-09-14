# Host-resident reconstruction arrays in the VCD loop: plan

Status: FOLDED IN on 2026-09-14.  The plan of record is now
`plans/features/device_memory_tiers/device_memory_tiers_plan.md`, which
carries this design as its Tier 2 and cites the numbers below.  The text
below is unchanged from the revision of 2026-09-13.

Original status: DRAFT, written 2026-09-13 and revised the same day after a
three-reviewer panel (accuracy, reasoning, style).  Greg asked the question
it answers on 2026-09-13.  Can every reconstruction-shaped array in
`_vcd_recon` stay in host memory, with its rows moved to the device as the
projections need them, while the sinogram-shaped arrays stay on the device?
And what would that save?  The numbers come from the memory ledger, recorded
in `plans/experiments/features/host_resident_recon/hr1_ledger_residents.md`,
and from the LEAP comparison record in
`plans/experiments/features/leap_comparison/results/leap_benchmark_results.md`.
File paths are given from the root of the mbirtorch repository.

## Executive summary

The loop holds three reconstruction-shaped arrays for its whole run: the
reconstruction, the Hessian diagonal, and in prox mode the prox input.
Every projector call already takes subset-sized rows, so the projectors
need no change.  What changes is where those three arrays live and how each
subset's rows reach the device.

The change has two tiers.  Tier 1 moves the two read-only arrays, the
Hessian diagonal and the prox input.  Their rows can be copied ahead of
need, because nothing in the loop writes them.  Tier 2 moves the
reconstruction.  Its rows and their prior neighbors are gathered on the
host per subset, and the scaled update comes back per subset.  The qGGMRF
function splits into a gather step and a compute step, and the exchange of
boundary slices between devices is no longer needed.

The saving is the three arrays.  At the 2048-class they are 64 GB, or 96 GB
in prox mode, and they are 40 to 85 percent of the arrays the loop holds
for its whole run, depending on the case.  They are a smaller share of the
loop's peak, because the peak also holds the back projection's transients.
Priced on the Mac with the torch projection bodies, the share of the peak
is 20 to 40 percent.  The share on an H100 lies between those two figures,
and the first increment prices it there.  Section 3 gives the table.

The time cost is throughput, not waiting.  The subsets of a partition are
disjoint, so a subset's own rows are never stale, and only the neighbor
rows the previous subset changed must be refreshed.  The traffic per
iteration is about six reconstruction volumes to the device and one back.
Section 4 bounds it with the measured iteration times of the LEAP record.
Without overlap it is up to a fifth of an iteration at the 512-class and up
to a tenth at the 2048-class.  With overlap it is smaller, by an amount the
first increment measures.

The recommendation is one host-resident layout that holds both tiers.  The
automatic device policy chooses it only after the device-resident layout
has failed to fit at every admitted device count, and `configure_devices`
gains an explicit switch.  The read-only tier is not made unconditional,
because a host operation per subset is a cost the project has measured
before, and the first increment measures it here before any code lands.
Section 7 lists the decisions.

## Status by increment

| Increment | Delivers | Status |
|---|---|---|
| 0 | The H100 measurements: the ledger with the kernel bodies, the two transfer rates, the per-subset host cost, and their record | Not started |
| 1a | The row-source interface with the device implementation, a pure refactor | Not started |
| 1b | The host implementation for the read-only arrays, behind a switch | Not started |
| 2 | The reconstruction on the host, one device | Not started |
| 3 | The reconstruction on the host, several devices | Not started |
| 4 | The pre-loop phases on the host, if Increment 0 shows they bound the peak | Not started |
| 5 | The ledger, the device policy, the host budget, and the documentation | Not started |

## Rule for the code

Nothing from this plan goes into the code.  Docstrings, comments, and
Sphinx pages describe what the code does, in plain words.  They do not
name this plan or its increments, they do not use its vocabulary as labels,
and they carry no dates, names, or history.  A comment explains a rule the
code follows only when a later change would otherwise break it, and then it
states the rule itself.

## Terms

The plan uses these words in one sense each.

- A **row** is one pixel's values through every slice.  The loop's
  reconstruction-shaped arrays have shape `(num_pixels, num_slices)`, so a
  row is one line of such an array.  The code calls the same thing a
  cylinder.
- A **band** is the range of slices one device owns on a layout with
  several devices.
- A **resident** is an array the loop holds for its whole run.  The
  ledger calls the set of them the persistent set.
- The **widest subset step** is the loop phase with the largest charge per
  device.  It is what the device policy fits to memory.
- The **device-resident layout** is the layout of today.  The
  **host-resident layout** is the one this plan adds.
- **Page-locked memory** is host memory the operating system may not move
  or swap out.  A device can copy from it without a staging pass, and the
  copy can run asynchronously.  The word "pinned" is not used here, because
  the code already uses it for the pinned device count.
- A **size class** names the volume: the 2048-class is a 2048-cubed
  reconstruction.
- **The ledger** is the memory ledger in `mbirtorch/_memory_ledger.py`.

## 1. What the loop holds today

The ledger lists six residents: the error sinogram, the weights, the
reconstruction, the Hessian diagonal, the prox input, and the partitions.
The prox input is present only in prox mode.  The partitions sit on the
lead device and are small.  The measured sinogram is not a resident.  Each
subset step adds transients: the gathered prior rows, the update direction,
the delta sinogram, which is the size of a full sinogram, and the
projection's view batch.  On a weighted run the back projection step also
holds a weighted error sinogram, again the size of a full sinogram.

The hr1 record prices the residents at the 2048-class, in gigabytes:

| Quantity | Dense, qGGMRF | Dense, prox | Sparse, qGGMRF | Sparse, prox |
|---|---|---|---|---|
| Error sinogram and weights | 96 | 96 | 16 | 16 |
| Reconstruction-shaped residents | 64 | 96 | 64 | 96 |
| Persistent set | 160 | 192 | 80 | 112 |
| Widest subset step, priced with the torch bodies | 328 | 360 | 208 | 240 |

Dense means 3072 views of 2048 by 2048.  Sparse means 512 views.  The
widest subset step is the back projection at the coarsest granularity, and
88 GB of each widest-step total is the torch bodies' back batch, which the
Triton kernels on an H100 do not have.

The loop reads or writes the reconstruction-shaped residents in five
places:

- the prior gathers the subset's rows and its four in-plane neighbors'
  rows from the reconstruction, or in prox mode the subset's rows of the
  reconstruction and of the prox input;
- the update direction gathers the subset's rows of the Hessian diagonal;
- the positivity constraint, when on, gathers the subset's rows of the
  reconstruction again;
- the state application adds the scaled update into the reconstruction in
  place;
- the per-iteration statistics take the ell-1 norm of the whole
  reconstruction, in the loop body rather than in the subset updater.

The four in-plane neighbor gathers are four separate subset-sized sets,
one per offset, in `mbirtorch/qggmrf.py`.  On several devices a sixth
place reads the reconstruction: the halo staging takes one boundary slice
from each band once per pass over the partition.

The phases before the loop also use reconstruction-shaped device buffers.
The direct reconstruction scatters a full back projection into one.  The
Hessian diagonal is scattered the same way.  The initial forward
projection gathers the whole starting volume at once on one device, and
in pixel batches only on several devices, where the batching belongs to
the device-to-device transfer of rows.

On several devices the reconstruction is divided by slice.  Each device
computes its own band of every subset step.  The prior needs slices that
belong to a neighboring band, and a halo exchange copies them once per
pass over the partition.  The forward projection needs the update direction
over all slices, so each device that owns views receives the other bands'
rows by device-to-device transfer.  The back projection produces partial
sums, which are added onto the device that owns each band.  The line
search combines its partial sums on the lead device.

## 2. The design

### 2.1 Tier 1: the read-only arrays

The Hessian diagonal and the prox input are stored in host memory as
`(num_pixels, num_slices)` tensors.  The rows move to the device in two
steps, once per subset and per band.  First, `index_select` gathers the
band's rows on the host into a page-locked staging buffer.  Second, a side
stream copies the buffer to the device.  The update direction and the prox
gradient then use the rows exactly as they do today.

Nothing in the loop writes these two arrays.  The gather and the copy for
subset `k + 1` are therefore issued while subset `k` runs, and the copy
finishes before the next subset needs the rows.  The subset order of a pass
is known when the pass starts, so the next subset is always known.

A caller may pass the prox input as a device array, as a Plug-and-Play
loop does today.  Such an input is copied to the host once, at the start of
the call.

### 2.2 Tier 2: the reconstruction

The reconstruction is stored in host memory.  The qGGMRF gradient and
Hessian function splits into a gather step and a compute step.  The gather
step takes the subset's rows and the four neighbor sets.  It does this the
same way whether the reconstruction is on the host or on a device.  On the
host it also gathers a band's boundary slices directly, so the halo
exchange is not needed.  The compute step is the existing arithmetic on the
gathered rows, and it is the same on both placements.  The denoiser calls
the same function and keeps the device placement.

The positivity constraint reuses the subset's rows.  The state application
copies the scaled update to the host and adds it into the reconstruction
there.  The error sinogram update stays on the device.  The per-iteration
ell-1 statistic runs on the host.  The initial forward projection reads the
starting volume from the host in pixel batches, which the single-device
forward projection does not do today and Increment 2 adds.

The traffic is not serialized by the subset order.  The subsets of a
partition are disjoint, so subset `k + 1`'s own rows are unchanged by
subset `k`'s update, and they can be gathered a whole pass ahead.  The
neighbor rows of subset `k + 1` that belong to subset `k` are the only
stale ones.  For a random partition with `S` subsets they are about one
part in `S` of the neighbor gather: at the default finest granularity of
128 subsets, about one percent.  The gather for subset `k + 1` is therefore
issued while subset `k` runs, and the stale rows are patched after subset
`k`'s update, either from the device, which holds the scaled update, or by
a second small gather.  The intersection of the two index sets is computed
once per pair of consecutive subsets per pass.  What remains is
throughput: the host must gather, and the link must carry, six
reconstruction volumes per iteration.

### 2.3 One loop, two placements

The loop keeps one body.  A small interface, the row source, supplies the
rows of a reconstruction-shaped array for a subset and a band, and accepts
the update for them.  One implementation reads and writes a device tensor,
as today.  The other gathers from a host tensor into page-locked staging
buffers, copies, and adds the update back.  The loop, the compute steps of
the kernels, and the multi-device flow are written once against the
interface.  The row source separates placement from arithmetic, as the
projectors separate their driver from their compute bodies.

### 2.4 Several devices

On several devices, the host gathers the rows for every band in one pass
and copies each band's rows to its own device.  Each band's update returns
to the host.  The halo exchange goes away, and so does the slice placement
of the reconstruction-shaped residents.  Four things are unchanged:

- each device computes its own band of every subset step;
- the device-to-device transfer of the update direction to the devices
  that own views;
- the band reduce of the back projection;
- the combination of the line search's partial sums on the lead device.

One host serves every device.  The link traffic per device falls with the
device count, but the host gather does not, so the host's gather rate
bounds the layout at large device counts.  Increment 0 measures the rate
under the thread layout the loop uses.

### 2.5 Page-locked memory and streams

Only the staging buffers are page-locked.  They are subset-sized, and the
rows move through them in pixel batches of a fixed size, so their total is
bounded by the batch and not by the subset.  The backing arrays stay in
ordinary host memory.  This matters on a cluster node: page-locked memory
cannot be reclaimed, so page-locking a large fraction of an allocation
invites a late kill by the memory limit when a transient arrives, not a
clean failure at the allocation.  A page-locked allocation that fails at
once falls back to ordinary memory with one warning that names the cost.

Copies run on one side stream per device, and the consumer waits on the
copy's event.  The next subset's read-only rows are held in a second set
of buffers while the current subset's are in use.  A page-locked copy is
assumed to be about twice as fast as a pageable one.  Increment 0 measures
both.

### 2.6 The ledger and the device policy

The ledger gains a host budget beside its device budget.  The host budget
holds the host-resident arrays, the staging buffers, the gather
temporaries, and the caller's own arrays: the sinogram the caller passed
and the reconstruction the call returns, both of which sit in host memory
on every run.  The device plan drops the residents that moved.

The policy chooses the host-resident layout as a second pass.  The
device-resident search runs first, over its speed-ordered device counts,
as it does today.  Only when no admitted count fits does the policy price
the host-resident layout, and it chooses that layout when it fits.  Host
residency never replaces a device-resident layout that fits, because the
device-resident layout is faster at every size.  `configure_devices` gains
an explicit switch that forces either layout.

Two details follow.  Today the policy makes no choice when fewer than two
devices are visible and leaves a single-device overflow to the allocator's
own error.  With host residency as a candidate, a single device needs the
ledger too, so that path gains a ledger build when the switch is not set.
And the host-memory check must hold: at the dense 2048-class the three
moved arrays are 96 GB, the caller's sinogram is 48 GB, and the returned
reconstruction is 32 GB, which together exceed the 126 GB of host memory a
Gautschi allocation provides per GPU requested, so such a run requests two
GPUs for their host memory.

The ledger's calibration mode measures device peaks only.  Increment 5
adds a host high-water counter beside it, so the host budget is measured
the way the device plans are.

### 2.7 The interface and the callers

`recon`, `prox_map`, and `recon_direct` keep their signatures and return
numpy by default, as now.  On a host-resident run `output_sharded=True` is
refused with a clear error, because there is no device form to return.
This follows the rule that a request the layout cannot honor is refused
rather than answered with something else.  A Plug-and-Play loop that passes
volumes as device arrays then passes them through host memory instead.
The MACE4D design already works this way, and its planned `denoise_stack`
moves its batches between host and device.

The checkpoint contract changes in one place.  `_vcd_recon` with
`return_checkpoint=True` returns the loop's own final tensors without a
copy.  On a host-resident run the Hessian diagonal in that dictionary is a
host tensor, and a resume uploads its rows again.  The resume path skips
the phases before the loop, so Increment 4 does not apply to it.

The specialized reconstructions, `recon_split_sino` and
`recon_plastic_metal`, call the engine several times per run.  The staging
buffers are cached per model, not per call, so those runs pay the
page-locked allocation once.  `compute_prior_loss` reads the whole
reconstruction, which on a host-resident run is one full upload per
iteration.  It is a debugging path and stays as it is.

## 3. Memory

The saving is the reconstruction-shaped residents.  As shares of the two
quantities the ledger prices, from the hr1 record at the 2048-class:

| Case | Share of the persistent set | Share of the widest subset step, torch bodies | Saving in GB |
|---|---|---|---|
| Dense, qGGMRF | 40% | 20% | 64 |
| Dense, prox | 50% | 27% | 96 |
| Sparse, qGGMRF | 80% | 31% | 64 |
| Sparse, prox | 85% | 40% | 96 |

The first column is what the arrays are.  The second is their share of a
peak that the Mac's torch bodies inflate by 88 GB.  The share of the peak
on an H100 lies between the two columns, and Increment 0 reads it from the
ledger there.  For the dense cases the fractions hold at every size,
because the sinogram grows with the volume.  For a fixed view count they
do not, because the sinogram then grows as the square of the size and the
reconstruction as the cube.  On several devices each device holds nearly
its share of every resident, apart from the partitions on the lead device
and the terms a transfer adds.

Four alternatives were considered.

- Gathering the union of the subset and its neighbors once, instead of
  five overlapping sets, saves little.  For a random subset of density `p`
  the union is `1 - (1 - p)^5` of the volume against `5 p`: at the finest
  default granularity the saving is under two percent.
- A spatially blocked partition would cut the neighbor traffic several
  fold, but it changes the VCD trajectory and every baseline.  Not
  considered further.
- Storing the Hessian diagonal in half precision on the device halves one
  32 GB array with no traffic and no new machinery.  The array is a
  preconditioner, so the fixed point is unchanged, but the trajectory and
  every baseline change.  It is a separate, smaller option, and Section 7
  raises it.
- `recon_split_sino` already exists and divides both sinogram-shaped and
  reconstruction-shaped arrays by the number of parts, two for cone beam.
  It is approximate at the seam and not available for every geometry.  For
  a dense scan it removes more than this plan does.  For a sparse-view or
  prox-heavy run this plan removes more.  The two compose.

Two larger levers lie outside this plan.  The delta sinogram transient
and the weighted error sinogram are each the size of a full sinogram.  And
the direct reconstruction holds three full sinograms at once: the
sinogram, the weights, and the filtered sinogram.

## 4. Time

Tier 1's copies overlap the loop entirely, but the host operations that
issue them do not come free.  A host operation per subset is the mechanism
behind a measured 35 percent slowdown recorded in the engineering lessons,
and the loop's subset updater is written so that it has no host
synchronization at any device count.  Increment 0 measures the cost of the
added host operations at demo sizes before any code lands.

Tier 2's traffic per iteration, for the qGGMRF prior, is six
reconstruction volumes to the device and one back.  The six are the
subset's rows, four sets of neighbor rows, and the Hessian rows.  For a
prox map it is three volumes in and one back.  The partitions cover 0.78
of the pixel grid under the default mask, so one volume of traffic at the
2048-class is 25 GB.  The table gives the traffic against the steady-state
iteration times the LEAP record measured on one H100, with the 2048-class
iteration time extrapolated from the 512-to-1024 ratio of 11.2.

| Size class | Traffic in, GB | Traffic out, GB | Iteration time, s | Traffic time at 20 and 10 GB/s, s | Share of the iteration, no overlap |
|---|---|---|---|---|---|
| 512 | 2.3 | 0.4 | 1.59 | 0.36 | 22% |
| 1024 | 18.8 | 3.1 | 17.7 | 2.8 | 16% |
| 2048 | 151 | 25 | 198, extrapolated | 23 | 11% |

The traffic time assumes 20 GB/s for page-locked copies over the host link
and 10 GB/s for the host gather of random rows, and it adds the two.  Both
rates are assumptions until Increment 0 measures them.  The measured
iteration times grow more slowly than the fourth power of the size, so the
share falls with size by less than a pure scaling argument suggests.

The share with overlap is smaller.  The gather and the copy for the next
subset run while the current subset projects, and what remains on the
critical path is the patch of the stale neighbor rows and the issue cost of
the host operations.  Increment 0 measures that remainder.  On several
devices the host gather does not divide by the device count, so the
share grows with the count.

## 5. Increments

### Increment 0: the measurements

Files: scripts and sbatch files under
`plans/experiments/features/host_resident_recon/`, with companion `.md`
records.

One job on an H100 node, about two GPU-hours, with four parts.

- The ledger, priced on the H100 with the Triton kernel bodies, for the
  three hr1 geometries, over every phase.  It reports the widest subset
  step, the dominant phase of the run, and the reconstruction-shaped share
  of each.  This replaces the torch-body column of Section 3.
- The two rates, at the 1024- and 2048-class, from three measurements.
  The first is the host `index_select` of one subset's six row sets into
  page-locked staging, at granularities 16 and 128.  The second is the copy
  of those rows to the device, page-locked and pageable.  The third is the
  copy of one subset's update back with the host `index_add_`.  The largest
  page-locked allocation the node grants is recorded too.
- The per-subset host cost, at the 256- and 512-class: the current loop
  run warm, with the host gather, the copy, and the event issued beside the
  device gather and the result discarded, against the current loop.  This
  isolates the added host operations without changing any result.
- One steady-state iteration time at the 2048-class on four devices,
  device-resident, to replace the extrapolation in Section 4.

Increment 0 ends when the record holds all four.

### Increment 1a: the row-source interface

Files: `mbirtorch/tomography_model.py`, `mbirtorch/_sharding.py`, and the
tests.

Write the interface of Section 2.3 with the device implementation only,
and move the loop onto it.  This is a refactor.

Tests: every existing test passes unchanged, and the demo-size cells of the
timing harness show no change beyond their run-to-run spread.

Increment 1a ends when both hold.

### Increment 1b: the read-only arrays on the host

Files: `mbirtorch/tomography_model.py`, `mbirtorch/_memory_ledger.py`, and
the tests.

Write the host implementation for the Hessian diagonal and the prox input:
the page-locked staging buffers with the pageable fallback, the side-stream
copies with events, and the second buffer set for the next subset.  Put it
behind an explicit switch on `configure_devices`.  The policy does not use
it yet.  Drop the two persistent terms from the ledger's device plan behind
the same switch.

Increment 1b has three tests:

- a reconstruction and a prox map with the switch on equal the
  device-resident results to a relative maximum difference of 1e-6, on one
  device and on two CPU workers;
- the fallback path is exercised by forcing the allocation to fail;
- the ledger's device totals drop by the two terms.

Increment 1b ends when these pass, the full suite passes unchanged with the
switch off, and the per-subset cost measured in Increment 0 is within
whatever bound Greg sets on it.

### Increment 2: the reconstruction on the host, one device

Files: `mbirtorch/qggmrf.py`, `mbirtorch/denoising.py`,
`mbirtorch/tomography_model.py`, `mbirtorch/projectors.py`, and the tests.

Split the qGGMRF function into gather and compute steps, with the compute
step equal to the current arithmetic on the gathered rows and the denoiser
kept on the device placement.  Move the reconstruction onto the row
source, with the prefetch-and-patch of Section 2.2 and the update added on
the host.  Move the ell-1 statistic to the host.  Add pixel batching to the
single-device initial forward projection so it can read the starting
volume from the host.

Increment 2 has four tests:

- the compute step on gathered rows equals the current function to 1e-7
  relative;
- a reconstruction with stop threshold zero equals the device-resident
  form to 1e-6 relative on every device, for the qGGMRF and prox priors
  and with positivity on;
- the denoiser tests pass unchanged;
- the baseline tests pass on both layouts.

Increment 2 ends when these pass.

### Increment 3: the reconstruction on the host, several devices

Files: `mbirtorch/tomography_model.py`, `mbirtorch/_sharding.py`, and the
sharded tests.

Gather each band's rows with its boundary slices on the host, remove the
halo exchange for the host layout, and return each band's update to the
host.  Keep the device-to-device transfer of the update direction and the
band reduce as they are.

Increment 3 has three tests:

- the sharded tests pass on the host layout with `['cpu', 'cpu']` workers;
- cross-count agreement holds to 1e-3, the tolerance already used for
  compiled runs with uneven splits;
- a run on two devices equals a run on one to that tolerance.

Increment 3 ends when these pass and, on a two-GPU node, a run at a size
that does not fit on one device with the device-resident layout completes
with the host-resident one.

### Increment 4: the phases before the loop

Files: `mbirtorch/tomography_model.py`, `mbirtorch/projectors.py`.

Scatter the direct reconstruction and the Hessian diagonal into host
buffers in pixel batches.  This increment runs only if the H100 pricing of
Increment 0 shows a pre-loop phase bounding the run's peak on the
host-resident layout.

Increment 4 ends when the run's dominant phase, priced on the H100, is a
loop phase.

### Increment 5: the ledger, the policy, and the documentation

Files: `mbirtorch/_memory_ledger.py`, `mbirtorch/tomography_model.py`,
`docs/source/`.

Price both layouts, add the host budget with the caller's arrays, add the
host high-water counter to the calibration mode, make the policy choose
as Section 2.6 says including the single-device path, wire the switch to
the policy, refuse `output_sharded=True` on the host layout, document the
checkpoint contract, and update the device-policy and memory pages.

Increment 5 has three tests:

- on faked device and host sizes the policy chooses host residency exactly
  when no device-resident count fits and the host layout does, and never
  otherwise;
- the switch overrides it both ways;
- the host counter's reading is within the ledger's band on a small run.

Increment 5 ends when these pass and the documentation builds without
warnings.

## 6. Risks

| Risk | Severity | How to handle it |
|---|---|---|
| The per-subset host operations slow the loop at demo sizes | Medium | Increment 0 measures them before any code; the layout is chosen for capacity only, so demo runs never take it unless forced. |
| The host gather rate bounds the layout, and does not divide by the device count | Medium | Increment 0 measures it under the loop's thread layout; the policy prices it. |
| Page-locking too much host memory brings a late kill by the node's memory limit | Medium | Only the bounded staging buffers are page-locked; the host budget counts the caller's arrays. |
| The run's peak is a pre-loop phase on the H100 too | Medium | Increment 0 prices it; Increment 4 exists for that case and is skipped otherwise. |
| Two placements let the loop's forms drift apart | Medium | One loop body against one interface, and every test runs both forms. |
| The 2048-class iteration time is an extrapolation | Low | Increment 0 measures one. |
| The ell-1 statistic on the host moves the stopping rule at 1e-7 | Low | Already documented for the chunked sum; the tests run with the threshold at zero. |
| The checkpoint dictionary changes type for the Hessian diagonal | Low | Documented in Increment 5; the resume path uploads the rows. |

## 7. Decisions

1. One host-resident layout holding both tiers, chosen by the automatic
   policy only when no device-resident count fits, or by the explicit
   switch.  Recommended yes.  This replaces the earlier idea of moving the
   read-only arrays unconditionally.
2. Page-locked memory for the staging buffers only, with the backing
   arrays in ordinary host memory.  Recommended yes.
3. `output_sharded=True` refused on a host-resident run.  Recommended yes.
4. Increment 0 before any code, about two GPU-hours.  Recommended yes.
5. The half-precision Hessian diagonal as a separate option.  It saves
   16 GB at the 2048-class with no machinery and changes trajectories.  Not
   part of this plan.  Whether to try it is Greg's call.
6. Whether Increment 4 runs at all is decided by Increment 0's pricing.
