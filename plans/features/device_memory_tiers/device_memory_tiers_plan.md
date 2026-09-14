# Device memory in the VCD loop, in tiers: plan

Status: DRAFT, written 2026-09-14 and revised the same day after a
three-reviewer panel (accuracy, reasoning, style).  Greg took all seven decisions on 2026-09-14.  This plan supersedes
the host-resident plan at
`plans/features/device_memory_tiers/host_resident_layout_plan.md` as the
plan of record.  That plan's design becomes Tier 2 here, and its measured
numbers are cited from that file rather than repeated.  The new numbers come
from the reproduction of the ORNL comparison at
`plans/features/leap_comparison/ornl_reproduction.md` and from the ledger
pricing recorded in Section 1.  File paths are given from the root of the
mbirtorch repository unless a path starts with `plans/`.

## Executive summary

This plan reduces the device memory a reconstruction needs, in three tiers
of increasing cost.  Tier 0 applies to every reconstruction.  Tiers 1 and 2
apply only when the reconstruction does not otherwise fit.  A
reconstruction of the ORNL Inconel scan on four H100 cards shows what each
tier acts on.

That reconstruction reported 39.7 GiB on its busiest card.  Of that, 29.0
GiB was the peak of arrays in use, 9.4 GiB was memory the allocator held
but was not using, and the rest was the CUDA context.  The peak of arrays
in use is set by one setup step, the formation of the error sinogram, which
holds five sinogram-sized arrays at once.  The next phases below it are the
forward projection of each subset's update and the per-iteration error
statistics, each holding two sinogram-sized temporaries.  The arrays the
loop holds for its whole run are 15.3 GiB per card.

Tier 0 removes the temporaries that set the peak.  Four changes do this.  The error sinogram is formed in place.  The
subset forward projection accumulates into its destination instead of
assembling a copy.  The update subtracts with a scaled operand instead of
a temporary.  The error statistics reduce in chunks.  Each change keeps
the arithmetic.  The first increment measures which phases hold the peak
on the H100 before any of them is made.  Tier 0
also settles the allocator's reserve.  The setting that stops the reserve from growing was measured on this
scan: it halved the reserve at no time cost, and this plan decides how the
package offers it.  Together these changes are expected to bring the peak of arrays in use
from 29 GiB per card to about 24 GiB, and the reported figure from 40 GiB
to about 29 GiB.

Tier 1 applies when the device-resident layout does not fit at any device
count.  Today that case ends in a preflight failure whose message tells the
user to call `recon_split_sino` by hand.  In this plan the device policy
makes that choice itself, and `recon` runs the split.  The split
reconstructs the sinogram in row parts, each a smaller problem, and
stitches them.  It halves every array of the reconstruction, the residents
and the transients alike, so it is a two-fold lever, and the code for it
exists for cone beam and parallel beam.

Tier 2 is the host-resident layout of the host-resident plan.  It moves the
reconstruction-shaped arrays to host memory and streams their rows per
subset.  It is the larger saving for sparse-view scans, where those arrays
are most of the persistent set, and it combines with the split.

The policy prices each candidate with the memory ledger and takes the
cheapest one that fits.  At first the candidates are tried in a fixed
order.  Later a measured cost comparison chooses among them.  `recon` stays
one method.  What changes is the device plan the policy hands it, and the
explicit override is a model parameter.  Section 9 records the decisions.  Greg took all seven on 2026-09-14.

## Status by increment

| Increment | Tier | Delivers | Status |
|---|---|---|---|
| 1 | 0 | Measurements on the ORNL scan: the peak attributed to the loop's phases, the allocator setting on and off with the device count pinned, and a steady-state time per iteration | Measured 2026-09-14 (jobs 16406732 and 16409250); record in `dm1_record.md` |
| 2 | 0 | The co-live sinogram shards removed at the sites Increment 1 confirms, with the ledger updated | All five sites committed 2026-09-14 (mbirtorch a225319 and 3101e93), suite and goldens green; the H100 verification of the new peak and time is running (job 16416780) |
| 3 | 0 | The allocator setting as documentation, a hint line, and the opt-in call, plus a run-log line separating memory in use from the allocator's cache | `get_memory_stats` now reports the pool's peak and its unused part (mbirtorch 23c4a43); the rest not started |
| 4 | 1 | The device plan, the `recon` dispatch, the split mode priced by the ledger, and the explicit override | Not started |
| 5 | 2 | The host-resident mode: the host-resident plan's increments, starting with its H100 measurement | Not started |
| 6 | 1 and 2 | Both modes together, and the cost comparison that chooses between them | Not started |
| 7 | all | The user guide section on memory modes and the developer notes | Not started |

Each increment ends with a review before the next starts.  Increment 1's attribution, in `dm1_record.md`, confirmed the ledger: the
initial error state is the run's peak by 4.1 GiB over the next phase, and
the subset back projection is below it.

## Rule for the code

Comments and docstrings say what the code does and why, in plain English.
They carry no dates, names, plan numbers, tier numbers, or references to
this document.  A reader of the code should not be able to tell that a plan
existed.

## Terms

- **Persistent set**: the arrays the loop holds for its whole run.  On the
  device-resident layout they are the error sinogram, the weights, the
  reconstruction, and the Hessian diagonal, plus the partitions and a small
  workspace.
- **Transient**: an array that exists within one phase or one subset step.
- **Phase**: one named stretch of a reconstruction that the ledger prices on
  its own, such as the error sinogram formation, the Hessian diagonal, or a
  subset step's forward projection.
- **Allocated, reserved, and the reserve**: torch's allocator counts the
  bytes in live tensors as allocated, and the bytes it holds from the driver
  as reserved.  The reserve is the difference.  `nvidia-smi` and NVML report
  the reserved bytes plus the CUDA context.
- **Layout**: which device holds which arrays.  The device-resident layout
  is today's.  The host-resident layout is the host-resident plan's.
- **Mode**: what `recon` runs.  `standard` is today's loop on the
  device-resident layout.  `split` reconstructs the sinogram in row parts
  with `recon_split_sino`.  `host_resident` runs the loop on the
  host-resident layout.  `both` splits and runs each part host-resident.
- **Part**: one piece of a split.  Cone beam has two parts that overlap in
  rows and slices.  Parallel beam has any number of parts, each holding its
  kept rows plus an overlap on each interior side.
- **Device plan**: what the device policy returns: a device list, a mode,
  and the ledger that priced them.  The word plan alone means this document.
- **Ledger**: the memory ledger in `mbirtorch/_memory_ledger.py`, which prices
  every array of a candidate layout before any large allocation.
- **Speed rules**: the measured thresholds in the device policy that order
  the device counts to try.  They reorder the counts and never remove one.

## 1. Where the memory goes

The figures below are for the ORNL scan on H100 cards, with weights and an
initial reconstruction supplied.  That scan is 2132 views of 1456 x 1840
detector pixels on a 1360 x 1360 x 1296 volume.  The measured rows come from
`ornl_reproduction.md` Sections 3.1 and 3.2.  The four-device run is the
full 15-iteration reproduction, and the two-device run is the five-iteration
ablation.  The modeled rows come from the ledger, priced with the torch
bodies and with the bodies' batch terms removed, so that what remains does
not depend on the pricing device.  The script is `price_phases.py` in
`plans/experiments/features/device_memory_tiers/`.

| quantity | four devices, busiest card | two devices, busiest card |
|---|---|---|
| persistent set, modeled | 15.3 GiB | 30.4 GiB |
| error sinogram formation, modeled | 29.06 GiB | 57.89 GiB |
| subset forward projection, modeled, without the kernel batch | 27.23 GiB | 53.53 GiB |
| initial forward projection, modeled | 26.66 GiB | 52.41 GiB |
| per-iteration error statistics, modeled | 25.97 GiB | 51.71 GiB |
| peak allocated, measured | 29.03 GiB | 57.86 GiB |
| peak reserved, measured | 38.45 GiB | 69.84 GiB |
| NVML peak, measured | 39.67 GiB | 71.09 GiB |

The measured peak of allocated memory matches the error sinogram formation
phase to 0.03 GiB at both device counts.  That phase holds the sinogram, the
weights, the forward projection of the initial reconstruction, a scaled
copy of that projection, and the error sinogram, five sinogram-sized shards
at once, plus the initial reconstruction.  The subset forward projection
phase holds the delta sinogram and an assembled copy of it beside the
persistent set.  The error statistics phase holds two sinogram-sized
temporaries of its reduction.  The subset back projection, which holds the
product of the weights and the error sinogram, sits below all of these,
because the loop frees that product before the forward projection that
follows it.  The reserve is 9.4 GiB on four devices, a quarter of the NVML
figure, and 12.0 GiB on two devices, a sixth of it.

Two things follow from the table.  First, the sinogram-shaped arrays are
the larger share of this scan's persistent set, because the sinogram is 2.4
times the volume.  Second, the peak is set by temporaries, not by the
persistent set, and the temporaries are sinogram-sized.  The host-resident
plan's cases have a different balance.  In its sparse-view 2048-class case
the reconstruction-shaped arrays are 80 to 85 percent of the persistent set.

## 2. Tier 0

### 2.1 The co-live sinogram shards

Four sites hold sinogram-sized temporaries that the arithmetic does not
need.  Increment 1 confirms which of them hold the measured peak on the
H100, and Increment 2 changes those, in the order of their modeled size.

1. The error sinogram formation, `_initial_error_state` in
   `mbirtorch/tomography_model.py`, computes `sinogram - alpha * fwd` per
   shard while `fwd` is still alive, so the scaled projection and the
   difference are two extra shards.  Scaling `fwd` in place by `-alpha` and
   adding the sinogram into it forms the error in the buffer `fwd` already
   owns.  The operations are the same multiply and the same addition, so the
   result is identical bit for bit.  This removes two shards from the phase
   that holds the measured peak: 10.6 GiB per card on four devices and 21.3
   GiB on two.
2. The subset forward projection on several devices holds the delta
   sinogram and an assembled copy of it, the `forward assembly` term of the
   ledger.  Accumulating each device's contribution into the destination
   removes one shard from that phase.  The summation order changes only if
   the assembly today sums in a different order, which Increment 2 checks
   before it chooses a gate.
3. The update, `_apply_update` in the same file, subtracts
   `alpha * delta_sinogram` from the error sinogram through a temporary.
   Subtracting with the scaled operand of torch's `sub_` forms no temporary.
   torch may fuse the multiply and the subtraction into one rounding, so
   this site is gated at the project's relative thresholds rather than bit
   for bit.  The ledger does not charge this temporary today.  Increment 2 adds
   the charge before removing the temporary, so the ledger describes the
   code at every step.
4. The per-iteration error statistics, `get_forward_model_loss`, reduce
   `error * error * weights` through two full temporaries.  Reducing in
   chunks of views bounds the temporary to one chunk, which is the pattern
   the project already uses where a whole-array reduction was too large.
5. The dot products that set the scaling of the initial reconstruction,
   in the same function as site 1, hold the weighted projection and the
   product temporary of their reduction beside the sinogram, the weights,
   and the projection.  That is five sinogram-sized shards, the same count
   as the error formation held, so a weighted run's peak does not fall
   until this site changes too.  Chunking the two reductions removes the
   product temporary and changes only the order of a summation.  Dropping the weighted projection as well needs the reduction rewritten
   as `sum(w * f * f)`.  That form changes the scale in its last digits,
   and with it every later iterate slightly.  Decision 4 chose both steps,
   provided the time per iteration does not grow, because the
   reconstruction does not depend on the initial scale at convergence.

What the implementation of sites 1 to 4 found: the multi-device forward
projection already accumulates into its destination, so site 2 had no copy
to remove.  The ledger's `forward assembly` term is a deliberate margin
calibrated against measured peaks, and it stays.  The ledger's charge for
the weighted projection in the dot products was missing on the
multi-device branch and is now charged, which is why the plan's earlier
modeled figure for that phase was too low.  The update uses `addcmul_`
with the step as a tensor operand, because `sub_` with a scaled operand
reads its scale as a number and would synchronize with the host at every
subset step.

After sites 1 to 4, the ledger prices the error formation at 18.3 GiB
and the error statistics at 15.2 GiB per card on four devices, but the dot
products stay at 28.9 GiB until site 5 changes, and the measured
attribution agrees: a weighted run's peak stays at the initial error state
until then.  After site 5, the modeled
peak of arrays in use on four devices falls to at most the initial forward
projection's 26.66 GiB.
Whether it falls further depends on what that phase holds beyond the
persistent set.  Increment 1's attribution says whether that phase is real
on the H100.  The expectation of about 24 GiB rests on the same
attribution.  It is an expectation, not a result.

The fused weighted error, which an earlier draft placed here, is not in
this plan.  The product it removes is freed before the peak.  Fusing it into the
back kernel would therefore save nothing at the peak.  It would also add a
second gather per tap to a kernel that is limited by memory traffic.  A later attribution may show the back projection phase holding the peak.
The cheaper remedy then is to multiply per view batch in the driver
`sparse_back_project_view_range`.  That bounds the transient to one batch
for every body without a kernel change.

### 2.2 The allocator reserve

torch's caching allocator keeps freed blocks for reuse and, by default,
allocates a new segment when a request does not fit an existing free
block.  Over a loop of varying transients the reserve grows past the peak
of live tensors.  The setting `expandable_segments` lets the allocator grow
a segment in place instead.  The setting is read once, when the allocator
is first configured, which happens at the first allocation.  It is set
through the environment variable `PYTORCH_ALLOC_CONF`, or the older
`PYTORCH_CUDA_ALLOC_CONF`, before that point.  torch 2.13 also has a private
function that changes the settings at run time.  The project measured the
setting on an H100 at the 1024-class in
`plans/experiments/archive/torch_port/mg53_host_cost_split.md`: it moved no
wall time.  Increment 1 measured it on this scan, with the device count pinned to
four and the setting on in two runs.  The reserved peak on the busiest
card fell from 38.45 GiB to 33.87 GiB, so the reserve fell from 9.4 GiB
to 4.8 GiB, and the NVML peak fell from 39.66 GiB to 35.08 GiB.  The peak
of arrays in use was unchanged at 29.03 GiB.  The 15 iterations took
567.6 s at the default and 565.3 and 566.6 s with the setting, so the
setting cost no time.  The record is `dm1_record.md` in
`plans/experiments/features/device_memory_tiers/`.

The setting does not empty the reserve.  The allocator still caches
freed blocks by design, so the setting removes the growth from
fragmentation, not the cache.  The measurement bears this out: 4.8 GiB of
reserve remained with the setting on.

How the package offers the setting is Decision 3.  Setting the environment
variable at import would change the allocator for every torch user in the
process.  It would also do nothing when the package is imported after CUDA
has initialized, so behavior would depend on import order.  The plan's
recommendation is a documented setting plus one log line at the preflight
when it is unset, with the private run-time function as an opt-in that Increment 3
measures.

Two more facts bear on the decision.  torch's notes for the setting have
carried corner-case restrictions, the long-standing one being the sharing
of device tensors between processes, and none of them touches mbirtorch's
own workload.  LEAP neither uses nor needs the setting.  Its library
allocates device memory directly with `cudaMalloc` inside each call and
frees it at the end, so it holds no cache and shows no reserve, and it
pays a fresh allocation on every call instead.

Increment 3 also adds one line to the reconstruction log reporting, per
device, the peak of memory in use and the allocator's cache beyond it.
torch's counters separate the two inside the process, and nothing outside
the process can, so this line is where users get the split that
`nvidia-smi` cannot show.

Releasing cached memory at phase boundaries is not part of this plan.  It
returns free blocks to the driver and changes what NVML reports and what a
neighboring process can take, but not this run's peak, and the loop
reacquires the blocks at once.

## 3. Tier 1: the device plan and the split mode

### 3.1 What the code does today

`_apply_device_policy` in `mbirtorch/tomography_model.py` is called from
inside `_vcd_recon`, the engine shared by `recon` and `prox_map`, and from
the direct reconstructions with their own workload.  When the layout is
automatic and at least two devices are visible, it searches the device
counts in the order the speed rules give.  It prices each count with the
ledger and settles on the first that fits.  The settled record is kept per model and
per shape pair.  With one visible device it returns without a search.  When
no count fits, it raises `MemoryPreflightError` with a list of remedies.
For geometries that have `recon_split_sino`, the list names that function.
The list is built in `_memory_remedies`.  An explicit device list from
`configure_devices`, or the process-wide pin `MBIRTORCH_NUM_DEVICES`, skips
the search.

`recon_split_sino` exists for cone beam and for parallel beam.  For cone
beam it splits the detector rows into two overlapping parts and stitches
the results.  For parallel beam it splits into any number of parts.  Each part holds
its kept rows plus an overlap on each interior side.  The parallel form
already chooses the part count by pricing the largest part with the
ledger.  Both
build their parts with `copy_ct_model`.  A part with an automatic layout
chooses its own device count when it runs, and a part copied from a model
with an explicit layout keeps that layout.  The split refuses sharded
inputs and the `output_sharded` argument, returns per-part entries in its
dictionary instead of `recon_params`, and falls back to a standard
reconstruction when a part would be too thin.  `recon_plastic_metal`
already chooses `recon_split_sino` over `recon` for cone beam at its call
site.

### 3.2 The device plan

The policy moves above `_vcd_recon`, so that `recon` and `prox_map` decide
the mode before the engine runs, and the engine reads the settled record.
The policy returns a device plan: the devices, the mode, and the ledger.
The search order is:

1. the device counts in today's order, in `standard` mode;
2. if none fits, the same counts in `split` mode, for geometries that have
   `recon_split_sino` and for calls the split can serve;
3. if none fits, the same counts in `host_resident` mode, once Tier 2
   exists;
4. if none fits, `both`.

The search is aware of the call.  A call with sharded inputs, a call with
`output_sharded=True`, and the prox map, which has no split, exclude the
split candidates, and the settled record keys on the workload as well as
the shapes.  The single-device case runs the search too, over the modes at
a count of one, which is the case in which the split matters most.  The
first candidate that fits is settled, and the run log and the returned
dictionary name the mode.

`recon` reads the device plan and chooses a code path from the mode.
`standard` runs `_vcd_recon`, and `split` runs `recon_split_sino`.  The
parts of a split run in `standard` mode, or in `host_resident` mode once
that mode exists, and a part is never split again.  In automatic split mode
the dictionary carries `recon_params` for the whole as well as the per-part
entries.  The fallback to a standard reconstruction for a thin part
becomes a refusal, because the standard mode is what was refused.

### 3.3 Pricing the split

The split is priced as the ledger of its largest part, at each device
count.  For cone beam the part is a copy of the model holding half the
detector rows plus the overlap, and for parallel beam it is the largest
part at the chosen part count.  Both copies are made the way
`recon_split_sino` makes them, and parallel beam's existing part pricing
moves into the policy rather than being duplicated.  The parts run at the
device count the parent priced, carried to them as an explicit device list,
so that the pricing binds.

The host check counts what the split holds on the host.  It counts the
sinogram and weights the caller passed, the reconstructions of the parts
that are held until the stitch, and the stitched result.  The parts' slices
of the sinogram and weights are views, not copies.  The total is compared
against the host budget rule the host-resident plan states.

### 3.4 The explicit override

The override is a model parameter, `memory_mode`, set through `set_params`,
with the four mode names and a default of automatic.  It is not an argument
of `configure_devices`, because that call's device count defaults to one and
an explicit call would pin the run to one device.  An explicit mode is
settled without a search, and the ledger still runs its check.  A
process-wide pin, `MBIRTORCH_MEMORY_MODE`, mirrors the device-count pin for
test suites and nightlies.  The mode is propagated to the parts of a split by
the split code itself, since `copy_ct_model` carries only an explicit device
list.

### 3.5 What changes for a user

A reconstruction that fails today with the preflight message runs instead,
in the split mode, with a log line that says so.  The result of a split
reconstruction differs slightly from the standard result, because the parts
are stitched across an overlap.  The device budget is read from free memory
at call time, so on a shared card the same script could take different
modes on different days.  Decision 1 chooses between two answers.  The
first is that the automatic choice takes the split, records the mode in the
returned dictionary and the log, and offers the pin for reproducibility.
The second is that a refusal stays a refusal, with a message that names
`memory_mode='split'` as the one-line remedy.  The plan recommends the
first, because it is what the remedy text asks users to do by hand today.

## 4. Tier 2: the host-resident mode

The design is the one in the host-resident plan, and its Sections 1 and 2
remain the reference.  Those sections cover five things: the residents
table, the row source, the prefetch of the read-only arrays, the gather and
compute split of the reconstruction, and the page-locked staging buffers.
Increment 5 of this plan is that plan's increments 0 through 5, in their
order, beginning with its H100 measurement.  That measurement prices the
layout with the kernel bodies, measures the two transfer rates, and
measures the per-subset host cost.  Its decisions stand as written there,
with one change: its first decision chose the host-resident layout after
the device-resident search fails, and here the split candidates come
between the two.

Two seams are stated so the work is not done twice.  Increment 4 builds the
device plan with `host_resident` as a name the search does not yet offer,
and Increment 5 fills it in.  The host-resident plan's refusal of
`output_sharded` becomes one of the call conditions the search reads.

## 5. Choosing between the split and the host-resident mode

The two modes reduce different arrays.  The split halves every term of the
ledger, residents and transients alike, but a cone-beam split has two parts
and no more, so its saving is at most two-fold.  The host-resident mode
removes the reconstruction-shaped arrays whatever their share, which is
small on the ORNL scan and most of the persistent set on the sparse-view
2048-class case.  Increment 4 uses the fixed order of Section 3.2, split
first, because the split exists today and its cost is bounded by its two
passes.  Increment 6 replaces the fixed order with a cost comparison once
both modes have measured time overheads: the split's from Increment 4, the
host-resident mode's from Increment 5.  Among the candidates that fit, the
comparison picks the one with the smallest predicted time.  The speed rules
order device counts but give no time estimate, so the comparison needs the
two measured overheads and nothing else.

## 6. The measurement of Increment 1

The job runs on four H100 cards with the ORNL harness in
`plans/experiments/features/leap_comparison/ornl/`, with the device count
pinned to four so that every run has the layout the full reproduction
chose.

- The peak attributed to phases.  A sampler thread reads torch's allocated
  bytes on every card every few milliseconds and keeps the maximum per
  phase.  Thin wrappers on the model's own methods name the phases.  The
  phases are the direct reconstruction, the error sinogram formation, the
  Hessian diagonal, and within a subset step the forward projection, the
  back projection, and the rest.  Four iterations cover the first four granularities.  The
  library is not changed.  The sampler can miss a transient shorter than its
  period, so torch's own peak counter is reported beside it.
- The allocator setting.  The full 15-iteration reconstruction runs once at
  the allocator's default and twice with `expandable_segments`.  Each run
  records peak allocated, peak reserved, and the NVML peak per card, and the
  total time.
- The weighted product's cost.  On one card, for one four-device shard, the
  time of the whole-shard multiply and of the sparse back projection at the
  finest granularity.  This sizes the per-view-batch alternative of Section
  2.1 should it ever be needed.
- A steady-state time per iteration.  The log lines carry no timestamps,
  so the per-iteration figure is the difference between a 30-iteration run
  and a 15-iteration run divided by 15, both at the default allocator.
  Measured: 1104.5 s and 567.6 s, so 35.8 s per iteration, with about 31 s
  of one-time cost.  This is the reference for every later time gate.

The record is a companion file beside the scripts in
`plans/experiments/features/device_memory_tiers/`.  The job takes about four
GPU-hours.

## 7. Tests and exit conditions

Increment 2:

- The in-place error formation gives the same error sinogram bit for bit
  as today's expression.  The check runs on CPU and on CUDA when available,
  on one device and on a two-device placement when two devices exist.
- The scaled subtraction and any change to the assembly agree with today's
  values at the project's existing relative gates for those paths.
- The ledger's modeled run peak falls by the removed terms, the ledger test
  says so, and the ledger charges the update temporary in the code it
  describes.
- The measured peak of arrays in use on the ORNL scan falls to within one
  GiB of the ledger's new modeled peak.  The exit condition is on the run
  peak, not on any single phase.

Increment 3:

- With the setting on, the reserved peak on the ORNL scan is at least the
  amount Increment 1 measured below the default run's.  The per-iteration
  time is within the spread of the repeated runs.
- The log line appears when the setting is unset and not otherwise, and the
  opt-in function changes the setting only when called.

Increment 4:

- A geometry and size that fail the preflight today instead run in split
  mode.  The log and the returned dictionary name the mode.  The result
  matches `recon_split_sino` called directly with the same part count and
  device list.
- A size that fits runs in standard mode, unchanged, with the same result
  as before the change.
- The explicit parameter and the pin run the named mode, and an unknown name
  raises.
- A call with sharded inputs or `output_sharded=True`, and a prox map, are
  never offered the split, and refuse with the preflight message when the
  standard mode does not fit.
- The parts never split again, and a part that would be too thin refuses.
- The host check refuses a split whose host arrays exceed the budget.

Increments 5 and 6: the host-resident plan's tests, plus a test that the
cost comparison picks the cheaper mode on two constructed cases.

## 8. Risks

| risk | how the plan meets it |
|---|---|
| The attribution finds the peak elsewhere than the ledger says | Increment 2 follows the measured attribution, not the model, and the ledger is corrected where the two differ |
| `expandable_segments` interacts with the Triton kernels or with multi-device streams | Increment 1 runs the real loop with the setting on, twice, and checks values and time |
| Applying an allocator setting from a library surprises a host program | Documented setting and a log line by default; the run-time function only as an opt-in |
| The split mode changes results relative to the standard mode | Decision 1; the mode is recorded in the dictionary and the log, and a pin fixes it |
| The mode depends on free memory at call time | The same decision; the pin gives test suites and nightlies a fixed mode |
| The parts of a split choose layouts the parent did not price | The parent's device count is carried to the parts as an explicit list |
| `both` mode holds two overlapping host-resident parts | Its host check counts both parts' arrays, and each part keeps its own staging buffers |
| A resumed reconstruction changes mode between runs | The dictionary records the mode, and a resume with a different mode warns |
| Host memory limits on cluster nodes | The host check counts the caller's arrays, the parts' results, and the stitched result |

## 9. Decisions for Greg

Each decision is a question, with the context needed to answer it and a
recommendation.

1. When a reconstruction does not fit on the devices at any device count,
   may the library switch to the split reconstruction on its own?  Today it
   refuses, and the error message tells the user to call `recon_split_sino`
   by hand.  An automatic switch does what that message asks.  Two things
   weigh against it.  The split's result differs slightly from the standard
   one at the stitched seam.  And the choice reads free device memory at
   call time, so the same script could take different paths on different
   days, for example on a shared card.  Recommendation: switch
   automatically, name the mode in the log and in the returned dictionary,
   and provide the pin `MBIRTORCH_MEMORY_MODE` so tests and nightlies can
   fix the mode.  The alternative is to keep refusing with a clearer message.
   Decision (Greg, 2026-09-14): switch automatically, as recommended.

2. Where does the manual override live?  `configure_devices` is the
   existing switch for device choices, but its device count defaults to
   one, so adding the mode there would make an explicit mode silently pin
   the run to one device.  Recommendation: a model parameter,
   `set_params(memory_mode=...)`, together with the pin above.
   Decision (Greg, 2026-09-14): agreed, `set_params`.

3. How is the allocator setting delivered to users?  The setting
   `expandable_segments` removed 4.6 GiB of reserve per card on the ORNL
   scan at no time cost.  Setting it inside the package at import would
   change the allocator for every torch user in the process, and it does
   nothing when torch has already touched a device.  Recommendation:
   document the setting, print one hint line before a reconstruction when
   it is unset, and provide an explicit opt-in call.  The alternative is to
   set it at import when it is unset.
   Decision (Greg, 2026-09-14): the documented setting plus the hint line,
   as recommended.  Section 2.2 records the downsides of the import-time
   alternative that the choice weighed.

4. How far should the initial-scale computation change?  For a weighted
   scan, the computation of the scale applied to the initial reconstruction
   holds five sinogram-sized arrays at once, and after the committed
   changes it is what sets the run's peak.  Its two reductions can be
   computed in chunks, which removes one of the five arrays and changes
   only the order in which numbers are added.  A further rewrite removes a
   second array but changes the scale in its last digits, and with it every
   later iterate slightly.  Recommendation: chunk the reductions and stop
   there, unless the next
   measurement shows the peak still at this computation by more than one
   array's size.
   Decision (Greg, 2026-09-14): take both steps, provided the time does
   not grow, because the result does not depend on the initial scale at
   convergence.

5. Do the parts of a split reconstruction use the device count the parent
   chose when pricing them?  If each part chooses its own count when it
   runs, the parent's memory check was only advisory.  Recommendation: yes, carry the count to the parts as an explicit device
   list.
   Decision (Greg, 2026-09-14): yes.

6. Do the host-resident layout plan's six decisions stand?  They were made
   when that layout was the only alternative to a refusal, and the one
   change here is that the split is tried first.  Recommendation: yes, with
   that one change.
   Decision (Greg, 2026-09-14): yes.

7. Is the fused weighted error dropped for good?  The measurement closed
   the question: the product it would remove is freed before the peak, and
   removing it would save at most two percent of an iteration.
   Recommendation: yes.  Section 2.1 records the
   cheaper alternative if the peak ever moves to the back projection.
   Decision (Greg, 2026-09-14): yes, dropped.

## 10. Files

- This plan.
- The host-resident plan: `plans/features/device_memory_tiers/host_resident_layout_plan.md`.
- The ORNL reproduction: `plans/features/leap_comparison/ornl_reproduction.md`
  and the harness under `plans/experiments/features/leap_comparison/ornl/`.
- The Increment 1 scripts and their record under
  `plans/experiments/features/device_memory_tiers/`: `price_phases.py`,
  `dm1_phase_peaks.py`, `dm1_weighted_product_cost.py`, and
  `dm1_allocator_and_product.sbatch`.
