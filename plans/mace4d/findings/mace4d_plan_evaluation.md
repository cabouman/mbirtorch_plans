# Evaluation of the MACE4D migration plan, and a design that fits torch

Status: written 2026-09-13 and reviewed the same day.  Section 7 records
the decisions taken, and `plans/mace4d/plan.md` beside this page is
the plan of record that follows from them.  This page evaluates the plan
in `mbirtorch/plans/mace4d_migration_plan.md` (revised 2026-09-11) and
proposes changes where the plan copies a jax mechanism that torch does not
need.  The sources read for this evaluation are the plan, the mbirjax module
`mbirjax/mace4d.py` and its test, the mbirtorch modules `denoising.py`,
`qggmrf.py`, `projectors.py`, `tomography_model.py`, and `_sharding.py`, and
the MACE work already in the mbirtorch repository at `experiments/drunet/`
with its records in `plans/mace_poc/` and `plans/multi_slice_fusion/`.  The measurements in Section 5 come
from three scripts in `plans/mace4d/experiments/`.  Each script has
a companion `.md` with the run record.

## Executive summary

The plan is thorough and correct about where the difficulty is.  The outer
algorithm ports almost unchanged, and the one hard part is denoising
hundreds of hyperplane volumes as one program, which jax does with `vmap`.
A hyperplane volume is the 3D array obtained from the 4D reconstruction by
fixing one spatial coordinate, so it holds every frame.

The plan's recommended answer for that part, stacking the volumes along the
row axis and running the existing single-image sweep, is not equivalent to
the jax code.  It changes three things: every volume shares one line-search
step size, the last frame of one volume becomes a prior neighbor of the
first frame of the next, and the random pixel partition is drawn over the
stacked image.  The first two change the values, and the second changes
the minimizer, not only the path to it.  The plan's proposed repair, a row
of zeros between volumes, makes the second effect worse rather than removing
it, because a zero neighbor pulls the frame-edge voxels toward zero.
Section 5 gives the measured sizes of these effects.

The recommended answer is the one torch uses everywhere: an explicit leading
batch dimension.  The denoiser sweep becomes a function of a tensor of shape
`(num_volumes, num_pixels, num_slices)`, with per-volume step sizes, a
per-volume stopping test, and converged volumes frozen by a mask.  This is
exactly what jax's `vmap` over a `while_loop` computes, so the batched
result equals per-volume denoising and the equality is testable without a
reference.  It needs two batched functions and one public method on
`QGGMRFDenoiser`, and no change to the existing single-image path.

Four more changes make the port fit mbirtorch rather than mbirjax.  The
prior agents become a public method of the denoiser, so `mace4d.py` uses no
denoiser internals and no thread-local caches.  The data-fit agents follow
the conventions already validated in the nn_priors MACE work: initialize a
frame model once, pass the cumulative iteration count, and warm start from
the agent's own previous output.  The temporal filter becomes one small
matrix applied by a matrix multiply, on the host or on the device.  The
host arrays become torch tensors, so the consensus update runs on every
core, and the orientation permutations happen on the device per batch
rather than on the host per orientation.

One alternative architecture deserves a later experiment but not the port.
Three 3D qGGMRF priors in consensus have the same equilibrium as one 4D
qGGMRF prior with fixed direction weights (Section 4).  A single 4D agent
would need no batching, no permutations, and would reuse the existing
sharded denoiser.  The hyperplane form is kept for the port because it is
the form the neural-network prior program needs.

Section 7 records the decisions taken at review, and Section 8 the
discussion items the review raised.  The largest decision is that the port
reproduces mbirjax's method, not its trajectories, which the retire_jax
plan also implies.

## 1. What the plan gets right

The division of the work is sound.  The frame construction, the agent
weights, the consensus update, the task assignment, the logging, and the
application workflow are host-side Python and port as written.  The
dependency inventory in the plan's Section 3 is accurate.  Each of the
twelve "available with no change" items was checked against the current
mbirtorch source, and none disagreed.

The plan correctly identifies the two facts that block a direct
translation of the batched denoiser: the in-place subset update and the
per-volume stopping test.  It also correctly insists on a per-volume
reference (its Option C) as the thing every batched form is checked
against.

The plan's risk table is right to name host memory.  The loop holds twelve
arrays of the full 4D volume in host memory: eight agent states, one
scratch array, the consensus point `z`, the consensus average, and the
previous consensus average.  One more full-size temporary appears during
the change test.  At thirty frames of a 512-cubed volume each array is
16 GB, so the loop needs about 210 GB of host memory.  A gautschi node
provides about 126 GB of host memory per GPU requested, so a production run
at that size needs at least two GPUs for its host memory alone.  This is
the same in mbirjax and is not a port problem, but the demo and the
documentation should state it.

## 2. Where the plan mimics jax rather than torch

### 2.1 The batched denoiser

The plan's Option A stacks the `P` volumes of shape `(T, d1, d2)` into one
image of shape `(P * T, d1, d2)` and runs the existing sweep.  The plan
lists one consequence: the row above the first frame of volume `p + 1` is
the last frame of volume `p`, so the prior couples them.  Two further
consequences are larger and are not in the plan.

The line search is shared.  In `vcd_subset_denoiser` the step size `alpha`
is a ratio of four sums over every pixel in the subset.  In the stacked
image a subset spans every volume, so one `alpha` serves all of them.  In
the jax code each volume has its own `alpha`, because `vmap` gives each
volume its own sums.  The two forms have the same minimizer but different
paths.  The sweep stops after at most fifteen iterations, so the results
differ by the distance each path still has to go.

The partition is different.  The plan's Option A draws a fresh random
partition over the stacked image, so the subsets no longer contain the same
pixels in every volume, and the subset count changes from
`min(16, T * d1 // 64)` to sixteen for any realistic size.  This is a third
change of path.

The plan's repair for the coupling is a row of zeros between volumes,
excluded from the partition.  This does not restore the reflected boundary
condition.  With no separator, an edge voxel's out-of-range neighbor clamps
to the voxel itself, so the neighbor difference is zero and the prior
exerts no pull.  With a zero separator, the neighbor difference equals the
voxel value, so the prior pulls every frame-edge voxel toward zero.  The
repair replaces a coupling between neighboring volumes with a bias toward
zero at both ends of every volume.

The plan's Option B, a functional rewrite plus `torch.func.vmap`, has the
cost the plan names and one more.  An out-of-place `index_add` copies the
whole batch tensor once per subset, so a sixteen-subset sweep moves sixteen
full copies per iteration where the in-place form moves one sixteenth of
one.  In torch, `vmap` is an added layer.  A leading batch dimension is the
native form, and Section 3.1 recommends it.

### 2.2 Thread-local caches, memory probing, and the out-of-memory retry

Three mechanisms in `mace4d.py` exist because of jax and should not be
ported.  The thread-local denoiser cache exists because a jax model that is
not pinned to one device spreads itself over every visible GPU, and four
such models running at once deadlock in NCCL.  A `QGGMRFDenoiser` pinned
with `configure_devices(devices=[dev])` and used from one thread has no such
problem, so one denoiser per orientation, created in the main thread and
handed to its device's worker, is enough.

The batch size is chosen by reading the device's free memory and catching
a `RESOURCE_EXHAUSTED` error.  mbirtorch replaced this style with the
memory ledger, which predicts a plan's peak before allocating (the
comparison document, item 4).  The batched sweep is the existing denoise
plan at image shape `(B * T, d1, d2)`, so the ledger can price a candidate
`B` directly, and the batch size follows from the largest `B` whose plan
fits.  No retry loop is needed.

The last partial batch is padded to the fixed batch size in jax so that
one compiled program serves every block.  In torch the compiled sweep
specializes once per distinct shape, so a smaller final batch costs one
more compile per orientation, which the recompile budget covers.  Padding
is still the simpler choice, and it costs the arithmetic of the padded
volumes.  Either is acceptable.

### 2.3 The data-fit agent

`_run_prox_task` calls `prox_map` with `do_initialization=True` and
`first_iteration=0` on every MACE iteration.  Each call therefore
regenerates the pixel partitions from the global random generator, recomputes
the regularization statistics from the same sinogram, and restarts the
partition sequence at its coarse entries.  The same sinogram gives the same
statistics every time, so this is wasted work.  The coarse-to-fine schedule
restarting every iteration also means the agent is never a fixed operator.

The nn_priors MACE work settled the better convention, recorded in
mbirtorch's `experiments/drunet/agents.py`: initialize once per frame model, pass the
cumulative iteration count as `first_iteration` so the partition sequence
walks coarse to fine once and then stays fine, and warm start each call
from the agent's own previous output.  `prox_map` gained the
`first_iteration` handling this needs on this branch (commit 2be05a7).
The port should adopt it.  The cost is that the mbirtorch trajectory will
not match the mbirjax trajectory, which Section 7 raises as a decision.

### 2.4 The temporal filter

The filter is linear along the frame axis and touches nothing else, so it
equals multiplication by one matrix of size `num_frames` by `num_frames`.
Script m4d3 builds that matrix by filtering the identity and checks it
against the scipy implementation.  The matrix form removes the four scipy
passes per iteration, runs on any device, and needs no chunking.  It also
makes the filter's behavior visible.  The matrix is a projection, and its
rank says how many temporal modes survive.  For the three-frame test case
the matrix is zero, which is why the plan's tests must set `dejitter=False`.

### 2.5 Host arrays and permutations

NumPy runs one elementwise operation on one core.  The consensus update is
about a dozen full-array operations per iteration, all on that one core.
Torch CPU tensors run the same in-place operations on every core.  Script m4d2
measures the difference.  The change is mechanical: the same in-place
sequence with torch calls, or the fused form using `add_` with `alpha`,
which also removes the scratch array.

The orientation permutation `np.ascontiguousarray(np.transpose(x, perm))`
copies the whole 4D array on one core, three times per iteration.  Each
batch needs only a slab of the array: all frames for a block of `z` for
the XY-t orientation, and similarly for the others.  Copying the slab to
the device and permuting there costs the same transfer and no host copy.

### 2.6 The recompile budget

Stage 6 says every worker thread must raise its own recompile budget.
The library already does this.  The wrapper that `maybe_compile` returns
raises the budget on the calling thread at the first sight of each input
shape, before the call that compiles.  The port should verify that no thread exceeds the
budget, as the plan says, but needs no code for it.

### 2.7 The comparison against mbirjax

Stage 7 adds a stored-array comparison against an mbirjax run.  The
retire_jax plan removes the jax reference from the test suite, and the
agent conventions in Section 2.3 change the trajectory anyway.  Three
reference-free checks replace it:

- The batched denoiser must equal the per-volume loop.  With the design of
  Section 3.1 the two are the same arithmetic, so the tolerance is float
  rounding rather than an iterated tolerance.
- The temporal filter matrix must match the scipy filter on random data.
- A whole-loop equality gate, in the style of the nn_priors gate.  With one
  frame the three orientation priors reduce to one standard 3D qGGMRF prior,
  so the consensus must reproduce `recon` on that frame's views.  Section
  3.6 gives the strength relation.

## 3. The recommended design

### 3.1 `QGGMRFDenoiser.denoise_stack`

Add one public method that denoises a stack of same-shaped volumes with
shared parameters.  Its signature follows `denoise`:

```python
denoise_stack(stack, sigma_noise, init_stack=None, max_iterations=15,
              stop_threshold_change_pct=0.2, batch_size=None)
    -> (denoised_stack, info)
```

The method sets the regularization parameters once from the stack merged
to 3D, `stack.reshape(-1, d1, d2)`, through the same row-subsampled path
`denoise` uses.  The mbirjax workaround that avoids
`auto_set_regularization_params` is not needed.  The mbirtorch path
subsamples the rows of whatever image it is given, so the merged stack
gives statistics over every volume.  The floor on `sigma_x` is kept.

Two batched functions carry the arithmetic.  `qggmrf.py` gains a batched
gradient and Hessian for a flat tensor of shape `(B, num_pixels, S)`, with
gathers along the pixel axis and the cylinder differences along the last
axis.  `denoising.py` gains a batched subset update whose four line-search
sums reduce over the pixel and slice axes, so `alpha` has shape `(B,)`.  A
boolean mask of active volumes zeroes the step of a converged volume.  The
sweep loop keeps a per-volume `nmae`, marks a volume inactive when its
`nmae` falls below the threshold, and stops when no volume is active or at
`max_iterations`.  Both functions go through `maybe_compile`.  The mask is
a tensor argument, so changing it triggers no recompile.

The existing `vcd_subset_denoiser` and the single-image path of `denoise`
stay as they are.  The plan's Stage 0, which extracts the single-device
sweep into a method, is no longer needed, because MACE4D calls
`denoise_stack` rather than a sweep function.

The batch size comes from the memory ledger when the device reports its
memory and from a fixed value otherwise, as Section 2.2 describes.  The
stack is processed in batches, each batch placed on the device, swept, and
returned to the host.

### 3.2 The data-fit agents

Each frame keeps one `TomographyModel` pinned to one device, with its
sinogram and weights placed on that device once.  The agent calls
`prox_map` with four settings: `do_initialization` true on the first call
only, the cumulative iteration count as `first_iteration`, `max_iterations`
equal to that count plus the per-call iteration count, and the agent's own
previous output as `init_recon`.  This is `ForwardProxAgent`
in mbirtorch's `experiments/drunet/agents.py` with a device argument, and the port can
move that class into `mace4d.py` or a small agents module.

### 3.3 The consensus loop

The loop keeps the eight agent states as float32 torch tensors in host
memory and runs the in-place update with torch calls.  The structure of
the mbirjax loop is otherwise kept: tasks read the states, the update runs
after the barrier, and the change statistic decides stopping.

### 3.4 The temporal filter

A function builds the filter matrix once from `num_frames`,
`frames_per_rotation`, the harmonics rule, and the band width, by
filtering the identity with the DCT-I recipe.  The loop applies the matrix
to the assembled prox results by one matrix multiply on the host.  The
prior agents apply it on the device to each batch before denoising, since
a batch holds all frames of its slab.

### 3.5 Devices and threads

The device pool helpers and the one-thread-per-device executors port as
planned.  The compile lock in `projectors.py` serializes the first
iteration's compiles across threads, which is the intended behavior.

### 3.6 Tests

The plan's test table stands, with three substitutions.  The batched
denoiser test compares `denoise_stack` with a loop of single-volume sweeps
at float rounding tolerance.  The filter test compares the matrix with the
scipy filter.  The stored-array comparison becomes the one-frame equality
gate.  With `num_frames=1` and `dejitter=False`, MACE4D's consensus must
match `recon` on the first frame's views.  Take agent weights
`[1/2, 1/6, 1/6, 1/6]`, denoiser sigma `s`, and prox sigma `sigma_prox`.
The equilibrium then minimizes the data term plus
`(2 s^2 / (3 sigma_prox^2))` times the standard prior.  Setting `s` to
`sigma_prox` times the square root of `3/2` therefore gives equality.  The prior's
`sigma_x` must be pinned to the value `recon` used.

## 4. An alternative: one 4D prior agent

For proximal-map agents, the consensus equilibrium minimizes the weighted
sum of the agents' objectives, each weighted by its agent weight times its
sigma squared.  Each orientation prior is a 3D qGGMRF over its hyperplane
volumes with equal direction weights of one sixth.  Their sum is a 4D
qGGMRF prior.  Every frame-axis neighbor pair appears in all three
orientations, and every spatial-axis pair appears in two.  After
normalization, the equivalent 4D prior has direction weight one sixth along the frame axis and
one ninth along each spatial axis.  With agent weights `[1/2, 1/2]` and the
same denoiser sigma, a single 4D agent has the same equilibrium as the three
orientation agents with weights `[1/2, 1/6, 1/6, 1/6]`.

The single agent would need one new gradient and Hessian function with six
in-plane neighbor offsets instead of four, on a flat image of shape
`(T * nx * ny, nz)`.  Everything else exists: the sweep, the partition
generator, the single-device path, and the slice-sharded multi-device path
with its halo exchange along `z`.  There would be no batching, no
permutations, no per-volume stopping, and the filter would run twice per
iteration instead of four times.  The prior arithmetic per iteration is
about half that of the three orientation sweeps.

The cost is a change of method.  Multi-slice fusion exists to build a 4D
prior from lower-dimensional denoisers, and the neural-network prior program
on this branch needs exactly the hyperplane-batch form.  The inexact
proximal maps also mean the two forms reach their common equilibrium along
different paths.  The number of MACE iterations to a fixed change may
therefore differ.  The recommendation is to build the hyperplane form
first.  One comparison on a small problem can follow, because the 4D agent
is a small addition once `denoise_stack` and the loop exist.

A gradient-based 4D solver using the differentiable projectors was also
considered and set aside.  It would replace VCD, whose convergence rate is
the reason the library is fast, and it would not reuse the memory ledger.

## 5. Measurements

Three scripts in `plans/mace4d/experiments/` measured the claims of
Sections 2 and 3 on the Mac (Apple M3 Max, torch 2.13.0, `cpu` and `mps`, no
CUDA).  Each script has a companion `.md` with the full run record, and every
number below is copied from those records.  The Mac numbers show which forms
are equivalent and which are not.  They do not predict H100 speed, which is
decision 7 in Section 7.

### 5.1 The batched denoiser (m4d1)

The correctness case denoised 8 volumes of shape `(6, 24, 24)` with shared
constants and 2 subsets.  The reference is the production sweep run once per
volume.  The difference is `max|y - y_ref| / max|y_ref|`.

| Form | cpu | mps |
|---|---|---|
| Option D, leading batch dimension | 0 | 1.07e-07 |
| Option A1, stacked rows, repeated partition | 2.67e-02 | 2.67e-02 |
| Option A1, interior frames only | 6.24e-03 | 6.24e-03 |
| Option A1, edge frames only | 2.67e-02 | 2.67e-02 |
| Option A2, stacked rows, fresh partition | 2.67e-02 | 2.67e-02 |
| Option A3, stacked rows, zero separators | 2.79e-02 | 2.79e-02 |

Three exactness checks passed on both devices.  The batched gradient and
Hessian equal the package values bit for bit.  The per-volume iteration
counts equal the reference's, volume by volume.  A batch holding one volume
equals the reference bit for bit when both run without compilation.  With
compilation on `mps` that one-volume batch differs from the reference by
5.95e-08, so the compiler orders the reduction differently there.

These results say that Option D is the same arithmetic as the per-volume
loop.  They also say that Option A is not close.  Every variant misses the
plan's tolerance of 1e-3 by a factor of about 27.  On interior frames the
miss is a factor of about 6.  Those frames have no neighbor in another
volume, so this part comes from the shared step size and the shared
stopping test.  On edge frames the miss is four times larger, which adds
the coupling between neighboring volumes.  The zero separator of A3 did not
reduce the difference.  It increased it.

The timing case denoised 64 volumes of shape `(8, 64, 64)` with 8 subsets,
after one warm-up run.

| Form | cpu seconds | mps seconds |
|---|---|---|
| Reference loop | 0.719 | 0.716 |
| Option D | 0.141 | 0.032 |
| Option A1 | 0.119 | 0.342 |

The volumes in Option D needed 4 to 6 iterations, 5.25 on average, so 12.5
percent of the volume iterations it executed were on frozen volumes.  The
reference loop is bound by its call count: about 2690 subset calls and 336
host reads of the image norm.  On `mps` Option D is 22 times faster than the
loop and 10 times faster than Option A1.  The cause of A1's slow `mps` time
was not investigated, because A1 is excluded on correctness.

### 5.2 The host consensus update (m4d2)

The update ran on arrays of shape `(8, 192, 192, 192)`, 0.226 GB each, with
10 threads.

| Implementation | seconds per update | max difference from numpy |
|---|---|---|
| numpy in-place, as in mbirjax | 0.272 | reference |
| torch in-place with one scratch tensor | 0.157 | 0 |
| torch in-place with no scratch tensor | 0.108 | 2.9e-06 on `W`, 4.8e-07 on `xbar` |

The change statistic differed from numpy's by 1.08e-05 relative.  These
results say that the torch form with a scratch tensor reproduces the numpy
values exactly at 1.7 times the speed.  The form without a scratch tensor is
2.5 times faster than numpy and allocates no full-size array per update,
where the numpy form allocates two.

### 5.3 The filter matrix (m4d3)

The matrix was built by filtering unit impulses along the frame axis with a
copy of the mbirjax filter, at period 6 with harmonics and band width 1.

| Frames | max difference, matrix against filter | rank at float32 tolerance |
|---|---|---|
| 3 | 0 | 0 |
| 12 | 7.2e-07 | 4 |
| 30 | 1.2e-06 | 22 |

The inputs had largest values between 4.0 and 4.7.  The matrix is symmetric
to 6.0e-08 and idempotent to 1.5e-07, so it is an orthogonal projection.
The rank is the frame count minus the number of DCT-I modes the filter
removes.  At period 6 the filter removes eight modes whenever the frame
count allows it, and at three frames it removes every mode and returns
zeros.  The documentation should state this fact.  With the default period, a run
needs well over eight frames for the filter to keep any temporal content.

The timing used an array of shape `(30, 256, 256, 64)`, 0.5 GB.

| Form | seconds |
|---|---|
| scipy filter, as in mbirjax | 1.526 |
| scipy filter with `workers=-1` | 0.210 |
| torch matrix multiply on the CPU | 0.044 |
| torch matrix multiply on `mps`, transfer excluded | 0.025 |

The scipy result and the CPU matrix result differ by at most 2.1e-06.  These
results say that the matrix form is exact to float32 rounding and 35 times
faster than the filter as written.

## 6. Revised stages

The stages keep the plan's numbering where the content is unchanged.  The
version 2 plan beside this page restates them in their final form.

| Stage | Content | Change from the plan |
|---|---|---|
| 0 | Batched gradient and Hessian in `qggmrf.py`, batched subset update and `denoise_stack` in `denoising.py`, with the equality test against the per-volume loop | Replaces the extraction of `_denoise_single_device` |
| 1 | Done: the measurements in Section 5 | Replaces the option measurement |
| 2 | Frame construction and device-pool helpers | Unchanged |
| 2b | The shared MACE module of Section 8.3: agent protocol, loop with the folding update, `ForwardProxAgent`, hyperplane-stack agent, filter matrix builder, with the nn_priors gate as a test | New |
| 3 | `MACE4DModel` as a composition of the frame models, the agents, and the shared loop on host tensors | Built on Stage 2b |
| 4 | The three prior agents as hyperplane-stack agents over `denoise_stack`, with the filter applied per batch on the device and one task per batch | Public method instead of internals; no thread-local cache; ledger batch size |
| 5 | The filter matrix and its test | Matrix instead of scipy passes |
| 6 | Fixed prox tasks, queued denoise batches (Section 8.1), and logging | Hybrid assignment; the budget item is dropped |
| 7 | Tests with the three substitutions of Section 3.6 | No mbirjax reference |
| 8 | Documentation and registration | Unchanged |
| 9 | `save_volume_as_gif` and the demo | Unchanged |

## 7. Decisions

Greg ruled on these at the review on 2026-09-13.

1. Method, not trajectory.  DECIDED: the trajectory may change; the fixed
   point is what matters.
2. `denoise_stack` as a public method of `QGGMRFDenoiser`.  DECIDED: yes.
3. Per-volume stopping by freezing converged volumes.  A frozen volume is
   one whose own stopping test has passed.  Its step size is set to zero
   from then on, so its values no longer change while the other volumes in
   the batch keep iterating.  This is what jax's vmapped `while_loop` does.
   The alternatives are to shrink the batch as volumes converge, which
   changes the tensor shapes and costs a recompile per new size, or to use
   one shared stopping test for the whole batch, which costs the same
   arithmetic as freezing and gives a result that is not exactly the
   per-volume one.  DECIDED: freeze.
4. The filter as a matrix.  DECIDED: yes, provided the results match,
   including the behavior at the two ends of the frame axis.  The matrix
   is built by applying the scipy filter to a unit impulse at every frame,
   so it reproduces the filter's end behavior by construction.  The test
   will check impulses at the first and last frames explicitly, in
   addition to random inputs.
5. The one-frame equality gate.  DECIDED: yes.
6. The 4D-agent comparison.  DECIDED: yes, as a follow-up.  Greg added
   that the agents may later include bilateral-filter and total-variation
   priors and denoisers, in 3D and in 4D.  Section 8.3 carries this into
   the shared agent design.
7. The first cluster measurement.  DECIDED, as recommended: after Stage 0
   lands, one job on one H100 with synthetic data at two sizes, twelve frames of a
   256-cubed volume and twenty-four frames of a 512-cubed volume.  The job
   sweeps the batch size of `denoise_stack` over 4, 16, 64, and the largest
   size that fits, and records the time per volume, the peak device memory,
   and the per-volume loop time at the same sizes.  It also times one
   `prox_map` per frame at each size.  The batch-size knee sets the default and calibrates the ledger's
   batch sizing.  The cost is about one GPU-hour.  With the per-batch tasks
   of Section 8.1 the mbirjax cost constant is not needed.

## 8. Discussion items from the review

### 8.1 Fixed versus dynamic task assignment

The mbirjax code assigns tasks to devices once per run, by a least-loaded
rule over estimated costs, and each device runs its tasks on one thread.
The assignment has three benefits.  A frame stays on one device, so its
sinogram and weights are uploaded once and its model keeps its cached prox
initialization; re-pinning a model to another device drops that cache and
rebuilds its projectors.  The run is reproducible, because the same device
runs the same task every iteration.  And there is no scheduler to reason
about.

The cost of the fixed assignment is its dependence on the cost estimate.
The three denoise tasks are large, and their cost relative to a prox task
is a constant measured on one GPU at one size.  When the constant is wrong
for the GPU or the size at hand, one device finishes late and the
iteration waits for it.

A dynamic assignment, where an idle device pulls the next task from a
shared queue, balances itself without an estimate.  For the prox tasks it
is not attractive, because a moving frame pays the re-pinning cost above.
For the denoise work it is natural.  `denoise_stack` already processes an
orientation in batches, each batch carries its own data to the device,
and the denoiser constants are small.  So the recommendation is a hybrid:
the prox tasks stay fixed to their devices, and each orientation's denoise
becomes one task per batch, pulled from a shared queue by whichever device
worker is idle.  The balance is then at the granularity of one batch.
Each device holds one configured denoiser per orientation, and compiles
all three orientation shapes on the first iteration rather than one, which
is a one-time cost.  A fallback, if the queue is unwanted, is to re-balance
the fixed assignment after the first iteration from the measured task
times that the task log already records.

DECIDED at review: the hybrid assignment.

### 8.2 Host memory

The mbirjax loop holds twelve full-size arrays and one full-size temporary
at its peak: four `W`, four `X`, the scratch array, the consensus point
`z`, the consensus average `xbar`, its previous value, and the temporary
that the change test forms.  A folding update reduces this to eight.  The
two forms below compute the same `W` and `xbar`.  The step numbers match
across them, so the forms can be read line against line.

```
# Form A, the mbirjax update.  Every named array has the full 4D size.

X[k] = agent_k(W[k])  for k in 0..3            # 1. four outputs, all kept until step 3
z = zeros()                                     # 2. the consensus point
for k in 0..3:
    scratch = 2 * X[k] - W[k]                   #    one scratch array, reused
    z += beta[k] * scratch
for k in 0..3:                                  # 3. the Mann step
    W[k] += 2 * rho * (z - X[k])                #    through the scratch array
xbar_prev = xbar                                # 4. the consensus average
xbar = sum_k beta[k] * X[k]                     #    a new array
change = norm(xbar - xbar_prev) / norm(xbar_prev)     # 5. one full-size temporary

# Peak: W x4, X x4, scratch, z, xbar, xbar_prev, and the temporary of step 5.
```

```
# Form B, the folding update.  Full-size arrays: W x4, z, xbar_new, xbar_prev.
# Each agent's output arrives one region at a time (one frame, or one slab of
# hyperplanes), as a tensor of that region's size only.

z.zero_(); xbar_new.zero_()
for each completed task of agent k, with region r and output x:    # 1. per region
    with the lock:
        z[r] += beta[k] * (2 * x - W[k][r])     # 2. W[k][r] still holds the agent's input
        xbar_new[r] += beta[k] * x              # 4. the consensus average, accumulated
        W[k][r] -= 2 * rho * x                  # 3a. the part of the Mann step that needs x
    # x is released here
for k in 0..3:                                  # 3b. after every task has arrived
    W[k] += 2 * rho * z                         #     W[k] now equals W_old + 2 rho (z - X[k])
change = chunked_norm(xbar_new - xbar_prev) / chunked_norm(xbar_prev)   # 5. no full temporary
xbar_new, xbar_prev = xbar_prev, xbar_new       # swap the two buffers

# Peak: W x4, z, xbar_new, xbar_prev, plus one region-sized tensor per task in flight.
# With the temporal filter on, the data-fit agent also keeps the stack of prox
# outputs until every frame has arrived, because the filter acts along the frame
# axis: one more full-size array, eight in total.
```

Step 3 is the same arithmetic in both forms, because `W - 2 rho X + 2 rho z`
equals `W + 2 rho (z - X)`.  Steps 2 and 4 form the same sums in a
different order, so the two forms agree to float32 rounding, which decision
1 allows.  Form B needs the lock because the regions of different agents
overlap: one frame of the data-fit agent and one slab of hyperplanes share
voxels.  Each fold is a region-sized in-place operation, so the lock is held
briefly.

Three further levers exist.  Memory-mapped state arrays give capacity at
a large speed cost and are a fallback only.  On gautschi, host memory
grows with the number of GPUs requested, at about 126 GB per GPU.
Half-precision state is not recommended without a study: the stopping
threshold is 0.2 percent and float16 carries about three significant
digits.

DECIDED at review: the folding update, with this section as its record.

### 8.3 Shared code with the nn_priors work

The loop and the agents in mbirtorch's `experiments/drunet/` are the same objects
MACE4D needs, so they should move into a package module, for example
`mbirtorch/mace.py`, and MACE4D should be built on it.  The module holds
five things.  The agent protocol: a callable from an array to an array of
the same shape, bound to one device.  `ForwardProxAgent`, with a device
argument and the sinogram and weights placed once.  A hyperplane-stack
agent that permutes a 4D array, applies the filter matrix, calls a stack
denoiser in batches on its device, and permutes back; the stack denoiser
is a parameter, so qGGMRF, total-variation, bilateral, and network
denoisers plug in.  A whole-volume agent form for 3D and 4D denoisers,
which is where a 4D qGGMRF, TV, or bilateral prior would go.  And one
MACE loop that takes the states wherever they live: device tensors for the
nn_priors case, host tensors with per-device workers for MACE4D, with the
folding update of Section 8.2 in both cases.  The filter matrix builder
goes beside them.  `MACE4DModel` is then a composition of the frame
models, the agents, the loop, the logging, and the initialization cache,
and the nn_priors scripts import from the package.  This adds one stage
before the module is written.

DECIDED at review: agreed.

A later side discussion settled the loop's form.  The loop is one small
class, `MACE`, with `step`, `run`, and a checkpoint pair.  The agents are
independent classes on the protocol, and there is no hierarchy of MACE
forms.  Section 2.2 of the version 2 plan gives it.

### 8.4 A batch dimension in the reconstruction loop

The question is whether a leading axis over independent reconstructions,
sharing a recon shape but each with its own sinogram and view geometry,
would make `_vcd_recon` more efficient, as it does for the denoiser.
MACE4D's frames are such a set.  The projector bodies and the Triton
kernels take one problem's voxel cylinders and view geometry per call, so
a problem axis would be new through the bodies, the kernels, the error
sinogram, the Hessian, the line search, and the memory ledger, which would
price every resident array once per problem.  The step size would become
one value per problem, as in the batched denoiser.  Subsets cannot be
batched, because each subset update reads the error sinogram the previous
one wrote.

The batch pays where one problem underfills the GPU: small frames,
downsampled runs, demo sizes, and parameter sweeps.  It pays nothing where
one problem already saturates a device, and multi-GPU sharding already
supplies the capacity dimension.  The decisive measurement would be the GPU utilization during one
frame's `prox_map` at the sizes MACE4D runs.

DECIDED at review: dropped.  It is complicated, and it has little or no
benefit at the sizes that matter.
