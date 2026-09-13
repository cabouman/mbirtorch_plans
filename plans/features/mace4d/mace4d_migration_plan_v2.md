# MACE4D in mbirtorch: migration plan, version 2

This plan replaces the plan of 2026-09-11, which is kept beside it as
`mace4d_migration_plan.md`.  The first plan describes what the mbirjax
code computes and which mbirtorch pieces it can use.  Those parts are still
correct, and this plan refers to them rather than repeating them.  What
changed is the design of the port.  The reasons and the measurements behind
the changes are in `mace4d_plan_evaluation.md`, also beside this plan, and
the decisions were taken at the review of 2026-09-13.  This plan was written
the same day.

The mbirjax sources are `mbirjax/mbirjax/mace4d.py` (1040 lines),
`mbirjax/tests/test_mace4d.py` (305 lines), and
`mbirjax/docs/source/usr_mace4d.rst` (61 lines).  The MACE loop and agents
already written for mbirtorch are `mbirtorch/experiments/drunet/mace.py` and
`agents.py`, with their records in `mbirtorch_plans/plans/nn_priors/`.

## Summary

The port is built from a shared MACE module rather than as a translation of
`mace4d.py`.  Three pieces of code carry it.

`QGGMRFDenoiser.denoise_stack` denoises a stack of same-shaped volumes with
shared parameters.  It is the batched sweep with an explicit leading batch
dimension, per-volume step sizes, and a per-volume stopping test that
freezes a converged volume.  It reproduces per-volume denoising exactly, and
it is the mbirtorch form of what jax's `vmap` computed.

`mbirtorch/mace.py` holds the MACE loop, the agent protocol, and the
agents: the data-fit proximal agent, the single-volume qGGMRF agent, and the
hyperplane-stack agent that permutes a 4D array, filters it along the frame
axis, denoises the stack in batches, and permutes back.  The loop folds each
agent's output into the consensus as it arrives, so it holds eight full-size
arrays instead of twelve, and it schedules the data-fit tasks on fixed
devices and the denoise batches through a shared queue.  The same loop and
agents serve the neural-network prior work, which moves its loop into the
package.

`mbirtorch/mace4d.py` holds `MACE4DModel`, composed from the per-frame
models, one data-fit agent over the frames, three hyperplane agents, the
initialization cache, and the log files.  Its public interface is the one
mbirjax has.

The values along the iteration differ from mbirjax, because the data-fit
agent follows the fixed-operator conventions of the nn_priors work and the
consensus update is folded.  The fixed point is what the tests check, and
every test is reference-free: the batched denoiser against a loop of single
volumes, the filter matrix against the scipy filter, and the whole loop
against `recon` through two equality gates.

The work is seven stages plus one cluster measurement, in the order of
Section 4.  The estimated size is about 2,000 lines including tests, of
which about 600 form the shared module that the nn_priors program also uses.

## Status by stage

| Stage | Delivers | Status |
|---|---|---|
| 0 | `denoise_stack`, the two batched functions, `auto_batch_size`, and their tests | Not started |
| 1 | The three measurement scripts and their records | Done |
| 2 | `construct_time_frame_models` and the device helpers, with tests | Not started |
| 3 | `mbirtorch/mace.py` with its tests, and the drunet scripts moved onto it | Not started |
| 4 | `MACE4DModel` in `mbirtorch/mace4d.py` | Not started |
| 5 | `tests/test_mace4d.py`, including the one-frame equality gate | Not started |
| 6 | The documentation pages and the lazy export | Not started |
| 7 | `save_volume_as_gif` and the demo, run on the phantom dataset | Not started |
| 8 | The H100 measurement and its record | Not started |

## Rule for the code

Nothing from this plan goes into the code.  Docstrings, comments, and
Sphinx pages describe what the code does, in plain words.  They do not
name this plan or its stages, they do not use its vocabulary as labels,
and they carry no history: no dates, no names, and no reference to
mbirjax.  A comment explains a rule the code follows only when a later
change would otherwise break it, and then it states the rule itself.

## 1. What MACE4D computes

MACE4D reconstructs a time sequence of volumes from one continuous scan of
a moving object.  The scan's views are divided into overlapping angular
windows, one per time frame.  `frames_per_rotation` (default 6) sets the
angular step between windows, and `frame_overlap_factor` (default 2.0) sets
each window's span in units of that step, so by default a window spans 120
degrees and each view belongs to two frames.  Trailing views that cannot
fill a window are discarded.  Each frame has its own `TomographyModel`,
built with `copy_ct_model` on the window's angles.  Section 1.1 of the
first plan gives the arithmetic and the 24-view test case, which yields 5
frames with `view_slices[1] == slice(4, 12)`.

The reconstruction is the consensus of four agents on the 4D array of
shape `(num_frames, nx, ny, nz)`.  Agent 0 is the data fit: one proximal
map per frame.  Agents 1 to 3 are qGGMRF denoisers on the three sets of
hyperplane volumes that hold the frame axis and two spatial axes.  A
hyperplane volume is the 3D array obtained by fixing one spatial
coordinate, so it holds every frame.  The XY-t set has `nz` volumes of
shape `(T, nx, ny)`, the YZ-t set has `nx` volumes of shape `(T, ny, nz)`,
and the XZ-t set has `ny` volumes of shape `(T, nx, nz)`.  The agent
weights are `[1 - w, w/3, w/3, w/3]` for a scalar `mace_prior_weight` `w`
(default 0.5), or `[1 - (w1 + w2 + w3), w1, w2, w3]` for a list.

Each iteration evaluates the four agents on their inputs `W_k`, forms the
consensus point `z = sum_k beta_k (2 X_k - W_k)`, moves each input by
`W_k += 2 rho (z - X_k)`, and reports the consensus average
`xbar = sum_k beta_k X_k`.  The iteration stops when the relative change
of `xbar` falls below `stop_threshold_change_pct` (default 0.2) or after
`max_iterations` (default 10).  The initial 4D image comes from the caller,
from a cache file in `init_dir`, or from a per-frame reconstruction with 15
iterations.  One denoiser noise level is estimated from the initial image
and shared by the three prior agents.

The overlapping windows imprint a periodic modulation along the frame axis
with period `frames_per_rotation`.  A filter that zeroes the DCT-I modes at
that period and its harmonics removes it.  The filter is applied to the
stack of prox outputs and to each prior agent's input.  When `log_dir` is
given, three files record the run settings, the per-iteration timing, and
the per-task timing.  Sections 1.2 to 1.9 of the first plan give the
details of each of these, with line references into the mbirjax source.

## 2. The design

### 2.1 `QGGMRFDenoiser.denoise_stack`

```python
denoise_stack(stack, sigma_noise=None, init_stack=None, max_iterations=15,
              stop_threshold_change_pct=0.2, batch_size=None)
    -> (denoised_stack, info)
```

`stack` is an array of shape `(P, d0, d1, d2)`, where `(d0, d1, d2)` is
the denoiser's image shape.  A numpy input returns numpy.  A tensor input
returns a tensor on the input's device, while the sweep runs on the
denoiser's device.  `init_stack` is an optional initial image per volume
and defaults to the input.  `sigma_noise` follows `denoise`: `None`
estimates it from the stack merged to 3D, `stack.reshape(-1, d1, d2)`, and
`sigma_y` is kept equal to it.  When `auto_regularize_flag` is set, the
regularization parameters are set once from the merged stack through the
same row-subsampled path `denoise` uses.  That path subsamples the rows of
whatever image it is given, so the merged stack gives statistics over every
volume.  The mbirjax workaround is therefore not needed.

One pixel partition is drawn from `granularity[partition_sequence[0]]`
over `(d0, d1)`, placed on the device, and used by every volume.  It comes
from the global numpy generator, so a seeded call is reproducible.

The sweep runs on a flat tensor of shape `(B, d0 * d1, d2)`.  Two batched
functions carry the arithmetic.
`qggmrf.qggmrf_gradient_and_hessian_batched(flat, recon_shape,
pixel_indices, qggmrf_params)` repeats the formulas of
`qggmrf_gradient_and_hessian_at_indices` with every gather along the pixel
axis and the cylinder differences along the last axis, with no halos.
`denoising.vcd_subset_denoiser_batched(flat_image, flat_error_image,
pixel_indices, fm_constant, qggmrf_params, image_shape, active)` repeats
`vcd_subset_denoiser` with the four line-search sums reduced over the pixel
and slice axes, so the step size `alpha` has shape `(B,)`.  The boolean
tensor `active` zeroes the step of an inactive volume.  Both functions go
through `maybe_compile`.  `active` is a tensor argument, so changing it
triggers no recompile.  The prototype of both functions is in
`plans/experiments/features/mace4d/m4d1_batched_denoiser_options.py`,
where they equal the package values bit for bit.

The stopping test is per volume.  After each iteration the loop computes
each volume's `nmae` as its accumulated step ell-1 over its own ell-1,
using a chunked sum as `image_ell1` does.  A volume whose `nmae` falls
below the threshold becomes inactive, its step is zero from then on, and
its values no longer change.  The loop stops when no volume is active or at
`max_iterations`.  This gives the same values and the same per-volume
iteration counts as running the volumes one at a time.

The stack is processed in batches of `batch_size` volumes.  The last batch
is padded to the full size by repeating its last volume, so one compiled
shape serves every batch of a call.  The padded results are discarded.
`batch_size=None` chooses the size.  On CUDA it is the largest batch
whose denoise plan fits the device's free memory.  The plan is priced by
the memory ledger for one volume's image shape, scaled by the batch count,
with the ledger's usual headroom.  On CPU and MPS it is the whole stack.
The rule is exposed as `QGGMRFDenoiser.auto_batch_size(volume_shape)` so
that the hyperplane agent can use it.  No out-of-memory retry exists,
because the library predicts rather than catches.

`denoise_stack` runs on the denoiser's lead device and does not use a
multi-device layout.  A denoiser configured with more than one device
raises `ValueError` from this method.  `info` holds the per-volume
iteration counts, the per-volume `nmae` history in percent, the
regularization parameters, and the batch size.  No log file is written.  At
`verbose >= 1` one line per call reports the volume count, the batch size,
and the range of iteration counts.

`vcd_subset_denoiser`, the single-image path of `denoise`, and their
golden tests stay as they are.

### 2.2 The shared module: `mbirtorch/mace.py`

The module holds the agent protocol, the loop, three agents, and the
filter matrix builder.

An agent is a callable from a tensor to a tensor of the same shape, bound
to one device.  It moves its input to its device, computes, and returns
its output on the input's device.  Two conventions, carried from the
nn_priors work, make the equilibrium well defined: an agent may follow a
schedule early but must be a fixed operator in the tail, and an agent that
runs an inner solve warm-starts it from its own previous output.  For the
loop's scheduling, an agent may also provide `tasks(w)`, a list of tasks
that together produce its output.  A task has three parts.  Its `device` is a fixed device, or `None` for
any device in the pool.  Its `region` is a tuple of slices into the array,
or `None` for the whole array.  Its `run(device)` method returns the
output for its region on the input's device.  An agent that
provides no `tasks` is wrapped as one task on its own device over the whole
array.  An agent may declare `fold_after_all = True`.  Its task outputs are
then kept by the agent until all its tasks are done, after which
`pieces()` yields `(region, output)` pairs.  The data-fit agent uses this
when the temporal filter is on, because the filter acts along the frame
axis.  Every other output is folded as its task completes.

The loop is

```python
mace(agents, x0, mu=None, rho=0.5, max_iterations=30,
     stop_threshold_change_pct=0.0, devices=None, callback=None)
    -> (x_bar, info)
```

The states `W_k` are tensors on the device of `x0`: the host for MACE4D,
a device for the nn_priors case.  `devices=None` runs every task inline,
in order.  A list of devices creates one worker thread per device.  Each
worker first runs the tasks fixed to its device, in order, then pulls tasks
with `device=None` from a shared queue until the queue is empty.  This is
the hybrid assignment: the data-fit tasks stay on their devices, and the
denoise batches balance themselves at batch granularity.

The consensus update is folded.  The loop keeps `z`, `xbar_new`, and
`xbar_prev` beside the `W_k`.  When a task of agent `k` completes with
region `r` and output `x`, the loop takes one lock and performs three
region-sized in-place operations: `z[r] += mu_k (2 x - W_k[r])`,
`xbar_new[r] += mu_k x`, and `W_k[r] -= 2 rho x`.  It also accumulates
`sum((x - xbar_prev[r])^2)` for agent `k`, which gives a per-agent
consensus spread against the previous average at no extra memory.  The
output is then released.  After every task of the iteration has arrived,
one pass sets `W_k += 2 rho z` for each agent, which completes the Mann
step, and the change statistic `norm(xbar_new - xbar_prev) /
norm(xbar_prev)` is computed in chunks.  The two average buffers are then
swapped and `z` and the new buffer are zeroed.  Section 8.2 of the
evaluation shows this form beside the mbirjax form and gives the algebra.

`info` holds, per iteration, the change in percent, the spread per agent,
the wall time, and one row per task with the agent index, the task index,
the device index, and the start and end times relative to the iteration
start.  `callback(iteration, x_bar)` is called after each iteration when
given.  The loop stops when the change falls below the threshold or at
`max_iterations`.

### 2.3 The agents

`ForwardProxAgent(model, sinogram, weights=None, sigma_prox=None,
inner_iterations=3, init_recon=None, device=None)` is the class in
`experiments/drunet/agents.py` with a device argument.  When `device` is
given, the agent pins the model to it with `configure_devices` and places
the sinogram and weights on it once through `prepare_sino_for_devices`.
Each call runs `prox_map` with `do_initialization` true on the first call
only, the cumulative iteration count as `first_iteration`,
`max_iterations` equal to that count plus `inner_iterations`, a stop
threshold of zero, and the agent's previous output as `init_recon`.  The
partition sequence therefore walks coarse to fine once and then stays on
its last entry.

`QGGMRFDenoiserAgent(image_shape, sigma_noise, pinned_params=None,
inner_iterations=8, like_model=None, use_ror_mask=False, device=None)` is
the single-volume agent of the same file, with the device rule above.

`HyperplaneAgent(axis, make_stack_denoiser, batch_size=None,
filter_matrix=None)` is new.  `axis` is the spatial axis that is fixed to
cut the hyperplanes: 3 for XY-t, 1 for YZ-t, 2 for XZ-t, in the
`(t, x, y, z)` order of the 4D array.  `make_stack_denoiser(device)` returns
a callable from a stack tensor on that device to its denoised stack.  The
agent calls it once per device in the pool, from the worker that first
needs it, and keeps the result.  `tasks(w)` returns one task per batch,
with `device=None` and `region` the slab of hyperplanes along `axis`.  A
task copies its slab to the device and permutes it so that the hyperplane
index comes first and the frame index second.  It applies the filter
matrix along the frame axis when one is given, calls the stack denoiser,
permutes back, and returns the slab on the input's device.  `batch_size=None` asks the
stack denoiser for its automatic size on the first device that runs one.
The agent is generic in its denoiser, so a total-variation, bilateral, or
network stack denoiser plugs in the same way as the qGGMRF one.

The data-fit agent of MACE4D is private to `mace4d.py`.  It holds one
`ForwardProxAgent` per frame, each pinned to the frame's device, and its
`tasks(W_0)` returns one task per frame with a fixed device and the frame's
region.  With `dejitter` off, each frame folds as it completes.  With
`dejitter` on, the agent sets `fold_after_all`, writes each frame into its
own stack buffer, and `pieces()` applies the filter matrix to that stack
one slab at a time and yields the slabs.

### 2.4 The temporal filter

`temporal_filter_matrix(num_frames, period, harmonics=True, band_width=1)`
returns the `num_frames` by `num_frames` float32 matrix of the filter.  It
is built by applying the DCT-I recipe of the mbirjax filter to a unit
impulse at every frame, with `scipy.fft.dct` and `idct` of type 1 and
orthonormal scaling.  Script m4d3 showed that the matrix reproduces the
filter to float32 rounding, is a symmetric projection, and has rank equal
to the frame count minus the number of zeroed modes.  With the default
period the filter removes eight modes whenever the frame count allows it,
and with three frames it removes every mode.  The documentation states
this, and the tests set `dejitter=False` on small frame counts.

The matrix is applied by one matrix multiply along the frame axis: by the
data-fit agent on the host, slab by slab, and by each hyperplane task on
the device, where the batch holds every frame of its slab.

### 2.5 Devices, threads, and compilation

`gpu_devices()`, `cpu_devices()`, and `default_devices()` sit next to
`_resolve_device` in `tomography_model.py`.  `gpu_devices()` returns every
CUDA device, or the MPS device when CUDA is absent and MPS is present, or
nothing.  The pool resolution in `mace.py` accepts `None` (every GPU, or one
CPU device), `'cpu'`, `'gpu'`, an integer count, a sequence of integer
indices into the default pool, and a sequence of devices, which passes
through unchanged.  An explicit sequence may repeat a device, such as
`['cpu', 'cpu']`, which gives two workers on one device.  The tests use
this to exercise the threaded path without a second GPU.  The string
`'cpu'` gives one device, where jax gave several virtual ones.  A comment
marks this divergence.

Each frame model is pinned to one device for the run.  Every device worker
compiles the batched sweep for each orientation shape it runs, so a device
compiles up to three shapes on the first iteration.  The compile lock in
`projectors.py` serializes these compiles.  The per-thread recompile
budget is raised by `maybe_compile` on the calling thread, so no code is
needed for it.  Stage 4 checks that no thread exceeds the budget.

### 2.6 Memory

The host holds eight full-size float32 arrays: four `W_k`, `z`, the two
average buffers, and the prox stack when the filter is on.  At thirty
frames of a 512-cubed volume each array is 16 GB, so a run of that size
needs about 130 GB of host memory.  On gautschi, host memory comes at about
126 GB per GPU requested, so such a run requests at least two GPUs.  The
demo and the documentation state the array count and the rule.

On a device, a hyperplane task holds one batch: the input slab, the
permuted copy, the sweep's state, and the subset temporaries of the
denoise plan.  A data-fit task holds one frame's `prox_map`.  The batch
size is what keeps a device within its memory, and the ledger sets it.

## 3. The public interface

### 3.1 `MACE4DModel`

```python
MACE4DModel(ct_model, frames_per_rotation=6, frame_overlap_factor=2.0, num_frames=None)
.set_params(no_warning=False, no_compile=False, **kwargs) -> bool
.recon(sinogram, weights=None, init_recon=None, max_iterations=10,
       stop_threshold_change_pct=0.2, init_dir=None, log_dir=None) -> (recon, recon_dict)
.set_device_pool(devices=None)
.devices  (property)
```

The attributes `model_list`, `view_slices`, `num_frames`, and
`recon_shape` are set in the constructor and read by application code.
`recon` returns a numpy array of shape `(num_frames,) + recon_shape` and a
dictionary with the keys `recon_params`, `timing`, `notes`, and
`model_params`, where `recon_params['iterations completed']` is the
iteration count, under exactly that name.  A `num_frames` below 1 raises.
A sinogram, weights, or `init_recon` of the wrong shape raises before any
computation.

### 3.2 Parameters

| Name | Default | Where it is set | Meaning |
|---|---|---|---|
| `frames_per_rotation` | 6 | Constructor, then fixed | Frames per rotation, and the filter period |
| `frame_overlap_factor` | 2.0 | Constructor, then fixed | Frames that share a view |
| `num_frames` | None, meaning all | Constructor | Use only the first frames |
| `mace_prior_weight` | 0.5 | `set_params`, validated when set | A scalar or a list of three |
| `rho_mann` | 0.5 | `set_params` | The consensus step |
| `prox_num_iterations` | 3 | `set_params` | VCD iterations per data-fit call |
| `prox_stop_threshold` | 0.02 | `set_params` | Stop threshold of the initialization reconstruction |
| `sigma_prox` | None, meaning automatic | `set_params`, registered with `no_warning` | Passed to every `prox_map` |
| `dejitter` | True | `set_params` | Apply the temporal filter |
| `dejitter_verbose` | 0 | `set_params` | Printing for the filter only |
| `verbose` | 1 | `set_params` | 0 silent, 1 progress, 2 debugging |
| `max_iterations` | 10 | `recon` | Consensus iterations |
| `stop_threshold_change_pct` | 0.2 | `recon` | 0 runs every iteration |
| `init_dir` | None | `recon` | Cache directory for `init_recon.npy` |
| `log_dir` | None | `recon` | Directory for the three log files |

Two behaviors of `set_params` are kept from mbirjax.  `sigma_prox` is set
with `no_warning=True`, which suppresses the base class warning about
disabling automatic regularization, because this model runs no
reconstruction of its own.  `mace_prior_weight` is validated when it is set.

Three fixed values are kept from mbirjax and stated in the code: 15
iterations for the initialization reconstruction, 15 iterations and a 0.2
percent threshold for each denoiser sweep, and a floor of 1e-6 on the
estimated `sigma_x`.  The subset count of each hyperplane denoiser is
`max(1, min(granularity[0], num_pixels // 64))` with `num_pixels = T * d1`,
because a subset with fewer than about 64 pixels makes the line search
compute zero over zero on flat regions.  The hyperplane agent sets this as
the denoiser's `granularity`.

### 3.3 Differences a user can see

- The data-fit agent runs exactly `prox_num_iterations` VCD iterations per
  call, with a stop threshold of zero, and walks the partition sequence
  coarse to fine across the whole run instead of restarting it every
  iteration.  `prox_stop_threshold` applies to the initialization
  reconstruction only.
- The consensus update is folded, so the float32 rounding of `W` and
  `xbar` differs from mbirjax's.
- `task_log.csv` gains a `part` field, the batch index of a denoise task,
  blank for a prox task, because a denoise orientation is now several
  tasks.
- `set_device_pool('cpu')` gives one device.
- The filter is a matrix.  Its values match the scipy filter to float32
  rounding.

## 4. Stages

Each stage names its files, its tests, and the condition at which it ends.
A stage does not start until the stages it depends on have ended.  Section
7 gives the order.

### Stage 0: `denoise_stack`

Files: `mbirtorch/qggmrf.py`, `mbirtorch/denoising.py`,
`tests/test_denoiser.py`.

Write `qggmrf_gradient_and_hessian_batched`, `vcd_subset_denoiser_batched`,
`QGGMRFDenoiser.auto_batch_size`, and `QGGMRFDenoiser.denoise_stack` as
Section 2.1 specifies.  Leave `vcd_subset_denoiser` and `denoise` as they
are.

Tests, in `tests/test_denoiser.py`:

- The batched gradient and Hessian equal the single-image function exactly
  (`torch.equal`) on each volume of a stack, for subset 0 of a seeded
  partition, with compilation off.
- `denoise_stack` on six volumes of shape `(8, 10, 12)` equals a loop of
  `denoise` calls on the same volumes with the same pinned parameters and
  the same seeded partition.  The noise amplitude differs per volume, so
  the volumes converge at different iterations.  The match is bit for bit
  on the CPU and within a relative maximum difference of 1e-6 on every
  other device.  The per-volume iteration counts are equal, and at least
  two distinct counts occur.
- A numpy input returns numpy and a tensor input returns a tensor on the
  input's device.
- `batch_size=2` on five volumes gives the same result as one batch, which
  exercises the padded last batch.
- A denoiser configured with two devices raises from `denoise_stack`.

Stage 0 ends when these tests pass on every available device and the
existing denoiser tests pass unchanged.

### Stage 1: measurements

Done.  The three scripts and their records are in
`plans/experiments/features/mace4d/`, and Section 5 of the evaluation
summarizes them.

### Stage 2: the utilities

Files: `mbirtorch/utilities.py`, `mbirtorch/tomography_model.py`,
`tests/test_utilities.py`.

Port `construct_time_frame_models` as Section 3.1 of the first plan
describes, keeping the median angle step, the discarded trailing views, the
four `ValueError` conditions, and the printing suppressed after the first
frame.  Add `gpu_devices`, `cpu_devices`, and `default_devices` next to
`_resolve_device`.

Three tests: the 24-view case gives 5 frames and `view_slices[1] ==
slice(4, 12)`, a `frames_per_rotation` large enough to give a stride below
one view raises, and `default_devices()` is nonempty while `cpu_devices()`
has one entry.

Stage 2 ends when these tests pass.

### Stage 3: the shared module

Files: a new `mbirtorch/mace.py`, `mbirtorch/__init__.py`, a new
`tests/test_mace.py`, and `experiments/drunet/`.

Write the agent protocol, the loop with the folding update and the hybrid
scheduling, the pool resolution, `ForwardProxAgent`, `QGGMRFDenoiserAgent`,
`HyperplaneAgent`, and `temporal_filter_matrix`, as Sections 2.2 to 2.5
specify.  Export the public names through the lazy loader, with the
matching lines in the `TYPE_CHECKING` block that the export test enforces.
Then make `experiments/drunet/mace.py` and `agents.py` import from the
package, and confirm that `run_qggmrf_gate.py` still passes its 2D gate.

Tests, in `tests/test_mace.py`:

- The folding update equals the plain formulas on small random tensors, for
  four agents with random regions that cover the array, to a relative
  maximum difference of 1e-6, and the change statistic agrees to 1e-5.
- The two-worker path: agents whose tasks record the thread that ran them,
  with `devices=['cpu', 'cpu']`, show both workers running tasks and the
  queue emptied.
- The filter matrix equals the copied scipy filter on random inputs and on
  unit impulses at the first and last frames, for 12 and 30 frames at
  period 6, to 1e-5.  The matrix for 3 frames is zero.
- `HyperplaneAgent` with a stack denoiser that adds a constant, on a small
  4D array, equals the same operation done without batching, for each of
  the three axes and with a batch size that does not divide the slab count.
- The 2D equality gate of the nn_priors work as a test: MACE with
  `ForwardProxAgent` and `QGGMRFDenoiserAgent` at matched sigma and pinned
  prior parameters reproduces `recon` on a small problem within 1 percent
  NRMSE, on the CPU, in under a minute.

Stage 3 ends when these tests pass and the drunet gate script runs from the
package.

### Stage 4: `MACE4DModel`

Files: a new `mbirtorch/mace4d.py`.

Write the constructor and parameter registration, the host helpers
(`_normalize_prior_weights`, `_validate_sinogram`, `_expected_init_shape`,
`_validate_init_recon`, `_load_cached_init`, `_run_settings`,
`_write_run_info`), `set_device_pool`, `devices`, the data-fit agent of
Section 2.3, the construction of the three hyperplane agents, the global
sigma estimate, the qGGMRF configuration once per run and orientation, the
initialization, the call to `mace`, and the three log files.

The qGGMRF configuration for an orientation runs once per `recon`: permute
the initial image, merge it to 3D, set `sigma_noise` to the global sigma,
call `auto_set_regularization_params` on the row subsample, floor
`sigma_x`, set the subset count, and set `auto_regularize_flag=False`.  The
resulting parameters go into the `make_stack_denoiser` closure, which
creates a pinned `QGGMRFDenoiser` per device.  The frame-to-device
assignment is round robin over the pool.

Stage 4 ends when a reconstruction with 3 frames, one device, and
`dejitter=False` runs, gives finite values of shape `(3,) + recon_shape`,
writes the three log files, and a run with `set_device_pool(['cpu',
'cpu'])` shows tasks from both workers in `task_log.csv`.  It also ends
only when a check confirms that no worker thread exceeded the recompile
budget, read from torch's recompile counters after the two-worker run.

### Stage 5: tests

Files: a new `tests/test_mace4d.py`.

The table in Section 5 lists the groups.  The one-frame equality gate is
the whole-loop check.

Stage 5 ends when every group passes on every available device.

### Stage 6: documentation and registration

Files: `mbirtorch/__init__.py`, `docs/source/usr_mace4d.rst`,
`docs/source/usr_denoising.rst`, `docs/source/usr_api.rst`,
`docs/source/usr_api_overview.rst`, `docs/source/refs.bib`.

Export `MACE4DModel` through the lazy loader with its `TYPE_CHECKING`
line.  Port `usr_mace4d.rst`, keeping the description of the windows, the
trade between signal-to-noise ratio and temporal resolution, the filter,
and the citation, and adding a section on the building blocks in
`mbirtorch.mace`.  State the host memory rule of Section 2.6 and the
frame-count rule of the filter.  Add `denoise_stack` to the denoising page.
Add the page to `usr_api.rst` and the overview.

Stage 6 ends when the documentation builds without warnings and
`from mbirtorch import MACE4DModel` works through the lazy loader.

### Stage 7: `save_volume_as_gif` and the demo

Files: `mbirtorch/utilities.py`, `tests/test_utilities.py`, a new
`demo/demo_10_mace4d.py`.

Port `save_volume_as_gif` as Section 3.6 of the first plan describes, with
one change: write the GIF with Pillow, which matplotlib already depends on,
rather than adding imageio.  Import Pillow inside the function, as
`gen_text_phantom` in `mbirtorch/utilities.py` does, so that importing
mbirtorch does not require Pillow.  Write the demo on the driver's sequence:
`get_sino_and_model` with `auto_crop=True`, `MACE4DModel`, `gen_weights`
with `transmission_root`, `recon` with `init_dir` and `log_dir`, `np.save`,
and the GIFs.  Run it on the 4DCT phantom dataset with `num_frames=3` and
`downsampling=4`, and confirm the four behaviors the driver depends on:
`num_frames` and `view_slices` readable after construction,
`recon_params['iterations completed']` present, `run_info.txt` present and
appendable, and a second run loading the cached initialization.

Stage 7 ends when the demo runs end to end and writes the reconstruction,
the three log files, and the GIFs.

### Stage 8: the H100 measurement

Files: a script and an sbatch file under
`plans/experiments/features/mace4d/`, and a companion `.md`.

After Stage 0, run one job on one H100 with synthetic data at two sizes:
twelve frames of a 256-cubed volume and twenty-four frames of a 512-cubed
volume.  Sweep the batch size of `denoise_stack` over 4, 16, 64, and the
largest size that fits, recording the time per volume, the peak device
memory, and the per-volume loop time at the same sizes.  Time one
`prox_map` per frame at each size.  The batch-size knee sets the default
and calibrates `auto_batch_size`.  The cost is about one GPU-hour.

Stage 8 ends when the record is written and `auto_batch_size` is checked
against the measured peak memory.

## 5. Tests for `MACE4DModel`

| Group | What it checks |
|---|---|
| Prior weights | `0.5` gives `[0.5, 1/6, 1/6, 1/6]`; `[0.1, 0.2, 0.3]` gives `[0.4, 0.1, 0.2, 0.3]`; `1.5`, `-0.1`, `[0.5, 0.5, 0.5]`, and `[0.1, 0.2]` raise. |
| Device pool | `None`, `1`, an explicit list, `[0]`, and `'cpu'` resolve; a count above the pool raises; `'tpu'` raises; `set_device_pool(1)` gives one device; `['cpu', 'cpu']` gives two. |
| Construction | `num_frames` is 5, `len(model_list)` is 5, `view_slices[1]` is `slice(4, 12)`; a wrong sinogram shape and wrong weights shape raise before any computation. |
| Parameters | `dejitter` defaults to True; `set_params(rho_mann=0.25, dejitter=False)` reads back; `mace_prior_weight=1.5` raises when set. |
| A complete reconstruction | Seeded, `num_frames=3`, `dejitter=False`, one device: shape, finite values, three log files, the four dictionary keys, one timing entry, `iterations completed` equal to 1, `weights` recorded as `unit (weights=None)`, `init_recon.npy` written then reused; `stop_threshold_change_pct=1e9` stops after one iteration; a supplied `init_recon` is recorded as provided by the caller; a wrong `init_recon` shape raises. |
| Two workers | `set_device_pool(['cpu', 'cpu'])`, `num_frames=4`: `task_log.csv` holds rows from device 0 and device 1, and the result equals the one-device result to 1e-5. |
| The initialization cache | Absent file returns None silently; wrong shape returns None with one warning containing `invalid`; a valid file returns float32. |
| The one-frame equality gate | `num_frames=1`, `dejitter=False`, agent weights `[1/2, 1/6, 1/6, 1/6]`, the denoiser sigma set to `sigma_prox` times the square root of `3/2`, `sigma_x` pinned to the value `recon` used: the consensus after 30 iterations matches `recon` on the frame's views within 1 percent NRMSE. |

The smooth sinogram of the mbirjax test is kept as the test data, for the
reason its comment gives: a random sinogram reconstructs to a volume with
extreme values, on which the line search computes zero over zero.

## 6. Risks

| Risk | Severity | How to handle it |
|---|---|---|
| The folding update's lock serializes the folds | Low | Each fold is region-sized and in place; the task log shows the wait if it grows. |
| A worker exceeds the recompile budget with three orientation shapes per device | Medium | Stage 4 reads torch's recompile counters after the two-worker run; the floor of 64 covers eight shapes on eight devices. |
| Compiled reductions differ across devices at about 1e-7 | Low | The batched-versus-loop test gates at 1e-6 off the CPU and bit for bit on it. |
| The batch-size rule misprices the sweep on CUDA | Medium | Stage 8 checks `auto_batch_size` against the measured peak; the ledger's headroom applies. |
| Small hyperplane volumes make the line search compute zero over zero | Low | The subset floor of Section 3.2. |
| The one-frame gate fails at 1 percent | Medium | The nn_priors gates passed at 0.65 and 0.23 percent; a failure is diagnosed one agent at a time, with the qGGMRF single-volume agent in place of the three hyperplane agents first. |
| Host memory at production size | Medium | Eight arrays, stated in the documentation; request GPUs for host memory on gautschi. |
| The filter removes every temporal mode at small frame counts | Low | The tests set `dejitter=False`; the documentation states the rule. |
| The end-to-end phantom run fails only at full scale | Medium | Stage 7 runs small first; the cache and the logs make a failed full run resumable. |

## 7. Size and order

| Stage | Content | Approximate size |
|---|---|---|
| 0 | `denoise_stack` and the two batched functions, with tests | 350 lines |
| 1 | Measurements | Done |
| 2 | Frame construction and device helpers, with tests | 160 lines |
| 3 | The shared module, with tests, and the drunet scripts moved onto it | 700 lines |
| 4 | `MACE4DModel` | 450 lines |
| 5 | Tests for `MACE4DModel` | 300 lines |
| 6 | Documentation and registration | 100 lines of reStructuredText |
| 7 | `save_volume_as_gif` and the demo | 350 lines |
| 8 | The H100 measurement | A script, an sbatch file, and a record |

The total is about 2,000 lines.  Stages 0 and 2 are independent and come
first.  Stage 3 needs Stage 0 for its qGGMRF-based tests.  Stage 4 needs
Stages 2 and 3.  Stages 5, 6, and 7 need Stage 4.  Stage 8 needs Stage 0
only and can run in parallel with Stages 2 to 7.  Each stage is a separate
Opus specification, reviewed before the next starts.

## 8. Follow-ups

- The single 4D prior agent.  Section 4 of the evaluation shows that one 4D
  qGGMRF agent has the same equilibrium as the three orientation agents.
  Its direction weights are one sixth along the frame axis and one ninth
  along each spatial axis, with agent weights `[1/2, 1/2]` and the same
  sigma.  It
  needs one gradient and Hessian function with six in-plane offsets and
  reuses the sharded denoiser.  One comparison on a small problem, on
  iterations to a fixed change and on the result, decides whether it
  becomes an option.
- Total-variation and bilateral-filter agents, in 3D as whole-volume agents
  and in 4D through `HyperplaneAgent` or a 4D implementation, on the agent
  protocol of Section 2.2.
- Moving the state to the devices for problems whose host memory is the
  limit, once the eight-array form is in use.
