
### Group 1: questions that change results at production size

**13. Fix the denoiser's pixel grouping for the whole run.**  Each qGGMRF
denoiser updates its pixels in groups, and the assignment of pixels to
groups is drawn at random.  Today `denoise_stack` draws a new assignment
every time it is called.  The 4D reconstruction calls each denoiser once
per iteration, so the denoiser behaves slightly differently each time.  On
the test problems this does not matter, because each slice through the
frames has only 30 pixels and therefore one group.  At production size a
slice has hundreds of pixels and 16 groups.  Two calls of the same
denoiser on the same input then differ by about 1.5 percent, which is
larger than the 0.2 percent stop threshold of the reconstruction.  These
results indicate that the reconstruction cannot converge below that level
at production size.  mbirjax draws the assignment once per run.  The fix
is to let the caller pass the assignment to `denoise_stack`.  Ziyun built
and tested that fix on 2026-09-15 and then rolled it back, because it
changes a Stage 0 interface before your review.  Options: an optional
`partition` argument on `denoise_stack`, or an assignment stored on the
denoiser object.  Recommendation: the argument, because the caller can
see what was used.  Until this is decided, `MACE4DModel` is verified only
at one group per slice.

**17. Make a seeded run reproducible on more than one worker.**  A run on
one worker reproduces itself exactly.  A run on two workers does not.  The
reason is that each frame's data-fit step draws its pixel assignment from
the shared numpy random generator on the worker thread that runs it, so the
order of the draws follows the thread timing.  On a production problem the
effect is about one percent.  It is a reproducibility question, not an
accuracy question.  Options: seed each frame's draw by its frame index,
which needs `prox_map` to accept an assignment or a generator; or document
that a seeded run is reproducible on one worker only, and check that
behavior with a test.  Recommendation: the first, together with question
13, because both need the same kind of argument.  The measurement is in
the Stage 4 fixes table.

### Group 1 plans

Revised on 2026-09-19 against mbirtorch `main` at 69d4972 (version 0.1.0),
and revised again the same day after a three-reviewer panel.  Nothing of
the earlier plans is in the code: `denoise_stack` still draws its pixel
partition inside every call, `MACE4DModel` neither draws nor passes one,
and `ForwardProxAgent` still initializes `prox_map` on the worker that
runs iteration 0.  The related refactors listed at the end are deferred.

Both questions have the same cause: something that should be drawn once
per run is drawn on every call, or on whichever thread runs the call.  Two
mechanisms fix them.  The denoiser gets an explicit initialization with a
cache, following `prox_map` and its `prox_data`.  Every random draw a
data-fit agent or a denoiser makes is derived from a seed and a name, so
the thread that makes the draw and the moment it makes it no longer
matter.

**Plan for 13: initialize the denoiser once.**

`QGGMRFDenoiser` gains `initialize_denoiser(image=None, sigma_noise=None,
partition=None, rng=None)` and a cache, `denoise_data`, that mirrors
`prox_data`.  The method first settles the device layout, as `denoise`
does, so that nothing settled later can discard what it stores.  It then
computes and stores everything that depends on the image or on a random
draw.

- The noise level: `sigma_noise` when given, else estimated from `image`,
  else the model's current value.  `sigma_y` is set equal to it.  A fresh
  model with none of the three raises `ValueError`.
- The regularization parameters: when `auto_regularize_flag` is on they
  are estimated from `image`, and `image` is then required, or
  `ValueError`.  A 3D image, or a `Shards` image, takes the row-subsampled
  path `denoise` uses, with `_volume_shape` deciding the case.  A 4D stack
  takes `auto_set_regularization_params_from_stack`, which refuses shards.
  When the flag is off the current parameters are read.
- The partition: `partition` when given, else drawn over the image shape
  at `granularity[partition_sequence[0]]` with the model's own
  `use_ror_mask`, from `rng` when given and from the global generator
  otherwise.  A supplied partition is copied to a contiguous int64 tensor
  on the model's device, and validated: two dimensions, integer values in
  range, no index twice within a row, and the union of the rows equal to
  the pixel set the mask allows.  Repeats across rows are legal, since a
  drawn partition pads its last rows that way.

The cache holds the partition, the regularization parameters, the noise
level, and a signature: `recon_shape`, `granularity`, `partition_sequence`,
`use_ror_mask`, and the device.  Whether the partition was supplied is
kept on the model outside the cache.

`denoise` and `denoise_stack` gain `do_initialization=True`.  True, the
default, calls `initialize_denoiser` with the call's image and
`sigma_noise`, so a user who denoises a different volume on every call
sees the same draws and the same values as before.  False reuses the
cache.  A cache that does not exist yet is built whatever the flag says,
as `prox_map` does, except when a supplied partition was discarded by a
device-layout change, which raises and names `initialize_denoiser`.  On a
False call a `sigma_noise` given to the call replaces the cached one,
because `fm_constant` is a scalar that costs nothing to rederive, and the
reported parameters carry the value used.  A False call whose signature no
longer matches the cache redraws the partition and logs it when the cached
partition was drawn, and raises when it was supplied.  The statistics are
never recomputed on a False call.  `denoise`'s `use_ror_mask` argument
becomes `None` by default, meaning the model's current value, so a False
call does not change the signature by accident.  The sharded path of
`denoise` keeps making its per-device copies from the cached partition.

Two facts of the parameter machinery must be handled or the cache is a
no-op.  Every parameter registered through the `no_warning` path carries
the recompile flag, and `sigma_noise` is one of them, so every
`set_params(sigma_noise=...)`, which both methods make on every call, runs
`refresh_device_bindings` and clears the model's caches.  The denoiser
therefore registers `sigma_noise` without the recompile flag at
construction.  And `_invalidate_device_caches`, which a real layout change
still calls, clears `denoise_data` as it clears `prox_data`.  A difference
from `prox_map` is documented in both places: the denoiser's cache notices
a change of `granularity`, `partition_sequence`, or `use_ror_mask`;
`prox_data` does not.

The compiled instance of the stack sweep is keyed by a counter assigned at
construction rather than by `id(self)`, so a freed denoiser's compiled
artifact is never handed to a later object at the same address.

`QGGMRFDenoiserAgent` in `mace.py` gains `seed=None`.  On a call whose
model has no cache it calls `initialize_denoiser` with the call's image,
its `sigma_noise`, and a generator derived from the seed when one is
given.  Every call then passes `do_initialization=False`.  The 3D agent
becomes a fixed operator, and with a seed its partition no longer depends
on the worker that draws it.  With `seed=None` the draw comes from the
global generator, as the whole agent does today, so no existing script's
draws move.  The one existing test whose values move is
`test_mace_reproduces_the_standard_reconstruction`, because its agent
stops redrawing.  It is a 1 percent gate, and its new trace is recorded.

In `mace4d.py`, `_configure_orientation` draws the orientation's partition
once, with `use_ror_mask=False` and the subset count it already computes,
from a generator derived from the run's seed and the orientation's name,
and returns it beside the parameters.  The stack-denoiser factory hands it
to every per-device denoiser through `initialize_denoiser(partition=...)`,
with the parameters already set and `auto_regularize_flag` off, so no
image is needed there, and every call passes `do_initialization=False`.
Every worker's copy of one orientation then sweeps with one partition,
which the shared queue requires.  The run settings record that the
partitions were drawn once.

**Plan for 17: draws derived from a seed.**

Main-thread initialization does not make a run reproducible, because the
VCD loop draws the order in which it visits the subsets from the global
generator on every iteration, on the worker, in the public method
`vcd_partition_iterator`, and the per-frame reconstruction that
initializes the run draws its partitions and orders there too.  Nor does
saving a generator's stream position make a resumed run identical, because
a fresh process has an empty `prox_data` and draws its partitions again
from wherever the stream stands.  The plan therefore derives every draw
from a seed and a name, `np.random.default_rng([seed, name])`, so that a
draw depends on neither the thread nor the moment.

- `gen_pixel_partition` and `gen_set_of_pixel_partitions` in
  `vcd_utils.py` gain `rng=None`.  None draws from `np.random` exactly as
  now.  A `numpy.random.Generator` draws from it with the same call
  sequence.  The module docstring stops saying that the generators always
  use the global state.
- `vcd_partition_iterator`, `_vcd_recon`, `initialize_recon`, `recon`, and
  `prox_map` in `tomography_model.py` gain `rng=None` and pass it through.
  `TomographyModel` gains `initialize_prox(...)`, the initialization
  `prox_map` performs on its first call, moved into a method that stores
  `prox_data`, so an agent can initialize with one generator and sweep
  with another.  Every default is None, so every existing test and golden
  sees the same draws as before.
- `ForwardProxAgent` gains `seed=None`.  With a seed, a call whose model
  has no `prox_data` first runs `initialize_prox` with the generator named
  `init`, and every call sweeps with the generator named by its iteration;
  the per-frame initialization reconstruction in `_compute_init_recon`
  uses the generator named `initial image`.  With `seed=None` every
  generator is None and the agent draws from the global generator as it
  does today.  `state_dict` saves the seed, and `load_state_dict` restores
  it only when the key is present, because the data-fit agent passes a
  partial dict on every frame to install the warm start.  A resumed run
  in a new process therefore makes the same draws as the original.
- `MACE4DModel.recon` gains `seed=None`.  None draws one integer from the
  global generator at the start of the call, so `np.random.seed(k)` before
  `recon` fixes the run.  An integer is used as given.  Each frame agent
  gets the seed named by its frame index, each orientation partition is
  drawn from the seed named by its orientation, and the run settings
  record the seed.  Nothing else in `recon` draws.

Two runs of one seed on the same pool of identical devices then differ
only where the folding update adds task outputs in a different order,
which is float rounding, and where a rounding difference crosses the
line-search clamp or a stopping threshold, which the loop can amplify.
Ten consensus iterations were measured to turn a rounding kick into a
few 1e-6 relative.  The gate below allows for it.

**Tests.**  Few, fast, and on the CPU where they compare float
trajectories.

- `tests/test_denoiser.py`, extending the stack-versus-loop test at 16
  subsets: two `do_initialization=False` sweeps on one denoiser agree to a
  relative maximum difference of 1e-6, two `do_initialization=True` sweeps
  without reseeding differ by more than 1e-3, the cache survives a False
  call that passes `sigma_noise`, and a supplied partition is the one the
  cache holds.
- `tests/test_prox_map.py`: two calls with a fresh generator of one seed
  agree to 1e-6, and one with another seed differs by more than 1e-6.
- `tests/test_mace.py`, extending the two-worker test with stub agents that
  record the draws of a generator named by seed and iteration: the
  recorded sequences are equal between the inline run and the two-worker
  run; and a resume check, four steps against two steps, save, rebuild,
  load, two more, with equal draw sequences and averages within 1e-6.
- `tests/test_mace4d.py`, extending the end-to-end test on the CPU only,
  two iterations, stop threshold zero: two CPU workers against one agree
  to a relative maximum difference of 1e-4, where the defect measured 0.69
  to 0.80 on this problem and the rounding noise a few 1e-6.

**Rule for the implementer.**  Docstrings and comments describe what the
code does.  None of this section's history, names, dates, measurements,
or "today" goes into them.  The three refactors below stay out.

**Size.**  About 110 lines in `denoising.py`, 45 in `tomography_model.py`
and `vcd_utils.py`, 40 in `mace.py`, 25 in `mace4d.py`, and 90 of tests.

**Status, 2026-09-20.**  Both plans are implemented and staged in the
mbirtorch checkout on `prerelease`, in nine files, awaiting Greg's review
and commit.  The five test files give 33 passed, the goldens 8 passed, and
the whole suite 169 passed with 92 skipped and no failure.  The new tests
observed, on the CPU: two reused initializations of the stack sweep differ
by 0, two fresh draws by 1.5e-2; `prox_map` with one seed twice differs by
1.2e-7 and with another seed by 1.1e-2; seeded draws on two workers equal
the inline draws, with the averages 1.4e-7 apart, and a resumed run matches
to 0; two CPU workers against one on a seeded four-frame run differ by
6.0e-8 against the gate of 1e-4.  One reading of the plan was settled by
the implementer: a seed sequence accepts integers only, so a string name is
read as an integer before the generator is built.

**Related refactors, deferred.**  Three duplications were noticed while
reading the denoiser.  They are separate increments and none is part of the
plans above.

- The single-image and the batched forms of the subset update and of the
  qGGMRF gradient and Hessian can each become one function, because the
  single image is the batch with the batch axis absent: gathers as
  `flat[..., idx, :]`, reductions over `dim=(-2, -1)`, the update as
  `index_add_(-2, ...)`, the step as `alpha[..., None, None]`, with
  `active` and the halos optional.  The single-device path of `denoise`
  then runs the stack sweep on `image[None]`, and its own loop goes away.
  The m4d1 record shows the one-volume batch equal to the 2D path bit for
  bit in eager mode on the CPU and within 6e-8 under compilation on MPS.
- `denoise_stack` clips a `batch_size` larger than the stack, so the agent
  factory in `mace4d.py` pads a short slab itself to keep one compiled
  shape.  If `denoise_stack` pads instead of clipping, the factory's copy
  of the padding goes.
- `_denoise_sharded` repeats the update's formulas inline because its line
  search is distributed across devices.  Splitting the update into a piece
  that computes the direction and the four partial sums and a piece that
  applies a given step, with the single-device update as a compiled
  wrapper around both, removes the last copy of the formulas.  This
  touches every reconstruction's prior path and belongs in its own
  increment with a bitwise gate.

### Group 2: design choices where the torch version and mbirjax differ

**11. Public parameters for the denoiser strength.**  The one-frame test
needs the denoiser noise level and `sigma_x` set to specific values.  The
class has no parameter for either, so the test sets them through a
subclass that exists only in the test file.  The test therefore proves the
reconstruction but not the automatic estimates a user gets.  Two public
parameters, `denoiser_sigma` and `denoiser_sigma_x` with None meaning
automatic, would let the test use the public path and let a user set the
strength.  Recommendation: add the two parameters.  They are about ten
lines, and setting a strength is a reasonable thing for a user to want.

**9. A repeated GPU in the device list.**  The device list accepts the
same device twice, which gives two worker threads on one device.  On the
CPU this is how the tests exercise the threaded code.  On an Apple GPU two
workers crash inside Metal.  On a CUDA GPU two workers share one stream and
gain nothing.  Recommendation: refuse a repeated GPU in `set_device_pool`
and keep repeats for the CPU.

**12. The denoiser stop threshold with the warm start on.**  With the
denoiser warm start on, each denoiser call starts from its previous output
and meets the 0.2 percent stop threshold after two or three iterations,
before it has finished.  The reconstruction then converges more slowly and
settles 2.5 percent from the result with the warm start off.  The
threshold was tuned for a start from the input.  Options for Stage 8 to
measure: a tighter threshold when the warm start is on, or a minimum
number of iterations per call.  Recommendation: the minimum number of
iterations, because it keeps the threshold's meaning.

### Group 2 plans

Decided with Greg on 2026-09-15.  Questions 11 and 9 are settled here.
Question 12 gets a stopgap now and its measurement in Stage 8.

**Plan for 11: use the standard parameters, and add `sigma_noise`.**
`MACE4DModel` is a `ParameterHandler`, so it already carries `sharpness`,
`sigma_x`, `p`, `q`, `T`, `qggmrf_nbr_wts`, and `auto_regularize_flag`
with the library defaults.  Today it forwards none of them to the three
denoisers, which are built fresh in `_configure_orientation` with the
denoiser class defaults, so a user has no knob on the prior strength
except the agent weight.  The plan forwards them.  `sharpness` scales the
automatic `sigma_x`, as it does for every model.  The class registers
`sharpness=0` as its own default, which is the denoiser class's default and
what mbirjax ran, so the automatic values do not change.  When
`auto_regularize_flag` is off, which setting `sigma_x` through `set_params`
does as for every model, the denoisers take `sigma_x` from the class
instead of estimating it.  `p`, `q`, `T`, and the neighbor weights are
copied to the denoisers as they are.  One parameter is new: `sigma_noise`,
default None, which keeps the automatic estimate from the initial image as
mbirjax made it.  A float pins the denoisers' noise level, which is the
strength of the prior agents.  `sigma_prox` keeps its special handling for
the frame models.  Both values are already written to the run settings.
The one-frame gate then sets `sigma_noise` to `sigma_prox` times the
square root of 3/2 and `sigma_x` to the reference reconstruction's value
through `set_params`, and the test-only subclass goes.  About 25 lines in
`mace4d.py` and the test edit.

**Plan for 9: refuse a repeated GPU.**  `set_device_pool` raises
`ValueError`, naming the device, when a CUDA or MPS device appears more
than once in an explicit list.  A repeated CPU entry stays allowed, since
that is how the tests exercise the threaded path.  The docstring states
the rule.  Tests: `['mps', 'mps']` on a Mac and `['cuda:0', 'cuda:0']` on a
CUDA machine raise; `['cpu', 'cpu']` is accepted.  About 10 lines.

**Plan for 12: a stopgap now, the measurement in Stage 8.**  The page's
finding needs one correction: the reconstruction converges more slowly
with the denoiser warm start on, not off.  The cause is the stopping test,
which compares one sweep's change to the image norm.  From a warm start
the remaining distance to the new proximal solution is small, so the first
sweeps are small and the 0.2 percent test fires after two or three of
them, whether or not that distance has been closed.  From a cold start the
sweeps never reached 0.2 percent and always ran the cap of 15, so that
operator is not converged either.  Neither result is known to be more
correct.

The stopgap, decided by Greg: the denoiser stop threshold in `mace4d.py`
goes from 0.2 to 0.05 percent, and the cap stays at 15.  With the warm
start off nothing changes in practice, because the sweeps did not reach
0.2 percent within the cap and will not reach 0.05.  With the warm start
on, the early iterations run to the cap and the later ones, once the input
moves little between calls, stop at the tighter tolerance.  The threshold
is recorded in the run settings.  One constant and its comment.

The measurement, in Stage 8: run the denoiser to a converged reference,
a tight threshold with a high cap, and measure each setting's distance
from it, per call and in the final result, on a production-sized problem.
The candidate remedy after that is an inner tolerance that tightens with
the outer loop, set by the driver between steps from the previous
consensus change and capped at the default early on, which the agent
protocol already allows.  A minimum iteration count is rejected: it is a
different mechanism from the threshold and does not keep its meaning.
Question 12 stays open until the measurement is in.

### Group 3: choices already in the code

**1. A point budget for the strength estimate.**  The estimator that sets
`sigma_x` from a set of volumes holds about 16 times its input in host
memory.  At production size that is about 10 GB per slicing direction,
which fits the target machine.  Options: cap the estimator by sampling the
two spatial axes, or state the cost in Section 2.6 of the plan.
Recommendation: cap it, since the noise estimator already caps itself at
five million points.

**2. The `auto_batch_size` interface.**  It returns None on CPU and Apple
GPUs, where no memory budget can be read.  It takes an `init_supplied`
flag, because a supplied starting image costs one more array per volume.
It takes an optional `volume_shape` that must equal the denoiser's shape.
Recommendation: keep as it is.

**3. The hashes that guard the widening floors.**  A test records a hash
of each source file that sets a widening floor, so that a change to those
files is noticed.  Six of the hashes are out of date, one because of the
Stage 0 additions and five from before.  The test passes when a hash is
out of date, by design.  Recommendation: run the script that records the
current hashes, and note in the commit which files it updated.

**4. The Stage 2 defaults.**  The frame constructor prints nothing, the
device helpers ignore the device pin, and a model without one angle per
view is refused.  Recommendation: accept all three.

**6. The checkpoint saves one average buffer.**  The plan wrote "two
average buffers".  The second buffer is the running sum, which is zero
between iterations, so the checkpoint saves only the average.
Recommendation: accept.

**10. The expected agreement of the end-to-end check with mbirjax.**  The
plan expects the two libraries to agree to between 1e-3 and 1e-2.  On the
100-pixel test problem the random pixel grouping alone changes a result by
9e-2, so that level cannot be met with the default groupings.  With one
group per slice the two libraries agree to 7e-7.  Recommendation: make the
one-group configuration the check of record and correct Section 5 of the
plan.

### Group 3 plans

Decided with Greg on 2026-09-15.  Questions 3, 6, and 10 are accepted as
recommended.  Questions 1, 2, and 4 change the code as described below.

**Plan for 1: reduce the estimator's memory, then cap it.**  The
neighbor-difference estimator `_get_estimate_of_recon_std` in
`denoising.py` builds three int64 index arrays with `np.where`, stacks four
gathered copies, and takes `np.std` over the stack, which is roughly 16
times its input.  `recon` uses a different estimator, a typical sinogram
value over a typical path length on a subsample of at most 20 views, which
costs nothing.  Two changes.  First, rewrite the denoiser's estimator with
shifted views and a two-pass mean and squared deviation, so it needs about
four arrays of the input's size.  `np.roll` reproduces the wrap of the
`inds - 1` indexing exactly, and the values equal the current ones to
rounding.  Second, cap the points on the stack path:
`auto_set_regularization_params_from_stack` chooses whole volumes and a
contiguous block in the two spatial axes so that the total is at most five
million points, the budget the noise estimator already uses, which keeps
every neighbor difference between adjacent voxels.  Tests: the rewritten
estimator agrees with the current one to 1e-6 on the golden image and on a
random stack, and the cap gives the same `sigma_x` to 1e-3 on a stack
where it binds.  About 50 lines.

**Plan for 2: drop the extra array.**  `denoise_stack` today keeps four
device arrays per volume through the sweep when an initial stack is given
and three without, because it clones the initial copy and keeps the input.
The initial copy is fresh once it has crossed from the host, so it becomes
the working image without a clone.  A caller's tensor that already lives on
the sweep device is cloned, so it is never written.  The input is released
once the residual is formed, because the identity forward model carries the
data term in the residual.  The sweep then holds two arrays per volume in
both cases, with three at placement, and `init_supplied` goes from
`auto_batch_size` and from the ledger's stack pricing.  The None result off
CUDA stays and is documented.  An MPS budget reader can come later if a
problem needs it.  Tests: the existing batch-size tests without the flag,
and a check that the caller's tensors are unchanged after a call.  About 20
lines.

**Plan for 3.**  Run the script that records the widening-floor hashes,
and name the updated files in the commit.

**Plan for 4: honor the device pin.**  The silent constructor and the
refusal of a model without one angle per view stay.  The automatic pool
changes: `resolve_device_pool(None)` and the ceiling of the integer form
cap the default devices at `MBIRTORCH_NUM_DEVICES` when it is set, read
through the ledger's `pinned_device_count`, because the variable pins the
count for the whole process and the reconstruction policy already treats it
as explicit.  Explicit lists stay the caller's, and `gpu_devices` and its
siblings still report the hardware.  Tests: with the pin at 1,
`set_device_pool()` gives one device and `set_device_pool(2)` raises with a
message that names the pin.  About 10 lines.

**Plans for 6 and 10.**  The checkpoint saves one average buffer.  The
one-subset configuration is the check of record against mbirjax, and
Section 5 of the plan is corrected to say so.

### The denoiser strength across orientations

Decided with Greg on 2026-09-15: one strength for the whole 4D volume.
This replaces the decision of 2026-09-15 in the progress page's table that
gave each slicing direction its own `sigma_x`.  The three orientation
priors add up to one 4D qGGMRF prior only when they share `sigma_x`,
`sigma_noise`, and `sharpness`.  Per-orientation estimates make the
effective prior anisotropic by an amount set by sampling rather than by
design, and the mbirjax quirk that gives the XZ-t direction the YZ-t value
is the same problem in another form.

The plan.  `recon` estimates `sigma_x` once, from the initial image
treated as a stack of frames, each of the frame shape, through
`auto_set_regularization_params_from_stack`, so the neighbor differences
run along x, y, and z within a frame.  With the filter on, the filtered
initial image is used, as already decided, because it is the image the
denoisers see.  The value, floored as now and scaled by the class's
`sharpness` under plan 11, goes to all three denoisers as a pinned value,
and a user-set `sigma_x` replaces it.  `sigma_noise` is already one
value.  The run settings record one `sigma_x` in place of the three-entry
list.  The one-frame gate is unchanged, since it pins `sigma_x`.  The
check against mbirjax pins `sigma_x` to the mbirjax value through the new
parameter, so it stays at the one-subset agreement of 7e-7 rather than
the 1.5e-3 that a different `sigma_x` gave.

The time direction gets a stopgap parameter now.  `qggmrf_nbr_wts` weights
the row, column, and slice directions, and in every hyperplane volume the
row axis is the frame axis.  The class gains `nbr_weight_time`, default
1.0, and forwards `qggmrf_nbr_wts=[nbr_weight_time, 1, 1]` to the three
denoisers.  The six direction weights are normalized to sum to one inside
each denoiser, so the parameter sets the weight of a frame neighbor
relative to a spatial neighbor and leaves the overall prior scale to
`sigma_x`.  The default reproduces mbirjax.  The run settings record the
value, and also the ratio of the temporal to the spatial neighbor-difference
spread of the initial image, taken from the same subsample the estimator
reads, so that a user trying values has a number to look at.  A sweep of
this parameter, and any automatic rule for it, are contingent on later
approval: the right value likely depends on the overlap factor, the frame
rate, and the object, and a rule fitted to one data set is not expected to
hold across data sets.  About 20 lines, plus the run-settings entry.
