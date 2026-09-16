
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

Both questions have the same cause: something that should be drawn or
computed once per run is drawn on every call, and on whichever thread runs
the call.  The fix for both is to initialize once, explicitly, in the main
thread, following the pattern `prox_map` already uses with
`do_initialization` and its cached `prox_data`.  The plans below were
written on 2026-09-15 and are proposals for review.  No code has changed.

**Plan for 13: initialize the denoiser once.**  `QGGMRFDenoiser` gains a
method `initialize_denoiser(image, sigma_noise=None, partition=None)` and
a cache, `denoise_data`, that mirrors `prox_data`.  The method computes and
stores everything that depends on the image or on a random draw: the noise
estimate when `sigma_noise` is None; the regularization parameters when
`auto_regularize_flag` is on, from a row subsample of a 3D image or from
the whole-volume subsample of a 4D stack, as Stage 0 found necessary; the
pixel partition, drawn from the global generator or taken from the
`partition` argument, and placed on the device; and the batch size for a
stack.  `denoise` and `denoise_stack` gain `do_initialization=True`.  The
default keeps today's behavior for a user who denoises a different volume
on every call.  With `False` the cached data is reused, so the denoiser is
a fixed operator across calls, which is what a MACE agent needs.

Three rules say when the cache is rebuilt.  The caller's flag is the
primary control, and a cache that does not exist yet is built whatever the
flag says, as `prox_map` does.  The scalar constants, `fm_constant` from
`sigma_noise` and the qGGMRF parameter tuple, are rederived on every call,
so a strength schedule that changes `sigma_noise` between calls costs
nothing.  The cache carries a signature of the parameters the partition was
built from, `granularity`, `partition_sequence`, `use_ror_mask`, and the
device, and a `False` call whose signature no longer matches redraws the
partition and logs it.  The device-layout invalidation that clears
`prox_data` clears this cache too.  The statistics are never recomputed
unless initialization is requested.

The `partition` argument is the reason the cache alone is not enough.  A
hyperplane agent holds one denoiser per worker device, and the shared queue
may send a slab to a different device on the next iteration, so every copy
of one orientation must use the same partition.  `MACE4DModel` draws each
orientation's partition once, in the main thread, when it configures that
orientation from the initial image, and hands it to every per-device copy
through `initialize_denoiser`.  This is Ziyun's rolled-back argument moved
from `denoise_stack`, where it would be passed on every call, to the
initialization, where it is passed once and recorded.

What changes where.  `denoising.py`: the method, the cache, the flag on
both public methods, and the two openings shrink to a validation, an
optional initialization, and the sweep.  `mace4d.py`:
`_configure_orientation` becomes one `initialize_denoiser` call after the
subset count is set, and the stack-denoiser factory passes the partition
and calls `denoise_stack` with `do_initialization=False`.  Tests: nine
calls of a three-iteration run receive one partition per orientation, which
is the test Ziyun wrote and rolled back; a `False` call on a never
initialized denoiser initializes; a `False` call after a change of
`granularity` redraws once; and a repeated call of one denoiser on one
input gives a difference of zero.  The end-to-end check against mbirjax
stays at one subset per volume (question 10), because mbirjax redraws its
own partitions on every proximal map and no setting on our side removes
that.  The Stage 4 verification then extends from one subset to the
production count.

**Plan for 17: draw the frame partitions in the main thread.**  Each
`ForwardProxAgent` gains an `initialize` method that runs the model's
`initialize_recon` and stores its result as the model's `prox_data`,
exactly as `prox_map` does on its first call.  The data-fit agent calls
`initialize` on its frame agents in frame order, in the main thread,
before it dispatches the tasks of iteration 0, and every task then calls
`prox_map` with `do_initialization=False`.  The draws from the global
generator now happen in one thread in a fixed order, so a seeded run
reproduces itself on any number of workers.  The denoiser partitions are
already drawn in the main thread under the plan for 13.  No new argument
on `prox_map` is needed, and no per-frame seeding.  The cost is `T`
sequential initializations at iteration 0, each a partition draw and a
statistics subsample of one frame's sinogram, which is small beside one
proximal map.  Test: two workers against one at the default partitions
give a difference of zero, where the measurement of 2026-09-15 gave 0.89
on the small problem.  The existing one-subset test then returns to the
default partitions.

**Related refactors, not Group 1 questions.**  Three duplications were
noticed while reading the denoiser for these plans.  They are separate
increments and none blocks the two plans above.

- The single-image and the batched forms of the subset update and of the
  qGGMRF gradient and Hessian can each become one function, because the
  single image is the batch with the batch axis absent: gathers as
  `flat[..., idx, :]`, reductions over `dim=(-2, -1)`, the update as
  `index_add_(-2, ...)`, the step as `alpha[..., None, None]`, with
  `active` and the halos optional.  The single-device path of `denoise`
  then runs the stack sweep on `image[None]`, and its own loop goes away.
  The m4d1 record shows the one-volume batch equal to the 2D path bit for
  bit in eager mode on the CPU and within 6e-8 under compilation on MPS,
  so the existing goldens gate the change.  About a hundred lines leave.
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
  increment with a bitwise gate, best done when the retire_jax baselines
  are regenerated.

**Order and size.**  The plan for 13 first, since the port is verified
only at one subset per volume until it lands.  About 120 lines are moved or
added in `denoising.py` and `mace4d.py` and about 60 lines of tests.  The
plan for 17 with it, about 30 lines.  Then the first two refactors as one
increment, and the third when the baselines are regenerated.  All of this
comes before Stage 6.

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
