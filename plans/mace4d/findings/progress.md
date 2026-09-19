# MACE4D port: progress for review

Written 2026-09-14, revised the same day after Stage 4.  This page summarizes the work on the port of MACE4D
from mbirjax to mbirtorch since the plan of record,
`plans/mace4d/plan.md`, was written on 2026-09-13.  It lists what
each stage delivered, what the tests observed, which design choices were
made on the way, and which decisions are still open.  The detailed records
it points to are beside this page and in
`plans/mace4d/experiments/`.

Note of 2026-09-19: the m4d4, m4d5, and m4d7 records and the stage 0, 3, and 4 panel reviews
cited below were never committed to this repository; see the note under "Status by stage" in
`plans/mace4d/plan.md`.

## Where the work stands

Stages 0 through 5 of the nine are done.  Stage 1 was done before the
port, Stages 0, 2, 3, and 4 are pushed to the mbirtorch branch
`mace_4d_dev`, and Stage 5 is pushed too.
Stages 0, 3, and 4 were reviewed by a four-member panel.  Stage 2 was
checked against mbirjax exactly.  Stage 4 was also checked against mbirjax
end to end; in the one configuration in which the two libraries compute the
same iteration, they agree to float32 rounding.  Every test of this port
passes on the CPU and on MPS.  The whole suite passes except one wall-clock
gate of the geometry viewer, which fails under the load of eight workers
and passes when run alone.  Greg has reviewed none of the stages yet, and
Stage 5 waits for that review.

| Stage | Delivers | State | mbirtorch commits |
| --- | --- | --- | --- |
| 0 | `denoise_stack`, the batched qGGMRF sweep, `auto_batch_size`, and the volume-subsample statistics | pushed | `79d5321`, `b80f6f5` |
| 1 | The three measurement scripts | done before the port | none |
| 2 | `construct_time_frame_models` and the device helpers | pushed | `3c6236a` |
| 3 | `mbirtorch/mace.py`: the `MACE` class, the agents, the filter matrix; the drunet scripts moved onto it | pushed | `42e0991`, and `2763640` for the `denoise_stack` compile key |
| 4 | `mbirtorch/mace4d.py`: `MACE4DModel`, the data-fit agent, the filter utilities moved in; the first tests | pushed | `7893f22` (the filter move), `8304b25` (the class), `d7112b8` (the fixes after the panel) |
| 5 | `tests/test_mace4d.py` completed: the unit groups and the one-frame equality gate | pushed | `ef13956` |
| 6 to 8 | The documentation, the demo, the H100 measurement | not started | |

No CUDA device was available for any of this work.  Every result below is
from the CPU and from MPS on one Mac.  The CUDA paths of `auto_batch_size`
and of the batched sweep have not run.

## Stage 0: the stack denoiser

**What it delivers.**  `QGGMRFDenoiser.denoise_stack` denoises a stack of
same-shaped volumes at once.  Each volume has its own step size and its own
stopping test, so the result equals a loop of single-volume calls.  Two
batched functions carry the arithmetic.  `auto_batch_size` prices one
volume's denoise plan with the memory ledger and scales it by the batch
count.  A follow-up the next day added
`auto_set_regularization_params_from_stack`, which sets the regularization
parameters from a subsample of about 20 whole volumes.

**Test results.**  23 new tests in `tests/test_denoiser.py`.

| Check | cpu | mps | Tolerance |
| --- | --- | --- | --- |
| Batched gradient and Hessian against the single-image function | 0 | 0 | 1e-7 |
| `denoise_stack` against a loop of `denoise`, six volumes, iteration counts equal | 5.9e-8 | 5.9e-8 | 1e-6 |
| Batches of 2 against one batch of 5 | 5.8e-8 | 0 | 1e-6 |
| Against mbirjax's batched hyperplane denoiser, four cases, partitions and iteration counts identical | 2.0e-7 | 2.0e-7 | 1e-3 |

The record is `plans/mace4d/experiments/m4d4_denoise_stack_check.md`.

**Findings.**  The CPU rounds the elementwise qGGMRF chain differently on a
3D tensor than on a 2D one when the element count is not a multiple of 8,
at about 5e-8; MPS agrees bit for bit.  The row-subsampled `sigma_x` path
the plan first specified gave 1.6 to 2.2 times mbirjax's value on the test
stacks, moving the result by 5 to 8 percent, because the row neighbors in
the merged stack lie several frames apart.  The volume subsample restored
mbirjax's value to every digit for stacks of at most 39 volumes and to 0.3
percent for a 60-volume stack.

**Design choices taken.**  `auto_batch_size` returns None on CPU and MPS,
where no memory budget can be read, and `denoise_stack` takes None as the
whole stack.  It takes an `init_supplied` flag, because a supplied initial
stack is one more array per volume.  The batched functions are not exported
from the package.  The `sigma_x` floor of 1e-6 is applied inside the new
statistics method, so `MACE4DModel` will apply none of its own.

**Panel review.**  All four reviewers: merge after fixes, no numerical
defect.  Record: `stage0_panel_review.md`.  The two threading findings were
resolved in Stage 3.  The remaining ones are listed under the decisions and
the follow-ups below.

## Stage 2: the frame construction and the device helpers

**What it delivers.**  `construct_time_frame_models` splits a scan into
overlapping windows of consecutive views and builds one model per window.
`gpu_devices`, `cpu_devices`, and `default_devices` report the hardware.

**Test results.**  5 new tests in `tests/test_utilities.py`.  The view
slices of five parameter sets match mbirjax exactly, including 240 views at
2.5 degrees with the angles stored modulo 360, which is how the NSI
preprocessing stores them, and its monotonic form.  Record:
`plans/mace4d/experiments/m4d5_time_frames_check.md`.

**Design choices taken.**  The verbose suppression of the mbirjax function
was left out, because mbirtorch prints nothing when a model is built or
copied.  The device helpers do not read `MBIRTORCH_NUM_DEVICES`.  A model
without a one-dimensional angle vector is refused with a clear error.  The
function is not exported from the package.

**Findings.**  A frame span or stride whose ratio is exactly one half sits
on a float32 rounding boundary; both libraries round it the same way, and
the tests keep away from it.  At the default overlap factor a stride below
one view is reported as a span below one view, in both libraries.

## Stage 3: the shared MACE module

**What it delivers.**  `mbirtorch/mace.py` holds the `MACE` class, the
agent protocol with tasks and regions, the `mace` one-call function, the
agents `ForwardProxAgent`, `QGGMRFDenoiserAgent`, and `HyperplaneAgent`,
and the device-pool resolver.  It also held the filter matrix builder and
its application until Stage 4 moved them into `mace4d.py`.  The loop adds each agent's output into the
consensus as it arrives, so it holds three full-size arrays beyond the agent
inputs.  A device pool runs one worker thread per entry with a shared queue.
The loop can be saved and resumed.  The drunet scripts import from the
package.

**Test results.**  35 tests in `tests/test_mace.py`.

| Check | Observed | Tolerance |
| --- | --- | --- |
| The folded update against the formulas written in full, four agents, random regions, three steps | 5.8e-7 on `W`, 1.9e-7 on the average | 1e-6 |
| Two workers against the inline run | 2.1e-7 | 1e-6 |
| Checkpoint round trip | 0 | 1e-6 |
| Filter matrix against the transform filter, 12 and 30 frames | 1.5e-7 | 1e-5 |
| Equality gate: prox agent and qGGMRF agent reproduce `recon`, start 2.5 percent away | 0.60 percent after 30 iterations | 1 percent |
| The drunet gate script from the package | NRMSE 0.00647, the value before the move | 0.01 |
| Whole suite, eight workers | 1099 passed, 143 skipped | |

**Findings.**  A 30-iteration `recon` is not a valid gate reference.  On the
64-view problem it is 2.5 percent from the 100-iteration result, and MACE
converges past it.  The gate test uses a 100-iteration reference, whose own
error is 0.05 percent.  The same rule applies to the one-frame gate of
Stage 5.  The old drunet loop and the new module agree to four digits on the
same problem, and the partition schedule changes the trace only in the
fourth digit, because the default sequence reaches its finest partition by
the third iteration.

**Panel review.**  All four reviewers: merge after fixes.  Record:
`stage3_panel_review.md`.  Five confirmed defects were fixed the same day,
each with a test: the average handed to callbacks was a buffer the next step
zeroed; a failed step left the state inconsistent without notice; the
hyperplane warm start passed zeros on the first call; the spread statistic
allocated two region-sized temporaries under the lock; and a model on
several devices crashed the agents.

**Design choices taken.**  One stack denoiser per worker thread.  The
change statistic is infinite when the previous average is zero, so a zero
start never stops the loop.  `state_dict` saves the average and not the
accumulation buffer, which is zero between steps.  The compiled instance of
`denoise_stack` is keyed by denoiser object, which resolves a Stage 0
finding.  The `mace` function is reached as `mbirtorch.mace.mace`, because
the package attribute is the module.

**Decisions Ziyun made after the panel** are in the table "Decisions Ziyun
made" below, dated 2026-09-14.

## Stage 4: `MACE4DModel`

**What it delivers.**  `mbirtorch/mace4d.py` holds `MACE4DModel`, the
public 4D class.  Its constructor builds the frame models and registers the
ten reconstruction parameters.  `recon` validates its inputs, takes the
initial image from the caller, the cache, or a per-frame reconstruction,
estimates one denoiser noise level, configures the three hyperplane
denoisers from the initial image, and runs the shared `MACE` loop over four
agents on the device pool.  The data-fit agent is private to the module.
It runs one `ForwardProxAgent` per frame on the frame's device.  The frame
outputs are kept in one host stack, and that stack is each frame's warm
start.  When the filter is on, the agent applies it to that stack.  The three log files are
written as in mbirjax, with the mean denoiser iteration count added to the
timing log and a `part` field to the task log.  The two filter utilities
moved from `mace.py` into this module, with their tests; the hyperplane
agent applies its matrix through a private helper, so the loop module does
not import the 4D module.

**Test results.**  10 tests in `tests/test_mace4d.py`, three of them the
moved filter tests.

| Check | Observed | Tolerance |
| --- | --- | --- |
| Three frames, one iteration, one device: finite, the 4D shape, the three log files, the four result keys, the cache written then read | passes on cpu and mps; `sigma_x` per orientation agrees between the devices to 7 digits | |
| Two CPU workers against one, with one pixel subset so the thread order of the random draws cannot matter | 0 | 1e-5 |
| Both workers take tasks | rows from workers 0 and 1 | |
| Compiled bodies after a run in a fresh process: no fallback where the machine can compile; variants per function against the recompile budget | mps, one worker: at most 5 variants of 64, no fallback; cpu, two workers: nothing compiles here, and every fallback is the toolchain's | |
| The data-fit agent's filter path against the filter applied to the stacked outputs, with stub frame agents; the warm start is the unfiltered output | 0 | 1e-6 |
| The filter path end to end, four frames at period 4 | finite, two iterations | |
| Whole suite, eight workers | 1104 passed and 143 skipped.  One wall-clock gate of the geometry viewer failed under the load of eight workers.  Run alone it passed at 52 ms against its 100 ms limit.  It behaved the same way before Stage 3 was pushed. | |

**Check against mbirjax.**  Record:
`plans/mace4d/experiments/m4d7_mace4d_check.md`.  The check was
run in a configuration in which the two libraries compute the same
iteration.  It uses one pixel subset, so mbirjax's redraw of the partitions
on every proximal map cannot change the result.  In that configuration the
torch model, with its own `sigma_x` per orientation, is 1.5e-3 from
mbirjax.  When the XZ-t denoiser is given the YZ-t `sigma_x`, which is what
mbirjax does, the two agree to 7e-7 on both devices.  With the default
partitions the difference is 1.1e-1.  Two identical mbirjax proximal maps
differ by 9.3e-2 on this problem, so that difference is the size of
mbirjax's own redraw and not a difference between the libraries.  The
plan's expected level of 1e-3 to 1e-2 assumed a larger problem.

**Findings.**

- **torch's inductor cannot build CPU kernels on this Mac.**  clang finds
  no C++ standard headers under the installed SDK, so a compiled body in a
  CPU run on this machine falls back to eager after its first call.  MPS
  compiles.  The Stage 4 compile check therefore holds on MPS and, on the
  CPU, checks only that every fallback is the toolchain's.  The CPU test
  results of this port on this machine are eager results.
- **Two worker threads on one MPS device crash inside Metal**, with a
  command-encoder assertion and a segmentation fault.  One worker per MPS
  device runs.  The two-worker check is on the CPU, as the plan wrote it.
  Whether a repeated GPU entry in the pool should be refused is a question
  for Greg; on CUDA it is untested.
- **mbirjax shares one denoiser between YZ-t and XZ-t** whenever the frame
  volume is square in x and y, which is the case for the default cone-beam
  reconstructions of both libraries, because its denoiser cache is keyed by
  volume shape and configured once.
  The XZ-t orientation therefore runs with the YZ-t `sigma_x`.  The port
  computes each orientation's own value.  A decision for Greg.
- **The partition redraw is not a small perturbation on a small problem.**
  The Stage 5 one-frame gate compares against `recon`, not against mbirjax,
  and is unaffected.
- **The two warm starts, measured.**  On a converging problem (64 views of
  a Shepp-Logan cone-beam scan, 32 by 32 by 4 volumes, three frames of 21
  views, 60 iterations, one initial image for every setting), the prox warm
  start was needed for convergence: without it the consensus change sat at
  7 percent from iteration 3 on, because three inner iterations from the
  input do not reach the proximal map.  The denoiser warm start cut the
  sweeps from the cap of 15 to about 3 iterations per volume.  The
  consensus change then settled at 0.04 percent, and the result was 2.5
  percent from the result of the default settings.  These results indicate
  that the 0.2 percent stop threshold is not a valid test with the warm
  start on: it measures the size of one sweep, and a warm-started sweep is
  small from its first iteration.  With the cold start the sweeps never
  reached the threshold and always ran the cap.  The saving in time was 25
  percent here, where the volumes are small and the overhead per call
  dominates; at production size the sweeps dominate and the saving should
  approach the ratio of the largest iteration counts in a batch.  A tighter
  threshold, or a minimum sweep count, when the warm start is on is a Stage
  8 experiment.  The script is not yet a record.

**Panel review.**  Four Opus reviewers on 2026-09-14, all "merge after
fixes".  Record: `stage4_panel_review.md`, with the raw reports beside it.
The two largest findings are ones the test problem cannot show.  The
hyperplane denoisers draw a new pixel partition on every call, so at more
than one subset per hyperplane volume they are not fixed operators, and
the loop cannot settle below their call-to-call change; mbirjax draws that
partition once per run.  And the shipped defaults on a short scan build the
zero filter matrix and return an all-zero result while reporting success.
The compile-budget check was also found to read counters that earlier
tests left behind.

**Fixes after the panel, 2026-09-15, pushed as `d7112b8`.**  Taken one finding at a
time with Ziyun.

| Finding | State |
| --- | --- |
| 1, the denoisers redraw their partition | Open.  A fix was built and measured: `denoise_stack` took an optional `partition`, the class drew one per orientation and reused it, and a test confirmed that nine calls of a three-iteration run received one partition per orientation, where a redrawn partition moved a 16-subset result by 1.7e-2.  Ziyun rolled the change to `denoising.py` back the same day, with doubts about changing the Stage 0 interface, and the class change went with it.  Decision 13 stands.  Until it is made, `MACE4DModel` is verified only at one subset per hyperplane volume. |
| 2, the zero filter matrix | Fixed.  With fewer frames than the period, or a matrix that removes every mode, `recon` turns the filter off, warns once, and records it in the run settings.  Ziyun chose this over an error. |
| 3, the compile check | Fixed.  It is one test in a fresh process.  On this Mac the CPU run compiles nothing and its seven fallbacks are the toolchain's; the MPS run holds at most 5 variants per function of the budget floor of 64, no fallback. |
| 4, two workers not reproducible | Open, decision 17.  Measured on 2026-09-15 on the 24-view problem at default partitions, after the denoiser partition was pinned: one worker against its repeat 0; two workers against one 0.89; two workers against their repeat 1.01.  The cause is the proximal maps' partition draws on the worker threads; the size is inflated by the 100-pixel problem.  Ziyun chose to record it and decide with Greg; nothing changed in the code. |
| 5, one host array more than Section 2.6 | Fixed.  The data-fit agent adopts the initial image as its stack when `recon` read or computed it, and copies it only when the caller supplied it, so the caller's array is never written.  The counts are the plan's 7, 8, and 11, or one more when the caller supplies the initial image. |
| 6, the filter error after the initialization | Fixed with 2: the filter is decided before any computation. |
| 7, the short last slab compiles a second shape | Fixed.  The stack denoiser pads a short slab by repeating its last volume to the batch size and discards the padded results, so one compiled shape serves every slab of an orientation.  A test forces a batch size of 3 on hyperplane counts of 8 and 10: all eleven sweeps see three volumes, and the result equals the whole-orientation run to 2.1e-9. |
| 8, `dejitter_verbose` | Fixed, by Ziyun's choice.  Every run with the filter on writes one line to `run_info.txt`, for example "removes periods of 6, 3, 2 frames; 8 of 30 modes removed, 22 kept", and `dejitter_verbose` logs the same line.  The kept count is the trace of the filter matrix.  Decision 16 is closed. |
| The smaller findings | Done on 2026-09-15, except two that touch Stage 3 code.  The `task_log.csv` column is `worker`; `set_params` returns nothing; the result's shared storage has its comment; the noise estimate's denoiser is pinned to the pool's first device; the `num_frames` check runs before the frames are built.  Tests added: the filter changes the result, the slab loop runs in several slabs, the data-fit agent's checkpoint round trip, a constant initial image is refused.  Ziyun decided on 2026-09-15 that the filter functions stay in `mace4d.py`, so the four-line copy of the application in `mace.py` stays; the data-fit agent's clearing of a frame agent's private attribute stays as it is. |
| The writing items | Done on 2026-09-15 in the code, the tests, this page, and the m4d7 record. |
| After the fixes | The whole suite: 1110 passed, 143 skipped, and the geometry viewer's wall-clock gate as before.  The m4d7 comparison rerun: the same numbers. |

**Design choices taken.**  With the filter on, the denoiser strengths are
estimated from the filtered initial image, the image the denoisers see;
the noise level is estimated from the unfiltered image, as in mbirjax.
The frame agents' warm start is the shared host
stack, which is the initial image itself when `recon` made that image and a
copy when the caller supplied it, loaded into each frame agent before its
call and the agent's own device copy released after it.  A one-device pool still runs one worker
thread, so every run takes the same path.  Frames go to pool entries round
robin.  The initialization reconstruction runs one thread per pool entry.
`sigma_x` per orientation and the batch sizes are written to `run_info.txt`.
Four points were left open in the Stage 4 specification, and the default
was taken for each.  The sampled `sigma_x` is accepted and recorded.
Logging goes through the instance logger.  No typing alias was added, and
no checkpoint directory was added.

## Stage 5: the tests

**What it delivers.**  `tests/test_mace4d.py` grows from 14 to 24 tests
and covers every group of the plan's Section 5 table.  The unit groups are
ported from mbirjax's tests: the prior weights, the model's device pool,
the frame construction, the parameters, and the initialization cache.  The
reconstruction group hangs off the existing 3-frame run.  One more run
covers the settings no test had used: both warm starts the other way
round, a given `sigma_prox`, a list prior weight, and verbose logging.  A
stub test covers the data-fit agent without a stack.  And the one-frame
equality gate is the whole-loop check.

**The gate.**  With one frame the three priors denoise the same volume
along three axis pairs.  At the agent weights `[1/2, 1/6, 1/6, 1/6]`, with
the denoiser sigma at `sigma_prox` times the square root of 3/2 and
`sigma_x` pinned to the value `recon` used, the consensus must reproduce
`recon` on the frame's views within 1 percent NRMSE after 40 iterations;
the plan wrote 30, and Ziyun chose 40 on 2026-09-15 for the wider margin.

| Quantity | Value |
| --- | --- |
| Problem | 64-view Shepp-Logan cone-beam scan, 32 channels, 4 detector rows, noise at seed 0, transmission-root weights |
| Frame | one frame at one frame per rotation, so it holds all 64 views |
| Reference | `recon` for 200 iterations; against `recon` for 100 iterations it is 3.8e-3, asserted below 5e-3 |
| Start | `recon` for 30 iterations, 5.9 percent from the reference, asserted above 1 percent |
| NRMSE to the reference at 10, 20, 30, 40 iterations | 3.0, 1.6, 0.82, 0.45 percent |
| Gate | below 1 percent at 40; passes with a margin of 0.55 percent |
| Run time of the gate test alone | 91 seconds on this CPU |

**Findings.**

- **A limited-angle frame cannot be the gate's problem.**  The first
  choice, one frame of 21 views spanning 120 degrees, converges slowly:
  `recon` at 100 and 200 iterations differ by 1.5 percent, and at one pixel
  subset by 5.2 percent, so no reference converged below the tolerance was
  reachable.  The full-rotation frame converges to 0.38 percent.
- **One frame is the one-subset regime by itself.**  The hyperplane subset
  rule gives `T * d1 = 32` pixels per volume and so one subset, whatever
  the scan model's partitions.  The scan model therefore keeps its default
  partition sequence, and the gate sits inside the regime the class is
  verified in (decision 13) without any special setting.
- **The margin at 30 iterations was thin**, 0.82 against 1 percent, with
  the 200-iteration reference.  Ziyun chose 40 iterations, where it is 0.45
  percent, for about 10 seconds more.  The plan's Section 6 lists a failure
  of this gate as a medium risk; the risk is real but the gate holds.
- **The denoiser sigma and `sigma_x` are pinned through a test-only
  subclass**, decision 11's default.  The gate therefore proves the loop is
  right and says nothing about the shipped automatic values.

**Test results.**  The file: 24 passed in 3 minutes 19 seconds, against
the budget of three minutes; the gate test takes 79 seconds and the
3-frame run 30 seconds on MPS and 22 on the CPU.  The whole suite with
eight workers: 1118 passed, 143 skipped, and the geometry viewer's
wall-clock gate as before; the suite's wall time is unchanged at about
three minutes, because the runner spreads the file's tests over its
workers.  The budget overrun is 19 seconds; the prompt's default is to
reduce the gate's detector rows from 4 to 2, which needs the reference's
convergence re-measured, and that is left for Ziyun to decide.

## Decisions Ziyun made

Each was taken on the date given, is in the code or the records, and
stands unless Greg asks for a change.

| Date | Question | Decision |
| --- | --- | --- |
| 2026-09-14 | Who sizes the hyperplane batches | `MACE4DModel` computes each orientation's batch size from its configured denoiser's `auto_batch_size()` and passes it to `HyperplaneAgent`; the agent's None stays one task per orientation. |
| 2026-09-14 | A resume with other `mu` or `rho` | `load_state_dict` warns and keeps the loop's values. |
| 2026-09-14 | The `mace` function's traces changed definition | Keep the new definitions, which measure against the previous average; dated notes on the two nn_priors findings pages. |
| 2026-09-14, 2026-09-15 | Where the filter utilities live | Both move to `mace4d.py` (commit `7893f22`) and stay there; the loop module keeps a private four-line copy of the application rather than import the 4D module. |
| 2026-09-15 | Fewer frames than the filter period | The filter turns itself off with a warning and a note in the run settings, rather than raise. |
| 2026-09-15 | What `dejitter_verbose` does | Every filtered run records the periods removed and the modes kept; the parameter logs the same line. |
| 2026-09-15 | The panel's fix for the denoiser partition redraw | Built, then rolled back: changing `denoise_stack` is Greg's call (decision 13 below). |
| 2026-09-15 | Two workers not reproducible | Record it and decide with Greg (decision 17 below); no code change. |
| 2026-09-15 | Two panel items that touch Stage 3 code | Left as they are: the data-fit agent clears a frame agent's private attribute, and the filter application exists in both modules. |
| 2026-09-15 | How the Stage 5 gate pins the denoiser parameters | A test-only subclass, the default of decision 11; the public parameters stay Greg's call. |
| 2026-09-15 | The gate's iteration count | 40 rather than the plan's 30, for a margin of 0.55 percent against the 1 percent tolerance. |
| 2026-09-15 | Starting Stages 4 and 5 before a review | Proceed; the panels serve as the review in the meantime. |
| 2026-09-15 | The image the denoiser strength is estimated from (was open question 14) | The filtered initial image when the filter is on, as mbirjax does, since that is the image the denoisers see.  Changed in the code the same day, commit `2ef9b82`; the noise level itself is still estimated from the unfiltered image, as in mbirjax. |
| 2026-09-15 | The strength of the XZ-t denoiser (was open question 8) | One strength per slicing direction, estimated from that direction's own slices, which is what the code does.  mbirjax shares the YZ-t value with XZ-t through its cache; Stage 6 says so in the documentation. |

## Decisions that wait for Greg

Eleven questions are open.  Each one is a choice that the code cannot
settle on its own.  For each, this section states the question, what the
code does today, the options, and a recommendation.  The numbers are the
ones the questions were given when first raised, so the numbers missing
here are the closed questions in the table above.  "The torch version"
below means the new PyTorch implementation of MACE4D; the rest of this
page calls it the port.

The questions fall into three groups.  The first group changes results at
production size and should be decided first.  The second group is design
choices where the torch version and mbirjax differ.  The third group is
choices already in the code that need a yes or a no.

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

Also to confirm: the fourteen decisions in the table above.

## Follow-ups not yet done

- Four items from the Stage 0 panel are still open.  Two tests are
  missing: `denoise_stack` with `sigma_noise=None`, and the sizing branch of
  `auto_batch_size` on the CPU with a patched budget reader.  The wording
  "row subsample" is stale in the first section of the m4d4 record and in
  its compare script.  The `stack_ell1` docstring claims exact equality,
  which holds only below one chunk.  Three small items remain: the
  empty-stack message, `sigma_prox` not being floored, and the read-only
  numpy warning.
- Stage 3 writing: the sentence-level items in
  `stage3_panel_review_reports.md` are still open.
- The Stage 3 gate test runs about 55 seconds alone and about 70 under the
  eight-worker suite, against the plan's "under a minute".  A smaller
  problem would leave less room between the measured error and the gate's
  tolerance.
- The warm-start measurement of the Stage 4 findings is a script in the
  session's scratch directory; it should become an m4d record with its
  companion page before Stage 8 builds on it.
- An end-to-end check against mbirjax with the filter on has not been run.
  It needs a scan of at least 12 frames at the default period, since both
  libraries build a zero filter below that; the 24-view test scan gives 5.
- Stage 6 documentation: state that each denoiser's strength is estimated
  from its own slicing direction, and that mbirjax gives the XZ-t direction
  the YZ-t value.
- This Mac's C++ toolchain: torch's inductor cannot build CPU kernels here,
  so CPU tests run eagerly.  A toolchain fix would let the CPU compile
  checks run.

## Stage 5 as planned

Stage 5 was done on 2026-09-15 from `stage5_prompt.md` beside this page.
The plan for it, kept for the record, was: Stage 5 completes `tests/test_mace4d.py` against the table in Section 5 of
the plan, and ends when every group passes on every available device.  The
plan for it, in order:

1. **Fill the unit groups from mbirjax's tests**: the prior weights, the
   model's `set_device_pool` and `devices`, construction (5 frames from the
   24-view model, `view_slices[1] == slice(4, 12)`), and the parameters
   (defaults, round trip, `mace_prior_weight=1.5` refused when set).  These
   are cheap and need no reconstruction.
2. **Complete the reconstruction group** on the one 3-frame run Stage 4
   already makes: `stop_threshold_change_pct=1e9` stops after one
   iteration, a supplied `init_recon` is recorded as provided by the caller,
   and the cache group (absent file, wrong shape with one warning containing
   `invalid`, valid file as float32) as direct calls of the helper.
3. **The one-frame equality gate**, the whole-loop check.  `num_frames=1`,
   `dejitter=False`, agent weights `[1/2, 1/6, 1/6, 1/6]`, the denoiser sigma set
   to `sigma_prox` times the square root of 3/2, `sigma_x` pinned to the
   value `recon` used, 30 iterations, within 1 percent NRMSE of `recon` on
   the frame's views.  Three rules from the earlier stages apply.  The
   reference must be a converged `recon`, 100 iterations, with its own
   convergence measured against 200.  The start must be more than the
   tolerance away, a 30-iteration `recon`, so that the loop is seen to move.
   The problem must be one on which the loop converges: the 64-view
   Shepp-Logan scan at 32 by 32 by 4 of the warm-start measurement does,
   and the 24-view test problem does not.  The denoiser parameters are
   pinned through a test-only subclass (decision 11).  Budget: under two
   minutes on the CPU; the Stage 3 gate takes 55 seconds alone.
4. **Keep the test time short.**  One 3-frame run and one gate run are
   the two expensive tests; every other check hangs off them or is a unit
   test.
5. **Close the stage.**  Run the whole suite, update the status row,
   update this page, and write a status report.  Then Stage 6, the
   documentation and the lazy export of `MACE4DModel`.

## Repositories

mbirtorch, branch `mace_4d_dev`; the local branch and the remote are both
at `2ef9b82`, which estimates the denoiser strength from the filtered initial
image; `ef13956` holds the Stage 5 tests in `tests/test_mace4d.py`;
`d7112b8` holds the fixes after the Stage 4 panel in `mbirtorch/mace4d.py`,
`mbirtorch/mace.py`, and `tests/test_mace4d.py`.
Stage 4 is two commits: `7893f22` moves the filter utilities and changes
five files, `mbirtorch/__init__.py`, `mbirtorch/mace.py`,
`mbirtorch/mace4d.py`, `tests/test_mace.py`, and `tests/test_mace4d.py`;
`8304b25` adds the class and its tests to `mbirtorch/mace4d.py` and
`tests/test_mace4d.py`.  Stage 3 is two commits:
`42e0991` holds `mbirtorch/mace.py`, `tests/test_mace.py`,
`mbirtorch/__init__.py`, `experiments/drunet/mace.py`, and
`experiments/drunet/agents.py`; `2763640` holds a four-line change to
`mbirtorch/denoising.py` that keys the compiled stack sweep by denoiser
object, kept separate because it answers a Stage 0 panel finding that Greg
has not ruled on.  The whole suite after Stage 4: 1104 passed, 143 skipped,
and the geometry viewer's wall-clock gate, which fails under the load of
eight workers and passes alone at half its limit.

mbirtorch_plans, branch `main`; the local branch and the remote are both
at `d23fad2`.  The
mace4d files of this work are unstaged: the edits to
`plans/mace4d/plan.md` and to the two nn_priors findings pages, and
the untracked records, prompts, panel pages, scripts, and output files under
`plans/mace4d/` and `plans/mace4d/experiments/`.  Ziyun
has no push rights to this repository.
