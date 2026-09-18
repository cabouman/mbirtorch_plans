# Runtime detector offsets: can `det_channel_offset` and `det_row_offset` stop triggering a recompile?

Answer: yes. The change is small and it works. The prototype is seven files, 92
added lines and 14 removed. Every measured value is bit-identical to the code
it replaces, and the targeted and full test suites pass unchanged.

Measured on this Mac (Darwin 25.5.0, Apple silicon), torch 2.13.0, python 3.11,
`torch.set_num_threads(1)`, models pinned to CPU. Triton is not installed in
this environment, so every measurement below exercises the torch bodies. The
default device here is MPS, whose queue is asynchronous; every timing below
pins the model to CPU and consumes the result, so `perf_counter` measures the
projection and not the dispatch.

All paths are absolute. Line numbers are as of the start of this work.

---

## Part 1 -- the mechanism

### 1.1 What the flag does, step by step

`Param` carries the flag:

`/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch/mbirtorch/_utils.py:47-53`
holds the dataclass, and lines 63-64 gave both offsets `recompile_flag=True`.

`ParameterHandler.set_params`
(`/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch/mbirtorch/parameter_handler.py:260-345`)
reads the flag of each key it is given (line 306), and if any flagged key is
present it sets one boolean `recompile` (line 316). At the end of the method
(lines 341-342) that boolean does exactly one thing: it calls
`self.refresh_device_bindings()`. Nothing else in the package reads the flag.

`TomographyModel.refresh_device_bindings`
(`/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch/mbirtorch/tomography_model.py:874-889`)
rebuilds the two `Placement` objects from the current `sinogram_shape` and
`recon_shape`, checks that no device would be left idle, drops the device
caches (`prox_data`, `_dc_damping_cache`), and calls `create_projectors()`.

`TomographyModel.create_projectors`
(`.../tomography_model.py:257-258`) constructs a new `Projectors`
(`/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch/mbirtorch/projectors.py:400-445`),
which asks the geometry for its bodies, binds one compiled instance per device
through `maybe_compile`, and re-places the view parameters on every device.

### 1.2 The three possible costs, separated

**(a) Recomputing derived geometry tensors or Python constants -- yes, and it is
cheap.** `Projectors.__init__` rebuilds the per-device view-parameter tensors
(`.../projectors.py:433-445`) and the two `Placement` objects. Nothing derived
from the OFFSETS is recomputed, because nothing is derived from them: the psf
radii come from the pitches and voxel sizes only
(`/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch/mbirtorch/parallel_beam.py:172-179`
and `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch/mbirtorch/cone_beam.py:530-560`),
and the cone damping profile's cache key lists `recon_slice_offset` but neither
detector offset (`.../cone_beam.py:440-448`). Measured: `set_params` with the
rebuild ran in 0.02-0.32 ms across every cell below, and a bare
`create_projectors()` in 0.887 ms.

**(b) Rebuilding the torch.compile wrappers -- NO, this does not happen.** The
compiled callables live in a MODULE-LEVEL cache keyed by (function, device
index) (`.../projectors.py:41` and `156-171`), so a rebuilt `Projectors` gets the
same wrapper objects back, complete with their `seen_keys` set. Measured
directly: after `create_projectors()` the cache key list was unchanged and the
next forward took 0.336 ms.

**(c) Triton kernel re-JIT -- does not arise for the offsets.** Both Triton
parallel bodies compute the geometry in EAGER python before the launch and pass
the kernel only tensors:
`/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch/mbirtorch/triton_parallel.py:633-635`
(forward) and `:299-302` (back) call `_parallel_hfan_math`, and the offset never
crosses into the kernel. The cone bodies do the same at
`/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch/mbirtorch/triton_cone.py:391-397`
and `:680-686`. So neither offset is a `tl.constexpr` and neither is even a
runtime scalar kernel argument -- they are consumed before the launch.

**(b') The cost that actually exists: ONE dynamo retrace per body.** With the
guard log on, the failing guard is named exactly:

```
Recompiling function _parallel_forward_view_batch in .../mbirtorch/parallel_beam.py:56
    triggered by the following guard failure(s):
    - 0/0: det_channel_offset == 0.0
      # n_p = (x + det_channel_offset) / delta_det_channel + det_center_channel
      # .../mbirtorch/parallel_beam.py:44 in _parallel_hfan_math
```

Dynamo specialized the first trace on the literal value, so the first CHANGE
away from the build value fails that guard and the body is traced and codegen'd
again. After that one retrace it does not happen again, because
`torch._dynamo.config.specialize_float` is **False** in this torch, so the
retrace promotes the float to a symbolic float that fits every later value.

### 1.3 How the offsets reach the kernels

They are ordinary Python floats passed as keyword arguments, rebuilt from the
live parameters on EVERY projection call -- not captured in a closure, not
frozen at build time. `Projectors.sparse_forward_project_view_range` calls
`m._view_batch_args()` at `.../projectors.py:561` and the back loop at `:621`;
the geometries build the dict at
`.../parallel_beam.py:278-291` and `.../cone_beam.py:397-414`.

Inside the compiled bodies each offset appears in exactly one arithmetic
expression, on data:

- parallel: `n_p = (x + det_channel_offset) / delta_det_channel + det_center_channel`
  (`.../parallel_beam.py:44`)
- cone channel: same shape of expression at `.../cone_beam.py:101`
- cone row: `m0 = (pixel_mag * z_at_slice_0 + det_row_offset) / delta_det_row + det_center_row`
  (`.../cone_beam.py:156`)

**Neither offset takes part in any shape or index-range computation.** The
confirmation is concrete: the tap loops run over `range(-psf_radius,
psf_radius + 1)` (`.../cone_beam.py:290` and the fan in
`/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch/mbirtorch/horizontal_fan.py`),
and `psf_radius` comes only from the detector pitches and voxel sizes. The
detector indices a voxel lands on are `torch.round(n_p)` clamped into range --
data, not shapes. So dynamo never needs either offset as a constant.

One place does treat a changed offset as a new key without needing to:
`_shape_key` (`.../projectors.py:107-118`) stringifies any non-tensor argument,
so each distinct float value is a new key, and each first-sight key takes the
process-wide compile lock once (`.../projectors.py:161`). That costs one lock
acquisition and one set entry per new value, not a compile.

### 1.4 Dynamo's treatment of Python floats in this torch

`torch._dynamo.config.specialize_float` is **False** by default here (torch
2.13.0). A five-line toy confirms the resulting pattern -- specialize once,
generalize on the second value, then never again:

| offset passed | wall time | dynamo frames |
|---|---:|---:|
| 0.00 | 2973.8 ms | 2 |
| 0.25 | 476.3 ms | 3 |
| 0.50 | 0.058 ms | 3 |
| 0.75 | 0.025 ms | 3 |
| 1.00 | 0.026 ms | 3 |
| 1.25 | 0.026 ms | 3 |

So today's code is already NOT recompiling per value. It pays one retrace on
the first change, and the whole exposure is that this behaviour is a torch
DEFAULT, not something the package states. On a torch with
`specialize_float=True` (the default before torch 2.7) every distinct value
would retrace, and after `_RECOMPILE_LIMIT_FLOOR` (64,
`.../projectors.py:70`) variants the body would fall back to eager for good.

### 1.5 The lead developer's question: is the flag still doing anything real?

**(a) What the rebuild does that a plain attribute change would miss.** Only two
things, and neither involves the compiled code. First, the two `Placement`
objects are rebuilt from the current shapes, which is what tells the sharded
drivers how many views and slices each device owns. Second, `prox_data` and
`_dc_damping_cache` are dropped. Everything else the rebuild touches is either
re-derived per call anyway (`_view_batch_args`) or served from a module-level
cache that the rebuild cannot invalidate (the compiled wrappers). No geometry
value is captured in a closure or a tuple at build time; the mbirjax design the
flags came from does not survive in this code.

**(b) Would dynamo's guards catch a changed geometry value on their own?**
For a value that is read per call and passed in as an argument, yes -- and the
cost is exactly one guarded retrace against the per-function budget (measured
below), not a silent stale value. That covers every scalar in
`_view_batch_args`. It does NOT cover the placements, which no guard can see.

**(c) The experiment.** Flag turned off in the live model (which is the same
dictionary `set_params` reads), value changed, forward projection compared
against a model built with that value from the start:

| model | parameter | new value | error | result vs a freshly built model | dynamo frames added |
|---|---|---|---|---|---:|
| parallel | `det_channel_offset` | 0.5 | none | identical (0.0 abs) | 1 |
| cone | `det_row_offset` | 0.5 | none | identical (0.0 abs) | 1 |
| cone | `det_channel_offset` | 0.5 | none | identical (0.0 abs) | 1 |
| parallel | `recon_shape` | (16, 16, 24) | none | identical (0.0 abs) | 1 |

On ONE device even `recon_shape` comes out right, because `_view_batch_args`
re-reads the shape per call. The rebuild's real job shows up only when the
model is sharded. Cone on two virtual CPU devices, recon slice count changed
from 24 to 18:

| flag | recon placement `axis_len` after the change | `back_project` output shape | expected |
|---|---:|---|---|
| True (today) | 18 | (32, 32, 18) | (32, 32, 18) |
| False | 24 | **(32, 32, 24)** | (32, 32, 18) |

With the flag off the back projection came back the wrong size, with no error
and no warning. That is the silent-stale-value case, and it is why the flag
must stay on shape changers.

**Which parameters need what.**

- **Genuinely need the rebuild**: `sinogram_shape`, `recon_shape`. They set the
  placement axis lengths, and a stale placement is silently wrong on the
  sharded path (measured above). Anything that changes the device layout goes
  through `configure_devices`, which does its own rebuild.
- **Could drop the flag today, with no other change**: `delta_det_channel`,
  `delta_det_row`, `delta_voxel`, `voxel_row_aspect`, `voxel_slice_aspect`.
  They are read per call and passed as arguments, so a change is picked up
  correctly and costs at most one guarded retrace. The caution is that they DO
  feed integer values -- the psf radii, and the cone transient width -- which
  are recomputed per call from the same live parameters, so they stay
  consistent; but a psf radius change legitimately produces a different graph
  and should retrace.
- **Could drop the flag and also stop costing a retrace, once they are runtime
  inputs**: `det_channel_offset`, `det_row_offset` (done here), and by the same
  argument `recon_slice_offset`, `source_detector_dist`, `source_iso_dist` --
  see the "remaining work" section for the one caveat on the distances.

---

## Part 2 -- the prototype

### 2.1 The change

Three moves, chosen because Part 1 showed the offsets are pure data:

1. Both offsets lose the recompile flag
   (`mbirtorch/_utils.py`). Nothing else in the model is derived from them, so
   no rebuild is needed.
2. Each geometry hands its offsets to the bodies as **0-dimensional float32
   tensors on the calling device** instead of Python floats. torch.compile
   guards a tensor by dtype, device, shape and layout -- all identical for
   every value -- so no guard can fail. The tensors are memoized per (name,
   value, device) by a new `TomographyModel._runtime_scalar`, so the steady
   state allocates nothing per call.
3. The two driver loops pass the device of the view parameters they are about
   to use, so each device of a sharded run builds the scalar where its own
   arrays live. `_view_batch_args` gained an optional `device=None`; every
   other caller (the memory ledger, the kernel-availability self-check, fifteen
   test sites) keeps working unchanged.

The public API is untouched: `set_params(det_channel_offset=...)` still works
and the projectors read the current value at call time, as they always did.

**The subtle part, and it bit once.** The first version memoized on (value,
device) only. Both cone offsets default to 0.0, so both arguments got the SAME
tensor object, and dynamo installed an ALIASING guard:

```
Recompiling function _cone_forward_view_batch ...
    - 0/0: det_channel_offset is det_row_offset
      # m0 = (pixel_mag * z_at_slice_0 + det_row_offset) / delta_det_row \
      # .../mbirtorch/cone_beam.py:156 in _cone_vertical_affine
```

Changing one offset broke the identity and the body retraced anyway. Putting
the parameter NAME in the memo key fixes it; the reasoning is written into the
`_runtime_scalar` docstring so nobody removes it later.

### 2.2 Recompiles and wall time over five distinct `det_channel_offset` values

Each row: `set_params`, then one forward and one back projection. Model warmed
first, so the table is about changing the offset, not about the first compile.
Sinogram (48, 24, 32); parallel recon (32, 32, 24), cone recon (32, 32, 24).
Inductor's on-disk cache was empty at the start of each run.

**Parallel beam**

| offset | before: total | before: retraces | after: total | after: retraces |
|---:|---:|---:|---:|---:|
| 0.00 | 7.6 ms | 0 | 7.9 ms | 0 |
| 0.25 | **2140.5 ms** | **2** | 7.6 ms | 0 |
| 0.50 | 7.8 ms | 0 | 7.5 ms | 0 |
| 0.75 | 7.3 ms | 0 | 6.2 ms | 0 |
| 1.00 | 6.8 ms | 0 | 6.6 ms | 0 |
| **five-value total** | **2170.0 ms** | **2** | **35.7 ms** | **0** |

**Cone beam**

| offset | before: total | before: retraces | after: total | after: retraces |
|---:|---:|---:|---:|---:|
| 0.00 | 50.0 ms | 0 | 48.1 ms | 0 |
| 0.25 | **7266.6 ms** | **2** | 46.2 ms | 0 |
| 0.50 | 49.3 ms | 0 | 47.1 ms | 0 |
| 0.75 | 47.0 ms | 0 | 46.7 ms | 0 |
| 1.00 | 47.1 ms | 0 | 47.2 ms | 0 |
| **five-value total** | **7459.9 ms** | **2** | **235.2 ms** | **0** |

Parallel is 61x faster over the sweep, cone 32x. The whole saving is the single
retrace of each body; the row at 0.25 is where it lands. `set_params` itself
went from 0.02-0.30 ms to 0.00-0.04 ms, because it no longer rebuilds the
projectors.

The retrace is much cheaper when inductor's on-disk cache already holds the
graph: measured 40.4 ms for the parallel forward on a warm cache against
1049 ms cold. So the cost a user actually feels ranges from tens of
milliseconds to several seconds per body, depending on the cache.

One side effect worth noting: the cone warm-up compiled 5 dynamo frames after
the change against 7 before. Passing a float under `specialize_float=False`
makes dynamo build an extra wrapper frame per float input; passing a tensor
does not.

### 2.3 Value parity

Forward and back projections at three offsets (0.0, 0.37, -0.62), both
geometries, compared array by array against the pre-change worktree state
(recorded first, then restored with `git stash`), same seeds, same CPU device:

| array | max abs difference | relative | bitwise equal |
|---|---:|---:|:--:|
| all 12 arrays (parallel and cone, forward and back, three offsets each) | 0.000e+00 | 0.000e+00 | yes |

Worst relative difference over the twelve arrays: 0.000e+00. **Bit-identical**,
not merely within 1e-6. That answers the float32-versus-Python-float worry
directly: a Python float added to a float32 tensor is already rounded to
float32 by torch's wrapped-number rule, so replacing it with a float32 scalar
tensor changes no arithmetic.

The sharded path was checked separately, on one and two devices, sweeping five
offsets against a freshly built reference model each time:

| geometry | devices | worst relative difference | retraces |
|---|---:|---:|---:|
| parallel | 1 | 0.000e+00 | 0 |
| parallel | 2 | 0.000e+00 | 0 |
| cone | 1 | 0.000e+00 | 0 |
| cone | 2 | 0.000e+00 | 0 |

(The cone rows swept `det_row_offset` and `det_channel_offset` together, in
opposite directions, which is also the case that exposed the aliasing guard.)

### 2.4 Tests

Targeted set: `tests/test_adjoint.py`, `tests/test_triton_parallel.py`,
`tests/test_triton_cone.py`, `tests/test_view_batching.py`,
`tests/test_device_policy.py`, `tests/test_params_and_paths.py`,
`tests/test_hdf5_family.py`, `tests/test_sharding.py`,
`tests/test_memory_ledger.py`, `tests/test_cone.py`,
`tests/test_multiaxis.py`, `tests/test_translation.py`.

A grep of `tests/` for the two offset names points at
`tests/test_triton_parallel.py:67-83` (the kernel value self-check
deliberately drives `n_p` off the channel grid with an offset),
`tests/test_device_policy.py:992`, `tests/test_hdf5_family.py:103-114`, and the
preprocessing tests, all of which are in the set.

| run | result |
|---|---|
| targeted set, before | 340 passed, 83 skipped, 29 deselected, in 60.4 s |
| targeted set, after | 340 passed, 83 skipped, 29 deselected, in 59.7 s |
| full suite (`pytest tests/ -q`), after | 669 passed, 96 skipped, 93 deselected, in 151.8 s |

No failures, and no test changed its outcome. The full suite was run once,
after the change, and exited 0.

### 2.5 Per-call overhead

256 views, 256x256 recon (51040 pixels after the ROR mask), CPU, one thread,
best of five, result consumed each time:

| cell | before | after | change |
|---|---:|---:|---:|
| 1 detector row, offset at the build value | 138.07 ms | 141.21 ms | +2.3% |
| 1 detector row, offset changed to 0.37 | 140.10 ms | 139.80 ms | -0.2% |
| 8 detector rows, offset at the build value | 466.57 ms | 468.45 ms | +0.4% |
| 8 detector rows, offset changed to 0.37 | 467.80 ms | 474.16 ms | +1.4% |

Spread within a single run was 1-3%, so none of these differences is
distinguishable from run-to-run noise. The output sums were identical to the
last digit in every cell. **No measurable per-call cost.** That is what the
memo buys: in the steady state the scalar tensor is looked up in a dict, not
allocated.

### 2.6 Diff size

```
 mbirtorch/_utils.py             | 10 ++++++--
 mbirtorch/cone_beam.py          | 10 ++++++--
 mbirtorch/multiaxis_parallel.py |  5 +++-
 mbirtorch/parallel_beam.py      |  8 ++++--
 mbirtorch/projectors.py         | 12 ++++++---
 mbirtorch/tomography_model.py   | 56 +++++++++++++++++++++++++++++++++++++++--
 mbirtorch/translation_model.py  |  5 +++-
 7 files changed, 92 insertions(+), 14 deletions(-)
```

The full patch is saved beside this file as `runtime_offsets.diff` (221 lines).
Most of the 56 lines in `tomography_model.py` are the `_runtime_scalar`
docstring.

---

## Part 3 -- what is left, and the risks

### Remaining work

**Cone beam is done** -- it was in scope if time allowed, and it is included
above with its own measurements. Both cone offsets are runtime inputs.

**Multi-axis parallel and translation are NOT done.** Their
`_view_batch_args` now accepts the `device` argument the drivers pass, and
ignores it: they still hand their offsets over as Python floats
(`mbirtorch/multiaxis_parallel.py:271-283`,
`mbirtorch/translation_model.py:287-299`). Because the two offsets lost their
recompile flag globally, those geometries no longer rebuild their projectors on
an offset change -- which is correct, they do not need to -- but they still pay
the one dynamo retrace per body on the first change. Making them match is the
same two-line edit per geometry that parallel and cone got. It was left out
only to keep the prototype small; it should be done before this ships, so the
behaviour is uniform.

**The Triton kernels need no work at all**, and that is worth stating plainly
rather than leaving as a to-do. Both offsets are consumed in eager python
before any launch, in `_parallel_hfan_math`, `_cone_horizontal_data` and
`_cone_vertical_affine`. No kernel takes either offset as an argument, as a
`tl.constexpr` or otherwise, so no Triton cache key changes when an offset
changes. The prototype's tensors flow through those same eager helpers
unchanged, which is why `tests/test_triton_parallel.py` and
`tests/test_triton_cone.py` pass without modification -- though note that
Triton is not installed here, so those files' kernel tests skipped and the
kernel path itself has NOT been exercised end to end on this machine. It should
be run once on a CUDA box before shipping.

**Not attempted**: `recon_slice_offset` and the source distances (below).

### Risks

**1. The `_utils.py` header forbids this edit.** The file opens with "The names,
values, and recompile flags of those defaults are fixed by an external
reference and must not be changed here," and the same warning sits above the
dictionary. The prototype changes two of those flags anyway, because the task
asked for it. Whoever owns that external reference has to agree, or the flags
have to move somewhere the reference does not govern. This is the one thing in
the change that is a decision rather than a measurement.

**2. torch.compile guards on tensor metadata.** Swapping a float for a tensor
trades a value guard for metadata guards on dtype, device, shape and layout.
Those are constant here by construction, so nothing can fail -- but the
aliasing guard the prototype hit is proof that the guard set is wider than
"dtype and shape", and that two scalars sharing one object is enough to
reintroduce a retrace. The name in the memo key is what prevents it, and a test
should pin that: change one cone offset, leave the other, assert zero new
dynamo frames. No such test exists yet.

**3. The memo holds device tensors keyed by value.** It is capped at 256
entries and emptied wholesale when full
(`TomographyModel._RUNTIME_SCALAR_CACHE_MAX`). A caller sweeping thousands of
offsets on a GPU therefore allocates at most 256 four-byte tensors before the
memo resets. The reset drops references, so nothing leaks, but it does mean the
call right after a reset allocates again. On a fixed offset -- every production
recon -- the memo holds one entry per device and never resets.

**4. Numerics.** No risk was found, and it was checked rather than assumed: all
twelve parity arrays came back bit-identical, in both geometries and both
directions. The reason is that torch already rounds a Python float to the
tensor's dtype before the operation, so float32-scalar-tensor arithmetic and
Python-float arithmetic are the same arithmetic here. This would NOT hold for a
scalar that torch promotes differently -- for example a float64 scalar tensor,
which would promote the whole expression to float64.

**5. Triton scalar arguments and kernel caching.** Not exercised on this
machine (no Triton). The argument that it cannot matter is structural -- the
offsets never reach a kernel -- but "structural" is not "measured", and a CUDA
run of `tests/test_triton_parallel.py` and `tests/test_triton_cone.py` should
confirm it.

**6. Multi-device correctness.** Checked on two virtual CPU devices, where the
values came back bit-identical to a freshly built model with zero retraces. Two
CPU devices exercise the placement and per-device argument plumbing but NOT a
real cross-device copy, so a two-GPU run is still owed.

**7. Dropping the flag removes a rebuild that something else might have been
relying on.** The rebuild also dropped `prox_data` and `_dc_damping_cache`. The
damping cache's key does not include either detector offset
(`.../cone_beam.py:440-448`), and `prox_data` holds the proximal-map input, not
geometry -- so neither is stale after an offset change. This was read, not
tested; a plug-and-play loop that changes an offset mid-run would be the case to
watch.

### Could `recon_slice_offset` and the source distances follow the same route?

**`recon_slice_offset`: yes, and it is the easiest next one.** It appears in
`_cone_vertical_affine` (`.../cone_beam.py:151`) and
`_multiaxis_vertical_terms` (`mbirtorch/multiaxis_parallel.py:78`) in exactly
the same shape of expression as `det_row_offset` -- added to a coordinate, then
divided by a pitch. It takes part in no shape computation. One caveat, and it is
real: unlike the detector offsets, `recon_slice_offset` DOES appear in the cone
damping profile's cache key (`.../cone_beam.py:446`), so it needs that cache
invalidated when it changes. That is a small explicit hook, not a rebuild.

**The source distances: probably, but not for free.** `source_detector_dist`
reaches the bodies and is already converted to a tensor inside them
(`.../cone_beam.py:93-95`, `:280`), so the tensor form is natural. But unlike
the offsets it feeds `get_psf_radii` (`.../cone_beam.py:530-560`), which
returns INTEGER tap counts that set the length of the `range(-psf_radius,
psf_radius + 1)` loops the bodies unroll. Those integers must stay Python
constants -- they are the graph's shape. So the distances can become runtime
inputs only in their arithmetic role, while the psf radii they imply stay
build-time constants, and a distance change large enough to move a radius must
still retrace. That is correct behaviour, not a defect, but it means the
distances are a genuinely different case from the offsets and need their own
measurement before anyone assumes they are free.

`magnification` follows the distances, since it is derived from them.

---

## How to reproduce

Scripts are in this directory. Run each from the worktree root with
`PYTHONPATH=.` and `/Users/gbuzzard/miniforge3/envs/mbirtorch/bin/python`.

- `toy_float.py` -- the five-line float-specialization demonstration (1.4)
- `diagnose.py` -- names the parallel guard and splits the rebuild's cost (1.2)
- `diag_cone.py` -- names the cone guard, including the aliasing one (2.1)
- `measure.py --what sweep` -- the five-value tables (2.2); set `MEASURE_TAG`
  and point `TORCHINDUCTOR_CACHE_DIR` at an empty directory for a cold run
- `measure.py --what parity --out <file.npz>` then `compare_parity.py
  <before.npz> <after.npz>` -- the parity table (2.3)
- `percall.py` -- the per-call bench (2.5)
- `flag_experiment.py` and `flag_sharded2.py` -- the flag experiments (1.5)
- `check_sharded_after.py` -- the multi-device check (2.3)
