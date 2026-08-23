# Porting mbirjax_metrics to mbirtorch_metrics: implementation plan

Status: accepted 2026-08-23.  Greg accepted the recommendations for decisions 1
through 4.  Decision 5 is settled: the repository is `cabouman/mbirtorch_metrics`,
and it is cloned to `Research/mbirtorch_metrics`.  Implementation is under way,
one increment at a time, with a commit after each.

## Executive summary

The goal is a new repository, `mbirtorch_metrics`, that records mbirtorch and
nothing else.  It takes its structure from `mbirjax_metrics` at commit
`e37bc93e` (2026-08-08), which is the last commit before the torch nightly was
added to that repository.  At that commit the repository has one nightly, one
measurement engine, one dashboard, and one set of entry points.  The new
repository keeps that shape and puts mbirtorch where mbirjax was.  This is open
item G3.

The port is mostly assembly rather than new code, because the torch pieces
already exist.  `mbirjax_metrics` at its current tip carries a full torch
nightly built as a sibling of the jax one: `run_torch_regression.sh`,
`torch_regression.env`, `lib_torch_env.sh`, `torch_run_configs.env`, the torch
enable/disable/status scripts, and the measurement layer
`torch_backend_writer.py`.  Each of those becomes the single, unqualified file
of its kind in the new repository.  For example `run_torch_regression.sh`
becomes `run_regression.sh`.

Three pieces do need real work.  The first is merging the torch measurement
layer into one engine file, described in section 4.  The second is porting
`add_run.sh`, the manual backfill entry point, which has no torch sibling today.
The third is the data migration, which needs a careful rename if we adopt
decision 1 below.

The measured data moves but the mbirjax data does not.  There are 120 tracked
result files under `results/gpu-torch/` and `results/cpu-torch/` (3.2 MB) and
five branch markers under `state/gpu-torch/` and `state/cpu-torch/`.  Those move
to the new repository.  The jax results, the jax state, and the
`experiments/` tree stay behind.

## 1. Decisions needed before coding

### Decision 1: the platform keys

The measured data files are named and keyed `gpu-torch` and `cpu-torch`.  That
qualifier exists so torch rows can sit beside jax rows in one repository.  The
new repository has only torch rows, so the qualifier carries no information
there.

**Recommendation: rename the keys to `gpu` and `cpu`.**  The rename lets the
dashboard drop the two-backend machinery it grew on 2026-08-04.  That machinery
is three things: the `PLAT_IS_TORCH` test, the second history row, and the
family restriction in the correctness analyzer.  The rename also fixes a label
that would otherwise read oddly, `device_label: GPU-TORCH (NVIDIA H100 80GB
HBM3)`.

The rename touches four places in each migrated file: the directory name, the
file name, the top-level `platform:` field, and the per-cell `platform:` fields.
It also touches the `sizes:` keys recorded in each run's config block, which the
gate reads back by platform key.  Increment 2 does this with a script and
verifies it by building the dashboard before and after.

The cost of not renaming is a permanent `-torch` suffix on every path in a
torch-only repository, plus dead branching in the dashboard.

### Decision 2: one engine file or two layers

Today the torch measurement layer, `torch_backend_writer.py`, sits on top of the
jax engine's decision layer in `performance_tracking.py`.  That split exists for
one reason: two backends in one repository must share one gate model, or the two
gates drift apart.  The new repository has one backend, so the reason is gone.

**Recommendation: merge them into one `performance_tracking.py`.**  The merge
also removes duplication that exists today.  `torch_backend_writer.py` carries
its own `time_op`, `run_measure_loop`, `peak_memory_mb`, and `git_provenance`,
each shadowing a function of the same name in `scaling_common.py`.  In a
torch-only repository each of those collapses to one definition.

The alternative is to keep two files.  That is less work now and leaves a
structure that no longer has a reason.

### Decision 3: environment variable and knob names

The torch scripts qualify their names to stay disjoint from the jax ones.
Examples are `REG_TORCH_LIB_ROOT`, `TORCH_TRACKED_BRANCHES`, and
`TORCH_SLURM_ACCOUNT`.

**Recommendation: drop the qualifier throughout**, so `REG_TORCH_LIB_ROOT`
becomes `REG_LIB_ROOT` and `TORCH_TRACKED_BRANCHES` becomes `TRACKED_BRANCHES`.
mbirtorch's own variables keep their names, so `MBIRTORCH_NUM_DEVICES` and
`MBIRTORCH_MEMORY_CALIBRATION` are unchanged.

### Decision 4: the dependency canary

The jax nightly runs a dependency canary, and it is switched on
(`DEP_CANARY_ENABLED=1`).  The canary fires when PyPI ships a jax newer than the
one last measured.  It then upgrades jax in the shared environment and
re-measures a fixed branch.  A performance change caused by the new jax is
therefore attributed to it, rather than merely reported.  The torch nightly has
no equivalent.  It has a different feature, the dependency-watch watchdog, which
checks the Python test matrix against the versions torch publishes wheels for.

A torch canary is worth having, because mbirtorch declares `torch>=2.13` with no
upper bound, so any torch release can change measured performance.  Building it
means writing `check_torch_release.py` against the PyPI API and adapting the
canary block in the wrapper.

**Recommendation: port the canary in a later increment, switched off at
first.**  Increment 6 covers it.  Say if you would rather drop it entirely.

### Decision 5: the GitHub repository (settled)

The repository is `github.com/cabouman/mbirtorch_metrics`.  Note the owner: it is
`cabouman`, not `gbuzzard`, which is where `mbirjax_metrics` lives.  Two values
follow from that.  The push URL in `regression.env` is
`https://github.com/cabouman/mbirtorch_metrics.git`.  The published dashboard is
at `https://cabouman.github.io/mbirtorch_metrics/`.

Greg created the repository and cloned it to `Research/mbirtorch_metrics`.  It
starts with a `main` branch holding a LICENSE and nothing else.  No history is
imported from `mbirjax_metrics`.  Each migrated result file carries its own
dates, so the per-night commit history there adds nothing.

## 2. What the new repository contains

The layout follows the reference commit exactly.

```
README.md                     rewritten for mbirtorch
LICENSE, .gitignore           copied, with the jax-only ignore lines dropped
.github/workflows/pages.yml   copied; only the repository name changes
.claude/dashboard_orientation.md   copied and updated

action_scripts/               entry points and the run knobs
  run_configs.env             from torch_run_configs.env, names de-qualified
  add_run.sh                  ported from the jax version (new work)
  run_one_night.sh            from run_one_torch_night.sh
  enable_nightly.sh, disable_nightly.sh, status_nightly.sh
  build_dashboard.sh, clear_correctness.sh, create_token.sh
  README.md

tooling/regression/           the nightly wrapper
  run_regression.sh           from run_torch_regression.sh
  regression.env              from torch_regression.env
  lib_env.sh                  from lib_torch_env.sh
  lib_mac_entry.sh            copied from the current tip (see section 5)
  enable_nightly.sh, disable_nightly.sh, status_nightly.sh
  nightly_regression.slurm, com.mbirtorch.regression.plist
  cluster_preamble.sh.example, sbatch_submit.sh, recent_runs.py
  README.md

tooling/scaling_tests/        the measurement engine
  scaling_common.py           the harness, jax parts removed
  performance_tracking.py     decision layer plus the torch measurement layer
  run_nightly.py              nightly entry point
  run_performance_local.py    manual launcher
  measure_one_cell.py         single-cell reproducer
  regression_to_table.py      the per-run browsable table
  test_gate.py                gate unit tests

tooling/dashboard/            unchanged in role
  build_dashboard.py, clear_correctness.py
  dashboard.js, dashboard.css, template.html, vendor/
  README.md

results/<plat>/<branch>/      the migrated torch time series
state/<plat>/<branch>         the migrated branch markers
state/README.md
```

## 3. What is not migrated

Four groups stay in `mbirjax_metrics`.

The jax measured data stays.  That is `results/gpu/`, `results/cpu/`,
`state/gpu/`, and `state/cpu/`.

The experiment trees stay.  `experiments/partition_sequence/` is a jax
partition-sequence study with 175 files, and `experiments/profiling/` is a
jax and XLA profiling toolkit with 20 files that reads HLO and runs Nsight
Compute.  Neither has a torch counterpart.  `dev_scripts/` exists only to serve
those two trees, so it stays as well.

The jax release watch stays.  `check_jax_release.py` and the `JAX_LAST_REVIEWED`
knob have no meaning in the new repository.  A torch counterpart is decision 4.

The two annotation files start empty.  `results/annotations.yaml` has no torch
entries today, so the new file starts with its header comment and no entries.
`results/correctness_acks.yaml` has `cleared_through: 2026-07-25`, which predates
the first torch run on 2026-08-05, so the new file starts with no watermark.

## 4. The engine merge, in detail

This section supports decision 2.  The reference commit's
`performance_tracking.py` is 1,403 lines and splits cleanly in two.

The decision half is backend-independent and carries over unchanged.  It holds
these pieces: `Config`, `fingerprint`, `_crop_to_true_shape`, `parse_size_label`,
`_file_tag`, `update_records`, `gate_run` with its `_gate_*` and `_compare_cell`
helpers, `_expected_cells`, `_find_priors`, `_apply_mem_window`, `_print_gate`,
`_print_summary`, `_assert_platform_matches_out_dir`, and the four input
generators `make_cylinders`, `make_sinogram`, `make_noisy_image`, and
`make_weights`.  The current torch writer already imports exactly this set.  That
import list is the evidence that the split is real.

The measurement half is jax-specific and is replaced.  It is `make_model`,
`make_indices`, `to_device`, the five op bodies, `build_partitions`, `run_vcd`,
`path_info`, `measure_cell_group`, `_probe_sharding_by_geom`, `worker_setup`,
`run_worker`, `_inline_setup`, `_git_provenance`, `_mbirjax_policy_defaults`,
`run`, and `main`.  `torch_backend_writer.py` supplies a torch counterpart for
each, plus three guards the jax engine does not have: the platform-claim check,
the per-row device pin and assertion, and the refusal to measure under
`MBIRTORCH_MEMORY_CALIBRATION`.

`scaling_common.py` loses its jax half.  Removed: `mbirjax_git_branch`,
`mbirjax_pkg_dir`, `toolchain_info`, `compile_cache_dir`, `compile_cache_env`,
`allocator_env`, `apply_env`, `build_worker_env`, `beta_root`, `gpus`,
`detect_platform`, `pick_devices`, `device_label`, `peak_memory_mb`,
`build_setup_result`, `print_setup_banner`, and `default_device_counts`.  The
torch equivalents move in from the writer.  Kept unchanged: `run_worker`,
`write_worker_result`, `is_oom`, `sample_gpu_health`, `throttled_gpus`,
`_GpuSampler`, `annotate_speedups`, `annotate_mem_fraction`, `save_yaml`,
`load_yaml`, `size_label`, `installed_packages`, `pyproject_version`,
`uniform_env`, `gpu_topology`, and the two plotting functions.

`time_op`, `run_measure_loop`, and `peak_memory_mb` exist in both files today.
The torch definitions win, and the jax ones are deleted.

## 5. Features to take from after the reference commit

The reference commit predates 52 commits on `mbirjax_metrics`.  Most of those
commits are nightly data.  Reviewing the rest is the second half of this
session's task, and section 8 below covers it properly.  Two items are already
identified, because the port cannot be correct without them.

`lib_mac_entry.sh` must come across.  It landed after the reference commit and
fixes a silent failure of the macOS nightly.  A launchd process cannot read
anything under `~/Documents` because of macOS privacy protection, and the
metrics repository commonly lives there.  The failure is silent and cost 51
consecutive nights before it was diagnosed.  The fix keeps an entry clone
outside `~/Documents` and points the launchd agent at that clone.  The new
repository runs its `cpu` nightly on the Mac through launchd, so it needs this.

The `dashboard.js` change from after the reference commit must **not** come
across.  That change makes the dashboard open on the newest non-torch run, so
that active mbirtorch development does not own the landing view.  In a
torch-only repository the correct behavior is the reference commit's, which is
to open on the newest run of any platform.

## 6. Increments

Each increment ends with a check, and I stop for review before the next one.

**Increment 1: the repository skeleton.**  Create `Research/mbirtorch_metrics`,
run `git init`, and add `LICENSE`, `.gitignore`, `README.md`, and the empty
`results/` and `state/` trees with their READMEs.  Check: the tree exists and
`git status` lists only intended files.

**Increment 2: migrate the measured data.**  Copy the 120 result files and five
state markers, applying the decision 1 rename.  Check: build the dashboard from
the new repository, and build it from `mbirjax_metrics` with `ONLY_PLATFORMS`
set to the two torch keys.  The run count, the cell count, and the per-cell
times and memories must agree.

**Increment 3: the dashboard.**  Copy `tooling/dashboard/` from the reference
commit.  Remove the two-backend machinery.  Retarget the text, the title, and
the links to mbirtorch.  Check: the dashboard builds from the migrated data and
shows every run, and the correctness banner matches what the source repository
shows for the torch rows.

**Increment 4: the engine.**  Build `scaling_common.py` and
`performance_tracking.py` as described in section 4, then port `run_nightly.py`,
`run_performance_local.py`, `measure_one_cell.py`, `regression_to_table.py`, and
`test_gate.py`.  Check: `test_gate.py` passes; re-running
`regression_to_table.py` on a migrated run file reproduces the migrated
`_table.yaml`; and replaying the gate on a migrated night against its recorded
prior reproduces the recorded verdict.  The replay check needs no GPU.

**Increment 5: the nightly wrapper and the entry points.**  Port the wrapper,
its configuration files, the schedule scripts, and the `action_scripts/` entry
points, including the new `add_run.sh`.  Check: run the smoke path (`REG_SMOKE=1`) on the Mac and on a Gautschi
interactive session.  Then run a full pass with pushing switched off
(`REG_NO_PUSH=1`), and confirm it writes a result file and a state marker
locally.

**Increment 6: the dependency canary.**  Only if decision 4 says yes.  Write
`check_torch_release.py` and adapt the canary block, switched off.  Check: the
release check prints the expected line against the live PyPI index, and the
nightly is unchanged with the canary off.

**Increment 7: publish and cut over.**  Push to the new GitHub remote, set Pages
to build from Actions, and confirm the published dashboard.  Then run one forced
night (`REG_FORCE=1`) on the cluster as the end-to-end check.  Then disable the
torch nightly in `mbirjax_metrics` and decide what happens to its torch files
and rows.  Check: the published dashboard shows the forced night, and
`status_nightly.sh` reports the new schedule as active.

## 7. Risks

The migrated data and the new engine must agree on the cell coordinates.  The
gate looks up expected cells by platform key in the config block stored inside
each run file.  If the decision 1 rename misses those keys, the first new night
compares against a prior it cannot match, and every cell reads as new.  The
increment 2 check catches this, because the dashboard reads the same fields.

The first night after cutover re-measures whatever the migrated state markers do
not cover.  Migrating `state/` means the first scheduled night is a no-change
night, which is the quiet outcome.  The forced run in increment 7 is what
actually exercises the path.

Two nightlies must not write to one place during the cutover.  The torch nightly
in `mbirjax_metrics` has to be disabled before or at the same time as the new one
is enabled.  Otherwise both measure the same mbirtorch branches and the series
splits across two repositories.

## 8. The second task: reviewing the commits after the reference commit

After the port is working, review `e37bc93e..HEAD` on `mbirjax_metrics` for
features worth carrying forward.  The 52 commits break down as follows.  Most
are nightly data commits that touch only `results/` and `state/`.  The code
commits touch 29 files, of which 22 are the torch nightly itself and are already
covered by this plan.  The remainder to review is small: `lib_mac_entry.sh` and
the `enable_nightly.sh` change that uses it (section 5, already accepted), the
`recent_runs.py` column-width change, the `dashboard.js` landing-view change
(section 5, rejected), and the two README additions.

This review is a separate step, and I will write its result into this file when
the port is done.
