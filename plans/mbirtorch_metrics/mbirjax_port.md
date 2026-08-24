# Porting mbirjax_metrics to mbirtorch_metrics: implementation plan

Status: increments 1 through 6 are built and pushed.  The Mac is cut over.  The cluster and the
published page wait on two things only a repository admin or token owner can do.  See the results
section at the end.  Greg accepted the recommendations for decisions 1 through 4 on 2026-08-23.  Decision 5 is settled: the repository is `cabouman/mbirtorch_metrics`,
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

---

# What was built (2026-08-23)

Increments 1 through 6 are done, committed, and pushed to
`github.com/cabouman/mbirtorch_metrics`.  Increment 7, the cutover, is not started, because its
steps change running schedules and repository settings.  Section "The cutover, still to do" lists
them.

The repository holds 128 tracked data files and about 4,800 lines of tooling.  The measured data is
39 runs and 1,900 cells, covering 2026-08-05 through 2026-08-23 on three branches.

## The increments, with what verified each one

**Increment 1: the repository skeleton.**  `.gitignore`, `README.md`, and `state/README.md`.

**Increment 2: migrate the measured data.**  120 result files and five branch markers moved, with
the platform keys renamed from `gpu-torch` and `cpu-torch` to `gpu` and `cpu`.

Verification compared the migrated tree against the source, file by file.  The 120 files map one to
one.  Each migrated file differs from its source only by the four renamed tokens.  All 1,900 cells
compare identical on geometry, operator, size, device count, time, and memory.  Every run's platform
key is consistent with the `sizes` key in its own config block, which is what the gate reads back.

**Increment 3: the dashboard.**  The dashboard sources came from the reference commit.  The
two-backend machinery was removed, which is the `PLAT_IS_TORCH` test, the second history row, and
the family restriction in the correctness analyzer.

Two schema keys were also renamed in the migrated data during this increment, because the dashboard
reader forces the choice.  `mbirjax_version` became `mbirtorch_version` and `jax_available` became
`torch_available`.  The dashboard now reads `toolchain.torch`, so every migrated run displays its
torch version and its package version; before the rename both were blank.

Verification built the dashboard from both repositories and compared the results.  The source build
was restricted to the two torch platform keys.  The two builds agree on every run, cell, gate
verdict, fingerprint, and correctness finding, after normalizing the platform token inside the
gate's references to prior files.  The published page was then loaded in a browser: 39 runs, no
console errors, and every panel populated.

**Increment 4: the measurement engine.**  `torch_backend_writer.py` and the decision half of
`performance_tracking.py` merged into one engine file of 1,458 lines.  `scaling_common.py` lost its
jax half and gained the torch memory and timing rulers, dropping from 1,190 lines to 549.  Three
functions that existed in both files collapsed to one definition each: `time_op`,
`run_measure_loop`, and `peak_memory_mb`.  `time_op` also lost an unused `model` argument.
Plotting, `correctness_metrics`, and `annotate_mem_fraction` were dropped as unused, which also
removes matplotlib from the harness dependencies.

Four checks verified the engine.  The gate unit tests pass, six of six.  The gate was replayed over
all 39 migrated runs against their recorded priors, and every verdict, hard finding, and soft
finding matched what the file recorded.  Every companion `_table.yaml` re-rendered byte-identically.
A smoke sweep ran end to end on CPU and wrote a run file, a record book, and a table, and the
dashboard parser read the result.

**Increment 5: the nightly wrapper and the entry points.**  The wrapper, its configuration, the
schedule scripts, and the `action_scripts/` entry points, including `add_run.sh`, which had no torch
version before.

Two guards were added that the source repository did not need.  The wrapper now refuses to reuse a
metrics clone whose origin is a different repository, and `status_nightly.sh` refuses to read runs
from one.  Both matter at cutover, because `~/.mbirtorch/regression/metrics` currently holds a clone
of `mbirjax_metrics`.  Without the first guard the wrapper would rebase this repository's work onto
another repository's history and then fail to find its own script.

Verification ran the wrapper on the Mac.  Fire-on-change read the migrated markers and skipped an
unchanged branch.  A forced run cloned the tip, installed it, measured one cell, and wrote the run
file under its commit-time name, matching the migrated file for the same commit.  `status_nightly.sh`
reported the schedule, the recent runs, and the correctness summary.

**Increment 6: the torch-release watch and the dependency canary.**  `check_torch_release.py` is
new.  The watch runs every night and warns when PyPI carries a torch newer than
`TORCH_LAST_REVIEWED`, with a second line saying whether that version can install on the env's
Python.  The canary is the measuring half and ships switched off.

Verification exercised both states.  The watch is silent when `TORCH_LAST_REVIEWED` equals the PyPI
latest, and prints both lines when it does not.  The canary stayed inert with `DEP_CANARY_ENABLED=0`.

## The review of commits after the reference commit

The review is complete, and it found nothing further to port.  Twenty files changed between
`e37bc93e` and the tip of `mbirjax_metrics`, outside `results/` and `state/`.  Seventeen of them are
the torch nightly itself, which this port took at its final state, so every fix inside them came
along.

Three shared files changed, and each was decided.  `lib_mac_entry.sh` and the `enable_nightly.sh`
change that uses it are the macOS entry-clone fix, and both were ported.  The `recent_runs.py`
change widens a column to fit hyphenated platform keys, which this repository does not have.  The
`dashboard.js` change makes the dashboard open on the newest non-torch run, which is wrong here.

## Departures from the plan as written

Three things differ from the plan above, and each is a simplification the plan did not anticipate.

`run_nightly.py` was not ported.  In the reference commit it is a thin, environment-driven entry
point over `performance_tracking.run`, kept separate because `main()` served manual runs.  The
merged engine has one entry point, so `main()` reads the environment directly and the extra file
would add nothing.

The two dashboard entry scripts landed in increment 3 rather than increment 5.  `build_dashboard.sh`
and `clear_correctness.sh` belong to the dashboard, and increment 3 needed the first one to run its
own check.

The schema key rename in increment 3 was not in the plan.  It follows from decision 1 in the same
way the platform rename does, and the dashboard forced the choice.

## The cutover, still to do

Five steps remain, and each one changes something outside this repository.

1. Turn on GitHub Pages for the repository, with the source set to GitHub Actions.  The workflow is
   already in place and the build already runs locally.
2. Create the push token on the cluster.  It must grant write access to
   `cabouman/mbirtorch_metrics`, so the mbirjax nightly's token will not serve.  The path
   `regression.env` expects is `~/.config/mbirtorch/metrics_credentials`.
3. Run one forced night on the cluster as the end-to-end check, with `REG_FORCE=1`.
4. Disable the torch nightly in `mbirjax_metrics`, then enable this one.  The order matters: while
   both are enabled they measure the same mbirtorch branches and the series splits across two
   repositories.
5. Decide what happens to the torch files and rows left in `mbirjax_metrics`.  They can stay frozen
   or be removed.

One detail belongs to step 4.  Both nightlies use `~/.mbirtorch/regression` as their work directory,
and the clone inside it currently belongs to `mbirjax_metrics`.  The new wrapper detects that and
re-clones, so no manual cleanup is needed.

---

# The cutover (2026-08-23)

The Mac is cut over and running against the new repository.  The cluster is not, and one blocker
explains both remaining steps: no credential on the cluster can push to
`cabouman/mbirtorch_metrics`.

## Step 1: GitHub Pages, blocked by permission

Enabling Pages needs admin permission on the repository.  `gbuzzard` has `write`.  Only `cabouman`
can turn it on, or grant admin.

The workflow itself is proven.  It has already run once, on the increment-1 push.  Its build step
succeeded and produced the dashboard.  Only the `configure-pages` step failed, because Pages is off.
Turning Pages on, with the source set to GitHub Actions, is the whole of what remains.

## Step 2: the push token, blocked for a specific reason

The credential on the cluster is scoped to the old repository.  A dry-run push to
`cabouman/mbirtorch_metrics` using `~/.config/mbirjax/metrics_credentials` returns
"Permission to cabouman/mbirtorch_metrics.git denied to gbuzzard", which is HTTP 403.

SSH is not an alternative.  The cluster has no SSH key for GitHub, and `ssh -T git@github.com`
returns "Permission denied (publickey)".

The new token must be a CLASSIC token with the `repo` scope, not a fine-grained one.  A fine-grained
token can only target repositories owned by whoever created it, or by an organization that opted in.
`cabouman` is a user account, not an organization, so a fine-grained token owned by `gbuzzard`
cannot reach this repository.  A fine-grained token created by `cabouman` would also work.

Once the token exists, `action_scripts/create_token.sh` writes it to
`~/.config/mbirtorch/metrics_credentials`, which is the path `regression.env` expects.

## Step 3: the forced runs, done on both machines

**The Mac ran a real night**, not a trial.  It measured `prerelease` and `greg_dev` on CPU, wrote
both runs, and pushed them.  The push proves the Mac's keychain credential reaches the new
repository.

The `greg_dev` run is the strongest single check in the whole port.  Its gate compared a fresh
measurement against a MIGRATED prior, `regression_cpu_20260821T211839Z_42574f83.yaml`, and returned
PASS with no changes.  Seven cells set new best-ever records against migrated baselines.

**The cluster ran a forced night** on `main`, in an isolated work directory with pushing off.  It
took 20 minutes 48 seconds on four H100s and swept n=1, n=2, and n=4.

That run reproduced the migrated run's gate exactly.  Both flag the same 14 hard items, on the same
cells, with the same numbers to every digit reported.  The measured memory is therefore identical
between the old engine and the ported one.  The gate says FAIL because a forced re-measure of one
commit overwrites that commit's file, so the comparison falls back to the previous commit's run from
2026-08-13.  Those 14 items are the pre-existing memory regression the old nightly already recorded
on 2026-08-21.  They are not a defect in the port.

## Step 4: the schedules, Mac done and cluster held

On the Mac the old torch agent was unloaded and the new one loaded.  It runs daily at 10:00, from an
entry clone of the new repository at `~/.mbirtorch/entry`.  The mbirjax agent was not touched.

On the cluster nothing was changed.  The `mbirtorch-nightly` scrontab block still runs the old
wrapper into `mbirjax_metrics`, and GPU coverage continues there.  Enabling the new cluster schedule
before the token exists would measure for up to four hours, fail to push, and lose the result, every
night, without an alert.  A push failure is a warning rather than an error, so the waste would be
silent.  Holding the cluster costs nothing except a few more frozen rows in the old repository,
which step 5 leaves in place anyway.

## One defect found and fixed during the cutover

`lib_mac_entry.sh` refreshed an existing entry clone in place without checking its origin.
`~/.mbirtorch/entry` held a clone of `mbirjax_metrics`, so the refresh would have left the agent
running that repository's `run_regression.sh`, which is the mbirjax nightly.  The fixed version
replaces a clone whose origin does not match, and the fix fired correctly during the cutover.  This
is the third place that check was needed; the wrapper and the status script already had it.

## What remains

Three commands, in this order, after the token exists:

1. On the cluster, `action_scripts/create_token.sh` from a checkout of the new repository.
2. In `mbirjax_metrics` on the cluster, `action_scripts/disable_torch_nightly.sh`.
3. In `mbirtorch_metrics` on the cluster, `action_scripts/enable_nightly.sh`.

Separately, `cabouman` turns on GitHub Pages with the source set to GitHub Actions.

## Two things left in a changed state

The trial sent one notify email, subject `[mbirtorch-nightly] gpu regression: main`.  It reports the
14 hard items above, which are already known.

Both `mbirtorch_regression` conda environments, on the Mac and on the cluster, have their editable
install pointing at a clone that has since been deleted.  Each nightly reinstalls as its first step,
so both self-heal on the next run.

---

# Two questions about the live dashboard (2026-08-24)

Greg raised two things after the dashboard went live.  One was the pending cutover, one was a real
gap in the cell set, and one part was not a defect at all.

## The GPU rows stop before the newest commits

The cause is the pending cluster cutover, not the port.  The cluster nightly still runs the old
wrapper from `mbirjax_metrics`, because `~/.config/mbirtorch/metrics_credentials` does not exist.
One GPU run had landed there since the migration: `greg_dev` at commit `eee646af`, measured
2026-08-24.

That run is now back-filled into this repository, using the same renames the migration used.  The
dashboard went from 41 runs to 42.  The back-fill covers GPU only.  The CPU nightly already writes
here, so copying its runs from the old repository would replace measurements this repository made
with duplicates of the same commits.

Back-filling is a stopgap.  Every cluster night adds another stranded GPU run until the token exists
and the cutover happens.

## Prerelease shows one GPU entry, and that is correct

`results/gpu-torch/prerelease/` in `mbirjax_metrics` has exactly one run, committed 2026-08-23, and
git shows no deletions there.  So one entry is faithful.

The reason there is only one is fire-on-change.  `prerelease` was added to the torch tracked
branches, removed, then added again, and its tip moved only once while it was tracked.  The
long prerelease series in the old dashboard is the mbirjax one under `results/gpu/prerelease/`,
which is jax data and was deliberately not migrated.

## Translation and multiaxis were genuinely missing

The design record fixed the cell set on 2026-08-08 with this sentence: "mbirtorch has no
translation or multiaxis geometry, so those two rows do not exist and their absence is not a gap."
That sentence is now stale.  mbirtorch exports `TranslationModel` and `MultiAxisParallelModel`, and
both build and project.  The port carried the 2026-08-08 cell set forward faithfully, so the gap
came across with it.

Both geometries are now in the engine, at the mbirjax engine's own sizes and ops, so the rows sit at
coordinates the two backends share.  Each runs `direct_filter`, `forward`, and `back`.  Neither runs
vcd, because their reconstruction is the shared qGGMRF outer loop already tracked under parallel and
cone.  Both are held at one device, like the denoiser, and `SINGLE_DEVICE_GEOMETRIES` names that set
in one place so the gate does not expect multi-device cells for them.

No dashboard change was needed.  The dashboard already knew both geometries, including a History
group called "translation + multiaxis" that has been empty until now.

The cost is small.  The 21 new GPU cells took 167 seconds on one H100, against a full night of about
21 minutes.  The 15 new CPU cells took 71 seconds on the Mac.  No cell failed on either platform.

## What the new rows showed on their first measurement

The multiaxis forward projector is 3.28 times slower at the non-dividing size than at the dividing
one.  It reads 305.9 ms at 512x448x384 and 1004.6 ms at 513x449x385, on one H100.

Every other geometry pays far less for the same step.  Parallel pays 1.04 times on forward, cone
pays 1.03 times, and multiaxis itself pays only 1.14 times on its filter and 1.17 times on its back
projection.  So the penalty is specific to the multiaxis forward projector at the non-dividing size.

These numbers come from one run on one node, so they need a second run before anyone acts on them.
The nightly will supply that on its own.  Surfacing this is what the non-dividing cell is for.

## One measurement lesson, recorded because it nearly misled the cost estimate

A first pass measured the new CPU cells on a Gautschi login node and read 18,450 ms for the
multiaxis filter at 96x80x64.  The same cell on the Mac reads 5.7 ms.  The login node is shared and
contended, so it is not a ruler.  Measure on the node the nightly measures on, or on a batch
allocation, and never on a login node.

## Also changed

`HARNESS_DEPS` now installs PyYAML alongside ruamel.yaml.  `recent_runs.py`, which
`status_nightly.sh` calls, reuses the dashboard reader and needs PyYAML.  The dedicated env did not
carry it, so the recent-runs table fell back to listing filenames on a machine that had nothing else
with PyYAML.

## Evidence that the Mac cutover holds

The Mac nightly fired on its own at 10:00 on 2026-08-24, updated its clone of this repository, and
reported all three branches unchanged.  No hand-holding was involved.

---

# The push credential, and a 403 that was not the token (2026-08-24)

The first token check on the cluster failed with "Permission to cabouman/mbirtorch_metrics.git
denied to gbuzzard", which is HTTP 403.  The token was correct.  The wiring was not.

git consults every configured `credential.helper` in config order, which is system, then global,
then local, then the command line, and it uses the FIRST credential returned.  The cluster account
has a global `credential.helper = store`, which reads `~/.git-credentials`.  That file matches by
HOST alone, so its `github.com` entry answers every github.com request.  On this account that entry
holds a token for the other metrics repository, so it answered first and the push was refused.

Adding a helper on the command line does not displace the global one.  It appends to the list.  An
EMPTY value resets the list, so the fix is to reset first and then add the file:

```
git -c credential.helper= -c credential.helper="store --file=$TOKEN_FILE" push --dry-run
```

Measured on 2026-08-24 in a throwaway clone: the single-helper form returns 403, and the
reset-then-add form returns "Everything up-to-date".  The token is the same in both.

The nightly had the same defect.  `run_regression.sh` wired its helper with one
`git config credential.helper ...` call, which appends rather than replaces.  The first scheduled
cluster night would therefore have measured for up to four hours and then failed to push.  A push failure is a warning rather than
an error, so the loss would have been silent.  The wrapper now resets the list before adding
`TOKEN_FILE`, and the two token documents explain why the reset is required.

One thing was left alone deliberately.  The `github.com` entry in `~/.git-credentials` is stored
under the literal username `YOUR_GITHUB_USERNAME`, which is a template line that was pasted
verbatim at some point.  GitHub ignores the username for a personal access token, so the entry still
works, and the mbirjax nightly may depend on it.  Changing a shared global credential file is the
account owner's call, not the harness's.


---

# Retiring the mbirjax nightly on the Mac (2026-08-24)

The Mac now runs one nightly.  `com.mbirjax.regression`, which ran at 09:00, is unloaded and its
agent file is removed.  `com.mbirtorch.regression` still runs at 10:00 against this repository.
Nothing else changed: no config, no results, and no cluster schedule.  Running
`action_scripts/enable_nightly.sh` from the mbirjax_metrics checkout puts it back.

## The retired nightly had already stopped working

This matters for reading the mbirjax dashboard, so it is recorded here.  The mbirjax macOS nightly
produced its last completed run on 2026-08-17.  Every run since then aborted, and the logs name two
causes.

The first cause is the network.  Two runs, on 2026-08-18 and 2026-08-20, ended at
`FATAL: clone metrics failed`, with `Could not resolve host: github.com` in the error log.  The Mac
apparently had no working DNS at 09:00.

The second cause is a shell bug.  Every run after that aborted at
`run_regression.sh: line 184: CHANGED_BR[@]: unbound variable`, immediately after the
dependency-canary line.  The abort left no completion line, so the log simply stops.

The mbirjax CPU series is therefore about a week stale.  Retiring that nightly ended nothing that
was still working.

## The same bug was latent in this repository

macOS ships bash 3.2.  Under `set -u`, bash 3.2 raises "unbound variable" when it expands
`"${arr[@]}"` on an EMPTY array.  `${#arr[@]}` is safe, and bash 4.4 and later are safe both ways.
The cluster runs bash 5.1, so only the macOS nightly is exposed.

This wrapper had the unsafe form in three places, and two action scripts had it as well.  It had
never fired, for one reason: the two risky paths are the dependency canary, which ships switched
off, and `--sbatch` with no other argument, which is only ever used on the cluster.  Switching the
canary on would have aborted the macOS nightly on the first night with no changed branch.

Every such expansion now uses the `${arr[@]+"${arr[@]}"}` form, which expands to nothing when the
array is empty.  Verified under `/bin/bash` 3.2: the old form aborts, the new form runs, and the
wrapper completes its no-change path.
