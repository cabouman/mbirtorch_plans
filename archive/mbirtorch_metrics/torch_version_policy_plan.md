# Torch version policy — plan and record

Status: IMPLEMENTED 2026-09-03 (rev 2 after review; the verification log is at the end).  Two
repos: the metrics harness (`mbirtorch_metrics/tooling/regression/`, `action_scripts/`,
`state/<plat>/`) and the library's dependency watch and CI (`mbirtorch/ci/`, `.github/`).
Predecessor design: `plans/torch_port/closed/python_matrix_nightly_check.md` (the watch); this
plan changes what the watch proposes and adds the validation it was waiting for.

## The policy

1. **Every new torch version is validated automatically**, minor or patch, within a night of
   appearing on PyPI: the CPU suite through CI on the newest torch (dispatched the night the
   release appears), the GPU and Triton paths through the nightly's dependency canary, which
   upgrades the regression env and re-measures a fixed commit under the new torch so any change
   is attributed to torch.
2. **Approval is an outcome, not a decision.**  A version is *validated* when the canary's
   measurement passes the gate with no warning, the engine did not abort, and the test suite
   passed under it; *rejected* otherwise.  A time WARN rejects by default
   (`DEP_CANARY_WARN_IS_REJECT=1`): the jax 0.10.2 precedent, a 3–9× slower forward projection,
   was a pure time regression, which the gate classes as soft.  The newest validated version is
   what the nightly runs on, and the wrapper re-pins the env to it at every start.  Humans see
   rejections in the alert mail and override one by editing the ledger line.
3. **The floor moves by rule, rarely.**  The watch proposes a floor advance only when the ledger
   holds more than `FLOOR_WINDOW` (4) validated minors; the proposed floor is the oldest of the
   newest four, so about a year of releases stays installable, and a minor with no validated
   version is never proposed.  Merging stays human.  Exclusions below the new floor are dropped
   in the same pull request.
4. **Exclusions only for demonstrated failures.**  A rejected version at or above the floor gets a
   `!=X.Y.Z` proposal from the watch, with the ledger's evidence in the body.  A later patch
   release is a new candidate and is validated on its own.

## The ledger

`state/<plat>/torch_ledger.txt` in the metrics repo: the publish block already stages
`state/<plat>`, the GPU and CPU nightlies never touch each other's file, and a machine-written
file stays out of the hand-edited `action_scripts/`.  Append-only lines, latest line per version
wins; hand-seeded once per platform with the env's torch at enablement:

```
# <version> <state> <date> <evidence>
2.13.0 validated 2026-09-03 bootstrap:env-at-canary-enablement python=3.12
2.14.0 candidate 2026-09-04 pypi python=3.12
2.14.0 validated 2026-09-04 regression_gpu_20260904T030512Z_26bd0ea9_g0001.yaml python=3.12
2.14.1 unavailable 2026-10-02 index=https://download.pytorch.org/whl/cu130 env=2.14.0 python=3.12
```

States: `candidate` (installed, verdict pending), `validated`, `rejected`, `unavailable` (PyPI has
it, the pinned wheel index does not).  A release is NEW while its latest line is none of
`validated`/`rejected`, so a night killed between the candidate line and the verdict retries,
and `unavailable` retries nightly until the index serves it; the old `torch_seen` files are
retired.  `TORCH_LAST_REVIEWED` is empty and defaults to the ledger's newest version.
Helpers: `tooling/regression/lib_ledger.sh` (`ledger_state`, `ledger_runs_on`, `ledger_newest`,
`ledger_append` idempotent per (version, state), `ledger_versions_in_state`), pure bash with a
portable version sort, exercised by `test_lib_ledger.sh`.

## Harness changes

- **Startup invariant** (`run_regression.sh`, after the env activates): the installed torch must
  equal `ledger_runs_on`; if not, re-pin from the pinned index, and refuse to measure (exit 2,
  Slurm FAIL mail) when the re-pin fails.  A failed restore can therefore never persist silently.
- **Upgrade, then decide.**  On a NEW release: upgrade the env, verify the installed torch is the
  release; if so it is *effective* (generation bump, candidate line, canary branch measured even
  if unmoved); if not, an `unavailable` line, the env re-pinned to `runs_on`, and the night
  proceeds on the current torch.  A ledger line on a quiet night no longer dies at the
  no-change exit: the night falls through to the publish block with an empty branch list.
- **Every measurement under a new dependency set is tagged**: the loop passes `REG_DEP_GEN` and
  `REG_RUN_REASON` (`torch-step` for the unmoved canary tip, `code-step` when the tip moved and
  the torch step measured the previous tip, `commit` otherwise).
- **The verdict** after the loop: the torch step's gate (when it ran) plus the canary branch's
  engine rc, abort marker, GATE WARN marker and test result, all under the new torch.  Rejection
  writes the ledger line, restores `runs_on`, moves every run file written tonight under the
  rejected torch to `results/<plat>/<branch>/rejected/` (the prior-selection and dashboard globs
  are non-recursive, so they leave the series), and restores those branches' state files from
  the last push so they re-measure tomorrow on the restored torch.  The alert mail is forced on
  any new `rejected` or `unavailable` line, once per version.
- **`lib_env.sh`**: the torch requirement is read from the clone's `pyproject.toml` with
  `tomllib` (fallback `torch>=2.13` with a loud log), pre-installed from the pinned index so an
  exclusion cannot make the editable install re-resolve torch from PyPI, and checked after the
  install with `packaging`; `reg_upgrade_torch` verifies the import; `reg_pin_torch` installs an
  exact version from the index (PEP 440 matches `2.13.0` to `2.13.0+cu130`); the 14-day
  `reg_upgrade_all` holds torch with a constraints file and adds the index as an extra index.
- **Knobs**: `DEP_CANARY_ENABLED=1`, `DEP_CANARY_WARN_IS_REJECT=1`, `TORCH_LAST_REVIEWED=""`,
  `METRICS_URL` override-able as a trial seam.
- **Watchdog**: the checker's new `verdict UNKNOWN (ledger ...)` line gets its own arm.

## Watch and CI changes (`mbirtorch`)

- `ci/dependency_watch.py`: reads the GPU ledger (`--ledger-file` for tests and dry runs);
  `parse_torch_exclusions`, `parse_ledger`; `divergence()` takes the ledger and pyproject's
  exclusions: floor by the window rule, exclusions for rejected versions, `ledger_known`; an
  unreadable ledger proposes nothing torch-related and prints `verdict UNKNOWN`, never
  `verdict none`; `compose()` rewrites the whole torch line (floor, surviving exclusions, new
  exclusions) and cites the ledger evidence; the PR body's plan pointer is corrected.
- New-release trigger: `last-release` beside the two-night state; a change emits
  `new_release=true` and the workflow dispatches `ci.yml` on `prerelease`.  An empty previous
  value (cache miss) is not a new release.
- `ci.yml`: a `floor` job on the oldest matrix Python installs the pyproject requirement pinned
  to the floor minor (`torch==<floor>.*` plus the exclusions) from the CPU index and runs the
  suite.
- Tests: ledger parsing, the window rule at and below the window, an unknown ledger, exclusion
  proposal, the pyproject edits (exclusion added, sub-floor exclusions dropped), and the two
  existing floor-advance tests rewritten for the rule.

## Review outcome (2026-09-03)

Opus reviewed (its first successful launch today): eighteen findings, eight must-fix, verdict
"implement with the must-fix changes".  All adopted: the quiet-night ledger publish (1), the
`torch_seen` retirement (2, 17), the constrained full refresh (3), the rejected-run set-aside
(4), tomllib plus post-install check (5), the torch-step-decides note (6), WARN rejects (7),
forced mail on rejected/unavailable (8), the floor job installs the pyproject requirement and
the rule never names an unvalidated minor (9), sub-floor exclusions dropped on advance (10),
`FLOOR_WINDOW=4` from validated minors (11), the ledger under `state/` (12), hand-seeded
bootstrap (13), the watchdog arm (14), cache-miss handling (15), the startup invariant (16),
and the trial hazards (18): knobs exported, a plain venv, trial ledger lines discarded.

## Second review (default model, same day) and what it changed

Seventeen findings, six must-fix, verdict "implement with the must-fix changes".  Adopted:
the engine's prior lookup used the bare tag while the file is saved with the `_gNNNN` suffix,
so a generation re-measure gated against the previous commit and blamed code on torch — fixed
in `performance_tracking.py`; the candidate is what the pinned wheel index resolves
(`pip index versions`), not PyPI's newest, which only feeds the `unavailable` line; the
verdict run happens BEFORE the branch loop (with the test suite), so no other branch is ever
measured under an unvalidated torch and a clone or install failure is "no outcome, retry",
not a rejection; `validated` is written only if the installed torch after the run is still
the candidate (a branch's own pyproject exclusion can pull it back); WARN lines always go into
the alert mail; the ledger merges by union (`.gitattributes`); `DEP_CANARY_PLATFORMS="gpu"`
keeps the Mac's canary off; the ledger parser fails closed on any unreadable line; the floor
rule gained hysteresis (`FLOOR_SLACK=1`: a proposal about every two minors, moving the floor by
two); the weekly probe reports `torch.runs_on`, pending and rejected versions and flags an env
that disagrees with the ledger; the watch line's stale "bump TORCH_LAST_REVIEWED" advice is
replaced.  Not adopted: dropping the CI dispatch (kept, it buys a night of latency for the CPU
matrix), a `--system-site-packages` trial venv (the other review's plain venv is safer), and
`!=2.14.*` exclusions (a later patch is its own candidate; churn is bounded by rejections).

## What stays human

Merging the watch's pull requests, answering a rejection mail when a slowdown was intended (edit
the ledger line to `validated`; the invariant then moves the env), and moving
`TORCH_INDEX_URL_gpu` when torch's default CUDA build moves on: the `unavailable` line and its
mail are the reminder, and the cluster probe's driver fact says whether the next index is allowed.

## Cost

Per new torch version: one CI run and one canary measurement, about twenty minutes of four
GPUs; a rejection costs one more measuring night for the branches set aside.  Quiet nights add
one `pip install -U torch` that finds nothing.

## Verification log

- Local, 2026-09-03: `test_lib_ledger.sh` passes (idempotent append, latest-line-wins, version
  order without `sort -V`, comments ignored); `ci/test_dependency_watch.py` 19 tests pass
  (ledger parsing fails closed, the window rule with slack at and below the threshold, an
  unknown ledger, exclusion proposals, the pyproject edits); every edited shell file passes
  `bash -n`, both workflows parse; a `--dry-run` of the watch against the live CPU index with
  the seed ledger reports one validated version, "3.15 not on the runners yet", and
  `verdict none`.  The engine's gate tests need `ruamel.yaml`, which the dev env lacks (a
  pre-existing collection error, unrelated to the one-line prior fix).
- Cluster trial (job 15860580): a scratch clone of the cluster checkout carrying the working
  tree as one commit, `METRICS_URL` pointed at it, a plain venv, `REG_NO_PUSH=1`, a scratch
  pip cache (the home quota is at 73%), on 4 GPUs.  Result, h016, 14:20–15:14:
  - the startup invariant found no torch in the fresh venv and pinned 2.13.0 from the cu130
    index (96 s); the canary resolved 2.14.0 as the index's newest, upgraded, confirmed the
    installed torch, wrote `candidate`; the verdict run measured `main @ b8bbd8a0` as a
    `torch-step`: the suite passed (751 passed, 14 skipped, 10 min), the engine sweep ran (39
    min) and gated WARN against the 08-21 prior; the ledger gained `rejected`, the `_g0001`
    run and its table moved to `results/gpu/main/rejected/`, the venv went back to 2.13.0 and
    the shared env was never touched (2.13.0+cu130 throughout); the wrapper then began the
    14-day full refresh, which any fresh work directory triggers, and the job was cancelled
    there to save the 4-GPU sweep — so the publish and mail blocks ran only by inspection.
  - **The rejection was false.**  All 52 warning lines were "new cell, no baseline (not
    gated)": the sweep was widened on 08-27 and the 08-21 baseline lacks those cells.  The
    coarse rule "any GATE WARN rejects" was replaced by the second review's version: only a
    soft line reporting a time, speedup or memory change against the prior counts, and the
    ledger evidence says `warn=perf` or `warn=info`.  Run on the trial's engine log the
    refined classification yields no performance warning, so torch 2.14.0 would have been
    validated by that run; the real verdict is tonight's nightly.
- Expected first night (03:00): the verdict run for 2.14.0 (about 50 min) plus the first full
  refresh (about 40 min), within the 4 h walltime; the ledger is pushed with the results, the
  watch reads it the next morning and, with two validated minors, proposes nothing.
