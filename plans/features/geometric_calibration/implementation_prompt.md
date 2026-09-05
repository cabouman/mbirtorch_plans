# Implementation prompt: geometric calibration, the reconstruction estimator

This file starts a new session on the geometric calibration feature.  Paths are relative to the
root of the `mbirtorch_plans` repository unless said otherwise.  mbirtorch file paths follow the
plans' convention: they are given from the package directory, so
`mbirtorch/preprocess/geometry_calibration.py` means
`mbirtorch/mbirtorch/preprocess/geometry_calibration.py` in the sibling repository.

## Current task

Both plans read APPROVED as of 2026-09-05:
`plans/features/geometric_calibration/geometric_calibration_plan_v2.md` and
`estimate_by_recon_plan.md` in the same directory.  The superseded stamps on
`closed/geometric_calibration_plan.md` and `closed/status_2026-09-05.md` are in place.  If any
of those four status lines reads otherwise, stop and ask Greg before running anything.

Execute `geometric_calibration_plan_v2.md` from its Increment 1,
which is the estimator plan of `estimate_by_recon_plan.md`.  Its sub-increments are numbered 1.1
to 1.5, and the first, 1.1, is one cluster job: the fine far-slice sweep that settles whether
reconstruction quality on the no-metal NSI scan prefers 0.130 or 0.167 degrees, and that picks
the blur default from data.  Stop for Greg's review at the end of every increment and
sub-increment.

The public function under construction is `estimate_geometry_from_recon`.  The two
`estimate_by_recon` file names keep their earlier short form.

## Where to run

Start the session with the `mbirtorch` repository as the working directory, so that the project
memory for that directory loads.  The memory index records the local test environment, the
cluster access that works, and the shared-checkout protocol.  `mbirtorch_plans` and
`mbirtorch_metrics` are sibling checkouts; `mbirjax` is a read-only reference.
`mbirtorch_plans/.claude/initial_prompt.md` describes the roles of these repositories.

## Read first, in this order

The files below are the required background, and the first two are in
`mbirtorch_plans/.claude/`:

1. `mbirtorch_plans/.claude/claude_prompt.md`: how Greg wants to work, and what he relies on the
   session for.
2. `mbirtorch_plans/.claude/writing_style.md`: mandatory for every durable record, code comment,
   and summary.  Reread it before drafting anything, and again before the revision pass it
   requires.
3. Skim `mbirtorch_plans/.claude/lessons.md`: engineering rules from the sharding effort and the
   torch port.
4. `plans/features/geometric_calibration/geometric_calibration_plan_v2.md`: the plan of record,
   including its "Terms and scans" section.
5. `plans/features/geometric_calibration/estimate_by_recon_plan.md` and `estimate_by_recon.md`:
   the estimator's plan and its design, including the mechanism section the gates rest on.
6. `plans/features/geometric_calibration/executive_summary_2026-09-05.md`: the state of the
   feature and the evidence in short form.
7. `mbirtorch/preprocess/geometry_calibration.py`: the module as it stands, with v1's Increment
   6 edits staged and uncommitted.
8. `mbirtorch_plans/.claude/cluster_use.md` and `mbirtorch_plans/.claude/gpu-resources.md`: how
   to run cluster jobs.

The closed campaign's pages are in `plans/features/geometric_calibration/closed/`, and its
experiment scripts and records in `plans/experiments/features/geometric_calibration/closed/`.
Read them when a cited number or a harness pattern is needed, not as a prerequisite.

## Harness starting points

The pointers below say where the working patterns are:

- The fine-sweep job of sub-increment 1.1 is the far-slice job rerun on a fine grid.
  `plans/experiments/features/geometric_calibration/closed/real_scan_rotation_recon.py` shows
  the slice reconstruction and figure pattern, and `closed/real_scan_band_height.py` in the same
  directory shows the loading, recording, and memory pattern.
- The scan-loader modules those jobs import are in that same `closed/` directory:
  `real_scan_validation.py` and `real_scan_followup.py`.  A new job script at the experiments
  directory's top level cannot import them directly; either add the `closed/` directory to
  `sys.path` at the top of the script, or rely on the cluster layout, where everything is flat.
- On gautschi the submit directory is `/scratch/gautschi/buzzard/leap_cmp`, which holds the
  extracted scans, the venv with both mbirtorch and LEAP, flat copies of the loader modules, and
  the batch-file precedents.  The local `closed/` copies and the cluster copies have diverged in
  location, so before submitting a job that imports a loader, compare the cluster copy of each
  imported module against the local `closed/` copy and report any difference.
- The live synthetic harness is
  `plans/experiments/features/geometric_calibration/rotation_zero_point_synthetic.py`; the
  estimator plan's sub-increment 1.2 extends it with a no-slab phantom.
- The LEAP cross-generation harness is `leap_axis_tilt.py` in the same directory, with its
  record's caution about asymmetric objects.
- Real-scan jobs on the NSI scans request two GPUs for host memory, as every batch file in
  `closed/` shows.

## Constraints the plans record

Follow the increments in the order the plan of record states, and stop for Greg's review at each
stop it names.  Each increment names its files, its tests, and its gates; a gate that needs a
GPU runs as a cluster batch job.  A sinogram correction must not allocate a second full-size
sinogram.  Do not put references to the plan or to increments in code or comments.  Do not
duplicate geometry arithmetic that the model classes own.  Scripts and job files go in
`plans/experiments/features/geometric_calibration/`, with run parameters at the top and no
command-line arguments.  Findings pages go in `plans/features/geometric_calibration/`.  A
measured number appears in a durable record only after it was read from its source in the same
session, with the source cited beside it.  This repository ignores `.png`, `.sbatch`, and
`.jsonl` files, so a record transcribes what it needs from them.

Three validation rules and two cautions from the closed campaign apply to every new measurement.
A synthetic rotation is injected at four times the detector resolution or through LEAP's
modular-beam projector, never by resampling at the detector's own resolution with the kernel
under test.  Real-scan gates use quantities that need no ground truth where possible.  A deep
score minimum is not a right answer without such a check.  The two cautions: a deterministic
search returns points of its own lattice on a flat curve, so identical digits across runs mean
the lattice and not the data (`closed/real_scan_band_reach.md`); and rotations that displace the
edge pixel by less than one pixel sit where the resampling kernel's own bias dominates
(`rotation_zero_point_synthetic.md`).

## Working with the shared checkouts

Other sessions edit and commit in the `mbirtorch` checkout at the same time.  Do code work on
the `geometric_calibration` branch, never on `greg_dev` directly.  Do not run the full test
suite while another session may be running it.  Before reporting, check `git status` and report
the staged list as it is at that moment.

## Git protocol: stage, but do not commit without authorization

Stage a file with `git add` when it is finished and verified.  Do not run `git commit` or
`git push` unless Greg authorizes it in the conversation, and an authorization covers only the
files and the commit it names.  When authorized, use the commit attribution line the session's
instructions specify, and report the commit hash.

## Reviews and delegation

Use Opus subagents for drafting, harness writing, inventories, and result extraction.  Keep the
main session for judgment: it reviews a harness before the harness runs, rules on results, and
reads the final record.  Have a subagent review new estimator code against the module and the
projector source before the tests are final.  Before a findings page or a plan revision is
called done, run a panel of three reviewers on it.  The three charges are accuracy against
sources, reasoning, and style.  Apply their findings in one pass, then read the result yourself.

## Reporting

Keep chat summaries short and plain, in the style guide's form.  Lead with the outcome, then
what was verified, then what is staged, then what is left.  Put measurements in a findings page,
not in the chat.  `estimators_in_action.py` in the experiments directory draws figures when Greg
asks to see an estimator working.
