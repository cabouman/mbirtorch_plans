# Implementation prompt: geometric calibration, sub-increment 1.2

This file starts a new session on the geometric calibration feature.  Paths are relative to the
root of the `mbirtorch_plans` repository unless said otherwise.  mbirtorch file paths follow the
plans' convention: they are given from the package directory, so
`mbirtorch/preprocess/geometry_calibration.py` means
`mbirtorch/mbirtorch/preprocess/geometry_calibration.py` in the sibling repository.

## Current task

Both plans read APPROVED as of 2026-09-05: `geometric_calibration_plan_v2.md` and
`estimate_by_recon_plan.md` in this directory.  Sub-increment 1.1 is complete: the fine far-slice
sweep ran as gautschi jobs 15970242 (bilinear kernel), 15971893 (Fourier kernel), and 15980973
(analysis); its record is `plans/experiments/features/geometric_calibration/recon_sweep_fine.md`
and its findings page is `increment_1_1_findings.md` here.  Greg reviewed both on 2026-09-05 and
decided the three design questions they raise:

- The scoring kernel is the Fourier shear kernel.  The projector-side rotation is the larger
  change and stays parked.  How the kernel handles a truncated projection (padding and a taper at
  the row ends) is an open implementation item; the findings page lists the options.
- The slices are combined by fitting the offset-error model to the per-slice minima, not by a
  trimmed curve, and the gates are stated on the fit.
- The even/odd view split runs by default, as the noise reference and the repeatability floor.

The next work is sub-increment 1.2 of `estimate_by_recon_plan.md`: the score and the slice
chooser, with synthetic tests and the score-resolution measurement, built on those three
decisions.  Stop for Greg's review at the end of every increment and sub-increment.  The 1.1
deliverables were staged and not committed when this file was written; check `git status` in both
repositories first, because Greg may have committed them, and never rebuild what is already there.
If a plan's status line reads otherwise than APPROVED, stop and ask Greg before running anything.

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
   the estimator's plan and its design.  Where sub-increment 1.2's text assumes a trimmed-curve
   combination or a resampling kernel, the 1.1 decisions above supersede it.
6. `plans/features/geometric_calibration/increment_1_1_findings.md`: the kernel finding, the
   gates, the decisions, and the numbers 1.2 and 1.3 inherit (the half width and the undecided
   calibration cases).
7. `plans/experiments/features/geometric_calibration/recon_sweep_fine.md`: the evidence behind
   the findings page, including "Units and terms" for the lattice, the noise split, and the
   offset-error model.
8. `mbirtorch/preprocess/geometry_calibration.py`: the module as it stands.  Its docstrings were
   shortened to caller-facing text and four passages now state the fine-sweep result; those edits
   are part of the staged 1.1 set.
9. `mbirtorch_plans/.claude/cluster_use.md` and `mbirtorch_plans/.claude/gpu-resources.md`: how
   to run cluster jobs.

The closed campaign's pages are in `plans/features/geometric_calibration/closed/`, and its
experiment scripts and records in `plans/experiments/features/geometric_calibration/closed/`.
Read them when a cited number or a harness pattern is needed, not as a prerequisite.

## Harness starting points

The pointers below say where the working patterns are:

- `plans/experiments/features/geometric_calibration/recon_sweep_fine.py` is the fine-sweep job:
  the even/odd split, the identity check, the per-slice scoring, and the recording pattern.
  `recon_sweep_fine_fourier.py` beside it holds the Fourier kernel the estimator adopts
  (`fourier_shift`, `fourier_rotation_kernel`, and the row-margin widening); sub-increment 1.2
  lifts that kernel into the module.  `recon_sweep_fine_analysis.py` holds the offset-error model
  fit (`fit_offset_model`) and the noise-split arithmetic.  `recon_sweep_fine_tables.py` prints a
  job's record as tables.
- `recon_sweep_fine_slice_views.py` and `recon_sweep_fine_feature_profiles.py` beside them draw
  the candidate-comparison figures from a results directory's saved stacks: the crops with score
  curves, the groove profile, and the tooth-and-gap modulation.  Their measured numbers are in
  `recon_sweep_fine_views.md`.
- The scan-loader modules those jobs import are in the `closed/` subdirectory:
  `real_scan_validation.py` and `real_scan_followup.py`.  A job script at the experiments
  directory's top level adds `closed/` to `sys.path`; on the cluster everything is flat.
- On gautschi the submit directory is `/scratch/gautschi/buzzard/leap_cmp`, which holds the
  extracted scans, the venv with both mbirtorch and LEAP, flat copies of the loader modules, and
  the batch-file precedents.  The fine-sweep results, including every even-view and odd-view
  slice stack, are in `results_recon_sweep_fine` and `results_recon_sweep_fine_fourier` there;
  reuse them before reconstructing anything again.  Before submitting a job that imports a
  loader, compare the cluster copy of each imported module against the local `closed/` copy and
  report any difference.
- The live synthetic harness is
  `plans/experiments/features/geometric_calibration/rotation_zero_point_synthetic.py`; the
  estimator plan's sub-increment 1.2 extends it with a no-slab phantom.  The LEAP
  cross-generation harness is `leap_axis_tilt.py` in the same directory.
- `zeiss_metadata_probe.py` and its record `zeiss_metadata_probe.md` at the experiments top level
  answer what the Zeiss files hold: no vendor tilt (the format's `CameraFineRotation` field is
  zero in both files), applied per-view alignment shift totals, an unparsed shift decomposition,
  and per-view positions of every stage axis.
- Real-scan jobs on the NSI scans request two GPUs for host memory, as every such batch file
  shows.

## Constraints the plans record

Follow the increments in the order the plan of record states, and stop for Greg's review at each
stop it names.  Each increment names its files, its tests, and its gates; a gate that needs a
GPU runs as a cluster batch job.  The scoring kernel must not smooth by an amount that depends on
the fractional displacement; that rules out the module's bilinear kernel for scoring, and the
correction path in `apply_calibration` is a separate question that no measurement has settled.  A
sinogram correction must not allocate a second full-size sinogram.  Do not put references to the
plan or to increments in code or comments.  Do not duplicate geometry arithmetic that the model
classes own.  Scripts and job files go in `plans/experiments/features/geometric_calibration/`,
with run parameters at the top and no command-line arguments.  Findings pages go in this
directory.  A measured number appears in a durable record only after it was read from its source
in the same session, with the source cited beside it.  This repository ignores `.png`, `.sbatch`,
`.jsonl`, `.json`, and `.log` files, so a record transcribes what it needs from them.

Three validation rules and two cautions from the closed campaign apply to every new measurement.
A synthetic rotation is injected at four times the detector resolution or through LEAP's
modular-beam projector, never by resampling at the detector's own resolution with the kernel
under test.  Real-scan gates use quantities that need no ground truth where possible.  A deep
score minimum is not a right answer without such a check.  The two cautions: a deterministic
search returns points of its own lattice on a flat curve, so identical digits across runs mean
the lattice and not the data (`closed/real_scan_band_reach.md`); and rotations that displace the
edge pixel by less than one pixel sit where the resampling kernel's own bias dominates
(`rotation_zero_point_synthetic.md`).  The fine sweep added a third caution, recorded in
`increment_1_1_findings.md`: a displacement-dependent resampling kernel places score minima at
whole-pixel displacements, and a trimmed curve over such minima can be narrow, repeatable, and
wrong.

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
