# Implementation prompt: geometric calibration utilities, Increment 3

This file starts a new session that continues the geometric calibration plan at Increment 3.  It
points to the files that hold the details and repeats none of them.  Paths are relative to the
root of the `mbirtorch_plans` repository unless a path starts with `mbirtorch/`, which means the
sibling `mbirtorch` repository.

## Where to run

Start the session with the `mbirtorch` repository as the working directory, so that the project
memory for that directory loads.  The memory index records the local test environment, the
cluster access that works, the LEAP environment on the cluster, and the shared-checkout
protocol.  `mbirtorch_plans` and `mbirtorch_metrics` are sibling checkouts; `mbirjax` is a
read-only reference.  `.claude/initial_prompt.md` describes the roles of these repositories.

## Read first, in this order

1. `.claude/claude_prompt.md`: how Greg wants to work, and what he relies on the session for.
2. `.claude/writing_style.md`: mandatory for every durable record, code comment, and summary.
   Reread it before drafting anything, and again before the revision pass it requires.
   `.claude/writing_style_charlie.md` holds more examples of the same rules.
3. `.claude/lessons.md`: engineering rules from the sharding effort and the torch port.
4. `plans/features/geometric_calibration/geometric_calibration_plan.md`: the plan of record,
   version 3, accepted on 2026-09-03, with one correction after acceptance recorded at its end.
5. `plans/features/geometric_calibration/increment_1_findings.md` and
   `increment_2_findings.md`: what Increments 1 and 2 built, what was measured, the decisions
   Greg made, and the questions Increment 3 answers.  Read both in full.  The experiment
   records they cite are in `plans/experiments/features/geometric_calibration/`.
6. `mbirtorch/mbirtorch/preprocess/geometry_calibration.py` and
   `mbirtorch/tests/test_geometry_calibration.py`: the code as it stands, with 40 test cases.
   The module's docstrings state the contracts, and the conjugate-view section at its end is
   what Increment 3 extends.
7. `.claude/cluster_use.md` and `.claude/gpu-resources.md`: how to run cluster jobs.  The
   record `calibration_512_gautschi.md` in the experiments directory shows a working job of this
   feature, and the memory note on cluster access gives the environment that holds LEAP and
   mbirtorch together.

## What Increment 3 is, as Greg decided it on 2026-09-04

Greg answered the three questions at the end of `increment_2_findings.md`, and Increment 3
differs from the plan's text in these ways:

1. Extend the conjugate-view method to short scans of 180 degrees plus the fan angle.  Only the
   rays measured from both sides are paired, and the rest are excluded from the comparison.  The
   findings page of Increment 2 gives the geometry: the paired rays form a triangle in the view
   and channel plane, the views that hold any are two wedges each twice the full fan angle wide,
   and the paired rays are about a tenth of all rays at a 20 degree fan.  Three cautions from
   that page apply.  The score's rectangular region, circular shift, and per-pair normalization
   need a mask per pair.  The pairs are one-sided, which on a full rotation raised the error
   tenfold, so score each pair in both directions or interpolate the partner views with a cubic
   kernel, and measure which.  A parallel-beam scan over exactly a half rotation has no opposite
   views at all.  The coverage check must change from refusing such a scan to accepting it when
   enough rays have partners.  Golden-angle view orderings over a full rotation already work, and
   a test covers them.
2. The derivative-filter method is not needed for anything the feature now requires, and the
   question of whether it is useful for something else is open.  The conjugate method covers the
   channel offset and the detector rotation on full rotations, and the residual method of
   Increment 4 is the fallback for any geometry.  The findings page names what would settle the
   question: a measurement of the derivative score's minimum on a short scan, with Parker
   weighting, and whether a cone-beam initializer is wanted on real data with lateral truncation.
   If the session finds a use, propose it with its cost before building it; otherwise leave it
   out and say so on the findings page.
3. Offset scans, whose detector is displaced by hundreds of channels, are deferred to the work
   that adds the direct-reconstruction weighting they also need.

The search machinery the plan assigned to Increment 3, the coarse pass and the golden-section
polish, exists already in `_search_minimum`, and the search window moves when the coarse
minimum sits at an edge.  The cluster record already holds the wall times the plan asks for at
N = 512.  The plan's gate of about fifteen evaluations is exceeded by the search as built, which
costs 35 evaluations and twice that with the second pass on cone beam, so the gates of Increment
3 need restating on the findings page before they are claimed.

## Constraints the plan records

Follow the plan's increments in order, and stop for Greg's review at the end of each increment.
Each increment names its files, its tests, and its gates.  A gate that needs a GPU runs on the
cluster as a batch job.

A sinogram correction must not allocate a second full-size sinogram.  Stream view batches and
write the result in place, or stream into the device-resident sinogram.  `reduce_sinogram` and
`apply_calibration` show the pattern.

Do not put references to the plan, to increments, or to release timing in code or comments; they
go stale.  Do not duplicate geometry arithmetic that the model classes own; `recon_slice_z`,
`nearest_recon_slice`, and `pixel_magnification_bounds` were added to the models for that
reason, and a new need should be met the same way.

Scripts and job files for this work go in `plans/experiments/features/geometric_calibration/`,
with run parameters at the top of each script and no command-line arguments.  Findings pages and
status updates go in `plans/features/geometric_calibration/`.  Run detail goes in script comments
or in a companion `.md` file with the same base name as the script.  A measured number appears in
a durable record only after it was read from its source in the same session, with the source
cited beside it.  This repository ignores `.png`, `.sbatch`, and `.jsonl` files, so a record
transcribes what it needs from them.

## Working with the shared checkouts

Other sessions edit and commit in the `mbirtorch` checkout at the same time.  Do code work on the
`geometric_calibration` branch, never on `greg_dev` directly.  Do not run the full test suite
while another session may be running it.  Before reporting, check `git status` and report the
staged list as it is at that moment.

## Git protocol: stage, but do not commit without authorization

Stage a file with `git add` when it is finished and verified, so that `git status` shows what is
ready for review.  Do not run `git commit` or `git push` unless Greg authorizes it in the
conversation.  An authorization covers only the files and the commit it names; it does not carry
over to later work.  When authorized, use the commit attribution line that the session's
instructions specify, and report the commit hash.

## Reviews and delegation

Use Opus subagents for drafting, harness writing, inventories, and result extraction, and keep
the main session for judgment: reviewing a harness before it runs, ruling on results, and reading
the final record.  Have a subagent review new code against the projector source before the tests
are final; the Increment 2 review found the conjugate-ray sign and three defects that way.
Before a findings page or a plan revision is called done, run a panel of three reviewers on it:
one for accuracy against sources, one for reasoning and design, and one for style against
`.claude/writing_style.md`.  Apply the reviews in one pass, then read the result yourself.

## Reporting

Keep chat summaries short and plain, in the style guide's form.  Lead with the outcome, then what
was verified, then what is staged, then what is left.  Put measurements in a findings page, not in
the chat.  When Greg asks to see an estimator in action, `estimators_in_action.py` in the
experiments directory draws figures that can be sent to him.
