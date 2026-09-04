# Implementation prompt: geometric calibration utilities

This file starts a new session that implements the geometric calibration plan.  It points to the
files that hold the details and repeats none of them.  Paths are relative to the root of the
`mbirtorch_plans` repository unless a path starts with `mbirtorch/`, which means the sibling
`mbirtorch` repository.

## Where to run

Start the session with the `mbirtorch` repository as the working directory, so that the project
memory for that directory loads.  The memory index records the local test environment, the cluster
access that works, and the lessons from earlier sessions.  `mbirtorch_plans` and `mbirtorch_metrics`
are sibling checkouts; `mbirjax` is a read-only reference.  `.claude/initial_prompt.md` describes
the roles of these repositories.

## Read first, in this order

1. `.claude/claude_prompt.md`: how Greg wants to work, and what he relies on the session for.
2. `.claude/writing_style.md`: mandatory for every durable record, code comment, and summary.
   Reread it before drafting anything, and again before the revision pass it requires.
   `.claude/writing_style_charlie.md` holds more examples of the same rules.
3. `.claude/lessons.md`: engineering rules from the sharding effort and the torch port.
4. `plans/features/geometric_calibration/geometric_calibration_plan.md`: the plan of record,
   version 3, accepted on 2026-09-03.  It holds the design, the API, the workflows, the
   parameter-system changes, the increments with their gates, the validation plan, and the risks.
5. `plans/features/runtime_offsets/runtime_offsets_findings.md` and the diff beside it: the
   prototype that makes the detector offsets call-time inputs.  The prototype also sits in a git
   worktree of the `mbirtorch` repository under `.claude/worktrees/`, which may have been removed;
   the diff is the durable record.
6. `plans/features/leap_comparison/leap_comparison.md`: the comparison that motivates this work.
   Its first two entries under "High-value LEAP features missing in mbirtorch" are the relevant
   ones.  The LEAP inventory beside it, in `leap_comparison_sources/`, records what LEAP's
   calibration utilities compute, with pinned source links.
7. `.claude/cluster_use.md` and `.claude/gpu-resources.md`: how to run cluster jobs when an
   increment's gate needs a GPU.  `plans/experiments/features/leap_comparison/` holds working
   job scripts and the environment recipe from the LEAP benchmark, which can be copied.

## Two decisions to confirm with Greg before starting

The plan marks two items for Greg to confirm: the split of the parameter table into user-facing
and research parameters, and the order of the two strands (the calibration strand and the
parameter-system strand are independent).  Ask at the start of the session, then follow the plan.

## Constraints the plan records

Follow the plan's increments in order within a strand, and stop for Greg's review at the end of
each increment.  Each increment names its files, its tests, and its gates.  A gate that needs a
GPU runs on the cluster as a batch job.

One constraint was added after the panel review and is recorded in the plan's "Candidate
evaluation" subsection: a sinogram correction must not allocate a second full-size sinogram.
Stream view batches and write the result in place, or stream into the device-resident sinogram.

Scripts and job files for this work go in `plans/experiments/features/geometric_calibration/`.
Findings pages and status updates go in `plans/features/geometric_calibration/`.  Run detail goes
in script comments or in a companion `.md` file with the same base name as the script, not in the
plan.  A measured number appears in a durable record only after it was read from its source in
the same session, with the source cited beside it.

## Working with the shared checkouts

Other sessions edit and commit in the `mbirtorch` checkout at the same time.  Do code work on a
branch or in a git worktree, never on `greg_dev` directly.  Do not run the full test suite while
another session may be running it.  Before reporting, check `git status` and report the staged
list as it is at that moment, because another session may have committed or changed it.

## Git protocol: stage, but do not commit without authorization

Stage a file with `git add` when it is finished and verified, so that `git status` shows what is
ready for review.  Do not run `git commit` or `git push` unless Greg authorizes it in the
conversation.  An authorization covers only the files and the commit it names; it does not carry
over to later work.  When authorized, use the commit attribution line that the session's
instructions specify, and report the commit hash.

## Reviews and delegation

Use Opus subagents for drafting, harness writing, inventories, and result extraction, and keep
the main session for judgment: reviewing a harness before it runs, ruling on results, and reading
the final record.  Before a findings page or a plan revision is called done, run a panel of three
reviewers on it: one for accuracy against sources, one for reasoning and design, and one for style
against `.claude/writing_style.md`.  Apply the reviews in one pass, then read the result yourself.

## Reporting

Keep chat summaries short and plain, in the style guide's form.  Lead with the outcome, then what
was verified, then what is staged, then what is left.  Put measurements in a findings page, not in
the chat.
