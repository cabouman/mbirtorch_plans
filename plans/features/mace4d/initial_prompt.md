# Starting prompt for the MACE4D port

You are helping a student port MACE4D, a 4D reconstruction method, from
mbirjax to mbirtorch.  The design is decided and written down.  Your job is
to implement it stage by stage, test each stage as the plan says, and report
in plain English.  Read this file first, then the files it names, in order.

## The repositories

Three checkouts sit side by side.

- `mbirtorch/` is the library you change.  Work on the branch
  `mace4d_jax2torch`, which holds the MACE loop and agents in
  `experiments/drunet/` that the plan builds on.  Install it in a Python
  environment as the repository's README describes, and run the tests from
  the repository root with `pytest` (see `dev_scripts/run_tests.sh`).  CPU
  only is fine for every stage but the last.
- `mbirtorch_plans/` holds the plans and the measurement records.  This
  file is in `plans/features/mace4d/`, beside the plan.
- `mbirjax/` is the old library.  It is the reference for what the code
  must compute, and nothing else.  It is no longer supported, and no
  change goes into it.  Clone it beside the other two, read-only, from
  `github.com/cabouman/mbirjax`.

## Read these, in this order

1. `plans/features/mace4d/mace4d_migration_plan_v2.md` in `mbirtorch_plans`.
   This is the plan of record.  Section 2 gives the design, Section 4 the
   stages with their tests and exit conditions, and the status table near
   the top says where the work stands.
2. `plans/features/mace4d/mace4d_plan_evaluation.md`, beside it.  It
   explains why the design is what it is, and Section 5 holds the
   measurements that decided it.  Section 8.2 has the consensus update in
   two forms; the second is the one you implement.
3. `plans/features/mace4d/mace4d_migration_plan.md`, the first plan.
   Sections 1 and 3 describe what the mbirjax code computes and which
   mbirtorch functions it can use, with line references.  Its design
   sections are superseded.
4. `.claude/writing_style.md` in `mbirtorch_plans`.  Every record, report,
   and status message follows it.
5. In `mbirtorch`: `mbirtorch/denoising.py`, `mbirtorch/qggmrf.py`,
   `mbirtorch/projectors.py` (the part about `maybe_compile`), and
   `experiments/drunet/mace.py` and `agents.py`.
6. In `mbirjax`: `mbirjax/mace4d.py` and `tests/test_mace4d.py`.
7. The prototype of the batched denoiser:
   `plans/experiments/features/mace4d/m4d1_batched_denoiser_options.py`
   in `mbirtorch_plans`.  Its two batched functions equal the package
   formulas bit for bit, and Stage 0 starts from them.

**IMPORTANT — workflow protocol:** stage only (`git add` by explicit file
name), never `git commit` unless the user directs it (the user commits from
PyCharm).  Shared checkouts — never `git add -A`; verify staged-file
lists at report time.  Cluster jobs are pre-authorized during the agreed
investigation.  Durable records and summary status reports in Alley style — reread
`.claude/writing_style.md` before drafting; plan entries and chat
summaries stay short and plain, with run detail in script comments or a
companion `.md` beside the script.  Have opus carry out well-defined
plans, then review.

Read for orientation (code and measured results over recollection or .md files):
1. `.claude/claude_prompt.md`, `.claude/cluster_use.md`.

## How the work is organized

The plan has eight stages.  Do them one at a time, in the order Section 7
of the plan gives.  A stage is done when its exit condition in Section 4
holds and its tests pass.  When a stage is done, update its row in the
plan's status table, write a short status report, and stop for Greg's
review before starting the next stage.  Commit on a branch of your own
clone at the end of each stage.

Before changing a file, say what you are about to change and why, in a few
sentences.  When the plan and the code disagree, or when the plan leaves
something open, ask rather than guess.  A question that changes the design
goes to Greg.

## Rules for the code

- Plain English in docstrings and comments.  They describe what the code
  does.  They do not name the plan or its stages, they do not use the
  plan's vocabulary as labels, and they carry no dates, names, or history.
  A comment states a rule the code follows only when a later change would
  otherwise break it.
- Minimal, local changes.  `vcd_subset_denoiser`, the single-image path
  of `denoise`, and their golden tests stay as they are.  New behavior
  goes into new functions.
- No new dependencies.  The GIF writer uses Pillow, imported inside the
  function as `gen_text_phantom` in `mbirtorch/utilities.py` does.
- Every name exported through the lazy loader in `mbirtorch/__init__.py`
  needs its line in the `TYPE_CHECKING` block; a test enforces this.
- Keep the arithmetic of the existing functions where you copy it.  The
  batched functions repeat the formulas of the single-image ones with a
  batch axis added, and nothing rearranged.

## Rules for tests and measurements

- Exact equality is never the gate for a computed float value.  Gate on
  the relative maximum difference, `max|a - b| / max|b|`, at the
  tolerance the plan states, and print the difference the test observed.
  Exact equality is right only for data movement, such as a permutation
  and its inverse.
- Seed the global numpy generator before any comparison that draws a
  pixel partition; the partitions come from it.
- Run the tests of the stage on every device available to you, and the
  whole suite before you declare a stage done.  The golden tests are
  opt-in and are being retired by another plan; leave that machinery
  alone.
- Stage 8 runs on a cluster GPU.  It needs Greg's go and the group's
  cluster guide, `.claude/cluster_use.md` in `mbirtorch_plans`.

## Rules for writing

- Status reports and records: short sentences, one idea each, ordinary
  words, no metaphors.  Open with the conclusion.  Numbers go in a small
  table or on their own line, never packed into a paragraph.
- A measurement script gets a companion `.md` with the same base name,
  holding the command, the environment, and the numbers, as the m4d
  scripts in `plans/experiments/features/mace4d/` do.  Run detail lives
  there, not in the plan and not in code comments.
- Copy a number into a record only from the file that produced it, never
  from memory.

## Start here: Stage 0

Stage 0 adds `QGGMRFDenoiser.denoise_stack` and the two batched functions
it runs on.  Section 2.1 of the plan specifies them, and the Stage 0 entry
in Section 4 lists the tests and the exit condition.  The steps:

1. Read `vcd_subset_denoiser` and the single-image sweep inside `denoise`
   in `mbirtorch/denoising.py`, and `qggmrf_gradient_and_hessian_at_indices`
   in `mbirtorch/qggmrf.py`, until you can say what each line does.
2. Read `grad_hess_batched` and `subset_update_batched` in the m4d1
   prototype, and `run_option_d`, which is the sweep with the per-volume
   stopping test.  Move them into the package under the names the plan
   gives, with docstrings written for a user of the library.
3. Write `auto_batch_size` and `denoise_stack` as Section 2.1 specifies.
   For the batch sizing, read how `denoise` prices its plan through
   `_apply_device_policy` and `_memory_ledger.py`, and price one volume's
   plan the same way.
4. Write the Stage 0 tests in `tests/test_denoiser.py`.
5. Run `tests/test_denoiser.py` on every device you have, then the whole
   suite.  Report the observed differences from the tests, the batch
   sizes chosen on your devices, and anything that surprised you.

Do not start Stage 2 or Stage 3 until Stage 0 has been reviewed.
