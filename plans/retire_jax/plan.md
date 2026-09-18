# Retiring mbirjax as the reference, plan of record

Date: 2026-09-12, revised 2026-09-13.  Status: APPROVED direction (Greg, 2026-09-13).
The four decisions in the last section are open, and nothing has been built.  Decision
recorded on 2026-09-12: mbirjax is no longer supported, so neither backward
compatibility with mbirjax nor a later port of mbirtorch changes back to it is a
requirement.

## Executive summary

mbirjax is no longer supported, so the tests that compared mbirtorch with it lose
their reference.  The 93 golden tests are kept and their reference changes: mbirtorch
generates its own baseline archives, the archives are committed, and every CI job
runs the tests.  Two consequences follow.

- Drift protection improves.  Today the golden tests are opt-in and never run in CI.
  After the change they run on every pull request, with tolerances about ten times
  tighter than today, set from the disagreement measured across the Mac and the CI
  platforms.
- mbirjax leaves the repository.  The jax generator scripts, the opt-in marker, the
  fixture files written by mbirjax, and the comments and pages that name mbirjax go.
  The credit page keeps its acknowledgment.

A baseline blesses current behavior, bugs included.  Tests that compute the answer
independently are therefore added afterwards: dense projection matrices on tiny
cells, finite differences of the prior's loss, and closed-form sinograms.

The work has four increments, each carried out from a written specification and
reviewed before the next starts:

1. the baseline script, the committed archives, the converted tests, and the removal
   of the opt-in machinery and the jax scripts;
2. tolerances measured on the Mac and in CI;
3. the four decisions applied and every remaining mention of mbirjax rewritten or
   deleted;
4. the correctness tests, added one operator group at a time.

## Status by increment

| Increment | Delivers | Status |
|---|---|---|
| 1 | `tests/generate_baselines.py`, two committed archives, 93 tests reading them, the marker and jax scripts removed, the maintenance page rewritten | Not started |
| 2 | Tolerances set from the measured disagreement on the Mac and in the three CI jobs | Not started |
| 3 | The four decisions applied, remaining mbirjax mentions rewritten or deleted | Not started |
| 4 | Correctness tests without a reference, one operator group per specification | Not started |

## Rule for the code

Nothing from this plan goes into the code.  Docstrings, comments, and Sphinx pages
describe what the code does, in plain words.  They do not name this plan or its
increments, they do not use its vocabulary as labels, and they carry no history: no
dates, no names, and no statement that mbirjax was once the reference.  A comment
explains a rule the code follows only when a later change would otherwise break it,
and then it states the rule itself.  The credit page is the one place that names
mbirjax, subject to the third decision.

Citations.  File paths are given from the root of the mbirtorch repository, so
`tests/test_cone.py` means `mbirtorch/tests/test_cone.py` in the sibling checkout.
The port's original parity gates are described in
`plans/archive/torch_port/port_plan.md` in this repository.  This plan closes them.

## Purpose

The golden tests compared mbirtorch with mbirjax to prove the port.  The port is
complete, and mbirtorch now changes in ways mbirjax will not follow.  The automatic
reconstruction geometry moved on 2026-09-12 to center the volume on the illuminated
detector band, and the geometry viewer and the point projector exist only in
mbirtorch.  Every such change either weakens the comparison or forces a matching
change in mbirjax.  The goal of this plan is to keep the protection against drift
that the golden tests give, and to remove mbirjax from the repository.

Two kinds of protection are wanted, and they are distinct.  Regression protection
means that an unintended numerical change from a refactor, a kernel change, or a
dependency update is caught, by comparing against a baseline of mbirtorch's own
outputs.  Correctness protection means that a wrong answer is caught regardless of
history, by comparing against an independent computation such as a dense matrix or
a closed form.  The golden tests gave only the first kind, with mbirjax as the
baseline.  This plan keeps the first kind with mbirtorch as the baseline, and adds
the second kind where the operators allow it.

## What exists today

The golden material is a family of files.  One archive,
`tests/goldens/golden_64x64x64.npz`, holds the projector, prior, and reconstruction
references for all five geometries.  A second archive,
`tests/goldens/preprocess_goldens.npz`, holds the preprocess chain, the metal artifact
reduction chain, stripe removal and segmentation, the hyperspectral and view-selection
modules, the demo data, and the split-sinogram reconstruction.  Five small HDF5 files
written by mbirjax pin the on-disk formats.  Two scripts, `tests/generate_goldens.py`
and `tests/generate_preprocess_goldens.py`, write these files, and both run only in
the mbirjax conda environment.

| Fact | Value |
|---|---|
| Golden tests | 93, in 14 test files |
| Run by default | No: the pytest options in `pyproject.toml` deselect the `goldens` marker |
| Run in CI | No: the archives are gitignored, and CI has no mbirjax |
| Result on Greg's Mac, 2026-09-12 | 93 passed in 51 s |
| Archive sizes | 10.7 MB and 1.1 MB, plus 130 KB of HDF5 files |
| Parallel-beam single operations, measured against tolerance | about 1e-6 against 1e-4 |
| Parallel-beam five-iteration reconstruction, measured against tolerance | about 1e-5 against 1e-3 |

These facts have two consequences.  The gate runs only when someone passes the marker
on a machine that holds the archives, so the pull-request checks enforce none of it.
The tolerances are about a hundred times looser than the measured agreement.  They are
loose because two frameworks computing in float32 cannot agree more closely.  A drift
fifty times larger than today's disagreement would therefore pass.  The comment above
the pytest options already says these tests retire once intentional divergence begins.

The 93 tests fall into these groups:

- single projector operations for each geometry, namely sparse forward, sparse back,
  full forward, and the Hessian diagonal;
- the direct reconstructions, FBP and FDK;
- the automatic reconstruction geometry and the automatic regularization values;
- the seeded five-iteration reconstruction with its per-iteration traces;
- the denoiser and the phantom generator;
- the preprocess chain from scans to sinogram, metal artifact reduction, and stripe
  removal and segmentation;
- the hyperspectral and view-selection modules;
- the demo data and the split-sinogram reconstruction;
- the HDF5 files, five of which read files written by mbirjax.

## The design

### Baseline archives generated by mbirtorch

One script, `tests/generate_baselines.py`, replaces the two jax scripts and runs in the
mbirtorch environment.  It writes two archives, `tests/baselines/models.npz` and
`tests/baselines/preprocess.npz`, with the same keys the golden archives hold today,
except that the jax version is replaced by the mbirtorch commit, the torch version,
and the platform.  The archives are committed, so a checkout has them and CI runs the
tests.  The standard therefore becomes mbirtorch at a named commit.

The archives must stay small enough to commit repeatedly.  The models archive is
10.7 MB today because its cells are 64 and 48 voxels wide.  The script shrinks the
cells, for example to 32 and 24, and the increment measures the result.  The target is
under 3 MB for both archives together.  A cell must stay large enough that every
test's quantity is still exercised, for example that a five-iteration reconstruction
still moves through several partitions.

Generation is reproducible.  The script runs on CPU and pins torch to one thread.
Float32 sums differ at about 1e-7 between thread counts, and one thread makes the
archive repeatable on the same machine.  When an archive already exists, the script
prints the largest relative change for every key against it and refuses to overwrite
unless it is run with `--replace`.  A commit that replaces an archive quotes those
numbers in its message, so a re-baseline is a reviewed step and not a side effect.

### The tests

The 93 tests keep their bodies and change their reference.  Each test file loses the
skip that fires when the archive is missing.  The archive is committed, so a missing
archive means a broken checkout, and the test fails.  The `goldens` marker, the pytest
option that deselects it, and the `RUN_GOLDENS` branch of `dev_scripts/run_tests.sh`
are removed, and the tests run in every job of the CI workflow.
`tests/test_vs_goldens.py` becomes `tests/test_parallel.py`, beside `test_cone.py`,
`test_multiaxis.py`, and `test_translation.py`.  Docstrings and comments that describe
a comparison with mbirjax describe a comparison with the baseline instead.

Tolerances are measured rather than guessed.  Every test keeps printing its measured
value.  The archives are generated on Greg's Mac, and the tests then run on the Mac
and in the three CI jobs, which cover two Python versions and the oldest supported
torch on Linux.  Each tolerance is set to ten times the largest disagreement observed
across those runs, rounded up to a power of ten.  The disagreement comes from platform
differences in float32 arithmetic, so the tolerances are expected to land near 1e-5
for single operations and 1e-4 for iterated results.  That is about ten times tighter
than today.  A later torch update that moves a result past its tolerance is a signal
to examine, and the re-baseline is the deliberate step above.

### Correctness tests without a reference

A baseline blesses current behavior, bugs included, so the second kind of protection
is added for the operators that allow an independent computation.  Each of these
tests knows the answer without any archive:

- A dense projection matrix built on a tiny cell, by forward projecting one voxel at
  a time, checks the sparse forward projection, the back projection, and the Hessian
  diagonal against matrix products, for every geometry.
- Finite differences of the qGGMRF loss check its gradient and Hessian functions on a
  small random volume.
- A sphere or cylinder phantom has a closed-form sinogram in parallel and cone
  geometry, and the projection must match it to discretization accuracy.
- Cone beam at a very large source distance must agree with parallel beam.
- FBP of a smooth phantom must recover it to a tolerance set by the discretization.

The adjoint tests, the triton kernel parity tests, and the point-projection tests are
already of this kind, and the increment that adds the rest first records which
operators they cover.

### Removing mbirjax from the repository

The jax generator scripts, the gitignore lines for the golden directory, the marker,
and the `RUN_GOLDENS` branch go.  The section of `docs/source/dev_maintenance.rst`
that describes the golden procedure is rewritten to describe the baseline script.  The
five tests that read HDF5 files written by mbirjax go with their fixture files, and
the loader's acceptance of the tag `mbirjax_preprocessing_v1` goes with them, subject
to the second decision below.

Comments, docstrings, and Sphinx sources that name mbirjax are rewritten to state the
rule itself.  A comment that says the code pads rows the way mbirjax does becomes a
comment that says how the code pads rows.  Sphinx source comments that record a
divergence from mbirjax's pages are deleted.  Three kinds of mention stay: the credit
and citation on `docs/source/credits.rst`, the name of the shared performance
dashboard repository, and the pending documentation pages, which are outside this
plan.

## Increments

Each increment is carried out from a written specification, reviewed, and staged by
explicit file name.  Every test suite involved passes before staging.

1. **The baseline and the converted tests.**  Write `tests/generate_baselines.py`,
   generate both archives at the current commit, convert the 14 test files to read
   them with today's tolerances, remove the skip logic, the marker, the pytest option,
   and the run-script branch, delete the two jax scripts and the gitignore lines, and
   rewrite the unit-test section of the maintenance page.  Gate: the full suite passes
   on the Mac with the 93 converted tests included and no test skipped for a missing
   archive, the archives together are under 3 MB, and Sphinx builds with warnings as
   errors.
2. **Measured tolerances.**  Commit the archives, run the CI workflow on the branch by
   dispatch or through a draft pull request to `prerelease`, collect the printed values
   from the Mac and the three CI jobs, and set every tolerance by the rule above.
   Gate: CI is green with the new tolerances, and each test module's docstring records
   the measured floors, as `test_vs_goldens.py` does today.
3. **The removal.**  Apply the decisions below, and rewrite or delete every remaining
   mention of mbirjax outside the three kinds that stay.  Gate: a case-insensitive
   search of the repository for the name returns only those mentions, and the full
   suite and the documentation build pass.
4. **Correctness tests.**  Add the tests of the third design section, one operator
   group per specification, starting with the dense-matrix check because it covers
   the most operators.  Gate per group: the new tests pass on CPU and on the Mac's
   MPS device, and their tolerances are stated with the reason for each.

Increments 1 and 2 remove the dependence on mbirjax.  Increment 3 finishes the
removal.  Increment 4 is open-ended and can proceed alongside other work.

## Decisions for Greg

1. **Where the baseline lives.**  The plan commits the archives to the repository and
   keeps them small.  The alternative generates them at test time from a pinned
   mbirtorch commit in a git worktree, which stores no binaries and allows tighter
   tolerances, but the pinned commit ages as torch moves and doubles the CI install.
   Recommendation: commit the archives.
2. **Files written by mbirjax.**  Five tests read HDF5 files that mbirjax wrote, and the
   cone preprocessing loader accepts the tag `mbirjax_preprocessing_v1`.  With no
   backward compatibility required, the tests, the fixture files, and the tag can go.
   Recommendation: remove all three.
3. **The credit.**  `docs/source/credits.rst` states that MBIRTorch is a port of
   MBIRJAX and asks users to cite it.  That is attribution rather than support.
   Recommendation: keep it.
4. **Cell sizes.**  The archive size target of 3 MB decides the cells.
   Recommendation: let the first increment measure sizes for cells of 32 and 24
   voxels and report before the archives are committed.

## Out of scope

The pending documentation pages under `docs/source/_pending/`, which were copied
from mbirjax for modules not yet written in mbirtorch, and the mace4d migration plan
in the mbirtorch repository concern ports from mbirjax and are separate decisions.
The shared performance dashboard keeps its repository name.  The partition sequence
still draws from numpy's global random generator, which was kept for the
iteration-for-iteration comparison.  Changing it is now possible but is not part of
this plan.
