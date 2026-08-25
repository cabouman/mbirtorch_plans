# mg64 run record

Two submissions.  The finding and its reading are in
`plans/torch_port/active/multigpu_findings.md` §1.51; this file holds
the run detail.

## mg64, the row-bound change verified (job 15487827)

* 2026-08-24, node h000, one H100 80GB, 1 minute 17 seconds by sacct,
  exit 0.  The card read 1980 MHz and 40 C at the end of the job.
* The library under test is the working checkout's tip `eee646a` plus
  the row-bound change to `triton_multiaxis.py` and the note change to
  `_utils.py`, both copied to the scratch tree and md5-verified.  The
  probe records the sha256 of the kernel file it ran against:
  be7ba7dd0291669ec12207a563886acd867edc0a431fca4eb8a0e790a3debbab.
  torch 2.13.0+cu130, triton 3.7.1.
* Cells, models and inputs are mg62's; see that run record.
* Two gates in one job.  The CUDA-only kernel tests ran first and a
  failure would have stopped the job before it measured anything:
  `test_triton_multiaxis`, `test_triton_cone`, `test_triton_parallel`,
  `test_kernels_sharded`, `test_multiaxis` and `test_adjoint`, reading
  166 passed and 16 skipped in 47 seconds.  The probe then ran.

## The first submission, and why it was rerun (job 15487634)

The first submission held the kernel-to-torch-body comparison to 1e-5
at every cell and reported three of four cells failing, at 1.26e-05,
2.35e-05 and 2.48e-05.  The gate was wrong, not the values.

`tests/test_triton_multiaxis.py` records the reason at its
multi-row-chunk test and gates that test at 1e-4.  The trapezoid weight
subtracts two row coordinates of size about `num_rows_r`, and the
kernel and the torch body round the `m0 + slope * k` forming them
differently, so the weight carries an absolute perturbation of about
`num_rows_r` times float32 eps.  That is 5e-5 at 448 rows.  The
readings scale with the row count as that predicts: 6.2e-06 at 113
rows, 1.26e-05 at 224, 2.33e-05 at 448 and 2.48e-05 at 449.

The rerun uses the test file's own two-tier rule, splitting at the
forward kernel's row tile, and adds the comparison that settles the
question directly: the kernel's PREVIOUS behaviour against the same
torch body.  Everything else is unchanged, and the timings agree
between the two submissions.

## Part A, what Triton compiled at each bound

Both bounds compiled in one process, before any other launch warmed
either divisibility class.

| bound | registers | spills | shared | PTX bytes | cubin bytes |
|---|---|---|---|---|---|
| 464, a multiple of 16 | 32 | 0 | 0 | 40,494 | 41,480 |
| 449, not a multiple of 16 | 32 | 0 | 0 | 42,028 | 44,424 |

The two variants use the same registers and neither spills.  The
unspecialized one carries 4 percent more PTX and 7 percent more cubin.
What the specialization buys was not isolated beyond that.

## Part B, values against the torch body

The torch body is the value reference.  Both the edited kernel and the
kernel's previous behaviour are compared against it, at a view batch of
8 views, which keeps the torch body's gather transient small.

| cell | rows | gate | edited | previous |
|---|---|---|---|---|
| 129x113x97 | 113 | 1e-05 | 6.24e-06 | 6.21e-06 |
| 256x224x192 | 224 | 1e-04 | 1.26e-05 | 1.26e-05 |
| 512x448x384 | 448 | 1e-04 | 2.33e-05 | 2.34e-05 |
| 513x449x385 | 449 | 1e-04 | 2.48e-05 | 2.48e-05 |

The two columns agree at every cell, so the gap against the torch body
is the kernel's own and the change did not move it.

## Part C, values against the kernel's previous behaviour

| cell | difference |
|---|---|
| 129x113x97 | 4.84e-07 |
| 256x224x192 | 8.57e-07 |
| 512x448x384 | 6.94e-07 |
| 513x449x385 | 6.77e-07 |

All four sit at the kernel's own repeat-to-repeat spread, which comes
from its atomic adds.

## Part D, times end to end

Through `sparse_forward_project`, against mg62's readings on the same
node for the same cells.

| cell | mg62, before | mg64, after | ratio | transient |
|---|---|---|---|---|
| 129x113x97 | 5.0 ms | 2.3 ms | 0.46 | 21.2 MB |
| 256x224x192 | 20.4 ms | 20.1 ms | 0.99 | 147.0 MB |
| 512x448x384 | 307.0 ms | 307.1 ms | 1.00 | 758.3 MB |
| 513x449x385 | 1003.4 ms | 316.6 ms | 0.32 | 765.6 MB |

Both cells whose detector row count is a multiple of 16 are unchanged.
Both cells whose row count is not are faster, and the smaller of them
by a factor of 2.2 rather than 3.2 because at 113 rows the kernel is
not yet the whole cost.  The non-dividing penalty is now 1.03 times.

## Result file

`rows/mg64_row_bound_verify_h000_20260824.json`, md5
77b2dab29cdad7d676cdb023b13fee5b, verified after the copy.
