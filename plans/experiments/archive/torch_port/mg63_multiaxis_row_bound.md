# mg63 run record

One run, first submission.  The finding and its reading are in
`plans/torch_port/active/multigpu_findings.md` §1.51; this file holds
the run detail.

## mg63, the row-bound remedy measured (job 15487437)

* 2026-08-24, node h000, one H100 80GB, 58 seconds by sacct, exit 0.
  The card read 1980 MHz and 43 C at the end of the job.
* Same library, environment and model construction as mg62; see that
  run record.  Both cells asserted the Triton forward body bound
  before any measurement.
* The probe reimplements the shipped forward wrapper with one change:
  the kernel's row-mask bound and the grid's row extent take the
  padded row count instead of the real one.  The geometry builders
  keep the real row count, because the slice-to-row map is anchored on
  the detector the model has.  No library file was edited.
* Protocol is mg62's: one untimed warm-up, five timed calls, device
  synchronize around each.

## Part A, the remedy end to end

Timed through `sparse_forward_project`, which is the call the
dashboard measures.

| cell | shipped | row bound padded | ratio | transient | value difference |
|---|---|---|---|---|---|
| 512x448x384 | 306.1 ms | 306.3 ms | 1.00 | 758.3 MB | 9.82e-07 |
| 513x449x385 | 1002.8 ms | 314.4 ms | 3.19 | 765.6 MB | 1.16e-06 |

The dividing cell is unchanged, which is the design: a row count
already a multiple of 16 takes the path it took before.  The value
differences sit inside the design's 1e-5 single-shot gate and are the
size of this kernel's own repeat-to-repeat spread, which comes from
its atomic adds.

## Part B, what Triton compiled

The walk reports a compiled variant only when a launch produced a new
one.  Bounds 464, 448 and 447 produced none, because Part A had
already compiled the two divisible-bound variants and 447 shares its
specialization class with 449.  The one new variant, at bound 449,
reports 32 registers, 0 spills, 0 bytes of shared memory, 42,028 bytes
of PTX and 44,424 bytes of cubin.

Register spilling is therefore not the mechanism.  What the divisible
bound buys was not isolated further.

## Part C, where the threshold sits

The row bound swept at the dividing cell, everything else fixed.

| bound | median | multiple of 16 | multiple of 8 |
|---|---|---|---|
| 440 | 229.04 ms | no | yes |
| 441 | 229.95 ms | no | no |
| 444 | 232.42 ms | no | no |
| 446 | 232.39 ms | no | no |
| 447 | 232.44 ms | no | no |
| 448 | 73.64 ms | yes | yes |
| 449 | 230.33 ms | no | no |
| 450 | 230.96 ms | no | no |
| 452 | 232.45 ms | no | no |
| 456 | 232.97 ms | no | yes |
| 460 | 235.34 ms | no | no |
| 464 | 75.74 ms | yes | yes |

Only the two multiples of 16 are fast.  Multiples of 8 buy nothing.

Bounds below the real row count do slightly less work and bounds above
it slightly more, at most 3.6 percent either way, which is small
against the factor being measured.

## Part D, the back body under the same ablation

| cell | variant | median | multiple of 16 |
|---|---|---|---|
| 512x448x384 | baseline, bound 448 | 34.90 ms | yes |
| 512x448x384 | bound 447 | 34.88 ms | no |
| 512x448x384 | bound 449 | 34.94 ms | no |
| 513x449x385 | baseline, bound 449 | 40.53 ms | no |
| 513x449x385 | bound 448 | 40.55 ms | yes |
| 513x449x385 | bound 450 | 40.53 ms | no |
| 513x449x385 | bound 464 | 40.59 ms | yes |

The back body's row bound moves nothing.  Its own non-dividing
penalty, 40.53 against 34.90 ms, is 1.16 times and matches the
dashboard's 1.17 times, so that residue is real and has another cause.

## Result file

`rows/mg63_multiaxis_row_bound_h000_20260824.json`, md5
d807237d16ecc8119a51736c059dd3f3, verified after the copy.
