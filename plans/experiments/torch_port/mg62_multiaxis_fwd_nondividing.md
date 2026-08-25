# mg62 run record

One run, first submission.  The finding and its reading are in
`plans/torch_port/active/multigpu_findings.md` §1.51; this file holds
the run detail.

## mg62, what the non-dividing size costs the multiaxis forward (job 15487404)

* 2026-08-24, node h000, one H100 80GB, 1 minute 32 seconds by sacct,
  exit 0.  The card read 1980 MHz and 39 to 42 C at both ends of the
  job, so it was not throttled.
* The library under test is the committed tip `eee646a` at
  /scratch/gautschi/buzzard/torch_p3/mbirtorch_src, installed editable
  into the shared environment at job start.  All 40 package files were
  md5-verified against the working checkout before submission.
  torch 2.13.0+cu130, triton 3.7.1.  Both cells asserted the Triton
  forward body bound.
* The models and inputs are the metrics harness's own, so the run
  measures the surface the dashboard measures: azimuths evenly spaced
  over half a turn, one elevation of 25 degrees for every view, the
  recon shape taken from a plain parallel model at the same sinogram
  size, full field-of-view indices under the region-of-reconstruction
  mask, and cylinders from `numpy` generator seed 0.
* Protocol: one untimed warm-up, then five timed calls, each bracketed
  by a device synchronize and each result freed before the next
  allocation.  Rows carry the minimum, median and maximum.
* Memory is reported two ways.  Several cells stay alive in one
  process, so the peak counter reads resident plus transient; the
  transient alone is the figure comparable across cells and against the
  dashboard.

## Part A, the dashboard numbers reproduced

| cell | median | transient | view batch |
|---|---|---|---|
| 129x113x97 | 5.0 ms | 53.2 MB | 128 |
| 256x224x192 | 20.4 ms | 148.2 MB | 128 |
| 512x448x384 | 307.0 ms | 758.3 MB | 128 |
| 513x449x385 | 1003.4 ms | 766.4 MB | 128 |

The dashboard read 306.7 ms and 1005.1 ms at the last two cells, so
the probe reproduces it.

## Part B, where one projection's time sits

| cell | body calls | in bodies | assembling | elsewhere |
|---|---|---|---|---|
| 512x448x384 | 4 | 305.9 ms | 0.8 ms | 0.1 ms |
| 513x449x385 | 5 | 1002.7 ms | 0.8 ms | 0.1 ms |

Per call, the dividing cell reads 74.0, 78.9, 79.3 and 73.7 ms.  The
non-dividing cell reads 230.7, 268.1, 270.6, 230.9 and 2.4 ms.  The
last call is the ragged batch of one view.

## Part C, the compiled kernels

The cache walk did not reach Triton's compiled objects in this run and
reported nothing.  mg63 fixed the walk.

## Part D, one integer changed at a time

Each arm holds the cell's own arguments at a view batch of 128 views
and changes one integer.  The two band arms add zero columns past the
slice count, so they preserve values and their outputs were compared;
the rest change a bound and are timing probes only.

| cell | variant | median | ratio | value difference |
|---|---|---|---|---|
| 512x448x384 | baseline | 73.86 ms | 1.00 | 0 |
| 512x448x384 | band 449 | 74.20 ms | 1.00 | 7.19e-07 |
| 512x448x384 | num_rows_r 448 | 74.01 ms | 1.00 | |
| 512x448x384 | num_rows_r 447 | 231.82 ms | 3.14 | |
| 512x448x384 | num_channels 383 | 74.00 ms | 1.00 | |
| 512x448x384 | num_slices 447 | 73.96 ms | 1.00 | |
| 512x448x384 | num_pixels 115152 | 74.03 ms | 1.00 | |
| 512x448x384 | views 1 | 1.17 ms | | |
| 513x449x385 | baseline | 231.32 ms | 1.00 | 0 |
| 513x449x385 | band 464 | 230.88 ms | 1.00 | 7.73e-07 |
| 513x449x385 | band 450 | 230.90 ms | 1.00 | 6.77e-07 |
| 513x449x385 | num_rows_r 464 | 76.90 ms | 0.33 | |
| 513x449x385 | num_rows_r 448 | 74.66 ms | 0.32 | |
| 513x449x385 | num_channels 384 | 230.57 ms | 1.00 | |
| 513x449x385 | num_slices 448 | 231.01 ms | 1.00 | |
| 513x449x385 | num_pixels 115776 | 230.96 ms | 1.00 | |
| 513x449x385 | views 1 | 2.40 ms | | |

The band arms allocate their padded copy once per arm rather than once
per call, so their rows price the kernel and not the copy.  Part E
priced the copy.

## Part E, the band-padding remedy

| cell | median | transient | value difference |
|---|---|---|---|
| 512x448x384 | 307.4 ms | 758.3 MB | 1.03e-06 |
| 513x449x385 | 1004.6 ms | 971.3 MB | 9.67e-07 |

Padding the values band changes no time and costs 205 MB.  This arm
tested the hypothesis the run started with, which Part D had already
refuted.

## Result file

`rows/mg62_multiaxis_fwd_nondividing_h000_20260824.json`, md5
384fb517ccb08ab7d353fdd78c0ba6c0, verified after the copy.
