# m4d4: the data-fit step against `prox_partition_advance`

Status: done 2026-09-20.  All three jobs completed ten iterations.

## Question

The 4D loop's data-fit step is the slow step in the timing recorded on slide 23
of the API deck (see `plan.md`, status update of 2026-09-19).  The suspected
cause is the partition schedule: with `prox_partition_advance=1.0` each MACE
iteration starts one entry further along the model's partition sequence, so
from the fourth iteration on every proximal-map call runs at the finest
partition, 128 subsets.  This experiment runs the same reconstruction at
`prox_partition_advance` 0.0, 0.5, and 1.0 and compares the time of the
data-fit step and the reconstructions.

## Setup

- Cluster harness: `/scratch/gautschi/buzzard/mace4d_adv/` on gautschi.
  `src/` is a clone of mbirtorch at `v0.1.1` (d75f13e), installed editable
  into a bare venv over the conda environment `~/.conda/envs/mbirtorch`
  (torch 2.13.0+cu130).  The clone is not named `mbirtorch`, because a
  directory of that name beside the script shadows the package as a
  namespace package.
- Data: `/depot/bouman/data/Lilly/4DCT/Phantom_30s_Run1_Dec2024/`, the 30 s
  phantom scan, at full resolution.
- Script: `m4d4_Lilly_recon_4d.py`, a copy of `nsi_4d/Lilly_recon_4d.py`
  from mbirtorch_applications (9a965ed) with one flag added,
  `--prox_partition_advance`, which is passed to `MACE4DModel.set_params`
  and written to `run_info.txt`.
- Job: `mace4d_adv.sbatch` in the harness (job files stay out of this
  repository), one job per value, four H100s of one node, 56 cores,
  `OMP_NUM_THREADS=56`, a compile cache per run.  The run command is

      python -u Lilly_recon_4d.py \
        --data_path /depot/bouman/data/Lilly/4DCT/Phantom_30s_Run1_Dec2024/ \
        --downsampling 1 --max_mace_iterations 10 --stop_threshold_change_pct 0 \
        --prox_partition_advance $ADV --output_path runs/adv_$ADV/output

  with every other flag at its default (six frames per rotation, overlap
  factor 2.0, all frames, `transmission_root` weights, sharpness 1.0 on the
  frames, denoiser sharpness 0.0, `nbr_weight_time` 1.0).  The stop
  threshold is zero so that every run completes ten iterations and the
  per-iteration times compare directly.  Each run computes its own initial
  image.
- Jobs: 16518019 (0.0), 16518020 (0.5), 16518021 (1.0), submitted
  2026-09-20.  Outputs under `runs/adv_<value>/`: `output/` holds the recon
  and the GIFs, `logs/<stem>/` holds `run_info.txt`, `timing_log.csv`, and
  `task_log.csv`; the Slurm log is `logs/adv_<jobid>.log`.

## What to record

- From `timing_log.csv` of each run: the data-fit sum, the denoise sum, and
  the wall time of each iteration, and the loop total.
- The relative difference between the three reconstructions, and a slice or
  GIF from each for a visual check.
- The subset counts the data-fit calls actually used, from the frames'
  partition sequence and the advance value.

## Results

The files under `results/m4d4/` are the source of every number here:
`timing_summary.csv` (per iteration and sums), `timing_log_adv_<v>.csv`
and `run_info_adv_<v>.txt` (each run's own log and settings),
`recon_differences.csv`, and `middle_slices.png`.

**The subset counts of the three data-fit calls of each iteration**, from
the default granularity and partition sequence:

| advance | it 1 | it 2 | it 3 | it 4 | it 5 | it 6 | it 7 | it 8 | it 9 | it 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 4/16/64 | 4/16/64 | 4/16/64 | 4/16/64 | 4/16/64 | 4/16/64 | 4/16/64 | 4/16/64 | 4/16/64 | 4/16/64 |
| 0.5 | 4/16/64 | 4/16/64 | 16/64/128 | 16/64/128 | 64/128/128 | 64/128/128 | 128/128/128 | 128/128/128 | 128/128/128 | 128/128/128 |
| 1.0 | 4/16/64 | 16/64/128 | 64/128/128 | 128/128/128 | 128/128/128 | 128/128/128 | 128/128/128 | 128/128/128 | 128/128/128 | 128/128/128 |

**Time over the ten iterations (seconds, sums over the four workers except
the wall columns):**

| advance | data fit | denoise | wall of the parallel part | wall per iteration total | whole run |
|---|---|---|---|---|---|
| 0.0 | 4119 | 2283 | 3094 | 3328 | 1.24 h |
| 0.5 | 6334 | 2050 | 3453 | 3663 | 1.32 h |
| 1.0 | 7171 | 2064 | 3673 | 3894 | 1.39 h |

The whole run includes the initial image, about 18 minutes in every run.
From iteration 4 on, one iteration's data fit takes 382 to 449 s at 0.0 and
780 to 792 s at 1.0; at 0.5 it climbs from 531 s to 781 s as the calls
reach the 128-subset entries.  The denoising time does not depend on the
advance.  The wall time gains less than the data fit does, because the four
workers also take the denoise slabs while the data fit runs.  Every denoised
volume ran the full 15 iterations in every run, so the denoiser's 0.05
percent stop rule never fired.

**The reconstructions.**  The consensus change at iteration 10 is 0.90
percent (0.0), 0.94 percent (0.5), and 0.99 percent (1.0), so no run had
reached the usual 0.2 percent stop, and the coarse schedule was not behind.
The three results differ pairwise by 0.63 to 0.69 percent in relative RMS,
less than one iteration's own change, with per-frame NRMSE between 0.54 and
1.7 percent.  The relative maximum difference is large, 0.34 to 0.36, so
single voxels differ a lot while the volumes agree; the middle slice of the
middle frame looks the same in all three (`middle_slices.png`).

## Against the run of 2026-09-17

Ziyun's run is job 16482614; its `run_info.txt`, job script, and
`timing_log.csv` are under `results/16482614/`.  What they say:

- Same scan, frame shape (260, 260, 728), 99 frames of 48 views, four
  H100s and 56 cores on one node, ten iterations completed, the driver's
  defaults (advance 1.0, stop threshold 0.2 percent).
- The initial image was read from the cache that job 16478297 wrote before
  it ran out of memory in the denoise phase of its first iteration: the
  batch estimate returned 921 volumes against 728 hyperplanes, so the
  batch was a whole orientation, and the device held four 18.15 GiB stacks
  at once.  That is the failure job 16522860 here reproduced.
- The code was Ziyun's working tree with an uncommitted patch: the batch
  estimate charges the output stack, the filter's output is made
  contiguous, and an orientation is cut into equal slabs.  Batch sizes
  were 364, 130, 130, two slabs per orientation.  The commit the tree
  stood on is printed in the job's `.out` log, which is not here.
- No thread setting was exported, so the run had one host thread, the
  same as every run here.  The compile caches were job-scoped, so cold, as
  here.
- The `.out` log (`16482614.out`) pins the rest: node h007, GPUs 2 to 5,
  mbirtorch at c06f165 with six modified files (the patch above), the
  driver at c1e2576, and **torch 2.14.0+cu130**.  This harness runs
  Greg's conda environment with torch 2.13.0+cu130.

Per iteration at iterations 4 to 10, from the two timing logs:

| run | total | parallel part | denoise | data fit | ten iterations |
|---|---|---|---|---|---|
| 2026-09-17, advance 1.0, batches 364/130/130 | 152.2 s | 150.0 s | 59.4 s | 533.9 s | 24.2 min |
| 2026-09-20, v0.1.1, advance 1.0, batches 73/26/26 | 405.6 s | 383.5 s | 196.8 s | 785.4 s | 64.9 min |
| 2026-09-20, v0.1.1, advance 0.0, batches 73/26/26 | 340.1 s | 316.2 s | 227.6 s | 417.2 s | 55.5 min |

At the same advance the denoising is 3.3 times slower here, the data fit
1.5 times, and the time outside the parallel part 22 s against 2.2 s.  The
first iteration, which carries the compile, shows the same ratio in the
data fit (429 s against 268 s), so it is not a warm-cache effect.  The
slab budget accounts for about 15 percent of it (below).  What remained
open after this was the code and the environment; the runs below settle
the code, and the `.out` log names the environment difference: torch
2.14.0+cu130 there, 2.13.0+cu130 here.

## Follow-up: isolating the difference from 2026-09-17

Greg's rule for this part: reproduce the 2026-09-17 result first, then
change one variable at a time.  The variables are the commit, the slab
budget, the advance, and the host thread count.

What the first attempts taught, 2026-09-21:

- Job 16522860 (5e66e28, the commit before the slab budget, advance 0.0)
  failed at its first iteration with a GPU out-of-memory error.  That
  code's automatic batch size put a whole orientation in one task (728,
  260, and 260 hyperplanes), and an 18 GiB allocation did not fit beside
  the data-fit work.  The commit is therefore not usable for an A/B at
  advance 0.0; the reproduction and the code A/B use 7a80bbe instead, the
  commit with the 2 GB slab budget, which Ziyun's second run of that day
  used (job 16484769, 27 min 58 s of wall time in all against 25 min 17 s
  for the run before it, so the budget cost little).
- Every job so far ran with one host thread.  The job script exported
  `OMP_NUM_THREADS=56`, yet torch reported one thread inside the job,
  while the same export on a login node gives 56.  The cause is not yet
  found; the script now prints the CPU affinity and the thread-related
  environment at the start of each job.  So the thread count did not
  differ between these runs and the 2026-09-17 runs, whose script
  (`nsi_4d/test_script_4d.sh`) exports nothing about threads, and the
  thread job 16527753 was cancelled as a duplicate of the baseline.
- Job 16532129, a reproduction at 5e66e28, was cancelled for the same
  out-of-memory reason before it started.

**The slab budget, done 2026-09-21** (job 16522861, v0.1.1, advance 0.0,
`denoise_slab_gb=8`, batch sizes 243, 87, 87, one host thread; files
`timing_log_adv_0.0_slab8.csv` and `run_info_adv_0.0_slab8.txt`).  Per
iteration at iterations 4 to 10, against the 2 GB baseline:

| slab budget | total | parallel part | denoise | data fit | ten iterations |
|---|---|---|---|---|---|
| 2 GB (baseline) | 340.1 s | 316.2 s | 227.6 s | 417.2 s | 55.5 min |
| 8 GB | 290.5 s | 269.3 s | 186.5 s | 363.9 s | 48.7 min |

The larger slabs save 15 percent of the iteration, and the data fit gains
as well as the denoising, since fewer denoise tasks interleave with the
frames on each GPU.  The result is unchanged (final change 0.90 percent in
both).  So the slab budget is worth a larger default, but it explains
little of the gap to 2026-09-17, in line with Ziyun's own 2 GB run.

**The code, done 2026-09-21** (job 16532496, 7a80bbe at advance 0.0,
2 GB slabs, one host thread; files `timing_log_adv_0.0_code0917b.csv` and
`run_info_adv_0.0_code0917b.txt`).  Per iteration at iterations 4 to 10,
against the v0.1.1 baseline:

| commit | total | parallel part | denoise | data fit | ten iterations |
|---|---|---|---|---|---|
| v0.1.1 (baseline) | 340.1 s | 316.2 s | 227.6 s | 417.2 s | 55.5 min |
| 7a80bbe | 301.2 s | 279.5 s | 203.1 s | 373.1 s | 50.5 min |

The commit of 2026-09-17 is about 11 percent faster on every component,
with the same result (final change 0.90 percent in both).  That is the
size of the code's share; it is not the factor of 2.6.

**The reproduction, done 2026-09-21** (job 16532495, 7a80bbe, advance
1.0, the driver's defaults including the 0.2 percent stop threshold, 2 GB
slabs, one host thread, node h007, the node of Ziyun's run; files
`timing_log_repro_0917b.csv` and `run_info_repro_0917b.txt`).  Per
iteration at iterations 4 to 10:

| run | total | parallel part | denoise | data fit | ten iterations |
|---|---|---|---|---|---|
| Ziyun, 16482614, patched tree, batches 364/130/130 | 152.2 s | 150.0 s | 59.4 s | 533.9 s | 24.2 min |
| here, 7a80bbe, batches 73/26/26 | 406.5 s | 385.1 s | 196.4 s | 790.7 s | 65.0 min |
| here, v0.1.1, batches 73/26/26 | 405.6 s | 383.5 s | 196.8 s | 785.4 s | 64.9 min |

The commit of 2026-09-17 in this harness runs exactly as v0.1.1 does,
and 2.6 times slower than Ziyun's run of the same day on the same node.
The job's diagnostic line shows 56 CPUs in the affinity mask and
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` in the environment, the values
the partition's prolog sets.  So the code, the node, the thread count,
and the advance are all the same as Ziyun's, and the batch sizes differ
by an amount the slab test bounds at about 15 percent.  What differs is
the torch version: 2.13.0+cu130 here against 2.14.0+cu130 in Ziyun's
environment, from the `.out` log.  The next single-variable test is the
same job in an environment with torch 2.14.0+cu130 and the Triton it
brings.

**Ziyun's batch sizes, done 2026-09-21** (job 16542319, 7a80bbe, advance
1.0, `denoise_slab_gb=9.8`, batches 364, 130, 130 as in Ziyun's run;
files `timing_log_repro_0917b_slab9.8.csv` and
`run_info_repro_0917b_slab9.8.txt`): at iterations 4 to 10, 396.6 s
total, 375.0 s parallel part, 167.7 s denoising, 776.0 s data fit, 63.2
min for ten iterations.  Against the 2 GB reproduction (406.5, 385.1,
196.4, 790.7) the batch sizes are worth a few percent.  So the slab choice
in Ziyun's patch is not the cause either.

**The thread count, found 2026-09-21.**  Two six-minute jobs settled how
the count is set inside a job on this partition (logs `thr_16553806.log`
and `thr_16553807.log` in the harness).  A job that sources
`~/load_conda_cuda.sh`, as every job here does, finds `OMP_NUM_THREADS=1`
and `MKL_NUM_THREADS=1` afterward; the login shell has neither, and the
variables are set by what that file sources, not by Slurm.  Torch follows
both variables: with `MKL_NUM_THREADS=1` still set, exporting
`OMP_NUM_THREADS=56` leaves torch at one thread, which is why the first
runs here reported one thread despite their export.  Unsetting both gives
torch the cores of the allocation, and so does `torch.set_num_threads`.
Ziyun's script sources nothing and exports nothing, so Ziyun's run had
torch's default, the 56 cores of the allocation.  That is also the only
way the 2.2 s serial tail of that run can be arithmetic over four 19.5 GB
host arrays.

The mechanism, from `mace.py`: a task's recorded time includes its fold
into the host arrays, and the last data-fit task of an iteration also
carries the temporal filter over the whole stack and the fold of every
piece.  All of that runs on torch's host threads, so one thread inflates
every component, denoising most, since its slabs are strided regions of
the host arrays.

The torch version is cleared: job 16543006 (torch 2.14.0+cu130, one
thread) runs iteration for iteration like the torch 2.13 runs.

**The host thread count, done 2026-09-21** (job 16554094: the
reproduction job again, 7a80bbe, advance 1.0, 2 GB slabs, the 0.2 percent
stop threshold, torch 2.13, with both thread variables unset so that torch
used the 56 cores of the allocation; node h004; files
`timing_log_repro_0917b_thr56.csv` and `run_info_repro_0917b_thr56.txt`).
Per iteration at iterations 4 to 10:

| run | total | parallel part | serial tail | denoise | data fit | ten iterations |
|---|---|---|---|---|---|---|
| Ziyun, 2026-09-17, 56 threads, batches 364/130/130 | 152.2 s | 150.0 s | 2.3 s | 59.4 s | 533.9 s | 24.2 min |
| here, one thread | 406.5 s | 385.1 s | 21.3 s | 196.4 s | 790.7 s | 65.0 min |
| here, 56 threads | 180.2 s | 178.1 s | 2.2 s | 60.4 s | 645.3 s | 27.0 min |

Freeing the host threads brings the denoising and the serial tail to
Ziyun's values exactly and the iteration total to within about 15
percent, which is the size of the remaining differences: Ziyun's larger
batches (a few percent, measured above), the code after c06f165 (about
11 percent), and a different node.  The 2.6 times is the host thread
count.
**The torch version, done 2026-09-21** (job 16543006: the reproduction
job in `venv_t214`, torch 2.14.0+cu130 with Triton 3.8.0 in place of
torch 2.13.0+cu130 with Triton 3.7.1, one host thread as before; files
`timing_log_repro_0917b_t214.csv` and `run_info_repro_0917b_t214.txt`).
At iterations 4 to 10: 461.5 s total,
437.8 s parallel part, 242.2 s
denoising, 881.3 s data fit, 71.9 min for ten
iterations, against 406.5, 385.1, 196.4, 790.7, and 65.0 for torch 2.13.
The torch version is not the cause.
- Job 16542319, the reproduction with Ziyun's batch sizes: 7a80bbe,
  advance 1.0, the 0.2 percent stop threshold, and `denoise_slab_gb=9.8`,
  which the slab rule turns into batches of 364, 130, and 130.  Run
  directory `runs/repro_0917b_slab9.8/`.  If it lands near 152 s per
  iteration, the difference is in the code after 7a80bbe; if it stays
  near 400 s, the difference is in the environment or in Ziyun's patch.

## Conclusion

**Update, 2026-09-21.**  The gap between the runs of 2026-09-20 and the run
of 2026-09-17 is the number of host threads torch had, one against 56.
The loop folds every task's output into host arrays and filters the
data-fit output on the host, inside the timed tasks, and that work runs on
torch's intra-op threads.  On gautschi a job that sources
`~/load_conda_cuda.sh` gets `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`,
and torch honors both, so such a job runs the 4D loop 2.6 times slower
than one that leaves the variables unset.  The commit, the slab budget,
the batch sizes, and the torch version each account for at most 15
percent.  Open for Greg: whether the library should set torch's thread
count itself when it finds one thread and many cores, or whether the
documentation should say to leave the variables unset; and the choice of
`prox_partition_advance` below, which stands as measured at one thread
and should be re-timed at 56.


The slow data-fit step is the partition schedule.  With the default advance
of 1.0 every data-fit call runs at 128 subsets from the fourth iteration on,
and the data fit costs 1.7 times what it costs when the calls stay at 4, 16,
and 64 subsets.  After ten iterations the three schedules give
reconstructions that differ by less than one iteration's change, and the
coarse schedule is not converging more slowly.

Open for Greg: whether the default `prox_partition_advance` becomes 0.0,
which is the schedule mbirjax used, or a small value such as 0.25.  This
experiment stops at ten iterations; it does not say which schedule reaches
the lowest cost at convergence.
