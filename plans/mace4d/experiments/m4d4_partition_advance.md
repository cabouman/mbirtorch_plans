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

## Conclusion

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
