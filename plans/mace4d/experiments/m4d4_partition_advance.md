# m4d4: the data-fit step against `prox_partition_advance`

Status: submitted 2026-09-20, results pending.

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

Pending.
