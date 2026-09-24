# recon4k_gautschi: the 4K benchmark job on gautschi

Scripts: `experiments/parallel_4k/stream_recon_parallel.py`,
`experiments/parallel_4k/make_synthetic_sinogram.py`, and
`experiments/parallel_4k/gautschi_4k.sbatch` in the mbirtorch repository,
branch `parallel_4k_script`.  The batch file was rewritten on 2026-09-23 to
run the streaming harness on phantom sinograms written to scratch, and the
copies on gautschi match the repository (md5 checked).  Nothing is queued.  Copies with the same md5 sums are at
`/scratch/gautschi/buzzard/recon4k/` on gautschi.  The job runs with
`recon4k/venv/bin/python`, which imports the clone `recon4k/mbirtorch_src`
at prerelease 498294d.

Submitted 2026-09-23 as job 16629880 with `-A bouman -p ai -q normal -N1
--gpus-per-node=8 --cpus-per-task=112 -t 04:00:00`.  The scheduler's
estimated start was 2026-09-25 at 21:33.

## What it measures

1. Anchor cells of 900 views at 1024, 1536, and 2048 rows and channels, each
   reconstructed as one band from a phantom sinogram on scratch: 1024 and
   1536 on 1 and 4 GPUs, 2048 on 4 and 8 GPUs.  Each cell runs with 3 and
   with 6 iterations, and each run repeats once, so the second run is warm.
2. The 4K case, 900 x 4096 x 4096, on 8 GPUs and then on 4 GPUs, with the
   band count the harness chooses at its default margin of 0.3.  Five
   slices of each result are checked against the phantom.

## Where the results land

The job log is `/scratch/gautschi/buzzard/recon4k/logs/recon4k_16629880.log`.
Each reconstruction writes `<name>_progress.json` in `recon4k/results/` with
each band's read, reconstruction, and write times, its iteration count, and
the peak host memory so far; the job log holds the peak allocated and
reserved memory of each GPU.  The anchor volumes are deleted after each run.
The 4K volumes stay as `results/recon_4k_n8.h5` and `results/recon_4k_n4.h5`,
275 GB each.

## How to read it

The steady-state cost of one iteration at a cell is the 6-iteration warm
time minus the 3-iteration warm time, divided by 3.  The 4K per-iteration
cost follows from the bands' `recon_s` values and iteration counts, less
the setup.  Compare the measured GPU peaks with the plan mode's numbers to
check the ledger at 4K.  The peak host RSS should be about one band of the
sinogram plus one reconstructed band.

## Result

Cancelled on 2026-09-23 before it started, pending discussion of the streaming
harness.  Nothing was measured.
