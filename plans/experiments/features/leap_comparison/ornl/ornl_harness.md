# Harness for reproducing the ORNL LEAP versus mbirtorch comparison

Companion to the scripts in this directory and to the findings page
`plans/features/leap_comparison/ornl_reproduction.md`.  This file holds the
run detail: environment, commands, job identifiers, and what each script
does.  The findings page holds the results and their interpretation.

## Inputs on gautschi

- Collaborators' code: `~/PycharmProjects/leapMBIR` on Greg's account, a
  clone of `code.ornl.gov/o2x/leap_recon` (Obaid Rahman, last commit
  2026-09-12).  The comparison scripts are `LEAP_(MBIRtorch_vs_LEAP).py` and
  `MBIRtorch_(MBIRtorch_vs_LEAP).py`.  The harness imports a copy of their
  `utils` package unchanged.
- Data: `/depot/bouman/data/ORNL/mbirtorch_vs_leap/`.  The folder
  `483-103_2132x8/` holds the raw counts (`UncorrectedSingle.uint16`, 2232
  frames of 1456 x 1840: 10 dark, 90 bright, 2132 views) and the Metrotom
  log their reader mines.  The tiff `483-103_2132x8_proj_BHC_SC_2132vws.tiff`
  holds the corrected projections as float16 (2132, 1456, 1840).  The zip
  file beside them duplicates both.

## Environment

- Working directory: `/scratch/gautschi/buzzard/leap_ornl/`.
- `venv/`: a bare virtual environment over the conda environment
  `~/.conda/envs/mbirtorch` (Python 3.11, torch 2.13.0+cu130).  It holds an
  editable install of `mbirtorch_src/`, a clone of `cabouman/mbirtorch` at
  the `greg_dev` tip 41fca86 (2026-09-12).  A `.pth` path line adds the conda
  environment's site-packages and the site-packages of
  `/scratch/gautschi/buzzard/leap_cmp/venv`, where LEAP 1.26 was built from
  source on 2026-09-02, so `leapctype` resolves from there.  `nvidia-ml-py`
  was installed into the older venv for the sampler.
- LEAP's shared library links the cluster's CUDA 12.6.1 module, so every job
  sources `~/load_conda_cuda.sh` first.
- `data/`: `weights_vrc_float32.npy` (21.3 GiB) and `params.pkl`, written by
  the first run of the loader.  Rebuilding them takes about 90 s and 86 GiB
  of host memory.

## Scripts

- `ornl_data.py`: loads the scan as the collaborators' scripts do and returns
  float32 projections, weights, and their parameter dictionaries in (view,
  row, column) order.  `--view-step` and `--det-step` in the drivers shrink
  the problem for smoke tests.
- `run_mbirtorch.py`: the mbirtorch arm.  Options: `--iterations`,
  `--stop-pct` (default 0.2, the library default their script keeps),
  `--num-devices` (0 leaves the choice to the library), `--fdk-repeats`,
  `--save-volume`, `--tag`.
- `run_leap.py`: the LEAP arm with their `MBIR` loop.  Options:
  `--iterations`, `--beta` (0.7), `--gpus` (0 keeps all visible cards),
  `--max-slices` (LEAP's chunk cap, default 128), `--fdk-repeats`,
  `--save-volume`, `--tag`.  Debug logging is on for the Lipschitz projection
  and the first two iterations, which records LEAP's chunk ranges.
- `gpu_sampler.py`: NVML sampler, 50 ms period, per card and combined, both
  the card total and this process's share; host peak from `getrusage`.
- `leap_chunks.py`: prices LEAP's forward and back projection jobs for this
  geometry from the slab formulas in LEAP's `parameters.cpp`.
- `price_ornl_geometry.py`: prices mbirtorch's device arrays for this
  geometry with the memory ledger on one, two, and four devices.  Runs
  locally from the mbirtorch repository root.
- `fdk_pieces.py`, `fdk_pieces.sbatch`: the FDK split into sinogram
  placement, filter, back projection, and volume gather for mbirtorch, and
  FBP, back projection, and filter for LEAP, on four H100s.
- `leap_transfer.py`: LEAP's forward, back, and FBP with host arrays and
  with device tensors on one card, plus the raw copy rates; run inside the
  interactive job.
- `smoke_h005.sh`: both arms at one eighth of the views and 4 x 4 detector
  blocks inside an interactive job, run as
  `srun --overlap --jobid=<id> bash smoke_h005.sh`.
- `full_mbirtorch.sbatch`, `full_leap.sbatch`: the reproduction, four H100s,
  15 iterations, two timed FDK runs.
- `ablation.sbatch`: two H100s, five iterations per arm: LEAP on one card,
  on two, on two with the chunk cap at 32, and mbirtorch pinned to two
  devices with the stop rule off.
- `msf_ornl.py`, `msf_finish.py`, `render_early.py`: the multi-slice
  fusion reconstruction of the scan with DRUNet priors, the step that makes
  its metrics and images, and a renderer for a run in progress.  The run
  detail is in `msf_ornl.md` beside them.  `msf_profiles.py` plots
  intensity profiles across pores from the saved central slices.

Outputs go to `out/` as `<arm>_<tag>_results.json`, central slices in an
`.npz`, and the full volume as float32 `.npy` when `--save-volume` is given.
Job logs go to `logs/`.  Copies of the results files and logs are in
`results/` here.  The repository's ignore rules exclude `.json`, `.log`,
`.png`, and `.sbatch` files, so those copies and the batch scripts are not
committed.  The cluster directory keeps the originals.

## Runs on 2026-09-14

| run | job | node | elapsed | notes |
|---|---|---|---|---|
| smoke, both arms | interactive job 16395336 | h005 | minutes | `smoke.log`; the first attempt crashed at the end on a saving bug |
| ablation, four arms | 16396043 | h001 | 39 min | `ablation_16396043.log` |
| full, mbirtorch | 16396079 | h008 | 12 min | `full_mbirtorch_16396079.log` |
| full, LEAP | 16396080 | h008 | 15 min | `full_leap_16396080.log` |
| FDK parts, both codes | 16401543 | h011 | 5 min | `fdk_pieces_16401543.log` |
| LEAP transfer share | interactive job 16395336 | h005 | 8 min | `leap_transfer.log` |
| fusion smoke pass and crashed full run | 16417144 | four H100s | 4 min | `msf_ornl_16417144.log`; see `msf_ornl.md` |
| fusion full run | 16421729 | h013 | up to 6 h | `msf_ornl_16421729.log`; see `msf_ornl.md` |
| fusion finisher | 16426291 | one H100 | minutes | runs after 16421729; see `msf_ornl.md` |

The results files are named `<arm>_<tag>_results.json` with tags `full`,
`abl_leap_g1`, `abl_leap_g2`, `abl_leap_g2_s32`, and `abl_mbirtorch_d2`.
The figures `central_slices_full.png` and `central_slices_full_zoom.png`
come from `slices_png.py`, which matched the LEAP slice to the mbirtorch
slice by a half-turn rotation and a column flip.

Jobs 16396022 and 16396023 were submitted with 100 iterations and cancelled
before they started, once the slides showed that the collaborators' table is
for 15 iterations.
