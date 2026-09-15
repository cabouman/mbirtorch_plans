# Multi-slice fusion of the ORNL Inconel scan: the run record

Companion to `msf_ornl.py`, `msf_finish.py`, and `render_early.py` in this
directory, and to the addendum for this run in
`plans/nn_priors/multi_slice_fusion.md`.  This file holds the run detail and
the numbers.  The addendum holds their interpretation.

## What runs

`msf_ornl.py` reconstructs the scan with MACE.  The data-fit proximal map is
the forward agent, and three DRUNet denoiser agents, one per slice
orientation, are the prior, at consensus weights (1/2, 1/6, 1/6, 1/6).  The
construction and the algorithm choices follow the Lilly script
`mbirtorch_applications/nsi/msf_recon.py` and the findings in
`plans/nn_priors/multi_slice_fusion_findings.md`.

The difference from the Lilly run is scale.  The volume is (1360, 1360,
1296) float32, 8.9 GiB, so the consensus state lives in host memory: four
agent inputs, four agent outputs, and the consensus, nine volumes in all.
The proximal agent passes host arrays to `prox_map`, whose model runs on all
four cards.  Each DRUNet agent streams batches of slices through one card and
writes the denoised slices back into a host volume, so no card holds a whole
volume for the denoisers.  The script writes a checkpoint of the consensus
every five iterations and the traces every iteration, so a run that the
walltime ends still leaves a usable result.

The settings of the full run: initialized at the 15-iteration standard
reconstruction saved by the comparison run, `out/mbirtorch_full_recon.npy`;
`sigma_scaled` 0.02, which is 0.00788 in reconstruction units; intensity
scale 2.54, which maps the standard reconstruction's 99.9th percentile,
0.35448, to 0.9 of the network's range; rho 0.5; three inner iterations per
proximal step; 30 outer iterations requested.  The percentile is taken with
numpy over every fourth voxel along each axis, because torch's quantile
refuses a tensor of this size.

The choice of 0.02 follows Greg's request to start at the light end of the
regularization range and keep sharp edges.  On the Lilly scan 0.04 gave
crisper edges than 0.075 and looked like the better operating point, so this
run goes one step lighter than that.

`msf_finish.py` makes the metrics and images from whatever the run left
behind: the final fusion volume when it exists, otherwise the newest
checkpoint.  It computes the weighted sinogram residual on one card, saves
the central slices, and writes the side-by-side image and a results file.
`render_early.py` renders the central slices and the traces from the newest
checkpoint without a GPU, for a look at a run in progress.

## Jobs on 2026-09-14

| job | node | what | outcome |
|---|---|---|---|
| 16417144 | four H100s | a smoke pass at one quarter of the detector size, then the full run | the smoke pass ran clean in 4 minutes; the full run crashed at the intensity scale |
| 16421729 | h013, four H100s | the full run, walltime 6 hours | 25 iterations in 5 h 17 min; cancelled at 00:37 on 2026-09-15, right after the checkpoint at iteration 24, because the walltime would have ended the run before the next checkpoint |
| 16426291 | one H100 | `msf_finish.py sigma002`, queued to run after 16421729 ends | see the results below |

The smoke pass, at `sigma_scaled` 0.04 and three iterations on a (340, 340,
324) volume, gave residuals of 0.05920 for the standard reconstruction,
0.07820 for the postprocessing, and 0.06416 for the fusion, in 43.5 s.  The
full run in the same job stopped with `RuntimeError: quantile() input tensor
is too large`, which the numpy percentile above fixes.  Job 16421729 is the
resubmission without the smoke pass.

## Timeline of the full run, job 16421729

| stage | time |
|---|---|
| loading the projections and weights | 41 s |
| residual of the standard reconstruction | 87 s |
| three-orientation postprocessing | 371 s |
| residual of the postprocessing | 83 s |
| first fusion iteration | 693 s |
| later fusion iterations | 730 to 743 s each |

One outer iteration costs 12.2 minutes, 728 to 743 s after the first, so
the six-hour walltime would have ended the loop after about 28 iterations,
before the checkpoint at iteration 29.  The run was cancelled once the
checkpoint at iteration 24 was on disk, so the reported fusion result is
the consensus after 25 iterations.

The residuals from the run's own log: standard reconstruction 0.05925,
three-orientation postprocessing 0.08075.

## Consensus traces

The spread is the root-mean-square disagreement among the agents' outputs
about the consensus, and the change is the root-mean-square step of the
consensus, both in reconstruction units.

| iteration | spread | change |
|---|---|---|
| 0 | 3.42e-2 | 0 |
| 1 | 2.96e-2 | 8.37e-3 |
| 2 | 2.61e-2 | 9.58e-3 |
| 3 | 2.36e-2 | 9.91e-3 |
| 4 | 2.28e-2 | 9.26e-3 |
| 5 | 2.20e-2 | 8.46e-3 |
| 6 | 2.12e-2 | 7.79e-3 |
| 7 | 2.04e-2 | 7.32e-3 |
| 8 | 1.94e-2 | 7.01e-3 |
| 9 | 1.86e-2 | 6.72e-3 |
| 10 | 1.78e-2 | 6.44e-3 |
| 11 | 1.71e-2 | 6.15e-3 |
| 12 | 1.64e-2 | 5.86e-3 |
| 13 | 1.58e-2 | 5.61e-3 |
| 14 | 1.53e-2 | 5.31e-3 |
| 15 | 1.49e-2 | 5.04e-3 |
| 16 | 1.46e-2 | 4.80e-3 |
| 17 | 1.44e-2 | 4.60e-3 |
| 18 | 1.41e-2 | 4.42e-3 |
| 19 | 1.39e-2 | 4.27e-3 |
| 20 | 1.37e-2 | 4.12e-3 |
| 21 | 1.36e-2 | 3.98e-3 |
| 22 | 1.34e-2 | 3.86e-3 |
| 23 | 1.33e-2 | 3.75e-3 |
| 24 | 1.32e-2 | 3.64e-3 |

Both traces fall at every iteration without oscillation, so rho 0.5 holds
on this data.  The spread was still falling by just under one percent per
iteration at the end, so the run had not settled by iteration 24.  For
comparison, the Lilly runs settled near 8e-3 at `sigma_scaled` 0.075 and
5.2e-3 at 0.04 within 30 iterations, on a volume with one thirteenth as
many voxels.

## Early look

At 20:56 `render_early.py` ran inside the job's allocation with
`srun --overlap --jobid=16421729` on the iteration-4 checkpoint.  Its
images are `out/msf/early_slices.png`, `early_slices_zoom.png`, and
`early_traces.png`.  In the zoom the grain of the standard reconstruction is
gone from both denoised panels, the pores survive, and the fusion panel's
pores are darker and better defined than the postprocessing's.

## Results

The finisher, job 16426291 on h012, ran from the iteration-24 checkpoint
in 2.5 minutes on one card.  It found 25 iterations recorded and a final
spread of 1.32e-2.

| volume | weighted sinogram residual |
|---|---|
| standard reconstruction, 15 iterations | 0.05925 |
| three-orientation DRUNet postprocessing of it | 0.08075 |
| multi-slice fusion, 25 iterations, `sigma_scaled` 0.02 | 0.06063 |

The fusion result fits the data almost as well as the standard
reconstruction, 2.3 percent worse, while the postprocessing of the same
volume fits it 36 percent worse.  Whether the 2.3 percent is noise that
the standard reconstruction fits and the fusion rejects, or detail that the
prior removes, the residual alone cannot say.  The slices carry that
judgment.

The images are `ornl_sigma002_slices.png` and
`ornl_sigma002_slices_zoom.png`, the central axial slice and a zoom of it
for the three volumes.  The grain of the standard reconstruction is gone
from both denoised volumes.  The postprocessing lightens the small pores,
so the faintest of them fade toward the background.  The fusion keeps the
pores dark and compact, close to their appearance in the standard
reconstruction, with the grain removed.  The part's outer edge and the
bright band near its surface look the same in all three.

## Outputs

All under `out/msf/` in the cluster directory, none committed here:
`ornl_sigma002_postproc.npy`, `ornl_sigma002_checkpoint.npy` (the newest
checkpoint, overwritten every five iterations), `ornl_sigma002_traces.npz`,
and from the finisher `ornl_sigma002_finish_results.json`,
`ornl_sigma002_slices.npz`, `ornl_sigma002_slices.png`, and
`ornl_sigma002_slices_zoom.png`.  The job logs are
`logs/msf_ornl_<job>.log` and `logs/msf_finish_<job>.log`.
