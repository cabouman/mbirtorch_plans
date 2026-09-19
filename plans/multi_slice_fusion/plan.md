# Plan for multi-slice fusion with a network denoiser prior

Status: ACTIVE
Updated: 2026-09-18
Code: mbirtorch prerelease d90fd69 holds the fusion prototype (27c3cf5, 26bd0ea); the ORNL run script carries its own MACE loop
Next step: Move the fusion onto mbirtorch.mace as a library function, and decide the stronger prior (0.03 or 0.04) for the unsettled ORNL run.

This plan realizes the first queued follow-up of `plans/mace_poc/findings/mace_poc_findings.md`: move
the validated MACE loop from the 2D proof of concept to a 3D problem, with
three orientation denoiser agents fused with the forward model — the
multi-slice fusion construction.  The framing decisions carry over from the
proof of concept unchanged (section 1).  The plan was drafted 2026-08-27 and
awaits review before work starts.  Code continues in `experiments/drunet/`
in the mbirtorch repo; documentation lands beside this plan.

## Executive summary

The proof of concept left this program cheap to build.  The MACE loop in
mbirtorch's `experiments/drunet/mace.py` already takes any number of agents with
weights, so fusion is a parameter change, not a rewrite; the forward-prox
and DRUNet agents are validated; `prox_map` and `denoise` are natively 3D
with warm starts; and the equality-gate method (MACE with the qGGMRF agent
must reproduce the standard recon) transfers to 3D as-is, so the 3D plumbing
is validated before any network runs.

The question the program answers: on a noisy 3D cone-beam problem, does
fusing three slice orientations of a 2D denoiser beat (a) the same denoiser
as a single-orientation prior, which regularizes nothing through-plane, and
(b) the postprocessing uses of the same network — with the standard qGGMRF
recon as the incumbent throughout.

The new work is small and concentrated: a 3D problem setup, a `slice_axis`
parameter on the DRUNet agent so the same network serves all three
orientations, a data-consistency metric beside NRMSE, and the comparison
runs.  The one genuinely new algorithmic choice is the agent weights; the
default keeps the forward model at half the consensus weight and splits the
other half evenly across the three orientation agents, so the two-agent case
is recovered when the orientations collapse to one, and the weights stay
exposed for sweeps.

The work is four increments.  Increment 1 gates quantitatively (the 3D
equality gate at about 1% NRMSE, as in 2D) and measures runtimes so the
later increments are sized to the machine; the network increments gate on
recorded sweep tables and side-by-side comparisons rather than exact values.

## 1. Locked decisions carried from the proof of concept

- Agents are callables from a volume tensor to a volume tensor on one fixed
  device, every knob bound at construction.
- The denoiser strength (sigma_noise, or sigma_scaled in the network's
  scale) is the one user-facing regularization knob; sigma_prox stays at the
  model's auto value.
- Operators may follow a schedule early (the forward agent walks its
  partition sequence coarse to fine) but must be fixed in the tail.
- Each agent warm-starts its inner solve from its own previous output.
- The loop is the weighted Mann iteration; rho = 0.5 default, damping is the
  first remedy if the iteration oscillates.
- The qGGMRF MBIR baseline is the STANDARD recon; "direct recon" means the
  FDK-style `recon_direct` initialization.

## 2. The fusion construction

Four agents on one volume: the data-fit proximal map, and three DRUNet
agents that slice the volume along axis 0, 1, and 2 respectively.  The
DRUNet agent gains a `slice_axis` argument: move the chosen axis to the
batch position, denoise the resulting stack of 2D slices (reflect-padded to
multiples of 8, which matters now — two of the three orientations see
non-square slices), and move the axis back.  The intensity scale and the
region-of-reconstruction mask are applied to the volume, not the slices, so
they are shared by all orientations; per-orientation strengths are allowed
by construction (three separate agents) but start equal.

Default weights: mu = (1/2, 1/6, 1/6, 1/6).  The forward model keeps the
same half of the consensus it has in the two-agent case; the prior half is
split evenly.  Weights are plan parameters, not constants.

The problem is the full 3D version of the demo cone-beam setup the 2D work
restricted to its middle slice: same geometry and dosage-based noise model,
detector and view counts chosen in increment 1 after the runtime
measurement.  Metrics: NRMSE vs phantom, the weighted sinogram residual
(data consistency, computed by forward-projecting the result), and visual
through-plane inspection in the viewer via axis transposition — the
through-plane artifact structure is what single-orientation priors miss, so
it is checked directly, not only through one aggregate number.

## 3. Increments

Status (updated as work proceeds):

| # | Increment | Status |
|---|---|---|
| 1 | 3D problem and 3D qGGMRF gate | done 2026-08-27; `cone_beam_3d.py` and `run_qggmrf_gate.py --problem 3d`; gate PASS at NRMSE 0.0023 vs the standard recon (spread 6.9e-6, matched sigma, 30 iterations) in 84 s wall on mps; runtime probe `measure_3d_runtimes.py` at (128,128,128): standard recon 2 iterations 4.4 s cold / 4 iterations 2.2 s warm, prox_map 3 iterations 2.1 s, qGGMRF denoise 8 iterations 1.2 s, DRUNet about 1 s per 128-slice orientation stack — the full demo scale is kept and every run stays interactive on the Mac |
| 2 | slice_axis DRUNet agent and single-orientation baselines | done 2026-08-27; slice_axis landed with the moved-axis equality check; initial fixed-strength comparison reviewed by Greg, then the strength sweep over {0.05, 0.075, 0.10, 0.125, 0.15} (`run_fusion_sweep.py`): post-1 best 0.129 at 0.10, post-3 best 0.122 at 0.075, mace-1 best 0.121 at 0.075 at 30 iterations (0.124 confirmed at 60), all vs standard 0.363; grids in `experiments/drunet/output/fusion_sweep.npz` |
| 3 | multi-slice fusion runs and sweeps | done 2026-08-27; fusion (default weights 1/2, 1/6, 1/6, 1/6) is best at EVERY grid strength; best 0.0913 at sigma_scaled 0.075 (30 iterations), confirmed 0.0925 at 60 (spread 6.8e-3 — a stable equilibrium with small persistent inter-orientation disagreement); about 25% below the best non-fusion method at convergence; the weight sweep stayed unused since fusion separated at the defaults; data-consistency metric landed — denoised results sit at the noise floor (rms_w 0.0715-0.0736 vs 0.0729) while the standard recon overfits (0.0521); the sandbox gained PROBLEM='3d' with the fusion panel |
| 4 | findings and follow-ups | done 2026-08-27; `plans/multi_slice_fusion/findings/multi_slice_fusion_findings.md`, with the follow-up queue carried there |

Addendum 2026-08-27 — first real-data run (the queue's real-data follow-up): multi-slice
fusion on the NSI Lilly Autoinjector scan (downsample 3, view subsample 5, recon
(626, 626, 467) at 102.94 um), on a gautschi H100, via the self-contained
`mbirtorch_applications/nsi/msf_recon.py` (preprocessing duplicated from Lilly_recon.py;
MACE loop and agents inlined; initialized at the existing standard recon).  30 fusion
iterations took 799 s (~27 s each); consensus spread settled near 8e-3, the same
equilibrium signature as the demo runs.  Weighted sinogram residual rms_w(y - Ax):
standard 0.0657, three-orientation postprocessing 0.0656, fusion 0.0614 — though part of
the fusion improvement is simply its 90 additional warm-started prox iterations on a
15-iteration standard recon, so the visual comparison carries the real evaluation.
A second run at sigma_scaled 0.04 (Greg's call after seeing the 0.075 slices) matched
the data fit (rms_w 0.0613 vs 0.0614) with visibly crisper edges — the spring coils
resolve individually — while keeping the streak suppression; consensus spread settled
lower too (5.2e-3 vs 8.1e-3), consistent with the weaker prior.  0.04 looks like the
better operating point on this data.  Volumes and traces:
`mbirtorch_applications/nsi/output/msf/` on gautschi.  Env notes:
deepinv + cu130 torchvision installed --no-deps into the gautschi `mbirtorch` conda env;
DRUNet weights staged at `~/.cache/torch/hub/checkpoints/`.

Addendum 2026-09-15 — second real-data run, at production scale: multi-slice fusion on
the ORNL Inconel scan (2132 views of 1456 x 1840, reconstruction (1360, 1360, 1296)
float32, 8.9 GiB), on four gautschi H100s, via
`surveys/leap_comparison/experiments/ornl/msf_ornl.py`.  The run record
`surveys/leap_comparison/experiments/ornl/msf_ornl.md`, beside the script, holds the settings, the timeline, and the traces.  What is
new in the script is where the state lives.  The consensus state is nine host volumes, the
proximal agent hands host arrays to `prox_map` on four cards, and each DRUNet agent streams
slice batches through one card, so the loop runs on a volume with thirteen times the
voxels of the Lilly run.  The strength was `sigma_scaled` 0.02, the light end of the
range, at Greg's request to keep sharp edges; the other settings match the Lilly run.  One
outer iteration took 12.2 minutes, and the run stopped after 25 iterations, at its
iteration-24 checkpoint, because the six-hour walltime would have ended it before the
next checkpoint.  Weighted sinogram residual: standard 0.05925, three-orientation
postprocessing 0.08075, fusion 0.06063.  The consensus spread reached 1.32e-2 and was
still falling by just under one percent per iteration, so this run did not settle, where
the Lilly runs settled near 5e-3 to 8e-3 within 30 iterations.  In the central slice the
grain of the standard reconstruction is gone from both denoised volumes; the
postprocessing lightens the small pores, and the fusion keeps them dark and compact.
Whether to run a stronger prior (0.03 or 0.04) is Greg's call after seeing the slices.
Volumes and traces: `/scratch/gautschi/buzzard/leap_ornl/out/msf/` on gautschi.

**Increment 1: 3D problem and 3D qGGMRF gate (quantitative gate).**
A 3D problem module beside `cone_beam_2d.py` (the same generator without the
mid-slice restriction), and the equality gate at 3D: MACE with the forward
prox and the pinned qGGMRF denoiser at matched strengths must reproduce the
standard recon within about 1% NRMSE, initialized at the standard recon as
in 2D.  Record wall-clock for the standard recon, one prox call, and one
qGGMRF denoise at this scale, and size the sweep increments (and the
Mac-vs-cluster question) from those measurements.  Exit: gate passes; the
runtime table is recorded in the run records.

**Increment 2: slice_axis DRUNet agent and single-orientation baselines
(recorded comparisons).**
Add `slice_axis` to DRUNetAgent with a small unit check (denoising a volume
along axis 0 equals denoising its transpose along axis 2, transposed back).
Baselines on the 3D problem, each over a strength sweep: DRUNet
postprocessing of the standard recon slice-wise along one axis; the
three-orientation-average postprocessing; and MACE with a single-orientation
DRUNet prior.  Exit: baseline table and volumes saved under
mbirtorch's `experiments/drunet/output/`.

**Increment 3: multi-slice fusion runs and sweeps (recorded comparisons).**
N=4 MACE at the default weights over the strength sweep; compare against
every increment-2 baseline by NRMSE, data consistency, and through-plane
visuals.  If fusion does not separate from the single-orientation prior, a
weight sweep and per-orientation strengths are the first two levers, in that
order.  The sandbox gains the fusion recon as a fourth panel.  Exit: fusion
vs baselines table saved; sandbox updated.

**Increment 4: findings and follow-ups (record).**
Findings doc beside this plan, numbers quoted from the run records.
Expected follow-up queue: device-form (Shards) exchange between agents for
volumes past single-device scale; sigma_prox and rho tuning or schedules;
the 2.5D pseudo-RGB variant; correlated-noise denoisers if streak residue
shows.

## 4. Risks and named assumptions

- **A1: the network transfers to through-plane slices.**  DRUNet was trained
  on natural images; the axis-0 and axis-1 slices of a CT volume have
  different statistics from the in-plane slices.  Per-orientation strengths
  are the built-in mitigation, and increment 2's baselines make any
  orientation asymmetry visible before fusion runs.
- **A2: convergence with three network agents.**  Nothing guarantees the
  stack of three CNN agents is nonexpansive.  The remedy order is fixed:
  smaller rho first, then fewer inner prox iterations; the consensus-spread
  trace is the detector.
- **A3: runtime on the Mac.**  Every outer iteration runs the 2D network
  over three full stacks of slices plus a 3D prox.  Increment 1 measures
  before the sweeps are sized; if the Mac is the constraint, sweeps move to
  the cluster and the Mac keeps the sandbox-scale runs.
- **A4: memory at 3D.**  The loop holds one w and one X volume per agent
  (eight volumes at N=4) plus agent internals — fine at demo scale on one
  device, and the reason production scale stays out of scope here.
- **A5: the 3D gate needs the same pinning discipline as 2D.**  sigma_x from
  the standard recon's record, auto-regularization off, and the
  sigma_y-follows-sigma_noise sync (already in the library since the proof
  of concept).

## 5. Out of scope

- Production-scale volumes and the Shards (multi-device) exchange between
  agents; this plan stays at single-device demo scale.
- Denoiser training or fine-tuning, and video/2.5D networks beyond the
  pseudo-RGB follow-up note.
- Time-resolved (4D) fusion.
- Posterior sampling / diffusion machinery.
