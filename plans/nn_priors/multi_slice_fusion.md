# Plan for multi-slice fusion with a network denoiser prior

This plan realizes the first queued follow-up of `mace_poc_findings.md`: move
the validated MACE loop from the 2D proof of concept to a 3D problem, with
three orientation denoiser agents fused with the forward model — the
multi-slice fusion construction.  The framing decisions carry over from the
proof of concept unchanged (section 1).  The plan was drafted 2026-08-27 and
awaits review before work starts.  Code continues in `experiments/drunet/`
in the mbirtorch repo; documentation lands beside this plan.

## Executive summary

The proof of concept left this program cheap to build.  The MACE loop in
`experiments/drunet/mace.py` already takes any number of agents with
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
| 4 | findings and follow-ups | done 2026-08-27; `multi_slice_fusion_findings.md`, with the follow-up queue carried there |

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
`experiments/drunet/output/`.

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
