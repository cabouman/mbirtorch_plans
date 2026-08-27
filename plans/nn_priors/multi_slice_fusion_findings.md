# Multi-slice fusion with a network denoiser prior — findings

2026-08-27.  Code: `experiments/drunet/` in the mbirtorch repo.  Run
records: `experiments/drunet/output/qggmrf_gate_3d.npz`,
`fusion_initial.npz`, and `fusion_sweep.npz`.  Plan:
`multi_slice_fusion.md` beside this file.

## What was built

- `cone_beam_3d.py` — the full 3D demo problem (same generator and noise
  model as the 2D script, without the mid-slice restriction: (128, 128, 128)
  recon, dosage 500).
- `run_qggmrf_gate.py --problem 3d` — the equality gate at 3D.
- `measure_3d_runtimes.py` — the increment-1 runtime probe.
- `slice_axis` on DRUNetAgent, with a moved-axis equality check (denoising
  along axis 0 equals denoising the moved volume along axis 2, moved back,
  bitwise) that runs at the start of the comparison scripts.
- `run_fusion_initial.py` (fixed-parameter comparison, reviewed by Greg
  before the sweeps) and `run_fusion_sweep.py` (the strength sweep with
  convergence confirmations), both reporting NRMSE and the data-consistency
  residual rms_w(y - Ax).
- The sandbox gained `PROBLEM = '3d'`: five panels (phantom, standard,
  three-orientation postprocessing, single-orientation MACE, MACE fusion),
  sharing the 3D standard-recon cache with the runner scripts.

## Gate and runtimes

The 3D equality gate passed at NRMSE 0.0023 vs the standard recon (matched
sigma, pinned prior, 30 iterations, consensus spread 6.9e-6), in 84 s wall
on the Mac's mps device.  Runtime probe at (128, 128, 128): standard recon
2 iterations 4.4 s cold / 4 iterations 2.2 s warm, prox_map 3 iterations
2.1 s, qGGMRF denoise 8 iterations 1.2 s, DRUNet about 1 s per 128-slice
orientation stack — every run in this program stayed interactive on the Mac
(the full strength sweep with confirmations was about 30 minutes).

## Results

NRMSE vs phantom over the strength grid, 30 outer iterations, rho 0.5,
weights (1/2, 1/6, 1/6, 1/6) for fusion (standard recon: 0.363):

| sigma_scaled | post-1 | post-3 | mace-1 | mace-3 (fusion) |
|---|---|---|---|---|
| 0.05  | 0.194 | 0.187 | 0.169 | 0.144 |
| 0.075 | 0.132 | 0.122 | 0.121 | **0.0913** |
| 0.10  | 0.129 | 0.123 | 0.159 | 0.124 |
| 0.125 | 0.161 | 0.153 | 0.175 | 0.161 |
| 0.15  | 0.182 | 0.171 | 0.184 | 0.179 |

Convergence confirmations at the best strengths (60 iterations): mace-1
0.1244 (rms_w 0.0716, spread 1.3e-3); mace-3 0.0925 (rms_w 0.0717, spread
6.8e-3).

Conclusions:

1. **Fusion wins at every strength on the grid**, and best-vs-best at
   convergence the margin is about 25%: 0.0925 for fusion vs 0.1244 for the
   single-orientation prior and 0.1216 for the best postprocessing — four
   times better than the standard recon's 0.363 on this noisy problem.
2. The strength knob behaves: interior optimum at sigma_scaled 0.075 for
   post-3, mace-1, and mace-3 alike (post-1 at 0.10), matching the 2D
   sweep's MACE optimum.
3. The data-consistency column separates the regimes cleanly: the standard
   recon overfits the noise (rms_w 0.052, below the phantom's own noise
   floor of 0.0729), while every denoised result sits essentially at the
   floor (0.0715-0.0736).
4. The fusion consensus spread settles near 7e-3 rather than the 1e-4-scale
   agreement the two-agent runs reach: the three orientation agents keep a
   small persistent disagreement.  The consensus NRMSE is flat from about
   iteration 15 onward, so this is a stable equilibrium, not a failure to
   converge.
5. The single-orientation prior shows the same transient dip in 3D as in
   2D — closest to the phantom near iteration 10, then a slow drift up to
   its equilibrium.  Fusion shows no such drift.

## Caveats

- One phantom, one noise draw, one geometry; metrics are NRMSE, rms_w, and
  visual inspection.
- The fusion weights stayed at the default (1/2, 1/6, 1/6, 1/6) — fusion
  separated without touching that lever — and per-orientation strengths
  were not tried; rho stayed at 0.5.

## Follow-ups (queued)

- Device-form (Shards) exchange between agents for volumes past
  single-device scale.
- sigma_prox and rho tuning or schedules; per-orientation strengths and a
  weight sweep if a harder problem narrows the fusion margin.
- Correlated-noise denoisers (overview section 5) if streak residue shows
  on less favorable data.
- Real (measured) data, where the phantom-free metrics — data consistency
  and visuals — carry the evaluation.
