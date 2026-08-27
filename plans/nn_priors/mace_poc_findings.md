# MACE with a neural-network denoiser prior — proof-of-concept findings

2026-08-27.  Code: `experiments/drunet/` in the mbirtorch repo.  Run records:
`experiments/drunet/output/qggmrf_gate.npz` and `drunet_sweep.npz`.  Plan:
`mace_poc_plan.md` beside this file.

## What was built

- `mace.py` — the N-agent weighted Mann loop; agents exchange torch tensors
  on one fixed device.
- `agents.py` — ForwardProxAgent (wraps `prox_map`; its cumulative iteration
  count walks the partition sequence coarse to fine), QGGMRFDenoiserAgent
  (pinned prior parameters; `sigma_noise` is the strength knob), DRUNetAgent
  (pretrained grayscale DRUNet via deepinv, fixed intensity scale, output
  blended through the region-of-reconstruction mask), and `load_drunet`.
- Three small library fixes, each with a test: `denoise` keeps
  `sigma_y = sigma_noise` unconditionally; `prox_map`'s cached resume path
  honors `first_iteration`; the denoiser accepts an all-zero image (its NMAE
  statistic divided by the zero norm).

## Gate: MACE with the qGGMRF agent reproduces direct recon

On the 2D noisy cone-beam problem (128 x 128, dosage 500), matched strengths
and pinned prior parameters:

- NRMSE(MACE consensus, direct recon) = 0.0065 at 30 outer iterations
  (3 prox + 8 denoise inner iterations; consensus spread 8.5e-6).
  PASS at the 1% tolerance.
- Matched-sigma sweep at 0.5x / 1x / 2x the auto sigma_prox lands at
  0.0027 / 0.0065 / 0.0087 vs direct — the equilibrium is sigma-independent
  to under 1%, which bounds the damage from the inexact fixed-iteration
  inner solves.
- From-zero initialization descends monotonically, 0.63 to 0.027 in 30
  iterations, still closing — convergence from far away is well behaved.

## DRUNet: postprocessing vs prior

NRMSE vs phantom on the same problem (the incumbent direct qGGMRF recon:
0.385):

- DRUNet as postprocessing of the direct recon: best 0.124 at
  sigma_scaled 0.10.
- DRUNet as the MACE prior: best 0.139 at 30 outer iterations; refining the
  grid and running to convergence (60 iterations, spread 6.2e-4) gives
  0.112 at sigma_scaled 0.075.  The prior formulation beats postprocessing
  once the equilibrium is actually reached.
- The strength knob behaves: a clear interior optimum, and below it
  (sigma_scaled 0.02) the MACE recon is under-regularized and WORSE than
  direct (0.419) — expected, since the qGGMRF prior is absent from the MACE
  formulation and the weak denoiser is the only regularization.
- Coarse sweep at 30 iterations (postproc / MACE): 0.02 -> 0.310/0.419,
  0.05 -> 0.223/0.160, 0.10 -> 0.124/0.139, 0.15 -> 0.188/0.182,
  0.20 -> 0.211/0.204.

## Caveats

- One problem instance, one noise draw, NRMSE only — no visual or
  data-consistency comparison yet.
- sigma_prox stayed at its auto value and rho at 0.5; only the denoiser
  strength was swept.
- Two agents, one device, 2D.

## Follow-ups (queued)

- Multi-slice fusion: three orientation denoiser agents plus the forward
  agent on a 3D problem — a parameter change to the loop, new DRUNet slicing.
- 3D scale: exchange volumes between agents in the divided device form
  (prox_map and denoise both already accept it via configure_devices(like=)).
- Sweep sigma_prox and rho; strength/rho schedules (eventually-frozen rule).
- Correlated-noise denoisers (overview section 5) if streak residue shows in
  3D.
- Report data-consistency (forward-model RMSE) alongside NRMSE, and add
  visual comparisons.
