# MACE with a neural-network denoiser prior — proof-of-concept plan

Status: ACTIVE 2026-08-27.  Code lives in the `mbirtorch` repo at
`experiments/drunet/` (starting point: `cone_beam_2d.py`, the 2D noisy cone-beam
problem).  Background survey: `denoiser_overview.md` beside this plan.

## Goal

Run MACE (plug-and-play consensus equilibrium) on the 2D cone-beam problem, with the
tomography model's proximal map as the forward agent and a denoiser as the prior agent.
Develop and validate the loop with the qGGMRF denoiser, which has an exact answer to
check against, then swap in DRUNet.  Applying the denoiser once to the finished recon
(postprocessing) is kept as a baseline, not the method.

## Design decisions

- **General N-agent loop from the start.**  Weighted Mann iteration on the stacked
  operator:

      X_i = F_i(w_i)                      # agent evaluations
      z   = sum_i mu_i (2 X_i - w_i)
      w_i += 2 rho (z - X_i)
      x_bar = sum_i mu_i X_i              # consensus estimate

  With N=2, mu=(1/2,1/2), rho=1/2 this is PnP-ADMM / Douglas-Rachford.  Multi-slice
  fusion later (1 forward + 3 denoiser agents) is then a parameter change, not a rewrite.
- **Agent interface.**  An agent is a callable volume -> volume, where the
  volume is a torch tensor of shape (rows, cols, slices) on one fixed device
  -- in and out, so a GPU run stays on the GPU end to end (prox_map and
  denoise both accept and return device tensors via output_sharded=True).
  Every knob is bound at construction.  Three wrappers:
  ForwardProxAgent (wraps `TomographyModel.prox_map`), QGGMRFAgent (wraps
  `QGGMRFDenoiser.denoise`), DRUNetAgent (wraps the pretrained net; the `deepinv`
  dependency stays inside this one wrapper).
- **Knobs.**  The denoiser's `sigma_noise` is THE regularization-strength knob a user
  sweeps; the forward agent's `sigma_prox` stays fixed at its auto value.  When both
  agents are proximal maps the equilibrium solves f + (sigma_noise/sigma_prox)^2 h, so
  the ratio sets the strength, and holding sigma_prox fixed makes sigma_noise behave
  the way it does for an ordinary denoiser.
- **Eventually-frozen operators.**  Schedules are allowed early — the forward agent
  walks coarse-to-fine partition granularity across MACE iterations — but every agent
  must become a fixed operator in the tail so the equilibrium is well defined.
  (Denoising-strength schedules are future work.)  No per-call auto-estimation inside
  agents: prior parameters pinned, DRUNet intensity scale fixed once, inner-iteration
  counts fixed (stop thresholds set to 0).
- **Warm starts.**  Each agent's inner solve initializes from that agent's own previous
  OUTPUT, via `prox_map(init_recon=...)` and `denoise(init_image=...)`.  Outputs
  converge to the consensus; inputs converge to consensus + a nonzero dual offset, so
  the previous input is the wrong warm start.  DRUNet is feedforward — nothing to warm
  start.

## Correctness gates (the qGGMRF phase)

With matched strengths sigma_noise = sigma_prox = sigma, and the denoiser's prior
parameters pinned to the direct recon's values (sigma_x, p, q, T, and the neighbor
weights, read from the recon's `recon_dict`), the equilibrium solves the SAME objective
as `recon()`.  So:

1. **Equality gate.**  MACE consensus vs the direct qGGMRF recon: NRMSE within about
   1% (the proxes are inexact 3-iteration solves in float32, so the gate is a
   tolerance, not bitwise).  Initializing at the direct recon is a valid gate: the
   fixed point sits at w_i = x* + u_i with nonzero duals, so the loop still has to
   find the duals, and any drift of the consensus away from x* directly measures
   equilibrium mismatch.  A from-zero run is an optional robustness check.
2. **Sigma-independence.**  With exact matched proxes the equilibrium does not depend
   on sigma.  Sweep sigma and report how far the answer moves — that drift measures
   the damage from the inexact inner solves.

## Library changes needed (small, each with a test)

1. **`QGGMRFDenoiser.denoise`: sync sigma_y <- sigma_noise unconditionally.**  For the
   identity forward model, sigma_y IS sigma_noise by definition — it is not an
   estimate.  Today the sync lives inside the `auto_regularize_flag`-gated path
   (`auto_set_sigma_y` is only reached from `auto_set_regularization_params`), so a
   pinned denoiser (auto_regularize_flag=False — the correct agent configuration)
   silently keeps a stale sigma_y and the sigma_noise knob goes dead.  Move the sync
   out of the gated path so sigma_noise is the primary control in both configurations.
2. **`prox_map` resume path: honor first_iteration.**  With do_initialization=False,
   the cached partition sequence is replayed from its start on every call, and
   first_iteration only labels the log lines — although the docstring promises a
   "partition-sequence offset for restarts."  (On the initializing path,
   `initialize_recon` does slice `partition_sequence[first_iteration:]`.)  Fix: on the
   resume path, recompute the sequence from the model's partition_sequence parameter
   exactly as `initialize_recon` computes it (extend to max_iterations by repeating the
   last element, then drop the first first_iteration entries); the cache keeps the
   expensive pieces — the pixel partitions and the regularization estimates.  A
   PnP loop that passes its cumulative inner-iteration count then walks coarse-to-fine
   and ends on the 128-subset partitions (the default granularity list already ends
   at 128).  Fallback without a library change: call with do_initialization=True and a
   growing first_iteration, which re-runs initialization each call — acceptable at 2D
   scale, wasteful at 3D.
3. **`QGGMRFDenoiser.denoise`: accept an all-zero image** (found by the from-zero gate
   run): the NMAE convergence statistic divided by the zero image's norm and raised
   ZeroDivisionError.  Both denoiser paths now give nan instead, matching _vcd_recon's
   stated convention for a zero recon.

## Increments

Status (updated as work proceeds):

| # | Increment | Status |
|---|---|---|
| 1 | Loop, agents, qGGMRF gate | done 2026-08-27; gate PASS at NRMSE 0.0065 vs the standard recon (consensus spread 8.5e-6); matched-sigma sweep at 0.5x/2x lands at 0.0027/0.0087, so inexact-prox drift is under 1%; from-zero descends monotonically to 0.027 at 30 iterations; the run surfaced library change 3; records in `experiments/drunet/output/qggmrf_gate.npz` |
| 2 | DRUNet agent and sweep | done 2026-08-27; standard recon 0.385, best postprocessing 0.124 (sigma_scaled 0.10), best MACE prior 0.112 (sigma_scaled 0.075, 60 iterations, spread 6.2e-4); deepinv 0.4.1 added to the miniforge test env; records in `experiments/drunet/output/drunet_sweep.npz` |
| 3 | Findings and follow-ups | done 2026-08-27; `mace_poc_findings.md`, with the follow-up queue carried there; the first queued item is planned in `multi_slice_fusion.md` |
| 4 | Parameter sandbox (added at Greg's direction) | done 2026-08-27; `experiments/drunet/sandbox.py` compares the standard recon, DRUNet postprocessing, and the DRUNet MACE recon side by side, editing constants at the top of the file, with the standard recon cached across runs; the "standard recon" naming rule was adopted here and in the runner scripts' output ("direct recon" is reserved for the FDK-style `recon_direct`) |

**Increment 1: loop, agents, gate.**  `experiments/drunet/mace.py` (the loop plus
per-iteration traces: consensus spread max_i ||X_i - x_bar|| / ||x_bar||, and NRMSE
vs phantom), `agents.py`, and `run_qggmrf_gate.py` reusing the data setup from
`cone_beam_2d.py`.  Includes the library changes above.  Exit: both gates run and the
equality gate passes.

**Increment 2: DRUNet agent and sweep.**  Install `deepinv` in the local test env.
DRUNetAgent: grayscale weights; fixed intensity scale c chosen once from the initial
recon, applied as D(c v, c sigma)/c; pad/crop to multiples of 8 (128 already is).
Sweep sigma_noise with sigma_prox fixed.  Compare {standard qGGMRF recon, DRUNet
postprocessing of that recon, MACE-DRUNet} by NRMSE and side-by-side viewer.  Exit:
sweep table and best-sigma comparison saved under `experiments/drunet/output/`.

**Increment 3: findings and follow-ups.**  Short findings doc here (numbers quoted
from the run records), with the follow-up queue: multi-slice fusion (N=4 agents), 3D
volumes with device-form I/O between agents, rho and sigma schedules,
correlated-noise denoisers (overview section 5).

**Increment 4: parameter sandbox.**  Added after increment 3: an
edit-the-constants-and-run script showing the three reconstructions side by side,
with the number of iterations and the denoiser strength as the headline parameters
and the standard recon cached across runs.

## Mechanics

- Seed numpy (partitions come from the global RNG) and torch once per run; all gates
  are tolerance-based.
- rho = 0.5 default and exposed; the first remedy for a non-converging CNN agent is a
  smaller rho.
- Agents exchange single-device torch tensors; the host is touched only at the edges
  (data generation, caching, saving, viewing).  Multi-device (Shards) exchange between
  agents is deliberately out of scope until 3D.
