# Inner iterations of the MACE denoiser agent

Script: `mace_inner_iterations.py`.  Measured on mbirtorch at b0b1879
(`greg_dev` at 5f62506 plus PR #15), on CPU with 4 threads, 2026-10-03.
Results: `results/mace_inner_iterations/traces.csv` (NRMSE and MACE's
percent change after each MACE iteration, one column per case) and
`summary.txt`.

## Setup

The problem is the one of `tests/test_mace.py`: a 2D cone-beam scan of a
Shepp-Logan phantom with 64 views, 64 detector channels, and one detector
row.  The image has 64 by 64 by 1 voxels.  MACE runs two agents with equal
weights and \( \rho = 0.5 \).  The forward agent runs 3 VCD iterations per
call with its warm start.  `QGGMRFDenoiserAgent` runs the count in the first
column of the tables, starting from its previous output.  Every run starts
from a 30-iteration reconstruction and runs 60 MACE iterations.

With equal weights, the MACE solution minimizes the data term plus the prior,
whatever `sigma_prox` is **[derived]**.  So a 400-iteration reconstruction is
the reference for every run.  Its NRMSE against a 200-iteration
reconstruction was \( 5 \times 10^{-5} \).

The denoiser agent's strength is \( r = \sigma_x / \sigma_{prox} \).  With
mbirtorch's automatic parameters, `sigma_prox` and `sigma_x` were both 0.02208,
so \( r = 1 \).  The two are equal because `auto_set_sigma_x` and
`auto_set_sigma_prox` apply the same formula to the same estimate.  A second
set of runs multiplies `sigma_prox` by 5, so that \( r = 0.2 \), near the
strength of the MACE4D denoisers.

## Results

The NRMSE is against the reference.  The start had an NRMSE of 0.0251.

At \( r = 1 \), the automatic pairing:

| Inner | NRMSE at 10 | at 20 | at 40 | at 60 | Iterations to 0.01 | to 0.003 | s per iteration | Denoiser share of time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.01737 | 0.01169 | 0.00553 | 0.00271 | 25 | 58 | (0.520) | (0.18) |
| 2 | 0.01645 | 0.01052 | 0.00453 | 0.00204 | 22 | 51 | 0.439 | 0.02 |
| 4 | 0.01618 | 0.01020 | 0.00427 | 0.00188 | 21 | 49 | 0.440 | 0.04 |
| 8 | 0.01616 | 0.01017 | 0.00425 | 0.00186 | 21 | 49 | 0.457 | 0.08 |
| 16 | 0.01616 | 0.01017 | 0.00424 | 0.00186 | 21 | 49 | 0.495 | 0.14 |
| 32 | 0.01616 | 0.01017 | 0.00424 | 0.00185 | 21 | 49 | 0.559 | 0.25 |

At \( r = 0.2 \), with `sigma_prox` multiplied by 5:

| Inner | NRMSE at 10 | at 20 | at 40 | at 60 | Iterations to 0.01 | to 0.003 | to 0.001 | s per iteration | Denoiser share of time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.09209 | 0.04110 | 0.01823 | 0.00669 | 46 | - | - | 0.411 | 0.01 |
| 2 | 0.06417 | 0.03067 | 0.00880 | 0.00244 | 33 | 56 | - | 0.410 | 0.02 |
| 4 | 0.05811 | 0.02370 | 0.00393 | 0.00090 | 30 | 44 | 59 | 0.418 | 0.04 |
| 8 | 0.05854 | 0.01708 | 0.00361 | 0.00087 | 28 | 43 | 59 | 0.449 | 0.08 |
| 16 | 0.05105 | 0.01664 | 0.00363 | 0.00088 | 28 | 43 | 59 | 0.479 | 0.14 |
| 32 | 0.05086 | 0.01661 | 0.00363 | 0.00088 | 28 | 43 | 59 | 0.559 | 0.25 |

The first run includes the compilation of both agents, so its time and share,
in parentheses, are not comparable with the others.

These results indicate that the default of 8 inner iterations is enough on
this problem at both strengths.  At \( r = 1 \), 4 iterations gave the same
NRMSE as 32 to within 0.00003 at every reported iteration.  At \( r = 0.2 \),
8 iterations reached each target at the same iteration as 32.  Fewer inner
iterations slowed the loop: at \( r = 0.2 \), one inner iteration needed 46
MACE iterations to reach 0.01, against 28.

The test has two limits.  The image is a single 64 by 64 slice, so the
denoiser's share of the time is small and dominated by the cost per call.
And the two strengths bracket only part of the range.  From the standalone
measurements in `denoise_convergence.md`, the iterations a cold start needs
grow about as \( 1 / r^2 \), so an agent at \( r = 0.1 \) or below would need
more than 8 **[inferred]**.
