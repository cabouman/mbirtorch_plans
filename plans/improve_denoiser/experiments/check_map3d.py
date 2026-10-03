"""Checks of qggmrf_map3d, the reference solver of denoise_convergence.py.

The script runs four checks on a small volume:
  1. The gradient of cost_and_gradient matches autograd, for three priors.
  2. The prior of cost_and_gradient matches mbirtorch's qggmrf_loss.
  3. For q = 2, solve matches the exact minimizer that exact_quadratic
     computes with the DCT.
  4. At the minimizer that solve finds for q = 1.2, mbirtorch's gradient of
     the cost is as small as the module's gradient.  So the minimizer of the
     module's cost is also a minimizer of mbirtorch's cost.

Run on any machine with mbirtorch installed:
    python check_map3d.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

from mbirtorch import qggmrf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qggmrf_map3d  # noqa: E402

# ---- Run parameters ----------------------------------------------------------
SHAPE = (24, 20, 8)             # rows, columns, slices
SEED = 0
STEP = 3.0                      # height of the edge across the rows, in units of the noise
GRADIENT_PRIORS = [(2.0, 1.2, 1.0, 0.1), (2.0, 2.0, 1.0, 0.3), (1.8, 1.1, 0.5, 0.2)]   # (p, q, T, r)
LOSS_PRIOR = (2.0, 1.2, 1.0, 0.1)                     # (p, q, T, sigma_x) for check 2
QUADRATIC_RATIOS = (0.05, 0.1, 0.4)                   # r for check 3
FIXED_POINT_RATIOS = (0.05, 0.2)                      # r for check 4
RMS_BOUND = 1e-10               # the rms gradient at which solve stops


def autograd_cost(v, z, r, p, q, T):
    """The cost of qggmrf_map3d in units of sigma_y, written for autograd,
    with |d| smoothed at 0 so that its derivative exists."""
    cost = 0.5 * torch.sum((v - z) ** 2)
    for axis in range(3):
        d = torch.diff(v, dim=axis)
        a = torch.sqrt(d * d + 1e-30)
        t = (a / (T * r)) ** (p - q) if q != p else torch.ones_like(a)
        cost = cost + qggmrf_map3d.PAIR_WEIGHT * torch.sum(a ** p / (p * r ** p) / (1 + t))
    return cost


def main():
    rng = np.random.default_rng(SEED)
    y = rng.standard_normal(SHAPE) + STEP * (np.indices(SHAPE)[0] > SHAPE[0] // 2)
    sigma_y = 1.0
    b = qggmrf.get_b_from_nbr_wts([1, 1, 1])

    print('Check 1: gradient against autograd')
    for p, q, T, r in GRADIENT_PRIORS:
        v = torch.tensor(y + 0.3 * rng.standard_normal(SHAPE), requires_grad=True)
        z = torch.tensor(y)
        value, gradient = qggmrf_map3d.cost_and_gradient(v.detach(), z, r, p, q, T)
        reference = autograd_cost(v, z, r, p, q, T)
        reference.backward()
        print(f'  p {p:g}, q {q:g}, T {T:g}, r {r:g}: cost difference {abs(float(reference.detach()) - value):.1e}, '
              f'largest gradient difference {float((v.grad - gradient).abs().max()):.1e}')

    print('Check 2: prior against mbirtorch\'s qggmrf_loss')
    p, q, T, sigma_x = LOSS_PRIOR
    x = y + 0.3 * rng.standard_normal(SHAPE)
    mbir_prior = float(qggmrf.qggmrf_loss(x, (b, sigma_x, p, q, T)))
    value, _ = qggmrf_map3d.cost_and_gradient(torch.tensor(x), torch.tensor(x), sigma_x, p, q, T)
    print(f'  mbirtorch {mbir_prior:.12g}, qggmrf_map3d {value:.12g}, '
          f'relative difference {abs(mbir_prior - value) / value:.1e}')

    print('Check 3: solve against the exact minimizer for q = 2')
    for r in QUADRATIC_RATIOS:
        solution, info = qggmrf_map3d.solve(y, sigma_y, r * sigma_y, 2.0, 2.0, 1.0, rms_bound=RMS_BOUND)
        exact = qggmrf_map3d.exact_quadratic(y, sigma_y, r * sigma_y)
        print(f'  r {r:g}: rms difference {np.sqrt(np.mean((solution - exact) ** 2)):.1e}, '
              f'largest {np.abs(solution - exact).max():.1e}, {info["iterations"]} iterations')

    print('Check 4: mbirtorch\'s gradient at the minimizer for q = 1.2')
    pixels = torch.arange(SHAPE[0] * SHAPE[1])
    for r in FIXED_POINT_RATIOS:
        solution, info = qggmrf_map3d.solve(y, sigma_y, r * sigma_y, 2.0, 1.2, 1.0, rms_bound=RMS_BOUND)
        flat = torch.tensor(solution.reshape(-1, SHAPE[2]))
        prior_grad, _ = qggmrf.qggmrf_gradient_and_hessian_at_indices(
            flat, SHAPE, pixels, (b, r * sigma_y, 2.0, 1.2, 1.0))
        total = prior_grad + (flat - torch.tensor(y.reshape(-1, SHAPE[2]))) / sigma_y ** 2
        print(f'  r {r:g}: rms gradient {float(total.pow(2).mean().sqrt()):.1e} from mbirtorch and '
              f'{info["distance_bound"]:.1e} from qggmrf_map3d')


if __name__ == '__main__':
    main()
