"""The MAP estimate of mbirtorch's 3D qGGMRF denoising model, found by L-BFGS in float64.

QGGMRFDenoiser.denoise, with its default neighbor weights, minimizes

    J(x) = |y - x|^2 / (2 sigma_y^2) + (1/6) sum over neighbor pairs (i, j) of rho(x_i - x_j),
    rho(d) = |d|^p / (p sigma_x^p) * u^(q - p) / (1 + u^(q - p)),   u = |d| / (T sigma_x).

The pairs are the nearest neighbors along each of the three axes.  No pair
crosses the boundary of the volume.  This is the 3D form of pcdrecon's
plans/image_mace/experiments/qggmrf_map.py, with two differences: the
gradient is computed from its formula rather than by autograd, and the
solver is scipy's L-BFGS-B.

All computation runs on the image divided by sigma_y.  In that unit the data
term is |z - v|^2 / 2, which is 1-strongly convex, and the prior is convex for
1 <= q <= p <= 2.  So for any v, the distance to the minimizer is at most the
norm of the gradient at v.  solve() reports this bound, so the accuracy of
each reference is checked rather than assumed.

exact_quadratic() gives the minimizer for q = p = 2 with the DCT.
"""

import numpy as np
import torch
from scipy import fft, optimize

PAIR_WEIGHT = 1.0 / 6.0     # mbirtorch's weight for each pair, with qggmrf_nbr_wts = [1, 1, 1]


def potential(d, r, p, q, T):
    """rho and its derivative at the neighbor differences d, in the unit of sigma_y.

    r is sigma_x / sigma_y.  With t = u^(p - q), rho = |d|^p / (p r^p) / (1 + t)
    and rho' = sign(d) |d|^(p - 1) / r^p * (1 + (q / p) t) / (1 + t)^2.  Both
    are finite at d = 0.
    """
    a = d.abs()
    t = torch.ones_like(a) if q == p else (a / (T * r)) ** (p - q)
    rho = a ** p / (p * r ** p) / (1 + t)
    drho = torch.sign(d) * a ** (p - 1) / r ** p * (1 + (q / p) * t) / (1 + t) ** 2
    return rho, drho


def cost_and_gradient(v, z, r, p, q, T):
    """J and its gradient at v, for float64 tensors v and z in the unit of sigma_y."""
    residual = v - z
    value = 0.5 * torch.sum(residual * residual)
    gradient = residual.clone()
    for axis in range(3):
        n = v.shape[axis]
        if n < 2:
            continue
        rho, drho = potential(torch.diff(v, dim=axis), r, p, q, T)
        value = value + PAIR_WEIGHT * torch.sum(rho)
        # The pair (i, i + 1) adds rho'(v[i + 1] - v[i]) to the gradient at i + 1
        # and subtracts it at i.
        gradient.narrow(axis, 1, n - 1).add_(PAIR_WEIGHT * drho)
        gradient.narrow(axis, 0, n - 1).sub_(PAIR_WEIGHT * drho)
    return float(value), gradient


def solve(y, sigma_y, sigma_x, p=2.0, q=1.2, T=1.0, init=None, history=20,
          rms_bound=1e-7, max_iterations=20000):
    """The minimizer of J for the 3D image y.

    Args:
        y: 3D image, in any unit
        sigma_y, sigma_x, p, q, T: the model's parameters, sigma_y and sigma_x in
            the unit of y
        init: starting image, or None to start from y
        history: L-BFGS history length
        rms_bound: stop when the rms of the gradient falls below this, which
            bounds the rms distance to the minimizer, in units of sigma_y
        max_iterations: L-BFGS iteration limit

    Returns:
        (x, info).  x is the minimizer in the unit of y.  info holds
        'iterations', 'evaluations', 'message', and 'distance_bound', the bound
        on the rms distance from x to the minimizer, in units of sigma_y.
    """
    shape = np.shape(y)
    num = int(np.prod(shape))
    z = torch.from_numpy(np.asarray(y, dtype=np.float64) / sigma_y)
    start = np.asarray(y if init is None else init, dtype=np.float64).ravel() / sigma_y
    r = sigma_x / sigma_y

    def fun(flat):
        value, gradient = cost_and_gradient(torch.from_numpy(flat.reshape(shape)), z, r, p, q, T)
        return value, gradient.numpy().ravel()

    # scipy's gtol bounds the largest gradient component, which bounds the rms.
    result = optimize.minimize(fun, start, jac=True, method='L-BFGS-B',
                               options={'maxcor': history, 'maxiter': max_iterations,
                                        'maxfun': 2 * max_iterations, 'ftol': 0.0,
                                        'gtol': rms_bound})
    _, gradient = fun(result.x)
    info = {'iterations': int(result.nit), 'evaluations': int(result.nfev),
            'message': str(result.message),
            'distance_bound': float(np.linalg.norm(gradient) / np.sqrt(num))}
    return result.x.reshape(shape) * sigma_y, info


def exact_quadratic(y, sigma_y, sigma_x):
    """The minimizer of J for q = p = 2, by the DCT.

    For q = p = 2, rho(d) = d^2 / (4 sigma_x^2), so J's gradient is zero where
    x + c L x = y, with c = sigma_y^2 / (12 sigma_x^2) and L the Laplacian of
    the grid's neighbor graph.  The graph has no pairs across the boundary, so
    the type-2 DCT diagonalizes L.  Its eigenvalues are 4 sin^2(pi k / (2 n))
    along each axis of length n, summed over the axes.
    """
    y = np.asarray(y, dtype=np.float64)
    c = sigma_y ** 2 / (12.0 * sigma_x ** 2)
    eigenvalues = [4.0 * np.sin(np.pi * np.arange(n) / (2.0 * n)) ** 2 for n in y.shape]
    total = (eigenvalues[0][:, None, None] + eigenvalues[1][None, :, None]
             + eigenvalues[2][None, None, :])
    return fft.idctn(fft.dctn(y, type=2, norm='ortho') / (1.0 + c * total),
                     type=2, norm='ortho')
