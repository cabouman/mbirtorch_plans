"""Scale-equivariance test of QGGMRFDenoiser.denoise (step 1 of the plan).

For each scale c, the noisy image c*y is denoised with sigma_noise = c*sigma_y
and, in the fixed-prior case, sigma_x = c*sigma_x.  Every run uses the same
pixel partition and a fixed number of iterations.  A correct denoiser returns
c times the result for c = 1.

The script runs the released update and a patched update, in which the
second derivative of the data term is fm_constant instead of 1.  The patch is
applied by replacing denoising.vcd_subset_denoiser in this process only.

Usage: python scale_equivariance.py [num_iterations]
"""
import sys

import numpy as np
import torch

import mbirtorch
from mbirtorch import denoising, qggmrf as _qggmrf

released_update = denoising.vcd_subset_denoiser


def patched_update(flat_image, flat_error_image, pixel_indices,
                   fm_constant, qggmrf_params, image_shape):
    """vcd_subset_denoiser with forward_hess = fm_constant."""
    prior_grad, prior_hess = _qggmrf.qggmrf_gradient_and_hessian_at_indices(
        flat_image, image_shape, pixel_indices, qggmrf_params)
    cur_error_image = flat_error_image[pixel_indices]
    forward_grad = -fm_constant * cur_error_image
    forward_hess = fm_constant
    delta = -((forward_grad + prior_grad) / (forward_hess + prior_hess))
    prior_linear = torch.sum(prior_grad * delta)
    prior_quadratic_approx = torch.sum(prior_hess * delta ** 2)
    forward_linear = fm_constant * torch.sum(cur_error_image * delta)
    forward_quadratic = fm_constant * torch.sum(delta * delta)
    alpha = ((forward_linear - prior_linear)
             / (forward_quadratic + prior_quadratic_approx + denoising._F32_EPS))
    alpha = torch.clamp(alpha, denoising._F32_EPS, 1.5)
    delta = alpha * delta
    flat_image.index_add_(0, pixel_indices, delta)
    cur_error_image = cur_error_image - delta
    flat_error_image.index_copy_(0, pixel_indices, cur_error_image)
    return flat_image, flat_error_image, torch.sum(torch.abs(delta)), alpha


def make_phantom(shape=(256, 256, 4), sigma_y=182.0, seed=0):
    """A water disk at 1000 HU in air at 0 HU, with three inserts, plus white
    Gaussian noise of standard deviation sigma_y.  Returns (clean, noisy)."""
    rows, cols, slices = shape
    r, c = np.meshgrid(np.arange(rows) - rows / 2, np.arange(cols) - cols / 2,
                       indexing='ij')
    clean = np.zeros((rows, cols), np.float32)
    clean[r ** 2 + c ** 2 < (0.45 * rows) ** 2] = 1000.0
    for (r0, c0, value) in [(-50, 0, 1300.0), (40, -50, 1050.0), (40, 50, 1800.0)]:
        clean[(r - r0) ** 2 + (c - c0) ** 2 < 20 ** 2] = value
    clean = np.repeat(clean[:, :, None], slices, axis=2)
    noise = sigma_y * np.random.RandomState(seed).randn(*shape).astype(np.float32)
    return clean, clean + noise


def run(noisy, sigma_noise, sigma_x, num_iterations):
    """Denoise with a fixed partition and a fixed number of iterations.
    sigma_x None means the automatic value.  Returns (image, sigma_x, alphas)."""
    denoiser = mbirtorch.QGGMRFDenoiser(noisy.shape)
    denoiser.configure_devices(devices=['cpu'])
    denoiser.set_params(no_warning=True, verbose=0, use_ror_mask=False)
    if sigma_x is not None:
        denoiser.set_params(no_warning=True, auto_regularize_flag=False,
                            sigma_x=sigma_x)
    np.random.seed(0)
    out, d = denoiser.denoise(noisy, sigma_noise=sigma_noise,
                              max_iterations=num_iterations,
                              stop_threshold_change_pct=0.0,
                              logfile_path=None, print_logs=False)
    return (np.asarray(out, np.float64), denoiser.get_params('sigma_x'),
            d['recon_params']['alpha_values'])


def main():
    num_iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    sigma_y = 182.0
    clean, noisy = make_phantom(sigma_y=sigma_y)
    sigma_x_fixed = 0.1 * sigma_y      # r = sigma_x / sigma_y = 0.1
    scales = [1e-3, 1.0, 1e3]
    print(f'iterations = {num_iterations}, sigma_y = {sigma_y} at c = 1')
    for name, update in [('released', released_update), ('patched', patched_update)]:
        denoising.vcd_subset_denoiser = update
        for prior in ['fixed sigma_x', 'automatic sigma_x']:
            results = {}
            for c in scales:
                sx = c * sigma_x_fixed if prior == 'fixed sigma_x' else None
                results[c] = run(c * noisy, c * sigma_y, sx, num_iterations)
            ref = results[1.0][0]
            moved_ref = np.sqrt(np.mean((ref - noisy) ** 2))
            print(f'\n{name} update, {prior}')
            print('     c   sigma_x/c   mean alpha   rms(x_c/c - y)   '
                  'rms(x_c/c - x_1)   rms(x_c/c - clean)')
            for c in scales:
                x, sx, alphas = results[c]
                xc = x / c
                print(f'{c:6.0e}   {sx / c:9.3f}   {np.mean(alphas):10.3f}   '
                      f'{np.sqrt(np.mean((xc - noisy) ** 2)):14.3f}   '
                      f'{np.sqrt(np.mean((xc - ref) ** 2)):16.3f}   '
                      f'{np.sqrt(np.mean((xc - clean) ** 2)):18.3f}')
            print(f'(rms distance moved by x_1 from y: {moved_ref:.3f} HU)')
    denoising.vcd_subset_denoiser = released_update


if __name__ == '__main__':
    main()
