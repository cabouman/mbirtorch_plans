"""Compare three ways to denoise a stack of independent 3D volumes in torch.

MACE4D denoises P independent volumes of shape (T, D1, D2) with one shared set
of qGGMRF denoiser constants.  Option C runs the production single-image sweep
once per volume.  Option D adds a leading batch dimension to the sweep and runs
all volumes at once, with a per-volume step size and a per-volume stop test.
Option A stacks the volumes along the row axis and runs the production sweep
unchanged on the one tall image, in three variants.

Nothing in the mbirtorch package is modified.  The batched functions here are
copies of the package formulas with every gather moved to dimension 1.

Run parameters are the constants below; the script takes no arguments.
"""

import time

import numpy as np
import torch

import mbirtorch
from mbirtorch import projectors as _projectors
from mbirtorch import qggmrf as _qggmrf
from mbirtorch import vcd_utils
from mbirtorch._memory_ledger import image_ell1
from mbirtorch.denoising import vcd_subset_denoiser
from mbirtorch.projectors import maybe_compile

# ── run parameters ───────────────────────────────────────────────────────────
CORRECTNESS_CASE = dict(P=8, T=6, D1=24, D2=24)
TIMING_CASE = dict(P=64, T=8, D1=64, D2=64)
SIGMA = 0.05
MAX_ITERS = 15
STOP_PCT = 0.2
SEED = 0
DEVICES = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else [])
TIMING_REPEATS = 2

# If the estimated time for one device's timing block exceeds this many
# seconds, P is halved for that device and the output says so.
TIMING_BUDGET_SEC = 600.0

# Same value the package uses in vcd_subset_denoiser.
EPS = float(np.finfo(np.float32).eps)
MAX_ALPHA = 1.5
STOP_THRESH = STOP_PCT / 100.0
SIGMA_X_FLOOR = 1e-6


# ── test data ────────────────────────────────────────────────────────────────
def make_stack(num_vols, num_frames, d1, d2):
    """A stack of num_vols noisy copies of one ramp volume.

    The noise amplitude rises with the volume index, so the volumes reach the
    stopping threshold at different iterations.
    """
    rng = np.random.default_rng(3)
    base = np.linspace(0, 1, num_frames * d1 * d2).reshape(num_frames, d1, d2)
    denom = (num_vols - 1) if num_vols > 1 else 1
    x = np.empty((num_vols, num_frames, d1, d2), dtype=np.float32)
    for p in range(num_vols):
        noise_amp = 0.02 + 0.06 * p / denom
        x[p] = (base + noise_amp * rng.normal(size=(num_frames, d1, d2))).astype(np.float32)
    return x


# ── shared denoiser constants ────────────────────────────────────────────────
def build_constants(x, device):
    """Configure a denoiser for one volume shape and return the sweep constants.

    This mirrors _configure_denoiser and _denoise_constants in the mbirjax
    MACE4D code, using the mbirtorch equivalents.  Statistics come from the
    whole stack merged into one 3D array.
    """
    num_vols, num_frames, d1, d2 = x.shape
    denoiser = mbirtorch.QGGMRFDenoiser((num_frames, d1, d2))
    denoiser.set_params(no_warning=True, verbose=0, use_ror_mask=False,
                        sigma_noise=SIGMA)

    image_for_stats = x.reshape(-1, d1, d2)
    sino_indicator = denoiser._get_sino_indicator(image_for_stats)
    denoiser.auto_set_sigma_y(image_for_stats, sino_indicator)
    recon_std = denoiser._get_estimate_of_recon_std(image_for_stats, sino_indicator)
    if not np.isfinite(recon_std):
        recon_std = 0.0
    denoiser.auto_set_sigma_x(recon_std)
    sigma_x = denoiser.get_params('sigma_x')
    if not np.isfinite(sigma_x) or sigma_x < SIGMA_X_FLOOR:
        denoiser.set_params(no_warning=True, sigma_x=SIGMA_X_FLOOR)

    fm_constant = 1.0 / (denoiser.get_params('sigma_y') ** 2.0)
    nbr_wts, sigma_x, p_par, q_par, t_par = denoiser.get_params(
        ['qggmrf_nbr_wts', 'sigma_x', 'p', 'q', 'T'])
    qggmrf_params = (_qggmrf.get_b_from_nbr_wts(nbr_wts), sigma_x, p_par, q_par, t_par)

    granularity = denoiser.get_params('granularity')
    num_pixels = num_frames * d1
    num_subsets = max(1, min(granularity[0], num_pixels // 64))
    np.random.seed(SEED)
    partition = vcd_utils.gen_set_of_pixel_partitions(
        (num_frames, d1, d2), [num_subsets], device=device, use_ror_mask=False)[0]

    return dict(denoiser=denoiser, fm_constant=fm_constant,
                qggmrf_params=qggmrf_params, partition=partition,
                num_subsets=num_subsets, granularity=granularity,
                sigma_x=sigma_x, sigma_y=denoiser.get_params('sigma_y'),
                compile_enabled=denoiser.compile_enabled,
                image_shape=(num_frames, d1, d2))


# ── the batched formulas (Option D) ──────────────────────────────────────────
def grad_hess_batched(flat, image_shape, pixel_indices, qggmrf_params):
    """qGGMRF gradient and Hessian diagonal for a batch of flat images.

    flat has shape (P, num_pixels, S).  This repeats the formulas of
    qggmrf_gradient_and_hessian_at_indices with every gather along dimension 1.
    There are no halos.
    """
    b, sigma_x, p, q, t_par = qggmrf_params
    num_rows, num_cols = image_shape[0], image_shape[1]

    cylinders = flat[:, pixel_indices]                       # (P, N, S)
    left_val = cylinders[..., :1]
    right_val = cylinders[..., -1:]
    delta = torch.cat((cylinders[..., :1] - left_val,
                       cylinders[..., 1:] - cylinders[..., :-1],
                       right_val - cylinders[..., -1:]), dim=-1)   # (P, N, S+1)

    b_tilde_2 = _qggmrf.get_2_b_tilde(delta, 1.0, qggmrf_params)
    b_tilde_2_delta = b_tilde_2 * delta

    b_slice_plus, b_slice_minus = b[4], b[5]
    gradient = (-b_slice_plus * b_tilde_2_delta[..., 1:]
                + b_slice_minus * b_tilde_2_delta[..., :-1])
    hessian = (b_slice_plus * b_tilde_2[..., 1:]
               + b_slice_minus * b_tilde_2[..., :-1])

    row_index = pixel_indices // num_cols
    col_index = pixel_indices % num_cols
    xs0 = cylinders

    offsets_and_b = [((1, 0), b[0]), ((-1, 0), b[1]),
                     ((0, 1), b[2]), ((0, -1), b[3])]
    for (dr, dc), b_value in offsets_and_b:
        r = (row_index + dr).clamp(0, num_rows - 1)
        c = (col_index + dc).clamp(0, num_cols - 1)
        neighbor = flat[:, r * num_cols + c]                 # (P, N, S)
        delta = xs0 - neighbor
        b_tilde_2 = _qggmrf.get_2_b_tilde(delta, b_value, qggmrf_params)
        gradient = gradient + b_tilde_2 * delta
        hessian = hessian + b_tilde_2

    return gradient, hessian


def subset_update_batched(flat_image, flat_error_image, pixel_indices,
                          fm_constant, qggmrf_params, image_shape, active):
    """One VCD subset update for a batch of flat images, with per-volume scalars.

    Mutates both state tensors in place.  A volume whose active entry is False
    gets step size zero, so its state does not change.
    """
    prior_grad, prior_hess = grad_hess_batched(flat_image, image_shape,
                                               pixel_indices, qggmrf_params)

    cur_error_image = flat_error_image[:, pixel_indices]
    forward_grad = -fm_constant * cur_error_image
    forward_hess = 1

    delta_recon_at_indices = -((forward_grad + prior_grad)
                               / (forward_hess + prior_hess))

    prior_linear = torch.sum(prior_grad * delta_recon_at_indices, dim=(1, 2))
    prior_quadratic_approx = torch.sum(prior_hess * delta_recon_at_indices ** 2,
                                       dim=(1, 2))

    delta_sinogram = delta_recon_at_indices
    forward_linear = fm_constant * torch.sum(cur_error_image * delta_sinogram,
                                             dim=(1, 2))
    forward_quadratic = fm_constant * torch.sum(delta_sinogram * delta_sinogram,
                                                dim=(1, 2))

    alpha_numerator = forward_linear - prior_linear
    alpha_denominator = forward_quadratic + prior_quadratic_approx + EPS
    alpha = alpha_numerator / alpha_denominator
    alpha = torch.clamp(alpha, EPS, MAX_ALPHA)
    alpha = torch.where(active, alpha, torch.zeros_like(alpha))

    delta_scaled = alpha[:, None, None] * delta_recon_at_indices
    flat_image.index_add_(1, pixel_indices, delta_scaled)

    cur_error_image = cur_error_image - delta_scaled
    flat_error_image.index_copy_(1, pixel_indices, cur_error_image)
    ell1_for_subset = torch.sum(torch.abs(delta_scaled), dim=(1, 2))
    return flat_image, flat_error_image, ell1_for_subset, alpha


# ── sweeps ───────────────────────────────────────────────────────────────────
def sync(device):
    if device == 'mps':
        torch.mps.synchronize()


def production_sweep(flat_image, flat_error_image, partition, fm_constant,
                     qggmrf_params, image_shape, subset_fn, max_iters=MAX_ITERS):
    """The single-image sweep of QGGMRFDenoiser.denoise, one global stop test."""
    num_iters = 0
    with torch.no_grad():
        for i in range(max_iters):
            ell1_accum = 0.0
            alpha_accum = 0.0
            for k in range(partition.shape[0]):
                flat_image, flat_error_image, ell1_subset, alpha_subset = subset_fn(
                    flat_image, flat_error_image, partition[k], fm_constant,
                    qggmrf_params, image_shape)
                ell1_accum = ell1_accum + ell1_subset
                alpha_accum = alpha_accum + alpha_subset
            image_l1 = float(image_ell1(flat_image))
            nmae = float(ell1_accum) / image_l1 if image_l1 else float('nan')
            num_iters = i + 1
            if nmae < STOP_THRESH:
                break
    return flat_image, num_iters


def run_option_c(x, const, device, subset_fn):
    """Option C: the production sweep, once per volume."""
    num_vols, num_frames, d1, d2 = x.shape
    y = np.empty_like(x)
    iters = []
    for p in range(num_vols):
        flat_image = torch.as_tensor(x[p]).to(device).reshape(
            num_frames * d1, d2).clone().contiguous()
        flat_error_image = torch.zeros_like(flat_image)
        flat_image, n = production_sweep(
            flat_image, flat_error_image, const['partition'], const['fm_constant'],
            const['qggmrf_params'], const['image_shape'], subset_fn)
        y[p] = flat_image.reshape(num_frames, d1, d2).cpu().numpy()
        iters.append(n)
    return y, iters


def run_option_d(x, const, device, subset_fn):
    """Option D: one sweep over an explicit leading batch dimension."""
    num_vols, num_frames, d1, d2 = x.shape
    partition = const['partition']
    flat = torch.as_tensor(x).to(device).reshape(
        num_vols, num_frames * d1, d2).clone().contiguous()
    flat_error = torch.zeros_like(flat)
    active = torch.ones(num_vols, dtype=torch.bool, device=flat.device)
    iters = np.zeros(num_vols, dtype=int)
    with torch.no_grad():
        for i in range(MAX_ITERS):
            ell1_accum = torch.zeros(num_vols, dtype=flat.dtype, device=flat.device)
            alpha_accum = torch.zeros(num_vols, dtype=flat.dtype, device=flat.device)
            for k in range(partition.shape[0]):
                flat, flat_error, ell1_subset, alpha_subset = subset_fn(
                    flat, flat_error, partition[k], const['fm_constant'],
                    const['qggmrf_params'], const['image_shape'], active)
                ell1_accum = ell1_accum + ell1_subset
                alpha_accum = alpha_accum + alpha_subset
            image_l1 = torch.sum(torch.abs(flat), dim=(1, 2))
            nmae = ell1_accum / image_l1
            active_host = active.cpu().numpy()
            iters[active_host] = i + 1
            active = active & ~(nmae < STOP_THRESH)
            if not bool(active.any()):
                break
    y = flat.reshape(num_vols, num_frames, d1, d2).cpu().numpy()
    return y, iters


def replicated_partition(partition, num_vols, rows_per_vol, d1):
    """Repeat one per-volume partition across the volumes of a stacked image."""
    parts = []
    for k in range(partition.shape[0]):
        pieces = [partition[k] + p * rows_per_vol * d1 for p in range(num_vols)]
        parts.append(torch.sort(torch.cat(pieces))[0])
    return torch.stack(parts)


def run_option_a1(x, const, device, subset_fn):
    """Option A1: volumes stacked on the row axis, per-volume partition repeated."""
    num_vols, num_frames, d1, d2 = x.shape
    stacked = x.reshape(num_vols * num_frames, d1, d2)
    part = replicated_partition(const['partition'], num_vols, num_frames, d1)
    flat_image = torch.as_tensor(stacked).to(device).reshape(
        num_vols * num_frames * d1, d2).clone().contiguous()
    flat_error = torch.zeros_like(flat_image)
    image_shape = (num_vols * num_frames, d1, d2)
    flat_image, n = production_sweep(flat_image, flat_error, part,
                                     const['fm_constant'], const['qggmrf_params'],
                                     image_shape, subset_fn)
    y = flat_image.reshape(num_vols, num_frames, d1, d2).cpu().numpy()
    return y, n


def run_option_a2(x, const, device, subset_fn):
    """Option A2: volumes stacked on the row axis, one fresh partition of the stack."""
    num_vols, num_frames, d1, d2 = x.shape
    stacked = x.reshape(num_vols * num_frames, d1, d2)
    granularity = const['granularity']
    num_pixels = num_vols * num_frames * d1
    num_subsets = max(1, min(granularity[0], num_pixels // 64))
    np.random.seed(SEED)
    part = vcd_utils.gen_set_of_pixel_partitions(
        (num_vols * num_frames, d1, d2), [num_subsets], device=device,
        use_ror_mask=False)[0]
    flat_image = torch.as_tensor(stacked).to(device).reshape(
        num_pixels, d2).clone().contiguous()
    flat_error = torch.zeros_like(flat_image)
    image_shape = (num_vols * num_frames, d1, d2)
    flat_image, n = production_sweep(flat_image, flat_error, part,
                                     const['fm_constant'], const['qggmrf_params'],
                                     image_shape, subset_fn)
    y = flat_image.reshape(num_vols, num_frames, d1, d2).cpu().numpy()
    return y, n, num_subsets


def run_option_a3(x, const, device, subset_fn):
    """Option A3: volumes stacked with one zero row of frames between them."""
    num_vols, num_frames, d1, d2 = x.shape
    block = num_frames + 1
    stacked = np.zeros((num_vols * block, d1, d2), dtype=np.float32)
    for p in range(num_vols):
        stacked[p * block:p * block + num_frames] = x[p]
    part = replicated_partition(const['partition'], num_vols, block, d1)
    flat_image = torch.as_tensor(stacked).to(device).reshape(
        num_vols * block * d1, d2).clone().contiguous()
    flat_error = torch.zeros_like(flat_image)
    image_shape = (num_vols * block, d1, d2)
    flat_image, n = production_sweep(flat_image, flat_error, part,
                                     const['fm_constant'], const['qggmrf_params'],
                                     image_shape, subset_fn)
    out = flat_image.reshape(num_vols * block, d1, d2).cpu().numpy()
    y = np.stack([out[p * block:p * block + num_frames] for p in range(num_vols)])
    return y, n


# ── comparison helpers ───────────────────────────────────────────────────────
def rel_max(a, b):
    """max|a - b| / max|b|, in float64 on the host."""
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    denom = np.max(np.abs(b64))
    return float(np.max(np.abs(a64 - b64)) / denom) if denom > 0 else float('nan')


def max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64)
                               - np.asarray(b, dtype=np.float64))))


# ── correctness ──────────────────────────────────────────────────────────────
def run_correctness(device):
    case = CORRECTNESS_CASE
    num_vols, num_frames, d1, d2 = case['P'], case['T'], case['D1'], case['D2']
    print('')
    print('=' * 78)
    print('CORRECTNESS CASE on {}: P={} T={} D1={} D2={}'
          .format(device, num_vols, num_frames, d1, d2))
    print('=' * 78)

    x = make_stack(num_vols, num_frames, d1, d2)
    const = build_constants(x, device)
    print('sigma_y            = {:.8g}'.format(const['sigma_y']))
    print('sigma_x            = {:.8g}'.format(const['sigma_x']))
    print('fm_constant        = {:.8g}'.format(const['fm_constant']))
    print('num_subsets        = {}'.format(const['num_subsets']))
    print('partition shape    = {}'.format(tuple(const['partition'].shape)))
    print('compile_enabled    = {}'.format(const['compile_enabled']))
    print('stop_thresh        = {:.6g}'.format(STOP_THRESH))

    compiled_prod = maybe_compile(vcd_subset_denoiser, const['compile_enabled'])
    compiled_batch = maybe_compile(subset_update_batched, const['compile_enabled'])

    y_ref, iters_ref = run_option_c(x, const, device, compiled_prod)
    y_batch, iters_batch = run_option_d(x, const, device, compiled_batch)

    print('')
    print('-- check 1: Option D against the reference loop --')
    d_rel = rel_max(y_batch, y_ref)
    print('D vs ref rel_max   = {:.6e}'.format(d_rel))
    print('D vs ref max abs   = {:.6e}'.format(max_abs(y_batch, y_ref)))

    print('')
    print('-- check 2: iteration counts lane by lane --')
    print('iters_ref          = {}'.format(list(int(v) for v in iters_ref)))
    print('iters_batch        = {}'.format(list(int(v) for v in iters_batch)))
    print('iteration counts equal = {}'.format(
        list(int(v) for v in iters_ref) == list(int(v) for v in iters_batch)))

    print('')
    print('-- check 3: batched gradient and Hessian against the package, eager --')
    flat0 = torch.as_tensor(x).to(device).reshape(
        num_vols, num_frames * d1, d2).clone().contiguous()
    with torch.no_grad():
        g_b, h_b = grad_hess_batched(flat0, const['image_shape'],
                                     const['partition'][0], const['qggmrf_params'])
        grad_equal, hess_equal = [], []
        grad_rel, hess_rel = [], []
        for p in range(num_vols):
            g_p, h_p = _qggmrf.qggmrf_gradient_and_hessian_at_indices(
                flat0[p], const['image_shape'], const['partition'][0],
                const['qggmrf_params'])
            grad_equal.append(bool(torch.equal(g_b[p], g_p)))
            hess_equal.append(bool(torch.equal(h_b[p], h_p)))
            grad_rel.append(rel_max(g_b[p].cpu().numpy(), g_p.cpu().numpy()))
            hess_rel.append(rel_max(h_b[p].cpu().numpy(), h_p.cpu().numpy()))
    print('gradient exactly equal per lane = {}'.format(grad_equal))
    print('hessian  exactly equal per lane = {}'.format(hess_equal))
    print('gradient rel_max max over lanes = {:.6e}'.format(max(grad_rel)))
    print('hessian  rel_max max over lanes = {:.6e}'.format(max(hess_rel)))

    print('')
    print('-- check 4: a single-lane batch against the reference for that lane --')
    x0 = x[:1]
    y_ref0_c, _ = run_option_c(x0, const, device, compiled_prod)
    y_d1_c, _ = run_option_d(x0, const, device, compiled_batch)
    same_c = bool(np.array_equal(y_d1_c, y_ref0_c))
    print('compiled: bitwise equal = {}'.format(same_c))
    if not same_c:
        print('compiled: rel_max       = {:.6e}'.format(rel_max(y_d1_c, y_ref0_c)))
    y_ref0_e, _ = run_option_c(x0, const, device, vcd_subset_denoiser)
    y_d1_e, _ = run_option_d(x0, const, device, subset_update_batched)
    same_e = bool(np.array_equal(y_d1_e, y_ref0_e))
    print('eager:    bitwise equal = {}'.format(same_e))
    if not same_e:
        print('eager:    rel_max       = {:.6e}'.format(rel_max(y_d1_e, y_ref0_e)))

    print('')
    print('-- Option A: stacked on the row axis, production sweep unchanged --')
    y_a1, n_a1 = run_option_a1(x, const, device, compiled_prod)
    y_a2, n_a2, ns_a2 = run_option_a2(x, const, device, compiled_prod)
    y_a3, n_a3 = run_option_a3(x, const, device, compiled_prod)
    a1_rel = rel_max(y_a1, y_ref)
    a2_rel = rel_max(y_a2, y_ref)
    a3_rel = rel_max(y_a3, y_ref)
    print('A1 vs ref rel_max  = {:.6e}   global iterations = {}'.format(a1_rel, n_a1))
    print('A2 vs ref rel_max  = {:.6e}   global iterations = {}   num_subsets = {}'
          .format(a2_rel, n_a2, ns_a2))
    print('A3 vs ref rel_max  = {:.6e}   global iterations = {}'.format(a3_rel, n_a3))

    if num_frames >= 3:
        interior = slice(1, num_frames - 1)
        a1_int = rel_max(y_a1[:, interior], y_ref[:, interior])
        edge_b = np.concatenate([y_a1[:, :1], y_a1[:, num_frames - 1:]], axis=1)
        edge_r = np.concatenate([y_ref[:, :1], y_ref[:, num_frames - 1:]], axis=1)
        a1_edge = rel_max(edge_b, edge_r)
        print('A1 interior frames (t=1..T-2) rel_max = {:.6e}'.format(a1_int))
        print('A1 edge frames (t=0 and t=T-1) rel_max = {:.6e}'.format(a1_edge))
    else:
        a1_int, a1_edge = float('nan'), float('nan')
        print('A1 interior and edge split skipped: T < 3')

    return dict(d_rel=d_rel, a1_rel=a1_rel, a2_rel=a2_rel, a3_rel=a3_rel,
                a1_int=a1_int, a1_edge=a1_edge,
                iters_ref=list(int(v) for v in iters_ref),
                iters_batch=list(int(v) for v in iters_batch),
                iters_equal=(list(int(v) for v in iters_ref)
                             == list(int(v) for v in iters_batch)),
                grad_equal=all(grad_equal), hess_equal=all(hess_equal),
                p1_bitwise_compiled=same_c, p1_bitwise_eager=same_e,
                n_a1=n_a1, n_a2=n_a2, n_a3=n_a3)


# ── timing ───────────────────────────────────────────────────────────────────
def time_call(fn, device):
    sync(device)
    t0 = time.perf_counter()
    out = fn()
    sync(device)
    return time.perf_counter() - t0, out


def run_timing_at(device, num_vols):
    """Time the three options on the timing case at the given volume count."""
    case = TIMING_CASE
    num_frames, d1, d2 = case['T'], case['D1'], case['D2']
    x = make_stack(num_vols, num_frames, d1, d2)
    const = build_constants(x, device)
    compiled_prod = maybe_compile(vcd_subset_denoiser, const['compile_enabled'])
    compiled_batch = maybe_compile(subset_update_batched, const['compile_enabled'])

    print('P={} T={} D1={} D2={}   num_subsets={}'
          .format(num_vols, num_frames, d1, d2, const['num_subsets']))

    t_warm_ref, _ = time_call(lambda: run_option_c(x, const, device, compiled_prod), device)
    print('warm-up reference loop : {:.3f} s'.format(t_warm_ref))
    t_ref_first, _ = time_call(lambda: run_option_c(x, const, device, compiled_prod), device)
    est = t_ref_first * TIMING_REPEATS * 3
    print('one timed reference run: {:.3f} s   estimated block total {:.1f} s'
          .format(t_ref_first, est))
    if est > TIMING_BUDGET_SEC and num_vols > 1:
        print('Estimated block total is above the {:.0f} s budget.  '
              'Halving P and starting the block over.'.format(TIMING_BUDGET_SEC))
        return None

    t_ref = [t_ref_first]
    for _ in range(TIMING_REPEATS - 1):
        t, _ = time_call(lambda: run_option_c(x, const, device, compiled_prod), device)
        t_ref.append(t)

    t_warm_d, (_, iters_d) = time_call(
        lambda: run_option_d(x, const, device, compiled_batch), device)
    print('warm-up Option D       : {:.3f} s'.format(t_warm_d))
    t_d = []
    for _ in range(TIMING_REPEATS):
        t, _ = time_call(lambda: run_option_d(x, const, device, compiled_batch), device)
        t_d.append(t)

    t_warm_a1, (_, n_a1) = time_call(
        lambda: run_option_a1(x, const, device, compiled_prod), device)
    print('warm-up Option A1      : {:.3f} s'.format(t_warm_a1))
    t_a1 = []
    for _ in range(TIMING_REPEATS):
        t, _ = time_call(lambda: run_option_a1(x, const, device, compiled_prod), device)
        t_a1.append(t)

    print('')
    print('mean of {} timed runs, seconds:'.format(TIMING_REPEATS))
    print('  reference loop : {:.3f}'.format(float(np.mean(t_ref))))
    print('  Option D       : {:.3f}'.format(float(np.mean(t_d))))
    print('  Option A1      : {:.3f}'.format(float(np.mean(t_a1))))
    print('Option D per-volume iterations: min={} max={} mean={:.2f}'
          .format(int(np.min(iters_d)), int(np.max(iters_d)), float(np.mean(iters_d))))
    print('Option A1 global iterations   : {}'.format(n_a1))
    return dict(P=num_vols, t_ref=float(np.mean(t_ref)), t_d=float(np.mean(t_d)),
                t_a1=float(np.mean(t_a1)),
                iters_d_min=int(np.min(iters_d)), iters_d_max=int(np.max(iters_d)),
                iters_d_mean=float(np.mean(iters_d)), n_a1=n_a1)


def run_timing(device):
    print('')
    print('=' * 78)
    print('TIMING CASE on {}'.format(device))
    print('=' * 78)
    num_vols = TIMING_CASE['P']
    while True:
        result = run_timing_at(device, num_vols)
        if result is not None:
            return result
        num_vols = max(1, num_vols // 2)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print('m4d1_batched_denoiser_options')
    print('torch {}   numpy {}'.format(torch.__version__, np.__version__))
    print('mbirtorch at {}'.format(mbirtorch.__file__))
    print('devices: {}'.format(DEVICES))
    print('torch.get_num_threads() = {}'.format(torch.get_num_threads()))
    print('MAX_ITERS={}  STOP_PCT={}  SIGMA={}  SEED={}  TIMING_REPEATS={}'
          .format(MAX_ITERS, STOP_PCT, SIGMA, SEED, TIMING_REPEATS))

    correctness = {}
    timing = {}
    for device in DEVICES:
        correctness[device] = run_correctness(device)
    for device in DEVICES:
        timing[device] = run_timing(device)

    print('')
    print('=' * 78)
    print('COMPILE ERRORS')
    print('=' * 78)
    errors = _projectors._COMPILE_ERRORS
    if errors:
        for key, value in errors.items():
            print('{}: {}'.format(key, value))
    else:
        print('(none: no compiled body fell back to eager)')

    for device in DEVICES:
        c = correctness[device]
        t = timing[device]
        print('')
        print('=' * 78)
        print('SUMMARY for {}'.format(device))
        print('=' * 78)
        print('{:<34} {:>16}'.format('quantity', 'value'))
        print('{:<34} {:>16.6e}'.format('D vs ref rel_max', c['d_rel']))
        print('{:<34} {:>16.6e}'.format('A1 vs ref rel_max', c['a1_rel']))
        print('{:<34} {:>16.6e}'.format('A1 interior frames rel_max', c['a1_int']))
        print('{:<34} {:>16.6e}'.format('A1 edge frames rel_max', c['a1_edge']))
        print('{:<34} {:>16.6e}'.format('A2 vs ref rel_max', c['a2_rel']))
        print('{:<34} {:>16.6e}'.format('A3 vs ref rel_max', c['a3_rel']))
        print('{:<34} {:>16}'.format('D iteration counts match ref', str(c['iters_equal'])))
        print('{:<34} {:>16}'.format('batched gradient exact', str(c['grad_equal'])))
        print('{:<34} {:>16}'.format('batched hessian exact', str(c['hess_equal'])))
        print('{:<34} {:>16}'.format('P=1 bitwise, compiled', str(c['p1_bitwise_compiled'])))
        print('{:<34} {:>16}'.format('P=1 bitwise, eager', str(c['p1_bitwise_eager'])))
        print('{:<34} {:>16}'.format('timing P', t['P']))
        print('{:<34} {:>16.3f}'.format('time reference loop (s)', t['t_ref']))
        print('{:<34} {:>16.3f}'.format('time Option D (s)', t['t_d']))
        print('{:<34} {:>16.3f}'.format('time Option A1 (s)', t['t_a1']))
        print('{:<34} {:>16}'.format('D iterations min/max',
                                     '{}/{}'.format(t['iters_d_min'], t['iters_d_max'])))
        print('{:<34} {:>16.2f}'.format('D iterations mean', t['iters_d_mean']))


if __name__ == '__main__':
    main()
