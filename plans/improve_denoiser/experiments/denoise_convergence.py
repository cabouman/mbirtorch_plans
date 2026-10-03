"""Convergence of QGGMRFDenoiser.denoise to the MAP estimate, and its stopping rules (step 4 of the plan).

For each case in CASES, the script denoises a test volume made by
phantom3d.py with the denoiser's own VCD update, one iteration at a time.  The
volume is the 3D test volume, or its middle slice as a 2D image ('slice').
After each iteration the script records:
  distance            rms distance to the MAP estimate
  change              rms change of the image in this iteration
  gradient_statistic  rms of the gradient of J at each voxel just before the
                      voxel's update, over the iteration, times sigma_y
  exact_gradient      rms of the gradient of J at the image after the
                      iteration, times sigma_y.  Computed at the iterations
                      in BOUND_ITERATIONS only, because it costs about 0.3 s.
  percent_change      denoise's stopping statistic, 100 ||change||_1 / ||x||_1,
                      for the volume with each offset in OFFSETS added
  alpha               mean step size over the subsets
  flat_noise          rms error against the clean volume over its flat region
  seconds             time of the iteration's updates, without the statistics
Every quantity but percent_change, alpha, and seconds is in units of sigma_y.

The exact gradient is an upper bound on the distance.  In the unit of
sigma_y, J is |z - v|^2 / 2 plus a convex prior, so J is 1-strongly convex,
and the rms distance from any image to the minimizer is at most the rms of
its gradient [derived].  The gradient statistic is a statistic the VCD loop
could compute with one more reduction per subset, because each subset update
already computes the gradient at its voxels.

The MAP estimate comes from qggmrf_map3d.solve, whose distance bound is
recorded.  The loop is the one of denoise's single-device path, with the same
pixel partition and the same compiled update.  The gradient statistic is computed
by a second compiled function before each subset update, so the image follows
denoise's path exactly.  A check confirms that the loop's image after
CHECK_ITERATIONS iterations equals the output of denoise.

The VCD update does not change when a constant is added to the noisy image,
because it depends only on differences of neighbors and on y - x.  So adding
an offset changes only ||x||_1 in the stopping statistic, and percent_change
can be computed for each offset from one run [derived].  This holds with
sigma_y and sigma_x fixed.  The automatic sigma_x and the estimate of sigma_y
change with the offset, so the offset columns of the 'auto' and 'defaults'
cases are not those of denoise(image + offset).

Each case's record goes to results/denoise_convergence/<case>.csv, and the
case's parameters go to <case>_settings.txt beside it, in JSON.  A case whose
files exist is not run again.  The MAP estimates are cached in the system's
temporary folder.  The analysis at the end reads the files.

Run on any machine with mbirtorch installed:
    python denoise_convergence.py
"""

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

import mbirtorch
from mbirtorch import denoising, qggmrf
from mbirtorch.projectors import maybe_compile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qggmrf_map3d  # noqa: E402
from phantom3d import make_phantom  # noqa: E402

# ---- Run parameters ----------------------------------------------------------
SHAPE = (256, 256, 32)          # rows, columns, slices
SIGMA = 182.0                   # noise per voxel of the test volume, in HU
# Each case is (name, volume, noise, r).  volume is '3d' for the test volume
# or 'slice' for its middle slice.  noise is a kind of phantom3d.py.
# r = sigma_x / sigma_y sets sigma_x with sigma_noise = SIGMA.  'auto' keeps
# sigma_noise = SIGMA and lets the denoiser choose sigma_x.  'defaults' lets
# the denoiser estimate both, as a call of denoise(image) does.
CASES = [('fdk_r0.05', '3d', 'fdk', 0.05), ('fdk_r0.1', '3d', 'fdk', 0.1), ('fdk_r0.2', '3d', 'fdk', 0.2),
         ('fdk_r0.4', '3d', 'fdk', 0.4), ('fdk_auto', '3d', 'fdk', 'auto'),
         ('fdk_defaults', '3d', 'fdk', 'defaults'), ('white_r0.1', '3d', 'white', 0.1),
         ('white_auto', '3d', 'white', 'auto'), ('fdkz_r0.1', '3d', 'fdk_z', 0.1),
         ('fdkz_auto', '3d', 'fdk_z', 'auto'), ('slice_fdk_r0.1', 'slice', 'fdk', 0.1),
         ('slice_fdk_auto', 'slice', 'fdk', 'auto'), ('slice_white_r0.1', 'slice', 'white', 0.1)]
MAX_ITERATIONS = 4000           # the loop stops here at the latest
STOP_DISTANCE = 1e-4            # or when the distance falls below this, in units of sigma_y
OFFSETS = (-1000.0, 0.0, 1000.0, 10000.0)   # added to the volume for the stopping statistic
CHECK_ITERATIONS = 15
REFERENCE_BOUND = 1e-7          # the rms gradient at which L-BFGS stops, in units of sigma_y
BOUND_ITERATIONS = set(range(1, 31)) | set(range(40, MAX_ITERATIONS + 1, 10))
RESULTS = HERE / 'results' / 'denoise_convergence'
CACHE = Path(tempfile.gettempdir()) / 'improve_denoiser_cache'
# The analysis reports the iteration at which each rule stops.
DISTANCE_TARGETS = (0.3, 0.1, 0.03, 0.01, 0.003)
PERCENT_CHANGE_THRESHOLDS = (0.2, 0.05)
CHANGE_THRESHOLDS = (0.01, 0.003, 0.001)
GRADIENT_THRESHOLDS = (0.03, 0.01)
ESTIMATE_THRESHOLDS = (0.03, 0.01)
ESTIMATE_SPAN = 5               # iterations over which the contraction factor is measured


def subset_gradient_squared(flat_image, flat_error_image, pixel_indices, fm_constant, qggmrf_params,
                            image_shape):
    """The sum of the squared gradient of J over the voxels of one subset,
    computed as vcd_subset_denoiser computes the gradient's two terms."""
    prior_grad, _ = qggmrf.qggmrf_gradient_and_hessian_at_indices(
        flat_image, image_shape, pixel_indices, qggmrf_params)
    return torch.sum((prior_grad - fm_constant * flat_error_image[pixel_indices]) ** 2)


def test_volume(volume, noise):
    """Return (clean, noisy, flat) for a case: the test volume, or its middle
    slice with a slice axis of length 1."""
    clean, noisy, flat = make_phantom(SHAPE, noise=noise, sigma=SIGMA)
    if volume == 'slice':
        middle = slice(SHAPE[2] // 2, SHAPE[2] // 2 + 1)
        clean, noisy, flat = clean[:, :, middle], noisy[:, :, middle], flat[:, :, middle]
    return np.ascontiguousarray(clean), np.ascontiguousarray(noisy), np.ascontiguousarray(flat)


def settle(noisy, r):
    """Return a denoiser, its partition, and its (sigma_y, sigma_x), settled as denoise would."""
    denoiser = mbirtorch.QGGMRFDenoiser(noisy.shape)
    denoiser.configure_devices(devices=['cpu'])
    denoiser.set_params(no_warning=True, verbose=0)
    if r not in ('auto', 'defaults'):
        denoiser.set_params(no_warning=True, sigma_x=r * SIGMA, auto_regularize_flag=False)
    np.random.seed(0)     # the partition is drawn from the global state
    data = denoiser.initialize_denoiser(image=noisy, sigma_noise=None if r == 'defaults' else SIGMA)
    sigma_y = float(denoiser.get_params('sigma_y'))
    sigma_x = float(denoiser.get_params('sigma_x'))
    return denoiser, data['partition'], sigma_y, sigma_x


def map_estimate(name, noisy, sigma_y, sigma_x, p, q, T):
    """Return the MAP estimate and its solver's information, from the cache when present."""
    path = CACHE / f'{name}_{sigma_y:.6g}_{sigma_x:.6g}_{p:g}_{q:g}_{T:g}.npz'
    if path.exists():
        saved = np.load(path, allow_pickle=True)
        return saved['reference'], saved['info'].item()
    start = time.time()
    reference, info = qggmrf_map3d.solve(noisy, sigma_y, sigma_x, p, q, T, rms_bound=REFERENCE_BOUND)
    info['seconds'] = time.time() - start
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, reference=reference, info=np.array(info, dtype=object))
    return reference, info


def run_case(name, volume, noise, r):
    """Run one case and write its record."""
    clean, noisy, flat = test_volume(volume, noise)
    shape = noisy.shape
    denoiser, partition, sigma_y, sigma_x = settle(noisy, r)
    p, q, T = (float(v) for v in denoiser.get_params(['p', 'q', 'T']))
    print(f'{name}: sigma_y {sigma_y:.2f}, sigma_x {sigma_x:.3f}, r {sigma_x / sigma_y:.4f}, '
          f'q {q}, T {T}, {partition.shape[0]} subsets', flush=True)
    reference, info = map_estimate(name, noisy, sigma_y, sigma_x, p, q, T)
    print(f'  reference: {info}', flush=True)

    num_slices = shape[2]
    num_voxels = int(np.prod(shape))
    b = qggmrf.get_b_from_nbr_wts(denoiser.get_params('qggmrf_nbr_wts'))
    qggmrf_params = (b, sigma_x, p, q, T)
    noisy_t = torch.as_tensor(noisy, dtype=torch.float32)
    flat_image = noisy_t.clone().reshape(-1, num_slices).contiguous()
    flat_error = (noisy_t.reshape(-1, num_slices) - flat_image).contiguous()
    update = maybe_compile(denoising.vcd_subset_denoiser, denoiser.compile_enabled)
    gradient_squared = maybe_compile(subset_gradient_squared, denoiser.compile_enabled)
    fm_constant_t, qggmrf_params_t = denoising._as_device_scalars(1.0 / sigma_y ** 2, qggmrf_params,
                                                                  flat_image)
    reference_t = torch.as_tensor(reference.reshape(-1, num_slices))
    clean_t = torch.as_tensor(clean.reshape(-1, num_slices))
    flat_t = torch.as_tensor(flat.reshape(-1, num_slices))
    z_t = torch.as_tensor(noisy / sigma_y)
    previous = flat_image.clone()

    def rms(x):
        return float(torch.sqrt(torch.mean(x.double() ** 2))) / sigma_y

    rows = []
    checked = None
    with torch.no_grad():
        for iteration in range(1, MAX_ITERATIONS + 1):
            seconds, ell1, alpha, squares = 0.0, 0.0, 0.0, 0.0
            for k in range(partition.shape[0]):
                squares = squares + gradient_squared(flat_image, flat_error, partition[k], fm_constant_t,
                                                     qggmrf_params_t, shape)
                tick = time.time()
                flat_image, flat_error, ell1_subset, alpha_subset = update(
                    flat_image, flat_error, partition[k], fm_constant_t, qggmrf_params_t, shape)
                ell1, alpha = ell1 + ell1_subset, alpha + alpha_subset
                seconds += time.time() - tick
            ell1 = float(ell1)
            image64 = flat_image.double()
            if iteration in BOUND_ITERATIONS:
                _, gradient = qggmrf_map3d.cost_and_gradient(image64.reshape(shape) / sigma_y, z_t,
                                                             sigma_x / sigma_y, p, q, T)
                bound = float(torch.sqrt(torch.mean(gradient ** 2)))
            else:
                bound = float('nan')
            norms = [float(torch.sum(torch.abs(image64 + c))) for c in OFFSETS]
            rows.append([iteration, rms(image64 - reference_t), rms(flat_image - previous),
                         sigma_y * float(torch.sqrt(squares / num_voxels)), bound]
                        + [100.0 * ell1 / n for n in norms]
                        + [float(alpha) / partition.shape[0], rms((image64 - clean_t)[flat_t]), seconds])
            previous.copy_(flat_image)
            if iteration == CHECK_ITERATIONS:
                checked = flat_image.reshape(shape).numpy().copy()
            if rows[-1][1] < STOP_DISTANCE:
                break

    # The check: denoise with the same partition gives the loop's image.
    output, _ = denoiser.denoise(noisy, sigma_noise=sigma_y, max_iterations=CHECK_ITERATIONS,
                                 stop_threshold_change_pct=0.0, logfile_path=None,
                                 print_logs=False, do_initialization=False)
    check = float(np.abs(np.asarray(output) - checked).max()) if checked is not None else None
    print(f'  {len(rows)} iterations, distance {rows[-1][1]:.2e}; largest difference from denoise '
          f'after {CHECK_ITERATIONS} iterations: {check}', flush=True)

    columns = (['iteration', 'distance', 'change', 'gradient_statistic', 'exact_gradient']
               + [f'percent_change_offset_{c:g}' for c in OFFSETS] + ['alpha', 'flat_noise', 'seconds'])
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savetxt(RESULTS / f'{name}.csv', np.array(rows), delimiter=',', header=','.join(columns),
               comments='', fmt='%.6e')
    settings = {'name': name, 'volume': volume, 'noise': noise, 'r_setting': r, 'shape': shape,
                'sigma': SIGMA,
                'sigma_y': sigma_y, 'sigma_x': sigma_x, 'r': sigma_x / sigma_y, 'p': p, 'q': q, 'T': T,
                'subsets': int(partition.shape[0]), 'reference': info,
                'noisy_flat_noise': float(np.sqrt(np.mean((noisy - clean)[flat] ** 2)) / sigma_y),
                'map_flat_noise': float(np.sqrt(np.mean((reference - clean)[flat] ** 2)) / sigma_y),
                'map_distance_from_noisy': float(np.sqrt(np.mean((reference - noisy) ** 2)) / sigma_y),
                'check_iterations': CHECK_ITERATIONS, 'check_max_abs_difference': check,
                'mbirtorch_version': mbirtorch.__version__, 'torch_threads': torch.get_num_threads()}
    (RESULTS / f'{name}_settings.txt').write_text(json.dumps(settings, indent=1))


def first(mask):
    """The index of the first True entry, or None."""
    hits = np.flatnonzero(mask)
    return int(hits[0]) if hits.size else None


def analyze():
    """Print, for each case, the iterations needed to reach each distance, and
    where each stopping rule stops."""
    for name, _, _, _ in CASES:
        path = RESULTS / f'{name}.csv'
        if not path.exists():
            continue
        names = path.read_text().splitlines()[0].split(',')
        values = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
        data = {n: values[:, i] for i, n in enumerate(names)}
        settings = json.loads((RESULTS / f'{name}_settings.txt').read_text())
        iteration, distance, change = data['iteration'], data['distance'], data['change']
        print(f'\n{name}: r = {settings["r"]:.3f}, sigma_y = {settings["sigma_y"]:.1f}; '
              f'rms error over the flat region {settings["noisy_flat_noise"]:.3f} in y and '
              f'{settings["map_flat_noise"]:.3f} in the MAP estimate; distance from y to the MAP '
              f'estimate {settings["map_distance_from_noisy"]:.3f}; all in units of sigma_y')
        at15 = CHECK_ITERATIONS - 1
        print(f'  after {CHECK_ITERATIONS} iterations: distance {distance[at15]:.4f}, flat-region error '
              f'{data["flat_noise"][at15]:.4f}; median seconds per iteration '
              f'{np.median(data["seconds"]):.3f}')
        cells = []
        for target in DISTANCE_TARGETS:
            k = first(distance < target)
            cells.append(f'{target:g}: {"-" if k is None else int(iteration[k])}')
        print('  iterations to reach a distance of ' + ', '.join(cells))

        def report(label, k):
            if k is None:
                print(f'  {label}: does not stop')
            else:
                print(f'  {label}: stops at {int(iteration[k])}, distance {distance[k]:.4f}')
        for threshold in PERCENT_CHANGE_THRESHOLDS:
            for c in OFFSETS:
                report(f'percent change < {threshold} at offset {c:g}',
                       first(data[f'percent_change_offset_{c:g}'] < threshold))
        for threshold in CHANGE_THRESHOLDS:
            report(f'change < {threshold}', first(change < threshold))
        for threshold in GRADIENT_THRESHOLDS:
            report(f'gradient statistic < {threshold}', first(data['gradient_statistic'] < threshold))
        # The contraction factor f over the last ESTIMATE_SPAN iterations
        # estimates the remaining distance as change f / (1 - f).
        span = ESTIMATE_SPAN
        factor = np.full_like(change, np.nan)
        factor[span:] = (change[span:] / change[:-span]) ** (1.0 / span)
        estimate = np.where(factor < 1, change * factor / (1 - factor), np.inf)
        for threshold in ESTIMATE_THRESHOLDS:
            report(f'estimated distance < {threshold}', first(estimate < threshold))
        # How well each statistic tracks the distance, from iteration 10 on.
        late = iteration >= 10
        for label, statistic in (('change', change), ('estimated distance', estimate),
                                 ('gradient statistic', data['gradient_statistic']),
                                 ('exact gradient', data['exact_gradient'])):
            valid = late & np.isfinite(statistic)
            ratio = statistic[valid] / distance[valid]
            print(f'  {label} / distance from iteration 10: median {np.median(ratio):.3f}, '
                  f'range {ratio.min():.3f} to {ratio.max():.3f}')


def main():
    torch.manual_seed(0)
    for name, volume, noise, r in CASES:
        if not (RESULTS / f'{name}.csv').exists():
            run_case(name, volume, noise, r)
    analyze()


if __name__ == '__main__':
    main()
