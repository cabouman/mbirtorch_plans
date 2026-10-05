"""Step 6 of improve_qggmrf_denoiser.md: does the tomography update share the
denoiser's scaling problem, and does the clamp on its step size alpha bind?

Each VCD subset update of recon and prox_map moves along a direction and
scales it by a step size alpha from a line search.  alpha is clamped to the
interval [eps, max_alpha].  max_alpha defaults to 1.5, and the translation
model sets it to 1.3.

Test 'scale': for each scale c, recon(c y) with sigma_y = c sigma_y0 and
sigma_x = c sigma_x0 must equal c recon(y).  The test runs with these values
pinned, and with the automatic values, in parallel and cone beam.  The
scales are powers of 2, so the scaling adds no rounding.  The models compile
their kernels, as they do by default.

Test 'alpha': max_alpha is set to 1e30, which removes the clamp, and the
alpha of every subset is recorded.  If no alpha of a run exceeds the
model's default max_alpha, the default clamp would not have changed that
run.  The cases are:
- recon and the proximal map, in parallel and cone beam, at sharpness -1, 1
  (the default) and 3, on a noiseless sinogram and on the same sinogram with
  noise;
- recon and the proximal map started from a converged image, so that the
  coarse partitions at the start of the partition sequence update an image
  whose remaining error is small, as in a warm-started call;
- recon with every pixel in one subset at every iteration, which couples the
  updated voxels most;
- recon and the proximal map on a noisy sinogram in the translation
  geometry, whose recon has 4 rows, and in the multiaxis geometry with an
  elevation of 30 degrees, at the three sharpness values.
The recording replaces torch.clamp in this process, so these models run with
compile_mode='off'.  For each case whose largest alpha exceeds the model's
default max_alpha, the case runs for 5, 10, 20 and 40 iterations with the
default max_alpha, with 1.5 and with no clamp, and the results are compared
by their cost.

Test 'epsilon': the line search adds a small constant to its denominator.
One proximal map runs with the default constant and with 1e-30 in its place,
to show how the constant lowers alpha once the updates are small.

Test 'damping': the cone-beam direction damps the slice mean of the gradient
over each subset.  The cone-beam proximal map at sharpness -1 runs with the
clamp removed, with the default damping and with the damping turned off, to
show whether the damping causes the values of alpha above 1.5.

Test 'prefix': runs of 5, 20 and 40 iterations of recon at sharpness 1, in
the translation geometry and in cone beam, are compared by their alpha and
data misfit in each shared iteration.  Equal values show that a shorter run
repeats the first iterations of a longer one, as the comparison of costs in
the test 'alpha' assumes.

Test 'overlap': the cross terms of the line search come from voxels that
share rays.  For 12 voxels near the center of each recon, the test projects
the voxel and each of its neighbors along the three axes, and reports the
inner product of the two projections divided by the product of their norms.

Test 'offset': the cone-beam proximal map at sharpness -1 starts from a
recon result, at the 128-subset partitions, with an input equal to that
result plus a constant.  The gradient of each slice of a subset is then
mostly its mean, which is the case in which the line search undoes the
damping.  The test reports the mean alpha of each iteration with the clamp,
and the alpha of the subsets with the clamp removed.

Usage: python alpha_clamp.py
[scale | alpha | epsilon | damping | prefix | overlap | offset]
(no argument runs all seven)
Results: results/alpha_clamp/scale.csv, alpha.csv, alpha_iterations.csv,
clamp_effect.csv, epsilon.csv, damping.csv, prefix.csv, overlap.csv, and
offset.csv.
"""
import csv
import os
import sys
import time

import numpy as np
import torch

import mbirtorch
from mbirtorch import qggmrf as _qggmrf
from mbirtorch import vcd_utils

# A small problem, so that every case runs in under a minute on a CPU.
CELL = dict(num_views=64, num_det_rows=16, num_det_channels=64)
GEOMETRIES = ('parallel', 'cone')
OTHER_GEOMETRIES = ('translation', 'multiaxis')
# The translation geometry of tests/test_translation.py, with a 7 by 7 grid
# of translations in place of 4 by 4, a spacing of 3 along z in place of 2,
# and a 64 by 64 detector in place of 40 by 32.  The recon then has 4 rows.
# With the 16 by 64 detector of CELL, generate_demo_data gives a translation
# recon only 1 row thick.
TRANSLATION = dict(num_x_translations=7, num_z_translations=7, spacing=3.0,
                   num_det_rows=64, num_det_channels=64,
                   source_detector_dist=128.0, source_iso_dist=32.0)
MULTIAXIS_ELEVATION_DEGREES = 30.0
# The noise is white and Gaussian, 30 dB below the rms of the sinogram.  The
# automatic sigma_y assumes this signal-to-noise ratio.
SNR_DB = 30.0
SCALES = (2.0 ** -10, 1.0, 2.0 ** 10)
SCALE_ITERATIONS = 10
ALPHA_ITERATIONS = 40
WARM_ITERATIONS = 10
EFFECT_ITERATIONS = (5, 10, 20, 40)
REFERENCE_ITERATIONS = 160
SHARPNESS = (-1.0, 1.0, 3.0)
NO_CLAMP = 1e30
DEFAULT_CLAMP = 1.5
OVERLAP_VOXELS = 12
# The offset test starts at iteration 3, the first with 128 subsets.
OFFSETS = (0.01, 0.03, 0.1)
OFFSET_FIRST_ITERATION = 3
OFFSET_ITERATIONS = 5
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'alpha_clamp')

# The line search calls torch.clamp(alpha, eps, max_alpha) once per subset.
# The replacement records alpha whenever max_alpha is NO_CLAMP.
_real_clamp = torch.clamp
_recorded = []


def _recording_clamp(input, min=None, max=None, **kwargs):
    if isinstance(max, float) and max == NO_CLAMP:
        _recorded.append(input.detach().clone())
    return _real_clamp(input, min, max, **kwargs)


def add_noise(sinogram):
    sigma = 10 ** (-SNR_DB / 20) * float(np.sqrt(np.mean(sinogram ** 2)))
    noise = np.random.RandomState(1).randn(*sinogram.shape).astype(np.float32)
    return sinogram + sigma * noise


def make_data(geometry, noisy):
    """Return the sinogram of the geometry, with or without noise, and the
    parameters of the geometry.  The phantom is Shepp-Logan, except in the
    translation geometry, which uses a pattern of dots."""
    if geometry == 'translation':
        translation_vectors = mbirtorch.gen_translation_vectors(
            TRANSLATION['num_x_translations'], TRANSLATION['num_z_translations'],
            x_spacing=TRANSLATION['spacing'], z_spacing=TRANSLATION['spacing'])
        params = dict(translation_vectors=translation_vectors)
        sinogram_shape = (translation_vectors.shape[0], TRANSLATION['num_det_rows'],
                          TRANSLATION['num_det_channels'])
        model = make_model(geometry, sinogram_shape, params, 'off')
        phantom = mbirtorch.gen_translation_phantom(model.get_params('recon_shape'),
                                                    'dots', None, fill_rate=0.05)
        sinogram = model.forward_project(phantom)
    else:
        extra = (dict(elevation_degrees=MULTIAXIS_ELEVATION_DEGREES)
                 if geometry == 'multiaxis' else {})
        _, sinogram, params = mbirtorch.generate_demo_data(
            model_type=geometry, object_type='shepp-logan', **CELL, **extra)
    sinogram = np.asarray(sinogram, dtype=np.float32)
    if noisy:
        sinogram = add_noise(sinogram)
    return sinogram, params


def make_model(geometry, sinogram_shape, params, compile_mode):
    """Return a model of the geometry on the CPU."""
    if geometry == 'parallel':
        model = mbirtorch.ParallelBeamModel(sinogram_shape, params['angles'],
                                            compile_mode=compile_mode)
    elif geometry == 'cone':
        model = mbirtorch.ConeBeamModel(sinogram_shape, params['angles'],
                                        source_detector_dist=params['source_detector_dist'],
                                        source_iso_dist=params['source_iso_dist'],
                                        compile_mode=compile_mode)
    elif geometry == 'translation':
        model = mbirtorch.TranslationModel(
            sinogram_shape, params['translation_vectors'],
            source_detector_dist=TRANSLATION['source_detector_dist'],
            source_iso_dist=TRANSLATION['source_iso_dist'], compile_mode=compile_mode)
    else:
        model = mbirtorch.MultiAxisParallelModel(sinogram_shape, params['angles'],
                                                 compile_mode=compile_mode)
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0)
    return model


def run(model, sinogram, num_iterations, prox_input=None, init=None):
    """Run recon, or prox_map when prox_input is given, for a fixed number of
    iterations from a fixed seed.  init None starts from the direct
    reconstruction, as recon does by default.  Returns the result and its
    dictionary."""
    np.random.seed(0)
    init = None if init is None else init.astype(np.float32)
    if prox_input is None:
        out, out_dict = model.recon(sinogram, init_recon=init,
                                    max_iterations=num_iterations,
                                    stop_threshold_change_pct=0.0,
                                    logfile_path=None, print_logs=False)
    else:
        out, out_dict = model.prox_map(prox_input.astype(np.float32), sinogram,
                                       init_recon=init, max_iterations=num_iterations,
                                       stop_threshold_change_pct=0.0,
                                       logfile_path=None, print_logs=False)
    return np.asarray(out, dtype=np.float64), out_dict


def write_csv(name, rows):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {path}')


def scale_test():
    """Compare c recon(y) with recon(c y) at each scale c.

    Each scaled run is compared with the run at c = 1 that has the same
    prior setting.  The first run of each geometry, which compiles the
    kernels, is kept as a separate reference, because a run that compiles
    can differ from later runs in the last bits."""
    rows = []
    for geometry in GEOMETRIES:
        sinogram, params = make_data(geometry, noisy=True)
        # The automatic values at c = 1 are the pinned values at c = 1.
        model = make_model(geometry, sinogram.shape, params, 'auto')
        first_run, _ = run(model, sinogram, SCALE_ITERATIONS)
        sigma_y0, sigma_x0 = model.get_params(['sigma_y', 'sigma_x'])
        largest = float(np.max(np.abs(first_run)))
        for prior in ('pinned', 'automatic'):
            unscaled = None
            # The run at c = 1 comes first, so that the other scales can be
            # compared with it.
            for c in sorted(SCALES, key=lambda s: s != 1.0):
                model = make_model(geometry, sinogram.shape, params, 'auto')
                if prior == 'pinned':
                    model.set_params(no_warning=True, auto_regularize_flag=False,
                                     sigma_y=c * sigma_y0, sigma_x=c * sigma_x0)
                out, out_dict = run(model, c * sinogram, SCALE_ITERATIONS)
                if unscaled is None:
                    unscaled = out
                sigma_y, sigma_x = model.get_params(['sigma_y', 'sigma_x'])
                row = dict(geometry=geometry, prior=prior, scale=c,
                           sigma_y_over_c_sigma_y0=sigma_y / (c * sigma_y0),
                           sigma_x_over_c_sigma_x0=sigma_x / (c * sigma_x0),
                           max_difference_to_c1_over_max=float(
                               np.max(np.abs(out / c - unscaled))) / largest,
                           max_difference_to_first_run_over_max=float(
                               np.max(np.abs(out / c - first_run))) / largest,
                           last_mean_alpha=float(out_dict['recon_params']['alpha_values'][-1]))
                rows.append(row)
                print(row)
    write_csv('scale.csv', rows)


def subsets_per_iteration(model, num_iterations):
    """Return the number of subsets of each iteration of the partition
    sequence.  A partition has at most one subset per pixel."""
    num_pixels = len(vcd_utils.gen_full_indices(model.get_params('recon_shape'),
                                                use_ror_mask=model.get_params('use_ror_mask')))
    granularity = model.get_params('granularity')
    sequence = vcd_utils.gen_partition_sequence(model.get_params('partition_sequence'),
                                                max_iterations=num_iterations)
    return [min(int(granularity[k]), num_pixels) for k in sequence]


def cost(model, sinogram, image, prox_input):
    """Return the cost that recon or prox_map minimizes, with weights of 1:
    the data term plus the qGGMRF prior or the proximal term."""
    sigma_y = model.get_params('sigma_y')
    error = sinogram - np.asarray(model.forward_project(image.astype(np.float32)),
                                  dtype=np.float64)
    data_term = float(np.sum(error ** 2)) / (2 * sigma_y ** 2)
    if prox_input is None:
        qggmrf_nbr_wts, sigma_x, p, q, T = model.get_params(
            ['qggmrf_nbr_wts', 'sigma_x', 'p', 'q', 'T'])
        b = _qggmrf.get_b_from_nbr_wts(qggmrf_nbr_wts)
        prior_term = float(_qggmrf.qggmrf_loss(image.astype(np.float32),
                                               (b, sigma_x, p, q, T)))
    else:
        sigma_prox = model.get_params('sigma_prox')
        prior_term = float(np.sum((image - prox_input) ** 2)) / (2 * sigma_prox ** 2)
    return data_term + prior_term


def configured_model(case, sinogram, params, max_alpha=None):
    """Return an uncompiled model set up for the case, and the model's own
    default max_alpha.  max_alpha None keeps the default."""
    model = make_model(case['geometry'], sinogram.shape, params, 'off')
    default_clamp = model.get_params('max_alpha')
    model.set_params(no_warning=True, sharpness=case['sharpness'])
    if max_alpha is not None:
        model.set_params(no_warning=True, max_alpha=max_alpha)
    if case['partition'] == 'one subset':
        model.set_params(no_warning=True, granularity=[1], partition_sequence=[0])
    return model, default_clamp


def alpha_case(case, sinogram, params, prox_input, init):
    """Run one case with the clamp removed and summarize the alpha of every
    subset.  Returns (summary row, per-iteration rows, result, model)."""
    model, default_clamp = configured_model(case, sinogram, params, NO_CLAMP)
    _recorded.clear()
    start = time.perf_counter()
    out, out_dict = run(model, sinogram, case['iterations'], prox_input, init)
    seconds = time.perf_counter() - start
    alphas = torch.stack(_recorded).double().cpu().numpy().ravel()
    counts = subsets_per_iteration(model, case['iterations'])
    if sum(counts) != alphas.size:
        raise RuntimeError(f'{alphas.size} alphas recorded for {sum(counts)} subsets')
    per_iteration = np.split(alphas, np.cumsum(counts)[:-1])
    # The mean of each iteration must match the mean the recon reports.
    reported = np.asarray(out_dict['recon_params']['alpha_values'], dtype=np.float64)
    mean_check = float(np.max(np.abs(np.array([a.mean() for a in per_iteration]) - reported)))
    largest_by_iteration = np.array([a.max() for a in per_iteration])
    worst = int(np.argmax(largest_by_iteration))
    summary = dict(case['row'],
                   subsets=alphas.size, default_clamp=default_clamp,
                   max_alpha=float(alphas.max()),
                   iteration_of_max=worst + 1, subsets_in_that_iteration=counts[worst],
                   alpha_p99=float(np.percentile(alphas, 99)),
                   fraction_above_1=float(np.mean(alphas > 1.0)),
                   fraction_above_default_clamp=float(np.mean(alphas > default_clamp)),
                   fraction_above_1p5=float(np.mean(alphas > DEFAULT_CLAMP)),
                   largest_fraction_above_default_clamp_in_an_iteration=float(
                       max(np.mean(a > default_clamp) for a in per_iteration)),
                   min_alpha=float(alphas.min()),
                   last_iteration_mean=float(per_iteration[-1].mean()),
                   sigma_y=model.get_params('sigma_y'),
                   sigma_x=model.get_params('sigma_x'),
                   sigma_prox=(model.get_params('sigma_prox')
                               if case['operation'] == 'prox' else ''),
                   cost=cost(model, sinogram, out, prox_input),
                   mean_check=mean_check, seconds=round(seconds, 1))
    iteration_rows = [dict(case=case['row']['case'], iteration=i + 1, subsets=counts[i],
                           mean_alpha=float(a.mean()), max_alpha=float(a.max()),
                           min_alpha=float(a.min()),
                           fraction_above_default_clamp=float(np.mean(a > default_clamp)))
                      for i, a in enumerate(per_iteration)]
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()})
    return summary, iteration_rows, out


def clamp_effect(case, sinogram, params, prox_input, init, default_clamp, alpha_run_cost):
    """Compare the cost after k iterations, for each k in EFFECT_ITERATIONS,
    with the model's default max_alpha, with 1.5, and with no clamp.

    The reference cost is the cost after REFERENCE_ITERATIONS with the
    default max_alpha, and each row reports how far a run's cost is above
    it.  A run of k iterations repeats the first k iterations of a longer
    run, because the seed and the partition sequence are the same; the test
    'prefix' checks this.  The run with no clamp at ALPHA_ITERATIONS must
    reproduce alpha_run_cost."""
    model, _ = configured_model(case, sinogram, params)
    out, _ = run(model, sinogram, REFERENCE_ITERATIONS, prox_input, init)
    reference_cost = cost(model, sinogram, out, prox_input)
    rows = []
    for max_alpha in sorted({default_clamp, DEFAULT_CLAMP}) + [NO_CLAMP]:
        for k in EFFECT_ITERATIONS:
            model, _ = configured_model(case, sinogram, params, max_alpha)
            out, _ = run(model, sinogram, k, prox_input, init)
            value = cost(model, sinogram, out, prox_input)
            rows.append(dict(case=case['row']['case'],
                             max_alpha='none' if max_alpha == NO_CLAMP else max_alpha,
                             iterations=k, cost=value,
                             excess_cost=value - reference_cost,
                             reference_cost=reference_cost,
                             reference_iterations=REFERENCE_ITERATIONS,
                             alpha_run_cost=alpha_run_cost))
            print(rows[-1])
    return rows


def make_cases():
    """Return the cases of the alpha test, in an order in which the recon at
    the default sharpness comes before the cases that use its result."""
    cases = []

    def add(geometry, noisy, operation, sharpness, start, partition, iterations):
        name = (f"{geometry} {operation}, sharpness {sharpness:g}, "
                f"{'noisy' if noisy else 'noiseless'}, {start} start, {partition} partitions")
        cases.append(dict(geometry=geometry, noisy=noisy, operation=operation,
                          sharpness=sharpness, start=start, partition=partition,
                          iterations=iterations,
                          row=dict(case=name, geometry=geometry, operation=operation,
                                   sharpness=sharpness, noisy=noisy, start=start,
                                   partition=partition, iterations=iterations)))

    for geometry in GEOMETRIES:
        for noisy in (False, True):
            for operation in ('recon', 'prox'):
                for sharpness in SHARPNESS:
                    add(geometry, noisy, operation, sharpness, 'direct', 'default',
                        ALPHA_ITERATIONS)
        for operation in ('recon', 'prox'):
            add(geometry, True, operation, 1.0, 'converged', 'default', WARM_ITERATIONS)
        add(geometry, True, 'recon', 3.0, 'direct', 'one subset', ALPHA_ITERATIONS)
    for geometry in OTHER_GEOMETRIES:
        for operation in ('recon', 'prox'):
            for sharpness in SHARPNESS:
                add(geometry, True, operation, sharpness, 'direct', 'default',
                    ALPHA_ITERATIONS)
    return cases


def alpha_test():
    """Record alpha with the clamp removed, case by case."""
    summaries, iteration_rows, effects = [], [], []
    data = {}
    # The recon at the default sharpness, from the direct start, is the input
    # of the proximal maps and the start of the converged-start cases.
    default_recons = {}
    torch.clamp = _recording_clamp
    try:
        for case in make_cases():
            key = (case['geometry'], case['noisy'])
            if key not in data:
                data[key] = make_data(*key)
            sinogram, params = data[key]
            prox_input = default_recons[key] if case['operation'] == 'prox' else None
            init = default_recons[key] if case['start'] == 'converged' else None
            summary, rows, out = alpha_case(case, sinogram, params, prox_input, init)
            summaries.append(summary)
            iteration_rows.extend(rows)
            if (case['operation'] == 'recon' and case['sharpness'] == 1.0
                    and case['start'] == 'direct' and case['partition'] == 'default'):
                default_recons[key] = out
            if summary['max_alpha'] > summary['default_clamp']:
                effects.extend(clamp_effect(case, sinogram, params, prox_input, init,
                                            summary['default_clamp'], summary['cost']))
    finally:
        torch.clamp = _real_clamp
    write_csv('alpha.csv', summaries)
    write_csv('alpha_iterations.csv', iteration_rows)
    if effects:
        write_csv('clamp_effect.csv', effects)


def epsilon_test():
    """Run one proximal map with the default constant in the line search
    denominator and with 1e-30 in its place.  The constant is the module
    global _F32_EPS of tomography_model, which the subset updater reads at
    each call."""
    from mbirtorch import tomography_model
    sinogram, params = make_data('parallel', noisy=True)
    model = make_model('parallel', sinogram.shape, params, 'off')
    prox_input, _ = run(model, sinogram, ALPHA_ITERATIONS)
    default_eps = tomography_model._F32_EPS
    rows = []
    try:
        for eps in (default_eps, 1e-30):
            tomography_model._F32_EPS = eps
            model = make_model('parallel', sinogram.shape, params, 'off')
            out, out_dict = run(model, sinogram, ALPHA_ITERATIONS, prox_input=prox_input)
            alphas = out_dict['recon_params']['alpha_values']
            change = out_dict['recon_params']['stop_threshold_change_pct']
            for i in (9, 19, 29, 39):
                rows.append(dict(case='parallel prox, sharpness 1, noisy', eps=eps,
                                 iteration=i + 1, mean_alpha=float(alphas[i]),
                                 percent_change=float(change[i]),
                                 final_cost=cost(model, sinogram, out, prox_input)))
                print(rows[-1])
    finally:
        tomography_model._F32_EPS = default_eps
    write_csv('epsilon.csv', rows)


def damping_test():
    """Run the cone-beam proximal map at sharpness -1 on the noisy sinogram,
    with the clamp removed, with the default damping of the slice mean and
    with the damping turned off.  Setting _dc_damping to None on the model
    turns the damping off."""
    sinogram, params = make_data('cone', noisy=True)
    model = make_model('cone', sinogram.shape, params, 'off')
    prox_input, _ = run(model, sinogram, ALPHA_ITERATIONS)
    case = dict(geometry='cone', sharpness=-1.0, partition='default')
    rows = []
    torch.clamp = _recording_clamp
    try:
        for damping in ('default', 'off'):
            model, _ = configured_model(case, sinogram, params, NO_CLAMP)
            if damping == 'off':
                model._dc_damping = None
            _recorded.clear()
            out, _ = run(model, sinogram, ALPHA_ITERATIONS, prox_input=prox_input)
            alphas = torch.stack(_recorded).double().cpu().numpy().ravel()
            # The damping factors exist once the run has built the projectors.
            s_range = ''
            if damping == 'default':
                profiles, _ = model._dc_damping_slice_profile()
                s = torch.cat(profiles).double().cpu().numpy()
                s_range = f'{s.min():.3f} to {s.max():.3f}'
            rows.append(dict(case='cone prox, sharpness -1, noisy', damping=damping,
                             damping_factor_range=s_range, subsets=alphas.size,
                             max_alpha=float(alphas.max()),
                             alpha_p99=float(np.percentile(alphas, 99)),
                             fraction_above_1p5=float(np.mean(alphas > DEFAULT_CLAMP)),
                             cost=cost(model, sinogram, out, prox_input)))
            print(rows[-1])
    finally:
        torch.clamp = _real_clamp
    write_csv('damping.csv', rows)


def prefix_test():
    """Compare runs of 5, 20 and 40 iterations of recon at sharpness 1 by
    their alpha and data misfit in each shared iteration."""
    rows = []
    case_lengths = (5, 20, ALPHA_ITERATIONS)
    for geometry in ('translation', 'cone'):
        sinogram, params = make_data(geometry, noisy=True)
        case = dict(geometry=geometry, sharpness=1.0, partition='default')
        history = {}
        for k in case_lengths:
            model, _ = configured_model(case, sinogram, params)
            _, out_dict = run(model, sinogram, k)
            history[k] = (np.asarray(out_dict['recon_params']['alpha_values']),
                          np.asarray(out_dict['recon_params']['fm_rmse']))
        longest = history[ALPHA_ITERATIONS]
        for k in case_lengths[:-1]:
            rows.append(dict(case=f'{geometry} recon, sharpness 1, noisy',
                             iterations=k, compared_with=ALPHA_ITERATIONS,
                             same_alpha=bool(np.array_equal(history[k][0], longest[0][:k])),
                             same_fm_rmse=bool(np.array_equal(history[k][1], longest[1][:k]))))
            print(rows[-1])
    write_csv('prefix.csv', rows)


def voxel_projection(model, row, col, slice_index):
    """Return the projection of one voxel of value 1, as a flat array."""
    num_cols, num_slices = model.get_params('recon_shape')[1:]
    pixel_indices = torch.tensor([row * num_cols + col], dtype=torch.int64)
    voxels = torch.zeros((1, num_slices), dtype=torch.float32)
    voxels[0, slice_index] = 1.0
    projection = model.sparse_forward_project(voxels, pixel_indices)
    return np.asarray(projection, dtype=np.float64).ravel()


def overlap_test():
    """Report how strongly neighboring voxels share rays: the inner product of
    their projections divided by the product of the norms.  The weights are
    constant, so this ratio is the normalized entry of the Hessian of the data
    term."""
    neighbors = {'adjacent rows': (1, 0, 0), 'adjacent columns': (0, 1, 0),
                 'adjacent slices': (0, 0, 1)}
    rows = []
    for geometry in ('parallel', 'cone', 'translation'):
        sinogram, params = make_data(geometry, noisy=False)
        model = make_model(geometry, sinogram.shape, params, 'off')
        num_rows, num_cols, num_slices = model.get_params('recon_shape')
        rng = np.random.default_rng(0)
        values = {name: [] for name in neighbors}
        # The voxels lie in the middle half of each axis, so that every
        # neighbor exists and projects onto the detector.
        for _ in range(OVERLAP_VOXELS):
            row = int(rng.integers(num_rows // 4, max(num_rows // 4 + 1, 3 * num_rows // 4 - 1)))
            col = int(rng.integers(num_cols // 4, 3 * num_cols // 4 - 1))
            slice_index = int(rng.integers(num_slices // 4, 3 * num_slices // 4 - 1))
            projection = voxel_projection(model, row, col, slice_index)
            for name, (d_row, d_col, d_slice) in neighbors.items():
                other = voxel_projection(model, row + d_row, col + d_col, slice_index + d_slice)
                values[name].append(float(projection @ other) / float(
                    np.sqrt((projection @ projection) * (other @ other))))
        for name, v in values.items():
            rows.append(dict(geometry=geometry, recon_shape=str(model.get_params('recon_shape')),
                             neighbor=name, voxels=len(v), min_overlap=min(v),
                             median_overlap=float(np.median(v)), max_overlap=max(v)))
            print(rows[-1])
    write_csv('overlap.csv', rows)


def offset_prox_map(model, sinogram, start, offset):
    """Run the proximal map of the offset test and return its dictionary."""
    np.random.seed(0)
    _, out_dict = model.prox_map(start + offset, sinogram, init_recon=start,
                                 max_iterations=OFFSET_FIRST_ITERATION + OFFSET_ITERATIONS,
                                 first_iteration=OFFSET_FIRST_ITERATION,
                                 stop_threshold_change_pct=0.0,
                                 logfile_path=None, print_logs=False)
    return out_dict


def offset_test():
    """Run the cone-beam proximal map at sharpness -1 whose input is its start
    image plus a constant, with the clamp and with the clamp removed."""
    sinogram, params = make_data('cone', noisy=True)
    model, _ = configured_model(dict(geometry='cone', sharpness=1.0, partition='default'),
                                sinogram, params)
    start, _ = run(model, sinogram, ALPHA_ITERATIONS)
    start = start.astype(np.float32)
    case = dict(geometry='cone', sharpness=-1.0, partition='default')
    rows = []
    for offset in OFFSETS:
        model, default_clamp = configured_model(case, sinogram, params)
        means = np.asarray(offset_prox_map(model, sinogram, start, offset)
                           ['recon_params']['alpha_values'], dtype=np.float64)
        model, _ = configured_model(case, sinogram, params, NO_CLAMP)
        _recorded.clear()
        torch.clamp = _recording_clamp
        try:
            offset_prox_map(model, sinogram, start, offset)
        finally:
            torch.clamp = _real_clamp
        alphas = torch.stack(_recorded).double().cpu().numpy().ravel()
        for i, (mean, a) in enumerate(zip(means, np.split(alphas, len(means)))):
            rows.append(dict(case='cone prox, sharpness -1, noisy, input = start + offset',
                             offset=offset, iteration=OFFSET_FIRST_ITERATION + i + 1,
                             subsets=a.size, mean_alpha_over_max_alpha=mean / default_clamp,
                             largest_alpha_without_clamp=float(a.max()),
                             fraction_above_1p5_without_clamp=float(np.mean(a > DEFAULT_CLAMP))))
            print(rows[-1])
    write_csv('offset.csv', rows)


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    torch.set_num_threads(4)
    if which in ('scale', 'all'):
        scale_test()
    if which in ('alpha', 'all'):
        alpha_test()
    if which in ('epsilon', 'all'):
        epsilon_test()
    if which in ('damping', 'all'):
        damping_test()
    if which in ('prefix', 'all'):
        prefix_test()
    if which in ('overlap', 'all'):
        overlap_test()
    if which in ('offset', 'all'):
        offset_test()
