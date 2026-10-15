"""Where three stopping rules stop the qGGMRF denoiser on prerelease.

The script compares three stopping rules on the code of the mbirtorch branch
prerelease.  That branch has Charlie's changes of 2026-10-11 and 2026-10-12:
the exact prior term in the line search, the data-term Hessian fix, and
max_alpha in the denoiser.  The three rules are:
  percent change          the rule of denoise today: stop when
                          100 ||change||_1 / ||x||_1 falls below a threshold
  percent change, DC      Charlie's alternative: the same rule with the mean of
                          x removed from the denominator,
                          100 ||change||_1 / ||x - mean(x)||_1
  gradient statistic      the rule of greg_dev: stop when sigma_y times the rms
                          gradient of the cost falls below a threshold.  Each
                          voxel's gradient is taken just before the voxel's
                          update in the sweep, as greg_dev computes it.
Here "change" is the change of the image in one iteration.

Each case denoises the test volume of phantom3d.py with white noise.  The
cases differ from a base case in one setting each:
  r         the ratio sigma_x / sigma_y, which sets the strength of the prior.
            The base case has r = 0.2, close to the automatic value for this
            volume with white noise.  A smaller r gives a stronger prior.
  sigma     the noise level.  The base case has 182 HU per voxel.  A case
            with 20 HU keeps r = 0.2, so sigma_x scales with the noise level.
  offset    a constant added to the image.  The VCD update does not change
            when a constant is added to the noisy image and the starting
            image, because the update depends only on differences of
            neighbors and on y - x.  So the percent change for each offset is
            computed from the base run, by adding the offset to ||x||_1.
            check_offsets confirms with denoise, on a smaller volume, that an
            offset does not change the update.
  warm      the start of the call.  A cold start begins at the noisy image.
            A warm start begins at the MAP estimate for a previous input, as
            a MACE loop does when a denoiser call starts from its previous
            output.  The new input is the previous input plus a smooth
            perturbation.

The loop is the single-device loop of QGGMRFDenoiser.denoise, with the same
pixel partition and the same compiled subset update.  A second compiled
function computes the gradient statistic before each subset update, so the
image follows the path of denoise exactly.  A check confirms that the loop's
image after CHECK_ITERATIONS iterations equals the output of denoise, and
that the loop's percent change equals the one that denoise reports.

The distance is the rms difference between the image and the MAP estimate, in
units of sigma_y.  The MAP estimate comes from qggmrf_map3d.solve, which
reports a bound on its own error.  Each case's record goes to
results/stopping_example/<case>.csv, with its settings in
<case>_settings.txt.  A case whose files exist is not run again.  The MAP
estimates are cached in the system's temporary folder.  The analysis at the
end reads the files and prints the tables of stopping_example.md.  It also
applies the rules to the real-data replays of mace4d_stopping.md, and it draws
results/stopping_example/stopping_example.png.  Last, it draws the middle slice
of each case in IMAGE_CASES where each rule stops, in
results/stopping_example/stopping_images_<case>.png.

Run from any folder except one that holds a folder named mbirtorch, with the
prerelease branch of mbirtorch installed:
    python stopping_example.py
"""

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

import mbirtorch
from mbirtorch import denoising
from mbirtorch import qggmrf
from mbirtorch.projectors import maybe_compile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qggmrf_map3d  # noqa: E402
from phantom3d import make_phantom  # noqa: E402

# ---- Run parameters ----------------------------------------------------------
SHAPE = (256, 256, 32)          # rows, columns, slices, as in denoise_convergence.py
NOISE = 'white'
# Each case is (name, sigma, r, warm).  sigma is the noise per voxel, in HU.
# warm is True for a warm start.
CASES = [('r0.4', 182.0, 0.4, False), ('r0.2', 182.0, 0.2, False), ('r0.1', 182.0, 0.1, False),
         ('r0.05', 182.0, 0.05, False), ('sigma20_r0.2', 20.0, 0.2, False),
         ('warm_r0.2', 182.0, 0.2, True), ('warm_r0.1', 182.0, 0.1, True)]
OFFSETS = (-1000.0, 0.0, 1000.0, 10000.0)   # added to x for the percent change of the base case
BASE_CASE = 'r0.2'
# The warm start: the new input is the previous input plus white noise blurred
# by a Gaussian of WARM_BLUR voxels along each axis, scaled to an rms of
# WARM_RMS times sigma_y.
WARM_BLUR = 3.0
WARM_RMS = 0.2
WARM_SEED = 2
MAX_ITERATIONS = 1500           # the loop stops here at the latest
STOP_DISTANCE = 1e-3            # or when the distance falls below this, in units of sigma_y
CHECK_ITERATIONS = 15
REFERENCE_BOUND = 1e-7          # the rms gradient at which L-BFGS stops, in units of sigma_y
RESULTS = HERE / 'results' / 'stopping_example'
CACHE = Path(tempfile.gettempdir()) / 'improve_denoiser_cache'
DISTANCE_TARGETS = (0.03, 0.01)  # Table 3 gives the iterations to reach these distances
CAP = 15                        # the default max_iterations of denoise on prerelease
IMAGE_CASES = {                 # the cases that draw_images draws, with a note on each figure
    'warm_r0.1': 'The light and dark patches of the MAP estimate come from the change.',
    'r0.05': 'With r\u00a0=\u00a00.05, the prior is strong, and the MAP estimate blurs the edges.',
}
IMAGE_HALF_WIDTH = 100.0        # the gray window of draw_images is water +- this, in HU


def subset_gradient_squared(flat_image, flat_error_image, pixel_indices, fm_constant,
                            qggmrf_params, image_shape):
    """Return the sum, over the voxels of one subset, of the squared gradient of
    the cost.  The two terms of the gradient are those that vcd_subset_denoiser
    computes: prior_grad and -fm_constant * cur_error_image."""
    prior_grad, _, _ = qggmrf.qggmrf_terms_at_indices(
        flat_image, image_shape, pixel_indices, qggmrf_params)
    return torch.sum((prior_grad - fm_constant * flat_error_image[pixel_indices]) ** 2)


def settle(noisy, sigma, sigma_x):
    """Return a denoiser and its pixel partition, with sigma_noise = sigma and the
    given sigma_x.  The partition comes from the global numpy state, seeded here."""
    denoiser = mbirtorch.QGGMRFDenoiser(noisy.shape)
    denoiser.configure_devices(devices=['cpu'])
    denoiser.set_params(no_warning=True, verbose=0, sigma_x=sigma_x, auto_regularize_flag=False)
    np.random.seed(0)
    data = denoiser.initialize_denoiser(image=noisy, sigma_noise=sigma)
    return denoiser, data['partition']


def map_estimate(tag, noisy, sigma_y, sigma_x, p, q, T, init=None):
    """Return the MAP estimate for the input ``noisy`` and the solver's information.
    The result is cached under ``tag``."""
    path = CACHE / f'stopping_example_{tag}_{sigma_y:.6g}_{sigma_x:.6g}_{p:g}_{q:g}_{T:g}.npz'
    if path.exists():
        saved = np.load(path, allow_pickle=True)
        return saved['reference'], saved['info'].item()
    start = time.time()
    reference, info = qggmrf_map3d.solve(noisy, sigma_y, sigma_x, p, q, T, init=init,
                                         rms_bound=REFERENCE_BOUND)
    info['seconds'] = time.time() - start
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, reference=reference, info=np.array(info, dtype=object))
    return reference, info


def warm_perturbation(shape, sigma_y):
    """Return a smooth perturbation with an rms of WARM_RMS * sigma_y."""
    rng = np.random.default_rng(WARM_SEED)
    field = ndimage.gaussian_filter(rng.standard_normal(shape), WARM_BLUR, mode='wrap')
    return field * (WARM_RMS * sigma_y / field.std())


def case_inputs(sigma, r, warm):
    """Return the denoiser and the arrays of one case: the denoiser, its partition,
    the input of the call, the starting image, the MAP estimate, and the solver's
    information about the MAP estimate."""
    _, noisy, _ = make_phantom(SHAPE, noise=NOISE, sigma=sigma)
    sigma_x = r * sigma
    denoiser, partition = settle(noisy, sigma, sigma_x)
    sigma_y = float(denoiser.get_params('sigma_y'))
    p, q, T = (float(v) for v in denoiser.get_params(['p', 'q', 'T']))
    previous_map, previous_info = map_estimate(f'cold_r{r:g}', noisy, sigma_y, sigma_x, p, q, T)
    if warm:
        call_input = noisy + warm_perturbation(noisy.shape, sigma_y)
        reference, info = map_estimate(f'warm_r{r:g}', call_input, sigma_y, sigma_x, p, q, T,
                                       init=previous_map)
        return denoiser, partition, call_input, previous_map, reference, info
    return denoiser, partition, noisy, noisy, previous_map, previous_info


def run_case(name, sigma, r, warm):
    """Run one case and write its record."""
    denoiser, partition, call_input, start_image, reference, info = case_inputs(sigma, r, warm)
    shape = call_input.shape
    sigma_x = r * sigma
    sigma_y = float(denoiser.get_params('sigma_y'))
    p, q, T = (float(v) for v in denoiser.get_params(['p', 'q', 'T']))
    max_alpha = float(denoiser.get_params('max_alpha'))
    positivity = bool(denoiser.get_params('positivity_flag'))
    print(f'{name}: sigma_y {sigma_y:.1f}, sigma_x {sigma_x:.2f}, r {r}, q {q}, T {T}, '
          f'max_alpha {max_alpha}, positivity {positivity}, {partition.shape[0]} subsets',
          flush=True)
    print(f'  reference: {info}', flush=True)

    num_slices = shape[2]
    num_voxels = int(np.prod(shape))
    b = qggmrf.get_b_from_nbr_wts(denoiser.get_params('qggmrf_nbr_wts'))
    qggmrf_params = (b, sigma_x, p, q, T)
    fm_constant = 1.0 / sigma_y ** 2
    input_t = torch.as_tensor(call_input, dtype=torch.float32)
    flat_image = torch.as_tensor(start_image, dtype=torch.float32).reshape(-1, num_slices).clone()
    flat_error = (input_t.reshape(-1, num_slices) - flat_image).contiguous()
    update = maybe_compile(denoising.vcd_subset_denoiser, denoiser.compile_enabled)
    gradient_squared = maybe_compile(subset_gradient_squared, denoiser.compile_enabled)
    pair_lists = denoising._subset_pair_lists(partition, shape, flat_image.device)
    reference_t = torch.as_tensor(reference.reshape(-1, num_slices))

    def rms(x):
        return float(torch.sqrt(torch.mean(x.double() ** 2))) / sigma_y

    rows = []
    checked = None
    distance = rms(flat_image.double() - reference_t)
    rows.append([0, distance] + [np.nan] * (5 + len(OFFSETS)))
    with torch.no_grad():
        for iteration in range(1, MAX_ITERATIONS + 1):
            ell1, alpha, squares, seconds = 0.0, 0.0, 0.0, 0.0
            for k in range(partition.shape[0]):
                squares = squares + gradient_squared(flat_image, flat_error, partition[k],
                                                     fm_constant, qggmrf_params, shape)
                tick = time.time()
                flat_image, flat_error, ell1_subset, alpha_subset = update(
                    flat_image, flat_error, partition[k], fm_constant, qggmrf_params, max_alpha,
                    positivity, shape, pair_lists[k])
                ell1, alpha = ell1 + ell1_subset, alpha + alpha_subset
                seconds += time.time() - tick
            ell1 = float(ell1)
            # This is the denominator of denoise's percent change, computed as denoise does.
            image_ell1 = float(denoising.image_l1_norm(flat_image))
            image64 = flat_image.double()
            offset_norms = [float(torch.sum(torch.abs(image64 + c))) for c in OFFSETS]
            dc_norm = float(torch.sum(torch.abs(image64 - image64.mean())))
            distance = rms(image64 - reference_t)
            rows.append([iteration, distance, 100.0 * ell1 / image_ell1, 100.0 * ell1 / dc_norm,
                         sigma_y * float(torch.sqrt(squares / num_voxels)),
                         float(alpha) / partition.shape[0], seconds]
                        + [100.0 * ell1 / n for n in offset_norms])
            if iteration == CHECK_ITERATIONS:
                checked = flat_image.reshape(shape).numpy().copy()
            # The loop runs at least CHECK_ITERATIONS iterations, for the check below.
            if distance < STOP_DISTANCE and iteration >= CHECK_ITERATIONS:
                break

    # The check: denoise, with the same partition and the same start, gives the loop's
    # image and the loop's percent change.
    output, info_dict = denoiser.denoise(
        call_input.astype(np.float32), sigma_noise=sigma_y,
        init_image=start_image.astype(np.float32), max_iterations=CHECK_ITERATIONS,
        stop_threshold_change_pct=0.0, logfile_path=None, print_logs=False,
        do_initialization=False)
    image_difference = float(np.abs(np.asarray(output) - checked).max()) / sigma_y
    # denoise stores its percent change at each iteration under this key.
    reported = np.asarray(info_dict['recon_params']['stop_threshold_change_pct'][:CHECK_ITERATIONS])
    loop_percent = np.array([row[2] for row in rows[1:CHECK_ITERATIONS + 1]])
    percent_difference = float(np.max(np.abs(reported - loop_percent) / loop_percent))
    print(f'  {len(rows) - 1} iterations, distance {rows[-1][1]:.2e}.  Check after '
          f'{CHECK_ITERATIONS} iterations: largest image difference from denoise '
          f'{image_difference:.1e} sigma_y, largest relative difference of the percent '
          f'change {percent_difference:.1e}', flush=True)

    columns = (['iteration', 'distance', 'percent_change', 'percent_change_dc',
                'gradient_statistic', 'alpha', 'seconds']
               + [f'percent_change_offset_{c:g}' for c in OFFSETS])
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savetxt(RESULTS / f'{name}.csv', np.array(rows), delimiter=',', header=','.join(columns),
               comments='', fmt='%.6e')
    settings = {'name': name, 'sigma': sigma, 'r': r, 'warm': warm, 'shape': shape, 'noise': NOISE,
                'sigma_y': sigma_y, 'sigma_x': sigma_x, 'p': p, 'q': q, 'T': T,
                'max_alpha': max_alpha, 'positivity_flag': positivity,
                'subsets': int(partition.shape[0]), 'reference': info,
                'start_distance': rows[0][1],
                'warm_blur': WARM_BLUR if warm else None, 'warm_rms': WARM_RMS if warm else None,
                'check_iterations': CHECK_ITERATIONS,
                'check_image_difference_sigma': image_difference,
                'check_percent_change_relative_difference': percent_difference,
                'mbirtorch_version': mbirtorch.__version__, 'mbirtorch_file': mbirtorch.__file__,
                'torch_version': torch.__version__, 'torch_threads': torch.get_num_threads()}
    (RESULTS / f'{name}_settings.txt').write_text(json.dumps(settings, indent=1))


def check_offsets():
    """Check that an offset added to the input and to the starting image does not
    change the output of denoise, apart from the offset itself.  The check uses a
    smaller volume, 64 by 64 by 16 voxels, with the settings of the base case."""
    _, noisy, _ = make_phantom((64, 64, 16), noise=NOISE, sigma=182.0)
    noisy = noisy.astype(np.float32)
    denoiser, _ = settle(noisy, 182.0, 0.2 * 182.0)
    outputs = {}
    for c in OFFSETS:
        shifted = (noisy + c).astype(np.float32)
        output, _ = denoiser.denoise(shifted, sigma_noise=182.0, init_image=shifted,
                                     max_iterations=10, stop_threshold_change_pct=0.0,
                                     logfile_path=None, print_logs=False,
                                     do_initialization=False)
        outputs[c] = np.asarray(output, dtype=np.float64) - c
    largest = max(np.abs(outputs[c] - outputs[0.0]).max() for c in OFFSETS) / 182.0
    print(f'Offset check: after 10 iterations, the largest difference between the output '
          f'with an offset, minus the offset, and the output without one: {largest:.1e} '
          f'sigma_y', flush=True)


def first(mask):
    """Return the index of the first True entry, or None."""
    hits = np.flatnonzero(mask)
    return int(hits[0]) if hits.size else None


def load(name):
    """Return the columns of a case's record as a dict, and its settings."""
    path = RESULTS / f'{name}.csv'
    names = path.read_text().splitlines()[0].split(',')
    values = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
    settings = json.loads((RESULTS / f'{name}_settings.txt').read_text())
    return {n: values[:, i] for i, n in enumerate(names)}, settings


def stop(data, column, threshold):
    """Return (iteration, distance) where a rule stops, or (None, None).  Row 0 is
    the starting image, which has no statistic."""
    k = first(data[column][1:] < threshold)
    if k is None:
        return None, None
    return int(data['iteration'][k + 1]), float(data['distance'][k + 1])


def cell(iteration, distance):
    """Format a stop as 'iteration (distance)'."""
    return 'none' if iteration is None else f'{iteration} ({distance:.3f})'


def case_label(sigma, r, warm):
    """Return the name of a case in the tables, such as 'warm start, r = 0.1'."""
    parts = ['warm start'] if warm else []
    if sigma != 182.0:
        parts.append(f'noise {sigma:g} HU')
    parts.append(f'r = {r:g}' + (', the base case' if (sigma, r, warm) == (182.0, 0.2, False)
                                 else ''))
    return ', '.join(parts)


def analyze():
    """Print the tables of stopping_example.md."""
    cases = [case for case in CASES if (RESULTS / f'{case[0]}.csv').exists()]
    records = {name: load(name)[0] for name, _, _, _ in cases}

    print('\nTable 1. Where each rule stops at the thresholds of denoise and greg_dev: '
          'the iteration, and the distance there.')
    print('| Case | Distance at the start | percent change < 0.2 | mean removed < 0.2 '
          f'| gradient statistic < 0.03 | Distance after {CAP} iterations |')
    print('|---|---:|---|---|---|---:|')
    for name, sigma, r, warm in cases:
        data = records[name]
        cells = [cell(*stop(data, column, threshold)) for column, threshold in
                 (('percent_change', 0.2), ('percent_change_dc', 0.2),
                  ('gradient_statistic', 0.03))]
        after_cap = data['distance'][CAP]
        after_cap = '<0.001' if after_cap < 0.0005 else f'{after_cap:.3f}'
        print(f'| {case_label(sigma, r, warm)} | {data["distance"][0]:.2f} | '
              + ' | '.join(cells) + f' | {after_cap} |')

    print('\nTable 2. The base case with an offset added to the image.')
    data = records[BASE_CASE]
    print('| Offset | percent change < 0.2 | percent change < 0.05 | mean removed < 0.2 '
          '| gradient statistic < 0.03 |')
    print('|---:|---|---|---|---|')
    for c in OFFSETS:
        # One run gives every row, because the offset changes only ||x||_1.
        column = f'percent_change_offset_{c:g}'
        offset = f'{c:+g}' if c else '0'
        print(f'| {offset} | {cell(*stop(data, column, 0.2))} | {cell(*stop(data, column, 0.05))} '
              f'| {cell(*stop(data, "percent_change_dc", 0.2))} '
              f'| {cell(*stop(data, "gradient_statistic", 0.03))} |')

    print('\nTable 3. Where each rule stops at stricter thresholds, and the iterations '
          'needed to reach a distance of 0.03 and 0.01.')
    print('| Case | percent change < 0.05 | mean removed < 0.05 | gradient statistic < 0.01 '
          '| Iterations to 0.03 | Iterations to 0.01 |')
    print('|---|---|---|---|---:|---:|')
    for name, sigma, r, warm in cases:
        data = records[name]
        cells = [cell(*stop(data, column, threshold)) for column, threshold in
                 (('percent_change', 0.05), ('percent_change_dc', 0.05),
                  ('gradient_statistic', 0.01))]
        reach = [first(data['distance'] < target) for target in DISTANCE_TARGETS]
        reach = ['-' if k is None else str(int(data['iteration'][k])) for k in reach]
        print(f'| {case_label(sigma, r, warm)} | ' + ' | '.join(cells + reach) + ' |')

    print('\nTable 4. The percent change where the gradient rule at 0.03 stops, and the '
          'median ratio of each statistic to the distance from iteration 5 on.')
    print('| Case | Stop of the gradient rule | Percent change there '
          '| Ratio of the percent change to the distance '
          '| Ratio of the gradient statistic to the distance |')
    print('|---|---|---:|---:|---:|')
    smallest = np.inf
    for name, sigma, r, warm in cases:
        data = records[name]
        iteration, distance = stop(data, 'gradient_statistic', 0.03)
        late = data['iteration'] >= 5
        ratios = [np.median(data[column][late] / data['distance'][late])
                  for column in ('percent_change', 'gradient_statistic')]
        every = data['iteration'] >= 1
        smallest = min(smallest, np.min(data['gradient_statistic'][every]
                                        / data['distance'][every]))
        print(f'| {case_label(sigma, r, warm)} | {cell(iteration, distance)} '
              f'| {data["percent_change"][iteration]:.3f} | {ratios[0]:.3g} | {ratios[1]:.3g} |')
    print(f'The smallest ratio of the gradient statistic to the distance, over every '
          f'iteration of every case: {smallest:.3g}')

    # The ratio of the norm of x with an offset or with the mean removed to the norm of
    # x, in the first iteration of the base case.  The text of stopping_example.md
    # uses these ratios.
    data = records[BASE_CASE]
    norms = {f'offset {c:+g}': data['percent_change'][1] / data[f'percent_change_offset_{c:g}'][1]
             for c in OFFSETS if c != 0}
    norms['mean removed'] = data['percent_change'][1] / data['percent_change_dc'][1]
    print('Norm of x in the base case after 1 iteration, relative to ||x||_1: '
          + ', '.join(f'{key} {value:.4f}' for key, value in norms.items()))


def analyze_mace4d():
    """Print where the rules stop on the real-data calls of mace4d_stopping.md.

    The GPU test of mace4d_stopping.md replayed 58 sampled denoiser calls of
    MACE4D on the Lilly 4DCT phantom scan, for up to 400 iterations each.  Its
    file trajectories.csv holds the distance, the percent change, and the
    gradient statistic of each replay after each iteration.  The replays ran on
    greg_dev, whose line search uses an upper bound on the prior term instead of
    the exact prior term of prerelease.  The file does not hold the norm of x
    minus its mean, so the rule with the mean removed is not available here.
    Calls with the same input have the same replay, so each replay is counted once.
    """
    folder = HERE / 'results' / 'mace4d_stopping'
    calls = np.genfromtxt(folder / 'calls_summary.csv', delimiter=',', names=True, dtype=None,
                          encoding='utf-8')
    trajectories = np.genfromtxt(folder / 'trajectories.csv', delimiter=',', names=True,
                                 dtype=None, encoding='utf-8')
    groups = {}
    for call in calls:
        if call['label'].startswith('frame'):
            group = f'one frame, r = {call["r"]:.2f}'
        else:
            start = 'warm' if call['warm_start'] else 'cold'
            group = f'{start} start, r = {call["r"]:.2f}'
        rows = trajectories[(trajectories['label'] == call['label'])
                            & (trajectories['iteration'] == call['iteration'])
                            & (trajectories['orientation'] == call['orientation'])]
        rows = np.sort(rows, order='step')
        key = tuple(np.round(rows['distance'][:5], 12))
        groups.setdefault(group, {})[key] = rows
    rules = [('percent change < 0.2', 'percent_change', 0.2),
             ('percent change < 0.05', 'percent_change', 0.05),
             ('gradient statistic < 0.03', 'gradient_statistic', 0.03),
             ('gradient statistic < 0.01', 'gradient_statistic', 0.01)]
    print('\nTable 5. The rules on the replays of the real-data calls of mace4d_stopping.md, '
          'with no cap: the range of the stop iteration, and the range of the distance there.')
    print('| Calls | Number | ' + ' | '.join(label for label, _, _ in rules) + ' |')
    print('|---|---:|' + '---|' * len(rules))
    for group, replays in sorted(groups.items()):
        cells = []
        for _, column, threshold in rules:
            stops = []
            for rows in replays.values():
                k = first(rows[column] < threshold)
                if k is not None:
                    stops.append((int(rows['step'][k]), float(rows['distance'][k])))
            iterations = [s[0] for s in stops]
            distances = [f'{d:.3f}' for d in sorted(s[1] for s in stops)]
            span = (str(iterations[0]) if min(iterations) == max(iterations)
                    else f'{min(iterations)} to {max(iterations)}')
            reach = (distances[0] if distances[0] == distances[-1]
                     else f'{distances[0]} to {distances[-1]}')
            cells.append(f'{span} ({reach})')
        print(f'| {group} | {len(replays)} | ' + ' | '.join(cells) + ' |')


def plot():
    """Draw where three rules stop in each case, and save the figure as
    RESULTS / 'stopping_example.png'.  This repository does not keep PNG files."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rules = [('percent change < 0.2', 'percent_change', 0.2, '#2a78d6'),
             ('percent change, mean removed, < 0.2', 'percent_change_dc', 0.2, '#eb6834'),
             ('gradient statistic < 0.03', 'gradient_statistic', 0.03, '#1baf7a')]
    # Each row is (label, case, offset or None).
    rows = [('r = 0.4', 'r0.4', None), ('r = 0.2 (base)', 'r0.2', None),
            ('r = 0.1', 'r0.1', None), ('r = 0.05', 'r0.05', None),
            ('noise 20 HU', 'sigma20_r0.2', None),
            ('offset -1000 (HU)', 'r0.2', -1000.0), ('offset +1000', 'r0.2', 1000.0),
            ('offset +10000', 'r0.2', 10000.0),
            ('warm start, r = 0.2', 'warm_r0.2', None), ('warm start, r = 0.1', 'warm_r0.1', None)]
    ink, muted, grid = '#0b0b0b', '#52514e', '#e4e3df'
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 5.6), sharey=True,
                                gridspec_kw={'width_ratios': [1.25, 1.0]})
    for index, (label, case, offset) in enumerate(rows):
        data, _ = load(case)
        y = len(rows) - 1 - index
        for shift, (_, column, threshold, color) in zip((0.22, 0.0, -0.22), rules):
            if offset is not None and column == 'percent_change':
                column = f'percent_change_offset_{offset:g}'
            iteration, distance = stop(data, column, threshold)
            for axis, value in zip(axes, (distance, iteration)):
                axis.plot(value, y + shift, 'o', color=color, markersize=8,
                          markeredgecolor='white', markeredgewidth=1.5, zorder=3)
    for axis in axes:
        axis.set_xscale('log')
        axis.grid(True, which='major', axis='x', color=grid, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ('top', 'right', 'left'):
            axis.spines[side].set_visible(False)
        axis.spines['bottom'].set_color(muted)
        axis.tick_params(colors=muted, length=0)
        for y in range(len(rows) - 1):
            axis.axhline(y + 0.5, color=grid, linewidth=0.8, zorder=0)
    axes[0].set_yticks(range(len(rows)), [label for label, _, _ in reversed(rows)], color=ink)
    axes[0].set_ylim(-0.6, len(rows) - 0.4)
    axes[0].set_xlabel('rms distance to the MAP estimate at the stop, in units of sigma_y',
                       color=ink)
    axes[1].set_xlabel('iteration at the stop', color=ink)
    handles = [plt.Line2D([], [], marker='o', linestyle='', color=color, markersize=8,
                          markeredgecolor='white', markeredgewidth=1.5, label=label)
               for label, _, _, color in rules]
    figure.legend(handles=handles, loc='upper center', ncol=3, frameon=False,
                  labelcolor=ink, bbox_to_anchor=(0.55, 1.0))
    figure.suptitle('Where each stopping rule stops the denoiser on prerelease', x=0.02,
                    ha='left', y=1.06, color=ink, fontsize=13)
    figure.tight_layout()
    path = RESULTS / 'stopping_example.png'
    figure.savefig(path, dpi=160, bbox_inches='tight', facecolor='white')
    print(f'\nFigure: {path}')


def draw_images(name):
    """Draw the middle slice of one case where the percent change at 0.2 and the
    gradient rule at 0.03 stop, beside the starting image and the MAP estimate, and
    the difference of each from the MAP estimate.  The figure goes to
    RESULTS / 'stopping_images_<name>.png'.

    Both images come from denoise itself, with the partition of the case.  The
    first uses the defaults of denoise on prerelease: a threshold of 0.2 on the
    percent change, and at most CAP iterations.  prerelease has no gradient rule,
    so the second runs the number of iterations at which the gradient rule stops
    in the case's record."""
    import textwrap

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sigma, r, warm = next((s, r, w) for n, s, r, w in CASES if n == name)
    data, _ = load(name)
    gradient_stop, gradient_distance = stop(data, 'gradient_statistic', 0.03)
    denoiser, _, call_input, start_image, reference, _ = case_inputs(sigma, r, warm)
    sigma_y = float(denoiser.get_params('sigma_y'))

    def denoised(max_iterations, threshold):
        output, info_dict = denoiser.denoise(
            call_input.astype(np.float32), sigma_noise=sigma_y,
            init_image=start_image.astype(np.float32), max_iterations=max_iterations,
            stop_threshold_change_pct=threshold, logfile_path=None, print_logs=False,
            do_initialization=False)
        iterations = len(info_dict['recon_params']['stop_threshold_change_pct'])
        return np.asarray(output, dtype=np.float64), iterations

    percent_image, percent_stop = denoised(CAP, 0.2)
    gradient_image, _ = denoised(gradient_stop, 0.0)

    def rms_hu(image):
        return float(np.sqrt(np.mean((image - reference) ** 2)))

    def iterations(count):
        return f'{count} iteration{"" if count == 1 else "s"}'

    print(f'Images of {name}: the percent change stopped denoise after '
          f'{iterations(percent_stop)}, {rms_hu(percent_image) / sigma_y:.3f} sigma_y from '
          f'the MAP estimate.  The gradient rule stops after {iterations(gradient_stop)}, '
          f'{rms_hu(gradient_image) / sigma_y:.3f} sigma_y away '
          f'(the record gives {gradient_distance:.3f}).', flush=True)

    # Each column: the title of the image, the name in the title of its difference
    # from the MAP estimate, and the image.
    columns = [('Start of the call', 'Start', start_image),
               (f'Percent change < 0.2:\nstops after {iterations(percent_stop)}',
                'Percent change', percent_image),
               (f'Gradient statistic < 0.03:\nstops after {iterations(gradient_stop)}',
                'Gradient rule', gradient_image),
               ('MAP estimate of this call', None, reference)]
    # The notes beside the differences.  A no-break space keeps a number on the line of
    # its unit when textwrap breaks the lines.
    if warm:
        change_hu = WARM_RMS * sigma_y
        notes = [f'The input of this call is the noisy image plus a smooth random change '
                 f'with an rms of {WARM_RMS:g}\u00a0$\\sigma_y$ ({change_hu:.0f}\u00a0HU).',
                 'The call starts from the MAP estimate for the noisy image alone.']
    else:
        notes = ['The input of this call is the noisy image, and the call starts from it.']
    notes += [IMAGE_CASES[name], 'Each rms is over the whole volume.']
    middle = SHAPE[2] // 2
    window = (1000.0 - IMAGE_HALF_WIDTH, 1000.0 + IMAGE_HALF_WIDTH)
    # The color range of the differences is 3 times the rms difference of the percent
    # change image, rounded up to a multiple of 10 HU.
    limit = 10.0 * np.ceil(0.3 * rms_hu(percent_image))
    ink = '#0b0b0b'
    figure = plt.figure(figsize=(13.5, 7.9))
    figure.subplots_adjust(left=0.02, right=0.94, top=0.86, bottom=0.03)
    grid = figure.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.05], hspace=0.3,
                               wspace=0.08)
    for column, (title, difference_name, image) in enumerate(columns):
        axis = figure.add_subplot(grid[0, column])
        shown = axis.imshow(image[:, :, middle], cmap='gray', vmin=window[0],
                            vmax=window[1], interpolation='nearest')
        axis.set_title(title, color=ink, fontsize=11)
        axis.set_xticks([])
        axis.set_yticks([])
        if difference_name is None:
            continue
        axis = figure.add_subplot(grid[1, column])
        contrast = axis.imshow(image[:, :, middle] - reference[:, :, middle], cmap='RdBu_r',
                               vmin=-limit, vmax=limit, interpolation='nearest')
        axis.set_title(f'{difference_name} minus MAP:\nrms {rms_hu(image):.1f} HU '
                       f'({rms_hu(image) / sigma_y:.3f} $\\sigma_y$)', color=ink, fontsize=11)
        axis.set_xticks([])
        axis.set_yticks([])
    axis = figure.add_subplot(grid[1, 3])
    axis.axis('off')
    axis.text(0.06, 0.5, '\n\n'.join(textwrap.fill(note, 34) for note in notes),
              transform=axis.transAxes, va='center', ha='left', color=ink, fontsize=10.5)
    figure.colorbar(shown, cax=figure.add_subplot(grid[0, 4]),
                    label='image value (water 1000, air 0)')
    figure.colorbar(contrast, cax=figure.add_subplot(grid[1, 4]), label='difference (HU)')
    start = 'warm start' if warm else 'cold start'
    figure.suptitle(f'Middle slice of the 3D phantom: white noise of {sigma:g} HU, '
                    f'r = {r:g}, {start}', x=0.02, ha='left', color=ink, fontsize=13)
    path = RESULTS / f'stopping_images_{name}.png'
    figure.savefig(path, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(figure)
    print(f'Figure: {path}')


def main():
    torch.manual_seed(0)
    for name, sigma, r, warm in CASES:
        if not (RESULTS / f'{name}.csv').exists():
            run_case(name, sigma, r, warm)
    check_offsets()
    analyze()
    analyze_mace4d()
    plot()
    for name in IMAGE_CASES:
        draw_images(name)


if __name__ == '__main__':
    main()
