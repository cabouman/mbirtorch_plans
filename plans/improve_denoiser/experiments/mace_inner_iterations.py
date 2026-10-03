"""Inner iterations of the MACE denoiser agent (step 4 of the plan).

QGGMRFDenoiserAgent runs a fixed number of VCD iterations per call, 8 by
default, starting from its previous output.  This script runs the MACE loop
of tests/test_mace.py with each count in INNER_COUNTS and records the NRMSE
of the consensus average against a converged reconstruction after each MACE
iteration.  The forward agent is held fixed: 3 inner iterations, with its
warm start.

With equal agent weights and the same sigma_prox in both agents, the MACE
solution minimizes the data term plus the prior, for any sigma_prox
[derived].  So sigma_prox changes only how fast the loop converges, and the
reconstruction is the reference for every case.  For the denoiser agent,
sigma_prox takes the place of sigma_y, so its prior is set by the ratio
r = sigma_x / sigma_prox.  The agent has no automatic parameters, so the
script passes it the model's automatic sigma_prox and sigma_x, as
tests/test_mace.py does.  Then r = 1, because both are 0.2 * 2^sharpness
times the same estimate (tomography_model.py, auto_set_sigma_x and
auto_set_sigma_prox).  PROX_SCALES multiplies sigma_prox in both agents, so a
scale of 5 runs the denoiser agent at r = 0.2.

The problem is the 2D cone-beam scan of tests/test_mace.py: a Shepp-Logan
phantom, 64 views, 64 detector channels, one detector row.

Run on any machine with mbirtorch installed:
    python mace_inner_iterations.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

import mbirtorch
from mbirtorch.mace import MACE, ForwardProxAgent, QGGMRFDenoiserAgent

HERE = Path(__file__).resolve().parent

# ---- Run parameters ----------------------------------------------------------
INNER_COUNTS = (1, 2, 4, 8, 16, 32)
PROX_SCALES = (1.0, 5.0)        # sigma_prox is this times the automatic value
MACE_ITERATIONS = 60
FORWARD_INNER = 3
REFERENCE_ITERATIONS = 400
NRMSE_TARGETS = (0.01, 0.003, 0.001)
RESULTS = HERE / 'results' / 'mace_inner_iterations'


def problem():
    """The sinogram, weights, and a model factory of tests/test_mace.py."""
    num_views, det = 64, 64
    phantom, sinogram, params = mbirtorch.generate_demo_data(
        model_type='cone', object_type='shepp-logan', num_views=num_views,
        num_det_rows=det, num_det_channels=det, target_max_attenuation=6.0)
    noise_std = np.sqrt(np.exp(sinogram) / 500.0)
    rng = np.random.default_rng(0)
    sinogram = (sinogram + noise_std * rng.standard_normal(sinogram.shape)).astype(np.float32)
    sinogram = sinogram[:, det // 2:det // 2 + 1]

    def make_model():
        model = mbirtorch.ConeBeamModel(sinogram.shape, params['angles'],
                                        source_detector_dist=params['source_detector_dist'],
                                        source_iso_dist=params['source_iso_dist'])
        model.configure_devices(devices=['cpu'])
        model.set_params(no_warning=True, sharpness=1.0, verbose=0)
        return model
    weights = mbirtorch.gen_weights(sinogram, weight_type='transmission_root')
    return sinogram, weights, make_model


def nrmse(x, reference):
    return float(torch.linalg.vector_norm(x - reference) / torch.linalg.vector_norm(reference))


def main():
    sinogram, weights, make_model = problem()
    model = make_model()
    np.random.seed(0)
    half, _ = model.recon(sinogram, weights=weights, max_iterations=REFERENCE_ITERATIONS // 2,
                          stop_threshold_change_pct=0.0, print_logs=False, logfile_path=None,
                          output_sharded=True)
    np.random.seed(0)
    reference, reference_dict = make_model().recon(
        sinogram, weights=weights, max_iterations=REFERENCE_ITERATIONS, stop_threshold_change_pct=0.0,
        print_logs=False, logfile_path=None, output_sharded=True)
    regularization = reference_dict['recon_params']['regularization_params']
    sigma_prox_auto = float(regularization['sigma_prox'])
    sigma_x = float(regularization['sigma_x'])
    print(f'reference: NRMSE between {REFERENCE_ITERATIONS // 2} and {REFERENCE_ITERATIONS} '
          f'iterations {nrmse(half, reference):.2e}; shape {tuple(reference.shape)}; '
          f'sigma_prox {sigma_prox_auto:.4g}, sigma_x {sigma_x:.4g}', flush=True)

    # The start is a 30-iteration reconstruction, as in tests/test_mace.py.
    np.random.seed(0)
    start, _ = make_model().recon(sinogram, weights=weights, max_iterations=30,
                                  stop_threshold_change_pct=0.0, print_logs=False,
                                  logfile_path=None, output_sharded=True)
    print(f'start: NRMSE {nrmse(start, reference):.4f}', flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    summary, columns = {}, {}
    for scale in PROX_SCALES:
        sigma_prox = scale * sigma_prox_auto
        print(f'\nsigma_prox = {scale:g} times automatic, so the denoiser agent has r = '
              f'{sigma_x / sigma_prox:.3f}')
        print(f'{"inner":>6}' + ''.join(f'{"NRMSE at " + str(k):>16}' for k in (10, 20, 40, MACE_ITERATIONS))
              + ''.join(f'{"iters to " + format(t, "g"):>15}' for t in NRMSE_TARGETS)
              + f'{"s per iter":>12}{"denoise share":>15}')
        for inner in INNER_COUNTS:
            np.random.seed(0)
            forward = ForwardProxAgent(make_model(), sinogram, weights=weights, sigma_prox=sigma_prox,
                                       inner_iterations=FORWARD_INNER, init_recon=start.clone())
            denoiser = QGGMRFDenoiserAgent(tuple(reference.shape), sigma_noise=sigma_prox,
                                           pinned_params={'sigma_x': sigma_x}, inner_iterations=inner,
                                           like_model=model, seed=0,
                                           use_ror_mask=model.get_params('use_ror_mask'))
            trace = []
            tick = time.time()
            with MACE([forward, denoiser], start.clone(), rho=0.5) as loop:
                _, info = loop.run(max_iterations=MACE_ITERATIONS,
                                   callback=lambda i, xb: trace.append(nrmse(xb, reference)))
            seconds = (time.time() - tick) / MACE_ITERATIONS
            # Each task row is (agent, task, device, start, end); agent 1 is the denoiser.
            agent_seconds = np.zeros(2)
            for rows in info['tasks']:
                for agent, _, _, begin, end in rows:
                    agent_seconds[agent] += end - begin
            share = agent_seconds[1] / agent_seconds.sum()
            reached = []
            for target in NRMSE_TARGETS:
                hits = np.flatnonzero(np.array(trace) < target)
                reached.append(str(int(hits[0]) + 1) if hits.size else '-')
            print(f'{inner:>6}' + ''.join(f'{trace[k - 1]:>16.5f}' for k in (10, 20, 40, MACE_ITERATIONS))
                  + ''.join(f'{r:>15}' for r in reached) + f'{seconds:>12.3f}{share:>15.2f}', flush=True)
            columns[f'nrmse_scale_{scale:g}_inner_{inner}'] = trace
            columns[f'change_pct_scale_{scale:g}_inner_{inner}'] = [float(v) for v in info['change_pct']]
            summary[f'scale_{scale:g}_inner_{inner}'] = {
                'r': sigma_x / sigma_prox, 'seconds_per_iteration': seconds,
                'denoiser_share': float(share)}
    # One column per case and quantity, one row per MACE iteration.
    np.savetxt(RESULTS / 'traces.csv', np.array(list(columns.values())).T, delimiter=',',
               header=','.join(columns), comments='', fmt='%.6e')
    (RESULTS / 'summary.txt').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
