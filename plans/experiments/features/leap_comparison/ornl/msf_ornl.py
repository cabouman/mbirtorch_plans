"""Multi-slice fusion reconstruction of the ORNL Inconel scan.

Runs MACE with the data-fit proximal map as the forward agent and three
DRUNet denoiser agents, one per slice orientation, at consensus weights
(1/2, 1/6, 1/6, 1/6).  The construction and every algorithm choice follow
the NSI fusion script in mbirtorch_applications/nsi/msf_recon.py and the
findings in mbirtorch_plans/plans/nn_priors/multi_slice_fusion_findings.md.

What differs here is scale.  The volume is (1360, 1360, 1296) float32, about
9 GiB, so the consensus state (four agent inputs, four outputs, and the
consensus itself) lives in HOST memory as cpu tensors.  The proximal agent
passes host arrays to prox_map, whose model runs on all four GPUs.  Each
DRUNet agent streams batches of slices through one GPU and writes the
denoised slices back into a host volume, so no card ever holds a whole
volume for the denoisers.

The reconstruction is initialized at the existing standard MBIR recon (the
15-iteration qGGMRF result saved by the comparison runs), and that same
volume sets the fixed intensity scale: its 99.9th-percentile value maps to
0.9 in the network's [0, 1] range.  sigma_scaled is the one regularization
knob.  There is no ground truth, so the reported metric is the weighted
sinogram residual rms_w(y - Ax), plus saved volumes and slice images for
visual comparison.  With no --init, the script computes its own standard
recon first, which is how the reduced-size smoke run works.

Outputs under out/msf/: the fusion volume, the three-orientation
postprocessing of the standard recon, central slices of all three volumes,
per-iteration traces, and a slice comparison image.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import mbirtorch
import ornl_data

OUT_DIR = '/scratch/gautschi/buzzard/leap_ornl/out/msf/'
LOG_DIR = '/scratch/gautschi/buzzard/leap_ornl/logs/'


def log(msg):
    print(f'[msf {time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ------------------------------------------------------------- MACE loop --
def mace(agents, x0, mu, rho=0.5, num_iterations=30, callback=None):
    """Weighted Mann iteration for consensus equilibrium over agents that
    map a host volume tensor to a host volume tensor.  At a fixed point
    every agent output equals the consensus, so the per-iteration spread
    max_i ||X_i - x_bar|| / ||x_bar|| is the convergence trace."""
    def norm(x):
        return float(torch.sqrt(torch.sum(x * x)))
    W = [x0.clone() for _ in agents]
    info = {'consensus_spread': [], 'consensus_change': [],
            'iteration_seconds': []}
    previous_x_bar = None
    with torch.no_grad():
        for iteration in range(num_iterations):
            t0 = time.time()
            X = [agent(w) for agent, w in zip(agents, W)]
            x_bar = sum(m * x for m, x in zip(mu, X))
            z = sum(m * (2.0 * x - w) for m, x, w in zip(mu, X, W))
            W = [w + 2.0 * rho * (z - x) for w, x in zip(W, X)]
            x_bar_norm = norm(x_bar)
            info['consensus_spread'].append(
                max(norm(x - x_bar) for x in X) / x_bar_norm)
            info['consensus_change'].append(
                norm(x_bar - previous_x_bar) / x_bar_norm
                if previous_x_bar is not None else 0.0)
            info['iteration_seconds'].append(time.time() - t0)
            previous_x_bar = x_bar
            log(f'iteration {iteration}: spread '
                f'{info["consensus_spread"][-1]:.2e}, change '
                f'{info["consensus_change"][-1]:.2e}, '
                f'{info["iteration_seconds"][-1]:.1f} s')
            if callback is not None:
                callback(iteration, x_bar, info)
    return x_bar, info


# ---------------------------------------------------------------- agents --
class ForwardProxAgent:
    """Proximal map of the data-fit term via TomographyModel.prox_map, with
    warm starts from its own previous output and the cumulative iteration
    count passed as first_iteration, so the model's partition sequence walks
    coarse to fine across MACE iterations.  Host arrays in and out; the
    model itself runs on its own devices."""

    def __init__(self, model, sinogram, weights=None, sigma_prox=None,
                 inner_iterations=3, init_recon=None, logfile_path=None):
        self.model = model
        self.sinogram = sinogram
        self.weights = weights
        self.sigma_prox = sigma_prox
        self.inner_iterations = inner_iterations
        self.logfile_path = logfile_path
        self._previous_output = init_recon
        self._iterations_done = 0

    def __call__(self, v):
        first = self._iterations_done
        output, _ = self.model.prox_map(
            np.ascontiguousarray(v.numpy()), self.sinogram,
            sigma_prox=self.sigma_prox,
            weights=self.weights, init_recon=self._previous_output,
            do_initialization=(first == 0),
            max_iterations=first + self.inner_iterations,
            first_iteration=first,
            stop_threshold_change_pct=0.0,
            logfile_path=self.logfile_path,
            print_logs=False)
        self._previous_output = output
        self._iterations_done = first + self.inner_iterations
        return torch.from_numpy(np.ascontiguousarray(output))


def load_drunet(device):
    """The pretrained grayscale DRUNet (weights from the local cache, or
    downloaded on first use)."""
    from deepinv.models import DRUNet
    net = DRUNet(in_channels=1, out_channels=1, pretrained='download',
                 device=device)
    net.eval()
    return net


class DRUNetAgent:
    """DRUNet applied to the volume's slices along ``slice_axis``.

    The volume stays in host memory.  Batches of slices move to this
    agent's device, are scaled by the fixed intensity scale, reflect-padded
    to multiples of 8, denoised, and written back into a host output
    volume.  The scale is shared by all three orientation agents."""

    def __init__(self, net, sigma_noise, intensity_scale, device,
                 slice_batch=8, slice_axis=2):
        self.net = net
        self.sigma_noise = float(sigma_noise)
        self.intensity_scale = float(intensity_scale)
        self.device = device
        self.slice_batch = slice_batch
        self.slice_axis = slice_axis

    def __call__(self, v):
        import torch.nn.functional as functional
        moved = torch.moveaxis(v, self.slice_axis, 0)
        out = torch.empty_like(v)
        out_moved = torch.moveaxis(out, self.slice_axis, 0)
        height, width = moved.shape[-2:]
        pad_rows = (-height) % 8
        pad_cols = (-width) % 8
        sigma_scaled = self.intensity_scale * self.sigma_noise
        with torch.no_grad():
            for start in range(0, moved.shape[0], self.slice_batch):
                batch = moved[start:start + self.slice_batch]
                x = (batch.to(self.device) * self.intensity_scale).unsqueeze(1)
                if pad_rows or pad_cols:
                    x = functional.pad(x, (0, pad_cols, 0, pad_rows),
                                       mode='reflect')
                y = self.net(x, sigma_scaled)
                y = y[..., :height, :width].squeeze(1) / self.intensity_scale
                out_moved[start:start + self.slice_batch] = y.cpu()
        return out


# ------------------------------------------------------------------ main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--view-step', type=int, default=1)
    ap.add_argument('--det-step', type=int, default=1)
    ap.add_argument('--init', default=None,
                    help='existing standard recon (float32 .npy); with no '
                         'value the script computes one first')
    ap.add_argument('--sigma-scaled', type=float, default=0.04)
    ap.add_argument('--iterations', type=int, default=30)
    ap.add_argument('--rho', type=float, default=0.5)
    ap.add_argument('--inner-prox', type=int, default=3)
    ap.add_argument('--slice-batch', type=int, default=8)
    ap.add_argument('--tag', default='full')
    args = ap.parse_args()

    assert torch.cuda.is_available(), 'NOT ON GPU'
    print('library under test:', mbirtorch.__file__, flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    proj, weights, params = ornl_data.load(args.view_step, args.det_step)
    cone = params['proj_params']['cone_params']
    vol = params['vol_params']
    mis = params['miscalib']
    angles = params['proj_params']['angles']
    recon_shape = (int(vol['n_vox_x']), int(vol['n_vox_y']), int(vol['n_vox_z']))
    model = mbirtorch.ConeBeamModel(proj.shape, angles,
                                    source_detector_dist=cone['src_orig'] + cone['orig_det'],
                                    source_iso_dist=cone['src_orig'])
    model.set_params(recon_shape=recon_shape, snr_db=30, sharpness=1.0,
                     positivity_flag=False,
                     delta_det_channel=cone['pix_y'], delta_det_row=cone['pix_x'],
                     delta_voxel=vol['vox_xy'], det_channel_offset=-mis['delta_u'],
                     det_row_offset=-mis['delta_v'], use_ror_mask=False, verbose=0)

    # The standard recon: the initialization and the intensity scale.
    if args.init:
        standard_np = np.load(args.init)
        if tuple(standard_np.shape) != recon_shape:
            raise ValueError(f'--init has shape {standard_np.shape}, the '
                             f'model expects {recon_shape}')
        log(f'standard recon loaded from {args.init}')
    else:
        log('computing the standard recon (no --init given)')
        np.random.seed(0)
        standard_np, _ = model.recon(proj, weights=weights,
                                     max_iterations=15,
                                     stop_threshold_change_pct=0.0,
                                     print_logs=False)
        standard_np = standard_np.astype(np.float32)
    standard = torch.from_numpy(np.ascontiguousarray(standard_np))
    del standard_np

    # The fixed intensity scale.  The percentile runs in numpy because
    # torch.quantile refuses inputs past a few tens of millions of elements,
    # which the strided subsample of a full-size volume exceeds.
    subsample = standard[::4, ::4, ::4].numpy()
    robust_max = float(np.quantile(subsample, 0.999))
    intensity_scale = 0.9 / robust_max
    sigma_recon = args.sigma_scaled / intensity_scale
    log(f'sigma_scaled {args.sigma_scaled:g} ({sigma_recon:.5f} in recon '
        f'units), intensity scale {intensity_scale:.2f} '
        f'(robust max {robust_max:.5f})')

    # One DRUNet per orientation, each on its own card so the proximal
    # model's lead card is not also the denoising card.
    n_dev = torch.cuda.device_count()
    denoise_devices = [f'cuda:{min(a + 1, n_dev - 1)}' for a in range(3)]
    agents_dn = []
    for axis, dev in zip((0, 1, 2), denoise_devices):
        agents_dn.append(DRUNetAgent(load_drunet(dev), sigma_recon,
                                     intensity_scale, dev,
                                     slice_batch=args.slice_batch,
                                     slice_axis=axis))

    def rms_w(volume):
        """Weighted rms sinogram residual, reduced a view block at a time so
        no third sinogram-sized array is formed on the host."""
        ax = model.forward_project(np.ascontiguousarray(volume.numpy()))
        num = 0.0
        den = 0.0
        for v0 in range(0, proj.shape[0], 64):
            r = proj[v0:v0 + 64] - ax[v0:v0 + 64]
            w = weights[v0:v0 + 64]
            num += float(np.sum(w * r * r))
            den += float(np.sum(w))
        del ax
        return float(np.sqrt(num / den))

    results = {'tag': args.tag, 'sigma_scaled': args.sigma_scaled,
               'iterations': args.iterations, 'rho': args.rho,
               'inner_prox': args.inner_prox,
               'intensity_scale': intensity_scale,
               'recon_shape': list(recon_shape)}

    log('data consistency of the standard recon')
    results['rms_w_standard'] = rms_w(standard)
    log(f'rms_w standard: {results["rms_w_standard"]:.5f}')

    # The three-orientation postprocessing of the standard recon, for the
    # side-by-side comparison.
    t0 = time.time()
    postproc = sum(agent(standard) for agent in agents_dn) / 3.0
    log(f'postprocessing took {time.time() - t0:.1f} s')
    results['rms_w_postproc'] = rms_w(postproc)
    log(f'rms_w postprocessing: {results["rms_w_postproc"]:.5f}')
    np.save(os.path.join(OUT_DIR, f'ornl_{args.tag}_postproc.npy'),
            postproc.numpy())

    # The MACE fusion run.
    np.random.seed(0)
    forward = ForwardProxAgent(
        model, proj, weights=weights, sigma_prox=None,
        inner_iterations=args.inner_prox, init_recon=standard.numpy().copy(),
        logfile_path=os.path.join(LOG_DIR, f'msf_ornl_{args.tag}.log'))
    agents = [forward] + agents_dn
    mu = [0.5, 1 / 6, 1 / 6, 1 / 6]

    def checkpoint(iteration, x_bar, info):
        np.savez(os.path.join(OUT_DIR, f'ornl_{args.tag}_traces.npz'),
                 **{k: np.array(v) for k, v in info.items()})
        if (iteration + 1) % 5 == 0 and iteration + 1 < args.iterations:
            np.save(os.path.join(OUT_DIR, f'ornl_{args.tag}_checkpoint.npy'),
                    x_bar.numpy())
            log(f'checkpoint saved at iteration {iteration}')

    t0 = time.time()
    fusion, info = mace(agents, standard, mu, rho=args.rho,
                        num_iterations=args.iterations, callback=checkpoint)
    results['mace_seconds'] = time.time() - t0
    log(f'fusion took {results["mace_seconds"]:.1f} s for '
        f'{args.iterations} iterations')
    results['rms_w_fusion'] = rms_w(fusion)
    log(f'rms_w fusion: {results["rms_w_fusion"]:.5f}')
    results['final_spread'] = info['consensus_spread'][-1]

    np.save(os.path.join(OUT_DIR, f'ornl_{args.tag}_fusion.npy'),
            fusion.numpy())
    mid = [n // 2 for n in recon_shape]
    np.savez(os.path.join(OUT_DIR, f'ornl_{args.tag}_slices.npz'),
             standard=standard[:, :, mid[2]].numpy(),
             postproc=postproc[:, :, mid[2]].numpy(),
             fusion=fusion[:, :, mid[2]].numpy(),
             standard_row=standard[mid[0]].numpy(),
             fusion_row=fusion[mid[0]].numpy())
    with open(os.path.join(OUT_DIR, f'ornl_{args.tag}_results.json'), 'w') as f:
        json.dump(results, f, indent=1)

    # A quick side-by-side of the central axial slice.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s = standard[:, :, mid[2]].numpy()
    lo, hi = np.percentile(s, [1, 99.5])
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for ax_, (img, title) in zip(axes, [
            (s, 'standard qGGMRF recon'),
            (postproc[:, :, mid[2]].numpy(), 'DRUNet postprocessing'),
            (fusion[:, :, mid[2]].numpy(),
             f'multi-slice fusion, sigma {args.sigma_scaled:g}')]):
        ax_.imshow(img, cmap='gray', vmin=lo, vmax=hi)
        ax_.set_title(title)
        ax_.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'ornl_{args.tag}_slices.png'), dpi=110)
    log('MSF RUN DONE')


if __name__ == '__main__':
    main()
