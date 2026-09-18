# Overview of denoisers for PnP/MACE with CT

The landscape sorts into a few clear buckets, and one option stands out for a proof of concept. 
Short version: **use DRUNet (grayscale, noise-level-conditioned) from the `deepinv` package, 
applied slice-wise, then fuse three orientations** — it's the strongest widely-used plug-and-play 
denoiser, the weights auto-download, and its continuous noise-strength input is exactly the knob 
you want to sweep. Here's the survey behind that.

## 1. Pretrained 2D grayscale denoisers (the quick win)

This bucket is mature and the weights are genuinely easy to get:

- **DRUNet** (Zhang et al., the DPIR prior) — U-Net that takes the noise level σ as an extra input channel, so one set of weights covers a continuous range of strengths. Trained exactly as a Gaussian-denoising prior for plug-and-play work. Grayscale pretrained weights are in `deepinv` (`pip install deepinv`, auto-downloads) and in the original KAIR repo.
- **DnCNN / FFDNet** — older, lighter, also in `deepinv`/KAIR with grayscale weights. Fine as a sanity baseline; DRUNet dominates them in quality.
- **SwinIR / Restormer / SCUNet** — stronger transformer-era nets with grayscale weights, but mostly trained at fixed σ ∈ {15, 25, 50}/255 or as "blind" denoisers. A fixed or hidden noise level is a real drawback here: for a reconstruction prior you want the strength to be an explicit, sweepable parameter.

The σ-conditioning is the reason DRUNet is the right building block: when this graduates from post-processing PoC to a prior agent inside the reconstruction, σ *is* the regularization-strength knob, and you can sweep it instead of guessing.

The whole PoC is roughly this (API from memory — check against the installed `deepinv`, but it's been stable):

```python
import torch, deepinv as dinv

d = dinv.models.DRUNet(in_channels=1, out_channels=1, pretrained="download").eval().to(dev)

def denoise_axis(vol01, sigma, axis, bs=16):          # vol01 scaled to ~[0,1]
    x = torch.moveaxis(vol01, axis, 0).unsqueeze(1)   # (N, 1, H, W)
    with torch.no_grad():
        y = torch.cat([d(b, sigma) for b in x.split(bs)])
    return torch.moveaxis(y.squeeze(1), 0, axis)

fused = sum(denoise_axis(vol01, 0.03, ax) for ax in (0, 1, 2)) / 3
```

Gotchas: scale the volume into ~[0,1] with a robust max (99.9th percentile, not the max) and remember the scale to invert; U-Nets want H,W padded to multiples of 8; sweep σ over roughly 0.01–0.10 rather than picking one.

## 2. Getting from 2D slices to a volume

Three escalating versions, all reusing the same 2D net:

1. **Slice-wise along z** — the minimal PoC; leaves through-plane streaking untouched.
2. **Three orientations, averaged** (the last line above) — a crude one-shot consensus; already visibly better on directional artifacts and nearly free to try.
3. **Proper multi-slice fusion via MACE** — the three orientation denoisers as separate agents equilibrated with the forward-model agent, as in your fusion work. Same component, just moved inside the iterations; I'd only go here after the post-processing PoC looks promising.

There's also a cheap **2.5D trick**: feed three adjacent slices into the *color* DRUNet's RGB channels and keep the middle output channel. It gives the net some through-plane context without any retraining. Worth one experiment, not more.

## 3. Video denoisers as volume denoisers

Your instinct is right, and there is one practical pretrained option: **FastDVDnet** (official PyTorch repo with weights). It takes 5 consecutive frames and denoises the middle one, so sliding it along z makes it effectively a 2.5D denoiser with ±2 slices of context. Caveats: it's trained on RGB natural video with i.i.d. Gaussian noise, so you replicate the grayscale slice into three channels, and its advantage over slice-wise DRUNet on CT content is likely modest. The newer video transformers (VRT/RVRT) also publish weights but are heavy and awkward for a PoC. I'd rank this as "one comparison run," not the primary path.

## 4. Denoisers inside generative models — direct answer to your question

Diffusion/score models **are** standalone-usable denoisers: the ε-prediction (or score) network is an MMSE Gaussian denoiser via Tweedie's formula, so you pick the timestep whose σ matches your noise level and do one forward pass. The ones you can actually do this with are the **pixel-space** models with public weights — OpenAI's guided-diffusion, EDM (`deepinv` wraps guided-diffusion as `DiffUNet`). The ones you *can't* use this way are the latent-space models (Stable Diffusion family): their U-Net denoises in a learned latent, not in pixels.

The practical punchline, though: DRUNet is trained on the same objective — Gaussian denoising conditioned on σ — minus the sampling machinery and the natural-image/RGB/fixed-256² baggage. So "a denoiser extracted from a generative model" and "DRUNet" are nearly the same object, and DRUNet is the low-friction version. I'd only reach for an actual diffusion net if you later want the generative posterior-sampling machinery (DPS-style), which is a different project.

## 5. Self-supervised, trained on your own volume

Worth having in reserve because reconstruction noise is correlated and streaky — not the AWGN any of the above saw in training. **Noise2Void / structN2V** (via the CAREamics package) trains on the noisy volume itself, natively in 3D, with structured-noise masks that tolerate streaks; **ZS-N2N** (zero-shot Noise2Noise) trains a tiny net per-volume in minutes. If DRUNet leaves obvious correlated residue, this is the next lever before committing to training anything supervised.

## 6. CT-specific pretrained nets, and baselines

Low-dose-CT nets trained on the AAPM/Mayo challenge data (RED-CNN, CTformer, WGAN-VGG) match CT noise statistics better, but the weights live in unofficial repos of varying quality and are 2D anyway — I'd skip them for the PoC. As a non-learned volumetric reference, **BM4D** (`pip install bm4d`, CPU) is the honest baseline that tells you what a true 3D prior buys; slow at production scale but fine on PoC crops. And of course the incumbent comparison is whatever prior the reconstruction is currently running.

## What I'd actually do

1. `pip install deepinv`, DRUNet grayscale, slice-wise on a modest crop of a real recon, sweep σ — an afternoon, and step 1 of everything else.
2. Add the three-orientation average (fusion-lite).
3. One calibration run each of FastDVDnet and BM4D to see whether through-plane context is buying anything.
4. If correlated noise survives, try structN2V/ZS-N2N on-volume before considering MACE integration or any training.

Happy to build step 1–2 as a small script in `experiments/` whenever you want.