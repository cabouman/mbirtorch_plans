"""The distance of denoise from the MAP estimate on the 2D slice of the plan.

The plan reports that on the synthetic slice of pcdrecon's check_solver.py, at
r = 0.1, 15 iterations of the fixed denoise left the image 69.6 HU from the
MAP estimate.  The 3D test volume of denoise_convergence.py came much closer
in 15 iterations.  This script repeats the plan's measurement with the
current code, to check that the difference comes from the image and not from
the code.

The slice is rebuilt here from check_solver.py's recipe: 500 by 500 pixels
of 0.42 mm, a water ellipse at 1000 with six inserts, and fdk noise of 182 HU
per pixel.  denoise runs on the slice in two array shapes: (rows, columns, 1)
and (1, rows, columns), the shape pcdrecon's test used.

Results on 2026-10-03, mbirtorch at b0b1879, 4 CPU threads:
    distance from the noisy slice to the MAP estimate: 155.8 HU
    shape (500, 500, 1), 15 iterations: 68.0 HU, which is 0.373 sigma_y
    shape (1, 500, 500), 15 iterations: 69.6 HU, which is 0.383 sigma_y

Run on any machine with mbirtorch installed:
    python plan_slice_check.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import ndimage

import mbirtorch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qggmrf_map3d  # noqa: E402

# ---- Run parameters ----------------------------------------------------------
SIZE = 500                      # pixels on a side
PIXEL_MM = 0.42
NOISE = 182.0                   # HU per pixel
EDGE_MM = 0.7                   # the 10 to 90% width of the edges
IODINE_HU_PER_MG_ML = 26.04
SEED = 1
R = 0.1                         # sigma_x / sigma_y
ITERATIONS = 15


def synthetic_slice():
    """The noisy slice of check_solver.py, in HU plus 1000."""
    down, right = (np.indices((SIZE, SIZE)) - (SIZE - 1) / 2.0) * PIXEL_MM
    image = np.zeros((SIZE, SIZE))
    body = (right / (0.45 * SIZE * PIXEL_MM)) ** 2 + (down / (0.35 * SIZE * PIXEL_MM)) ** 2 <= 1
    image[body] = 1000.0
    values = [1000 + IODINE_HU_PER_MG_ML * c for c in (5, 10, 15, 20)] + [1029.0, 1500.0]
    for index, value in enumerate(values):
        angle = 2 * np.pi * index / len(values)
        center = (0.17 * SIZE * PIXEL_MM * np.sin(angle), 0.25 * SIZE * PIXEL_MM * np.cos(angle))
        image[np.hypot(down - center[0], right - center[1]) <= 11.0] = value
    image = ndimage.gaussian_filter(image, EDGE_MM / 2.563 / PIXEL_MM)
    frequency = np.hypot(*np.meshgrid(np.fft.fftfreq(SIZE), np.fft.fftfreq(SIZE), indexing='ij'))
    amplitude = np.sqrt(frequency) * np.where(frequency < 0.5, 0.5 * (1 + np.cos(2 * np.pi * frequency)), 0.0)
    rng = np.random.default_rng(SEED)
    noise = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((SIZE, SIZE))) * amplitude))
    return image + noise * NOISE / noise.std()


def main():
    warnings.filterwarnings('ignore', message='You are directly setting regularization parameters')
    noisy = synthetic_slice()
    sigma_x = R * NOISE
    reference, _ = qggmrf_map3d.solve(noisy[:, :, None], NOISE, sigma_x)
    reference = reference[:, :, 0]
    print(f'distance from the noisy slice to the MAP estimate: '
          f'{np.sqrt(np.mean((noisy - reference) ** 2)):.1f} HU')
    for shape in ((SIZE, SIZE, 1), (1, SIZE, SIZE)):
        image = noisy.reshape(shape)
        denoiser = mbirtorch.QGGMRFDenoiser(shape)
        denoiser.set_params(verbose=0, auto_regularize_flag=False, sigma_y=NOISE, sigma_x=sigma_x,
                            p=2.0, q=1.2, T=1.0)
        np.random.seed(SEED)    # the partition is drawn from the global state
        output, _ = denoiser.denoise(image, sigma_noise=NOISE, max_iterations=ITERATIONS,
                                     stop_threshold_change_pct=0.0)
        distance = np.sqrt(np.mean((np.asarray(output).reshape(SIZE, SIZE) - reference) ** 2))
        print(f'shape {shape}, {ITERATIONS} iterations: {distance:.1f} HU, which is '
              f'{distance / NOISE:.3f} sigma_y')


if __name__ == '__main__':
    main()
