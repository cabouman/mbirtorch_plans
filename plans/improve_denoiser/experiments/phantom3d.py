"""A 3D test volume for the denoiser experiments of step 4.

The volume follows the synthetic slice of pcdrecon's check_solver.py, made 3D.
Air is 0 and water is 1000, so the values are HU plus 1000.  A body ellipse of
water holds six cylindrical inserts: four iodine concentrations, muscle, and
bone.  Each insert ends a few slices inside the volume, so the volume has
edges along the slice axis as well as within each slice.  A Gaussian blur
softens every edge.

Three kinds of noise can be added, each with a standard deviation of sigma
per voxel.  'white' noise is independent from voxel to voxel.  'fdk' noise has
the spectrum of an FDK image within each slice.  Its amplitude is |f|^(1/2)
times a Hann window that ends at the Nyquist frequency.  So its power is zero
at zero frequency, peaks at 0.15 cycles per voxel, and is zero at the Nyquist
frequency.  Neighbors within a slice have a correlation of 0.57.  The 'fdk'
noise is independent from slice to slice.  'fdk_z' noise is 'fdk' noise
filtered along the slice axis by a Gaussian of SLICE_BLUR slices, which gives
neighbors in adjacent slices a correlation of 0.78.
"""

import numpy as np
from scipy import ndimage

IODINE_HU_PER_MG_ML = 26.04     # the value pcdrecon's check_solver.py uses, at 70 keV
SLICE_BLUR = 1.0                # standard deviation, in slices, of the blur of 'fdk_z' noise


def make_phantom(shape=(256, 256, 32), noise='fdk', sigma=182.0, edge_sigma_px=0.65, seed=1):
    """Return (clean, noisy, flat).

    Args:
        shape: (rows, columns, slices)
        noise: 'fdk', 'fdk_z', or 'white'
        sigma: noise standard deviation per voxel
        edge_sigma_px: standard deviation in voxels of the Gaussian blur of the
            edges.  0.65 voxels gives the 10 to 90% edge width of 0.7 mm at
            the 0.42 mm pixel of pcdrecon's test slice.
        seed: seed of the noise

    Returns:
        clean and noisy are float64 volumes.  flat is a boolean mask of the
        voxels at least 3 voxels from every edge.
    """
    rows, cols, slices = shape
    down, right = np.indices((rows, cols)) - (np.array([rows, cols])[:, None, None] - 1) / 2.0
    body = (right / (0.45 * cols)) ** 2 + (down / (0.35 * rows)) ** 2 <= 1
    clean = np.zeros(shape)
    clean[body] = 1000.0
    values = [1000.0 + IODINE_HU_PER_MG_ML * c for c in (5, 10, 15, 20)] + [1029.0, 1500.0]
    radius = 0.052 * rows
    inserts = np.zeros(shape, dtype=bool)
    for index, value in enumerate(values):
        angle = 2 * np.pi * index / len(values)
        center = (0.17 * rows * np.sin(angle), 0.25 * cols * np.cos(angle))
        disk = np.hypot(down - center[0], right - center[1]) <= radius
        first, last = 3 + 2 * (index % 3), slices - 3 - 2 * (index % 2)
        cylinder = np.zeros(shape, dtype=bool)
        cylinder[disk, first:last] = True
        clean[cylinder] = value
        inserts |= cylinder
    clean = ndimage.gaussian_filter(clean, edge_sigma_px)

    rng = np.random.default_rng(seed)
    if noise == 'white':
        noise_volume = rng.standard_normal(shape)
    elif noise in ('fdk', 'fdk_z'):
        frequency = np.hypot(*np.meshgrid(np.fft.fftfreq(rows), np.fft.fftfreq(cols), indexing='ij'))
        amplitude = np.sqrt(frequency) * np.where(frequency < 0.5,
                                                  0.5 * (1 + np.cos(2 * np.pi * frequency)), 0.0)
        white = rng.standard_normal(shape)
        noise_volume = np.real(np.fft.ifft2(np.fft.fft2(white, axes=(0, 1))
                                            * amplitude[:, :, None], axes=(0, 1)))
        if noise == 'fdk_z':
            noise_volume = ndimage.gaussian_filter1d(noise_volume, SLICE_BLUR, axis=2, mode='wrap')
    else:
        raise ValueError(f'noise must be fdk, fdk_z, or white, not {noise}')
    noise_volume *= sigma / noise_volume.std()

    flat = ndimage.binary_erosion(np.repeat(body[:, :, None], slices, axis=2), iterations=3)
    flat &= ~ndimage.binary_dilation(inserts, iterations=3)
    return clean, clean + noise_volume, flat
