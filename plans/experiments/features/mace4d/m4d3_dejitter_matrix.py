"""Express the MACE4D temporal dejitter filter as one matrix and time both forms.

The mbirjax filter _dejitter_4d_dct zeroes a few DCT-I modes along the frame
axis and leaves the spatial axes alone, so it is a linear map on the frame axis
only.  That map is one N by N matrix, where N is the number of frames.  This
script copies the filter, builds its matrix by filtering the identity, checks
that the matrix reproduces the filter, and times the two forms.

Nothing in the mbirtorch package is modified.  Run parameters are the constants
below; the script takes no arguments.
"""

import time

import numpy as np
import torch
from scipy.fft import dct, idct

# ── run parameters ───────────────────────────────────────────────────────────
N_LIST = (3, 12, 30)
PERIOD = 6
HARMONICS = True
BAND_WIDTH = 1
CHECK_SPATIAL = (16, 16, 8)
TIMING_SHAPE = (30, 256, 256, 64)
TIMING_PERIOD = 6
TIMING_REPEATS = 2
SEED = 5
USE_MPS = torch.backends.mps.is_available()


# ── the mbirjax filter, copied, with a workers argument added ────────────────
def _dejitter_4d_dct(
    recon_4d,
    period,
    harmonics=True,
    band_width=1,
    dtype=np.float32,
    chunk_size=None,
    verbose=False,
    workers=None,
):
    """Remove periodic temporal jitter from a 4D reconstruction via DCT-I filtering.

    Copied from mbirjax/mace4d.py.  The only change is the workers argument,
    which is handed to scipy's dct and idct so the transform can use several
    threads.  workers=None calls dct and idct exactly as the original does.

    Args:
        recon_4d (ndarray): 4D volume, shape (time, x, y, z).
        period (float or int): Main jitter period in frames.
        harmonics (bool or list of int): True removes the main period and all
            harmonics with period/h >= 2.  False removes only the main period.
            A list specifies explicit harmonic indices h to remove.
        band_width (int): Number of DCT-I modes to zero on each side of the
            target mode.
        dtype (np.dtype): Working dtype.
        chunk_size (int or None): Process the last spatial axis in chunks of
            this size.  None processes the whole axis in one pass.
        verbose (bool): Print the modes being zeroed.
        workers (int or None): Passed to scipy dct and idct when not None.

    Returns:
        ndarray: Dejittered volume, same shape as recon_4d.
    """
    recon_4d = np.asarray(recon_4d)
    N = recon_4d.shape[0]
    spatial_shape = recon_4d.shape[1:]

    if harmonics is False:
        harmonic_list = [1]
    elif harmonics is True:
        max_h = int(np.floor(period / 2))
        harmonic_list = list(range(1, max_h + 1))
    else:
        harmonic_list = list(harmonics)

    periods_to_remove = [period / h for h in harmonic_list]

    if verbose:
        print("Input shape:", recon_4d.shape)
        print("Periods to remove:", periods_to_remove)

    Z = spatial_shape[-1]
    if chunk_size is None:
        chunk_size = Z

    kw = {} if workers is None else {"workers": workers}

    recon_dejittered = np.empty((N,) + spatial_shape, dtype=dtype)
    for z0 in range(0, Z, chunk_size):
        z1 = min(z0 + chunk_size, Z)
        block = np.asarray(recon_4d[..., z0:z1], dtype=dtype)
        C = dct(block, type=1, norm="ortho", axis=0, **kw)
        for p in periods_to_remove:
            k_center = 2 * (N - 1) / p
            k0 = int(round(k_center))
            lo = max(0, k0 - band_width)
            hi = min(C.shape[0], k0 + band_width + 1)
            if lo < hi:
                C[lo:hi, ...] = 0
            if verbose and z0 == 0:
                actual_period = 2 * (N - 1) / k0 if k0 != 0 else np.inf
                print(
                    f"  Removed period {p:.3g}: "
                    f"k~{k_center:.2f}, rounded k={k0}, "
                    f"actual period~{actual_period:.3g}, "
                    f"zeroed k={lo}:{hi - 1}"
                )
        recon_dejittered[..., z0:z1] = idct(C, type=1, norm="ortho", axis=0,
                                            **kw).astype(dtype, copy=False)
        del block, C
    return recon_dejittered


# ── the matrix of the filter ─────────────────────────────────────────────────
def dejitter_matrix(num_frames, period, harmonics=HARMONICS, band_width=BAND_WIDTH):
    """Apply the filter to each unit frame impulse to get the columns of M."""
    matrix = np.zeros((num_frames, num_frames), dtype=np.float32)
    for j in range(num_frames):
        e_j = np.zeros((num_frames, 1, 1, 1), dtype=np.float32)
        e_j[j] = 1.0
        out = _dejitter_4d_dct(e_j, period, harmonics=harmonics,
                               band_width=band_width, dtype=np.float32)
        matrix[:, j] = out.reshape(num_frames)
    return matrix


def max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64)
                               - np.asarray(b, dtype=np.float64))))


# ── part 1 and part 2: agreement and matrix properties ───────────────────────
def check_matrix(num_frames, rng):
    print('')
    print('-' * 70)
    print('N = {}  period = {}  harmonics = {}  band_width = {}'
          .format(num_frames, PERIOD, HARMONICS, BAND_WIDTH))
    print('-' * 70)
    matrix = dejitter_matrix(num_frames, PERIOD)

    x = rng.standard_normal((num_frames,) + CHECK_SPATIAL, dtype=np.float32)
    y_filter = _dejitter_4d_dct(x, PERIOD, harmonics=HARMONICS,
                                band_width=BAND_WIDTH, dtype=np.float32)
    y_matrix = np.einsum('ij,j...->i...', matrix, x)
    diff = max_abs(y_matrix, y_filter)
    print('max |x|                        = {:.6e}'.format(float(np.max(np.abs(x)))))
    print('max abs difference matrix vs filter = {:.6e}'.format(diff))

    m64 = matrix.astype(np.float64)
    sym = float(np.max(np.abs(m64 - m64.T)))
    idem = float(np.max(np.abs(m64 @ m64 - m64)))
    rank_default = int(np.linalg.matrix_rank(m64))
    # M is built in float32, so its zero singular values come out near 1e-8.
    # The default tolerance is a float64 one and counts them as nonzero, so a
    # float32 tolerance is reported as well.
    sing = np.linalg.svd(m64, compute_uv=False)
    tol_f32 = max(m64.shape) * np.finfo(np.float32).eps * (sing[0] if sing.size else 0.0)
    rank_f32 = int(np.linalg.matrix_rank(m64, tol=tol_f32))
    print('max |M - M^T|                  = {:.6e}'.format(sym))
    print('max |M M - M|                  = {:.6e}'.format(idem))
    print('rank(M), numpy default tol     = {}   N = {}'.format(rank_default, num_frames))
    print('rank(M), float32 tol {:.3e} = {}'.format(tol_f32, rank_f32))
    print('singular values: {}'.format(
        np.array2string(sing, precision=4, suppress_small=False, max_line_width=100)))

    if num_frames == 3:
        print('M for N = 3:')
        with np.printoptions(precision=8, suppress=False):
            print(matrix)

    return dict(N=num_frames, diff=diff, sym=sym, idem=idem,
                rank_default=rank_default, rank_f32=rank_f32)


# ── part 3: timing ───────────────────────────────────────────────────────────
def time_call(fn, repeats, device=None):
    if device == 'mps':
        torch.mps.synchronize()
    fn()
    if device == 'mps':
        torch.mps.synchronize()
    times = []
    for _ in range(repeats):
        if device == 'mps':
            torch.mps.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if device == 'mps':
            torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), out


def run_timing(rng):
    num_frames = TIMING_SHAPE[0]
    print('')
    print('=' * 70)
    print('TIMING on x of shape {} float32 ({:.3f} GB)'
          .format(TIMING_SHAPE, 4 * int(np.prod(TIMING_SHAPE)) / 1e9))
    print('=' * 70)
    x = rng.standard_normal(TIMING_SHAPE, dtype=np.float32)
    matrix = dejitter_matrix(num_frames, TIMING_PERIOD)

    t_scipy, y_scipy = time_call(
        lambda: _dejitter_4d_dct(x, TIMING_PERIOD, harmonics=HARMONICS,
                                 band_width=BAND_WIDTH, dtype=np.float32,
                                 chunk_size=None),
        TIMING_REPEATS)
    print('scipy filter, chunk_size=None     : {:.3f} s'.format(t_scipy))

    t_workers, _ = time_call(
        lambda: _dejitter_4d_dct(x, TIMING_PERIOD, harmonics=HARMONICS,
                                 band_width=BAND_WIDTH, dtype=np.float32,
                                 chunk_size=None, workers=-1),
        TIMING_REPEATS)
    print('scipy filter, workers=-1          : {:.3f} s'.format(t_workers))

    m_t = torch.from_numpy(matrix)
    x_t = torch.from_numpy(x)
    x_flat = x_t.reshape(num_frames, -1)

    def cpu_matmul():
        return torch.matmul(m_t, x_flat).reshape(x_t.shape)

    t_cpu, y_cpu = time_call(cpu_matmul, TIMING_REPEATS)
    print('torch CPU matrix form             : {:.3f} s'.format(t_cpu))

    diff_cpu = max_abs(y_cpu.numpy(), y_scipy)
    print('max abs difference scipy vs torch CPU matrix = {:.6e}'.format(diff_cpu))

    t_mps = float('nan')
    if USE_MPS:
        m_m = m_t.to('mps')
        x_m = x_flat.to('mps')
        torch.mps.synchronize()
        t_mps, _ = time_call(lambda: torch.matmul(m_m, x_m),
                             TIMING_REPEATS, device='mps')
        print('torch mps matrix form (matmul only): {:.3f} s'.format(t_mps))
        del m_m, x_m
        torch.mps.empty_cache()
    else:
        print('torch mps matrix form             : skipped, mps not available')

    return dict(t_scipy=t_scipy, t_workers=t_workers, t_cpu=t_cpu, t_mps=t_mps,
                diff_cpu=diff_cpu)


def main():
    print('m4d3_dejitter_matrix')
    import scipy
    print('torch {}   numpy {}   scipy {}'
          .format(torch.__version__, np.__version__, scipy.__version__))
    print('torch.get_num_threads() = {}   mps available = {}'
          .format(torch.get_num_threads(), USE_MPS))
    print('N_LIST={}  PERIOD={}  HARMONICS={}  BAND_WIDTH={}'
          .format(N_LIST, PERIOD, HARMONICS, BAND_WIDTH))
    print('check array spatial shape = {}'.format(CHECK_SPATIAL))

    rng = np.random.default_rng(SEED)
    rows = [check_matrix(n, rng) for n in N_LIST]
    timing = run_timing(rng)

    print('')
    print('=' * 70)
    print('SUMMARY: matrix properties')
    print('=' * 70)
    print('{:>4} {:>16} {:>14} {:>14} {:>12} {:>10}'
          .format('N', 'matrix-filter', 'max|M-M^T|', 'max|MM-M|',
                  'rank default', 'rank f32'))
    for r in rows:
        print('{:>4} {:>16.6e} {:>14.6e} {:>14.6e} {:>12} {:>10}'
              .format(r['N'], r['diff'], r['sym'], r['idem'],
                      r['rank_default'], r['rank_f32']))

    print('')
    print('=' * 70)
    print('SUMMARY: timing on {}'.format(TIMING_SHAPE))
    print('=' * 70)
    print('{:<38} {:>10}'.format('form', 'seconds'))
    print('{:<38} {:>10.3f}'.format('scipy filter, chunk_size=None', timing['t_scipy']))
    print('{:<38} {:>10.3f}'.format('scipy filter, workers=-1', timing['t_workers']))
    print('{:<38} {:>10.3f}'.format('torch CPU matrix form', timing['t_cpu']))
    print('{:<38} {:>10.3f}'.format('torch mps matrix form', timing['t_mps']))
    print('max abs difference scipy vs torch CPU matrix = {:.6e}'
          .format(timing['diff_cpu']))


if __name__ == '__main__':
    main()
