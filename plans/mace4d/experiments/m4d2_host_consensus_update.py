"""Time the host-side consensus update that MACE4D runs once per iteration.

The update reads four pairs of full-size 4D arrays, W and X, and writes new W,
a new consensus average xbar, and the percent change of xbar.  Three
implementations of the same arithmetic are compared: the numpy in-place form
copied from the mbirjax MACE4D loop, a torch CPU form with the same sequence of
in-place operations, and a torch CPU form that uses no scratch array.

Nothing in the mbirtorch package is modified.  Run parameters are the constants
below; the script takes no arguments.
"""

import resource
import sys
import time

import numpy as np
import torch

# ── run parameters ───────────────────────────────────────────────────────────
SHAPE = (8, 192, 192, 192)
REPEATS = 3
BETA = [0.5, 1 / 6, 1 / 6, 1 / 6]
RHO = 0.5
SEED = 11

NUM_AGENTS = 4
BYTES_PER_ARRAY = int(np.prod(SHAPE)) * 4


# ── data ─────────────────────────────────────────────────────────────────────
def make_arrays():
    """The same random W and X for every implementation, from a fixed seed."""
    rng = np.random.default_rng(SEED)
    w = [rng.standard_normal(SHAPE, dtype=np.float32) for _ in range(NUM_AGENTS)]
    x = [rng.standard_normal(SHAPE, dtype=np.float32) for _ in range(NUM_AGENTS)]
    return w, x


# ── implementation (a): numpy in-place, copied from mbirjax ──────────────────
def update_numpy(W, X, xbar_prev, scratch, beta, rho):
    """The consensus update exactly as the mbirjax MACE4D loop writes it."""
    z = np.zeros_like(X[0])
    for k in range(NUM_AGENTS):
        np.multiply(X[k], 2.0, out=scratch)
        scratch -= W[k]
        scratch *= beta[k]
        z += scratch
    for k in range(NUM_AGENTS):
        np.subtract(z, X[k], out=scratch)
        scratch *= (2.0 * rho)
        W[k] += scratch

    xbar = np.zeros_like(X[0])
    for k in range(NUM_AGENTS):
        np.multiply(X[k], beta[k], out=scratch)
        xbar += scratch
    denom = np.linalg.norm(xbar_prev)
    change_pct = (100.0 * np.linalg.norm(xbar - xbar_prev) / denom
                  if denom > 0 else np.inf)
    return xbar, float(change_pct)


# ── implementation (b): torch CPU, same operation sequence, one scratch ──────
def update_torch_scratch(W, X, xbar_prev, scratch, beta, rho):
    """The same operation sequence in torch, with one scratch tensor."""
    z = torch.zeros_like(X[0])
    for k in range(NUM_AGENTS):
        torch.mul(X[k], 2.0, out=scratch)
        scratch -= W[k]
        scratch *= beta[k]
        z += scratch
    for k in range(NUM_AGENTS):
        torch.sub(z, X[k], out=scratch)
        scratch *= (2.0 * rho)
        W[k] += scratch

    xbar = torch.zeros_like(X[0])
    for k in range(NUM_AGENTS):
        torch.mul(X[k], beta[k], out=scratch)
        xbar += scratch
    denom = torch.linalg.vector_norm(xbar_prev)
    torch.sub(xbar, xbar_prev, out=scratch)
    numer = torch.linalg.vector_norm(scratch)
    denom_f = float(denom)
    change_pct = (100.0 * float(numer) / denom_f if denom_f > 0 else float('inf'))
    return xbar, change_pct


# ── implementation (c): torch CPU in-place, no scratch array ─────────────────
def update_torch_no_scratch(W, X, xbar_prev, z, xbar, beta, rho):
    """The same arithmetic with scaled in-place adds, so no scratch is needed.

    z and xbar are reused across calls rather than allocated.  The difference
    xbar - xbar_prev is written into z, which is dead by that point.
    """
    z.zero_()
    for k in range(NUM_AGENTS):
        z.add_(X[k], alpha=2.0 * beta[k]).sub_(W[k], alpha=beta[k])
    for k in range(NUM_AGENTS):
        W[k].add_(z, alpha=2.0 * rho).sub_(X[k], alpha=2.0 * rho)

    xbar.zero_()
    for k in range(NUM_AGENTS):
        xbar.add_(X[k], alpha=beta[k])
    denom = torch.linalg.vector_norm(xbar_prev)
    torch.sub(xbar, xbar_prev, out=z)
    numer = torch.linalg.vector_norm(z)
    denom_f = float(denom)
    change_pct = (100.0 * float(numer) / denom_f if denom_f > 0 else float('inf'))
    return xbar, change_pct


# ── drivers ──────────────────────────────────────────────────────────────────
def run_numpy():
    W, X = make_arrays()
    scratch = np.empty(SHAPE, dtype=np.float32)
    xbar_prev = X[0].copy()
    changes = []
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        xbar, change = update_numpy(W, X, xbar_prev, scratch, BETA, RHO)
        changes.append(change)
        xbar_prev = xbar
    elapsed = time.perf_counter() - t0
    return W, xbar_prev, changes, elapsed


def run_torch_scratch():
    W_np, X_np = make_arrays()
    W = [torch.from_numpy(a) for a in W_np]
    X = [torch.from_numpy(a) for a in X_np]
    scratch = torch.empty(SHAPE, dtype=torch.float32)
    xbar_prev = X[0].clone()
    changes = []
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        xbar, change = update_torch_scratch(W, X, xbar_prev, scratch, BETA, RHO)
        changes.append(change)
        xbar_prev = xbar
    elapsed = time.perf_counter() - t0
    return W, xbar_prev, changes, elapsed


def run_torch_no_scratch():
    W_np, X_np = make_arrays()
    W = [torch.from_numpy(a) for a in W_np]
    X = [torch.from_numpy(a) for a in X_np]
    z = torch.empty(SHAPE, dtype=torch.float32)
    xbar = torch.empty(SHAPE, dtype=torch.float32)
    xbar_prev = X[0].clone()
    changes = []
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        _, change = update_torch_no_scratch(W, X, xbar_prev, z, xbar, BETA, RHO)
        changes.append(change)
        # Swap the two buffers instead of copying, so no new array is made.
        xbar, xbar_prev = xbar_prev, xbar
    elapsed = time.perf_counter() - t0
    return W, xbar_prev, changes, elapsed


def max_abs_diff(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)))


def peak_memory_gb():
    """Peak resident memory of this process.  ru_maxrss is bytes on macOS."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1e9 if sys.platform == 'darwin' else 1e6
    return raw / scale, raw


def main():
    print('m4d2_host_consensus_update')
    print('torch {}   numpy {}'.format(torch.__version__, np.__version__))
    print('torch.get_num_threads() = {}'.format(torch.get_num_threads()))
    print('SHAPE={}  REPEATS={}  RHO={}'.format(SHAPE, REPEATS, RHO))
    print('BETA={}'.format(BETA))
    print('one array = {} elements = {:.3f} GB float32'
          .format(int(np.prod(SHAPE)), BYTES_PER_ARRAY / 1e9))

    print('')
    print('-- (a) numpy in-place --')
    W_a, xbar_a, changes_a, t_a = run_numpy()
    print('total {:.3f} s over {} updates -> {:.3f} s per update'
          .format(t_a, REPEATS, t_a / REPEATS))
    print('change percent per update: {}'.format(
        ['{:.8g}'.format(c) for c in changes_a]))
    gb, raw = peak_memory_gb()
    print('peak resident memory after (a): {:.3f} GB (ru_maxrss={})'.format(gb, raw))

    print('')
    print('-- (b) torch CPU, one scratch tensor --')
    W_b, xbar_b, changes_b, t_b = run_torch_scratch()
    print('total {:.3f} s over {} updates -> {:.3f} s per update'
          .format(t_b, REPEATS, t_b / REPEATS))
    print('change percent per update: {}'.format(
        ['{:.8g}'.format(c) for c in changes_b]))
    w_diff_b = max(max_abs_diff(W_b[k].numpy(), W_a[k]) for k in range(NUM_AGENTS))
    xbar_diff_b = max_abs_diff(xbar_b.numpy(), xbar_a)
    print('max abs difference on W vs (a)    = {:.6e}'.format(w_diff_b))
    print('max abs difference on xbar vs (a) = {:.6e}'.format(xbar_diff_b))
    print('W within 1e-5    = {}'.format(w_diff_b < 1e-5))
    print('xbar within 1e-5 = {}'.format(xbar_diff_b < 1e-5))
    gb, raw = peak_memory_gb()
    print('peak resident memory after (b): {:.3f} GB (ru_maxrss={})'.format(gb, raw))
    del W_b, xbar_b

    print('')
    print('-- (c) torch CPU in-place, no scratch --')
    W_c, xbar_c, changes_c, t_c = run_torch_no_scratch()
    print('total {:.3f} s over {} updates -> {:.3f} s per update'
          .format(t_c, REPEATS, t_c / REPEATS))
    print('change percent per update: {}'.format(
        ['{:.8g}'.format(c) for c in changes_c]))
    w_diff_c = max(max_abs_diff(W_c[k].numpy(), W_a[k]) for k in range(NUM_AGENTS))
    xbar_diff_c = max_abs_diff(xbar_c.numpy(), xbar_a)
    print('max abs difference on W vs (a)    = {:.6e}'.format(w_diff_c))
    print('max abs difference on xbar vs (a) = {:.6e}'.format(xbar_diff_c))
    print('W within 1e-5    = {}'.format(w_diff_c < 1e-5))
    print('xbar within 1e-5 = {}'.format(xbar_diff_c < 1e-5))
    gb, raw = peak_memory_gb()
    print('peak resident memory after (c): {:.3f} GB (ru_maxrss={})'.format(gb, raw))

    print('')
    print('-- change statistic agreement --')
    for i in range(REPEATS):
        ref = changes_a[i]
        if ref != 0:
            rel_b = abs(changes_b[i] - ref) / abs(ref)
            rel_c = abs(changes_c[i] - ref) / abs(ref)
            print('update {}: (a)={:.8g}  rel diff (b)={:.3e}  rel diff (c)={:.3e}'
                  .format(i + 1, ref, rel_b, rel_c))
        else:
            print('update {}: (a)=0  (b)={:.8g}  (c)={:.8g}  '
                  '(relative test not defined at zero)'
                  .format(i + 1, changes_b[i], changes_c[i]))

    print('')
    print('=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print('{:<30} {:>12} {:>16} {:>16}'.format(
        'implementation', 's/update', 'max abs W diff', 'max abs xbar diff'))
    print('{:<30} {:>12.3f} {:>16} {:>16}'.format(
        '(a) numpy in-place', t_a / REPEATS, 'reference', 'reference'))
    print('{:<30} {:>12.3f} {:>16.3e} {:>16.3e}'.format(
        '(b) torch, one scratch', t_b / REPEATS, w_diff_b, xbar_diff_b))
    print('{:<30} {:>12.3f} {:>16.3e} {:>16.3e}'.format(
        '(c) torch, no scratch', t_c / REPEATS, w_diff_c, xbar_diff_c))
    gb, raw = peak_memory_gb()
    print('peak resident memory for the whole run: {:.3f} GB (ru_maxrss={})'
          .format(gb, raw))


if __name__ == '__main__':
    main()
