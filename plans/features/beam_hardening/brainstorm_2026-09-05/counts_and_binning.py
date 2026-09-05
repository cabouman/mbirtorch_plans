import itertools, math
import numpy as np

# 1. Column counts of the existing monomial model (mar.py:764-779).
def exps(num_metal, max_order):
    out = []
    for c in itertools.product(range(max_order + 1), repeat=num_metal):
        if 0 < sum(c) <= max_order:
            out.append(c)
    return out
print("existing H column counts: 1 + cross(order-1) + metal(order)")
for K in (1, 2, 3):
    for order in (2, 3, 4):
        nc = len(exps(K, order - 1)); nm = len(exps(K, order))
        print(f"  K={K} order={order}: {1 + nc + nm} columns  (cross {nc}, metal-only {nm})")

# 2. Reduced problem sizes at a 2K production scan.
V, R, C = 1800, 2000, 2000
full = V * R * C * 4 / 1e9
print(f"\nfull sinogram {full:.1f} GB; 2K^3 volume {2000**3*4/1e9:.1f} GB")
for stride, b, rows in ((4, 2, 64), (4, 4, 32), (4, 4, 128)):
    n = (V // stride) * (rows) * (C // b)
    print(f"  stride {stride}, bin {b}, {rows} binned rows in window: reduced sino {n*4/1e6:.0f} MB; "
          f"slab image 8 slices {(C//b)**2*8*4/1e6:.0f} MB; whole-extent reduced sino {(V//stride)*(R//b)*(C//b)*4/1e9:.2f} GB")

# 3. Binning bias for a quadratic hardening model, disk object, parallel rays.
#    y = a p + b p^2 with p the chord length of a disk of radius R_px (in pixels),
#    sampled at fine channel centers, then averaged over bins of size B.  Fit (a, b)
#    to (mean_bin(p), mean_bin(y)) by least squares over rays that hit the disk.
def disk_chord(u, R):
    return 2.0 * np.sqrt(np.clip(R**2 - u**2, 0, None))
def fit_ab(p, y):
    A = np.stack([p, p**2], axis=1)
    return np.linalg.lstsq(A, y, rcond=None)[0]
a, b = 1.0, -0.05
print(f"\nbinning bias of a quadratic fit, y = {a} p + ({b}) p^2, p in units of R (chord max = 2R -> use p/R)")
for R in (800.0, 50.0):
    u = np.arange(-1024, 1024) + 0.5
    p = disk_chord(u, R) / R           # normalized so p in [0, 2]
    y = a * p + b * p**2
    for B in (1, 2, 4):
        n = (len(u) // B) * B
        pb = p[:n].reshape(-1, B).mean(1)
        yb = y[:n].reshape(-1, B).mean(1)
        keep = pb > 0
        ab = fit_ab(pb[keep], yb[keep])
        print(f"  R={R:4.0f} px, bin {B}: fitted a={ab[0]:.5f} b={ab[1]:.5f}  "
              f"rel err in b {100*(ab[1]-b)/b:+.2f}%  (quadratic excess unchanged: sum b p^2 bias)")
