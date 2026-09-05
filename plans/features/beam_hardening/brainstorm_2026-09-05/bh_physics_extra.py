import sys, numpy as np
sys.path.insert(0, sys.argv[1])
exec(open(sys.argv[2]).read().split('CASES = [')[0])  # reuse the helper definitions only
from scipy.optimize import least_squares
np.set_printoptions(linewidth=200)

# (a) realistic off-manifold probes on the centered-rod object
E, w = spectrum(200, ('Cu', 0.9)); mu_p, mu_fe = mu_of('PMMA', E), mu_of('Fe', E)
th_ang = np.linspace(0, np.pi, 180, endpoint=False); s_off = np.linspace(-4.2, 4.2, 421)
TH, S = np.meshgrid(th_ang, s_off, indexing='ij')
def chord(R, cx, cy):
    sc = S - (cx * np.cos(TH) + cy * np.sin(TH))
    return np.where(np.abs(sc) < R, 2 * np.sqrt(np.maximum(R**2 - sc**2, 0)), 0.0)
Lp_full = chord(4.0, 0, 0); Lm_c = chord(0.3, 0.0, 0.0); Lp_c = np.maximum(Lp_full - Lm_c, 0)
y_c = y_poly(E, w, [mu_p, mu_fe], [Lp_c, Lm_c]); metal_c = (Lm_c > 0).ravel()
pn, mn = (Lp_c / Lp_c.max()).ravel(), (Lm_c / Lm_c.max()).ravel(); yv = y_c.ravel(); H = marpy_columns(pn, mn)
print('(a) centered rod, 200 kV 0.9 mm Cu: corrected-plastic error on off-manifold probes (m=0.5 of max, p at 0.9x and 1.1x the manifold value)')
for beta in (2e-4, 2e-3, 2e-2):
    th = marpy_ridge_fit(H, yv, beta)
    Sp_all = th[0] + th[1] * mn + th[2] * mn ** 2
    corrected = (yv - th[3] * mn - th[4] * mn ** 2 - th[5] * mn ** 3) / np.maximum(Sp_all, 0.1 * Sp_all.mean())
    scale = np.sum(corrected[~metal_c] * pn[~metal_c]) / np.sum(pn[~metal_c] ** 2)
    out = []
    for fac in (0.9, 1.1):
        probe_m = 0.5; probe_p = fac * pn[metal_c].mean()
        y_probe = float(y_poly(E, w, [mu_p, mu_fe], [probe_p * Lp_c.max(), probe_m * Lm_c.max()])[0])
        corr = (y_probe - th[3] * probe_m - th[4] * probe_m ** 2 - th[5] * probe_m ** 3) / (th[0] + th[1] * probe_m + th[2] * probe_m ** 2) / scale
        out.append(f'p={fac:.1f}x: corrected {corr:.4f} vs truth {probe_p:.4f} ({100*(corr/probe_p-1):+.1f} %)')
    print(f'  beta={beta:.0e}: ' + '; '.join(out))
# the exact inversion has no such error: solve y = f(p, m) for p by bisection at the probe
probe_m = 0.5 * Lm_c.max(); probe_p = 0.9 * Lp_c[Lm_c > 0].mean()
y_probe = float(y_poly(E, w, [mu_p, mu_fe], [probe_p, probe_m])[0])
lo, hi = 0.0, 20.0
for _ in range(60):
    mid = 0.5 * (lo + hi)
    if float(y_poly(E, w, [mu_p, mu_fe], [mid, probe_m])[0]) < y_probe: lo = mid
    else: hi = mid
print(f'  exact inversion of the true model at the same probe: p = {0.5*(lo+hi):.4f} cm vs truth {probe_p:.4f} cm')

# (b) misspecification: truth = Kramers + 1.5 mm Al inherent + 0.9 mm Cu, GOS 0.2 mm; model = Kramers + 1 mm Be + Cu(t) with CsI 0.6 mm, knobs t and kV
def spectrum2(kv, cu_mm, al_inh=1.5, gos_mm=0.2, nE=500):
    E = np.linspace(8.0, kv - 0.5, nE); n = (kv - E) / E
    n *= np.exp(-xraydb.material_mu('Al', E * 1e3) * al_inh / 10.0)
    n *= np.exp(-xraydb.material_mu('Cu', E * 1e3) * cu_mm / 10.0)
    D = E * (1.0 - np.exp(-xraydb.material_mu('Gd2O2S', E * 1e3, density=7.32) * gos_mm / 10.0))
    w = n * D; return E, w / w.sum()
Et, wt = spectrum2(200, 0.9)
pg, mg = np.meshgrid(np.linspace(0, 8, 41), np.linspace(0, 1.0, 26), indexing='ij')
yt = y_poly(Et, wt, [mu_of('PMMA', Et), mu_of('Fe', Et)], [pg, mg]).ravel(); p, m = pg.ravel(), mg.ravel()
print(f'(b) misspecified physical family; truth mean energy {mean_energy(Et, wt):.1f} keV, y max {yt.max():.2f}')
def model_y(x, two_knob):
    kv = 200.0 if not two_knob else float(np.clip(x[1], 120, 300)); cu = float(np.exp(x[0]))
    Em, wm = spectrum(kv, ('Cu', cu)); return y_poly(Em, wm, [mu_of('PMMA', Em), mu_of('Fe', Em)], [p, m])
for two in (False, True):
    x0 = np.array([np.log(0.9)] + ([200.0] if two else []))
    res = least_squares(lambda x: model_y(x, two) - yt, x0, diff_step=1e-3, max_nfev=200)
    knobs = f'Cu {np.exp(res.x[0]):.2f} mm' + (f', kV {res.x[1]:.0f}' if two else ' at the file kV')
    print(f'  {"2" if two else "1"}-knob fit ({knobs}): rms {np.sqrt(np.mean(res.fun**2)):.4f}, max |r| {np.abs(res.fun).max():.4f}')
    # extrapolation of this fit: fit on m<=0.5 then check m in (0.5,1]
    sel = m <= 0.5
    res2 = least_squares(lambda x: (model_y(x, two) - yt)[sel], x0, diff_step=1e-3, max_nfev=200)
    r2 = model_y(res2.x, two) - yt
    print(f'     fit on m<=0.5 cm only: error on m in (0.5,1] max |r| {np.abs(r2[~sel]).max():.4f}, at (p=0,m=1) {r2[(p==0)&(m==1)][0]:+.4f}')
# add the polynomial fitted to the same misspecified truth on the uniform grid for reference
H = marpy_columns(p, m); th = np.linalg.lstsq(H, yt, rcond=None)[0]; r = H @ th - yt
print(f'  mar.py cubic on the same truth: rms {np.sqrt(np.mean(r**2)):.4f}, max |r| {np.abs(r).max():.4f}; inflection of the metal cubic at m = {-2*th[4]/(6*th[5]) if th[5] != 0 else np.nan:.2f} cm')

# (c) base-spectrum LSE fit error without scatter (single material, utilities.py form with 4 equally spaced bins)
E, w = spectrum(200, ('Cu', 0.9)); Lp = np.linspace(0, 8, 81); y0 = y_poly(E, w, [mu_of('PMMA', E)], [Lp])
def lse1(x, L):
    t0 = x[0]; lw = x[1:]; lw = lw - np.logaddexp.reduce(lw); i = np.arange(1, len(lw) + 1)
    return -np.logaddexp.reduce(lw[None, :] - np.outer(L, i * t0), axis=1)
for N in (2, 3, 4):
    r = least_squares(lambda x: lse1(x, Lp) - y0, np.concatenate([[0.05], np.zeros(N)]), max_nfev=6000).fun
    print(f'(c) single-material equally spaced LSE, {N} bins ({N+1} params), PMMA 0-8 cm at 200 kV 0.9 mm Cu: rms {np.sqrt(np.mean(r**2)):.5f} max {np.abs(r).max():.5f}')
E1, w1 = spectrum(100, ('Al', 1.0)); y1 = y_poly(E1, w1, [mu_of('PMMA', E1)], [Lp])
for N in (2, 3, 4, 6):
    r = least_squares(lambda x: lse1(x, Lp) - y1, np.concatenate([[0.05], np.zeros(N)]), max_nfev=6000).fun
    print(f'    same at 100 kV 1 mm Al (y max {y1.max():.2f}): {N} bins: rms {np.sqrt(np.mean(r**2)):.5f} max {np.abs(r).max():.5f}')

# (d) Poisson log bias
rng = np.random.default_rng(0)
for N in (20, 100, 1000):
    k = np.maximum(rng.poisson(N, 400000).astype(float), 0.5)
    print(f'(d) Poisson N={N}: mean(-log(k/N)) = {np.mean(-np.log(k / N)):+.4f} (delta-method 1/(2N) = {1/(2*N):.4f}), std {np.std(-np.log(k/N)):.3f}')
