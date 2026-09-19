"""Beam-hardening physics numbers for the brainstorm report.

Spectrum: Kramers bremsstrahlung N(E) ~ (Emax - E)/E, filtered by inherent Be + Al, plus an added
filter (Cu or Al).  No characteristic lines (W K lines at 59/67 keV are omitted).
Attenuation: Elam tables via xraydb (close to NIST XCOM in 20-250 keV, a few percent).
Detector: energy-integrating CsI scintillator, D(E) = E * (1 - exp(-mu_CsI(E) t)).
"""
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
import xraydb
from scipy.optimize import least_squares
np.set_printoptions(linewidth=200)

def spectrum(kv, filt=('Al', 1.0), be_mm=1.0, csi_mm=0.6, nE=500):
    E = np.linspace(8.0, kv - 0.5, nE)  # keV
    n = (kv - E) / E
    n *= np.exp(-xraydb.material_mu('Be', E * 1e3, density=1.85) * be_mm / 10.0)
    fname, fmm = filt
    n *= np.exp(-xraydb.material_mu(fname, E * 1e3) * fmm / 10.0)
    mu_csi = xraydb.material_mu('CsI', E * 1e3, density=4.51)
    D = E * (1.0 - np.exp(-mu_csi * csi_mm / 10.0))
    w = n * D
    w /= w.sum()
    return E, w

MATS = {'PMMA': ('C5H8O2', 1.19), 'Al': ('Al', 2.70), 'Fe': ('Fe', 7.87), 'Ti': ('Ti', 4.51), 'Cu': ('Cu', 8.96)}

def mu_of(name, E):
    f, rho = MATS[name]
    return xraydb.material_mu(f, E * 1e3, density=rho)  # 1/cm

def y_poly(E, w, mus, Ls):
    Ls = [np.atleast_1d(np.asarray(L, dtype=float)) for L in Ls]
    shape = np.broadcast(*Ls).shape
    expo = np.zeros(shape + (E.size,))
    for mu, L in zip(mus, Ls):
        expo = expo + np.broadcast_to(L, shape)[..., None] * mu[None, :]
    return -np.log(np.sum(w * np.exp(-expo), axis=-1))

def mean_energy(E, w, mus=(), Ls=()):
    expo = np.zeros(E.size)
    for mu, L in zip(mus, Ls):
        expo = expo + L * mu
    ww = w * np.exp(-expo)
    return (E * ww).sum() / ww.sum()

def lse_free(params, J, p, m):
    a = np.exp(params[:J]); b = np.exp(params[J:2*J]); lw = params[2*J:3*J]
    lw = lw - np.logaddexp.reduce(lw)
    z = lw[None, :] - np.outer(p, a) - np.outer(m, b)
    return -np.logaddexp.reduce(z, axis=1)

def marpy_columns(p, m):
    return np.stack([p, p * m, p * m ** 2, m, m ** 2, m ** 3], axis=1)

def marpy_ridge_fit(H, y, beta, alpha=1.0):
    """Unconstrained version of mar.py's fit: P = HtH + beta*tr(HtH)/tr(W) * W, W = diag(1 + deg^alpha)."""
    deg = np.array([1, 2, 3, 1, 2, 3], dtype=float)  # total degree of p, pm, pm^2, m, m^2, m^3
    W = np.diag(1 + deg ** alpha)
    HtH = H.T @ H
    lam = beta * np.trace(HtH) / np.trace(W)
    return np.linalg.solve(HtH + lam * W, H.T @ y)

CASES = [(200, ('Cu', 0.9), 'the NSI scans: 200 kV, 0.9 mm Cu'),
         (150, ('Al', 1.0), 'a lighter case: 150 kV, 1 mm Al'),
         (100, ('Al', 1.0), 'a soft case: 100 kV, 1 mm Al')]

for kv, filt, label in CASES:
    E, w = spectrum(kv, filt)
    mu_p, mu_fe, mu_al = mu_of('PMMA', E), mu_of('Fe', E), mu_of('Al', E)
    print(f'\n===== {label}; Kramers + 1 mm Be, CsI 0.6 mm energy-integrating; mean energy {mean_energy(E, w):.1f} keV')
    Ls = np.array([0.1, 1, 2, 4, 6, 8, 10])
    yp = y_poly(E, w, [mu_p], [Ls])
    print('PMMA: L (cm)     ', Ls)
    print('      y          ', np.round(yp, 3))
    print('      y/L (1/cm) ', np.round(yp / Ls, 4), ' ratio to L->0:', np.round((yp / Ls) / (yp[0] / Ls[0]), 3))
    for name, mu, Lm in (('Fe', mu_fe, np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0])), ('Al', mu_al, np.array([0.1, 0.5, 1, 2, 3, 5]))):
        ym = y_poly(E, w, [mu], [Lm])
        print(f'{name}: L (cm)   ', Lm)
        print('      y        ', np.round(ym, 3), ' y/L', np.round(ym / Lm, 3))
        print('      mean energy behind:', np.round([mean_energy(E, w, [mu], [l]) for l in Lm], 1), ' photons/pixel at I0=1e5:', np.round(1e5 * np.exp(-ym), 0))
    P = np.array([2.0, 5.0, 8.0])
    for name, mu, Mlist in (('Fe', mu_fe, [0.1, 0.3, 0.5, 1.0]), ('Al', mu_al, [0.5, 1.0, 2.0])):
        print(f'--- cross terms, PMMA x {name}, at p = {P} cm')
        for m in Mlist:
            ypm = y_poly(E, w, [mu_p, mu], [P, m]); yp0 = y_poly(E, w, [mu_p], [P]); y0m = float(y_poly(E, w, [mu], [m])[0])
            inter = ypm - yp0 - y0m
            h = 1e-3
            dydp_m = (y_poly(E, w, [mu_p, mu], [P + h, m]) - ypm) / h
            dydp_0 = (y_poly(E, w, [mu_p], [P + h]) - yp0) / h
            print(f'  m={m:4.2f} cm: y(0,m)={y0m:.3f}; interaction y(p,m)-y(p,0)-y(0,m)={np.round(inter, 3)} = {np.round(100 * inter / y0m, 1)} % of y(0,m); '
                  f'dy/dp behind metal / dy/dp without = {np.round(dydp_m / dydp_0, 3)}')
    # Fit quality on a uniform presented set
    pg, mg = np.meshgrid(np.linspace(0, 8, 41), np.linspace(0, 1.0, 26), indexing='ij')
    yv = y_poly(E, w, [mu_p, mu_fe], [pg, mg]).ravel(); p, m = pg.ravel(), mg.ravel()
    H = marpy_columns(p, m)
    th, *_ = np.linalg.lstsq(H, yv, rcond=None); r = H @ th - yv
    print(f'--- fits over a uniform grid: PMMA p<=8 cm, Fe m<=1 cm (y from 0 to {yv.max():.2f})')
    print(f'  mar.py cubic (linear in p, 6 cols): rms {np.sqrt(np.mean(r**2)):.4f}, max |r| {np.abs(r).max():.4f}; theta={np.round(th, 4)}')
    H3 = np.stack([p, m, p**2, p*m, m**2, p**3, p**2*m, p*m**2, m**3], axis=1)
    th3, *_ = np.linalg.lstsq(H3, yv, rcond=None); r3 = H3 @ th3 - yv
    print(f'  full cubic in (p,m) (9 cols): rms {np.sqrt(np.mean(r3**2)):.4f}, max |r| {np.abs(r3).max():.4f}')
    sel = m == 0
    print(f'  plastic-only rays: mar.py cubic max |r| {np.abs(r[sel]).max():.4f} (true y at 8 cm {yv[sel][-1]:.3f}, fit {(H@th)[sel][-1]:.3f}); full cubic max |r| {np.abs(r3[sel]).max():.4f}')
    for J in (2, 3, 4):
        x0 = np.concatenate([np.log(np.linspace(0.12, 0.30, J)), np.log(np.linspace(1.0, 20.0, J)), np.zeros(J)])
        res = least_squares(lambda x: lse_free(x, J, p, m) - yv, x0, max_nfev=6000, xtol=1e-13, ftol=1e-13)
        print(f'  free shared-bin LSE J={J} ({3*J-1} params): rms {np.sqrt(np.mean(res.fun**2)):.5f}, max |r| {np.abs(res.fun).max():.5f}')
    for J in (3, 4, 6, 10):
        Ej = np.linspace(E[0] + 5, E[-1] - 5, J); aj = mu_of('PMMA', Ej); bj = mu_of('Fe', Ej)
        def phys(lw):
            lw = lw - np.logaddexp.reduce(lw); z = lw[None, :] - np.outer(p, aj) - np.outer(m, bj)
            return -np.logaddexp.reduce(z, axis=1)
        res = least_squares(lambda x: phys(x) - yv, np.zeros(J), max_nfev=6000, xtol=1e-13, ftol=1e-13)
        print(f'  physical LSE J={J} bins at fixed energies, weights only ({J-1} params): rms {np.sqrt(np.mean(res.fun**2)):.5f}, max |r| {np.abs(res.fun).max():.5f}')
    sel_fit = m <= 0.5
    thx, *_ = np.linalg.lstsq(H[sel_fit], yv[sel_fit], rcond=None); rx = H @ thx - yv
    print(f'  extrapolation, mar.py cubic fit on m<=0.5 cm: error on m in (0.5,1] max |r| {np.abs(rx[~sel_fit]).max():.4f}, at (p=0,m=1) {rx[(p==0)&(m==1)][0]:+.4f}')
    J = 3; x0 = np.concatenate([np.log(np.linspace(0.12, 0.30, J)), np.log(np.linspace(1.0, 20.0, J)), np.zeros(J)])
    res = least_squares(lambda x: lse_free(x, J, p[sel_fit], m[sel_fit]) - yv[sel_fit], x0, max_nfev=6000, xtol=1e-13, ftol=1e-13)
    rl = lse_free(res.x, J, p, m) - yv
    print(f'  extrapolation, free LSE J=3 fit on m<=0.5 cm: error on m in (0.5,1] max |r| {np.abs(rl[~sel_fit]).max():.4f}, at (p=0,m=1) {rl[(p==0)&(m==1)][0]:+.4f}')
    # Scatter offset degeneracy, single material
    Lp = np.linspace(0, 8, 81)
    for s in (0.005, 0.02):
        ys = -np.log(np.exp(-y_poly(E, w, [mu_p], [Lp])) + s)
        def lse1(x, L, zero_bin=False):
            t0 = x[0]; lw = x[1:]; lw = lw - np.logaddexp.reduce(lw)
            i = np.arange(len(lw)) if zero_bin else np.arange(1, len(lw) + 1)
            return -np.logaddexp.reduce(lw[None, :] - np.outer(L, i * t0), axis=1)
        r_a = least_squares(lambda x: lse1(x, Lp) - ys, np.concatenate([[0.05], np.zeros(4)]), max_nfev=6000).fun
        r_b = least_squares(lambda x: lse1(x, Lp, True) - ys, np.concatenate([[0.05], np.zeros(5)]), max_nfev=6000).fun
        print(f'  scatter s={s}: y(8 cm) {ys[-1]:.3f} vs no scatter {float(y_poly(E, w, [mu_p], [8.0])[0]):.3f}; 4-bin LSE (no zero bin) rms {np.sqrt(np.mean(r_a**2)):.4f} max {np.abs(r_a).max():.4f}; with a zero-attenuation bin rms {np.sqrt(np.mean(r_b**2)):.5f}')

# ---- Realistic presented set: PMMA disk radius 4 cm, Fe rod radius 0.3 cm at 2 cm off center; parallel rays
print('\n===== presented-set experiment: PMMA disk R=4 cm with a 0.3 cm radius Fe rod 2 cm off center, 200 kV 0.9 mm Cu')
E, w = spectrum(200, ('Cu', 0.9)); mu_p, mu_fe = mu_of('PMMA', E), mu_of('Fe', E)
th_ang = np.linspace(0, np.pi, 180, endpoint=False); s_off = np.linspace(-4.2, 4.2, 421)
TH, S = np.meshgrid(th_ang, s_off, indexing='ij')
def chord(R, cx, cy):
    sc = S - (cx * np.cos(TH) + cy * np.sin(TH))  # offset of the ray relative to the disk center
    return np.where(np.abs(sc) < R, 2 * np.sqrt(np.maximum(R**2 - sc**2, 0)), 0.0)
Lm = chord(0.3, 2.0, 0.0); Lp_full = chord(4.0, 0, 0); Lp = np.maximum(Lp_full - Lm, 0)
y = y_poly(E, w, [mu_p, mu_fe], [Lp, Lm])
metal = Lm > 0
print(f'  rays: {y.size}, metal rays: {metal.sum()} ({100*metal.mean():.1f} %); on metal rays plastic chord ranges {Lp[metal].min():.2f}..{Lp[metal].max():.2f} cm, metal chord up to {Lm.max():.2f} cm')
# normalized like mar.py: p = Lp/max, m = Lm/max
pn, mn = (Lp / Lp.max()).ravel(), (Lm / Lm.max()).ravel(); yv = y.ravel()
H = marpy_columns(pn, mn)
for beta in (0.0, 2e-4, 2e-3, 2e-2):
    th = marpy_ridge_fit(H, yv, beta) if beta > 0 else np.linalg.lstsq(H, yv, rcond=None)[0]
    r = H @ th - yv
    Sp = th[0] + th[1] * mn + th[2] * mn ** 2
    corrected = (yv - th[3] * mn - th[4] * mn ** 2 - th[5] * mn ** 3) / np.maximum(Sp, 0.1 * Sp.mean())
    truth = pn  # the corrected plastic should be the plastic path in normalized units
    scale = np.sum(corrected[~metal.ravel()] * truth[~metal.ravel()]) / np.sum(truth[~metal.ravel()] ** 2)
    err = corrected / scale - truth
    print(f'  beta={beta:6.0e}: fit rms {np.sqrt(np.mean(r**2)):.4f} (max {np.abs(r).max():.4f}); theta={np.round(th, 3)}; '
          f'corrected-plastic error on metal rays: rms {np.sqrt(np.mean(err[metal.ravel()]**2)):.4f} max {np.abs(err[metal.ravel()]).max():.4f} (in units of max plastic path)')
# the same, but with a rod-centered geometry, where p is nearly constant on metal rays
Lm_c = chord(0.3, 0.0, 0.0); Lp_c = np.maximum(Lp_full - Lm_c, 0); y_c = y_poly(E, w, [mu_p, mu_fe], [Lp_c, Lm_c]); metal_c = (Lm_c > 0).ravel()
pn, mn = (Lp_c / Lp_c.max()).ravel(), (Lm_c / Lm_c.max()).ravel(); yv = y_c.ravel(); H = marpy_columns(pn, mn)
print(f'  centered rod: plastic chord on metal rays ranges {Lp_c[Lm_c>0].min():.3f}..{Lp_c[Lm_c>0].max():.3f} cm (nearly constant), so p*m and m are nearly collinear')
for beta in (0.0, 2e-4, 2e-3, 2e-2):
    th = marpy_ridge_fit(H, yv, beta) if beta > 0 else np.linalg.lstsq(H, yv, rcond=None)[0]
    r = H @ th - yv
    Sp = th[0] + th[1] * mn + th[2] * mn ** 2
    corrected = (yv - th[3] * mn - th[4] * mn ** 2 - th[5] * mn ** 3) / np.maximum(Sp, 0.1 * Sp.mean())
    scale = np.sum(corrected[~metal_c] * pn[~metal_c]) / np.sum(pn[~metal_c] ** 2); err = corrected / scale - pn
    # off-manifold probe: the same theta applied at p half the manifold value (a partial-volume ray)
    probe_m = 0.5; probe_p = 0.5 * pn[metal_c].mean()
    y_probe = float(y_poly(E, w, [mu_p, mu_fe], [probe_p * Lp_c.max(), probe_m * Lm_c.max()])[0])
    corr_probe = (y_probe - th[3] * probe_m - th[4] * probe_m ** 2 - th[5] * probe_m ** 3) / (th[0] + th[1] * probe_m + th[2] * probe_m ** 2) / scale
    print(f'  beta={beta:6.0e}: fit rms {np.sqrt(np.mean(r**2)):.4f}; theta={np.round(th, 3)}; on-manifold corrected error rms {np.sqrt(np.mean(err[metal_c]**2)):.4f}; '
          f'off-manifold probe (p at half the manifold value, m=0.5): corrected {corr_probe:.4f} vs truth {probe_p:.4f}')
