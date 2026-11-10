"""
esf_lightcone.py -- wide excursion-set (ESF) reionization lightcones, from which pencil beams of our 43-galaxy
footprint are carved (make_pencil_pool.py). Pure numpy/scipy: runs locally, no 21cmFAST.

WHY
---
The v1/v2 priors draw bubbles inside the ~30 x 30 cMpc footprint itself. A volume that small cannot carry a
continuous reionization history: its local ionized fraction IS the imposed global one, slab by slab. A pencil cut
out of a much wider lightcone instead has a "messy" local history -- more or less ionized than the global mean,
depending on the large-scale density around it -- which is what a real survey sees.

THE MODEL (Furlanetto, Zaldarriaga & Hernquist 2004; the ionization step of 21cmFAST, Mesinger+11)
--------------------------------------------------------------------------------------------------
1. Linear density field delta (z = 0) on a periodic grid: Gaussian, Eisenstein & Hu (1998) no-wiggle P(k), Planck18,
   sigma_8 = 0.8102. The grid is a LIGHTCONE: axis 2 is the line of sight, comoving coordinate ell (cMpc, the mock
   frame of generate_bubble_mocks.py), and every LOS cell has its own redshift z(ell) and growth factor D(z).
2. Smooth delta with real-space top hats of radius R = R_MAX ... dx. Extended Press-Schechter collapse fraction of
   halos above M_min inside each sphere:
       f_coll(R) = erfc[ (delta_c / D(z) - delta_R) / sqrt(2 (sigma^2(M_min) - sigma^2(R))) ].
3. A cell is ionized if zeta(z) f_coll(R) >= 1 for any R (21cmFAST's centre-flagging), or, at the cell scale,
   if the partial ionization zeta f_coll(dx) >= 1/2 (the pipeline binarizes x_HI at 0.5).
   Hence with   F = max( max_R f_coll(R), 2 f_coll(dx) )   (stored as log10 F, one field per M_min):
       ionized  <=>  zeta(z) F >= 1.
   One ESF run therefore serves EVERY ionizing efficiency: zeta(z) = zeta_7 ((1 + z) / 8)^alpha sets the global
   history (zeta_7 its timing, alpha its duration), and is chosen per mock to hit a target (x_HI(7), z_end).
   M_min sets which halos drive reionization: larger M_min = rarer, more biased sources = larger bubbles (the role
   the Lu+24 gradual/rapid parameter plays in v2).

Approximations: linear density (no 2LPT), no recombinations, no photon conservation, instantaneous zeta(z) (as
21cmFAST without inhomogeneous recombinations). Check against a real 21cmFAST box: `python esf_lightcone.py validate`.

Usage
-----
python esf_lightcone.py generate --n_real 40 --out_dir ~/Documents/Lya_bubbles/esf_lightcones
python esf_lightcone.py validate --real_box real_vs_mock_fields.npz --out esf_validation.png
"""
import os
import sys
import json
import time
import argparse

import numpy as np
from scipy import fft as sfft
from scipy.special import erfc

_trapz = getattr(np, 'trapezoid', None) or np.trapz      # numpy >= 2 name; the cluster numpy only has np.trapz

REPO = os.path.dirname(os.path.abspath(__file__))

# ---- cosmology: Planck 2018 (astropy Planck18) ----
H = 0.6766
OM, OB, NS, SIGMA8, TCMB = 0.30966, 0.04897, 0.9665, 0.8102, 2.7255
RHO_M = OM * 2.775e11 * H ** 2               # mean matter density today [M_sun / cMpc^3]
DELTA_C = 1.686

# ---- ESF settings ----
M_MIN = (1e8, 1e9, 1e10)                     # [M_sun] halo mass thresholds of the ionizing sources (one F per value)
R_MAX = 50.0                                 # [cMpc] largest smoothing scale (21cmFAST R_BUBBLE_MAX)
R_STEP = 1.1                                 # R_{i+1} = R_i / R_STEP
DX = 2.0                                     # [cMpc] cell size
L_T = 400.0                                  # [cMpc] transverse box side
LOS_PAD = 60.0                               # [cMpc] beyond the sightlines on both LOS ends (> R_MAX: FFT wrap)
Z_END = 5.3


# ---------------------------------------------------------------------------------------------------------------
# linear theory
# ---------------------------------------------------------------------------------------------------------------
def transfer_eh_nowiggle(k):
    """Eisenstein & Hu (1998) zero-baryon-oscillation transfer function, eqs. 26-31. k in 1/cMpc."""
    omh2, obh2, fb = OM * H ** 2, OB * H ** 2, OB / OM
    theta = TCMB / 2.7
    s = 44.5 * np.log(9.83 / omh2) / np.sqrt(1 + 10 * obh2 ** 0.75)                 # sound horizon [cMpc]
    a_g = 1 - 0.328 * np.log(431 * omh2) * fb + 0.38 * np.log(22.3 * omh2) * fb ** 2
    gamma = OM * H * (a_g + (1 - a_g) / (1 + (0.43 * k * s) ** 4))
    q = k * theta ** 2 / (gamma * H)
    l0 = np.log(2 * np.e + 1.8 * q)
    c0 = 14.2 + 731 / (1 + 62.5 * q)
    return l0 / (l0 + c0 * q ** 2)


def tophat(x):
    """Fourier transform of a unit real-space top hat, 3 (sin x - x cos x) / x^3 (-> 1 for x -> 0)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.ones_like(x)
    big = x > 1e-3
    xb = x[big]
    out[big] = 3 * (np.sin(xb) - xb * np.cos(xb)) / xb ** 3
    return out


_KG = np.logspace(-5, 3.5, 6000)


def _sigma2_unnorm(R):
    w = tophat(_KG[None] * np.atleast_1d(R)[:, None])
    return _trapz(_KG ** (2 + NS) * transfer_eh_nowiggle(_KG) ** 2 * w ** 2, _KG, axis=1) / (2 * np.pi ** 2)


_AMP = SIGMA8 ** 2 / _sigma2_unnorm(8.0 / H)[0]


def pk_lin(k):
    """Linear matter power spectrum at z = 0 [cMpc^3], k in 1/cMpc."""
    return _AMP * k ** NS * transfer_eh_nowiggle(k) ** 2


def sigma2(R):
    """Variance of the z = 0 linear field in top hats of radius R [cMpc]."""
    return _AMP * _sigma2_unnorm(R)


def r_of_m(m):
    return (3 * np.asarray(m) / (4 * np.pi * RHO_M)) ** (1 / 3)


_GROWTH_TABLE = None


def growth(z):
    """Linear growth factor of flat LCDM, D(z = 0) = 1 (Heath 1977 integral; tabulated once)."""
    global _GROWTH_TABLE
    if _GROWTH_TABLE is None:
        def d_unnorm(a):
            ag = np.linspace(1e-4, a, 4000)
            e = np.sqrt(OM / ag ** 3 + 1 - OM)
            return 2.5 * OM * np.sqrt(OM / a ** 3 + 1 - OM) * _trapz(1 / (ag * e) ** 3, ag)
        zt = np.linspace(0, 40, 2000)
        dt = np.array([d_unnorm(1 / (1 + zz)) for zz in zt])
        _GROWTH_TABLE = (zt, dt / dt[0])
    return np.interp(np.atleast_1d(z).astype(float), *_GROWTH_TABLE)


_S2MIN = {}


def mean_fcoll(z, m_min):
    """Global EPS collapse fraction above m_min."""
    if m_min not in _S2MIN:
        _S2MIN[m_min] = sigma2(r_of_m(m_min))[0]
    return erfc(DELTA_C / (growth(z) * np.sqrt(2 * _S2MIN[m_min])))


# ---------------------------------------------------------------------------------------------------------------
# fields
# ---------------------------------------------------------------------------------------------------------------
def kgrid(shape, dx):
    kx = 2 * np.pi * np.fft.fftfreq(shape[0], dx)
    ky = 2 * np.pi * np.fft.fftfreq(shape[1], dx)
    kz = 2 * np.pi * np.fft.rfftfreq(shape[2], dx)
    return np.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2).astype(np.float32)


def gaussian_field_k(shape, dx, rng, workers=4):
    """Fourier modes of a Gaussian z = 0 linear density field with power pk_lin (periodic box)."""
    noise = rng.standard_normal(shape, dtype=np.float32)
    dk = sfft.rfftn(noise, workers=workers)
    k = kgrid(shape, dx)
    amp = np.zeros_like(k)
    nz = k > 0
    amp[nz] = np.sqrt(pk_lin(k[nz].astype(np.float64)) / dx ** 3).astype(np.float32)   # white noise -> P(k)
    return (dk * amp).astype(np.complex64), k


def esf_logF(delta_k, k, shape, d_los, m_min=M_MIN, r_max=R_MAX, dx=DX, keep_R=None, workers=4, verbose=True):
    """log10 F (see module docstring) for every cell, one float32 field per entry of m_min.
    d_los: growth factor of every LOS cell (axis 2), or a scalar for a coeval box.
    keep_R: also return the z = 0 linear field smoothed at this radius (for galaxy-bias studies)."""
    radii = [r_max]
    while radii[-1] / R_STEP > dx:
        radii.append(radii[-1] / R_STEP)
    radii.append(dx)
    s2min = sigma2(r_of_m(m_min))                                  # (n_m,)
    s2R = sigma2(np.array(radii))
    dc_over_d = (DELTA_C / np.broadcast_to(np.asarray(d_los, np.float32), (shape[2],))).astype(np.float32)
    amin = [np.full(shape, np.inf, np.float32) for _ in m_min]
    kept = None
    t0 = time.perf_counter()
    for i, R in enumerate(radii):
        dR = sfft.irfftn(delta_k * tophat(k * R).astype(np.float32), s=shape, workers=workers).astype(np.float32)
        if keep_R is not None and kept is None and R <= keep_R:
            kept = dR.copy()
        num = dc_over_d[None, None, :] - dR                     # delta_c / D - delta_R
        for j in range(len(m_min)):
            arg = num * np.float32(1 / np.sqrt(2 * (s2min[j] - s2R[i])))
            if i == len(radii) - 1:                              # cell scale: partial ionization >= 1/2
                fpart = 2 * erfc(arg)
                F = np.maximum(erfc(amin[j]), fpart)
                amin[j] = np.log10(np.maximum(F, 1e-30)).astype(np.float32)
            else:
                np.minimum(amin[j], arg, out=amin[j])
    if verbose:
        print(f"[esf] {len(radii)} radii {r_max:.0f}..{dx:.1f} cMpc, {np.prod(shape) / 1e6:.1f}M cells, "
              f"{time.perf_counter() - t0:.0f} s", flush=True)
    return (amin, kept) if keep_R is not None else amin


# ---------------------------------------------------------------------------------------------------------------
# lightcone geometry: the mock frame of generate_bubble_mocks.py
# ---------------------------------------------------------------------------------------------------------------
def survey_geometry(repo=REPO, z_hi=7.3):
    """Galaxy positions in the mock frame and the LOS coordinate of z = Z_END. Needs pandas (JWST_inference env)."""
    sys.path.insert(0, repo)
    import generate_bubble_mocks as gbm
    cat = gbm.load_real_catalog(repo, z_hi)
    geom = gbm.build_geometry(cat)
    return dict(x_gal=cat['x_gal'], y_gal=cat['y_gal'], z_gal=cat['z_gal'], z0=cat['z0'], z_mean=cat['z_mean'],
                redshift=cat['redshift'], z_end_offset=geom['z_end_offset'])


def los_redshift(ell, z0, z_mean):
    """Redshift of LOS coordinate ell (bubble_prior_v2.LOSRedshift)."""
    from astropy.cosmology import Planck18
    import astropy.units as u
    zg = np.linspace(4.0, 9.5, 6000)
    dg = Planck18.comoving_distance(zg).to(u.Mpc).value
    return np.interp(Planck18.comoving_distance(z0).to(u.Mpc).value + z_mean + np.asarray(ell), dg, zg)


def lightcone_frame(g, dx=DX, l_t=L_T, pad=LOS_PAD):
    """Grid shape, LOS cell centres ell and their redshifts, for the survey geometry g."""
    lo = g['z_end_offset'] - pad
    hi = g['z_gal'].max() + pad
    n_los = int(np.ceil((hi - lo) / dx))
    n_los += n_los % 2                                         # even, for rfft
    n_t = int(round(l_t / dx))
    ell = lo + (np.arange(n_los) + 0.5) * dx
    return (n_t, n_t, n_los), ell, los_redshift(ell, g['z0'], g['z_mean'])


# ---------------------------------------------------------------------------------------------------------------
# global history tables: x_HI(z) of the whole lightcone for any (zeta_7, alpha)
# ---------------------------------------------------------------------------------------------------------------
N_QUANT = 401


def slice_quantiles(logF):
    """(n_los, N_QUANT) quantiles of log10 F in every LOS slice (the global ionization state of that slice)."""
    q = np.linspace(0, 1, N_QUANT)
    return np.quantile(logF.reshape(-1, logF.shape[2]), q, axis=0).T.astype(np.float32)


def xhi_history(qtab, z_los, log_zeta7, alpha):
    """Neutral fraction of every LOS slice: fraction of cells with log F + alpha log((1+z)/8) < -log zeta_7."""
    thr = -log_zeta7 - alpha * np.log10((1 + z_los) / 8.0)               # (n_los,)
    n_q = qtab.shape[1]
    k = np.clip((qtab < thr[:, None]).sum(axis=1), 1, n_q - 1)           # thr lies in [qtab[k-1], qtab[k]]
    rows = np.arange(len(thr))
    lo, hi = qtab[rows, k - 1], qtab[rows, k]
    frac = np.clip((thr - lo) / np.maximum(hi - lo, 1e-12), 0, 1)
    out = (k - 1 + frac) / (n_q - 1)
    out[thr <= qtab[:, 0]] = 0.0
    out[thr >= qtab[:, -1]] = 1.0
    return out


# ---------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------
def run_generate(a):
    os.makedirs(a.out_dir, exist_ok=True)
    gpath = os.path.join(a.out_dir, 'geometry.npz')
    if not os.path.exists(gpath):
        np.savez(gpath, **survey_geometry())
    g = dict(np.load(gpath))
    shape, ell, z_los = lightcone_frame(g, a.dx, a.l_t)
    d_los = growth(z_los).astype(np.float32)
    keep = (ell >= g['z_end_offset'] - 10) & (ell <= g['z_gal'].max() + 10)     # slices free of FFT wrap
    meta = dict(shape=shape, dx=a.dx, l_t=a.l_t, m_min=list(M_MIN), r_max=R_MAX, seed0=a.seed)
    with open(os.path.join(a.out_dir, 'config.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(f"[generate] lightcone {shape} cells of {a.dx} cMpc = {shape[0] * a.dx:.0f}^2 x {shape[2] * a.dx:.0f} cMpc, "
          f"z {z_los[0]:.2f}..{z_los[-1]:.2f}", flush=True)
    for r in range(a.first, a.first + a.n_real):
        out = os.path.join(a.out_dir, f'lc_{r:03d}.npz')
        if os.path.exists(out):
            print(f"[generate] {out} exists, skipping", flush=True)
            continue
        t0 = time.perf_counter()
        rng = np.random.default_rng([a.seed, r])
        dk, k = gaussian_field_k(shape, a.dx, rng, a.workers)
        logF, d5 = esf_logF(dk, k, shape, d_los, dx=a.dx, keep_R=5.0, workers=a.workers, verbose=False)
        qt = np.stack([slice_quantiles(lf) for lf in logF])                      # (n_m, n_los, N_QUANT)
        tmp = out[:-4] + '.tmp.npz'
        np.savez(tmp, logF=np.stack(logF).astype(np.float16), qtab=qt, ell=ell, z_los=z_los, d_los=d_los,
                 keep=keep, m_min=np.array(M_MIN), dx=a.dx, delta5=d5.astype(np.float16))
        os.replace(tmp, out)
        print(f"[generate] realization {r}: {time.perf_counter() - t0:.0f} s -> {out}", flush=True)


def run_validate(a):
    """Coeval ESF box with the size, resolution and neutral fraction of a real 21cmFAST box (z = 6.5), side by side:
    slice, and the mean-free-path bubble size distribution of both."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = np.load(a.real_box)
    real_ion = d['real_ionized'] > 0.5
    dx_r = float(d['cell_size'])
    n = real_ion.shape[0]
    x_hi = 1 - real_ion.mean()
    rng = np.random.default_rng(1)
    dk, k = gaussian_field_k((n, n, n), dx_r, rng)
    logF = esf_logF(dk, k, (n, n, n), growth(6.5)[0], dx=dx_r)
    out = {}
    for j, m in enumerate(M_MIN):
        thr = np.quantile(logF[j], x_hi)                        # zeta chosen to match the real neutral fraction
        out[m] = logF[j] >= thr
        print(f"[validate] M_min {m:.0e}: zeta = {10 ** -thr:.1f}, x_HI = {1 - out[m].mean():.3f} (real {x_hi:.3f})")

    def mfp(ion, n_ray=40000):
        """Mean-free-path BSD: from random ionized cells, distance along a random axis direction to neutral."""
        idx = np.argwhere(ion)
        idx = idx[rng.integers(0, len(idx), n_ray)]
        ax = rng.integers(0, 3, n_ray)
        sg = rng.choice([-1, 1], n_ray)
        L = np.zeros(n_ray)
        pos = idx.copy()
        alive = np.ones(n_ray, bool)
        for step in range(1, n):
            pos[np.arange(n_ray), ax] = (idx[np.arange(n_ray), ax] + sg * step) % n
            hit = alive & ~ion[pos[:, 0], pos[:, 1], pos[:, 2]]
            L[hit] = step * dx_r
            alive &= ~hit
            if not alive.any():
                break
        L[alive] = n * dx_r
        return L

    fig, axs = plt.subplots(1, 5, figsize=(22, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.3]))
    sl = n // 2
    axs[0].imshow(real_ion[:, :, sl].T, origin='lower', cmap='Greys_r', extent=[0, n * dx_r] * 2)
    axs[0].set_title(f'21cmFAST (z=6.5)\nx_HI = {x_hi:.2f}')
    for i, m in enumerate(M_MIN):
        axs[i + 1].imshow(out[m][:, :, sl].T, origin='lower', cmap='Greys_r', extent=[0, n * dx_r] * 2)
        axs[i + 1].set_title(f'ESF, M_min = 1e{np.log10(m):.0f} M$_\\odot$')
    for ax in axs[:4]:
        ax.set_xlabel('cMpc')
    bins = np.logspace(np.log10(dx_r), np.log10(n * dx_r / 2), 40)
    c = 0.5 * (bins[1:] + bins[:-1])
    for lab, ion, col in [('21cmFAST', real_ion, 'k')] + [(f'ESF 1e{np.log10(m):.0f}', out[m], cc)
                                                       for m, cc in zip(M_MIN, ['#1f77b4', '#ff7f0e', '#2ca02c'])]:
        h, _ = np.histogram(mfp(ion), bins)
        axs[4].plot(c, h * c / h.sum() / np.diff(np.log(bins)), color=col, lw=2, label=lab)
    axs[4].set_xscale('log')
    axs[4].set_xlabel('R [cMpc] (mean free path)')
    axs[4].set_ylabel('R dP/dR')
    axs[4].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f"[validate] saved {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('generate')
    p.add_argument('--n_real', type=int, default=40)
    p.add_argument('--first', type=int, default=0)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--dx', type=float, default=DX)
    p.add_argument('--l_t', type=float, default=L_T)
    p.add_argument('--workers', type=int, default=4, help='FFT threads')
    p.add_argument('--out_dir', default=os.path.expanduser('~/Documents/Lya_bubbles/esf_lightcones'))
    v = sub.add_parser('validate')
    v.add_argument('--real_box', default=os.path.join(REPO, 'real_vs_mock_fields.npz'))
    v.add_argument('--out', default='esf_validation.png')
    a = ap.parse_args()
    a.out_dir = os.path.expanduser(getattr(a, 'out_dir', '') or '')
    run_generate(a) if a.cmd == 'generate' else run_validate(a)


if __name__ == '__main__':
    main()
