"""
make_pencil_pool.py -- the "pencil-beam" prior: mocks of our 43 sightlines carved out of the wide ESF lightcones of
esf_lightcone.py, in the pool format of the exact likelihood (exact_pixel_likelihood_chunked.py).

One mock = one lightcone realization, one source model and one global history, and one placement of the survey:
  M_min   ~ uniform over esf_lightcone.M_MIN (which halos drive reionization; bubble sizes)
  x_hi_7  ~ U(0.06, 0.95)   global neutral fraction at z = 7 (same prior as v2)
  alpha   ~ U(ALPHA_LO, alpha_max(M_min))   ionizing efficiency zeta(z) = zeta_7 ((1+z)/8)^alpha: the duration.
            alpha_max keeps zeta(z) <f_coll>(z) growing with time over the whole lightcone, so the global gas only
            ever ionizes (an ESF with a fast-falling zeta would re-neutralize).
  zeta_7  follows: the x_hi_7 quantile of log F in the z = 7 slice of that lightcone.
  placement: the 43-galaxy footprint (transverse positions, rotated by a multiple of 90 deg and possibly mirrored)
            shifted to a uniformly random transverse position (the boxes are periodic transversely).
Sightline points as in v1/v2 (generate_bubble_mocks.py): 75 points from z = 5.3 to each galaxy; a point is ionized
if zeta(z) F >= 1 in its cell.

Derived per mock (outputs, not inputs): z_end (where the global x_HI falls to 0.05; < 5.3 is stored as 5.3 - ...,
see code), tau_e (box history below z = 7.4, the analytic ESF history above it), the LOCAL neutral fraction of the
pencil around the galaxies vs the global one, and the linear overdensity at each galaxy (R = 5 cMpc, for later
galaxy-bias weighting). The Lyman-alpha-forest cut z_end < 6.05 of v2 is NOT built in: apply it (or not) as a
reweighting in the analysis, like the tau prior.

Output (one batch file per lightcone realization):
  <out_dir>/mocks/train_batch_XXXXX.npz : masks (n, 43, 75) uint8, 1 = ionized, column 0 at z_end end (as v2),
                                          x_ion, x_hi_7, z_end, tau_e, alpha, log_zeta7, m_min, realization, ...
  <out_dir>/pool/train_batch_XXXXX.npz  : theta = 1 - masks[:, :, ::-1] flattened (n, 43*75) uint8, n_gal, n_los

Usage
-----
python make_pencil_pool.py --lc_dir ~/Documents/Lya_bubbles/esf_lightcones --n_per 3000 \\
    --out_dir ~/Documents/Lya_bubbles/esf_pencils
"""
import os
import glob
import time
import argparse

import numpy as np

import esf_lightcone as esf

X_HI_7_RANGE = (0.06, 0.95)
ALPHA_LO = -10.0
X_HI_END = 0.05
Z_END_MIN = 5.3                 # with --z_end_max: also require the history to end by z = 5.3 (v2 range)
N_LOS = 75


def alpha_max(m_min, margin=0.9):
    """Largest alpha for which zeta(z) <f_coll>(z) still grows with time at every z of the lightcone."""
    zz = np.linspace(5.2, 7.5, 50)
    slope = np.gradient(np.log(esf.mean_fcoll(zz, m_min)), np.log(1 + zz))     # d ln f / d ln(1+z) < 0
    return margin * np.min(-slope)


_TAU = {}


def tau_e(z_box, xhi_box, log_zeta7, alpha, m_min):
    """Thomson optical depth: the box's global history over its z range, below it fully ionized, above it the
    analytic ESF history x_ion = 1 - exp(-c zeta(z) <f_coll>(z)), c matched to the box at its highest z."""
    if not _TAU:
        from astropy import constants as const, units as u
        from astropy.cosmology import Planck18 as Cosmo
        z = np.linspace(0, 30, 3000)
        Y = 0.245
        f_he = Y / (4 * (1 - Y))
        n_h0 = (Cosmo.critical_density0 * Cosmo.Ob0 * (1 - Y) / const.m_p).to(u.m ** -3).value
        _TAU.update(z=z, f_he=f_he, kern=const.c.value * const.sigma_T.value * n_h0 * (1 + z) ** 2
                    / Cosmo.H(z).to(1 / u.s).value, fcoll={m: esf.mean_fcoll(z, m) for m in esf.M_MIN})
    z, f_he = _TAU['z'], _TAU['f_he']
    fc = _TAU['fcoll'][m_min]
    zeta = 10 ** log_zeta7 * ((1 + z) / 8) ** alpha
    zt = z_box[-1]
    f_top = 10 ** log_zeta7 * ((1 + zt) / 8) ** alpha * np.interp(zt, z, fc)
    c = -np.log(max(xhi_box[-1], 1e-6)) / f_top
    xe = np.where(z < z_box[0], 1.0, np.interp(z, z_box, 1 - xhi_box))
    hi = z > zt
    xe[hi] = 1 - np.exp(-c * zeta[hi] * fc[hi])
    ne = xe * (1 + f_he) + np.where(z < 3, f_he, 0.0)
    return esf._trapz(ne * _TAU['kern'], z)


def footprint_transform(x, y, k):
    """Rotation by k * 90 deg (k = 0..3), mirrored for k = 4..7."""
    if k >= 4:
        x = -x
    for _ in range(k % 4):
        x, y = -y, x
    return x, y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lc_dir', default=os.path.expanduser('~/Documents/Lya_bubbles/esf_lightcones'))
    ap.add_argument('--out_dir', default=os.path.expanduser('~/Documents/Lya_bubbles/esf_pencils'))
    ap.add_argument('--n_per', type=int, default=3000, help='mocks per lightcone realization')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--z_end_max', type=float, default=None,
                    help='keep only histories ending (x_HI = 0.05) at or below this z, e.g. 6.05 (Lya forest)')
    a = ap.parse_args()
    for sub in ('mocks', 'pool'):
        os.makedirs(os.path.join(a.out_dir, sub), exist_ok=True)

    g = np.load(os.path.join(a.lc_dir, 'geometry.npz'))
    xg, yg, zg, z_end_off = g['x_gal'], g['y_gal'], g['z_gal'], float(g['z_end_offset'])
    n_gal = len(xg)
    # sightline points (n_gal, N_LOS) along the LOS, as generate_bubble_mocks / bubble_prior_v2
    ell_pts = np.stack([np.linspace(z_end_off, zg[i], N_LOS) for i in range(n_gal)])
    amax = {m: alpha_max(m) for m in esf.M_MIN}
    print('[pencil] alpha_max per M_min:', {f'{m:.0e}': round(v, 2) for m, v in amax.items()}, flush=True)

    files = sorted(f for f in glob.glob(os.path.join(a.lc_dir, 'lc_*.npz')) if not f.endswith('.tmp.npz'))
    for r, path in enumerate(files):
        name = f'train_batch_{r:05d}.npz'
        out_m, out_p = os.path.join(a.out_dir, 'mocks', name), os.path.join(a.out_dir, 'pool', name)
        if os.path.exists(out_p):
            continue
        t0 = time.perf_counter()
        d = np.load(path)
        logF, qtab, ell, z_los, keep = d['logF'], d['qtab'], d['ell'], d['z_los'], d['keep']
        delta5 = d['delta5']
        dx = float(d['dx'])
        n_t = logF.shape[1]
        il = np.clip(np.round((ell_pts - ell[0]) / dx).astype(int), 0, len(ell) - 1)     # (n_gal, N_LOS)
        zcell = z_los[il]
        i7 = int(np.argmin(np.abs(z_los - 7.0)))
        il_gal = il[:, -1]
        q = np.linspace(0, 1, esf.N_QUANT)
        zk = z_los[keep]
        loc_pts = (zcell >= 6.8)                                 # "near the galaxies": z >= 6.8 sightline points

        rng = np.random.default_rng([a.seed, r])
        n = a.n_per
        jm = rng.integers(0, len(esf.M_MIN), n)
        x7 = rng.uniform(*X_HI_7_RANGE, n)
        alpha = np.array([rng.uniform(ALPHA_LO, amax[esf.M_MIN[j]]) for j in jm])
        rot = rng.integers(0, 8, n)
        off = rng.uniform(0, n_t * dx, (n, 2))
        masks = np.zeros((n, n_gal, N_LOS), np.uint8)
        lz7 = np.zeros(n)
        z_end = np.zeros(n)
        tau = np.zeros(n)
        xhi_loc = np.zeros(n)
        xhi_glob = np.zeros(n)
        dgal = np.zeros((n, n_gal), np.float32)
        x_ion = np.zeros(n)                                      # global ionized fraction at the field-mean z0
        for i in range(n):
            while True:                                          # redraw until z_end <= --z_end_max (if set)
                j = jm[i]
                lz7[i] = -np.interp(x7[i], q, qtab[j, i7])       # log10 zeta_7
                hist = esf.xhi_history(qtab[j], z_los, lz7[i], alpha[i])
                hk = hist[keep]
                x_ion[i] = 1 - np.interp(float(g['z0']), z_los, hist)
                below = np.flatnonzero(hk <= X_HI_END)
                # z_end: highest z with x_HI <= 0.05; if the box never gets there, extrapolate linearly below 5.3
                if len(below):
                    z_end[i] = zk[below].max()
                else:
                    s = np.polyfit(zk[:20], hk[:20], 1)
                    z_end[i] = (X_HI_END - s[1]) / s[0] if s[0] > 0 else 0.0
                if a.z_end_max is None or Z_END_MIN <= z_end[i] <= a.z_end_max:
                    break
                jm[i] = rng.integers(0, len(esf.M_MIN))
                x7[i] = rng.uniform(*X_HI_7_RANGE)
                alpha[i] = rng.uniform(ALPHA_LO, amax[esf.M_MIN[jm[i]]])
            tau[i] = tau_e(zk, np.maximum.accumulate(hk), lz7[i], alpha[i], esf.M_MIN[j])
            xr, yr = footprint_transform(xg, yg, rot[i])
            ix = (np.floor((xr + off[i, 0]) / dx).astype(int)) % n_t
            iy = (np.floor((yr + off[i, 1]) / dx).astype(int)) % n_t
            f = logF[j][ix[:, None], iy[:, None], il].astype(np.float32)          # (n_gal, N_LOS)
            ion = f + alpha[i] * np.log10((1 + zcell) / 8) >= -lz7[i]
            masks[i] = ion
            xhi_loc[i] = 1 - ion[loc_pts].mean()
            xhi_glob[i] = np.mean(hist[il][loc_pts])
            dgal[i] = delta5[ix, iy, il_gal]
        np.savez(out_m, masks=masks, x_ion=x_ion, x_hi_7=x7, z_end=z_end, tau_e=tau, alpha=alpha, log_zeta7=lz7,
                 m_min=np.array(esf.M_MIN)[jm], realization=np.full(n, r), rot=rot, offset=off,
                 xhi_local=xhi_loc, xhi_global_local=xhi_glob, delta5_gal=dgal, realized=masks.mean(axis=(1, 2)))
        theta = (1 - masks[:, :, ::-1]).reshape(n, -1).astype(np.uint8)
        np.savez(out_p, theta=theta, n_gal=n_gal, n_los=N_LOS)
        print(f"[pencil] {os.path.basename(path)}: {n} mocks in {time.perf_counter() - t0:.0f} s, "
              f"mean neutral {1 - masks.mean():.3f}, local-global x_HI(z>6.8) scatter "
              f"{np.std(xhi_loc - xhi_glob):.3f}", flush=True)


if __name__ == '__main__':
    main()
