"""
calibrate_bsd_lu24.py -- tie the bubble-size distribution of the Boolean bubble prior to the ionized fraction,
using the 21cmFAST bubble-size measurements of Lu et al. 2024 (MNRAS 528, 4872; arXiv:2304.11192).

WHAT LU+24 MEASURE
------------------
In (1.6 Gpc)^3 21cmFAST boxes at z = 7, 8, 9 and x_HI = 0.1 ... 0.9, for 50 000 random positions: the distance
from the position along a random direction to the first neutral (or partially ionized) cell -- the
"mean-free-path" (MFP) bubble size (Mesinger & Furlanetto 2007). R = 0 if the start cell itself is not ionized.
Two reionization models bracket the plausible range at fixed x_HI:
    gradual ("fg"): driven by faint galaxies, M_min = 5e8 Msun  -> many small bubbles
    rapid   ("bg"): driven by bright galaxies, M_min = 1e11 Msun -> fewer, larger bubbles
Data: https://github.com/ting-yi-lu/bubble_size_overdensities_paper (files "*_rall_z7": random positions, z = 7).

WHY A CALIBRATION, NOT A DIRECT COPY
------------------------------------
Our prior (generate_bubble_mocks.py) is a Boolean model: spheres with random radii, centres Poisson with density
lambda = -ln(1 - x_ion) / <V>. Ionized REGIONS are unions of overlapping spheres, so their MFP sizes exceed the sphere
radii, increasingly so as x_ion grows. We therefore measure the MFP distribution OF THE BOOLEAN MODEL with Lu+24's
method (random ionized start points, isotropic directions, distance to the first point outside all spheres) and
choose the sphere-radius distribution, at each x_ion, so that the two MFP distributions agree.

RADIUS DISTRIBUTION AND THE TWO FITTED NUMBERS
----------------------------------------------
Radii are log-normal, ln r ~ N(ln(s * R0), sigma^2), truncated to [R0 s e^-3.5sigma, R0 s e^+3.5sigma]
(R0 = 10 cMpc is only a unit). Two exact properties make the fit cheap:
  * SCALE: stretching every radius by s (and thinning the centres by s^3, which keeps x_ion) stretches every MFP by
    exactly s. So the MFP distribution at any s is the one at s = 1 times s, and s follows from the medians:
        s(x_ion) = median_Lu24(x_ion) / median_Boolean(x_ion, sigma; s = 1).
  * SHAPE: the quantile RATIOS q16/q50 and q84/q50 of the MFP do not depend on s, only on (x_ion, sigma).
For each x_ion we compute the Boolean MFP at s = 1 on a grid of sigma, pick sigma whose ratios match Lu+24's best
(least squares in log), and then s from the medians. This is done for the gradual and the rapid model.

OUTPUT
------
bsd_lu24_calibration.npz: for each model, x_ion grid (= 1 - x_HI of Lu+24), sigma(x_ion), s(x_ion), and the
16/50/84% MFP quantiles of Lu+24 and of the calibrated Boolean model (a check of the match).

Usage: python calibrate_bsd_lu24.py --lu24_dir ~/Downloads/lu24_bubble_lists [--n_rays 2000] [--test]
"""
import os
import re
import glob
import time
import argparse

import numpy as np

R0 = 10.0              # radius unit [cMpc]: the median sphere radius at s = 1
N_SIG_TRUNC = 3.5      # log-normal truncated at +- 3.5 sigma
L_RAY = 800.0          # maximum MFP measured [cMpc]; Lu+24's 84th percentile is < 530 cMpc at all x_HI
SIGMA_GRID = np.round(np.arange(0.1, 1.01, 0.15), 2)   # 0.10, 0.25, ..., 1.00
X_GRID = np.round(np.arange(0.1, 0.91, 0.1), 2)       # common x_ion grid; Lu+24's x_ion are within +-0.015 of it
S_GRID = 0.3           # the (x_ion, sigma) grid is computed at s = 0.3 (far below the L_RAY cap) and divided by it
# (np.round removes floating-point noise from arange, e.g. 0.30000000000000004, so the grid values give clean
#  cache keys like '0.30_0.40')


def load_lu24(lu24_dir, model):
    """{x_HI: array of R > 0 [cMpc]} from Lu+24's random-position lists at z = 7."""
    # One file per neutral fraction in <lu24_dir>/<model>_rall_z7/, with x_HI in the file name ("..._nf0.5_...").
    # Each file is a CSV with a header line; column 1 is the MFP size R of one random position.
    out = {}
    for f in sorted(glob.glob(os.path.join(lu24_dir, f'{model}_rall_z7', '*'))):
        x_hi = float(re.search(r'_nf([0-9.]+)_', os.path.basename(f)).group(1))
        R = np.loadtxt(f, delimiter=',', skiprows=1, usecols=1)
        out[x_hi] = R[R > 0]          # R = 0: the start position was not ionized; we compare ionized starts only
    return out


class LogNormalRadii:
    """ln r ~ N(ln(s R0), sigma^2) truncated at +- N_SIG_TRUNC sigma (sampled by rejection-free inverse CDF)."""

    def __init__(self, s, sigma):
        self.s, self.sigma = s, sigma
        self.r_max = s * R0 * np.exp(N_SIG_TRUNC * sigma)   # largest possible radius

    def draw(self, n, rng):
        # uniform u in [Phi(-c), Phi(c)] -> Z = Phi^-1(u) is a standard normal restricted to |Z| < c; r = s R0 e^{sigma Z}
        from scipy.special import ndtr, ndtri
        lo, hi = ndtr(-N_SIG_TRUNC), ndtr(N_SIG_TRUNC)
        z = ndtri(lo + rng.uniform(size=n) * (hi - lo))
        return self.s * R0 * np.exp(self.sigma * z)

    def mean_volume(self, rng, n=400_000):
        # <4/3 pi r^3> by Monte Carlo (~1% accurate at sigma = 1). bubble_prior_v2.BSD.mean_volume is the exact,
        # analytic version of the same quantity.
        return np.mean(4 / 3 * np.pi * self.draw(n, rng) ** 3)


def boolean_mfp(x_ion, radii_dist, n_rays, rng):
    """MFP sizes of the Boolean model: random ionized start points, isotropic directions, distance along the ray
    to the first point not inside any sphere (capped at L_RAY).

    The model is statistically isotropic and homogeneous, so every ray can start at the origin and point along +z,
    each in its own independent realization. Only spheres whose centres lie within r_max of the ray segment can
    touch it, so each realization is generated only inside the cylinder rho < r_max, -r_max < z < L_RAY + r_max
    (Poisson count with the same density lambda). A realization whose origin is not inside a sphere is discarded:
    Lu+24's R > 0 sample starts at ionized points."""
    r_max = radii_dist.r_max
    # centre density giving covered fraction x_ion: neutral fraction exp(-lambda <V>) = 1 - x_ion
    lam = -np.log(1 - x_ion) / radii_dist.mean_volume(rng)
    vol = np.pi * r_max ** 2 * (L_RAY + 2 * r_max)          # volume of the cylinder that can hold relevant centres
    out = np.empty(n_rays)
    k = 0                                                   # number of accepted rays so far
    while k < n_rays:
        # --- one independent realization of the spheres inside the cylinder ---
        n_b = rng.poisson(lam * vol)
        rho = r_max * np.sqrt(rng.uniform(size=n_b))           # uniform in the disc
        zc = rng.uniform(-r_max, L_RAY + r_max, size=n_b)
        r = radii_dist.draw(n_b, rng)
        # --- the interval [t0, t1] of the z axis (the ray) that each sphere covers ---
        # a sphere whose centre is rho from the axis cuts it only if r > rho, in a chord of half-length sqrt(r^2 - rho^2)
        h2 = r ** 2 - rho ** 2                                 # squared half-chord on the z axis
        hit = h2 > 0
        t0, t1 = zc[hit] - np.sqrt(h2[hit]), zc[hit] + np.sqrt(h2[hit])
        # --- keep only realizations whose start point (the origin) is ionized ---
        if not np.any((t0 <= 0) & (t1 >= 0)):
            continue                                           # origin not ionized
        # --- merge the intervals ahead of the origin and find the first gap ---
        keep = t1 > 0                                          # intervals ending behind the origin are irrelevant
        o = np.argsort(t0[keep])                               # sort by start; the first one contains the origin
        a_s, b_s = t0[keep][o], t1[keep][o]
        reach = np.maximum.accumulate(b_s)                     # furthest point covered by the first j chords
        # Interval j+1 starting beyond reach[j] leaves an uncovered gap; everything before it is one connected
        # ionized stretch from the origin to reach[j]. Strict '>' joins intervals that merely touch.
        gap = np.nonzero(a_s[1:] > reach[:-1])[0]              # first chord that starts beyond the covered part
        cur = reach[gap[0]] if gap.size else reach[-1]
        # spheres were only generated up to L_RAY + r_max, so anything beyond L_RAY is not trustworthy: cap it
        out[k] = min(cur, L_RAY)
        k += 1
    return out


def quantiles(a):
    # the three numbers compared between Lu+24 and the Boolean model
    return np.percentile(a, [16, 50, 84])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lu24_dir', required=True)
    ap.add_argument('--n_rays', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--test', action='store_true', help='time one Boolean MFP measurement per sigma and exit')
    ap.add_argument('--output', default='bsd_lu24_calibration.npz')
    ap.add_argument('--cache', default='bsd_lu24_grid_cache.npz', help='resumable cache of the (x_ion, sigma) grid')
    ap.add_argument('--sigma_fixed', type=float, default=None,
                    help='use this log-width at every x_ion (must be on SIGMA_GRID) and fit only s(x_ion). The '
                         'Boolean MFP distribution cannot become as broad as Lu+24 above sigma ~ 1 (its q84/q50 '
                         'saturates near 2.7 vs their 2.85-3.45), and letting sigma jump between grid values makes '
                         's(x_ion) jagged; a fixed sigma = 1.0 gives a smooth s(x_ion) matching median and q16.')
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    if args.test:
        # timing / sanity run: a few MFP measurements at the most expensive x_ion, then exit
        for sigma in (0.2, 0.6, 1.0):
            t = time.perf_counter()
            q = quantiles(boolean_mfp(0.9, LogNormalRadii(1.0, sigma), args.n_rays, rng))
            print(f'[test] x_ion 0.9, sigma {sigma}: MFP 16/50/84% {q[0]:.1f}/{q[1]:.1f}/{q[2]:.1f} cMpc '
                  f'({time.perf_counter() - t:.1f}s)', flush=True)
        return

    # Lu+24's MFP samples: lu['fg'] (gradual) and lu['bg'] (rapid), each {x_HI: array of R > 0}
    lu = {m: load_lu24(args.lu24_dir, m) for m in ('fg', 'bg')}
    # Boolean MFP quantiles at s = 1 on the (x_ion, sigma) grid, at every x_ion where Lu+24 have data. Computed at
    # s = S_GRID and divided by S_GRID (exact by the scaling property), so that no MFP reaches the L_RAY cap.
    # Each grid point is cached in --cache (resumable). Lu+24's x_ion are then matched by linear interpolation of
    # the Boolean quantiles in x_ion between grid points.
    # cache: {'<x_ion>_<sigma>': [q16, q50, q84] at s = 1}; saved after every point so a killed run loses nothing
    cache = dict(np.load(args.cache)) if os.path.exists(args.cache) else {}
    for x_ion in X_GRID:
        for sigma in SIGMA_GRID:
            key = f'{x_ion:.2f}_{sigma:.2f}'
            if key not in cache:
                m = boolean_mfp(x_ion, LogNormalRadii(S_GRID, sigma), args.n_rays, rng)
                if np.percentile(m, 84) > 0.9 * L_RAY:
                    print(f'[grid] WARNING x_ion {x_ion}, sigma {sigma}: MFP near the L_RAY cap', flush=True)
                cache[key] = quantiles(m) / S_GRID
                np.savez(args.cache, **cache)
        print(f'[grid] x_ion {x_ion:.2f} done', flush=True)

    def qb_at(x_ion, sigma):
        # Boolean quantiles (s = 1) at any x_ion for a grid sigma: linear interpolation along the X_GRID column
        grid = np.array([cache[f'{x:.2f}_{sigma:.2f}'] for x in X_GRID])       # (n_x, 3)
        return np.array([np.interp(x_ion, X_GRID, grid[:, j]) for j in range(3)])

    # qb[(x_ion, sigma)]: Boolean quantiles at every Lu+24 x_ion (both models) and every grid sigma
    qb = {}
    for model in lu:
        for x_hi in lu[model]:
            for sigma in SIGMA_GRID:
                qb[round(1 - x_hi, 4), sigma] = qb_at(round(1 - x_hi, 4), sigma)

    res = {}
    for model in ('fg', 'bg'):
        rows = []
        for x_hi, R in sorted(lu[model].items()):
            x_ion = round(1 - x_hi, 4)
            ql = quantiles(R)                                   # Lu+24's q16, q50, q84 at this x_ion
            # SHAPE: the log quantile ratios ln(q16/q50), ln(q84/q50) do not depend on s; pick the grid sigma whose
            # Boolean ratios are closest (least squares). Skipped when --sigma_fixed is given.
            target = np.log([ql[0] / ql[1], ql[2] / ql[1]])
            cost = [np.sum((np.log([qb[x_ion, sg][0] / qb[x_ion, sg][1], qb[x_ion, sg][2] / qb[x_ion, sg][1]])
                            - target) ** 2) for sg in SIGMA_GRID]
            sigma = SIGMA_GRID[int(np.argmin(cost))] if args.sigma_fixed is None else args.sigma_fixed
            # SCALE: Boolean MFPs scale exactly with s, so s = Lu+24 median / Boolean median at s = 1
            s = ql[1] / qb[x_ion, sigma][1]
            # check: rerun the Boolean MFP at the fitted (s, sigma) and compare its quantiles with Lu+24's
            check = quantiles(boolean_mfp(x_ion, LogNormalRadii(s, sigma), args.n_rays, rng))
            rows.append((x_ion, sigma, s, *ql, *check))
            print(f'[{model}] x_ion {x_ion:.2f}: sigma {sigma:.1f}, s {s:.3f} (median sphere radius '
                  f'{s * R0:.1f} cMpc); MFP 16/50/84%: Lu+24 {ql[0]:.1f}/{ql[1]:.1f}/{ql[2]:.1f}, '
                  f'Boolean {check[0]:.1f}/{check[1]:.1f}/{check[2]:.1f} cMpc', flush=True)
        res[model] = np.array(sorted(rows))                 # sorted by x_ion (first column), as np.interp needs
    np.savez(args.output, gradual=res['fg'], rapid=res['bg'],
             columns=np.array(['x_ion', 'sigma', 's', 'lu_q16', 'lu_q50', 'lu_q84',
                               'bool_q16', 'bool_q50', 'bool_q84']),
             R0=R0, n_sig_trunc=N_SIG_TRUNC, l_ray=L_RAY)
    print(f'saved {args.output}')


if __name__ == '__main__':
    main()
