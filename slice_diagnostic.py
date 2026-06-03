"""
1-D likelihood slice diagnostic for the Lyman-alpha bubble inference.

Answers the key question: does the likelihood peak at the true parameter values
regardless of noise level, or does the peak shift (→ the likelihood is biased)?

Usage
-----
python slice_diagnostic.py --n_gal 70 --seed 7 --n_pts 41
python slice_diagnostic.py --n_gal 70 --seed 7 --noise 5e-19  # single noise level
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.special import logsumexp, gammaln
from astropy.cosmology import Planck18 as Cosmo
from astropy import constants as const
import astropy.units as u

from venv.speed_up import get_content, calculate_taus_post_batched
from venv.galaxy_prop import get_mock_data, get_js, tau_CGM, p_EW
from venv.helpers import full_res_flux, perturb_flux, z_at_proper_distance, I

TRUE_MU      = np.array([0.0, 0.0, 0.0, 10.0])
PRIOR_LO     = np.array([-10.0, -10.0, -10.0,  1.0])
PRIOR_HI     = np.array([ 10.0,  10.0,  10.0, 20.0])
PARAM_NAMES  = ['x_bub', 'y_bub', 'z_bub', 'r_bub']
N_BINS       = 11
N_INSIDE_TAU = 200
N_ITER_BUB   = 1
NU_STUDENT   = 3.0
MAIN_DIR     = '/groups/astro/ivannik/programs/Lyman-alpha-bubbles/'
wave_em      = np.linspace(1214, 1225., 100) * u.Angstrom
wave_Lya     = 1215.67 * u.Angstrom

NOISE_LEVELS = [5e-19, 2e-19, 1e-19, 5e-20, 2e-20]


def build_state(n_gal: int, noise: float, seed: int):
    """Build the precomputed arrays for (n_gal, noise, seed). Returns a namespace."""
    import types
    np.random.seed(seed)
    Muv_mock = np.ones(n_gal) * -18.5
    beta     = -2.0 * np.ones(n_gal)

    tau_mock, x_gal, y_gal, z_gal, *_ = get_mock_data(
        n_gal=n_gal, z_start=7.5, r_bubble=10, dist=15,
    )
    redshifts = np.array([
        z_at_proper_distance(-z_gal[i] / (1 + 7.5) * u.Mpc, 7.5)
        for i in range(n_gal)
    ])
    one_J = get_js(z=7.5, muv=Muv_mock, n_iter=n_gal)
    area_factor = np.array([
        np.trapz(one_J[0][i] * tau_CGM(Muv_mock[i], main_dir=MAIN_DIR), wave_em.value)
        / np.trapz(one_J[0][i], wave_em.value)
        for i in range(n_gal)
    ])
    _, la_e = p_EW(Muv_mock.flatten(), beta.flatten())
    la_e = la_e.reshape(np.shape(Muv_mock)) / area_factor

    continuum = (
        la_e[:, np.newaxis] * one_J[0][:n_gal] * np.exp(-tau_mock)
        * tau_CGM(Muv_mock, main_dir=MAIN_DIR)
        / (4 * np.pi * Cosmo.luminosity_distance(7.5).to(u.cm).value ** 2)
    )
    full_flux = full_res_flux(continuum, 7.5)
    full_flux += np.random.normal(0, noise, np.shape(full_flux))
    flux_noise_mock = perturb_flux(full_flux, N_BINS)

    cont_filled = get_content(
        Muv_mock.flatten(), redshifts,
        x_gal, y_gal, z_gal,
        n_iter_bub=N_ITER_BUB, n_inside_tau=N_INSIDE_TAU,
        include_muv_unc=False, fwhm_true=False,
        redshift=7.5, xh_unc=True, high_prob_emit=False,
        EW_fixed=False, cache=None, AH22_model=False,
        main_dir=MAIN_DIR, cache_dir=None, gauss_distr=False,
    )
    R_H      = np.array([(const.c / Cosmo.H(redshifts[i])).to(u.Mpc).value for i in range(n_gal)])
    tau_cgm  = np.array([tau_CGM(Muv_mock[i], main_dir=MAIN_DIR) for i in range(n_gal)])
    j_s      = np.array([cont_filled.j_s_full[i] for i in range(n_gal)])
    raw_af   = np.array([
        np.trapz(j_s[i] * tau_cgm[i], wave_em.value, axis=1) /
        np.trapz(j_s[i], wave_em.value, axis=1)
        for i in range(n_gal)
    ])
    af = np.where(raw_af < 1e-20, 1e-5, raw_af)

    r_alpha_val = 6.25e8 / (4 * np.pi * (const.c / wave_Lya).to(u.Hz).value)
    tau_gp      = 7.16e5 * ((1 + redshifts) / 10) ** 1.5
    tau_wv_pref = tau_gp * r_alpha_val / np.pi * 0.65
    z_wv        = wave_em.value[np.newaxis, :] / 1216 * (1 + redshifts[:, np.newaxis]) - 1
    I_z_end     = I((1 + 5.3) / (1 + z_wv))
    ooz         = 1215.67 / (wave_em.value[np.newaxis, :] * (1 + redshifts[:, np.newaxis]))
    red_up      = np.array([cont_filled.first_bubble_encounter_redshift_up_full[i] for i in range(n_gal)])
    I_red_up    = I((1 + red_up[:, :, np.newaxis]) * ooz[:, np.newaxis, :])

    la_flux = np.array([cont_filled.la_flux_out_full[i] for i in range(n_gal)])
    com_fact = np.array([cont_filled.com_fact[i] for i in range(n_gal)])
    tau_prec = np.array([cont_filled.tau_prec_full[i] for i in range(n_gal)])
    z_up     = np.array([cont_filled.first_bubble_encounter_coord_z_up_full[i] for i in range(n_gal)])
    z_lo     = np.array([cont_filled.first_bubble_encounter_coord_z_lo_full[i] for i in range(n_gal)])
    red_lo   = np.array([cont_filled.first_bubble_encounter_redshift_lo_full[i] for i in range(n_gal)])

    base_cont = (
        (la_flux / af)[:, :, np.newaxis]
        * j_s * np.exp(-tau_prec)
        * tau_cgm[:, np.newaxis, :]
        * com_fact[:, np.newaxis, np.newaxis]
    )

    _spec_res      = wave_Lya.value * (1 + 7.5) / 2700
    _full_bins     = np.arange(wave_em.value[0]*(1+7.5), wave_em.value[-1]*(1+7.5), _spec_res)
    _max_bins_full = len(_full_bins)
    _wave_em_dig   = np.digitize(wave_em.value * (1+7.5), _full_bins)
    _trapz_weights = np.zeros((100, _max_bins_full))
    for _i in range(_max_bins_full):
        _idx = np.where(_wave_em_dig == _i + 1)[0]
        if len(_idx) < 2:
            continue
        _x = wave_em.value[_idx]
        _w = np.empty(len(_idx))
        _w[0]  = (_x[1] - _x[0]) / 2
        _w[-1] = (_x[-1] - _x[-2]) / 2
        if len(_idx) > 2:
            _w[1:-1] = (_x[2:] - _x[:-2]) / 2
        _trapz_weights[_idx, _i] = _w
    _bins_rebin        = np.linspace(wave_em.value[0]*(1+7.5), wave_em.value[-1]*(1+7.5), N_BINS+1)
    _wave_em_dig_rebin = np.digitize(_full_bins, _bins_rebin)
    _rebin_matrix      = (_wave_em_dig_rebin[:, np.newaxis] == np.arange(1, N_BINS+1)[np.newaxis, :]).astype(float)
    direct_matrix      = _trapz_weights @ _rebin_matrix
    noise_per_bin      = np.sqrt(_rebin_matrix.sum(axis=0)) * noise

    tau_post_out = calculate_taus_post_batched(
        redshifts, redshifts,
        z_up.copy(), red_up,
        z_lo.copy(), red_lo,
        z_per_gal=z_wv, tau_wv_pref_per_gal=tau_wv_pref,
        I_z_end_per_gal=I_z_end, I_red_up_all=I_red_up,
    )
    tau_out = tau_post_out.copy()
    bad_out = np.any(tau_out[:, :, 30:] - tau_out[:, :, 29:-1] > 0, axis=2)
    for _g in np.where(np.any(bad_out, axis=1))[0]:
        _ratio = (1 + redshifts[_g]) / (1 + z_wv[_g])
        tau_out[_g, bad_out[_g]] = np.clip(
            tau_wv_pref[_g] * _ratio**1.5 * (I(_ratio) - I_z_end[_g]), 0, np.inf,
        )
    tau_out[tau_out < 0] = np.inf
    tau_out = np.nan_to_num(tau_out, nan=np.inf)
    base_cont_outside = base_cont * np.exp(-tau_out)
    del tau_post_out, tau_out, bad_out

    flux_outside = (
        base_cont_outside.reshape(n_gal * N_INSIDE_TAU, 100) @ direct_matrix
    ).reshape(n_gal, N_INSIDE_TAU, N_BINS)

    s = types.SimpleNamespace(
        x_gal_mock=x_gal, y_gal_mock=y_gal, z_gal_mock=z_gal,
        redshifts=redshifts, R_H=R_H,
        tau_prec=tau_prec, z_up=z_up, red_up=red_up,
        z_lo=z_lo, red_lo=red_lo,
        z_wv=z_wv, tau_wv_pref=tau_wv_pref,
        I_z_end=I_z_end, I_red_up=I_red_up,
        base_cont=base_cont, base_cont_outside=base_cont_outside,
        flux_outside=flux_outside, direct_matrix=direct_matrix,
        noise_per_bin=noise_per_bin, obs_flux=flux_noise_mock,
    )
    return s


def log_likelihood(s, xb, yb, zb, rb):
    dx = s.x_gal_mock - xb
    dy = s.y_gal_mock - yb
    dz = s.z_gal_mock - zb
    inside = dx**2 + dy**2 + dz**2 < rb**2
    dist_arr = np.where(inside, dz + np.sqrt(np.where(inside, rb**2 - dx**2 - dy**2, 0.0)), 0.0)
    z_end_bub_arr = s.redshifts - np.where(inside, dist_arr / s.R_H, 0.0)
    inside_gals = np.where(inside)[0]

    if len(inside_gals) == 0:
        continuum_all = s.base_cont_outside
    else:
        tau_post_in = calculate_taus_post_batched(
            s.redshifts[inside_gals], z_end_bub_arr[inside_gals],
            s.z_up[inside_gals].copy(), s.red_up[inside_gals],
            s.z_lo[inside_gals].copy(), s.red_lo[inside_gals],
            z_per_gal=s.z_wv[inside_gals],
            tau_wv_pref_per_gal=s.tau_wv_pref[inside_gals],
            I_z_end_per_gal=s.I_z_end[inside_gals],
            I_red_up_all=s.I_red_up[inside_gals],
        )
        tau_now = s.tau_prec[inside_gals] + tau_post_in
        bad = np.any(tau_now[:, :, 30:] - tau_now[:, :, 29:-1] > 0, axis=2)
        if np.any(bad):
            for _ii, g in enumerate(inside_gals):
                if not np.any(bad[_ii]):
                    continue
                ratio_g = (1 + z_end_bub_arr[g]) / (1 + s.z_wv[g])
                tau_now[_ii, bad[_ii]] = np.clip(
                    s.tau_wv_pref[g] * ratio_g**1.5 * (I(ratio_g) - s.I_z_end[g]),
                    0, np.inf,
                )
        tau_now[tau_now < 0] = np.inf
        tau_now = np.nan_to_num(tau_now, nan=np.inf)
        continuum_all = s.base_cont_outside.copy()
        continuum_all[inside_gals] = s.base_cont[inside_gals] * np.exp(-tau_now)

    predicted = s.flux_outside.copy()
    if len(inside_gals) > 0:
        flat_in = continuum_all[inside_gals].reshape(len(inside_gals) * N_INSIDE_TAU, 100)
        predicted[inside_gals] = (flat_in @ s.direct_matrix).reshape(
            len(inside_gals), N_INSIDE_TAU, N_BINS
        )
    np.nan_to_num(predicted, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    diffs  = s.obs_flux[:, np.newaxis, :] - predicted
    _log_norm = (gammaln((NU_STUDENT + 1) / 2) - gammaln(NU_STUDENT / 2)
                 - 0.5 * np.log(np.pi * NU_STUDENT) - np.log(s.noise_per_bin))
    log_p = (
        logsumexp(
            -(NU_STUDENT + 1) / 2 * np.log1p((diffs / s.noise_per_bin) ** 2 / NU_STUDENT),
            axis=1,
        )
        - np.log(N_INSIDE_TAU)
        + _log_norm
    )
    return float(log_p.sum())


def run_slices(n_gal: int, seed: int, noise_levels, n_pts: int, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    r_grid = np.linspace(1, 20, n_pts)
    # Also scan each position param
    grids = [
        np.linspace(PRIOR_LO[i], PRIOR_HI[i], n_pts) for i in range(3)
    ] + [r_grid]

    colors = cm.plasma(np.linspace(0.1, 0.9, len(noise_levels)))
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    all_peaks = {}
    for ni, noise in enumerate(noise_levels):
        print(f"\nBuilding state: n_gal={n_gal}, noise={noise:.1e}, seed={seed}...", flush=True)
        s = build_state(n_gal, noise, seed)
        print(f"  State ready. Computing 4 slices × {n_pts} points...", flush=True)

        peaks = []
        for pi, (ax, pname, pgrid) in enumerate(zip(axes, PARAM_NAMES, grids)):
            lls = []
            for v in pgrid:
                p = TRUE_MU.copy().astype(float)
                p[pi] = v
                lls.append(log_likelihood(s, *p))
            lls = np.array(lls)
            lls -= lls.max()   # normalise so peak is at 0
            ax.plot(pgrid, lls, lw=1.5, color=colors[ni], label=f'{noise:.0e}')
            peak_v = pgrid[np.argmax(lls)]
            peaks.append(peak_v)
            print(f"  {pname}: peak at {peak_v:.2f}  (truth={TRUE_MU[pi]:.1f})", flush=True)

        all_peaks[noise] = peaks

    for pi, (ax, pname) in enumerate(zip(axes, PARAM_NAMES)):
        ax.axvline(TRUE_MU[pi], color='red', ls='--', lw=1.5, label='truth')
        ax.set_xlabel(pname, fontsize=12)
        ax.set_title(pname)
    axes[0].set_ylabel('Δ log L  (peak = 0)', fontsize=11)
    axes[-1].legend(title='noise', fontsize=7, loc='lower left')
    fig.suptitle(
        f"1-D likelihood slices through truth  |  n_gal={n_gal}, seed={seed}, "
        f"N_INSIDE_TAU={N_INSIDE_TAU}, ν={NU_STUDENT}",
        fontsize=11
    )
    fig.tight_layout()
    fname = os.path.join(outdir, f'slices_ngal{n_gal:03d}_seed{seed:02d}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"\nSaved {fname}", flush=True)

    # Print summary table
    print("\n── Peak positions (truth = [0, 0, 0, 10]) ────────────────────────────")
    header = f"{'noise':>10s}  " + "  ".join(f"{p:>8s}" for p in PARAM_NAMES)
    print(header)
    for noise, peaks in sorted(all_peaks.items()):
        row = f"{noise:>10.1e}  " + "  ".join(f"{v:>8.2f}" for v in peaks)
        print(row)

    return all_peaks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gal',  type=int,   default=70)
    parser.add_argument('--seed',   type=int,   default=7)
    parser.add_argument('--n_pts',  type=int,   default=41,  help='grid points per slice')
    parser.add_argument('--noise',  type=float, default=None,
                        help='single noise level; if omitted, runs all NOISE_LEVELS')
    parser.add_argument('--outdir', type=str,   default='slice_results')
    args = parser.parse_args()

    noise_levels = [args.noise] if args.noise is not None else NOISE_LEVELS
    run_slices(args.n_gal, args.seed, noise_levels, args.n_pts, args.outdir)