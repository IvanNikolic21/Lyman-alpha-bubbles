"""
P-P (coverage) test for the Lyman-alpha bubble inference.

For each seed: generate fresh mock galaxies with the same true bubble
(0, 0, 0, 10), run nested sampling, record the posterior quantile at which
each true parameter falls.  Under correct, unbiased inference these quantiles
are uniform on [0, 1].

Usage
-----
# Run 20 seeds sequentially, save per-seed files, plot at the end:
python pp_test.py --n_seeds 20 --output_dir pp_results/

# Single-seed run (useful as a cluster array job):
python pp_test.py --seed 3 --output_dir pp_results/

# Make the P-P plot from already-saved results without running new inference:
python pp_test.py --plot_only --output_dir pp_results/
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import argparse
import glob
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import dynesty
from dynesty.utils import resample_equal
from scipy.special import logsumexp
from scipy.stats import binom, percentileofscore
from astropy.cosmology import Planck18 as Cosmo
from astropy import constants as const
import astropy.units as u

from venv.speed_up import get_content, calculate_taus_post_batched
from venv.galaxy_prop import get_mock_data, get_js, tau_CGM, p_EW
from venv.helpers import full_res_flux, perturb_flux, z_at_proper_distance, I

# ── Fixed settings (must match dynesty_tutorial.py) ───────────────────────────
TRUE_MU      = np.array([0.0, 0.0, 0.0, 10.0])
PRIOR_LO     = np.array([-10.0, -10.0, -10.0,  1.0])
PRIOR_HI     = np.array([ 10.0,  10.0,  10.0, 20.0])
NDIM         = 4
N_DATA       = 50
N_INSIDE_TAU = 50
N_ITER_BUB   = 1
N_BINS       = 11
NOISE        = 5e-20
ADDITIVE     = 1e-18
BW_KDE       = 0.12
N_WORKERS    = 24
NLIVE_PP     = 100   # reduced from 300 for speed; increase if posteriors look jagged
DLOGZ_PP     = 1.0   # looser than production 0.5
MAIN_DIR     = '/groups/astro/ivannik/programs/Lyman-alpha-bubbles/'

wave_em  = np.linspace(1214, 1225., 100) * u.Angstrom
wave_Lya = 1215.67 * u.Angstrom
PARAM_NAMES = ['x_bub', 'y_bub', 'z_bub', 'r_bub']


def run_single_inference(seed: int, n_workers: int = N_WORKERS) -> np.ndarray:
    """
    Generate fresh mock data with `seed`, run nested sampling, return the
    posterior quantile (0–1) at which each true parameter falls.
    Returns shape (NDIM,).
    """
    np.random.seed(seed)
    print(f"\n[seed {seed}] Generating mock data...", flush=True)

    Muv_mock = np.ones(N_DATA) * -18.5
    beta     = -2.0 * np.ones(N_DATA)

    tau_mock, x_gal_mock, y_gal_mock, z_gal_mock, *_ = get_mock_data(
        n_gal=N_DATA, z_start=7.5, r_bubble=10, dist=15,
    )
    redshifts_of_mocks = np.array([
        z_at_proper_distance(-z_gal_mock[i] / (1 + 7.5) * u.Mpc, 7.5)
        for i in range(N_DATA)
    ])

    one_J = get_js(z=7.5, muv=Muv_mock, n_iter=N_DATA)
    area_factor = np.array([
        np.trapz(one_J[0][i] * tau_CGM(Muv_mock[i], main_dir=MAIN_DIR), wave_em.value)
        / np.trapz(one_J[0][i], wave_em.value)
        for i in range(N_DATA)
    ])
    _, la_e = p_EW(Muv_mock.flatten(), beta.flatten())
    la_e = la_e.reshape(np.shape(Muv_mock)) / area_factor

    continuum = (
        la_e[:, np.newaxis] * one_J[0][:N_DATA] * np.exp(-tau_mock)
        * tau_CGM(Muv_mock, main_dir=MAIN_DIR)
        / (4 * np.pi * Cosmo.luminosity_distance(7.5).to(u.cm).value ** 2)
    )
    full_flux = full_res_flux(continuum, 7.5)
    full_flux += np.random.normal(0, NOISE, np.shape(full_flux))
    flux_noise_mock = perturb_flux(full_flux, N_BINS)   # (N_DATA, N_BINS)

    print(f"[seed {seed}] Building precomputed arrays...", flush=True)
    cont_filled = get_content(
        Muv_mock.flatten(), redshifts_of_mocks,
        x_gal_mock, y_gal_mock, z_gal_mock,
        n_iter_bub=N_ITER_BUB, n_inside_tau=N_INSIDE_TAU,
        include_muv_unc=False, fwhm_true=False,
        redshift=7.5, xh_unc=True, high_prob_emit=False,
        EW_fixed=False, cache=None, AH22_model=False,
        main_dir=MAIN_DIR, cache_dir=None, gauss_distr=False,
    )

    # Per-galaxy fixed quantities
    _R_H_per_gal = np.array([
        (const.c / Cosmo.H(redshifts_of_mocks[i])).to(u.Mpc).value
        for i in range(N_DATA)
    ])
    _tau_cgm_per_gal     = np.array([tau_CGM(Muv_mock[i], main_dir=MAIN_DIR) for i in range(N_DATA)])
    _j_s_per_gal         = np.array([cont_filled.j_s_full[i] for i in range(N_DATA)])
    _raw_af              = np.array([
        np.trapz(_j_s_per_gal[i] * _tau_cgm_per_gal[i], wave_em.value, axis=1) /
        np.trapz(_j_s_per_gal[i], wave_em.value, axis=1)
        for i in range(N_DATA)
    ])
    _area_factor_per_gal = np.where(_raw_af < 1e-20, 1e-5, _raw_af)
    _obs_mag_per_gal     = 5 * np.log10(10**18.7 * (ADDITIVE + 2 * flux_noise_mock))

    _r_alpha_val         = 6.25e8 / (4 * np.pi * (const.c / wave_Lya).to(u.Hz).value)
    _tau_gp_per_gal      = 7.16e5 * ((1 + redshifts_of_mocks) / 10) ** 1.5
    _tau_wv_pref_per_gal = _tau_gp_per_gal * _r_alpha_val / np.pi * 0.65
    _z_wv_per_gal        = (wave_em.value[np.newaxis, :] / 1216
                            * (1 + redshifts_of_mocks[:, np.newaxis]) - 1)
    _I_z_end_per_gal     = I((1 + 5.3) / (1 + _z_wv_per_gal))
    _I_red_up_all        = I(
        (1 + np.array([cont_filled.first_bubble_encounter_redshift_up_full[i]
                       for i in range(N_DATA)])[:, :, np.newaxis])
        * (1215.67 / (wave_em.value[np.newaxis, :] * (1 + redshifts_of_mocks[:, np.newaxis])))[:, np.newaxis, :]
    )

    # Spectral matrices
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
    _direct_matrix     = _trapz_weights @ _rebin_matrix
    _noise_per_bin     = np.sqrt(_rebin_matrix.sum(axis=0)) * NOISE

    # Stacked sightline arrays
    _tau_prec_all    = np.array([cont_filled.tau_prec_full[i] for i in range(N_DATA)])
    _z_up_all        = np.array([cont_filled.first_bubble_encounter_coord_z_up_full[i] for i in range(N_DATA)])
    _red_up_all      = np.array([cont_filled.first_bubble_encounter_redshift_up_full[i] for i in range(N_DATA)])
    _z_lo_all        = np.array([cont_filled.first_bubble_encounter_coord_z_lo_full[i] for i in range(N_DATA)])
    _red_lo_all      = np.array([cont_filled.first_bubble_encounter_redshift_lo_full[i] for i in range(N_DATA)])
    _la_flux_out_all = np.array([cont_filled.la_flux_out_full[i] for i in range(N_DATA)])
    _com_fact_all    = np.array([cont_filled.com_fact[i] for i in range(N_DATA)])

    _base_continuum = (
        (_la_flux_out_all / _area_factor_per_gal)[:, :, np.newaxis]
        * _j_s_per_gal
        * np.exp(-_tau_prec_all)
        * _tau_cgm_per_gal[:, np.newaxis, :]
        * _com_fact_all[:, np.newaxis, np.newaxis]
    )

    # Outside-bubble precompute
    _tau_post_out = calculate_taus_post_batched(
        redshifts_of_mocks, redshifts_of_mocks,
        _z_up_all.copy(), _red_up_all,
        _z_lo_all.copy(), _red_lo_all,
        z_per_gal=_z_wv_per_gal, tau_wv_pref_per_gal=_tau_wv_pref_per_gal,
        I_z_end_per_gal=_I_z_end_per_gal, I_red_up_all=_I_red_up_all,
    )
    _tau_out = _tau_post_out.copy()
    _bad_out = np.any(_tau_out[:, :, 30:] - _tau_out[:, :, 29:-1] > 0, axis=2)
    for _g in np.where(np.any(_bad_out, axis=1))[0]:
        _ratio_g = (1 + redshifts_of_mocks[_g]) / (1 + _z_wv_per_gal[_g])
        _tau_out[_g, _bad_out[_g]] = np.clip(
            _tau_wv_pref_per_gal[_g] * _ratio_g**1.5 * (I(_ratio_g) - _I_z_end_per_gal[_g]),
            0, np.inf,
        )
    _tau_out[_tau_out < 0] = np.inf
    _tau_out = np.nan_to_num(_tau_out, nan=np.inf)
    _base_cont_eit_outside = _base_continuum * np.exp(-_tau_out)
    del _tau_post_out, _tau_out, _bad_out

    _flux_outside = (
        _base_cont_eit_outside.reshape(N_DATA * N_INSIDE_TAU, 100) @ _direct_matrix
    ).reshape(N_DATA, N_INSIDE_TAU, N_BINS)

    # ── Likelihood (deterministic — no random draws) ──────────────────────────
    def get_spectral_likelihood(xb, yb, zb, rb):
        dx    = x_gal_mock - xb
        dy    = y_gal_mock - yb
        dz    = z_gal_mock - zb
        inside        = dx**2 + dy**2 + dz**2 < rb**2
        dist_arr      = np.where(inside, dz + np.sqrt(np.where(inside, rb**2 - dx**2 - dy**2, 0.0)), 0.0)
        z_end_bub_arr = redshifts_of_mocks - np.where(inside, dist_arr / _R_H_per_gal, 0.0)
        inside_gals   = np.where(inside)[0]

        if len(inside_gals) == 0:
            continuum_all = _base_cont_eit_outside
        else:
            tau_post_in = calculate_taus_post_batched(
                redshifts_of_mocks[inside_gals], z_end_bub_arr[inside_gals],
                _z_up_all[inside_gals].copy(), _red_up_all[inside_gals],
                _z_lo_all[inside_gals].copy(), _red_lo_all[inside_gals],
                z_per_gal=_z_wv_per_gal[inside_gals],
                tau_wv_pref_per_gal=_tau_wv_pref_per_gal[inside_gals],
                I_z_end_per_gal=_I_z_end_per_gal[inside_gals],
                I_red_up_all=_I_red_up_all[inside_gals],
            )
            tau_now_in = _tau_prec_all[inside_gals] + tau_post_in
            bad = np.any(tau_now_in[:, :, 30:] - tau_now_in[:, :, 29:-1] > 0, axis=2)
            if np.any(bad):
                for _ii, g in enumerate(inside_gals):
                    if not np.any(bad[_ii]):
                        continue
                    ratio_g = (1 + z_end_bub_arr[g]) / (1 + _z_wv_per_gal[g])
                    tau_now_in[_ii, bad[_ii]] = np.clip(
                        _tau_wv_pref_per_gal[g] * ratio_g**1.5 * (I(ratio_g) - _I_z_end_per_gal[g]),
                        0, np.inf,
                    )
            tau_now_in[tau_now_in < 0] = np.inf
            tau_now_in = np.nan_to_num(tau_now_in, nan=np.inf)
            continuum_all = _base_cont_eit_outside.copy()
            continuum_all[inside_gals] = _base_continuum[inside_gals] * np.exp(-tau_now_in)

        predicted = _flux_outside.copy()
        if len(inside_gals) > 0:
            flat_in = continuum_all[inside_gals].reshape(len(inside_gals) * N_INSIDE_TAU, 100)
            predicted[inside_gals] = (flat_in @ _direct_matrix).reshape(
                len(inside_gals), N_INSIDE_TAU, N_BINS
            )

        model_mags = 5 * np.log10(10**18.7 * (ADDITIVE + 2 * predicted))
        valid      = np.isfinite(_obs_mag_per_gal) & np.all(np.isfinite(model_mags), axis=1)
        diffs      = _obs_mag_per_gal[:, np.newaxis, :] - model_mags
        _sigma_m   = (10 / np.log(10)) * _noise_per_bin / (ADDITIVE + 2 * predicted)
        _bw_eff    = np.sqrt(BW_KDE**2 + _sigma_m**2)
        log_kde    = (
            logsumexp(-0.5 * (diffs / _bw_eff) ** 2 - np.log(_bw_eff), axis=1)
            - np.log(N_INSIDE_TAU)
            - 0.5 * np.log(2 * np.pi)
        )
        return float(np.sum(log_kde[valid]))

    def log_likelihood(theta):
        return get_spectral_likelihood(theta[0], theta[1], theta[2], theta[3])

    def prior_transform(u):
        return PRIOR_LO + u * (PRIOR_HI - PRIOR_LO)

    # ── Run nested sampling ───────────────────────────────────────────────────
    print(f"[seed {seed}] Running dynesty (nlive={NLIVE_PP}, dlogz={DLOGZ_PP})...", flush=True)
    with mp.get_context('fork').Pool(n_workers) as pool:
        sampler = dynesty.NestedSampler(
            log_likelihood, prior_transform, ndim=NDIM,
            nlive=NLIVE_PP, pool=pool, queue_size=n_workers,
        )
        sampler.run_nested(print_progress=True, dlogz=DLOGZ_PP)

    results       = sampler.results
    weights       = np.exp(results.logwt - results.logz[-1])
    equal_samples = resample_equal(results.samples, weights)

    quantiles = np.array([
        percentileofscore(equal_samples[:, i], TRUE_MU[i]) / 100.0
        for i in range(NDIM)
    ])
    print(f"[seed {seed}] Quantiles: {np.round(quantiles, 3)}", flush=True)
    return quantiles


def plot_pp(quantiles_all: np.ndarray, output_path: str) -> None:
    """
    quantiles_all : (N_seeds, NDIM)
    Plots empirical CDF of posterior quantiles vs. the diagonal.
    Shaded bands show 68% and 95% tolerance intervals for a sample of size N.
    """
    n   = quantiles_all.shape[0]
    fig, axes = plt.subplots(1, NDIM, figsize=(16, 4), sharey=True)

    alpha = np.linspace(0, 1, 300)
    lo1   = binom.ppf(0.16, n, alpha) / n
    hi1   = binom.ppf(0.84, n, alpha) / n
    lo2   = binom.ppf(0.025, n, alpha) / n
    hi2   = binom.ppf(0.975, n, alpha) / n

    for i, (ax, label) in enumerate(zip(axes, PARAM_NAMES)):
        qs        = np.sort(quantiles_all[:, i])
        empirical = np.arange(1, n + 1) / n

        ax.fill_between(alpha, lo2, hi2, color='lightblue', label='95% band')
        ax.fill_between(alpha, lo1, hi1, color='steelblue', alpha=0.6, label='68% band')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='ideal')
        ax.step(qs, empirical, color='red', lw=1.5, where='post', label='empirical')
        ax.set_xlabel('Posterior quantile')
        ax.set_title(label)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if i == 0:
            ax.set_ylabel('Fraction ≤ quantile')
        ax.legend(fontsize=7)

    fig.suptitle(f'P-P coverage test  (N = {n} simulations,  nlive = {NLIVE_PP})')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved {output_path}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',       type=int,  default=None,
                        help='Single seed to run (omit to run --n_seeds sequentially)')
    parser.add_argument('--n_seeds',    type=int,  default=20)
    parser.add_argument('--output_dir', type=str,  default='pp_results')
    parser.add_argument('--n_workers',  type=int,  default=N_WORKERS)
    parser.add_argument('--plot_only',  action='store_true',
                        help='Skip inference, just plot from saved results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.plot_only:
        seeds = [args.seed] if args.seed is not None else list(range(args.n_seeds))
        for seed in seeds:
            out_file = os.path.join(args.output_dir, f'pp_seed_{seed}.npz')
            if os.path.exists(out_file):
                print(f"[seed {seed}] Already done, skipping.", flush=True)
                continue
            q = run_single_inference(seed, n_workers=args.n_workers)
            np.savez(out_file, quantiles=q, true_mu=TRUE_MU, seed=seed)
            print(f"[seed {seed}] Saved to {out_file}", flush=True)

    # Aggregate and plot whatever results exist
    files = sorted(glob.glob(os.path.join(args.output_dir, 'pp_seed_*.npz')))
    if len(files) < 2:
        print(f"Only {len(files)} result file(s) in {args.output_dir}; "
              "need at least 2 to plot.", flush=True)
    else:
        all_q = np.array([np.load(f)['quantiles'] for f in files])
        plot_pp(all_q, os.path.join(args.output_dir, 'pp_plot.png'))

        print(f"\nP-P summary ({len(files)} seeds):")
        ideal_std = 1 / np.sqrt(12)
        for i, name in enumerate(PARAM_NAMES):
            qs = all_q[:, i]
            print(f"  {name:6s}  mean = {qs.mean():.3f} (ideal 0.500)  "
                  f"std = {qs.std():.3f} (ideal {ideal_std:.3f})")