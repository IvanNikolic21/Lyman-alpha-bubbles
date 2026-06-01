"""
Production grid: vary N_DATA × NOISE to map how inference quality scales.

N_DATA  : 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70
NOISE   : 2e-20, 5e-20, 8e-20, 1e-19, 2e-19, 5e-19  (per full-res pixel)
True bubble : (0, 0, 0, 10), fixed seed=42 per combination.

Usage
-----
# Single combination:
python production_run.py --n_gal 30 --noise 5e-20 --output_dir prod_results/

# SLURM array (90 jobs):
#   python production_run.py --job_id $SLURM_ARRAY_TASK_ID --output_dir prod_results/
#   job_id 0..89, maps row-major over N_GAL_GRID × NOISE_GRID

# Full grid sequentially (slow, ~90 × run-time):
python production_run.py --all --output_dir prod_results/

# Summary plots from saved results (no new inference):
python production_run.py --plot_only --output_dir prod_results/
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
import types as _types
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import dynesty
from dynesty.utils import resample_equal
from scipy.special import logsumexp
from astropy.cosmology import Planck18 as Cosmo
from astropy import constants as const
import astropy.units as u

from venv.speed_up import get_content, calculate_taus_post_batched
from venv.galaxy_prop import get_mock_data, get_js, tau_CGM, p_EW
from venv.helpers import full_res_flux, perturb_flux, z_at_proper_distance, I

# ── Grid definition ───────────────────────────────────────────────────────────
N_GAL_GRID  = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
NOISE_GRID  = [2e-20, 5e-20, 8e-20, 1e-19, 2e-19, 5e-19]

def job_id_to_params(job_id: int):
    return N_GAL_GRID[job_id // len(NOISE_GRID)], NOISE_GRID[job_id % len(NOISE_GRID)]

def params_to_filename(n_gal: int, noise: float, output_dir: str) -> str:
    return os.path.join(output_dir, f'prod_ngal{n_gal:03d}_noise{noise:.2e}.npz')

# ── Fixed settings ────────────────────────────────────────────────────────────
TRUE_MU      = np.array([0.0, 0.0, 0.0, 10.0])
PRIOR_LO     = np.array([-10.0, -10.0, -10.0,  1.0])
PRIOR_HI     = np.array([ 10.0,  10.0,  10.0, 20.0])
NDIM         = 4
N_INSIDE_TAU = 100
N_ITER_BUB   = 1
N_BINS       = 11
ADDITIVE     = 1e-18
BW_KDE       = 0.12
N_WORKERS    = 50
NLIVE        = 300
DLOGZ        = 0.5
SEED         = 42
MAIN_DIR     = '/groups/astro/ivannik/programs/Lyman-alpha-bubbles/'
PARAM_NAMES  = ['x_bub', 'y_bub', 'z_bub', 'r_bub']

wave_em  = np.linspace(1214, 1225., 100) * u.Angstrom
wave_Lya = 1215.67 * u.Angstrom

# ── Module-level state (populated before fork, inherited by workers) ──────────
_S = _types.SimpleNamespace()


def _prior_transform(u):
    return PRIOR_LO + u * (PRIOR_HI - PRIOR_LO)


def _log_likelihood(theta):
    s  = _S
    xb, yb, zb, rb = theta
    dx    = s.x_gal_mock - xb
    dy    = s.y_gal_mock - yb
    dz    = s.z_gal_mock - zb
    inside        = dx**2 + dy**2 + dz**2 < rb**2
    dist_arr      = np.where(inside, dz + np.sqrt(np.where(inside, rb**2 - dx**2 - dy**2, 0.0)), 0.0)
    z_end_bub_arr = s.redshifts - np.where(inside, dist_arr / s.R_H, 0.0)
    inside_gals   = np.where(inside)[0]

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

    n_gal = s.x_gal_mock.shape[0]
    predicted = s.flux_outside.copy()
    if len(inside_gals) > 0:
        flat_in = continuum_all[inside_gals].reshape(len(inside_gals) * N_INSIDE_TAU, 100)
        predicted[inside_gals] = (flat_in @ s.direct_matrix).reshape(
            len(inside_gals), N_INSIDE_TAU, N_BINS
        )

    model_mags = 5 * np.log10(10**18.7 * (ADDITIVE + 2 * predicted))
    valid      = np.isfinite(s.obs_mag) & np.all(np.isfinite(model_mags), axis=1)
    diffs      = s.obs_mag[:, np.newaxis, :] - model_mags
    sigma_m    = (10 / np.log(10)) * s.noise_per_bin / (ADDITIVE + 2 * predicted)
    bw_eff     = np.sqrt(BW_KDE**2 + sigma_m**2)
    log_kde    = (
        logsumexp(-0.5 * (diffs / bw_eff) ** 2 - np.log(bw_eff), axis=1)
        - np.log(N_INSIDE_TAU)
        - 0.5 * np.log(2 * np.pi)
    )
    return float(np.sum(log_kde[valid]))


# ── Single run ────────────────────────────────────────────────────────────────

def run_single(n_gal: int, noise: float, n_workers: int = N_WORKERS) -> dict:
    """Run inference for one (n_gal, noise) combination. Returns result dict."""
    np.random.seed(SEED)
    print(f"\n[n_gal={n_gal}, noise={noise:.2e}] Generating mock data...", flush=True)

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
    flux_noise_mock = perturb_flux(full_flux, N_BINS)   # (n_gal, N_BINS)

    print(f"[n_gal={n_gal}, noise={noise:.2e}] Building precomputed arrays...", flush=True)
    cont_filled = get_content(
        Muv_mock.flatten(), redshifts,
        x_gal, y_gal, z_gal,
        n_iter_bub=N_ITER_BUB, n_inside_tau=N_INSIDE_TAU,
        include_muv_unc=False, fwhm_true=False,
        redshift=7.5, xh_unc=True, high_prob_emit=False,
        EW_fixed=False, cache=None, AH22_model=False,
        main_dir=MAIN_DIR, cache_dir=None, gauss_distr=False,
    )

    # Per-galaxy fixed quantities
    R_H = np.array([
        (const.c / Cosmo.H(redshifts[i])).to(u.Mpc).value for i in range(n_gal)
    ])
    tau_cgm  = np.array([tau_CGM(Muv_mock[i], main_dir=MAIN_DIR) for i in range(n_gal)])
    j_s      = np.array([cont_filled.j_s_full[i] for i in range(n_gal)])
    raw_af   = np.array([
        np.trapz(j_s[i] * tau_cgm[i], wave_em.value, axis=1) /
        np.trapz(j_s[i], wave_em.value, axis=1)
        for i in range(n_gal)
    ])
    af       = np.where(raw_af < 1e-20, 1e-5, raw_af)
    obs_mag  = 5 * np.log10(10**18.7 * (ADDITIVE + 2 * flux_noise_mock))

    r_alpha_val  = 6.25e8 / (4 * np.pi * (const.c / wave_Lya).to(u.Hz).value)
    tau_gp       = 7.16e5 * ((1 + redshifts) / 10) ** 1.5
    tau_wv_pref  = tau_gp * r_alpha_val / np.pi * 0.65
    z_wv         = wave_em.value[np.newaxis, :] / 1216 * (1 + redshifts[:, np.newaxis]) - 1
    I_z_end      = I((1 + 5.3) / (1 + z_wv))
    ooz          = 1215.67 / (wave_em.value[np.newaxis, :] * (1 + redshifts[:, np.newaxis]))
    red_up_arr   = np.array([cont_filled.first_bubble_encounter_redshift_up_full[i] for i in range(n_gal)])
    I_red_up     = I((1 + red_up_arr[:, :, np.newaxis]) * ooz[:, np.newaxis, :])

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
    direct_matrix      = _trapz_weights @ _rebin_matrix
    noise_per_bin      = np.sqrt(_rebin_matrix.sum(axis=0)) * noise

    tau_prec  = np.array([cont_filled.tau_prec_full[i] for i in range(n_gal)])
    z_up      = np.array([cont_filled.first_bubble_encounter_coord_z_up_full[i] for i in range(n_gal)])
    red_up    = red_up_arr
    z_lo      = np.array([cont_filled.first_bubble_encounter_coord_z_lo_full[i] for i in range(n_gal)])
    red_lo    = np.array([cont_filled.first_bubble_encounter_redshift_lo_full[i] for i in range(n_gal)])
    la_flux   = np.array([cont_filled.la_flux_out_full[i] for i in range(n_gal)])
    com_fact  = np.array([cont_filled.com_fact[i] for i in range(n_gal)])

    base_cont = (
        (la_flux / af)[:, :, np.newaxis]
        * j_s * np.exp(-tau_prec)
        * tau_cgm[:, np.newaxis, :]
        * com_fact[:, np.newaxis, np.newaxis]
    )

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

    # Populate module-level state before forking
    _S.x_gal_mock     = x_gal
    _S.y_gal_mock     = y_gal
    _S.z_gal_mock     = z_gal
    _S.redshifts      = redshifts
    _S.R_H            = R_H
    _S.tau_prec       = tau_prec
    _S.z_up           = z_up
    _S.red_up         = red_up
    _S.z_lo           = z_lo
    _S.red_lo         = red_lo
    _S.z_wv           = z_wv
    _S.tau_wv_pref    = tau_wv_pref
    _S.I_z_end        = I_z_end
    _S.I_red_up       = I_red_up
    _S.base_cont      = base_cont
    _S.base_cont_outside = base_cont_outside
    _S.flux_outside   = flux_outside
    _S.direct_matrix  = direct_matrix
    _S.noise_per_bin  = noise_per_bin
    _S.obs_mag        = obs_mag

    print(f"[n_gal={n_gal}, noise={noise:.2e}] Running dynesty "
          f"(nlive={NLIVE}, dlogz={DLOGZ})...", flush=True)
    import time
    t0 = time.perf_counter()
    with mp.get_context('fork').Pool(n_workers) as pool:
        sampler = dynesty.NestedSampler(
            _log_likelihood, _prior_transform, ndim=NDIM,
            nlive=NLIVE, pool=pool, queue_size=n_workers,
        )
        sampler.run_nested(print_progress=True, dlogz=DLOGZ)
    wall_time = time.perf_counter() - t0

    results       = sampler.results
    weights       = np.exp(results.logwt - results.logz[-1])
    equal_samples = resample_equal(results.samples, weights)

    post_mean   = equal_samples.mean(axis=0)
    post_median = np.median(equal_samples, axis=0)
    post_std    = equal_samples.std(axis=0)
    post_p16    = np.percentile(equal_samples, 16, axis=0)
    post_p84    = np.percentile(equal_samples, 84, axis=0)

    print(f"[n_gal={n_gal}, noise={noise:.2e}] Done in {wall_time:.1f}s", flush=True)
    for _pi, _pn in enumerate(PARAM_NAMES):
        print(f"  {_pn:6s}  median={post_median[_pi]:.3f}  "
              f"std={post_std[_pi]:.3f}  truth={TRUE_MU[_pi]:.1f}  "
              f"[{post_p16[_pi]:.2f}, {post_p84[_pi]:.2f}]", flush=True)

    return dict(
        n_gal=n_gal, noise=noise,
        posterior_samples=equal_samples,
        post_mean=post_mean, post_median=post_median, post_std=post_std,
        post_p16=post_p16, post_p84=post_p84,
        true_mu=TRUE_MU,
        logz=results.logz[-1], logzerr=results.logzerr[-1],
        ncall=results.ncall.sum(), wall_time=wall_time,
    )


# ── Summary plots ─────────────────────────────────────────────────────────────

def plot_grid(output_dir: str) -> None:
    files = sorted(glob.glob(os.path.join(output_dir, 'prod_ngal*.npz')))
    if not files:
        print("No result files found.", flush=True)
        return

    records = [dict(np.load(f, allow_pickle=True)) for f in files]

    noise_vals  = sorted(set(float(r['noise']) for r in records))
    n_gal_vals  = sorted(set(int(r['n_gal']) for r in records))
    colors      = cm.plasma(np.linspace(0.15, 0.85, len(noise_vals)))

    fig, axes = plt.subplots(2, NDIM, figsize=(18, 8), sharex=True)

    for ni, noise in enumerate(noise_vals):
        subset = sorted([r for r in records if float(r['noise']) == noise],
                        key=lambda r: int(r['n_gal']))
        ngals  = [int(r['n_gal'])   for r in subset]
        stds   = np.array([r['post_std']    for r in subset])   # (N, 4)
        biases = np.array([np.abs(r['post_median'] - r['true_mu']) for r in subset])  # (N, 4)
        label  = f'{noise:.0e}'

        for pi in range(NDIM):
            axes[0, pi].plot(ngals, stds[:, pi],   color=colors[ni], lw=1.5,
                             marker='o', ms=4, label=label)
            axes[1, pi].plot(ngals, biases[:, pi], color=colors[ni], lw=1.5,
                             marker='o', ms=4, label=label)

    for pi, pname in enumerate(PARAM_NAMES):
        axes[0, pi].set_title(pname)
        axes[0, pi].set_ylabel('Posterior std')
        axes[1, pi].set_ylabel('|Median − truth|')
        axes[1, pi].set_xlabel('N galaxies')
        for row in range(2):
            axes[row, pi].axhline(0, color='k', lw=0.5, ls='--')

    axes[0, -1].legend(title='Noise (flux)', fontsize=8, bbox_to_anchor=(1.02, 1))
    fig.suptitle('Inference quality vs. N galaxies and noise level\n'
                 'True bubble: (0, 0, 0, 10)')
    fig.tight_layout()
    out_path = os.path.join(output_dir, 'prod_summary.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved {out_path}", flush=True)

    # Also save a compact table to stdout
    print(f"\n{'n_gal':>6}  {'noise':>8}  "
          + "  ".join(f"std_{p:4s}" for p in PARAM_NAMES)
          + "  " + "  ".join(f"bias_{p:4s}" for p in PARAM_NAMES))
    for r in sorted(records, key=lambda r: (float(r['noise']), int(r['n_gal']))):
        std_str  = "  ".join(f"{r['post_std'][i]:8.3f}" for i in range(NDIM))
        bias_str = "  ".join(f"{abs(r['post_median'][i]-r['true_mu'][i]):9.3f}"
                             for i in range(NDIM))
        print(f"{int(r['n_gal']):6d}  {float(r['noise']):8.2e}  {std_str}  {bias_str}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gal',      type=int,   default=None)
    parser.add_argument('--noise',      type=float, default=None)
    parser.add_argument('--job_id',     type=int,   default=None,
                        help='0..89: maps to (n_gal, noise) row-major over grid')
    parser.add_argument('--all',        action='store_true',
                        help='Run full grid sequentially')
    parser.add_argument('--plot_only',  action='store_true')
    parser.add_argument('--output_dir', type=str,   default='prod_results')
    parser.add_argument('--n_workers',  type=int,   default=N_WORKERS)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.plot_only:
        plot_grid(args.output_dir)

    else:
        # Build list of (n_gal, noise) pairs to run
        if args.job_id is not None:
            pairs = [job_id_to_params(args.job_id)]
        elif args.n_gal is not None and args.noise is not None:
            pairs = [(args.n_gal, args.noise)]
        elif args.all:
            pairs = [(n, ns) for n in N_GAL_GRID for ns in NOISE_GRID]
        else:
            parser.error('Specify --n_gal + --noise, --job_id, or --all')

        for n_gal, noise in pairs:
            out_file = params_to_filename(n_gal, noise, args.output_dir)
            if os.path.exists(out_file):
                print(f"[n_gal={n_gal}, noise={noise:.2e}] Already done, skipping.",
                      flush=True)
                continue
            result = run_single(n_gal, noise, n_workers=args.n_workers)
            np.savez(out_file, **result)
            print(f"Saved {out_file}", flush=True)

        plot_grid(args.output_dir)