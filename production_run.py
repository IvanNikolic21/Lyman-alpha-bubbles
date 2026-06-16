"""
Production grid: vary N_DATA × NOISE × SEED to map inference quality.

N_DATA  : 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70
NOISE   : 2e-20, 5e-20, 8e-20, 1e-19, 2e-19, 5e-19  (per full-res pixel)
Seeds   : 0 .. n_seeds-1  (default 5 → 450 jobs total)
True bubble : (0, 0, 0, 10)

Results are aggregated across seeds at plot time:
  - Top row   : median posterior std ± 68% CI across seeds
  - Bottom row: SIGNED bias (median − truth) ± 68% CI across seeds
    A band that straddles zero consistently → unbiased inference.
    A median consistently offset → systematic model bias to investigate.

Usage
-----
# Single combination + seed:
python production_run.py --n_gal 30 --noise 5e-20 --seed 0 --output_dir prod_results/

# SLURM array (n_seeds=5 → 450 jobs, job_id 0..449):
#   python production_run.py --job_id $SLURM_ARRAY_TASK_ID --n_seeds 5 --output_dir prod_results/

# Full grid sequentially:
python production_run.py --all --n_seeds 5 --output_dir prod_results/

# Summary plots from saved results (no new inference):
python production_run.py --plot_only --output_dir prod_results/
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import time
import argparse
import glob
import multiprocessing as mp
import types as _types
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import dynesty
from dynesty.utils import resample_equal
from scipy.special import logsumexp, gammaln, stdtr
from astropy.cosmology import Planck18 as Cosmo
from astropy import constants as const
import astropy.units as u

from venv.speed_up import get_content, calculate_taus_post_batched
from venv.galaxy_prop import get_mock_data, get_js, tau_CGM, p_EW
from venv.helpers import full_res_flux, perturb_flux, z_at_proper_distance, I, \
    comoving_distance_from_source_Mpc
from venv.igm_prop import calculate_taus_i, tau_wv as _tau_wv_igm

# ── Grid definition ───────────────────────────────────────────────────────────
N_GAL_GRID  = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
NOISE_GRID  = [2e-20, 5e-20, 8e-20, 1e-19, 2e-19, 5e-19]
N_COMBOS    = len(N_GAL_GRID) * len(NOISE_GRID)   # 90

def job_id_to_params(job_id: int, n_seeds: int, n_gal_list=None):
    """job_id runs over combinations first, then seeds:
       job_id = comb_idx * n_seeds + seed_idx
       n_gal_list overrides N_GAL_GRID for subset runs."""
    gal_grid = n_gal_list if n_gal_list is not None else N_GAL_GRID
    max_id   = len(gal_grid) * len(NOISE_GRID) * n_seeds - 1
    if job_id > max_id:
        raise ValueError(
            f"job_id={job_id} exceeds max={max_id} "
            f"(gal_grid={len(gal_grid)} × noise={len(NOISE_GRID)} × n_seeds={n_seeds}). "
            f"Did you forget --n_seeds {n_seeds}?"
        )
    comb_idx = job_id // n_seeds
    seed     = job_id  % n_seeds
    n_gal    = gal_grid[comb_idx // len(NOISE_GRID)]
    noise    = NOISE_GRID[comb_idx  % len(NOISE_GRID)]
    return n_gal, noise, seed

def params_to_filename(n_gal: int, noise: float, seed: int, output_dir: str,
                       censored: bool = False, lae_first: bool = False,
                       two_bub_mock: bool = False) -> str:
    suffix = (("_censored" if censored else "")
              + ("_laefirst" if lae_first else "")
              + ("_2bubmock" if two_bub_mock else ""))
    return os.path.join(output_dir, f'prod_ngal{n_gal:03d}_noise{noise:.2e}_seed{seed:02d}{suffix}.npz')

# ── Fixed settings ────────────────────────────────────────────────────────────
TRUE_MU      = np.array([0.0, 0.0, 0.0, 10.0])
PRIOR_LO     = np.array([-10.0, -10.0, -10.0,  1.0])
PRIOR_HI     = np.array([ 10.0,  10.0,  10.0, 20.0])
NDIM         = 4
N_INSIDE_TAU = 200
N_ITER_BUB   = 1
N_BINS       = 11
NU_STUDENT   = 3.0    # Student-t degrees of freedom
N_WORKERS    = 50
NLIVE        = 300
DLOGZ        = 0.5
MAIN_DIR     = '/groups/astro/ivannik/programs/Lyman-alpha-bubbles/'
PARAM_NAMES  = ['x_bub', 'y_bub', 'z_bub', 'r_bub']
TRUE_2BUB_MU = np.array([-5.0, -5.0, -2.0, 7.0, 4.0, 5.0, 6.0, 5.0])
NDIM_2BUB       = 8
PARAM_NAMES_2BUB = ['x1_bub', 'y1_bub', 'z1_bub', 'r1_bub',
                    'x2_bub', 'y2_bub', 'z2_bub', 'r2_bub']
SNR_DET_THRESH = 1.0    # peak obs SNR below which a galaxy is treated as non-detected
LAE_EW_THRESH  = 25.0   # minimum observed EW (Å) to qualify as a LAE

wave_em  = np.linspace(1214, 1225., 100) * u.Angstrom
wave_Lya = 1215.67 * u.Angstrom

# ── Module-level state (populated before fork, inherited by workers) ──────────
_S = _types.SimpleNamespace()

_S_PERM_FIELDS = [
    'x_gal_mock', 'y_gal_mock', 'z_gal_mock', 'redshifts', 'R_H',
    'tau_prec', 'z_up', 'red_up', 'z_lo', 'red_lo',
    'z_wv', 'tau_wv_pref', 'I_z_end', 'I_red_up',
    'base_cont', 'base_cont_outside', 'flux_outside', 'obs_flux',
]

def _permute_S(perm):
    for f in _S_PERM_FIELDS:
        setattr(_S, f, getattr(_S, f)[perm])


def _prior_transform(u):
    return PRIOR_LO + u * (PRIOR_HI - PRIOR_LO)


def _prior_transform_2bub(u):
    p = np.empty(8)
    p[:4] = PRIOR_LO + u[:4] * (PRIOR_HI - PRIOR_LO)
    p[4:] = PRIOR_LO + u[4:] * (PRIOR_HI - PRIOR_LO)
    if p[3] < p[7]:          # enforce r1 >= r2 to break bubble-swap symmetry
        p[3], p[7] = p[7], p[3]
    return p


def _log_likelihood_2bub(theta):
    s = _S
    x1, y1, z1, r1, x2, y2, z2, r2 = theta

    dx1 = s.x_gal_mock - x1;  dy1 = s.y_gal_mock - y1;  dz1 = s.z_gal_mock - z1
    dx2 = s.x_gal_mock - x2;  dy2 = s.y_gal_mock - y2;  dz2 = s.z_gal_mock - z2

    in1 = dx1**2 + dy1**2 + dz1**2 < r1**2
    in2 = dx2**2 + dy2**2 + dz2**2 < r2**2
    inside = in1 | in2

    # LOS distance from galaxy to near face; use np.maximum to avoid sqrt of negatives
    dist1 = dz1 + np.sqrt(np.maximum(r1**2 - dx1**2 - dy1**2, 0.0))
    dist2 = dz2 + np.sqrt(np.maximum(r2**2 - dx2**2 - dy2**2, 0.0))

    # Near-face redshift per bubble; inf for galaxies not inside that bubble
    z_end1 = np.where(in1, s.redshifts - dist1 / s.R_H, np.inf)
    z_end2 = np.where(in2, s.redshifts - dist2 / s.R_H, np.inf)

    # Galaxy inside multiple bubbles: take whichever near face is closer to observer
    # (lower redshift → lower tau_IGM to observer)
    z_end_bub_arr = np.minimum(z_end1, z_end2)

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
    diffs     = s.obs_flux[:, np.newaxis, :] - predicted
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
    per_gal = log_p.sum(axis=1)

    if getattr(s, 'censored', False):
        obs_peak_snr = np.max(np.abs(s.obs_flux) / s.noise_per_bin, axis=-1)
        detected = obs_peak_snr > SNR_DET_THRESH
        if not np.all(detected):
            det_threshold = SNR_DET_THRESH * s.noise_per_bin
            z_cdf = (det_threshold - predicted) / s.noise_per_bin
            log_cdf = np.log(stdtr(NU_STUDENT, z_cdf))
            log_p_nondet = logsumexp(log_cdf.sum(axis=2), axis=1) - np.log(N_INSIDE_TAU)
            per_gal = np.where(detected, per_gal, log_p_nondet)

    return float(per_gal.sum())


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

    np.nan_to_num(predicted, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    diffs     = s.obs_flux[:, np.newaxis, :] - predicted         # (N_gal, N_INSIDE_TAU, N_BINS)
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
    per_gal = log_p.sum(axis=1)   # (n_gal,)

    if getattr(s, 'censored', False):
        obs_peak_snr = np.max(np.abs(s.obs_flux) / s.noise_per_bin, axis=-1)
        detected = obs_peak_snr > SNR_DET_THRESH
        if not np.all(detected):
            det_threshold = SNR_DET_THRESH * s.noise_per_bin
            z_cdf = (det_threshold - predicted) / s.noise_per_bin
            log_cdf = np.log(stdtr(NU_STUDENT, z_cdf))
            log_p_nondet = logsumexp(log_cdf.sum(axis=2), axis=1) - np.log(N_INSIDE_TAU)
            per_gal = np.where(detected, per_gal, log_p_nondet)

    return float(per_gal.sum())


# ── Two-bubble mock helpers ───────────────────────────────────────────────────

def _tau_for_gal_in_bub(xs_i, ys_i, zs_i, red_s,
                         xb, yb, zb, rb,
                         x_b_bg, y_b_bg, z_b_bg, r_bubs_bg):
    """Return (tau_arr (100,), z_end_bub) for galaxy i inside bubble (xb,yb,zb,rb)."""
    dist_i = zs_i - zb + np.sqrt(rb**2 - (xs_i - xb)**2 - (ys_i - yb)**2)
    z_end  = z_at_proper_distance(dist_i / (1 + red_s) * u.Mpc, red_s)
    tau    = calculate_taus_i(x_b_bg, y_b_bg, z_b_bg, r_bubs_bg,
                               red_s, z_end, n_iter=1,
                               x_pos=xs_i, y_pos=ys_i, dist=dist_i)
    if np.any(tau[:, 30:] - tau[:, 29:-1] > 0.0):
        inds_rm = np.where(np.any(tau[:, 30:] - tau[:, 29:-1] > 0.0, axis=1))[0]
        for idx in inds_rm:
            shift_sm = np.random.normal(0.0, 0.1)
            dc = comoving_distance_from_source_Mpc(red_s, z_end)
            tau[idx] = np.clip(
                _tau_wv_igm(wave_em, dist=np.abs(dc), zs=red_s, z_end=5.3, nf=0.65) + shift_sm,
                0, np.inf,
            )
    return np.nan_to_num(tau[0], np.inf), z_end


def _apply_second_bubble_tau(xs, ys, zs, redshifts, tau_mock,
                              xb1, yb1, zb1, rb1,
                              xb2, yb2, zb2, rb2,
                              x_b_bg, y_b_bg, z_b_bg, r_bubs_bg):
    """Return tau_mock updated for galaxies that benefit from bubble 2."""
    tau_out = tau_mock.copy()
    for i in range(len(xs)):
        in2 = (xs[i]-xb2)**2 + (ys[i]-yb2)**2 + (zs[i]-zb2)**2 < rb2**2
        if not in2:
            continue
        tau2, z_end2 = _tau_for_gal_in_bub(xs[i], ys[i], zs[i], redshifts[i],
                                             xb2, yb2, zb2, rb2,
                                             x_b_bg, y_b_bg, z_b_bg, r_bubs_bg)
        in1 = (xs[i]-xb1)**2 + (ys[i]-yb1)**2 + (zs[i]-zb1)**2 < rb1**2
        if not in1:
            tau_out[i] = tau2
        else:
            dist1  = zs[i] - zb1 + np.sqrt(rb1**2 - (xs[i]-xb1)**2 - (ys[i]-yb1)**2)
            z_end1 = z_at_proper_distance(dist1 / (1 + redshifts[i]) * u.Mpc, redshifts[i])
            if z_end2 < z_end1:
                tau_out[i] = tau2
    return tau_out


# ── Single run ────────────────────────────────────────────────────────────────

def run_single(n_gal: int, noise: float, seed: int, n_workers: int = N_WORKERS,
               censored: bool = False, lae_first: bool = False,
               two_bub_mu: np.ndarray = None) -> dict:
    """Run inference for one (n_gal, noise, seed) combination. Returns result dict.

    two_bub_mu : array of shape (8,) = [x1,y1,z1,r1, x2,y2,z2,r2], optional.
        When given, the mock is generated from two bubbles instead of TRUE_MU.
    """
    np.random.seed(seed)
    two_bub = two_bub_mu is not None
    true_mu_1 = two_bub_mu[:4] if two_bub else TRUE_MU

    print(f"\n[n_gal={n_gal}, noise={noise:.2e}, seed={seed}] Generating mock data"
          f"{' (2-bubble mock)' if two_bub else ''}...", flush=True)

    Muv_mock = np.ones(n_gal) * -18.5
    beta     = -2.0 * np.ones(n_gal)

    tau_mock, x_gal, y_gal, z_gal, x_b_bg, y_b_bg, z_b_bg, r_bubs_bg = get_mock_data(
        n_gal=n_gal, z_start=7.5,
        r_bubble=float(true_mu_1[3]),
        xb=float(true_mu_1[0]), yb=float(true_mu_1[1]), zb=float(true_mu_1[2]),
        dist=15,
    )
    redshifts = np.array([
        z_at_proper_distance(-z_gal[i] / (1 + 7.5) * u.Mpc, 7.5)
        for i in range(n_gal)
    ])

    if two_bub:
        x2, y2, z2, r2 = two_bub_mu[4:]
        tau_mock = _apply_second_bubble_tau(
            x_gal, y_gal, z_gal, redshifts, tau_mock,
            *true_mu_1, x2, y2, z2, r2,
            x_b_bg, y_b_bg, z_b_bg, r_bubs_bg,
        )
        n_in1 = int(((x_gal-true_mu_1[0])**2 + (y_gal-true_mu_1[1])**2
                     + (z_gal-true_mu_1[2])**2 < true_mu_1[3]**2).sum())
        n_in2 = int(((x_gal-x2)**2 + (y_gal-y2)**2
                     + (z_gal-z2)**2 < r2**2).sum())
        print(f"  Inside bub1: {n_in1}/{n_gal}   Inside bub2: {n_in2}/{n_gal}", flush=True)

    one_J = get_js(z=7.5, muv=Muv_mock, n_iter=n_gal)
    area_factor = np.array([
        np.trapz(one_J[0][i] * tau_CGM(Muv_mock[i], main_dir=MAIN_DIR), wave_em.value)
        / np.trapz(one_J[0][i], wave_em.value)
        for i in range(n_gal)
    ])
    ews, la_e = p_EW(Muv_mock.flatten(), beta.flatten())
    la_e = la_e.reshape(np.shape(Muv_mock)) / area_factor

    continuum = (
        la_e[:, np.newaxis] * one_J[0][:n_gal] * np.exp(-tau_mock)
        * tau_CGM(Muv_mock, main_dir=MAIN_DIR)
        / (4 * np.pi * Cosmo.luminosity_distance(7.5).to(u.cm).value ** 2)
    )
    full_flux = full_res_flux(continuum, 7.5)
    full_flux += np.random.normal(0, noise, np.shape(full_flux))
    flux_noise_mock = perturb_flux(full_flux, N_BINS)   # (n_gal, N_BINS)

    # ── Geometry diagnostics ─────────────────────────────────────────────────
    _inside = ((x_gal - true_mu_1[0])**2 + (y_gal - true_mu_1[1])**2
               + (z_gal - true_mu_1[2])**2 < true_mu_1[3]**2)
    print(f"  x: mean={x_gal.mean():+.2f}  y: mean={y_gal.mean():+.2f}  "
          f"z: mean={z_gal.mean():+.2f}", flush=True)
    print(f"  Inside bub1: {_inside.sum()}/{n_gal}"
          + (f"  z_mean={z_gal[_inside].mean():+.2f}" if _inside.any() else ""), flush=True)
    print(f"  tau_mock: min={tau_mock.min():.3f}  "
          f"mean={tau_mock.mean():.3f}  max={tau_mock.max():.3f}", flush=True)

    print(f"[n_gal={n_gal}, noise={noise:.2e}, seed={seed}] Building precomputed arrays...", flush=True)
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
    obs_flux = flux_noise_mock   # (n_gal, N_BINS) — raw flux, used in Gaussian likelihood

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

    # ── LAE-first: ensure ≥1 LAE per bubble, swap best B1 LAE to index 0 ────────
    lae_mask = np.array([], dtype=int)
    perm     = np.arange(n_gal)
    if lae_first:
        tau_cgm_arr = tau_CGM(Muv_mock, main_dir=MAIN_DIR)          # (n_gal, 100)
        j_s_arr  = np.array([cont_filled.j_s_full[i] for i in range(n_gal)])
        j_s_mean = j_s_arr.mean(axis=1)                              # (n_gal, 100)
        numer_ew = np.trapz(j_s_mean * np.exp(-tau_mock) * tau_cgm_arr, wave_em.value, axis=1)
        denom_ew = np.trapz(j_s_mean * tau_cgm_arr,                  wave_em.value, axis=1)
        ratio_ew = np.where(denom_ew > 1e-30, numer_ew / denom_ew, 0.0)  # fixed per galaxy

        if two_bub:
            # Separate inside masks for each bubble
            _x2, _y2, _z2, _r2 = two_bub_mu[4:]
            in_b1 = ((x_gal - true_mu_1[0])**2 + (y_gal - true_mu_1[1])**2
                     + (z_gal - true_mu_1[2])**2 < true_mu_1[3]**2)
            in_b2 = ((x_gal - _x2)**2 + (y_gal - _y2)**2
                     + (z_gal - _z2)**2 < _r2**2)

            # Resample EW draw until both bubbles have ≥1 LAE (positions/tau fixed)
            N_MAX_EW_ATTEMPTS = 30
            for _attempt in range(N_MAX_EW_ATTEMPTS):
                ew_eff  = ews * ratio_ew
                lae_b1  = np.where(in_b1 & (ew_eff > LAE_EW_THRESH))[0]
                lae_b2  = np.where(in_b2 & (ew_eff > LAE_EW_THRESH))[0]
                if len(lae_b1) > 0 and len(lae_b2) > 0:
                    break
                # Redraw EWs only; update flux accordingly
                ews, _la_e_new = p_EW(Muv_mock.flatten(), beta.flatten())
                la_e      = _la_e_new.reshape(np.shape(Muv_mock)) / area_factor
                _cont_new = (
                    la_e[:, np.newaxis] * one_J[0][:n_gal] * np.exp(-tau_mock)
                    * tau_cgm_arr
                    / (4 * np.pi * Cosmo.luminosity_distance(7.5).to(u.cm).value ** 2)
                )
                _ff_new       = full_res_flux(_cont_new, 7.5)
                _ff_new      += np.random.normal(0, noise, np.shape(_ff_new))
                flux_noise_mock = perturb_flux(_ff_new, N_BINS)
                obs_flux        = flux_noise_mock
            else:
                missing = ((['B1'] if len(lae_b1) == 0 else [])
                           + (['B2'] if len(lae_b2) == 0 else []))
                print(f"  WARNING: after {N_MAX_EW_ATTEMPTS} EW resamples, "
                      f"still no LAE in {missing}. Proceeding anyway.", flush=True)

            print(f"  lae_first (2-bub): B1 LAEs={len(lae_b1)}  B2 LAEs={len(lae_b2)}"
                  + (f"  EW_b1_max={ew_eff[in_b1].max():.1f}Å"
                     f"  EW_b2_max={ew_eff[in_b2].max():.1f}Å"
                     if in_b1.any() and in_b2.any() else ""), flush=True)

            # Swap best B1 LAE to index 0
            lae_mask = lae_b1
            if len(lae_mask) > 0 and lae_mask[0] != 0:
                k = lae_mask[0]
                perm[0], perm[k] = k, 0
                tau_mock = tau_mock[perm];  x_gal = x_gal[perm]
                y_gal = y_gal[perm];        z_gal = z_gal[perm]
                redshifts = redshifts[perm];  flux_noise_mock = flux_noise_mock[perm]
                print(f"  lae_first (2-bub): swapped B1 galaxy {k} "
                      f"(EW_eff={ew_eff[k]:.1f}Å) → index 0", flush=True)

        else:
            ew_eff   = ews * ratio_ew
            lae_mask = np.where(ew_eff > LAE_EW_THRESH)[0]
            if len(lae_mask) == 0:
                print(f"  WARNING: no LAE found (max EW_eff={ew_eff.max():.1f}Å); "
                      f"lae_first not applied.", flush=True)
            elif lae_mask[0] != 0:
                k = lae_mask[0]
                perm[0], perm[k] = k, 0
                tau_mock     = tau_mock[perm]
                x_gal        = x_gal[perm];  y_gal = y_gal[perm];  z_gal = z_gal[perm]
                redshifts    = redshifts[perm]
                flux_noise_mock = flux_noise_mock[perm]
                print(f"  lae_first: swapped galaxy {k} (EW_eff={ew_eff[k]:.1f}Å) → index 0",
                      flush=True)

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
    _S.obs_flux       = obs_flux
    _S.censored       = censored

    if lae_first and len(lae_mask) > 0 and lae_mask[0] != 0:
        _permute_S(perm)

    # Non-detection diagnostic — fixed property of the mock, not theta-dependent
    _peak_snr  = np.max(np.abs(_S.obs_flux) / _S.noise_per_bin, axis=1)
    _detected  = _peak_snr > SNR_DET_THRESH
    n_detected = int(_detected.sum())
    print(f"  Detection status (peak SNR > {SNR_DET_THRESH}): "
          f"{n_detected}/{n_gal} detected, {n_gal - n_detected} non-detected "
          f"({'censoring active' if censored else 'censoring OFF — all treated as detected'})",
          flush=True)

    print(f"[n_gal={n_gal}, noise={noise:.2e}, seed={seed}] Running dynesty "
          f"(nlive={NLIVE}, dlogz={DLOGZ})...", flush=True)
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

    print(f"[n_gal={n_gal}, noise={noise:.2e}, seed={seed}] Done in {wall_time:.1f}s", flush=True)
    for _pi, _pn in enumerate(PARAM_NAMES):
        print(f"  {_pn:6s}  median={post_median[_pi]:.3f}  "
              f"std={post_std[_pi]:.3f}  truth={TRUE_MU[_pi]:.1f}  "
              f"[{post_p16[_pi]:.2f}, {post_p84[_pi]:.2f}]", flush=True)

    return dict(
        n_gal=n_gal, noise=noise, seed=seed,
        posterior_samples=equal_samples,
        post_mean=post_mean, post_median=post_median, post_std=post_std,
        post_p16=post_p16, post_p84=post_p84,
        true_mu=true_mu_1,
        n_detected=n_detected,
        logz=results.logz[-1], logzerr=results.logzerr[-1],
        ncall=results.ncall.sum(), wall_time=wall_time,
    )


# ── Bayes factor run ──────────────────────────────────────────────────────────

def run_bayes_factor(n_gal: int, noise: float, seed: int,
                     n_workers: int = N_WORKERS,
                     censored: bool = False, lae_first: bool = False,
                     two_bub_mu: np.ndarray = None) -> dict:
    """Run M1 (1 bubble) and M2 (2 bubbles) on the same mock; return Bayes factor.

    two_bub_mu : optional (8,) array — when given, mock is generated from two bubbles.
    """
    # M1 — also populates _S which M2 will reuse
    print(f"\n=== Bayes factor run: n_gal={n_gal}, noise={noise:.2e}, seed={seed}"
          f"{' [2-bub mock]' if two_bub_mu is not None else ''} ===", flush=True)
    print("--- Model 1: single bubble ---", flush=True)
    result_m1 = run_single(n_gal, noise, seed, n_workers=n_workers,
                           censored=censored, lae_first=lae_first,
                           two_bub_mu=two_bub_mu)

    # M2 — _S already populated; fork pool inherits it
    print("--- Model 2: two bubbles ---", flush=True)
    t0 = time.perf_counter()
    with mp.get_context('fork').Pool(n_workers) as pool:
        sampler_m2 = dynesty.NestedSampler(
            _log_likelihood_2bub, _prior_transform_2bub, ndim=NDIM_2BUB,
            nlive=NLIVE, pool=pool, queue_size=n_workers,
        )
        sampler_m2.run_nested(print_progress=True, dlogz=DLOGZ)
    wall_time_m2 = time.perf_counter() - t0

    res2         = sampler_m2.results
    weights_m2   = np.exp(res2.logwt - res2.logz[-1])
    samples_m2   = resample_equal(res2.samples, weights_m2)
    logz_2       = float(res2.logz[-1])
    logzerr_2    = float(res2.logzerr[-1])
    logz_1       = float(result_m1['logz'])
    log_bf       = logz_2 - logz_1

    print(f"\n  log Z(M1) = {logz_1:.2f} ± {float(result_m1['logzerr']):.2f}", flush=True)
    print(f"  log Z(M2) = {logz_2:.2f} ± {logzerr_2:.2f}", flush=True)
    print(f"  log BF(M2/M1) = {log_bf:.2f}  →  BF = {np.exp(log_bf):.2f}  "
          f"({'M2 preferred' if log_bf > 0 else 'M1 preferred'})", flush=True)

    return dict(
        **result_m1,
        logz_2=logz_2, logzerr_2=logzerr_2,
        log_bf=log_bf,
        posterior_samples_m2=samples_m2,
        true_mu_m2=two_bub_mu if two_bub_mu is not None else np.full(8, np.nan),
        wall_time_m2=wall_time_m2,
    )


# ── Summary plots ─────────────────────────────────────────────────────────────

def plot_grid(output_dir: str, n_gal_list=None) -> None:
    files = sorted(glob.glob(os.path.join(output_dir, 'prod_ngal*.npz')))
    if not files:
        print("No result files found.", flush=True)
        return

    records    = [dict(np.load(f, allow_pickle=True)) for f in files]
    if n_gal_list is not None:
        records = [r for r in records if int(r['n_gal']) in n_gal_list]
    if not records:
        print("No records match the requested n_gal_list.", flush=True)
        return
    noise_vals = sorted(set(float(r['noise']) for r in records))
    colors     = cm.plasma(np.linspace(0.15, 0.85, len(noise_vals)))

    fig, axes = plt.subplots(2, NDIM, figsize=(18, 8), sharex=True)

    for ni, noise in enumerate(noise_vals):
        # Group by n_gal, aggregate over seeds
        by_ngal = {}
        for r in records:
            if float(r['noise']) != noise:
                continue
            ng = int(r['n_gal'])
            by_ngal.setdefault(ng, []).append(r)

        ngals = sorted(by_ngal)
        # Compute statistics per n_gal separately — seed counts may differ
        # if some jobs are still running.
        std_med  = np.zeros((len(ngals), NDIM))
        std_lo   = np.zeros((len(ngals), NDIM))
        std_hi   = np.zeros((len(ngals), NDIM))
        nbias_med = np.zeros((len(ngals), NDIM))
        nbias_lo  = np.zeros((len(ngals), NDIM))
        nbias_hi  = np.zeros((len(ngals), NDIM))
        for ki, ng in enumerate(ngals):
            seeds_here = by_ngal[ng]
            ss = np.array([s['post_std']                   for s in seeds_here])  # (n_s, 4)
            bs = np.array([s['post_median'] - s['true_mu'] for s in seeds_here])  # (n_s, 4)
            # Normalised bias: per-seed bias / per-seed std
            nbs = bs / np.where(ss > 0, ss, np.nan)                               # (n_s, 4)
            std_med[ki]   = np.median(ss,   axis=0)
            std_lo[ki]    = np.percentile(ss,   16, axis=0)
            std_hi[ki]    = np.percentile(ss,   84, axis=0)
            nbias_med[ki] = np.nanmedian(nbs,   axis=0)
            nbias_lo[ki]  = np.nanpercentile(nbs,  16, axis=0)
            nbias_hi[ki]  = np.nanpercentile(nbs,  84, axis=0)

        label = f'{noise:.0e}'
        for pi in range(NDIM):
            c = colors[ni]
            # std
            axes[0, pi].plot(ngals, std_med[:, pi], color=c, lw=1.5,
                             marker='o', ms=4, label=label)
            axes[0, pi].fill_between(ngals, std_lo[:, pi], std_hi[:, pi],
                                     color=c, alpha=0.2)
            # normalised bias
            axes[1, pi].plot(ngals, nbias_med[:, pi], color=c, lw=1.5,
                             marker='o', ms=4, label=label)
            axes[1, pi].fill_between(ngals, nbias_lo[:, pi], nbias_hi[:, pi],
                                     color=c, alpha=0.2)

    for pi, pname in enumerate(PARAM_NAMES):
        axes[0, pi].set_title(pname)
        axes[0, pi].set_ylabel('Posterior std  (median ± 68% CI over seeds)')
        axes[1, pi].set_ylabel('(Median − truth) / std  (± 68% CI over seeds)')
        axes[1, pi].set_xlabel('N galaxies')
        axes[1, pi].axhline( 0, color='k',    lw=0.8, ls='--')
        axes[1, pi].axhline( 1, color='gray', lw=0.7, ls=':')
        axes[1, pi].axhline(-1, color='gray', lw=0.7, ls=':')
        axes[0, pi].set_ylim(bottom=0)

    axes[0, -1].legend(title='Noise (flux)', fontsize=8, bbox_to_anchor=(1.02, 1))
    # Count max seeds seen for any single (n_gal, noise) combination
    from collections import Counter
    combo_counts = Counter((int(r['n_gal']), float(r['noise'])) for r in records)
    max_seeds = max(combo_counts.values()) if combo_counts else 1
    fig.suptitle(f'Inference quality vs. N galaxies and noise  '
                 f'(up to {max_seeds} seeds per combination)\n'
                 f'True bubble: (0, 0, 0, 10)  —  shaded band = 68% CI over seeds, dashed = zero bias')
    fig.tight_layout()
    out_path = os.path.join(output_dir, 'prod_summary.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved {out_path}", flush=True)


# ── Corner plots ──────────────────────────────────────────────────────────────

def _make_corner(samples, labels, truths, title, out_path):
    try:
        import corner
    except ImportError:
        print("Install corner: pip install corner")
        return
    fig = corner.corner(
        samples, labels=labels, truths=truths,
        truth_color='C1', show_titles=True, title_fmt='.2f',
        title_kwargs={'fontsize': 9},
        quantiles=[0.16, 0.5, 0.84],
    )
    fig.suptitle(title, y=1.01, fontsize=10)
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_corners(output_dir: str, triples, censored: bool = False,
                 lae_first: bool = False, bayes_factor: bool = False,
                 two_bub_mock: bool = False) -> None:
    for n_gal, noise, seed in triples:
        tag = f'ngal{n_gal:03d}_noise{noise:.2e}_seed{seed:02d}'
        suffix = (('_censored' if censored else '')
                  + ('_laefirst' if lae_first else '')
                  + ('_2bubmock' if two_bub_mock else ''))

        if bayes_factor:
            fname = os.path.join(output_dir, f'bf_{tag}{suffix}.npz')
        else:
            fname = params_to_filename(n_gal, noise, seed, output_dir,
                                       censored=censored, lae_first=lae_first,
                                       two_bub_mock=two_bub_mock)

        if not os.path.exists(fname):
            print(f"Not found: {fname}")
            continue

        d     = dict(np.load(fname, allow_pickle=True))
        n_det = int(d.get('n_detected', -1))
        det_str = (f'  detected: {n_det}/{n_gal} ({100*n_det/n_gal:.0f}%)'
                   if n_det >= 0 else '')

        # M1 corner
        _make_corner(
            samples   = d['posterior_samples'],
            labels    = PARAM_NAMES,
            truths    = d['true_mu'],
            title     = f'M1 (1 bubble)  n_gal={n_gal}, noise={noise:.1e}, seed={seed}{det_str}',
            out_path  = os.path.join(output_dir, f'corner_m1_{tag}{suffix}.png'),
        )

        # M2 corner — only present in BF files
        if 'posterior_samples_m2' in d:
            stored = d.get('true_mu_m2', np.full(8, np.nan))
            truths_m2 = stored if not np.all(np.isnan(stored)) \
                else np.concatenate([d['true_mu'], np.full(4, np.nan)])
            log_bf    = float(d['log_bf'])
            _make_corner(
                samples   = d['posterior_samples_m2'],
                labels    = PARAM_NAMES_2BUB,
                truths    = truths_m2,
                title     = (f'M2 (2 bubbles)  n_gal={n_gal}, noise={noise:.1e}, seed={seed}{det_str}\n'
                             f'log BF(M2/M1) = {log_bf:.2f}  →  BF = {np.exp(log_bf):.1f}'
                             f'  ({"M2 preferred" if log_bf > 0 else "M1 preferred"})'),
                out_path  = os.path.join(output_dir, f'corner_m2_{tag}{suffix}.png'),
            )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gal',      type=int,   default=None)
    parser.add_argument('--noise',      type=float, default=None)
    parser.add_argument('--seed',       type=int,   default=None)
    parser.add_argument('--n_seeds',    type=int,   default=5,
                        help='Seeds per combination (0..n_seeds-1).')
    parser.add_argument('--n_gal_list', type=int,   nargs='+', default=None,
                        help='Subset of N_GAL_GRID to run, e.g. --n_gal_list 10 30 50 70. '
                             'Determines job array size: len(n_gal_list)*6*n_seeds.')
    parser.add_argument('--job_id',     type=int,   default=None,
                        help='Maps to (n_gal, noise, seed) given n_seeds and n_gal_list.')
    parser.add_argument('--all',        action='store_true',
                        help='Run full grid sequentially')
    parser.add_argument('--plot_only',       action='store_true')
    parser.add_argument('--plot_n_gal_list', type=int, nargs='+', default=None,
                        help='Only include these N values in the summary plot.')
    parser.add_argument('--output_dir', type=str,   default='prod_results')
    parser.add_argument('--n_workers',  type=int,   default=N_WORKERS)
    parser.add_argument('--censored',   action='store_true',
                        help='Non-detected galaxies use censored (upper-limit) likelihood.')
    parser.add_argument('--lae_first',  action='store_true',
                        help='Condition mock on first galaxy having EW_eff > 25Å.')
    parser.add_argument('--corner',     action='store_true',
                        help='Make corner plots for --n_gal, --noise, --corner_seeds.')
    parser.add_argument('--corner_bf',  action='store_true',
                        help='Make M1+M2 corner plots from a BF run file.')
    parser.add_argument('--corner_seeds', type=int, nargs='+', default=None,
                        help='Seeds to plot corners for (defaults to --seed if set, else 0).')
    parser.add_argument('--bayes_factor', action='store_true',
                        help='Run M1 (1 bubble) + M2 (2 bubbles) on same mock; compute BF.')
    parser.add_argument('--two_bub_mock', action='store_true',
                        help='Generate mock from two bubbles (TRUE_2BUB_MU). '
                             'Use with --bayes_factor or alone.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.corner_bf:
        if args.n_gal is None or args.noise is None:
            parser.error('--corner_bf requires --n_gal and --noise')
        seeds = args.corner_seeds or ([args.seed] if args.seed is not None else [0])
        plot_corners(args.output_dir,
                     [(args.n_gal, args.noise, s) for s in seeds],
                     censored=args.censored, lae_first=args.lae_first,
                     bayes_factor=True, two_bub_mock=args.two_bub_mock)

    elif args.bayes_factor:
        if args.n_gal is None or args.noise is None or args.seed is None:
            parser.error('--bayes_factor requires --n_gal, --noise, and --seed')
        two_bub_mu = TRUE_2BUB_MU if args.two_bub_mock else None
        out_file = os.path.join(
            args.output_dir,
            f'bf_ngal{args.n_gal:03d}_noise{args.noise:.2e}_seed{args.seed:02d}'
            + ('_censored' if args.censored else '')
            + ('_laefirst' if args.lae_first else '')
            + ('_2bubmock' if args.two_bub_mock else '')
            + '.npz'
        )
        result = run_bayes_factor(
            args.n_gal, args.noise, args.seed,
            n_workers=args.n_workers,
            censored=args.censored, lae_first=args.lae_first,
            two_bub_mu=two_bub_mu,
        )
        np.savez(out_file, **result)
        print(f"Saved {out_file}", flush=True)

    elif args.corner:
        if args.n_gal is None or args.noise is None:
            parser.error('--corner requires --n_gal and --noise')
        seeds = args.corner_seeds or ([args.seed] if args.seed is not None else [0])
        plot_corners(args.output_dir,
                     [(args.n_gal, args.noise, s) for s in seeds],
                     censored=args.censored, lae_first=args.lae_first,
                     two_bub_mock=args.two_bub_mock)

    elif args.plot_only:
        plot_grid(args.output_dir, n_gal_list=args.plot_n_gal_list)

    else:
        # Build list of (n_gal, noise, seed) triples to run
        gal_grid = args.n_gal_list if args.n_gal_list is not None else N_GAL_GRID
        if args.job_id is not None:
            triples = [job_id_to_params(args.job_id, args.n_seeds, args.n_gal_list)]
        elif args.n_gal is not None and args.noise is not None and args.seed is not None:
            triples = [(args.n_gal, args.noise, args.seed)]
        elif args.all:
            triples = [(n, ns, s)
                       for n in gal_grid
                       for ns in NOISE_GRID
                       for s in range(args.n_seeds)]
        else:
            parser.error('Specify --n_gal + --noise + --seed, --job_id, or --all')

        for n_gal, noise, seed in triples:
            out_file = params_to_filename(n_gal, noise, seed, args.output_dir,
                                          censored=args.censored, lae_first=args.lae_first)
            if os.path.exists(out_file):
                print(f"[n_gal={n_gal}, noise={noise:.2e}] Already done, skipping.",
                      flush=True)
                continue
            result = run_single(n_gal, noise, seed, n_workers=args.n_workers,
                                 censored=args.censored, lae_first=args.lae_first)
            np.savez(out_file, **result)
            print(f"Saved {out_file}", flush=True)

        plot_grid(args.output_dir, n_gal_list=args.plot_n_gal_list)