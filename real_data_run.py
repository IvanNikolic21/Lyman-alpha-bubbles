"""
Bubble inference on a real EW catalog, likelihood computed directly on
equivalent width (no spectral flux binning).

Pipeline: load a fixed-width RA/Dec/zspec/EW catalog -> select a redshift
window (a single bubble can only plausibly explain galaxies in one coeval
slice) -> convert RA/Dec/z to comoving Mpc centered on the selected sample ->
derive the (x, y, z, r_bub) prior box from the data extent -> run dynesty with
a likelihood that compares model-predicted EW (intrinsic EW from the
population p_EW model, attenuated by the bubble's IGM/CGM transmission) to
the observed EW, using a censored (cumulative) likelihood for upper limits.

Usage
-----
python real_data_run.py --catalog table.dat --z_lo 6.8 --z_hi 7.3 \
    --nlive 300 --n_inside_tau 200 --output_dir real_data_results/
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import time
import argparse
import multiprocessing as mp
import types as _types
import numpy as np
import dynesty
from dynesty.utils import resample_equal
from scipy.special import logsumexp, gammaln, stdtr
from astropy.cosmology import Planck18 as Cosmo
from astropy import constants as const
import astropy.units as u

from lyabubbles.speed_up import get_content, calculate_taus_post_batched
from lyabubbles.galaxy_prop import tau_CGM
from lyabubbles.helpers import I
from lyabubbles.real_data import load_catalog, radec_to_comoving, data_driven_priors

# ── Fixed settings ────────────────────────────────────────────────────────────
NDIM         = 4
NDIM_2BUB    = 8
NDIM_3BUB    = 12
N_ITER_BUB   = 1
NU_STUDENT   = 3.0    # Student-t degrees of freedom, matches production_run.py
PARAM_NAMES      = ['x_bub', 'y_bub', 'z_bub', 'r_bub']
PARAM_NAMES_2BUB = ['x1_bub', 'y1_bub', 'z1_bub', 'r1_bub',
                    'x2_bub', 'y2_bub', 'z2_bub', 'r2_bub']
PARAM_NAMES_3BUB = ['x1_bub', 'y1_bub', 'z1_bub', 'r1_bub',
                    'x2_bub', 'y2_bub', 'z2_bub', 'r2_bub',
                    'x3_bub', 'y3_bub', 'z3_bub', 'r3_bub']

# EW upper limits carry no reported error. This sets how "soft" the censored
# cumulative likelihood is, in Angstrom. It is a modeling assumption with no
# data behind it (the catalog gives no error for these rows) -- tune freely.
EW_UL_SCALE  = 5.0

wave_em  = np.linspace(1214, 1225., 100) * u.Angstrom
wave_Lya = 1215.67 * u.Angstrom

# ── Module-level state (populated before fork, inherited by workers) ──────────
_S = _types.SimpleNamespace()


def _prior_transform(u_):
    s = _S
    return s.prior_lo + u_ * (s.prior_hi - s.prior_lo)


def _prior_transform_2bub(u_):
    """Same per-axis box as the 1-bubble prior, doubled, with r1 >= r2 enforced
    via a smooth ordering transform (r1 ~ Uniform(r_min, r_max), r2 ~ Uniform(
    r_min, r1)) rather than a discrete swap. A discrete swap of just the radii
    (leaving positions untouched) folds the unit-cube -> parameter mapping
    discontinuously right where the two independently-drawn radii cross --
    this wrecked dynesty's ellipsoidal bounding once r_max was tightened
    (sampling efficiency collapsed to ~0.04%, >500k calls without converging).
    The ordering transform is continuous everywhere, at the cost of a
    different (but equally valid) marginal prior on r2."""
    s = _S
    p = np.empty(8)
    p[:3]  = s.prior_lo[:3] + u_[:3]  * (s.prior_hi[:3] - s.prior_lo[:3])
    p[4:7] = s.prior_lo[:3] + u_[4:7] * (s.prior_hi[:3] - s.prior_lo[:3])
    r_lo, r_hi = s.prior_lo[3], s.prior_hi[3]
    p[3] = r_lo + u_[3] * (r_hi - r_lo)   # r1 ~ Uniform(r_lo, r_hi)
    p[7] = r_lo + u_[7] * (p[3] - r_lo)   # r2 ~ Uniform(r_lo, r1)  -> r2 <= r1, continuous
    return p


def _prior_transform_3bub(u_):
    """Same per-axis box tripled, with r1 >= r2 >= r3 via the same smooth
    ordering transform as `_prior_transform_2bub`."""
    s = _S
    p = np.empty(12)
    p[:3]   = s.prior_lo[:3] + u_[:3]   * (s.prior_hi[:3] - s.prior_lo[:3])
    p[4:7]  = s.prior_lo[:3] + u_[4:7]  * (s.prior_hi[:3] - s.prior_lo[:3])
    p[8:11] = s.prior_lo[:3] + u_[8:11] * (s.prior_hi[:3] - s.prior_lo[:3])
    r_lo, r_hi = s.prior_lo[3], s.prior_hi[3]
    p[3]  = r_lo + u_[3]  * (r_hi - r_lo)   # r1 ~ Uniform(r_lo, r_hi)
    p[7]  = r_lo + u_[7]  * (p[3] - r_lo)   # r2 ~ Uniform(r_lo, r1)
    p[11] = r_lo + u_[11] * (p[7] - r_lo)   # r3 ~ Uniform(r_lo, r2)
    return p


def _tau_now_for_inside(inside_gals, z_end_bub_arr):
    """Bubble-dependent total tau for the galaxies currently inside a bubble,
    with the same wavelength-monotonicity sanity fix used in production_run.py."""
    s = _S
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
    return np.nan_to_num(tau_now, nan=np.inf)


def _ew_pred_for_inside(inside_gals, tau_now):
    s = _S
    weighted  = s.j_s[inside_gals] * s.tau_cgm[inside_gals][:, np.newaxis, :] * np.exp(-tau_now)
    numerator = np.trapz(weighted, wave_em.value, axis=2)
    t_in      = numerator / s.j_s_trapz_denom[inside_gals]
    return s.ew_int[inside_gals] * t_in


def _ew_loglike_from_pred(ew_pred):
    """Combine model-predicted EW (n_gal, K) into the total log-likelihood,
    using a Student-t detection term and a censored (cumulative) term for
    upper limits, marginalized over the K MC draws via logsumexp."""
    s = _S
    diffs     = s.ew_obs[:, np.newaxis] - ew_pred                 # (n_gal, K)
    _log_norm = (gammaln((NU_STUDENT + 1) / 2) - gammaln(NU_STUDENT / 2)
                 - 0.5 * np.log(np.pi * NU_STUDENT) - np.log(s.ew_err))
    log_p_det = (
        -(NU_STUDENT + 1) / 2 * np.log1p((diffs / s.ew_err[:, np.newaxis]) ** 2 / NU_STUDENT)
        + _log_norm[:, np.newaxis]
    )
    per_gal_det = logsumexp(log_p_det, axis=1) - np.log(s.n_inside_tau)

    z_cdf       = (s.ew_obs[:, np.newaxis] - ew_pred) / EW_UL_SCALE
    log_cdf     = np.log(stdtr(NU_STUDENT, z_cdf))
    per_gal_ul  = logsumexp(log_cdf, axis=1) - np.log(s.n_inside_tau)

    per_gal = np.where(s.is_upper_limit, per_gal_ul, per_gal_det)
    return float(per_gal.sum())


def _log_likelihood_ew(theta):
    s = _S
    xb, yb, zb, rb = theta
    dx = s.x_gal - xb
    dy = s.y_gal - yb
    dz = s.z_gal - zb
    inside        = dx**2 + dy**2 + dz**2 < rb**2
    dist_arr      = np.where(inside, dz + np.sqrt(np.where(inside, rb**2 - dx**2 - dy**2, 0.0)), 0.0)
    z_end_bub_arr = s.redshifts - np.where(inside, dist_arr / s.R_H, 0.0)
    inside_gals   = np.where(inside)[0]

    ew_pred = s.ew_pred_outside.copy()   # (n_gal, K) — default, theta-independent

    if len(inside_gals) > 0:
        tau_now = _tau_now_for_inside(inside_gals, z_end_bub_arr)
        ew_pred[inside_gals] = _ew_pred_for_inside(inside_gals, tau_now)

    return _ew_loglike_from_pred(ew_pred)


def _log_likelihood_ew_2bub(theta):
    s = _S
    x1, y1, z1, r1, x2, y2, z2, r2 = theta

    dx1 = s.x_gal - x1;  dy1 = s.y_gal - y1;  dz1 = s.z_gal - z1
    dx2 = s.x_gal - x2;  dy2 = s.y_gal - y2;  dz2 = s.z_gal - z2

    in1 = dx1**2 + dy1**2 + dz1**2 < r1**2
    in2 = dx2**2 + dy2**2 + dz2**2 < r2**2
    inside = in1 | in2

    dist1 = dz1 + np.sqrt(np.maximum(r1**2 - dx1**2 - dy1**2, 0.0))
    dist2 = dz2 + np.sqrt(np.maximum(r2**2 - dx2**2 - dy2**2, 0.0))

    # Near-face redshift per bubble; inf for galaxies not inside that bubble.
    z_end1 = np.where(in1, s.redshifts - dist1 / s.R_H, np.inf)
    z_end2 = np.where(in2, s.redshifts - dist2 / s.R_H, np.inf)

    # Galaxy inside both bubbles: take whichever near face is closer to the
    # observer (lower redshift -> lower tau_IGM to observer).
    z_end_bub_arr = np.minimum(z_end1, z_end2)

    inside_gals = np.where(inside)[0]

    ew_pred = s.ew_pred_outside.copy()

    if len(inside_gals) > 0:
        tau_now = _tau_now_for_inside(inside_gals, z_end_bub_arr)
        ew_pred[inside_gals] = _ew_pred_for_inside(inside_gals, tau_now)

    return _ew_loglike_from_pred(ew_pred)


def _log_likelihood_ew_3bub(theta):
    s = _S
    x1, y1, z1, r1, x2, y2, z2, r2, x3, y3, z3, r3 = theta

    dx1 = s.x_gal - x1;  dy1 = s.y_gal - y1;  dz1 = s.z_gal - z1
    dx2 = s.x_gal - x2;  dy2 = s.y_gal - y2;  dz2 = s.z_gal - z2
    dx3 = s.x_gal - x3;  dy3 = s.y_gal - y3;  dz3 = s.z_gal - z3

    in1 = dx1**2 + dy1**2 + dz1**2 < r1**2
    in2 = dx2**2 + dy2**2 + dz2**2 < r2**2
    in3 = dx3**2 + dy3**2 + dz3**2 < r3**2
    inside = in1 | in2 | in3

    dist1 = dz1 + np.sqrt(np.maximum(r1**2 - dx1**2 - dy1**2, 0.0))
    dist2 = dz2 + np.sqrt(np.maximum(r2**2 - dx2**2 - dy2**2, 0.0))
    dist3 = dz3 + np.sqrt(np.maximum(r3**2 - dx3**2 - dy3**2, 0.0))

    z_end1 = np.where(in1, s.redshifts - dist1 / s.R_H, np.inf)
    z_end2 = np.where(in2, s.redshifts - dist2 / s.R_H, np.inf)
    z_end3 = np.where(in3, s.redshifts - dist3 / s.R_H, np.inf)

    # Galaxy inside multiple bubbles: take whichever near face is closer to
    # the observer (lower redshift -> lower tau_IGM to observer).
    z_end_bub_arr = np.minimum(np.minimum(z_end1, z_end2), z_end3)

    inside_gals = np.where(inside)[0]

    ew_pred = s.ew_pred_outside.copy()

    if len(inside_gals) > 0:
        tau_now = _tau_now_for_inside(inside_gals, z_end_bub_arr)
        ew_pred[inside_gals] = _ew_pred_for_inside(inside_gals, tau_now)

    return _ew_loglike_from_pred(ew_pred)


def build_state(catalog_path: str, z_lo: float, z_hi: float, n_inside_tau: int,
                z_min: float, muv_max: float, main_dir: str,
                r_max: float = None) -> dict:
    """Load the catalog, select the redshift window, convert coordinates,
    build the data-driven prior, and populate `_S` with everything the
    likelihood needs. Returns the (x, y, z, prior_lo, prior_hi, ra0, dec0, z0)
    metadata needed to interpret/rerun the fit."""
    cat = load_catalog(catalog_path, z_min=z_min, muv_max=muv_max)

    z_lo_eff  = -np.inf if z_lo is None else z_lo
    in_window = (cat.redshift >= z_lo_eff) & (cat.redshift <= z_hi)
    n_gal = int(in_window.sum())
    print(f"[build_state] z-window [{z_lo}, {z_hi}]: {n_gal} galaxies "
          f"(of {len(cat.redshift)} kept after catalog sanity filtering).", flush=True)
    if n_gal < 2:
        raise ValueError(f"Only {n_gal} galaxies in the requested z-window — "
                         f"need at least a few to constrain a bubble.")

    muv       = cat.muv[in_window]
    ew_obs    = cat.ew[in_window]
    ew_err    = cat.ew_err[in_window]
    is_ul     = cat.is_upper_limit[in_window]
    redshifts = cat.redshift[in_window]
    beta      = np.full(n_gal, -2.0)

    x_gal, y_gal, z_gal, ra0, dec0, z0, x_mean, y_mean, z_mean = radec_to_comoving(
        cat.ra[in_window], cat.dec[in_window], redshifts
    )
    prior_lo, prior_hi = data_driven_priors(x_gal, y_gal, z_gal, r_max=r_max)
    print(f"[build_state] field center: ra0={ra0:.5f} dec0={dec0:.5f} z0={z0:.4f}", flush=True)
    print(f"[build_state] prior box: x in [{prior_lo[0]:.2f}, {prior_hi[0]:.2f}], "
          f"y in [{prior_lo[1]:.2f}, {prior_hi[1]:.2f}], "
          f"z in [{prior_lo[2]:.2f}, {prior_hi[2]:.2f}] Mpc, "
          f"r_bub in [{prior_lo[3]:.2f}, {prior_hi[3]:.2f}] Mpc", flush=True)
    print(f"[build_state] {is_ul.sum()} upper limits, {(~is_ul).sum()} detections "
          f"in this window.", flush=True)

    cont_filled = get_content(
        muv, redshifts, x_gal, y_gal, z_gal,
        beta=beta, n_iter_bub=N_ITER_BUB, n_inside_tau=n_inside_tau,
        include_muv_unc=False, fwhm_true=False,
        redshift=z0, xh_unc=True, high_prob_emit=False,
        EW_fixed=False, cache=None, AH22_model=False,
        main_dir=main_dir, cache_dir=None, gauss_distr=False,
    )

    R_H      = np.array([(const.c / Cosmo.H(redshifts[i])).to(u.Mpc).value for i in range(n_gal)])
    tau_cgm  = np.array([tau_CGM(muv[i], main_dir=main_dir) for i in range(n_gal)])
    j_s      = np.array([cont_filled.j_s_full[i] for i in range(n_gal)])
    la_flux  = np.array([cont_filled.la_flux_out_full[i] for i in range(n_gal)])

    r_alpha_val = 6.25e8 / (4 * np.pi * (const.c / wave_Lya).to(u.Hz).value)
    tau_gp      = 7.16e5 * ((1 + redshifts) / 10) ** 1.5
    tau_wv_pref = tau_gp * r_alpha_val / np.pi * 0.65
    z_wv        = wave_em.value[np.newaxis, :] / 1216 * (1 + redshifts[:, np.newaxis]) - 1
    I_z_end     = I((1 + 5.3) / (1 + z_wv))
    ooz         = 1215.67 / (wave_em.value[np.newaxis, :] * (1 + redshifts[:, np.newaxis]))
    red_up_arr  = np.array([cont_filled.first_bubble_encounter_redshift_up_full[i] for i in range(n_gal)])
    I_red_up    = I((1 + red_up_arr[:, :, np.newaxis]) * ooz[:, np.newaxis, :])

    tau_prec = np.array([cont_filled.tau_prec_full[i] for i in range(n_gal)])
    # calculate_taus_prep (lyabubbles/speed_up.py) can hand back raw nan/-inf for
    # pathological sightline geometries (seen with real galaxy parameters,
    # not exercised by the validated synthetic-mock regime) -- sanitize using
    # the same "bad tau -> +inf (fully absorbed)" convention already used
    # for tau_now/tau_out everywhere else in this model.
    tau_prec = np.nan_to_num(tau_prec, nan=np.inf, posinf=np.inf, neginf=0.0)
    tau_prec[tau_prec < 0] = np.inf
    z_up     = np.array([cont_filled.first_bubble_encounter_coord_z_up_full[i] for i in range(n_gal)])
    red_up   = red_up_arr
    z_lo     = np.array([cont_filled.first_bubble_encounter_coord_z_lo_full[i] for i in range(n_gal)])
    red_lo   = np.array([cont_filled.first_bubble_encounter_redshift_lo_full[i] for i in range(n_gal)])

    # ── Baseline ("outside any bubble") tau and EW prediction, theta-independent ──
    tau_post_out = calculate_taus_post_batched(
        redshifts, redshifts, z_up.copy(), red_up, z_lo.copy(), red_lo,
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
    tau_total_outside = tau_prec + tau_out

    # ── Intrinsic EW per (galaxy, MC draw), recovered exactly from the
    # luminosity draw already used to build `la_flux` (see lyabubbles.galaxy_prop.p_EW:
    # lum_alpha = EW * C_const * L_UV_mean, with C_const/L_UV_mean deterministic
    # functions of Muv/beta only) ──────────────────────────────────────────────
    c_const    = (2.47e15 / 1216.0) * (1500.0 / 1216.0) ** (-beta - 2)
    l_uv_mean  = 10 ** (-0.4 * (muv - 51.6))
    ew_int     = la_flux / (c_const[:, np.newaxis] * l_uv_mean[:, np.newaxis])

    j_s_trapz_denom   = np.trapz(j_s, wave_em.value, axis=2)
    weighted_outside  = j_s * tau_cgm[:, np.newaxis, :] * np.exp(-tau_total_outside)
    numerator_outside = np.trapz(weighted_outside, wave_em.value, axis=2)
    t_outside         = numerator_outside / j_s_trapz_denom
    ew_pred_outside   = ew_int * t_outside

    _S.x_gal           = x_gal
    _S.y_gal           = y_gal
    _S.z_gal           = z_gal
    _S.redshifts       = redshifts
    _S.R_H             = R_H
    _S.tau_prec        = tau_prec
    _S.z_up            = z_up
    _S.red_up          = red_up
    _S.z_lo            = z_lo
    _S.red_lo          = red_lo
    _S.z_wv            = z_wv
    _S.tau_wv_pref     = tau_wv_pref
    _S.I_z_end         = I_z_end
    _S.I_red_up        = I_red_up
    _S.j_s             = j_s
    _S.tau_cgm         = tau_cgm
    _S.j_s_trapz_denom = j_s_trapz_denom
    _S.ew_int          = ew_int
    _S.ew_pred_outside = ew_pred_outside
    _S.ew_obs          = ew_obs
    _S.ew_err          = np.where(is_ul, 1.0, ew_err)   # placeholder for unused branch
    _S.is_upper_limit  = is_ul
    _S.n_inside_tau    = n_inside_tau
    _S.prior_lo        = prior_lo
    _S.prior_hi        = prior_hi

    return dict(
        n_gal=n_gal, ra0=ra0, dec0=dec0, z0=z0,
        x_mean=x_mean, y_mean=y_mean, z_mean=z_mean,
        prior_lo=prior_lo, prior_hi=prior_hi,
        x_gal=x_gal, y_gal=y_gal, z_gal=z_gal, redshifts=redshifts,
        ew_obs=ew_obs, ew_err=ew_err, is_upper_limit=is_ul,
    )


def _run_dynesty(loglike, prior_transform, ndim, param_names,
                 nlive: int, dlogz: float, n_workers: int, label: str,
                 sample: str = 'auto') -> dict:
    print(f"[{label}] Running dynesty (nlive={nlive}, dlogz={dlogz}, ndim={ndim}, "
          f"sample={sample})...", flush=True)
    t0 = time.perf_counter()
    with mp.get_context('fork').Pool(n_workers) as pool:
        sampler = dynesty.NestedSampler(
            loglike, prior_transform, ndim=ndim, nlive=nlive,
            pool=pool, queue_size=n_workers, sample=sample,
        )
        sampler.run_nested(print_progress=True, dlogz=dlogz)
    wall_time = time.perf_counter() - t0

    results       = sampler.results
    weights       = np.exp(results.logwt - results.logz[-1])
    equal_samples = resample_equal(results.samples, weights)

    post_mean   = equal_samples.mean(axis=0)
    post_median = np.median(equal_samples, axis=0)
    post_std    = equal_samples.std(axis=0)
    post_p16    = np.percentile(equal_samples, 16, axis=0)
    post_p84    = np.percentile(equal_samples, 84, axis=0)
    post_map    = results.samples[np.argmax(results.logl)]   # max-likelihood point among raw nested samples

    print(f"[{label}] Done in {wall_time:.1f}s", flush=True)
    for _pi, _pn in enumerate(param_names):
        print(f"  {_pn:6s}  median={post_median[_pi]:.3f}  map={post_map[_pi]:.3f}  "
              f"std={post_std[_pi]:.3f}  [{post_p16[_pi]:.2f}, {post_p84[_pi]:.2f}]", flush=True)

    return dict(
        posterior_samples=equal_samples,
        post_mean=post_mean, post_median=post_median, post_std=post_std,
        post_p16=post_p16, post_p84=post_p84, post_map=post_map,
        logz=results.logz[-1], logzerr=results.logzerr[-1],
        ncall=results.ncall.sum(), wall_time=wall_time,
    )


def run(catalog_path: str, z_lo: float, z_hi: float, n_inside_tau: int,
        nlive: int, dlogz: float, n_workers: int, z_min: float, muv_max: float,
        main_dir: str, r_max: float = None) -> dict:
    meta = build_state(catalog_path, z_lo, z_hi, n_inside_tau, z_min, muv_max,
                       main_dir, r_max=r_max)
    fit  = _run_dynesty(_log_likelihood_ew, _prior_transform, NDIM, PARAM_NAMES,
                        nlive, dlogz, n_workers, label='run')
    return dict(**meta, **fit)


def run_bayes_factor(catalog_path: str, z_lo: float, z_hi: float, n_inside_tau: int,
                     nlive: int, dlogz: float, n_workers: int, z_min: float,
                     muv_max: float, main_dir: str, r_max: float = None) -> dict:
    """Fit M1 (1 bubble) and M2 (2 bubbles) to the same galaxy sample and
    compare their Bayesian evidence. `_S` is built once and shared by both
    fits (only the likelihood/prior/ndim differ)."""
    meta = build_state(catalog_path, z_lo, z_hi, n_inside_tau, z_min, muv_max,
                       main_dir, r_max=r_max)

    print("--- Model 1: single bubble ---", flush=True)
    fit1 = _run_dynesty(_log_likelihood_ew, _prior_transform, NDIM, PARAM_NAMES,
                        nlive, dlogz, n_workers, label='M1')

    print("--- Model 2: two bubbles ---", flush=True)
    # 'rslice' (slice sampling) handles the narrow, mostly-empty 8D likelihood
    # here far better than the default -- the default collapsed to <0.2%
    # sampling efficiency once r_bub's ceiling was tightened to a physical scale.
    fit2 = _run_dynesty(_log_likelihood_ew_2bub, _prior_transform_2bub, NDIM_2BUB,
                        PARAM_NAMES_2BUB, nlive, dlogz, n_workers, label='M2',
                        sample='rslice')

    log_bf = float(fit2['logz']) - float(fit1['logz'])
    print(f"\n  log Z(M1) = {float(fit1['logz']):.2f} +/- {float(fit1['logzerr']):.2f}", flush=True)
    print(f"  log Z(M2) = {float(fit2['logz']):.2f} +/- {float(fit2['logzerr']):.2f}", flush=True)
    print(f"  log BF(M2/M1) = {log_bf:.2f}  ->  BF = {np.exp(log_bf):.2f}  "
          f"({'M2 preferred' if log_bf > 0 else 'M1 preferred'})", flush=True)

    return dict(
        **meta,
        posterior_samples=fit1['posterior_samples'],
        post_mean=fit1['post_mean'], post_median=fit1['post_median'],
        post_std=fit1['post_std'], post_p16=fit1['post_p16'],
        post_p84=fit1['post_p84'], post_map=fit1['post_map'],
        logz=fit1['logz'], logzerr=fit1['logzerr'], ncall=fit1['ncall'],
        wall_time=fit1['wall_time'] + fit2['wall_time'],
        posterior_samples_m2=fit2['posterior_samples'],
        post_mean_m2=fit2['post_mean'], post_median_m2=fit2['post_median'],
        post_std_m2=fit2['post_std'], post_p16_m2=fit2['post_p16'],
        post_p84_m2=fit2['post_p84'], post_map_m2=fit2['post_map'],
        logz_m2=fit2['logz'], logzerr_m2=fit2['logzerr'],
        log_bf=log_bf,
    )


def run_model_comparison(catalog_path: str, z_lo: float, z_hi: float, n_inside_tau: int,
                         nlive: int, dlogz: float, n_workers: int, z_min: float,
                         muv_max: float, main_dir: str, r_max: float = None) -> dict:
    """Fit M1 (1 bubble), M2 (2 bubbles), and M3 (3 bubbles) to the same galaxy
    sample and compare their Bayesian evidence pairwise."""
    meta = build_state(catalog_path, z_lo, z_hi, n_inside_tau, z_min, muv_max,
                       main_dir, r_max=r_max)

    print("--- Model 1: single bubble ---", flush=True)
    fit1 = _run_dynesty(_log_likelihood_ew, _prior_transform, NDIM, PARAM_NAMES,
                        nlive, dlogz, n_workers, label='M1')

    print("--- Model 2: two bubbles ---", flush=True)
    fit2 = _run_dynesty(_log_likelihood_ew_2bub, _prior_transform_2bub, NDIM_2BUB,
                        PARAM_NAMES_2BUB, nlive, dlogz, n_workers, label='M2',
                        sample='rslice')

    print("--- Model 3: three bubbles ---", flush=True)
    fit3 = _run_dynesty(_log_likelihood_ew_3bub, _prior_transform_3bub, NDIM_3BUB,
                        PARAM_NAMES_3BUB, nlive, dlogz, n_workers, label='M3',
                        sample='rslice')

    log_bf_21 = float(fit2['logz']) - float(fit1['logz'])
    log_bf_32 = float(fit3['logz']) - float(fit2['logz'])
    log_bf_31 = float(fit3['logz']) - float(fit1['logz'])
    print(f"\n  log Z(M1) = {float(fit1['logz']):.2f} +/- {float(fit1['logzerr']):.2f}", flush=True)
    print(f"  log Z(M2) = {float(fit2['logz']):.2f} +/- {float(fit2['logzerr']):.2f}", flush=True)
    print(f"  log Z(M3) = {float(fit3['logz']):.2f} +/- {float(fit3['logzerr']):.2f}", flush=True)
    print(f"  log BF(M2/M1) = {log_bf_21:.2f}  ->  BF = {np.exp(log_bf_21):.2f}", flush=True)
    print(f"  log BF(M3/M2) = {log_bf_32:.2f}  ->  BF = {np.exp(log_bf_32):.2f}", flush=True)
    print(f"  log BF(M3/M1) = {log_bf_31:.2f}  ->  BF = {np.exp(log_bf_31):.2f}", flush=True)
    best = max([('M1', float(fit1['logz'])), ('M2', float(fit2['logz'])),
               ('M3', float(fit3['logz']))], key=lambda t: t[1])[0]
    print(f"  Highest evidence: {best}", flush=True)

    return dict(
        **meta,
        posterior_samples=fit1['posterior_samples'],
        post_mean=fit1['post_mean'], post_median=fit1['post_median'],
        post_std=fit1['post_std'], post_p16=fit1['post_p16'],
        post_p84=fit1['post_p84'], post_map=fit1['post_map'],
        logz=fit1['logz'], logzerr=fit1['logzerr'], ncall=fit1['ncall'],
        wall_time=fit1['wall_time'] + fit2['wall_time'] + fit3['wall_time'],
        posterior_samples_m2=fit2['posterior_samples'],
        post_mean_m2=fit2['post_mean'], post_median_m2=fit2['post_median'],
        post_std_m2=fit2['post_std'], post_p16_m2=fit2['post_p16'],
        post_p84_m2=fit2['post_p84'], post_map_m2=fit2['post_map'],
        logz_m2=fit2['logz'], logzerr_m2=fit2['logzerr'],
        posterior_samples_m3=fit3['posterior_samples'],
        post_mean_m3=fit3['post_mean'], post_median_m3=fit3['post_median'],
        post_std_m3=fit3['post_std'], post_p16_m3=fit3['post_p16'],
        post_p84_m3=fit3['post_p84'], post_map_m3=fit3['post_map'],
        logz_m3=fit3['logz'], logzerr_m3=fit3['logzerr'],
        log_bf_21=log_bf_21, log_bf_32=log_bf_32, log_bf_31=log_bf_31,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog', type=str, required=True)
    parser.add_argument('--z_lo', type=float, default=None,
                        help='Lower edge of the redshift window for this bubble fit '
                             '(default: no extra lower cut beyond --z_min).')
    parser.add_argument('--z_hi', type=float, default=7.3,
                        help='Upper edge of the redshift window for this bubble fit. '
                             'Default 7.3 selects the main galaxy overdensity in this catalog.')
    parser.add_argument('--z_min', type=float, default=5.0,
                        help='Catalog-wide sanity filter (drop zspec <= z_min).')
    parser.add_argument('--muv_max', type=float, default=-18.0,
                        help='Catalog-wide sanity filter (drop muv >= muv_max).')
    parser.add_argument('--r_max', type=float, default=None,
                        help='Explicit r_bub prior ceiling (Mpc). Default: derived from '
                             'the transverse (x,y) extent only, NOT the line-of-sight depth '
                             '(a deep spectroscopic survey\'s LOS extent is typically far '
                             'larger than any plausible single bubble).')
    parser.add_argument('--n_inside_tau', type=int, default=200)
    parser.add_argument('--nlive', type=int, default=300)
    parser.add_argument('--dlogz', type=float, default=0.5)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--main_dir', type=str,
                        default='/groups/astro/ivannik/programs/Lyman-alpha-bubbles/')
    parser.add_argument('--output_dir', type=str, default='real_data_results')
    parser.add_argument('--corner', action='store_true')
    parser.add_argument('--bayes_factor', action='store_true',
                        help='Also fit a 2-bubble model and compare evidence against '
                             'the 1-bubble model (M2 vs M1).')
    parser.add_argument('--three_bubble', action='store_true',
                        help='Fit 1-, 2-, and 3-bubble models and compare evidence '
                             'pairwise (M2/M1, M3/M2, M3/M1). Supersedes --bayes_factor.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    z_lo_tag = f'{args.z_lo:.2f}' if args.z_lo is not None else 'min'
    fname_stem = f'z{z_lo_tag}-{args.z_hi:.2f}'

    if args.three_bubble:
        result = run_model_comparison(
            args.catalog, args.z_lo, args.z_hi, args.n_inside_tau,
            args.nlive, args.dlogz, args.n_workers, args.z_min, args.muv_max,
            args.main_dir, r_max=args.r_max,
        )
        out_file = os.path.join(args.output_dir, f'mc_real_data_{fname_stem}.npz')
    elif args.bayes_factor:
        result = run_bayes_factor(
            args.catalog, args.z_lo, args.z_hi, args.n_inside_tau,
            args.nlive, args.dlogz, args.n_workers, args.z_min, args.muv_max,
            args.main_dir, r_max=args.r_max,
        )
        out_file = os.path.join(args.output_dir, f'bf_real_data_{fname_stem}.npz')
    else:
        result = run(
            args.catalog, args.z_lo, args.z_hi, args.n_inside_tau,
            args.nlive, args.dlogz, args.n_workers, args.z_min, args.muv_max,
            args.main_dir, r_max=args.r_max,
        )
        out_file = os.path.join(args.output_dir, f'real_data_{fname_stem}.npz')

    np.savez(out_file, **result)
    print(f"Saved {out_file}", flush=True)

    if args.corner:
        import corner
        import matplotlib.pyplot as plt

        fig = corner.corner(
            result['posterior_samples'], labels=PARAM_NAMES,
            show_titles=True, title_fmt='.2f', quantiles=[0.16, 0.5, 0.84],
        )
        fig.suptitle(f"M1 (1 bubble)  z in [{args.z_lo}, {args.z_hi}], "
                     f"n_gal={result['n_gal']}", y=1.01)
        corner_path = out_file.replace('.npz', '_corner_m1.png')
        fig.savefig(corner_path, bbox_inches='tight', dpi=150)
        print(f"Saved {corner_path}", flush=True)

        if 'posterior_samples_m2' in result:
            if args.three_bubble:
                log_bf_label = f"log BF(M2/M1) = {float(result['log_bf_21']):.2f}"
            else:
                log_bf = float(result['log_bf'])
                log_bf_label = (f"log BF(M2/M1) = {log_bf:.2f}  ->  BF = {np.exp(log_bf):.1f}  "
                               f"({'M2 preferred' if log_bf > 0 else 'M1 preferred'})")
            fig2 = corner.corner(
                result['posterior_samples_m2'], labels=PARAM_NAMES_2BUB,
                show_titles=True, title_fmt='.2f', quantiles=[0.16, 0.5, 0.84],
            )
            fig2.suptitle(
                f"M2 (2 bubbles)  z in [{args.z_lo}, {args.z_hi}], n_gal={result['n_gal']}\n"
                f"{log_bf_label}", y=1.03,
            )
            corner_path2 = out_file.replace('.npz', '_corner_m2.png')
            fig2.savefig(corner_path2, bbox_inches='tight', dpi=150)
            print(f"Saved {corner_path2}", flush=True)

        if 'posterior_samples_m3' in result:
            fig3 = corner.corner(
                result['posterior_samples_m3'], labels=PARAM_NAMES_3BUB,
                show_titles=True, title_fmt='.2f', quantiles=[0.16, 0.5, 0.84],
            )
            fig3.suptitle(
                f"M3 (3 bubbles)  z in [{args.z_lo}, {args.z_hi}], n_gal={result['n_gal']}\n"
                f"log BF(M3/M2) = {float(result['log_bf_32']):.2f}   "
                f"log BF(M3/M1) = {float(result['log_bf_31']):.2f}", y=1.03,
            )
            corner_path3 = out_file.replace('.npz', '_corner_m3.png')
            fig3.savefig(corner_path3, bbox_inches='tight', dpi=150)
            print(f"Saved {corner_path3}", flush=True)