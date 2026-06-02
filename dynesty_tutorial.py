"""
Dynesty tutorial: inferring the mean of a 2D Gaussian.

Install: pip install dynesty corner matplotlib

The three things you need for any dynesty problem:
  1. prior_transform(u)  -- maps [0,1]^N to your parameter space
  2. log_likelihood(theta) -- evaluates log p(data | theta)
  3. Run the sampler, extract weighted samples and log-evidence

This toy problem has an analytic answer, so you can immediately
verify that dynesty got it right.
"""

import os
# Limit BLAS/OpenMP to 1 thread per process — must be set before numpy is imported.
# With fork-based multiprocessing we get N_WORKERS processes, each single-threaded,
# which is far better than N_WORKERS × N_blas_threads threads all fighting for the same cores.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import warnings
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import dynesty
warnings.filterwarnings("ignore", category=DeprecationWarning)
from venv.speed_up import get_content, calculate_taus_post_batched
from dynesty import plotting as dyplot
from dynesty.utils import resample_equal
from astropy.cosmology import Planck18 as Cosmo
from astropy import constants as const
import astropy.units as u
from venv.galaxy_prop import get_muv, get_mock_data, get_js, tau_CGM, p_EW
from venv.helpers import full_res_flux, perturb_flux, z_at_proper_distance, I
from scipy.special import logsumexp
# ── Ground truth and fake data ────────────────────────────────────────────────

TRUE_MU = np.array([0,0,0,10])       # what we want to recover
#SIGMA   = np.array([0.5,  0.8])       # known measurement noise (per axis)
N_DATA  = 50   # 50 for production; 10 for a quick test run
main_dir='/groups/astro/ivannik/programs/Lyman-alpha-bubbles/'
wave_em = np.linspace(1214, 1225., 100) * u.Angstrom
wave_Lya = 1215.67 * u.Angstrom


rng  = np.random.default_rng(42)

Muv_mock = np.ones(N_DATA) * -18.5 #get_muv( #TODO:UVLF calculation needs to be improved
                #     n_gal=N_DATA,
                #     redshift=7.5,
                #     muv_cut=-18,
                # )
beta = -2.0 * np.ones(N_DATA)
tau_mock, x_gal_mock, y_gal_mock, z_gal_mock, x_bubbles_mock, y_bubbles_mock, z_bubbles_mock, r_bubs_mock = get_mock_data(
                n_gal=N_DATA,
                z_start=7.5,
                r_bubble=10,
                dist=15,
)

redshifts_of_mocks = np.zeros((N_DATA))
for i in range(N_DATA):
    red_s = z_at_proper_distance(
        - z_gal_mock[i] / (1 + 7.5) * u.Mpc, 7.5
    )
    redshifts_of_mocks[i] = red_s

tau_data_I = []
one_J = get_js(   #TODO: reformat this function
    z=7.5,        #TODO: get rid of implicit redshift dependence through inputs
    muv=Muv_mock,
    n_iter=N_DATA,
)
area_factor = np.array(
    [
        np.trapz(
            one_J[0][i_gal] * tau_CGM(Muv_mock[i_gal], main_dir=main_dir),
            wave_em.value
        ) / np.trapz(
            one_J[0][i_gal],
            wave_em.value
        ) for i_gal in range(N_DATA)
    ]
)

for i in range(len(tau_mock)):
    eit = np.exp(-tau_mock[i])
    tau_cgm_gal = tau_CGM(Muv_mock[i], main_dir=main_dir)
    tau_data_I.append(
        np.trapz(
            eit * tau_cgm_gal * one_J[0][i],
            wave_em.value)
    )

ew_factor, la_e = p_EW(
    Muv_mock.flatten(),
    beta.flatten(),
)

ew_factor = ew_factor.reshape((np.shape(Muv_mock)))
ew_factor_orig = np.copy(ew_factor)
ew_factor /= area_factor
la_e = la_e.reshape((np.shape(Muv_mock)))
la_e_orig = np.copy(la_e)
la_e /= area_factor  # new improvement
data = np.array(tau_data_I)

flux_noise_mock = np.zeros(
    (
        N_DATA,
        11
    )
)
one_J = one_J[0]
continuum = (
        la_e[:, np.newaxis] * one_J[:N_DATA,:] * np.exp(-tau_mock) * tau_CGM(
    Muv_mock, main_dir=main_dir) / (
                4 * np.pi * Cosmo.luminosity_distance(
            7.5
        ).to(u.cm).value ** 2)
)

full_flux_res = full_res_flux(continuum, 7.5)
flux_nonoise_save = np.copy(full_flux_res)

full_flux_res += np.random.normal(
    0,
    5e-20,
    np.shape(full_flux_res)
)
flux_noise_mock[:,  :] = perturb_flux(
    full_flux_res, 11
)
flux_mock = la_e / (
        4 * np.pi * Cosmo.luminosity_distance(
    redshifts_of_mocks).to(u.cm).value ** 2
)
flux_tau = flux_mock * tau_data_I
flux_tau += np.random.normal(0, 5e-20, np.shape(flux_tau))
#data = rng.normal(TRUE_MU, SIGMA, size=(N_DATA, 2))   # shape (N_DATA, 2)

NDIM = 4

# ── Prior ─────────────────────────────────────────────────────────────────────
#
# dynesty always works in the unit hypercube internally.
# prior_transform converts a point u in [0,1]^N to physical parameters.
# Here we use a wide uniform prior on each axis.

PRIOR_LO = np.array([-10.0, -10.0, -10.0, 1])
PRIOR_HI = np.array([ 10.0,  10.0,  10.0, 20])

def prior_transform(u):
    """Uniform prior: [0,1]^2  ->  [PRIOR_LO, PRIOR_HI]."""
    return PRIOR_LO + u * (PRIOR_HI - PRIOR_LO)

# ── Likelihood ────────────────────────────────────────────────────────────────


cont_filled = get_content(
    Muv_mock.flatten(),
    redshifts_of_mocks,
    x_gal_mock,
    y_gal_mock,
    z_gal_mock,
    n_iter_bub=1,
    n_inside_tau=50,
    include_muv_unc=False,
    fwhm_true=False,
    redshift=7.5,
    xh_unc=True,
    high_prob_emit=False,
    EW_fixed=False,
    cache=None,
    AH22_model=False,
    main_dir=main_dir,
    cache_dir=None,
    gauss_distr=False,
    #Tang_distr=False,
)

N_BINS = 11         # fixed spectral bins matching flux_noise_mock
NOISE = 5e-20       # per-pixel noise added to model spectra
N_ITER_BUB = 1     # must match get_content call above
N_INSIDE_TAU = 50  # must match get_content call above
N_WORKERS = 24      # parallel likelihood evaluations via multiprocessing

# ── Quantities that don't depend on bubble params — precompute once ───────────

# Full-resolution spectral grid (fixed redshift = 7.5)
_spec_res = wave_Lya.value * (1 + 7.5) / 2700
_full_bins = np.arange(
    wave_em.value[0] * (1 + 7.5),
    wave_em.value[-1] * (1 + 7.5),
    _spec_res,
)
_max_bins_full = len(_full_bins)

# Hubble radius per galaxy: R_H = c/H(z) in Mpc.
# Used to inline z_at_proper_distance without calling Cosmo.H() on every likelihood call.
_R_H_per_gal = np.array([
    (const.c / Cosmo.H(redshifts_of_mocks[i])).to(u.Mpc).value
    for i in range(N_DATA)
])

# CGM transmission per galaxy — depends only on fixed Muv, not on bubble params
_tau_cgm_per_gal = np.array([
    tau_CGM(Muv_mock[i], main_dir=main_dir)
    for i in range(N_DATA)
])  # (N_DATA, 100)

# Line-profile arrays stacked for all galaxies
_j_s_per_gal = np.array([
    cont_filled.j_s_full[i]
    for i in range(N_DATA)
])  # (N_DATA, N_SAMPLES, 100)

# Area factors per galaxy — ∫(J·τ_CGM)/∫J, also fixed
_raw_af = np.array([
    np.trapz(_j_s_per_gal[i] * _tau_cgm_per_gal[i], wave_em.value, axis=1) /
    np.trapz(_j_s_per_gal[i], wave_em.value, axis=1)
    for i in range(N_DATA)
])  # (N_DATA, N_SAMPLES)
_area_factor_per_gal = np.where(_raw_af < 1e-20, 1e-5, _raw_af)

# Observed flux per galaxy/bin — used directly in the Gaussian likelihood
_obs_flux_per_gal = flux_noise_mock  # (N_DATA, N_BINS)

# Fixed parts of tau_wv — lets calculate_taus_post_batched skip all Cosmo.H calls
_r_alpha_val = 6.25e8 / (4 * np.pi * (const.c / wave_Lya).to(u.Hz).value)
_tau_gp_per_gal = 7.16e5 * ((1 + redshifts_of_mocks) / 10) ** 1.5                # (N_DATA,)
_tau_wv_pref_per_gal = _tau_gp_per_gal * _r_alpha_val / np.pi * 0.65             # (N_DATA,), nf=0.65
_z_wv_per_gal = wave_em.value[np.newaxis, :] / 1216 * (1 + redshifts_of_mocks[:, np.newaxis]) - 1  # (N_DATA, 100)
_I_z_end_per_gal = I((1 + 5.3) / (1 + _z_wv_per_gal))                            # (N_DATA, 100)

# Trapz-weight matrix: continuum (N, 100) @ _trapz_weights → (N, _max_bins_full)
# Replaces the per-bin loop inside full_res_flux — one matmul per likelihood call.
_wave_em_dig_full = np.digitize(wave_em.value * (1 + 7.5), _full_bins)
_trapz_weights = np.zeros((100, _max_bins_full))
for _i in range(_max_bins_full):
    _idx = np.where(_wave_em_dig_full == _i + 1)[0]
    if len(_idx) < 2:
        continue
    _x = wave_em.value[_idx]
    _w = np.empty(len(_idx))
    _w[0] = (_x[1] - _x[0]) / 2
    _w[-1] = (_x[-1] - _x[-2]) / 2
    if len(_idx) > 2:
        _w[1:-1] = (_x[2:] - _x[:-2]) / 2
    _trapz_weights[_idx, _i] = _w

# Rebinning matrix: flux (N, _max_bins_full) @ _rebin_matrix → (N, N_BINS)
# Replaces the per-bin list comprehension inside perturb_flux — one matmul per likelihood call.
_bins_rebin = np.linspace(
    wave_em.value[0] * (1 + 7.5),
    wave_em.value[-1] * (1 + 7.5),
    N_BINS + 1,
)
_wave_em_dig_rebin = np.digitize(_full_bins, _bins_rebin)
_rebin_matrix = (
    _wave_em_dig_rebin[:, np.newaxis] == np.arange(1, N_BINS + 1)[np.newaxis, :]
).astype(float)  # (_max_bins_full, N_BINS)

# Combined trapz→rebin in one (100, N_BINS) matrix — skips the intermediate full-res array entirely
_direct_matrix = _trapz_weights @ _rebin_matrix  # (100, N_BINS)
# Each final bin sums M full-res pixels; summing M iid N(0,σ²) gives N(0,Mσ²)
_noise_per_bin = np.sqrt(_rebin_matrix.sum(axis=0)) * NOISE  # (N_BINS,)
_M_per_bin = _rebin_matrix.sum(axis=0).astype(int)
_sigma_m_floor = (10 / np.log(10)) * _noise_per_bin / ADDITIVE  # noise in mag at faint limit
print(f"Full-res pixels per bin:        {_M_per_bin}", flush=True)
print(f"Noise per bin (flux):           {_noise_per_bin}", flush=True)
print(f"Mag noise at faint limit:       {np.round(_sigma_m_floor, 4)}", flush=True)
print(f"sigma_m / BW_KDE (faint limit): {np.round(_sigma_m_floor / BW_KDE, 3)}", flush=True)

# Stacked cont_filled arrays — avoids per-galaxy attribute lookups inside the likelihood
_tau_prec_all    = np.array([cont_filled.tau_prec_full[i] for i in range(N_DATA)])                             # (N_DATA, N_INSIDE_TAU, 100)
_z_up_all        = np.array([cont_filled.first_bubble_encounter_coord_z_up_full[i] for i in range(N_DATA)])    # (N_DATA, N_INSIDE_TAU)
_red_up_all      = np.array([cont_filled.first_bubble_encounter_redshift_up_full[i] for i in range(N_DATA)])   # (N_DATA, N_INSIDE_TAU)
_z_lo_all        = np.array([cont_filled.first_bubble_encounter_coord_z_lo_full[i] for i in range(N_DATA)])    # (N_DATA, N_INSIDE_TAU)
_red_lo_all      = np.array([cont_filled.first_bubble_encounter_redshift_lo_full[i] for i in range(N_DATA)])   # (N_DATA, N_INSIDE_TAU)
_la_flux_out_all = np.array([cont_filled.la_flux_out_full[i] for i in range(N_DATA)])                         # (N_DATA, N_INSIDE_TAU)
_com_fact_all    = np.array([cont_filled.com_fact[i] for i in range(N_DATA)])                                  # (N_DATA,)

# one_over_onepz per galaxy — matches what calculate_taus_post_batched computes from z_sources
_ooz_per_gal = 1215.67 / (wave_em.value[np.newaxis, :] * (1 + redshifts_of_mocks[:, np.newaxis]))            # (N_DATA, 100)
# I((1+red_up)*ooz) is fixed (both red_up and ooz depend only on precomputed sightlines + galaxy redshifts).
# Precomputing it eliminates the O(N_INSIDE_TAU × 100) I() call inside every likelihood evaluation.
_I_red_up_all = I(
    (1 + _red_up_all[:, :, np.newaxis]) * _ooz_per_gal[:, np.newaxis, :]
)  # (N_DATA, N_INSIDE_TAU, 100)

# Fixed part of continuum: everything except exp(-tau_post), the only bubble-dependent factor.
# exp(-tau_prec) is included here so it never runs in the hot path.
_base_continuum = (
    (_la_flux_out_all / _area_factor_per_gal)[:, :, np.newaxis]  # (N_DATA, N_INSIDE_TAU, 1)
    * _j_s_per_gal                                                 # (N_DATA, N_INSIDE_TAU, 100)
    * np.exp(-_tau_prec_all)                                       # (N_DATA, N_INSIDE_TAU, 100)
    * _tau_cgm_per_gal[:, np.newaxis, :]                          # (N_DATA, 1, 100)
    * _com_fact_all[:, np.newaxis, np.newaxis]                     # (N_DATA, 1, 1)
)  # (N_DATA, N_INSIDE_TAU, 100)

# ── Outside-bubble precomputation ──────────────────────────────────────────
# When galaxy g is outside the main bubble, z_end_bubble = z_source (fixed).
# tau_post is then fully fixed per galaxy — precompute it so that at call time
# steps 2-4 are skipped entirely for outside galaxies.
_tau_post_outside = calculate_taus_post_batched(
    redshifts_of_mocks, redshifts_of_mocks,   # z_end = z_source for outside galaxies
    _z_up_all.copy(), _red_up_all,
    _z_lo_all.copy(), _red_lo_all,
    z_per_gal=_z_wv_per_gal,
    tau_wv_pref_per_gal=_tau_wv_pref_per_gal,
    I_z_end_per_gal=_I_z_end_per_gal,
    I_red_up_all=_I_red_up_all,
)  # (N_DATA, N_INSIDE_TAU, 100)
# Apply sanity checks to the outside-case tau now, before baking into the fixed array
_tau_out_check = _tau_post_outside.copy()
_bad_out = np.any(_tau_out_check[:, :, 30:] - _tau_out_check[:, :, 29:-1] > 0, axis=2)
for _g in np.where(np.any(_bad_out, axis=1))[0]:
    _ratio_g = (1 + redshifts_of_mocks[_g]) / (1 + _z_wv_per_gal[_g])
    _tau_out_check[_g, _bad_out[_g]] = np.clip(
        _tau_wv_pref_per_gal[_g] * _ratio_g**1.5 * (I(_ratio_g) - _I_z_end_per_gal[_g]),
        0, np.inf,
    )
_tau_out_check[_tau_out_check < 0] = np.inf
_tau_out_check = np.nan_to_num(_tau_out_check, nan=np.inf)
# _base_continuum already has exp(-tau_prec); multiply by exp(-tau_post_outside)
_base_cont_eit_outside = _base_continuum * np.exp(-_tau_out_check)  # (N_DATA, N_INSIDE_TAU, 100)
del _tau_post_outside, _tau_out_check, _bad_out

# Precompute the rebinned flux for outside-bubble galaxies — the matmul result is fixed.
# At call time only inside-galaxy rows need recomputing, shrinking the matmul by ~N_DATA/n_inside.
_flux_outside = (
    _base_cont_eit_outside.reshape(N_DATA * N_INSIDE_TAU, 100) @ _direct_matrix
).reshape(N_DATA, N_INSIDE_TAU, N_BINS)  # (N_DATA, N_INSIDE_TAU, N_BINS)


import time as _time
_NCALLS = 0

def get_spectral_likelihood(xb, yb, zb, rb):
    """
    Log-likelihood of observed spectra given bubble at (xb, yb, zb, rb).
    All galaxy-axis operations are batched — no Python loop over galaxies.
    """
    global _NCALLS
    _NCALLS += 1
    _profile = (_NCALLS <= 3)
    _t0 = _time.perf_counter() if _profile else None

    def _tick(label):
        if _profile:
            print(f"  [{_NCALLS}] {label}: {(_time.perf_counter()-_t0)*1e3:.1f} ms", flush=True)

    # ── 1. Bubble geometry — identify which galaxies are inside ───────────────
    dx = x_gal_mock - xb                                       # (N_DATA,)
    dy = y_gal_mock - yb
    dz = z_gal_mock - zb
    inside = dx**2 + dy**2 + dz**2 < rb**2
    dist_arr = np.where(inside, dz + np.sqrt(np.where(inside, rb**2 - dx**2 - dy**2, 0.0)), 0.0)
    z_end_bub_arr = redshifts_of_mocks - np.where(inside, dist_arr / _R_H_per_gal, 0.0)  # (N_DATA,)
    inside_gals = np.where(inside)[0]
    _tick("1-geometry")

    # ── 2-4. IGM tau + continuum — only inside-bubble galaxies need fresh work ─
    # Outside galaxies: tau_post is fixed (z_end=z_source) — use _base_cont_eit_outside directly.
    # Inside galaxies:  compute tau_post for the n_inside slice only, overwrite those rows.
    if len(inside_gals) == 0:
        continuum_all = _base_cont_eit_outside                 # no allocation — direct reference
        _tick("2-calculate_taus_post")
        _tick("3-tau_sanity")
        _tick("4-continuum")
    else:
        tau_post_in = calculate_taus_post_batched(             # (n_inside, N_INSIDE_TAU, 100)
            redshifts_of_mocks[inside_gals], z_end_bub_arr[inside_gals],
            _z_up_all[inside_gals].copy(), _red_up_all[inside_gals],
            _z_lo_all[inside_gals].copy(), _red_lo_all[inside_gals],
            z_per_gal=_z_wv_per_gal[inside_gals],
            tau_wv_pref_per_gal=_tau_wv_pref_per_gal[inside_gals],
            I_z_end_per_gal=_I_z_end_per_gal[inside_gals],
            I_red_up_all=_I_red_up_all[inside_gals],
        )
        _tick("2-calculate_taus_post")

        tau_now_in = _tau_prec_all[inside_gals] + tau_post_in # (n_inside, N_INSIDE_TAU, 100)
        bad = np.any(tau_now_in[:, :, 30:] - tau_now_in[:, :, 29:-1] > 0.0, axis=2)
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
        _tick("3-tau_sanity")

        continuum_all = _base_cont_eit_outside.copy()
        continuum_all[inside_gals] = _base_continuum[inside_gals] * np.exp(-tau_now_in)
        _tick("4-continuum")

    # ── 5. Flux + rebin — matmul only for inside-galaxy rows ─────────────────
    # Outside galaxies: use precomputed _flux_outside (fixed matmul result).
    # Inside galaxies:  recompute only those rows — matmul scales with n_inside, not N_DATA.
    predicted = _flux_outside.copy()
    if len(inside_gals) > 0:
        flat_in = continuum_all[inside_gals].reshape(len(inside_gals) * N_INSIDE_TAU, 100)
        predicted[inside_gals] = (flat_in @ _direct_matrix).reshape(
            len(inside_gals), N_INSIDE_TAU, N_BINS
        )
    _tick("5-flux_rebin")

    # ── 6. Gaussian likelihood in flux space ──────────────────────────────────
    # p(f_obs | θ) = (1/N) Σ_k N(f_obs ; predicted_k, σ_bin²)
    # NaN in predicted = sightline with complete absorption → zero flux.
    np.nan_to_num(predicted, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    diffs = _obs_flux_per_gal[:, np.newaxis, :] - predicted    # (N_DATA, N_INSIDE_TAU, N_BINS)
    log_p = (
        logsumexp(-0.5 * (diffs / _noise_per_bin) ** 2, axis=1)  # (N_DATA, N_BINS)
        - np.log(N_INSIDE_TAU)
        - np.log(_noise_per_bin)
        - 0.5 * np.log(2 * np.pi)
    )
    _tick("6-likelihood")
    return float(log_p.sum())


def log_likelihood(theta):
    return get_spectral_likelihood(theta[0], theta[1], theta[2], theta[3])

# ── Pre-run diagnostics ───────────────────────────────────────────────────────
print(f"\nTrue bubble params (x, y, z, r): {TRUE_MU}", flush=True)

# 1. Likelihood at truth — should be near the posterior peak
_ll_truth = get_spectral_likelihood(*TRUE_MU)
print(f"Log-likelihood at truth: {_ll_truth:.2f}", flush=True)

# 2. 1-D slices through truth — peak should sit on the red dashed line;
#    discontinuities flag tau-sanity fallback misfires.
_param_names  = ['x_bub', 'y_bub', 'z_bub', 'r_bub']
_param_ranges = [
    np.linspace(PRIOR_LO[0], PRIOR_HI[0], 31),
    np.linspace(PRIOR_LO[1], PRIOR_HI[1], 31),
    np.linspace(PRIOR_LO[2], PRIOR_HI[2], 31),
    np.linspace(PRIOR_LO[3], PRIOR_HI[3], 31),
]
_fig_sl, _axes_sl = plt.subplots(1, 4, figsize=(16, 4))
for _pi, (ax, pname, pgrid) in enumerate(zip(_axes_sl, _param_names, _param_ranges)):
    _lls = []
    for _v in pgrid:
        _p = TRUE_MU.copy().astype(float)
        _p[_pi] = _v
        _lls.append(get_spectral_likelihood(*_p))
    ax.plot(pgrid, _lls, lw=1.5)
    ax.axvline(TRUE_MU[_pi], color='red', ls='--', label='truth')
    ax.set_xlabel(pname)
    if _pi == 0:
        ax.set_ylabel('log L')
    ax.legend(fontsize=8)
_fig_sl.suptitle("1-D likelihood slices through truth (red = true value)")
_fig_sl.tight_layout()
_fig_sl.savefig("likelihood_slices.png", dpi=150, bbox_inches="tight")
print("Saved likelihood_slices.png", flush=True)

# Reset call counter so profiling covers the first sampler calls, not diagnostics
_NCALLS = 0
print("Starting dynesty sampler...", flush=True)

# ── Run dynesty ───────────────────────────────────────────────────────────────
#
# NestedSampler  = static nested sampling (fixed number of live points).
# DynamicNestedSampler = adapts live points during the run; better for
#                        posteriors, but a bit more setup. Try it later.
#
# nlive: number of live points. More = more accurate but slower.
#        ~200-500 is fine for low-dimensional problems.

CHECKPOINT_FILE = 'dynesty_checkpoint.pkl'

with mp.get_context('fork').Pool(N_WORKERS) as pool:
    if os.path.exists(CHECKPOINT_FILE):
        print(f"Resuming from {CHECKPOINT_FILE}", flush=True)
        sampler = dynesty.NestedSampler.restore(CHECKPOINT_FILE, pool=pool)
    else:
        sampler = dynesty.NestedSampler(
            log_likelihood,
            prior_transform,
            ndim=NDIM,
            nlive=300,   # 300 for production; 100 for quick test
            pool=pool,
            queue_size=N_WORKERS,
        )
    sampler.run_nested(
        print_progress=True,
        dlogz=0.5,
        checkpoint_file=CHECKPOINT_FILE,
        checkpoint_every=100,
    )
results = sampler.results

# ── Extract results ───────────────────────────────────────────────────────────
#
# results.samples  : raw samples, NOT equally weighted
# results.logwt    : log importance weights for each sample
# results.logz[-1] : log Bayesian evidence, log Z
# results.logzerr  : uncertainty on log Z
#
# To get posterior samples you need to account for the weights.
# resample_equal() draws N equally-weighted samples from the weighted set.

weights        = np.exp(results.logwt - results.logz[-1])   # normalized
equal_samples  = resample_equal(results.samples, weights)   # shape (M, NDIM)
posterior_mean = np.average(results.samples, weights=weights, axis=0)
posterior_std  = np.sqrt(
    np.average((results.samples - posterior_mean)**2, weights=weights, axis=0)
)

print("\n── Dynesty result ───────────────────────────────────")
print(f"  Posterior mean:    {posterior_mean}")
print(f"  Posterior std:     {posterior_std}")
print(f"  log Z:             {results.logz[-1]:.2f} +/- {results.logzerr[-1]:.2f}")

# ── Plots ─────────────────────────────────────────────────────────────────────

# 1) Run summary: shows how log-evidence accumulates and live-point history
fig1, axes1 = dyplot.runplot(results)
fig1.suptitle("Nested sampling run summary", y=1.01)
fig1.savefig("dynesty_run.png", dpi=150, bbox_inches="tight")

# 2) Corner plot of the posterior
fig2, axes2 = dyplot.cornerplot(
    results,
    labels=[r"$x_\mathrm{bub}$", r"$y_\mathrm{bub}$", r"$z_\mathrm{bub}$", r"$r_\mathrm{bub}$"],
    truths=TRUE_MU,
    show_titles=True,
    quantiles=[0.16, 0.5, 0.84],
)
fig2.suptitle("Posterior: Lyman-alpha bubble parameters", y=1.02)
fig2.savefig("dynesty_corner.png", dpi=150, bbox_inches="tight")

plt.show()
print("\nSaved dynesty_run.png and dynesty_corner.png")

# ── What to try next ──────────────────────────────────────────────────────────
#
# 1) Increase N_DATA: watch posterior_std shrink as 1/sqrt(N).
#
# 2) Switch to DynamicNestedSampler for better posterior accuracy:
#
#      sampler = dynesty.DynamicNestedSampler(log_likelihood, prior_transform, ndim=NDIM)
#      sampler.run_nested()
#
# 3) Add a correlated parameter (off-diagonal covariance) to see how
#    dynesty handles degeneracies.
#
# 4) Make the prior informative (Gaussian instead of uniform):
#    prior_transform then draws from that Gaussian given u via ppf.
#
# 5) For your bubble problem:
#    theta = [x_bub, y_bub, z_bub, r_bub]
#    prior_transform maps 4 unit-cube coords to physical ranges
#    log_likelihood wraps _get_likelihood for a single theta