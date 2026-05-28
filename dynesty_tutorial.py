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

import warnings
import numpy as np
import matplotlib.pyplot as plt
import dynesty
warnings.filterwarnings("ignore", category=DeprecationWarning)
from venv.speed_up import get_content, calculate_taus_post
from dynesty import plotting as dyplot
from dynesty.utils import resample_equal
from astropy.cosmology import Planck18 as Cosmo
import astropy.units as u
from venv.galaxy_prop import get_muv, get_mock_data, get_js, tau_CGM, p_EW
from venv.helpers import z_at_proper_distance, full_res_flux, perturb_flux, comoving_distance_from_source_Mpc
from venv.igm_prop import tau_wv
from sklearn.neighbors import KernelDensity
# ── Ground truth and fake data ────────────────────────────────────────────────

TRUE_MU = np.array([0,0,0,10])       # what we want to recover
#SIGMA   = np.array([0.5,  0.8])       # known measurement noise (per axis)
N_DATA  = 10   # 50 for production; 10 for a quick test run
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
    n_inside_tau=1,
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
ADDITIVE = 1e-18    # additive offset before log-transform to avoid log(0)
N_ITER_BUB = 1     # must match get_content call above
N_INSIDE_TAU = 1   # must match get_content call above


def get_spectral_likelihood(xb, yb, zb, rb):
    """
    Log-likelihood of observed spectra given bubble at (xb, yb, zb, rb).
    Returns a scalar: sum of log-likelihoods over all galaxies and spectral bins.

    For each galaxy:
      1. Compute IGM optical depth for the main bubble geometry.
      2. Build Monte Carlo ensemble of predicted spectra using precomputed
         outside-bubble taus from cont_filled.
      3. Fit a 1D KDE per spectral bin to the model predictions.
      4. Evaluate the observed spectrum under the KDE and accumulate log-prob.
    """
    spec_res = wave_Lya.value * (1 + 7.5) / 2700
    full_bins = np.arange(
        wave_em.value[0] * (1 + 7.5),
        wave_em.value[-1] * (1 + 7.5),
        spec_res,
    )
    max_bins_full = len(full_bins)

    log_like = 0.0

    for index_gal in range(N_DATA):
        xg = x_gal_mock[index_gal]
        yg = y_gal_mock[index_gal]
        zg = z_gal_mock[index_gal]
        red_s = redshifts_of_mocks[index_gal]
        muvi = Muv_mock[index_gal]

        # Far edge of the main bubble along this galaxy's LOS
        if (xg - xb)**2 + (yg - yb)**2 + (zg - zb)**2 < rb**2:
            dist = zg - zb + np.sqrt(rb**2 - (xg - xb)**2 - (yg - yb)**2)
            z_end_bub = z_at_proper_distance(dist / (1 + red_s) * u.Mpc, red_s)
        else:
            z_end_bub = red_s

        tau_cgm_in = tau_CGM(muvi, main_dir=main_dir)

        # Precompute area factors for all MC samples at once (vectorised over samples)
        j_s_all = np.array(cont_filled.j_s_full[index_gal])     # (N_SAMPLES, 100)
        area_factor_all = (
            np.trapz(j_s_all * tau_cgm_in[np.newaxis, :], wave_em.value, axis=1) /
            np.trapz(j_s_all, wave_em.value, axis=1)
        )
        area_factor_all = np.where(area_factor_all < 1e-20, 1e-5, area_factor_all)

        flux_to_save = np.zeros((N_ITER_BUB * N_INSIDE_TAU, max_bins_full))

        for n in range(N_ITER_BUB):
            sl = slice(n * N_INSIDE_TAU, (n + 1) * N_INSIDE_TAU)

            # Combine precomputed outside-bubble tau with main-bubble contribution
            tau_now_i = np.copy(cont_filled.tau_prec_full[index_gal][sl, :])
            tau_now_i += calculate_taus_post(
                red_s,
                z_end_bub,
                np.copy(cont_filled.first_bubble_encounter_coord_z_up_full[index_gal][sl]),
                np.copy(cont_filled.first_bubble_encounter_redshift_up_full[index_gal][sl]),
                np.copy(cont_filled.first_bubble_encounter_coord_z_lo_full[index_gal][sl]),
                np.copy(cont_filled.first_bubble_encounter_redshift_lo_full[index_gal][sl]),
                n_iter=N_INSIDE_TAU,
            )

            # Tau sanity: replace non-monotonic rows with a smooth fallback
            bad = np.any(tau_now_i[:, 30:] - tau_now_i[:, 29:-1] > 0.0, axis=1)
            if np.any(bad):
                dist_cm = comoving_distance_from_source_Mpc(red_s, z_end_bub)
                fallback = np.clip(
                    tau_wv(wave_em, dist=np.abs(dist_cm), zs=red_s, z_end=5.3, nf=0.65)
                    + np.random.normal(0.0, 0.1),
                    0, np.inf,
                )
                tau_now_i[bad] = fallback
            tau_now_i[tau_now_i < 0] = np.inf
            tau_now_i = np.nan_to_num(tau_now_i, nan=np.inf)

            eit_l = np.exp(-tau_now_i)
            lae_now = cont_filled.la_flux_out_full[index_gal][sl] / area_factor_all[sl]

            continuum_i = (
                lae_now[:, np.newaxis]
                * j_s_all[sl]
                * eit_l
                * tau_cgm_in[np.newaxis, :]
                * cont_filled.com_fact[index_gal]
            )
            flux_to_save[sl] = full_res_flux(continuum_i, 7.5)

        # Add noise and rebin to the fixed N_BINS used for the observed data
        noisy = flux_to_save + np.random.normal(0, NOISE, flux_to_save.shape)
        predicted = perturb_flux(noisy, N_BINS)   # (n_samples, N_BINS)
        observed = flux_noise_mock[index_gal]      # (N_BINS,)

        # 1D KDE per bin in magnitude space, accumulate log-prob
        for b in range(N_BINS):
            model_mag = 5 * np.log10(10**18.7 * (ADDITIVE + 2 * predicted[:, b]))
            obs_mag = 5 * np.log10(10**18.7 * (ADDITIVE + 2 * observed[b]))
            if not np.isfinite(obs_mag) or not np.all(np.isfinite(model_mag)):
                continue
            kde = KernelDensity(kernel='exponential', bandwidth=0.12).fit(
                model_mag.reshape(-1, 1)
            )
            log_like += kde.score_samples([[obs_mag]])[0]

    return log_like


def log_likelihood(theta):
    return get_spectral_likelihood(theta[0], theta[1], theta[2], theta[3])

print(f"True bubble params (x, y, z, r): {TRUE_MU}", flush=True)
print("Starting dynesty sampler...", flush=True)

# ── Run dynesty ───────────────────────────────────────────────────────────────
#
# NestedSampler  = static nested sampling (fixed number of live points).
# DynamicNestedSampler = adapts live points during the run; better for
#                        posteriors, but a bit more setup. Try it later.
#
# nlive: number of live points. More = more accurate but slower.
#        ~200-500 is fine for low-dimensional problems.

sampler = dynesty.NestedSampler(
    log_likelihood,
    prior_transform,
    ndim=NDIM,
    nlive=100,   # 300 for production; 100 for quick test
)

sampler.run_nested(print_progress=True, dlogz=0.5)
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