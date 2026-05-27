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

import numpy as np
import matplotlib.pyplot as plt
import dynesty
from main import _get_likelihood
from dynesty import plotting as dyplot
from dynesty.utils import resample_equal
from astropy.cosmology import Planck18 as Cosmo
import astropy.units as u
from venv.galaxy_prop import get_muv, get_mock_data, get_js, tau_CGM, p_EW
from venv.helpers import z_at_proper_distance, full_res_flux, perturb_flux
# ── Ground truth and fake data ────────────────────────────────────────────────

TRUE_MU = np.array([0,0,0,10])       # what we want to recover
#SIGMA   = np.array([0.5,  0.8])       # known measurement noise (per axis)
N_DATA  = 50
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

bins_arr = np.linspace(
        wave_em.value[0] * (1 + 7.5),
        wave_em.value[-1] * (1 + 7.5),
        11
    )

wave_em_dig_arr = np.digitize(
        wave_em.value * (1 + 7.5),
        bins_arr
)


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

#data = rng.normal(TRUE_MU, SIGMA, size=(N_DATA, 2))   # shape (N_DATA, 2)

NDIM = 4

# ── Prior ─────────────────────────────────────────────────────────────────────
#
# dynesty always works in the unit hypercube internally.
# prior_transform converts a point u in [0,1]^N to physical parameters.
# Here we use a wide uniform prior on each axis.

PRIOR_LO = np.array([-10.0, -10.0, 10, 1])
PRIOR_HI = np.array([ 10.0,  10.0, 10, 20])

def prior_transform(u):
    """Uniform prior: [0,1]^2  ->  [PRIOR_LO, PRIOR_HI]."""
    return PRIOR_LO + u * (PRIOR_HI - PRIOR_LO)

# ── Likelihood ────────────────────────────────────────────────────────────────

def log_likelihood(theta):
    """Log p(data | theta): data is iid Gaussian around theta."""
    #residuals = data - theta          # broadcast over N_DATA rows
    return _get_likelihood(
        None,
        theta[0],
        theta[1],
        theta[2],
        theta[3],
        x_gal_mock,
        y_gal_mock,
        z_gal_mock,
        data,
        10,
        redshift = 7.5,
        muv = Muv_mock,
        beta_data = beta,
        la_e_in = la_e,
        flux_int = None,
        flux_limit = 2e-19,
        like_on_flux = flux_noise_mock,
        n_inside_tau = 10,
        bins_tot = 11,
        cache = False,
        like_on_tau_full = False,
        noise_on_the_spectrum = 5e-20,
        consistent_noise = True,
        cont_filled = None,
        constrained_prior = False,
        reds_of_galaxies = redshifts_of_mocks,
        dir_name = None,
        main_dir = main_dir,
        cache_dir = None,
        la_e_orig = la_e_orig,
        prior_on_all = False,
    )
    #return -0.5 * np.sum((residuals / SIGMA) ** 2)

# ── Analytic answer (for comparison) ─────────────────────────────────────────
#
# With N iid Gaussian measurements and a flat prior the posterior is:
#   mu_posterior = mean(data, axis=0)
#   sigma_posterior = SIGMA / sqrt(N_DATA)

analytic_mean = data.mean(axis=0)
#analytic_std  = SIGMA / np.sqrt(N_DATA)

print("── Analytic answer ──────────────────────────────────")
print(f"  True mu:           {TRUE_MU}")
print(f"  Posterior mean:    {analytic_mean}")
#print(f"  Posterior std:     {analytic_std}")

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
    nlive=300,
)

sampler.run_nested(print_progress=True)
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
    labels=[r"$\mu_x$", r"$\mu_y$"],
    truths=TRUE_MU,
    show_titles=True,
    quantiles=[0.16, 0.5, 0.84],
)
fig2.suptitle("Posterior: 2D Gaussian mean", y=1.02)
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