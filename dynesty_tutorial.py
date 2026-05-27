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
from dynesty import plotting as dyplot
from dynesty.utils import resample_equal

# ── Ground truth and fake data ────────────────────────────────────────────────

TRUE_MU = np.array([2.5, -1.0])       # what we want to recover
SIGMA   = np.array([0.5,  0.8])       # known measurement noise (per axis)
N_DATA  = 30

rng  = np.random.default_rng(42)
data = rng.normal(TRUE_MU, SIGMA, size=(N_DATA, 2))   # shape (N_DATA, 2)

NDIM = 2

# ── Prior ─────────────────────────────────────────────────────────────────────
#
# dynesty always works in the unit hypercube internally.
# prior_transform converts a point u in [0,1]^N to physical parameters.
# Here we use a wide uniform prior on each axis.

PRIOR_LO = np.array([-10.0, -10.0])
PRIOR_HI = np.array([ 10.0,  10.0])

def prior_transform(u):
    """Uniform prior: [0,1]^2  ->  [PRIOR_LO, PRIOR_HI]."""
    return PRIOR_LO + u * (PRIOR_HI - PRIOR_LO)

# ── Likelihood ────────────────────────────────────────────────────────────────

def log_likelihood(theta):
    """Log p(data | theta): data is iid Gaussian around theta."""
    residuals = data - theta          # broadcast over N_DATA rows
    return -0.5 * np.sum((residuals / SIGMA) ** 2)

# ── Analytic answer (for comparison) ─────────────────────────────────────────
#
# With N iid Gaussian measurements and a flat prior the posterior is:
#   mu_posterior = mean(data, axis=0)
#   sigma_posterior = SIGMA / sqrt(N_DATA)

analytic_mean = data.mean(axis=0)
analytic_std  = SIGMA / np.sqrt(N_DATA)

print("── Analytic answer ──────────────────────────────────")
print(f"  True mu:           {TRUE_MU}")
print(f"  Posterior mean:    {analytic_mean}")
print(f"  Posterior std:     {analytic_std}")

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