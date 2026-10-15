"""
exact_pixel_likelihood.py -- per-galaxy likelihood contributions log p(x_g | theta) for every map of a
pool, computed directly from the forward model (no neural network).

WHY THIS IS POSSIBLE
--------------------
In the pixel-field simulator (add_mock_observations.py / sbi_pixel_field.py) galaxy g's observation x_g =
(reported EW, upper-limit flag) depends on theta ONLY through its own sightline column theta_g (75 bins),
plus random draws (intrinsic EW, velocity offset) that are independent between galaxies. Hence

    p(x | theta) = prod_g p(x_g | theta_g),       log-likelihood  L(theta) = sum_g l_g(theta).

For a pool of prior maps theta_i the posterior is the importance-weighted pool with w_i ~ exp(L(theta_i)),
exactly like the network-based `infer` but with the exact likelihood instead of the learned ratio. And
the posterior WITHOUT galaxy g is obtained exactly by dropping one term: w_i ~ exp(L(theta_i) - l_g(theta_i)).

THE LIKELIHOOD OF ONE GALAXY (same forward model as add_mock_observations.py, steps 1-4)
--------------------------------------------------------------------------------------
 1. tau_i(lambda) = sum_k theta_i,gk C_gk(lambda), where C_gk is the damping-wing optical depth of bin k
    if it were fully neutral (lyabubbles.lightcone_field.segments_to_tau applied to that single bin).
    This sum is EXACTLY segments_to_tau of the whole column, because theta is 0 or 1 and the clipping of
    negative pole artefacts in segments_to_tau is done per segment.
 2. M Monte Carlo draws m of the galaxy model, shared by all pool maps (common random numbers):
    intrinsic EW_m from p_EW(--ew_model), velocity offset dv_m ~ 10^N(delta_v_func(Muv, z0), 0.24), line
    profile J_m = Gaussian in velocity (centre dv_m, width dv_m, unit integral), CGM transmission T_CGM
    (0/1 step at the halo circular velocity, tau_CGM). These are the pipeline's own functions, extracted
    from lyabubbles/galaxy_prop.py (that module imports py21cmfast, so it is not imported as a whole).
 3. mu_im = EW_m * Int J_m T_CGM exp(-tau_i) dlambda / Int J_m dlambda     (noiseless observed EW)
 4. noise and 3-sigma cut, sigma_g from the catalog (sbi_real_data._per_galaxy_sigma):
      upper limit:  p = mean_m Phi((3 sigma_g - mu_im) / sigma_g)              (probability of a non-detection)
      detection  :  p = mean_m phi((EW_obs - mu_im) / sigma_g) / sigma_g       (density of the reported EW)
    The detection density is the simulator's only when EW_obs > 3 sigma_g; see "OFF-SUPPORT DETECTIONS".

OFF-SUPPORT DETECTIONS
----------------------
The simulator reports a detection only if EW_sim > 3 sigma_g. A real catalog detection with
EW_obs <= 3 sigma_g (e.g. JADES-1129: 9.3 +- 7.2 A) has probability ZERO under the simulator for every
theta: the network never saw such an input during training. Here we use the untruncated Gaussian density
above (which still prefers maps predicting EW ~ EW_obs) and list these galaxies in the output, so their
role can be checked by removing them (--groups).

VALIDATION (--validate): the pool's own simulated observations x_j are draws from p(x | theta_j), so the
predicted P(detection | theta_j) must match the detection frequency in the pool, and the probability
integral transform (PIT) of the simulated detected EWs must be uniform. This checks that this file
reproduces the simulator, independently of the real data.

APPROXIMATIONS: Monte Carlo with M draws (noise in l_g ~ 1/sqrt(M)); the pool limits the posterior
resolution exactly as for the network (ESS).

Runs locally in the JWST_inference conda env (numpy, scipy, astropy, pandas, h5py; no torch/py21cmfast).

Usage
-----
python exact_pixel_likelihood.py --ew_model exponential --pool_dir ~/Documents/Lya_bubblesx \\
    --validate --compare_nn ~/Documents/Lya_bubbles/SBI_on_mock_lightcones/pixel_infer.npz \\
    --groups 14-18 27-36 21,22,24,25 --output exact_attribution_exp.npz
Plot with plot_pixel_attribution.py (same file format as sbi_pixel_attribution.py).
"""
import os
import sys
import ast
import glob
import types
import argparse

import numpy as np
from scipy import integrate
from scipy.special import log_ndtr, logsumexp, ndtr, ndtri
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import Planck18 as Cosmo

if not hasattr(np, 'trapz'):          # numpy >= 2 renamed it; the cluster numpy still has np.trapz
    np.trapz = np.trapezoid

REPO = os.path.dirname(os.path.abspath(__file__))
# Register an empty `lyabubbles` package so that importing its submodules does NOT run
# lyabubbles/__init__.py (which imports galaxy_prop -> py21cmfast).
_pkg = types.ModuleType('lyabubbles')
_pkg.__path__ = [os.path.join(REPO, 'lyabubbles')]
sys.modules['lyabubbles'] = _pkg
from lyabubbles import real_data as rd               # catalog loader, radec_to_comoving
from lyabubbles import lightcone_field as lf         # bin_z_edges, segments_to_tau, Z_END_DEFAULT
from lyabubbles.helpers import wave_em, wave_to_dv   # rest-frame grid 1214-1225 A (100 points)

# Same mapping as real_data_run._EW_MODEL_FLAGS (copied: real_data_run imports dynesty/py21cmfast).
EW_MODEL_FLAGS = {
    'exponential':      dict(),
    'gauss':            dict(gauss_distr=True),
    'tang':             dict(Tang_distr=True, Tang_sample='z6.5-8.0'),
    'tang_z8.0-10.0':   dict(Tang_distr=True, Tang_sample='z8.0-10.0'),
    'tang_egs':         dict(Tang_distr=True, Tang_sample='z6.5-8.0_egs'),
    'tang_goods_a2744': dict(Tang_distr=True, Tang_sample='z6.5-8.0_goods_a2744'),
    'tang_muv':         dict(Tang_distr=True, Tang_sample='muv_interp'),
    'gagnon_hartman':   dict(GH_distr=True),
}


def galaxy_prop_functions():
    """Extract p_EW, delta_v_func and tau_CGM (and the constants they use) from galaxy_prop.py by
    executing only those definitions -- the source of truth stays the pipeline file."""
    tree = ast.parse(open(os.path.join(REPO, 'lyabubbles', 'galaxy_prop.py')).read())
    ns = dict(np=np, integrate=integrate, u=u, const=const, Cosmo=Cosmo, wave_em=wave_em,
              wave_to_dv=wave_to_dv, _tau_cgm_data_cache={})
    want = {'p_EW', 'delta_v_func', 'tau_CGM', '_tang_muv_interp_params', '_linterp_extrap',
            'TANG24_EW_PARAMS', '_TANG24A_MUV_ANCHORS', '_TANG24A_MUV_EMU', '_TANG24A_MUV_SIGMA',
            '_TANG24A_MUV_LOG_EMU', '_TANG24A_MUV_LOG_SIGMA'}
    for node in tree.body:
        name = node.name if isinstance(node, ast.FunctionDef) else (
            node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            else None)
        if name in want:
            exec(compile(ast.Module([node], []), 'galaxy_prop.py', 'exec'), ns)
    return ns['p_EW'], ns['delta_v_func'], ns['tau_CGM'], ns


def parse_groups(tokens, n_gal):
    """'14-18' -> [14..18]; '21,22,24' -> [21, 22, 24] (same syntax as sbi_pixel_attribution.py)."""
    groups = []
    for tok in tokens:
        members = []
        for part in tok.split(','):
            a, _, b = part.partition('-')
            members.extend(range(int(a), int(b) + 1) if b else [int(a)])
        if min(members) < 0 or max(members) >= n_gal:
            raise ValueError(f"group {tok!r} outside 0..{n_gal - 1}")
        groups.append(sorted(set(members)))
    return groups


def load_pool(pool_dir, split):
    """theta (n, n_gal*n_los) and x (n, 2 n_gal) from {split}_batch_*.npz, in sorted file order -- the
    same order as sbi_pixel_field.load_sims, so pool index i matches the network's pixel_infer.npz."""
    paths = sorted(glob.glob(os.path.join(pool_dir, f'{split}_batch_*.npz')))
    if not paths:
        raise FileNotFoundError(f"no {split}_batch_*.npz in {pool_dir}")
    ds = [np.load(p) for p in paths]
    return (np.concatenate([d['theta'] for d in ds]), np.concatenate([d['x'] for d in ds]),
            int(ds[0]['n_gal']), int(ds[0]['n_los']))


def weights_and_map(logL, theta_grid):
    """Self-normalised importance weights, ESS and P(neutral) map (as sbi_pixel_field.run_infer)."""
    w = np.exp(logL - logL.max())
    w /= w.sum()
    return w, 1.0 / np.sum(w ** 2), np.tensordot(w, theta_grid, axes=(0, 0))


def load_real_catalog(lya_catalog, properties_catalog, z_min, muv_max, prefer, z_hi):
    """Catalog arrays exactly as real_data_run._load_catalog_and_priors (z_lo = None)."""
    cat = rd.load_catalog_v2(lya_catalog, properties_catalog, z_min=z_min, muv_max=muv_max, prefer=prefer)
    win = cat.redshift <= z_hi
    muv, z_gal_red = cat.muv[win], cat.redshift[win]
    ew_obs, ew_err, is_ul = cat.ew[win], cat.ew_err[win], cat.is_upper_limit[win].astype(bool)
    _, _, z_gal, _, _, z0, _, _, _ = rd.radec_to_comoving(cat.ra[win], cat.dec[win], z_gal_red)
    sigma = np.where(is_ul, ew_obs / 3.0, ew_err)            # = sbi_real_data._per_galaxy_sigma
    return dict(muv=muv, z_gal_red=z_gal_red, z_gal=z_gal, z0=z0, ew_obs=ew_obs, is_ul=is_ul, sigma=sigma)


class GalaxyModel:
    """Steps 1-3 of the module docstring: the noiseless observed EW mu_im of galaxy g for every pool map
    theta_i and every Monte Carlo draw m. Draws are made galaxy by galaxy in order g = 0, 1, ... from
    numpy's global state (p_EW) and from `rng` (velocity offsets), so calling mu() for g = 0..n_gal-1 in
    order with the same seed gives identical draws in every script that uses this class."""

    def __init__(self, cat, ew_model, n_mc, seed):
        self.cat, self.ew_model, self.M = cat, ew_model, n_mc
        self.p_EW, self.delta_v_func, self.tau_CGM, self.ns = galaxy_prop_functions()
        self.wv = wave_em.value                               # rest-frame wavelengths [A]
        self.dv_grid = wave_to_dv(wave_em).value              # the same grid in km/s
        wq = np.zeros_like(self.wv)                           # trapezoid-rule weights on the grid
        wq[1:-1] = (self.wv[2:] - self.wv[:-2]) / 2
        wq[0] = (self.wv[1] - self.wv[0]) / 2; wq[-1] = (self.wv[-1] - self.wv[-2]) / 2
        self.wq = wq
        self.d_c0 = Cosmo.comoving_distance(cat['z0']).to(u.Mpc).value
        np.random.seed(seed)                                  # p_EW draws from numpy's global state
        self.rng = np.random.default_rng(seed)
        self._tr_cache = {}                                   # transmission(): (C, A) per (g, n_los, n_dv)

    def mu(self, g, theta_g, n_los):
        """theta_g: (n_pool, n_los) sightline columns of galaxy g. Returns mu (n_pool, M)."""
        c, M = self.cat, self.M
        z_b, z_e = lf.bin_z_edges(n_los, c['z_gal'][g], self.d_c0, z_end=lf.Z_END_DEFAULT)
        C = np.stack([lf.segments_to_tau(z_b[k:k + 1], z_e[k:k + 1], np.ones(1), c['z_gal_red'][g], self.wv)
                      for k in range(n_los)])               # (n_los, n_wave): tau of each bin if neutral
        tau = theta_g @ C                                     # (n_pool, n_wave)
        ew_int = np.asarray(self.p_EW(np.full(M, c['muv'][g]), np.full(M, -2.0),
                                      **EW_MODEL_FLAGS[self.ew_model])[0], dtype=float)
        dv = 10 ** self.rng.normal(self.delta_v_func(c['muv'][g], c['z0']), 0.24, M)  # get_js, fwhm_true=False
        J = np.exp(-0.5 * ((self.dv_grid[None, :] - dv[:, None]) / dv[:, None]) ** 2) / dv[:, None]
        J /= (J * self.wq).sum(axis=1, keepdims=True)         # unit integral over the grid
        A = J * self.tau_CGM(c['muv'][g], main_dir=REPO)[None, :] * self.wq   # profile x CGM x weights
        T = np.exp(-tau) @ A.T                                # (n_pool, M) IGM transmission
        return ew_int[None, :] * T                            # noiseless observed EW

    def transmission(self, g, theta_g, n_los, n_dv):
        """Deterministic version of steps 1-2 for the QUADRATURE likelihood: the IGM x CGM transmission
        T_ik = Int J_k T_CGM exp(-tau_i) dlambda of galaxy g for every pool map i, on n_dv velocity offsets
        at the equal-probability quantiles of the Mason+18 log-normal,
            log10 dv_k = mu_dv(Muv, z0) + 0.24 * Phi^-1((k + 1/2) / n_dv),   k = 0 .. n_dv - 1,
        so that mean_k h(dv_k) is a midpoint-rule integral over the dv distribution (no random noise).
        Returns T (n_pool, n_dv).
        C and A depend only on (g, n_los, n_dv), not on the pool, so they are computed once per galaxy and
        cached: a pool processed in chunks (exact_pixel_likelihood_chunked.py) does not rebuild them."""
        key = (g, n_los, n_dv)
        if key not in self._tr_cache:
            c = self.cat
            z_b, z_e = lf.bin_z_edges(n_los, c['z_gal'][g], self.d_c0, z_end=lf.Z_END_DEFAULT)
            C = np.stack([lf.segments_to_tau(z_b[k:k + 1], z_e[k:k + 1], np.ones(1), c['z_gal_red'][g], self.wv)
                          for k in range(n_los)])           # (n_los, n_wave): tau of each bin if neutral
            dv = 10 ** (self.delta_v_func(c['muv'][g], c['z0'])
                        + 0.24 * ndtri((np.arange(n_dv) + 0.5) / n_dv))
            J = np.exp(-0.5 * ((self.dv_grid[None, :] - dv[:, None]) / dv[:, None]) ** 2) / dv[:, None]
            J /= (J * self.wq).sum(axis=1, keepdims=True)     # unit integral over the grid
            A = J * self.tau_CGM(c['muv'][g], main_dir=REPO)[None, :] * self.wq
            self._tr_cache[key] = (C, A)
        C, A = self._tr_cache[key]
        return np.exp(-(theta_g @ C)) @ A.T                   # (n_pool, n_dv)

    def ew_distribution(self, g):
        """Intrinsic-EW distribution of galaxy g under --ew_model, for the quadrature likelihood."""
        return IntrinsicEW(self.ew_model, self.cat['muv'][g], self.ns)


def loglik_from_mu(mu, ew, upper_limit, s):
    """Step 4: log p(x_g | theta_i) for every pool map, from mu (n_pool, M). ew = reported EW (ignored
    for an upper limit, whose threshold 3 s is fixed by the catalog sigma)."""
    M = mu.shape[1]
    if upper_limit:
        return logsumexp(log_ndtr((3 * s - mu) / s), axis=1) - np.log(M)
    return logsumexp(-0.5 * ((ew - mu) / s) ** 2, axis=1) - np.log(M) - np.log(s * np.sqrt(2 * np.pi))


class IntrinsicEW:
    """The intrinsic Lya EW distribution that p_EW (lyabubbles/galaxy_prop.py) SAMPLES from, written as a
    point mass p0 at EW = 0 plus a continuous part with cumulative mass F(W) (F(inf) = 1 - p0) and tail
    mass S(W) = (1 - p0) - F(W) (kept separately so that tiny tail masses are accurate).

    exponential   p_EW: with probability A(Muv) an emitter, EW from (1/W_c) exp(-EW/W_c) restricted to
                  [0, 500] A; otherwise EW = 0 (Mason+18 / Treu+12 parametrisation):
                    A(Muv) = 0.65 + 0.1 tanh(3 (Muv + 20.75)),  W_c(Muv) = 31 + 12 tanh(4 (Muv + 20.25)).
    tang*         p_EW: always an emitter, log-normal EW (Tang+24) restricted to [0, 500] A.
                  For both, p_EW draws by inverse-CDF interpolation of the cumulative trapezoid of the density
                  on Ws = linspace(0, 500, 1000); F is built in EXACTLY the same way, so it is the sampler's
                  own (piecewise-linear) CDF, not the analytic one.
    gagnon_hartman  p_EW (sample_GH): log10 L_Lya = 0.42 (u1 + u2 + u3/3) + 42.47 with independent Gaussians
                  u1 ~ N(0.087 m - 0.51, 0.7), u2 ~ N(-0.57 m - 0.85, 0.49), u3 ~ N(-0.38 m - 0.31, 0.26),
                  m = Muv + 18.5, and EW = L_Lya / (C L_UV), C = 2.47e15 / 1216 (beta = -2),
                  L_UV = 10^(-0.4 (Muv - 51.6)). So log10 EW is exactly Gaussian: no grid needed.
    The copies of these formulas are checked against p_EW itself by --check_ew (KS test on samples)."""

    W_MAX = 500.0                                            # p_EW's EW grid upper limit [A]

    def __init__(self, ew_model, muv, ns):
        self.kind = 'grid'
        if ew_model == 'exponential':
            A = 0.65 + 0.1 * np.tanh(3 * (muv + 20.75))
            Wc = 31 + 12 * np.tanh(4 * (muv + 20.25))
            dens = lambda W: np.exp(-W / Wc) / Wc
            self.p0, mass = 1.0 - A, A
        elif ew_model.startswith('tang'):
            sample = EW_MODEL_FLAGS[ew_model]['Tang_sample']
            if sample == 'muv_interp':
                mu, sig = ns['_tang_muv_interp_params'](muv)
            else:
                ew0, sig = ns['TANG24_EW_PARAMS'][sample]
                mu = np.log(ew0)
            def dens(W):                                     # = p_Tang, 0 at W <= 0
                Wp = np.where(W > 0, W, 1.0)
                return np.where(W > 0, np.exp(-(np.log(Wp) - mu) ** 2 / 2 / sig ** 2)
                                / (np.sqrt(2 * np.pi) * sig * Wp), 0.0)
            self.p0, mass = 0.0, 1.0
        elif ew_model == 'gagnon_hartman':
            m = muv + 18.5
            mean_x0 = (0.087 * m - 0.51) + (-0.57 * m - 0.85) + (-0.38 * m - 0.31) / 3
            sd_x0 = np.sqrt(0.7 ** 2 + 0.49 ** 2 + (0.26 / 3) ** 2)
            log10_CLuv = np.log10(2.47e15 / 1216) - 0.4 * (muv - 51.6)
            self.kind, self.p0 = 'lognormal', 0.0
            self.ln_mu = np.log(10) * (0.42 * mean_x0 + 42.47 - log10_CLuv)   # mean of ln EW
            self.ln_sd = np.log(10) * 0.42 * sd_x0                            # sd of ln EW
            return
        else:
            raise NotImplementedError(f"quadrature likelihood not implemented for ew_model={ew_model!r}")
        Ws = np.linspace(0, self.W_MAX, 1000)                # p_EW's grid
        seg = 0.5 * (dens(Ws[1:]) + dens(Ws[:-1])) * np.diff(Ws)          # trapezoid mass of each cell
        cum = np.concatenate(([0.0], np.cumsum(seg)))
        tail = np.concatenate((np.cumsum(seg[::-1])[::-1], [0.0]))      # same, summed from the top
        self.Ws, self.F_grid, self.S_grid = Ws, mass * cum / cum[-1], mass * tail / cum[-1]

    def F(self, W):
        if self.kind == 'lognormal':
            return ndtr((np.log(np.maximum(W, 1e-300)) - self.ln_mu) / self.ln_sd)
        return np.interp(W, self.Ws, self.F_grid)

    def S(self, W):
        if self.kind == 'lognormal':
            return ndtr(-(np.log(np.maximum(W, 1e-300)) - self.ln_mu) / self.ln_sd)
        return np.interp(W, self.Ws, self.S_grid)

    def mass_between(self, a, b):
        """Continuous mass in [a, b] (a <= b), from F below the median and from S above it."""
        lo = self.F(b) - self.F(a)
        hi = self.S(a) - self.S(b)
        return np.where(self.F(a) < 0.5 * (1 - self.p0), lo, hi)


def loglik_quadrature(T, ew, upper_limit, s, dist, n_y=64, chunk=500):
    """log p(x_g | theta_i) for every pool map, integrating over the intrinsic EW NUMERICALLY instead of
    by Monte Carlo (the MC estimate is dominated by a few rare draws for bright detections).

    T (n_pool, n_dv): transmissions on velocity-offset quantiles (GalaxyModel.transmission); the dv
    integral is the mean over k. Write Y = EW_int * T_ik (noiseless observed EW), so that the continuous
    part of Y has cumulative mass F_Y(y) = F(y / T_ik). With the simulator's noise n ~ N(0, s^2):

    detection (reported EW = x > 3 s):
        p = p0 phi(x/s)/s + Int phi((x - y)/s)/s dF_Y(y)
          ~ p0 phi(x/s)/s + sum_j [F_Y(e_j+1) - F_Y(e_j)] phi((x - y_j)/s)/s
        on n_y equal cells e_j of [max(0, x - 8s), x + 8s] (outside, phi < 1e-14), y_j = cell midpoints.
        Using the exact cell MASSES makes the rule robust to any shape of p(EW); the only error is the
        variation of phi within a cell of width 16 s / n_y.
    upper limit (observation below 3 s):
        P = p0 Phi(3) + Int Phi((3s - y)/s) dF_Y(y) = p0 Phi(3) + Int_0^inf F_Y(y) phi((3s - y)/s)/s dy
        (integration by parts), midpoint rule on n_y cells of [0, 3s + 8s].
    Processed in chunks of `chunk` pool maps to bound memory (chunk x n_dv x n_y floats)."""
    n_pool = len(T)
    out = np.empty(n_pool)
    u = (np.arange(n_y) + 0.5) / n_y                          # cell midpoints in [0, 1]
    for a in range(0, n_pool, chunk):
        Tc = np.maximum(T[a:a + chunk], 1e-300)[:, :, None]   # (c, n_dv, 1)
        if upper_limit:
            hi = 11.0 * s
            y = u * hi                                        # (n_y,)
            integ = (dist.F(y[None, None, :] / Tc) * norm_pdf((3 * s - y) / s) / s).sum(-1) * (hi / n_y)
            p = dist.p0 * ndtr(3.0) + integ
        else:
            lo, hi = max(0.0, ew - 8 * s), ew + 8 * s
            e = lo + (hi - lo) * np.arange(n_y + 1) / n_y     # cell edges
            ym = 0.5 * (e[1:] + e[:-1])
            m = dist.mass_between(e[None, None, :-1] / Tc, e[None, None, 1:] / Tc)
            p = dist.p0 * norm_pdf(ew / s) / s + (m * norm_pdf((ew - ym) / s) / s).sum(-1)
        with np.errstate(divide='ignore'):
            out[a:a + chunk] = np.log(p.mean(axis=1))         # mean over dv quantiles
    return out


def norm_pdf(z):
    return np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ew_model', required=True, choices=list(EW_MODEL_FLAGS))
    ap.add_argument('--pool_dir', required=True, help='(theta, x) sims; theta is identical for all EW '
                                                      'models of the same mocks, x only used by --validate.')
    ap.add_argument('--pool_split', default='val', choices=['train', 'val'])
    ap.add_argument('--lya_catalog', default=os.path.join(REPO, 'tb_lya.txt'))
    ap.add_argument('--properties_catalog', default=os.path.join(REPO, 'sample_nirspec_properties.txt'))
    ap.add_argument('--prefer', default='grating', choices=['grating', 'prism'])
    ap.add_argument('--z_hi', type=float, default=7.3)       # defaults = sbi_pixel_field._add_catalog_args
    ap.add_argument('--z_min', type=float, default=5.0)
    ap.add_argument('--muv_max', type=float, default=-18.0)
    ap.add_argument('--n_mc', type=int, default=4000, help='M: Monte Carlo draws of the galaxy model.')
    ap.add_argument('--method', default='quad', choices=['quad', 'mc'],
                    help='quad: numerical integration over EW_int and dv (default, no MC noise); '
                         'mc: Monte Carlo with --n_mc draws (the original estimator).')
    ap.add_argument('--n_dv', type=int, default=32, help='quad: velocity-offset quantiles.')
    ap.add_argument('--n_y', type=int, default=64, help='quad: cells of the EW integral.')
    ap.add_argument('--check_ew', action='store_true',
                    help='KS-test the IntrinsicEW formulas against p_EW samples for every galaxy, then exit.')
    ap.add_argument('--validate', action='store_true', help='Check against the pool\'s simulated x.')
    ap.add_argument('--compare_nn', default=None, help='pixel_infer.npz of the network for the same pool.')
    ap.add_argument('--groups', nargs='*', default=[])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    # ---- catalog, exactly as real_data_run._load_catalog_and_priors (z_lo = None) ----
    cat = load_real_catalog(args.lya_catalog, args.properties_catalog, args.z_min, args.muv_max,
                            args.prefer, args.z_hi)
    ew_obs, is_ul, sigma, z_gal_red = cat['ew_obs'], cat['is_ul'], cat['sigma'], cat['z_gal_red']
    n_gal, z0 = len(ew_obs), cat['z0']
    off_support = (~is_ul) & (ew_obs <= 3 * sigma)           # detections the simulator cannot produce
    print(f"[exact] {n_gal} galaxies, {(~is_ul).sum()} detections, z0 = {z0:.4f}; ew_model = {args.ew_model}")
    for g in np.where(off_support)[0]:
        print(f"[exact] NOTE galaxy {g}: detection EW {ew_obs[g]:.1f} <= 3 sigma = {3 * sigma[g]:.1f} A -- "
              f"impossible under the simulator (see docstring, OFF-SUPPORT DETECTIONS)")

    if args.check_ew:                                         # IntrinsicEW vs the sampler itself
        _, _, _, ns = galaxy_prop_functions()
        p_EW = ns['p_EW']
        np.random.seed(args.seed)
        worst = 0.0
        for g in range(n_gal):
            draws = np.asarray(p_EW(np.full(20000, cat['muv'][g]), np.full(20000, -2.0),
                                    **EW_MODEL_FLAGS[args.ew_model])[0], dtype=float)
            dist = IntrinsicEW(args.ew_model, cat['muv'][g], ns)
            xs = np.sort(draws)
            cdf_model = dist.p0 * (xs >= 0) + dist.F(xs)      # total CDF incl. the point mass at 0
            emp_right = np.searchsorted(xs, xs, side='right') / len(xs)   # empirical CDF (ties: EW = 0)
            emp_left = np.searchsorted(xs, xs, side='left') / len(xs)     # its left limit
            cdf_left = dist.F(xs) + dist.p0 * (xs > 0)                    # model CDF just below xs
            ks = max(np.max(np.abs(cdf_model - emp_right)), np.max(np.abs(cdf_left - emp_left)))
            worst = max(worst, ks)
        crit = np.sqrt(-0.5 * np.log(0.05 / n_gal / 2)) / np.sqrt(20000)  # 95%, Bonferroni over galaxies
        print(f"[check_ew] {args.ew_model}: max KS distance over {n_gal} galaxies = {worst:.4f} "
              f"(95% threshold for the largest of {n_gal} tests with 20000 draws: {crit:.4f})")
        return

    # ---- pool ----
    theta, x_pool, pg, n_los = load_pool(os.path.expanduser(args.pool_dir), args.pool_split)
    if pg != n_gal:
        raise ValueError(f"pool has {pg} galaxies, catalog {n_gal}")
    n_pool = len(theta)
    theta_grid = theta.reshape(n_pool, n_gal, n_los).astype(np.float64)

    # ---- per-galaxy log-likelihood l_g(theta_i) ----
    model = GalaxyModel(cat, args.ew_model, args.n_mc, args.seed)
    M = args.n_mc
    loglik = np.empty((n_pool, n_gal))
    p_det_pred = np.empty((n_pool, n_gal))                    # P(detection | theta_i), for --validate
    pit = [None] * n_gal                                      # PIT of simulated detected EWs
    for g in range(n_gal):
        s = sigma[g]
        if args.method == 'quad':
            T = model.transmission(g, theta_grid[:, g, :], n_los, args.n_dv)
            loglik[:, g] = loglik_quadrature(T, ew_obs[g], is_ul[g], s, model.ew_distribution(g), args.n_y)
        if args.method == 'mc' or args.validate:              # the validation uses the MC draws
            mu = model.mu(g, theta_grid[:, g, :], n_los)      # (n_pool, M)
            if args.method == 'mc':
                loglik[:, g] = loglik_from_mu(mu, ew_obs[g], is_ul[g], s)
        if args.validate:
            p_det_pred[:, g] = ndtr((mu - 3 * s) / s).mean(axis=1)
            det = x_pool[:, n_gal + g] == 0
            xe = x_pool[det, g]
            num = (ndtr((xe[:, None] - mu[det]) / s) - ndtr((3 * s - mu[det]) / s)).mean(axis=1)
            pit[g] = num / np.maximum(p_det_pred[det, g], 1e-300)
        if (g + 1) % 10 == 0:
            print(f"[exact] {g + 1}/{n_gal} galaxies", flush=True)

    # ---- validation against the pool's simulated observations ----
    if args.validate:
        det_sim = x_pool[:, n_gal:] == 0
        z_score = (det_sim.mean(0) - p_det_pred.mean(0)) / np.sqrt(
            np.maximum(p_det_pred.mean(0) * (1 - p_det_pred.mean(0)), 1e-12) / n_pool)
        allpit = np.concatenate([p for p in pit if p is not None and len(p)])
        ks = np.abs(np.sort(allpit) - (np.arange(len(allpit)) + 0.5) / len(allpit)).max()
        print(f"[validate] detection frequency (pool) vs predicted, per galaxy: max |z| = "
              f"{np.abs(z_score).max():.2f}, mean |z| = {np.abs(z_score).mean():.2f} (MC + binomial noise: ~1)")
        print(f"[validate] PIT of {len(allpit)} simulated detected EWs: KS distance {ks:.4f} "
              f"(uniform at this n: ~{1.36 / np.sqrt(len(allpit)):.4f} at 95%)")
        if args.ew_model != 'exponential':
            print("[validate] NOTE: only meaningful if --pool_dir x was simulated with this --ew_model")

    # ---- posterior, attribution, comparison with the network ----
    L = loglik.sum(axis=1)
    w_full, ess_full, p_full = weights_and_map(L, theta_grid)
    print(f"[exact] exact-likelihood posterior: ESS {ess_full:.1f} of {n_pool}")
    if args.compare_nn:
        nn = np.load(os.path.expanduser(args.compare_nn))
        lr = nn['log_ratio'].astype(float)
        good = np.isfinite(L)
        print(f"[compare] network ESS {float(nn['ess']):.1f}; corr(log w_exact, log w_nn) over the pool = "
              f"{np.corrcoef(L[good], lr[good])[0, 1]:.3f}; max |dP| between the maps = "
              f"{np.abs(p_full - nn['marginal_map']).max():.3f}; mean P: exact {p_full.mean():.3f}, "
              f"network {nn['marginal_map'].mean():.3f}")
    sets = [[g] for g in range(n_gal)] + parse_groups(args.groups, n_gal)
    labels = [str(s[0]) if len(s) == 1 else ','.join(map(str, s)) for s in sets]
    p_minus = np.empty((len(sets), n_gal, n_los)); ess_minus = np.empty(len(sets))
    for si, members in enumerate(sets):
        _, ess_minus[si], p_minus[si] = weights_and_map(L - loglik[:, members].sum(axis=1), theta_grid)

    np.savez(args.output, p_full=p_full, p_minus=p_minus, influence=p_full[None] - p_minus,
             ess_full=ess_full, ess_minus=ess_minus, set_labels=np.array(labels),
             set_members=np.array([np.pad(s, (0, n_gal - len(s)), constant_values=-1) for s in sets]),
             x_obs=np.concatenate([ew_obs, is_ul.astype(float)]).astype(np.float32), n_draws=M,
             redshifts=z_gal_red, n_gal=n_gal, n_los=n_los, loglik=loglik.astype(np.float32),
             off_support=off_support, ew_model=args.ew_model)
    print(f"[exact] saved {args.output}")


if __name__ == '__main__':
    main()
