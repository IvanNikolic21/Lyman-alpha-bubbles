"""
bubble_prior_v2.py -- the second-generation prior on the ionization field along the 43 sightlines.

Same Boolean bubble model as generate_bubble_mocks.py (v1): ionized spheres with Poisson-distributed centres, a
sightline point is ionized if it lies inside any sphere. Three changes, each with its own prior parameter:

1. REDSHIFT EVOLUTION of the neutral fraction (v1: one x_ion for the whole volume z = 5.3 ... 7.3):
       x_HI(z) = 1/2 [1 + tanh((z - z_mid) / dz)],          x_ion(z) = 1 - x_HI(z).
   The prior is placed on the two numbers that matter for our data, not on (z_mid, dz) directly:
       x_HI at Z_REF = 7 (the galaxies' redshift)   x_hi_7 ~ U(X_HI_7_RANGE),
       end of reionization z_end, where x_HI = X_HI_END = 0.05,   z_end ~ U(Z_END_RANGE).
   They fix the tanh curve: with a_7 = artanh(2 x_hi_7 - 1) and a_e = artanh(2 X_HI_END - 1) = -1.472,
       dz = (Z_REF - z_end) / (a_7 - a_e),     z_mid = Z_REF - a_7 dz.
   (A uniform prior on z_mid in 5.8-10 instead -- the first version -- put the prior median of x_ion(z = 7) at 0.91,
   i.e. it strongly favoured an ionized z = 7 by itself.) Planck is NOT built in: each history's Thomson optical
   depth tau_e is stored (tau_reionization), so a CMB constraint can be applied afterwards as a deliberately broadened
   importance reweighting (Planck 2018 VI: tau = 0.054 +- 0.007). The bubble density varies along the line of sight,
       lambda(z) = -ln(1 - x_ion(z)) / <V>(z),
   so that a fraction x_ion(z) of space is covered at each z (Boolean model: covered fraction = 1 - exp(-lambda <V>);
   exact for slabs much thicker than the bubbles, approximate where bubbles are comparable to the evolution scale).

2. BUBBLE SIZES TIED TO THE IONIZED FRACTION (v1: one size distribution for all x_ion, stretched by k_factor):
   sphere radii are log-normal, ln r ~ N(ln(s R0), sigma^2) truncated at +- N_SIG_TRUNC sigma, with s(x_ion) and
   sigma(x_ion) from calibrate_bsd_lu24.py: the Boolean model's mean-free-path bubble sizes then match the 21cmFAST
   ones of Lu et al. 2024 (MNRAS 528, 4872) at every x_ion. A prior parameter zeta ~ U(ZETA_RANGE) interpolates between
   their two reionization models:
       ln s = (1 - zeta) ln s_gradual(x_ion) + zeta ln s_rapid(x_ion),   sigma likewise (linearly).
   zeta = 0: faint galaxies drive reionization (many small bubbles); zeta = 1: bright galaxies (fewer, larger).
   Outside the calibrated x_ion range [0.09, 0.90] the end values are used.

3. GALAXIES MAY HOST BUBBLES (v1: bubble centres are independent of the galaxies):
   each of the 43 galaxies, independently with probability f_host ~ U(F_HOST_RANGE), gets one additional sphere that
   contains it: radius from the size distribution at the galaxy's redshift, centre at a point drawn uniformly inside
   the sphere around the galaxy. f_host = 0 is the v1 assumption; the data decide (Lu+24 find galaxies 10-1000 times
   more likely than random points to sit in ionized regions -- we do not impose this, we let f_host measure it).

Outputs per mock, as v1: masks (n_gal, n_los) uint8, 1 = ionized, sightline points from z_end towards the galaxy;
plus all parameters (z_mid, dz, zeta, f_host), the number of bubbles, which galaxies got host bubbles, and the
realized ionized fraction along the sightlines in redshift slices.
"""
import numpy as np
from scipy.special import ndtr, ndtri       # standard-normal CDF Phi(x) and its inverse Phi^-1(p)
from scipy.spatial import cKDTree           # k-d tree: fast "which sightline points lie within r of a centre"
from astropy import units as u
from astropy.cosmology import Planck18 as Cosmo

# ---------------------------------------------------------------------------------------------------------------
# Prior ranges and fixed settings
# ---------------------------------------------------------------------------------------------------------------
Z_END = 5.3                     # low-z end of every sightline. Not used in this module: generate_bubble_mocks.py has
                                # its own Z_END = 5.3 and passes the corresponding LOS coordinate as z_end_offset
Z_REF = 7.0                     # redshift at which the neutral fraction is a prior parameter (≈ the galaxies' z)
X_HI_7_RANGE = (0.06, 0.95)     # lower bound must exceed X_HI_END: x_HI(7) < 0.05 would need z_end > 7
Z_END_RANGE = (5.3, 6.05)        # Lyman-α forest data tells us with high confidence EoR ends after z=6.0
X_HI_END = 0.05                 # "end of reionization" is DEFINED here as the redshift where x_HI = 0.05
ZETA_RANGE = (0.0, 1.0)         # 0 = Lu+24 gradual (faint-galaxy) bubble sizes, 1 = rapid (bright-galaxy) sizes
F_HOST_RANGE = (0.0, 1.0)       # probability that a galaxy gets its own host bubble; 0 = v1 (no association)
X_ION_CAP = 0.99           # lambda diverges as x_ion -> 1; above this the volume is treated as 99% ionized
SLAB = 2.0                 # cMpc: thickness of the line-of-sight slabs in which lambda is taken constant
Z_SLICES = np.array([5.3, 6.0, 6.5, 6.8, 7.0, 7.1, 7.2, 7.4])   # redshift slices for the realized fraction


def x_hi_of_z(z, z_mid, dz):
    """Neutral fraction of the tanh history."""
    # x_HI -> 1 at high z (z >> z_mid), x_HI -> 0 at low z; equals 1/2 at z = z_mid; dz sets how fast it changes
    # (from 0.88 to 0.12 between z_mid + dz and z_mid - dz). Works element-wise on arrays of z.
    return 0.5 * (1 + np.tanh((z - z_mid) / dz))


def draw_history(rng):
    """(z_mid, dz, x_hi_7, z_end): x_hi_7 and z_end from the prior, converted to the tanh parameters."""
    # the two prior parameters, both uniform
    x7, z_end = rng.uniform(*X_HI_7_RANGE), rng.uniform(*Z_END_RANGE)
    # Invert the tanh at the two anchor points. x_HI = (1 + tanh(a)) / 2  <=>  a = artanh(2 x_HI - 1), with
    # a = (z - z_mid) / dz. So the two conditions x_HI(Z_REF) = x7 and x_HI(z_end) = X_HI_END read
    #     (Z_REF - z_mid) / dz = a7,      (z_end - z_mid) / dz = ae.
    # Subtracting: (Z_REF - z_end) / dz = a7 - ae, which gives dz; then z_mid from the first condition.
    # a7 > ae always (x7 >= 0.06 > 0.05), so dz > 0. ae = artanh(-0.9) = -1.472 is a constant.
    a7, ae = np.arctanh(2 * x7 - 1), np.arctanh(2 * X_HI_END - 1)
    dz = (Z_REF - z_end) / (a7 - ae)
    return Z_REF - a7 * dz, dz, x7, z_end


def tau_reionization(z_mid, dz, z_max=30.0, n=3000):
    """Thomson optical depth of a tanh history (Planck18 cosmology): tau = c sigma_T int n_e / ((1+z) H) dz, with
    hydrogen and singly ionized helium following x_ion(z), and helium doubly ionized below z = 3 (as in CAMB)."""
    from astropy import constants as const
    z = np.linspace(0, z_max, n)                             # integration grid; dz_grid = 0.01
    xe = (1 - x_hi_of_z(z, z_mid, dz))                       # H II fraction
    Y = 0.245                                                # helium mass fraction
    f_he = Y / (4 * (1 - Y))                                 # He / H by number
    # free electrons per hydrogen nucleus: one per ionized H, one per singly ionized He (assumed to follow H),
    # and a second He electron below z = 3 (He II reionization, as CAMB's default)
    ne_per_nh = xe * (1 + f_he) + np.where(z < 3, f_he, 0.0)
    # comoving hydrogen number density today: n_H0 = rho_crit Omega_b (1 - Y) / m_p  [m^-3]
    n_h0 = (Cosmo.critical_density0 * Cosmo.Ob0 * (1 - Y) / const.m_p).to(u.m ** -3).value
    # d tau = c sigma_T n_e dt, with proper n_e = n_H0 (1+z)^3 ne_per_nh and dt = dz / ((1+z) H): one power of
    # (1+z) cancels, leaving (1+z)^2 / H(z). H in s^-1, so c [m/s] * sigma_T [m^2] * n [m^-3] / H [s^-1] is unitless.
    integrand = ne_per_nh * n_h0 * (1 + z) ** 2 / Cosmo.H(z).to(1 / u.s).value
    return (const.c.value * const.sigma_T.value) * np.trapz(integrand, z)


class BSD:
    """Calibrated log-normal sphere-radius distribution as a function of x_ion and zeta (calibrate_bsd_lu24.py)."""

    def __init__(self, calibration_path):
        # Lookup table written by calibrate_bsd_lu24.py: 'gradual' and 'rapid' are (9, 9) arrays, one row per Lu+24
        # x_ion value (sorted ascending, as np.interp needs), one column per quantity named in 'columns'.
        c = np.load(calibration_path)
        cols = list(c['columns'])
        g, r = c['gradual'], c['rapid']
        # find the columns by name, not by position, so a reordered calibration file is still read correctly
        ix, isg, i_s = cols.index('x_ion'), cols.index('sigma'), cols.index('s')
        # per model: x_ion grid, log-width sigma(x_ion), and ln s(x_ion). ln s, because s spans a factor ~36 and
        # params() interpolates and mixes the two models in ln s (a geometric mean of the two median radii)
        self.xg, self.sg_g, self.lns_g = g[:, ix], g[:, isg], np.log(g[:, i_s])
        self.xr, self.sg_r, self.lns_r = r[:, ix], r[:, isg], np.log(r[:, i_s])
        # R0: radius unit (median sphere radius = s R0 cMpc); c: truncation of the log-normal in units of sigma
        self.R0, self.c = float(c['R0']), float(c['n_sig_trunc'])

    def params(self, x_ion, zeta):
        """Median radius m = s R0 [cMpc] and log-width sigma at ionized fraction x_ion (array) for model zeta."""
        # np.interp holds the end values outside the grid, so x_ion < 0.09 or > 0.90 use the last calibrated point
        lns = (1 - zeta) * np.interp(x_ion, self.xg, self.lns_g) + zeta * np.interp(x_ion, self.xr, self.lns_r)
        sig = (1 - zeta) * np.interp(x_ion, self.xg, self.sg_g) + zeta * np.interp(x_ion, self.xr, self.sg_r)
        return np.exp(lns) * self.R0, sig

    def r_max(self, m, sig):
        # largest radius the truncated log-normal can produce: ln r_max = ln m + c sigma
        return m * np.exp(self.c * sig)

    def mean_volume(self, m, sig):
        """<4/3 pi r^3> of the truncated log-normal, analytic:
        E[e^{3 sig Z} | |Z| < c] = e^{9 sig^2 / 2} [Phi(c - 3 sig) - Phi(-c - 3 sig)] / [Phi(c) - Phi(-c)]."""
        # r = m e^{sig Z}, Z standard normal truncated to |Z| < c, so r^3 = m^3 e^{3 sig Z}. Completing the square in
        # the Gaussian integral gives the formula above: the e^{9 sig^2 / 2} is the untruncated log-normal moment,
        # the Phi ratio corrects for the truncation (the e^{3 sig Z} weight shifts the Gaussian by 3 sig).
        c = self.c
        return 4 / 3 * np.pi * m ** 3 * np.exp(4.5 * sig ** 2) * (ndtr(c - 3 * sig) - ndtr(-c - 3 * sig)) / (
            ndtr(c) - ndtr(-c))

    def draw(self, m, sig, rng):
        """One radius per element of m, sig (truncated log-normal by inverse CDF)."""
        # Inverse-CDF sampling of a truncated normal: a uniform u in [Phi(-c), Phi(c)] mapped through Phi^-1 is a
        # standard normal restricted to |Z| < c (no rejection loop). Then r = m e^{sig Z}.
        lo, hi = ndtr(-self.c), ndtr(self.c)
        return m * np.exp(sig * ndtri(lo + rng.uniform(size=np.shape(m)) * (hi - lo)))


class LOSRedshift:
    """Redshift of a line-of-sight coordinate of the mock frame: d_C(z) = d_C(z0) + z_mean + zc
    (lyabubbles.real_data.radec_to_comoving centres the LOS offsets d_C - d_C(z0) on their mean z_mean)."""
    # The mock frame's LOS coordinate zc is a comoving distance in cMpc, centred on the galaxies. To evaluate
    # x_ion(z) we need the redshift of each zc: undo the centring (add d_C(z0) + z_mean) to get the comoving
    # distance d_C, then invert d_C(z) by interpolating a fine table (d_C is monotonic in z).

    def __init__(self, z0, z_mean):
        zg = np.linspace(4.5, 9.0, 4000)                     # covers the sightlines (5.3 ... ~7.3) plus the padding
        self.zg, self.dg = zg, Cosmo.comoving_distance(zg).to(u.Mpc).value
        self.offset = Cosmo.comoving_distance(z0).to(u.Mpc).value + z_mean

    def __call__(self, zc):
        return np.interp(self.offset + np.asarray(zc), self.dg, self.zg)


def generate_one_mock_v2(rng, z_mid, dz, zeta, f_host, bsd, los_z, geom, x_gal, y_gal, z_gal, z_end_offset,
                         n_los):
    """One mock of the v2 prior. Returns (mask, n_bubbles, hosted, realized_in_slices)."""
    # Coordinates (cMpc) are the mock frame of generate_bubble_mocks.py: x, y transverse, z along the line of sight,
    # all centred on the galaxy sample. Larger z = further away = higher redshift.
    n_gal = len(x_gal)
    # sightline sample points, exactly as v1 (generate_bubble_mocks.generate_one_mock): n_los points from z_end
    # to the galaxy, at the galaxy's transverse position
    # pts has shape (n_gal * n_los, 3); rows i * n_los ... (i + 1) * n_los - 1 belong to galaxy i
    pts = np.concatenate([np.column_stack([np.full(n_los, x_gal[i]), np.full(n_los, y_gal[i]),
                                           np.linspace(z_end_offset, z_gal[i], n_los)]) for i in range(n_gal)])
    tree = cKDTree(pts)

    # ---- 1. background bubbles, slab by slab along the line of sight ----
    zc_lo, zc_hi = geom['z_lo'], geom['z_hi']                # evaluation range of the sightline points
    # First pass: slabs over the evaluation range only, just to find the largest possible sphere (pad) ...
    edges = np.arange(zc_lo, zc_hi + SLAB, SLAB)
    mid = 0.5 * (edges[1:] + edges[:-1])
    x_ion = np.minimum(1 - x_hi_of_z(los_z(mid), z_mid, dz), X_ION_CAP)
    m, sig = bsd.params(x_ion, zeta)
    pad = float(bsd.r_max(m, sig).max())                    # largest sphere anywhere: extend the slabs by it
    # ... second pass: extend the slab range by pad on both sides (rounded up to whole slabs), so spheres whose
    # centres lie just outside the evaluation range, but reach into it, are generated too
    n_ext = int(np.ceil(pad / SLAB))
    edges = np.arange(zc_lo - n_ext * SLAB, zc_hi + (n_ext + 1) * SLAB, SLAB)
    mid = 0.5 * (edges[1:] + edges[:-1])
    x_ion = np.minimum(1 - x_hi_of_z(los_z(mid), z_mid, dz), X_ION_CAP)   # target ionized fraction per slab
    m, sig = bsd.params(x_ion, zeta)                        # size distribution per slab (tied to that slab's x_ion)
    # Boolean model: covered fraction = 1 - exp(-lambda <V>) = x_ion  =>  lambda = -ln(1 - x_ion) / <V>
    lam = -np.log(1 - x_ion) / bsd.mean_volume(m, sig)      # bubbles per cMpc^3 in each slab
    rmax = bsd.r_max(m, sig)
    # a slab's spheres can only reach the sightlines if their centres are within r_max of the field transversely
    hx, hy = geom['x_half'] + rmax, geom['y_half'] + rmax
    counts = rng.poisson(lam * (2 * hx) * (2 * hy) * SLAB)   # Poisson number of centres per slab (density x volume)
    k = np.repeat(np.arange(len(mid)), counts)              # slab index of every centre, e.g. [0, 0, 1, 3, 3, 3, ...]
    # centres uniform inside their slab's box (each slab has its own transverse half-widths hx[k], hy[k])
    centers = np.column_stack([rng.uniform(-hx[k], hx[k]), rng.uniform(-hy[k], hy[k]),
                               rng.uniform(edges[k], edges[k + 1])])
    radii = bsd.draw(m[k], sig[k], rng)                     # each radius from its own slab's size distribution

    # ---- 3. host bubbles around galaxies ----
    hosted = rng.uniform(size=n_gal) < f_host               # each galaxy independently, with probability f_host
    if hosted.any():
        # sphere sizes from the size distribution at each hosted galaxy's own redshift
        xg = np.minimum(1 - x_hi_of_z(los_z(z_gal[hosted]), z_mid, dz), X_ION_CAP)
        mh, sh = bsd.params(xg, zeta)
        rh = bsd.draw(mh, sh, rng)
        # Centre = galaxy + offset, with the offset uniform inside a ball of radius rh, so the galaxy lies inside
        # its sphere but not necessarily at the centre. Uniform in a ball: direction isotropic (a normalised 3D
        # Gaussian vector) and distance rh * u^(1/3) (the volume within distance d grows as d^3).
        d = rng.normal(size=(hosted.sum(), 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        off = d * (rh * rng.uniform(size=hosted.sum()) ** (1 / 3))[:, None]   # uniform inside the sphere
        centers = np.vstack([centers, np.column_stack([x_gal[hosted], y_gal[hosted], z_gal[hosted]]) + off])
        radii = np.concatenate([radii, rh])

    # ---- mask: sightline points inside any sphere ----
    # query_ball_point returns, for every sphere, the indices of the sightline points within its radius;
    # a point is ionized if at least one sphere lists it
    inside = np.zeros(len(pts), bool)
    for idx in tree.query_ball_point(centers, radii):
        inside[idx] = True
    mask = inside.reshape(n_gal, n_los).astype(np.uint8)    # row i = galaxy i, column 0 = z_end end of its sightline

    # realized ionized fraction of the sightline points in each redshift slice of Z_SLICES (a check of the target
    # x_ion(z); NaN for a slice that contains no sightline points)
    zp = los_z(pts[:, 2])
    sl = np.digitize(zp, Z_SLICES) - 1                      # slice index of every point (-1 / 7 = outside)
    realized = np.array([inside[sl == j].mean() if np.any(sl == j) else np.nan for j in range(len(Z_SLICES) - 1)])
    return mask, len(radii), hosted, realized
