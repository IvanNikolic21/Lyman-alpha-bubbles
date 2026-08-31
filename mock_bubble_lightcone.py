"""
Prototype v2: fixes v1's edge-effect bug (bubble centers were confined to
the exact evaluation box, so bubbles with radius comparable to/larger than
the box's transverse half-width had most of their volume wasted outside it
-- realized ionized fraction came out ~0.54 against a 0.70 target, confirmed
systematic via a 20-seed variance check, not sampling noise). Fix: generate
bubble centers in a region PADDED by the max possible radius (R_HI) beyond
the real evaluation box, then only ray-trace sightlines within the
unpadded, real region.

Also adds what was decided while the padding-fix check ran: the target
ionized fraction is itself unknown and should vary across the mock
database, drawn from a CONTINUOUS prior (not a fixed value or discrete
grid) -- IONIZED_FRAC_PRIOR below, currently Uniform(0.1, 0.9) as a
starting default, easily adjustable. Per the same conversation, full
LOS/redshift evolution (bubbles growing as z decreases toward z_end=5.3) is
explicitly DEFERRED -- inference will likely focus on structure near the
actual galaxy cluster (z~6.9-7.3), not the full path to z_end, so a single
static ionized fraction per mock lightcone is the right scope for now.

SECOND continuous prior axis added after a real-vs-mock structural
comparison (see synthetic-bubble-mock-prior memory): ray-tracing real
galaxy skewers through a real 21cmFAST field vs. a fixed-BSD mock showed
the mock produces much more RIGID cross-galaxy bands (many galaxies flip
together across one oversized bubble) than the real field's finer,
individually-varying texture. Root cause, characterized quantitatively:
the fitted R_BUB_DIST_PARAMS' exponential tail is strongly volume-top-heavy
(the largest ~13% of bubbles by count contribute HALF the total ionized
volume). Rather than modeling explicit density-field correlation, this now
also varies the bubble-size distribution's TAIL WEIGHT itself as a second
continuous prior, K_FACTOR_PRIOR below (multiplies the fitted exponnorm K,
log-uniform since it's a multiplicative knob) -- confirmed via
compare_bsd_tailweight.py's sweep that bottom-heavy draws (K_FACTOR<1)
give visibly finer, more realistic-looking texture, and top-heavy draws
(K_FACTOR>1) make the rigidity worse, so covering this whole range in the
prior should give the training set much better texture diversity than a
single fixed BSD. NOTE (flagged, not resolved): varying K alone also
shifts the mean bubble radius, conflating tail shape with typical size --
acceptable for now, a cleaner isolated-tail-shape-only version would need
`scale` to co-vary inversely with `K` to hold the mean fixed.

Generates N_MOCKS example lightcones (each: independent x_ion AND K_FACTOR
draw + Poisson bubble field), ray-traces the SAME real galaxy catalog
through each (catalog positions are DATA, fixed; only the field/theta
varies -- matching how sbi_real_data.py/sbi_pixel_field.py's `simulate`
already treats the real vs. stochastic split), and visualizes how the
resulting pixel masks vary across both prior axes.
"""
import importlib.util
import sys

import numpy as np
from astropy import units as u
from astropy.cosmology import Planck18 as Cosmo
from scipy.stats import exponnorm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/Users/dxf836/Documents/git_code/Lyman-alpha-bubbles"
rng = np.random.default_rng(0)

# ============================== 1. load the real catalog (once, fixed) =====
spec = importlib.util.spec_from_file_location("real_data_standalone", f"{REPO}/lyabubbles/real_data.py")
rdmod = importlib.util.module_from_spec(spec)
sys.modules["real_data_standalone"] = rdmod
spec.loader.exec_module(rdmod)

cat = rdmod.load_catalog_v2(f"{REPO}/tb_lya.txt", f"{REPO}/sample_nirspec_properties.txt",
                            z_min=5.0, muv_max=-18.0, prefer="grating")
Z_HI = 7.3
in_window = cat.redshift <= Z_HI
ra, dec, redshift = cat.ra[in_window], cat.dec[in_window], cat.redshift[in_window]
n_gal = len(ra)

x_gal, y_gal, z_gal, ra0, dec0, z0, x_mean, y_mean, z_mean = rdmod.radec_to_comoving(ra, dec, redshift)

Z_END = 5.3
d_c0    = Cosmo.comoving_distance(z0).to(u.Mpc).value
d_c_end = Cosmo.comoving_distance(Z_END).to(u.Mpc).value
z_end_offset = (d_c_end - d_c0) - z_mean

PAD_BOX = 1.3
x_half = PAD_BOX * np.abs(x_gal).max()
y_half = PAD_BOX * np.abs(y_gal).max()
z_lo = z_end_offset - 10.0
z_hi = PAD_BOX * np.abs(z_gal).max()
eval_vol = (2 * x_half) * (2 * y_half) * (z_hi - z_lo)
print(f"[catalog] {n_gal} galaxies. Evaluation box: x_half={x_half:.2f} y_half={y_half:.2f} Mpc, "
      f"z in [{z_lo:.2f},{z_hi:.2f}] Mpc, volume={eval_vol:.3e} Mpc^3", flush=True)

# ============================== 2. bubble-size distribution + edge-effect fix
K_FIT, LOC, SCALE = 1.787595494643556, 8.57306023571156, 2.438386898231336   # real_data_run.py's R_BUB_DIST_PARAMS
R_LO, R_HI = 0.5, 60.0

def draw_radii(n, rng_, K, n_grid=4000):
    """K is now a per-mock draw (K_FIT * k_factor), not a module-level
    constant -- f_lo/f_hi depend on K too, so recomputed each call.

    Inverse-CDF sampling via a fine interpolation table, NOT
    exponnorm.ppf directly -- real bug found and fixed in
    generate_bubble_mocks.py (see that script's draw_radii docstring):
    exponnorm.ppf has no closed form and falls back to per-element
    numerical root-finding, which occasionally hangs for a very long time
    on extreme tail probabilities (this truncation's f_lo sits near the
    R_LO=0.5 bound, deep in the tail). exponnorm.cdf, by contrast, is
    closed-form and effectively instant -- building the CDF grid once per
    K and inverting via np.interp is both far faster AND removes the
    unpredictable-hang failure mode entirely."""
    r_grid = np.linspace(R_LO, R_HI, n_grid)
    cdf_grid = exponnorm.cdf(r_grid, K, LOC, SCALE)
    f_lo, f_hi = cdf_grid[0], cdf_grid[-1]
    u_ = rng_.uniform(size=n)
    return np.interp(f_lo + u_ * (f_hi - f_lo), cdf_grid, r_grid)

# EDGE-EFFECT FIX: generate bubble centers in a region padded by R_HI (the
# max possible radius) beyond the real evaluation box -- any bubble that
# COULD reach into the evaluation region is captured; bubbles centered
# beyond that genuinely cannot reach in, so the padding is exactly
# sufficient, not arbitrarily conservative.
PAD_R = R_HI
gx_half, gy_half = x_half + PAD_R, y_half + PAD_R
gz_lo, gz_hi = z_lo - PAD_R, z_hi + PAD_R
gen_vol = (2 * gx_half) * (2 * gy_half) * (gz_hi - gz_lo)

# ============================== 3. continuous priors: ionized fraction + BSD tail weight
IONIZED_FRAC_PRIOR = (0.1, 0.9)   # Uniform(lo, hi) -- ADJUSTABLE, first default
K_FACTOR_PRIOR = (0.3, 3.0)       # log-Uniform(lo, hi) -- multiplies K_FIT; see module docstring

N_MOCKS = 9
N_LOS = 75

mocks = []
for m in range(N_MOCKS):
    mrng = np.random.default_rng(1000 + m)
    x_ion = mrng.uniform(*IONIZED_FRAC_PRIOR)
    k_factor = np.exp(mrng.uniform(np.log(K_FACTOR_PRIOR[0]), np.log(K_FACTOR_PRIOR[1])))
    K = K_FIT * k_factor

    mean_vol = np.mean(4 / 3 * np.pi * draw_radii(300_000, np.random.default_rng(999), K) ** 3)
    lam = -np.log(1 - x_ion) / mean_vol
    n_b = mrng.poisson(lam * gen_vol)
    centers = np.column_stack([
        mrng.uniform(-gx_half, gx_half, n_b),
        mrng.uniform(-gy_half, gy_half, n_b),
        mrng.uniform(gz_lo, gz_hi, n_b),
    ])
    radii = draw_radii(n_b, mrng, K)

    mask_matrix = np.zeros((n_gal, N_LOS), dtype=int)
    for i in range(n_gal):
        z_samples = np.linspace(z_end_offset, z_gal[i], N_LOS)
        pts = np.column_stack([np.full(N_LOS, x_gal[i]), np.full(N_LOS, y_gal[i]), z_samples])
        d2 = np.sum((pts[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        mask_matrix[i] = np.any(d2 <= radii[None, :] ** 2, axis=1).astype(int)

    realized = mask_matrix.mean()
    mocks.append(dict(x_ion_target=x_ion, k_factor=k_factor, K=K, n_bubbles=n_b,
                      mask=mask_matrix, realized=realized))
    print(f"[mock {m}] target x_ion={x_ion:.3f}  k_factor={k_factor:.2f} (K={K:.2f})  "
          f"n_bubbles(padded gen. region)={n_b:4d}  realized pixel-ionized-frac={realized:.3f}",
          flush=True)

np.savez(f"{REPO}/mock_bubble_lightcone_v2.npz",
         redshift=redshift, x_gal=x_gal, y_gal=y_gal, z_gal=z_gal,
         masks=np.stack([mk["mask"] for mk in mocks]),
         x_ion_targets=np.array([mk["x_ion_target"] for mk in mocks]),
         k_factors=np.array([mk["k_factor"] for mk in mocks]),
         realized_fracs=np.array([mk["realized"] for mk in mocks]))
print(f"[saved] {REPO}/mock_bubble_lightcone_v2.npz", flush=True)

# ============================== 4. visualize: small multiples across the prior
INK_PRIMARY, INK_SECOND, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE = "#fcfcfb"
C_ION, C_NEU = "#1baf7a", "#383835"
cmap = matplotlib.colors.ListedColormap([C_NEU, C_ION])

order = np.argsort(redshift)
mocks_sorted = sorted(mocks, key=lambda mk: mk["x_ion_target"])

fig, axes = plt.subplots(3, 3, figsize=(13, 10))
fig.patch.set_facecolor(SURFACE)
for ax, mk in zip(axes.flat, mocks_sorted):
    ax.set_facecolor(SURFACE)
    ax.imshow(mk["mask"][order], aspect="auto", cmap=cmap, vmin=0, vmax=1,
              extent=[0, N_LOS, 0, n_gal], origin="lower")
    ax.set_title(f"x_ion={mk['x_ion_target']:.2f} (realized={mk['realized']:.2f})  "
                 f"k_factor={mk['k_factor']:.2f}",
                 color=INK_PRIMARY, fontsize=9.5, loc="left")
    ax.tick_params(colors=INK_MUTED, labelsize=7)
axes[-1, 1].set_xlabel("LOS bin (0=z_end, far=galaxy)", color=INK_SECOND, fontsize=9)
axes[1, 0].set_ylabel("galaxy (sorted by redshift)", color=INK_SECOND, fontsize=9)

from matplotlib.patches import Patch
fig.legend(handles=[Patch(facecolor=C_ION, label="ionized"), Patch(facecolor=C_NEU, label="neutral")],
           loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02),
           labelcolor=INK_SECOND)
fig.suptitle("Synthetic bubble mocks v2 -- continuous ionized-fraction + BSD tail-weight priors",
             color=INK_PRIMARY, fontsize=14, fontweight="bold", x=0.02, ha="left", y=1.01)
plt.tight_layout()
fig.savefig(f"{REPO}/mock_bubble_lightcone_v2.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
print(f"[saved] {REPO}/mock_bubble_lightcone_v2.png", flush=True)

# Realized vs target check across all mocks (should track the diagonal closely now)
fig2, ax2 = plt.subplots(figsize=(5, 5))
fig2.patch.set_facecolor(SURFACE); ax2.set_facecolor(SURFACE)
targets = np.array([mk["x_ion_target"] for mk in mocks])
realized = np.array([mk["realized"] for mk in mocks])
ax2.plot([0, 1], [0, 1], color=INK_MUTED, lw=1, ls="--", zorder=1)
ax2.scatter(targets, realized, color=C_ION, s=40, zorder=3, edgecolor=INK_PRIMARY, linewidth=0.5)
ax2.set_xlabel("target x_ion", color=INK_SECOND); ax2.set_ylabel("realized pixel-ionized fraction", color=INK_SECOND)
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
ax2.tick_params(colors=INK_MUTED)
ax2.set_title("Edge-effect fix check: realized vs. target", color=INK_PRIMARY, fontsize=11, loc="left")
plt.tight_layout()
fig2.savefig(f"{REPO}/mock_bubble_lightcone_v2_check.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
print(f"[saved] {REPO}/mock_bubble_lightcone_v2_check.png", flush=True)
