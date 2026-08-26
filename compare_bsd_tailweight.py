"""
Vary the bubble-size distribution's tail weight (K, the exponnorm exponential-
tail parameter relative to its Gaussian core) as a continuous prior axis, and
check how much it moves the cross-galaxy skewer rigidity found when comparing
the fixed-BSD mock against a real 21cmFAST field.

K_FACTOR multiplies the already-validated fitted K (R_BUB_DIST_PARAMS, see
bubble-size-prior memory) -- K_FACTOR=1.0 reproduces exactly what's already
been used. K_FACTOR<1 ("bottom-heavy"): suppresses the exponential tail,
more Gaussian-like, no single bubble dominates. K_FACTOR>1 ("top-heavy"):
heavier tail, more volume-dominated by rare giants -- expected to make the
rigid-banding effect WORSE, not better, which is itself a useful check.

NOTE (flagged honestly, not hidden): varying K alone also shifts the mean
radius (mean = loc + K*scale), so this conflates "tail shape" with "typical
size" rather than isolating tail shape at fixed mean. For a rough
exploratory check of whether tail-weight diversity helps, that coupling is
arguably fine (a "top-heavy" scenario colloquially IS "dominated by bigger
structures" for either reason) -- but flagged in case a cleaner isolated-
tail-shape version is wanted later (would need scale to co-vary inversely
with K to hold the mean fixed).

Reuses the already-validated padding-fix rasterization + ray-tracing
machinery from compare_real_vs_mock.py / compare_real_vs_mock_skewers.py.
"""
import importlib.util
import sys

import numpy as np
from scipy.stats import exponnorm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = "/Users/dxf836/Documents/git_code/Lyman-alpha-bubbles"

# ============================== 1. real galaxy positions + real/mock fields
spec = importlib.util.spec_from_file_location("real_data_standalone", f"{REPO}/lyabubbles/real_data.py")
rdmod = importlib.util.module_from_spec(spec)
sys.modules["real_data_standalone"] = rdmod
spec.loader.exec_module(rdmod)

cat = rdmod.load_catalog_v2(f"{REPO}/tb_lya.txt", f"{REPO}/sample_nirspec_properties.txt",
                            z_min=5.0, muv_max=-18.0, prefer="grating")
in_window = cat.redshift <= 7.3
ra, dec, redshift = cat.ra[in_window], cat.dec[in_window], cat.redshift[in_window]
n_gal = len(ra)
x_gal, y_gal, z_gal, *_ = rdmod.radec_to_comoving(ra, dec, redshift)

d = np.load(f"{REPO}/real_vs_mock_fields.npz")
real_ionized = d["real_ionized"]
CELL_SIZE = float(d["cell_size"])
N_CELL = real_ionized.shape[0]
BOX_LEN_MPC = N_CELL * CELL_SIZE
half = BOX_LEN_MPC / 2
target_x_ion = float(real_ionized.mean())
print(f"[setup] {n_gal} galaxies, box={BOX_LEN_MPC} Mpc, target_x_ion={target_x_ion:.4f}", flush=True)

N_LOS = 75

def ray_trace(field, x_gal, y_gal, n_gal, n_los, cell_size, half, n_cell):
    mask = np.zeros((n_gal, n_los), dtype=int)
    z_samples_mpc = np.linspace(-half + 0.5 * cell_size, half - 0.5 * cell_size, n_los)
    for i in range(n_gal):
        ix = int(np.clip(round((x_gal[i] + half) / cell_size), 0, n_cell - 1))
        iy = int(np.clip(round((y_gal[i] + half) / cell_size), 0, n_cell - 1))
        iz = np.clip(np.round((z_samples_mpc + half) / cell_size).astype(int), 0, n_cell - 1)
        mask[i] = field[ix, iy, iz]
    return mask

mask_real = ray_trace(real_ionized, x_gal, y_gal, n_gal, N_LOS, CELL_SIZE, half, N_CELL)

# ============================== 2. sweep the tail-weight K_FACTOR ===========
K_FIT, LOC, SCALE = 1.787595494643556, 8.57306023571156, 2.438386898231336
R_LO, R_HI = 0.5, 60.0
PAD_R = R_HI

VARIANTS = {"bottom-heavy (K x0.3)": 0.3, "baseline (fit, K x1.0)": 1.0, "top-heavy (K x3.0)": 3.0}

results = {}
for label, k_factor in VARIANTS.items():
    K = K_FIT * k_factor
    rng = np.random.default_rng(7)
    # Inverse-CDF via an interpolation table, NOT exponnorm.ppf directly --
    # .ppf has no closed form and occasionally hangs on extreme tail
    # probabilities (real bug found while building generate_bubble_mocks.py,
    # see that script's draw_radii docstring); .cdf is closed-form/instant.
    _r_grid = np.linspace(R_LO, R_HI, 4000)
    _cdf_grid = exponnorm.cdf(_r_grid, K, LOC, SCALE)
    f_lo, f_hi = _cdf_grid[0], _cdf_grid[-1]

    def draw_radii(n, rng_, _r_grid=_r_grid, _cdf_grid=_cdf_grid, f_lo=f_lo, f_hi=f_hi):
        u_ = rng_.uniform(size=n)
        return np.interp(f_lo + u_ * (f_hi - f_lo), _cdf_grid, _r_grid)

    sample_r = draw_radii(300_000, np.random.default_rng(999))
    mean_r = sample_r.mean()
    mean_vol = np.mean(4 / 3 * np.pi * sample_r ** 3)
    # volume-concentration stat, same as the earlier characterization
    vol = sample_r ** 3
    order = np.argsort(-vol)
    cum = np.cumsum(vol[order]) / vol.sum()
    n_for_half = np.searchsorted(cum, 0.5) + 1
    frac_for_half = 100 * n_for_half / len(sample_r)

    gen_half = half + PAD_R
    gen_vol = (2 * gen_half) ** 3
    lam = -np.log(1 - target_x_ion) / mean_vol
    n_b = rng.poisson(lam * gen_vol)
    centers = rng.uniform(-gen_half, gen_half, size=(n_b, 3))
    radii = draw_radii(n_b, rng)

    grid_1d = (np.arange(N_CELL) + 0.5) * CELL_SIZE - half
    mock_ion = np.zeros((N_CELL, N_CELL, N_CELL), dtype=bool)

    def cell_index(coord):
        return int(np.clip(round((coord + half) / CELL_SIZE), 0, N_CELL))

    for cx, cy, cz, r in zip(centers[:, 0], centers[:, 1], centers[:, 2], radii):
        i_lo, i_hi = max(0, cell_index(cx - r)), min(N_CELL, cell_index(cx + r) + 1)
        j_lo, j_hi = max(0, cell_index(cy - r)), min(N_CELL, cell_index(cy + r) + 1)
        k_lo, k_hi = max(0, cell_index(cz - r)), min(N_CELL, cell_index(cz + r) + 1)
        if i_lo >= i_hi or j_lo >= j_hi or k_lo >= k_hi:
            continue
        sub_x = grid_1d[i_lo:i_hi][:, None, None]
        sub_y = grid_1d[j_lo:j_hi][None, :, None]
        sub_z = grid_1d[k_lo:k_hi][None, None, :]
        d2 = (sub_x - cx) ** 2 + (sub_y - cy) ** 2 + (sub_z - cz) ** 2
        mock_ion[i_lo:i_hi, j_lo:j_hi, k_lo:k_hi] |= (d2 <= r ** 2)
    mock_ion = mock_ion.astype(np.float32)

    mask_mock = ray_trace(mock_ion, x_gal, y_gal, n_gal, N_LOS, CELL_SIZE, half, N_CELL)

    results[label] = dict(K=K, mean_r=mean_r, n_bubbles=n_b, frac_for_half=frac_for_half,
                          realized_field=mock_ion.mean(), mask=mask_mock)
    print(f"[{label}] K={K:.3f} mean_r={mean_r:.2f} Mpc  n_bubbles={n_b}  "
          f"{frac_for_half:.1f}% of bubbles account for half the volume  "
          f"realized_field_xion={mock_ion.mean():.3f}", flush=True)

# ============================== 3. visualize ==================================
INK_PRIMARY, INK_SECOND, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE = "#fcfcfb"
C_ION, C_NEU = "#1baf7a", "#383835"
cmap = matplotlib.colors.ListedColormap([C_NEU, C_ION])
order_gal = np.argsort(redshift)

fig, axes = plt.subplots(1, 4, figsize=(19, 6.2))
fig.patch.set_facecolor(SURFACE)

panels = [("Real 21cmFAST field", mask_real, None)] + \
         [(label, r["mask"], r) for label, r in results.items()]

for ax, (title, mask, r) in zip(axes, panels):
    ax.set_facecolor(SURFACE)
    ax.imshow(mask[order_gal], aspect="auto", cmap=cmap, vmin=0, vmax=1,
              extent=[0, N_LOS, 0, n_gal], origin="lower")
    subtitle = f"skewer x_ion={mask.mean():.3f}"
    if r is not None:
        subtitle += f"\nK={r['K']:.2f}, mean_r={r['mean_r']:.1f} Mpc, {r['frac_for_half']:.0f}%->half vol"
    ax.set_title(f"{title}\n{subtitle}", color=INK_PRIMARY, fontsize=9.5, loc="left")
    ax.set_xlabel("LOS bin", color=INK_SECOND, fontsize=9)
    ax.tick_params(colors=INK_MUTED, labelsize=7.5)
axes[0].set_ylabel("galaxy (sorted by redshift)", color=INK_SECOND, fontsize=9.5)

fig.legend(handles=[Patch(facecolor=C_ION, label="ionized"), Patch(facecolor=C_NEU, label="neutral")],
           loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.03),
           labelcolor=INK_SECOND)
fig.suptitle("Bubble-size-distribution tail-weight sweep vs. real field (same box, matched x_ion)",
             color=INK_PRIMARY, fontsize=14, fontweight="bold", x=0.02, ha="left", y=1.03)
plt.tight_layout()
fig.savefig(f"{REPO}/bsd_tailweight_sweep.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
print(f"[saved] {REPO}/bsd_tailweight_sweep.png", flush=True)
