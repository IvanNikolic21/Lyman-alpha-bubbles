"""
Real-vs-mock structural comparison (rough, per user's explicit framing).

Loads the real neutral_fraction field extracted from a cached single-box
snapshot (z=6.5, x_H=0.271, see extract_real_snapshot_field.py), binarizes
it (threshold=0.5, matching pixel-field-sbi's discretize_to_fixed_bins
convention: field value = local NEUTRAL fraction), generates a matched
synthetic bubble mock on the SAME 384 Mpc / 256^3 grid targeting the real
field's own REALIZED ionized fraction (not the nominal snapshot-table
value, for the fairest possible comparison), and compares:
  1. The isotropic 2-point correlation function xi(r) of the binary
     ionized field (FFT/Wiener-Khinchin estimator) -- this is the actual
     "is the mock's clustering structure realistic" question flagged in
     synthetic-bubble-mock-prior memory.
  2. A visual 2D slice through each field.

No py21cmfast needed -- pure numpy/scipy once the real field is a local .npy.
"""
import os

import numpy as np
from scipy.stats import exponnorm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/Users/dxf836/Documents/git_code/Lyman-alpha-bubbles"
rng = np.random.default_rng(42)

# ============================== 1. load + binarize the real field ==========
real_path = os.path.expanduser("~/Downloads/real_snapshot_z6.5000_field.npy")
real_neutral_frac = np.load(real_path)   # (256,256,256), continuous local neutral fraction
print(f"[real] loaded {real_path}, shape={real_neutral_frac.shape}, "
      f"mean(neutral)={real_neutral_frac.mean():.4f} (expected ~0.271)", flush=True)

BOX_LEN_MPC = 384.0
N_CELL = real_neutral_frac.shape[0]
CELL_SIZE = BOX_LEN_MPC / N_CELL
print(f"[real] box={BOX_LEN_MPC} Mpc, N_CELL={N_CELL}, cell_size={CELL_SIZE:.3f} Mpc", flush=True)

# theta=1 means NEUTRAL in this project's convention (pixel-field-sbi memory) --
# binarize then flip to get an ionized=1 field for a direct match to the mock.
real_binary_neutral = (real_neutral_frac >= 0.5).astype(np.float32)
real_ionized = 1.0 - real_binary_neutral
target_x_ion = float(real_ionized.mean())
print(f"[real] binarized (threshold=0.5): realized ionized fraction = {target_x_ion:.4f}", flush=True)

# ============================== 2. matched mock, rasterized on the SAME grid
R_BUB_DIST_PARAMS = (1.787595494643556, 8.57306023571156, 2.438386898231336)
R_LO, R_HI = 0.5, 60.0
# Inverse-CDF via an interpolation table, NOT exponnorm.ppf directly --
# .ppf has no closed form and occasionally hangs on extreme tail
# probabilities (real bug found while building generate_bubble_mocks.py,
# see that script's draw_radii docstring); .cdf is closed-form/instant.
_r_grid = np.linspace(R_LO, R_HI, 4000)
_cdf_grid = exponnorm.cdf(_r_grid, *R_BUB_DIST_PARAMS)
f_lo, f_hi = _cdf_grid[0], _cdf_grid[-1]

def draw_radii(n, rng_):
    u_ = rng_.uniform(size=n)
    return np.interp(f_lo + u_ * (f_hi - f_lo), _cdf_grid, _r_grid)

mean_vol = np.mean(4 / 3 * np.pi * draw_radii(300_000, np.random.default_rng(999)) ** 3)

half = BOX_LEN_MPC / 2
PAD_R = R_HI   # same edge-effect fix as mock_bubble_lightcone.py
gen_half = half + PAD_R
gen_vol = (2 * gen_half) ** 3

lam = -np.log(1 - target_x_ion) / mean_vol
n_b = rng.poisson(lam * gen_vol)
centers = rng.uniform(-gen_half, gen_half, size=(n_b, 3))
radii = draw_radii(n_b, rng)
print(f"[mock] targeting x_ion={target_x_ion:.4f} (matched to the real field's own realized value) "
      f"-> n_bubbles={n_b}", flush=True)

# Rasterize bubble-by-bubble via a local bounding-box slice (NOT a full
# (N_CELL^3, n_bubbles) distance matrix -- would be far too much memory at
# n_bubbles ~ thousands and N_CELL^3 ~ 16.7M).
grid_1d = (np.arange(N_CELL) + 0.5) * CELL_SIZE - half
mock_ion = np.zeros((N_CELL, N_CELL, N_CELL), dtype=bool)

def cell_index(coord):
    return int(np.clip(round((coord + half) / CELL_SIZE), 0, N_CELL))

for cx, cy, cz, r in zip(centers[:, 0], centers[:, 1], centers[:, 2], radii):
    i_lo, i_hi = cell_index(cx - r), cell_index(cx + r) + 1
    j_lo, j_hi = cell_index(cy - r), cell_index(cy + r) + 1
    k_lo, k_hi = cell_index(cz - r), cell_index(cz + r) + 1
    i_lo, i_hi = max(0, i_lo), min(N_CELL, i_hi)
    j_lo, j_hi = max(0, j_lo), min(N_CELL, j_hi)
    k_lo, k_hi = max(0, k_lo), min(N_CELL, k_hi)
    if i_lo >= i_hi or j_lo >= j_hi or k_lo >= k_hi:
        continue
    sub_x = grid_1d[i_lo:i_hi][:, None, None]
    sub_y = grid_1d[j_lo:j_hi][None, :, None]
    sub_z = grid_1d[k_lo:k_hi][None, None, :]
    d2 = (sub_x - cx) ** 2 + (sub_y - cy) ** 2 + (sub_z - cz) ** 2
    mock_ion[i_lo:i_hi, j_lo:j_hi, k_lo:k_hi] |= (d2 <= r ** 2)

mock_ion = mock_ion.astype(np.float32)
print(f"[mock] rasterized realized ionized fraction = {mock_ion.mean():.4f}", flush=True)

np.savez(f"{REPO}/real_vs_mock_fields.npz", real_ionized=real_ionized, mock_ion=mock_ion,
         target_x_ion=target_x_ion, cell_size=CELL_SIZE, n_bubbles=n_b)
print(f"[saved] {REPO}/real_vs_mock_fields.npz", flush=True)

# ============================== 3. isotropic 2-point correlation function ==
def compute_xi_isotropic(field, cell_size, r_max=120.0, n_bins=24):
    delta = field / field.mean() - 1.0
    fft = np.fft.fftn(delta)
    power = (fft * np.conj(fft)).real
    xi_grid = np.fft.ifftn(power).real / field.size
    N = field.shape[0]
    idx = np.fft.fftfreq(N, d=1.0) * N
    ix, iy, iz = np.meshgrid(idx, idx, idx, indexing="ij")
    r_grid = np.sqrt(ix ** 2 + iy ** 2 + iz ** 2) * cell_size
    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[1:] + r_edges[:-1])
    xi_r = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (r_grid >= r_edges[i]) & (r_grid < r_edges[i + 1])
        if mask.any():
            xi_r[i] = xi_grid[mask].mean()
    return r_centers, xi_r

print("[xi] computing real field's 2-point correlation function...", flush=True)
r_real, xi_real = compute_xi_isotropic(real_ionized, CELL_SIZE)
print("[xi] computing mock field's 2-point correlation function...", flush=True)
r_mock, xi_mock = compute_xi_isotropic(mock_ion, CELL_SIZE)

# ============================== 4. visualize =================================
INK_PRIMARY, INK_SECOND, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID = "#fcfcfb", "#e1e0d9"
C_REAL, C_MOCK = "#2a78d6", "#eb6834"
cmap_bin = matplotlib.colors.ListedColormap(["#383835", "#1baf7a"])

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
fig.patch.set_facecolor(SURFACE)

axA = axes[0]
axA.set_facecolor(SURFACE)
axA.imshow(real_ionized[:, :, N_CELL // 2], cmap=cmap_bin, vmin=0, vmax=1, origin="lower")
axA.set_title(f"Real field slice (z=6.5 snapshot)\nrealized x_ion={target_x_ion:.3f}",
              color=INK_PRIMARY, fontsize=10.5, loc="left")
axA.tick_params(colors=INK_MUTED, labelsize=7)

axB = axes[1]
axB.set_facecolor(SURFACE)
axB.imshow(mock_ion[:, :, N_CELL // 2], cmap=cmap_bin, vmin=0, vmax=1, origin="lower")
axB.set_title(f"Synthetic mock slice (same grid)\nrealized x_ion={mock_ion.mean():.3f}",
              color=INK_PRIMARY, fontsize=10.5, loc="left")
axB.tick_params(colors=INK_MUTED, labelsize=7)

axC = axes[2]
axC.set_facecolor(SURFACE)
axC.plot(r_real, xi_real, color=C_REAL, lw=2, label="real (21cmFAST)")
axC.plot(r_mock, xi_mock, color=C_MOCK, lw=2, label="synthetic mock")
axC.axhline(0, color=INK_MUTED, lw=0.8, ls="--")
axC.set_xlabel("separation r [Mpc]", color=INK_SECOND, fontsize=10)
axC.set_ylabel(r"$\xi(r)$  (ionization 2-pt correlation)", color=INK_SECOND, fontsize=10)
axC.set_title("2-point correlation function", color=INK_PRIMARY, fontsize=10.5, loc="left")
axC.legend(frameon=False, fontsize=9, labelcolor=INK_SECOND)
axC.grid(True, color=GRID, linewidth=0.8)
axC.tick_params(colors=INK_MUTED, labelsize=8)

fig.suptitle("Real 21cmFAST field vs. synthetic bubble mock -- rough structural comparison",
             color=INK_PRIMARY, fontsize=13.5, fontweight="bold", x=0.02, ha="left", y=1.03)
plt.tight_layout()
fig.savefig(f"{REPO}/real_vs_mock_comparison.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
print(f"[saved] {REPO}/real_vs_mock_comparison.png", flush=True)
