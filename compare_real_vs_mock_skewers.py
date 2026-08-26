"""
The actually-relevant comparison (per user): not a field-level statistic,
but the real object this pipeline trains on -- per-galaxy LOS skewers
(pixel-field-sbi's (n_gal, N_LOS) binary mask). Ray-traces the SAME real
galaxy catalog positions through BOTH the real 21cmFAST field and the
matched synthetic mock (both already generated/saved by
compare_real_vs_mock.py -> real_vs_mock_fields.npz), and compares the
resulting skewer masks side by side.

Scope: only one box-length (384 Mpc) of depth, not the full ~735 Mpc to
z_end=5.3 -- matches the user's own explicit recent guidance to focus on
structure near the actual galaxy cluster rather than the full LOS. Real
galaxies' small transverse footprint (~28x29 Mpc) comfortably fits inside
the 384 Mpc box with no wrapping needed; march each galaxy's sightline
across the box's full z-depth as one representative slice (no periodic
augmentation -- kept simple for this rough/illustrative comparison, per
"roughly, in the broad sense").
"""
import importlib.util
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = "/Users/dxf836/Documents/git_code/Lyman-alpha-bubbles"

# ============================== 1. real galaxy positions (fixed, real data)
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
x_gal, y_gal, z_gal, *_ = rdmod.radec_to_comoving(ra, dec, redshift)
print(f"[catalog] {n_gal} galaxies, transverse extent x in [{x_gal.min():.1f},{x_gal.max():.1f}], "
      f"y in [{y_gal.min():.1f},{y_gal.max():.1f}] Mpc -- comfortably inside the 384 Mpc box", flush=True)

# ============================== 2. load the real + mock fields (already generated)
d = np.load(f"{REPO}/real_vs_mock_fields.npz")
real_ionized = d["real_ionized"]   # (256,256,256), binary, 1=ionized
mock_ion = d["mock_ion"]
CELL_SIZE = float(d["cell_size"])
N_CELL = real_ionized.shape[0]
BOX_LEN_MPC = N_CELL * CELL_SIZE
half = BOX_LEN_MPC / 2
print(f"[fields] loaded real_vs_mock_fields.npz: box={BOX_LEN_MPC} Mpc, N_CELL={N_CELL}", flush=True)

# ============================== 3. ray-trace the real galaxy positions through each
N_LOS = 75

def ray_trace(field, x_gal, y_gal, n_gal, n_los, cell_size, half, n_cell):
    """One representative slice through `field`: each galaxy's fixed
    transverse (x,y) position, marched across the full box depth in z."""
    mask = np.zeros((n_gal, n_los), dtype=int)
    z_samples_mpc = np.linspace(-half + 0.5 * cell_size, half - 0.5 * cell_size, n_los)
    for i in range(n_gal):
        ix = int(np.clip(round((x_gal[i] + half) / cell_size), 0, n_cell - 1))
        iy = int(np.clip(round((y_gal[i] + half) / cell_size), 0, n_cell - 1))
        iz = np.clip(np.round((z_samples_mpc + half) / cell_size).astype(int), 0, n_cell - 1)
        mask[i] = field[ix, iy, iz]
    return mask

mask_real = ray_trace(real_ionized, x_gal, y_gal, n_gal, N_LOS, CELL_SIZE, half, N_CELL)
mask_mock = ray_trace(mock_ion, x_gal, y_gal, n_gal, N_LOS, CELL_SIZE, half, N_CELL)

print(f"[skewers] real: mean ionized fraction across skewers = {mask_real.mean():.3f}", flush=True)
print(f"[skewers] mock: mean ionized fraction across skewers = {mask_mock.mean():.3f}", flush=True)

# ============================== 4. visualize side by side ====================
INK_PRIMARY, INK_SECOND, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE = "#fcfcfb"
C_ION, C_NEU = "#1baf7a", "#383835"
cmap = matplotlib.colors.ListedColormap([C_NEU, C_ION])

order = np.argsort(redshift)
fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
fig.patch.set_facecolor(SURFACE)
for ax, mask, title, frac in [
    (axes[0], mask_real, "Real 21cmFAST field", mask_real.mean()),
    (axes[1], mask_mock, "Synthetic mock (matched x_ion)", mask_mock.mean()),
]:
    ax.set_facecolor(SURFACE)
    ax.imshow(mask[order], aspect="auto", cmap=cmap, vmin=0, vmax=1,
              extent=[0, N_LOS, 0, n_gal], origin="lower")
    ax.set_title(f"{title}\nskewer-mean ionized fraction={frac:.3f}", color=INK_PRIMARY, fontsize=11, loc="left")
    ax.set_xlabel("LOS bin (one box length, 384 Mpc)", color=INK_SECOND, fontsize=9.5)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
axes[0].set_ylabel("galaxy (sorted by redshift)", color=INK_SECOND, fontsize=9.5)

fig.legend(handles=[Patch(facecolor=C_ION, label="ionized"), Patch(facecolor=C_NEU, label="neutral")],
           loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.03),
           labelcolor=INK_SECOND)
fig.suptitle("Real galaxy skewers: real 21cmFAST field vs. synthetic mock (same box, matched x_ion)",
             color=INK_PRIMARY, fontsize=13.5, fontweight="bold", x=0.02, ha="left", y=1.02)
plt.tight_layout()
fig.savefig(f"{REPO}/real_vs_mock_skewers.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
print(f"[saved] {REPO}/real_vs_mock_skewers.png", flush=True)
