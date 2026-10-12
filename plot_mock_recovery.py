"""
Plots mock_recovery_demo.py's output: for each toy example, TRUE theta
(the ionization mask that actually generated that example's x) side-by-side
with the INFERRED marginal probability map from running inference against
it -- the "does this work on mock data, where we know the answer" check.

What to actually look for (corrected after a wrong first read of this same
plot): NOT whether the inferred map recovers the true GLOBAL ionized
fraction -- that was never the target of this method. Look at the FIRST
band nearest the source (low LOS bin index): per-galaxy variation there
(which galaxies show a purple/dark dip vs. which stay bright) is the part
that's expected to carry real signal, since damping-wing absorption
sensitivity is dominated by gas near each galaxy's own redshift -- and it
should visually track the true map's own first-band pattern. Bands further
from the source are expected to be much less informative or flat; getting
even faint traces there is a bonus, not a baseline to fail against.

One PDF per example (not a combined figure) -- large labels, all text
black, no editorializing captions with a built-in verdict.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = "/Users/dxf836/Documents/git_code/Lyman-alpha-bubbles"
RECOVERY_PATH = "/Users/dxf836/Documents/Lya_bubbles/SBI_on_mock_lightcones/mock_recovery_demo.npz"

BLACK = "black"
SURFACE = "white"
C_ION, C_NEU = "#1baf7a", "#383835"
CMAP_SEQ = "magma"


def plot_one_example(theta_true, marginal_map, ess, pool_size, example_idx, out_path):
    n_gal, n_los = theta_true.shape
    x_ion_true = 1.0 - theta_true.mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    fig.patch.set_facecolor(SURFACE)

    cmap_bin = matplotlib.colors.ListedColormap([C_ION, C_NEU])
    ax = axes[0]
    ax.set_facecolor(SURFACE)
    ax.imshow(theta_true, aspect="auto", cmap=cmap_bin, vmin=0, vmax=1,
              extent=[0, n_los, 0, n_gal], origin="lower")
    ax.set_title(f"TRUE ionization state (x_ion = {x_ion_true:.2f})",
                color=BLACK, fontsize=16, loc="left")
    ax.set_xlabel("LOS bin (0 = source, end = z_end)", color=BLACK, fontsize=14)
    ax.set_ylabel("galaxy", color=BLACK, fontsize=14)
    ax.tick_params(colors=BLACK, labelsize=12)

    ax = axes[1]
    ax.set_facecolor(SURFACE)
    im = ax.imshow(marginal_map, aspect="auto", cmap=CMAP_SEQ, vmin=0, vmax=1,
                   extent=[0, n_los, 0, n_gal], origin="lower")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("P(neutral)", color=BLACK, fontsize=14)
    cbar.ax.tick_params(colors=BLACK, labelsize=12)
    ax.set_title("INFERRED probability of neutral gas", color=BLACK, fontsize=16, loc="left")
    ax.set_xlabel("LOS bin (0 = source, end = z_end)", color=BLACK, fontsize=14)
    ax.tick_params(colors=BLACK, labelsize=12)

    fig.suptitle(f"Toy example {example_idx}: known-truth recovery check",
                 color=BLACK, fontsize=19, fontweight="bold", x=0.02, ha="left", y=1.02)
    fig.text(0.02, -0.05,
            f"This observation was generated from the TRUE map on the left (never shown to "
            f"inference). The right panel is inferred from only the resulting noisy observations, "
            f"evaluated against a held-out pool of {pool_size - 1:,} other simulated universes "
            f"(ESS = {ess:.0f}, {100 * ess / (pool_size - 1):.1f}% of pool).",
            color=BLACK, fontsize=13, ha="left", va="top", wrap=True)

    fig.legend(handles=[Patch(facecolor=C_ION, label="ionized"), Patch(facecolor=C_NEU, label="neutral")],
               loc="upper right", ncol=1, frameon=False, fontsize=13, bbox_to_anchor=(0.99, 1.05),
               labelcolor=BLACK)

    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    print(f"[saved] {out_path}", flush=True)


def main():
    d = np.load(RECOVERY_PATH)
    n_examples = int(d["n_examples"])
    pool_size = int(d["pool_size"])
    for i in range(n_examples):
        theta_true = d[f"example_{i}__theta_true"]
        marginal_map = d[f"example_{i}__marginal_map"]
        ess = float(d[f"example_{i}__ess"])
        out_path = f"{REPO}/mock_recovery_example_{i}.pdf"
        plot_one_example(theta_true, marginal_map, ess, pool_size, i, out_path)


if __name__ == "__main__":
    main()