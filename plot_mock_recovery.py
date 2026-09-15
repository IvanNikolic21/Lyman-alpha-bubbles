"""
Plots mock_recovery_demo.py's output: for each toy example, TRUE theta
(the ionization mask that actually generated that example's x) side-by-side
with the INFERRED marginal probability map from running inference against
it -- the "does this work on mock data, where we know the answer" check.

One figure per example (not one big grid) -- meant to become individual
slides directly, each self-contained with its own title/caption baked in
as the figure's own text, per the explicit feedback that prior figures
needed more description to be legible on their own.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/Users/dxf836/Documents/git_code/Lyman-alpha-bubbles"
RECOVERY_PATH = "/Users/dxf836/Documents/Lya_bubbles/SBI_on_mock_lightcones/mock_recovery_demo.npz"

INK_PRIMARY, INK_SECOND, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID = "#fcfcfb", "#e1e0d9"
C_ION, C_NEU = "#1baf7a", "#383835"
CMAP_SEQ = "magma"


def plot_one_example(theta_true, marginal_map, ess, pool_size, example_idx, out_path):
    n_gal, n_los = theta_true.shape
    x_ion_true = 1.0 - theta_true.mean()
    x_ion_est = 1.0 - marginal_map.mean()
    mae = np.abs(marginal_map - theta_true).mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.patch.set_facecolor(SURFACE)

    cmap_bin = matplotlib.colors.ListedColormap([C_ION, C_NEU])
    ax = axes[0]
    ax.set_facecolor(SURFACE)
    ax.imshow(theta_true, aspect="auto", cmap=cmap_bin, vmin=0, vmax=1,
              extent=[0, n_los, 0, n_gal], origin="lower")
    ax.set_title(f"TRUE ionization state\n(x_ion = {x_ion_true:.2f})",
                color=INK_PRIMARY, fontsize=12, loc="left")
    ax.set_xlabel("LOS bin (0=source, end=z_end)", color=INK_SECOND, fontsize=9.5)
    ax.set_ylabel("galaxy", color=INK_SECOND, fontsize=9.5)
    ax.tick_params(colors=INK_MUTED, labelsize=8)

    ax = axes[1]
    ax.set_facecolor(SURFACE)
    im = ax.imshow(marginal_map, aspect="auto", cmap=CMAP_SEQ, vmin=0, vmax=1,
                   extent=[0, n_los, 0, n_gal], origin="lower")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(neutral)")
    ax.set_title(f"INFERRED probability of neutral gas\n(implied x_ion = {x_ion_est:.2f})",
                color=INK_PRIMARY, fontsize=12, loc="left")
    ax.set_xlabel("LOS bin (0=source, end=z_end)", color=INK_SECOND, fontsize=9.5)
    ax.tick_params(colors=INK_MUTED, labelsize=8)

    fig.suptitle(f"Toy example {example_idx}: known-truth recovery check",
                 color=INK_PRIMARY, fontsize=15, fontweight="bold", x=0.02, ha="left", y=1.03)
    fig.text(0.02, -0.04,
            f"Method: this galaxy/observation set was generated from the TRUE map on the left "
            f"(never shown to the inference). The right panel is what the trained network infers "
            f"from ONLY the resulting noisy observations, evaluated against a held-out pool of "
            f"{pool_size:,} other simulated universes it was never trained on directly.\n"
            f"Recovery quality: effective sample size {ess:.0f}/{pool_size-1:,} "
            f"({100*ess/(pool_size-1):.1f}%) -- mean |inferred - true| per pixel = {mae:.3f}.",
            color=INK_SECOND, fontsize=9.5, ha="left", va="top", wrap=True)

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=C_ION, label="ionized"), Patch(facecolor=C_NEU, label="neutral")],
               loc="upper right", ncol=1, frameon=False, fontsize=9, bbox_to_anchor=(0.99, 1.06),
               labelcolor=INK_SECOND)

    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"[saved] {out_path}", flush=True)


def main():
    d = np.load(RECOVERY_PATH)
    n_examples = int(d["n_examples"])
    pool_size = int(d["pool_size"])
    for i in range(n_examples):
        theta_true = d[f"example_{i}__theta_true"]
        marginal_map = d[f"example_{i}__marginal_map"]
        ess = float(d[f"example_{i}__ess"])
        out_path = f"{REPO}/mock_recovery_example_{i}.png"
        plot_one_example(theta_true, marginal_map, ess, pool_size, i, out_path)


if __name__ == "__main__":
    main()