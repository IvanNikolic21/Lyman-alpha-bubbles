"""
Plotting for sbi_pixel_field.py's `infer` output (pixel_infer.npz) --
one function per panel, each self-contained (takes an `ax` + plain arrays,
draws into it), so any single panel can be restyled/rearranged without
touching the others. `main()` just assembles them into one figure via
gridspec; swap that layout freely.

No computation happens here beyond what's needed to draw (the actual
inference/statistics were already done by `sbi_pixel_field.py infer` and
saved to pixel_infer.npz) -- this mirrors the compute/plot split already
used for compute_bsd_survey_area.py/plot_bsd_survey_area.py.

Data sources:
  - INFER_PATH: pixel_infer.npz (marginal_map, joint_samples, weights, ess,
    pool_size, n_gal, n_los) -- synced from the cluster.
  - VAL_DIR: the SAME val-split mock batches used as the inference pool
    (bubble_mocks_run1_x's val_batch_*.npz, synced earlier), reloaded here
    to get the prior's own realized (unweighted) mean for comparison --
    confirmed elsewhere to be the exact same pool inference used.
  - Galaxy redshifts (for sort order only, not physics): loaded locally via
    the same importlib bypass used throughout this project to dodge
    lyabubbles/__init__.py's eager py21cmfast import.

Convention: theta/map values are P(neutral) -- 1=neutral, index 0 = nearest
each galaxy's own redshift (the source), index n_los-1 = nearest z_end.
"""
import glob
import importlib.util
import sys
import types

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = "/Users/dxf836/Documents/git_code/Lyman-alpha-bubbles"
INFER_PATH = "/Users/dxf836/Documents/Lya_bubbles/SBI_on_mock_lightcones/pixel_infer.npz"
VAL_DIR = "/Users/dxf836/Documents/Lya_bubblesx"
OUT_PATH = f"{REPO}/pixel_infer_diagnostics.png"

# ── Style (edit freely -- every panel function below reads only these) ─────
INK_PRIMARY, INK_SECOND, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID = "#fcfcfb", "#e1e0d9"
C_ION, C_NEU = "#1baf7a", "#383835"
C_WEIGHT = "#2a78d6"
CMAP_SEQ = "magma"      # prior/posterior maps
CMAP_DIV = "RdBu_r"     # posterior-prior difference


# ── Data loading (no plotting) ──────────────────────────────────────────────

def load_infer_results(infer_path=INFER_PATH):
    d = np.load(infer_path)
    return dict(marginal_map=d["marginal_map"], joint_samples=d["joint_samples"],
               weights=d["weights"], ess=float(d["ess"]), pool_size=int(d["pool_size"]),
               n_gal=int(d["n_gal"]), n_los=int(d["n_los"]))


def load_prior_pool_mean(n_gal, n_los, val_dir=VAL_DIR):
    """Unweighted mean of the SAME val pool used for inference -- the
    prior's own realized mean (not just its analytic expectation)."""
    paths = sorted(glob.glob(f"{val_dir}/val_batch_*.npz"))
    theta_pool = np.concatenate([np.load(p)["theta"] for p in paths], axis=0)
    return theta_pool.reshape(len(theta_pool), n_gal, n_los).mean(axis=0)


def load_galaxy_sort_order(n_gal, repo=REPO):
    """Redshift-sorted galaxy order for display only -- falls back to raw
    order if the catalog can't be loaded locally (e.g. wrong machine)."""
    try:
        pkg_stub = types.ModuleType("lyabubbles")
        pkg_stub.__path__ = [f"{repo}/lyabubbles"]
        sys.modules["lyabubbles"] = pkg_stub
        spec = importlib.util.spec_from_file_location("real_data_standalone", f"{repo}/lyabubbles/real_data.py")
        rdmod = importlib.util.module_from_spec(spec)
        sys.modules["real_data_standalone"] = rdmod
        spec.loader.exec_module(rdmod)
        cat = rdmod.load_catalog_v2(f"{repo}/tb_lya.txt", f"{repo}/sample_nirspec_properties.txt",
                                    z_min=5.0, muv_max=-18.0, prefer="grating")
        redshifts = cat.redshift[cat.redshift <= 7.3]
        assert len(redshifts) == n_gal
        return np.argsort(redshifts)
    except Exception as e:
        print(f"[catalog] could not load locally ({type(e).__name__}: {e}) -- using raw order",
              flush=True)
        return np.arange(n_gal)


# ── Panels -- each one self-contained: ax in, drawing out ──────────────────

def plot_map_panel(ax, matrix, title, order, n_los, n_gal, vmin=0.0, vmax=None,
                   cmap=CMAP_SEQ, xlabel="LOS bin (0=source, end=z_end)",
                   ylabel=None, colorbar=True):
    """One (n_gal, n_los) map -- reused for both the prior and posterior
    panels (same value range, same colormap, so they're visually
    comparable side by side)."""
    ax.set_facecolor(SURFACE)
    im = ax.imshow(matrix[order], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[0, n_los, 0, n_gal], origin="lower")
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, color=INK_PRIMARY, fontsize=10.5, loc="left")
    ax.set_xlabel(xlabel, color=INK_SECOND, fontsize=8.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECOND, fontsize=9)
    ax.tick_params(colors=INK_MUTED, labelsize=7.5)
    return im


def plot_diff_panel(ax, prior_map, posterior_map, order, n_los, n_gal,
                    xlabel="LOS bin (0=source, end=z_end)", title="Posterior - Prior"):
    """Posterior-minus-prior, diverging colormap centered on the actual
    max |difference| (not an arbitrary fixed scale)."""
    diff = posterior_map - prior_map
    lim = np.abs(diff).max()
    ax.set_facecolor(SURFACE)
    im = ax.imshow(diff[order], aspect="auto", cmap=CMAP_DIV, vmin=-lim, vmax=lim,
                   extent=[0, n_los, 0, n_gal], origin="lower")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, color=INK_PRIMARY, fontsize=10.5, loc="left")
    ax.set_xlabel(xlabel, color=INK_SECOND, fontsize=8.5)
    ax.tick_params(colors=INK_MUTED, labelsize=7.5)
    return im


def plot_weight_panel(ax, weights, ess, pool_size):
    """Importance-weight histogram (log-count) -- healthy shape is a peak
    near the uniform line with a smoothly decaying tail; a single spike
    near weight=1 would mean the pool is effectively one sample."""
    ax.set_facecolor(SURFACE)
    ax.hist(weights, bins=80, color=C_WEIGHT, alpha=0.85)
    ax.axvline(1 / pool_size, color=INK_MUTED, lw=1, ls="--", label="uniform (1/pool_size)")
    ax.set_yscale("log")
    ax.set_xlabel("importance weight", color=INK_SECOND, fontsize=9.5)
    ax.set_ylabel("count (log)", color=INK_SECOND, fontsize=9.5)
    ax.set_title(f"Importance weight distribution -- ESS={ess:.0f}/{pool_size} "
                f"({100*ess/pool_size:.1f}%)", color=INK_PRIMARY, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECOND)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


def plot_joint_sample_panel(ax, sample, order, n_los, n_gal, title=None):
    """One binary (n_gal, n_los) joint posterior sample -- theta=1 means
    NEUTRAL, so the reported x_ion is 1-mean(sample)."""
    cmap_bin = matplotlib.colors.ListedColormap([C_ION, C_NEU])
    ax.set_facecolor(SURFACE)
    ax.imshow(sample[order], aspect="auto", cmap=cmap_bin, vmin=0, vmax=1,
              extent=[0, n_los, 0, n_gal], origin="lower")
    if title is None:
        title = f"x_ion={1 - sample.mean():.2f}"
    ax.set_title(title, color=INK_PRIMARY, fontsize=9.5, loc="left")
    ax.set_xlabel("LOS bin", color=INK_SECOND, fontsize=8)
    ax.tick_params(colors=INK_MUTED, labelsize=7)


def plot_ionized_neutral_legend(fig, loc="lower center", bbox_to_anchor=(0.5, -0.01)):
    fig.legend(handles=[Patch(facecolor=C_ION, label="ionized"), Patch(facecolor=C_NEU, label="neutral")],
               loc=loc, ncol=2, frameon=False, fontsize=9, bbox_to_anchor=bbox_to_anchor,
               labelcolor=INK_SECOND)


# ── Assembly -- rearrange/resize freely, panels above don't care ───────────

def main():
    r = load_infer_results()
    prior_map = load_prior_pool_mean(r["n_gal"], r["n_los"])
    order = load_galaxy_sort_order(r["n_gal"])
    n_gal, n_los = r["n_gal"], r["n_los"]

    print(f"[compare] overall mean neutral-prob: prior={prior_map.mean():.4f}, "
          f"posterior={r['marginal_map'].mean():.4f}", flush=True)

    fig = plt.figure(figsize=(15, 13))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 0.55, 0.85], hspace=0.45, wspace=0.35)

    vmax = max(prior_map.max(), r["marginal_map"].max())
    plot_map_panel(fig.add_subplot(gs[0, 0]), prior_map, "Prior (pool mean)",
                  order, n_los, n_gal, vmax=vmax, ylabel="galaxy (sorted by redshift)")
    plot_map_panel(fig.add_subplot(gs[0, 1]), r["marginal_map"], "Posterior (weighted)",
                  order, n_los, n_gal, vmax=vmax)
    plot_diff_panel(fig.add_subplot(gs[0, 2]), prior_map, r["marginal_map"], order, n_los, n_gal)

    plot_weight_panel(fig.add_subplot(gs[1, :]), r["weights"], r["ess"], r["pool_size"])

    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(r["joint_samples"]), size=3, replace=False)
    for k, idx in enumerate(sample_idx):
        plot_joint_sample_panel(fig.add_subplot(gs[2, k]), r["joint_samples"][idx], order,
                                n_los, n_gal, title=f"Joint sample #{idx} "
                                f"(x_ion={1 - r['joint_samples'][idx].mean():.2f})")

    plot_ionized_neutral_legend(fig)
    fig.suptitle("Pixel-field NRE inference diagnostic (real x_obs, mock-trained ratio estimator)",
                 color=INK_PRIMARY, fontsize=15, fontweight="bold", x=0.02, ha="left", y=0.99)
    fig.savefig(OUT_PATH, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"[saved] {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()