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
CMAP_UNC = "viridis"    # uncertainty maps (Bernoulli std / bootstrap SEM) -- deliberately
                        # distinct from CMAP_SEQ so a probability map and an uncertainty
                        # map are never visually confusable at a glance


# ── Data loading (no plotting) ──────────────────────────────────────────────

def load_infer_results(infer_path=INFER_PATH):
    d = np.load(infer_path)
    return dict(marginal_map=d["marginal_map"], joint_samples=d["joint_samples"],
               weights=d["weights"], ess=float(d["ess"]), pool_size=int(d["pool_size"]),
               n_gal=int(d["n_gal"]), n_los=int(d["n_los"]))


def load_val_pool_grid(n_gal, n_los, val_dir=VAL_DIR):
    """(pool_size, n_gal, n_los) theta from the SAME val pool used for
    inference -- the raw material both the prior mean and the bootstrap
    uncertainty below are computed from."""
    paths = sorted(glob.glob(f"{val_dir}/val_batch_*.npz"))
    theta_pool = np.concatenate([np.load(p)["theta"] for p in paths], axis=0)
    return theta_pool.reshape(len(theta_pool), n_gal, n_los)


def load_prior_pool_mean(n_gal, n_los, val_dir=VAL_DIR):
    """Unweighted mean of the SAME val pool used for inference -- the
    prior's own realized mean (not just its analytic expectation)."""
    return load_val_pool_grid(n_gal, n_los, val_dir).mean(axis=0)


# ── Uncertainty on the marginal map -- two DIFFERENT things, both called
#    "confidence"/"width" colloquially but not interchangeable ────────────

def compute_bernoulli_std(marginal_map):
    """Intrinsic per-pixel spread of the posterior itself: each pixel's
    theta is strictly binary, so its posterior IS a weighted coin flip
    with P(neutral)=p -- sqrt(p*(1-p)) is that coin's own std, maximal at
    p=0.5 (maximally uncertain which state it's in) and zero at p=0 or 1
    (certain). This says nothing about how RELIABLE the estimate of p
    itself is -- see compute_marginal_bootstrap_std for that."""
    p = marginal_map
    return np.sqrt(p * (1 - p))


def compute_marginal_bootstrap_std(theta_pool_grid, weights, n_boot=200, seed=0):
    """Weighted-bootstrap standard error of the marginal_map ESTIMATE
    itself -- resample the pool many times (multinomial counts per
    replicate, matching how self-normalized importance sampling would
    behave if rerun) and see how much the resulting mean map varies. This
    is the ESS-driven "how much do I trust this specific number" question,
    NOT the intrinsic Bernoulli spread above -- a pixel can have LOW
    Bernoulli std (p near 0 or 1, seemingly confident) while still having
    HIGH bootstrap std if that near-0/1 estimate itself rests on very few
    effective samples.

    Implemented as one matrix multiply (counts @ theta_flat) rather than
    n_boot separate fancy-index copies -- ~200x fewer large-array
    allocations, matmul is BLAS-backed."""
    rng = np.random.default_rng(seed)
    n_pool, n_gal, n_los = theta_pool_grid.shape
    theta_flat = theta_pool_grid.reshape(n_pool, n_gal * n_los).astype(np.float32)
    counts = rng.multinomial(n_pool, weights, size=n_boot).astype(np.float32)
    boot_means_flat = (counts @ theta_flat) / n_pool
    return boot_means_flat.reshape(n_boot, n_gal, n_los).std(axis=0)


def _load_local_module(name, path):
    """importlib-load one file as `name`, bypassing package __init__.py
    (needed for lyabubbles.* files, whose real __init__.py eagerly imports
    py21cmfast -- not installed/needed in this local plotting env)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _stub_lyabubbles_package(repo=REPO):
    pkg_stub = types.ModuleType("lyabubbles")
    pkg_stub.__path__ = [f"{repo}/lyabubbles"]
    sys.modules["lyabubbles"] = pkg_stub


def load_galaxy_sort_order(n_gal, repo=REPO):
    """Redshift-sorted galaxy order for display only -- falls back to raw
    order if the catalog can't be loaded locally (e.g. wrong machine)."""
    try:
        _stub_lyabubbles_package(repo)
        rdmod = _load_local_module("real_data_standalone", f"{repo}/lyabubbles/real_data.py")
        cat = rdmod.load_catalog_v2(f"{repo}/tb_lya.txt", f"{repo}/sample_nirspec_properties.txt",
                                    z_min=5.0, muv_max=-18.0, prefer="grating")
        redshifts = cat.redshift[cat.redshift <= 7.3]
        assert len(redshifts) == n_gal
        return np.argsort(redshifts)
    except Exception as e:
        print(f"[catalog] could not load locally ({type(e).__name__}: {e}) -- using raw order",
              flush=True)
        return np.arange(n_gal)


def load_redshift_bin_edges(n_gal, n_los, repo=REPO, z_hi=7.3, z_end=5.3):
    """Real redshift bin edges per galaxy, shape (n_gal, n_los + 1),
    decreasing along axis 1 (index 0 edge = that galaxy's own redshift,
    last edge = z_end, shared by every galaxy). Same catalog filters as
    everywhere else in this pipeline (z_hi=7.3, z_min=5.0, muv_max=-18.0,
    prefer='grating', no z_lo) so the galaxy order/count matches
    marginal_map/joint_samples exactly.

    Needed because a fixed n_los=75 bins per galaxy, spanning EACH
    galaxy's own (slightly different) path length to z_end, means bin
    INDEX k is NOT the same real redshift for every galaxy -- only
    fractional position along each galaxy's own path. Galaxies here span
    z=6.91-7.27, so the distortion is modest but real; this is what lets a
    redshift-axis panel show skewers of genuinely different length,
    correctly aligned in real redshift rather than bin-index position."""
    _stub_lyabubbles_package(repo)
    rdmod = _load_local_module("real_data_standalone", f"{repo}/lyabubbles/real_data.py")
    _load_local_module("lyabubbles.helpers", f"{repo}/lyabubbles/helpers.py")
    lf = _load_local_module("lyabubbles.lightcone_field", f"{repo}/lyabubbles/lightcone_field.py")

    from astropy.cosmology import Planck18 as Cosmo
    from astropy import units as u

    cat = rdmod.load_catalog_v2(f"{repo}/tb_lya.txt", f"{repo}/sample_nirspec_properties.txt",
                                z_min=5.0, muv_max=-18.0, prefer="grating")
    in_window = cat.redshift <= z_hi
    ra, dec, redshift = cat.ra[in_window], cat.dec[in_window], cat.redshift[in_window]
    x_gal, y_gal, z_gal, ra0, dec0, z0, *_ = rdmod.radec_to_comoving(ra, dec, redshift)
    assert len(z_gal) == n_gal, f"{len(z_gal)} != {n_gal}"
    d_c0 = Cosmo.comoving_distance(z0).to(u.Mpc).value

    z_edges = np.empty((n_gal, n_los + 1))
    for g in range(n_gal):
        z_b, z_e = lf.bin_z_edges(n_los, z_gal[g], d_c0, z_end=z_end)
        z_edges[g] = np.concatenate([z_b, z_e[-1:]])
    return z_edges


# ── Panels -- each one self-contained: ax in, drawing out ──────────────────

def _pcolormesh_by_redshift(ax, matrix, z_edges, order, n_gal, n_los, cmap, vmin, vmax):
    """Real-redshift x-axis version of the map panels -- each galaxy row
    gets its OWN bin edges (z_edges[g], shape (n_los+1,)) rather than a
    shared bin-index axis, so galaxies with different z_gal (hence
    different true path length to z_end) render as rows of genuinely
    different physical width, aligned in real redshift rather than
    fractional position. A quadrilateral mesh (X, Y both (n_gal+1,
    n_los+1)) is needed rather than plain imshow/pcolormesh-with-1D-axes
    because each row's edges differ -- no single shared x-axis grid
    describes every row.

    z_end (the right-hand edge, per bin_z_edges/module docstring) is the
    SAME for every galaxy, so that side of the plot is a clean vertical
    line; each galaxy's OWN z_gal (the left-hand edge) varies, so that
    side is jagged -- literally "some skewers span a larger distance."
    invert_xaxis() keeps high-z/source on the left, matching the
    bin-index panels' left-to-right convention."""
    X = z_edges[order]                                    # (n_gal, n_los+1)
    X = np.vstack([X, X[-1:]])                             # (n_gal+1, n_los+1)
    Y = np.arange(n_gal + 1)[:, None] * np.ones(n_los + 1)  # (n_gal+1, n_los+1)
    C = matrix[order]                                      # (n_gal, n_los)
    im = ax.pcolormesh(X, Y, C, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
    ax.invert_xaxis()
    return im


def plot_map_panel(ax, matrix, title, order, n_los, n_gal, vmin=0.0, vmax=None,
                   cmap=CMAP_SEQ, xlabel=None, ylabel=None, colorbar=True, z_edges=None):
    """One (n_gal, n_los) map -- reused for both the prior and posterior
    panels (same value range, same colormap, so they're visually
    comparable side by side). Pass `z_edges` (from
    `load_redshift_bin_edges`) to render on a real-redshift x-axis instead
    of bin index -- see `_pcolormesh_by_redshift`."""
    ax.set_facecolor(SURFACE)
    if z_edges is None:
        im = ax.imshow(matrix[order], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=[0, n_los, 0, n_gal], origin="lower")
        xlabel = xlabel or "LOS bin (0=source, end=z_end)"
    else:
        im = _pcolormesh_by_redshift(ax, matrix, z_edges, order, n_gal, n_los, cmap, vmin, vmax)
        xlabel = xlabel or "redshift"
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, color=INK_PRIMARY, fontsize=10.5, loc="left")
    ax.set_xlabel(xlabel, color=INK_SECOND, fontsize=8.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECOND, fontsize=9)
    ax.tick_params(colors=INK_MUTED, labelsize=7.5)
    return im


def plot_diff_panel(ax, prior_map, posterior_map, order, n_los, n_gal,
                    xlabel=None, title="Posterior - Prior", z_edges=None):
    """Posterior-minus-prior, diverging colormap centered on the actual
    max |difference| (not an arbitrary fixed scale). Pass `z_edges` for
    the real-redshift x-axis version, same as plot_map_panel."""
    diff = posterior_map - prior_map
    lim = np.abs(diff).max()
    ax.set_facecolor(SURFACE)
    if z_edges is None:
        im = ax.imshow(diff[order], aspect="auto", cmap=CMAP_DIV, vmin=-lim, vmax=lim,
                       extent=[0, n_los, 0, n_gal], origin="lower")
        xlabel = xlabel or "LOS bin (0=source, end=z_end)"
    else:
        im = _pcolormesh_by_redshift(ax, diff, z_edges, order, n_gal, n_los, CMAP_DIV, -lim, lim)
        xlabel = xlabel or "redshift"
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


def plot_joint_sample_panel(ax, sample, order, n_los, n_gal, title=None, z_edges=None,
                            xlabel=None):
    """One binary (n_gal, n_los) joint posterior sample -- theta=1 means
    NEUTRAL, so the reported x_ion is 1-mean(sample). Pass `z_edges` for
    the real-redshift x-axis version, same convention as plot_map_panel/
    plot_diff_panel."""
    cmap_bin = matplotlib.colors.ListedColormap([C_ION, C_NEU])
    ax.set_facecolor(SURFACE)
    if z_edges is None:
        ax.imshow(sample[order], aspect="auto", cmap=cmap_bin, vmin=0, vmax=1,
                  extent=[0, n_los, 0, n_gal], origin="lower")
        xlabel = xlabel or "LOS bin"
    else:
        _pcolormesh_by_redshift(ax, sample, z_edges, order, n_gal, n_los, cmap_bin, 0, 1)
        xlabel = xlabel or "redshift"
    if title is None:
        title = f"x_ion={1 - sample.mean():.2f}"
    ax.set_title(title, color=INK_PRIMARY, fontsize=9.5, loc="left")
    ax.set_xlabel(xlabel, color=INK_SECOND, fontsize=8)
    ax.tick_params(colors=INK_MUTED, labelsize=7)


def plot_ionized_neutral_legend(fig, loc="lower center", bbox_to_anchor=(0.5, -0.01)):
    fig.legend(handles=[Patch(facecolor=C_ION, label="ionized"), Patch(facecolor=C_NEU, label="neutral")],
               loc=loc, ncol=2, frameon=False, fontsize=9, bbox_to_anchor=bbox_to_anchor,
               labelcolor=INK_SECOND)


# ── Assembly -- rearrange/resize freely, panels above don't care ───────────

def main():
    r = load_infer_results()
    n_gal, n_los = r["n_gal"], r["n_los"]
    theta_pool_grid = load_val_pool_grid(n_gal, n_los)
    prior_map = theta_pool_grid.mean(axis=0)
    order = load_galaxy_sort_order(n_gal)

    print(f"[compare] overall mean neutral-prob: prior={prior_map.mean():.4f}, "
          f"posterior={r['marginal_map'].mean():.4f}", flush=True)

    z_edges = load_redshift_bin_edges(n_gal, n_los)

    bernoulli_std = compute_bernoulli_std(r["marginal_map"])
    bootstrap_std = compute_marginal_bootstrap_std(theta_pool_grid, r["weights"])
    print(f"[uncertainty] Bernoulli std range [{bernoulli_std.min():.3f}, {bernoulli_std.max():.3f}], "
          f"bootstrap SEM range [{bootstrap_std.min():.4f}, {bootstrap_std.max():.4f}]", flush=True)

    fig = plt.figure(figsize=(15, 24))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(6, 3, height_ratios=[1, 1, 1, 0.55, 0.85, 0.85], hspace=0.55, wspace=0.35)

    vmax = max(prior_map.max(), r["marginal_map"].max())
    plot_map_panel(fig.add_subplot(gs[0, 0]), prior_map, "Prior (pool mean) -- bin index",
                  order, n_los, n_gal, vmax=vmax, ylabel="galaxy (sorted by redshift)")
    plot_map_panel(fig.add_subplot(gs[0, 1]), r["marginal_map"], "Posterior (weighted) -- bin index",
                  order, n_los, n_gal, vmax=vmax)
    plot_diff_panel(fig.add_subplot(gs[0, 2]), prior_map, r["marginal_map"], order, n_los, n_gal)

    # Same three panels, real-redshift x-axis -- each galaxy's row now
    # spans its own actual redshift range (z_gal down to the shared
    # z_end=5.3) instead of a fixed 75 bins regardless of true path
    # length, so shared ionized/neutral structure across galaxies aligns
    # in real distance rather than fractional LOS position.
    plot_map_panel(fig.add_subplot(gs[1, 0]), prior_map, "Prior (pool mean) -- redshift",
                  order, n_los, n_gal, vmax=vmax, ylabel="galaxy (sorted by redshift)",
                  z_edges=z_edges)
    plot_map_panel(fig.add_subplot(gs[1, 1]), r["marginal_map"], "Posterior (weighted) -- redshift",
                  order, n_los, n_gal, vmax=vmax, z_edges=z_edges)
    plot_diff_panel(fig.add_subplot(gs[1, 2]), prior_map, r["marginal_map"], order, n_los, n_gal,
                    z_edges=z_edges)

    # Two DIFFERENT notions of "how confident" -- see compute_bernoulli_std/
    # compute_marginal_bootstrap_std's docstrings for why they're not the
    # same thing. Both plotted with plot_map_panel directly -- an
    # uncertainty map is still just an (n_gal, n_los) map, no new panel
    # function needed, just a different matrix/cmap/vmax.
    plot_map_panel(fig.add_subplot(gs[2, 0]), bernoulli_std,
                  "Intrinsic spread: sqrt(p(1-p))\n(coin-flip uncertainty at the posterior's own p)",
                  order, n_los, n_gal, vmax=0.5, cmap=CMAP_UNC,
                  ylabel="galaxy (sorted by redshift)")
    plot_map_panel(fig.add_subplot(gs[2, 1]), bootstrap_std,
                  "Estimation SEM (weighted bootstrap, 200 resamples)\n(how much p_hat itself would wobble on a rerun)",
                  order, n_los, n_gal, cmap=CMAP_UNC)
    ax_ratio = fig.add_subplot(gs[2, 2])
    plot_map_panel(ax_ratio, bootstrap_std / np.clip(bernoulli_std, 1e-6, None),
                  "SEM / intrinsic std\n(>~0.3-0.5 = estimate itself is shaky, not just the coin flip)",
                  order, n_los, n_gal, cmap=CMAP_UNC, vmin=None)

    plot_weight_panel(fig.add_subplot(gs[3, :]), r["weights"], r["ess"], r["pool_size"])

    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(r["joint_samples"]), size=3, replace=False)
    for k, idx in enumerate(sample_idx):
        plot_joint_sample_panel(fig.add_subplot(gs[4, k]), r["joint_samples"][idx], order,
                                n_los, n_gal, title=f"Joint sample #{idx} -- bin index "
                                f"(x_ion={1 - r['joint_samples'][idx].mean():.2f})")
    # Same samples, real-redshift x-axis -- same reasoning as the map panels.
    for k, idx in enumerate(sample_idx):
        plot_joint_sample_panel(fig.add_subplot(gs[5, k]), r["joint_samples"][idx], order,
                                n_los, n_gal, title=f"Joint sample #{idx} -- redshift "
                                f"(x_ion={1 - r['joint_samples'][idx].mean():.2f})",
                                z_edges=z_edges)

    plot_ionized_neutral_legend(fig)
    fig.suptitle("Pixel-field NRE inference diagnostic (real x_obs, mock-trained ratio estimator)",
                 color=INK_PRIMARY, fontsize=15, fontweight="bold", x=0.02, ha="left", y=0.99)
    fig.savefig(OUT_PATH, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"[saved] {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()