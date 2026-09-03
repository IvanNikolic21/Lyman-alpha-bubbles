"""
JWST proposal plot, plotting step -- reads compute_bsd_survey_area.py's
saved .npz (raw MFP ray results, already sanity-checked) and renders the
figure. No physics/statistics computed here beyond simple histogramming;
all the actual measurement happened in the compute script.

Layout: ONE wide-context 2D field slice (top -- both survey footprints
overlaid as outlines, same location, for direct size comparison) + one wide
summary panel (bottom -- the actual quantitative evidence: the MFP bubble-
size distribution for all three cases, current/proposed/full_box).

Real thing discovered while building this, worth knowing before trusting the
top panel: this snapshot is 70% ionized (x_H=0.27), i.e. well past the
percolation transition -- the field is topologically an IONIZED SEA with
scattered NEUTRAL ISLANDS, not isolated round "bubbles" in a neutral
background. An earlier version of this script tried to show two separate
crops (current-area-sized and proposed-area-sized) each centered on "a
bubble" -- both came back nearly solid green with no visible boundary at
all, because at this ionization state there mostly isn't a small, clean,
isolated ionized region to crop around; large stretches of the box are all
one connected ionized network. The MFP measurement itself (bottom panel) is
unaffected by this -- mean free path from a random ionized point to the
nearest neutral cell is well-defined regardless of topology -- but the
naive "crop around a bubble" illustration doesn't visually work for THIS
snapshot. Fixed by switching to one wide context slice with both footprints
outlined at true relative scale, which shows the real (connected/percolated)
structure honestly instead of forcing a misleading "isolated bubble" crop.
If a genuinely bubble-like (pre-percolation, more neutral) snapshot is
wanted for the illustration instead, that needs a different cached redshift
-- flagged, not fetched here, since the agreed plan was to reuse the one
snapshot already extracted.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from scipy.ndimage import distance_transform_edt

REPO = "/Users/dxf836/Documents/git_code/Lyman-alpha-bubbles"

# Must match compute_bsd_survey_area.py's own Z_SNAPSHOT for the run you want
# to plot -- only picks WHICH results/.npz to load; FIELD_PATH below is then
# re-derived from that .npz's OWN recorded z_snapshot (not from this
# constant directly), so a mismatch between the two is caught loudly instead
# of silently loading the wrong field for a given result set.
Z_SNAPSHOT = 7.2436
RESULTS_PATH = f"{REPO}/bsd_survey_area_results_z{Z_SNAPSHOT:.4f}.npz"
OUT_PATH = f"{REPO}/bsd_survey_area_proposal_plot_z{Z_SNAPSHOT:.4f}.png"

INK_PRIMARY, INK_SECOND, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID = "#fcfcfb", "#e1e0d9"
C_ION, C_NEU = "#1baf7a", "#383835"
C_CURRENT, C_PROPOSED, C_TRUTH = "#eb6834", "#2a78d6", "#0b0b0b"
cmap_bin = matplotlib.colors.ListedColormap([C_NEU, C_ION])


def load_results():
    d = np.load(RESULTS_PATH, allow_pickle=True)
    cases = {}
    for label in ["full_box", "current", "proposed"]:
        cases[label] = dict(
            dist=d[f"{label}__dist"], status=d[f"{label}__status"],
            side_mpc=float(d[f"{label}__side_mpc"]), n_tiles=int(d[f"{label}__n_tiles"]),
        )
    meta = dict(cell_size=float(d["cell_size"]), n_cell=int(d["n_cell"]),
               box_len_mpc=float(d["box_len_mpc"]), threshold=float(d["threshold"]),
               z_snapshot=float(d["z_snapshot"]))
    if abs(meta["z_snapshot"] - Z_SNAPSHOT) > 1e-6:
        raise ValueError(f"{RESULTS_PATH} was computed at z={meta['z_snapshot']}, but this "
                         f"script's Z_SNAPSHOT={Z_SNAPSHOT} -- update one to match.")
    field_path = f"/Users/dxf836/Downloads/real_snapshot_z{meta['z_snapshot']:.4f}_field.npy"
    return cases, meta, field_path


def find_largest_bubble_center(ionized, margin_cells=40, safe_border_cells=0):
    """Euclidean distance transform: for every ionized cell, distance (in
    cells) to the nearest non-ionized cell. Its argmax is the center of the
    single largest inscribed sphere in the field -- a natural, non-cherry-
    picked choice of 'a big bubble' to illustrate the crop effect with.

    Real bug caught before trusting this for a proposal figure: a plain
    (non-periodic) distance_transform_edt treats the array boundary as if
    nothing exists beyond it, so an ionized region merely touching the
    box edge gets an artificially INFLATED distance there (no real neutral
    cell to bound it in that direction) -- the very first run picked
    exactly such an edge cell (j=0) as the "largest bubble", which is an
    edge artifact, not necessarily a genuine large structure. 21cmFAST
    boxes are periodic, so fixed by padding with wrap-around cells
    (`margin_cells`, comfortably larger than R_HI=60 Mpc's established
    bubble-radius prior ceiling) before running the transform, then reading
    back only the interior (un-padded) region -- this gives the same answer
    a fully periodic transform would, for any true bubble smaller than the
    margin."""
    padded = np.pad(ionized, margin_cells, mode="wrap")
    edt_padded = distance_transform_edt(padded)
    edt = edt_padded[margin_cells:-margin_cells, margin_cells:-margin_cells, margin_cells:-margin_cells]
    ci, cj, ck = np.unravel_index(np.argmax(edt), edt.shape)
    return int(ci), int(cj), int(ck), float(edt[ci, cj, ck])


def crop_centered(field2d_shape_n, center, n_cells_side):
    """Clamped start index for a `n_cells_side`-wide crop centered on
    `center`, kept fully inside [0, field2d_shape_n)."""
    lo = center - n_cells_side // 2
    lo = max(0, min(lo, field2d_shape_n - n_cells_side))
    return lo, lo + n_cells_side


def render_context_panel(ax, ionized, cell_size, n_cell, center_ijk, context_side_mpc,
                         footprints):
    """ONE wide slice centered on `center_ijk`, with each (label, side_mpc,
    color) in `footprints` drawn as a centered square outline at its true
    relative scale -- shows the real field structure once, rather than
    forcing two separately-scaled crops (see module docstring for why)."""
    ci, cj, ck = center_ijk
    n_cells_side = max(1, int(round(context_side_mpc / cell_size)))
    i_lo, i_hi = crop_centered(n_cell, ci, n_cells_side)
    j_lo, j_hi = crop_centered(n_cell, cj, n_cells_side)
    crop = ionized[i_lo:i_hi, j_lo:j_hi, ck]
    actual_side_mpc = n_cells_side * cell_size

    # Center of the crop IN PLOT COORDS (Mpc from the crop's own origin) --
    # not necessarily actual_side_mpc/2 if crop_centered had to clamp
    # against a box edge, so recovered from the actual crop indices used.
    center_x_mpc = (0.5 * (i_lo + i_hi) - i_lo) * cell_size
    center_y_mpc = (0.5 * (j_lo + j_hi) - j_lo) * cell_size

    ax.set_facecolor(SURFACE)
    ax.imshow(crop.T, origin="lower", cmap=cmap_bin, vmin=0, vmax=1,
              extent=[0, actual_side_mpc, 0, actual_side_mpc])
    for label, side_mpc, color in footprints:
        rect = Rectangle((center_x_mpc - side_mpc / 2, center_y_mpc - side_mpc / 2),
                         side_mpc, side_mpc, fill=False, edgecolor=color, linewidth=2.6,
                         label=label)
        ax.add_patch(rect)
    ax.set_xlabel("comoving Mpc", color=INK_SECOND, fontsize=9.5)
    ax.set_ylabel("comoving Mpc", color=INK_SECOND, fontsize=9.5)
    ax.set_title(f"Field slice ({actual_side_mpc:.0f} x {actual_side_mpc:.0f} Mpc context) "
                f"with both footprints overlaid at true scale",
                color=INK_PRIMARY, fontsize=11.5, loc="left")
    ax.tick_params(colors=INK_MUTED, labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color(INK_MUTED)


def bsd_curve(dist, n_bins=22):
    """d P / d(ln R): histogram in log-R, density-normalized -- the
    standard Mesinger & Furlanetto 2007 bubble-size-distribution
    convention, comparable in shape regardless of each case's ray count.

    n_bins deliberately modest: `dist` is quantized to multiples of the
    compute script's step size (0.75 Mpc, confirmed: only 150-350 distinct
    values across 200,000 rays per case), so a finer log-binning (an
    earlier 40-bin version) produces a jagged sawtooth purely from bins
    landing between quantization levels, not real distributional
    structure -- misleading in a proposal figure. 22 bins keeps bin width
    comfortably above the quantization step across nearly the whole range."""
    dist = dist[dist > 0]
    log_r = np.log(dist)
    counts, edges = np.histogram(log_r, bins=n_bins, density=True)
    centers_log = 0.5 * (edges[:-1] + edges[1:])
    return np.exp(centers_log), counts


def main():
    cases, meta, field_path = load_results()
    neutral_frac = np.load(field_path).astype(np.float32)
    ionized = neutral_frac < meta["threshold"]
    cell_size, n_cell = meta["cell_size"], meta["n_cell"]

    # Context window: comfortably bigger than both footprints AND the
    # field's own single largest bubble (~26 Mpc radius, see
    # find_largest_bubble_center), so that bubble's true edges are visible
    # too, not just the two footprint outlines.
    context_side_mpc = max(3.0 * cases["proposed"]["side_mpc"], 90.0)
    context_cells = int(round(context_side_mpc / cell_size))
    ci, cj, ck, edt_r_cells = find_largest_bubble_center(
        ionized, safe_border_cells=context_cells // 2 + 1)
    print(f"[context center] largest-bubble cell ({ci},{cj},{ck}), inscribed radius "
          f"~{edt_r_cells * cell_size:.1f} Mpc", flush=True)

    fig = plt.figure(figsize=(11.5, 12.5))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1], hspace=0.3)

    ax_a = fig.add_subplot(gs[0])
    footprints = [
        ("current (70 arcmin$^2$)", cases["current"]["side_mpc"], C_CURRENT),
        ("proposed (140 arcmin$^2$)", cases["proposed"]["side_mpc"], C_PROPOSED),
    ]
    render_context_panel(ax_a, ionized, cell_size, n_cell, (ci, cj, ck), context_side_mpc,
                        footprints)

    fig.legend(handles=[Patch(facecolor=C_ION, label="ionized"), Patch(facecolor=C_NEU, label="neutral"),
                        Patch(facecolor="none", edgecolor=C_CURRENT, linewidth=2.2, label="current footprint"),
                        Patch(facecolor="none", edgecolor=C_PROPOSED, linewidth=2.2, label="proposed footprint")],
               loc="upper center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.985),
               labelcolor=INK_SECOND)

    ax_c = fig.add_subplot(gs[1])
    ax_c.set_facecolor(SURFACE)
    style = {"full_box": (C_TRUTH, "-", 2.4, "full box (reference)"),
            "current": (C_CURRENT, "-", 2.2, "current area (70 arcmin$^2$)"),
            "proposed": (C_PROPOSED, "-", 2.2, "proposed area (140 arcmin$^2$)")}
    for label in ["full_box", "current", "proposed"]:
        r, dpdlnr = bsd_curve(cases[label]["dist"])
        color, ls, lw, leg = style[label]
        ax_c.plot(r, dpdlnr, color=color, ls=ls, lw=lw, label=leg)
        mean_r = cases[label]["dist"].mean()
        ax_c.axvline(mean_r, color=color, lw=1, ls=":", alpha=0.7)

    ax_c.set_xscale("log")
    ax_c.set_xlabel("bubble size R [comoving Mpc]", color=INK_SECOND, fontsize=10.5)
    ax_c.set_ylabel(r"$dP/d\ln R$", color=INK_SECOND, fontsize=10.5)
    ax_c.set_title("Mean-free-path bubble size distribution", color=INK_PRIMARY, fontsize=12, loc="left")
    ax_c.legend(frameon=False, fontsize=9.5, labelcolor=INK_SECOND, loc="upper right")
    ax_c.grid(True, color=GRID, linewidth=0.8)
    ax_c.tick_params(colors=INK_MUTED, labelsize=9)

    frac_capped_cur = (cases["current"]["status"] == "capped_by_area").mean()
    frac_capped_pro = (cases["proposed"]["status"] == "capped_by_area").mean()
    note = (f"{frac_capped_cur*100:.0f}% of current-area measurements never reach a true bubble "
           f"edge -- capped by the survey boundary\n"
           f"{frac_capped_pro*100:.0f}% still capped at the proposed area, vs. 0% for the "
           f"unrestricted reference")
    ax_c.text(0.02, 0.03, note, transform=ax_c.transAxes, fontsize=8.7, color=INK_SECOND,
             va="bottom", ha="left")

    fig.suptitle("Small survey areas bias the inferred bubble size distribution low",
                 color=INK_PRIMARY, fontsize=15, fontweight="bold", x=0.02, ha="left", y=0.995)
    fig.savefig(OUT_PATH, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    print(f"[saved] {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()