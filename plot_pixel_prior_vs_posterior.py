"""
Prior-vs-posterior comparison for the pixel-field SBI mode: puts the
training-set (prior-predictive) mean neutral-fraction map side by side with
`infer`'s posterior marginal map, plus their difference and a few coherent
joint-sample draws, in one composite figure.

The whole point of this script is the diff panel. The marginal map alone
mixes two very different things: (1) a step around the box-length crossing
distance that is baked into the SIMULATOR/PRIOR itself (present even with
zero conditioning on real data -- see the box-walk geometry note in
lyabubbles/lightcone_field.py), and (2) genuine signal from conditioning on
the real catalog's x_obs. Diffing against the prior-predictive mean is what
separates them -- see pixel-field-sbi project memory, 2026-08-14 entry.

Convention (lyabubbles/lightcone_field.py's discretize_to_fixed_bins):
theta=1 means NEUTRAL, theta=0 means IONIZED.

Usage
-----
python plot_pixel_prior_vs_posterior.py \\
    --infer_npz sbi_runs/pixel_v2/pixel_infer.npz \\
    --prior_npz sbi_runs/pixel_v2/train_mean_pixel_map_data.npz \\
    --output_dir sbi_runs/pixel_v2/plots

`--prior_npz` is produced by a one-off aggregation of the training sims,
e.g. (on the cluster, where the sim batches actually live):
    import numpy as np, plot_pixel_sims as pps
    theta, n_gal, n_los = pps.load_theta('sbi_runs/pixel_v2/sim', 'train')
    np.savez('train_mean_pixel_map_data.npz', mean_theta=theta.mean(axis=0))
"""
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

# ── Palette (dataviz skill's validated default, light mode) ────────────────
# Sequential blue ramp (100->700, light->dark) -- reused from the skill's
# reference palette rather than a fresh cividis call, so panels A/B/D read as
# one deliberate "P(neutral)" scale throughout the figure.
_SEQ_BLUE_STEPS = [
    '#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
    '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b',
]
SEQ_BLUE = LinearSegmentedColormap.from_list('seq_blue', _SEQ_BLUE_STEPS)

# Diverging blue<->red pair with a neutral gray midpoint, for the diff panel
# (polarity: more-neutral-than-prior vs. more-ionized-than-prior) -- never a
# sequential map for a signed quantity, per the dataviz skill's core rule.
_DIV_STEPS = ['#0d366b', '#2a78d6', '#86b6ef', '#f0efec', '#f3a9a8', '#e34948', '#8a1f1f']
DIV_BLUE_RED = LinearSegmentedColormap.from_list('div_blue_red', _DIV_STEPS)

INK_PRIMARY   = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED     = '#898781'
GRID_HAIRLINE = '#e1e0d9'
SURFACE       = '#fcfcfb'
# Two-color binary rendering for the joint-sample panels (status-style: a
# cell IS neutral or IS ionized, not a magnitude) -- surface for ionized,
# mid-ramp blue for neutral, rather than a continuous colormap on strictly
# binary data.
BINARY_IONIZED = SURFACE
BINARY_NEUTRAL = '#256abf'


def _style_axes(ax):
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_HAIRLINE)
    ax.set_facecolor(SURFACE)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--infer_npz', type=str, required=True, help='pixel_infer.npz from `infer`.')
    ap.add_argument('--prior_npz', type=str, required=True,
                    help='.npz with a `mean_theta` (n_gal, n_los) array from the training sims.')
    ap.add_argument('--n_joint_examples', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output_dir', type=str, required=True)
    args = ap.parse_args()

    d_post = np.load(args.infer_npz)
    marginal_map = d_post['marginal_map']           # (n_gal, n_los), P(neutral)
    joint_samples = d_post['joint_samples']          # (n_samples, n_gal, n_los)
    ess = float(d_post['ess'])
    pool_size = int(d_post['pool_size'])

    prior_map = np.load(args.prior_npz)['mean_theta']  # (n_gal, n_los), P(neutral)
    if prior_map.shape != marginal_map.shape:
        raise ValueError(f"prior map shape {prior_map.shape} != posterior map shape "
                         f"{marginal_map.shape} -- mismatched n_gal/n_los?")
    n_gal, n_los = marginal_map.shape
    diff_map = marginal_map - prior_map

    os.makedirs(args.output_dir, exist_ok=True)

    # Where the box-length crossing sits, for the ~median galaxy path length
    # implied by the prior's own step location -- purely a visual annotation,
    # derived from the data itself (first bin where the population-mean prior
    # drops by more than half its total step), not a hardcoded guess.
    prior_bin_mean = prior_map.mean(axis=0)
    step_drop = prior_bin_mean[0] - prior_bin_mean[-1]
    if step_drop > 1e-3:
        below = np.where(prior_bin_mean < prior_bin_mean[0] - 0.5 * step_drop)[0]
        step_bin = int(below[0]) if len(below) else None
    else:
        step_bin = None

    fig = plt.figure(figsize=(13, 8.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.85], hspace=0.55, wspace=0.35)

    # A/B share the same P(neutral) *quantity* but not the same range -- the
    # posterior sits entirely in the lower part of [0,1]. Pinning both to a
    # fixed [0,1] scale crushes the real structure (the near-source hump in
    # B especially) into a handful of visually indistinguishable shades.
    # Auto-scale each to its own data range instead; panel C (the diff) is
    # what carries the properly-normalized absolute comparison.
    pad_a = 0.03 * (prior_map.max() - prior_map.min())
    pad_b = 0.03 * (marginal_map.max() - marginal_map.min())
    panels = [
        (prior_map,    'A · Prior — mean over training draws',
         SEQ_BLUE, prior_map.min() - pad_a, prior_map.max() + pad_a, 'P(neutral)'),
        (marginal_map, f'B · Posterior — infer at real x_obs (ESS {100*ess/pool_size:.1f}%)',
         SEQ_BLUE, marginal_map.min() - pad_b, marginal_map.max() + pad_b, 'P(neutral)'),
        (diff_map,     'C · Posterior − Prior (genuine signal)',
         DIV_BLUE_RED, -0.4, 0.4, 'Δ P(neutral)'),
    ]
    for col, (arr, title, cmap, vmin, vmax, cbar_label) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(arr, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        if step_bin is not None:
            ax.axvline(step_bin, color=INK_SECONDARY, lw=1, ls=(0, (3, 2)))
        ax.set_title(title, fontsize=10, color=INK_PRIMARY, loc='left')
        ax.set_xlabel('LOS bin (0=source, N_LOS−1=z_end)', fontsize=8, color=INK_SECONDARY)
        if col == 0:
            ax.set_ylabel('galaxy index', fontsize=8, color=INK_SECONDARY)
        _style_axes(ax)
        cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label(cbar_label, fontsize=7.5, color=INK_SECONDARY)
        cbar.ax.tick_params(labelsize=7, colors=INK_MUTED)
        cbar.outline.set_edgecolor(GRID_HAIRLINE)

    # ── Bottom row: a few coherent joint-sample draws (binary, two-color) ──
    rng = np.random.default_rng(args.seed)
    n_examples = min(args.n_joint_examples, len(joint_samples))
    idx = rng.choice(len(joint_samples), size=n_examples, replace=False)
    binary_cmap = LinearSegmentedColormap.from_list('binary_status', [BINARY_IONIZED, BINARY_NEUTRAL])
    for col, i in enumerate(idx):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(joint_samples[i], aspect='auto', cmap=binary_cmap, vmin=0, vmax=1,
                 interpolation='nearest')
        if step_bin is not None:
            ax.axvline(step_bin, color=INK_SECONDARY, lw=0.75, ls=(0, (3, 2)), alpha=0.6)
        ax.set_title(f'D · joint sample {i}  (neutral frac={joint_samples[i].mean():.2f})',
                    fontsize=8.5, color=INK_PRIMARY, loc='left')
        ax.set_xlabel('LOS bin', fontsize=7.5, color=INK_SECONDARY)
        if col == 0:
            ax.set_ylabel('galaxy index', fontsize=7.5, color=INK_SECONDARY)
        _style_axes(ax)

    legend_handles = [
        Patch(facecolor=BINARY_NEUTRAL, edgecolor=GRID_HAIRLINE, label='neutral'),
        Patch(facecolor=BINARY_IONIZED, edgecolor=GRID_HAIRLINE, label='ionized'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=2, frameon=False,
              fontsize=8, bbox_to_anchor=(0.5, -0.02), labelcolor=INK_SECONDARY)

    fig.suptitle('Pixel-field SBI: prior vs. posterior neutral-fraction structure',
                fontsize=13, color=INK_PRIMARY, x=0.02, ha='left', y=1.01)
    fig.text(0.02, 0.965,
            'dashed line: box-length crossing distance -- a prior/geometry artifact common to A, B, D, not inferred structure. '
            'Panels A/B are each scaled to their own data range; C carries the properly-normalized absolute comparison.',
            fontsize=8, color=INK_SECONDARY, ha='left')

    out_path = os.path.join(args.output_dir, 'prior_vs_posterior.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out_path}")
    if step_bin is not None:
        print(f"Detected prior step at bin {step_bin} (annotated with dashed line).")
    print(f"Overall: prior mean={prior_map.mean():.3f}, posterior mean={marginal_map.mean():.3f}, "
          f"diff={marginal_map.mean() - prior_map.mean():+.3f}")


if __name__ == '__main__':
    main()
