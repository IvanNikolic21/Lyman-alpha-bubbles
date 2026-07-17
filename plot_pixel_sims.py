"""
Visual sanity check for `sbi_pixel_field.py simulate` output: plots a handful
of individual simulated binary ionization masks (theta, shape n_gal x n_los)
as heatmaps, plus an aggregate mean-ionized-fraction-per-pixel map and a
histogram of per-draw ionized fraction across the whole batch.

Standalone (numpy/matplotlib only, no `lyabubbles`/`real_data_run` import) so
it runs anywhere the .npz files are, without needing py21cmfast installed.

Usage
-----
python plot_pixel_sims.py --sims_dir sbi_runs/pixel/sims --prefix train \\
    --n_examples 6 --output_dir sbi_runs/pixel/plots
"""
import os
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_theta(sims_dir, prefix):
    paths = sorted(glob.glob(os.path.join(sims_dir, f"{prefix}_batch_*.npz")))
    if not paths:
        raise FileNotFoundError(f"No {prefix}_batch_*.npz files found in {sims_dir}")
    thetas = []
    n_gal = n_los = None
    for p in paths:
        d = np.load(p)
        thetas.append(d['theta'])
        n_gal, n_los = int(d['n_gal']), int(d['n_los'])
    theta = np.concatenate(thetas, axis=0)
    return theta.reshape(len(theta), n_gal, n_los), n_gal, n_los


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sims_dir', type=str, required=True)
    ap.add_argument('--prefix', type=str, default='train', choices=['train', 'val'])
    ap.add_argument('--n_examples', type=int, default=6,
                    help='Number of individual simulated draws to plot as heatmaps.')
    ap.add_argument('--seed', type=int, default=0, help='Which draws to pick, if randomizing.')
    ap.add_argument('--output_dir', type=str, required=True)
    args = ap.parse_args()

    theta, n_gal, n_los = load_theta(args.sims_dir, args.prefix)
    n_sim = len(theta)
    print(f"Loaded {n_sim} sims, {n_gal} galaxies x {n_los} LOS bins each, "
          f"from {args.sims_dir} (prefix={args.prefix!r})")

    ionized_frac_per_draw = theta.mean(axis=(1, 2))
    print(f"Ionized fraction per draw: mean={ionized_frac_per_draw.mean():.3f}, "
          f"std={ionized_frac_per_draw.std():.3f}, "
          f"range=[{ionized_frac_per_draw.min():.3f}, {ionized_frac_per_draw.max():.3f}]")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Individual example draws ────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    n_examples = min(args.n_examples, n_sim)
    idx = rng.choice(n_sim, size=n_examples, replace=False)
    ncols = min(3, n_examples)
    nrows = int(np.ceil(n_examples / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False,
                             constrained_layout=True)
    for ax, i in zip(axes.flat, idx):
        im = ax.imshow(theta[i], aspect='auto', cmap='cividis', vmin=0, vmax=1,
                       interpolation='nearest')
        ax.set_title(f"sim {i}  (ionized frac={theta[i].mean():.2f})", fontsize=9)
        ax.set_xlabel('LOS bin (0=near source, -1=near z_end)')
        ax.set_ylabel('galaxy index')
    for ax in axes.flat[n_examples:]:
        ax.axis('off')
    fig.colorbar(im, ax=axes, shrink=0.6, label='0=neutral, 1=ionized')
    fig.suptitle(f"{args.prefix}: {n_examples} example simulated ionization masks "
                f"({n_gal} gal x {n_los} LOS bins)")
    out1 = os.path.join(args.output_dir, f'{args.prefix}_example_masks.png')
    fig.savefig(out1, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out1}")

    # ── Aggregate: mean ionized fraction per pixel across the whole batch ──
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(theta.mean(axis=0), aspect='auto', cmap='cividis', vmin=0, vmax=1,
                   interpolation='nearest')
    ax.set_xlabel('LOS bin (0=near source, -1=near z_end)')
    ax.set_ylabel('galaxy index')
    ax.set_title(f"{args.prefix}: mean ionized fraction per pixel, across {n_sim} draws")
    fig.colorbar(im, ax=ax, label='mean(theta)')
    out2 = os.path.join(args.output_dir, f'{args.prefix}_mean_pixel_map.png')
    fig.savefig(out2, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out2}")

    # ── Histogram of per-draw ionized fraction ──────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ionized_frac_per_draw, bins=30, color='steelblue', edgecolor='k', alpha=0.8)
    ax.set_xlabel('ionized fraction (mean over all pixels in one draw)')
    ax.set_ylabel('count')
    ax.set_title(f"{args.prefix}: per-draw ionized fraction, {n_sim} sims")
    out3 = os.path.join(args.output_dir, f'{args.prefix}_ionized_frac_hist.png')
    fig.savefig(out3, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out3}")


if __name__ == '__main__':
    main()
