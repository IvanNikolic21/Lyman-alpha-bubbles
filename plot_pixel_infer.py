"""
Visual inspection of `sbi_pixel_field.py infer` output (`pixel_infer.npz`):
the marginal per-pixel ionization probability map, a handful of resampled
joint map draws, and the importance-weight distribution behind the reported
effective sample size (ESS) -- low ESS means a few draws dominate the
weighted average/resample, which the weight histogram makes directly visible.

Standalone (numpy/matplotlib only), runs anywhere `pixel_infer.npz` is.

Usage
-----
python plot_pixel_infer.py --infer_npz sbi_runs/m1_smoketest/pixel_infer.npz \\
    --n_examples 6 --output_dir sbi_runs/m1_smoketest/plots
"""
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--infer_npz', type=str, required=True, help='pixel_infer.npz from `infer`.')
    ap.add_argument('--n_examples', type=int, default=6,
                    help='Number of resampled joint_samples draws to plot as heatmaps.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output_dir', type=str, required=True)
    args = ap.parse_args()

    d = np.load(args.infer_npz)
    marginal_map = d['marginal_map']
    joint_samples = d['joint_samples']
    weights = d['weights']
    ess = float(d['ess'])
    pool_size = int(d['pool_size'])
    n_gal, n_los = marginal_map.shape

    print(f"Loaded {args.infer_npz}: marginal_map {marginal_map.shape}, "
          f"{len(joint_samples)} joint_samples, pool_size={pool_size}, "
          f"ESS={ess:.1f} ({100 * ess / pool_size:.2f}% of pool)")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Marginal per-pixel probability map ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(marginal_map, aspect='auto', cmap='cividis', vmin=0, vmax=1,
                   interpolation='nearest')
    ax.set_xlabel('LOS bin (0=near source, -1=near z_end)')
    ax.set_ylabel('galaxy index')
    ax.set_title(f"Marginal P(ionized) per pixel -- ESS={ess:.1f}/{pool_size} "
                f"({100 * ess / pool_size:.1f}%)")
    fig.colorbar(im, ax=ax, label='P(ionized)')
    out1 = os.path.join(args.output_dir, 'infer_marginal_map.png')
    fig.savefig(out1, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out1}")

    # ── Example resampled joint draws ───────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    n_examples = min(args.n_examples, len(joint_samples))
    idx = rng.choice(len(joint_samples), size=n_examples, replace=False)
    ncols = min(3, n_examples)
    nrows = int(np.ceil(n_examples / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False,
                             constrained_layout=True)
    for ax, i in zip(axes.flat, idx):
        im2 = ax.imshow(joint_samples[i], aspect='auto', cmap='cividis', vmin=0, vmax=1,
                        interpolation='nearest')
        ax.set_title(f"joint sample {i}", fontsize=9)
        ax.set_xlabel('LOS bin')
        ax.set_ylabel('galaxy index')
    for ax in axes.flat[n_examples:]:
        ax.axis('off')
    fig.colorbar(im2, ax=axes, shrink=0.6, label='0=neutral, 1=ionized')
    fig.suptitle(f"{n_examples} resampled joint map draws (SIR from the reweighted pool)")
    out2 = os.path.join(args.output_dir, 'infer_joint_examples.png')
    fig.savefig(out2, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out2}")

    # ── Importance-weight concentration behind the ESS number ─────────────
    sorted_w = np.sort(weights)[::-1]
    cumfrac = np.cumsum(sorted_w)
    n_for_50 = int(np.searchsorted(cumfrac, 0.5) + 1)
    n_for_90 = int(np.searchsorted(cumfrac, 0.9) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(weights, bins=50, color='steelblue', edgecolor='k', alpha=0.8)
    axes[0].set_xlabel('self-normalized importance weight')
    axes[0].set_ylabel('count')
    axes[0].set_title('Weight distribution')

    axes[1].plot(np.arange(1, len(sorted_w) + 1), cumfrac, color='steelblue')
    axes[1].axhline(0.5, color='gray', ls='--', lw=1)
    axes[1].axhline(0.9, color='gray', ls='--', lw=1)
    axes[1].set_xlabel('pool draws, sorted by weight (descending)')
    axes[1].set_ylabel('cumulative weight fraction')
    axes[1].set_title(f"{n_for_50} draws carry 50% of the weight, "
                      f"{n_for_90} draws carry 90%\n(pool size {pool_size}, ESS {ess:.1f})")
    fig.tight_layout()
    out3 = os.path.join(args.output_dir, 'infer_weight_concentration.png')
    fig.savefig(out3, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out3}")
    print(f"{n_for_50} of {pool_size} pool draws carry 50% of the total weight; "
          f"{n_for_90} carry 90%.")


if __name__ == '__main__':
    main()
