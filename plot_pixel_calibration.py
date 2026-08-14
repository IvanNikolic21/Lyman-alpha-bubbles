"""
Reliability diagram for `sbi_pixel_calibrate.py`'s output (`pixel_calibration.npz`):
predicted P(neutral) vs. empirical neutral frequency, binned, plus a Brier
score and ECE summary. Standalone (numpy/matplotlib only, no torch needed) --
the scoring step needs the cluster, this plotting step doesn't.

Usage
-----
python plot_pixel_calibration.py --calib_npz sbi_runs/pixel_v2/pixel_calibration.npz \\
    --output_dir sbi_runs/pixel_v2/plots
"""
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK_PRIMARY, INK_SECONDARY, INK_MUTED = '#0b0b0b', '#52514e', '#898781'
GRID_HAIRLINE, SURFACE, BLUE = '#e1e0d9', '#fcfcfb', '#2a78d6'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--calib_npz', type=str, required=True)
    ap.add_argument('--output_dir', type=str, required=True)
    args = ap.parse_args()

    d = np.load(args.calib_npz)
    bin_edges = d['bin_edges']
    bin_mean_pred = d['bin_mean_pred']
    emp_freq = d['emp_freq']
    bin_count = d['bin_count']
    brier = float(d['brier'])
    ece = float(d['ece'])
    n_queries = int(d['n_queries'])
    all_pred = d['all_pred']

    print(f"Loaded {args.calib_npz}: {n_queries} queries, {len(all_pred)} pixel pairs, "
          f"Brier={brier:.4f}, ECE={ece:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=SURFACE,
                             gridspec_kw={'width_ratios': [1.1, 1]})

    # ── Reliability diagram ──────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(SURFACE)
    ax.plot([0, 1], [0, 1], color=INK_MUTED, lw=1, ls=(0, (4, 2)), label='perfect calibration')
    valid = bin_count > 0
    ax.plot(bin_mean_pred[valid], emp_freq[valid], 'o-', color=BLUE, lw=2, ms=6,
           label='observed')
    # Point size encodes how many pixel-pairs landed in each bin -- a thin
    # reliability curve built from a handful of points in some bins is much
    # less trustworthy than one built from thousands.
    sizes = 20 + 180 * (bin_count[valid] / bin_count[valid].max())
    ax.scatter(bin_mean_pred[valid], emp_freq[valid], s=sizes, color=BLUE, alpha=0.25, zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('predicted P(neutral)', color=INK_SECONDARY)
    ax.set_ylabel('empirical neutral frequency', color=INK_SECONDARY)
    ax.set_title(f'Reliability diagram — Brier={brier:.4f}, ECE={ece:.4f}',
                fontsize=11, color=INK_PRIMARY, loc='left')
    ax.tick_params(colors=INK_MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID_HAIRLINE)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc='upper left')

    # ── Predicted-probability histogram (what the marginal-map values look
    # like across the calibration set, and how much weight is behind each
    # reliability-diagram point) ───────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(SURFACE)
    ax2.hist(all_pred, bins=30, color=BLUE, alpha=0.8, edgecolor=GRID_HAIRLINE)
    ax2.set_xlabel('predicted P(neutral)', color=INK_SECONDARY)
    ax2.set_ylabel('pixel-pair count', color=INK_SECONDARY)
    ax2.set_title('Predicted-probability distribution', fontsize=11, color=INK_PRIMARY, loc='left')
    ax2.tick_params(colors=INK_MUTED)
    for spine in ax2.spines.values():
        spine.set_color(GRID_HAIRLINE)

    fig.tight_layout()
    out_path = os.path.join(args.output_dir, 'pixel_calibration_reliability.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out_path}")

    print("\nPer-bin detail:")
    for b in range(len(bin_edges) - 1):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if bin_count[b] == 0:
            print(f"  [{lo:.1f},{hi:.1f}) n=0 (empty)")
        else:
            print(f"  [{lo:.1f},{hi:.1f})  n={bin_count[b]:6d}  "
                  f"mean_pred={bin_mean_pred[b]:.3f}  empirical={emp_freq[b]:.3f}  "
                  f"|diff|={abs(emp_freq[b] - bin_mean_pred[b]):.3f}")


if __name__ == '__main__':
    main()
