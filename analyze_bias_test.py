"""
Analyze mock_bias_test.py output.

Usage
-----
python analyze_bias_test.py bias_M1/           # all bias_M1_seed*.npz in that dir
python analyze_bias_test.py bias_M1/ --n_bub 1 --out bias_M1_analysis.png
"""

import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_seeds(result_dir, n_bub):
    pattern = os.path.join(result_dir, f'bias_M{n_bub}_seed*.npz')
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f'No files matching {pattern}')
    print(f'Loading {len(files)} seed files from {result_dir}')
    return [dict(np.load(f, allow_pickle=False)) for f in files]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir')
    parser.add_argument('--n_bub', type=int, default=1)
    parser.add_argument('--out',   type=str, default=None)
    args = parser.parse_args()

    results = load_seeds(args.result_dir, args.n_bub)
    n_seeds = len(results)

    # ── Infer param names from array shapes ──────────────────────────────────
    _NAMES = {
        1: ['x_bub', 'y_bub', 'z_bub', 'r_bub'],
        2: ['x1', 'y1', 'z1', 'r1', 'x2', 'y2', 'z2', 'r2'],
        3: ['x1', 'y1', 'z1', 'r1', 'x2', 'y2', 'z2', 'r2', 'x3', 'y3', 'z3', 'r3'],
    }
    param_names = _NAMES[args.n_bub]
    ndim = len(param_names)

    truth   = np.array([r['theta_truth']  for r in results])   # (S, D)
    median  = np.array([r['post_median']  for r in results])
    p16     = np.array([r['post_p16']     for r in results])
    p84     = np.array([r['post_p84']     for r in results])
    map_est = np.array([r['post_map']     for r in results])
    n_ins   = np.array([r['n_inside']     for r in results])
    try:
        n_lae   = np.array([r['n_lae_inside'] for r in results])
        has_lae = True
    except KeyError:
        has_lae = False

    sigma   = (p84 - p16) / 2.0                  # approximate 1-sigma
    resid   = (median - truth) / np.where(sigma > 0, sigma, 1.0)   # normalized bias

    # ── Print text summary ───────────────────────────────────────────────────
    print(f'\nBias summary: M{args.n_bub}, {n_seeds} seeds')
    print(f'  n_inside per seed:   {n_ins}')
    if has_lae:
        print(f'  n_lae_inside:        {n_lae}')
    print()
    hdr = f"{'param':8s}  {'mean bias':>10s}  {'std bias':>10s}  "
    hdr += f"{'mean |bias|/σ':>14s}  {'coverage 68%':>12s}"
    print(hdr)
    print('-' * len(hdr))
    for pi, pn in enumerate(param_names):
        bias      = median[:, pi] - truth[:, pi]
        norm      = resid[:, pi]
        in_68     = ((truth[:, pi] >= p16[:, pi]) & (truth[:, pi] <= p84[:, pi])).mean()
        print(f'{pn:8s}  {bias.mean():+10.3f}  {bias.std():10.3f}  '
              f'{np.abs(norm).mean():14.3f}  {in_68:12.2%}')

    # ── Plots ────────────────────────────────────────────────────────────────
    ncols = min(ndim, 4)
    nrows_top = (ndim + ncols - 1) // ncols
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows_top + 3))
    gs  = gridspec.GridSpec(nrows_top + 1, ncols, figure=fig,
                            hspace=0.45, wspace=0.35)

    for pi, pn in enumerate(param_names):
        row, col = divmod(pi, ncols)
        ax = fig.add_subplot(gs[row, col])

        lo = min(truth[:, pi].min(), p16[:, pi].min())
        hi = max(truth[:, pi].max(), p84[:, pi].max())
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', lw=0.8, alpha=0.5)

        yerr_lo = median[:, pi] - p16[:, pi]
        yerr_hi = p84[:, pi] - median[:, pi]
        ax.errorbar(truth[:, pi], median[:, pi],
                    yerr=[yerr_lo, yerr_hi],
                    fmt='o', ms=5, lw=1.2, capsize=3, color='C0', label='median ± 68%')
        ax.scatter(truth[:, pi], map_est[:, pi],
                   marker='x', s=40, color='C1', zorder=5, label='MAP')

        ax.set_xlabel(f'truth {pn}')
        ax.set_ylabel(f'recovered {pn}')
        ax.set_title(pn)
        if pi == 0:
            ax.legend(fontsize=7)

    # Bottom row: normalized residuals per parameter
    ax_res = fig.add_subplot(gs[nrows_top, :])
    positions = np.arange(ndim)
    for pi, pn in enumerate(param_names):
        ax_res.scatter(np.full(n_seeds, pi) + np.random.uniform(-0.15, 0.15, n_seeds),
                       resid[:, pi], alpha=0.7, s=20, color='C0')
    ax_res.axhline(0,  color='k',   lw=0.8, ls='--')
    ax_res.axhline(+1, color='grey', lw=0.6, ls=':')
    ax_res.axhline(-1, color='grey', lw=0.6, ls=':')
    ax_res.set_xticks(positions)
    ax_res.set_xticklabels(param_names, rotation=30, ha='right')
    ax_res.set_ylabel('(median − truth) / σ_posterior')
    ax_res.set_title('Normalized residuals  (unbiased → scatter around 0, σ ≈ 1)')

    fig.suptitle(f'Bias test M{args.n_bub}  ({n_seeds} seeds, noiseless mock)',
                 fontsize=12, y=1.01)

    out_path = args.out or os.path.join(
        args.result_dir, f'bias_M{args.n_bub}_analysis.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f'\nPlot saved to {out_path}')


if __name__ == '__main__':
    main()