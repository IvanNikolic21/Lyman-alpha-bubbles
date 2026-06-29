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


# ── Ionized fraction ─────────────────────────────────────────────────────────

def f_ion_samples(posterior_samples, prior_lo, prior_hi, n_bub, n_mc=20000, seed=0):
    """MC estimate of ionized volume fraction for each posterior sample.

    Draws n_mc points uniformly in the prior box, checks which fall inside
    at least one bubble, returns the fraction for each posterior sample.

    Parameters
    ----------
    posterior_samples : (n_post, ndim)
    prior_lo, prior_hi : (4,) — the 1-bubble prior bounds; x/y/z box same for all models
    n_bub : 1, 2, or 3
    n_mc  : MC points per posterior sample (20 000 gives < 1% MC error on f_ion)

    Returns
    -------
    f_ions : (n_post,)
    """
    rng = np.random.default_rng(seed)
    # Sample points uniform in the (x, y, z) prior box
    pts = rng.uniform(size=(n_mc, 3)) * (prior_hi[:3] - prior_lo[:3]) + prior_lo[:3]

    n_post = len(posterior_samples)
    f_ions = np.empty(n_post)

    for i, theta in enumerate(posterior_samples):
        inside = np.zeros(n_mc, dtype=bool)
        for b in range(n_bub):
            xb, yb, zb, rb = theta[b * 4: b * 4 + 4]
            d2 = ((pts[:, 0] - xb) ** 2
                  + (pts[:, 1] - yb) ** 2
                  + (pts[:, 2] - zb) ** 2)
            inside |= d2 < rb ** 2
        f_ions[i] = inside.mean()

    return f_ions


def f_ion_truth(theta_truth, prior_lo, prior_hi, n_bub, n_mc=100000, seed=1):
    """High-accuracy MC estimate of the true ionized fraction for one theta."""
    return f_ion_samples(
        theta_truth[np.newaxis, :], prior_lo, prior_hi, n_bub,
        n_mc=n_mc, seed=seed,
    )[0]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir')
    parser.add_argument('--n_bub', type=int, default=1)
    parser.add_argument('--out',   type=str, default=None)
    parser.add_argument('--n_mc',  type=int, default=20000,
                        help='MC points for ionized-fraction estimate (default 20000)')
    args = parser.parse_args()

    results = load_seeds(args.result_dir, args.n_bub)
    n_seeds = len(results)

    _NAMES = {
        1: ['x_bub', 'y_bub', 'z_bub', 'r_bub'],
        2: ['x1', 'y1', 'z1', 'r1', 'x2', 'y2', 'z2', 'r2'],
        3: ['x1', 'y1', 'z1', 'r1', 'x2', 'y2', 'z2', 'r2', 'x3', 'y3', 'z3', 'r3'],
    }
    param_names = _NAMES[args.n_bub]
    ndim = len(param_names)

    truth   = np.array([r['theta_truth']  for r in results])
    median  = np.array([r['post_median']  for r in results])
    p16     = np.array([r['post_p16']     for r in results])
    p84     = np.array([r['post_p84']     for r in results])
    map_est = np.array([r['post_map']     for r in results])
    n_ins   = np.array([r['n_inside']     for r in results])
    try:
        n_lae = np.array([r['n_lae_inside'] for r in results])
        has_lae = True
    except KeyError:
        has_lae = False

    prior_lo = results[0]['prior_lo']
    prior_hi = results[0]['prior_hi']

    sigma = (p84 - p16) / 2.0
    resid = (median - truth) / np.where(sigma > 0, sigma, 1.0)

    # ── Ionized fraction ─────────────────────────────────────────────────────
    print('Computing ionized fractions...', flush=True)
    f_ion_true_all = np.array([
        f_ion_truth(truth[s], prior_lo, prior_hi, args.n_bub)
        for s in range(n_seeds)
    ])
    f_ion_post_all = []   # list of (n_post,) arrays, one per seed
    for s, r in enumerate(results):
        post = r['posterior_samples']   # (n_post, ndim)
        f_ion_post_all.append(
            f_ion_samples(post, prior_lo, prior_hi, args.n_bub, n_mc=args.n_mc, seed=s)
        )

    f_ion_post_median = np.array([np.median(f) for f in f_ion_post_all])
    f_ion_post_p16    = np.array([np.percentile(f, 16) for f in f_ion_post_all])
    f_ion_post_p84    = np.array([np.percentile(f, 84) for f in f_ion_post_all])

    # ── Text summary ─────────────────────────────────────────────────────────
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
        bias  = median[:, pi] - truth[:, pi]
        norm  = resid[:, pi]
        in_68 = ((truth[:, pi] >= p16[:, pi]) & (truth[:, pi] <= p84[:, pi])).mean()
        print(f'{pn:8s}  {bias.mean():+10.3f}  {bias.std():10.3f}  '
              f'{np.abs(norm).mean():14.3f}  {in_68:12.2%}')

    print()
    print('Ionized fraction f_ion:')
    bias_fion = f_ion_post_median - f_ion_true_all
    in_68_fion = ((f_ion_true_all >= f_ion_post_p16) &
                  (f_ion_true_all <= f_ion_post_p84)).mean()
    print(f"  truth range:    [{f_ion_true_all.min():.3f}, {f_ion_true_all.max():.3f}]")
    print(f"  mean bias:      {bias_fion.mean():+.4f}")
    print(f"  std  bias:      {bias_fion.std():.4f}")
    print(f"  coverage 68%:   {in_68_fion:.2%}")

    # ── Plots ────────────────────────────────────────────────────────────────
    ncols = min(ndim, 4)
    nrows_top = (ndim + ncols - 1) // ncols
    # Extra rows: normalized residuals + f_ion
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows_top + 7))
    gs  = gridspec.GridSpec(nrows_top + 2, ncols, figure=fig,
                            hspace=0.50, wspace=0.35)

    for pi, pn in enumerate(param_names):
        row, col = divmod(pi, ncols)
        ax = fig.add_subplot(gs[row, col])

        lo  = min(truth[:, pi].min(), p16[:, pi].min())
        hi  = max(truth[:, pi].max(), p84[:, pi].max())
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', lw=0.8, alpha=0.5)

        ax.errorbar(truth[:, pi], median[:, pi],
                    yerr=[median[:, pi] - p16[:, pi], p84[:, pi] - median[:, pi]],
                    fmt='o', ms=5, lw=1.2, capsize=3, color='C0', label='median ± 68%')
        ax.scatter(truth[:, pi], map_est[:, pi],
                   marker='x', s=40, color='C1', zorder=5, label='MAP')
        ax.set_xlabel(f'truth {pn}')
        ax.set_ylabel(f'recovered {pn}')
        ax.set_title(pn)
        if pi == 0:
            ax.legend(fontsize=7)

    # Normalized residuals
    ax_res = fig.add_subplot(gs[nrows_top, :])
    for pi in range(ndim):
        ax_res.scatter(
            np.full(n_seeds, pi) + np.random.uniform(-0.15, 0.15, n_seeds),
            resid[:, pi], alpha=0.7, s=20, color='C0',
        )
    ax_res.axhline(0,  color='k',    lw=0.8, ls='--')
    ax_res.axhline(+1, color='grey', lw=0.6, ls=':')
    ax_res.axhline(-1, color='grey', lw=0.6, ls=':')
    ax_res.set_xticks(np.arange(ndim))
    ax_res.set_xticklabels(param_names, rotation=30, ha='right')
    ax_res.set_ylabel('(median − truth) / σ')
    ax_res.set_title('Normalized residuals')

    # Ionized fraction: truth vs recovered, one point per seed
    ax_fion = fig.add_subplot(gs[nrows_top + 1, :])
    lo_f  = min(f_ion_true_all.min(), f_ion_post_p16.min()) - 0.02
    hi_f  = max(f_ion_true_all.max(), f_ion_post_p84.max()) + 0.02
    ax_fion.plot([lo_f, hi_f], [lo_f, hi_f], 'k--', lw=0.8, alpha=0.5)
    ax_fion.errorbar(
        f_ion_true_all, f_ion_post_median,
        yerr=[f_ion_post_median - f_ion_post_p16, f_ion_post_p84 - f_ion_post_median],
        fmt='o', ms=6, lw=1.4, capsize=4, color='C2',
        label='posterior median ± 68%',
    )
    for s in range(n_seeds):
        ax_fion.annotate(f's{s}', (f_ion_true_all[s], f_ion_post_median[s]),
                         fontsize=6, alpha=0.6, xytext=(3, 3), textcoords='offset points')
    ax_fion.set_xlabel('truth $f_{\\rm ion}$')
    ax_fion.set_ylabel('recovered $f_{\\rm ion}$')
    ax_fion.set_title(
        f'Ionized fraction  (mean bias {bias_fion.mean():+.3f}, '
        f'std {bias_fion.std():.3f}, coverage {in_68_fion:.0%})'
    )
    ax_fion.legend(fontsize=8)

    fig.suptitle(f'Bias test M{args.n_bub}  ({n_seeds} seeds, noiseless mock)',
                 fontsize=12, y=1.005)

    out_path = args.out or os.path.join(
        args.result_dir, f'bias_M{args.n_bub}_analysis.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f'\nPlot saved to {out_path}')


if __name__ == '__main__':
    main()