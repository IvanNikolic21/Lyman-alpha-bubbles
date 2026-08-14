"""
P-P (coverage) test for `real_data_run.py`'s EW likelihood, built on top of
`mock_bias_test.py`'s per-seed output.

`mock_bias_test.py` already draws each seed's truth bubble from the CURRENT
prior (`rdr._prior_transform*`, i.e. the EMG-truncated r_bub prior + real
catalog geometry, no more /n_bub split -- see the bubble-size-prior work) and
saves the full equal-weighted posterior for that seed. This script only adds
the missing piece: for each seed and parameter, the quantile at which the
truth falls in that seed's marginal posterior. Under correct, unbiased
inference these quantiles are uniform on [0, 1] -- deviations show up as
S-curves against the diagonal, same diagnostic as the original `pp_test.py`
(which calibrated the old flux-space mock pipeline, not this one).

Does NOT itself run dynesty -- run `mock_bias_test.py` first (needs the
cluster: py21cmfast etc.), then point this script at its --output_dir.

Usage
-----
# after: python mock_bias_test.py --n_bub 1 --n_seeds 25 \
#            --nlive 300 --dlogz 0.5 --n_inside_tau 1000 --output_dir bias_M1_pp/
python mock_pp_test.py --input_dir bias_M1_pp/ --n_bub 1 --output_dir bias_M1_pp/

Caveats (inherited from mock_bias_test.py's design, not fixed here)
---------------------------------------------------------------------
- Mock EW is noiseless and all-detection (no upper limits) -- doesn't probe
  the actual 33/43-upper-limit regime of the real catalog.
- fesc reweighting is disabled for these mocks (`_S.fesc_has` forced False),
  so this test calibrates the EW+prior likelihood only, not the fesc term.
"""

import argparse
import glob
import os

import numpy as np
from scipy.stats import binom, percentileofscore
import matplotlib.pyplot as plt

PARAM_NAMES = {
    1: ['x_bub', 'y_bub', 'z_bub', 'r_bub'],
    2: ['x1_bub', 'y1_bub', 'z1_bub', 'r1_bub',
        'x2_bub', 'y2_bub', 'z2_bub', 'r2_bub'],
    3: ['x1_bub', 'y1_bub', 'z1_bub', 'r1_bub',
        'x2_bub', 'y2_bub', 'z2_bub', 'r2_bub',
        'x3_bub', 'y3_bub', 'z3_bub', 'r3_bub'],
}


def compute_quantiles(input_dir: str, n_bub: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (quantiles, seeds): quantiles is (n_seeds, ndim)."""
    files = sorted(glob.glob(os.path.join(input_dir, f'bias_M{n_bub}_seed*.npz')))
    if not files:
        raise FileNotFoundError(
            f"No bias_M{n_bub}_seed*.npz files found in {input_dir}. "
            "Run mock_bias_test.py first."
        )

    param_names = PARAM_NAMES[n_bub]
    ndim = len(param_names)
    all_q, seeds = [], []
    for f in files:
        d = np.load(f)
        truth   = d['theta_truth']          # (ndim,)
        samples = d['posterior_samples']    # (n_samp, ndim)
        if truth.shape[0] != ndim:
            raise ValueError(f"{f}: theta_truth has {truth.shape[0]} dims, expected {ndim} for M{n_bub}.")
        q = np.array([
            percentileofscore(samples[:, i], truth[i]) / 100.0
            for i in range(ndim)
        ])
        all_q.append(q)
        seeds.append(int(d['seed']))

    return np.array(all_q), np.array(seeds)


def plot_pp(quantiles_all: np.ndarray, param_names: list, output_path: str,
           nlive: int = None) -> None:
    """Same tolerance-band construction as the original pp_test.py, generalized
    to arbitrary ndim/param_names."""
    n     = quantiles_all.shape[0]
    ndim  = quantiles_all.shape[1]
    fig, axes = plt.subplots(1, ndim, figsize=(4 * ndim, 4), sharey=True)
    if ndim == 1:
        axes = [axes]

    alpha = np.linspace(0, 1, 300)
    lo1   = binom.ppf(0.16, n, alpha) / n
    hi1   = binom.ppf(0.84, n, alpha) / n
    lo2   = binom.ppf(0.025, n, alpha) / n
    hi2   = binom.ppf(0.975, n, alpha) / n

    for i, (ax, label) in enumerate(zip(axes, param_names)):
        qs        = np.sort(quantiles_all[:, i])
        empirical = np.arange(1, n + 1) / n

        ax.fill_between(alpha, lo2, hi2, color='lightblue', label='95% band')
        ax.fill_between(alpha, lo1, hi1, color='steelblue', alpha=0.6, label='68% band')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='ideal')
        ax.step(qs, empirical, color='red', lw=1.5, where='post', label='empirical')
        ax.set_xlabel('Posterior quantile')
        ax.set_title(label)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if i == 0:
            ax.set_ylabel('Fraction ≤ quantile')
        ax.legend(fontsize=7)

    title = f'P-P coverage test (EW likelihood, EMG r_bub prior)  N = {n} seeds'
    if nlive is not None:
        title += f'  nlive = {nlive}'
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved {output_path}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',  type=str, required=True,
                        help='Directory of bias_M{n_bub}_seed*.npz files from mock_bias_test.py')
    parser.add_argument('--n_bub',      type=int, default=1, choices=[1, 2, 3])
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Defaults to --input_dir')
    parser.add_argument('--nlive',      type=int, default=None,
                        help='Only used to annotate the plot title.')
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    quantiles, seeds = compute_quantiles(args.input_dir, args.n_bub)
    param_names = PARAM_NAMES[args.n_bub]

    print(f"Loaded {len(seeds)} seeds: {seeds.tolist()}")
    plot_pp(quantiles, param_names, os.path.join(output_dir, f'pp_plot_M{args.n_bub}.png'),
            nlive=args.nlive)

    print(f"\nP-P summary ({len(seeds)} seeds):")
    ideal_std = 1 / np.sqrt(12)
    for i, name in enumerate(param_names):
        qs = quantiles[:, i]
        print(f"  {name:8s}  mean = {qs.mean():.3f} (ideal 0.500)  "
              f"std = {qs.std():.3f} (ideal {ideal_std:.3f})")

    out_npz = os.path.join(output_dir, f'pp_quantiles_M{args.n_bub}.npz')
    np.savez(out_npz, quantiles=quantiles, seeds=seeds, param_names=param_names)
    print(f"Saved {out_npz}", flush=True)