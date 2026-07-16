"""
Simulation-Based Calibration (SBC) for the SBI mode -- checks that a trained
NPE posterior (`sbi_real_data.py train`) is actually calibrated before it's
trusted on real data. Adapted from `pp_test.py`'s P-P plot / binomial-band
machinery, generalized to arbitrary ndim/param_names so one implementation
serves M1/M2/M3, and fed by `sbi_real_data.py`'s held-out `val_batch_*.npz`
files instead of `pp_test.py`'s from-scratch mock generation.

This is a HARD GATE: `sbi_real_data.py infer`'s output on real data should
not be trusted until this passes (near-uniform ranks within the binomial
tolerance bands for every parameter). Systematic deviation (ranks pushed
toward 0/1, or U/bathtub-shaped) means under/over-dispersion, insufficient
training simulations, or a mismatch between the simulator's `x` and the real
`x_obs` construction -- see `sbi_real_data.py`'s module docstring.

Usage
-----
python sbi_calibrate.py --posterior sbi_runs/m1/posterior_m1.pt \\
    --sims_dir sbi_runs/m1/sims --output_dir sbi_runs/m1
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

import sbi_real_data as sbird   # reuses load_sims(); no py21cmfast-heavy state needed here


def compute_ranks(posterior, theta_val, x_val, n_posterior_samples, device='cpu'):
    """Per-parameter rank statistic for each held-out (theta, x) pair:
    fraction of posterior samples below the true theta. Uniform ranks under
    correct calibration -- same principle as `pp_test.py`'s
    `percentileofscore`, fed SBI posterior samples instead of dynesty's
    `equal_samples`. `device` should match whatever the posterior's network
    was loaded onto (see `sbi_real_data.py run_infer`'s `map_location` note)."""
    import torch
    ndim = theta_val.shape[1]
    ranks = np.empty((len(theta_val), ndim))
    for i in range(len(theta_val)):
        samples = posterior.sample(
            (n_posterior_samples,), x=torch.as_tensor(x_val[i], dtype=torch.float32, device=device),
            show_progress_bars=False,
        ).detach().cpu().numpy()
        ranks[i] = (samples < theta_val[i]).mean(axis=0)
        if (i + 1) % 50 == 0:
            print(f"[sbc] {i + 1}/{len(theta_val)} held-out sims ranked", flush=True)
    return ranks


def plot_pp(ranks, param_names, output_path, title_extra=''):
    """Generalizes `pp_test.py`'s `plot_pp` to arbitrary ndim/param_names."""
    n, ndim = ranks.shape
    fig, axes = plt.subplots(1, ndim, figsize=(4 * ndim, 4), sharey=True, squeeze=False)
    axes = axes[0]

    alpha = np.linspace(0, 1, 300)
    lo1 = binom.ppf(0.16, n, alpha) / n
    hi1 = binom.ppf(0.84, n, alpha) / n
    lo2 = binom.ppf(0.025, n, alpha) / n
    hi2 = binom.ppf(0.975, n, alpha) / n

    for i, (ax, label) in enumerate(zip(axes, param_names)):
        qs = np.sort(ranks[:, i])
        empirical = np.arange(1, n + 1) / n

        ax.fill_between(alpha, lo2, hi2, color='lightblue', label='95% band')
        ax.fill_between(alpha, lo1, hi1, color='steelblue', alpha=0.6, label='68% band')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='ideal')
        ax.step(qs, empirical, color='red', lw=1.5, where='post', label='empirical')
        ax.set_xlabel('Posterior rank')
        ax.set_title(str(label))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if i == 0:
            ax.set_ylabel('Fraction ≤ rank')
        ax.legend(fontsize=7)

    fig.suptitle(f'SBC coverage test (N = {n} held-out sims){title_extra}')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved {output_path}", flush=True)


def check_gate(ranks, param_names):
    """Pass/fail per parameter: does the empirical rank CDF stay within the
    95% binomial tolerance band across the whole [0, 1] range (not just at
    one point)? Returns True only if every parameter passes -- the gate
    `sbi_real_data.py infer` should wait on."""
    n = ranks.shape[0]
    all_ok = True
    for i, name in enumerate(param_names):
        qs = np.sort(ranks[:, i])
        empirical = np.arange(1, n + 1) / n
        lo2 = binom.ppf(0.025, n, qs) / n
        hi2 = binom.ppf(0.975, n, qs) / n
        breach = (empirical < lo2) | (empirical > hi2)
        frac_breach = breach.mean()
        status = "FAIL" if frac_breach > 0.05 else "pass"
        all_ok &= (status == "pass")
        print(f"  {str(name):8s}  {status}  ({frac_breach * 100:.1f}% of the empirical CDF "
              f"outside the 95% band)", flush=True)
    return bool(all_ok)


def _try_sbi_native_sbc(posterior, theta_val, x_val, param_names, output_dir, n_bub):
    """Optional cross-check: some `sbi` versions ship
    `sbi.analysis.run_sbc`/`sbc_rank_plot`, which may be more efficient/
    numerically robust than the hand-rolled version above. Best-effort --
    silently skipped if unavailable, since the hand-rolled check above is the
    one this script's gate actually depends on and doesn't need this to work."""
    try:
        import torch
        from sbi.analysis import run_sbc, sbc_rank_plot
    except ImportError:
        print("[sbc] sbi.analysis.run_sbc not available in this sbi version -- "
              "skipping native cross-check (not required, hand-rolled check above stands).",
              flush=True)
        return

    try:
        ranks_native, dap_samples = run_sbc(
            torch.as_tensor(theta_val, dtype=torch.float32),
            torch.as_tensor(x_val, dtype=torch.float32),
            posterior, num_posterior_samples=200,
        )
        fig, _ = sbc_rank_plot(ranks_native, num_posterior_samples=200, plot_type="cdf")
        out_path = os.path.join(output_dir, f'sbc_native_plot_m{n_bub}.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"[sbc] native sbi SBC cross-check saved to {out_path}", flush=True)
    except Exception as e:
        print(f"[sbc] native sbi SBC cross-check failed ({e}) -- not fatal, "
              f"hand-rolled check above is the one that gates `infer`.", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--posterior', type=str, required=True, help='.pt from `sbi_real_data.py train`.')
    parser.add_argument('--sims_dir', type=str, required=True,
                        help='--output_dir from `sbi_real_data.py simulate` (uses its val_batch_*.npz).')
    parser.add_argument('--n_posterior_samples', type=int, default=500)
    parser.add_argument('--n_val_max', type=int, default=None,
                        help='Cap the number of held-out sims used (SBC cost scales linearly '
                             'with this) -- useful for a quick check before committing to the '
                             'full held-out set.')
    parser.add_argument('--device', type=str, default='cpu',
                        help="Should match the --device the posterior was trained with -- "
                             "map_location handles moving the checkpoint if it doesn't.")
    parser.add_argument('--skip_native_check', action='store_true')
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    import torch
    checkpoint = torch.load(args.posterior, weights_only=False, map_location=args.device)
    posterior = checkpoint['posterior']
    n_bub = checkpoint['n_bub']
    param_names = checkpoint['param_names']

    theta_val, x_val = sbird.load_sims(args.sims_dir, 'val')
    if args.n_val_max is not None:
        theta_val, x_val = theta_val[:args.n_val_max], x_val[:args.n_val_max]
    print(f"[sbc] {len(theta_val)} held-out validation sims for M{n_bub}", flush=True)

    ranks = compute_ranks(posterior, theta_val, x_val, args.n_posterior_samples, device=args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(os.path.join(args.output_dir, f'sbc_ranks_m{n_bub}.npz'),
             ranks=ranks, param_names=param_names)
    plot_pp(ranks, param_names, os.path.join(args.output_dir, f'sbc_plot_m{n_bub}.png'),
           title_extra=f'  M{n_bub}')

    print(f"\nSBC gate check (M{n_bub}):", flush=True)
    passed = check_gate(ranks, param_names)
    print(f"\n{'PASSED' if passed else 'FAILED'} -- "
          f"{'safe to run `infer` on real data' if passed else 'do NOT trust `infer` on real data yet -- iterate on N_SIM/noise model/network before proceeding'}",
          flush=True)

    if not args.skip_native_check:
        _try_sbi_native_sbc(posterior, theta_val, x_val, param_names, args.output_dir, n_bub)
