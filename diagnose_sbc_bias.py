"""
Diagnose an SBC failure by comparing true vs. posterior-predicted theta
directly on a handful of held-out sims, and surfacing sbi's own rejection-
sampler acceptance-rate warnings (usually only visible as a UserWarning,
easy to miss in a cluster log) -- written after `sbi_calibrate.py`'s M1
SBC came back failing hard (69-91% of the rank CDF outside the 95% band)
with a shape (low-rank excess on every parameter) consistent with the
rejection sampler truncating the accepted sample set near the prior
boundary, not plain undertraining.

Usage
-----
python diagnose_sbc_bias.py --posterior sbi_runs/m1/posterior_m1.pt \\
    --sims_dir sbi_runs/m1_smoketest/sim --n_check 20 \\
    --lya_catalog /groups/astro/ivannik/programs/Lyman-alpha-bubbles/tb_lya.txt \\
    --properties_catalog /groups/astro/ivannik/programs/Lyman-alpha-bubbles/sample_nirspec_properties.txt
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'

import warnings
import argparse
import numpy as np

import sbi_real_data as sbird


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sbird._add_catalog_args(ap)
    ap.add_argument('--posterior', type=str, required=True)
    ap.add_argument('--sims_dir', type=str, required=True)
    ap.add_argument('--n_check', type=int, default=20,
                    help='Number of held-out val sims to check (kept small -- this samples '
                         'the posterior fresh for each one, same cost as one SBC data point).')
    ap.add_argument('--n_posterior_samples', type=int, default=1000)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    import torch
    checkpoint = torch.load(args.posterior, weights_only=False, map_location=args.device)
    posterior = checkpoint['posterior']
    n_bub = checkpoint['n_bub']
    param_names = checkpoint['param_names']

    sbird.rdr._load_catalog_and_priors(
        args.lya_catalog, args.properties_catalog, args.z_lo, args.z_hi,
        args.z_min, args.muv_max, args.main_dir, r_max=args.r_max, prefer=args.prefer,
        legacy_catalog_path=args.legacy_catalog,
    )
    prior_lo, prior_hi = sbird.rdr._S.prior_lo, sbird.rdr._S.prior_hi
    print(f"Prior box: lo={prior_lo}, hi={prior_hi}\n")

    theta_val, x_val = sbird.load_sims(args.sims_dir, 'val')
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(theta_val), size=min(args.n_check, len(theta_val)), replace=False)

    print(f"{'param':>8s}  {'true':>9s}  {'post_mean':>9s}  {'post_med':>9s}  "
          f"{'bias(mean-true)':>16s}  {'near_hi_bound':>13s}  {'near_lo_bound':>13s}")

    biases = {name: [] for name in param_names}
    n_near_hi_bound = {name: 0 for name in param_names}
    for i in idx:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            samples = posterior.sample(
                (args.n_posterior_samples,),
                x=torch.as_tensor(x_val[i], dtype=torch.float32, device=args.device),
                show_progress_bars=False,
            ).detach().cpu().numpy()
            for w in caught:
                if 'accept' in str(w.message).lower() or 'reject' in str(w.message).lower():
                    print(f"  [sim {i}] sbi warning: {w.message}")

        post_mean = samples.mean(axis=0)
        post_med = np.median(samples, axis=0)
        true = theta_val[i]

        for k, name in enumerate(param_names):
            bias = post_mean[k] - true[k]
            biases[name].append(bias)
            frac_range = (prior_hi[k % 4] - prior_lo[k % 4])
            near_hi = np.mean(samples[:, k] > prior_hi[k % 4] - 0.02 * frac_range)
            near_lo = np.mean(samples[:, k] < prior_lo[k % 4] + 0.02 * frac_range)
            if near_hi > 0.05:
                n_near_hi_bound[name] += 1
            print(f"{name:>8s}  {true[k]:9.3f}  {post_mean[k]:9.3f}  {post_med[k]:9.3f}  "
                  f"{bias:16.3f}  {near_hi:13.2%}  {near_lo:13.2%}")
        print()

    print("=" * 70)
    print("Summary over checked sims:")
    for name in param_names:
        b = np.array(biases[name])
        print(f"  {name:8s}  mean(post_mean - true) = {b.mean():+.3f}  "
              f"(std {b.std():.3f})  -- {n_near_hi_bound[name]}/{len(idx)} sims had "
              f">5% of posterior samples within 2% of the prior's upper bound")


if __name__ == '__main__':
    main()
