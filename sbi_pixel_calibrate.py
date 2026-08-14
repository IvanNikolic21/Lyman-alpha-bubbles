"""
Per-pixel calibration check for the pixel-field SBI mode (NRE) -- the
practical substitute for a full SBC gate, which is intractable for a
~3225-dim binary theta (see pixel-field-sbi project memory: "no calibration
procedure exists yet ... genuinely open problem").

Instead of a multivariate rank test over the whole theta vector, this checks
MARGINAL per-pixel calibration via a reliability diagram: for many held-out
(theta, x) pairs (not just the one real x_obs `infer` uses), score each held-
out x against a disjoint held-out pool -- same importance-reweighting
machinery `infer` uses -- to get a predicted P(neutral) map, then compare
against that instance's OWN true theta. Pooling this over many queries and
every pixel gives (predicted probability, actual 0/1 outcome) pairs a
reliability diagram / Brier score / ECE can be computed from directly.

Needs `torch`/`sbi` (forward passes through the trained ratio estimator
only, no training) -- same class of cluster-only script as `sbi_calibrate.py`
and `sbi_pixel_field.py infer`. Not tested locally (no torch/sbi available
here); syntax-checked only. Run on the cluster.

Query/pool split: the chosen `--pool_split` (default val) is itself split
into `--n_queries` query instances and the REMAINING instances as the
reweighting pool for every query -- disjoint by construction, so no query's
own (theta, x) pair ever appears in its own pool (that would leak the answer
into the reweighting and bias the calibration check to look better than it
is).

Usage
-----
python sbi_pixel_calibrate.py --n_los 75 \\
    --ratio sbi_runs/pixel_v2/ratio_estimator.pt \\
    --pool_dir sbi_runs/pixel_v2/sim --pool_split val \\
    --n_queries 500 --n_bins 10 \\
    --lya_catalog tb_lya.txt --properties_catalog sample_nirspec_properties.txt \\
    --output_dir sbi_runs/pixel_v2

Then plot locally (numpy/matplotlib only, no torch needed) with
`plot_pixel_calibration.py` against the saved `pixel_calibration.npz`.
"""
import os
import time
import argparse

import numpy as np

import real_data_run as rdr
import sbi_pixel_field as spf   # reuses load_sims() and _add_catalog_args()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    spf._add_catalog_args(ap)
    ap.add_argument('--ratio', type=str, required=True, help='ratio_estimator.pt from `train_nre`.')
    ap.add_argument('--pool_dir', type=str, required=True, help='--output_dir from `simulate`.')
    ap.add_argument('--pool_split', type=str, default='val', choices=['train', 'val'])
    ap.add_argument('--n_queries', type=int, default=500,
                    help='Held-out instances scored against a disjoint pool. Cost scales '
                         'roughly linearly with this -- start smaller (e.g. 100-200) for a '
                         'first pass.')
    ap.add_argument('--n_bins', type=int, default=10, help='Reliability-diagram probability bins.')
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output_dir', type=str, required=True)
    args = ap.parse_args()

    import torch

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(f"--device {args.device!r} requested but torch.cuda.is_available() "
                           f"is False on this machine/allocation -- pass --device cpu instead "
                           f"if this is a CPU-only node (this script is forward-pass-only, no "
                           f"training, CPU is normally fine).")

    checkpoint = torch.load(args.ratio, weights_only=False, map_location=args.device)
    ratio_estimator = checkpoint['ratio_estimator']
    n_gal, n_los = checkpoint['n_gal'], checkpoint['n_los']

    # Only used to validate n_gal matches the ratio estimator -- calibration
    # doesn't need the real catalog's x_obs at all, unlike `infer`.
    rdr._load_catalog_and_priors(
        args.lya_catalog, args.properties_catalog, args.z_lo, args.z_hi,
        args.z_min, args.muv_max, args.main_dir, r_max=args.r_max, prefer=args.prefer,
        legacy_catalog_path=args.legacy_catalog,
    )
    if len(rdr._S.x_gal) != n_gal:
        raise ValueError(f"Catalog has {len(rdr._S.x_gal)} galaxies but the ratio estimator "
                         f"was trained on {n_gal} -- check --lya_catalog/--properties_catalog/"
                         f"--z_lo/--z_hi match the `simulate` run that produced --pool_dir.")

    theta_all, x_all, pool_n_gal, pool_n_los = spf.load_sims(args.pool_dir, args.pool_split)
    if pool_n_gal != n_gal or pool_n_los != n_los:
        raise ValueError(f"--pool_dir sims ({pool_n_gal} gal x {pool_n_los} los) don't match "
                         f"the ratio estimator ({n_gal} gal x {n_los} los).")
    n_total = len(theta_all)

    rng = np.random.default_rng(args.seed)
    n_queries = min(args.n_queries, max(1, n_total // 2))
    if n_queries < args.n_queries:
        print(f"[calibrate] requested {args.n_queries} queries but pool_split only has "
              f"{n_total} instances -- capping to {n_queries} so the remaining pool stays "
              f"at least half of {n_total}.", flush=True)
    query_idx = rng.choice(n_total, size=n_queries, replace=False)
    pool_mask = np.ones(n_total, dtype=bool)
    pool_mask[query_idx] = False
    pool_idx = np.where(pool_mask)[0]
    print(f"[calibrate] {n_queries} queries, {len(pool_idx)}-instance disjoint pool "
          f"(from {args.pool_split} split, {n_total} total).", flush=True)

    theta_pool = theta_all[pool_idx]
    theta_pool_grid = theta_pool.reshape(len(theta_pool), n_gal, n_los)
    theta_pool_t = torch.as_tensor(theta_pool, dtype=torch.float32, device=args.device)

    theta_query = theta_all[query_idx].reshape(n_queries, n_gal, n_los)
    x_query = x_all[query_idx]

    all_pred, all_true = [], []
    t0 = time.perf_counter()
    print_every = max(1, n_queries // 20)
    with torch.no_grad():
        for qi in range(n_queries):
            x_q_t = torch.as_tensor(x_query[qi], dtype=torch.float32, device=args.device)
            x_q_tiled = x_q_t.unsqueeze(0).expand(len(theta_pool), -1)
            log_ratio = ratio_estimator(theta=theta_pool_t, x=x_q_tiled).squeeze(-1)
            log_ratio = log_ratio.detach().cpu().numpy()

            log_w = log_ratio - log_ratio.max()
            w = np.exp(log_w)
            w /= w.sum()
            pred_map = np.tensordot(w, theta_pool_grid, axes=(0, 0))   # (n_gal, n_los)

            all_pred.append(pred_map.ravel())
            all_true.append(theta_query[qi].ravel())

            if (qi + 1) % print_every == 0 or qi == n_queries - 1:
                elapsed = time.perf_counter() - t0
                rate = (qi + 1) / elapsed if elapsed > 0 else 0.0
                eta = (n_queries - qi - 1) / rate if rate > 0 else float('nan')
                print(f"[calibrate] {qi + 1}/{n_queries} queries scored "
                      f"({rate:.2f}/s, ETA {eta:.0f}s)", flush=True)

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)

    # ── Reliability diagram + summary stats ─────────────────────────────────
    bin_edges = np.linspace(0, 1, args.n_bins + 1)
    bin_idx = np.clip(np.digitize(all_pred, bin_edges) - 1, 0, args.n_bins - 1)
    emp_freq = np.full(args.n_bins, np.nan)
    bin_mean_pred = np.full(args.n_bins, np.nan)
    bin_count = np.zeros(args.n_bins, dtype=int)
    for b in range(args.n_bins):
        m = bin_idx == b
        bin_count[b] = int(m.sum())
        if m.sum() > 0:
            emp_freq[b] = all_true[m].mean()
            bin_mean_pred[b] = all_pred[m].mean()

    brier = float(np.mean((all_pred - all_true) ** 2))
    valid = bin_count > 0
    ece = float(np.sum(bin_count[valid] / bin_count.sum()
                       * np.abs(emp_freq[valid] - bin_mean_pred[valid])))

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'pixel_calibration.npz')
    np.savez(
        out_path,
        all_pred=all_pred, all_true=all_true,
        bin_edges=bin_edges, bin_mean_pred=bin_mean_pred, emp_freq=emp_freq, bin_count=bin_count,
        brier=brier, ece=ece, n_queries=n_queries, n_bins=args.n_bins, n_pool=len(pool_idx),
    )
    print(f"[calibrate] Brier score = {brier:.4f} (0=perfect, 0.25=uninformative-at-p0.5 baseline)")
    print(f"[calibrate] ECE = {ece:.4f} (weighted mean |predicted - empirical| across bins)")
    print(f"[calibrate] saved {out_path} -- {len(all_pred)} (pred, true) pixel pairs from "
          f"{n_queries} queries x {n_gal}x{n_los} pixels each.", flush=True)


if __name__ == '__main__':
    main()
