"""
Truth-vs-inferred ionization maps for held-out validation instances of the
pixel-field SBI (NRE), to test whether the posterior is biased against NEUTRAL
structure (on real data it never assigns P(neutral) > 0.36 anywhere).

For each of --n_queries held-out (theta_true, x) pairs from the val split, the
instance's x is scored against a DISJOINT pool (the rest of the split) with the
trained ratio estimator -- exactly as `infer` / `sbi_pixel_calibrate.py` do --
giving its posterior marginal P(neutral) map, a few SIR joint samples, and its ESS.
Saves everything needed to plot truth vs inference locally (numpy/matplotlib only)
with the lightcone-domain notebook, plus pooled neutral-recovery statistics:

  * for pixels that are truly NEUTRAL, the distribution of predicted P(neutral)
  * per instance: true neutral fraction vs posterior-mean neutral fraction
  * the same restricted to near-source bins (where the real-data signal is)

Cluster-only (torch/sbi), forward passes only. Same CLI as sbi_pixel_calibrate.py.

Usage
-----
python sbi_pixel_validation_maps.py --n_los 75 \\
    --ratio sbi_runs/pixel_v2/ratio_estimator.pt \\
    --pool_dir sbi_runs/pixel_v2/sim --pool_split val \\
    --n_queries 300 --n_maps 24 --n_joint 5 \\
    --lya_catalog tb_lya.txt --properties_catalog sample_nirspec_properties.txt \\
    --output_dir sbi_runs/pixel_v2
"""
import os
import time
import argparse

import numpy as np

import real_data_run as rdr
import sbi_pixel_field as spf   # load_sims(), _add_catalog_args()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    spf._add_catalog_args(ap)
    ap.add_argument('--ratio', type=str, required=True)
    ap.add_argument('--pool_dir', type=str, required=True)
    ap.add_argument('--pool_split', type=str, default='val', choices=['train', 'val'])
    ap.add_argument('--n_queries', type=int, default=300,
                    help='Held-out instances scored (for the pooled bias statistics).')
    ap.add_argument('--n_maps', type=int, default=24,
                    help='Of those, how many to save in full (true theta, marginal, joint samples).')
    ap.add_argument('--n_joint', type=int, default=5, help='SIR joint samples saved per map.')
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output_dir', type=str, required=True)
    args = ap.parse_args()

    import torch
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(f"--device {args.device!r} requested but CUDA is not available here; "
                           f"use --device cpu (forward passes only).")

    checkpoint = torch.load(args.ratio, weights_only=False, map_location=args.device)
    ratio_estimator = checkpoint['ratio_estimator']
    n_gal, n_los = checkpoint['n_gal'], checkpoint['n_los']

    rdr._load_catalog_and_priors(
        args.lya_catalog, args.properties_catalog, args.z_lo, args.z_hi,
        args.z_min, args.muv_max, args.main_dir, r_max=args.r_max, prefer=args.prefer,
        legacy_catalog_path=args.legacy_catalog,
    )
    if len(rdr._S.x_gal) != n_gal:
        raise ValueError(f"Catalog has {len(rdr._S.x_gal)} galaxies, ratio estimator {n_gal}.")

    theta_all, x_all, pg, pl = spf.load_sims(args.pool_dir, args.pool_split)
    if (pg, pl) != (n_gal, n_los):
        raise ValueError(f"pool sims ({pg}x{pl}) don't match the ratio estimator ({n_gal}x{n_los}).")
    n_total = len(theta_all)

    rng = np.random.default_rng(args.seed)
    n_queries = min(args.n_queries, max(1, n_total // 2))
    query_idx = rng.choice(n_total, size=n_queries, replace=False)
    pool_mask = np.ones(n_total, dtype=bool); pool_mask[query_idx] = False
    pool_idx = np.where(pool_mask)[0]
    theta_pool = theta_all[pool_idx]
    theta_pool_grid = theta_pool.reshape(len(theta_pool), n_gal, n_los)
    theta_pool_t = torch.as_tensor(theta_pool, dtype=torch.float32, device=args.device)
    print(f"[valmaps] {n_queries} queries vs a disjoint {len(pool_idx)}-instance pool "
          f"({args.pool_split} split).", flush=True)

    n_maps = min(args.n_maps, n_queries)
    true_maps = np.empty((n_maps, n_gal, n_los), np.float32)
    pred_maps = np.empty((n_maps, n_gal, n_los), np.float32)
    joint = np.empty((n_maps, args.n_joint, n_gal, n_los), np.float32)
    x_maps = x_all[query_idx[:n_maps]].astype(np.float32)

    ess = np.empty(n_queries)
    true_frac = np.empty(n_queries); pred_frac = np.empty(n_queries)
    true_frac_near = np.empty(n_queries); pred_frac_near = np.empty(n_queries)
    n_near = 5                                          # ~50 cMpc in front of each galaxy
    pred_on_true_neutral, pred_on_true_ionized = [], []

    t0 = time.perf_counter()
    with torch.no_grad():
        for qi in range(n_queries):
            x_q = torch.as_tensor(x_all[query_idx[qi]], dtype=torch.float32, device=args.device)
            lr = ratio_estimator(theta=theta_pool_t, x=x_q.unsqueeze(0).expand(len(theta_pool), -1))
            lr = lr.squeeze(-1).detach().cpu().numpy()
            w = np.exp(lr - lr.max()); w /= w.sum()
            pred = np.tensordot(w, theta_pool_grid, axes=(0, 0))          # P(neutral), (n_gal, n_los)
            true = theta_all[query_idx[qi]].reshape(n_gal, n_los)

            ess[qi] = 1.0 / np.sum(w ** 2)
            true_frac[qi], pred_frac[qi] = true.mean(), pred.mean()
            true_frac_near[qi], pred_frac_near[qi] = true[:, :n_near].mean(), pred[:, :n_near].mean()
            pred_on_true_neutral.append(pred[true > 0.5])
            pred_on_true_ionized.append(pred[true <= 0.5])

            if qi < n_maps:
                true_maps[qi], pred_maps[qi] = true, pred
                pick = rng.choice(len(w), size=args.n_joint, replace=True, p=w)   # SIR
                joint[qi] = theta_pool_grid[pick]

            if (qi + 1) % max(1, n_queries // 20) == 0:
                el = time.perf_counter() - t0
                print(f"[valmaps] {qi + 1}/{n_queries} ({(qi + 1) / el:.2f}/s)", flush=True)

    pn = np.concatenate(pred_on_true_neutral); pi = np.concatenate(pred_on_true_ionized)
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, 'pixel_validation_maps.npz')
    np.savez_compressed(
        out, true_maps=true_maps, pred_maps=pred_maps, joint=joint, x_maps=x_maps,
        query_idx=query_idx, ess=ess, true_frac=true_frac, pred_frac=pred_frac,
        true_frac_near=true_frac_near, pred_frac_near=pred_frac_near, n_near=n_near,
        pred_on_true_neutral=pn.astype(np.float32), pred_on_true_ionized=pi.astype(np.float32),
        n_gal=n_gal, n_los=n_los, n_pool=len(pool_idx))

    # quick text summary -- the numbers that answer "is neutral structure biased away?"
    print(f"[valmaps] global neutral fraction: true {true_frac.mean():.3f} vs posterior "
          f"{pred_frac.mean():.3f} (mean over {n_queries} queries; mean diff "
          f"{np.mean(pred_frac - true_frac):+.3f} +/- {np.std(pred_frac - true_frac) / np.sqrt(n_queries):.3f})")
    print(f"[valmaps] near-source ({n_near} bins): true {true_frac_near.mean():.3f} vs posterior "
          f"{pred_frac_near.mean():.3f}")
    for thr in (0.5, 0.68, 0.8):
        print(f"[valmaps] truly-neutral pixels with P(neutral) > {thr}: {np.mean(pn > thr):.3%}; "
              f"truly-ionized pixels with P(neutral) > {thr}: {np.mean(pi > thr):.3%}")
    print(f"[valmaps] max P(neutral) over all scored pixels: {max(pn.max(), pi.max()):.3f}; "
          f"median ESS {np.median(ess):.0f}")
    print(f"[valmaps] saved {out}", flush=True)


if __name__ == '__main__':
    main()
